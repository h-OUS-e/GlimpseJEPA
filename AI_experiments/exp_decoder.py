"""Decoder-conditioning sweep: does decoding z_t+1 get easier if the decoder also sees the
CURRENT frame? Train one frozen JEPA backbone, then train 7 decoder variants on its latents
and compare AR-decode quality before/after DMT (like train_quick.py).

Variants (all decode the predicted next-latent z_pred; conditioned ones also see the current frame):
    baseline  - MLP(z_pred)                       no conditioning (the current train_quick decoder)
    A_concat  - MLP([z_pred, z_cur])              concat current encoder-latent (idea A)
    B_resid   - cur_img + MLP([z_pred, z_cur])    residual off current frame (idea B)
    C_xattn   - z_pred queries, img tokens K/V    cross-attention ViT (idea C)
    D_convcat - MLP([z_pred, convenc(cur_img)])   conv-encode current image, concat (idea D)
    E_convadd - MLP(proj(z_pred) + convenc(img))  conv-encode current image, ADD (idea E)
    F_vitvae  - ViT encode img + z cond -> decode ViT-VAE style (idea F)

Conditioning convention at eval: conditioned decoders see the GROUND-TRUTH current frame
(the agent re-observes each step) and decode the AR-PREDICTED next latent. The baseline must
rebuild everything from the predicted latent alone. A 'copy current frame' floor is plotted as
reference: a residual/conditioned decoder that simply copies the current frame sits on that floor.

DMT shifts the predictor's output distribution, so a decoder trained BEFORE DMT then evaluated on
post-DMT AR latents sees a manifold it never trained on. To isolate that, every decoder is trained
TWICE -- once on the pre-DMT model, once on the post-DMT model -- and BOTH are evaluated on the same
post-DMT model. 'trained after DMT' should win if the manifold shift is what hurts recon.

Run: "C:/Users/Ous/miniconda3/envs/ML/python.exe" exp_decoder.py
"""

import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from einops import rearrange
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from glimpse import rollout
from jepa import JEPA
from ml_layers import ARPredictor, ActionEncoder, ImageEncoder, MLP_Projector, ViTBlock

# ================================================
#                CONFIG
# ================================================
SCALE_S, TRANS_S = 0.2, 0.1
BATCH = 64
T_MAX = 10
T_OOD = 2 * T_MAX          # OOD horizon to expose AR drift
Z_IMG = 36
Z_ACT = 3
IMG = 28
PATCH = 7                  # 28/7 = 4x4 = 16 tokens for the ViT/x-attn decoders
WINDOW = 3

BACKBONE_STEPS = 3000
DEC_STEPS = 2000
DMT_STEPS = 500
LR, DEC_LR, DMT_LR = 4e-4, 1e-3, 1e-4
LAMBD_SIG = 0.09

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

date_tag = datetime.now().strftime("%y_%m%d")
OUT = os.path.join("out", "plots", f"{date_tag}-decoder-sweep")
os.makedirs(OUT, exist_ok=True)

# ================================================
#                DATA
# ================================================
train_ds = torchvision.datasets.MNIST(root="./dataset", train=True, download=True, transform=transforms.ToTensor())
val_ds = torchvision.datasets.MNIST(root="./dataset", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, drop_last=True)


def fresh_train_iter():
    return iter(train_loader)


def next_imgs(it):
    """Pull the next image batch, restarting the iterator at epoch end."""
    try:
        imgs, _ = next(it)
    except StopIteration:
        it = fresh_train_iter()
        imgs, _ = next(it)
    return imgs.to(DEV).squeeze(1), it


# ================================================
#                BACKBONE (frozen after training)
# ================================================
def build_backbone():
    """JEPA backbone mirroring train_quick (no memory). Recon is left to the swept decoders."""
    enc = ImageEncoder(IMG * IMG, 512, Z_IMG, depth=3)
    act_enc = ActionEncoder(3, emb_dim=Z_ACT)
    predictor = ARPredictor(num_frames=T_MAX, depth=4, heads=4, mlp_dim=512, input_dim=Z_IMG,
                            hidden_dim=512, output_dim=512, action_dim=Z_ACT, window=WINDOW)
    projector_pred = MLP_Projector(input_dim=512, output_dim=Z_IMG, hidden_dim=256, norm_fn=nn.BatchNorm1d)
    projector = nn.LayerNorm(Z_IMG, elementwise_affine=False)
    return JEPA(enc, predictor, act_enc, decoder=None, projector=projector,
                projector_pred=projector_pred).to(DEV)


def train_backbone(model):
    """Train encoder+predictor with latent MSE + SigReg only (no recon, so latents aren't biased
    toward any one decoder)."""
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    it = fresh_train_iter()
    model.train()
    for step in tqdm(range(BACKBONE_STEPS), desc="backbone"):
        imgs, it = next_imgs(it)
        with torch.no_grad():
            _, acts, inp, tgt = rollout(imgs, T_MAX, SCALE_S, TRANS_S, device=DEV)
        z_pred, z_img, _ = model(inp, acts, ar_steps=0)
        z_tgt, _ = model.encode(tgt)
        loss = model.mse(z_pred, z_tgt, mean=False) + LAMBD_SIG * model.sigreg_loss(z_img)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


# ================================================
#                DECODER VARIANTS
# ================================================
def mlp_head(in_dim, hidden=512, depth=2):
    """Vector -> 784 logits-then-sigmoid MLP body shared by the flat decoders."""
    layers = [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
    layers += [nn.Linear(hidden, IMG * IMG)]
    return nn.Sequential(*layers)


def patchify(img):
    """(B,1,H,W) -> (B, N, patch*patch) non-overlapping patch tokens."""
    return rearrange(img, "b o (h p1) (w p2) -> b (h w) (o p1 p2)", p1=PATCH, p2=PATCH)


class ConvImgEnc(nn.Module):
    """Small CNN: current frame -> latent vector (for the conv-VAE-style decoders)."""
    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(inplace=True),   # 28 -> 14
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(inplace=True),  # 14 -> 7
            nn.Flatten(), nn.Linear(32 * 7 * 7, out_dim),
        )

    def forward(self, img):  # (B,1,H,W) -> (B,out_dim)
        return self.net(img)


def conv_head(hidden):
    """4x4 token grid -> 28x28 logits via transposed convs (np=4 path from ViTSpatialDecoder)."""
    return nn.Sequential(
        nn.ConvTranspose2d(hidden, hidden, 4, 1, 0), nn.GELU(),  # 4 -> 7
        nn.ConvTranspose2d(hidden, 32, 4, 2, 1), nn.GELU(),      # 7 -> 14
        nn.ConvTranspose2d(32, 1, 4, 2, 1),                      # 14 -> 28
    )


class BaseDecoder(nn.Module):
    """Uniform interface dec(z, cur_img, cur_z) -> (B,T,1,H,W) in [0,1]. Subclasses set _decode
    on flattened (B*T) tensors; the wrapper handles (B,T,...) folding."""
    def forward(self, z, cur_img, cur_z):
        B, T = z.shape[:2]
        z = rearrange(z, "b t d -> (b t) d")
        cur_z = rearrange(cur_z, "b t d -> (b t) d")
        cur_img = rearrange(cur_img, "b t o h w -> (b t) o h w")
        out = self._decode(z, cur_img, cur_z)             # (B*T,1,H,W)
        return rearrange(out, "(b t) o h w -> b t o h w", b=B)


class Baseline(BaseDecoder):
    """MLP(z_pred) only - no conditioning."""
    def __init__(self):
        super().__init__()
        self.net = mlp_head(Z_IMG)

    def _decode(self, z, cur_img, cur_z):
        return self.net(z).sigmoid().view(-1, 1, IMG, IMG)


class ConcatLatent(BaseDecoder):
    """MLP([z_pred, z_cur]) - concat current encoder-latent (idea A)."""
    def __init__(self):
        super().__init__()
        self.net = mlp_head(Z_IMG * 2)

    def _decode(self, z, cur_img, cur_z):
        return self.net(torch.cat([z, cur_z], -1)).sigmoid().view(-1, 1, IMG, IMG)


class Residual(BaseDecoder):
    """cur_img + MLP([z_pred, z_cur]) - decode the change off the current frame (idea B)."""
    def __init__(self):
        super().__init__()
        self.net = mlp_head(Z_IMG * 2)

    def _decode(self, z, cur_img, cur_z):
        delta = self.net(torch.cat([z, cur_z], -1)).view(-1, 1, IMG, IMG)
        return (cur_img + delta).clamp(0, 1)


class CrossAttn(BaseDecoder):
    """z_pred -> query tokens, current-frame patch tokens -> K/V; cross-attend then conv head (idea C)."""
    def __init__(self, hidden=64, heads=4, depth=2):
        super().__init__()
        self.np = IMG // PATCH
        n = self.np ** 2
        self.img_embed = nn.Linear(PATCH * PATCH, hidden)
        self.img_pos = nn.Parameter(torch.randn(1, n, hidden) * 0.02)
        self.to_q = nn.Linear(Z_IMG, n * hidden)
        self.q_pos = nn.Parameter(torch.randn(1, n, hidden) * 0.02)
        self.attn = nn.ModuleList([nn.MultiheadAttention(hidden, heads, batch_first=True) for _ in range(depth)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.ff = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden * 4),
                                               nn.GELU(), nn.Linear(hidden * 4, hidden)) for _ in range(depth)])
        self.head = conv_head(hidden)
        self.hidden = hidden

    def _decode(self, z, cur_img, cur_z):
        kv = self.img_embed(patchify(cur_img)) + self.img_pos              # (B,N,h) current-frame tokens
        q = self.to_q(z).view(-1, self.np ** 2, self.hidden) + self.q_pos  # (B,N,h) queries from z_pred
        for attn, norm, ff in zip(self.attn, self.norm, self.ff):
            q = q + attn(norm(q), kv, kv, need_weights=False)[0]
            q = q + ff(q)
        grid = rearrange(q, "b (h w) d -> b d h w", h=self.np)
        return self.head(grid).sigmoid()


class ConvConcat(BaseDecoder):
    """MLP([z_pred, convenc(cur_img)]) - conv-encode current image, concat (idea D)."""
    def __init__(self, enc_dim=64):
        super().__init__()
        self.enc = ConvImgEnc(enc_dim)
        self.net = mlp_head(Z_IMG + enc_dim)

    def _decode(self, z, cur_img, cur_z):
        return self.net(torch.cat([z, self.enc(cur_img)], -1)).sigmoid().view(-1, 1, IMG, IMG)


class ConvAdd(BaseDecoder):
    """MLP(proj(z_pred) + convenc(cur_img)) - conv-encode current image, ADD the two latents (idea E)."""
    def __init__(self, fuse_dim=Z_IMG):
        super().__init__()
        self.enc = ConvImgEnc(fuse_dim)
        self.z_proj = nn.Linear(Z_IMG, fuse_dim)
        self.net = mlp_head(fuse_dim)

    def _decode(self, z, cur_img, cur_z):
        fused = self.z_proj(z) + self.enc(cur_img)
        return self.net(fused).sigmoid().view(-1, 1, IMG, IMG)


class ViTVAE(BaseDecoder):
    """ViT-encode the current frame to tokens, FiLM-condition every token on z_pred, ViT-decode (idea F)."""
    def __init__(self, hidden=64, heads=4, depth=2):
        super().__init__()
        self.np = IMG // PATCH
        n = self.np ** 2
        self.embed = nn.Linear(PATCH * PATCH, hidden)
        self.pos = nn.Parameter(torch.randn(1, n, hidden) * 0.02)
        self.z_cond = nn.Linear(Z_IMG, hidden)                 # z -> per-token additive code
        self.enc_blocks = nn.ModuleList([ViTBlock(hidden, heads) for _ in range(depth)])
        self.dec_blocks = nn.ModuleList([ViTBlock(hidden, heads) for _ in range(depth)])
        self.head = conv_head(hidden)

    def _decode(self, z, cur_img, cur_z):
        x = self.embed(patchify(cur_img)) + self.pos
        for blk in self.enc_blocks:
            x = blk(x)
        x = x + self.z_cond(z).unsqueeze(1)                    # condition tokens on the predicted latent
        for blk in self.dec_blocks:
            x = blk(x)
        grid = rearrange(x, "b (h w) d -> b d h w", h=self.np)
        return self.head(grid).sigmoid()


DECODERS = {
    "baseline": Baseline,
    "A_concat": ConcatLatent,
    "B_resid": Residual,
    "C_xattn": CrossAttn,
    "D_convcat": ConvConcat,
    "E_convadd": ConvAdd,
    "F_vitvae": ViTVAE,
}


# ================================================
#                TRAIN / EVAL HELPERS
# ================================================
def train_decoder(model, dec):
    """Train one decoder on frozen backbone latents. Decode both the encoder-target latent and the
    teacher-forced predictor latent (cover the prediction manifold), conditioned on the current frame."""
    opt = torch.optim.Adam(dec.parameters(), lr=DEC_LR)
    it = fresh_train_iter()
    dec.train()
    for step in range(DEC_STEPS):
        imgs, it = next_imgs(it)
        with torch.no_grad():
            _, acts, inp, tgt = rollout(imgs, T_MAX, SCALE_S, TRANS_S, device=DEV)
            z_img, _ = model.encode(inp)               # current-frame latent (conditioning)
            z_tgt, _ = model.encode(tgt)               # perfect next latent
            z_tf, _, _ = model(inp, acts, ar_steps=0)  # teacher-forced predicted next latent
        tgt_f = tgt.float()
        loss = F.mse_loss(dec(z_tgt, inp, z_img), tgt_f) + F.mse_loss(dec(z_tf, inp, z_img), tgt_f)
        opt.zero_grad(); loss.backward(); opt.step()
    dec.eval()
    return dec


@torch.no_grad()
def ar_decode(model, dec, inp, acts, tgt):
    """AR rollout -> decode predicted next latents conditioned on the GT current frame.
    Returns decoded frames (B,T,1,H,W) and per-step pixel MSE (T,)."""
    z_ar, _, _ = model(inp, acts, ar_steps=inp.size(1))
    z_img, _ = model.encode(inp)
    decoded = dec(z_ar, inp, z_img)
    perstep = ((decoded - tgt.float()) ** 2).mean(dim=(0, 2, 3, 4))
    return decoded, perstep


@torch.no_grad()
def eval_val_ar(model):
    """Mean val AR latent MSE over the full val set (drift metric)."""
    tot, n = 0.0, 0
    for imgs, _ in val_loader:
        imgs = imgs.to(DEV).squeeze(1)
        _, acts, inp, tgt = rollout(imgs, T_MAX, SCALE_S, TRANS_S, device=DEV)
        z_tgt, _ = model.encode(tgt)
        z_pred, _, _ = model(inp, acts, ar_steps=inp.size(1))
        tot += model.mse(z_pred, z_tgt, mean=False).item(); n += 1
    return tot / max(n, 1)


def run_dmt(model):
    """DAgger Memory Training: freeze all but the predictor, unroll AR and regress to the frozen
    encoder trajectory. Cuts AR latent drift (decoders stay frozen)."""
    params = model.freeze_for_dmt()
    opt = torch.optim.AdamW(params, lr=DMT_LR)
    model.eval(); model.predictor.train()
    it = fresh_train_iter()
    for step in tqdm(range(DMT_STEPS), desc="dmt"):
        imgs, it = next_imgs(it)
        with torch.no_grad():
            _, acts, inp, tgt = rollout(imgs, T_MAX, SCALE_S, TRANS_S, device=DEV)
        loss = model.dmt_loss(inp, acts, tgt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


# ================================================
#                PLOTS
# ================================================
def plot_grid(name, tgt, dec_pre, dec_post):
    """Per trajectory: target / decoder-trained-pre-DMT / decoder-trained-post-DMT rows (both decode
    the SAME post-DMT AR latents); columns = rollout steps."""
    n, Tn = 4, tgt.size(1)
    fig, axes = plt.subplots(3 * n, Tn, figsize=(Tn * 0.55, 3 * n * 0.55))
    rows = [("tgt", tgt), ("dec@pre", dec_pre), ("dec@post", dec_post)]
    for b in range(n):
        for k, (lbl, src) in enumerate(rows):
            r = 3 * b + k
            for t in range(Tn):
                axes[r, t].imshow(src[b, t, 0].cpu(), cmap="gray", vmin=0, vmax=1)
                axes[r, t].set_xticks([]); axes[r, t].set_yticks([])
                if t == T_MAX - 1 and Tn > T_MAX:
                    for sp in axes[r, t].spines.values():
                        sp.set_color("red"); sp.set_linewidth(1.5)
            axes[r, 0].set_ylabel(f"t{b}\n{lbl}", fontsize=7)
    plt.suptitle(f"{name}: decoder trained pre- vs post-DMT, both on post-DMT model "
                 f"(T={Tn}, red=train horizon {T_MAX})", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"grid_{name}.png"), dpi=110)
    plt.close(fig)


def plot_perstep(perstep_pre, perstep_post, copy_floor):
    """Two panels (decoder trained pre- vs post-DMT, both on the post-DMT model): per-step recon MSE
    for every decoder + copy-frame floor."""
    steps = range(1, T_OOD + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    panels = [(axes[0], perstep_pre, "decoder trained BEFORE DMT"),
              (axes[1], perstep_post, "decoder trained AFTER DMT")]
    for ax, data, title in panels:
        for name, ps in data.items():
            ax.plot(steps, ps.cpu(), marker="o", ms=3, label=name)
        ax.plot(steps, copy_floor.cpu(), "k--", lw=1, label="copy current frame")
        ax.axvline(T_MAX, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("rollout step"); ax.set_yscale("log"); ax.set_title(title); ax.grid(alpha=0.3)
    axes[0].set_ylabel("decoder recon MSE (eval on post-DMT model)")
    axes[1].legend(fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "perstep_recon.png"), dpi=120)
    plt.close(fig)


def plot_summary(perstep_pre, perstep_post):
    """Grouped bars: mean OOD recon MSE per decoder, trained before vs after DMT (both eval post-DMT)."""
    names = list(perstep_pre.keys())
    ood = slice(T_MAX, T_OOD)  # steps past the train horizon
    pre = [perstep_pre[n][ood].mean().item() for n in names]
    post = [perstep_post[n][ood].mean().item() for n in names]
    x = torch.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.2, pre, 0.4, label="trained before DMT")
    ax.bar(x + 0.2, post, 0.4, label="trained after DMT")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("mean OOD recon MSE"); ax.set_yscale("log")
    ax.set_title(f"OOD recon (steps {T_MAX+1}-{T_OOD}): decoder trained pre- vs post-DMT (both eval post-DMT)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "summary_ood.png"), dpi=120)
    plt.close(fig)


# ================================================
#                MAIN
# ================================================
def main():
    print(f"device={DEV}  out={OUT}")
    model = build_backbone()
    train_backbone(model)
    for p in model.parameters():
        p.requires_grad_(False)
    ar_before = eval_val_ar(model)
    print(f"backbone trained. val AR latent MSE = {ar_before:.4f}")

    # Decoders trained on the PRE-DMT prediction manifold
    dec_pre = {}
    for name, cls in DECODERS.items():
        dec = train_decoder(model, cls().to(DEV))
        dec_pre[name] = dec
        print(f"  pre-DMT decoder {name} ({sum(p.numel() for p in dec.parameters())/1e3:.0f}k params)")

    # DMT finetunes the predictor (encoder + decoders untouched), shifting the AR latent manifold
    run_dmt(model)
    ar_after = eval_val_ar(model)
    print(f"DMT: val AR latent MSE {ar_before:.4f} -> {ar_after:.4f}")

    # Decoders trained AGAIN, now on the POST-DMT manifold (fresh instances)
    dec_post = {}
    for name, cls in DECODERS.items():
        dec = train_decoder(model, cls().to(DEV))
        dec_post[name] = dec
        print(f"  post-DMT decoder {name}")

    # Fixed viz batch (post-DMT), rolled to the OOD horizon
    viz_imgs, _ = next(iter(val_loader))
    viz_imgs = viz_imgs.to(DEV).squeeze(1)
    with torch.no_grad():
        _, viz_acts, viz_inp, viz_tgt = rollout(viz_imgs, T_OOD, SCALE_S, TRANS_S, device=DEV)
    # 'copy current frame' floor: predict frame t+1 by showing frame t
    copy_floor = ((viz_inp.float() - viz_tgt.float()) ** 2).mean(dim=(0, 2, 3, 4))

    # Evaluate BOTH decoder sets on the SAME post-DMT model
    out_pre, perstep_pre, out_post, perstep_post = {}, {}, {}, {}
    for name in DECODERS:
        out_pre[name], perstep_pre[name] = ar_decode(model, dec_pre[name], viz_inp, viz_acts, viz_tgt)
        out_post[name], perstep_post[name] = ar_decode(model, dec_post[name], viz_inp, viz_acts, viz_tgt)

    print(f"\n{'decoder':10s} {'OOD pre':>9s} {'OOD post':>9s} {'gain':>7s}")
    ood = slice(T_MAX, T_OOD)
    for name in DECODERS:
        a, b = perstep_pre[name][ood].mean().item(), perstep_post[name][ood].mean().item()
        print(f"{name:10s} {a:9.4f} {b:9.4f} {100*(a-b)/a:6.1f}%")

    # --- plots ---
    for name in DECODERS:
        plot_grid(name, viz_tgt.float(), out_pre[name], out_post[name])
    plot_perstep(perstep_pre, perstep_post, copy_floor)
    plot_summary(perstep_pre, perstep_post)
    print(f"\nsaved plots to {OUT}")


if __name__ == "__main__":
    main()
