"""Three architectural ideas to fight content loss in the glimpse JEPA.

Baseline = current best config (enc_norm LayerNorm + memory via AdaLN, "mem36_LN").
Idea A   = memory injected into the predictor INPUT (content channel), AdaLN sees action only.
Idea B   = spatially-structured latent (7x7 token grid) + spatiotemporal predictor + conv decoder.
Idea C   = slow content latent (causal cumulative) + fast pose latent predicted via AdaLN.

Validation rollouts are HONEST autoregressive: only the seed frame + actions are given,
and any memory/content is rebuilt causally from PREDICTED latents (no true-frame leak,
unlike the original evaluate() which fed memory from the true frames).

Run:  "C:/Users/Ous/miniconda3/envs/ML/python.exe" exp_ideas.py
"""

import os, json, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from einops import rearrange
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from glimpse import rollout
from jepa import JEPA, SIGReg
from ml_layers import (ImageEncoder, ActionEncoder, Decoder, ARPredictor,
                       MLP_Projector, MemoryPredictor)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = "exp_out/ideas"
os.makedirs(OUT, exist_ok=True)

# ---- fixed hyperparams (match exp_memory for comparability) ----
T = 10
H = W = 28
Z_IMG = 36
Z_ACT = 3
Z_MEM = 36
SCALE_S, TRANS_S = 0.2, 0.1
LR = 4e-4
STEPS = 800
SEED_DATA = 1234
LAMBD_SIG = 0.09
LAMBD_REC = 0.1

_DS = {}


def _datasets():
    if not _DS:
        _DS["tr"] = torchvision.datasets.MNIST(root="./dataset", train=True, download=True, transform=transforms.ToTensor())
        _DS["va"] = torchvision.datasets.MNIST(root="./dataset", train=False, download=True, transform=transforms.ToTensor())
    return _DS["tr"], _DS["va"]


def make_train_loader(seed=SEED_DATA):
    tr, _ = _datasets()
    g = torch.Generator().manual_seed(seed)
    return DataLoader(tr, batch_size=64, shuffle=True, drop_last=True, generator=g)


def val_batch():
    _, va = _datasets()
    return next(iter(DataLoader(va, batch_size=64, shuffle=False, drop_last=True)))[0].to(DEV).squeeze(1)


# =====================================================================
#  Idea A: memory as CONTENT (injected into predictor input, not AdaLN)
# =====================================================================
class JEPA_MemContent(JEPA):
    """DIFF vs baseline JEPA.predict:
       baseline: cond = cat(action, memory) -> AdaLN carries memory (a scale/shift channel).
       here:     cond = action only; memory is added to the predictor INPUT x via a
                 zero-init mem_proj, so attention can use it as content.
    """
    def __init__(self, *args, z_mem=Z_MEM, z_img=Z_IMG, **kw):
        super().__init__(*args, **kw)
        self.mem_proj = nn.Linear(z_mem, z_img)
        nn.init.zeros_(self.mem_proj.weight); nn.init.zeros_(self.mem_proj.bias)

    def predict(self, z_img, z_action, z_memory, ar_steps=0):
        Tn = z_img.size(1)
        cond = z_action  # DIFF: action only (no memory in AdaLN)
        mem = self.mem_proj(z_memory) if z_memory is not None else None
        if not ar_steps:
            x = z_img if mem is None else z_img + mem
            return self.project(self.predictor(x, cond))
        ar_steps = min(ar_steps, Tn)
        z_in = z_img[:, :1]
        preds = []
        for t in range(ar_steps):
            x = z_in if mem is None else z_in + mem[:, :t + 1]
            raw = self.predictor(x, cond[:, :t + 1])[:, -1:]
            pred = self.project(raw)
            preds.append(pred)
            z_in = torch.cat([z_in, pred], dim=1)
        preds = torch.cat(preds, dim=1)
        if ar_steps >= Tn:
            return preds
        z_full = torch.cat([z_in, z_img[:, ar_steps + 1:]], dim=1)
        xf = z_full if mem is None else z_full + mem
        tail = self.project(self.predictor(xf, cond)[:, ar_steps:])
        return torch.cat([preds, tail], dim=1)


class MemModel(nn.Module):
    """Wraps baseline / Idea-A JEPA with a uniform train + honest-AR interface."""
    def __init__(self, mode="adaln", name="baseline"):
        super().__init__()
        self.name = name
        self.mode = mode
        enc = ImageEncoder(H * W, 512, Z_IMG, depth=3)
        ae = ActionEncoder(3, emb_dim=Z_ACT)
        action_dim = Z_ACT + Z_MEM if mode == "adaln" else Z_ACT  # DIFF: A drops memory from AdaLN
        pr = ARPredictor(num_frames=T, depth=4, heads=4, mlp_dim=512, input_dim=Z_IMG,
                         hidden_dim=512, output_dim=512, action_dim=action_dim)
        pj = MLP_Projector(input_dim=512, output_dim=Z_IMG, hidden_dim=256, norm_fn=nn.BatchNorm1d)
        dc = Decoder(z_dim=Z_IMG, hidden_dim=512, h=H, w=W, depth=2)
        mp = MemoryPredictor(Z_IMG, Z_MEM, hidden_dim=256, depth=2, heads=4)
        projector = nn.LayerNorm(Z_IMG, elementwise_affine=False)  # enc_norm
        cls = JEPA if mode == "adaln" else JEPA_MemContent
        self.jepa = cls(enc, pr, ae, decoder=dc, projector=projector,
                        projector_pred=pj, memory_predictor=mp)

    def forward_train(self, inp, actions, tgt):
        m = self.jepa
        z_pred, z_img, _ = m(inp, actions, ar_steps=0)  # teacher forced
        z_tgt, _ = m.encode(tgt)
        loss_mse = m.mse(z_pred, z_tgt, mean=False)
        loss_sig = m.sigreg_loss(z_img)
        loss_rec = m.recon_loss(z_img.detach(), inp)
        loss = loss_mse + LAMBD_SIG * loss_sig + LAMBD_REC * loss_rec
        return loss, {"mse": loss_mse.item(), "sig": loss_sig.item(), "rec": loss_rec.item()}

    @torch.no_grad()
    def eval_ar(self, inp, actions, tgt):
        m = self.jepa
        z_tgt, _ = m.encode(tgt)
        tvar = z_tgt.var().item() + 1e-8
        z_img, z_action = m.encode(inp, actions)

        # teacher forced (uses true-frame memory, fine: all frames are inputs here)
        z_tf, _, _ = m(inp, actions, ar_steps=0)

        # HONEST AR: only seed observed; memory rebuilt from predicted latents each step
        z_in = z_img[:, :1]
        for t in range(T):
            mem = m.predict_memory(z_in) if m.memory_predictor is not None else None
            if self.mode == "adaln":
                cond = z_action[:, :t + 1] if mem is None else torch.cat([z_action[:, :t + 1], mem], dim=-1)
                raw = m.predictor(z_in, cond)[:, -1:]
            else:
                x = z_in + (m.mem_proj(mem) if mem is not None else 0)
                raw = m.predictor(x, z_action[:, :t + 1])[:, -1:]
            pred = m.project(raw)
            z_in = torch.cat([z_in, pred], dim=1)
        z_ar = z_in[:, 1:]  # frames 1..T

        rec_enc, rec_tf, rec_ar = m.decode(z_img), m.decode(z_tf), m.decode(z_ar)
        return self._metrics(z_tf, z_ar, z_tgt, tvar, z_img, inp, tgt, rec_enc, rec_tf, rec_ar)

    def _metrics(self, z_tf, z_ar, z_tgt, tvar, z_lat, inp, tgt, rec_enc, rec_tf, rec_ar):
        mse_tf = ((z_tf - z_tgt) ** 2).mean(-1).mean(0)  # (T,)
        mse_ar = ((z_ar - z_tgt) ** 2).mean(-1).mean(0)
        met = {
            "nmse_tf": mse_tf.mean().item() / tvar,
            "nmse_ar": mse_ar.mean().item() / tvar,
            "mse_ar_t": (mse_ar / tvar).cpu().tolist(),
            "px_enc": F.mse_loss(rec_enc, inp.float()).item(),
            "px_tf": F.mse_loss(rec_tf, tgt.float()).item(),
            "px_ar": F.mse_loss(rec_ar, tgt.float()).item(),
            "z_std": z_lat.reshape(-1, z_lat.size(-1)).std(0).mean().item(),
        }
        return met, (inp, tgt, rec_enc, rec_tf, rec_ar)


# =====================================================================
#  Idea B: spatial token latent (7x7 grid) + spatiotemporal predictor
# =====================================================================
class SpatialEncoder(nn.Module):
    """28x28 image -> (B, N=49, C) tokens. Non-affine LN per token pins scale for SigReg."""
    def __init__(self, c=8):
        super().__init__()
        self.c = c
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.GELU(),   # 28->14
            nn.Conv2d(32, c, 3, stride=2, padding=1),              # 14->7
        )
        self.norm = nn.LayerNorm(c, elementwise_affine=False)

    def forward(self, img):  # img: (BT, 1, H, W)
        h = self.net(img)                      # (BT, C, 7, 7)
        h = rearrange(h, "b c x y -> b (x y) c")
        return self.norm(h)                    # (BT, 49, C)


class SpatialDecoder(nn.Module):
    """(B*, N=49, C) tokens -> (B*, 1, 28, 28)."""
    def __init__(self, c=8):
        super().__init__()
        self.c = c
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c, 32, 3, stride=2, padding=1, output_padding=1), nn.GELU(),  # 7->14
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),  # 14->28
        )

    def forward(self, tok):  # (..., 49, C)
        lead = tok.shape[:-2]
        x = rearrange(tok.reshape(-1, 49, self.c), "b (x y) c -> b c x y", x=7)
        out = self.net(x)
        return out.view(*lead, 1, 28, 28)


def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


class STBlock(nn.Module):
    """Spatiotemporal transformer block with AdaLN-zero on the action.
    Tokens are flat (B, T*N, H); a block-causal mask lets frame t attend to all tokens <= t."""
    def __init__(self, dim, heads=4, mlp=4):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp), nn.GELU(), nn.Linear(dim * mlp, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c, mask):  # x:(B,L,H)  c:(B,L,H) per-token action embed  mask:(L,L)
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, -1)
        h = _modulate(self.n1(x), sh1, sc1)
        q, k, v = self.qkv(h).chunk(3, -1)
        q, k, v = (rearrange(t, "b l (h d) -> b h l d", h=self.h) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        a = rearrange(a, "b h l d -> b l (h d)")
        x = x + g1 * self.proj(a)
        x = x + g2 * self.mlp(_modulate(self.n2(x), sh2, sc2))
        return x


class SpatialPredictor(nn.Module):
    """Predicts next-frame tokens from current tokens + action. Block-causal over T frames."""
    def __init__(self, c=8, hidden=64, depth=4, heads=4, n=49):
        super().__init__()
        self.n, self.hidden = n, hidden
        self.in_proj = nn.Linear(c, hidden)
        self.act_proj = nn.Linear(3, hidden)
        self.sp_pos = nn.Parameter(torch.randn(1, 1, n, hidden) * 0.02)
        self.tp_pos = nn.Parameter(torch.randn(1, T, 1, hidden) * 0.02)
        self.blocks = nn.ModuleList([STBlock(hidden, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, c)

    def _mask(self, Tn, device):
        # block-causal: token in frame i may attend to tokens in frames j<=i
        fi = torch.arange(Tn, device=device).repeat_interleave(self.n)
        return (fi[None, :] <= fi[:, None])  # (L,L) bool, True=keep

    def forward(self, tok, action):  # tok:(B,T,N,C) action:(B,T,3) -> pred tokens (B,T,N,C)
        B, Tn, N, _ = tok.shape
        x = self.in_proj(tok) + self.sp_pos + self.tp_pos[:, :Tn]
        x = rearrange(x, "b t n h -> b (t n) h")
        c = self.act_proj(action)[:, :, None, :].expand(B, Tn, N, self.hidden)
        c = rearrange(c, "b t n h -> b (t n) h")
        mask = self._mask(Tn, tok.device)
        for blk in self.blocks:
            x = blk(x, c, mask)
        x = self.out(self.norm(x))
        return rearrange(x, "b (t n) c -> b t n c", t=Tn)


class SpatialModel(nn.Module):
    """Idea B. DIFF vs baseline: latent is a 7x7 token grid, not a 36-vector; the predictor
    moves content with spatial+temporal attention; decoder is convolutional."""
    def __init__(self, c=8, name="B_spatial"):
        super().__init__()
        self.name = name
        self.c = c
        self.enc = SpatialEncoder(c)
        self.pred = SpatialPredictor(c)
        self.dec = SpatialDecoder(c)
        self.sigreg = SIGReg()

    def encode(self, frames):  # (B,T,1,H,W) -> (B,T,49,C)
        B = frames.size(0)
        z = self.enc(rearrange(frames.float(), "b t c h w -> (b t) c h w"))
        return rearrange(z, "(b t) n c -> b t n c", b=B)

    def forward_train(self, inp, actions, tgt):
        z = self.encode(inp)
        z_tgt = self.encode(tgt)
        z_pred = self.pred(z, actions)
        loss_mse = F.mse_loss(z_pred, z_tgt.detach())
        # SigReg on flattened tokens per frame
        zf = rearrange(z, "b t n c -> t b (n c)")
        loss_sig = self.sigreg(zf)
        rec = self.dec(z.detach())
        loss_rec = F.mse_loss(rec, inp.float())
        loss = loss_mse + LAMBD_SIG * loss_sig + LAMBD_REC * loss_rec
        return loss, {"mse": loss_mse.item(), "sig": loss_sig.item(), "rec": loss_rec.item()}

    @torch.no_grad()
    def eval_ar(self, inp, actions, tgt):
        z = self.encode(inp)
        z_tgt = self.encode(tgt)
        tvar = z_tgt.var().item() + 1e-8
        z_tf = self.pred(z, actions)
        # honest AR: seed tokens only
        z_in = z[:, :1]
        for t in range(T):
            pred = self.pred(z_in, actions[:, :t + 1])[:, -1:]
            z_in = torch.cat([z_in, pred], dim=1)
        z_ar = z_in[:, 1:]
        rec_enc, rec_tf, rec_ar = self.dec(z), self.dec(z_tf), self.dec(z_ar)
        mse_tf = ((z_tf - z_tgt) ** 2).mean((-1, -2)).mean(0)
        mse_ar = ((z_ar - z_tgt) ** 2).mean((-1, -2)).mean(0)
        met = {
            "nmse_tf": mse_tf.mean().item() / tvar,
            "nmse_ar": mse_ar.mean().item() / tvar,
            "mse_ar_t": (mse_ar / tvar).cpu().tolist(),
            "px_enc": F.mse_loss(rec_enc, inp.float()).item(),
            "px_tf": F.mse_loss(rec_tf, tgt.float()).item(),
            "px_ar": F.mse_loss(rec_ar, tgt.float()).item(),
            "z_std": z.std().item(),
        }
        return met, (inp, tgt, rec_enc, rec_tf, rec_ar)


# =====================================================================
#  Idea C: slow content latent + fast pose latent (MilliVid-style)
# =====================================================================
class PoseContentModel(nn.Module):
    """Idea C. DIFF vs baseline: factor the latent into a SLOW content code (causal cumulative
    mean over observed frames -> the digit identity, barely drifts) and a FAST pose code
    (predicted autoregressively via AdaLN). Decoder reads (content, pose)."""
    def __init__(self, dc=32, dp=8, name="C_slowfast"):
        super().__init__()
        self.name = name
        self.dc, self.dp = dc, dp
        self.frame_enc = ImageEncoder(H * W, 256, 128, depth=2)
        self.to_pose = nn.Linear(128, dp)
        self.to_content = nn.Linear(128, dc)
        self.pose_norm = nn.LayerNorm(dp, elementwise_affine=False)
        self.content_norm = nn.LayerNorm(dc, elementwise_affine=False)
        self.content_to_pose = nn.Linear(dc, dp)  # content injected into predictor input
        self.pred = ARPredictor(num_frames=T, depth=4, heads=4, mlp_dim=256, input_dim=dp,
                                hidden_dim=128, output_dim=dp, action_dim=Z_ACT)
        self.dec = Decoder(z_dim=dc + dp, hidden_dim=512, h=H, w=W, depth=2)
        self.sigreg = SIGReg()

    def encode(self, frames):
        B = frames.size(0)
        h = self.frame_enc(rearrange(frames.float(), "b t c h w -> (b t) c h w"))
        h = rearrange(h, "(b t) d -> b t d", b=B)
        pose = self.pose_norm(self.to_pose(h))            # (B,T,dp) fast
        cemb = self.to_content(h)                          # (B,T,dc)
        content = self.content_norm(torch.cumsum(cemb, 1) / torch.arange(1, frames.size(1) + 1, device=h.device).view(1, -1, 1))  # causal cummean -> slow
        return pose, content

    def _predict_pose(self, pose, content, actions):
        x = pose + self.content_to_pose(content)  # content as input context
        return self.pred(x, actions)

    def forward_train(self, inp, actions, tgt):
        pose, content = self.encode(inp)
        pose_t, _ = self.encode(tgt)
        pose_pred = self._predict_pose(pose, content, actions)
        loss_mse = F.mse_loss(pose_pred, pose_t.detach())
        loss_sig = self.sigreg(rearrange(pose, "b t d -> t b d")) + self.sigreg(rearrange(content, "b t d -> t b d"))
        rec = self.dec(torch.cat([content, pose], -1).detach())
        loss_rec = F.mse_loss(rec, inp.float())
        loss = loss_mse + LAMBD_SIG * loss_sig + LAMBD_REC * loss_rec
        return loss, {"mse": loss_mse.item(), "sig": loss_sig.item(), "rec": loss_rec.item()}

    @torch.no_grad()
    def eval_ar(self, inp, actions, tgt):
        pose, content = self.encode(inp)
        pose_t, content_t = self.encode(tgt)
        tvar = pose_t.var().item() + 1e-8
        pose_tf = self._predict_pose(pose, content, actions)
        # honest AR: content frozen at the seed (only seed observed); roll out pose
        c0 = content[:, :1]                       # content from seed only
        p_in = pose[:, :1]
        for t in range(T):
            x = p_in + self.content_to_pose(c0.expand(-1, p_in.size(1), -1))
            pred = self.pred(x, actions[:, :t + 1])[:, -1:]
            p_in = torch.cat([p_in, pred], dim=1)
        pose_ar = p_in[:, 1:]
        c_rep = c0.expand(-1, T, -1)
        rec_enc = self.dec(torch.cat([content, pose], -1))
        rec_tf = self.dec(torch.cat([content_t, pose_tf], -1))  # TF uses observed content per step
        rec_ar = self.dec(torch.cat([c_rep, pose_ar], -1))      # AR uses frozen seed content
        mse_tf = ((pose_tf - pose_t) ** 2).mean(-1).mean(0)
        mse_ar = ((pose_ar - pose_t) ** 2).mean(-1).mean(0)
        met = {
            "nmse_tf": mse_tf.mean().item() / tvar,
            "nmse_ar": mse_ar.mean().item() / tvar,
            "mse_ar_t": (mse_ar / tvar).cpu().tolist(),
            "px_enc": F.mse_loss(rec_enc, inp.float()).item(),
            "px_tf": F.mse_loss(rec_tf, tgt.float()).item(),
            "px_ar": F.mse_loss(rec_ar, tgt.float()).item(),
            "z_std": pose.std().item(),
        }
        return met, (inp, tgt, rec_enc, rec_tf, rec_ar)


# =====================================================================
#  Generic train loop + plotting
# =====================================================================
def train_model(model, steps=STEPS, seed=0, val_imgs=None):
    model = model.to(DEV)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loader = make_train_loader(SEED_DATA)
    it = iter(loader)
    curve = []
    model.train()
    t0 = time.time()
    for step in range(steps):
        try:
            imgs, _ = next(it)
        except StopIteration:
            it = iter(loader); imgs, _ = next(it)
        imgs = imgs.to(DEV).squeeze(1)
        with torch.no_grad():
            _, actions, inp, tgt = rollout(imgs, T, SCALE_S, TRANS_S, device=DEV)
        loss, parts = model.forward_train(inp, actions, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0:
            curve.append((step, parts["mse"]))
    dt = time.time() - t0
    model.eval()
    with torch.no_grad():
        _, actions, inp, tgt = rollout(val_imgs, T, SCALE_S, TRANS_S, device=DEV)
        met, pack = model.eval_ar(inp, actions, tgt)
    met["train_s"] = dt
    met["n_params"] = sum(p.numel() for p in model.parameters())
    print(f"[{model.name:12s}] nmse_tf {met['nmse_tf']:.4f}  nmse_ar {met['nmse_ar']:.4f} | "
          f"px_enc {met['px_enc']:.4f}  px_tf {met['px_tf']:.4f}  px_ar {met['px_ar']:.4f} | "
          f"z_std {met['z_std']:.2f} ({dt:.0f}s)")
    return met, curve, pack


def plot_rollout(name, pack, n=3):
    inp, tgt, rec_enc, rec_tf, rec_ar = [p.cpu() for p in pack]
    rows = 5
    fig, ax = plt.subplots(rows * n, T, figsize=(T * 0.8, rows * n * 0.8))
    labels = ["inp", "dec(z)", "tgt", "dec(TF)", "dec(AR)"]
    srcs = [inp, rec_enc, tgt, rec_tf, rec_ar]
    for b in range(n):
        for r, (lab, src) in enumerate(zip(labels, srcs)):
            for t in range(T):
                a = ax[rows * b + r, t]
                a.imshow(src[b, t, 0], cmap="gray", vmin=0, vmax=1)
                a.set_xticks([]); a.set_yticks([])
            ax[rows * b + r, 0].set_ylabel(f"{lab}", fontsize=7)
    plt.suptitle(f"{name}: rows = inp / dec(z_enc) / tgt / dec(TF) / dec(AR-honest)", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}_rollout.png", dpi=90); plt.close(fig)


def plot_training(curves):
    plt.figure(figsize=(8, 4))
    for name, c in curves.items():
        xs = [s for s, _ in c]; ys = [v for _, v in c]
        plt.plot(xs, ys, label=name)
    plt.xlabel("step"); plt.ylabel("train latent MSE"); plt.yscale("log")
    plt.title("Training (latent MSE)"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/training.png", dpi=100); plt.close()


def plot_eval(results):
    names = list(results)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    # per-step AR latent nmse
    for n in names:
        ax[0].plot(range(1, T + 1), results[n]["mse_ar_t"], marker="o", label=n)
    ax[0].set_xlabel("rollout step"); ax[0].set_ylabel("AR latent nMSE")
    ax[0].set_title("Per-step AR latent nMSE (honest)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    # latent nmse bars
    x = range(len(names)); w = 0.35
    ax[1].bar([i - w / 2 for i in x], [results[n]["nmse_tf"] for n in names], w, label="TF")
    ax[1].bar([i + w / 2 for i in x], [results[n]["nmse_ar"] for n in names], w, label="AR")
    ax[1].set_xticks(list(x)); ax[1].set_xticklabels(names, rotation=20, fontsize=8)
    ax[1].set_ylabel("latent nMSE"); ax[1].set_title("Latent nMSE"); ax[1].legend()
    # pixel recon bars
    ax[2].bar([i - w / 2 for i in x], [results[n]["px_tf"] for n in names], w, label="px TF")
    ax[2].bar([i + w / 2 for i in x], [results[n]["px_ar"] for n in names], w, label="px AR")
    ax[2].set_xticks(list(x)); ax[2].set_xticklabels(names, rotation=20, fontsize=8)
    ax[2].set_ylabel("pixel MSE"); ax[2].set_title("Pixel recon (AR = honest)"); ax[2].legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/eval_compare.png", dpi=100); plt.close()


# =====================================================================
#  Diagnostics
# =====================================================================
def diagnostics(base_model, val_imgs, steps=1500):
    """(1) probe decoder: can a strong decoder reproduce the TRUE latent? (encoder ceiling)
       (2) per-step latent error vs newly-revealed content + action magnitude."""
    m = base_model.jepa
    # ---- probe decoder on frozen encoder ----
    for p in m.parameters():
        p.requires_grad_(False)
    probe = Decoder(z_dim=Z_IMG, hidden_dim=512, h=H, w=W, depth=3).to(DEV)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    loader = make_train_loader(SEED_DATA); it = iter(loader)
    for _ in range(steps):
        try:
            imgs, _ = next(it)
        except StopIteration:
            it = iter(loader); imgs, _ = next(it)
        imgs = imgs.to(DEV).squeeze(1)
        with torch.no_grad():
            _, _, _, tgt = rollout(imgs, T, SCALE_S, TRANS_S, device=DEV)
            z_tgt, _ = m.encode(tgt)
        rec = probe(z_tgt)
        loss = F.mse_loss(rec, tgt.float())
        opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        _, actions, inp, tgt = rollout(val_imgs, T, SCALE_S, TRANS_S, device=DEV)
        z_tgt, _ = m.encode(tgt)
        px_true = F.mse_loss(probe(z_tgt), tgt.float()).item()
        px_joint = F.mse_loss(m.decode(z_tgt), tgt.float()).item()

    # ---- per-step error vs revealed content ----
    with torch.no_grad():
        met, _ = base_model.eval_ar(inp, tgt=tgt, actions=actions)
        err_t = met["mse_ar_t"]
        # newly revealed = pixel mass appearing in tgt that wasn't in inp (content the model can't know)
        revealed = F.relu(tgt.float() - inp.float()).flatten(2).sum(-1).mean(0).cpu().tolist()  # (T,)
        act_mag = actions.norm(dim=-1).mean(0).cpu().tolist()  # (T,)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(["dec(z_true)\nprobe", "dec(z_true)\njoint"], [px_true, px_joint])
    ax[0].set_ylabel("pixel MSE"); ax[0].set_title("Is the TRUE latent decodable?\n(encoder ceiling vs joint decoder)")
    ln1 = ax[1].plot(range(1, T + 1), err_t, "o-", color="tab:red", label="AR latent nMSE")
    ax2 = ax[1].twinx()
    ln2 = ax2.plot(range(1, T + 1), revealed, "s--", color="tab:blue", label="newly-revealed content")
    ln3 = ax2.plot(range(1, T + 1), act_mag, "^:", color="tab:green", label="|action|")
    ax[1].set_xlabel("rollout step"); ax[1].set_ylabel("AR latent nMSE", color="tab:red")
    ax2.set_ylabel("revealed content / |action|")
    ax[1].set_title("Error tracks newly-revealed (unknowable) content")
    lns = ln1 + ln2 + ln3
    ax[1].legend(lns, [l.get_label() for l in lns], fontsize=8, loc="upper left")
    plt.tight_layout(); plt.savefig(f"{OUT}/diagnostics.png", dpi=100); plt.close()
    print(f"[diagnostics] probe px(true latent) {px_true:.4f}  joint px(true latent) {px_joint:.4f}")
    return {"px_true_probe": px_true, "px_true_joint": px_joint, "err_t": err_t, "revealed": revealed, "act_mag": act_mag}


def main(steps=STEPS, diag_steps=1500):
    vb = val_batch()
    results, curves, packs = {}, {}, {}
    models = {
        "baseline": MemModel("adaln", "baseline"),
        "A_memcontent": MemModel("content", "A_memcontent"),
        "B_spatial": SpatialModel(name="B_spatial"),
        "C_slowfast": PoseContentModel(name="C_slowfast"),
    }
    base_for_diag = None
    for name, model in models.items():
        met, curve, pack = train_model(model, steps=steps, val_imgs=vb)
        results[name], curves[name], packs[name] = met, curve, pack
        plot_rollout(name, pack)
        if name == "baseline":
            base_for_diag = model
    diag = diagnostics(base_for_diag, vb, steps=diag_steps)
    plot_training(curves)
    plot_eval(results)
    with open(f"{OUT}/results.json", "w") as f:
        json.dump({"results": results, "diag": {k: diag[k] for k in ("px_true_probe", "px_true_joint", "err_t", "revealed", "act_mag")}}, f, indent=2)
    print("\nsaved plots + results to", OUT)
    return results, diag


if __name__ == "__main__":
    import sys
    s = int(sys.argv[1]) if len(sys.argv) > 1 else STEPS
    ds = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    main(steps=s, diag_steps=ds)
