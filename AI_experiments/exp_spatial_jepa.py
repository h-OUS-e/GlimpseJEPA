"""Spatial-latent JEPA: ViT token grid + spatiotemporal residual predictor + ViT probe decoder.

Design decisions (from the sweeps):
- latent = 16 tokens x C=8 = 128 floats (ViT patch-7), per-token LayerNorm. ~50x smaller than predictor.
- predictor: spatiotemporal transformer, block-causal across time (frame t attends to all tokens <= t),
  AdaLN on action, predicts the RESIDUAL from the current frame's tokens (consecutive glimpses are close).
- recon: DETACHED probe decoder (visualization only, like LeWorldModel) -> prediction + gentle SigReg
  fully shape the tokens; spatial tokens already decode sharply.
- block-causal temporal attention gives the predictor direct access to past content, so Idea A's
  separate memory channel is subsumed (added back only if rollout still drifts).

Validation rollout is HONEST autoregressive (seed tokens only).

Run: "C:/Users/Ous/miniconda3/envs/ML/python.exe" exp_spatial_jepa.py [steps]
"""
import sys, os, json
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import exp_ideas as E
from exp_vit_sweep import ViTBlock
from glimpse import rollout

DEV, T = E.DEV, E.T
OUT = "exp_out/spatial"; os.makedirs(OUT, exist_ok=True)
STEPS = 1500
N_TOK, C = 16, 8           # patch-7 -> 4x4=16 tokens, 8 dims each -> 128-float latent
SIG_W, REC_W = 0.05, 1.0   # gentle SigReg; recon is detached so its weight only scales the decoder


class ViTSpatialEncoder(nn.Module):
    """Image -> (B, 16, C) patch tokens. Per-token non-affine LN pins scale for SigReg."""
    def __init__(self, patch=7, c=C, hidden=64, depth=2, heads=4, img=28):
        super().__init__()
        self.p, self.np = patch, img // patch
        n = self.np ** 2
        self.embed = nn.Linear(patch * patch, hidden)
        self.pos = nn.Parameter(torch.randn(1, n, hidden) * 0.02)
        self.blocks = nn.ModuleList([ViTBlock(hidden, heads) for _ in range(depth)])
        self.to_latent = nn.Linear(hidden, c)
        self.norm = nn.LayerNorm(c, elementwise_affine=False)

    def forward(self, img):  # (BT,1,28,28) -> (BT,16,C)
        x = self.embed(rearrange(img, "b o (h p1) (w p2) -> b (h w) (o p1 p2)", p1=self.p, p2=self.p)) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(self.to_latent(x))


class ViTSpatialDecoder(nn.Module):
    """(B,16,C) -> logits (B,1,28,28). ViT blocks give the tokens global context, then a CONV render
    head upsamples the 4x4 token grid (4->7->14->28) so neighbors blend -> no per-patch seams/gridding
    (the old Linear+unpatchify decoded each token's 7x7 block independently)."""
    def __init__(self, patch=7, c=C, hidden=64, depth=2, heads=4, img=28):
        super().__init__()
        self.np = img // patch  # 4x4 token grid
        n = self.np ** 2
        self.from_latent = nn.Linear(c, hidden)
        self.pos = nn.Parameter(torch.randn(1, n, hidden) * 0.02)
        self.blocks = nn.ModuleList([ViTBlock(hidden, heads) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden, 4, 1, 0), nn.GELU(),  # 4->7
            nn.ConvTranspose2d(hidden, 32, 4, 2, 1), nn.GELU(),      # 7->14
            nn.ConvTranspose2d(32, 1, 4, 2, 1),                      # 14->28 (logits)
        )

    def forward(self, tok):  # (B,16,C) -> (B,1,28,28) logits
        x = self.from_latent(tok) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = rearrange(x, "b (h w) d -> b d h w", h=self.np)
        return self.head(x)


class STPredictor(nn.Module):
    """Spatiotemporal residual predictor. Block-causal over T*N tokens, AdaLN on action."""
    def __init__(self, c=C, hidden=128, depth=4, heads=4, n=N_TOK):
        super().__init__()
        self.n = n
        self.in_proj = nn.Linear(c, hidden)
        self.act_proj = nn.Linear(3, hidden)
        self.sp_pos = nn.Parameter(torch.randn(1, 1, n, hidden) * 0.02)
        self.tp_pos = nn.Parameter(torch.randn(1, T, 1, hidden) * 0.02)
        self.blocks = nn.ModuleList([E.STBlock(hidden, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, c)

    def _mask(self, Tn, device):
        fi = torch.arange(Tn, device=device).repeat_interleave(self.n)  # frame index per token
        return fi[None, :] <= fi[:, None]  # block-causal, True=keep

    def forward(self, tokens, action):  # (B,T,N,C),(B,T,3) -> next-frame tokens (B,T,N,C)
        B, Tn, Nn, _ = tokens.shape
        h = self.in_proj(tokens) + self.sp_pos + self.tp_pos[:, :Tn]
        h = rearrange(h, "b t n d -> b (t n) d")
        c = self.act_proj(action)[:, :, None, :].expand(B, Tn, Nn, -1)
        c = rearrange(c, "b t n d -> b (t n) d")
        mask = self._mask(Tn, tokens.device)
        for blk in self.blocks:
            h = blk(h, c, mask)
        delta = rearrange(self.out(self.norm(h)), "b (t n) c -> b t n c", t=Tn)
        return tokens + delta  # residual: predict change from the current frame's tokens


class SpatialJEPA(nn.Module):
    def __init__(self, c=C, name="spatial_jepa"):
        super().__init__()
        self.name = name
        self.enc = ViTSpatialEncoder(c=c)
        self.pred = STPredictor(c=c)
        self.dec = ViTSpatialDecoder(c=c)
        self.sigreg = E.SIGReg()

    def encode(self, frames):  # (B,T,1,28,28) -> (B,T,16,C)
        B = frames.size(0)
        z = self.enc(rearrange(frames.float(), "b t o h w -> (b t) o h w"))
        return rearrange(z, "(b t) n c -> b t n c", b=B)

    def decode(self, tokens):  # (B,T,16,C) -> (B,T,1,28,28) probs
        B = tokens.size(0)
        logits = self.dec(rearrange(tokens, "b t n c -> (b t) n c"))
        return rearrange(torch.sigmoid(logits), "(b t) o h w -> b t o h w", b=B)

    def forward_train(self, inp, actions, tgt):
        z_in = self.encode(inp)
        z_tgt = self.encode(tgt).detach()
        z_pred = self.pred(z_in, actions)
        loss_mse = F.mse_loss(z_pred, z_tgt)
        loss_sig = self.sigreg(rearrange(z_in, "b t n c -> t b (n c)"))
        logits = self.dec(rearrange(z_in.detach(), "b t n c -> (b t) n c"))  # detached probe
        inp_f = rearrange(inp.float(), "b t o h w -> (b t) o h w")
        loss_rec = F.binary_cross_entropy_with_logits(logits, inp_f)
        loss = loss_mse + SIG_W * loss_sig + REC_W * loss_rec
        return loss, {"mse": loss_mse.item(), "sig": loss_sig.item(), "rec": loss_rec.item()}

    @torch.no_grad()
    def eval_ar(self, inp, actions, tgt):
        z_in = self.encode(inp)
        z_tgt = self.encode(tgt)
        tvar = z_tgt.var().item() + 1e-8
        z_tf = self.pred(z_in, actions)
        # honest AR: seed tokens only
        z_seq = z_in[:, :1]
        for t in range(T):
            pred = self.pred(z_seq, actions[:, :t + 1])[:, -1:]
            z_seq = torch.cat([z_seq, pred], dim=1)
        z_ar = z_seq[:, 1:]
        rec_enc, rec_tf, rec_ar = self.decode(z_in), self.decode(z_tf), self.decode(z_ar)
        mse_tf = ((z_tf - z_tgt) ** 2).mean((-1, -2)).mean(0)
        mse_ar = ((z_ar - z_tgt) ** 2).mean((-1, -2)).mean(0)
        met = {
            "nmse_tf": mse_tf.mean().item() / tvar,
            "nmse_ar": mse_ar.mean().item() / tvar,
            "mse_ar_t": (mse_ar / tvar).cpu().tolist(),
            "px_enc": F.mse_loss(rec_enc, inp.float()).item(),
            "px_tf": F.mse_loss(rec_tf, tgt.float()).item(),
            "px_ar": F.mse_loss(rec_ar, tgt.float()).item(),
            "z_std": z_in.std().item(),
        }
        return met, (inp, tgt, rec_enc, rec_tf, rec_ar)


def plot_rollout(name, pack, n=2):
    inp, tgt, rec_enc, rec_tf, rec_ar = [p.cpu() for p in pack]
    rows, labels = 5, ["inp", "dec(z)", "tgt", "dec(TF)", "dec(AR)"]
    srcs = [inp, rec_enc, tgt, rec_tf, rec_ar]
    fig, ax = plt.subplots(rows * n, T, figsize=(T * 1.4, rows * n * 1.4))
    for b in range(n):
        for r, (lab, src) in enumerate(zip(labels, srcs)):
            for t in range(T):
                a = ax[rows * b + r, t]; a.imshow(src[b, t, 0], cmap="gray", vmin=0, vmax=1)
                a.set_xticks([]); a.set_yticks([])
            ax[rows * b + r, 0].set_ylabel(lab, fontsize=11)
    plt.suptitle(f"{name}: inp / dec(z) / tgt / dec(TF) / dec(AR-honest)", fontsize=12)
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}_rollout.png", dpi=120); plt.close(fig)


def plot_compare(results, curves):
    names = list(results)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for n in names:
        ax[0].plot(range(1, T + 1), results[n]["mse_ar_t"], marker="o", label=n)
    ax[0].set_xlabel("rollout step"); ax[0].set_ylabel("AR latent nMSE")
    ax[0].set_title("Per-step AR latent nMSE (honest)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    x = range(len(names)); w = 0.35
    ax[1].bar([i - w / 2 for i in x], [results[n]["px_tf"] for n in names], w, label="px TF")
    ax[1].bar([i + w / 2 for i in x], [results[n]["px_ar"] for n in names], w, label="px AR")
    ax[1].set_xticks(list(x)); ax[1].set_xticklabels(names, rotation=15, fontsize=8)
    ax[1].set_ylabel("pixel MSE"); ax[1].set_title("Pixel recon (AR = honest)"); ax[1].legend()
    for n in names:
        c = curves[n]; ax[2].plot([s for s, _ in c], [v for _, v in c], label=n)
    ax[2].set_xlabel("step"); ax[2].set_ylabel("train latent MSE"); ax[2].set_yscale("log")
    ax[2].set_title("Training"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/compare.png", dpi=100); plt.close(fig)


def main(steps=STEPS):
    vb = E.val_batch()
    models = {
        "flat_baseline": E.MemModel("adaln", "flat_baseline"),  # current arch, for reference
        "spatial_jepa": SpatialJEPA("spatial_jepa"),
    }
    results, curves = {}, {}
    for name, m in models.items():
        print(f"-- {name}  ({sum(p.numel() for p in m.parameters())/1e6:.2f}M params) --")
        met, curve, pack = E.train_model(m, steps=steps, val_imgs=vb)
        results[name], curves[name] = met, curve
        plot_rollout(name, pack)
    plot_compare(results, curves)
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved plots + results to", OUT)
    return results


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else STEPS)
