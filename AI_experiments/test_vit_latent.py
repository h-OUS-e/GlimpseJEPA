"""Does a ViT make as sharp a spatial latent as the conv one?

Pure reconstruction (BCE), same protocol as diag_blur Exp2, on glimpse frames.
Compares: conv 7x7x8 (reference) vs ViT patch-4 (49 tokens) at C=8 and C=16.
A ViT latent is attractive because the predictor is already a transformer -> shared token space.

Run: "C:/Users/Ous/miniconda3/envs/ML/python.exe" test_vit_latent.py
"""
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import exp_ideas as E
from glimpse import rollout

DEV, T, OUT = E.DEV, E.T, E.OUT
STEPS = 1500


class ViTBlock(nn.Module):
    """Pre-LN transformer block, full (bidirectional) attention."""
    def __init__(self, dim, heads=4, mlp=4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp), nn.GELU(), nn.Linear(dim * mlp, dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.n2(x))
        return x


class ViTAE(nn.Module):
    """ViT autoencoder. Latent = grid of patch tokens (B, N, C) -> a spatial latent."""
    def __init__(self, patch=4, c=8, hidden=128, depth=4, heads=4, img=28):
        super().__init__()
        self.p, self.c, self.np = patch, c, img // patch  # tokens per side
        n = self.np * self.np
        pdim = patch * patch
        # encoder
        self.embed = nn.Linear(pdim, hidden)
        self.enc_pos = nn.Parameter(torch.randn(1, n, hidden) * 0.02)
        self.enc_blocks = nn.ModuleList([ViTBlock(hidden, heads) for _ in range(depth)])
        self.to_latent = nn.Linear(hidden, c)
        # decoder
        self.from_latent = nn.Linear(c, hidden)
        self.dec_pos = nn.Parameter(torch.randn(1, n, hidden) * 0.02)
        self.dec_blocks = nn.ModuleList([ViTBlock(hidden, heads) for _ in range(depth)])
        self.to_pix = nn.Linear(hidden, pdim)

    def encode(self, img):  # (B,1,28,28) -> (B,N,C)
        x = rearrange(img, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=self.p, p2=self.p)
        x = self.embed(x) + self.enc_pos
        for blk in self.enc_blocks:
            x = blk(x)
        return self.to_latent(x)

    def decode(self, z):  # (B,N,C) -> logits (B,1,28,28)
        x = self.from_latent(z) + self.dec_pos
        for blk in self.dec_blocks:
            x = blk(x)
        x = self.to_pix(x)
        return rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                         h=self.np, w=self.np, p1=self.p, p2=self.p, c=1)

    def recon(self, img):
        return torch.sigmoid(self.decode(self.encode(img)))

    def loss(self, img):
        return F.binary_cross_entropy_with_logits(self.decode(self.encode(img)), img)


class ConvAE(nn.Module):
    def __init__(self, c=8):
        super().__init__()
        self.en, self.de = E.SpatialEncoder(c), E.SpatialDecoder(c)

    def recon(self, img):
        return self.de(self.en(img))

    def loss(self, img):
        return F.binary_cross_entropy(self.recon(img).clamp(1e-6, 1 - 1e-6), img)


def train(model, steps=STEPS):
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = E.make_train_loader(); it = iter(loader)
    model.train()
    for _ in range(steps):
        try: imgs, _ = next(it)
        except StopIteration: it = iter(loader); imgs, _ = next(it)
        imgs = imgs.to(DEV).squeeze(1)
        with torch.no_grad():
            _, _, _, tgt = rollout(imgs, T, E.SCALE_S, E.TRANS_S, device=DEV)
            frames = rearrange(tgt.float(), "b t c h w -> (b t) c h w")
        loss = model.loss(frames)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def main():
    vb = E.val_batch()
    with torch.no_grad():
        _, _, _, tgt = rollout(vb, T, E.SCALE_S, E.TRANS_S, device=DEV)
    ev = tgt[:, 5].float()  # mid-rollout frame across the batch

    cfgs = [
        ("conv_7x7x8", ConvAE(c=8)),
        ("vit_p4_c8", ViTAE(patch=4, c=8)),
        ("vit_p4_c16", ViTAE(patch=4, c=16)),
    ]
    rows, labels, res = [ev], ["target"], {}
    for name, m in cfgs:
        train(m)
        with torch.no_grad():
            rec = m.recon(ev)
            px = F.mse_loss(rec, ev).item()
        npar = sum(p.numel() for p in m.parameters())
        res[name] = px; rows.append(rec); labels.append(f"{name}\n{px:.4f}")
        print(f"  {name:14s} px(MSE) {px:.4f}  ({npar/1e3:.0f}k params)")

    # image grid
    n = 8
    fig, ax = plt.subplots(len(rows), n, figsize=(n * 0.9, len(rows) * 0.9))
    for r, (lab, im) in enumerate(zip(labels, rows)):
        for j in range(n):
            a = ax[r, j]; a.imshow(im[j, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            a.set_xticks([]); a.set_yticks([])
        ax[r, 0].set_ylabel(lab, fontsize=8)
    plt.suptitle("Spatial latent: conv vs ViT (autoencoder recon, BCE)", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{OUT}/test_vit_latent.png", dpi=100); plt.close(fig)
    print("\nsaved", f"{OUT}/test_vit_latent.png")


if __name__ == "__main__":
    main()
