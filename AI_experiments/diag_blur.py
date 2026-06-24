"""Why are decodes blurry? Isolate the three sources, holding others fixed.

Exp 1 (decoder/objective): freeze the trained baseline encoder, decode the TRUE latent with
       {MLP,Conv} x {MSE,BCE}. Same latent -> any sharpness difference is the decoder+objective.
Exp 2 (bottleneck):        train plain autoencoders (BCE+Conv) with latent {vec36, vec128, spatial7x7x8}.
       No prediction, no SigReg -> any difference is latent capacity/structure.

Run: "C:/Users/Ous/miniconda3/envs/ML/python.exe" diag_blur.py
"""
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import exp_ideas as E
from glimpse import rollout
from ml_layers import ImageEncoder

DEV, T, OUT = E.DEV, E.T, E.OUT
S_DEC, S_AE = 1500, 1500


class MLPDec(nn.Module):
    """logits out (no final sigmoid) so BCEWithLogits works; sigmoid applied for MSE/eval."""
    def __init__(self, z, hid=512, depth=3):
        super().__init__()
        L = [nn.Linear(z, hid), nn.ReLU(True)]
        for _ in range(depth - 1): L += [nn.Linear(hid, hid), nn.ReLU(True)]
        L += [nn.Linear(hid, 28 * 28)]
        self.net = nn.Sequential(*L)
    def forward(self, z):
        return self.net(z.reshape(-1, z.size(-1))).view(*z.shape[:-1], 1, 28, 28)


class ConvDec(nn.Module):
    def __init__(self, z, c=32):
        super().__init__()
        self.fc = nn.Linear(z, c * 7 * 7); self.c = c
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c, c, 3, 2, 1, output_padding=1), nn.GELU(),  # 7->14
            nn.ConvTranspose2d(c, 1, 3, 2, 1, output_padding=1))             # 14->28 logits
    def forward(self, z):
        lead = z.shape[:-1]
        x = self.fc(z.reshape(-1, z.size(-1))).view(-1, self.c, 7, 7)
        return self.net(x).view(*lead, 1, 28, 28)


def loss_fn(logits, tgt, kind):
    return F.binary_cross_entropy_with_logits(logits, tgt) if kind == "bce" else F.mse_loss(torch.sigmoid(logits), tgt)


def train(params, step_fn, steps):
    opt = torch.optim.Adam(params, lr=1e-3)
    loader = E.make_train_loader(); it = iter(loader)
    for _ in range(steps):
        try: imgs, _ = next(it)
        except StopIteration: it = iter(loader); imgs, _ = next(it)
        imgs = imgs.to(DEV).squeeze(1)
        with torch.no_grad():
            _, _, _, tgt = rollout(imgs, T, E.SCALE_S, E.TRANS_S, device=DEV)
        loss = step_fn(tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def grid(rows, labels, title, path, n=8):
    fig, ax = plt.subplots(len(rows), n, figsize=(n * 0.9, len(rows) * 0.9))
    for r, (lab, im) in enumerate(zip(labels, rows)):
        for j in range(n):
            a = ax[r, j]; a.imshow(im[j, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            a.set_xticks([]); a.set_yticks([])
        ax[r, 0].set_ylabel(lab, fontsize=8)
    plt.suptitle(title, fontsize=10); plt.tight_layout()
    plt.savefig(path, dpi=100); plt.close(fig)


def main():
    vb = E.val_batch()
    base = E.MemModel("adaln", "baseline")
    E.train_model(base, steps=1200, val_imgs=vb)
    enc = base.jepa
    for p in enc.parameters(): p.requires_grad_(False)

    with torch.no_grad():
        _, _, _, tgt = rollout(vb, T, E.SCALE_S, E.TRANS_S, device=DEV)
        z_true, _ = enc.encode(tgt)
    ev_tgt = tgt[:, 5]            # mid-rollout frame across the batch
    ev_z = z_true[:, 5]

    # ---- Exp 1: decoder x objective, latent FIXED (the baseline 36-d true latent) ----
    print("\n== Exp 1: decode TRUE 36-d latent, vary decoder+objective ==")
    rows, labels, res1 = [ev_tgt], ["target"], {}
    for arch in ["MLP", "Conv"]:
        for kind in ["mse", "bce"]:
            dec = (MLPDec(36) if arch == "MLP" else ConvDec(36)).to(DEV)
            train(dec.parameters(), lambda t: loss_fn(dec(enc.encode(t)[0]), t.float(), kind), S_DEC)
            with torch.no_grad():
                rec = torch.sigmoid(dec(ev_z)); px = F.mse_loss(rec, ev_tgt.float()).item()
            res1[f"{arch}-{kind}"] = px; rows.append(rec); labels.append(f"{arch}-{kind}\n{px:.4f}")
            print(f"  {arch}-{kind:3s}  px(MSE) {px:.4f}")
    grid(rows, labels, "Exp1: same 36-d TRUE latent, different decoder/objective", f"{OUT}/diag_blur_obj.png")

    # ---- Exp 2: latent capacity/structure (plain autoencoder, BCE) ----
    print("\n== Exp 2: plain autoencoder (BCE), vary latent capacity ==")
    rows2, labels2, res2 = [ev_tgt], ["target"], {}
    for name, z in [("vec36", 36), ("vec128", 128), ("spatial7x7x8", "sp")]:
        if z == "sp":
            en = E.SpatialEncoder(c=8).to(DEV); de = E.SpatialDecoder(c=8).to(DEV)  # de already sigmoids
            def step(t):
                B = t.size(0); x = rearrange(t.float(), "b k c h w -> (b k) c h w")
                rec = de(en(x)).view(B, T, 1, 28, 28)
                return F.binary_cross_entropy(rec.clamp(1e-6, 1 - 1e-6), t.float())
            train(list(en.parameters()) + list(de.parameters()), step, S_AE)
            with torch.no_grad():
                rec = de(en(ev_tgt)).view(-1, 1, 28, 28)
        else:
            en = ImageEncoder(784, 512, z, depth=3).to(DEV); de = ConvDec(z).to(DEV)
            def encv(t):
                B = t.size(0); x = rearrange(t.float(), "b k c h w -> (b k) c h w")
                return rearrange(en(x), "(b k) d -> b k d", b=B)
            train(list(en.parameters()) + list(de.parameters()),
                  lambda t: F.binary_cross_entropy_with_logits(de(encv(t)), t.float()), S_AE)
            with torch.no_grad():
                rec = torch.sigmoid(de(en(ev_tgt)))  # ImageEncoder flattens internally
        px = F.mse_loss(rec, ev_tgt.float()).item()
        res2[name] = px; rows2.append(rec); labels2.append(f"{name}\n{px:.4f}")
        print(f"  {name:14s} px(MSE) {px:.4f}")
    grid(rows2, labels2, "Exp2: autoencoder recon (BCE), different latent capacity", f"{OUT}/diag_blur_cap.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(list(res1), list(res1.values())); ax[0].set_title("Exp1: decoder/objective (36-d true latent)")
    ax[0].set_ylabel("pixel MSE"); ax[0].tick_params(axis="x", labelrotation=20)
    ax[1].bar(list(res2), list(res2.values())); ax[1].set_title("Exp2: latent capacity (BCE AE)")
    ax[1].set_ylabel("pixel MSE"); ax[1].tick_params(axis="x", labelrotation=20)
    plt.tight_layout(); plt.savefig(f"{OUT}/diag_blur_summary.png", dpi=100); plt.close(fig)
    print("\nsaved diag_blur_*.png to", OUT)


if __name__ == "__main__":
    main()
