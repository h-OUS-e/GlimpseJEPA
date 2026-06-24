"""ViT spatial-latent autoencoder: size sweep + conditional-decoder test.

Goal: smallest latent (few tokens x small C) and fewest params that still reconstructs sharply.
Reference points: old flat latent = 36 floats; conv-c8 spatial = 392 floats; full pixels = 784 floats.

Self-contained (only imports exp_ideas for data/rollout + ml_layers for the predictor param count).
Does NOT modify jepa.py / ml_layers.py.

Phase 1 (sweep):  "...python.exe" exp_vit_sweep.py sweep
Phase 2 (cond):   "...python.exe" exp_vit_sweep.py cond p7 4    # patch, C of the small latent to augment
"""
import sys, json
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import exp_ideas as E
from glimpse import rollout
from ml_layers import ARPredictor

DEV, T, OUT = E.DEV, E.T, E.OUT
STEPS = 1200
PATCH2TOK = {14: 4, 7: 16, 4: 49}  # patch size -> tokens per image (img=28)


class ViTBlock(nn.Module):
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


class CrossBlock(nn.Module):
    """Decoder block: self-attn over output tokens + cross-attn to conditioning tokens."""
    def __init__(self, dim, heads=4, mlp=4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.sa = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.nq = nn.LayerNorm(dim); self.ca = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp), nn.GELU(), nn.Linear(dim * mlp, dim))

    def forward(self, x, cond):
        h = self.n1(x); x = x + self.sa(h, h, h, need_weights=False)[0]
        q = self.nq(x); x = x + self.ca(q, cond, cond, need_weights=False)[0]
        x = x + self.mlp(self.n2(x))
        return x


class ViTAE(nn.Module):
    """ViT autoencoder with independently sized encoder/decoder.

    Latent = (B, N, C) grid of patch tokens -> latent_floats = N*C.
    cond_mode: 'none' | 'add' (prev image patch-embedded, added at matching positions)
               | 'xattn' (decoder cross-attends to prev image patches -> can warp/translate).
    """
    COND_PATCH = 4  # prev image is patchified at patch-4 (49 tokens) for rich content

    def __init__(self, patch=7, c=8, enc_hidden=64, enc_depth=2, dec_hidden=64, dec_depth=2,
                 heads=4, img=28, cond_mode="none"):
        super().__init__()
        self.p, self.c, self.mode = patch, c, cond_mode
        self.np = img // patch
        n = self.np * self.np
        pdim = patch * patch
        # encoder
        self.embed = nn.Linear(pdim, enc_hidden)
        self.enc_pos = nn.Parameter(torch.randn(1, n, enc_hidden) * 0.02)
        self.enc_blocks = nn.ModuleList([ViTBlock(enc_hidden, heads) for _ in range(enc_depth)])
        self.to_latent = nn.Linear(enc_hidden, c)
        # decoder
        self.from_latent = nn.Linear(c, dec_hidden)
        self.dec_pos = nn.Parameter(torch.randn(1, n, dec_hidden) * 0.02)
        if cond_mode == "add":
            self.cond_embed = nn.Linear(pdim, dec_hidden)
            self.dec_blocks = nn.ModuleList([ViTBlock(dec_hidden, heads) for _ in range(dec_depth)])
        elif cond_mode == "xattn":
            ncp = (img // self.COND_PATCH) ** 2
            self.cond_embed = nn.Linear(self.COND_PATCH ** 2, dec_hidden)
            self.cond_pos = nn.Parameter(torch.randn(1, ncp, dec_hidden) * 0.02)
            self.dec_blocks = nn.ModuleList([CrossBlock(dec_hidden, heads) for _ in range(dec_depth)])
        else:
            self.dec_blocks = nn.ModuleList([ViTBlock(dec_hidden, heads) for _ in range(dec_depth)])
        self.to_pix = nn.Linear(dec_hidden, pdim)

    def _patchify(self, img, p):
        return rearrange(img, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=p, p2=p)

    def encode(self, img):
        x = self.embed(self._patchify(img, self.p)) + self.enc_pos
        for blk in self.enc_blocks:
            x = blk(x)
        return self.to_latent(x)

    def decode(self, z, cond=None):
        x = self.from_latent(z) + self.dec_pos
        if self.mode == "add" and cond is not None:
            x = x + self.cond_embed(self._patchify(cond, self.p))
        if self.mode == "xattn":
            ct = self.cond_embed(self._patchify(cond, self.COND_PATCH)) + self.cond_pos
            for blk in self.dec_blocks:
                x = blk(x, ct)
        else:
            for blk in self.dec_blocks:
                x = blk(x)
        x = self.to_pix(x)
        return rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                         h=self.np, w=self.np, p1=self.p, p2=self.p, c=1)

    def recon(self, img, cond=None):
        return torch.sigmoid(self.decode(self.encode(img), cond))

    def loss(self, img, cond=None):
        return F.binary_cross_entropy_with_logits(self.decode(self.encode(img), cond), img)

    @property
    def latent_floats(self):
        return self.np * self.np * self.c

    def part_params(self):
        enc = sum(p.numel() for n, p in self.named_parameters() if n.startswith(("embed", "enc_", "to_latent")))
        dec = sum(p.numel() for p in self.parameters()) - enc
        return enc, dec


class ViTAE_CLS(nn.Module):
    """CLS-token autoencoder (the paper's choice): latent = a SINGLE d-dim vector (the CLS token).
    Decoder broadcasts that one vector to all output positions + learned pos, then ViT-decodes.
    latent_floats = d (one vector), vs the spatial AE's N*C grid.
    """
    def __init__(self, d=128, patch=4, enc_hidden=64, enc_depth=2, dec_hidden=64, dec_depth=2,
                 heads=4, img=28):
        super().__init__()
        self.p, self.d = patch, d
        self.np = img // patch
        n = self.np * self.np
        pdim = patch * patch
        # encoder + CLS
        self.embed = nn.Linear(pdim, enc_hidden)
        self.enc_pos = nn.Parameter(torch.randn(1, n, enc_hidden) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, enc_hidden) * 0.02)
        self.enc_blocks = nn.ModuleList([ViTBlock(enc_hidden, heads) for _ in range(enc_depth)])
        self.to_latent = nn.Linear(enc_hidden, d)
        # decoder: broadcast single vector -> N tokens
        self.from_latent = nn.Linear(d, dec_hidden)
        self.dec_pos = nn.Parameter(torch.randn(1, n, dec_hidden) * 0.02)
        self.dec_blocks = nn.ModuleList([ViTBlock(dec_hidden, heads) for _ in range(dec_depth)])
        self.to_pix = nn.Linear(dec_hidden, pdim)

    def encode(self, img):  # -> (B, d)  the CLS vector only
        x = self.embed(rearrange(img, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=self.p, p2=self.p)) + self.enc_pos
        x = torch.cat([self.cls.expand(x.size(0), 1, -1), x], dim=1)
        for blk in self.enc_blocks:
            x = blk(x)
        return self.to_latent(x[:, 0])

    def decode(self, z, cond=None):  # z: (B, d)
        t = self.from_latent(z)[:, None, :].expand(-1, self.np * self.np, -1) + self.dec_pos
        for blk in self.dec_blocks:
            t = blk(t)
        x = self.to_pix(t)
        return rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                         h=self.np, w=self.np, p1=self.p, p2=self.p, c=1)

    def recon(self, img, cond=None):
        return torch.sigmoid(self.decode(self.encode(img)))

    def loss(self, img, cond=None):
        return F.binary_cross_entropy_with_logits(self.decode(self.encode(img)), img)

    @property
    def latent_floats(self):
        return self.d

    def part_params(self):
        enc = sum(p.numel() for n, p in self.named_parameters() if n.startswith(("embed", "enc_", "cls", "to_latent")))
        dec = sum(p.numel() for p in self.parameters()) - enc
        return enc, dec


def predictor_params():
    pr = ARPredictor(num_frames=T, depth=4, heads=4, mlp_dim=512, input_dim=E.Z_IMG,
                     hidden_dim=512, output_dim=512, action_dim=E.Z_ACT + E.Z_MEM)
    return sum(p.numel() for p in pr.parameters())


def train(model, steps=STEPS, cond=False):
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = E.make_train_loader(); it = iter(loader)
    model.train()
    for _ in range(steps):
        try: imgs, _ = next(it)
        except StopIteration: it = iter(loader); imgs, _ = next(it)
        imgs = imgs.to(DEV).squeeze(1)
        with torch.no_grad():
            _, _, inp, tgt = rollout(imgs, T, E.SCALE_S, E.TRANS_S, device=DEV)
            tg = rearrange(tgt.float(), "b t c h w -> (b t) c h w")
            cd = rearrange(inp.float(), "b t c h w -> (b t) c h w") if cond else None
        loss = model.loss(tg, cd)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def eval_frames(cond=False):
    vb = E.val_batch()
    with torch.no_grad():
        _, _, inp, tgt = rollout(vb, T, E.SCALE_S, E.TRANS_S, device=DEV)
    return tgt[:, 5].float(), (inp[:, 5].float() if cond else None)


def run_sweep():
    pp = predictor_params()
    print(f"reference: predictor params {pp/1e6:.2f}M | old latent 36 floats | conv-c8 392 floats | pixels 784\n")
    ev, _ = eval_frames()
    # name: (patch, c, enc_h, enc_d, dec_h, dec_d)
    cfgs = [
        # latent SIZE at small fixed capacity (h64 d2)
        ("t4_c8",     14, 8, 64, 2, 64, 2),
        ("t16_c4",     7, 4, 64, 2, 64, 2),
        ("t16_c8",     7, 8, 64, 2, 64, 2),
        ("t16_c16",    7, 16, 64, 2, 64, 2),
        ("t49_c2",     4, 2, 64, 2, 64, 2),
        ("t49_c8",     4, 8, 64, 2, 64, 2),
        # CAPACITY at fixed latent ~128 (t16_c8)
        ("t16_c8_h32", 7, 8, 32, 2, 32, 2),
        ("t16_c8_h128", 7, 8, 128, 2, 128, 2),
        ("t16_c8_d4",  7, 8, 64, 4, 64, 4),
        # ASYMMETRIC: tiny latent, heavier decoder
        ("t4_c8_bigdec",  14, 8, 64, 2, 128, 4),
        ("t16_c4_bigdec",  7, 4, 32, 2, 128, 4),
    ]
    rows, labels, res = [ev], ["target"], {}
    for name, p, c, eh, ed, dh, dd in cfgs:
        m = ViTAE(patch=p, c=c, enc_hidden=eh, enc_depth=ed, dec_hidden=dh, dec_depth=dd)
        train(m)
        with torch.no_grad():
            rec = m.recon(ev); px = F.mse_loss(rec, ev).item()
        enc_p, dec_p = m.part_params()
        lf = m.latent_floats
        res[name] = {"px": px, "latent_floats": lf, "enc_k": enc_p / 1e3, "dec_k": dec_p / 1e3,
                     "tot_k": (enc_p + dec_p) / 1e3}
        rows.append(rec); labels.append(f"{name}\n{lf}f {px:.4f}")
        print(f"  {name:16s} latent {lf:4d}f  px {px:.4f}  enc {enc_p/1e3:5.0f}k dec {dec_p/1e3:5.0f}k tot {(enc_p+dec_p)/1e3:5.0f}k")

    # image grid
    n = 8
    fig, ax = plt.subplots(len(rows), n, figsize=(n * 0.85, len(rows) * 0.85))
    for r, (lab, im) in enumerate(zip(labels, rows)):
        for j in range(n):
            a = ax[r, j]; a.imshow(im[j, 0].cpu(), cmap="gray", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
        ax[r, 0].set_ylabel(lab, fontsize=7)
    plt.suptitle("ViT spatial latent sweep (autoencoder recon, BCE)", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{OUT}/vit_sweep_grid.png", dpi=95); plt.close(fig)

    # scatter: latent floats vs recon, size~params
    fig, axx = plt.subplots(figsize=(8, 5))
    for name, r in res.items():
        axx.scatter(r["latent_floats"], r["px"], s=max(20, r["tot_k"]), alpha=0.6)
        axx.annotate(f"{name}\n{r['tot_k']:.0f}k", (r["latent_floats"], r["px"]), fontsize=6,
                     xytext=(3, 3), textcoords="offset points")
    for x, lab in [(36, "old=36"), (392, "conv-c8=392"), (784, "pixels=784")]:
        axx.axvline(x, ls="--", c="gray", alpha=0.5); axx.text(x, axx.get_ylim()[1], lab, fontsize=7, rotation=90, va="top")
    axx.set_xscale("log"); axx.set_yscale("log")
    axx.set_xlabel("latent floats (N tokens x C)"); axx.set_ylabel("recon pixel MSE")
    axx.set_title("Smaller latent <-> sharper? (marker size ~ total params)"); axx.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/vit_sweep_scatter.png", dpi=100); plt.close(fig)
    with open(f"{OUT}/vit_sweep.json", "w") as f:
        json.dump({"predictor_params": pp, "configs": res}, f, indent=2)
    print("\nsaved vit_sweep_grid.png / vit_sweep_scatter.png / vit_sweep.json")


def run_cond(patch, c):
    """Latent-only vs latent+previous-image decoder, at a small latent. Tests whether handing the
    decoder the clean previous frame lets a tiny latent stay sharp."""
    ev, cond = eval_frames(cond=True)
    n_tok = PATCH2TOK[patch]; lf = n_tok * c
    print(f"conditional test at latent {lf}f (patch {patch}, C {c})")
    out_rows, labels = [ev, cond], ["target", "prev (cond)"]
    res = {}
    for name, mode in [("latent_only", "none"), ("prev_img_add", "add"), ("prev_img_xattn", "xattn")]:
        use_cond = mode != "none"
        m = ViTAE(patch=patch, c=c, enc_hidden=64, enc_depth=2, dec_hidden=64, dec_depth=2, cond_mode=mode)
        train(m, cond=use_cond)
        with torch.no_grad():
            rec = m.recon(ev, cond if use_cond else None); px = F.mse_loss(rec, ev).item()
        res[name] = px; out_rows.append(rec); labels.append(f"{name}\n{px:.4f}")
        print(f"  {name:16s} px {px:.4f}")
    n = 8
    fig, ax = plt.subplots(len(out_rows), n, figsize=(n * 0.9, len(out_rows) * 0.9))
    for r, (lab, im) in enumerate(zip(labels, out_rows)):
        for j in range(n):
            a = ax[r, j]; a.imshow(im[j, 0].cpu(), cmap="gray", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
        ax[r, 0].set_ylabel(lab, fontsize=8)
    plt.suptitle(f"Conditional decoder at {lf}-float latent: does the prev image rescue sharpness?", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{OUT}/vit_cond_grid.png", dpi=100); plt.close(fig)
    print("saved vit_cond_grid.png")


def run_cls_sweep():
    """CLS-token latent sweep: vary the single-vector dim d, at small (matched) and strong (paper-like)
    decoder capacity. Overlay against the spatial sweep on the same latent-floats axis."""
    pp = predictor_params()
    print(f"reference: predictor {pp/1e6:.2f}M | spatial sweet spot t16_c8 = 128f @ 0.0026\n")
    ev, _ = eval_frames()
    cfgs = [  # (name, d, enc_h, enc_d, dec_h, dec_d, cap)
        ("cls16",  16, 64, 2, 64, 2, "small"),
        ("cls32",  32, 64, 2, 64, 2, "small"),
        ("cls64",  64, 64, 2, 64, 2, "small"),
        ("cls128", 128, 64, 2, 64, 2, "small"),
        ("cls256", 256, 64, 2, 64, 2, "small"),
        ("cls32_big",  32, 128, 4, 128, 4, "strong"),
        ("cls128_big", 128, 128, 4, 128, 4, "strong"),
        ("cls192_big", 192, 128, 4, 128, 4, "strong"),  # paper uses 192-d CLS
    ]
    rows, labels, res = [ev], ["target"], {}
    for name, d, eh, ed, dh, dd, cap in cfgs:
        m = ViTAE_CLS(d=d, patch=4, enc_hidden=eh, enc_depth=ed, dec_hidden=dh, dec_depth=dd)
        train(m)
        with torch.no_grad():
            rec = m.recon(ev); px = F.mse_loss(rec, ev).item()
        ep, dp = m.part_params()
        res[name] = {"px": px, "latent_floats": d, "cap": cap, "tot_k": (ep + dp) / 1e3}
        rows.append(rec); labels.append(f"{name}\n{d}f {px:.4f}")
        print(f"  {name:12s} d {d:4d}  px {px:.4f}  {cap:6s}  tot {(ep+dp)/1e3:5.0f}k")

    # image grid
    n = 8
    fig, ax = plt.subplots(len(rows), n, figsize=(n * 0.85, len(rows) * 0.85))
    for r, (lab, im) in enumerate(zip(labels, rows)):
        for j in range(n):
            a = ax[r, j]; a.imshow(im[j, 0].cpu(), cmap="gray", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
        ax[r, 0].set_ylabel(lab, fontsize=7)
    plt.suptitle("CLS-token latent sweep (autoencoder recon, BCE)", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{OUT}/cls_sweep_grid.png", dpi=95); plt.close(fig)

    # overlay vs spatial
    spatial = {}
    try:
        spatial = json.load(open(f"{OUT}/vit_sweep.json"))["configs"]
    except Exception:
        pass
    fig, axx = plt.subplots(figsize=(8.5, 5.5))
    sm = [(r["latent_floats"], r["px"]) for n_, r in res.items() if r["cap"] == "small"]
    bg = [(r["latent_floats"], r["px"]) for n_, r in res.items() if r["cap"] == "strong"]
    if sm: axx.plot(*zip(*sorted(sm)), "o-", c="tab:red", label="CLS (small dec h64d2)")
    if bg: axx.plot(*zip(*sorted(bg)), "s-", c="tab:orange", label="CLS (strong dec h128d4)")
    if spatial:
        sp = sorted((r["latent_floats"], r["px"]) for r in spatial.values())
        axx.plot(*zip(*sp), "^--", c="tab:blue", alpha=0.7, label="spatial grid (h64d2)")
    axx.axhline(0.011, ls=":", c="gray"); axx.text(axx.get_xlim()[1], 0.011, " old flat-36", fontsize=7, va="bottom")
    axx.set_xscale("log"); axx.set_yscale("log")
    axx.set_xlabel("latent floats"); axx.set_ylabel("recon pixel MSE")
    axx.set_title("CLS vector vs spatial grid: recon at equal latent budget"); axx.legend(fontsize=8); axx.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/cls_vs_spatial.png", dpi=100); plt.close(fig)
    with open(f"{OUT}/cls_sweep.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nsaved cls_sweep_grid.png / cls_vs_spatial.png / cls_sweep.json")


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if phase == "sweep":
        run_sweep()
    elif phase == "cls":
        run_cls_sweep()
    elif phase == "cond":
        patch = int(sys.argv[2][1:]) if len(sys.argv) > 2 else 7  # e.g. "p7" -> 7
        c = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        run_cond(patch, c)
