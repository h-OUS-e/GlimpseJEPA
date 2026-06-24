"""Controlled experiments to diagnose the memory-latent regression.

Holds data + action RNG identical across configs so differences are purely
architectural. Trains each config for a fixed budget, then reports latent MSE
(teacher-forced and full AR rollout) plus pixel recon, and saves figures.

Run:  "C:/Users/Ous/miniconda3/envs/ML/python.exe" exp_memory.py
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from glimpse import rollout
from jepa import JEPA
from ml_layers import ARPredictor, ActionEncoder, ImageEncoder, Decoder, MLP_Projector, MemoryPredictor

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = "exp_out"
os.makedirs(OUT, exist_ok=True)

# ---- fixed experiment hyperparams ----
T = 10
H = W = 28
Z_IMG = 36
Z_ACT = 3
SCALE_S, TRANS_S = 0.2, 0.1
LR = 4e-4
STEPS = 800
SEED_DATA = 1234
SEED_TRAIN = 0
LAMBD_SIG = 0.09
LAMBD_REC = 0.1


_TRAIN_DS = None
_VAL_DS = None


def _datasets():
    global _TRAIN_DS, _VAL_DS
    if _TRAIN_DS is None:
        _TRAIN_DS = torchvision.datasets.MNIST(root="./dataset", train=True, download=True, transform=transforms.ToTensor())
        _VAL_DS = torchvision.datasets.MNIST(root="./dataset", train=False, download=True, transform=transforms.ToTensor())
    return _TRAIN_DS, _VAL_DS


def make_train_loader(seed=SEED_DATA):
    """Fresh loader with its own seeded generator -> identical batch order per call,
    independent of any RNG consumed elsewhere (fixes the stateful-generator confound)."""
    train_ds, _ = _datasets()
    g = torch.Generator().manual_seed(seed)
    return DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True, generator=g)


def make_loaders():
    train_loader = make_train_loader(SEED_DATA)
    _, val_ds = _datasets()
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=True)
    return train_loader, val_loader


def build_model(z_mem, enc_norm=False):
    """z_mem=0 -> no-memory baseline; else memory predictor with that dim.
    enc_norm=True puts a non-affine LayerNorm on the encoder output (via JEPA.projector)
    to pin the latent scale to ~unit (the broken-SigReg workaround under test)."""
    enc = ImageEncoder(H * W, 512, Z_IMG, depth=3)
    ae = ActionEncoder(3, emb_dim=Z_ACT)
    action_dim = Z_ACT + (z_mem if z_mem else 0)
    pr = ARPredictor(num_frames=T, depth=4, heads=4, mlp_dim=512, input_dim=Z_IMG,
                     hidden_dim=512, output_dim=512, action_dim=action_dim)
    pj = MLP_Projector(input_dim=512, output_dim=Z_IMG, hidden_dim=256, norm_fn=torch.nn.BatchNorm1d)
    dc = Decoder(z_dim=Z_IMG, hidden_dim=512, h=H, w=W, depth=2)
    mp = MemoryPredictor(Z_IMG, z_mem, hidden_dim=256, depth=2, heads=4) if z_mem else None
    projector = nn.LayerNorm(Z_IMG, elementwise_affine=False) if enc_norm else None
    return JEPA(enc, pr, ae, decoder=dc, projector=projector, projector_pred=pj, memory_predictor=mp).to(DEV)


@torch.no_grad()
def evaluate(model, val_batch):
    model.eval()
    imgs = val_batch
    seed, actions, inp, tgt = rollout(imgs, T, SCALE_S, TRANS_S, device=DEV)
    z_tgt, _ = model.encode(tgt)
    tgt_var = z_tgt.var().item() + 1e-8  # for scale-invariant normalization

    zp_tf, z_img, _ = model(inp, actions, ar_steps=0)
    zp_ar, _, _ = model(inp, actions, ar_steps=T)

    mse_tf_t = ((zp_tf - z_tgt) ** 2).mean(dim=-1).mean(dim=0)   # (T,)
    mse_ar_t = ((zp_ar - z_tgt) ** 2).mean(dim=-1).mean(dim=0)   # (T,)

    # decodes: encoder recon, teacher-forced pred, AR pred
    rec_enc = model.decode(z_img)
    rec_tf = model.decode(zp_tf)
    rec_ar = model.decode(zp_ar)
    px_enc = F.mse_loss(rec_enc, inp.float()).item()
    px_tf = F.mse_loss(rec_tf, tgt.float()).item()
    px_ar = F.mse_loss(rec_ar, tgt.float()).item()

    # collapse + magnitude diagnostics
    zf = z_img.reshape(-1, z_img.size(-1))
    z_std = zf.std(0).mean().item()
    a, b = zf[:50], zf[50:100]
    cos = F.cosine_similarity(a, b).mean().item()
    act_std = actions.std().item()
    mem_std = None
    if model.memory_predictor is not None:
        zm = model.predict_memory(z_img)
        mem_std = zm.std().item()

    return {
        "mse_tf": mse_tf_t.mean().item(),
        "mse_ar": mse_ar_t.mean().item(),
        "nmse_tf": mse_tf_t.mean().item() / tgt_var,   # scale-invariant
        "nmse_ar": mse_ar_t.mean().item() / tgt_var,
        "tgt_var": tgt_var,
        "mse_tf_t": mse_tf_t.cpu().tolist(),
        "mse_ar_t": mse_ar_t.cpu().tolist(),
        "px_enc": px_enc,
        "px_tf": px_tf,
        "px_ar": px_ar,
        "z_std": z_std,
        "cos": cos,
        "act_std": act_std,
        "mem_std": mem_std,
    }, (inp, tgt, rec_enc, rec_tf, rec_ar)


def _row(axes, r, imgs, b, label):
    for t in range(T):
        axes[r, t].imshow(imgs[b, t, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[r, t].set_xticks([]); axes[r, t].set_yticks([])
    axes[r, 0].set_ylabel(label, fontsize=7)


def save_decode_grid(name, pack, n=3):
    inp, tgt, rec_enc, rec_tf, rec_ar = pack
    rows = 5
    fig, axes = plt.subplots(rows * n, T, figsize=(T * 0.8, rows * n * 0.8))
    for b in range(n):
        base = rows * b
        _row(axes, base + 0, inp, b, f"t{b}\ninp")
        _row(axes, base + 1, rec_enc, b, "dec(z)")
        _row(axes, base + 2, tgt, b, "tgt")
        _row(axes, base + 3, rec_tf, b, "dec(TF)")
        _row(axes, base + 4, rec_ar, b, "dec(AR)")
    plt.suptitle(f"{name}: inp / dec(z_enc) | tgt / dec(pred_TF) / dec(pred_AR)", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}_decode.png", dpi=90)
    plt.close(fig)


def train_one(cfg, train_loader, val_imgs, seed=SEED_TRAIN):
    name, z_mem = cfg["name"], cfg["z_mem"]
    lambd_sig = cfg.get("lambd_sig", LAMBD_SIG)
    # fresh loader seeded identically per config so data order is not affected by run order
    train_loader = make_train_loader(SEED_DATA)
    torch.manual_seed(seed)
    model = build_model(z_mem, enc_norm=cfg.get("enc_norm", False))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    ar_curriculum = cfg.get("ar_curriculum", False)
    torch.manual_seed(seed)  # rollout/sigreg RNG
    model.train()
    it = iter(train_loader)
    curve = []
    t0 = time.time()
    for step in range(STEPS):
        try:
            imgs, _ = next(it)
        except StopIteration:
            it = iter(train_loader); imgs, _ = next(it)
        imgs = imgs.to(DEV).squeeze(1)
        with torch.no_grad():
            seed, actions, inp, tgt = rollout(imgs, T, SCALE_S, TRANS_S, device=DEV)
        # ar curriculum: ramp ar_steps 0->T over training so the predictor learns on its own rollouts
        ar = min(T, 1 + (step * T) // STEPS) if ar_curriculum else 0
        zp, z_img, _ = model(inp, actions, ar_steps=ar)
        z_tgt, _ = model.encode(tgt)
        loss_mse = model.mse(zp, z_tgt, mean=False)
        loss_sig = model.sigreg_loss(z_img)
        loss_rec = model.recon_loss(z_img.detach(), inp)
        loss = loss_mse + lambd_sig * loss_sig + LAMBD_REC * loss_rec
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0:
            curve.append((step, loss_mse.item()))
    dt = time.time() - t0

    metrics, pack = evaluate(model, val_imgs)
    metrics["train_s"] = dt
    metrics["n_params"] = sum(p.numel() for p in model.parameters())
    save_decode_grid(name, pack)
    print(f"[{name:12s}] nmse_tf {metrics['nmse_tf']:.4f}  nmse_ar {metrics['nmse_ar']:.4f}  | "
          f"px_enc {metrics['px_enc']:.4f}  px_tf {metrics['px_tf']:.4f}  px_ar {metrics['px_ar']:.4f}  | "
          f"z_std {metrics['z_std']:.2f}  cos {metrics['cos']:.2f}  ({dt:.0f}s)")
    return name, metrics, curve, model


def probe_decoder(model, train_loader, steps=3000, lr=1e-3):
    """Freeze encoder, train a fresh strong decoder on (z_target -> target_image).
    Tests whether the encoder latents are decodable at all (info content),
    isolating that from the weak joint decoder."""
    for p in model.parameters():
        p.requires_grad_(False)
    dec = Decoder(z_dim=Z_IMG, hidden_dim=512, h=H, w=W, depth=3).to(DEV)
    opt = torch.optim.Adam(dec.parameters(), lr=lr)
    it = iter(train_loader)
    dec.train()
    for step in range(steps):
        try:
            imgs, _ = next(it)
        except StopIteration:
            it = iter(train_loader); imgs, _ = next(it)
        imgs = imgs.to(DEV).squeeze(1)
        with torch.no_grad():
            _, _, _, tgt = rollout(imgs, T, SCALE_S, TRANS_S, device=DEV)
            z_tgt, _ = model.encode(tgt)
        rec = dec(z_tgt)
        loss = F.mse_loss(rec, tgt.float())
        opt.zero_grad(); loss.backward(); opt.step()
    return dec, loss.item()


def main(configs, tag="round1"):
    train_loader, val_loader = make_loaders()
    val_imgs = next(iter(val_loader))[0].to(DEV).squeeze(1)  # fixed val batch

    results = {}
    curves = {}
    for cfg in configs:
        name, metrics, curve, _ = train_one(cfg, train_loader, val_imgs)
        results[name] = metrics
        curves[name] = curve

    # per-step AR MSE comparison
    plt.figure(figsize=(8, 4))
    for name, m in results.items():
        plt.plot(range(T), m["mse_ar_t"], marker="o", label=f"{name} (AR mean {m['mse_ar']:.3f})")
    plt.xlabel("rollout step t"); plt.ylabel("latent MSE")
    plt.title(f"Per-step AR latent MSE ({tag})"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/{tag}_mse_ar_per_step.png", dpi=100); plt.close()

    # training mse curves
    plt.figure(figsize=(8, 4))
    for name, c in curves.items():
        xs = [s for s, _ in c]; ys = [v for _, v in c]
        plt.plot(xs, ys, label=name)
    plt.xlabel("step"); plt.ylabel("train mse"); plt.yscale("log")
    plt.title(f"Train MSE ({tag})"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/{tag}_train_mse.png", dpi=100); plt.close()

    with open(f"{OUT}/{tag}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "round2":
        STEPS = 1500
        configs = [
            {"name": "nomem", "z_mem": 0},
            {"name": "mem36", "z_mem": 36},
        ]
        main(configs, tag="round2")
    elif len(sys.argv) > 1 and sys.argv[1] == "round3":
        # Decoder-probe diagnostic: is the latent decodable with a strong frozen-encoder decoder?
        STEPS = 1500
        train_loader, val_loader = make_loaders()
        val_imgs = next(iter(val_loader))[0].to(DEV).squeeze(1)
        for cfg in [{"name": "nomem", "z_mem": 0}, {"name": "mem36", "z_mem": 36}]:
            name, metrics, _, model = train_one(cfg, train_loader, val_imgs)
            dec, dloss = probe_decoder(model, train_loader, steps=3000)
            # decode true target latents and AR-pred latents with the strong probe decoder
            model.eval()
            with torch.no_grad():
                _, actions, inp, tgt = rollout(val_imgs, T, SCALE_S, TRANS_S, device=DEV)
                z_tgt, _ = model.encode(tgt)
                zp_ar, z_img, _ = model(inp, actions, ar_steps=T)
                rec_tgt = dec(z_tgt)      # decode the TRUE latent (upper bound for this encoder)
                rec_ar = dec(zp_ar)       # decode AR predictions with the strong decoder
                rec_enc = dec(z_img)
                px_tgt = F.mse_loss(rec_tgt, tgt.float()).item()
                px_ar = F.mse_loss(rec_ar, tgt.float()).item()
            save_decode_grid(f"{name}_probe", (inp, tgt, rec_enc, rec_tgt, rec_ar))
            print(f"[{name:12s} PROBE] decoder_train_mse {dloss:.4f}  "
                  f"px(true latent) {px_tgt:.4f}  px(AR pred) {px_ar:.4f}")
    elif len(sys.argv) > 1 and sys.argv[1] == "round7":
        # How to lower AR MSE: test AR curriculum and longer training (all enc_norm + mem36)
        _, val_loader = make_loaders()
        val_imgs = next(iter(val_loader))[0].to(DEV).squeeze(1)
        import statistics as st
        agg = {}
        runs = [
            ("base_1200", {"z_mem": 36, "enc_norm": True}, 1200),
            ("curr_1200", {"z_mem": 36, "enc_norm": True, "ar_curriculum": True}, 1200),
            ("curr_2400", {"z_mem": 36, "enc_norm": True, "ar_curriculum": True}, 2400),
        ]
        for name, base_cfg, steps in runs:
            STEPS = steps
            rows = []
            for s in [0, 1]:
                cfg = dict(base_cfg); cfg["name"] = name
                _, m, _, _ = train_one(cfg, None, val_imgs, seed=s)
                rows.append(m)
                print(f"    {name} seed {s}: nmse_tf {m['nmse_tf']:.3f} nmse_ar {m['nmse_ar']:.3f} "
                      f"px_ar {m['px_ar']:.4f}")
            na = [r["nmse_ar"] for r in rows]; nt = [r["nmse_tf"] for r in rows]; px = [r["px_ar"] for r in rows]
            agg[name] = {"nmse_tf": nt, "nmse_ar": na, "px_ar": px}
            print(f"[{name:10s}] nmse_tf {st.mean(nt):.3f}  nmse_ar {st.mean(na):.3f}+-{st.pstdev(na):.3f}  px_ar {st.mean(px):.4f}")
        with open(f"{OUT}/round7_lower_mse.json", "w") as f:
            json.dump(agg, f, indent=2)
    elif len(sys.argv) > 1 and sys.argv[1] == "round6":
        # H7: normalize encoder output (non-affine LayerNorm) to pin latent scale.
        STEPS = 1200
        _, val_loader = make_loaders()
        val_imgs = next(iter(val_loader))[0].to(DEV).squeeze(1)
        seeds = [0, 1, 2]
        import statistics as st
        agg = {}
        cfgs = [
            {"name": "nomem_LN", "z_mem": 0, "enc_norm": True},
            {"name": "mem36_LN", "z_mem": 36, "enc_norm": True},
        ]
        last_pack = {}
        for cfg in cfgs:
            rows = []
            for s in seeds:
                nm, m, _, model = train_one(cfg, None, val_imgs, seed=s)
                rows.append(m)
                print(f"    {cfg['name']} seed {s}: nmse_ar {m['nmse_ar']:.3f}  z_std {m['z_std']:.2f}  "
                      f"px_tf {m['px_tf']:.4f}  px_ar {m['px_ar']:.4f}")
            na = [r["nmse_ar"] for r in rows]; zs = [r["z_std"] for r in rows]
            pt = [r["px_tf"] for r in rows]; px = [r["px_ar"] for r in rows]
            agg[cfg["name"]] = {"nmse_ar": na, "z_std": zs, "px_tf": pt, "px_ar": px}
            print(f"[{cfg['name']:10s}] nmse_ar {st.mean(na):.3f}+-{st.pstdev(na):.3f}  "
                  f"z_std {st.mean(zs):.2f}+-{st.pstdev(zs):.2f}  px_tf {st.mean(pt):.4f}  px_ar {st.mean(px):.4f}")
        with open(f"{OUT}/round6_encnorm.json", "w") as f:
            json.dump(agg, f, indent=2)
    elif len(sys.argv) > 1 and sys.argv[1] == "round5":
        # H6: does stronger SigReg pin z_std ~1 and remove instability?
        STEPS = 1000
        _, val_loader = make_loaders()
        val_imgs = next(iter(val_loader))[0].to(DEV).squeeze(1)
        seeds = [0, 1]
        import statistics as st
        agg = {}
        for zmem in [0, 36]:
            for ls in [0.09, 1.0, 5.0]:
                name = f"{'nomem' if zmem==0 else 'mem36'}_ls{ls}"
                rows = []
                for s in seeds:
                    _, m, _, _ = train_one({"name": name, "z_mem": zmem, "lambd_sig": ls}, None, val_imgs, seed=s)
                    rows.append(m)
                na = [r["nmse_ar"] for r in rows]; zs = [r["z_std"] for r in rows]; px = [r["px_ar"] for r in rows]
                agg[name] = {"nmse_ar": na, "z_std": zs, "px_ar": px}
                print(f"[{name:14s}] nmse_ar {st.mean(na):.3f}+-{st.pstdev(na):.3f}  "
                      f"z_std {st.mean(zs):.2f}+-{st.pstdev(zs):.2f}  px_ar {st.mean(px):.4f}")
        with open(f"{OUT}/round5_sigreg.json", "w") as f:
            json.dump(agg, f, indent=2)
    elif len(sys.argv) > 1 and sys.argv[1] == "round4":
        # H5: is memory unstable across seeds? (data order now fixed per config)
        STEPS = 1200
        _, val_loader = make_loaders()
        val_imgs = next(iter(val_loader))[0].to(DEV).squeeze(1)
        seeds = [0, 1, 2]
        agg = {}
        for cfg in [{"name": "nomem", "z_mem": 0}, {"name": "mem36", "z_mem": 36}]:
            rows = []
            for s in seeds:
                _, metrics, _, _ = train_one(cfg, None, val_imgs, seed=s)
                rows.append(metrics)
                print(f"    seed {s}: nmse_ar {metrics['nmse_ar']:.3f}  z_std {metrics['z_std']:.2f}  px_ar {metrics['px_ar']:.4f}")
            import statistics as st
            agg[cfg["name"]] = {
                "nmse_ar": [r["nmse_ar"] for r in rows],
                "z_std": [r["z_std"] for r in rows],
                "px_ar": [r["px_ar"] for r in rows],
            }
            na = agg[cfg["name"]]["nmse_ar"]; zs = agg[cfg["name"]]["z_std"]
            print(f"[{cfg['name']}] nmse_ar mean {st.mean(na):.3f} std {st.pstdev(na):.3f} | "
                  f"z_std mean {st.mean(zs):.2f} std {st.pstdev(zs):.2f}")
        with open(f"{OUT}/round4_variance.json", "w") as f:
            json.dump(agg, f, indent=2)
    else:
        configs = [
            {"name": "nomem", "z_mem": 0},
            {"name": "mem36", "z_mem": 36},
            {"name": "mem8", "z_mem": 8},
            {"name": "mem2", "z_mem": 2},
        ]
        main(configs, tag="round1")
