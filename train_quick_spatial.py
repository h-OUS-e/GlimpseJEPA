"""
Quick training for the spatial-latent JEPA (jepa.SpatialJEPA): ViT token grid + spatiotemporal
residual predictor + conv-head decoder, with a DMT post-finetune to cut AR rollout drift.

Counterpart to train_quick.py (which trains the flat-vector JEPA). Same shape: globals -> data ->
model -> wandb -> train loop with viz -> history plots, then a DMT phase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import wandb

from einops import rearrange
from glimpse import rollout
from jepa import SpatialJEPA
from ml_layers import ViTSpatialDecoder
from vis_utils import plot_glimpse_frames


#================================================
#                GLOBAL VARS
#================================================
scale_sensitivity = 0.2
translation_sensitivity = 0.1
batch_size = 64
T_max = 10
lr = 4e-4
epochs = 20
viz_every = 1

# spatial-latent params (16 tokens x C = 128-float latent at patch 7)
patch = 7
z_dim_token = 8
img_hw = 28
sig_w = 0.05   # gentle per-token SigReg (recon is a detached probe -> rec weight only scales decoder)
rec_w = 1.0

# DMT post-finetune (on-policy imitation of the frozen encoder trajectory)
dmt_epochs = 4
dmt_lr = 1e-4

T_ood = 2 * T_max  # OOD rollout length (trained on T_max, tested at 2x)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#================================================
#                DATASET PREP
#================================================
train_ds = torchvision.datasets.MNIST(root='./dataset', train=True,  download=True, transform=transforms.ToTensor())
val_ds   = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True)


#================================================
#                MODEL AND OPTIMIZER
#================================================
# max_frames must cover the OOD horizon (predictor has temporal position embeddings)
model = SpatialJEPA(c=z_dim_token, patch=patch, img=img_hw, max_frames=T_ood,
                    sig_w=sig_w, rec_w=rec_w).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
print(f"SpatialJEPA params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")


#================================================
#                WANDB INIT
#================================================
wandb.init(
    project="glimpse-jepa",
    config={
        "model": "SpatialJEPA",
        "scale_sensitivity": scale_sensitivity,
        "translation_sensitivity": translation_sensitivity,
        "batch_size": batch_size,
        "T_max": T_max,
        "lr": lr,
        "epochs": epochs,
        "patch": patch,
        "z_dim_token": z_dim_token,
        "n_tokens": (img_hw // patch) ** 2,
        "sig_w": sig_w,
        "rec_w": rec_w,
        "dmt_epochs": dmt_epochs,
        "dmt_lr": dmt_lr,
    },
)
global_step = 0
hist = {"step": [], "train_mse": [],
        "epoch": [], "train_loss": [], "val_nmse_tf": [], "val_nmse_ar": [], "val_px_ar": []}


#================================================
#                EVAL + VIZ HELPERS
#================================================
@torch.no_grad()
def evaluate(loader):
    """Teacher-forced + honest-AR latent nMSE (scale-invariant) and AR pixel recon over the loader."""
    model.eval()
    nt = na = pa = nb = 0.0
    last = None
    for imgs, _ in loader:
        imgs = imgs.to(device).squeeze(1)
        _, actions, inp, tgt = rollout(imgs, T_max, scale_sensitivity, translation_sensitivity, device=device)
        z_in, z_tgt = model.encode(inp), model.encode(tgt)
        tvar = z_tgt.var().item() + 1e-8
        z_tf = model.predict(z_in, actions)
        z_ar = model.ar_rollout(z_in[:, :1], actions)
        nt += ((z_tf - z_tgt) ** 2).mean().item() / tvar
        na += ((z_ar - z_tgt) ** 2).mean().item() / tvar
        pa += F.mse_loss(model.decode(z_ar), tgt.float()).item()
        nb += 1
        last = (inp, tgt, z_in, z_tf, z_ar)
    return nt / nb, na / nb, pa / nb, last


def viz_grid(pack, n=3, title="rollout"):
    """One labeled grid -> rows: inp / dec(z_enc) / tgt / dec(TF) / dec(AR-honest); cols = time.
       inp:    glimpse fed to the model        dec(z): autoencode of encoder latent (no prediction)
       tgt:    true next frame                 dec(TF): decode of teacher-forced 1-step prediction
       dec(AR): decode of honest AR rollout (only the seed is real)."""
    inp, tgt, z_in, z_tf, z_ar = pack
    srcs = [inp.cpu(), model.decode(z_in).detach().cpu(), tgt.cpu(),
            model.decode(z_tf).detach().cpu(), model.decode(z_ar).detach().cpu()]
    labels = ["inp", "dec(z)", "tgt", "dec(TF)", "dec(AR)"]
    Tn = inp.size(1)
    fig, ax = plt.subplots(5 * n, Tn, figsize=(Tn * 0.8, 5 * n * 0.8))
    for b in range(n):
        for r, (lab, src) in enumerate(zip(labels, srcs)):
            for t in range(Tn):
                a = ax[5 * b + r, t]; a.imshow(src[b, t, 0], cmap="gray", vmin=0, vmax=1)
                a.set_xticks([]); a.set_yticks([])
            ax[5 * b + r, 0].set_ylabel(lab, fontsize=7)
    plt.suptitle(title, fontsize=9); plt.tight_layout()
    return fig


def latent_viz(z_in, n=2):
    """Latent token-norm map: |token| reshaped to the spatial grid (e.g. 4x4), one row/sample over time.
       Shows WHERE the latent puts activation -> a quick read on what the spatial tokens encode."""
    g = int(round(z_in.size(2) ** 0.5))
    norm = rearrange(z_in.norm(dim=-1), "b t (h w) -> b t h w", h=g)
    Tn = z_in.size(1)
    fig, ax = plt.subplots(n, Tn, figsize=(Tn * 0.7, n * 0.7), squeeze=False)
    for b in range(n):
        for t in range(Tn):
            a = ax[b, t]; a.imshow(norm[b, t].detach().cpu(), cmap="viridis")
            a.set_xticks([]); a.set_yticks([])
        ax[b, 0].set_ylabel(f"s{b}", fontsize=7)
    plt.suptitle("latent token-norm map (grid per frame)", fontsize=9); plt.tight_layout()
    return fig


def log_viz(pack, tag):
    """Labeled rollout grid + latent token-norm map as wandb images."""
    fig_r, fig_l = viz_grid(pack, title=tag), latent_viz(pack[2])
    out = {f"{tag}/rollout": wandb.Image(fig_r), f"{tag}/latent": wandb.Image(fig_l)}
    plt.close(fig_r); plt.close(fig_l)
    return out


@torch.no_grad()
def collapse_stats(z_in):
    zf = z_in.reshape(-1, z_in.size(-1))
    a, b = zf[:50], zf[50:100]
    return zf.std(0).mean().item(), F.cosine_similarity(a, b).mean().item()


#================================================
#                TRAIN LOOP (teacher forced)
#================================================
for epoch in tqdm(range(epochs), desc="epochs"):
    model.train()
    train_loss_sum, train_batches = 0.0, 0
    for imgs, _ in tqdm(train_loader, desc=f"train {epoch}", leave=False):
        imgs = imgs.to(device).squeeze(1)
        with torch.no_grad():
            _, actions, inp, tgt = rollout(imgs, T_max, scale_sensitivity, translation_sensitivity, device=device)
        loss, parts = model.loss(inp, actions, tgt)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        train_loss_sum += loss.item(); train_batches += 1
        hist["step"].append(global_step); hist["train_mse"].append(parts["mse"])
        wandb.log({"train/loss": loss.item(), "train/loss_mse": parts["mse"],
                   "train/loss_sigreg": parts["sig"], "train/loss_recon": parts["rec"],
                   "epoch": epoch}, step=global_step)
        global_step += 1

    # ---- validation ----
    nmse_tf, nmse_ar, px_ar, pack = evaluate(val_loader)
    z_std, cos = collapse_stats(pack[2])
    train_avg = train_loss_sum / max(train_batches, 1)
    print(f"epoch {epoch:4d}  train {train_avg:.4f}  nmse_tf {nmse_tf:.4f}  nmse_ar {nmse_ar:.4f}  "
          f"px_ar {px_ar:.4f}  z_std {z_std:.2f}  cos {cos:.2f}")
    hist["epoch"].append(epoch); hist["train_loss"].append(train_avg)
    hist["val_nmse_tf"].append(nmse_tf); hist["val_nmse_ar"].append(nmse_ar); hist["val_px_ar"].append(px_ar)

    epoch_log = {"epoch": epoch, "train/avg_loss": train_avg, "val/nmse_tf": nmse_tf,
                 "val/nmse_ar": nmse_ar, "val/px_ar": px_ar,
                 "collapse/z_std": z_std, "collapse/cos_sim": cos}
    if epoch % viz_every == 0:
        epoch_log.update(log_viz(pack, "viz"))
        # OOD rollout: trained on T_max, roll out to T_ood
        with torch.no_grad():
            _, a_o, inp_o, tgt_o = rollout(imgs, T_ood, scale_sensitivity, translation_sensitivity, device=device)
            z_ar_o = model.ar_rollout(model.encode(inp_o)[:, :1], a_o)
            epoch_log["viz/ood_rollout"] = wandb.Image(
                plot_glimpse_frames(model.decode(z_ar_o).squeeze(2).detach().cpu()))
        plt.close("all")
    wandb.log(epoch_log, step=global_step)


#================================================
#                DMT POST-FINETUNE
#================================================
# Freeze encoder + decoder, unroll the predictor on its OWN predictions, regress to the frozen
# encoder trajectory. Cuts AR drift without touching the (preserved) teacher-forced encoder.
print("\n--- DMT finetune ---")
params = model.freeze_for_dmt()
dmt_opt = torch.optim.AdamW(params, lr=dmt_lr)
for epoch in tqdm(range(dmt_epochs), desc="dmt"):
    model.train()
    for imgs, _ in tqdm(train_loader, desc=f"dmt {epoch}", leave=False):
        imgs = imgs.to(device).squeeze(1)
        with torch.no_grad():
            _, actions, inp, tgt = rollout(imgs, T_max, scale_sensitivity, translation_sensitivity, device=device)
        loss = model.dmt_loss(inp, actions, tgt)
        dmt_opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); dmt_opt.step()
        wandb.log({"dmt/loss": loss.item()}, step=global_step); global_step += 1

nmse_tf, nmse_ar, px_ar, pack = evaluate(val_loader)
print(f"after DMT  nmse_tf {nmse_tf:.4f}  nmse_ar {nmse_ar:.4f}  px_ar {px_ar:.4f}")
wandb.log({"val/nmse_tf_dmt": nmse_tf, "val/nmse_ar_dmt": nmse_ar, "val/px_ar_dmt": px_ar,
           **log_viz(pack, "viz_dmt")}, step=global_step)


#================================================
#            TRAINING HISTORY PLOTS
#================================================
fig_hist, ax = plt.subplots(figsize=(8, 4))
ax.plot(hist["step"], hist["train_mse"], lw=0.8, alpha=0.8)
ax.set_xlabel("step"); ax.set_ylabel("train latent MSE"); ax.set_yscale("log")
ax.set_title("Training history: per-step latent MSE"); ax.grid(alpha=0.3)
plt.tight_layout(); fig_hist.savefig("train_history_spatial.png", dpi=110)

fig_mse, ax = plt.subplots(figsize=(8, 4))
ax.plot(hist["epoch"], hist["val_nmse_tf"], marker="o", label="val nMSE (teacher-forced)")
ax.plot(hist["epoch"], hist["val_nmse_ar"], marker="o", label="val nMSE (autoregressive)")
ax.plot(hist["epoch"], hist["val_px_ar"], marker=".", ls="--", alpha=0.6, label="val px AR")
ax.set_xlabel("epoch"); ax.set_ylabel("latent nMSE / px"); ax.set_yscale("log")
ax.set_title("Spatial JEPA: latent nMSE over training"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); fig_mse.savefig("latent_mse_history_spatial.png", dpi=110)

wandb.log({"history/train_mse": wandb.Image(fig_hist),
           "history/latent_mse": wandb.Image(fig_mse)}, step=global_step)
plt.close(fig_hist); plt.close(fig_mse)


#================================================
#        POST-TRAINING PROBES (on the DMT model)
#================================================
def _next(it_holder):
    """Infinite train batch iterator helper."""
    try:
        imgs, _ = next(it_holder[0])
    except (StopIteration, IndexError):
        it_holder[:] = [iter(train_loader)]; imgs, _ = next(it_holder[0])
    return imgs.to(device).squeeze(1)


def _grid_target_pred(target, pred, n=6, title=""):
    """Rows alternate target / decoded-prediction; columns = rollout steps."""
    Tn = target.size(1)
    fig, ax = plt.subplots(2 * n, Tn, figsize=(Tn * 0.8, 2 * n * 0.8))
    for b in range(n):
        for t in range(Tn):
            ax[2 * b, t].imshow(target[b, t, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            ax[2 * b + 1, t].imshow(pred[b, t, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            for r in (2 * b, 2 * b + 1):
                ax[r, t].set_xticks([]); ax[r, t].set_yticks([])
        ax[2 * b, 0].set_ylabel(f"t{b}\ntgt", fontsize=7); ax[2 * b + 1, 0].set_ylabel("pred", fontsize=7)
    plt.suptitle(title, fontsize=9); plt.tight_layout()
    return fig


for p in model.parameters():  # freeze the trained model for all probes below
    p.requires_grad_(False)
model.eval()

# --- 1. Decoder probe: train a STRONG fresh decoder on the frozen DMT model (z_target -> image),
#        then decode the AR predictions. Tests how decodable the latents really are. ---
print("\n--- decoder probe ---")
probe_dec = ViTSpatialDecoder(patch=patch, c=z_dim_token, hidden=128, depth=3, img=img_hw).to(device)
pd_opt = torch.optim.Adam(probe_dec.parameters(), lr=1e-3)
hold = [iter(train_loader)]
for step in range(4000):
    imgs = _next(hold)
    with torch.no_grad():
        _, _, _, tgt = rollout(imgs, T_max, scale_sensitivity, translation_sensitivity, device=device)
        zt = model.encode(tgt)
    logits = probe_dec(rearrange(zt, "b t n c -> (b t) n c"))
    loss = F.binary_cross_entropy_with_logits(logits, rearrange(tgt.float(), "b t o h w -> (b t) o h w"))
    pd_opt.zero_grad(); loss.backward(); pd_opt.step()
probe_dec.eval()
with torch.no_grad():
    imgs = next(iter(val_loader))[0].to(device).squeeze(1)
    _, a, inp, tgt = rollout(imgs, T_max, scale_sensitivity, translation_sensitivity, device=device)
    z_ar = model.ar_rollout(model.encode(inp)[:, :1], a); B = z_ar.size(0)
    rec = rearrange(torch.sigmoid(probe_dec(rearrange(z_ar, "b t n c -> (b t) n c"))),
                    "(b t) o h w -> b t o h w", b=B)
    px = F.mse_loss(rec, tgt.float()).item()
print(f"decoder probe: px(strong dec, AR pred) {px:.4f}")
fig = _grid_target_pred(tgt, rec, n=6, title=f"Decoder probe (strong dec): target vs decode(AR pred)  px {px:.4f}")
fig.savefig("probe_decoder_spatial.png", dpi=100); wandb.log({"probe/decoder": wandb.Image(fig)}, step=global_step); plt.close(fig)

# --- 2. Inverse-dynamics probe: recover the action a_t from (z_t, z_{t+1}). High fidelity => the
#        latents carry action-relevant info. Tokens are flattened to a vector for the probe. ---
print("--- inverse-dynamics probe ---")
Z, A = [], []
hold = [iter(train_loader)]
with torch.no_grad():
    for _ in range(20):
        imgs = _next(hold)
        _, actions, inp, tgt = rollout(imgs, T_max, scale_sensitivity, translation_sensitivity, device=device)
        zi = model.encode(inp).flatten(2); zt = model.encode(tgt).flatten(2)  # (B,T,N*C)
        Z.append(torch.cat([zi, zt], dim=-1)); A.append(actions)
Z = torch.cat(Z); A = torch.cat(A); Bn, Tn = Z.shape[:2]
X = Z.reshape(-1, Z.size(-1)); y = A.reshape(-1, 3)
perm = torch.randperm(X.size(0)); ntr = int(0.8 * X.size(0))
Xtr, Xva, ytr, yva = X[perm[:ntr]], X[perm[ntr:]], y[perm[:ntr]], y[perm[ntr:]]
probe = nn.Sequential(nn.Linear(X.size(-1), 64), nn.ReLU(), nn.Linear(64, 3)).to(device)
po = torch.optim.Adam(probe.parameters(), lr=1e-3)
for step in range(2000):
    l = F.mse_loss(probe(Xtr), ytr); po.zero_grad(); l.backward(); po.step()
with torch.no_grad():
    val = F.mse_loss(probe(Xva), yva).item()
    pred_a = probe(X).reshape(Bn, Tn, 3)
print(f"inverse-dynamics probe: val MSE {val:.4f}")
names = ["log_scale", "dx", "dy"]; ntj = 3
fig, axes = plt.subplots(ntj, 3, figsize=(12, 2.5 * ntj), sharex=True)
for b in range(ntj):
    for d in range(3):
        axes[b, d].plot(range(Tn), A[b, :, d].cpu(), label="target", lw=2)
        axes[b, d].plot(range(Tn), pred_a[b, :, d].cpu(), label="pred", lw=2, ls="--")
        if b == 0: axes[b, d].set_title(names[d])
        if d == 0: axes[b, d].set_ylabel(f"traj {b}")
        axes[b, d].axhline(0, color="gray", lw=0.5)
    axes[b, 0].legend(fontsize=8)
plt.suptitle(f"Inverse-dynamics probe (val MSE {val:.4f})"); plt.tight_layout()
fig.savefig("probe_invdyn_spatial.png", dpi=100); wandb.log({"probe/inverse_dynamics": wandb.Image(fig)}, step=global_step); plt.close(fig)

# --- 3. Surprise / violation-of-expectation: per-step AR latent error under perturbations. A good
#        world model should spike at the perturbation step (teleport) or stay high after (swap/invert). ---
print("--- surprise eval ---")
@torch.no_grad()
def measure_surprise(inp, actions, tgt):
    z_pred = model.ar_rollout(model.encode(inp)[:, :1], actions)
    return ((z_pred - model.encode(tgt)) ** 2).mean(dim=(-1, -2))  # (B,T)

@torch.no_grad()
def rollout_with_perturbation(imgs, t_perturb, mode):
    _, actions, inp, tgt = rollout(imgs, T_max, scale_sensitivity, translation_sensitivity, device=device)
    if mode == "none":
        return actions, inp, tgt
    B = imgs.shape[0]
    if mode == "teleport":          # physical violation: target jumps to a random glimpse state
        from glimpse import Glimpse
        rs = torch.randn(B, 3, device=device)
        g = Glimpse(imgs, log_scale=rs[:, 0:1], x=rs[:, 1:2], y=rs[:, 2:3], T_max=1)
        tgt[:, t_perturb, 0] = g.transform(t=-1)
    elif mode == "swap_digit":      # identity violation: underlying digit swaps from t_perturb on
        other = next(iter(train_loader))[0].squeeze(1)[:B].to(device)
        _, _, _, ot = rollout(other, T_max, scale_sensitivity, translation_sensitivity, device=device)
        tgt[:, t_perturb:] = ot[:, t_perturb:]
    elif mode == "invert":          # visual violation: intensities invert from t_perturb on
        tgt[:, t_perturb:] = 1.0 - tgt[:, t_perturb:]
    return actions, inp, tgt

t_perturb = T_max // 2
modes, colors = ["none", "teleport", "swap_digit", "invert"], ["gray", "tab:red", "tab:orange", "tab:blue"]
curves = {m: [] for m in modes}
for _ in range(10):
    imgs = next(iter(val_loader))[0].squeeze(1).to(device)
    for m in modes:
        a, inp, tgt = rollout_with_perturbation(imgs, t_perturb, m)
        curves[m].append(measure_surprise(inp, a, tgt).cpu())
curves = {m: torch.cat(curves[m], 0) for m in modes}
fig = plt.figure(figsize=(8, 4))
for m, c in zip(modes, colors):
    s = curves[m]; mean, std = s.mean(0), s.std(0); t = torch.arange(s.size(-1))
    plt.plot(t, mean, label=m, color=c, lw=2); plt.fill_between(t, mean - std, mean + std, color=c, alpha=0.15)
plt.axvline(t_perturb, color="k", ls="--", lw=0.7, label="perturbation")
plt.xlabel("step t"); plt.ylabel("surprise (latent MSE)"); plt.title("Violation-of-expectation: per-step surprise")
plt.legend(fontsize=8); plt.tight_layout()
fig.savefig("probe_surprise_spatial.png", dpi=100); wandb.log({"probe/surprise": wandb.Image(fig)}, step=global_step); plt.close(fig)

wandb.finish()
