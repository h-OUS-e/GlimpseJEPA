"""Decoupled-memory JEPA: separate "what to remember" from "how to update".

Motivation (arxiv 2606.06479): an RNN-style memory that only ever predicts the
SINGLE next state stays lazy. If a glimpse drifts into a black box the next state
is a black box, so the memory learns to hold nothing. Forcing the memory state
m_t to predict MANY future states z_{t+1..t+H} makes it compress useful history.

Four modules (everything here is self-contained; ml_layers.py / jepa.py untouched):
  E_m  memory encoder  : reuse MemoryPredictor      z_0..z_t           -> m_t
  D_m  memory decoder  : NEW (bottleneck)           m_t, a_{t..t+H-1}  -> z_{t+1..t+H}
  P    world predictor : reuse ARPredictor (content) z_t, a_t, m_t      -> z_{t+1}
  U    memory updater  : NEW                         m_t, z_{t+1}, a_t  -> m_{t+1}

D_m sees ONLY m_t + future actions (never z_{>t}) -> the bottleneck that forces a
rich m_t. The T x H many-future objective stays fully PARALLEL: unfold the future
actions/targets into per-source windows and fold the source-time axis into the
batch, so D_m runs as one batched causal pass.

Training is joint with stop-grad: z is detached BOTH as the D_m target AND at the
E_m input, so no memory-loss gradient reaches the encoder (equivalent to a frozen
encoder for the memory path, in a single run).

Configs contrasted:
  nomem         z_mem=0 baseline (no memory)
  mem36_single  memory feeds P but trained only on next-step (naive RNN memory)
  mem36_multi   adds D_m many-future objective + U  (the decoupled design)

Layout mirrors train_quick.py so runs are directly comparable: each config writes
into its OWN dated folder out/plots/<YY_MMDD>-decoupledMem-<name>/ with the SAME
filenames as train_quick.py, plus the memory-specific extras:
  train_history.png        per-step train latent MSE        (== train_quick.py)
  latent_mse_history.png   val TF vs AR latent MSE / epoch  (== train_quick.py)
  recon_history.png        val recon (enc/pred) / epoch     (== train_quick.py)
  decode.png               inp / dec(z) / tgt / dec(TF) / dec(AR) grid
  mem_horizon_mse.png      D_m error vs horizon offset k (multi only)
  per_step_ar_mse.png      per-step AR latent MSE along the rollout
A shared out/plots/<YY_MMDD>-decoupledMem-compare/ holds cross-config overlays
(per-step AR, val AR history) and results.json.

Budget matches train_quick.py: 20 epochs over full MNIST (~937 steps/epoch).

Run:  "C:/Users/Ous/miniconda3/envs/ML/python.exe" exp_decoupled_mem.py
      ...exp_decoupled_mem.py check        # fast unit checks (bottleneck + stop-grad)
      ...exp_decoupled_mem.py 5            # quick smoke run with EPOCHS=5
"""

import os
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from glimpse import rollout
from jepa import JEPA
from ml_layers import (
    ARPredictor, ActionEncoder, ImageEncoder, Decoder, MLP_Projector,
    MemoryPredictor, Transformer, ConditionalBlock,
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- hyperparams matched to train_quick.py for a direct frame of reference ----
T = 10
H = W = 28
Z_IMG = 36
Z_ACT = 3
SCALE_S, TRANS_S = 0.3, 0.25   # bumped from 0.2/0.1 so glimpses actually move when unrolling
LR = 4e-4
EPOCHS = 20            # same budget as train_quick.py
BATCH = 64
VAL_BATCHES = 50       # cap val for speed; means match full val within noise
SEED_DATA = 1234
SEED_TRAIN = 0
LAMBD_SIG = 0.09
LAMBD_REC = 0.1
MEM_HIDDEN = 256
HORIZON = T            # D_m predicts up to T steps ahead (masked where it runs off the end)
T_OOD = 2 * T          # extrapolation horizon for the DMT before/after viz (20 vs trained 10)
DMT_STEPS = 400        # DMT post-finetune iterations per config
DMT_LR = 1e-4

DATE_TAG = datetime.now().strftime("%y_%m%d")


def plot_dir_for(name):
    """Per-config dated folder, sibling to train_quick.py's out/plots/<date>-<tag>/."""
    d = os.path.join("out", "plots", f"{DATE_TAG}-decoupledMem-{name}")
    os.makedirs(d, exist_ok=True)
    return d


# ============================================================================
#  New modules (local; nothing added to the shared library files)
# ============================================================================

class MemoryDecoder(nn.Module):
    """Decode a horizon of future latents from a single memory vector.

    Bottleneck: the only trajectory signal is m_t (a prepended seed token); the
    rest is learned horizon queries modulated by the future action window. A
    causal transformer means query k attends to seed + actions a_t..a_{t+k-1}.
    """

    def __init__(self, z_dim_memory, z_dim_img, action_dim, horizon,
                 hidden_dim=MEM_HIDDEN, depth=2, heads=4, dim_head=64, mlp_dim=512):
        super().__init__()
        self.H = horizon
        self.seed_proj = nn.Linear(z_dim_memory, hidden_dim)
        self.seg_seed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos = nn.Parameter(torch.zeros(1, horizon, hidden_dim))  # learned horizon queries
        nn.init.normal_(self.seg_seed, std=0.02)
        nn.init.normal_(self.pos, std=0.02)
        # ConditionalBlock -> AdaLN action conditioning, same machinery as ARPredictor
        self.transformer = Transformer(
            hidden_dim, hidden_dim, z_dim_img, depth, heads, dim_head, mlp_dim,
            block_class=ConditionalBlock, action_dim=action_dim,
        )

    def forward(self, m_t, action_win):
        """
        Args:
            m_t: (N, Z_mem) memory vectors (N = B*T source steps)
            action_win: (N, H, A) future actions a_{t..t+H-1}
        Returns:
            (N, H, Z_img) predictions of z_{t+1..t+H}
        """
        N = m_t.size(0)
        seed = self.seed_proj(m_t).unsqueeze(1) + self.seg_seed       # (N, 1, hidden)
        queries = self.pos.expand(N, -1, -1)                          # (N, H, hidden)
        x = torch.cat([seed, queries], dim=1)                        # (N, H+1, hidden)
        # cond: zero for seed, then a_t..a_{t+H-1} aligned so query k sees a_t..a_{t+k-1}
        zero_a = action_win.new_zeros(N, 1, action_win.size(-1))
        cond = torch.cat([zero_a, action_win], dim=1)                # (N, H+1, A)
        out = self.transformer(x, cond)                              # causal SDPA inside
        return out[:, 1:]                                            # drop seed -> (N, H, Z_img)


class MemoryUpdater(nn.Module):
    """Predict m_{t+1} from (m_t, z_{t+1}, a_t). Only its rollout is sequential (eval)."""

    def __init__(self, z_dim_memory, z_dim_img, action_dim, hidden_dim=MEM_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim_memory + z_dim_img + action_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, z_dim_memory),
        )

    def forward(self, m_t, z_next, a_t):
        """m_t/(B,*,Z_mem), z_next/(B,*,Z_img), a_t/(B,*,A) -> (B,*,Z_mem)."""
        return self.net(torch.cat([m_t, z_next, a_t], dim=-1))


# ============================================================================
#  Model: thin JEPA subclass adding the decoupled-memory objectives
# ============================================================================

class DecoupledMemJEPA(JEPA):
    """JEPA + memory decoder/updater. Base class is untouched; new behavior lives here."""

    def __init__(self, *args, memory_decoder=None, memory_updater=None, horizon=HORIZON, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_decoder = memory_decoder
        self.memory_updater = memory_updater
        self.horizon = horizon

    def memory_forward(self, image, action):
        """Encode, build memory (z detached), predict next state via P (content mode)."""
        z_img, z_action = self.encode(image, action)
        # E_m: detach z at the input so the memory path can't reshape the encoder
        m = self.memory_predictor(z_img.detach()) if self.memory_predictor is not None else None
        z_pred = self.predict(z_img, z_action, m, ar_steps=0)
        return z_pred, z_img, z_action, m

    def memory_decode_loss(self, z_img, z_action, m, return_per_k=False):
        """Many-future objective: from each m_t predict z_{t+1..t+H} in one parallel pass.

        Targets are stop-grad (z is the frozen-for-memory target). Returns a scalar,
        or (scalar, per_k) where per_k[k] is the masked MSE at horizon offset k.
        """
        B, Tt, Zi = z_img.shape
        Hh = self.horizon

        # targets tgt[b,t,k] = z_img[b, t+1+k]
        z_pad = F.pad(z_img, (0, 0, 0, Hh))                          # (B, T+H, Zi)
        tgt = z_pad.unfold(1, Hh, 1)[:, 1:Tt + 1]                    # (B, T, Zi, H)
        tgt = rearrange(tgt, "b t z h -> b t h z").detach()         # stop-grad target

        # future action window awin[b,t,k] = z_action[b, t+k]
        a_pad = F.pad(z_action, (0, 0, 0, Hh - 1))                  # (B, T+H-1, A)
        awin = a_pad.unfold(1, Hh, 1)[:, :Tt]                       # (B, T, A, H)
        awin = rearrange(awin, "b t a h -> b t h a")               # (B, T, H, A)

        # valid where t+1+k < T (target frame exists)
        t_idx = torch.arange(Tt, device=z_img.device)[:, None]
        k_idx = torch.arange(Hh, device=z_img.device)[None, :]
        valid = ((t_idx + 1 + k_idx) < Tt)[None].expand(B, Tt, Hh)  # (B, T, H)

        # fold source-time into batch -> one batched causal pass
        m_flat = rearrange(m, "b t z -> (b t) z")
        awin_flat = rearrange(awin, "b t h a -> (b t) h a")
        pred = self.memory_decoder(m_flat, awin_flat)               # (B*T, H, Zi)
        pred = rearrange(pred, "(b t) h z -> b t h z", b=B)

        se = (pred - tgt).square().sum(-1)                          # (B, T, H)
        denom = valid.sum().clamp_min(1)
        loss = (se * valid).sum() / denom
        if not return_per_k:
            return loss
        per_k = (se * valid).sum(dim=(0, 1)) / valid.sum(dim=(0, 1)).clamp_min(1)
        return loss, per_k                                          # per_k: (H,)

    def memory_update_loss(self, m, z_img, z_action):
        """Train U to match E_m's next memory (teacher-forced, fully parallel)."""
        m_hat = self.memory_updater(m[:, :-1], z_img[:, 1:].detach(), z_action[:, :-1])
        return F.mse_loss(m_hat, m[:, 1:].detach())

    def rollout_latents(self, image, action, steps):
        """Honest generative rollout from the true first frame only (grad flows through).

        Works for any config and any length: nomem -> plain AR; single -> memory
        recomputed by E_m over the predicted prefix (no leak); multi -> memory rolled
        forward by U. Returns predicted latents (B, steps, Z_img).
        """
        z_img, z_action = self.encode(image, action)
        z_in = z_img[:, :1]                                        # true seed only
        use_u = self.memory_updater is not None
        if use_u:
            m_t = self.memory_predictor(z_in)[:, 0]
            m_list = [m_t]
        preds = []
        for t in range(steps):
            if self.memory_predictor is None:
                x = z_in
            elif use_u:
                x = torch.cat([z_in, torch.stack(m_list, dim=1)], dim=-1)
            else:
                x = torch.cat([z_in, self.memory_predictor(z_in)], dim=-1)  # E_m over predicted prefix
            raw = self.predictor(x, z_action[:, :t + 1])[:, -1:]
            z_next = self.project(raw)
            preds.append(z_next)
            if use_u:
                m_t = self.memory_updater(m_t, z_next[:, 0], z_action[:, t])
                m_list.append(m_t)
            z_in = torch.cat([z_in, z_next], dim=1)
        return torch.cat(preds, dim=1)                            # (B, steps, Z_img)

    @torch.no_grad()
    def predict_with_updater(self, image, action):
        """Eval wrapper (no grad) around the honest rollout at the training horizon."""
        return self.rollout_latents(image, action, T), None

    def dmt_loss_ood(self, inp, actions, tgt, steps):
        """DMT: unroll on own predictions, regress to the FROZEN encoder trajectory."""
        with torch.no_grad():
            z_target, _ = self.encode(tgt)
        z_pred = self.rollout_latents(inp, actions, steps)
        return F.mse_loss(z_pred, z_target.detach())

    def freeze_for_dmt(self):
        """Freeze everything except the rollout chain (predictor + updater). Returns its params."""
        for p in self.parameters():
            p.requires_grad_(False)
        train = list(self.predictor.parameters())
        for p in self.predictor.parameters():
            p.requires_grad_(True)
        if self.memory_updater is not None:
            for p in self.memory_updater.parameters():
                p.requires_grad_(True)
            train += list(self.memory_updater.parameters())
        return train


# ============================================================================
#  Data / model construction (mirrors exp_memory.py)
# ============================================================================

_TRAIN_DS = None
_VAL_DS = None


def _datasets():
    global _TRAIN_DS, _VAL_DS
    if _TRAIN_DS is None:
        _TRAIN_DS = torchvision.datasets.MNIST(root="./dataset", train=True, download=True, transform=transforms.ToTensor())
        _VAL_DS = torchvision.datasets.MNIST(root="./dataset", train=False, download=True, transform=transforms.ToTensor())
    return _TRAIN_DS, _VAL_DS


def make_train_loader(seed=SEED_DATA):
    """Fresh loader with its own seeded generator -> identical batch order per call."""
    train_ds, _ = _datasets()
    g = torch.Generator().manual_seed(seed)
    return DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True, generator=g)


def make_val_loader():
    _, val_ds = _datasets()
    return DataLoader(val_ds, batch_size=BATCH, shuffle=False, drop_last=True)


def build_model(cfg):
    """cfg: {z_mem, multi}. z_mem=0 -> no-memory baseline; multi -> add D_m + U."""
    z_mem = cfg["z_mem"]
    multi = cfg.get("multi", False)
    enc = ImageEncoder(H * W, 512, Z_IMG, depth=3)
    ae = ActionEncoder(3, emb_dim=Z_ACT)
    # content mode: memory rides the predictor INPUT, action stays the cond
    pred_input_dim = Z_IMG + (z_mem if z_mem else 0)
    pr = ARPredictor(num_frames=T, depth=4, heads=4, mlp_dim=512, input_dim=pred_input_dim,
                     hidden_dim=512, output_dim=512, action_dim=Z_ACT)
    pj = MLP_Projector(input_dim=512, output_dim=Z_IMG, hidden_dim=256, norm_fn=torch.nn.BatchNorm1d)
    dc = Decoder(z_dim=Z_IMG, hidden_dim=512, h=H, w=W, depth=2)
    projector = nn.LayerNorm(Z_IMG, elementwise_affine=False)  # pin latent scale for SigReg
    mp = MemoryPredictor(Z_IMG, z_mem, hidden_dim=MEM_HIDDEN, depth=2, heads=4) if z_mem else None
    md = MemoryDecoder(z_mem, Z_IMG, Z_ACT, HORIZON) if (z_mem and multi) else None
    mu = MemoryUpdater(z_mem, Z_IMG, Z_ACT) if (z_mem and multi) else None
    model = DecoupledMemJEPA(
        enc, pr, ae, decoder=dc, projector=projector, projector_pred=pj,
        memory_predictor=mp, mem_mode="content",
        memory_decoder=md, memory_updater=mu, horizon=HORIZON,
    )
    return model.to(DEV)


# ============================================================================
#  Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(model, val_batch):
    """Latent nMSE (teacher-forced + AR), pixel recon, collapse diagnostics, packs."""
    model.eval()
    seed, actions, inp, tgt = rollout(val_batch, T, SCALE_S, TRANS_S, device=DEV)
    z_tgt, _ = model.encode(tgt)
    tgt_var = z_tgt.var().item() + 1e-8

    zp_tf, z_img, _ = model(inp, actions, ar_steps=0)
    zp_ar, _, _ = model(inp, actions, ar_steps=T)

    mse_tf_t = ((zp_tf - z_tgt) ** 2).mean(dim=-1).mean(dim=0)   # (T,)
    mse_ar_t = ((zp_ar - z_tgt) ** 2).mean(dim=-1).mean(dim=0)   # (T,)

    rec_enc = model.decode(z_img)
    rec_tf = model.decode(zp_tf)
    rec_ar = model.decode(zp_ar)
    px_enc = F.mse_loss(rec_enc, inp.float()).item()
    px_tf = F.mse_loss(rec_tf, tgt.float()).item()
    px_ar = F.mse_loss(rec_ar, tgt.float()).item()

    zf = z_img.reshape(-1, z_img.size(-1))
    z_std = zf.std(0).mean().item()
    a, b = zf[:50], zf[50:100]
    cos = F.cosine_similarity(a, b).mean().item()

    # decoupled-memory diagnostics: U-rollout AR drift + per-horizon decoder error
    nmse_upd = None
    horizon_mse = None
    if model.memory_decoder is not None:
        zp_upd, _ = model.predict_with_updater(inp, actions)
        nmse_upd = (((zp_upd - z_tgt) ** 2).mean()).item() / tgt_var
        m = model.memory_predictor(z_img)
        _, per_k = model.memory_decode_loss(z_img, model.encode(inp, actions)[1], m, return_per_k=True)
        horizon_mse = (per_k / tgt_var).cpu().tolist()

    metrics = {
        "mse_tf": mse_tf_t.mean().item(),
        "mse_ar": mse_ar_t.mean().item(),
        "nmse_tf": mse_tf_t.mean().item() / tgt_var,
        "nmse_ar": mse_ar_t.mean().item() / tgt_var,
        "nmse_upd": nmse_upd,
        "tgt_var": tgt_var,
        "mse_tf_t": mse_tf_t.cpu().tolist(),
        "mse_ar_t": mse_ar_t.cpu().tolist(),
        "horizon_mse": horizon_mse,
        "px_enc": px_enc, "px_tf": px_tf, "px_ar": px_ar,
        "z_std": z_std, "cos": cos,
    }
    return metrics, (inp, tgt, rec_enc, rec_tf, rec_ar)


# ============================================================================
#  Plotting (each save site documents what the figure shows)
# ============================================================================

def _row(axes, r, imgs, b, label):
    for t in range(T):
        axes[r, t].imshow(imgs[b, t, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[r, t].set_xticks([]); axes[r, t].set_yticks([])
    axes[r, 0].set_ylabel(label, fontsize=7)


def save_decode_grid(plot_dir, name, pack, n=3):
    """Rows per sample: input / dec(z_enc) / target / dec(pred_TF) / dec(pred_AR)."""
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
    plt.savefig(os.path.join(plot_dir, "decode.png"), dpi=90)
    plt.close(fig)


def save_train_quick_plots(plot_dir, hist):
    """The three train_quick.py figures, same filenames/titles/series for comparability."""
    # 1. train_history.png: per-step train latent MSE
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist["step"], hist["train_mse"], lw=0.8, alpha=0.8)
    ax.set_xlabel("step"); ax.set_ylabel("train latent MSE"); ax.set_yscale("log")
    ax.set_title("Training history: per-step latent MSE"); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(os.path.join(plot_dir, "train_history.png"), dpi=110); plt.close(fig)

    # 2. latent_mse_history.png: val TF vs AR latent MSE (+ train loss) per epoch
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist["epoch"], hist["val_mse_tf"], marker="o", label="val MSE (teacher-forced)")
    ax.plot(hist["epoch"], hist["val_mse_ar"], marker="o", label="val MSE (autoregressive)")
    ax.plot(hist["epoch"], hist["train_loss"], marker=".", ls="--", alpha=0.6, label="train total loss")
    if any(v is not None for v in hist["val_mse_upd"]):
        ax.plot(hist["epoch"], hist["val_mse_upd"], marker="o", label="val MSE (U-rollout)")
    ax.set_xlabel("epoch"); ax.set_ylabel("latent MSE / loss"); ax.set_yscale("log")
    ax.set_title("Latent MSE over training"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(os.path.join(plot_dir, "latent_mse_history.png"), dpi=110); plt.close(fig)

    # 3. recon_history.png: val decoder recon MSE (encoder vs predicted latents) per epoch
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist["epoch"], hist["val_recon_enc"], marker="o", label="recon (encoder latents)")
    ax.plot(hist["epoch"], hist["val_recon_pred"], marker="o", label="recon (predicted latents)")
    ax.set_xlabel("epoch"); ax.set_ylabel("pixel MSE"); ax.set_yscale("log")
    ax.set_title("Decoder reconstruction over training"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(os.path.join(plot_dir, "recon_history.png"), dpi=110); plt.close(fig)


def save_extras(plot_dir, name, metrics):
    """Memory-specific figures: per-step AR drift and (multi) D_m error vs horizon."""
    plt.figure(figsize=(8, 4))
    plt.plot(range(T), metrics["mse_ar_t"], marker="o", label=f"AR mean {metrics['mse_ar']:.3f}")
    plt.xlabel("rollout step t"); plt.ylabel("latent MSE")
    plt.title(f"Per-step AR latent MSE ({name})"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "per_step_ar_mse.png"), dpi=110); plt.close()

    hm = metrics["horizon_mse"]
    if hm is not None:
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, len(hm) + 1), hm, marker="o")
        plt.xlabel("horizon offset k (predict z_{t+k})"); plt.ylabel("decoder nMSE")
        plt.title(f"Memory-decoder error vs horizon ({name})"); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "mem_horizon_mse.png"), dpi=110); plt.close()


def plot_dmt_decode(plot_dir, target, dec_before, dec_after, n=4, title=""):
    """target/before/after rows over T_OOD frames; red box marks the trained horizon."""
    Tn = target.size(1)
    fig, axes = plt.subplots(3 * n, Tn, figsize=(Tn * 0.6, 3 * n * 0.6))
    rows = [("tgt", target), ("before", dec_before), ("after", dec_after)]
    for b in range(n):
        for k, (lbl, src) in enumerate(rows):
            r = 3 * b + k
            for t in range(Tn):
                axes[r, t].imshow(src[b, t, 0].cpu(), cmap="gray", vmin=0, vmax=1)
                axes[r, t].set_xticks([]); axes[r, t].set_yticks([])
                if t == T - 1 and Tn > T:  # mark train-horizon boundary
                    for sp in axes[r, t].spines.values():
                        sp.set_color("red"); sp.set_linewidth(1.5)
            axes[r, 0].set_ylabel(f"t{b}\n{lbl}", fontsize=7)
    plt.suptitle(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "dmt_decode_before_after.png"), dpi=110)
    plt.close(fig)


def run_dmt_and_viz(model, plot_dir, name, seed=SEED_TRAIN):
    """DMT post-finetune + before/after decode rolled out to T_OOD (beyond training T)."""
    val_imgs = next(iter(make_val_loader()))[0].to(DEV).squeeze(1)
    with torch.no_grad():
        _, viz_act, viz_inp, viz_tgt = rollout(val_imgs, T_OOD, SCALE_S, TRANS_S, device=DEV)

    @torch.no_grad()
    def decode_and_perstep(m):
        m.eval()
        z_pred = m.rollout_latents(viz_inp, viz_act, T_OOD)
        dec = m.decode(z_pred)
        per = ((dec - viz_tgt.float()) ** 2).mean(dim=(0, 2, 3, 4))  # (T_OOD,)
        return dec, per

    dec_before, per_before = decode_and_perstep(model)

    params = model.freeze_for_dmt()
    opt = torch.optim.AdamW(params, lr=DMT_LR)
    model.eval()
    for p in params:
        p.requires_grad_(True)
    train_loader = make_train_loader(SEED_DATA)
    torch.manual_seed(seed)
    it = iter(train_loader)
    for _ in tqdm(range(DMT_STEPS), desc=f"dmt-{name}", leave=False):
        try:
            imgs, _ = next(it)
        except StopIteration:
            it = iter(train_loader); imgs, _ = next(it)
        imgs = imgs.to(DEV).squeeze(1)
        with torch.no_grad():
            _, acts, inp, tgt = rollout(imgs, T, SCALE_S, TRANS_S, device=DEV)  # DMT at trained horizon T
        loss = model.dmt_loss_ood(inp, acts, tgt, T)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    dec_after, per_after = decode_and_perstep(model)
    plot_dmt_decode(plot_dir, viz_tgt.float(), dec_before, dec_after, n=4,
                    title=f"{name}: DMT decode AR (T={T_OOD}, red=train horizon T={T})")

    # per-step recon before vs after, with the train-horizon boundary marked
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, T_OOD + 1), per_before.cpu(), marker="o", label="before DMT")
    plt.plot(range(1, T_OOD + 1), per_after.cpu(), marker="o", label="after DMT")
    plt.axvline(T, color="k", ls="--", lw=0.7, label="train horizon")
    plt.xlabel("rollout step"); plt.ylabel("decoder recon MSE"); plt.yscale("log")
    plt.title(f"Per-step AR recon: before vs after DMT ({name})")
    plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir, "dmt_perstep_recon.png"), dpi=110); plt.close()
    for p in model.parameters():
        p.requires_grad_(True)


# ============================================================================
#  Validation (epoch-level, mirrors train_quick.py's val loop)
# ============================================================================

@torch.no_grad()
def validate(model, val_loader):
    """Per-epoch val metrics matching train_quick.py (+ U-rollout AR for multi)."""
    model.eval()
    sums = {"loss": 0.0, "mse_tf": 0.0, "mse_ar": 0.0, "rec_enc": 0.0, "rec_pred": 0.0, "mse_upd": 0.0}
    n = 0
    has_upd = model.memory_updater is not None
    for imgs, _ in val_loader:
        imgs = imgs.to(DEV).squeeze(1)
        seed, actions, inp, tgt = rollout(imgs, T, SCALE_S, TRANS_S, device=DEV)
        z_tgt, _ = model.encode(tgt)
        z_ar, _, _ = model(inp, actions, ar_steps=T)
        z_tf, z_img, _ = model(inp, actions, ar_steps=0)
        mse_ar = model.mse(z_ar, z_tgt, mean=False)
        mse_tf = model.mse(z_tf, z_tgt, mean=False)
        rec_enc = model.recon_loss(z_img, inp)
        rec_pred = model.recon_loss(z_ar, tgt)
        sigreg = model.sigreg_loss(z_img)
        sums["loss"] += (mse_ar + LAMBD_SIG * sigreg).item()
        sums["mse_tf"] += mse_tf.item(); sums["mse_ar"] += mse_ar.item()
        sums["rec_enc"] += rec_enc.item(); sums["rec_pred"] += rec_pred.item()
        if has_upd:
            z_upd, _ = model.predict_with_updater(inp, actions)
            sums["mse_upd"] += model.mse(z_upd, z_tgt, mean=False).item()
        n += 1
        if n >= VAL_BATCHES:
            break
    out = {k: v / max(n, 1) for k, v in sums.items()}
    out["mse_upd"] = out["mse_upd"] if has_upd else None
    return out


# ============================================================================
#  Training (epoch loop, same budget/cadence as train_quick.py)
# ============================================================================

def train_one(cfg, val_loader, val_imgs, seed=SEED_TRAIN, epochs=EPOCHS):
    name = cfg["name"]
    lambd_sig = cfg.get("lambd_sig", LAMBD_SIG)
    plot_dir = plot_dir_for(name)
    train_loader = make_train_loader(SEED_DATA)
    torch.manual_seed(seed)
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    hist = {"step": [], "train_mse": [], "epoch": [], "train_loss": [],
            "val_mse_tf": [], "val_mse_ar": [], "val_mse_upd": [],
            "val_recon_enc": [], "val_recon_pred": []}
    torch.manual_seed(seed)  # rollout/sigreg RNG
    global_step = 0
    t0 = time.time()
    for epoch in tqdm(range(epochs), desc=name):
        model.train()
        train_loss_sum, train_batches = 0.0, 0
        for imgs, _ in train_loader:
            imgs = imgs.to(DEV).squeeze(1)
            with torch.no_grad():
                seed_g, actions, inp, tgt = rollout(imgs, T, SCALE_S, TRANS_S, device=DEV)
            zp, z_img, z_action, m = model.memory_forward(inp, actions)
            z_tgt, _ = model.encode(tgt)
            loss_mse = model.mse(zp, z_tgt, mean=False)
            loss_sig = model.sigreg_loss(z_img)
            loss_rec = model.recon_loss(z_img.detach(), inp)
            loss = loss_mse + lambd_sig * loss_sig + LAMBD_REC * loss_rec
            # decoupled-memory objectives (multi only)
            if model.memory_decoder is not None:
                loss = loss + model.memory_decode_loss(z_img, z_action, m)
                loss = loss + model.memory_update_loss(m, z_img, z_action)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss_sum += loss.item(); train_batches += 1
            hist["step"].append(global_step); hist["train_mse"].append(loss_mse.item())
            global_step += 1

        v = validate(model, val_loader)
        hist["epoch"].append(epoch)
        hist["train_loss"].append(train_loss_sum / max(train_batches, 1))
        hist["val_mse_tf"].append(v["mse_tf"]); hist["val_mse_ar"].append(v["mse_ar"])
        hist["val_mse_upd"].append(v["mse_upd"])
        hist["val_recon_enc"].append(v["rec_enc"]); hist["val_recon_pred"].append(v["rec_pred"])
    dt = time.time() - t0

    metrics, pack = evaluate(model, val_imgs)
    metrics["train_s"] = dt
    metrics["n_params"] = sum(p.numel() for p in model.parameters())
    metrics["val_mse_ar_final"] = hist["val_mse_ar"][-1]
    metrics["val_mse_tf_final"] = hist["val_mse_tf"][-1]
    save_train_quick_plots(plot_dir, hist)
    save_decode_grid(plot_dir, name, pack)
    save_extras(plot_dir, name, metrics)
    run_dmt_and_viz(model, plot_dir, name, seed=seed)  # DMT before/after, rolled out to T_OOD
    with open(os.path.join(plot_dir, "history.json"), "w") as f:
        json.dump({"hist": hist, "metrics": {k: metrics[k] for k in
                  ("nmse_tf", "nmse_ar", "nmse_upd", "px_ar", "z_std", "n_params",
                   "val_mse_ar_final", "val_mse_tf_final", "horizon_mse")}}, f, indent=2)
    upd = f" nmse_upd {metrics['nmse_upd']:.3f}" if metrics["nmse_upd"] is not None else ""
    print(f"[{name:12s}] nmse_tf {metrics['nmse_tf']:.4f}  nmse_ar {metrics['nmse_ar']:.4f}{upd}  | "
          f"val_mse_ar {metrics['val_mse_ar_final']:.4f}  px_ar {metrics['px_ar']:.4f}  "
          f"z_std {metrics['z_std']:.2f}  params {metrics['n_params']/1e3:.0f}k  ({dt:.0f}s)")
    return name, metrics, hist


def main(configs, epochs=EPOCHS):
    val_loader = make_val_loader()
    val_imgs = next(iter(val_loader))[0].to(DEV).squeeze(1)

    results, hists = {}, {}
    for cfg in configs:
        name, metrics, hist = train_one(cfg, val_loader, val_imgs, epochs=epochs)
        results[name] = metrics
        hists[name] = hist

    # cross-config comparison folder
    cmp_dir = plot_dir_for("compare")
    # per-step AR latent MSE overlay (final)
    plt.figure(figsize=(8, 4))
    for name, m in results.items():
        plt.plot(range(T), m["mse_ar_t"], marker="o", label=f"{name} (AR mean {m['mse_ar']:.3f})")
    plt.xlabel("rollout step t"); plt.ylabel("latent MSE")
    plt.title("Per-step AR latent MSE (final)"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(cmp_dir, "per_step_ar_mse.png"), dpi=110); plt.close()
    # val AR latent MSE over epochs overlay
    plt.figure(figsize=(8, 4))
    for name, h in hists.items():
        plt.plot(h["epoch"], h["val_mse_ar"], marker="o", label=name)
    plt.xlabel("epoch"); plt.ylabel("val AR latent MSE"); plt.yscale("log")
    plt.title("Val AR latent MSE over training"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(cmp_dir, "val_ar_history.png"), dpi=110); plt.close()

    with open(os.path.join(cmp_dir, "results.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "mse_ar_t" and kk != "mse_tf_t"}
                   for k, v in results.items()}, f, indent=2)
    print(f"\nplots written under out/plots/{DATE_TAG}-decoupledMem-*/")
    return results


# ============================================================================
#  Fast unit checks: bottleneck/causality + stop-grad (verification steps 2-3)
# ============================================================================

def _check():
    torch.manual_seed(0)
    model = build_model({"name": "chk", "z_mem": 36, "multi": True})
    model.eval()
    imgs = torch.rand(4, H, W, device=DEV)
    _, actions, inp, _ = rollout(imgs, T, SCALE_S, TRANS_S, device=DEV)
    z_img, z_action = model.encode(inp, actions)
    m = model.memory_predictor(z_img)

    # (a) scalar loss
    loss = model.memory_decode_loss(z_img, z_action, m)
    assert loss.dim() == 0, "decode loss must be a scalar"

    # (b) bottleneck + causality: D_m output for source t, offset k must NOT depend on
    #     z_img[:, >t] (only m feeds it) nor on actions a_{>t+k}.
    with torch.no_grad():
        _, per_k0 = model.memory_decode_loss(z_img, z_action, m, return_per_k=True)
        m_flat = rearrange(m, "b t z -> (b t) z")
        a_pad = F.pad(z_action, (0, 0, 0, HORIZON - 1))
        awin = rearrange(a_pad.unfold(1, HORIZON, 1)[:, :T], "b t a h -> (b t) h a")
        base = model.memory_decoder(m_flat, awin.clone())
        awin2 = awin.clone(); awin2[:, -1] += 10.0  # perturb the LAST future action
        out2 = model.memory_decoder(m_flat, awin2)
        # only the last horizon step may change; earlier predictions must be identical
        early_same = torch.allclose(base[:, :-1], out2[:, :-1], atol=1e-5)
    print(f"[check] decode loss scalar OK | causal (early preds unchanged by future action): {early_same}")

    # (c) stop-grad: L_dec alone must put ZERO gradient on the encoder.
    model.zero_grad()
    z_img2, z_action2 = model.encode(inp, actions)
    m2 = model.memory_predictor(z_img2.detach())
    model.memory_decode_loss(z_img2, z_action2, m2).backward()
    enc_grad = sum(p.grad.abs().sum().item() for p in model.encoder.parameters() if p.grad is not None)
    print(f"[check] encoder grad from L_dec (must be 0.0): {enc_grad:.6e}")
    assert early_same and enc_grad == 0.0, "bottleneck/stop-grad check FAILED"
    print("[check] all good.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        _check()
    else:
        epochs = int(sys.argv[1]) if len(sys.argv) > 1 else EPOCHS
        configs = [
            {"name": "nomem", "z_mem": 0},
            {"name": "mem36_single", "z_mem": 36, "multi": False},
            {"name": "mem36_multi", "z_mem": 36, "multi": True},
        ]
        main(configs, epochs=epochs)
