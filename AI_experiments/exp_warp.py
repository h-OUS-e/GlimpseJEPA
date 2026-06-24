"""Token-warp predictor vs the re-predict (ST) predictor, on AR rollout drift.

WarpPredictor predicts the next frame's tokens by TRANSPORTING the current tokens: each output
position is an action-steered soft combination (attention) of the current content tokens, so content
is moved rather than re-synthesized from scratch. Output is a convex-ish combo of real tokens ->
stays on the content manifold -> should resist the late-step mush we see with free re-prediction.

Compares flat baseline / spatial-ST / spatial-warp, each with DMT post-finetune, judged on AR
(latent nMSE + pixel) with before/after rollout grids. Reuses encoder/decoder + DMT harness.

Run: "C:/Users/Ous/miniconda3/envs/ML/python.exe" exp_warp.py [pretrain] [dmt]
"""
import sys, json
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import exp_ideas as E
from exp_spatial_jepa import SpatialJEPA, ViTSpatialEncoder, ViTSpatialDecoder, plot_rollout, OUT
from exp_dmt import dmt, eval_ar
from glimpse import rollout

DEV, T = E.DEV, E.T


class WarpPredictor(nn.Module):
    """Next tokens = action-steered transport of current tokens (+ small gated refine).

    queries: learned per-position, modulated by the action (where to pull content from).
    keys:    current content tokens + position.   values: the current tokens themselves.
    output:  softmax(q.k) @ tokens  -> content moved to new positions, never invented.
    """
    def __init__(self, c=8, hidden=64, n=16):
        super().__init__()
        self.n, self.scale = n, hidden ** -0.5
        self.in_proj = nn.Linear(c, hidden)
        self.pos = nn.Parameter(torch.randn(1, n, hidden) * 0.02)
        self.q = nn.Parameter(torch.randn(1, n, hidden) * 0.02)
        self.act = nn.Linear(3, 2 * hidden)
        nn.init.zeros_(self.act.weight); nn.init.zeros_(self.act.bias)  # start action-agnostic
        self.to_q = nn.Linear(hidden, hidden)
        self.to_k = nn.Linear(hidden, hidden)
        self.refine = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, hidden), nn.GELU(), nn.Linear(hidden, c))
        self.gate = nn.Parameter(torch.zeros(1))

    def _step(self, tok, act):  # tok (B,N,C), act (B,3) -> (B,N,C)
        h = self.in_proj(tok) + self.pos
        shift, sc = self.act(act).chunk(2, dim=-1)
        q = self.to_q(self.q * (1 + sc[:, None, :]) + shift[:, None, :])
        k = self.to_k(h)
        attn = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)  # (B,N,N) routing
        transported = attn @ tok                                            # move content
        return transported + self.gate * self.refine(transported)

    def forward(self, tokens, action):  # (B,T,N,C),(B,T,3) -> (B,T,N,C)  per-frame transport
        B, Tn, N, C = tokens.shape
        out = self._step(tokens.reshape(B * Tn, N, C), action.reshape(B * Tn, 3))
        return out.reshape(B, Tn, N, C)


class SpatialJEPAWarp(SpatialJEPA):
    """Same ViT encoder/decoder + losses + honest-AR eval as SpatialJEPA; predictor is the warp."""
    def __init__(self, c=8, name="spatial_warp"):
        super().__init__(c=c, name=name)
        self.pred = WarpPredictor(c=c)


def main(pre=1500, dmt_steps=500):
    vb = E.val_batch()
    configs = [
        ("flat_baseline", lambda: E.MemModel("adaln", "flat_baseline"), "mem"),
        ("spatial_ST", lambda: SpatialJEPA(c=8, name="spatial_ST"), "spatial"),
        ("spatial_warp", lambda: SpatialJEPAWarp(c=8, name="spatial_warp"), "spatial"),
    ]
    res = {}
    for name, build, kind in configs:
        m = build().to(DEV)
        print(f"-- {name} ({sum(p.numel() for p in m.parameters())/1e6:.2f}M) --")
        E.train_model(m, steps=pre, val_imgs=vb)
        before, pack_b = eval_ar(m, vb); plot_rollout(f"{name}_before", pack_b)
        dmt(m, kind, steps=dmt_steps)
        after, pack_a = eval_ar(m, vb); plot_rollout(f"{name}_after", pack_a)
        res[name] = {"before": before, "after": after}
        print(f"[{name:14s}] nmse_ar {before['nmse_ar']:.3f}->{after['nmse_ar']:.3f} | "
              f"px_ar {before['px_ar']:.4f}->{after['px_ar']:.4f} | px_enc {after['px_enc']:.4f}")

    names = list(res)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for name in names:
        ax[0].plot(range(1, T + 1), res[name]["before"]["mse_ar_t"], "--", alpha=0.45)
        ax[0].plot(range(1, T + 1), res[name]["after"]["mse_ar_t"], "-", label=name)
    ax[0].set_xlabel("rollout step"); ax[0].set_ylabel("AR latent nMSE")
    ax[0].set_title("Per-step AR nMSE (dashed=before DMT, solid=after)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    x = range(len(names)); w = 0.35
    ax[1].bar([i - w/2 for i in x], [res[n]["before"]["px_ar"] for n in names], w, label="before DMT")
    ax[1].bar([i + w/2 for i in x], [res[n]["after"]["px_ar"] for n in names], w, label="after DMT")
    ax[1].set_xticks(list(x)); ax[1].set_xticklabels(names, rotation=12, fontsize=8)
    ax[1].set_ylabel("AR pixel MSE"); ax[1].set_title("AR pixel recon"); ax[1].legend()
    ax[2].bar([i - w/2 for i in x], [res[n]["before"]["nmse_ar"] for n in names], w, label="before DMT")
    ax[2].bar([i + w/2 for i in x], [res[n]["after"]["nmse_ar"] for n in names], w, label="after DMT")
    ax[2].set_xticks(list(x)); ax[2].set_xticklabels(names, rotation=12, fontsize=8)
    ax[2].set_ylabel("AR latent nMSE"); ax[2].set_title("AR latent nMSE (not cross-comparable)"); ax[2].legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/warp_compare.png", dpi=100); plt.close(fig)
    with open(f"{OUT}/warp_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nsaved warp_compare.png + per-model rollout grids to", OUT)


if __name__ == "__main__":
    pre = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    ds = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    main(pre, ds)
