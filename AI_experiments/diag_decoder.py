"""Decisive test: is the wrong-transform / blur in predictions the DECODER or the PREDICTOR?

(a) Train a strong decoder on the TARGET-latent manifold, decode A's TF/AR predictions with it,
    and compare to the joint decoder. If transforms snap back -> decoder was the culprit.
(b) Linear-probe the absolute glimpse params (log_scale, x, y) from z_enc. High R^2 -> the encoder
    fully preserves the transform, so any transform error in recon is decoding, not representation.

Run:  "C:/Users/Ous/miniconda3/envs/ML/python.exe" diag_decoder.py
"""
import torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import exp_ideas as E
from glimpse import rollout
from ml_layers import Decoder

DEV, T, Z = E.DEV, E.T, E.Z_IMG
vb = E.val_batch()

# --- train Idea A ---
A = E.MemModel("content", "A_memcontent")
E.train_model(A, steps=1200, val_imgs=vb)
m = A.jepa
for p in m.parameters():
    p.requires_grad_(False)


def a_latents(inp, actions):
    """z_enc(inp), TF prediction, honest-AR prediction, z_enc(tgt)."""
    z_img, z_act = m.encode(inp, actions)
    z_tf, _, _ = m(inp, actions, ar_steps=0)
    z_in = z_img[:, :1]
    for t in range(T):
        mem = m.predict_memory(z_in)
        x = z_in + m.mem_proj(mem)
        raw = m.predictor(x, z_act[:, :t + 1])[:, -1:]
        z_in = torch.cat([z_in, m.project(raw)], 1)
    return z_img, z_tf, z_in[:, 1:]


# --- (a) strong decoder trained on target + predicted latents (prediction manifold) ---
probe = Decoder(z_dim=Z, hidden_dim=512, h=28, w=28, depth=3).to(DEV)
opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
loader = E.make_train_loader(); it = iter(loader)
for step in range(2500):
    try:
        imgs, _ = next(it)
    except StopIteration:
        it = iter(loader); imgs, _ = next(it)
    imgs = imgs.to(DEV).squeeze(1)
    with torch.no_grad():
        _, actions, inp, tgt = rollout(imgs, T, E.SCALE_S, E.TRANS_S, device=DEV)
        z_img, z_tf, z_ar = a_latents(inp, actions)
        z_tgt, _ = m.encode(tgt)
        # cover both the target manifold and the model's own prediction manifold
        z_train = torch.cat([z_tgt, z_tf, z_ar], 0)
        img_train = torch.cat([tgt, tgt, tgt], 0).float()
    rec = probe(z_train)
    loss = F.mse_loss(rec, img_train)
    opt.zero_grad(); loss.backward(); opt.step()
probe.eval()

# --- eval grid on a fixed val rollout ---
torch.manual_seed(0)
_, actions, inp, tgt = rollout(vb, T, E.SCALE_S, E.TRANS_S, device=DEV)
with torch.no_grad():
    z_img, z_tf, z_ar = a_latents(inp, actions)
    joint_ar = m.decode(z_ar); probe_ar = probe(z_ar)
    joint_tf = m.decode(z_tf); probe_tf = probe(z_tf)
    px = lambda a, b: F.mse_loss(a, b.float()).item()
    print(f"px(AR) joint {px(joint_ar, tgt):.4f} -> probe {px(probe_ar, tgt):.4f} | "
          f"px(TF) joint {px(joint_tf, tgt):.4f} -> probe {px(probe_tf, tgt):.4f}")

rows = [("inp", inp), ("tgt", tgt), ("joint dec(AR)", joint_ar),
        ("strong dec(AR)", probe_ar), ("strong dec(TF)", probe_tf)]
n = 3
fig, ax = plt.subplots(len(rows) * n, T, figsize=(T * 0.8, len(rows) * n * 0.8))
for b in range(n):
    for r, (lab, src) in enumerate(rows):
        for t in range(T):
            a = ax[len(rows) * b + r, t]
            a.imshow(src[b, t, 0].cpu(), cmap="gray", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
        ax[len(rows) * b + r, 0].set_ylabel(lab, fontsize=7)
plt.suptitle("Decoder vs predictor: joint decoder vs strong decoder on A's predictions", fontsize=9)
plt.tight_layout(); plt.savefig(f"{E.OUT}/diag_decoder.png", dpi=90); plt.close()

# --- (b) linear probe: recover absolute glimpse params (log_scale, x, y) from z_enc ---
with torch.no_grad():
    # absolute state of inp[t] = cumulative sum of actions up to t (seed = 0)
    cum = torch.cumsum(actions, dim=1)                       # (B,T,3) abs state AFTER action t
    abs_state = torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], 1)  # state of inp[t]
    Zf = z_img.reshape(-1, Z); Yf = abs_state.reshape(-1, 3)
    # closed-form ridge linear probe with bias
    Xb = torch.cat([Zf, torch.ones(Zf.size(0), 1, device=DEV)], 1)
    W = torch.linalg.lstsq(Xb, Yf).solution
    pred = Xb @ W
    ss_res = ((Yf - pred) ** 2).sum(0)
    ss_tot = ((Yf - Yf.mean(0)) ** 2).sum(0) + 1e-8
    r2 = (1 - ss_res / ss_tot).cpu().tolist()
    print(f"pose-recoverability R^2 from z_enc  log_scale {r2[0]:.3f}  x {r2[1]:.3f}  y {r2[2]:.3f}")
print("saved", f"{E.OUT}/diag_decoder.png")
