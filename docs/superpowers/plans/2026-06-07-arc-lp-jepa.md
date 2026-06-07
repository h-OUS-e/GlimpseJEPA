# ARC-AGI Latent Program JEPA (LP-JEPA) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, in one self-contained notebook, a JEPA that learns ARC-AGI transformations by treating demonstration pairs as a latent program applied to a test input — latent-only first (Phase 1), then with a discrete grid decoder for exact-match (Phase 2).

**Architecture:** Two transformer encoders (a grid/state encoder `E`, a program encoder `P` over I/O pairs), an AdaLN-conditioned predictor `F` that applies program `z` to state `a`, an EMA stop-grad target encoder `Ē`, and a Phase-2 decoder `G`. Anti-collapse via EMA target + SIGReg computed across augmentation views. Pretrain on ARC-train subset + dihedral/color augmentation, then test-time train (no-freeze) on unseen tasks, compared against frozen+z-gradient-ascent (LPN) and from-scratch.

**Tech Stack:** Python (env `C:/Users/Ous/miniconda3/envs/ML/python.exe`), PyTorch, einops, scikit-learn (TSNE), matplotlib, Jupyter. ARC-AGI-1 dataset from `fchollet/ARC-AGI`.

**Spec:** `docs/superpowers/specs/2026-06-07-arc-latent-program-jepa-design.md`

---

## Conventions for this notebook plan

- **Single deliverable:** `arc_lp_jepa.ipynb` at repo root. Do **not** create or edit any `.py` file.
- Each task adds one or more **cells** to the notebook in order. "Verification" steps are cells containing `assert`s and/or plots; run them in the `ML` kernel.
- **Run cells with the `ML` env kernel.** When running the notebook headless to check a task, use:
  `C:/Users/Ous/miniconda3/envs/ML/python.exe -m jupyter nbconvert --to notebook --execute --inplace arc_lp_jepa.ipynb`
- **Commit** after each task: `git add arc_lp_jepa.ipynb` (the dataset under `data/` stays gitignored).
- **Global constants** (define once in Task 2, reused everywhere): `MAX_H = MAX_W = 30`, `NUM_COLORS = 10`, `PAD_ID = 10`, `VOCAB = 11`, `D = 256`, `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`.

---

## File / cell structure

The notebook is organized into titled markdown sections, one per task:

1. **Setup & imports** (Task 1)
2. **Dataset download** (Task 1)
3. **Tokenizer** (Task 2)
4. **Augmentation & batch protocol** (Task 3)
5. **Grid encoder `E`** (Task 4)
6. **Program encoder `P`** (Task 5)
7. **Predictor `F` + EMA target `Ē`** (Task 6)
8. **SIGReg-over-views + collapse monitor** (Task 7)
9. **LP-JEPA module + single-task overfit — GATE G1** (Task 8)
10. **Pretrain + TSNE — GATE G2** (Task 9)
11. **Test-time training harness — GATE G3** (Task 10)
12. **Decoder `G` + exact-match — GATE G4** (Task 11)
13. **Hard task + ablations — GATE G5** (Task 12)

---

## Task 1: Setup, imports, and ARC-AGI-1 download (Gate G0a)

**Files:**
- Create: `arc_lp_jepa.ipynb`

- [ ] **Step 1: Imports cell**

```python
import json, math, os, random, copy, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0); random.seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| torch:", torch.__version__)
```

- [ ] **Step 2: Download ARC-AGI-1 (shell cell)**

```python
import subprocess, os
ARC_DIR = Path("data/arc")
if not ARC_DIR.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/fchollet/ARC-AGI.git", str(ARC_DIR)],
        check=True,
    )
print("train tasks:", len(list((ARC_DIR / "data/training").glob("*.json"))))
print("eval  tasks:", len(list((ARC_DIR / "data/evaluation").glob("*.json"))))
```

- [ ] **Step 3: Loader cell**

```python
def load_arc_split(split):
    """Return dict task_id -> {'train': [(in,out),...], 'test': [(in,out),...]}."""
    out = {}
    for p in sorted((ARC_DIR / "data" / split).glob("*.json")):
        d = json.loads(p.read_text())
        out[p.stem] = {
            "train": [(np.array(e["input"]), np.array(e["output"])) for e in d["train"]],
            "test":  [(np.array(e["input"]), np.array(e["output"])) for e in d["test"]],
        }
    return out

TRAIN_TASKS = load_arc_split("training")
EVAL_TASKS  = load_arc_split("evaluation")
print(len(TRAIN_TASKS), len(EVAL_TASKS))
```

- [ ] **Step 4: Verify (run cell)**

```python
assert len(TRAIN_TASKS) == 400 and len(EVAL_TASKS) == 400
g = next(iter(TRAIN_TASKS.values()))["train"][0][0]
assert g.ndim == 2 and g.min() >= 0 and g.max() <= 9
print("OK: 400/400 tasks, grids are 2D int 0-9")
```

Expected: prints `OK: 400/400 tasks ...` with no assertion error.

- [ ] **Step 5: Commit**

```bash
git add arc_lp_jepa.ipynb
git commit -m "feat(arc): notebook setup + ARC-AGI-1 download/loader"
```

---

## Task 2: Grid tokenizer with round-trip (Gate G0b)

**Files:**
- Modify: `arc_lp_jepa.ipynb` (add tokenizer section)

- [ ] **Step 1: Constants + tokenizer cell**

```python
MAX_H = MAX_W = 30
NUM_COLORS = 10
PAD_ID = 10           # padding/color sentinel
VOCAB = 11            # colors 0..9 + PAD
N_CELLS = MAX_H * MAX_W          # 900
N_TOKENS = N_CELLS                # cells only; shape carried separately
D = 256

def pad_grid(grid):
    """np (h,w) int -> (padded (30,30) long, h, w). Out-of-grid cells = PAD_ID."""
    h, w = grid.shape
    assert 1 <= h <= MAX_H and 1 <= w <= MAX_W, f"grid {h}x{w} too big"
    out = np.full((MAX_H, MAX_W), PAD_ID, dtype=np.int64)
    out[:h, :w] = grid
    return torch.from_numpy(out), h, w

def grid_to_tokens(grid):
    """np (h,w) -> dict(tokens (900,), pad_mask (900,) True=pad, h, w)."""
    padded, h, w = pad_grid(grid)
    tokens = padded.reshape(-1)            # (900,)
    pad_mask = tokens.eq(PAD_ID)           # True where padding
    return {"tokens": tokens, "pad_mask": pad_mask,
            "h": torch.tensor(h), "w": torch.tensor(w)}

def tokens_to_grid(tokens, h, w):
    """tokens (900,) + shape -> np (h,w) int. Inverse of grid_to_tokens."""
    grid = tokens.reshape(MAX_H, MAX_W)[:h, :w]
    return grid.cpu().numpy().astype(np.int64)
```

- [ ] **Step 2: Round-trip verification cell**

```python
ok = 0
for t in list(TRAIN_TASKS.values())[:50]:
    for (gi, go) in t["train"]:
        for g in (gi, go):
            enc = grid_to_tokens(g)
            rt = tokens_to_grid(enc["tokens"], int(enc["h"]), int(enc["w"]))
            assert np.array_equal(rt, g), "round-trip mismatch"
            ok += 1
print(f"OK: {ok} grids round-tripped identically")
```

Expected: prints a positive count, no assertion error. **This is Gate G0.**

- [ ] **Step 3: Commit**

```bash
git add arc_lp_jepa.ipynb
git commit -m "feat(arc): grid tokenizer with identity round-trip (Gate G0)"
```

---

## Task 3: Augmentation + batch protocol (Gate G0c)

**Files:**
- Modify: `arc_lp_jepa.ipynb`

- [ ] **Step 1: Augmentation cell**

```python
def dihedral(grid, k):
    """k in 0..7: 4 rotations x optional flip. Returns np grid."""
    g = np.rot90(grid, k % 4)
    if k >= 4:
        g = np.fliplr(g)
    return np.ascontiguousarray(g)

def color_perm(grid, perm):
    """perm: length-10 array, a permutation of colors 0..9. Background 0 kept fixed."""
    return perm[grid]

def random_perm(keep_bg=True):
    p = np.arange(NUM_COLORS)
    rest = p[1:] if keep_bg else p
    np.random.shuffle(rest)
    if keep_bg:
        p = np.concatenate([[0], rest])
    else:
        p = rest
    return p

def augment_pair(inp, out, k=None, perm=None):
    """Apply the SAME dihedral + color perm to both halves of a pair."""
    if k is None: k = random.randint(0, 7)
    if perm is None: perm = random_perm()
    return color_perm(dihedral(inp, k), perm), color_perm(dihedral(out, k), perm)
```

- [ ] **Step 2: Batch builder cell (leave-one-out program protocol)**

```python
def encode_pair_grids(pair):
    """(in,out) np -> two token dicts."""
    gi, go = pair
    return grid_to_tokens(gi), grid_to_tokens(go)

def sample_episode(task, n_demo=None, augment=True):
    """Build one training episode from a task.
    Returns: demos (list of (in_tok,out_tok)), query (in_tok,out_tok).
    Query is one held-out pair; demos are the rest. With augmentation, a shared
    (k,perm) is drawn per episode so the program is invariant to it."""
    pairs = list(task["train"])
    random.shuffle(pairs)
    q = pairs[0]
    demos = pairs[1:]
    if n_demo is not None:
        demos = demos[:n_demo]
    if augment:
        k, perm = random.randint(0, 7), random_perm()
        demos = [augment_pair(i, o, k, perm) for (i, o) in demos]
        q = augment_pair(q[0], q[1], k, perm)
    demos_tok = [encode_pair_grids(p) for p in demos]
    q_tok = encode_pair_grids(q)
    return demos_tok, q_tok

def make_views(task, n_views=16):
    """Many augmented (in,out) token pairs of a task, for SIGReg batch dimension."""
    pairs = list(task["train"])
    views = []
    for _ in range(n_views):
        i, o = random.choice(pairs)
        ai, ao = augment_pair(i, o)
        views.append(encode_pair_grids((ai, ao)))
    return views
```

- [ ] **Step 3: Collate helper cell (stack token dicts to batched tensors)**

```python
def collate_tokens(token_dicts):
    """list of token dicts -> dict of stacked tensors on DEVICE."""
    return {
        "tokens":   torch.stack([t["tokens"]   for t in token_dicts]).to(DEVICE),
        "pad_mask": torch.stack([t["pad_mask"] for t in token_dicts]).to(DEVICE),
        "h":        torch.stack([t["h"]        for t in token_dicts]).to(DEVICE),
        "w":        torch.stack([t["w"]        for t in token_dicts]).to(DEVICE),
    }
```

- [ ] **Step 4: Verification cell**

```python
task = next(iter(TRAIN_TASKS.values()))
demos, q = sample_episode(task)
assert len(demos) >= 1 and len(q) == 2
b = collate_tokens([d[0] for d in demos])
assert b["tokens"].shape[1] == N_CELLS and b["pad_mask"].dtype == torch.bool
# augmentation preserves the I/O relation count
v = make_views(task, n_views=8)
assert len(v) == 8
print("OK: episodes, views, collation shapes valid")
```

Expected: `OK: episodes, views, collation shapes valid`.

- [ ] **Step 5: Commit**

```bash
git add arc_lp_jepa.ipynb
git commit -m "feat(arc): dihedral/color augmentation + leave-one-out batch protocol"
```

---

## Task 4: Grid encoder `E` (Gate G1a)

**Files:**
- Modify: `arc_lp_jepa.ipynb`

- [ ] **Step 1: Encoder cell**

```python
class GridEncoder(nn.Module):
    """ViT-style transformer over 30x30 color tokens -> pooled latent (B, D)."""
    def __init__(self, d=D, depth=4, heads=4, mlp_mult=4):
        super().__init__()
        self.color_emb = nn.Embedding(VOCAB, d)
        self.row_emb = nn.Embedding(MAX_H, d)
        self.col_emb = nn.Embedding(MAX_W, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * mlp_mult,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.tf = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d)
        rows = torch.arange(MAX_H).repeat_interleave(MAX_W)   # (900,)
        cols = torch.arange(MAX_W).repeat(MAX_H)              # (900,)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)

    def forward(self, tokens, pad_mask):
        # tokens (B,900) long, pad_mask (B,900) True=pad
        B = tokens.size(0)
        x = self.color_emb(tokens) + self.row_emb(self.rows) + self.col_emb(self.cols)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                       # (B, 901, D)
        # CLS never masked
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=tokens.device),
                          pad_mask], dim=1)
        x = self.tf(x, src_key_padding_mask=mask)
        return self.norm(x[:, 0])                            # (B, D)
```

- [ ] **Step 2: Verification cell**

```python
E = GridEncoder().to(DEVICE)
demos, q = sample_episode(next(iter(TRAIN_TASKS.values())))
qb = collate_tokens([q[0]])
a = E(qb["tokens"], qb["pad_mask"])
assert a.shape == (1, D) and torch.isfinite(a).all()
print("OK: GridEncoder ->", tuple(a.shape))
```

Expected: `OK: GridEncoder -> (1, 256)`.

- [ ] **Step 3: Commit**

```bash
git add arc_lp_jepa.ipynb
git commit -m "feat(arc): ViT-style grid encoder E"
```

---

## Task 5: Program encoder `P` (Gate G1b)

**Files:**
- Modify: `arc_lp_jepa.ipynb`

- [ ] **Step 1: Program encoder cell**

```python
class ProgramEncoder(nn.Module):
    """Encode one (input,output) demo pair -> z_i (B, D). Mean over pairs gives z."""
    def __init__(self, d=D, depth=4, heads=4, mlp_mult=4):
        super().__init__()
        self.color_emb = nn.Embedding(VOCAB, d)
        self.row_emb = nn.Embedding(MAX_H, d)
        self.col_emb = nn.Embedding(MAX_W, d)
        self.seg_emb = nn.Embedding(2, d)            # 0=input half, 1=output half
        self.cls = nn.Parameter(torch.zeros(1, 1, d)); nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * mlp_mult,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.tf = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d)
        rows = torch.arange(MAX_H).repeat_interleave(MAX_W)
        cols = torch.arange(MAX_W).repeat(MAX_H)
        self.register_buffer("rows", rows); self.register_buffer("cols", cols)

    def _embed(self, tokens, seg):
        return (self.color_emb(tokens) + self.row_emb(self.rows)
                + self.col_emb(self.cols) + self.seg_emb.weight[seg])

    def forward(self, in_tok, in_mask, out_tok, out_mask):
        # each (B,900). Concatenate input and output halves into one sequence.
        B = in_tok.size(0)
        xi = self._embed(in_tok, 0)
        xo = self._embed(out_tok, 1)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, xi, xo], dim=1)               # (B, 1801, D)
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=in_tok.device),
                          in_mask, out_mask], dim=1)
        x = self.tf(x, src_key_padding_mask=mask)
        return self.norm(x[:, 0])                          # (B, D)

def encode_program(P, demos_tok, exclude=None):
    """Mean of per-pair latents over a list of (in_tok,out_tok). Leave-one-out via
    `exclude` (index to drop). Returns (1, D)."""
    idxs = [i for i in range(len(demos_tok)) if i != exclude]
    ins  = collate_tokens([demos_tok[i][0] for i in idxs])
    outs = collate_tokens([demos_tok[i][1] for i in idxs])
    z_i = P(ins["tokens"], ins["pad_mask"], outs["tokens"], outs["pad_mask"])  # (k, D)
    return z_i.mean(0, keepdim=True)                                            # (1, D)
```

- [ ] **Step 2: Verification cell**

```python
P = ProgramEncoder().to(DEVICE)
demos, q = sample_episode(next(iter(TRAIN_TASKS.values())))
z = encode_program(P, demos)
assert z.shape == (1, D)
z_loo = encode_program(P, demos, exclude=0)   # leave-one-out path runs
assert z_loo.shape == (1, D)
print("OK: ProgramEncoder + mean/LOO ->", tuple(z.shape))
```

Expected: `OK: ProgramEncoder + mean/LOO -> (1, 256)`.

- [ ] **Step 3: Commit**

```bash
git add arc_lp_jepa.ipynb
git commit -m "feat(arc): program encoder P with mean + leave-one-out aggregation"
```

---

## Task 6: Predictor `F` + EMA target `Ē` (Gate G1c)

**Files:**
- Modify: `arc_lp_jepa.ipynb`

- [ ] **Step 1: Predictor cell (AdaLN conditioning on z)**

```python
class Predictor(nn.Module):
    """Apply program z to state a via AdaLN-zero modulation -> predicted out-latent."""
    def __init__(self, d=D, hidden=4 * D):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.film = nn.Linear(d, 2 * d)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, a, z):
        # a (B,D) state, z (B,D) program
        gamma, beta = self.film(z).chunk(2, dim=-1)
        h = self.norm(a) * (1 + gamma) + beta
        return a + self.mlp(h)                 # residual: start near identity
```

- [ ] **Step 2: EMA target cell**

```python
class EMA:
    """Maintains an EMA copy of a module; used as stop-grad target encoder."""
    def __init__(self, model, decay=0.996):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.ema.parameters(), model.parameters()):
            e.mul_(self.decay).add_(m, alpha=1 - self.decay)
        for e, m in zip(self.ema.buffers(), model.buffers()):
            e.copy_(m)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        return self.ema(*args, **kwargs)
```

- [ ] **Step 3: Verification cell**

```python
Fp = Predictor().to(DEVICE)
a = E(qb["tokens"], qb["pad_mask"])
pred = Fp(a, z)
assert pred.shape == (1, D)
Et = EMA(E)
t0 = Et(qb["tokens"], qb["pad_mask"]).clone()
# after an EMA update toward a (slightly) changed E, target stays finite & shaped
Et.update(E)
t1 = Et(qb["tokens"], qb["pad_mask"])
assert t1.shape == (1, D) and torch.isfinite(t1).all()
print("OK: Predictor + EMA target wired")
```

Expected: `OK: Predictor + EMA target wired`.

- [ ] **Step 4: Commit**

```bash
git add arc_lp_jepa.ipynb
git commit -m "feat(arc): AdaLN predictor F + EMA stop-grad target"
```

---

## Task 7: SIGReg-over-views + collapse monitor (Gate G1d)

**Files:**
- Modify: `arc_lp_jepa.ipynb`

- [ ] **Step 1: SIGReg function cell (operates on a flat batch of embeddings)**

```python
def sigreg_stat(z, num_proj=256, knots=17):
    """Epps-Pulley Gaussianity statistic over z (N, D). Larger = less Gaussian.
    Adapted from jepa.py SIGReg; here N is the augmentation-view batch."""
    N, Dz = z.shape
    t = torch.linspace(0, 3, knots, device=z.device)
    dt = 3 / (knots - 1)
    phi = torch.exp(-0.5 * t ** 2)
    w = torch.full((knots,), 2 * dt, device=z.device); w[[0, -1]] = dt
    w = w * phi
    A = torch.randn(Dz, num_proj, device=z.device)
    A = A / (A.norm(dim=0, keepdim=True) + 1e-6)
    x_t = (z @ A).unsqueeze(-1) * t                 # (N, num_proj, knots)
    err = (x_t.cos().mean(0) - phi).square() + x_t.sin().mean(0).square()  # (num_proj,knots)
    return ((err @ w) * N).mean()                   # scalar
```

- [ ] **Step 2: Collapse monitor cell**

```python
@torch.no_grad()
def collapse_stats(z):
    """z (N, D) -> dict of std/norm/cosine health metrics."""
    z = z.reshape(-1, z.size(-1))
    std = z.std(0).mean().item()
    norm = z.norm(dim=-1).mean().item()
    n = min(50, z.size(0) // 2)
    cos = F.cosine_similarity(z[:n], z[n:2 * n]).mean().item() if n > 0 else float("nan")
    return {"z_std": std, "z_norm": norm, "cos": cos}
```

- [ ] **Step 3: Verification cell**

```python
# views -> program latents -> SIGReg + collapse stats
task = next(iter(TRAIN_TASKS.values()))
views = make_views(task, n_views=24)
vin = collate_tokens([v[0] for v in views]); vout = collate_tokens([v[1] for v in views])
zv = P(vin["tokens"], vin["pad_mask"], vout["tokens"], vout["pad_mask"])  # (24, D)
s = sigreg_stat(zv); cs = collapse_stats(zv)
assert torch.isfinite(s) and cs["z_std"] > 0
print(f"OK: sigreg={s.item():.3f} stats={cs}")
```

Expected: prints a finite sigreg value and `z_std > 0`.

- [ ] **Step 4: Commit**

```bash
git add arc_lp_jepa.ipynb
git commit -m "feat(arc): SIGReg-over-views + collapse monitor"
```

---

## Task 8: LP-JEPA module + single-task latent overfit — GATE G1

**Files:**
- Modify: `arc_lp_jepa.ipynb`

- [ ] **Step 1: Assemble module cell**

```python
class LPJepa(nn.Module):
    """Bundles E, P, F and the EMA target for the latent (Phase 1) objective."""
    def __init__(self):
        super().__init__()
        self.E = GridEncoder()
        self.P = ProgramEncoder()
        self.F = Predictor()
        self.target = None   # set after .to(device)

    def init_target(self, decay=0.996):
        self.target = EMA(self.E, decay)

    def state(self, tok):                 # tok = collated dict
        return self.E(tok["tokens"], tok["pad_mask"])

    @torch.no_grad()
    def target_state(self, tok):
        return self.target(tok["tokens"], tok["pad_mask"])
```

- [ ] **Step 2: Episode-loss cell (leave-one-out latent objective)**

```python
def episode_loss(model, demos_tok, q_tok, lam=0.09, use_target=True):
    """Predict the query out-latent from query in-state + LOO program; latent MSE
    + SIGReg over the program views. demos_tok includes the query for LOO."""
    # program from demos excluding the query (query is appended as last, excluded)
    all_pairs = demos_tok + [q_tok]
    z = encode_program(model.P, all_pairs, exclude=len(all_pairs) - 1)   # (1, D)
    qin  = collate_tokens([q_tok[0]]); qout = collate_tokens([q_tok[1]])
    a = model.state(qin)                                                  # (1, D)
    pred = model.F(a, z)                                                  # (1, D)
    tgt = model.target_state(qout) if use_target else model.E(
        qout["tokens"], qout["pad_mask"]).detach()
    mse = F.mse_loss(pred, tgt)
    # SIGReg across per-pair program embeddings
    ins  = collate_tokens([p[0] for p in all_pairs])
    outs = collate_tokens([p[1] for p in all_pairs])
    z_all = model.P(ins["tokens"], ins["pad_mask"], outs["tokens"], outs["pad_mask"])
    reg = sigreg_stat(z_all) if z_all.size(0) >= 4 else torch.zeros((), device=pred.device)
    return mse + lam * reg, {"mse": mse.item(), "sigreg": float(reg)}
```

- [ ] **Step 3: Single-task overfit cell (Gate G1 driver)**

```python
def overfit_one_task(task, steps=400, lr=3e-4, lam=0.09):
    model = LPJepa().to(DEVICE); model.init_target()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    held = task["train"][-1]                      # a pair never used as query in training
    train_pairs = task["train"][:-1]
    losses = []
    for step in range(steps):
        random.shuffle(train_pairs)
        q = train_pairs[0]; demos = train_pairs[1:]
        demos_tok = [encode_pair_grids(p) for p in demos]
        q_tok = encode_pair_grids(q)
        loss, logs = episode_loss(model, demos_tok, q_tok, lam=lam)
        opt.zero_grad(); loss.backward(); opt.step(); model.target.update(model.E)
        losses.append(logs["mse"])
    # held-out latent error: program from ALL train pairs, predict the held pair
    model.eval()
    with torch.no_grad():
        demos_tok = [encode_pair_grids(p) for p in train_pairs]
        h_tok = encode_pair_grids(held)
        z = encode_program(model.P, demos_tok)
        a = model.state(collate_tokens([h_tok[0]]))
        pred = model.F(a, z)
        tgt = model.target_state(collate_tokens([h_tok[1]]))
        held_err = F.mse_loss(pred, tgt).item()
        cs = collapse_stats(z)
    return model, losses, held_err, cs

# pick a simple same-shape task by inspection helper:
def is_same_shape(task):
    return all(i.shape == o.shape for i, o in task["train"])

easy_id = next(tid for tid, t in TRAIN_TASKS.items()
               if is_same_shape(t) and len(t["train"]) >= 4)
print("easy task:", easy_id)
m, losses, held_err, cs = overfit_one_task(TRAIN_TASKS[easy_id])
plt.plot(losses); plt.title("train latent MSE"); plt.xlabel("step"); plt.show()
print(f"held-out latent err={held_err:.4f}  collapse={cs}")
```

- [ ] **Step 4: Gate G1 assertion cell**

```python
# G1 passes if training error fell substantially AND latents did not collapse.
assert losses[-1] < 0.5 * (sum(losses[:10]) / 10), "train MSE did not drop"
assert cs["z_std"] > 1e-2, "latents collapsed (z_std too small)"
assert cs["cos"] < 0.95, "latents collapsed (near-identical directions)"
print("GATE G1 PASS")
```

Expected: `GATE G1 PASS`. If it fails on collapse, lower `decay`, raise `lam`, or check the EMA target is being updated.

- [ ] **Step 5: Commit**

```bash
git add arc_lp_jepa.ipynb
git commit -m "feat(arc): LP-JEPA module + single-task latent overfit (Gate G1)"
```

---

## Task 9: Pretrain on training subset + TSNE — GATE G2

**Files:**
- Modify: `arc_lp_jepa.ipynb`

- [ ] **Step 1: Pretrain loop cell**

```python
def pretrain(task_ids, steps=4000, lr=3e-4, lam=0.09, log_every=200):
    model = LPJepa().to(DEVICE); model.init_target()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    tasks = [TRAIN_TASKS[t] for t in task_ids if len(TRAIN_TASKS[t]["train"]) >= 3]
    hist = []
    for step in range(steps):
        task = random.choice(tasks)
        demos, q = sample_episode(task, augment=True)
        loss, logs = episode_loss(model, demos, q, lam=lam)
        opt.zero_grad(); loss.backward(); opt.step(); model.target.update(model.E)
        if step % log_every == 0:
            hist.append((step, logs["mse"], logs["sigreg"]))
            print(f"step {step:5d}  mse {logs['mse']:.4f}  sigreg {logs['sigreg']:.3f}")
    return model, hist

# hold out a chunk of training tasks as 'unseen' for later TTT
all_ids = list(TRAIN_TASKS.keys())
random.shuffle(all_ids)
UNSEEN_IDS = all_ids[:40]
PRETRAIN_IDS = all_ids[40:240]      # 200 tasks for the prior
model_pt, hist = pretrain(PRETRAIN_IDS, steps=4000)
```

- [ ] **Step 2: TSNE-of-programs cell**

```python
from sklearn.manifold import TSNE

@torch.no_grad()
def collect_program_latents(model, task_ids, per_task=8):
    Z, labels = [], []
    for li, tid in enumerate(task_ids):
        t = TRAIN_TASKS[tid]
        for _ in range(per_task):
            demos, _ = sample_episode(t, augment=True)
            z = encode_program(model.P, demos)
            Z.append(z.cpu()); labels.append(li)
    return torch.cat(Z).numpy(), np.array(labels)

vis_ids = PRETRAIN_IDS[:12]
Z, labels = collect_program_latents(model_pt, vis_ids)
emb = TSNE(n_components=2, perplexity=15, init="pca").fit_transform(Z)
plt.figure(figsize=(7, 6))
plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab20", s=18)
plt.title("TSNE of program latents z (color = task)"); plt.show()
```

- [ ] **Step 3: Gate G2 clustering metric cell**

```python
from sklearn.metrics import silhouette_score
sil = silhouette_score(Z, labels)
print(f"silhouette (program clustering by task) = {sil:.3f}")
assert sil > 0.0, "no program clustering — prior too weak (consider more steps / re-arc)"
print("GATE G2 PASS")
```

Expected: a positive silhouette score and a TSNE plot showing same-task points grouping. Document the score in the markdown cell above the plot.

- [ ] **Step 4: Commit**

```bash
git add arc_lp_jepa.ipynb
git commit -m "feat(arc): pretrain loop + TSNE program clustering (Gate G2)"
```

---

## Task 10: Test-time training harness (no-freeze vs frozen+z-search vs scratch) — GATE G3

**Files:**
- Modify: `arc_lp_jepa.ipynb`

- [ ] **Step 1: Held-out query latent metric cell**

```python
@torch.no_grad()
def query_latent_err(model, task):
    """Program from ALL train pairs; predict the task's TEST pair out-latent."""
    demos_tok = [encode_pair_grids(p) for p in task["train"]]
    qi, qo = task["test"][0]
    q_tok = encode_pair_grids((qi, qo))
    z = encode_program(model.P, demos_tok)
    a = model.state(collate_tokens([q_tok[0]]))
    pred = model.F(a, z)
    tgt = model.target_state(collate_tokens([q_tok[1]])) if model.target else \
          model.E(*[collate_tokens([q_tok[1]])[k] for k in ("tokens", "pad_mask")])
    return F.mse_loss(pred, tgt).item()
```

- [ ] **Step 2: No-freeze TTT cell (the twist) with early-stop on a left-out demo**

```python
def ttt_no_freeze(base_model, task, steps=200, lr=1e-4, lam=0.09, patience=30):
    """Continue training weights on the task's augmented demos. Early-stop on a
    left-out demo's latent error (NOT train loss)."""
    model = copy.deepcopy(base_model); model.init_target()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    pairs = list(task["train"])
    val_pair = pairs[-1]; train_pairs = pairs[:-1]
    best, best_state, bad = float("inf"), None, 0
    for step in range(steps):
        random.shuffle(train_pairs)
        q = train_pairs[0]; demos = train_pairs[1:]
        k, perm = random.randint(0, 7), random_perm()
        demos = [augment_pair(i, o, k, perm) for i, o in demos]
        q = augment_pair(q[0], q[1], k, perm)
        loss, _ = episode_loss(model, [encode_pair_grids(p) for p in demos],
                               encode_pair_grids(q), lam=lam)
        opt.zero_grad(); loss.backward(); opt.step(); model.target.update(model.E)
        # validation on the left-out demo
        with torch.no_grad():
            dt = [encode_pair_grids(p) for p in train_pairs]
            vt = encode_pair_grids(val_pair)
            z = encode_program(model.P, dt)
            a = model.state(collate_tokens([vt[0]]))
            verr = F.mse_loss(model.F(a, z), model.target_state(
                collate_tokens([vt[1]]))).item()
        if verr < best:
            best, best_state, bad = verr, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model
```

- [ ] **Step 3: Frozen + z-gradient-ascent baseline cell (LPN)**

```python
def z_search(model, task, steps=200, lr=0.1):
    """Freeze weights; optimize z to reduce demo reconstruction error in latent
    space, then apply to the query. Leakage-free (uses only demo outputs)."""
    model = model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    demos = [encode_pair_grids(p) for p in task["train"]]
    z = encode_program(model.P, demos).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    ins  = collate_tokens([d[0] for d in demos])
    outs = collate_tokens([d[1] for d in demos])
    with torch.no_grad():
        a_demo = model.E(ins["tokens"], ins["pad_mask"])             # (k, D)
        t_demo = model.target_state(outs) if model.target else \
                 model.E(outs["tokens"], outs["pad_mask"])
    for _ in range(steps):
        pred = model.F(a_demo, z.expand(a_demo.size(0), -1))
        loss = F.mse_loss(pred, t_demo)
        opt.zero_grad(); loss.backward(); opt.step()
    # apply optimized z to the test query
    qi, qo = task["test"][0]; q_tok = encode_pair_grids((qi, qo))
    with torch.no_grad():
        a = model.state(collate_tokens([q_tok[0]]))
        pred = model.F(a, z)
        tgt = model.target_state(collate_tokens([q_tok[1]]))
        return F.mse_loss(pred, tgt).item()
```

- [ ] **Step 4: Compare three regimes cell (Gate G3 driver)**

```python
results = {"frozen": [], "z_search": [], "no_freeze": [], "scratch": []}
for tid in UNSEEN_IDS[:10]:
    t = TRAIN_TASKS[tid]
    if len(t["train"]) < 3: continue
    results["frozen"].append(query_latent_err(model_pt, t))
    results["z_search"].append(z_search(copy.deepcopy(model_pt), t))
    results["no_freeze"].append(query_latent_err(ttt_no_freeze(model_pt, t), t))
    scratch = LPJepa().to(DEVICE); scratch.init_target()
    results["scratch"].append(query_latent_err(ttt_no_freeze(scratch, t), t))
for k, v in results.items():
    print(f"{k:10s}  mean query latent err = {np.mean(v):.4f}")
```

- [ ] **Step 5: Gate G3 assertion cell**

```python
assert np.mean(results["no_freeze"]) <= np.mean(results["frozen"]), \
    "no-freeze did not beat frozen"
print("GATE G3 PASS — no-freeze >= frozen baseline on latent metric")
print("(also compare to z_search and scratch above)")
```

Expected: `GATE G3 PASS`. Record the four means in a markdown cell as the headline Phase-1 result.

- [ ] **Step 6: Commit**

```bash
git add arc_lp_jepa.ipynb
git commit -m "feat(arc): TTT harness — no-freeze vs z-search vs scratch (Gate G3)"
```

---

## Task 11: Decoder `G` + exact-match (Phase 2) — GATE G4

**Files:**
- Modify: `arc_lp_jepa.ipynb`

- [ ] **Step 1: Decoder cell (non-autoregressive, z-conditioned)**

```python
class GridDecoder(nn.Module):
    """z + input grid -> output shape logits + per-cell color logits (parallel)."""
    def __init__(self, d=D, depth=4, heads=4, mlp_mult=4):
        super().__init__()
        self.color_emb = nn.Embedding(VOCAB, d)
        self.row_emb = nn.Embedding(MAX_H, d); self.col_emb = nn.Embedding(MAX_W, d)
        self.film = nn.Linear(d, 2 * d); nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        layer = nn.TransformerEncoderLayer(d, heads, d * mlp_mult,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, depth)
        self.color_head = nn.Linear(d, NUM_COLORS)
        self.h_head = nn.Linear(d, MAX_H); self.w_head = nn.Linear(d, MAX_W)
        self.shape_tok = nn.Parameter(torch.zeros(1, 1, d)); nn.init.normal_(self.shape_tok, std=0.02)
        rows = torch.arange(MAX_H).repeat_interleave(MAX_W)
        cols = torch.arange(MAX_W).repeat(MAX_H)
        self.register_buffer("rows", rows); self.register_buffer("cols", cols)

    def forward(self, in_tok, in_mask, z):
        B = in_tok.size(0)
        x = self.color_emb(in_tok) + self.row_emb(self.rows) + self.col_emb(self.cols)
        gamma, beta = self.film(z).chunk(2, dim=-1)
        x = x * (1 + gamma).unsqueeze(1) + beta.unsqueeze(1)        # condition on z
        st = self.shape_tok.expand(B, -1, -1)
        x = torch.cat([st, x], dim=1)
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=in_tok.device), in_mask], 1)
        x = self.tf(x, src_key_padding_mask=mask)
        shape_feat, cell_feat = x[:, 0], x[:, 1:]
        return self.h_head(shape_feat), self.w_head(shape_feat), self.color_head(cell_feat)
```

- [ ] **Step 2: Decoder loss + exact-match cell**

```python
def decode_loss(G, z, in_tok_dict, out_grid_dict):
    h_logit, w_logit, color_logit = G(in_tok_dict["tokens"], in_tok_dict["pad_mask"], z)
    h_t = out_grid_dict["h"] - 1; w_t = out_grid_dict["w"] - 1     # 0-indexed targets
    loss_shape = F.cross_entropy(h_logit, h_t) + F.cross_entropy(w_logit, w_t)
    valid = ~out_grid_dict["pad_mask"]                            # (B,900)
    color_t = out_grid_dict["tokens"].clamp(max=NUM_COLORS - 1)
    ce = F.cross_entropy(color_logit.reshape(-1, NUM_COLORS),
                         color_t.reshape(-1), reduction="none").reshape(color_t.shape)
    loss_color = (ce * valid).sum() / valid.sum().clamp(min=1)
    return loss_shape + loss_color

@torch.no_grad()
def exact_match(G, z, in_tok_dict, target_grid):
    h_logit, w_logit, color_logit = G(in_tok_dict["tokens"], in_tok_dict["pad_mask"], z)
    h = int(h_logit.argmax()) + 1; w = int(w_logit.argmax()) + 1
    pred = color_logit.argmax(-1)[0]                              # (900,)
    grid = tokens_to_grid(pred, h, w)
    return int(grid.shape == target_grid.shape and np.array_equal(grid, target_grid)), grid
```

- [ ] **Step 3: Train decoder on the easy task (Gate G4 driver)**

```python
def train_decoder_on_task(model, task, steps=600, lr=3e-4):
    G = GridDecoder().to(DEVICE)
    opt = torch.optim.AdamW(G.parameters(), lr=lr)
    pairs = list(task["train"])
    for step in range(steps):
        i, o = random.choice(pairs)
        ai, ao = augment_pair(i, o)
        with torch.no_grad():
            demos = [encode_pair_grids(p) for p in pairs]
            z = encode_program(model.P, demos)
        in_d = collate_tokens([grid_to_tokens(ai)])
        out_d = collate_tokens([grid_to_tokens(ao)])
        loss = decode_loss(G, z, in_d, out_d)
        opt.zero_grad(); loss.backward(); opt.step()
    return G

m_easy = ttt_no_freeze(model_pt, TRAIN_TASKS[easy_id])
G_easy = train_decoder_on_task(m_easy, TRAIN_TASKS[easy_id])
qi, qo = TRAIN_TASKS[easy_id]["test"][0]
with torch.no_grad():
    demos = [encode_pair_grids(p) for p in TRAIN_TASKS[easy_id]["train"]]
    z = encode_program(m_easy.P, demos)
    em, grid = exact_match(G_easy, z, collate_tokens([grid_to_tokens(qi)]), qo)
print("easy-task exact match on held-out test pair:", em)
```

- [ ] **Step 4: Plot predicted vs target grid cell**

```python
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
ax[0].imshow(qi, cmap="tab10", vmin=0, vmax=9); ax[0].set_title("input")
ax[1].imshow(grid, cmap="tab10", vmin=0, vmax=9); ax[1].set_title("predicted")
ax[2].imshow(qo, cmap="tab10", vmin=0, vmax=9); ax[2].set_title("target")
for a in ax: a.axis("off")
plt.show()
```

- [ ] **Step 5: Gate G4 assertion cell**

```python
# G4 passes if the decoder produces a non-degenerate grid of the right shape.
assert grid.shape == qo.shape, "predicted shape wrong"
assert len(np.unique(grid)) > 1, "decoder collapsed to a single color"
print("GATE G4 PASS (exact_match =", em, ")")
```

Expected: a correct-shape, multi-color prediction; `exact_match` may be 0 or 1 — record it. Gate is about a working decoder, not guaranteed solve.

- [ ] **Step 6: Commit**

```bash
git add arc_lp_jepa.ipynb
git commit -m "feat(arc): z-conditioned grid decoder + exact-match (Gate G4)"
```

---

## Task 12: Hard task + ablations — GATE G5

**Files:**
- Modify: `arc_lp_jepa.ipynb`

- [ ] **Step 1: Pick a hard task + config cell**

```python
# hard = compositional / shape-changing task with enough demos
hard_id = next(tid for tid, t in TRAIN_TASKS.items()
               if not is_same_shape(t) and len(t["train"]) >= 4 and tid in UNSEEN_IDS) \
          if any(not is_same_shape(TRAIN_TASKS[t]) for t in UNSEEN_IDS) else \
          next(tid for tid, t in TRAIN_TASKS.items() if not is_same_shape(t) and len(t["train"]) >= 4)
print("hard task:", hard_id)

ABLATIONS = [
    {"name": "full",        "lam": 0.09, "use_target": True},
    {"name": "no_target",   "lam": 0.09, "use_target": False},
    {"name": "lam_0",       "lam": 0.0,  "use_target": True},
    {"name": "lam_0.3",     "lam": 0.3,  "use_target": True},
]
```

- [ ] **Step 2: Ablation runner cell**

```python
def run_ablation(cfg, task_ids, eval_ids, steps=2000):
    model = LPJepa().to(DEVICE); model.init_target()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    tasks = [TRAIN_TASKS[t] for t in task_ids if len(TRAIN_TASKS[t]["train"]) >= 3]
    for step in range(steps):
        task = random.choice(tasks)
        demos, q = sample_episode(task, augment=True)
        loss, _ = episode_loss(model, demos, q, lam=cfg["lam"],
                               use_target=cfg["use_target"])
        opt.zero_grad(); loss.backward(); opt.step()
        if cfg["use_target"]: model.target.update(model.E)
    errs = [query_latent_err(model, TRAIN_TASKS[t]) for t in eval_ids
            if len(TRAIN_TASKS[t]["train"]) >= 3]
    z, labels = collect_program_latents(model, task_ids[:8], per_task=6)
    from sklearn.metrics import silhouette_score
    return {"name": cfg["name"], "query_err": float(np.mean(errs)),
            "silhouette": float(silhouette_score(z, labels)),
            "collapse": collapse_stats(torch.tensor(z))}
```

- [ ] **Step 3: Run all ablations + results table cell (Gate G5 driver)**

```python
rows = [run_ablation(c, PRETRAIN_IDS[:60], UNSEEN_IDS[:8], steps=1500)
        for c in ABLATIONS]
print(f"{'config':12s} {'query_err':>10s} {'silhouette':>11s} {'z_std':>8s}")
for r in rows:
    print(f"{r['name']:12s} {r['query_err']:>10.4f} {r['silhouette']:>11.3f} "
          f"{r['collapse']['z_std']:>8.3f}")
```

- [ ] **Step 4: Easy-vs-hard exact-match summary cell**

```python
summary = {}
for tag, tid in [("easy", easy_id), ("hard", hard_id)]:
    m = ttt_no_freeze(model_pt, TRAIN_TASKS[tid])
    G = train_decoder_on_task(m, TRAIN_TASKS[tid])
    qi, qo = TRAIN_TASKS[tid]["test"][0]
    demos = [encode_pair_grids(p) for p in TRAIN_TASKS[tid]["train"]]
    with torch.no_grad():
        z = encode_program(m.P, demos)
        em, _ = exact_match(G, z, collate_tokens([grid_to_tokens(qi)]), qo)
    summary[tag] = em
print("exact-match summary:", summary)
```

- [ ] **Step 5: Gate G5 assertion cell**

```python
# G5 passes when the ablation table and easy/hard summary are populated and the
# 'full' config does not collapse (z_std healthy).
full = next(r for r in rows if r["name"] == "full")
assert full["collapse"]["z_std"] > 1e-2, "full config collapsed"
assert set(summary.keys()) == {"easy", "hard"}
print("GATE G5 PASS — ablation table + easy/hard exact-match recorded")
```

Expected: `GATE G5 PASS`, an ablation table, and easy/hard exact-match numbers. Write a final markdown cell interpreting: did stop-grad/EMA matter? which λ? did no-freeze beat z-search? easy vs hard.

- [ ] **Step 6: Commit**

```bash
git add arc_lp_jepa.ipynb
git commit -m "feat(arc): hard task + anti-collapse ablations + results (Gate G5)"
```

---

## Self-review notes (coverage vs spec)

- Spec §4 architecture (E, P, F, Ē, G) → Tasks 4,5,6,11. Two-encoder design honored.
- Spec §5 losses (latent MSE + SIGReg; Phase-2 CE) → Tasks 8,11.
- Spec §6 anti-collapse (EMA/stop-grad, SIGReg-over-views, λ sweep, toggles) → Tasks 6,7,12.
- Spec §7 data/protocol (download, dihedral+color aug, leave-one-out, splits) → Tasks 1,3,9.
- Spec §8 training (pretrain, no-freeze TTT w/ early-stop, z-search, scratch) → Tasks 9,10.
- Spec §9 eval/viz (latent err, TSNE, z-search overlay, exact-match, grid plots) → Tasks 9,10,11,12.
- Spec §10 gates G0–G5 → Tasks 2,8,9,10,11,12.
- Constraint "one notebook, no .py edits" honored throughout; constants defined once (Task 2) and reused.
- Open risk (spec §11): if G2 silhouette is poor, the plan flags re-arc as the follow-up; if G1 collapses, tune decay/λ.
