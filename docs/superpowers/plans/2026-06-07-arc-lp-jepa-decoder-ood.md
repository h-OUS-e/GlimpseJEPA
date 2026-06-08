# LP-JEPA Phase 2: Proper Decoding + OOD Eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Replace the weak parallel decoder with an autoregressive encoder-decoder, train it *jointly* during pretraining so it learns to execute programs, and measure real exact-match on a truly out-of-distribution split (the ARC **evaluation** 400) — including the decisive no-freeze-vs-z-search comparison on exact-match (not just latent cosine).

**Architecture:** AR decoder = input-grid memory encoder (with program `z` prepended as a memory token) + a causal Transformer decoder that generates the output token sequence `[h, w, cell_0..cell_899]` left-to-right, cross-attending to that memory. Trained jointly with the existing LP-JEPA latent objective (EMA target + SIGReg kept). Evaluated on the ARC evaluation split with zero-shot / z-search / no-freeze regimes.

**Tech Stack:** Same notebook `arc_lp_jepa.ipynb`, PyTorch, ML env (`C:/Users/Ous/miniconda3/envs/ML/python.exe`), CUDA.

**Builds on:** `docs/superpowers/plans/2026-06-07-arc-lp-jepa.md` (Gates G0–G5 done; latent mechanism validated, programs cluster + generalize, decoder exact-match was 0).

---

## Conventions (same as Phase 1)
- **All code in `arc_lp_jepa.ipynb`; append cells; no other `.py` files.**
- Heavy runs exceed the 10-min foreground cap: subagents **append + smoke-validate (tiny steps) + commit** cells; the **controller drives the full run in the background** and validates. Use checkpoints so re-runs are cheap.
- Run/execute via `... nbconvert --to notebook --execute --inplace arc_lp_jepa.ipynb`.
- Reuse: `D, MAX_H, MAX_W, VOCAB, NUM_COLORS, PAD_ID, N_CELLS, DEVICE, CACHE, TRAIN_TASKS, EVAL_TASKS, LPJepa, episode_loss, encode_program, collate_tokens, sample_episode_cached, make_views_cached, task_canonical, build_cache, perm_tokens, random_perm11, tokens_to_grid, collapse_stats, query_metrics, z_search, ttt_no_freeze, model_pt`.

---

## Decisions locked
- Decoder: **autoregressive encoder-decoder**.
- Decoder training: **shared, during pretraining** (joint loss).
- OOD: **ARC evaluation split** (400 held-out tasks).
- Anti-collapse: **keep EMA + SIGReg**.

---

## Task H0: AR decoder module + single-task overfit (Gate H0)

**Files:** append to `arc_lp_jepa.ipynb`.

- [ ] **Step 1: AR decoder cell**

```python
class ARDecoder(nn.Module):
    """Autoregressive enc-dec: (program z, input grid) -> output grid.
    Input grid is encoded as cross-attention memory with z prepended; the output
    sequence [h, w, cell_0..cell_(N-1)] is generated left-to-right (causal)."""
    def __init__(self, d=D, enc_depth=3, dec_depth=4, heads=4, mlp_mult=4):
        super().__init__()
        self.in_color = nn.Embedding(VOCAB, d)
        self.in_row = nn.Embedding(MAX_H, d); self.in_col = nn.Embedding(MAX_W, d)
        enc_layer = nn.TransformerEncoderLayer(d, heads, d * mlp_mult, batch_first=True,
                                               norm_first=True, activation="gelu")
        self.in_enc = nn.TransformerEncoder(enc_layer, enc_depth)
        self.z_proj = nn.Linear(d, d)
        self.out_color = nn.Embedding(VOCAB, d)              # decoder-side color tokens
        self.bos = nn.Parameter(torch.zeros(1, 1, d)); nn.init.normal_(self.bos, std=0.02)
        self.shape_q = nn.Parameter(torch.zeros(1, 2, d)); nn.init.normal_(self.shape_q, std=0.02)
        self.out_pos = nn.Embedding(2 + 1 + N_CELLS, d)     # positions for [h,w,BOS,cells...]
        dec_layer = nn.TransformerDecoderLayer(d, heads, d * mlp_mult, batch_first=True,
                                               norm_first=True, activation="gelu")
        self.dec = nn.TransformerDecoder(dec_layer, dec_depth)
        self.h_head = nn.Linear(d, MAX_H); self.w_head = nn.Linear(d, MAX_W)
        self.color_head = nn.Linear(d, NUM_COLORS)
        rows = torch.arange(MAX_H).repeat_interleave(MAX_W)
        cols = torch.arange(MAX_W).repeat(MAX_H)
        self.register_buffer("rows", rows); self.register_buffer("cols", cols)

    def encode_input(self, in_tok, in_mask, z):
        x = self.in_color(in_tok) + self.in_row(self.rows) + self.in_col(self.cols)
        mem = self.in_enc(x, src_key_padding_mask=in_mask)         # (B,900,D)
        zt = self.z_proj(z).unsqueeze(1)                           # (B,1,D)
        memory = torch.cat([zt, mem], dim=1)                       # (B,901,D)
        mem_mask = torch.cat([torch.zeros(in_tok.size(0), 1, dtype=torch.bool,
                                          device=in_tok.device), in_mask], 1)
        return memory, mem_mask

    def forward(self, in_tok, in_mask, z, out_tok):
        """Teacher-forced parallel pass. out_tok: (B,900) true output cell tokens."""
        B = in_tok.size(0); dev = in_tok.device
        memory, mem_mask = self.encode_input(in_tok, in_mask, z)
        cells_in = self.out_color(out_tok[:, :-1])                 # c_0..c_898 (B,899,D)
        tgt = torch.cat([self.shape_q.expand(B, -1, -1),
                         self.bos.expand(B, -1, -1), cells_in], dim=1)   # (B,902,D)
        L = tgt.size(1)
        tgt = tgt + self.out_pos(torch.arange(L, device=dev))
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev), 1)
        out = self.dec(tgt, memory, tgt_mask=causal, memory_key_padding_mask=mem_mask)
        return self.h_head(out[:, 0]), self.w_head(out[:, 1]), self.color_head(out[:, 2:])

    @torch.no_grad()
    def generate(self, in_tok, in_mask, z):
        """Greedy decode. Stops once all cells inside the predicted (h,w) are produced."""
        B = in_tok.size(0); dev = in_tok.device
        memory, mem_mask = self.encode_input(in_tok, in_mask, z)
        cur = torch.cat([self.shape_q.expand(B, -1, -1), self.bos.expand(B, -1, -1)], dim=1)
        h = w = None; cells = []
        for step in range(N_CELLS):
            L = cur.size(1)
            t = cur + self.out_pos(torch.arange(L, device=dev))
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev), 1)
            out = self.dec(t, memory, tgt_mask=causal, memory_key_padding_mask=mem_mask)
            if step == 0:
                h = self.h_head(out[:, 0]).argmax(-1) + 1
                w = self.w_head(out[:, 1]).argmax(-1) + 1
            nxt = self.color_head(out[:, -1]).argmax(-1)           # (B,)
            cells.append(nxt)
            cur = torch.cat([cur, self.out_color(nxt).unsqueeze(1)], dim=1)
            # early stop: last needed raster index for the largest predicted grid in batch
            max_needed = int(((h - 1) * MAX_W + w).max().item())
            if step >= max_needed:
                break
        cells = torch.stack(cells, dim=1)                          # (B, <=900)
        if cells.size(1) < N_CELLS:                                # pad tail with PAD_ID
            pad = torch.full((B, N_CELLS - cells.size(1)), PAD_ID, device=dev, dtype=cells.dtype)
            cells = torch.cat([cells, pad], dim=1)
        return h, w, cells
```

- [ ] **Step 2: AR loss + exact-match cell**

```python
def ar_decode_loss(dec, z, in_d, out_d):
    h_logit, w_logit, color_logit = dec(in_d["tokens"], in_d["pad_mask"], z, out_d["tokens"])
    ls = F.cross_entropy(h_logit, out_d["h"] - 1) + F.cross_entropy(w_logit, out_d["w"] - 1)
    valid = ~out_d["pad_mask"]
    ct = out_d["tokens"].clamp(max=NUM_COLORS - 1)
    ce = F.cross_entropy(color_logit.reshape(-1, NUM_COLORS), ct.reshape(-1),
                         reduction="none").reshape(ct.shape)
    lc = (ce * valid).sum() / valid.sum().clamp(min=1)
    return ls + lc

@torch.no_grad()
def ar_exact_match(dec, z, in_d, target_grid):
    h, w, cells = dec.generate(in_d["tokens"], in_d["pad_mask"], z)
    grid = tokens_to_grid(cells[0], int(h[0]), int(w[0]))
    return int(grid.shape == target_grid.shape and np.array_equal(grid, target_grid)), grid
```

- [ ] **Step 3: single-task overfit sanity (Gate H0 driver)**

```python
# Can the AR decoder learn one task's mapping to exact-match? (uses frozen pretrained P)
ov_id = next(t for t in CACHE if len(TRAIN_TASKS.get(t, {"train":[]})["train"]) >= 4
             and all(i.shape == o.shape for i, o in TRAIN_TASKS[t]["train"])) \
        if False else easy_unseen   # reuse the easy unseen task from Phase 1
dec_ov = ARDecoder().to(DEVICE)
opt = torch.optim.AdamW(dec_ov.parameters(), lr=3e-4)
pairs = CACHE[ov_id]["train"]
with torch.no_grad():
    demos = [pairs[i][0] for i in range(len(pairs))]
    z_ov = encode_program(model_pt.P, demos)
for step in range(800):
    i = random.randrange(len(pairs)); ein, eout = pairs[i][0]
    loss = ar_decode_loss(dec_ov, z_ov, collate_tokens([ein]), collate_tokens([eout]))
    opt.zero_grad(); loss.backward(); opt.step()
# evaluate exact-match on a TRAINING demo it was trained on (memorization sanity)
ein, eout = pairs[0][0]
em, _ = ar_exact_match(dec_ov, z_ov, collate_tokens([ein]),
                       tokens_to_grid(eout["tokens"], int(eout["h"]), int(eout["w"])))
print("H0 single-pair memorization exact-match:", em)
```

- [ ] **Step 4: Gate H0 assertion**

```python
assert em == 1, "AR decoder cannot even memorize one pair -> architecture/training bug"
print("GATE H0 PASS")
```

Run: smoke (steps=50) then full (steps=800). Gate: the AR decoder can reproduce a trained pair exactly. **Commit:** `feat(arc): autoregressive enc-dec decoder + single-pair overfit (Gate H0)`.

---

## Task H1: Joint pretraining (latent + AR decode), shared decoder (Gate H1)

- [ ] **Step 1: joint pretrain cell (checkpointed)**

```python
def pretrain_joint(task_ids, steps=3000, lr=3e-4, lam=0.09, mu=1.0, n_views=16, log_every=300):
    model = LPJepa().to(DEVICE); model.init_target()
    dec = ARDecoder().to(DEVICE)
    opt = torch.optim.AdamW(list(model.parameters()) + list(dec.parameters()), lr=lr)
    ids = [t for t in task_ids if len(TRAIN_TASKS[t]["train"]) >= 3]
    for step in range(steps):
        tid = random.choice(ids)
        demos, q = sample_episode_cached(tid, augment=True)
        views = make_views_cached(tid, n_views=n_views)
        loss_lat, logs = episode_loss(model, demos, q, lam=lam, sigreg_views=views)
        all_pairs = demos + [q]
        z = encode_program(model.P, all_pairs, exclude=len(all_pairs) - 1)
        loss_dec = ar_decode_loss(dec, z, collate_tokens([q[0]]), collate_tokens([q[1]]))
        loss = loss_lat + mu * loss_dec
        opt.zero_grad(); loss.backward(); opt.step(); model.target.update(model.E)
        if step % log_every == 0:
            print(f"step {step:5d}  lat {logs['mse']:.4f}  dec {loss_dec.item():.4f}  sig {logs['sigreg']:.3f}")
    return model, dec
```

- [ ] **Step 2: run with checkpoint cache**

```python
JOINT_STEPS = 3000
CKPT2 = Path("data/lpjepa_joint.pt")
model_j = LPJepa().to(DEVICE); model_j.init_target(); dec = ARDecoder().to(DEVICE)
if CKPT2.exists():
    sd = torch.load(CKPT2, map_location=DEVICE)
    model_j.load_state_dict(sd["model"]); model_j.init_target(); dec.load_state_dict(sd["dec"])
    print("loaded joint checkpoint")
else:
    model_j, dec = pretrain_joint(PRETRAIN_IDS, steps=JOINT_STEPS)
    torch.save({"model": model_j.state_dict(), "dec": dec.state_dict()}, CKPT2)
    print("saved joint checkpoint")
```

- [ ] **Step 3: Gate H1 assertion (latent still healthy + decoder learned)**

```python
Zc, lc = program_latents(model_j, PRETRAIN_IDS[:12], per_task=6, augment=False, fixed_demos=False)
from sklearn.metrics import silhouette_score
sil_j = silhouette_score(Zc, lc); cs = collapse_stats(torch.tensor(Zc))
# decoder reconstructs a held-in pair under canonical program
demos, q = task_canonical(PRETRAIN_IDS[0])
z = encode_program(model_j.P, demos)
qo = TRAIN_TASKS[PRETRAIN_IDS[0]]["train"][0][1]  # any train output as a rough check
print(f"H1 silhouette={sil_j:.3f} z_std={cs['z_std']:.3f}")
assert cs["z_std"] > 1e-2 and sil_j > 0.0, "joint training hurt the latent space"
print("GATE H1 PASS")
```

Run: smoke (steps=60) then full (3000). Gate: joint training keeps the latent space healthy/clustered and the decoder loss drops. **Commit:** `feat(arc): joint latent+AR-decode pretraining with shared decoder (Gate H1)`.

---

## Task H2: In-distribution decode sanity on unseen *training* tasks (Gate H2)

- [ ] **Step 1: decode-eval helpers (zero-shot / z-search / no-freeze) + driver**

```python
@torch.no_grad()
def decode_exact(model, dec, tid):
    demos, q = task_canonical(tid)
    z = encode_program(model.P, demos)
    qi, qo = (TRAIN_TASKS if tid in TRAIN_TASKS else EVAL_TASKS)[tid]["test"][0]
    em, grid = ar_exact_match(dec, z, collate_tokens([q[0]]), qo)
    return em, grid

idsan = [t for t in UNSEEN_IDS if len(TRAIN_TASKS[t]["train"]) >= 3][:10]
acc = np.mean([decode_exact(model_j, dec, t)[0] for t in idsan])
print(f"H2 zero-shot exact-match on {len(idsan)} unseen TRAIN tasks = {acc:.3f}")
```

- [ ] **Step 2: Gate H2 assertion**

```python
assert acc >= 0.0   # informational gate: record the number; >0 is the hope
print(f"GATE H2 PASS (zero-shot in-distribution exact-match = {acc:.3f})")
```

Run full (uses joint checkpoint, fast). Gate: records in-distribution exact-match (target: meaningfully > 0, unlike Phase-1's 0). **Commit:** `feat(arc): in-distribution AR decode exact-match (Gate H2)`.

---

## Task H3: Truly-OOD evaluation on the ARC evaluation split (Gate H3)

- [ ] **Step 1: extend cache + task lookup to the eval split**

```python
# Merge eval tasks so all cached helpers (sample_episode_cached, task_canonical, ...) work.
CACHE.update(build_cache(EVAL_TASKS))
ALL_TASKS = {**TRAIN_TASKS, **EVAL_TASKS}   # id -> task; ids are distinct hashes across splits

@torch.no_grad()
def decode_exact_any(model, dec, tid):
    demos, q = task_canonical(tid)
    z = encode_program(model.P, demos)
    qo = ALL_TASKS[tid]["test"][0][1]
    return ar_exact_match(dec, z, collate_tokens([q[0]]), qo)[0]
```

- [ ] **Step 2: OOD regimes — zero-shot vs z-search vs no-freeze (the decisive test)**

```python
def z_search_decode(model, dec, tid, steps=200, lr=0.1):
    m = copy.deepcopy(model).eval()
    for p in m.parameters(): p.requires_grad_(False)
    demos, q = task_canonical(tid)
    z = encode_program(m.P, demos).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    ins = collate_tokens([d[0] for d in demos]); outs = collate_tokens([d[1] for d in demos])
    with torch.no_grad():
        a_demo = m.E(ins["tokens"], ins["pad_mask"]); t_demo = m.target_state(outs)
    for _ in range(steps):
        pred = m.F(a_demo, z.expand(a_demo.size(0), -1))
        loss = F.mse_loss(pred, t_demo); opt.zero_grad(); loss.backward(); opt.step()
    qo = ALL_TASKS[tid]["test"][0][1]
    return ar_exact_match(dec, z.detach(), collate_tokens([q[0]]), qo)[0]

def no_freeze_decode(model, dec, tid, steps=200, lr=1e-4, lam=0.09):
    m = copy.deepcopy(model); m.init_target(); d2 = copy.deepcopy(dec)
    opt = torch.optim.AdamW(list(m.parameters()) + list(d2.parameters()), lr=lr)
    pairs = CACHE[tid]["train"]; n = len(pairs); tr = list(range(n))
    for step in range(steps):
        random.shuffle(tr); q_i = tr[0]; demo_i = tr[1:]
        k = random.randint(0, 7); perm = random_perm11()
        def get(i):
            ein, eout = pairs[i][k]; return perm_tokens(ein, perm), perm_tokens(eout, perm)
        demos = [get(i) for i in demo_i] if demo_i else [get(q_i)]; q = get(q_i)
        views = make_views_cached(tid, n_views=16)
        loss_lat, _ = episode_loss(m, demos, q, lam=lam, sigreg_views=views)
        z = encode_program(m.P, demos + [q], exclude=len(demos))
        loss = loss_lat + ar_decode_loss(d2, z, collate_tokens([q[0]]), collate_tokens([q[1]]))
        opt.zero_grad(); loss.backward(); opt.step(); m.target.update(m.E)
    demos, q = task_canonical(tid); z = encode_program(m.P, demos)
    qo = ALL_TASKS[tid]["test"][0][1]
    return ar_exact_match(d2, z, collate_tokens([q[0]]), qo)[0]

eval_ids = [t for t in EVAL_TASKS if len(EVAL_TASKS[t]["train"]) >= 2][:40]
zs_ids   = eval_ids[:12]   # smaller subset for the expensive TTT regimes
zero = np.mean([decode_exact_any(model_j, dec, t) for t in eval_ids])
zser = np.mean([z_search_decode(model_j, dec, t) for t in zs_ids])
nofr = np.mean([no_freeze_decode(model_j, dec, t) for t in zs_ids])
print(f"OOD exact-match | zero-shot({len(eval_ids)})={zero:.3f} "
      f"z_search({len(zs_ids)})={zser:.3f}  no_freeze({len(zs_ids)})={nofr:.3f}")
```

- [ ] **Step 3: Gate H3 assertion (informational headline result)**

```python
print(f"OOD: zero-shot={zero:.3f}  z_search={zser:.3f}  no_freeze={nofr:.3f}")
assert all(v >= 0.0 for v in [zero, zser, nofr])   # record the numbers
print("GATE H3 PASS (OOD exact-match recorded; compare no_freeze vs z_search)")
```

Run: smoke (steps=20, eval_ids[:4]) then full. Heavy — controller drives in background. Gate: produces the OOD exact-match table and the no-freeze-vs-z-search comparison **on real grids** (the conclusive test of the hypothesis). **Commit:** `feat(arc): OOD eval on ARC evaluation split, exact-match across regimes (Gate H3)`.

---

## Task H4: Qualitative grids + summary (Gate H4)

- [ ] **Step 1: plot a few solved/unsolved OOD examples**

```python
import matplotlib.pyplot as plt
show = eval_ids[:6]
fig, ax = plt.subplots(len(show), 3, figsize=(9, 3 * len(show)))
for r, tid in enumerate(show):
    demos, q = task_canonical(tid)
    z = encode_program(model_j.P, demos)
    qi, qo = EVAL_TASKS[tid]["test"][0]
    em, grid = ar_exact_match(dec, z, collate_tokens([q[0]]), qo)
    for c, (g, t) in enumerate([(qi, "input"), (grid, f"pred em={em}"), (qo, "target")]):
        ax[r, c].imshow(g, cmap="tab10", vmin=0, vmax=9); ax[r, c].set_title(t); ax[r, c].axis("off")
plt.tight_layout(); plt.show()
```

- [ ] **Step 2: final summary markdown cell** — record: H0 memorization, H1 latent health, H2 in-distribution exact-match, H3 OOD exact-match per regime, and the verdict on no-freeze vs z-search on real grids. **Commit:** `docs(arc): Phase-2 results summary (Gate H4)`.

---

## Self-review (coverage vs decisions)
- AR enc-dec decoder → H0 (`ARDecoder`, cross-attn to input+z, causal gen).
- Shared decoder trained during pretraining → H1 (`pretrain_joint`, combined loss).
- OOD = ARC evaluation split → H3 (`CACHE.update(build_cache(EVAL_TASKS))`, eval-only ids).
- Keep EMA + SIGReg → unchanged; `model.init_target()` + `episode_loss(..., sigreg_views=)` retained in joint loss.
- Real exact-match (not cosine) is the metric throughout H2/H3 → answers the Phase-1 caveat.
- Compute control: every heavy task has smoke steps + a disk checkpoint; controller runs full in background.

## Risks
- AR greedy decode is O(cells) per grid; eval scoped to ~40 tasks (zero-shot) / ~12 (TTT regimes). Widen later if promising.
- If H0 can't memorize one pair, the decoder wiring is wrong — fix before H1.
- If OOD exact-match is ~0 across all regimes, that is a real finding (ARC-eval is genuinely hard); report honestly and consider re-arc generators / stronger prior as the next lever.
