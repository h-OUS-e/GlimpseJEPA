# ARC-AGI Latent Program JEPA (LP-JEPA) — Design

Date: 2026-06-07
Status: Approved (proceed to implementation plan, full G0–G5)

## 1. Goal

Test whether a small LeWorldModel-style JEPA can learn ARC-AGI **transformations**
(not pixels) by treating a task's demonstration pairs as a latent *program/action*
applied to a test input. Reframes the Latent Program Network (LPN, Bonnet &
Macfarlane, "Searching Latent Program Spaces", arXiv:2411.08706) into the repo's
action-conditioned JEPA, with two original twists:

1. **Latent-space matching** (JEPA) instead of pixel decoding — at least in Phase 1.
2. **Test-time *weight* training** (no freeze) instead of LPN's frozen model +
   gradient-ascent-over-`z` only. Hypothesis: continuing to train weights lets the
   model reach genuinely out-of-distribution programs that pure `z`-search cannot,
   because if a program does not exist in the learned latent manifold, ascent over
   `z` can never reach it.

Success is **staged**:
- **Phase 1 (latent-only):** validate the transformation mechanism; show program
  latents cluster by task (TSNE); show held-out latent prediction error drops on
  solved tasks. No decoder.
- **Phase 2 (grid):** add a discrete grid decoder; score real ARC exact-match
  (top-1/top-2) on held-out query pairs.

## 2. Hard constraints

- **All new code lives in ONE new self-contained notebook.** No other `.py` file in
  the repo is touched. Any encoder/decoder/predictor is (re)implemented in the
  notebook (concepts may be borrowed from `jepa.py` / `ml_layers.py` / LPN, but not
  imported as edits to those files).
- `LeWorldModelRef/` is reference-only; do not edit it.
- Use the conda env Python: `C:/Users/Ous/miniconda3/envs/ML/python.exe`.

## 3. Background: what we borrow from LPN

- Encoder/decoder are **small transformers** over grid tokens (not flat MLP).
- Grid tokenization: shape-prefix tokens + per-cell color tokens; factorized
  positional embedding `pe(i,j,c) = pr(i) + pc(j) + emb(c)`; padding masked,
  non-causal in encoder.
- Program latent = **mean** of per-pair latents; **leave-one-out** when supervising
  a pair (encode program from the *other* pairs) to stop output-copying leakage.
- VAE-style encoder (μ, Σ) + KL to N(0, I) is optional here; we default to a
  deterministic latent + SIGReg, with KL as an ablation.
- Test-time **gradient ascent over `z`** is kept as a *baseline* to compare against
  our no-freeze weight training.

## 4. Architecture (notebook modules)

```
Grid tokenizer
  grid (H≤30, W≤30, colors 0..9)
   → [shape_h, shape_w, cell_(0,0), cell_(0,1), ...]
   → factorized pos-emb pr(i)+pc(j)+colour(c); pad to fixed token budget; attn mask

Image encoder      E : grid → pooled CLS → a ∈ R^D            # the "state"
Program encoder    P : concat(input_grid, output_grid) per demo → z_i
                       z = mean_i z_i  (leave-one-out at train time)   # the "program/action"
Predictor          F : token a, conditioned on z via AdaLN (ConditionalBlock) → ẑ_out
Target encoder     Ē : EMA copy of E, stop-grad; target = Ē(output_grid)
Decoder (Phase 2)  G : z, input_grid → output shape tokens + per-cell colour logits
```

Notes:
- **Two encoders** (E for grids-as-states, P for I/O-pairs-as-programs) — approved
  over a single shared encoder. P must see *both* halves of a demo pair (the I/O
  difference defines the program).
- Differing input/output grid shapes are fine: E and Ē pool to fixed `D`. Only the
  Phase-2 decoder G must predict output **shape** explicitly (not assume input shape).
- Encoder choice: ViT-style transformer over grid tokens, per LPN. Final pick made
  in G1; start small (e.g. depth 4–6, dim 256, heads 4).

## 5. Losses

- **Phase 1:** `L = MSE(F(a, z), Ē(output)) + λ·SIGReg(embeddings)` with stop-grad
  on `Ē`. Optional `+ β·KL(z‖N(0,I))` ablation.
- **Phase 2:** add decoder cross-entropy on shape tokens + per-cell colors; track
  exact-match. Decoder loss grounds the latent and further fights collapse.

## 6. Anti-collapse

- Primary: **stop-grad + EMA target encoder** `Ē`, with SIGReg as secondary.
- **Tiny-batch SIGReg proposition:** the per-task batch is ~3 pairs, too small for a
  meaningful Gaussianity statistic. Run SIGReg across the **augmentation views** of a
  task (8 dihedral transforms × C color permutations → dozens of embeddings), and/or
  maintain an **EMA feature buffer** of recent embeddings to stabilize the statistic.
- Ablations: toggle stop-grad/EMA on/off; sweep λ ∈ {0, 0.01, 0.09, 0.3}; KL on/off.
- Collapse monitor (reuse the repo's existing probe): `z_std`, `z_norm`, pairwise
  cosine — log every eval.

## 7. Data & task protocol

- **Dataset:** ARC-AGI-1 (`fchollet/ARC-AGI`): `data/training/*.json` (400),
  `data/evaluation/*.json` (400). Download into `data/`.
- **Augmentation (the prior + the TTT batch source):** 8 dihedral transforms ×
  random color permutations per task.
- **Splits:** pretrain on a subset of the **training** 400; hold out specific tasks
  (and the evaluation split) as *unseen* for TTT — these test the no-freeze
  hypothesis.
- **Per task:** K demo pairs (I/O known) define the program; 1 query pair held out
  for scoring. Leave-one-out across K+1 pairs at pretraining time.
- **Easy/hard tasks:** one same-shape recolor/symmetry task (*easy*); one
  compositional multi-step task (*hard*). Concrete task IDs chosen in the plan at G5
  (pick from `data/training`, verify shapes by inspection).

## 8. Training procedures

- **Pretrain:** mean-program + leave-one-out; MSE-in-latent + EMA target + SIGReg(λ).
  Optionally one inner gradient-ascent step on `z` during training so the network is
  "search-aware" (LPN finding).
- **Test-time training (the twist):** per held-out task, continue training E/P/F
  (and optionally G) on the task's augmented demos. **Early-stop on a left-out demo**,
  not on the training loss (driving train loss → 0 memorizes and the held-out query
  does not improve). Compare three regimes:
  1. **no-freeze** (train weights) — ours,
  2. **frozen + z-gradient-ascent** — LPN baseline,
  3. **from-scratch per task** — control.

## 9. Evaluation & visualization

- **Phase 1:** per-task held-out latent prediction error; **TSNE of program latents
  `z`** colored by task (clusters = programs forming); optional 2D-latent toy task
  for a clean smooth-latent map (paper-style).
- **z-search overlay:** gradient ascent over `z` on the held-out query, compared with
  weight-training and with both combined.
- **Phase 2:** exact-match top-1/top-2 on held-out query for easy + hard task;
  decoded-grid plots across TTT steps showing progress.

## 10. Phased workflow (gates)

- **G0 — Data harness:** ARC download, grid tokenizer, augmentation, batch/protocol.
  *Gate: tokenizer round-trips a grid to identity.*
- **G1 — Encoders + JEPA forward:** E/P/F + EMA target; overfit a single easy task in
  latent space. *Gate: held-out-demo latent error drops; no collapse (healthy z_std).*
- **G2 — Pretrain + TSNE:** pretrain on a training subset; TSNE of `z`.
  *Gate: visible program clustering.*
- **G3 — TTT on unseen easy task:** no-freeze vs frozen+z-search vs scratch.
  *Gate: no-freeze ≥ baselines on latent metric.*
- **G4 — Decoder (Phase 2):** add G; exact-match on easy task.
  *Gate: nonzero exact-match on held-out query.*
- **G5 — Hard task + ablations:** SIGReg λ sweep, stop-grad/EMA toggles, KL on/off,
  easy vs hard. *Gate: results table + plots.*

## 11. Risks / open items

- ARC at full 30×30 + 10 colors is a large token budget; if G1 struggles, restrict
  to same-shape ≤ ~15×15 tasks first, then widen the canvas.
- SIGReg may still be too weak at TTT scale even with augmentation views; the
  EMA/stop-grad target is the real safety net.
- Pretraining on only 400 raw tasks (no re-arc generators) may give a weak prior;
  if clustering at G2 is poor, revisit adding re-arc as a follow-up.
- "Solved in latent space" ≠ "correct grid"; only Phase 2 (decoder + exact-match)
  confirms real ARC solving.

## 12. Out of scope (for now)

- re-arc procedural generators (follow-up if the 400-task prior is too weak).
- Leaderboard submission / private test set.
- Editing any existing repo `.py` file.
