# ARC-LeWM: Decode-by-Planning JEPA for ARC, Meta-Trained on the Public Train Split

**Date:** 2026-06-09
**Branch:** `arc-lewm`
**Code home:** `arc_lewm.ipynb` (new notebook at repo root; old `arc_lp_jepa.ipynb` untouched)
**Question under test:** Can the methods of LeWorldModel (LeWM) — end-to-end two-loss JEPA
training plus planning-as-inference — solve real ARC tasks when meta-trained episodically
on the public training split only, with no benchmark-specific data engineering?

## 1. Constraints (user-set)

- **No benchmark biasing.** No hand-designed transformation families (this rejects the
  Phase 3 approach of `arc_lp_jepa.ipynb`). Training data is the ARC-AGI-1 public
  training split (400 tasks) only — the benchmark's sanctioned training data.
- **Trained to learn fast.** The training objective is episodic/meta: every step is a
  mini ARC task (demo pairs -> predict a held-out pair), so few-shot adaptation is the
  core trained skill, not an afterthought.
- **Fast training.** Small model (~10–15M params), full pretrain budgeted at 30–60 min
  on the local GPU.
- **Success bar:** pixel-exact match on the held-out ARC evaluation split, top-2 guesses
  allowed (ARC rules). Honest reporting at every gate; no tuning on the eval split.

## 2. Framing: ARC task as a world-model episode

| LeWM (control) | ARC-LeWM |
|---|---|
| state observation (pixels) | test input grid `x` |
| action | latent program `z` from demo pairs |
| next-state embedding | output grid embedding `E(y)` |
| predictor F(emb, act_emb) | AdaLN predictor `F(E(x), z)` |
| planning: optimize actions vs latent goal cost | decoding: optimize the output *grid* vs predicted latent |

## 3. LeWM methods kept verbatim

1. **End-to-end, no EMA, no stop-grad teacher.** Target embeddings come from the same
   live encoder; gradients flow through both prediction and target branches
   (see `lejepa_forward` in `LeWorldModelRef/train.py`).
2. **Exactly two losses, one hyperparameter:**
   `loss = MSE(F(E(x), z), E(y)) + lambda * SIGReg(batch embeddings)`.
   SIGReg is the Epps–Pulley Gaussianity statistic over random projections
   (`LeWorldModelRef/module.py::SIGReg`), computed over a real batch — concretely, all
   register-token embeddings in the batch flattened to (B*K, d) vectors (>= 256 of
   them), plus the batch of program latents `z` as a second term with the same lambda.
   This fixes the silently-inactive per-episode SIGReg of the old notebook.
3. **AdaLN-zero ConditionalBlock predictor** and **BatchNorm-hidden MLP projectors**,
   lifted from `LeWorldModelRef/module.py`.
4. **Inference is optimization against a latent cost.** No pixel decoder anywhere.

## 4. Deliberate deviation (flagged): K latent tokens instead of single CLS

LeWM pools to a single CLS embedding. Pixel-exact recovery of a 30x30x10 grid through
one 256-d vector by gradient inversion is the dominant risk, so the grid encoder emits
**K = 16 register tokens** (d = 256) instead; the predictor predicts all K. Same losses,
same recipe otherwise. K is a tunable knob validated by the inversion gate (L3).
This deviation is generality-neutral: it encodes no ARC-specific prior.

## 5. Model components (~10–15M params total)

- **GridEncoder E:** linear token embedding over one-hot colors (so *soft* one-hot
  grids work natively at inference) + 2D positional embedding + small transformer;
  K = 16 learnable register tokens attend to cell tokens; output (K, 256).
- **Projector:** MLP with BatchNorm hidden layer (LeWM style), applied to encoder output.
- **ProgramEncoder P:** per-demo-pair, concat input/output register embeddings -> pair
  token; small transformer over pair tokens -> pooled program latent `z` (256-d).
  `z` plays LeWM's action-embedding role.
- **Predictor F:** stack of LeWM ConditionalBlocks; input = register tokens of `E(x)`,
  conditioning `c` = `z`; output = predicted register tokens of the output grid;
  pred_proj MLP on top (LeWM style).

## 6. Training

- **Episodic sampler:** sample task, sample k demos (2–4) + 1 held-out query pair from
  that task's train pairs. Batch many episodes per step.
- **Augmentation (task-agnostic only):** color permutation and dihedral-group transforms
  applied consistently across all grids of an episode. These are symmetry augmentations,
  not family engineering.
- **Throughput:** all grids pre-tokenized once into a GPU-resident padded tensor cache
  (pad to 30x30 + PAD class + size mask); target >= 50 episodes/s. This kills the known
  0.75 s/step CPU bottleneck from the previous notebook.
- **Optimizer:** AdamW + warmup-cosine (LeWM uses LinearWarmupCosineAnnealingLR).

## 7. Inference: decode-by-planning

Given test input `x` and demos -> `z`:
1. Target latents `T = F(E(x), z)` (frozen model).
2. Optimize relaxed grid logits `L` in R^(30 x 30 x 11) (10 colors + PAD for size) with
   Adam through the frozen encoder to minimize `|| E(softmax(L / tau)) - T ||^2`,
   annealing temperature `tau`; then argmax-discretize.
3. Multiple restarts: one initialized from the test input (many ARC outputs are
   near-copies of the input), the rest random. **Top-2 = the two best restarts by final
   cost** (deduplicated).
4. Output size: handled by PAD logits; fallback heuristic = infer size from the demo
   output-size pattern (consistent-size rule detection is task structure, not bias).

## 8. Gates (each cheap to fail, in order)

- **L0 — pipeline:** batched episodic loader works; >= 50 episodes/s.
- **L1 — single-batch overfit:** pred MSE drops, SIGReg term is active and nonzero,
  embedding std healthy (no collapse), all end-to-end with no EMA.
- **L2 — full pretrain:** 30–60 min GPU budget on the 400 train tasks + augmentations;
  run via background nbconvert; monitor pred/SIGReg losses and embedding stats.
- **L3 — encoder inversion (make-or-break):** recover grids from their *own* embeddings
  by relaxed optimization; >= 90% exact match required. If L3 fails after one diagnosis
  round (raise K, tune lambda/tau schedule), STOP and report honestly — do not tune
  around it. Documented fallback (separate decision, not in scope): Approach B = bolt a
  small AR decoder onto the same episodic core.
- **L4 — end-to-end exact match:** first on held-out pairs of train-split tasks
  (in-distribution few-shot), then ONCE on the full evaluation split (top-1 and top-2).
  Compare against the Phase 3 baseline (5 real-task solves) noting that baseline used
  hand-designed families, which this run forbids.

## 9. Error handling and honesty rules

- Eval split is touched exactly once, at L4, after all tuning is frozen.
- Every gate reports its raw numbers (including failures) in executed notebook output.
- Long runs go through background `nbconvert` (10-min foreground tool timeout).
- Checkpoints under `data/` (gitignored): `data/arc_lewm.pt`.

## 10. Out of scope

- re-arc / external generators (previously blocked; would also dilute the
  train-split-only claim).
- Test-time weight training (no-freeze TTT) — tested twice before, unsupported.
- Any procedural family generators.
