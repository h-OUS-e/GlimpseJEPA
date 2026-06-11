# ARC-LeWM Phase B: Joint AR Decoder as Latent Inverter

**Date:** 2026-06-10 · **Branch:** `arc-lewm` · **Code:** Phase B section appended to `arc_lewm.ipynb`
**Parent spec:** `2026-06-09-arc-lewm-design.md` (gates L0-L2 passed; L3 inversion FAILED 2/50 —
encoder folds grid space; decode-by-planning dead).

## 1. Question

Does adding ONE decoder loss to the LeWM recipe force pixel information into the latent, making
the (working) latent meta-learning decodable — and does that yield real exact-match ARC solves?

This is a flagged deviation from pure LeWM: loss becomes pred-MSE + λ·SIGReg + CE(decoder).
Everything else (episodic core, no EMA, no stop-grad, train-split-only data, symmetry-only
augmentation, single eval-split touch) is unchanged.

## 2. Design

- **GridDecoder** (~4 layers, d=192): autoregressive over the output canvas raster (BOS + cells,
  PAD beyond (h,w)); each block = causal self-attn + cross-attn to context + FFN. Context =
  [output latent tokens (K); input grid cell tokens embedded by the decoder itself + 2D pos].
- **Latent fed to the decoder during training:** per-sample coin flip (p=0.5) between the TRUE
  output embedding `E(y)` (gradient forces encoder injectivity — the direct fix for L3) and the
  PREDICTED latent `F(E(x), z)` (gradient trains the exact inference path, predictor included).
- **Warm start** encoder/program/predictor from `data/arc_lewm.pt` (L2 checkpoint); decoder fresh.
  Joint training ~20k steps, B=8, bf16 (the profiled VRAM-cliff-safe config).
- **Inference:** demos → z → predicted latent → batched greedy decode (PAD forced at off-grid
  positions — the Phase 2 generate() lesson). Output size from the demo-pattern heuristic
  (same-shape / constant / ratio), fallback 30×30 canvas. Top-2 = greedy + temperature-0.7
  sample, deduplicated.

## 3. Gates

- **B1** decoder sanity: joint single-batch overfit reaches teacher-forced accuracy ~1.0 and
  generate() exact-match 1.0 on the memorized pairs.
- **B2** joint pretrain: decode CE must escape the Phase 2 failure signature (plateau ≈3);
  monitor CE + teacher-forced exact-match + latent stats (no collapse, pred MSE stays healthy).
- **B3** decode from TRUE latents on held-out-from-gate train grids (the L3 replacement):
  exact-match reported; ≥0.5 means the latent is now invertible-by-decoder.
- **B4a** end-to-end on held-out test pairs of 100 train-split tasks (top-1/top-2).
- **B4b** evaluation split, ONCE, after all tuning frozen: tasks solved (all test pairs, top-2).

## 4. Honesty rules (unchanged)

Eval split touched exactly once (B4b). All gate numbers committed in executed output, pass or
fail. No hand-designed families. If B2/B3 fail, stop and report — same stop discipline as L3.
