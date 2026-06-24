# Memory-latent regression — debugging log

Metric notes:
- `nmse_*` = latent MSE / target-latent variance (scale-invariant; raw MSE is NOT
  comparable across configs because latent scale differs).
- `px_*` = pixel MSE of decoded images. `px_enc`=decode(z_enc), `px_tf`/`px_ar`=decode(pred).

## Hypotheses tested

| # | Hypothesis | Verdict | Evidence |
|---|------------|---------|----------|
| H1 | Action dilution (3 act vs 36 mem dims) makes it worse | **FALSIFIED** | Bigger memory gave *lower* latent MSE, opposite of dilution |
| H2 | Memory induces latent collapse, hurting decode | **FALSIFIED at convergence** | z_std gap (0.63 vs 5.0) was undertraining; at 1500 steps z_std ~equal |
| H3 | Decode blob = encoder lost info | **FALSIFIED** | Strong frozen-encoder decoder (3000 steps) recovers recognizable digits |
| -- | Raw latent-MSE comparison is valid | **FALSE (confound)** | latent scale (z_std) differs 8x across configs; must normalize |
| H5 | Memory is unstable / high variance (z_std blow-up) | **TESTING** | mem36 z_std swung 6.7 -> 13.3 across data orders; nmse_ar 0.21 -> 0.94 |

## Round 2 (1500 steps, identical data, scale-invariant)
- nomem:  nmse_tf 0.040  nmse_ar 0.231  px_ar 0.067  z_std 6.27
- mem36:  nmse_tf 0.063  nmse_ar 0.209  px_ar 0.072  z_std 6.70
=> At convergence memory is ~neutral (slightly better long-horizon AR, slightly worse TF/pixel).

## Round 3 (decoder probe, frozen encoder, 3000-step decoder)
- nomem PROBE: px(true latent) 0.046, px(AR pred) 0.066  -> recognizable digits
- mem36 PROBE: px(true latent) 0.054, px(AR pred) 0.089
=> Latents ARE decodable. In-loop decode blob = joint decoder undertrained (lambda=0.1, depth 2).
   NOTE: round-3 mem36 hit z_std 13.3 / nmse_ar 0.94 due to a data-order control bug
   (shared stateful DataLoader generator) -> motivates the variance test (H5).

## Two distinct problems identified
1. **Decode quality** (pre-existing, not memory): in-loop decoder undertrained.
2. **Memory stability** (the regression the user saw): memory appears seed/data sensitive
   with z_std blow-ups. Confirm with multi-seed, then fix at the source.

## Round 4 (multi-seed, data-control bug fixed)  -> ROOT CAUSE
Both configs are wildly unstable across seeds; the ENCODER LATENT SCALE is uncontrolled:
- nomem: nmse_ar 0.56+-0.21, z_std 6.3 / 14.4 / 10.9
- mem36: nmse_ar 0.84+-0.59, z_std 6.4 / 24.8 / 1.9
=> Root cause: latent scale runs free. The mechanism meant to control it (SigReg) is
   ineffective as written. Memory just amplifies the variance (extra capacity to inflate).

## Round 5 (SigReg weight sweep)  -> H6 FALSIFIED
Raising lambda_sig does NOT stabilize: nomem z_std goes UP (11->18->17); mem36 nmse_ar
EXPLODES (one seed 72). Confirms SigReg cannot pin the scale; leaning on it destabilizes.

## Round 6 (encoder-output non-affine LayerNorm)  -> FIX CONFIRMED (H7)
LayerNorm(Z_IMG, elementwise_affine=False) on the encoder output (via JEPA.projector):

| config    | nmse_ar      | z_std       | px_tf  | px_ar  | cos   |
|-----------|--------------|-------------|--------|--------|-------|
| nomem (no LN) | 0.56+-0.21 | 6-14 unstable | ~0.06 | 0.083 | 0.5-0.9 |
| mem36 (no LN) | 0.84+-0.59 | 2-25 unstable | ~0.07 | 0.083 | 0.5-0.9 |
| **nomem_LN**  | **0.26+-0.01** | **0.98+-0.00** | **0.029** | **0.052** | ~0 |
| **mem36_LN**  | **0.18+-0.03** | **0.98+-0.00** | **0.029** | **0.045** | ~0 |

- Variance collapses ~15x; z_std pinned to 0.98 every seed.
- Latent MSE halves; decode pixel error roughly halves; cos ~0 (no collapse).
- Decode is visually SHARP (recognizable digits), not blobs.
- Memory now genuinely HELPS once scale is controlled: mem36_LN beats nomem_LN on both
  latent MSE (0.18 vs 0.26) and decode (0.045 vs 0.052).

## Round 7 (lowering MSE further) — AR curriculum
Train with ar_steps ramped 0->T so the predictor learns on its own rollouts (mem36 + LN):
- base (ar_steps=0):   nmse_tf 0.054  nmse_ar 0.156  px_ar 0.035  (201s)
- AR curriculum:       nmse_tf 0.083  nmse_ar 0.086  px_ar 0.041  (768s)
=> Long-horizon AR latent MSE ~halves (0.156 -> 0.086). Cost: slightly worse one-step/pixel
   sharpness and ~4x slower (per-step rollout loop). Best single lever for long-horizon MSE.
   Other untested levers: larger z_dim_img (36 is a tight bottleneck), EMA target encoder,
   longer training, downweighting late steps (mse_weighted).

## Conclusion
The "memory made it worse" was a symptom, not the disease. The disease is an unconstrained
encoder latent scale (SigReg ineffective). Memory only amplified the existing instability.
SURGICAL FIX: normalize the encoder output with a non-affine LayerNorm (JEPA.projector).
This is the single change that yields low latent MSE + good decode + stability, and lets the
memory feature deliver its intended benefit.
