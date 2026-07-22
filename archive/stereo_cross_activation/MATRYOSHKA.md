# Functional Matryoshka test of the shared activation basis — GPT-2 small (2026-07-22)

Continues `ACTIVATION_PROBE.md`: a single shared PCA basis over per-layer-
centered residual-stream activations reconstructs every layer to ~0.30 rel
err at r=64 (shared/per-layer ratio 1.087 at r=32, flat in rank). PCA
prefixes are nested by construction, so the Matryoshka question is
**functional**: force the residual stream through the top-r shared subspace
at inference and see whether prediction degrades gracefully with r or falls
off a cliff.

## Setup

- Model: `openai-community/gpt2` (124M, 12 layers, d=768), float32 CPU.
- Fit: **wikitext-2-raw-v1 (train)**, 390 × 256 tok = 99,840 tokens, 15,000
  token vectors/layer subsampled — same recipe as the probe. Per-layer means
  kept; shared basis = eigvecs of the trace-normalized pooled covariance of
  per-layer-centered block outputs (`hidden_states[l+1]`), up to r=256.
- Eval: **wikitext-2-raw-v1 (test)**, held out, 40 × 256 tok = 10,240 tokens.
- Patch: forward hook on each block output (residual stream, after the
  block, `output[0]` of the tuple): `h → mean_l + P_r P_rᵀ (h − mean_l)`.
  - **ALL-LAYER**: all 12 block outputs patched (the stereo-cross claim).
  - **SINGLE-LAYER**: layer 6 only (all bases); layers 3, 9 shared-only.
- Controls: **random** orthonormal basis (one fixed 768×256 Q, nested
  prefixes, same per-layer means) = chance floor; **per-layer PCA** basis of
  same rank = layer-specific upper bound.
- Metrics vs the unpatched model on the same tokens: mean KL(base‖patched)
  over all positions, next-token top-1 agreement with the unpatched model,
  patched next-token accuracy, and ppl ratio (patched ppl / base ppl).
  Baseline: ppl **54.46**, top-1 accuracy **0.310**.
- Script `matryoshka_patch.py`, raw `results_matryoshka.json`, log
  `matryoshka_run.log`.

## Pre-registered fork

- **MATRYOSHKA_FUNCTIONAL**: all-layer degradation graceful — top-1
  agreement ≥ ~0.85 at r=128 and ≥ ~0.7 at r=64, monotone improvement with
  r, shared clearly beats random at every r and approaches per-layer PCA.
- **CLIFF / NOT_FUNCTIONAL**: all-layer agreement < ~0.5 even at r=128, or
  shared ≈ random.

## Results

### ALL-LAYER patching (shared vs random vs per-layer PCA)

| r | KL sh | agr sh | ppl× sh | KL rnd | agr rnd | ppl× rnd | KL pl | agr pl | ppl× pl |
|---|-------|--------|---------|--------|---------|----------|-------|--------|---------|
| 8 | 4.22 | 0.027 | 78.8 | 4.63 | 0.034 | 116 | 4.80 | 0.046 | 133 |
| 16 | 4.61 | 0.086 | 114 | 4.86 | 0.022 | 140 | 4.65 | 0.053 | 113 |
| 32 | 5.89 | 0.045 | 410 | 4.93 | 0.024 | 146 | 5.68 | 0.131 | 272 |
| 64 | 4.05 | 0.143 | 55.2 | 5.68 | 0.024 | 313 | 5.83 | 0.136 | 254 |
| 128 | 3.36 | 0.169 | 27.1 | 6.47 | 0.000 | 630 | 3.21 | 0.202 | 19.9 |
| 256 | 2.32 | 0.232 | 9.5 | 7.76 | 0.000 | 2444 | 2.32 | 0.245 | 8.2 |

### SINGLE-LAYER patching, layer 6

| r | KL sh | agr sh | ppl× sh | KL rnd | agr rnd | ppl× rnd | KL pl | agr pl | ppl× pl |
|---|-------|--------|---------|--------|---------|----------|-------|--------|---------|
| 8 | 3.90 | 0.123 | 52.1 | 6.12 | 0.032 | 465 | 3.31 | 0.160 | 25.7 |
| 16 | 3.08 | 0.166 | 20.7 | 5.59 | 0.031 | 287 | 2.76 | 0.216 | 14.2 |
| 32 | 2.45 | 0.234 | 11.3 | 5.80 | 0.029 | 351 | 2.17 | 0.289 | 7.9 |
| 64 | 1.72 | 0.324 | 5.3 | 5.52 | 0.023 | 249 | 1.49 | 0.377 | 4.0 |
| 128 | 1.01 | 0.446 | 2.67 | 5.42 | 0.045 | 256 | 0.82 | 0.502 | 2.09 |
| 256 | 0.45 | 0.615 | 1.51 | 4.66 | 0.075 | 119 | 0.32 | 0.676 | 1.29 |

Shared-basis single-layer patching at L3 / L9 (agr at r=64/128/256):
L3 0.291 / 0.456 / 0.679; L9 0.326 / 0.438 / 0.610 — same picture as L6.

### Verdict: **CLIFF / NOT_FUNCTIONAL** (all-layer), with two honest caveats

By the pre-registered thresholds this is the cliff branch: all-layer top-1
agreement is **0.169 at r=128** (bar was ≥ ~0.85; even the < 0.5 failure
line is missed by 3×), r=256 only reaches 0.232, ppl is 27× base at r=128,
and improvement is **not monotone** (r=32 is *worse* than r=16: ppl 410 vs
114; per-layer PCA shows the same r=32–64 spike). The ~0.30 per-layer
reconstruction error compounds through 12 successive projections and the
model's predictions collapse.

Caveats that matter for interpretation:

1. **Shared ≉ random.** The random control is dramatically worse everywhere
   (agreement → 0.000 at r≥128 while shared climbs to 0.23; random ppl
   *grows* with rank to 2444×, presumably because higher-rank random
   projections preserve more of the wrong geometry while still destroying
   the outlier/LN structure). The shared basis carries real functional
   signal.
2. **The cliff is not specific to the shared basis.** The per-layer PCA
   upper bound collapses almost identically under all-layer patching (agr
   0.245 vs 0.232 at r=256, KL 2.32 vs 2.32). So the failure mode is *hard
   low-rank projection applied at every block*, not the shared coordinate
   system per se — layer-specific coordinates would not have saved it. The
   probe's flat shared/per-layer ratio (~1.09) shows up functionally too:
   shared tracks per-layer within ~0.03 agreement at r≥128.
3. **Single-layer patching is genuinely graceful and Matryoshka-shaped**:
   monotone in r at every patched layer (L3/L6/L9), shared close to the
   per-layer upper bound (0.615 vs 0.676 agreement at r=256), random flat
   near 0.03–0.07. One projection anywhere in the stack degrades smoothly;
   twelve compounding projections do not.

So: the shared basis is a fine *descriptive* nested coordinate system and
even a usable *single-point* functional readout, but it is **not** a
functional Matryoshka bottleneck for the whole residual stream at these
ranks — no configuration tested comes close to the graceful-degradation
bar. Any future "run the model inside the shared subspace" claim would
need either much higher rank, error-aware soft projection, or fine-tuning
with the bottleneck in place.
