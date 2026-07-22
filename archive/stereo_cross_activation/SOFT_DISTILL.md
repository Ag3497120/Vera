# Can the all-layer shared bottleneck be made functional? — GPT-2 small (2026-07-22)

Continues `MATRYOSHKA.md`: hard projection of every block output onto the
top-r shared PCA basis collapses GPT-2 small (top-1 agreement 0.169 at
r=128, ppl 27×), but per-layer PCA collapses identically and single-layer
patching is graceful — so the failure is the *compounding* of 12 hard
projections, not the shared coordinates. Two phases here: (1) training-free
diagnostics that quantify the compounding directly and test soft
projections; (2) the real fix — freeze the bottleneck and fine-tune the
model weights through it ("bottleneck-aware distillation").

## Setup

- Model / data / basis: identical recipe to `matryoshka_patch.py` —
  `openai-community/gpt2`, shared basis = eigvecs of the trace-normalized
  pooled covariance of per-layer-centered block outputs, fit on
  wikitext-2-raw-v1 (train), 390 × 256 tok, 15,000 vectors/layer; per-layer
  means kept. Eval: wikitext-2-raw-v1 (test), same 40 × 256 tok = 10,240
  tokens as before. Device: MPS float32 (baseline reproduces the prior CPU
  run: ppl **54.466** vs 54.464, top-1 acc **0.3102**).
- Phase 1 patches (forward hooks, `h → mean_l + f(h − mean_l)`):
  - **Compounding curve**: hard shared projection at r=128 on k layers
    spread evenly, k ∈ {1,2,4,6,12}; layers = round((i+1)·12/(k+1)), so
    k=1 → L6, k=2 → L4,L8, k=4 → L2,5,7,10, k=6 → L2,3,5,7,9,10.
  - **Residual shrinkage**: f = P_r P_rᵀ + α(I − P_r P_rᵀ), α ∈
    {0.1, 0.25, 0.5}, all 12 layers, r ∈ {64,128}. α>0 leaks the full
    orthogonal complement, so this is a **diagnostic, not an honest
    bottleneck**.
  - **Wiener-style spectral soft projection**: f = V diag(w) Vᵀ over ALL
    768 shared components, w_i = λ_i/(λ_i+σ²), σ² set so Σw_i = r_eff ∈
    {64,128} (σ² = 0.0297 / 0.0116). A fixed linear map — an honest soft
    bottleneck variant.
- Phase 2: hard shared bottleneck (basis + means FROZEN) hooked on all 12
  block outputs; fine-tune all model weights with plain LM loss on
  wikitext-2 train (8,000 × 256 tok ≈ 2.05M tokens ≈ 1 epoch). AdamW
  lr 5e-5, linear warmup 50 steps then constant, grad clip 1.0, batch
  8 × 256, max 1000 steps, eval every 100 on a 16 × 256 validation slice
  (bottlenecked ppl + top-1 agreement vs the ORIGINAL unpatched GPT-2),
  early stop if <1% val-ppl improvement on 2 consecutive evals (never
  triggered). r=128 and r=64.
- Script `soft_and_distill.py`, raw `results_soft_distill.json`, logs
  `soft_distill_phase1.log` / `soft_distill_phase2.log`. Prior artifacts
  untouched.

## Pre-registered fork (Phase 2, r=128)

- **CONTAINER_VIABLE**: after short FT, ppl ≤ ~1.5× original baseline
  (≤ 82) and/or agreement ≥ 0.6 — "minimal training draws out the
  container" supported in activation space.
- **CONTAINER_NOT_RECOVERED**: FT barely helps — ppl still > 3× (> 163),
  agreement < 0.4.
- Between: state honestly.

Phase 2 runs regardless unless Phase 1 reaches agreement > 0.85 at an
honest bottleneck (it does not come close).

## Phase 1 results

### Compounding curve (hard shared projection, r=128, k layers)

| k | layers | KL | agr | ppl× | acc |
|---|--------|-----|------|------|------|
| 1 | 6 | 1.01 | 0.446 | 2.67 | 0.195 |
| 2 | 4,8 | 1.61 | 0.337 | 4.60 | 0.154 |
| 4 | 2,5,7,10 | 2.07 | 0.263 | 7.03 | 0.128 |
| 6 | 2,3,5,7,9,10 | 2.24 | 0.243 | 7.96 | 0.123 |
| 12 | all | 3.36 | 0.169 | 27.1 | 0.086 |

Monotone, smooth degradation in k with no single cliff — the compounding
claim is confirmed directly. (k=1 and k=12 reproduce the prior run's
single-L6 and all-layer numbers exactly.)

### Soft projections (all 12 layers)

| variant | KL | agr | ppl× | | KL | agr | ppl× |
|---|---|---|---|---|---|---|---|
| | **r=64** | | | | **r=128** | | |
| hard (reference) | 4.05 | 0.143 | 55.2 | | 3.36 | 0.169 | 27.1 |
| shrink α=0.1 (diagnostic) | 3.81 | 0.152 | 42.2 | | 3.11 | 0.184 | 21.0 |
| shrink α=0.25 (diagnostic) | 3.46 | 0.168 | 29.1 | | 2.76 | 0.210 | 14.7 |
| shrink α=0.5 (diagnostic) | 2.82 | 0.198 | 15.2 | | 2.15 | 0.266 | 7.8 |
| Wiener r_eff-matched | 4.63 | 0.107 | 108.0 | | 4.24 | 0.037 | 70.5 |

(hard r=64 row from `results_matryoshka.json`; hard r=128 = compounding
k=12 row above.)

Two findings:

1. **Shrinkage helps only modestly.** Even leaking *half* the orthogonal
   complement (α=0.5, not a bottleneck in any honest sense) only lifts
   r=128 agreement from 0.169 to 0.266 (ppl 7.8×). Error-aware leakage is
   not a route to a functional bottleneck.
2. **Wiener soft projection is strictly worse than the hard cutoff** at
   matched effective rank (agr 0.037 vs 0.169 at r_eff=128). The Wiener map
   attenuates *every* component (w_i < 1 even for the top outlier
   directions), and twelve successive applications of a strictly
   contractive map shrink the residual stream multiplicatively, destroying
   the scale/outlier structure that LayerNorm expects. Hard projection at
   least preserves the retained subspace exactly. Spectral softening does
   not fix compounding; it makes it worse.

## Phase 2 results (bottleneck-aware distillation)

Training curve (validation slice, bottlenecked ppl / agreement vs original):

| step | r=128 ppl | r=128 agr | r=64 ppl | r=64 agr |
|------|-----------|-----------|----------|----------|
| 0 | 1279.5 | 0.177 | 2712.5 | 0.151 |
| 100 | 226.5 | 0.285 | 354.7 | 0.244 |
| 300 | 165.7 | 0.311 | 259.2 | 0.269 |
| 500 | 144.1 | 0.321 | 228.4 | 0.283 |
| 1000 | 120.2 | 0.346 | 189.5 | 0.297 |

- Both runs used the full 1000 steps (2.05M tokens ≈ 1 epoch); the plateau
  early-stop never fired — val ppl was still improving ~3%/100 steps at
  budget end.
- Wall time: r=128 **2733 s** (45.6 min), r=64 2703 s, on MPS.
- Steps to 95% of the achieved val-ppl improvement: **300** (r=128, ~8 min
  wall) and 200 (r=64) — most of the recovery is fast; the tail is slow.

Final eval on the held-out wikitext-2 test set (same 10,240 tokens;
baseline ppl 54.47, acc 0.310):

| r | ppl | ppl× | agr vs orig | KL | own acc | pre-FT ppl× / agr |
|---|------|------|------|------|------|------|
| 128 | 136.7 | 2.51 | 0.341 | 2.13 | 0.244 | 27.1 / 0.169 |
| 64 | 212.9 | 3.91 | 0.289 | 2.58 | 0.215 | 55.2 / 0.143 |

Fork thresholds at r=128: ppl 136.7 > 82 (CONTAINER_VIABLE capability bar
missed, 2.51× vs the ≤ ~1.5× bar); agreement 0.341 < 0.6 (agreement bar
missed). But ppl 2.51× < 3× and the improvement is anything but "barely
helps" (27× → 2.5×, accuracy 0.086 → 0.244 = 79% of baseline), so
CONTAINER_NOT_RECOVERED is escaped on the capability side while its
agreement clause (< 0.4) still holds.

### Verdict: **between the forks — capability largely recovered, function-matching not**

One epoch of plain LM fine-tuning through the frozen hard r=128 all-layer
bottleneck takes the model from collapse (ppl 27×, agr 0.169) to a usable
language model (ppl 2.51×, own top-1 accuracy 79% of baseline), with 95% of
that recovery inside the first 300 steps (~8 min). So minimal training
*does* draw a working LM out of the shared-subspace container — the
structure is there to be re-activated, in sharp contrast to the
weight-space bridge (`results_gpt2_weight_bridge.json`), where training
found nothing because no shared structure existed.

But what gets drawn out is **not the original model**: top-1 agreement with
unpatched GPT-2 plateaus around 0.34 (< the 0.4 NOT_RECOVERED line, far
from the 0.6 VIABLE line). Caveats that matter:

1. **The objective never asked for agreement.** Plain LM loss lets the
   model settle into a *different* good predictor inside the constrained
   family; KL-distillation against the original logits would target the
   agreement clause directly and is the obvious next step.
2. **The budget capped the run, not a plateau.** Val ppl was still falling
   ~3%/100 steps at step 1000, so 2.51× is an upper bound on what this
   recipe reaches; the ≤ 1.5× bar might be reachable with a longer run,
   but "minimal training" was the point of the readout.
3. **Rank ordering is preserved through FT**: r=64 recovers less
   (3.91×, agr 0.289) than r=128 everywhere — the bottleneck is a real
   constraint, not a formality, even after adaptation.

Honest summary: the all-layer shared bottleneck is trainable-through
(capability container: yes, cheaply) but short fine-tuning with LM loss
does not recover the original model's function (identity container: no).
The claim "minimal training draws out the container" is supported for
*a* model in the shared subspace, not for *the* model.
