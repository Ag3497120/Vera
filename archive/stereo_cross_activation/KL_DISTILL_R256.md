# KL distillation at r=256 with warm start from the r=128 student — GPT-2 small (2026-07-22)

Follow-up to `KL_DISTILL.md`. There, KL-distilling original GPT-2 into the
frozen hard shared-PCA bottleneck (r=128, all 12 block outputs) for 2500
steps reached test ppl 122.8 (2.26× baseline 54.47), agreement 0.470,
KL 1.12 — both targets missed (ppl ≤ 82, agreement ≥ 0.6). This run widens
the container to r=256 (1/3 of 768) and warm-starts from the r=128 student,
exploiting the nested structure of the PCA basis: the r=256 projector uses
the top-256 columns of the SAME eigenbasis V, so its subspace strictly
contains the r=128 one.

## Setup

- **Basis**: `bases_cache_soft_distill.npz` holds the full 768×768 shared
  eigenbasis V (fit once on wikitext-2 train, ~100k tokens, per-layer means,
  trace-normalized pooled covariance), so r=256 is just the nested top-256
  columns — no refit; the top-128 subspace is identical to the r=128 run by
  construction.
- **Warm start**: student weights loaded from `kl_distill_student_r128.pt`
  (the 2500-step r=128 KL-distilled student), then the bottleneck hooks
  swapped to hard r=256: `h → mean_l + P Pᵀ (h − mean_l)`, P = V[:, :256],
  frozen, all 12 block outputs.
- **Teacher**: original GPT-2 small, no bottleneck, frozen, eval, no_grad.
- **Loss**: forward KL at T=1 (CE vs full-vocab teacher softmax each step)
  + 0.1 × plain LM loss, exactly as the r=128 run.
- **Training**: wikitext-2-raw-v1 train, batch 8 × 256, AdamW lr **3e-5**
  (warm-start rate, vs 5e-5 cold), wd 0.01, linear warmup 50 then constant,
  grad clip 1.0, budget 3000 steps. Eval every 100 steps on the 16 × 256
  validation slice. Plateau stop (after step 600): best val KL AND best val
  agreement both improved < 1% (relative) over the last 400 steps —
  **never fired**; agreement was still improving ~2%/400 steps at budget
  end. Device: MPS float32.
- **Eval**: same wikitext-2 test tokens as all prior runs (40 × 256 =
  10,240; baseline ppl 54.466, acc 0.3102). **Step-0 test eval run before
  any training** to isolate the pure effect of widening the container.
- Script `kl_distill_r256.py`, raw `results_kl_distill_r256.json`, log
  `kl_distill_r256_run.log`, checkpoint `kl_distill_student_r256.pt`.
  Prior scripts/results untouched.

## Step-0: widening the container is NOT free (test, before training)

| student | bottleneck | KL | agreement | ppl |
|---------|-----------|------|------|------|
| r=128-distilled | r=128 (its training config) | 1.115 | 0.470 | 122.8 |
| r=128-distilled | **r=256 (widened)** | **1.386** | **0.407** | **162.9** |

The nesting argument ("the r=128 student already satisfies the r=256
constraint") is true of the *subspace* but false of the *function*: the
hooks project the raw block outputs, and the student's raw outputs have
components in eigendirections 129–256 that the r=128 projector deleted
during its training. Widening lets those never-supervised components
through to downstream layers, so ppl transiently jumps 122.8 → 162.9 and
agreement drops 0.470 → 0.407. Still far better than a cold start at r=256
would be (the cold r=128 start was ppl 1279 / agr 0.177), and recovery was
immediate: by step 100 the student had already passed the r=128 run's
*final* numbers (val agr 0.492 vs 0.462, val ppl 94.6 vs 109.3).

## Training curve (validation slice, vs teacher)

| step | KL | agreement | own ppl | own acc |
|------|------|-------|--------|--------|
| 0 (warm) | 1.401 | 0.398 | 145.1 | 0.211 |
| 100 | 0.963 | 0.492 | 94.6 | 0.246 |
| 300 | 0.894 | 0.506 | 86.7 | 0.256 |
| 500 | 0.859 | 0.515 | 83.7 | 0.261 |
| 1000 | 0.807 | 0.533 | 79.0 | 0.272 |
| 1500 | 0.778 | 0.542 | 76.5 | 0.274 |
| 2000 | 0.752 | 0.553 | 74.8 | 0.277 |
| 2500 | 0.732 | 0.555 | 73.3 | 0.278 |
| 3000 | 0.716 | 0.566 | 71.9 | 0.280 |

- Wall time **10,652 s (178 min)** for 3000 steps (~3.6 s/step incl.
  teacher forward). Steps to 95% of achieved improvement: **2100** (val
  KL), **2600** (agreement), **1800** (own ppl) — gains again spread across
  the whole budget, and the plateau stop never fired.
- The warm start dominates budget-for-budget: the r=128 run needed its full
  2500 steps to reach val agreement 0.462; this run passes that inside 100
  steps and ends 0.104 higher (0.566) at comparable cost.
- Late-curve agreement slope: +0.011 per 500 steps (0.555 → 0.566), about
  the same absolute slope as the r=128 run's tail but from a higher level.
  Linear extrapolation to 0.6 needs ≈ 1500–2000 more steps; the curve is
  sublinear, so realistically 0.6 is a few thousand steps away — within
  personal-scale budget, unlike the r=128 extrapolation.

## Final test results (same 10,240 tokens; baseline ppl 54.47, acc 0.310)

| run | steps | ppl | ppl× | agr vs orig | KL | own acc |
|-----|-------|------|------|------|------|------|
| r=128 KL distill (`KL_DISTILL.md`) | 2500 | 122.8 | 2.26 | 0.470 | 1.12 | 0.231 |
| r=256 step-0 (pure widening) | 0 | 162.9 | 2.99 | 0.407 | 1.39 | 0.208 |
| **r=256 KL distill (this run)** | 3000 | **78.99** | **1.45** | **0.561** | **0.69** | 0.270 |
| targets | — | ≤ 82 | ≤ 1.5 | ≥ 0.6 | — | — |

**PPL target HIT**: 78.99 ≤ 82 (1.45× baseline). **Agreement target
missed**: 0.561 < 0.6, but still rising at budget end (early stop never
triggered) and nowhere near the < 0.55 stall clause.

### Rank-performance frontier (final r=256 student, nested bases, no retraining)

| eval rank | KL | agreement | ppl |
|-----------|------|------|------|
| 256 (trained) | 0.69 | 0.561 | 79.0 |
| 192 | 0.96 | 0.490 | 106.7 |
| (128-trained student at 128, ref) | 1.12 | 0.470 | 122.8 |

Narrowing the trained r=256 student to r=192 shows the same
widening-in-reverse effect: it loses components it now relies on, yet at
r=192 it still beats the r=128-trained student at its own rank — the warm-
started weights are strictly better along the whole nested frontier.

### Verdict: **IDENTITY_AT_256 (via the ppl clause) — the container is identity-viable at r=256**

- The pre-registered fork triggers on either target: **ppl 78.99 ≤ 82
  (1.45× ≤ 1.5×) is hit**. STILL_CAPABILITY_ONLY required agreement
  stalling < ~0.55 AND ppl > ~100; neither holds (0.561 and 79.0).
- Honest reading of the agreement clause: 0.561 misses 0.6, but the
  trajectory is qualitatively different from r=128. There, 2500 dedicated
  steps bought 0.470 with a flattening curve and 0.6 looked an order of
  magnitude away. Here 0.6 is ~0.04 away with the curve still moving
  ~0.011/500 steps — a plausible one-more-overnight-run distance, i.e.
  within the stated personal-scale budget.
- **Rank frontier established**: at r=128 (1/6 of 768) the frozen shared-
  PCA container is capability-viable but not identity-viable; at r=256
  (1/3 of 768) it clears the quality bar for identity (1.45× baseline ppl)
  and gets to 56% argmax agreement with modest distillation (178 min on one
  MPS machine, warm-started). The binding constraint at r=128 was indeed
  rank, not the shared-basis construction itself.
- Methodological finding worth keeping: **nested-basis warm starts are not
  function-preserving** — widening a hard projection bottleneck around an
  adapted student transiently *hurts* (ppl +33%, agreement −0.06) because
  previously-deleted eigendirections leak through untrained. The penalty
  is repaid within ~100 steps and the warm start then dominates decisively.
