# KL distillation into the frozen-bottleneck container — GPT-2 small (2026-07-22)

Decisive follow-up to `SOFT_DISTILL.md`. There, plain LM fine-tuning through
the frozen hard shared-PCA bottleneck (r=128, all 12 block outputs) recovered
capability (ppl 27.1× → 2.51× baseline) but not identity (top-1 agreement
with the original model 0.341), with two caveats: the LM objective never
asked for agreement, and the 1000-step budget capped the run before a
plateau. This run targets the agreement clause directly: KL-distill the
original GPT-2 (teacher) into the bottlenecked student with 2.5× the budget.

## Setup

- **Warm start: none.** `soft_and_distill.py` never saved its Phase-2
  checkpoint (`run_phase2` ends with `del model`), so the student starts
  from pretrained GPT-2 (lr 5e-5 per protocol, not the 3e-5 warm-start
  variant).
- **Student**: GPT-2 small + the SAME frozen r=128 hard bottleneck on all
  12 block outputs, `h → mean_l + P Pᵀ (h − mean_l)`, basis/means loaded
  from `bases_cache_soft_distill.npz` (not refit). Step-0 validation
  reproduces the prior run exactly (KL 3.3824, agr 0.1772, ppl 1279.5).
- **Teacher**: original GPT-2 small, no bottleneck, frozen, eval, no_grad.
- **Loss**: forward KL at T=1 — `CE(student_logits, softmax(teacher_logits))`
  with full-vocab soft targets each step (teacher forward per batch) —
  **plus 0.1 × plain LM loss** for stability. Train KL logged as
  CE − H(teacher).
- **Training**: wikitext-2-raw-v1 train, batch 8 × 256, AdamW lr 5e-5
  (wd 0.01), linear warmup 50 then constant, grad clip 1.0, budget 2500
  steps (≈ 5.1M tokens ≈ 2.5 epochs; data cycles with reshuffle). Eval
  every 100 steps on the 16 × 256 validation slice: KL, top-1 agreement vs
  teacher, own ppl/acc. Plateau early stop after step 600: best val KL AND
  best val agreement both improved < 1% (relative) over the last 300 steps.
  **Never fired** — agreement was still improving ~1.1%/300 steps at budget
  end. Device: MPS float32.
- **Eval**: same wikitext-2 test tokens as all prior runs (40 × 256 =
  10,240; baseline ppl 54.466, acc 0.3102).
- Script `kl_distill_bottleneck.py`, raw `results_kl_distill.json`, log
  `kl_distill_run.log`, checkpoint `kl_distill_student_r128.pt` (saved this
  time). Prior artifacts untouched. The optional r=64 secondary was skipped
  per protocol (r=128 hit neither target early).

## Pre-registered fork (r=128)

- **IDENTITY_RECOVERED**: agreement ≥ 0.6 AND/OR ppl ≤ 1.5× baseline
  (≤ ~82) — the shared-basis container can host a close functional copy of
  the original model with modest distillation.
- **CAPABILITY_ONLY_CONFIRMED**: agreement plateaus < ~0.45 despite the KL
  objective and longer budget — the hard r=128 all-layer bottleneck
  fundamentally loses too much token-level information.
- Between: state honestly.

## Training curve (validation slice, vs teacher)

| step | KL | agreement | own ppl | own acc |
|------|------|-------|--------|--------|
| 0 | 3.382 | 0.177 | 1279.5 | 0.089 |
| 100 | 1.861 | 0.320 | 249.4 | 0.168 |
| 300 | 1.607 | 0.364 | 184.2 | 0.195 |
| 500 | 1.495 | 0.387 | 160.6 | 0.204 |
| 1000 | 1.343 | 0.413 | 136.9 | 0.218 |
| 1500 | 1.258 | 0.434 | 124.6 | 0.229 |
| 2000 | 1.194 | 0.453 | 115.0 | 0.233 |
| 2500 | 1.144 | 0.462 | 109.3 | 0.238 |

- Wall time **8164 s (136 min)** for 2500 steps on MPS (~3.3 s/step incl.
  teacher forward). Steps to 95% of achieved improvement: **1600** (val KL),
  **2000** (agreement), 500 (own ppl) — unlike the LM-loss run (95% of ppl
  recovery inside 300 steps), the KL objective's gains are spread across
  the whole budget.
- The objective effect is visible at matched budget: at step 1000 the KL
  run has val agreement **0.413** vs the LM-loss run's **0.346**, and val
  KL **1.34** vs **2.19** — not merely a longer-budget artifact.
- Late-curve agreement slope: +0.010 per 500 steps and decelerating
  (0.453 → 0.462 over the last 500). Linear extrapolation to 0.6 would need
  ≳ 7000 more steps; the curve is sublinear, so realistically it is
  flattening in the high-0.4s on this data.

## Final test results (same 10,240 tokens; baseline ppl 54.47, acc 0.310)

| run | steps | ppl | ppl× | agr vs orig | KL | own acc |
|-----|-------|------|------|------|------|------|
| pre-FT (hard r=128, all layers) | 0 | 1476.4 | 27.1 | 0.169 | 3.36 | 0.086 |
| LM-loss FT (`SOFT_DISTILL.md`) | 1000 | 136.7 | 2.51 | 0.341 | 2.13 | 0.244 |
| **KL distill (this run)** | 2500 | **122.8** | **2.26** | **0.470** | **1.12** | 0.231 |
| targets | — | ≤ 82 | ≤ 1.5 | ≥ 0.6 | — | — |

KL distillation beats the LM-loss run on every teacher-matching metric —
agreement 0.341 → **0.470**, mean KL 2.13 → **1.12** (halved) — and even on
own ppl (136.7 → 122.8) despite ppl not being the objective. Own top-1
accuracy dips slightly (0.244 → 0.231): the student is matching the
teacher's full distribution rather than sharpening on the data, as expected.

**Both pre-registered targets missed**: agreement 0.470 < 0.6; ppl 122.8 >
82 (2.26× > 1.5×).

### Verdict: **between the forks, leaning CAPABILITY_ONLY — the KL objective helps substantially but the trajectory flattens far below identity**

- **IDENTITY_RECOVERED is clearly not reached.** With the objective aimed
  squarely at agreement and 2.5× the budget, the student agrees with the
  original on only 47% of next-token argmaxes (the original's own top-1
  accuracy is 31%, so this is far from a functional copy), and quality
  stays at 2.26× baseline ppl.
- **CAPABILITY_ONLY_CONFIRMED is not technically triggered either**: its
  plateau clause says < ~0.45, and agreement finished at 0.462 val / 0.470
  test, still improving ~1.1%/300 steps at budget end (which is exactly why
  the pre-registered early stop never fired).
- Honest reading: the level is a hair above the 0.45 line but the
  *shape* of the curve supports the capability-only conclusion. Getting
  from 0.34 to 0.47 took a direct KL objective plus 2.5× compute; each
  additional 500 steps now buys ~0.01 agreement and the slope is falling.
  Nothing in this curve suggests 0.6 is reachable at this rank on this
  data without an order-of-magnitude more distillation — at which point
  "modest distillation" (the identity-viable claim) is already false.
- Conclusion for the stereo-cross container: the frozen r=128 all-layer
  shared-PCA bottleneck holds *a* capable language model (drawn out
  cheaply, per `SOFT_DISTILL.md`) and can be pushed measurably toward the
  original with distillation, but it does **not** host a close functional
  copy of the original at modest cost. Container status stays
  **capability-viable, not identity-viable**, with the caveat that the
  agreement curve was still (slowly) rising when the budget ended.
