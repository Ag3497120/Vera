# Step 3 — Matryoshka KL distillation over nested ranks — GPT-2 small (2026-07-22)

Continues `KL_DISTILL_R256.md`. There, a single student distilled under a
frozen hard shared-PCA bottleneck at **r=256** reached test ppl **78.99**
(1.45× baseline) and, when narrowed post-hoc to r=192 without retraining,
still beat a dedicated r=128 student (106.7 vs 122.8). That “train wide,
then narrow” finding is the design seed for Matryoshka: jointly supervise
all nested granularities during training so **one checkpoint** is usable at
every r ∈ {8,16,32,64,128,192,256}.

## Setup

- **Student**: `kl_distill_student_r256.pt` warm start + hard shared
  bottleneck hooks on all 12 block outputs. Projection uses only
  `P[:, :r]` (same means) for a per-step sampled rank r.
  Basis: `bases_cache_soft_distill.npz` (frozen nested PCA; never refit).
- **Teacher**: original GPT-2 small, frozen, no bottleneck.
- **Loss**: forward KL at T=1 (CE vs teacher softmax) + 0.1 × LM — same
  recipe as `kl_distill_r256.py`.
- **Rank schedule**: linear preference toward larger ranks
  (p ∝ r over {8,16,32,64,128,192,256}): r8≈1.1%, …, r256≈36.8%.
- **Training**: AdamW lr 3e-5, wd 0.01, warmup 50, batch 8 × 256, MPS
  float32, eval every 100 on 16 × 256 val (tracked at r=256). **Total
  2500 steps** in two phases (see below).
- **Eval**: same 40 × 256 = 10,240 wikitext-2 test tokens; baseline ppl
  54.466. After training, the **same** checkpoint is evaluated at every
  fixed rank with no further training.
- Script `matryoshka_distill.py`, raw `results_matryoshka_distill.json`,
  log `matryoshka_distill_run.log`, checkpoint `matryoshka_student.pt`
  (local only).

### Phase note

Phase A (600 steps from the r256 student) early-stopped on the
val@r256 plateau criterion inherited from single-rank runs — inappropriate
here, because multi-rank training keeps r=256 nearly flat by design while
lower ranks keep improving. Phase B continued 1900 steps from the Phase-A
checkpoint with plateau disabled. Reported frontier is end of Phase B
(total 2500 steps). Combined train wall **~128 min** (36 + 92).

## Pre-registered fork

- **MATRYOSHKA_VIABLE**: at r=256, ppl ≤ ~1.6× baseline (≤87) OR not worse
  than prior r256 student by >10%; AND at r=128, beats dedicated r=128
  (ppl 122.8) by a clear margin (e.g. <110) **and/or** beats the
  “narrow after r256-only training” nested eval; AND degradation across
  ranks is monotone / graceful (no cliff between adjacent ranks).
- **MATRYOSHKA_WEAK**: multi-rank training helps little vs r256-only
  nested eval; small ranks still collapse.
- Between: state honestly.

## Step-0 control: r256-only student, nested eval (no Matryoshka)

| r | KL | agr | ppl | ppl× |
|---|-----|-----|------|------|
| 8 | 4.40 | 0.023 | 4937 | 90.6 |
| 16 | 4.82 | 0.087 | 6843 | 125.6 |
| 32 | 4.05 | 0.122 | 3120 | 57.3 |
| 64 | 2.54 | 0.226 | 604 | 11.1 |
| 128 | 1.28 | 0.416 | 150.9 | 2.77 |
| 192 | 0.96 | 0.490 | 106.7 | 1.96 |
| 256 | 0.69 | 0.561 | 79.0 | 1.45 |

Note: narrowing the r256-only student all the way to **r=128** (ppl 150.9)
is *worse* than the dedicated r=128 student (122.8). The II-5 “wide→narrow
beats dedicated” claim held at r=192, not at r=128. Also r16 > r8 in ppl
(non-monotone) — the unadapted frontier is not Matryoshka-shaped.

## Final Matryoshka frontier (same checkpoint, fixed r, no retraining)

| r | KL | agr | ppl | ppl× | vs r256-only nested |
|---|-----|-----|------|------|---------------------|
| 8 | 3.79 | 0.098 | 2346 | 43.1 | −52% ppl |
| 16 | 3.06 | 0.170 | 985 | 18.1 | −86% ppl |
| 32 | 2.49 | 0.233 | 540 | 9.9 | −83% ppl |
| 64 | 1.82 | 0.328 | 261 | 4.8 | −57% ppl |
| 128 | 1.06 | 0.479 | **113.4** | 2.08 | −25% vs nested; **−9.4 vs dedicated 122.8** |
| 192 | 0.85 | 0.525 | **92.8** | 1.70 | −13.9 vs nested 106.7 |
| 256 | 0.68 | 0.568 | **77.0** | **1.41** | −2.0 vs prior r256-only 79.0 |

- ppl and agreement are **strictly monotone** in r (no adjacent cliff;
  adjacent ppl ratios 1.2–2.4).
- Mid-run (600-step) frontier already showed the same shape at weaker
  absolute levels (r128 120.5, r256 80.0); Phase B bought another ~7 ppl
  at r128 and ~3 at r256.

## Comparisons (the ones that matter)

| comparison | Matryoshka | Control | Δ |
|------------|------------|---------|---|
| r=256 vs prior r256-only student | 77.03 | 78.99 | **−2.5%** (not worse; slightly better) |
| r=128 vs dedicated r=128 KL student | 113.43 | 122.83 | **−9.4** |
| r=128 vs r256-only nested@128 | 113.43 | 150.85 | **−37.4** |
| r=192 vs r256-only nested@192 | 92.80 | 106.70 | **−13.9** |

## Verdict: **MATRYOSHKA_VIABLE**

- **r=256 clause HIT**: ppl 77.03 ≤ 87 (1.41×), and not worse than the
  prior r256 student (actually −2.5%).
- **r=128 clause HIT via the and/or**: beats dedicated (113.4 < 122.8) and
  decisively beats r256-only nested (150.9). Misses the illustrative
  “clear margin <110” bar by 3.4 ppl — honest near-miss, not a miss of the
  pre-registered or-clause.
- **Graceful / monotone HIT**: ppl↓ and agr↑ at every adjacent step; the
  old all-layer patch cliff (MATRYOSHKA.md) does not reappear once the
  student is jointly supervised.
- **Caveat on small ranks**: r≤32 remain high in absolute ppl (r8≈2346,
  r16≈985) despite large relative gains — usable as a coarse continuum,
  not as a quality operating point. Linear schedule put only ~1–5% of
  steps on those ranks; a flatter schedule could push further.

So: one activation-bottleneck student, one checkpoint, seven nested
operating ranks with graceful degradation — and at every rank that prior
controls measured (128/192/256), Matryoshka matches or beats both the
dedicated narrow student and the “narrow after wide-only training”
baseline. Weight-compressed Matryoshka is left as a follow-up.
