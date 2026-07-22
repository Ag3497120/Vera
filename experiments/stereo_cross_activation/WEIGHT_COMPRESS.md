# Step 4 — turn the r=256 activation container into actual weight compression — GPT-2 small (2026-07-22)

Continues `KL_DISTILL_R256.md`. There, a frozen hard shared-PCA bottleneck at
all 12 block outputs (r=256) with KL distillation reached test ppl **78.99**
(1.45× baseline 54.47) and agreement **0.561**. That only constrains
*activations*; block weights remain full 768-d. This run re-expresses block
weights through the shared 256-d basis, measures real parameter savings, and
heals the compressed parameterization with teacher distillation.

## Setup

- **Student / basis**: `kl_distill_student_r256.pt` + frozen
  `P = V[:, :256]` from `bases_cache_soft_distill.npz` (same nested PCA as
  prior steps). Teacher: original GPT-2 small, frozen.
- **Eval**: wikitext-2-raw-v1 (test), 40 × 256 = 10,240 tokens (baseline ppl
  54.466). Device: MPS float32.
- **Phase A**: analytic input-side truncation of `c_attn` on blocks 1–11
  (block 0 input is raw embeddings). *Honest* uses
  `Q = orth(diag(g_ln1)[P | mean_in | 1])` (rank ≤258) plus bias fix
  `b' = b + b_ln (I−QQᵀ) W` (exact for the LN affine). *Naive* uses
  `P̃ = orth([P, 1])` with no g / mean / bias fix.
- **Phase B1**: add mid-block projection
  `h_mid → mean_mid + P Pᵀ (h_mid − mean_mid)` after the attention residual
  (fresh mid means, 200 × 256 calib tokens → `mid_means_weight_compress.npz`)
  on top of existing output projections; zero-shot.
- **Phase B2**: re-parameterize each block natively in 256-d coords `c`.
  LN computed **exactly** in coords via a v/w split (`v` = centered
  per-block mean, constant; `w` in global span(`[P | q1]`)); `diag(g)`,
  means, and biases fold into factorized matrices — no
  reconstruct-LN-project roundtrip. Residual projections → `B = W @ P`;
  block 0 hybrid (full `c_attn`, coords after attention residual). P stored
  once globally. B2 must match B1 up to float error.
- **Healing (B5)**: distill original GPT-2 into the compressed model
  (factorized matrices + embeddings + `ln_f` trainable; P and geometry
  buffers frozen). Loss = forward soft CE vs teacher + 0.1 × LM. AdamW
  lr 5e-5, wd 0.01, warmup 50, batch 8 × 256, ≤2000 steps, eval every 100
  on 16 × 256 val, plateau stop after step 600 if best KL and agreement
  both improve <1% over the last 400 steps.
- Script `weight_compress.py` (`--stage zeroshot|heal|all`), raw
  `results_weight_compress.json`, log `weight_compress_run.log`, mid means
  `mid_means_weight_compress.npz`, healed ckpt `weight_compress_healed.pt`
  (local; not published). Prior artifacts untouched.

## Pre-registered fork

- **COMPRESSION_REAL**: healed test ppl ≤ ~2× baseline (≤109) **and**
  ≥2.5× block-weight compression.
- **CONTAINER_ONLY**: healed test ppl > 3× baseline (>163).
- **BETWEEN**: otherwise, state honestly.

## Phase A — input-side compression is near-exact when LN is honest

| variant | mean weight frac discarded (L1–11) | test KL | agreement | ppl |
|---------|--------------------------------------|---------|-----------|-----|
| naive (`P̃`, no bias fix) | 0.805 | 1.709 | 0.379 | 212.3 |
| **honest** (`Q` + bias fix) | 0.803 | **0.693** | **0.561** | **78.99** |

Honest matches the untouched r=256 student bit-for-bit on the test metrics
(same KL / agreement / ppl as `baseline_r256_student`). Naive destroys the
model — LN's affine map really does take the input out of `span(P, 1)`.

## Phase B — full containment zeroshot (damage expected)

Mid-mean ‖·‖₂ grows with depth (L0 ≈ 20.5 → L11 ≈ 308.5). Adding mid-block
projections (B1) and the native-coord fold (B2):

| stage | test KL | agreement | ppl |
|-------|---------|-----------|-----|
| r=256 student (activation container only) | 0.693 | 0.561 | 78.99 |
| B1 zeroshot (output + mid projections) | 1.943 | 0.307 | 312.3 |
| B2 zeroshot (folded compressed weights) | 1.943 | 0.307 | 312.3 |

B2 vs B1 logit parity: max |Δ| **1.07e-3**, mean |Δ| **9.05e-5** — fold is
faithful; the damage is the mid-block container, not a folding bug.

**Parameter counts (B2):**

| scope | original | compressed | ratio |
|-------|----------|------------|-------|
| blocks total | 85,054,464 | 29,891,631 | **2.85×** |
| full model (incl. emb + ln_f) | 124,439,808 | 69,276,975 | **1.80×** |

Per-block (L1–11): 7,087,872 → 2,376,452; block 0 hybrid 3,552,770; global
container buffers 197,889. Trainable under heal: **69,073,152** (P frozen).

## Healing distillation (resume completed full 2000 steps)

Zeroshot already done; heal restarted with `--stage heal` after a mid-run
crash around step 120 (log separator `=== RESUME heal … (harness-bg) ===`).
Restarted from the folded unhealed init (no optimizer ckpt). Plateau stop
**never fired**.

### Validation curve

| step | KL | agreement | own ppl | own acc |
|------|------|-------|--------|--------|
| 0 (folded) | 1.965 | 0.300 | 280.6 | — |
| 100 | 1.309 | 0.422 | 125.5 | 0.220 |
| 200 | 1.178 | 0.445 | 109.5 | 0.229 |
| 400 | 1.071 | 0.468 | 99.1 | 0.241 |
| 600 | 1.009 | 0.484 | 92.2 | 0.248 |
| 800 | 0.978 | 0.497 | 90.4 | 0.250 |
| 1000 | 0.953 | 0.500 | 87.5 | 0.250 |
| 1200 | 0.932 | 0.506 | 86.2 | 0.254 |
| 1400 | 0.913 | 0.510 | 84.3 | 0.260 |
| 1600 | 0.897 | 0.514 | 83.9 | 0.257 |
| 1800 | 0.888 | 0.521 | 82.5 | 0.261 |
| 2000 | 0.872 | 0.521 | 81.1 | 0.259 |

- Wall time **5,070 s (~84.5 min)** for 2000 steps (~2.5 s/step incl.
  teacher forward + periodic val). Steps to 95% of achieved improvement:
  **1300** (val KL), **1600** (agreement), **700** (own ppl).
- Val ppl crossed the COMPRESSION_REAL ceiling (≤109) by step **200** and
  kept improving through the budget.

### Final test (10,240 tokens)

| metric | healed compressed | r=256 activation student | GPT-2 baseline |
|--------|-------------------|--------------------------|----------------|
| KL vs teacher | **0.833** | 0.693 | — |
| top-1 agreement | **0.522** | 0.561 | — |
| ppl | **88.99** (1.63×) | 78.99 (1.45×) | 54.47 |
| top-1 acc | 0.258 | 0.270 | 0.310 |

## Verdict: COMPRESSION_REAL

Healed test ppl **88.99 ≤ 109** (1.63× baseline) at **2.85× ≥ 2.5×**
block-weight compression (full-model 1.80×). The r=256 shared PCA container
is not activation-theatre only: after exact LN-in-coords folding and short
teacher distillation, the compressed weights recover near-student quality
under the pre-registered bar.

Caveats worth keeping honest: agreement (0.522) is still below the soft
0.6 aspiration from earlier container work; full-model ratio is pulled down
by untouched embeddings + `ln_f` (~39.4M params); heal used the full 2000
step budget (no plateau). Artifacts: `results_weight_compress.json`, this
note, `mid_means_weight_compress.npz`, `weight_compress_run.log`; local
ckpt `weight_compress_healed.pt`.
