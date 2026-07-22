# Trained bridge (joint GD) vs SVD — minimal-training probe

Script: `trained_bridge_gd.py` · Raw: `results_trained_bridge.json` · Log: `trained_bridge_run.log`

Same structure as the training-free protocol — `W_ℓ ≈ Ub C_ℓ Vbᵀ`, shared `Ub`/`Vb`, per-layer `C_ℓ`, **no** orthogonality constraint — but all factors jointly optimized with Adam (lr 1e-2, cosine decay, ≤2000 steps, plateau early-stop) starting from the frozen-PCA init. Qwen1.5-0.5B-Chat, `down_proj` L0–4, shape `(1024, 2816)`, float32 CPU.

## Question (pre-registered)

Does **minimal training** close the gap to per-layer truncated SVD that the training-free bridge lost (ratios 1.047 / 1.083 / 1.156 at r=32/64/128)?

## Results

### Same-rank comparison

| rank | SVD rel_fro | bridge step-0 | bridge final | ratio step-0 | ratio final | steps→95% | wall |
|------|-------------|---------------|--------------|--------------|-------------|-----------|------|
| 32 | 0.949 | 0.9929 | 0.9834 | **1.046** | **1.036** | 40 | 6.8 s |
| 64 | 0.911 | 0.9869 | 0.9720 | **1.083** | **1.067** | 37 | 11.4 s |
| 128 | 0.842 | 0.9733 | 0.9492 | **1.156** | **1.127** | 34 | 18.6 s |

Step-0 ratios reproduce the training-free numbers (1.047/1.083/1.156) ✓. Training recovers only ~20–25% of the gap; never approaches parity.

### Param-matched comparison (the fair one)

Bridge params `m·r + n·r + L·r²` vs SVD params `L·(m+n)·r_svd`, `r_svd` chosen for equality.

| rank | bridge params | r_svd | SVD rel_fro | bridge final | Δ (bridge − SVD) | verdict |
|------|---------------|-------|-------------|--------------|------------------|---------|
| 32 | 128,000 | 7 | 0.98323 | 0.98338 | +0.00015 | LOSE (hairline) |
| 64 | 266,240 | 14 | 0.97264 | 0.97197 | −0.00067 | WIN (hairline) |
| 128 | 573,440 | 30 | 0.95142 | 0.94916 | −0.00226 | WIN (hairline) |

### Optimization notes

- 95% of the total improvement lands in **~35–40 steps** (lr 1e-2); the rest is a long flat tail. Early stop fired at 500–1000 steps. Total wall clock for all three ranks: **~37 s** on CPU.
- lr=1e-3 sanity run (r=64) converges to the **same** final error (0.97198 vs 0.97197) in ~120 steps-to-95% — the plateau is a genuine optimum of this parameterization, not an optimizer artifact.

## Reading

1. **Same-rank: gap barely closes.** Final ratios 1.036 / 1.067 / 1.127 — still >1.05 at r=64/128, and the gap *grows* with rank exactly as in the training-free runs. Free (non-orthogonal) `Ub`/`Vb` and joint GD do not find shared structure the PCA init missed.
2. **Param-matched: exact parity, not a win.** Trained bridge lands within ±0.25% of the equal-param SVD at every rank — it converges onto the same error-per-param frontier as generic per-layer low-rank, with no bonus from sharing. The r=64/128 "wins" are hairline (0.07% / 0.24% rel_fro) and the r=32 case loses; there is no exploitable margin.
3. **Absolute quality stays terrible** in both columns (rel_fro 0.95–0.98 at matched params) — this whole regime is far below any usable reconstruction floor.

## Verdict (pre-registered fork)

**Fork B — dead.** Minimal training does *not* draw out shared cross-layer structure on this family+module: same-rank ratio stays >1.05 (r≥64), and param-matched performance is statistically indistinguishable from independent truncated SVD. The trained bridge is just a re-parameterization of the same low-rank budget, not a discovery of shared axes. Structural-axis compression via shared bridges on Qwen1.5-0.5B `down_proj` is closed even with training; **do not wire a trainable bridge into `jgen_forge`** on the strength of this probe.

(The only soft caveat: convergence is extremely cheap — ~40 steps — so if a future variant changes the *structure* (grouped bridges, activation-space loss), re-testing with training is nearly free.)
