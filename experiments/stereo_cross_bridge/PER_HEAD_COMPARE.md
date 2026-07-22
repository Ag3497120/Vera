# Per-head fork: specialization vs uniform RoPE

Same shared-bridge vs truncated-SVD protocol, but on **per-head row slices** of `q_proj` / `k_proj`.

- script: `shared_bridge_vs_svd_per_head.py`
- results: `results_per_head_L0-4.json`
- model: Qwen1.5-0.5B-Chat (HF cache), layers **0–4**
- config: `hidden=1024`, `n_q=16`, `n_kv=16` (**not GQA**), `head_dim=64`
- weight layout: HF Linear `(out, in)` → heads are **axis-0** chunks of size `head_dim`
- per-head matrix: **64 × 1024**; rank capped at `63` → swept **r=8,16,32** (r=128 impossible; fractions of `head_dim`: 12.5% / 25% / 50%)

Whole-matrix baselines (from `ATTN_COMPARE.md` / prior JSON):

| module | r=32 | r=64 | r=128 |
|--------|------|------|-------|
| MLP cluster (up/down/gate) | ~1.04–1.06 | **~1.08** | ~1.15–1.19 |
| k_proj (whole) | 1.212 | 1.324 | 1.529 |
| q_proj (whole) | — | 1.328 | 1.533 |

## Pre-registered fork

- **Specialization:** high variance of bridge/SVD across heads (CV ≳ 0.08), and/or some heads ≈ MLP (~1.08) while others remain extreme.
- **Uniform RoPE:** all heads similarly elevated vs MLP, low CV, no MLP-like heads.

Primary readout rank: **r=32** (largest feasible common rank across heads).

## Summary (bridge/SVD)

| module | r | mean | std | CV | min (head) | max (head) |
|--------|---|------|-----|----|------------|------------|
| q_proj | 8 | 1.291 | 0.027 | 0.021 | 1.244 (h1) | 1.360 (h7) |
| q_proj | 16 | 1.497 | 0.035 | 0.024 | 1.442 (h12) | 1.573 (h7) |
| q_proj | **32** | **2.044** | **0.084** | **0.041** | **1.933 (h10)** | **2.193 (h0)** |
| k_proj | 8 | 1.304 | 0.022 | 0.017 | 1.265 (h14) | 1.346 (h3) |
| k_proj | 16 | 1.515 | 0.030 | 0.020 | 1.466 (h10) | 1.566 (h3) |
| k_proj | **32** | **2.063** | **0.077** | **0.037** | **1.941 (h10)** | **2.181 (h4)** |

### Per-head ratios at r=32

**q_proj:**  
`[2.193, 1.936, 2.055, 2.120, 2.146, 2.008, 1.987, 2.142, 2.107, 1.984, 1.933, 1.958, 1.937, 1.991, 2.121, 2.081]`

**k_proj:**  
`[2.175, 2.015, 2.078, 2.112, 2.181, 2.024, 2.016, 2.079, 2.152, 1.995, 1.941, 1.983, 1.971, 2.011, 2.177, 2.091]`

Extremes exist (Δ≈0.26 at r=32) but are **small relative to the mean** (CV≈0.04). Every head is far above the MLP cluster (~1.08); none is MLP-like.

## Comparison notes

- At modest rank (**r=8**, 12.5% of `head_dim`), per-head means (~1.29–1.30) sit in the same band as whole-matrix `k`@r=32 / `q`@r=64 (~1.21–1.33) — still ≫ MLP.
- At **r=32** (50% of `head_dim`), SVD error falls hard while shared-bridge stays high → ratios inflate to ~2.0; this is expected when rank is a large fraction of the small output dim, and does **not** create MLP-like heads.
- Absolute spread grows with rank, but CV stays low (0.02→0.04). No evidence of a “shareable head subset.”

## Verdict

**Favors uniform RoPE structure over head functional specialization.**

- Reject specialization: CV≪0.08 at all ranks; min head @r=32 is still ~1.93 (≫ MLP 1.08).
- Support uniform-RoPE: all 16 heads for both `q` and `k` are elevated and tight around the mean.
- Consistent with `ATTN_COMPARE.md`: worst shareability tracks RoPE-sensitive projections as a class, not a few pathological heads.
