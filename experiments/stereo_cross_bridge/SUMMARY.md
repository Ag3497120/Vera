# Shared bridge experiment — summary (2026-07-21)

## Setup

- Model: Qwen1.5-0.5B-Chat (HF cache)
- Tensors: `mlp.down_proj` layers 0–4, shape `(1024, 2816)`
- Method: per-layer SVD factors → PCA shared `U_bridge`/`V_bridge` → `C_ℓ = U_bᵀ W_ℓ V_b` → compare to truncated SVD

## Results

| rank | SVD rel_fro | bridge rel_fro | bridge/SVD | params bridge/SVD | top-r energy (SVD) | verdict |
|------|-------------|----------------|------------|-------------------|--------------------|---------|
| 64 | 0.911 | 0.987 | 1.08 | 0.22 | 0.17 | COMPRESS_ONLY |
| 128 | 0.842 | 0.973 | 1.16 | 0.23 | 0.29 | COMPRESS_ONLY |
| 256 | 0.717 | 0.940 | 1.31 | 0.27 | 0.49 | WEAK_OR_LOSE |
| 512 | 0.490 | 0.853 | 1.74 | 0.33 | 0.76 | WEAK_OR_LOSE |

`C_valve` off-diagonal mass ≈ 0.99 → when free, the fit **uses** non-diagonal cross terms; they do not rescue quality to SVD parity.

## Interpretation

1. **Order-② first was the right test.** Shared axes alone do not match layer-wise SVD on these `down_proj`s.
2. Low rank: bridge ≈ SVD in *relative* badness (+8–16%) with ~5× fewer params → only a **compression story**, not a quality story (`rel_fro` still ≫ 0.35).
3. Higher rank: gap **widens** — layer-specific bases matter more as more energy is kept.
4. **Do not wire shared-bridge into `jgen_forge` yet.** Negative / weak result on this family+module.

## Next (if continuing)

- Group layers (e.g. every 4 layers share a bridge)
- Try other modules (`gate_proj` / `up_proj` / attn `o_proj`)
- Alternate init (Procrustes alignment of U_ℓ before PCA)
- Joint fine-tune of Ub/Vb (not frozen PCA) — costlier

Raw: `results.json`

---

## Hybrid follow-up: shared low-rank + residual SVD

Script: `hybrid_shared_plus_residual.py`  
`W ≈ Ub C Vb.T + SVD_r(W - Ub C Vb.T)`

| r_share:r_res | SVD(tot) | share-only | hybrid | hyb/svd err | hyb/svd params | verdict |
|---------------|----------|------------|--------|-------------|----------------|---------|
| 64:64 | 0.842 | 0.987 | 0.908 | 1.08 | 0.61 | HYBRID_NEAR |
| 64:128 | 0.778 | 0.987 | 0.841 | 1.08 | 0.74 | HYBRID_NEAR |
| 128:128 | 0.717 | 0.973 | 0.833 | 1.16 | 0.62 | HYBRID_NEAR |
| 128:256 | 0.601 | 0.973 | 0.713 | 1.19 | 0.74 | HYBRID_NEAR |

### Reading

- Hybrid **rescues** share-only (0.99 → ~0.71–0.91) by parking layer-specific energy in residual SVD — supports “partial shared subspace + layer-specific remainder”.
- Still **~8–19% worse** than a single independent SVD with the same total rank budget; saves ~26–39% params.
- Not yet `HYBRID_PROMISING` under a strict quality floor; good enough to keep as a **structural hypothesis**, not to ship into `jgen_forge`.

### Choice vs (2) joint opt / park

- Prefer continuing **(1)-style** (group bridges / other modules) before expensive joint GD.
- Parking down_proj stereo-cross compression remains valid if the goal is product memory/graphs rather than weight codec.

---

## Depth confound (L0–4 vs L19–23)

See `DEPTH_COMPARE.md`. Pre-reg prediction (deep worse) **missed**: bridge/SVD ratios are essentially identical (±0.01). Depth is a small confound for `down_proj`; gate/up_proj comparison can proceed without waiting on a depth rewrite.

Raw: `hybrid_results.json`
