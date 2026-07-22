# Depth confound: down_proj shallow vs deep

Same protocol (`shared_bridge_vs_svd.py`), Qwen1.5-0.5B-Chat (24 layers), module=`down_proj`.

- Shallow band: layers **0–4** → `results_down_proj_L0-4.json`
- Deep band: layers **19–23** → `results_down_proj_L19-23.json`

## Prediction (pre-registered)

Deep layers should share **worse** (higher `err(bridge)/err(svd)`), because task-/context-specific specialization overtakes the “landing pad” commonality of `down_proj`.

## Result: prediction largely **misses**

| rank | band | SVD rel_fro | bridge rel_fro | bridge/SVD | verdict |
|------|------|-------------|----------------|------------|---------|
| 32 | L0–4 | 0.949 | 0.993 | **1.047** | COMPRESS_ONLY |
| 32 | L19–23 | 0.953 | 0.995 | **1.044** | COMPRESS_ONLY |
| 64 | L0–4 | 0.911 | 0.987 | **1.083** | COMPRESS_ONLY |
| 64 | L19–23 | 0.913 | 0.990 | **1.084** | COMPRESS_ONLY |
| 128 | L0–4 | 0.842 | 0.973 | **1.156** | COMPRESS_ONLY |
| 128 | L19–23 | 0.838 | 0.976 | **1.165** | COMPRESS_ONLY |

Deltas (deep − shallow) on the ratio: **−0.003 / +0.001 / +0.009** at r=32/64/128.

Absolute reconstruction stays bad in both bands (bridge ~0.97–0.99). Depth does **not** move the needle in a way that would rewrite the shallow-band module story.

## Interpretation

1. **Depth confound is small** for this setup: within-band shared PCA bridges behave almost the same at the top and bottom of the stack.
2. Prior module order (**down_proj ≳ o_proj > q_proj**) was measured on L0–4; it is still “shallow-band,” but it is **unlikely** that depth alone was fabricating that order — at least for `down_proj`, deep ≈ shallow.
3. Surprising (relative to the pre-reg prediction): `down_proj`’s cross-layer shareability looks **stable through depth**, not progressively task-differentiated in the weight-subspace sense this probe measures.
4. Caveat: this is still **weight Frobenius / shared-subspace** evidence, not activation or task-transfer evidence. “Stable landing pad” is a weight-geometry claim, not a proof of functional invariance.

## What this unlocks next

Depth is no longer the blocking confound for interpreting new modules. Safe to run **gate_proj / up_proj** on the same shallow (and optionally deep) bands and place them as entrance-side counterparts to `down_proj`.
