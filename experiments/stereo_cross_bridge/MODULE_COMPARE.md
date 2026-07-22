# Module expansion: down_proj vs o_proj vs q_proj

Same protocol, Qwen1.5-0.5B-Chat, layers 0–4.

## Shared bridge only (err bridge / err SVD)

| module | r=64 | r=128 | r=256 | note |
|--------|------|-------|-------|------|
| **down_proj** | 1.08 | 1.16 | 1.31 | weakest gap at low r (COMPRESS_ONLY) |
| **o_proj** | 1.19 | 1.35 | 1.70 | similar shape, worse sooner |
| **q_proj** | 1.33 | 1.53 | 1.97 | **least shareable** among the three |

## Hybrid (r_share:r_res → hyb/svd err)

| module | 64:64 | 64:128 | 128:128 |
|--------|-------|--------|---------|
| **down_proj** | 1.08 NEAR | 1.08 NEAR | 1.16 NEAR |
| **o_proj** | 1.16 NEAR | 1.17 NEAR | 1.34 WEAK |
| **q_proj** | 1.20 NEAR | 1.19 NEAR | 1.38 WEAK |

## Claim strength

- Pattern **is not down_proj-only**: o_proj / q_proj also show share-only collapse + hybrid rescue.
- Pattern is **not equally strong**: down_proj ≥ o_proj > q_proj for shared low-rank structure (under this init).
- Still **no PROMISING quality floor** for forge; generalization claim should stay qualitative:
  “partial cross-layer subspace + layer residual” appears in MLP and attention projections, with module-dependent strength.

## Implication for layer grouping

Worth doing **first on down_proj** (strongest signal), optional later on o_proj. q_proj is the wrong place to hunt for shared-bridge wins.

---

## Entrance follow-up (gate / up)

See `ENTRANCE_COMPARE.md`. Shallow L0–4 shared-bridge ratios put **gate ≈ up ≈ down**, all clearly more shareable than **o > q**. Pre-registered `down > o > gate ≈ up > q` **fails** on the gate/up vs o placement.
