# Shared bridge vs independent SVD (立体十字 go/no-go)

## Question

Does **shared `U_bridge`/`V_bridge` + per-layer dense `C_valve`** reconstruct `down_proj` weights nearly as well as **per-layer truncated SVD**, at lower total storage?

## Run

```bash
cd experiments/stereo_cross_bridge
# uses HF cache Qwen2.5-0.5B* or STEREO_CROSS_MODEL_DIR
python3 shared_bridge_vs_svd.py --layers 5 --ranks 32,64,128
```

## Verdict rule (heuristic)

- `PROMISING`: bridge ≤ 1.25× SVD error **and** mean rel_fro < 0.35 **and** fewer params
- `COMPRESS_ONLY`: bridge ≈ SVD (≤1.25×) and fewer params, but absolute error still high
- `WEAK_OR_LOSE`: otherwise

Only after PROMISING should `jgen_forge` gain shared-bridge write path.
