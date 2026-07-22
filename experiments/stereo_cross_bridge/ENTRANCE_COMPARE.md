# Entrance MLP modules: gate_proj vs up_proj (+ shallow baseline)

Same protocol, Qwen1.5-0.5B-Chat, layers **0–4**.

- `results_gate_proj_L0-4.json`
- `results_up_proj_L0-4.json`
- baselines: `results_down_proj_L0-4.json`, `results_o_proj.json`, `results_q_proj.json`

## Pre-registered ordering (lower bridge/SVD = more shareable)

`down_proj > o_proj > gate_proj ≈ up_proj > q_proj`

## Observed bridge/SVD

| module | r=32 | r=64 | r=128 |
|--------|------|------|-------|
| up_proj | 1.039 | 1.075 | 1.147 |
| down_proj | 1.047 | 1.083 | 1.156 |
| gate_proj | 1.064 | 1.106 | 1.185 |
| o_proj | — | 1.189 | 1.345 |
| q_proj | — | 1.328 | 1.533 |

## Verdict on prediction

**Does not hold as written.** Actual cluster:

`up_proj ≈ down_proj ≈ gate_proj ≫ o_proj > q_proj`

- `gate_proj ≈ up_proj` holds (Δ ratio ≈ 0.025–0.039).
- MLP modules (gate/up/down) are the shareable cluster; attention `o`/`q` are worse.
- `gate`/`up` are **not** between `o` and `q` — they sit with / slightly above `down_proj`.

gate vs up: no meaningful difference for forge triage (both COMPRESS_ONLY; up slightly better).
