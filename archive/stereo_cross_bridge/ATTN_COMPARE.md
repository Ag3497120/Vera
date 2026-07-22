# Attention fork: v_proj (+ k_proj) vs MLP / o / q

Same protocol, Qwen1.5-0.5B-Chat, layers **0–4**.

- new: `results_v_proj_L0-4.json`, `results_k_proj_L0-4.json`
- baselines: `results_down_proj_L0-4.json`, `results_gate_proj_L0-4.json`, `results_up_proj_L0-4.json`, `results_o_proj.json`, `results_q_proj.json`

## Pre-registered fork

- **A)** If `v_proj ≈` MLP cluster (down/up/gate): supports “RoPE / position-sensitive projections hurt shareability more than attention-as-block.”
- **B)** If `v_proj ≈ q_proj` (or between `o` and `q`): supports “attention block / head structure layer-specificity” over RoPE alone.

## Observed bridge/SVD (lower = more shareable)

| module | r=32 | r=64 | r=128 |
|--------|------|------|-------|
| up_proj | 1.039 | 1.075 | 1.147 |
| down_proj | 1.047 | 1.083 | 1.156 |
| gate_proj | 1.064 | 1.106 | 1.185 |
| **v_proj** | **1.080** | **1.147** | **1.279** |
| o_proj | — | 1.189 | 1.345 |
| **k_proj** | **1.212** | **1.324** | **1.529** |
| q_proj | — | 1.328 | 1.533 |

Ordering (r=64/128):

`up ≈ down ≈ gate < v_proj ≤ o_proj ≪ k_proj ≈ q_proj`

## Verdict

**Favors A over B** (with nuance).

- Reject **B**: `v` is not ≈ `q`, and not between `o` and `q` — it is *more* shareable than `o` (`|v−o|≈0.04–0.07` vs `|v−q|≈0.18–0.26`).
- Strict **A** (`v ≈` MLP) does **not** hold: `v` sits just outside the MLP cluster (`|v−mlp_mean|≈0.06–0.12`).
- RoPE check: `k ≈ q` (Δ≈0.004 at r=64/128); both are the least shareable. Non-RoPE attention (`v`, `o`) is clearly better.

**Reading:** worst shareability tracks RoPE / position-sensitive projections (`k`,`q`), not “attention block” in general. Attention still costs something vs MLP (`MLP < v ≤ o`), but the fork data favor RoPE over head-structure-alone.
