# Orthogonal Procrustes: adjacent-layer weight alignment

## Question

Does adjacent-layer `W_{L+1} ≈ R* W_L` (orthogonal `R*`) hold for `q_proj` / `k_proj`, revealing a rotational shared structure that fixed `Ub/Vb + C` cannot see?

## Protocol

- model: `/Users/motonishikoudai/.cache/huggingface/hub/models--Qwen--Qwen1.5-0.5B-Chat/snapshots/4d14e384a4b037942bb3f3016665157c8bcb70ea`
- bands: L0-4, L19-23
- modules: q_proj, k_proj, v_proj, down_proj (v/down = controls)
- metric: relative Frobenius `||W_{L+1} - R W_L||_F / ||W_{L+1}||_F` (also raw `||W_{L+1}-W_L||_F / ||W_{L+1}||_F`)
- baselines: Gaussian same-shape (Frobenius-norm matched); **column-shuffle** and **entry-shuffle** of `W_L` (row-shuffle is invalid for left-Procrustes — absorbed into `R`); cross-module `q_L → v/k_{L+1}`
- fork (practical): `SUPPORTS` only if proc residual < 0.5 **and** real/null < 0.85 vs random, column-shuffle, and entry-shuffle. Tiny z-scores against a concentrated ~0.707 Gaussian floor do **not** count.

## Band L0-4

| module | raw mean | proc_O mean | ex-first | random mean | col-shuf | entry-shuf | real/rand | fork |
|--------|----------|-------------|----------|-------------|----------|------------|-----------|------|
| **q_proj** | 1.6413 | 1.0412 | 0.7581 | 0.7075 | 0.8954 | 0.8189 | 1.472 | `NO_STRUCTURE_UNDER_PROBE` |
| **k_proj** | 1.6038 | 1.0173 | 0.7694 | 0.7075 | 0.8856 | 0.8208 | 1.438 | `NO_STRUCTURE_UNDER_PROBE` |
| **v_proj** | 1.4193 | 0.7267 | 0.6977 | 0.7074 | 0.7577 | 0.7309 | 1.027 | `NO_STRUCTURE_UNDER_PROBE` |
| **down_proj** | 1.4205 | 1.0204 | 1.0107 | 1.0077 | 1.0126 | 1.0076 | 1.013 | `NO_STRUCTURE_UNDER_PROBE` |

### Per-pair (proc_O)

- **q_proj**: L0→1=1.8905, L1→2=0.8312, L2→3=0.7406, L3→4=0.7024
- **k_proj**: L0→1=1.7611, L1→2=0.8402, L2→3=0.7497, L3→4=0.7183
- **v_proj**: L0→1=0.8140, L1→2=0.7151, L2→3=0.6877, L3→4=0.6901
- **down_proj**: L0→1=1.0494, L1→2=1.0125, L2→3=1.0025, L3→4=1.0172

### Cross-module control (`q_L` → tgt `L+1`)

| src → tgt | proc_O mean | raw mean | ratio |
|-----------|-------------|----------|-------|
| q_proj → v_proj | 2.6129 | 3.2699 | 0.7563 |
| q_proj → k_proj | 1.0339 | 1.5989 | 0.6210 |

## Band L19-23

| module | raw mean | proc_O mean | ex-first | random mean | col-shuf | entry-shuf | real/rand | fork |
|--------|----------|-------------|----------|-------------|----------|------------|-----------|------|
| **q_proj** | 1.4226 | 0.7217 | 0.7386 | 0.7073 | 0.9571 | 0.8733 | 1.020 | `NO_STRUCTURE_UNDER_PROBE` |
| **k_proj** | 1.4160 | 0.7048 | 0.7059 | 0.7073 | 0.9112 | 0.8321 | 0.996 | `WEAK_SIGNAL` |
| **v_proj** | 1.4068 | 0.7103 | 0.7216 | 0.7075 | 0.7659 | 0.7064 | 1.004 | `NO_STRUCTURE_UNDER_PROBE` |
| **down_proj** | 1.4295 | 1.0327 | 1.0383 | 1.0077 | 1.0176 | 1.0100 | 1.025 | `NO_STRUCTURE_UNDER_PROBE` |

### Per-pair (proc_O)

- **q_proj**: L19→20=0.6713, L20→21=0.6852, L21→22=0.7345, L22→23=0.7960
- **k_proj**: L19→20=0.7014, L20→21=0.6963, L21→22=0.7239, L22→23=0.6976
- **v_proj**: L19→20=0.6765, L20→21=0.6783, L21→22=0.7310, L22→23=0.7555
- **down_proj**: L19→20=1.0158, L20→21=1.0177, L21→22=1.0262, L22→23=1.0712

### Cross-module control (`q_L` → tgt `L+1`)

| src → tgt | proc_O mean | raw mean | ratio |
|-----------|-------------|----------|-------|
| q_proj → v_proj | 0.8467 | 1.3459 | 0.6288 |
| q_proj → k_proj | 0.8686 | 1.4644 | 0.5932 |

## Verdict

**Fork: NO_STRUCTURE_UNDER_PROBE.**

Adjacent-layer full-W Procrustes does **not** uncover a usable rotational shared map `W_{L+1}≈R W_L` for q/k (nor for v/down controls). Residuals sit at the isotropic Gaussian floor (~0.707 for square attn; ~1.01 for rectangular down). "Uniformly bad shared-bridge" is therefore **not** hiding an orthogonal layer-to-layer structure that fixed `Ub/Vb+C` missed. Redesigning the forge bridge as fixed basis + per-layer `R` is **not** motivated by this probe.

### Key numbers (L0–4)

| module | raw | proc_O | ex-L0 | Gaussian null | col-shuf | entry-shuf |
|--------|-----|--------|-------|---------------|----------|------------|
| q_proj | 1.64 | 1.04 | 0.76 | 0.707 | 0.90 | 0.82 |
| k_proj | 1.60 | 1.02 | 0.77 | 0.707 | 0.89 | 0.82 |
| v_proj | 1.42 | 0.73 | 0.70 | 0.707 | 0.76 | 0.73 |
| down_proj | 1.42 | 1.02 | 1.01 | 1.008 | 1.01 | 1.01 |

### Notes

- **L0→L1** is an outlier for q/k (proc≈1.8–1.9); excluding it still leaves proc≈0.76–0.77, **above** the Gaussian null.
- Real pairs often beat **column/entry shuffle** of the same weights (spectrum/norm structure), but that is not enough: they do **not** beat independent Gaussians, and absolute residual stays ~0.7 (≈70% Frobenius unexplained).
- **v_proj** (no RoPE) is closest to the Gaussian floor; **not** worse than q/k — so there is no RoPE-specific Procrustes win.
- **Deep L19–23**: q/k/v all ≈0.70–0.72 vs random≈0.707 (k slightly under → `WEAK_SIGNAL` only by the mild rule; still nowhere near the practical SUPPORTS thresholds).
- **Cross-module** q→v is worse than same-module; q→k similar to same-module q — no evidence of a privileged q↔q rotational bridge.
- O(n) vs SO(n) residuals are identical for all reported pairs (det±1 does not change the fit).

## Artifacts

- JSON: `results_procrustes.json`
- log: `procrustes_run.log`
- script: `procrustes_layer_align.py`
