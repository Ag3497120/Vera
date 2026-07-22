# Activation-space shared structure — GPT-2 small (2026-07-22)

Continues `experiments/stereo_cross_bridge/`: shared-basis reconstruction of
**weights** lost to per-layer SVD (bridge/SVD 1.05–1.5, uniformly, even with
training). Question here: does the **activation** space of the same kind of
model share cross-layer structure that the weight space does not?

## Setup

- Model: `openai-community/gpt2` (124M, 12 layers, d=768), float32 CPU
- Corpus: **wikitext-2-raw-v1 (train)**, docs >200 chars, shuffled + packed
  into 390 × 256-token sequences = **99,840 tokens**; **15,000 token vectors
  per layer** subsampled (same random positions for every layer → paired CKA)
- Sites: `resid` = block output (`hidden_states[l+1]`, 768-d),
  `mlp_out` = MLP after `c_proj` before residual add (768-d),
  `mlp_hidden` = after GELU (3072-d)
- Centering: each layer mean-centered independently; the **shared** PCA basis
  comes from the trace-normalized pooled covariance (equal per-layer variance
  weight, so high-norm late layers don't dominate). Raw pooling reported in
  the JSON too — it changes ratios by <0.05, same story.
- Scripts: `activation_shared_probe.py` (Part A), `gpt2_weight_bridge.py`
  (Part B). Raw: `results_activation_probe.json`,
  `results_gpt2_weight_bridge.json`, log `activation_probe_run.log`.

## Part A — pre-registered prediction

Activation space shares much more than weight space: shared/per-layer error
ratio close to 1.0 (**<1.10 at r=32**) and adjacent-layer CKA high. If
instead ratio is weight-space-bad (**>1.3**) → evidence AGAINST the "probe
activations to find shared cross structure" direction.

## Part A — results

Shared-PCA vs per-layer-PCA relative reconstruction error (ratio = headline,
analog of bridge/SVD):

| site | r=16 | r=32 | r=64 | per-layer err @r=32 |
|------|------|------|------|---------------------|
| resid | 1.090 | **1.087** | 1.099 | 0.31 |
| mlp_out | 1.052 | **1.070** | 1.103 | 0.66 |
| mlp_hidden | 1.102 | **1.143** | 1.192 | 0.82 |

Linear CKA (paired tokens, column-centered):

| site | adjacent mean | adjacent min | off-diag mean |
|------|---------------|--------------|---------------|
| resid | 0.859 | 0.040 | 0.730 |
| mlp_out | 0.525 | 0.209 | 0.274 |
| mlp_hidden | 0.739 | 0.263 | 0.518 |

Resid adjacent CKA per pair: 0.43 (L0–1), **0.99–1.00 for L1–10**, 0.04
(L10–11). The mid-stack residual stream barely rotates; the two boundary
anomalies are the embedding-adjacent first block and the well-known GPT-2
final-block rotation toward the LM head. (Caveat: linear CKA on GPT-2 resid
is partly inflated by the few huge outlier dimensions.)

### Verdict (Part A)

**Prediction CONFIRMED for the 768-d sites, partial for mlp_hidden.**

- `resid` 1.087 and `mlp_out` 1.070 at r=32 are **<1.10** ✓; `mlp_hidden`
  1.143 misses the bar but is nowhere near the 1.3 failure line ✗→soft.
- Two qualitative differences from weight space:
  1. **Ratios are flat in rank** (resid: 1.090 → 1.087 → 1.099). In weight
     space the ratio *grew* with rank everywhere (1.05 → 1.5), the signature
     of layer-specific bases mattering more as energy is kept. Activations
     don't show that signature at these ranks.
  2. **Absolute quality is real.** Per-layer PCA err at r=32 is 0.31 on resid
     — activations are genuinely low-rank. Weight SVD at the same rank sat at
     0.83–0.95 rel_fro, below any usable floor. A shared 64-dim basis
     reconstructs every layer's residual stream to ~0.30 rel err.
- **Do not** read this as evidence against the activation-probe direction;
  it is the first positive sharing result in the series. Cheapest next step:
  shared basis + per-layer r×r valve (the exact bridge parameterization) on
  activations, and check whether cross-layer C_ℓ's are related.

## Part B — GPT-2 weight-space replication (RoPE fork)

Same protocol as `stereo_cross_bridge/shared_bridge_vs_svd.py`, reimplemented
in `gpt2_weight_bridge.py` for GPT-2 Conv1D weights (transposed to (out,in);
fused `c_attn` split into q/k/v thirds). Layers 0–4, ranks 32/64.

### Pre-registered fork

GPT-2 has **no RoPE**. In Qwen, q/k were worst (1.33–1.53) and
`ATTN_COMPARE.md` credited RoPE. If GPT-2 q/k are ALSO much worse than its
MLP → RoPE explanation weakens, attention-block layer-specificity
strengthens. If GPT-2 q/k ≈ its MLP → RoPE explanation strengthens.

### Observed bridge/SVD (lower = more shareable)

| module | r=32 | r=64 | (Qwen r=32) | (Qwen r=64) |
|--------|------|------|-------------|-------------|
| mlp_fc | 1.092 | 1.151 | 1.039–1.064 | 1.075–1.106 |
| mlp_proj | 1.113 | 1.170 | 1.047 | 1.083 |
| v | 1.135 | 1.240 | 1.080 | 1.147 |
| **q** | **1.151** | **1.272** | — | **1.328** |
| **k** | **1.178** | **1.298** | **1.212** | **1.324** |
| o | 1.197 | 1.326 | — | 1.189 |

Ordering (r=64): `mlp_fc < mlp_proj ≪ v < q ≈ k < o`.

### Verdict (Part B)

**RoPE explanation WEAKENS; attention-block layer-specificity strengthens.**

- GPT-2 q/k (1.27–1.30 at r=64) are clearly worse than GPT-2 MLP (1.15–1.17)
  **without any RoPE**. The MLP < attention gap does not require RoPE.
- Extra nail: in GPT-2 the *worst* module is `o` (1.326) — a projection that
  is position-independent in both families. The Qwen "q/k uniquely worst,
  o distinctly better" pattern does **not** replicate; here q/k/o cluster
  together above v above MLP.
- Soft caveat for RoPE: Qwen's q/k excess over its own o (≈0.14) is larger
  than GPT-2's (q/k ≈ o here), so RoPE may still add *some* layer-specificity
  on top — but it cannot be the primary explanation for attention < MLP
  shareability.
- Weight-space ratios on GPT-2 (1.09–1.33) replicate the Qwen negative
  result: shared weight bases lose everywhere, growing with rank. Part A's
  flat ~1.07–1.09 activation ratios with 3× lower absolute error is the
  contrast this experiment was designed to expose.
