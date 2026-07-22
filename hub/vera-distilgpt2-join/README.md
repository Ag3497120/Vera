---
license: mit
library_name: transformers
tags:
  - vera
  - stereo-cross
  - join
  - distilgpt2
  - bottleneck
datasets:
  - wikitext
base_model: distilbert/distilgpt2
pipeline_tag: text-generation
---

# Vera DistilGPT2 Join Partner (shared stereo-cross P)

Second model of the GPT-2 **join suite**: DistilGPT2 (6 layers, 768-d) distilled onto the **same frozen shared basis `P`** as [`Ag3497120/vera-gpt2-matryoshka`](https://huggingface.co/Ag3497120/vera-gpt2-matryoshka).

Used for **cross-model coordinate join** demos (Step 5 **JOIN_VIABLE**): mid-layer coords from GPT-2 → DistilGPT2 stay near solo quality; random/shuffle controls collapse.

## Bundle layout

| File | Role |
|------|------|
| `model.safetensors` | Fine-tuned DistilGPT2 weights |
| `vera_basis.npz` | DistilGPT2 per-layer `means` + **shared** `P` (same dictionary as GPT-2 Matryoshka) |
| `config.json` | `kind=hook_container`, `role=join_partner` |

## Download with the GPT-2 suite

```bash
python -m vera download-gpt2-suite
# pulls:
#   Ag3497120/vera-gpt2-matryoshka
#   Ag3497120/vera-distilgpt2-join
```

Or select **GPT-2 (Vera suite)** in `python -m vera ui` → Convert/Chat — both repos are fetched.

## Proven join snapshot

| Direction | Site | Real join vs solo | Random control |
|-----------|------|-------------------|----------------|
| GPT-2 → DistilGPT2 | late | ~1.01× solo ppl | tens–hundreds× worse |
| DistilGPT2 → GPT-2 | early | ~1.07× | collapse |

Details: [CROSS_MODEL_JOIN.md](https://github.com/Ag3497120/Vera/blob/main/experiments/stereo_cross_activation/CROSS_MODEL_JOIN.md) in the Vera repo.

## Load

```python
from vera.runtime import load_from_hf
partner = load_from_hf("Ag3497120/vera-distilgpt2-join", rank=256)
```

Must be paired with the Matryoshka GPT-2 bundle that shares the same `P` columns.

## Status

Research partner checkpoint for stereo-cross join — not a standalone chat model.
