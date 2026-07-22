---
license: mit
library_name: transformers
tags:
  - vera
  - stereo-cross
  - matryoshka
  - gpt2
  - bottleneck
datasets:
  - wikitext
base_model: openai-community/gpt2
pipeline_tag: text-generation
---

# Vera GPT-2 Matryoshka (stereo-cross container)

Personal-scale **stereo-cross** student: GPT-2 small weights **fine-tuned** under a frozen shared residual bottleneck (rank 256), with **Matryoshka** nested-rank training so one checkpoint works at ranks `{8,16,32,64,128,192,256}`.

This is **not** stock GPT-2. Geometry (`P`, per-layer means) is applied at runtime by Vera hooks (or load via `python -m vera`).

## Bundle layout

| File | Role |
|------|------|
| `model.safetensors` | Fine-tuned GPT-2 weights |
| `vera_basis.npz` | `means` `[12,768]`, `P` `[768,256]` (and full `V` if present) |
| `config.json` | Vera bundle metadata (`kind=hook_container`) |

## Install / download

```bash
pip install transformers safetensors torch huggingface_hub
python -m vera convert --package-gpt2   # or: download suite
python -m vera chat --model hf:Ag3497120/vera-gpt2-matryoshka
```

Selecting **GPT-2** in `python -m vera ui` downloads **this repo** together with the join partner [`Ag3497120/vera-distilgpt2-join`](https://huggingface.co/Ag3497120/vera-distilgpt2-join).

## Proven metrics (wikitext-2, research logs)

| Setting | Result |
|---------|--------|
| r=256 | ppl ≈ **77** (~1.41× vanilla GPT-2 baseline ~54.5) |
| Matryoshka | **MATRYOSHKA_VIABLE** — monotone degradation across ranks |
| Weights vs hooks | Weights trained; `P`/means frozen at inference |

Full chronicle: [Ag3497120/Vera](https://github.com/Ag3497120/Vera)

## How to load (Vera toolkit)

```python
from vera.runtime import load_from_hf
vm = load_from_hf("Ag3497120/vera-gpt2-matryoshka", rank=256)
print(vm.generate("The capital of France is", max_new=20))
vm.set_rank(64)  # Matryoshka lever
```

## Citation / status

Research prototype from the Vera stereo-cross program (2026). Not a general chat replacement for modern LLMs — a **reproducible container** for shared coordinates, memory, and join experiments.
