# Weights vs hooks — what Vera actually changes

## Short answer

**Both.** Distilled students change **weights**. The shared stereo-cross geometry (**P**, means) is applied with **runtime forward hooks** (unless you use the Step-4 folded compressor).

| Artifact | Weights | Runtime hooks |
|----------|---------|---------------|
| Stock GPT-2 | original | none |
| `matryoshka_student.pt` / `kl_distill_*.pt` | **fine-tuned** under bottleneck | **yes** (frozen P) |
| `student_b_distilgpt2.pt` | **fine-tuned** | **yes** |
| `weight_compress_healed.pt` | **re-parameterized + trained** | folded into matrices |
| Join / memory reinject | no | capture / inject coords |

## Convert scripts

```bash
# Package the proven GPT-2 Matryoshka student into bundles/gpt2-matryoshka
python -m vera convert --package-gpt2

# Interactive: list Ollama + convert/package/chat/verify/publish
python -m vera ui

# After HF login: publish so others can `hf:Ag3497120/vera-gpt2-matryoshka`
export HF_TOKEN=…
python -m vera publish --bundle gpt2-matryoshka
```

Selecting GPT-2 in the UI with **package-existing-student** uses the trained checkpoint (real weight change + hooks).  
**smoke-stock-gpt2** only fits a new P on stock weights (hooks only — poor quality until distilled).

## Ollama

Ollama serves GGUF — Vera cannot attach PyTorch hooks inside Ollama.  
Options in the CUI:

1. Map tag → Hugging Face id and smoke/distill a transformers bundle  
2. Use **Ollama as mouth** while Vera GPT-2 runs **coordinate memory** (`/mouth ollama:qwen3.5:9b` in chat)
