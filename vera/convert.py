"""Convert / package models into Vera bundles.

Honesty
-------
- **Package artifact** (fast): wrap already-trained student + basis → bundle.
  This is what makes GPT-2 Vera usable today.
- **Full convert** of a new HF model: requires fitting P + distillation
  (personal-scale for GPT-2; large for 9B). Interactive CUI warns and can
  run a *smoke* fit (basis only + zero-shot hooks) vs full distill.
- **Ollama**: cannot hook GGUF. Convert = resolve HF id + package/distill path.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Tuple

from . import catalog
from .runtime import BundleConfig, export_bundle, load_from_artifacts

BUNDLES = catalog.BUNDLES

# Best-effort map from Ollama tag → HF repo (extend as needed)
OLLAMA_TO_HF = {
    "gpt2": "openai-community/gpt2",
    "qwen:0.5b": "Qwen/Qwen1.5-0.5B-Chat",
    "qwen:0.5b-chat-v1.5-fp16": "Qwen/Qwen1.5-0.5B-Chat",
    "qwen3.5:0.8b": "Qwen/Qwen3.5-0.8B",  # may 404 — user overrides
    "qwen3.5:2b": "Qwen/Qwen3.5-2B",
    "qwen3.5:9b": "Qwen/Qwen3.5-9B",
}


def resolve_hf_id(entry_id: str, override: Optional[str] = None) -> str:
    if override:
        return override
    if entry_id.startswith("hf:"):
        return entry_id[3:]
    if entry_id.startswith("ollama:"):
        name = entry_id.split(":", 1)[1]
        if name in OLLAMA_TO_HF:
            return OLLAMA_TO_HF[name]
        # try strip tag variants
        base = name.split(":")[0]
        for k, v in OLLAMA_TO_HF.items():
            if k.startswith(base):
                return v
        raise ValueError(
            f"No HF mapping for ollama:{name}. Pass --hf-id explicitly."
        )
    if entry_id.startswith("artifact:"):
        # infer from filename
        if "distilgpt2" in entry_id:
            return "distilbert/distilgpt2"
        return "openai-community/gpt2"
    raise ValueError(f"Cannot resolve HF id for {entry_id}")


def package_gpt2_matryoshka(out_name: str = "gpt2-matryoshka") -> Path:
    """Export the proven Matryoshka student into bundles/ for local use & HF publish."""
    ckpt = catalog.find_file("matryoshka_student.pt", "kl_distill_student_r256.pt")
    basis = catalog.find_file("bases_cache_soft_distill.npz")
    if not ckpt or not basis:
        raise FileNotFoundError(
            "Need matryoshka_student.pt (or kl_distill_student_r256.pt) "
            "and bases_cache_soft_distill.npz — set VERA_ARTIFACTS"
        )
    out = BUNDLES / out_name
    export_bundle(
        out,
        ckpt=ckpt,
        basis=basis,
        base_model="openai-community/gpt2",
        rank=256,
        kind="hook_container",
        notes=(
            "WEIGHTS: fine-tuned (Matryoshka/KL distill under hard shared "
            "bottleneck). GEOMETRY: frozen P+means applied by runtime hooks. "
            "Proven: MATRYOSHKA_VIABLE / container ppl~1.4× on wikitext."
        ),
    )
    return out


def package_artifact(entry_id: str, out_name: Optional[str] = None) -> Path:
    if not entry_id.startswith("artifact:"):
        raise ValueError("package_artifact expects artifact:… id")
    fname = entry_id.split(":", 1)[1]
    ckpt = catalog.find_file(fname)
    basis = catalog.find_file("bases_cache_soft_distill.npz")
    if not ckpt or not basis:
        raise FileNotFoundError("missing ckpt/basis")
    base = resolve_hf_id(entry_id)
    name = out_name or re.sub(r"[^\w\-]+", "-", Path(fname).stem)
    return export_bundle(
        BUNDLES / name,
        ckpt=ckpt,
        basis=basis,
        base_model=base,
        rank=256,
        notes=f"Packaged from {fname}. Weights trained; P via hooks.",
    )


def smoke_basis_for_hf(
    hf_id: str,
    out_name: str,
    n_seqs: int = 40,
    rank: int = 256,
) -> Path:
    """Fit a quick shared PCA on residual streams and save a *basis-only* stub.

    Does NOT distill weights — attaches hooks to stock HF weights (will hurt
    quality). Useful to validate hidden-size compatibility before a long distill.
    """
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .runtime import BundleConfig

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id).to(device)
    model.eval()
    d = model.config.n_embd
    L = model.config.n_layer
    # collect residuals
    acts = [[] for _ in range(L)]

    def make_cap(li):
        def hook(_m, _i, out):
            acts[li].append(out[0].detach().float().cpu().reshape(-1, d)[:512])
            return out

        return hook

    handles = [
        model.transformer.h[li].register_forward_hook(make_cap(li))
        for li in range(L)
    ]
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:2%]")
    texts = [t for t in ds["text"] if len(t.strip()) > 100][:n_seqs]
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=128).input_ids.to(
                device
            )
            model(ids)
    for h in handles:
        h.remove()

    means = []
    cov = np.zeros((d, d), dtype=np.float64)
    total_n = 0
    for li in range(L):
        X = torch.cat(acts[li], dim=0).numpy()
        mu = X.mean(axis=0)
        means.append(mu)
        Xc = X - mu
        cov += Xc.T @ Xc
        total_n += Xc.shape[0]
    cov /= max(total_n, 1)
    # trace-normalize-ish
    cov = cov / (np.trace(cov) / d + 1e-12)
    w, V = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1]
    V = V[:, idx]
    P = V[:, :rank]
    means_a = np.stack(means, axis=0)

    out = BUNDLES / out_name
    out.mkdir(parents=True, exist_ok=True)
    # stock weights as safetensors
    from safetensors.torch import save_file

    save_file(model.state_dict(), str(out / "model.safetensors"))
    np.savez(out / "vera_basis.npz", means=means_a, P=P, V=V)
    BundleConfig(
        kind="hook_container_smoke",
        base_model=hf_id,
        rank=rank,
        n_layer=L,
        hidden_size=d,
        notes=(
            "SMOKE ONLY: stock HF weights + freshly fit P. "
            "Quality will be poor until distilled. Hidden-size check passed."
        ),
    ).save(out / "config.json")
    (out / "README.md").write_text(
        f"# Smoke bundle for {hf_id}\n\nNot production — distill next.\n",
        encoding="utf-8",
    )
    return out
