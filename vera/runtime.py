"""Vera runtime: weights vs hooks, and loadable container models.

What changed in the research artifacts
--------------------------------------
1) **Weights changed (trained)**  
   `matryoshka_student.pt`, `kl_distill_student_*.pt`, `student_b_distilgpt2.pt`,
   `weight_compress_healed.pt` are **fine-tuned / re-parameterized checkpoints**.
   They are not stock GPT-2 weights.

2) **Geometry applied at runtime (hooks)**  
   The shared basis `P` and per-layer means are **frozen**. Every block output is
   projected with a forward hook:
       h <- mean + (h - mean) @ P[:, :r] @ P[:, :r].T
   So the stereo-cross *constraint* is programmatic unless you export a
   folded architecture (Step 4 weight_compress).

3) **Join / memory reinject**  
   Purely runtime (capture / inject coords). No extra weight change.

Bundle layout (local or Hugging Face)
-------------------------------------
  vera_bundle/
    config.json          # base_model, rank, n_layer, kind
    model.safetensors    # or pytorch_model.bin — student weights
    vera_basis.npz      # means [L,D], V or P [D,R]
    README.md
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_HF_GPT2 = os.environ.get("VERA_HF_GPT2", "Ag3497120/vera-gpt2-matryoshka")
DEFAULT_RANK = 256


@dataclass
class BundleConfig:
    kind: str  # "hook_container" | "folded_compress"
    base_model: str
    rank: int
    n_layer: int
    hidden_size: int
    vera_version: str = "0.1"
    notes: str = ""

    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "BundleConfig":
        return BundleConfig(**json.loads(path.read_text(encoding="utf-8")))


class RankState:
    __slots__ = ("rank", "ablate", "inject_layer", "inject_coords", "mode")

    def __init__(self, rank: int = DEFAULT_RANK):
        self.rank = rank
        self.ablate = None
        self.inject_layer = -1
        self.inject_coords = None
        self.mode = "project"  # project | capture | inject


def _make_hook(li: int, mean, P_full, state: RankState):
    def hook(_mod, _inp, out):
        r = min(int(state.rank), P_full.shape[1])
        P = P_full[:, :r]
        h = out[0]
        hc = h - mean
        c = hc @ P
        if state.ablate:
            c = c.clone()
            c[..., list(state.ablate)] = 0
        if state.mode == "capture" and li == state.inject_layer:
            state.inject_coords = c.detach()
        if (
            state.mode == "inject"
            and li == state.inject_layer
            and state.inject_coords is not None
        ):
            donor = state.inject_coords
            t = min(donor.shape[1], c.shape[1])
            c = c.clone()
            c[:, :t] = donor[:, :t].to(c.dtype)
        h_new = mean + c @ P.T
        return (h_new,) + tuple(out[1:])

    return hook


@dataclass
class VeraModel:
    """HF CausalLM + stereo-cross hooks + mutable rank."""

    model: Any
    tokenizer: Any
    means: Any  # torch.Tensor [L,D]
    P: Any  # torch.Tensor [D,R]
    state: RankState
    hooks: List[Any]
    config: BundleConfig
    device: Any
    source: str

    def set_rank(self, rank: int):
        self.state.rank = int(rank)

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def encode_coords(self, text: str, layer: int = -1):
        import torch

        li = layer if layer >= 0 else self.config.n_layer - 1
        self.state.mode = "capture"
        self.state.inject_layer = li
        self.state.inject_coords = None
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            self.model(ids)
        c = self.state.inject_coords[0, -1].detach().float().cpu().numpy()
        self.state.mode = "project"
        return c

    def generate(self, prompt: str, max_new: int = 64, **kw) -> str:
        import torch

        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=max_new,
                do_sample=kw.get("do_sample", True),
                temperature=kw.get("temperature", 0.8),
                top_p=kw.get("top_p", 0.95),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True)

    def chat_turn(self, prompt: str, max_new: int = 80) -> str:
        return self.generate(prompt, max_new=max_new)


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_basis_npz(path: Path, rank: int):
    import torch

    z = np.load(path)
    if "P" in z.files:
        P_np = np.ascontiguousarray(z["P"][:, :rank])
    else:
        P_np = np.ascontiguousarray(z["V"][:, :rank])
    means = torch.from_numpy(z["means"]).float()
    P = torch.from_numpy(P_np).float()
    return means, P


def attach_hooks(model, means, P, state: RankState):
    import torch

    means = means.to(next(model.parameters()).device)
    P = P.to(next(model.parameters()).device)
    hooks = []
    n = model.config.n_layer
    for li in range(n):
        hooks.append(
            model.transformer.h[li].register_forward_hook(
                _make_hook(li, means[li], P, state)
            )
        )
    return hooks


def load_from_bundle(bundle_dir: Path, rank: Optional[int] = None) -> VeraModel:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bundle_dir = Path(bundle_dir)
    cfg = BundleConfig.load(bundle_dir / "config.json")
    r = int(rank or cfg.rank)
    device = _device()
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model).to(device)
    weights = bundle_dir / "model.safetensors"
    bin_path = bundle_dir / "pytorch_model.bin"
    if weights.is_file():
        from safetensors.torch import load_file

        sd = load_file(str(weights), device=str(device))
        model.load_state_dict(sd, strict=False)
    elif bin_path.is_file():
        sd = torch.load(bin_path, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)
    else:
        raise FileNotFoundError(f"No weights in {bundle_dir}")
    means, P = load_basis_npz(bundle_dir / "vera_basis.npz", r)
    means, P = means.to(device), P.to(device)
    state = RankState(r)
    hooks = attach_hooks(model, means, P, state)
    model.eval()
    return VeraModel(
        model=model,
        tokenizer=tok,
        means=means,
        P=P,
        state=state,
        hooks=hooks,
        config=cfg,
        device=device,
        source=str(bundle_dir),
    )


def load_from_artifacts(
    ckpt: Path,
    basis: Path,
    base_model: str = "openai-community/gpt2",
    rank: int = 256,
) -> VeraModel:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _device()
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=False)
    means, P = load_basis_npz(basis, rank)
    means, P = means.to(device), P.to(device)
    state = RankState(rank)
    hooks = attach_hooks(model, means, P, state)
    model.eval()
    cfg = BundleConfig(
        kind="hook_container",
        base_model=base_model,
        rank=rank,
        n_layer=model.config.n_layer,
        hidden_size=model.config.n_embd,
        notes="loaded from research artifacts",
    )
    return VeraModel(
        model, tok, means, P, state, hooks, cfg, device, f"{ckpt}"
    )


def load_from_hf(repo_id: Optional[str] = None, rank: Optional[int] = None) -> VeraModel:
    """Download a published Vera bundle from the Hub into cache, then load."""
    from huggingface_hub import hf_hub_download, snapshot_download

    repo = repo_id or DEFAULT_HF_GPT2
    try:
        root = Path(
            snapshot_download(
                repo_id=repo,
                allow_patterns=[
                    "config.json",
                    "model.safetensors",
                    "pytorch_model.bin",
                    "vera_basis.npz",
                    "*.json",
                    "README.md",
                ],
            )
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not download {repo}. Publish the bundle first "
            f"(python -m vera publish) or set VERA_HF_GPT2.\n{e}"
        ) from e
    return load_from_bundle(root, rank=rank)


def export_bundle(
    out_dir: Path,
    ckpt: Path,
    basis: Path,
    base_model: str,
    rank: int = 256,
    kind: str = "hook_container",
    notes: str = "",
) -> Path:
    """Package trained weights + basis into a portable Vera bundle."""
    import torch
    from transformers import AutoModelForCausalLM

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    # GPT-2 ties wte/lm_head — safetensors rejects shared storage unless handled.
    try:
        from safetensors.torch import save_model as st_save_model

        st_save_model(model, str(out_dir / "model.safetensors"))
    except Exception:
        torch.save(model.state_dict(), out_dir / "pytorch_model.bin")
    z = np.load(basis)
    if "P" in z.files:
        P = z["P"][:, :rank]
    else:
        P = z["V"][:, :rank]
    np.savez(
        out_dir / "vera_basis.npz",
        means=z["means"],
        P=P,
        V=z["V"] if "V" in z.files else P,
    )
    cfg = BundleConfig(
        kind=kind,
        base_model=base_model,
        rank=rank,
        n_layer=model.config.n_layer,
        hidden_size=model.config.n_embd,
        notes=notes
        or (
            "Weights: fine-tuned under stereo-cross constraint. "
            "Geometry: runtime hooks with frozen P/means."
        ),
    )
    cfg.save(out_dir / "config.json")
    (out_dir / "README.md").write_text(
        f"# Vera bundle\n\nbase={base_model} rank={rank} kind={kind}\n\n{cfg.notes}\n",
        encoding="utf-8",
    )
    return out_dir
