"""Discover local models: Ollama, Vera bundles, research artifacts, HF cache."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "stereo_cross_activation"
BUNDLES = ROOT / "bundles"
HF_GPT2_DEFAULT = os.environ.get("VERA_HF_GPT2", "Ag3497120/vera-gpt2-matryoshka")


@dataclass
class ModelEntry:
    id: str
    backend: str  # ollama | vera_bundle | artifact | huggingface
    name: str
    detail: str
    path: Optional[str] = None
    convertible: bool = False
    runnable_vera: bool = False
    notes: str = ""

    def to_dict(self):
        return asdict(self)


def artifact_dirs() -> List[Path]:
    dirs = []
    if os.environ.get("VERA_ARTIFACTS"):
        dirs.append(Path(os.environ["VERA_ARTIFACTS"]).expanduser())
    dirs.append(EXP)
    sib = ROOT.parent / "verantyx-cli" / "experiments" / "stereo_cross_activation"
    if sib.is_dir():
        dirs.append(sib)
    hard = Path("/Users/motonishikoudai/Projects/verantyx-cli/experiments/stereo_cross_activation")
    if hard.is_dir():
        dirs.append(hard)
    out, seen = [], set()
    for d in dirs:
        r = d.resolve() if d.exists() else d
        if r not in seen:
            seen.add(r)
            out.append(d)
    return out


def find_file(*names: str) -> Optional[Path]:
    for d in artifact_dirs():
        for n in names:
            p = d / n
            if p.is_file():
                return p
    return None


def list_ollama() -> List[ModelEntry]:
    if not shutil.which("ollama"):
        return []
    try:
        r = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=30
        )
    except Exception:
        return []
    if r.returncode != 0:
        return []
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return []
    entries = []
    for ln in lines[1:]:
        parts = ln.split()
        if not parts:
            continue
        name = parts[0]
        size = parts[2] if len(parts) > 2 else "?"
        # Ollama serves GGUF — cannot attach PyTorch stereo hooks in-process.
        # Convert = pull matching HF weights if user confirms; Use = API mouth.
        entries.append(
            ModelEntry(
                id=f"ollama:{name}",
                backend="ollama",
                name=name,
                detail=f"size~{size}",
                convertible=True,
                runnable_vera=False,
                notes=(
                    "Ollama runtime is GGUF — Vera hooks need HF/transformers "
                    "weights. 'Convert' maps to an HF id + builds a bundle; "
                    "'Use' can call Ollama as the speaking backend with Vera memory."
                ),
            )
        )
    return entries


def list_bundles() -> List[ModelEntry]:
    entries = []
    if not BUNDLES.is_dir():
        return entries
    for d in sorted(BUNDLES.iterdir()):
        if (d / "config.json").is_file():
            try:
                cfg = json.loads((d / "config.json").read_text())
                detail = f"{cfg.get('base_model')} r={cfg.get('rank')}"
            except Exception:
                detail = "bundle"
            entries.append(
                ModelEntry(
                    id=f"bundle:{d.name}",
                    backend="vera_bundle",
                    name=d.name,
                    detail=detail,
                    path=str(d),
                    convertible=False,
                    runnable_vera=True,
                    notes="Portable Vera bundle (trained weights + frozen P).",
                )
            )
    return entries


def list_artifacts() -> List[ModelEntry]:
    entries = []
    mapping = [
        ("matryoshka_student.pt", "openai-community/gpt2", "Matryoshka GPT-2 student"),
        ("kl_distill_student_r256.pt", "openai-community/gpt2", "KL r256 GPT-2 student"),
        ("student_b_distilgpt2.pt", "distilbert/distilgpt2", "Join partner DistilGPT2"),
        ("weight_compress_healed.pt", "openai-community/gpt2", "Folded compress (special)"),
    ]
    for fname, base, label in mapping:
        p = find_file(fname)
        if p:
            entries.append(
                ModelEntry(
                    id=f"artifact:{fname}",
                    backend="artifact",
                    name=label,
                    detail=base,
                    path=str(p),
                    convertible=True,
                    runnable_vera=True,
                    notes="Research checkpoint — weights fine-tuned; P via hooks.",
                )
            )
    return entries


def list_huggingface_presets() -> List[ModelEntry]:
    return [
        ModelEntry(
            id=f"hf:{HF_GPT2_DEFAULT}",
            backend="huggingface",
            name="Vera GPT-2 Matryoshka (Hub)",
            detail=HF_GPT2_DEFAULT,
            convertible=False,
            runnable_vera=True,
            notes="Downloads published bundle when selected. Publish via: python -m vera publish",
        ),
        ModelEntry(
            id="hf:openai-community/gpt2",
            backend="huggingface",
            name="Stock GPT-2 (no Vera)",
            detail="openai-community/gpt2",
            convertible=True,
            runnable_vera=False,
            notes="Vanilla weights — convert to create a Vera student (needs distill compute).",
        ),
    ]


def list_all() -> List[ModelEntry]:
    return list_bundles() + list_artifacts() + list_ollama() + list_huggingface_presets()


def print_catalog(entries: Optional[List[ModelEntry]] = None):
    entries = entries or list_all()
    print(f"{'#':>3}  {'backend':12}  {'name':40}  vera  convert")
    print("-" * 90)
    for i, e in enumerate(entries, 1):
        print(
            f"{i:>3}  {e.backend:12}  {e.name[:40]:40}  "
            f"{'Y' if e.runnable_vera else '-':4}  {'Y' if e.convertible else '-'}"
        )
        if e.notes:
            print(f"       · {e.notes[:88]}")
