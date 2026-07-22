"""Useful chat: Vera container + coordinate memory + optional Ollama mouth."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import catalog
from .runtime import VeraModel, load_from_bundle, load_from_artifacts, load_from_hf

MEM_DIR = catalog.ROOT / "bundles" / "_chat_memory"


class CoordMemory:
    def __init__(self, path: Path = MEM_DIR):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.path / "memories.jsonl"
        self.records: List[dict] = []
        if self.jsonl.is_file():
            for ln in self.jsonl.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    self.records.append(json.loads(ln))

    def add(self, text: str, coords: np.ndarray, meta: Optional[dict] = None):
        rec = {
            "id": len(self.records),
            "text": text,
            "coords": coords.astype(float).tolist(),
            "meta": meta or {},
        }
        self.records.append(rec)
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def search(self, query_coords: np.ndarray, k: int = 3) -> List[Tuple[float, dict]]:
        if not self.records:
            return []
        q = query_coords.astype(np.float64)
        q /= np.linalg.norm(q) + 1e-8
        scored = []
        for rec in self.records:
            c = np.asarray(rec["coords"], dtype=np.float64)
            c = c / (np.linalg.norm(c) + 1e-8)
            scored.append((float(q @ c), rec))
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[:k]


def load_active_vera(spec: str, rank: int = 256) -> VeraModel:
    if spec.startswith("bundle:"):
        name = spec.split(":", 1)[1]
        return load_from_bundle(catalog.BUNDLES / name, rank=rank)
    if spec.startswith("artifact:"):
        fname = spec.split(":", 1)[1]
        ckpt = catalog.find_file(fname)
        basis = catalog.find_file("bases_cache_soft_distill.npz")
        base = "distilbert/distilgpt2" if "distilgpt2" in fname else "openai-community/gpt2"
        return load_from_artifacts(ckpt, basis, base_model=base, rank=rank)
    if spec.startswith("hf:"):
        return load_from_hf(spec[3:], rank=rank)
    # bare bundle name
    p = catalog.BUNDLES / spec
    if p.is_dir():
        return load_from_bundle(p, rank=rank)
    raise ValueError(f"Cannot load Vera model spec {spec}")


def ollama_generate(model: str, prompt: str) -> str:
    import json as _json
    import urllib.request

    body = _json.dumps(
        {"model": model, "prompt": prompt, "stream": False}
    ).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = _json.loads(resp.read().decode())
    return data.get("response", "")


def run_chat(
    vera_spec: str,
    rank: int = 256,
    mouth: str = "vera",  # vera | ollama:<name>
    system: str = "",
):
    """Interactive chat loop leveraging container memory + generation."""
    print(f"[chat] loading {vera_spec} rank={rank} mouth={mouth}")
    vm = load_active_vera(vera_spec, rank=rank)
    mem = CoordMemory()
    print(
        "Commands: /rank N | /mem <text> | /search <q> | /mouth vera|ollama:name | /quit"
    )
    history = []
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user.startswith("/rank "):
            vm.set_rank(int(user.split()[1]))
            print(f"[rank] {vm.state.rank}")
            continue
        if user.startswith("/mouth "):
            mouth = user.split(maxsplit=1)[1].strip()
            print(f"[mouth] {mouth}")
            continue
        if user.startswith("/mem "):
            text = user[5:].strip()
            c = vm.encode_coords(text)
            mem.add(text, c, {"source": "user"})
            print(f"[mem] stored ({len(mem.records)} total)")
            continue
        if user.startswith("/search "):
            q = user[8:].strip()
            c = vm.encode_coords(q)
            hits = mem.search(c, k=5)
            for s, rec in hits:
                print(f"  {s:+.3f}  {rec['text'][:100]}")
            continue

        # retrieve memories into prompt (actually useful)
        qc = vm.encode_coords(user)
        hits = mem.search(qc, k=3)
        mem_block = ""
        if hits and hits[0][0] > 0.15:
            mem_block = "Known facts:\n" + "\n".join(
                f"- {rec['text']}" for _, rec in hits if _ > 0.15
            )
            print("[retrieved]")
            for s, rec in hits:
                if s > 0.15:
                    print(f"  {s:+.3f} {rec['text'][:90]}")

        prompt_parts = []
        if system:
            prompt_parts.append(system)
        if mem_block:
            prompt_parts.append(mem_block)
        if history:
            prompt_parts.append("Conversation:\n" + "\n".join(history[-6:]))
        prompt_parts.append(f"User: {user}\nAssistant:")
        prompt = "\n\n".join(prompt_parts)

        if mouth.startswith("ollama:"):
            name = mouth.split(":", 1)[1]
            # Ollama speaks; Vera only supplies retrieved memory in text form
            reply = ollama_generate(name, prompt)
        else:
            reply = vm.chat_turn(prompt, max_new=80).strip()

        print(f"bot> {reply}")
        history.append(f"User: {user}")
        history.append(f"Assistant: {reply}")
        # store turn in coord memory
        mem.add(f"Q: {user} | A: {reply[:200]}", vm.encode_coords(user + " " + reply))

    vm.clear_hooks()
