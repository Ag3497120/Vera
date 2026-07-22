"""Verify a Vera bundle + optional bench offer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .chat_app import load_active_vera


def quick_verify(spec: str, rank: int = 256) -> dict:
    """Cheap functional checks: generate, encode, memory self-retrieval."""
    vm = load_active_vera(spec, rank=rank)
    prompt = "The capital of France is"
    cont = vm.generate(prompt, max_new=16, do_sample=False)
    c1 = vm.encode_coords("The capital of Zentharia is Kaldrune.")
    c2 = vm.encode_coords("The capital of Zentharia is Kaldrune.")
    c3 = vm.encode_coords("Unrelated text about cooking pasta al dente.")
    def cos(a, b):
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)
        return float(a @ b)

    out = {
        "spec": spec,
        "rank": rank,
        "source": vm.source,
        "kind": vm.config.kind,
        "weights_note": vm.config.notes,
        "generation": {"prompt": prompt, "continuation": cont},
        "coord_self_sim": cos(c1, c2),
        "coord_diff_sim": cos(c1, c3),
        "self_gt_diff": cos(c1, c2) > cos(c1, c3),
    }
    vm.clear_hooks()
    return out


def print_verify(res: dict):
    print("=== Vera verify ===")
    for k in ("spec", "rank", "source", "kind"):
        print(f"  {k}: {res.get(k)}")
    print(f"  notes: {res.get('weights_note', '')[:120]}")
    g = res["generation"]
    print(f"  gen: {g['prompt']!r} → {g['continuation']!r}")
    print(f"  coord self-sim: {res['coord_self_sim']:.3f}")
    print(f"  coord diff-sim: {res['coord_diff_sim']:.3f}")
    print(f"  self > diff: {res['self_gt_diff']}")
    if res["self_gt_diff"]:
        print("  PASS smoke geometry")
    else:
        print("  WARN geometry weak (unexpected for distilled bundles)")
