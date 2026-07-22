"""Interactive CUI for Vera — list, convert, chat, verify, publish."""
from __future__ import annotations

import sys
from typing import List, Optional

from . import catalog, convert
from .chat_app import run_chat
from .publish_hf import publish_bundle
from .verify import print_verify, quick_verify


def _ask(prompt: str, default: Optional[str] = None) -> str:
    suf = f" [{default}]" if default is not None else ""
    s = input(f"{prompt}{suf}: ").strip()
    return s if s else (default or "")


def _pick(entries: List[catalog.ModelEntry]) -> Optional[catalog.ModelEntry]:
    catalog.print_catalog(entries)
    if not entries:
        return None
    s = _ask("Select # (or empty cancel)")
    if not s:
        return None
    try:
        i = int(s)
    except ValueError:
        return None
    if not 1 <= i <= len(entries):
        print("Invalid")
        return None
    return entries[i - 1]


def menu():
    print(
        """
╔══════════════════════════════════════════════════════════╗
║  VERA — stereo-cross workspace (interactive)            ║
║  Weights are trained; P/means applied by runtime hooks  ║
╚══════════════════════════════════════════════════════════╝
  1) List models (Ollama / bundles / artifacts / HF)
  2) Convert / package into Vera bundle
  3) Chat (Vera brain + memory; optional Ollama mouth)
  4) Verify (smoke) — optionally offer benches
  5) Publish bundle to Hugging Face
  6) Explain: weights vs hooks
  0) Quit
"""
    )


def explain_weights_vs_hooks():
    print(
        """
WHAT CHANGED IN THE RESEARCH ARTIFACTS
--------------------------------------
1) WEIGHTS CHANGED (trained)
   matryoshka_student.pt / kl_distill_*.pt / student_b_*.pt
   / weight_compress_healed.pt are fine-tuned or re-parameterized.
   They are NOT stock GPT-2.

2) GEOMETRY IS RUNTIME (hooks)
   Shared basis P and per-layer means are FROZEN. Each block output is
   projected every forward pass (or folded into matrices in Step-4 compress).

3) JOIN / MEMORY REINJECT
   Purely programmatic coord capture/inject — no extra training required
   at use time.

So: conversion scripts package (1)+(2). Distilling a NEW model updates
weights under the hook constraint; packaging alone only wraps an existing
student.
"""
    )


def do_convert():
    entries = [
        e
        for e in catalog.list_all()
        if e.convertible or e.backend in ("artifact", "ollama", "huggingface")
    ]
    print("\nConvertible / packageable sources:")
    e = _pick(entries)
    if not e:
        return
    print(f"\nSelected: {e.id}\n{e.notes}")
    if e.backend == "artifact":
        path = convert.package_artifact(e.id)
        print(f"Packaged → {path}")
        return
    if "Vera GPT-2" in e.name or e.detail.endswith("vera-gpt2-matryoshka"):
        path = convert.package_gpt2_matryoshka()
        print(f"Packaged local proven student → {path}")
        print("Publish (menu 5) so others can hf-download it.")
        return
    if e.detail == "openai-community/gpt2" or e.id.endswith("openai-community/gpt2"):
        mode = _ask(
            "Mode: package-existing-student / smoke-stock-gpt2",
            "package-existing-student",
        )
        if mode.startswith("package"):
            path = convert.package_gpt2_matryoshka()
            print(f"Packaged proven Matryoshka student → {path}")
        else:
            name = _ask("Bundle name", "gpt2-stock-smoke")
            path = convert.smoke_basis_for_hf("openai-community/gpt2", name)
            print(f"Smoke bundle → {path} (quality poor until distilled)")
        return
    if e.backend == "ollama":
        try:
            hf = convert.resolve_hf_id(e.id)
        except ValueError:
            hf = _ask("HF model id for this Ollama tag")
        print(
            f"Ollama '{e.name}' → HF '{hf}'\n"
            "GGUF cannot take Vera hooks. We can:\n"
            "  a) smoke-fit basis on HF weights (fast, low quality)\n"
            "  b) use Ollama only as chat mouth with Vera memory\n"
        )
        choice = _ask("Choice a/b", "b")
        if choice.lower().startswith("a"):
            if "9b" in e.name.lower() or "27" in e.name.lower():
                ok = _ask(
                    "WARNING: 9B+ needs serious GPU. Continue? y/N", "N"
                )
                if ok.lower() != "y":
                    return
            name = _ask("Bundle name", e.name.replace(":", "-").replace("/", "-"))
            try:
                path = convert.smoke_basis_for_hf(hf, name)
                print(f"Smoke bundle → {path}")
            except Exception as ex:
                print(f"Failed: {ex}")
        else:
            print("Use menu 3 chat with mouth=ollama:" + e.name)
        return
    hf = e.detail if e.backend == "huggingface" else _ask("HF id")
    name = _ask("Bundle name", hf.replace("/", "-"))
    path = convert.smoke_basis_for_hf(hf, name)
    print(f"Smoke bundle → {path}")


def do_chat():
    entries = [e for e in catalog.list_all() if e.runnable_vera]
    print("\nRunnable Vera models:")
    e = _pick(entries)
    if not e:
        if catalog.find_file("matryoshka_student.pt", "kl_distill_student_r256.pt"):
            if _ask("No bundle yet. Package GPT-2 Matryoshka now? y/N", "y").lower() == "y":
                convert.package_gpt2_matryoshka()
                bundles = catalog.list_bundles()
                e = bundles[0] if bundles else None
            else:
                return
        else:
            print("No runnable Vera model. Convert/package first, or set VERA_ARTIFACTS.")
            return
    if not e:
        return
    spec = e.id
    rank = int(_ask("Matryoshka rank", "256"))
    mouth = _ask("Mouth: vera | ollama:<name>", "vera")
    run_chat(spec, rank=rank, mouth=mouth)


def do_verify():
    entries = [e for e in catalog.list_all() if e.runnable_vera]
    seen, uniq = set(), []
    for e in entries:
        if e.id not in seen:
            seen.add(e.id)
            uniq.append(e)
    e = _pick(uniq)
    if not e:
        return
    rank = int(_ask("Rank", "256"))
    res = quick_verify(e.id, rank=rank)
    print_verify(res)
    out = catalog.ROOT / "bundles" / "_last_verify.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__import__("json").dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    ans = _ask("Run heavier benches (wikitext / join battery)? y/N", "N")
    if ans.lower() == "y":
        print(
            "Heavy benches:\n"
            "  python experiments/stereo_cross_activation/matryoshka_distill.py\n"
            "  python experiments/stereo_cross_activation/cross_model_join.py\n"
            "Feel-tests: python -m demo tour"
        )


def do_publish():
    bundles = catalog.list_bundles()
    if not bundles:
        if _ask("No bundles. Package GPT-2 Matryoshka first? y/N", "y").lower() == "y":
            convert.package_gpt2_matryoshka()
            bundles = catalog.list_bundles()
    e = _pick(bundles)
    if not e:
        return
    repo = _ask("HF repo id", catalog.HF_GPT2_DEFAULT)
    private = _ask("Private? y/N", "N").lower() == "y"
    try:
        publish_bundle(e.name, repo_id=repo, private=private)
    except Exception as ex:
        print(f"Publish failed: {ex}")
        print("Set HF_TOKEN or run huggingface-cli login, then retry.")


def main_loop():
    while True:
        menu()
        choice = _ask("Command", "1")
        if choice in ("0", "q", "quit"):
            break
        try:
            if choice == "1":
                catalog.print_catalog()
            elif choice == "2":
                do_convert()
            elif choice == "3":
                do_chat()
            elif choice == "4":
                do_verify()
            elif choice == "5":
                do_publish()
            elif choice == "6":
                explain_weights_vs_hooks()
            else:
                print("Unknown")
        except Exception as ex:
            print(f"[error] {ex}", file=sys.stderr)
