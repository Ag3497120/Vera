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
║  GPT-2 suite = 2 Hub downloads (Matryoshka + Join)      ║
╚══════════════════════════════════════════════════════════╝
  1) List models (Ollama / bundles / artifacts / HF)
  2) Convert / package / download GPT-2 suite
  3) Chat (Vera brain + memory; optional Ollama mouth)
  4) Verify (smoke) — optionally offer benches
  5) Publish to Hugging Face (gpt2-suite = 2 repos)
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

2) GEOMETRY IS RUNTIME (hooks)
   Shared basis P and per-layer means are FROZEN; applied every forward.

3) JOIN / MEMORY REINJECT — runtime coord capture/inject only.

GPT-2 Hub suite (2 repos):
  Ag3497120/vera-gpt2-matryoshka      — primary container
  Ag3497120/vera-distilgpt2-join      — join partner (same P)
"""
    )


def _gpt2_suite_actions(default: str = "package-local"):
    from . import gpt2_suite

    mode = _ask(
        "GPT-2 suite (2 repos). download-from-hub / package-local / publish-to-hub",
        default,
    )
    if mode.startswith("download"):
        gpt2_suite.download_gpt2_suite()
    elif mode.startswith("publish"):
        gpt2_suite.publish_gpt2_suite(
            private=_ask("Private? y/N", "N").lower() == "y"
        )
    else:
        a, b = gpt2_suite.package_gpt2_suite()
        print(f"Packaged with HF model-card READMEs:\n  {a}\n  {b}")


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

    if (
        e.id == "hf-suite:gpt2"
        or "Vera suite" in e.name
        or e.detail == "openai-community/gpt2"
        or e.id.endswith("openai-community/gpt2")
        or "vera-gpt2-matryoshka" in (e.detail or "")
        or "vera-distilgpt2-join" in (e.detail or "")
    ):
        _gpt2_suite_actions(
            "download-from-hub"
            if (
                e.backend == "huggingface"
                and (
                    e.id == "hf-suite:gpt2"
                    or "suite" in e.id
                    or "vera-gpt2" in (e.detail or "")
                    or "vera-distilgpt2" in (e.detail or "")
                )
            )
            else "package-local"
        )
        return

    if e.backend == "artifact":
        path = convert.package_artifact(e.id)
        print(f"Packaged → {path}")
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
                if _ask("WARNING: 9B+ needs GPU. Continue? y/N", "N").lower() != "y":
                    return
            name = _ask("Bundle name", e.name.replace(":", "-").replace("/", "-"))
            try:
                print(f"Smoke bundle → {convert.smoke_basis_for_hf(hf, name)}")
            except Exception as ex:
                print(f"Failed: {ex}")
        else:
            print("Use menu 3 chat with mouth=ollama:" + e.name)
        return

    hf = e.detail if e.backend == "huggingface" else _ask("HF id")
    name = _ask("Bundle name", str(hf).replace("/", "-"))
    print(f"Smoke bundle → {convert.smoke_basis_for_hf(hf, name)}")


def do_chat():
    entries = [e for e in catalog.list_all() if e.runnable_vera]
    print("\nRunnable Vera models:")
    e = _pick(entries)
    if not e:
        if catalog.find_file("matryoshka_student.pt", "kl_distill_student_r256.pt"):
            if _ask("Package GPT-2 suite now? y/N", "y").lower() == "y":
                from . import gpt2_suite

                gpt2_suite.package_gpt2_suite()
                e = next(
                    (x for x in catalog.list_bundles() if "gpt2-matryoshka" in x.name),
                    None,
                )
            else:
                return
        else:
            print("No runnable model. download-gpt2-suite or set VERA_ARTIFACTS.")
            return
    if not e:
        return
    if e.id == "hf-suite:gpt2":
        from . import gpt2_suite

        gpt2_suite.download_gpt2_suite()
        spec = "bundle:gpt2-matryoshka"
    else:
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
    spec = "bundle:gpt2-matryoshka" if e.id == "hf-suite:gpt2" else e.id
    if e.id == "hf-suite:gpt2":
        from . import gpt2_suite

        gpt2_suite.download_gpt2_suite()
    rank = int(_ask("Rank", "256"))
    res = quick_verify(spec, rank=rank)
    print_verify(res)
    out = catalog.ROOT / "bundles" / "_last_verify.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__import__("json").dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    if _ask("Run heavier benches? y/N", "N").lower() == "y":
        print(
            "  python experiments/stereo_cross_activation/matryoshka_distill.py\n"
            "  python experiments/stereo_cross_activation/cross_model_join.py"
        )


def do_publish():
    mode = _ask(
        "Publish: gpt2-suite (2 repos + READMEs) / single-bundle",
        "gpt2-suite",
    )
    if mode.startswith("gpt2"):
        from . import gpt2_suite

        gpt2_suite.publish_gpt2_suite(
            private=_ask("Private? y/N", "N").lower() == "y"
        )
        return
    bundles = catalog.list_bundles()
    if not bundles:
        if _ask("Package GPT-2 suite first? y/N", "y").lower() == "y":
            from . import gpt2_suite

            gpt2_suite.package_gpt2_suite()
            bundles = catalog.list_bundles()
    e = _pick(bundles)
    if not e:
        return
    default_repo = (
        catalog.HF_JOIN_DEFAULT
        if "distil" in e.name.lower()
        else catalog.HF_GPT2_DEFAULT
    )
    repo = _ask("HF repo id", default_repo)
    private = _ask("Private? y/N", "N").lower() == "y"
    try:
        from .gpt2_suite import _copy_card

        if "distil" in e.name.lower():
            _copy_card("vera-distilgpt2-join", catalog.BUNDLES / e.name)
        elif "gpt2" in e.name.lower():
            _copy_card("vera-gpt2-matryoshka", catalog.BUNDLES / e.name)
        publish_bundle(e.name, repo_id=repo, private=private)
    except Exception as ex:
        print(f"Publish failed: {ex}")
        print("Set HF_TOKEN or huggingface-cli login, then retry.")


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
