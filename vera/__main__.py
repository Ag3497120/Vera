"""python -m vera — CLI entry."""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog="vera",
        description=(
            "Vera stereo-cross toolkit: interactive CUI, convert/package, "
            "chat with memory, verify, publish to Hugging Face."
        ),
    )
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("ui", help="interactive CUI (default)")
    sub.add_parser("cui", help="alias of ui")

    p = sub.add_parser("models", help="list Ollama / bundles / artifacts / HF")
    p.set_defaults(cmd="models")

    p = sub.add_parser("convert", help="package GPT-2 Matryoshka or smoke-convert")
    p.add_argument("--package-gpt2", action="store_true")
    p.add_argument("--package-gpt2-suite", action="store_true",
                   help="package Matryoshka GPT-2 + DistilGPT2 join (2 bundles)")
    p.add_argument("--smoke-hf", type=str, default="", help="HF id for smoke basis")
    p.add_argument("--name", type=str, default="")

    sub.add_parser(
        "download-gpt2-suite",
        help="download BOTH Hub repos (Matryoshka GPT-2 + DistilGPT2 join)",
    )

    p = sub.add_parser("chat", help="useful chat with coord memory")
    p.add_argument("--model", type=str, default="bundle:gpt2-matryoshka")
    p.add_argument("--rank", type=int, default=256)
    p.add_argument("--mouth", type=str, default="vera")

    p = sub.add_parser("verify", help="smoke verify a model")
    p.add_argument("--model", type=str, default="bundle:gpt2-matryoshka")
    p.add_argument("--rank", type=int, default=256)

    p = sub.add_parser("publish", help="upload bundle(s) to Hugging Face")
    p.add_argument("--bundle", type=str, default="gpt2-matryoshka")
    p.add_argument("--gpt2-suite", action="store_true",
                   help="publish both GPT-2 suite repos with model cards")
    p.add_argument("--repo", type=str, default="")
    p.add_argument("--private", action="store_true")

    p = sub.add_parser("explain", help="weights vs hooks")

    # default to UI when no args
    if not argv:
        argv = ["ui"]

    args = ap.parse_args(argv)

    if args.cmd in (None, "ui", "cui"):
        from .cui import main_loop

        main_loop()
        return

    if args.cmd == "models":
        from .catalog import print_catalog

        print_catalog()
        return

    if args.cmd == "explain":
        from .cui import explain_weights_vs_hooks

        explain_weights_vs_hooks()
        return

    if args.cmd == "convert":
        from . import convert, gpt2_suite

        if args.package_gpt2_suite:
            a, b = gpt2_suite.package_gpt2_suite()
            print(f"Packaged suite:\n  {a}\n  {b}")
        elif args.package_gpt2 or not args.smoke_hf:
            path = convert.package_gpt2_matryoshka(args.name or "gpt2-matryoshka")
            print(f"Packaged → {path}")
        else:
            name = args.name or args.smoke_hf.replace("/", "-")
            path = convert.smoke_basis_for_hf(args.smoke_hf, name)
            print(f"Smoke → {path}")
        return

    if args.cmd == "download-gpt2-suite":
        from . import gpt2_suite

        gpt2_suite.download_gpt2_suite()
        return

    if args.cmd == "chat":
        from .chat_app import run_chat

        run_chat(args.model, rank=args.rank, mouth=args.mouth)
        return

    if args.cmd == "verify":
        from .verify import print_verify, quick_verify

        print_verify(quick_verify(args.model, rank=args.rank))
        return

    if args.cmd == "publish":
        from .publish_hf import publish_bundle
        from .catalog import HF_GPT2_DEFAULT
        from . import gpt2_suite

        if args.gpt2_suite:
            gpt2_suite.publish_gpt2_suite(private=args.private)
        else:
            publish_bundle(
                args.bundle,
                repo_id=args.repo or HF_GPT2_DEFAULT,
                private=args.private,
            )
        return


if __name__ == "__main__":
    main()
