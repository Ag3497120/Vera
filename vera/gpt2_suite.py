"""Download / publish the two-repo GPT-2 Vera suite on Hugging Face."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

from . import catalog, convert
from .runtime import export_bundle

HF_GPT2 = os.environ.get("VERA_HF_GPT2", "Ag3497120/vera-gpt2-matryoshka")
HF_JOIN = os.environ.get("VERA_HF_JOIN", "Ag3497120/vera-distilgpt2-join")
HUB_CARDS = catalog.ROOT / "hub"


def _copy_card(card_dir_name: str, bundle_dir: Path):
    src = HUB_CARDS / card_dir_name / "README.md"
    if src.is_file():
        shutil.copy2(src, bundle_dir / "README.md")


def package_gpt2_suite() -> Tuple[Path, Path]:
    """Build both local bundles and stamp HF model-card READMEs."""
    a = convert.package_gpt2_matryoshka("gpt2-matryoshka")
    _copy_card("vera-gpt2-matryoshka", a)

    ckpt_b = catalog.find_file("student_b_distilgpt2.pt")
    basis_p = catalog.find_file("bases_cache_soft_distill.npz")
    means_b = catalog.find_file("means_b_distilgpt2.npz")
    if not ckpt_b or not basis_p or not means_b:
        raise FileNotFoundError(
            "Need student_b_distilgpt2.pt, bases_cache_soft_distill.npz, "
            "means_b_distilgpt2.npz (set VERA_ARTIFACTS)"
        )
    # Shared P from GPT-2 basis, means from DistilGPT2 fit
    import numpy as np

    out_b = catalog.BUNDLES / "distilgpt2-join"
    out_b.mkdir(parents=True, exist_ok=True)
    # reuse export for weights
    export_bundle(
        out_b,
        ckpt=ckpt_b,
        basis=basis_p,
        base_model="distilbert/distilgpt2",
        rank=256,
        kind="hook_container",
        notes=(
            "JOIN PARTNER: DistilGPT2 fine-tuned on the same frozen P as "
            "vera-gpt2-matryoshka. Means are DistilGPT2-specific; P is shared."
        ),
    )
    # overwrite basis with Distil means + shared P
    z = np.load(basis_p)
    mb = np.load(means_b)
    P = z["V"][:, :256] if "V" in z.files else z["P"][:, :256]
    np.savez(
        out_b / "vera_basis.npz",
        means=mb["means"],
        P=P,
        V=z["V"] if "V" in z.files else P,
        shared_with="vera-gpt2-matryoshka",
    )
    cfg_path = out_b / "config.json"
    import json

    cfg = json.loads(cfg_path.read_text())
    cfg["role"] = "join_partner"
    cfg["pair_repo"] = HF_GPT2
    cfg["notes"] = (
        "JOIN PARTNER for GPT-2 suite. Same P as Matryoshka GPT-2; "
        "DistilGPT2-specific means. Weights fine-tuned; hooks at runtime."
    )
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    _copy_card("vera-distilgpt2-join", out_b)
    return a, out_b


def download_gpt2_suite(
    cache_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Download both HF repos into bundles/ (or cache_dir)."""
    from huggingface_hub import snapshot_download

    root = Path(cache_dir) if cache_dir else catalog.BUNDLES
    root.mkdir(parents=True, exist_ok=True)
    print(f"[download] {HF_GPT2} …")
    p1 = Path(
        snapshot_download(
            repo_id=HF_GPT2,
            local_dir=str(root / "gpt2-matryoshka"),
            local_dir_use_symlinks=False,
        )
    )
    print(f"[download] {HF_JOIN} …")
    p2 = Path(
        snapshot_download(
            repo_id=HF_JOIN,
            local_dir=str(root / "distilgpt2-join"),
            local_dir_use_symlinks=False,
        )
    )
    print(f"[download] done\n  A: {p1}\n  B: {p2}")
    return p1, p2


def publish_gpt2_suite(private: bool = False):
    """Package (if needed) and upload both repos with model-card READMEs."""
    from huggingface_hub import HfApi, login

    a, b = package_gpt2_suite()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("No HF_TOKEN — launching login…")
        login()
    api = HfApi(token=token or True)
    for path, repo in ((a, HF_GPT2), (b, HF_JOIN)):
        print(f"[publish] {path.name} → {repo}")
        api.create_repo(repo_id=repo, private=private, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=str(path),
            repo_id=repo,
            repo_type="model",
            commit_message="Publish Vera GPT-2 stereo-cross suite bundle",
        )
        print(f"  https://huggingface.co/{repo}")
    print("Consumers: python -m vera download-gpt2-suite")
