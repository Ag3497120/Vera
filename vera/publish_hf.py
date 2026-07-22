"""Publish a local Vera bundle to the Hugging Face Hub."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from . import catalog


def publish_bundle(
    bundle_name: str = "gpt2-matryoshka",
    repo_id: Optional[str] = None,
    private: bool = False,
):
    from huggingface_hub import HfApi, login

    repo = repo_id or os.environ.get("VERA_HF_GPT2", "Ag3497120/vera-gpt2-matryoshka")
    path = catalog.BUNDLES / bundle_name
    if not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m vera convert --package-gpt2"
        )
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("No HF_TOKEN in env. Launching huggingface-cli login…")
        login()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(path),
        repo_id=repo,
        repo_type="model",
        commit_message="Publish Vera stereo-cross GPT-2 Matryoshka bundle",
    )
    print(f"Published → https://huggingface.co/{repo}")
    print("Consumers: python -m vera chat --model hf:" + repo)
