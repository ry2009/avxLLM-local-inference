#!/usr/bin/env python3
"""Helper to download base models and LoRA adapters into the repo scratch dirs.

Examples
--------
python tools/download_assets.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --model-dir models/tinyllama-chat \
  --adapter-id theone049/agriqa-tinyllama-lora-adapter --adapter-dir adapters/agriqa
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover
    raise SystemExit("huggingface_hub is required.\nRun `pip install huggingface-hub`.") from exc


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download(repo_id: str, target: Path, revision: str | None, token: str | None) -> str:
    _ensure_dir(target)
    resolved = snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        revision=revision,
        token=token,
        resume_download=True,
    )
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download HF models/adapters into local scratch dirs")
    parser.add_argument("--model-id", help="Base model repo id to mirror", default=None)
    parser.add_argument("--model-dir", help="Destination folder under models/", default="models/base")
    parser.add_argument("--adapter-id", help="Adapter repo id to mirror", default=None)
    parser.add_argument("--adapter-dir", help="Destination folder under adapters/", default="adapters/sample")
    parser.add_argument("--revision", help="Optional git revision or tag", default=None)
    parser.add_argument(
        "--token", help="Explicit HF token (defaults to HF_TOKEN/HUGGINGFACE_HUB_TOKEN)", default=None
    )
    args = parser.parse_args(argv)

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    downloaded = []

    if args.model_id:
        target = Path(args.model_dir)
        resolved = download(args.model_id, target, args.revision, token)
        downloaded.append(("model", args.model_id, resolved))

    if args.adapter_id:
        target = Path(args.adapter_dir)
        resolved = download(args.adapter_id, target, args.revision, token)
        downloaded.append(("adapter", args.adapter_id, resolved))

    if not downloaded:
        parser.error("Provide at least --model-id or --adapter-id")

    for kind, repo_id, resolved in downloaded:
        print(f"Downloaded {kind} {repo_id} → {resolved}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
