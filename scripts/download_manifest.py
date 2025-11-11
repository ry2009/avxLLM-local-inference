#!/usr/bin/env python3
"""Download a set of base models and adapters defined in a manifest."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import snapshot_download


def _download(entry: Dict[str, str], token: Optional[str], revision: Optional[str]) -> Path:
    repo_id = entry["id"]
    target = Path(entry["dir"])
    target.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        revision=revision or entry.get("revision"),
        token=token,
        resume_download=True,
    )
    return Path(resolved)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download models/adapters from a manifest JSON file")
    parser.add_argument("manifest", default="configs/sample_assets.json")
    parser.add_argument("--token")
    parser.add_argument("--revision")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text())
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    downloaded = []
    for key in ("models", "adapters"):
        for entry in data.get(key, []):
            resolved = _download(entry, token, args.revision)
            downloaded.append({"type": key[:-1], "id": entry["id"], "path": str(resolved)})

    print(json.dumps(downloaded, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
