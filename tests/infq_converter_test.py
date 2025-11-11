#!/usr/bin/env python3
"""End-to-end test for safetensors -> INFQ conversion."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTER = REPO_ROOT / "tools" / "convert" / "safetensors_to_infq.py"


def create_base(path: Path) -> Path:
    data = {
        "layers.0.weight": np.linspace(-1.0, 1.0, 4, dtype=np.float32).reshape(2, 2),
        "layers.1.weight": np.array([[0.1, -0.2, 0.3]], dtype=np.float32),
    }
    save_file(data, str(path))
    return path


def create_adapter(path: Path) -> Path:
    data = {
        "A": np.array([[0.05, -0.03], [0.02, 0.01]], dtype=np.float32),
        "B": np.array([[0.5, -0.4, 0.3]], dtype=np.float32),
    }
    save_file(data, str(path))
    return path


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        base = tmp_dir / "base.safetensors"
        adapter = tmp_dir / "adapter.safetensors"
        create_base(base)
        create_adapter(adapter)

        import subprocess

        for threshold in (0.0, 6.0):
            out_dir = tmp_dir / f"infq_{threshold}"
            cmd = [
                "python3",
                str(CONVERTER),
                "--base",
                str(base),
                "--adapter",
                f"foo={adapter}",
                "--out",
                str(out_dir),
                "--outlier-threshold",
                str(threshold),
            ]
            subprocess.check_call(cmd)

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            assert manifest["version"] == 1
            assert len(manifest["tensors"]) == 2
            assert manifest["adapters"][0]["name"] == "foo"
            weights_file = out_dir / manifest["tensors"][0]["data_file"]
            assert weights_file.exists()


if __name__ == "__main__":
    main()
