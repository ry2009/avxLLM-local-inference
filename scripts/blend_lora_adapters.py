#!/usr/bin/env python3
"""Blend two LoRA adapters with a linear combination."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend two LoRA adapters into a new adapter")
    parser.add_argument("--adapter-a", required=True, help="Path to first adapter directory")
    parser.add_argument("--adapter-b", required=True, help="Path to second adapter directory")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for adapter A (0-1)")
    parser.add_argument("--output", required=True, help="Output adapter directory")
    return parser.parse_args(argv)


def load_adapter(path: Path) -> dict:
    file = path / "adapter_model.safetensors"
    if not file.exists():
        raise SystemExit(f"Adapter weights not found: {file}")
    return load_file(str(file))


def blend_states(state_a: dict, state_b: dict, alpha: float) -> dict:
    if state_a.keys() != state_b.keys():
        raise SystemExit("Adapters have mismatched keys")
    beta = 1.0 - alpha
    return {key: alpha * state_a[key] + beta * state_b[key] for key in state_a}


def copy_config(src: Path, dst: Path) -> None:
    config_src = src / "adapter_config.json"
    config_dst = dst / "adapter_config.json"
    data = json.loads(config_src.read_text())
    data["peft_type"] = data.get("peft_type", "LORA")
    config_dst.write_text(json.dumps(data, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    alpha = args.alpha
    if not 0.0 <= alpha <= 1.0:
        raise SystemExit("Alpha must be between 0 and 1")

    adapter_a = load_adapter(Path(args.adapter_a))
    adapter_b = load_adapter(Path(args.adapter_b))
    blended = blend_states(adapter_a, adapter_b, alpha)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(blended, str(out_dir / "adapter_model.safetensors"))
    copy_config(Path(args.adapter_a), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
