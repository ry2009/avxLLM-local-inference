#!/usr/bin/env python3
"""CPU-only RL demo using tiny GPT-2 and math prompts."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from peft_cpu_runtime.training import RLConfig
from peft_cpu_runtime.training.data import load_prompts
from peft_cpu_runtime.training.rl import train_policy_rl
from peft_cpu_runtime.training.rewards import get_reward

DEFAULT_CONFIG = Path("configs/rl_tiny.json")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CPU-scale RL fine-tuning demo")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=32, help="Override dataset sample count")
    parser.add_argument("--output", type=Path, help="Optional override for adapter output dir")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = RLConfig.from_json(args.config, get_reward)
    if args.output:
        cfg.output_dir = args.output
    dataset = cfg.dataset
    if args.limit:
        dataset.max_samples = args.limit
    prompts = load_prompts(dataset)
    adapter_path = train_policy_rl(prompts, cfg)
    print(f"RL adapter saved to {adapter_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
