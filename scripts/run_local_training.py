#!/usr/bin/env python3
"""Minimal LoRA training loop for macOS/Linux quickstarts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from peft_cpu_runtime.training import TrainingConfig, train_lora_adapter


def load_prompts(path: Path, limit: int) -> List[str]:
    prompts: List[str] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if "text" in record:
                sample = record["text"]
            else:
                prompt = record.get("prompt", "")
                completion = record.get("completion", "")
                sample = f"{prompt} {completion}".strip()
            if sample:
                prompts.append(sample)
            if len(prompts) >= limit:
                break
    if not prompts:
        raise SystemExit(f"No prompts found in {path}")
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny LoRA adapter for local inference experiments.")
    parser.add_argument("--base-model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--adapter-name", default="quickstart")
    parser.add_argument("--output-dir", default="adapters/quickstart-trained")
    parser.add_argument("--dataset", default="data/distill_math.jsonl")
    parser.add_argument("--limit", type=int, default=64, help="Number of samples to pull from the dataset")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    prompts = load_prompts(dataset_path, args.limit)

    cfg = TrainingConfig(
        base_model=args.base_model,
        adapter_name=args.adapter_name,
        output_dir=Path(args.output_dir),
        prompts=prompts,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )
    cfg.optimizer.lr = args.learning_rate
    cfg.lora.r = args.lora_r
    cfg.lora.alpha = args.lora_alpha
    cfg.lora.dropout = args.lora_dropout

    adapter_path = train_lora_adapter(prompts, cfg)
    print(f"Adapter saved to {adapter_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
