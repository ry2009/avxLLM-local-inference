#!/usr/bin/env python3
"""CPU-friendly causal LM pretraining helper."""
from __future__ import annotations

import argparse
from pathlib import Path

from peft_cpu_runtime.training import PretrainConfig, train_causal_lm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a tiny causal LM checkpoint on CPU.")
    parser.add_argument("--base-model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dataset", default="data/wiki_subset.jsonl", help="JSONL dataset with 'text' entries")
    parser.add_argument("--output-dir", default="checkpoints/tiny-pretrain-cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dtype", default="float32", help="Model dtype (float32, bf16, fp16)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cfg = PretrainConfig(
        base_model=args.base_model,
        dataset=Path(args.dataset),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        model_dtype=args.dtype,
    )
    cfg.optimizer.lr = args.learning_rate
    cfg.optimizer.weight_decay = args.weight_decay

    artifact = train_causal_lm(cfg)
    print(f"Pretrained weights saved to {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
