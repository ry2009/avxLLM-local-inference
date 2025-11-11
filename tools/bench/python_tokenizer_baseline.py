#!/usr/bin/env python3
"""Tokenization baseline using Hugging Face tokenizers (Python)."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

from tokenizers import Tokenizer


def read_prompts(path: Path) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise SystemExit(f"prompts file {path} is empty")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input_file", type=Path, default=Path("tools/bench/prompts.txt"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("reports/tokenizer_py.csv"))
    args = parser.parse_args()

    if args.threads <= 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ.setdefault("RAYON_NUM_THREADS", str(args.threads))

    tokenizer = Tokenizer.from_file(str(args.model))

    prompts = read_prompts(args.input_file)

    start = time.perf_counter()
    total_tokens = 0
    for prompt in prompts:
        encoding = tokenizer.encode(prompt, add_special_tokens=False)
        total_tokens += len(encoding.ids)
    elapsed = time.perf_counter() - start
    tokens_per_s = total_tokens / elapsed if elapsed else 0.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["bench", "threads", "tokens_per_s"])
        writer.writerow(["python_tokenizer", args.threads, f"{tokens_per_s:.6f}"])

    print(f"Python tokenizer tokens/sec: {tokens_per_s:.2f}")


if __name__ == "__main__":
    main()
