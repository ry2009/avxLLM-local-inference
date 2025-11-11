#!/usr/bin/env python3
"""Benchmark individual prompts and emit per-prompt metrics."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import snapshot_download

from peft_cpu_runtime import CpuPeftRuntime, InferenceRequest, InferenceTraceConfig, RequestBatch
from peft_cpu_runtime.training.utils import resolve_dtype


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompt-level benchmark with telemetry")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-dir", default="models/prompt-bench")
    parser.add_argument("--adapter-id")
    parser.add_argument("--adapter-name", default="bench")
    parser.add_argument("--adapter-dir", default="adapters/prompt-bench")
    parser.add_argument("--prompts", required=True, help="Path to text/JSONL prompts")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--token")
    parser.add_argument("--revision")
    parser.add_argument("--csv", default="reports/prompt_benchmark.csv")
    parser.add_argument("--json", default="reports/prompt_benchmark.json")
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--num-interop-threads", type=int)
    parser.add_argument("--token-cache-size", type=int, default=0)
    return parser.parse_args(argv)


def _download(repo_id: str, target: Path, token: Optional[str], revision: Optional[str]) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            revision=revision,
            token=token,
            resume_download=True,
        )
    )


def _load_prompts(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(f"Prompts file not found: {path}")
    prompts: List[str] = []
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompts.append(record.get("prompt") or record.get("text") or str(record))
    else:
        prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not prompts:
        raise SystemExit("Prompt file empty")
    return prompts


def _run_prompt(runtime: CpuPeftRuntime, prompt: str, adapter: Optional[str], max_new_tokens: int) -> Dict[str, float]:
    batch = RequestBatch(
        requests=[InferenceRequest(prompt=prompt, adapter_name=adapter)],
        trace_config=InferenceTraceConfig(max_new_tokens=max_new_tokens),
    )
    runtime.enable_profiling(True)
    _ = runtime.generate(batch)
    metrics = runtime.benchmark(batch, num_warmup=0, num_iters=1)
    return metrics


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    model_path = _download(args.model_id, Path(args.model_dir), token, args.revision)

    adapter_map = {}
    adapter_name = None
    if args.adapter_id:
        adapter_path = _download(args.adapter_id, Path(args.adapter_dir), token, args.revision)
        adapter_map[args.adapter_name] = str(adapter_path)
        adapter_name = args.adapter_name

    runtime = CpuPeftRuntime(
        base_model_id=str(model_path),
        adapter_map=adapter_map,
        torch_dtype=resolve_dtype(args.dtype),
        num_threads=args.num_threads,
        num_interop_threads=args.num_interop_threads,
        token_cache_size=args.token_cache_size,
    )
    prompts = _load_prompts(Path(args.prompts))

    rows = []
    for prompt in prompts:
        metrics = _run_prompt(runtime, prompt, adapter_name, args.max_new_tokens)
        rows.append(
            {
                "prompt": prompt,
                "tokens_per_second": metrics.get("tokens_per_second"),
                "avg_ttft_s": metrics.get("avg_ttft_s"),
                "seq_per_second": metrics.get("seq_per_second"),
            }
        )

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
