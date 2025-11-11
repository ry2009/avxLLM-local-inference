#!/usr/bin/env python3
"""Sweep prompt lengths/adapters to measure TPS/TTFT."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import snapshot_download

from peft_cpu_runtime import CpuPeftRuntime, InferenceRequest, InferenceTraceConfig, RequestBatch
from peft_cpu_runtime.training.utils import resolve_dtype


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a throughput sweep across prompt sizes/adapters")
    parser.add_argument("--model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--model-dir", default="models/throughput-base")
    parser.add_argument("--adapter", action="append", default=[], help="adapter_name=repo-or-path")
    parser.add_argument("--adapter-dir", default="adapters/throughput")
    parser.add_argument("--lengths", default="32,64,128,256", help="comma-separated prompt lengths")
    parser.add_argument("--prompts-file", help="Optional file with prompts to reuse")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--token")
    parser.add_argument("--revision")
    parser.add_argument("--out", default="reports/throughput_sweep.json")
    return parser.parse_args(argv)


def _download(repo_id: str, target: Path, token: Optional[str], revision: Optional[str]) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        revision=revision,
        token=token,
        resume_download=True,
    )
    return Path(resolved)


def _generate_prompts(lengths: List[int], prompt_file: Optional[Path]) -> Dict[int, List[str]]:
    if prompt_file and prompt_file.exists():
        lines = [line.strip() for line in prompt_file.read_text().splitlines() if line.strip()]
    else:
        lines = []
    prompts_by_len: Dict[int, List[str]] = {}
    for length in lengths:
        if lines:
            prompts = [line[:length] or line for line in lines]
        else:
            prompts = ["A" * length]
        prompts_by_len[length] = prompts
    return prompts_by_len


def _parse_lengths(spec: str) -> List[int]:
    values = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise SystemExit("No prompt lengths specified")
    return values


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    lengths = _parse_lengths(args.lengths)
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    model_path = _download(args.model_id, Path(args.model_dir), token, args.revision)
    prompts_by_len = _generate_prompts(lengths, Path(args.prompts_file) if args.prompts_file else None)

    adapter_specs: Dict[str, str] = {}
    for spec in args.adapter:
        if "=" not in spec:
            raise SystemExit(f"Adapter spec '{spec}' must be name=repo")
        name, value = spec.split("=", 1)
        adapter_specs[name.strip()] = value.strip()
    if not adapter_specs:
        adapter_specs["base"] = ""

    results = []
    for name, repo in adapter_specs.items():
        if repo:
            adapter_path = Path(repo)
            if not adapter_path.exists():
                adapter_path = _download(repo, Path(args.adapter_dir) / name, token, args.revision)
            adapter_map = {name: str(adapter_path)}
        else:
            adapter_map = {}
        runtime = CpuPeftRuntime(
            base_model_id=str(model_path),
            adapter_map=adapter_map,
            torch_dtype=resolve_dtype(args.dtype),
        )
        runtime.enable_profiling(True)
        for length, prompts in prompts_by_len.items():
            batch = RequestBatch(
                requests=[InferenceRequest(prompt=p, adapter_name=name if adapter_map else None) for p in prompts],
                trace_config=InferenceTraceConfig(max_new_tokens=args.max_new_tokens),
            )
            _ = runtime.generate(batch)
            metrics = runtime.benchmark(batch, num_warmup=0, num_iters=1)
            results.append(
                {
                    "adapter": name,
                    "length": length,
                    "tokens_per_second": metrics.get("tokens_per_second"),
                    "avg_ttft_s": metrics.get("avg_ttft_s"),
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
