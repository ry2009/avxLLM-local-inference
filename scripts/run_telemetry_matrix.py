#!/usr/bin/env python3
"""Run multiple adapters/prompts and aggregate telemetry metrics."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import snapshot_download

from peft_cpu_runtime import CpuPeftRuntime, InferenceRequest, InferenceTraceConfig, RequestBatch
from peft_cpu_runtime.training.utils import resolve_dtype


def _parse_adapter_spec(values: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for spec in values:
        if "=" not in spec:
            raise SystemExit(f"Adapter spec '{spec}' must be name=repo-or-path")
        name, repo = spec.split("=", 1)
        mapping[name.strip()] = repo.strip()
    return mapping


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


def _read_prompts(path: Path, limit: Optional[int]) -> List[str]:
    if not path.exists():
        raise SystemExit(f"Prompts file not found: {path}")
    prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if limit:
        prompts = prompts[:limit]
    if not prompts:
        raise SystemExit(f"No prompts found in {path}")
    return prompts


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple adapters and collect telemetry metrics")
    parser.add_argument("--model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--model-dir", default="models/telemetry-base")
    parser.add_argument("--adapter", action="append", default=[])
    parser.add_argument("--adapter-dir", default="adapters/telemetry")
    parser.add_argument("--prompts", default="data/math_prompts.jsonl")
    parser.add_argument("--prompts-limit", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--token")
    parser.add_argument("--revision")
    parser.add_argument("--out", default="reports/telemetry_matrix.json")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    adapter_specs = _parse_adapter_spec(args.adapter)
    if not adapter_specs:
        raise SystemExit("Provide at least one --adapter name=repo")

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    model_path = _download(args.model_id, Path(args.model_dir), token, args.revision)
    prompts = _read_prompts(Path(args.prompts), args.prompts_limit)

    results = []
    for name, repo in adapter_specs.items():
        if Path(repo).exists():
            adapter_path = Path(repo)
        else:
            adapter_path = _download(repo, Path(args.adapter_dir) / name, token, args.revision)
        runtime = CpuPeftRuntime(
            base_model_id=str(model_path),
            adapter_map={name: str(adapter_path)},
            torch_dtype=resolve_dtype(args.dtype),
        )
        runtime.enable_profiling(True)
        batch = RequestBatch(
            requests=[InferenceRequest(prompt=p, adapter_name=name) for p in prompts],
            trace_config=InferenceTraceConfig(max_new_tokens=args.max_new_tokens),
        )
        _ = runtime.generate(batch)
        metrics = runtime.benchmark(batch, num_warmup=0, num_iters=1)
        results.append(
            {
                "adapter": name,
                "repo": repo,
                "tokens_per_second": metrics.get("tokens_per_second"),
                "avg_ttft_s": metrics.get("avg_ttft_s"),
                "iterations": metrics.get("iterations"),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
