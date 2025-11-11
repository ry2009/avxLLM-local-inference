#!/usr/bin/env python3
"""Self-contained smoke test used for CI/lab reviews."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

from peft_cpu_runtime import CpuPeftRuntime, InferenceRequest, InferenceTraceConfig, RequestBatch
from peft_cpu_runtime.training.utils import resolve_dtype

DEFAULT_MODEL = "sshleifer/tiny-gpt2"
DEFAULT_REVISION = "refs/pr/1"  # Tiny GPT-2 safetensors branch
DEFAULT_PROMPT = "Write one sentence about AVX kernels." 


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CPU inference smoke test and emit telemetry.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--model-dir", default="models/ci-smoke")
    parser.add_argument("--adapter-id", help="Optional LoRA adapter repo")
    parser.add_argument("--adapter-name", default="ci-smoke")
    parser.add_argument("--adapter-dir", default="adapters/ci-smoke")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--token")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--metrics", default="reports/ci_smoke_metrics.json")
    parser.add_argument("--min-tps", type=float)
    parser.add_argument("--max-ttft", type=float)
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


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    model_path = _download(args.model_id, Path(args.model_dir), token, args.revision)

    adapter_map = {}
    if args.adapter_id:
        adapter_path = _download(args.adapter_id, Path(args.adapter_dir), token, args.revision)
        adapter_map[args.adapter_name] = str(adapter_path)

    runtime = CpuPeftRuntime(
        base_model_id=str(model_path),
        adapter_map=adapter_map,
        torch_dtype=resolve_dtype(args.dtype),
    )
    runtime.enable_profiling(True)

    requests = [
        InferenceRequest(prompt=args.prompt, adapter_name=args.adapter_name if adapter_map else None)
    ]
    batch = RequestBatch(
        requests=requests,
        trace_config=InferenceTraceConfig(max_new_tokens=args.max_new_tokens),
    )
    outputs = runtime.generate(batch)
    metrics = runtime.benchmark(batch, num_warmup=0, num_iters=1)

    payload = {"prompt": args.prompt, "outputs": outputs, "metrics": metrics}
    print(json.dumps(payload, indent=2))
    failures = []
    tps = metrics.get("tokens_per_second") or 0.0
    ttft = metrics.get("avg_ttft_s")
    if args.min_tps is not None and tps < args.min_tps:
        failures.append(f"TPS {tps:.2f} < min {args.min_tps}")
    if args.max_ttft is not None and ttft is not None and ttft > args.max_ttft:
        failures.append(f"TTFT {ttft:.2f}s > max {args.max_ttft}s")
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
