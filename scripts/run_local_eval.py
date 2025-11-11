#!/usr/bin/env python3
"""Evaluate a model/adapter on a local prompt file with telemetry output."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import snapshot_download

from peft_cpu_runtime import CpuPeftRuntime, InferenceRequest, InferenceTraceConfig, RequestBatch
from peft_cpu_runtime.training.utils import resolve_dtype


def _download(repo_id: str, target: Path, token: Optional[str], revision: Optional[str] = None) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        revision=revision,
        token=token,
        resume_download=True,
    )
    return Path(path)


def _load_prompts(path: Path, limit: Optional[int], field: str) -> List[str]:
    prompts: List[str] = []
    if not path.exists():
        raise SystemExit(f"Prompt file not found: {path}")
    if path.suffix.lower() in {".jsonl", ".jsonl.gz"}:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if field in record:
                    prompts.append(str(record[field]))
                elif "prompt" in record:
                    prompts.append(str(record["prompt"]))
                elif "text" in record:
                    prompts.append(str(record["text"]))
                else:
                    prompts.append(str(record))
                if limit and len(prompts) >= limit:
                    break
    else:
        prompts = path.read_text().splitlines()
        prompts = [p for p in prompts if p.strip()]
        if limit:
            prompts = prompts[:limit]
    if not prompts:
        raise SystemExit(f"No prompts loaded from {path}")
    return prompts


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local eval on a prompt dataset with telemetry.")
    parser.add_argument("--model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--model-dir", default="models/eval-base")
    parser.add_argument("--adapter-id")
    parser.add_argument("--adapter-name", default="eval-adapter")
    parser.add_argument("--adapter-dir", default="adapters/eval")
    parser.add_argument("--prompts-file", default="data/math_prompts.jsonl")
    parser.add_argument("--field", default="text", help="Field to read from JSON/JSONL entries")
    parser.add_argument("--max-prompts", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--token")
    parser.add_argument("--revision")
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--metrics-out", default="reports/local_eval_metrics.json")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    model_path = _download(args.model_id, Path(args.model_dir), token, args.revision)
    adapter_map: Dict[str, str] = {}
    if args.adapter_id:
        adapter_path = _download(args.adapter_id, Path(args.adapter_dir), token, args.revision)
        adapter_map[args.adapter_name] = str(adapter_path)

    prompts = _load_prompts(Path(args.prompts_file), args.max_prompts, args.field)

    runtime = CpuPeftRuntime(
        base_model_id=str(model_path),
        adapter_map=adapter_map,
        torch_dtype=resolve_dtype(args.dtype),
    )
    if args.telemetry:
        runtime.enable_profiling(True)

    requests = [
        InferenceRequest(prompt=prompt, adapter_name=args.adapter_name if adapter_map else None)
        for prompt in prompts
    ]
    batch = RequestBatch(
        requests=requests,
        trace_config=InferenceTraceConfig(max_new_tokens=args.max_new_tokens),
    )
    outputs = runtime.generate(batch)
    print(json.dumps({"prompts": prompts, "outputs": outputs}, indent=2))

    if args.telemetry:
        metrics = runtime.benchmark(batch, num_warmup=0, num_iters=1)
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"Saved telemetry to {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
