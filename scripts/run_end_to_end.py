#!/usr/bin/env python3
"""One-command pipeline: download base, train LoRA, and run eval prompts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import snapshot_download

from peft_cpu_runtime import CpuPeftRuntime, InferenceRequest, InferenceTraceConfig, RequestBatch
from peft_cpu_runtime.training import TrainingConfig, train_lora_adapter
from peft_cpu_runtime.training.config import DatasetConfig
from peft_cpu_runtime.training.utils import resolve_dtype

DEFAULT_BASE = "sshleifer/tiny-gpt2"
DEFAULT_DATASET = "data/distill_pairs.jsonl"
DEFAULT_EVAL = "data/math_prompts.jsonl"


def _download(repo_id: str, target: Path, token: Optional[str], revision: Optional[str]) -> Path:
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


def _load_jsonl(path: Path, limit: int) -> List[str]:
    prompts: List[str] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "text" in record:
                prompts.append(str(record["text"]))
            elif "prompt" in record:
                prompts.append(str(record["prompt"]))
            else:
                prompts.append(str(record))
            if len(prompts) >= limit:
                break
    if not prompts:
        raise SystemExit(f"No prompts found in {path}")
    return prompts


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download model, train adapter, run eval prompts.")
    parser.add_argument("--model-id", default=DEFAULT_BASE)
    parser.add_argument("--model-dir", default="models/pipeline-base")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-field", default="text")
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--adapter-name", default="pipeline-adapter")
    parser.add_argument("--output-dir", default="adapters/pipeline")
    parser.add_argument("--eval-prompts", default=DEFAULT_EVAL)
    parser.add_argument("--eval-limit", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--token")
    parser.add_argument("--revision")
    parser.add_argument("--reuse-adapter", help="Skip training and reuse this adapter path")
    parser.add_argument("--telemetry", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    model_path = _download(args.model_id, Path(args.model_dir), token, args.revision)

    if args.reuse_adapter:
        adapter_path = Path(args.reuse_adapter)
    else:
        cfg = TrainingConfig(
            base_model=str(model_path),
            adapter_name=args.adapter_name,
            output_dir=Path(args.output_dir),
            epochs=1,
            batch_size=1,
            max_seq_len=256,
        )
        cfg.dataset = DatasetConfig(
            path=Path(args.dataset),
            field=args.dataset_field,
            max_samples=args.train_samples,
            shuffle=False,
        )
        adapter_path = train_lora_adapter(cfg)

    prompts = _load_jsonl(Path(args.eval_prompts), args.eval_limit)
    adapter_map: Dict[str, str] = {args.adapter_name: str(adapter_path)}

    runtime = CpuPeftRuntime(
        base_model_id=str(model_path),
        adapter_map=adapter_map,
        torch_dtype=resolve_dtype(args.dtype),
    )
    if args.telemetry:
        runtime.enable_profiling(True)

    batch = RequestBatch(
        requests=[InferenceRequest(prompt=p, adapter_name=args.adapter_name) for p in prompts],
        trace_config=InferenceTraceConfig(max_new_tokens=args.max_new_tokens),
    )
    outputs = runtime.generate(batch)
    result = {"prompts": prompts, "outputs": outputs}
    print(json.dumps(result, indent=2))

    if args.telemetry:
        metrics = runtime.benchmark(batch, num_warmup=0, num_iters=1)
        print("\nTelemetry")
        print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
