from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, List

from peft_cpu_runtime import CpuPeftRuntime, InferenceRequest, InferenceTraceConfig, RequestBatch
from peft_cpu_runtime.training.utils import resolve_dtype

from .scorer import ScoreResult, load_scorer, pass_at_k


def _load_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def _load_prompts(path: Path) -> List[dict]:
    prompts: List[dict] = []
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                prompts.append(json.loads(line))
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            prompts.extend(payload)
        else:
            raise ValueError("JSON prompts file must contain an array of objects")
    else:
        raise ValueError(f"Unsupported prompts format: {path}")
    return prompts


def _normalise(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _compute_entropy(completions: List[str]) -> float:
    counts = Counter(_normalise(c) for c in completions)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p + 1e-12)
    return entropy


def run(config_path: Path) -> Path:
    cfg = _load_config(config_path)
    prompts_path = Path(cfg["prompts_file"])
    prompts = _load_prompts(prompts_path)
    answer_field = cfg.get("answer_field", "answer")

    adapter_name = cfg.get("adapter_name")
    adapter_path = cfg.get("adapter_path")
    if adapter_name and adapter_path:
        adapter_map = {adapter_name: adapter_path}
    else:
        adapter_map = {}

    dtype_spec = cfg.get("dtype")
    dtype = resolve_dtype(dtype_spec)
    dtype_name = str(dtype).replace("torch.", "")

    runtime = CpuPeftRuntime(
        base_model_id=cfg["base_model"],
        adapter_map=adapter_map,
        torch_dtype=dtype,
    )

    trace = InferenceTraceConfig(
        max_new_tokens=cfg.get("max_new_tokens", 64),
        temperature=cfg.get("temperature", 0.8),
        top_p=cfg.get("top_p", 0.95),
        do_sample=cfg.get("do_sample", True),
    )

    samples_per_prompt = cfg.get("samples_per_prompt", 32)
    k_values = sorted(set(cfg.get("k_values", [8, 16, 32])))

    default_scorer_name = cfg.get("scorer", "exact")

    output_dir = Path(cfg.get("output_dir", "reports/bon_runs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"bon_run_{timestamp}.jsonl"

    metadata = cfg.get("metadata", {})

    with output_path.open("w", encoding="utf-8") as fp:
        for prompt_obj in prompts:
            prompt_text = prompt_obj["prompt"]
            reference = prompt_obj.get(answer_field, "")
            prompt_scorer_name = prompt_obj.get("scorer", default_scorer_name)
            scorer = load_scorer(prompt_scorer_name)
            completions: List[str] = []
            scores: List[ScoreResult] = []
            durations: List[float] = []
            lengths: List[int] = []

            for _ in range(samples_per_prompt):
                request = InferenceRequest(prompt=prompt_text, adapter_name=adapter_name if adapter_map else None)
                batch = RequestBatch(requests=[request], trace_config=trace)
                start = perf_counter()
                outputs = runtime.generate(batch)
                duration = perf_counter() - start
                if not outputs:
                    continue
                completion = outputs[0]
                completions.append(completion)
                durations.append(duration)
                lengths.append(len(completion))
                scores.append(scorer(reference, completion))

            unique_frac = len({ _normalise(c) for c in completions }) / max(len(completions), 1)
            entropy = _compute_entropy(completions)
            avg_len = statistics.mean(len(c.strip()) for c in completions) if completions else 0.0
            avg_duration = statistics.mean(durations) if durations else 0.0
            avg_chars_per_sec = (
                statistics.mean(lengths) / avg_duration if avg_duration > 0.0 and lengths else 0.0
            )

            pass_metrics: Dict[str, float] = {}
            bool_mask = [score.correct for score in scores]
            pass_metrics["1"] = pass_at_k(bool_mask, 1) if bool_mask else 0.0
            for k in k_values:
                pass_metrics[str(k)] = pass_at_k(bool_mask, k) if bool_mask else 0.0

            sample_logs = []
            for completion, score, duration, length in zip(completions, scores, durations, lengths):
                sample_logs.append(
                    {
                        "completion": completion,
                        "correct": score.correct,
                        "details": score.details,
                        "duration_s": duration,
                        "length_chars": length,
                    }
                )

            record = {
                "prompt": prompt_text,
                "answer": reference,
                "samples": sample_logs,
                "metrics": {
                    "pass_at": pass_metrics,
                    "unique_frac": unique_frac,
                    "entropy": entropy,
                    "avg_completion_length": avg_len,
                    "avg_duration_s": avg_duration,
                    "avg_chars_per_sec": avg_chars_per_sec,
                },
                "config": {
                    "k_values": k_values,
                    "samples_per_prompt": samples_per_prompt,
                    "temperature": trace.temperature,
                    "top_p": trace.top_p,
                    "max_new_tokens": trace.max_new_tokens,
                    "dtype": dtype_name,
                },
                "adapter": {
                    "base_model": cfg["base_model"],
                    "adapter_name": adapter_name if adapter_map else None,
                    "adapter_path": adapter_path if adapter_map else None,
                },
                "scorer": prompt_scorer_name,
                "metadata": metadata,
                "timestamp": timestamp,
            }
            fp.write(json.dumps(record) + "\n")

    print(f"Wrote BoN run to {output_path}")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect BoN samples locally on CPU")
    parser.add_argument("--config", type=Path, required=True, help="Collector JSON config")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
