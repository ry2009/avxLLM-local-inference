from __future__ import annotations

import argparse
import importlib
import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Iterable, List, Optional

from . import (
    CpuPeftRuntime,
    DatasetConfig,
    InferenceRequest,
    InferenceTraceConfig,
    RequestBatch,
    TrainingConfig,
    train_lora_adapter,
    train_policy_rl,
    train_causal_lm,
)
from .training.config import PretrainConfig, RLConfig
from .training.rewards import get_reward as get_builtin_reward
from .training.utils import resolve_dtype


def _configure_infer_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="Base Hugging Face model ID or local path")
    parser.add_argument("prompts", nargs="+", help="Prompts to generate")
    parser.add_argument("--adapter", action="append", default=[], help="adapter_name=path")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--telemetry", action="store_true", help="Enable runtime telemetry")
    parser.add_argument("--telemetry-csv", default="reports/runtime_metrics.csv")
    parser.add_argument("--dtype", help="Model precision to load (e.g. float32, bf16, fp16)")


def _add_sft_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("sft", help="LoRA supervised fine-tuning")
    parser.add_argument("--config", type=Path, help="JSON file containing TrainingConfig")
    parser.add_argument("--base-model")
    parser.add_argument("--adapter-name")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--hf-cache-dir", type=Path)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--dtype", help="Model precision to load (e.g. float32, bf16, fp16)")
    return parser


def _add_rl_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("rl", help="Policy fine-tuning with lightweight RL")
    parser.add_argument("--config", type=Path, help="JSON file containing RLConfig")
    parser.add_argument("--base-model")
    parser.add_argument("--adapter-name")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-file", type=Path, help="JSONL/CSV file with prompts")
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--reward", required=False, help="Import path or file:func for reward callable")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--mini-batch-size", type=int)
    parser.add_argument("--kl-coef", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--hf-cache-dir", type=Path)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--dtype", help="Model precision to load (e.g. float32, bf16, fp16)")
    return parser


def _add_pretrain_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("pretrain", help="Causal LM pre-training on CPU")
    parser.add_argument("--config", type=Path, help="JSON file containing PretrainConfig")
    parser.add_argument("--base-model")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--dtype", help="Model precision to load (e.g. float32, bf16, fp16)")
    return parser


def load_adapter_map(values: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for spec in values:
        if "=" not in spec:
            raise SystemExit(f"Adapter spec '{spec}' must be name=path")
        name, path = spec.split("=", 1)
        mapping[name.strip()] = path.strip()
    return mapping


def _maybe_import_from_file(module_path: str) -> ModuleType:
    path = Path(module_path)
    if path.exists():
        module_name = f"reward_{path.stem}"
        loader = SourceFileLoader(module_name, str(path))
        module = ModuleType(module_name)
        loader.exec_module(module)
        return module
    return importlib.import_module(module_path)


def _load_reward(spec: str) -> Callable[[List[str], List[str], Optional[dict]], List[float]]:
    if spec.startswith("builtin:"):
        name = spec.split(":", 1)[1]
        try:
            return get_builtin_reward(name)
        except KeyError as exc:
            raise SystemExit(str(exc)) from exc
    if ":" not in spec:
        raise SystemExit("Reward spec must be module:function, file.py:function, or builtin:<name>")
    module_name, func_name = spec.split(":", 1)
    module = _maybe_import_from_file(module_name)
    fn = getattr(module, func_name, None)
    if fn is None:
        raise SystemExit(f"Reward function '{func_name}' not found in {module_name}")
    return fn


def _build_dataset_config(args, requires_file: bool) -> DatasetConfig:
    dataset = DatasetConfig()
    if args.train_file:
        dataset.path = args.train_file
    elif requires_file:
        raise SystemExit("--train-file is required for this command")
    if getattr(args, "text_field", None):
        dataset.field = args.text_field
    if getattr(args, "no_shuffle", False):
        dataset.shuffle = False
    return dataset


def _run_inference(args: argparse.Namespace) -> None:
    adapter_map = load_adapter_map(args.adapter)
    torch_dtype = resolve_dtype(args.dtype)
    runtime = CpuPeftRuntime(
        base_model_id=args.model,
        adapter_map=adapter_map,
        torch_dtype=torch_dtype,
    )
    if args.telemetry:
        from infeng.runtime import runtime as engine_runtime, EngineConfig

        engine_runtime.initialize(EngineConfig(enable_metrics=True))

    trace = InferenceTraceConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )

    requests = [InferenceRequest(prompt=p) for p in args.prompts]
    batch = RequestBatch(requests=requests, trace_config=trace)
    outputs = runtime.generate(batch)

    result = {"prompts": args.prompts, "outputs": outputs}
    print(json.dumps(result, indent=2))

    if args.telemetry:
        from infeng.telemetry import flush_csv

        Path(args.telemetry_csv).parent.mkdir(parents=True, exist_ok=True)
        flush_csv(args.telemetry_csv)


def _run_sft(args: argparse.Namespace) -> None:
    if args.config:
        cfg = TrainingConfig.from_json(args.config)
    else:
        if not (args.base_model and args.adapter_name and args.output_dir):
            raise SystemExit("--base-model, --adapter-name, --output-dir are required when no config file provided")
        cfg = TrainingConfig(
            base_model=args.base_model,
            adapter_name=args.adapter_name,
            output_dir=args.output_dir,
            prompts=args.prompt or [],
            epochs=args.epochs or 1,
            batch_size=args.batch_size or 1,
            max_seq_len=args.max_seq_len or 256,
        )
        cfg.optimizer.lr = args.lr or cfg.optimizer.lr
        cfg.optimizer.weight_decay = args.weight_decay or cfg.optimizer.weight_decay
        cfg.lora.r = args.lora_r or cfg.lora.r
        cfg.lora.alpha = args.lora_alpha or cfg.lora.alpha
        cfg.lora.dropout = args.lora_dropout or cfg.lora.dropout
        cfg.dataset = _build_dataset_config(args, requires_file=False)
    if args.hf_cache_dir:
        cfg.hf_cache_dir = args.hf_cache_dir
    if args.dtype:
        cfg.model_dtype = args.dtype
    override_dataset = _build_dataset_config(args, requires_file=False)
    if args.train_file:
        cfg.dataset.path = override_dataset.path
    if getattr(args, "text_field", None):
        cfg.dataset.field = override_dataset.field
    if getattr(args, "no_shuffle", False):
        cfg.dataset.shuffle = override_dataset.shuffle
    adapter_path = train_lora_adapter(cfg)
    print(json.dumps({"adapter": str(adapter_path)}, indent=2))


def _run_rl(args: argparse.Namespace) -> None:
    if args.config:
        rl_cfg = RLConfig.from_json(args.config, _load_reward)
    else:
        if not args.reward:
            raise SystemExit("--reward is required when no RL config file provided")
        reward_fn = _load_reward(args.reward)
        dataset = _build_dataset_config(args, requires_file=True)
        dataset.field = args.text_field or "prompt"
        rl_cfg = RLConfig(
            base_model=args.base_model,
            adapter_name=args.adapter_name,
            output_dir=args.output_dir,
            dataset=dataset,
            reward_fn=reward_fn,
            max_new_tokens=args.max_new_tokens or 64,
            epochs=args.epochs or 1,
            batch_size=args.batch_size or 1,
            mini_batch_size=args.mini_batch_size or 1,
            gamma=args.gamma or 0.99,
            kl_coef=args.kl_coef or 0.01,
        )
        if args.lr:
            rl_cfg.optimizer.lr = args.lr
        if args.weight_decay:
            rl_cfg.optimizer.weight_decay = args.weight_decay
    if args.hf_cache_dir:
        rl_cfg.hf_cache_dir = args.hf_cache_dir
    if args.dtype:
        rl_cfg.model_dtype = args.dtype
    override_dataset = _build_dataset_config(args, requires_file=False)
    if args.train_file:
        rl_cfg.dataset.path = override_dataset.path
    if getattr(args, "text_field", None):
        rl_cfg.dataset.field = override_dataset.field or rl_cfg.dataset.field
    if getattr(args, "no_shuffle", False):
        rl_cfg.dataset.shuffle = override_dataset.shuffle
    adapter_path = train_policy_rl(rl_cfg)
    print(json.dumps({"adapter": str(adapter_path)}, indent=2))


def _run_pretrain(args: argparse.Namespace) -> None:
    if args.config:
        cfg = PretrainConfig.from_json(args.config)
    else:
        dataset = _build_dataset_config(args, requires_file=True)
        cfg = PretrainConfig(
            base_model=args.base_model,
            output_dir=args.output_dir,
            dataset=dataset,
            epochs=args.epochs or 1,
            batch_size=args.batch_size or 1,
            max_seq_len=args.max_seq_len or 256,
        )
        if args.lr:
            cfg.optimizer.lr = args.lr
        if args.weight_decay:
            cfg.optimizer.weight_decay = args.weight_decay
    if args.dtype:
        cfg.model_dtype = args.dtype
    override_dataset = _build_dataset_config(args, requires_file=False)
    if args.train_file:
        cfg.dataset.path = override_dataset.path
    if getattr(args, "text_field", None):
        cfg.dataset.field = override_dataset.field or cfg.dataset.field
    if getattr(args, "no_shuffle", False):
        cfg.dataset.shuffle = override_dataset.shuffle
    out = train_causal_lm(cfg)
    print(json.dumps({"model": str(out)}, indent=2))


def _parse_with_subcommands(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="peft-cpu-run", description="CPU PEFT runtime")
    subparsers = parser.add_subparsers(dest="command")
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    _configure_infer_parser(infer_parser)
    _add_sft_parser(subparsers)
    _add_rl_parser(subparsers)
    _add_pretrain_parser(subparsers)
    return parser.parse_args(argv)


def main() -> None:
    commands = {"infer", "sft", "rl", "pretrain"}
    if len(sys.argv) > 1 and sys.argv[1] not in commands:
        parser = argparse.ArgumentParser(description="CPU LoRA inference CLI")
        _configure_infer_parser(parser)
        args = parser.parse_args()
        _run_inference(args)
        return

    args = _parse_with_subcommands(sys.argv[1:])
    if args.command == "infer":
        _run_inference(args)
    elif args.command == "sft":
        _run_sft(args)
    elif args.command == "rl":
        _run_rl(args)
    elif args.command == "pretrain":
        _run_pretrain(args)
    else:
        raise SystemExit("No command specified")


if __name__ == "__main__":
    main()

