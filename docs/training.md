# Local Training Toolkit

The CPU-first stack now ships with lightweight building blocks for fine-tuning and
policy optimisation directly on macOS Intel machines. All commands run entirely on the CPU
and lean on LoRA adapters to keep memory use small.

## Precision controls

Every training entry point now accepts a `--dtype` flag (and `model_dtype` inside the JSON configs)
to load the base model in the desired precision. Use `float32` for maximum portability, or switch to
`fp16`/`bf16` when you want parity between training and inference. Following Qi et al. (2025), the
default configs for the RL demos ship with `fp16` to minimise policy drift between rollout and update
engines while keeping everything on CPU.

## Supervised LoRA fine-tuning (SFT)

```bash
# JSON config (see examples/training/sft_config.json)
peft-cpu-run sft --config configs/my_sft.json

# One-off from CLI options
peft-cpu-run sft \
  --base-model sshleifer/tiny-gpt2 \
  --adapter-name math-sft \
  --output-dir adapters/math-sft \
  --prompt "2+2=" --prompt "solve for x: x+3=5" --dtype fp16 \
  --epochs 2 --batch-size 1 --max-seq-len 128 --lr 5e-5
```

## Lightweight RL (REINFORCE + KL guard)

```bash
peft-cpu-run rl \
  --base-model sshleifer/tiny-gpt2 \
  --adapter-name math-rl \
  --output-dir adapters/math-rl \
  --train-file data/math_prompts.jsonl \
  --text-field prompt \
  --reward builtin:exact_match \
  --epochs 1 --batch-size 2 --mini-batch-size 1 --max-new-tokens 48 \
  --dtype fp16
```

Reward callbacks receive `(prompts, completions, metadata)` and must return a list of floats.
Use `builtin:<name>` for the bundled helpers (`exact_match`, `length`), or point to
`package.module:function` / `path/to/file.py:function` for custom logic.

## Causal pre-training refresh

```bash
peft-cpu-run pretrain \
  --base-model sshleifer/tiny-gpt2 \
  --output-dir checkpoints/tiny-pretrain \
  --train-file data/wiki_subset.jsonl \
  --text-field text --epochs 1 --batch-size 2 --max-seq-len 256 \
  --dtype bf16
```

All training commands obey the shared dataset loader (`DatasetConfig`) so JSON/JSONL/CSV
are accepted without additional wiring.

For more complex workflows, prefer the JSON config files – each dataclass exposes a
`from_json` helper mirroring the CLI arguments.
