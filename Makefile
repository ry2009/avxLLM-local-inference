PYTHON ?= python3
VENV ?= .venv
BUILD_DIR ?= build
CMAKE_FLAGS ?= -DCMAKE_BUILD_TYPE=RelWithDebInfo -DINFENG_AVX2=ON -DINFENG_ENABLE_PYTHON=ON
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

.PHONY: setup cpp-config cpp-build cpp-test py-test test clean run-infer run-train run-pretrain

setup:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && pip install -e .

cpp-config:
	cmake -S . -B $(BUILD_DIR) -G Ninja $(CMAKE_FLAGS)

cpp-build: cpp-config
	cmake --build $(BUILD_DIR) --config RelWithDebInfo -j$(JOBS)

cpp-test:
	ctest --test-dir $(BUILD_DIR) --output-on-failure

py-test:
	. $(VENV)/bin/activate && pytest src/peft_cpu_runtime/tests

test: cpp-test py-test

clean:
	rm -rf $(BUILD_DIR)

run-infer:
	. $(VENV)/bin/activate && python scripts/run_local_inference.py --telemetry

run-train:
	. $(VENV)/bin/activate && python scripts/run_local_training.py

run-pretrain:
	. $(VENV)/bin/activate && python scripts/run_local_pretrain.py

run-eval:
	. $(VENV)/bin/activate && python scripts/run_local_eval.py --telemetry

run-pipeline:
	. $(VENV)/bin/activate && python scripts/run_end_to_end.py --telemetry

run-telemetry:
	. $(VENV)/bin/activate && python scripts/run_telemetry_matrix.py --adapter demo=theone049/agriqa-tinyllama-lora-adapter --prompts data/math_prompts.jsonl

run-throughput:
	. $(VENV)/bin/activate && python scripts/run_throughput_sweep.py --lengths 32,64,128 --adapter demo=theone049/agriqa-tinyllama-lora-adapter --prompts data/math_prompts.jsonl

run-ci-smoke:
	. $(VENV)/bin/activate && python scripts/run_ci_smoke.py

run-rl-demo:
	. $(VENV)/bin/activate && python scripts/run_rl_demo.py

run-rl-eval:
	. $(VENV)/bin/activate && python scripts/run_rl_eval.py --model-id checkpoints/tiny-pretrain-cpu --adapter adapters/rl-tiny

run-perf-dashboard:
	. $(VENV)/bin/activate && python scripts/generate_perf_dashboard.py

run-manifest:
	. $(VENV)/bin/activate && python scripts/download_manifest.py configs/sample_assets.json

check-mac:
	python scripts/check_mac_env.py
