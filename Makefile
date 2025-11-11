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

run-manifest:
	. $(VENV)/bin/activate && python scripts/download_manifest.py configs/sample_assets.json

check-mac:
	python scripts/check_mac_env.py
