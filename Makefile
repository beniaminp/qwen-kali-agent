SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
MODEL ?= mlx-community/Qwen3-14B-4bit
ADAPTER ?= adapters/qwen3-14b-all-layers
FUSED ?= fused/qwen3-14b-pentest-4bit
DATA ?= data/processed
TOOLS ?= examples/tools/kali-tools.json
JAVA_MODEL ?= mlx-community/Qwen2.5-Coder-14B-Instruct-4bit
JAVA_DATA ?= data/java/processed
JAVA_TOOLS ?= examples/java/tools.json
JAVA_ADAPTER ?= adapters/qwen25-coder-14b-java
JAVA_FUSED ?= fused/qwen25-coder-14b-java-4bit
JAVA_REPO ?= ../quarkus-quickstarts
REACT_MODEL ?= mlx-community/Qwen2.5-Coder-14B-Instruct-4bit
REACT_DATA ?= data/react/processed
REACT_TOOLS ?= examples/react/tools.json
REACT_ADAPTER ?= adapters/qwen25-coder-14b-react
REACT_FUSED ?= fused/qwen25-coder-14b-react-4bit
REACT_REPO ?= ../shadcn-ui/ui
UNIFIED_MODEL ?= mlx-community/Qwen3-14B-4bit
UNIFIED_DATA ?= data/unified/processed
UNIFIED_ADAPTER ?= adapters/qwen3-14b-unified
MKDOCS ?= $(PYTHON) -m mkdocs

.PHONY: help setup check data-check stats train-8b train-8b-all train-14b train-14b-last16 \
        check-lora sweep eval fuse serve serve-llama tunnel \
        java-check java-mine java-train-7b java-train-14b java-check-lora java-fuse \
        react-check react-mine react-train-7b react-train-14b react-check-lora react-fuse \
        unified-mix unified-check unified-train unified-check-lora \
        docs-install docs docs-serve clean

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Create the Python environment on macOS
	uv venv --python 3.12 .venv
	uv pip install -r requirements.txt
	$(PYTHON) scripts/check_env.py

check: ## Verify environment and memory headroom
	$(PYTHON) scripts/check_env.py

data-check: ## Validate train/valid/test JSONL
	$(PYTHON) scripts/validate_dataset.py $(DATA)/train.jsonl --tools $(TOOLS)
	$(PYTHON) scripts/validate_dataset.py $(DATA)/valid.jsonl --tools $(TOOLS)

stats: ## Token-length distribution for the training split
	$(PYTHON) scripts/token_stats.py $(DATA)/train.jsonl --model Qwen/Qwen3-14B

train-8b: ## Fast 8B iteration run
	mlx_lm.lora --config configs/qwen3-8b-fast.yaml

train-8b-all: ## 8B all-layer run
	mlx_lm.lora --config configs/qwen3-8b-all-layers.yaml

train-14b: ## Recommended 14B run (all layers)
	mlx_lm.lora --config configs/qwen3-14b-all-layers.yaml

train-14b-last16: ## 14B baseline (last 16 blocks)
	mlx_lm.lora --config configs/qwen3-14b-balanced.yaml

check-lora: ## Verify which modules the adapter actually targets
	$(PYTHON) scripts/check_lora.py --model $(MODEL) --num-layers -1 --rank 16 --scale 2

java-check: ## Validate the Java/Quarkus splits and token lengths
	$(PYTHON) scripts/validate_dataset.py $(JAVA_DATA)/train.jsonl --tools $(JAVA_TOOLS)
	$(PYTHON) scripts/validate_java_dataset.py $(JAVA_DATA)/train.jsonl --tools $(JAVA_TOOLS) --require-dict-arguments --strict
	$(PYTHON) scripts/token_stats.py $(JAVA_DATA)/train.jsonl --model Qwen/Qwen2.5-Coder-14B-Instruct

java-mine: ## Mine Java commit seeds from a permissive repo
	$(PYTHON) scripts/mine_java_commits.py --repo $(JAVA_REPO) \
	  --out data/java/interim/commit-seeds.jsonl --limit 300 --only-with-tests --license Apache-2.0

java-train-7b: ## Java/Quarkus 7B smoke run
	mlx_lm.lora --config configs/qwen2.5-coder-7b-java.yaml

java-train-14b: ## Java/Quarkus 14B run (24/48 blocks)
	mlx_lm.lora --config configs/qwen2.5-coder-14b-java.yaml

java-check-lora: ## Verify the Java adapter covers MLP projections
	$(PYTHON) scripts/check_lora.py --model $(JAVA_MODEL) --num-layers 24 --rank 16 --scale 2

java-fuse: ## Fuse the Java adapter
	mlx_lm.fuse --model $(JAVA_MODEL) --adapter-path $(JAVA_ADAPTER) --save-path $(JAVA_FUSED)

react-check: ## Validate the React/Tailwind splits and token lengths
	$(PYTHON) scripts/validate_dataset.py $(REACT_DATA)/train.jsonl --tools $(REACT_TOOLS)
	$(PYTHON) scripts/validate_react_dataset.py $(REACT_DATA)/train.jsonl --tools $(REACT_TOOLS) --require-dict-arguments --strict
	$(PYTHON) scripts/token_stats.py $(REACT_DATA)/train.jsonl --model Qwen/Qwen2.5-Coder-14B-Instruct

react-mine: ## Mine React commit seeds from a permissive repo
	$(PYTHON) scripts/mine_react_commits.py --repo $(REACT_REPO) \
	  --out data/react/interim/commit-seeds.jsonl --limit 300 --only-with-tests --license MIT

react-train-7b: ## React/Tailwind 7B long-context smoke run
	mlx_lm.lora --config configs/qwen2.5-coder-7b-react.yaml

react-train-14b: ## React/Tailwind 14B run (24/48 blocks, 3K)
	mlx_lm.lora --config configs/qwen2.5-coder-14b-react.yaml

react-check-lora: ## Verify the React adapter covers MLP projections
	$(PYTHON) scripts/check_lora.py --model $(REACT_MODEL) --num-layers 24 --rank 16 --scale 2

react-fuse: ## Fuse the React adapter
	mlx_lm.fuse --model $(REACT_MODEL) --adapter-path $(REACT_ADAPTER) --save-path $(REACT_FUSED)

unified-mix: ## Mix Kali + Java + React data for one Qwen3 adapter
	$(PYTHON) scripts/mix_datasets.py \
	  --source kali=$(DATA)/train.jsonl \
	  --source java=$(JAVA_DATA)/train.jsonl \
	  --source react=$(REACT_DATA)/train.jsonl \
	  --weights kali=0.40,java=0.30,react=0.30 \
	  --out $(UNIFIED_DATA) --arguments-format string

unified-check: ## Validate the unified split and token lengths
	$(PYTHON) scripts/validate_dataset.py $(UNIFIED_DATA)/train.jsonl
	$(PYTHON) scripts/token_stats.py $(UNIFIED_DATA)/train.jsonl --model Qwen/Qwen3-14B

unified-train: ## One Qwen3-14B adapter for all three tracks
	mlx_lm.lora --config configs/qwen3-14b-unified.yaml

unified-check-lora: ## Verify the unified adapter covers MLP projections
	$(PYTHON) scripts/check_lora.py --model $(UNIFIED_MODEL) --num-layers 24 --rank 16 --scale 2

sweep: ## Generate a small rank/LR/layer sweep
	$(PYTHON) scripts/make_sweep.py --base configs/qwen3-14b-all-layers.yaml \
	  --out configs/sweeps/r1 --rank 16,32 --lr 5e-5,1e-4 --layers all,16

eval: ## Evaluate the adapter against the golden set
	$(PYTHON) scripts/eval_toolcall.py --base-url http://127.0.0.1:8080/v1 \
	  --golden data/golden/golden.jsonl --out reports/eval.json

fuse: ## Fuse the adapter into a 4-bit MLX model
	mlx_lm.fuse --model $(MODEL) --adapter-path $(ADAPTER) --save-path $(FUSED)

serve: ## Serve the fused model with MLX (loopback only)
	mlx_lm.server --model $(FUSED) --host 127.0.0.1 --port 8080 \
	  --max-tokens 2048 --temp 0.2

serve-llama: ## Serve a GGUF with llama.cpp (best tool fidelity)
	llama-server -m models/qwen3-14b-pentest-Q4_K_M.gguf \
	  --host 127.0.0.1 --port 8080 -c 16384 -ngl 99 -fa auto --jinja

tunnel: ## Print the SSH tunnel command to run from Kali
	@echo "ssh -N -f -L 8080:127.0.0.1:8080 $$USER@$$(ipconfig getifaddr en0)"

docs-install: ## Install the docs toolchain (Material for MkDocs)
	uv pip install -r requirements-docs.txt

docs: ## Build the GitBook-style docs site
	$(MKDOCS) build --strict

docs-serve: ## Serve the docs locally with live reload
	$(MKDOCS) serve --dev-addr 127.0.0.1:8000

clean: ## Remove local build artefacts
	rm -rf fused mlx_models __pycache__ .pytest_cache site
