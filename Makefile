SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
MODEL ?= mlx-community/Qwen3-14B-4bit
ADAPTER ?= adapters/qwen3-14b-all-layers
FUSED ?= fused/qwen3-14b-pentest-4bit
DATA ?= data/processed
TOOLS ?= examples/tools/kali-tools.json
MKDOCS ?= $(PYTHON) -m mkdocs

.PHONY: help setup check data-check stats train-8b train-8b-all train-14b train-14b-last16 \
        check-lora sweep eval fuse serve serve-llama tunnel docs-install docs docs-serve clean

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
