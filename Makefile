SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
MODEL ?= mlx-community/Qwen3-14B-4bit
ADAPTER ?= adapters/qwen3-14b-pentest
FUSED ?= fused/qwen3-14b-pentest-4bit
DATA ?= data/processed
TOOLS ?= examples/tools/kali-tools.json

.PHONY: help setup check data-check stats train-8b train-14b eval fuse serve serve-llama tunnel clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create the Python environment on macOS
	uv venv --python 3.12 .venv
	uv pip install "mlx-lm[train]==0.32.2" "huggingface-hub[cli]" datasets openai pyyaml jsonschema
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

train-14b: ## Main 14B QLoRA run
	mlx_lm.lora --config configs/qwen3-14b-balanced.yaml

eval: ## Evaluate the adapter against the golden set
	$(PYTHON) scripts/eval_toolcall.py --base-url http://127.0.0.1:8080/v1 \
	  --golden data/golden/golden.jsonl --out reports/eval.json

fuse: ## Fuse the adapter into a 4-bit MLX model
	mlx_lm.fuse --model $(MODEL) --adapter-path $(ADAPTER) --save-path $(FUSED)

serve: ## Serve the fused model with MLX (loopback only)
	mlx_lm.server --model $(FUSED) --host 127.0.0.1 --port 8080 \
	  --max-tokens 2048 --chat-template-args '{"enable_thinking": false}'

serve-llama: ## Serve a GGUF with llama.cpp (best tool fidelity)
	llama-server -m models/qwen3-14b-pentest-Q4_K_M.gguf \
	  --host 127.0.0.1 --port 8080 -c 16384 -ngl 99 -fa auto --jinja

tunnel: ## Print the SSH tunnel command to run from Kali
	@echo "ssh -N -f -L 8080:127.0.0.1:8080 $$USER@$$(ipconfig getifaddr en0)"

clean: ## Remove local artefacts (keeps datasets and adapters)
	rm -rf fused mlx_models __pycache__ .pytest_cache
