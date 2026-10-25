# Qwen Kali Agent

**A practical, research-backed handbook for turning a Qwen open-weight model into an agentic Kali Linux / network-security assistant — trained locally on a 32 GB Apple M1 Pro, evaluated honestly, and served to an isolated Kali VM.**

This repository is documentation first, with runnable configs and scripts second. It is written to be useful months from now: every recommendation includes the reason, the memory math, the failure modes, and the primary sources.

**Live site:** <https://beniaminp.github.io/qwen-kali-agent/>

> **Status (25 October 2026).** Research, commands and links were checked against upstream documentation and source on 25 Oct 2026. The MLX training runs were **not** executed on the target Mac in the environment where this handbook was assembled, so treat every command as *verified but not hardware-tested*. Run the preflight in [`docs/03-macos-setup.md`](docs/03-macos-setup.md) before committing to a multi-day run.

> **Authorized testing only.** Everything here is for systems you own or have explicit written permission to test. The agent patterns in this repo are deliberately constrained by scope files, target allowlists, command confirmation and audit logs. See [`docs/12-safety-legal.md`](docs/12-safety-legal.md).

---

## The short answer

| Question | Recommendation |
|---|---|
| **What do I train locally?** | **Qwen3-14B**, 4-bit QLoRA with MLX-LM. It is the largest Qwen that trains *reliably* in 32 GB today, has mature tool calling, and a 32K native context. |
| **What is the safe fallback?** | **Qwen3-8B**, 4-bit QLoRA. Faster iteration, lower memory, same tool format. Use it to debug the dataset before spending days on the 14B. |
| **What if I rent a GPU?** | **Qwen3.5-9B** (or Qwen3.8-27B if you rent 48–80 GB). Much stronger agentic base, 262K context. A 5–10M-token run costs roughly $5–30 on an L40S/A100. |
| **What do I use as a teacher/judge?** | **Qwen3.8-27B-4bit** (best quality that fits for inference on 32 GB) and **Qwen3.6-35B-A3B-4bit** (much faster bulk generation). Do not try to train either locally. |
| **Training framework** | **MLX-LM** (`mlx_lm.lora`). Native Apple Silicon, quantized LoRA (QLoRA), no CUDA. |
| **LoRA recipe (Oct 2026)** | **All layers**, rank 16, `scale: 2.0` (= PEFT alpha 32), LR `1e-4`, effective batch 8, one to two epochs. Verify with `scripts/check_lora.py`; details in [`docs/17-finetune-2026-update.md`](docs/17-finetune-2026-update.md). |
| **Dataset size** | **2–5M tokens** for a first serious adapter; 10M+ only if evaluation proves it helps. Quality and trajectory realism dominate raw row count. |
| **Serving** | **`llama-server`** (llama.cpp) for the best tool-call fidelity; **`mlx_lm.server`** for the fastest local iteration; **Ollama** for convenience. All are unauthenticated — put them behind loopback + SSH, never on an untrusted LAN. |
| **Agent harness** | **OpenCode + MCP-Kali-Server** (small, typed tool surface) or **hackingBuddyGPT** (minimal research baseline). Avoid mega-tool MCP servers with a 14B. |
| **Evaluation** | **BFCL** (tool-call syntax), **KaliBench** (Kali CLI), **Cybench** (end-to-end CTF), plus a **240-task private golden set** and an unsafe-action rate gate. |

**Why not Qwen3.5/3.6/3.8 locally?** They use a hybrid Gated DeltaNet (GDN) architecture whose training path in MLX-LM is still immature as of October 2026. Open issues report first-backward OOMs, Metal descriptor leaks and step time scaling with trained GDN layers on 32–36 GB machines; the Metal VJP PR was closed unmerged. Unsloth also advises against 4-bit QLoRA for Qwen3.5. Qwen3-14B uses standard attention and is the reliable local choice. The full evidence is in [`docs/01-landscape.md`](docs/01-landscape.md) and [`docs/02-hardware.md`](docs/02-hardware.md).

---

## Documentation site

The handbook is also a **GitBook-style MkDocs site** ([Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) with a custom GitBook theme, search, grouped navigation and edit links). **Live:** <https://beniaminp.github.io/qwen-kali-agent/>

```bash
# from the repo root, with the project venv active
uv pip install -r requirements-docs.txt
mkdocs serve                  # http://127.0.0.1:8000
mkdocs build --strict         # static site in site/
```

Or use `make docs-serve` / `make docs`. `site/` is generated and git-ignored.

**Publishing:** `.github/workflows/docs.yml` runs `mkdocs build --strict` and publishes to the `gh-pages` branch on every push to `main`. GitHub Pages serves that branch at the live URL above. To deploy elsewhere, point Cloudflare Pages, Netlify or any static host at `site/`.

---

## What this repo contains

```
.
├── README.md                  ← you are here
├── docs/
│   ├── 01-landscape.md        Qwen model landscape and what to train
│   ├── 02-hardware.md         M1 Pro memory math, throughput, limits
│   ├── 03-macos-setup.md      reproducible environment from zero
│   ├── 04-data-strategy.md    capability taxonomy, mix, curriculum
│   ├── 05-datasets.md         catalog of tool-use, agentic and cyber data
│   ├── 06-data-format.md      canonical JSONL, Qwen templates, masking
│   ├── 07-synthetic-data.md   DeepSeek + OpenCode + execution-verified data
│   ├── 08-finetuning.md       full MLX QLoRA runbook and hyperparameters
│   ├── 09-evaluation.md       benchmarks, golden set, gates, contamination
│   ├── 10-export-serving.md   fuse, GGUF, Ollama, MLX/llama.cpp servers
│   ├── 11-kali-agent.md       Kali VM, MCP, agent loop, guardrails
│   ├── 12-safety-legal.md     authorization, law, licenses, release
│   ├── 13-troubleshooting.md  symptom → cause → fix
│   ├── 14-references.md       annotated source list
│   ├── 15-agentic-methods.md  SFT, rejection sampling, DPO, RL
│   ├── 16-network-security.md asset inventory, TLS, segmentation, packet evidence
│   ├── 17-finetune-2026-update.md all-layer LoRA, scale/LR correction, new configs
│   ├── appendix-a-configs.md  every config, explained
│   ├── appendix-b-scripts.md  what each script does
│   └── appendix-c-runbook.md  day-by-day execution plan
├── configs/                   ready-to-copy MLX YAML configs
├── scripts/                   data, training, eval and serving helpers
├── examples/                  tools, system prompt, scope, MCP configs
└── data/                      dataset layout and tiny examples
```

---

## Quickstart

### 0. Read the two decisions first

1. **Model:** Qwen3-14B local, Qwen3-8B fallback. If you can rent a GPU for a day, Qwen3.5-9B is a better base.
2. **Data:** start with 20–50k curated examples (~2–5M tokens), not 500k. A first adapter should take hours, not weeks.

### 1. Set up the Mac

```bash
xcode-select --install
brew install uv python@3.12 git-lfs jq
git clone https://github.com/beniaminp/qwen-kali-agent.git qwen-kali-agent
cd qwen-kali-agent
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install "mlx-lm[train]==0.32.2" "huggingface-hub[cli]" datasets
python scripts/check_env.py
```

Full details: [`docs/03-macos-setup.md`](docs/03-macos-setup.md).

### 2. Put data in the expected shape

```text
data/
├── raw/                 # downloaded datasets (git-ignored)
├── interim/             # converted JSONL
├── processed/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
└── cards/dataset-card.md
```

Each line is an OpenAI-style chat example with an optional `tools` array:

```json
{"messages":[{"role":"user","content":"List the open ports on 10.10.10.5."},{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"nmap_scan","arguments":"{\"target\":\"10.10.10.5\",\"flags\":\"-sV\"}"}}]}],"tools":[{"type":"function","function":{"name":"nmap_scan","description":"Run an nmap scan against an authorized target.","parameters":{"type":"object","properties":{"target":{"type":"string"},"flags":{"type":"string"}},"required":["target"]}}}]}
```

Validate before training:

```bash
python scripts/validate_dataset.py data/processed/train.jsonl --tools examples/tools/kali-tools.json
python scripts/token_stats.py data/processed/train.jsonl --model Qwen/Qwen3-14B
```

Details: [`docs/06-data-format.md`](docs/06-data-format.md).

### 3. Train

```bash
# Fast iteration on 8B first (hours)
mlx_lm.lora --config configs/qwen3-8b-fast.yaml

# Verify the adapter really covers attention + MLP layers
python scripts/check_lora.py --model mlx-community/Qwen3-14B-4bit \
  --num-layers -1 --rank 16 --scale 2

# The real run (14B, 4-bit QLoRA, all layers)
mlx_lm.lora --config configs/qwen3-14b-all-layers.yaml
```

Monitor loss, tokens/sec and peak memory. Stop when validation loss stops improving. Full runbook: [`docs/08-finetuning.md`](docs/08-finetuning.md).

### 4. Fuse and serve

```bash
# Merge adapter into the 4-bit MLX model
mlx_lm.fuse \
  --model mlx-community/Qwen3-14B-4bit \
  --adapter-path adapters/qwen3-14b-all-layers \
  --save-path fused/qwen3-14b-pentest-4bit

# Serve to the Kali VM over loopback + SSH tunnel
mlx_lm.server \
  --model fused/qwen3-14b-pentest-4bit \
  --host 127.0.0.1 --port 8080 \
  --max-tokens 2048 \
  --temp 0.2
```

For a GGUF/Ollama deployment, use llama.cpp's converter — `mlx_lm.fuse --export-gguf` only supports Llama/Mistral/Mixtral model types, **not Qwen3**. See [`docs/10-export-serving.md`](docs/10-export-serving.md).

### 5. Wire up Kali

```bash
# On Kali, forward the Mac's model server
ssh -N -f -L 8080:127.0.0.1:8080 macuser@192.168.64.1

# Smoke test
curl -s http://127.0.0.1:8080/v1/models | jq .
```

Then install OpenCode + MCP-Kali-Server inside the VM. Architecture, configs and guardrails: [`docs/11-kali-agent.md`](docs/11-kali-agent.md).

---

## The data strategy in one table

| Bucket | Share | What it teaches | Example sources |
|---|---:|---|---|
| General tool calling | 35–40% | schema adherence, argument JSON, parallel calls | xLAM, ToolACE, Hermes, Toucan |
| Multi-turn / agentic | 15–20% | observe → think → act, recovery | ToolACE, SWE-smith, Nemotron Terminal, OpenThoughts |
| Cyber QA + Kali CLI | 20% | domain vocabulary, flags, procedures | KaliBench train, Trendyol, filtered cyber QA |
| Synthetic verified trajectories | 15% | *your* tool surface and environment | generated with DeepSeek/OpenCode, executed in a lab |
| Refusal / scope / no-call | 10% | authorization checks, ask vs act, abstain | When2Call, curated scope dialogues |

Do **not** train on BFCL, KaliBench test, Cybench, NYU CTF or CyberMetric. Hold them out. The full catalog, licenses and conversion notes are in [`docs/05-datasets.md`](docs/05-datasets.md).

---

## Hardware reality check

| Resource | M1 Pro 32 GB |
|---|---|
| GPU | 16-core, ~5.2 TFLOPS FP32, ~200 GB/s |
| Default GPU working set | ~21.3 GiB (macOS reserves the rest) |
| Qwen3-14B 4-bit weights | ~8.3 GB |
| Realistic QLoRA peak (seq 2K, batch 1, grad-checkpoint) | ~18–22 GB |
| Practical ceiling before macOS swaps/panics | ~24–26 GB wired |
| Qwen3-14B QLoRA throughput (estimate) | 20–30 tokens/s |
| 5M tokens at 25 tok/s | ~56 hours |
| 14B 4-bit inference | ~15–17 tok/s at 1–4K context |

Training a 14B for 5M tokens is a multi-day job on this machine. That is why the handbook insists on an 8B debugging run and a small, high-quality dataset first. The math and sources are in [`docs/02-hardware.md`](docs/02-hardware.md).

---

## Non-negotiable safety rules

1. **Written authorization and a scope file before any scan.** No authorization, no tool call.
2. **The model never gets a raw unrestricted shell against the internet.** It gets typed tools behind a policy layer, or a shell in a disposable lab VM with no route to public networks.
3. **Target allowlist, command denylist, timeout, audit log, kill switch.** All four, every run.
4. **No real credentials, customer data, or live flags in training data.** Scrub and review.
5. **Treat generated commands as hostile until executed in a disposable sandbox.** The synthetic-data pipeline is built around this assumption.

A model is not a penetration tester. It is a probabilistic planner that must be constrained by deterministic policy. [`docs/11-kali-agent.md`](docs/11-kali-agent.md) and [`docs/12-safety-legal.md`](docs/12-safety-legal.md) explain how.

---

## Research provenance

The handbook synthesizes primary sources: Qwen model cards, the MLX-LM source and issue tracker, Hugging Face dataset cards and dataset-server statistics, the DeepSeek API docs, OpenCode docs/source, and the papers behind BFCL, KaliBench, Cybench, ToolACE, APIGen, When2Call, xLAM and SWE-smith. Every chapter ends with its sources; [`docs/14-references.md`](docs/14-references.md) is the consolidated list.

Where a number is an estimate rather than a measurement, it is marked **EST**. Where a claim could not be verified, it is marked **UNVERIFIED**. Do not silently turn either into a fact.

## License

Documentation and code: Apache-2.0. Model weights and datasets keep their own licenses; see [`docs/12-safety-legal.md`](docs/12-safety-legal.md).
