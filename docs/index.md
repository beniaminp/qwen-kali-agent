# Qwen Kali Agent Handbook

**Fine-tuning a Qwen model into an agentic Kali Linux assistant on a 32 GB M1 Pro — with the memory math, datasets, training runbook, evaluation, serving and safety controls.**

This site is the long-form handbook. The repository root has the same material in [`README.md`](../README.md), plus configs, scripts and examples.

---

## Start here

| If you want to… | Read |
|---|---|
| Understand which Qwen model to train and why | [01 — Landscape](01-landscape.md) |
| Know whether it fits in 32 GB | [02 — Hardware](02-hardware.md) |
| Set up the Mac from zero | [03 — macOS setup](03-macos-setup.md) |
| Design the data mix | [04 — Data strategy](04-data-strategy.md) |
| Find datasets and licenses | [05 — Datasets](05-datasets.md) |
| Get the JSONL/template exactly right | [06 — Data format](06-data-format.md) |
| Generate synthetic trajectories safely | [07 — Synthetic data](07-synthetic-data.md) |
| Run the training | [08 — Fine-tuning](08-finetuning.md) |
| Prove it works | [09 — Evaluation](09-evaluation.md) |
| Export to GGUF/Ollama/MLX | [10 — Export & serving](10-export-serving.md) |
| Wire it to Kali with MCP | [11 — Kali agent](11-kali-agent.md) |
| Stay legal and in scope | [12 — Safety & legal](12-safety-legal.md) |
| Fix a failure | [13 — Troubleshooting](13-troubleshooting.md) |
| See the sources | [14 — References](14-references.md) |

---

## The one-paragraph answer

Train **Qwen3-14B** with 4-bit QLoRA using **MLX-LM** on the M1 Pro. Use **Qwen3-8B** for the first smoke run. Build a **2–5M-token** mix of general tool calling, multi-turn agentic trajectories, Kali CLI data, execution-verified synthetic trajectories, and refusal/scope examples. Evaluate on **BFCL**, **KaliBench** (held-out test), **Cybench** and a private **240-task golden set**. Serve with **`llama-server`** for the best tool fidelity, or **`mlx_lm.server`** for speed, behind an **SSH tunnel** to the Kali VM. Put a deterministic **policy layer** between the model and the lab: scope file, allowlist, confirmation, audit log, kill switch. Publish only if the unsafe-action rate is zero and the licenses are clean.

---

## Warnings

> **Authorized testing only.** A model is not authorization. Use a written scope, a lab you own, and a policy layer that enforces both.

> **This is a handbook, not a tested product.** Commands were verified against upstream source/docs on 25 Oct 2026, but not executed on the target Mac in the environment where the handbook was assembled. Run the preflight and a 1-iteration smoke test before a multi-day run.

---

## Repository map

```text
README.md
docs/            this handbook
configs/         MLX YAML configs
scripts/         env, data, training, eval, serving helpers
examples/        tool schemas, system prompt, scope, MCP config, lab compose
data/            dataset layout and tiny examples
```

License: Apache-2.0. Model and dataset licenses are separate; see [12 — Safety & legal](12-safety-legal.md).
