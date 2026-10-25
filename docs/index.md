# Qwen Agent Handbook

**Fine-tuning a Qwen model into an agentic Kali Linux assistant on a 32 GB M1 Pro — with the memory math, datasets, training runbook, evaluation, serving and safety controls.**

This site is the long-form handbook. The repository root has the same material in [`README.md`](https://github.com/beniaminp/qwen-kali-agent/blob/main/README.md), plus configs, scripts and examples.

<div class="gb-cards">
  <a class="gb-card" href="01-landscape/"><strong>Choose the model</strong><span>Qwen3-14B locally, Qwen3-8B as the fallback, and why Qwen3.5 is a GPU job.</span></a>
  <a class="gb-card" href="08-finetuning/"><strong>Train the adapter</strong><span>MLX QLoRA runbook: flags, memory, checkpoints, loss reading.</span></a>
  <a class="gb-card" href="17-finetune-2026-update/"><strong>Use the 2026 recipe</strong><span>All-layer LoRA, the MLX scale trap, one epoch, new configs.</span></a>
  <a class="gb-card" href="04-data-strategy/"><strong>Build the data</strong><span>Tool-calling mix, agentic trajectories, negative examples, synthetic data.</span></a>
  <a class="gb-card" href="10-export-serving/"><strong>Serve it</strong><span>MLX, GGUF, llama.cpp, Ollama, and the SSH tunnel to Kali.</span></a>
  <a class="gb-card" href="11-kali-agent/"><strong>Wire up Kali</strong><span>MCP tools, scope policy, audit logs, and the lab loop.</span></a>
  <a class="gb-card" href="18-java-quarkus/"><strong>Java &amp; Quarkus</strong><span>Java 25, Quarkus 3.40, hexagonal architecture and SOLID.</span></a>
  <a class="gb-card" href="20-java-finetuning/"><strong>Fine-tune a coder</strong><span>Qwen2.5-Coder on Java/Quarkus data, evaluated with Maven and ArchUnit.</span></a>
  <a class="gb-card" href="22-react-tailwind/"><strong>React &amp; Tailwind</strong><span>React 19, Tailwind 4.3, accessible design systems and Vite 8.</span></a>
  <a class="gb-card" href="24-react-finetuning/"><strong>Fine-tune a UI coder</strong><span>Qwen2.5-Coder on React data, verified by TypeScript, Playwright and axe.</span></a>
  <a class="gb-card" href="26-one-model-for-all/"><strong>One Qwen3 for all</strong><span>Qwen3-14B + three adapters, or one mixed adapter for Kali, Java and React.</span></a>
  <a class="gb-card" href="27-gemma4-12b/"><strong>Gemma 4 12B track</strong><span>Apache-2.0, 256K context, vision, native tool calls and a separate MLX runbook.</span></a>
  <a class="gb-card" href="31-gemma-vs-qwen/"><strong>Compare Gemma vs Qwen</strong><span>Keep both stacks, same prompts and tools, and decide with the real verifier.</span></a>
</div>


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
| Go beyond SFT | [15 — Agentic methods](15-agentic-methods.md) |
| Test networks, not just web apps | [16 — Network security](16-network-security.md) |
| Fix the LoRA recipe | [17 — Fine-tuning update](17-finetune-2026-update.md) |
| Specialize in Java/Quarkus | [18 — Java landscape](18-java-quarkus.md) |
| Find Java datasets | [19 — Java datasets](19-java-datasets.md) |
| Fine-tune a Java coder | [20 — Java fine-tuning](20-java-finetuning.md) |
| Generate Java data with an LLM | [21 — Java synthetic data](21-java-synthetic.md) |
| Specialize in React/Tailwind | [22 — React & Tailwind](22-react-tailwind.md) |
| Find React datasets | [23 — React datasets](23-react-datasets.md) |
| Fine-tune a UI coder | [24 — React fine-tuning](24-react-finetuning.md) |
| Generate React data with an LLM | [25 — React synthetic data](25-react-synthetic.md) |
| Use one Qwen3 for all tracks | [26 — One Qwen3 for all](26-one-model-for-all.md) |
| Try Gemma 4 12B | [27 — Gemma 4 12B](27-gemma4-12b.md) |
| Fine-tune Gemma | [29 — Gemma fine-tuning](29-gemma4-finetuning.md) |
| Compare Gemma and Qwen | [31 — Gemma vs Qwen](31-gemma-vs-qwen.md) |

---

## The one-paragraph answer

**Three primary tracks.**

**Kali/agent:** train **Qwen3-14B** with 4-bit QLoRA using **MLX-LM** on the M1 Pro. Use **Qwen3-8B** for the first smoke run. Build a **2–5M-token** mix of general tool calling, multi-turn agentic trajectories, Kali CLI data, execution-verified synthetic trajectories, and refusal/scope examples. Evaluate on **BFCL**, **KaliBench** (held-out test), **Cybench** and a private **240-task golden set**. Serve with **`llama-server`** for the best tool fidelity, or **`mlx_lm.server`** for speed, behind an **SSH tunnel** to the Kali VM. Put a deterministic **policy layer** between the model and the lab: scope file, allowlist, confirmation, audit log, kill switch. Publish only if the unsafe-action rate is zero and the licenses are clean.

**Java/Quarkus:** train **Qwen2.5-Coder-14B-Instruct** with 4-bit QLoRA, adapting 24 of 48 blocks. Target **Java 25 LTS + Quarkus 3.40 LTS**, teach hexagonal architecture and SOLID, and verify every example with Maven tests and ArchUnit. Qwen3-Coder-30B-A3B is the teacher/judge, not a local training target. Start at [18 — Java landscape](18-java-quarkus.md).

**React/Tailwind:** train **Qwen2.5-Coder-14B-Instruct** with 4-bit QLoRA at 3K context. Target **React 19.3 + Tailwind CSS 4.3 + Vite 8 + Node 24 LTS**, teach component architecture, design tokens and accessibility, and verify every example with TypeScript, Vitest, Playwright and axe. Start at [22 — React & Tailwind](22-react-tailwind.md).

**Gemma 4 12B (optional second stack):** Apache-2.0, 11.95B dense, 256K context, text/image/audio, native tool calling. It is a strong React/vision base and a credible Java/Kali challenger. Train separate adapters and compare with Qwen on the same golden set; start at [27 — Gemma 4 12B](27-gemma4-12b.md).

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
