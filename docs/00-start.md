# Start here

**Pick the shortest path to a working adapter.** The handbook has three primary specializations and one alternative model family. Each track has the same shape: choose a base model, build or find data, train a small adapter, evaluate against a real verifier, then serve it behind guardrails.

<div class="gb-track-grid gb-track-grid--compact">
  <a class="gb-track gb-track--kali" href="../01-landscape/">
    <span class="gb-pill">Agent</span>
    <h3>Kali &amp; network security</h3>
    <p>Qwen3-14B, tool calling, scope policy and an execution-verified Kali loop.</p>
    <span class="gb-track__cta">Start the track</span>
  </a>
  <a class="gb-track gb-track--java" href="../18-java-quarkus/">
    <span class="gb-pill">Code</span>
    <h3>Java &amp; Quarkus</h3>
    <p>Java 25, Quarkus 3.40, SOLID and hexagonal architecture, verified with Maven.</p>
    <span class="gb-track__cta">Start the track</span>
  </a>
  <a class="gb-track gb-track--react" href="../22-react-tailwind/">
    <span class="gb-pill">UI</span>
    <h3>React &amp; Tailwind</h3>
    <p>React 19.3, Tailwind 4.3, accessible design systems, checked in a browser.</p>
    <span class="gb-track__cta">Start the track</span>
  </a>
</div>

---

## Pick your path

=== "Kali / agent"

    **Base:** Qwen3-14B 4-bit QLoRA, with Qwen3-8B for the smoke run.

    **Data:** 2–5M tokens of general tool calling, multi-turn agentic trajectories, Kali CLI data, execution-verified synthetic trajectories, and refusal/scope examples.

    **Train:** [08 — Fine-tuning](08-finetuning.md) and the corrected [17 — 2026 recipe](17-finetune-2026-update.md).

    **Evaluate:** BFCL, KaliBench (held-out), Cybench and a private 240-task golden set. Gate on unsafe-action rate.

    **Serve:** `llama-server` for tool fidelity, `mlx_lm.server` for speed, behind an SSH tunnel.

    **Guardrails:** scope file, allowlist, confirmation, audit log, kill switch. Start at [11 — Kali agent](11-kali-agent.md).

=== "Java / Quarkus"

    **Base:** Qwen2.5-Coder-14B-Instruct 4-bit QLoRA, 24 of 48 blocks, sequence 2,048.

    **Target:** Java 25 LTS + Quarkus 3.40 LTS. Teach hexagonal architecture, DDD boundaries and SOLID.

    **Data:** code, repository tasks, commit-derived repairs, tests and ArchUnit rules; verify with Maven.

    **Train:** [20 — Java fine-tuning](20-java-finetuning.md).

    **Evaluate:** compile rate, test pass rate, ArchUnit violations, review score and patch size.

    **Synthetic data:** [21 — Java synthetic](21-java-synthetic.md).

=== "React / Tailwind"

    **Base:** Qwen2.5-Coder-14B-Instruct 4-bit QLoRA at 3,072 context.

    **Target:** React 19.3 + Tailwind CSS 4.3 + Vite 8 + Node 24 LTS, with React Router 8 and TanStack Query 5.

    **Data:** components, pages, tests, refactors, accessibility fixes and design-to-code tasks.

    **Train:** [24 — React fine-tuning](24-react-finetuning.md).

    **Evaluate:** TypeScript, Vitest, Playwright, axe and a visual rubric.

    **Synthetic data:** [25 — React synthetic](25-react-synthetic.md).

=== "Gemma 4 12B"

    **Base:** Gemma 4 12B 4-bit QLoRA, 24 of 48 blocks.

    **Why:** Apache-2.0, 256K context, native vision/audio input and a credible React/vision base.

    **Trade-off:** Qwen3 has the stronger documented tool-call track record; test Gemma on your own Kali harness before switching.

    **Start:** [27 — Gemma 4 12B](27-gemma4-12b.md), then compare with [31 — Gemma vs Qwen](31-gemma-vs-qwen.md).

---

## The short answer

| Question | Recommendation |
|---|---|
| **What do I train locally?** | **Qwen3-14B**, 4-bit QLoRA with MLX-LM. Largest Qwen that trains reliably in 32 GB, mature tool calling, 32K native context. |
| **Safe fallback** | **Qwen3-8B**, 4-bit QLoRA. Faster iteration, same tool format. |
| **If you rent a GPU** | **Qwen3.5-9B**, or Qwen3.8-27B on 48–80 GB. Much stronger agentic base, 262K context. |
| **Teacher / judge** | Qwen3.8-27B-4bit for quality; Qwen3.6-35B-A3B-4bit for bulk generation. Do not train either on 32 GB. |
| **Training framework** | **MLX-LM** (`mlx_lm.lora`), native Apple Silicon, quantized LoRA. |
| **2026 LoRA recipe** | All layers, rank 16, `scale: 2.0` (= PEFT alpha 32), LR `1e-4`, effective batch 8, 1–2 epochs. |
| **Java track** | Qwen2.5-Coder-14B, 24/48 blocks, Java 25 + Quarkus 3.40. |
| **React track** | Qwen2.5-Coder-14B, 24/48 blocks, sequence 3K, React 19.3 + Tailwind 4.3. |
| **One Qwen3 for all** | Qwen3-14B + three adapters. One mixed adapter is possible; see [26 — One model for all](26-one-model-for-all.md). |
| **Gemma 4 12B** | Separate adapters, separate ports, same evaluation harness. See [31 — Gemma vs Qwen](31-gemma-vs-qwen.md). |
| **Dataset size** | 2–5M tokens for a first adapter; 10M+ only if evaluation proves it helps. |
| **Serving** | `llama-server` for tool fidelity, `mlx_lm.server` for speed, Ollama for convenience. Loopback + SSH only. |
| **Agent harness** | OpenCode + MCP-Kali-Server, or hackingBuddyGPT for a minimal baseline. |
| **Evaluation** | BFCL, KaliBench, Cybench, private golden set, unsafe-action gate. |

---

## Full chapter map

| If you want to… | Read |
|---|---|
| Understand which model to train and why | [01 — Landscape](01-landscape.md) |
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
| Understand the mathematics | [32 — Mathematics](32-mathematics.md) |
| Map math to real tensor shapes | [37 — Architecture anatomy](37-architecture.md) |
| Read the proofs | [38 — Derivations and proofs](38-derivations.md) |
| See how Qwen was built | [33 — How Qwen was built](33-qwen-creation.md) |
| See how Gemma 4 was built | [34 — How Gemma 4 was built](34-gemma4-creation.md) |
| Train a model from zero | [35 — From zero](35-from-zero.md) |
| Reproduce a 12B–14B | [36 — Reproducing 12B–14B](36-reproducing-12b-14b.md) |

---

## Before you spend a day training

- [ ] Read [02 — Hardware](02-hardware.md) and run `scripts/check_env.py`.
- [ ] Render one dataset row with the exact chat template ([06 — Data format](06-data-format.md)).
- [ ] Run `scripts/check_lora.py` and confirm attention **and** MLP projections are adapted.
- [ ] Do a 1-iteration or 40-iteration smoke run before the multi-day run.
- [ ] Freeze a golden set before training; do not evaluate on rows you trained on.
- [ ] Put the model behind a policy layer; a model is not authorization.

---

## Repository map

```text
README.md
docs/            this handbook
configs/         MLX YAML configs
scripts/         env, data, training, eval, serving helpers
examples/        tool schemas, system prompts, scope files, lab compose
data/            dataset layout and tiny examples
```

License: Apache-2.0. Model and dataset licenses are separate; see [12 — Safety & legal](12-safety-legal.md).
