# 01 — The Qwen landscape and what to train

*Last verified: 25 October 2026.*

> This chapter is the Kali/agent track. If you are specialising in **Java/Quarkus** or **React/Tailwind**, the base model and data are different: start at [18 — Java landscape](18-java-quarkus.md) or [22 — React & Tailwind](22-react-tailwind.md).

The Qwen family moves fast. This chapter fixes the vocabulary, lays out the open-weight options as they exist today, and explains why the rest of the handbook trains **Qwen3-14B** locally rather than a newer, larger, or more fashionable model.

The short version: **the best model you can train is not the best model you can run.** On a 32 GB M1 Pro, the training ceiling is much lower than the inference ceiling, and the newest Qwen architectures are the *least* mature in MLX's training path.

---

## 1. Three generations, two architectures

As of October 2026 there are three relevant generations:

| Generation | Examples | Architecture | Training status on MLX |
|---|---|---|---|
| **Qwen3** (Apr 2025) | Qwen3-4B/8B/14B/32B, Qwen3-30B-A3B, Qwen3-Coder-30B-A3B | Standard transformer: GQA + dense FFN or MoE | **Mature.** The safest local fine-tuning target. |
| **Qwen3.5 / 3.6 / 3.8** (Feb–Aug 2026) | Qwen3.5-4B/9B, Qwen3.6-35B-A3B, Qwen3.8-27B | Hybrid **Gated DeltaNet (GDN)** + gated full attention; native vision; 262K context | **Immature for training.** Inference works; LoRA training is experimental on 32 GB. |
| **Qwen3-Coder-Next** (Jan 2026) | 80B total / 3B active | Hybrid GDN + 512-expert MoE | Inference only on this hardware; 4-bit weights alone are ~45 GB. |

The generational jump is real: Qwen3.5-9B beats Qwen3-30B-A3B on agentic benchmarks, and Qwen3.8-27B is a genuinely strong long-horizon agent. But the architecture that makes them efficient at inference (GDN, a linear-attention recurrence) is also the architecture whose backward pass is not yet optimized in MLX. That is the central trade-off of this handbook.

---

## 2. The open-weight field

All models below are **Apache-2.0** unless noted. Benchmark numbers are vendor-reported and are useful for relative ordering, not for predicting your task.

| Model | Released | Total / active | Architecture | Native context | Notes |
|---|---|---:|---|---:|---|
| [`Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) | 2025-04 | 4.0B | Dense, 36 layers | 32K (131K YaRN) | Fast format-iteration model; weaker reasoning. |
| [`Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | 2025-04 | 8.2B | Dense, 36 layers, GQA 32Q/8KV | 32K (131K YaRN) | **Safest training fallback.** BFCL-v3 68.1. |
| [`Qwen3-14B`](https://huggingface.co/Qwen/Qwen3-14B) | 2025-04 | 14.8B | Dense, 40 layers, GQA 40Q/8KV | 32K (131K YaRN) | **Recommended local training target.** BFCL-v3 70.4. |
| [`Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B) | 2025-04 | 32.8B | Dense, 64 layers | 32K (131K YaRN) | Inference-only at 4-bit on 32 GB; training does not fit. |
| [`Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 2025-04 | 30.5B / 3.3B | MoE, 128 experts, 8 active | 32K | Inference fits; training is marginal. |
| [`Qwen3-30B-A3B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | 2025-07 | 30.5B / 3.3B | MoE, 48 layers, non-thinking | 262K | Strong general/agentic MoE; 4-bit inference ~17–18 GB; no local LoRA. |
| [`Qwen3-Coder-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | 2025-07 | 30.5B / 3.3B | MoE, 48 layers, non-thinking | 262K | Strong CLI/coding specialist. 4-bit inference ~17 GB. |
| [`Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) | 2026-02 | 4.7B | Hybrid GDN, 32 layers, vision | 262K | Strong small agent; training conditional. |
| [`Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) | 2026-02 | 9.7B | Hybrid GDN, 32 layers, vision | 262K | **Best quality/time if you rent a GPU.** BFCL-v4 66.1, TAU2 79.1. |
| [`Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 2026-04 | 36.0B / ~3B | Hybrid GDN + MoE, 256 experts | 262K | Excellent agentic/coding; 4-bit inference 19–24 GB. |
| [`Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | 2026-08 | 27.8B | Hybrid GDN, 64 layers, vision | 262K | Best dense model that fits 4-bit inference; **teacher/oracle**, not a local training target. |
| [`Qwen3-Coder-Next`](https://huggingface.co/Qwen/Qwen3-Coder-Next) | 2026-01 | 79.7B / ~3B | Hybrid GDN + 512-expert MoE | 262K | Too large for 32 GB even at 4-bit (~45 GB). |

**Terminology.** “Total / active” matters for MoE models: Qwen3.6-35B-A3B has 36B parameters but activates ~3B per token, so it generates quickly while still needing ~20 GB of RAM at 4-bit. “Native context” is the trained window; YaRN extension is possible but costs quality and memory.

---

## 3. What actually fits

Training needs the weights **plus** activations, logits, adapter gradients and optimizer state. The table below is the practical conclusion; the derivation is in [`02-hardware.md`](02-hardware.md).

| Target | 4-bit weights | Realistic QLoRA peak | Verdict on 32 GB |
|---|---:|---:|---|
| Qwen3-8B | ~4.6 GB | 10–12 GB | **Yes.** Fast, forgiving. |
| Qwen3-14B | ~8.3 GB | 18–22 GB | **Yes.** The local sweet spot. |
| Qwen3-32B | ~18.5 GB | >32 GB | No. |
| Qwen3.5-4B | ~2.6 GB | 12–15 GB | Conditional; GDN backward is slow. |
| Qwen3.5-9B | ~5.4 GB | OOM reported at 36 GB | **No / high risk.** |
| Qwen3.6-35B-A3B | ~20.2 GB | 30–35 GB | No. |
| Qwen3.8-27B | ~15.6 GB | 32–60 GB | No. |
| Qwen3-Coder-30B-A3B | ~17.2 GB | ~28–32 GB | Marginal; treat as inference-only. |
| Qwen3-Coder-Next | ~44.8 GB | — | No. |

The Qwen3.5/3.8 numbers are not hypothetical. MLX-LM issues report first-backward OOMs for 9B on a 36 GB machine, and 27B peaks of 60–117 GB before a Metal resource-limit failure. The GDN path falls back to a Python recurrence during training because the Metal kernel has no vector-Jacobian product; issue #1956 (Oct 2026) shows step time scaling with the number of trained GDN layers, and the Metal VJP PR #1217 was closed unmerged. Unsloth also advises against 4-bit QLoRA for Qwen3.5 dense/MoE because of quantization sensitivity. See [`02-hardware.md`](02-hardware.md) §4 and [17 — Fine-tuning update](17-finetune-2026-update.md).

**The consequence:** if you want the newer architecture, rent a CUDA GPU for a day, train there, and bring the merged weights back for 4-bit MLX inference. That is a perfectly good workflow and often cheaper than spending a week fighting the local training path.

---

## 4. Why Qwen3-14B is the local default

1. **It trains.** Standard attention, mature LoRA path, thousands of successful runs. The same cannot be said for GDN models today.
2. **It is large enough to be useful.** 14.8B parameters and 32K context are enough for multi-step tool use, Kali CLI vocabulary, and structured output. It is not a frontier reasoner, but the job is pattern-following inside a scaffold, not open-ended exploitation research.
3. **It has native tool calling.** Qwen3 was post-trained with tool use; BFCL-v3 70.4 is competitive with much larger 2024 models.
4. **It leaves memory for the rest of the system.** At ~18–22 GB peak, macOS, the terminal, and a small Kali VM can coexist. A 27B run leaves nothing.
5. **It is cheap to iterate.** An 8B debugging run is ~2× faster; the 14B is the graduation run, not the first run.

**When to choose Qwen3-8B instead:** your dataset is still changing daily, you have less than 2M tokens, or you need a 4K–8K sequence length. The 8B is also the control in the MLX issue reports — if a training bug appears, test it on 8B first.

**When to choose Qwen3.5-9B instead:** you will rent an L40S/A100 (or have access to one), you want 262K context, and you care about the strongest possible base. The 9B is a better model; it is just not a better *local training* model today.

**When to choose Qwen3-Coder-30B-A3B:** your agent is mostly code/CLI generation and you will use it for inference, not training. It is excellent at terminal work, but 17 GB of weights plus activations leaves little room for a serious QLoRA run.

---

## 5. Teacher and oracle models

Even if you train a 14B, a larger local model is useful for:

- generating candidate trajectories and hard negatives,
- grading outputs against a rubric,
- adjudicating disagreements in evaluation,
- producing a “best-of-N” reference for failure analysis.

| Role | Model | 4-bit memory | Why |
|---|---|---:|---|
| Quality oracle / judge | `mlx-community/Qwen3.8-27B-4bit` | 16–23 GB | Best dense quality that fits; slow (~8–10 tok/s), so use it selectively. |
| Bulk generator | `mlx-community/Qwen3.6-35B-A3B-4bit` | 19–24 GB | 40–50 tok/s at short context; 3B active. Good for volume. |
| Coding/CLI specialist | `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` | 17–19 GB | Strong terminal/code behavior; non-thinking. |
| Cheap formatter | Qwen3-8B/14B | 5–9 GB | Fast, good enough for JSON repair and simple classification. |

Do not use a model to judge its own training data without a deterministic check. Execution results, schema validation and human spot checks come first; the LLM judge is a tie-breaker, not ground truth.

---

## 6. Base vs instruct vs thinking

- **Base** models are completion models. Do not fine-tune them for tool calling unless you have a very large, carefully formatted corpus and a reason.
- **Instruct** models already follow chat templates and emit tool calls. QLoRA on an instruct model is the practical choice.
- **Thinking mode.** Qwen3 supports a `reasoning_content` block; Qwen3.5/3.6 add `preserve_thinking` and reasoning-effort controls. Thinking improves hard multi-step tasks but doubles latency and token cost. For a pentest agent, use thinking for planning and disable it for routine tool selection, or train two adapters (fast/no-think and deliberate/think).
- **Template fidelity is more important than raw model quality.** A 14B that emits valid JSON every time beats a 27B that emits almost-JSON. Chapter [`06-data-format.md`](06-data-format.md) is the most important chapter in this repo for that reason.

---

## 7. Local vs rented GPU

The local Mac is superb for **iteration and inference** and mediocre for **long training runs**. Rough 2026 economics:

| Job | Local M1 Pro | Rent an L40S/A100 |
|---|---|---|
| 1M-token experiment | 15–25 hours, $0 | ~20–40 minutes, ~$1–3 |
| 5M-token adapter | 2–4 days | ~2–4 hours, ~$5–15 |
| 20M-token adapter | 8–15 days | ~8–15 hours, ~$15–50 |
| Privacy / air-gap | Excellent | Poor unless dedicated |
| Iteration speed | Slow but unlimited | Fast, metered |

**Rule of thumb:** if the run is longer than ~24 hours, rent. If the data is sensitive or you need to iterate on the pipeline itself, stay local. The handbook supports both: the same JSONL and the same evaluation harness work with MLX (local) or PEFT/Unsloth/TRL (CUDA).

---

## 8. A decision tree

```text
Do you have a CUDA GPU or a $20 budget?
├─ Yes → Is 262K context or the strongest base important?
│         ├─ Yes → Qwen3.5-9B (bf16 LoRA, not 4-bit QLoRA) on L40S/A100
│         └─ No  → Qwen3-14B QLoRA on the rented GPU (faster, same recipe)
└─ No  → Train locally on the M1 Pro
          ├─ First adapter, dataset still moving → Qwen3-8B QLoRA
          ├─ Serious adapter, stable data       → Qwen3-14B QLoRA
          └─ Need newer architecture?           → Rent, or wait for MLX GDN training support
```

---

## 9. Benchmarks are not your task

Vendor benchmark tables compare models on BFCL, TAU, SWE-bench and Terminal-Bench. Those numbers are useful for eliminating bad choices, not for predicting whether a model will run `nmap -sV` with the right flags on your lab network. A model can score well on BFCL and still fail your tool schemas because:

- your tool names/descriptions are ambiguous,
- your arguments use enums or nested objects the model has not seen,
- your system prompt conflicts with the tool instructions,
- your context contains 30 tools when the model can reliably choose among 8,
- the model was trained on a different tool-call syntax.

The only benchmark that matters is a held-out set of **your** tasks, executed in **your** lab, with **your** tools. Chapter [`09-evaluation.md`](09-evaluation.md) builds it.

---

## 10. Sources

- Qwen3 technical report — <https://arxiv.org/abs/2505.09388>
- Qwen3 model cards — <https://huggingface.co/Qwen/Qwen3-14B>, <https://huggingface.co/Qwen/Qwen3-8B>
- Qwen3.5-9B — <https://huggingface.co/Qwen/Qwen3.5-9B>
- Qwen3.6-35B-A3B — <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>
- Qwen3.8-27B — <https://huggingface.co/Qwen/Qwen3.8-27B>
- Qwen3-Coder-30B-A3B — <https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct>
- Qwen3-Coder-Next — <https://huggingface.co/Qwen/Qwen3-Coder-Next>
- MLX-LM GDN training issues — <https://github.com/ml-explore/mlx-lm/issues/1185>, <https://github.com/ml-explore/mlx-lm/issues/1206>, <https://github.com/ml-explore/mlx-lm/issues/1956>
- MLX-LM repository — <https://github.com/ml-explore/mlx-lm>
