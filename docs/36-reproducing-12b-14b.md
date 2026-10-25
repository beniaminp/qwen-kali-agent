# 36 — Reproducing a 12B–14B model from scratch

*Last verified: 25 October 2026. This chapter is a planning document, not a claim that a 12B run is practical for an individual. It separates the architecture (reproducible) from the data and compute (not reproducible at home).*

The previous chapter built a 100M–1B model. This one scales the same recipe to the size of **Qwen3-14B**, **Qwen2.5-Coder-14B** and **Gemma 4 12B**. The goal is not to clone a checkpoint; it is to understand what a real 12B–14B pretraining project requires, and to choose the cheapest path to your actual goal.

---

## 1. The three target configurations

| | Qwen3-14B | Qwen2.5-Coder-14B | Gemma 4 12B |
|---|---:|---:|---:|
| Parameters | 14.8B | 14.7B | 11.95B |
| Layers | 40 | 48 | 48 |
| \(d\) | 5,120 | 5,120 | 3,840 |
| FFN | 17,408 | 13,824 | 15,360 |
| Heads / KV | 40 / 8 | 40 / 8 | 16 / 8 |
| Head dim | 128 | 128 | 256 (global 512) |
| Vocab | 151,669 | 151,646 | 262,144 |
| Tied embeddings | No | No | Yes |
| Context | 32K train / 128K YaRN | 128K | 256K |
| Norm | RMSNorm + QK-norm | RMSNorm | RMSNorm pre+post + QK-norm |
| MLP | SwiGLU | SwiGLU | GeGLU |
| Attention | full, GQA | full, GQA | 5:1 local/global, K=V, p-RoPE |

These are different models. “12B–14B” is a parameter band, not an architecture.

### 1.1 Parameter count

For a dense decoder:

\[
P \approx L\left(d^2 + 2dG d_h + 3d d_{\text{ff}}\right)
+ (1+\mathbf{1}[\text{untied}])\,Vd.
\]

- Qwen3-14B: \(P\approx14.77\text{B}\).
- Qwen2.5-Coder-14B: \(P\approx14.7\text{B}\).
- Gemma 4 12B: \(P\approx11.8\text{B}\) by this formula; official 11.95B.

Use `scripts/estimate_pretraining.py` to check your own config.

---

## 2. Compute budget

### 2.1 The formula

\[
C \approx 6ND,
\qquad
\text{GPU-hours} = \frac{C}{\text{peak FLOP/s}\times\text{MFU}\times3600}.
\]

For Qwen3-14B with \(D=36\text{T}\):

\[
C = 6 \times 14.77\times10^9 \times 36\times10^{12}
\approx 3.2\times10^{24}\ \text{FLOPs}.
\]

At 40% MFU on 10,000 H100-class GPUs (989 TFLOP/s peak):

\[
\text{GPU-hours}\approx2.24\times10^6,
\qquad
\text{wall time}\approx9.3\ \text{days},
\qquad
\text{cost}\approx\$5.6\text{M}
\]

at $2.50/GPU-hour, before failures, data, evaluation and staff.

### 2.2 What changes the number

- **MFU**: 25% vs 45% is nearly a 2× cost difference.
- **Sequence length**: long-context attention adds \(O(LT^2d)\) and lowers MFU.
- **Model FLOPs**: MoE uses active parameters for compute but stores all experts.
- **Failures**: add 20–50% for restarts, checkpoints and stragglers.
- **Data**: pipeline compute and storage are separate.
- **Post-training**: 5–20% additional compute, more for RL.

### 2.3 Chinchilla vs production

Chinchilla-optimal for 14B is roughly

\[
D\approx20N\approx280\text{B tokens}.
\]

Qwen3-14B uses **36T**, ~129× more. That is not a waste: a model served billions of times should be trained far past compute-optimal because inference cost dominates. For a one-off research model, 20–100 tokens/parameter is a saner starting point; for a product, overtrain.

---

## 3. Data: the hard part

A 14B model at 36T tokens is a data project first.

### 3.1 Volume

- 36T tokens at ~4 bytes/token in uint32 shards is ~144 TB before compression.
- Raw text, dedup indexes, quality scores and checkpoints add more.
- You need a storage cluster, not a laptop.

### 3.2 Pipeline

```text
crawl / licensed corpora
  → extraction (HTML, PDF, code, books)
  → language ID
  → normalization
  → exact dedup (hash)
  → near-dedup (MinHash/SimHash)
  → quality filter (classifier + heuristics)
  → toxicity/PII filter
  → benchmark decontamination
  → domain/mixture tagging
  → tokenize → shard → shuffle
```

### 3.3 Mixture

Qwen3 discloses domains but not weights: web, code, STEM, reasoning, books, multilingual, synthetic. Qwen2.5-Coder discloses a **70% code / 20% text / 10% math** final mixture. Gemma 4 discloses web/code/images/audio and a Jan 2025 cutoff, not weights.

For a 14B general model, a reasonable starting mixture is:

| Domain | Share |
|---|---:|
| Web (filtered) | 55–65% |
| Code | 15–20% |
| Math/STEM | 10–15% |
| Books/reference | 5–10% |
| Multilingual | 5–10% |
| Synthetic/instruction | 2–5% |

For a code model, invert toward code; for a domain model, add 10–30% domain data.

### 3.4 Curriculum

A three-stage curriculum is the public Qwen3 pattern:

1. **General** — most tokens, 4K context.
2. **Reasoning/STEM/code** — higher-quality subset, faster LR decay.
3. **Long context** — 32K sequences, hundreds of billions of tokens, RoPE base/ABF/YaRN.

Gemma 4 follows a Gemma-3-like pretraining recipe with a multimodal mixture and long-context design in the architecture. The exact curriculum is not public.

### 3.5 Data quality beats quantity

At 14B scale, a 10% quality improvement is worth more than a 10% token increase. Budget **50% of the project** for data: collection, cleaning, dedup, decontamination, mixture experiments and lineage.

---

## 4. Infrastructure

### 4.1 Hardware

A 14B dense pretraining run needs:

- 64–1,024 GPUs for a serious run; 1,000+ for a frontier token count;
- high-bandwidth interconnect (NVLink/InfiniBand);
- fast shared storage (parallel filesystem);
- checkpoint storage;
- a scheduler with preemption and restart.

A 12B/14B model does **not** fit on one 80 GB GPU for full training: weights (28 GB bf16) + optimizer (168 GB fp32 Adam) + gradients (28 GB) + activations exceed it. You need sharding.

### 4.2 Parallelism

| Technique | What it shards | When |
|---|---|---|
| **Data parallel (DP)** | batch | always |
| **FSDP / ZeRO-1/2/3** | optimizer, gradients, parameters | dense models |
| **Tensor parallel (TP)** | matrix dimensions | within a node, high bandwidth |
| **Pipeline parallel (PP)** | layers | across nodes |
| **Sequence/context parallel** | sequence | long context |
| **Expert parallel (EP)** | experts | MoE |

Typical 3D setup: TP within a node, PP across nodes, DP/FSDP across the rest. MoE adds all-to-all communication for expert routing, which is often the bottleneck.

### 4.3 Precision

- bf16 weights/activations for training;
- fp32 master weights and Adam states;
- fp32 layernorm/softmax where needed;
- gradient accumulation to reach the global batch;
- activation checkpointing for memory;
- FlashAttention for long sequences.

### 4.4 Throughput

Measure **tokens/sec/GPU** on a short run, then extrapolate:

\[
\text{wall time} =
\frac{D}{\text{tokens/sec/GPU}\times\text{GPUs}}.
\]

Do not trust vendor FLOPS alone. A 14B model at 2K context might do 3,000–8,000 tokens/sec/GPU on modern hardware depending on MFU and parallelism; at 32K it is much lower.

---

## 5. Stability

Large runs fail for boring reasons:

- loss spikes from bad data or high LR;
- attention logit explosion (QK-norm helps);
- MoE expert collapse;
- numerical overflow in fp16 (use bf16);
- checkpoint corruption;
- data loader imbalance;
- network stragglers.

Mitigations:

- QK-norm (Qwen3, Gemma 4);
- pre-norm residual;
- gradient clipping 1.0;
- LR warmup 1–2%;
- z-loss on logits;
- skip/yield bad batches;
- rollback to the last good checkpoint;
- monitor grad norm, loss scale, expert load, throughput;
- deterministic restart with the same data order.

A 14B run without a rollback plan is a gamble.

---

## 6. Post-training

Pretraining gives a base model. The assistant behaviour comes from post-training.

### 6.1 SFT

- 100k–1M+ instruction/response pairs (Qwen2.5 disclosed 1M+).
- Code, math, general, multilingual, tool use, safety.
- Mask the prompt; train the completion.
- Pack sequences for efficiency; watch cross-example contamination.

### 6.2 Reasoning cold start

- math/code/logic problems with verified answers or tests;
- filter unverifiable and too-easy queries;
- long-CoT traces in the target format.

### 6.3 RL

- **GRPO** with rule-based verifiers for math/code;
- **general RL** with rule, model and human rewards;
- **agentic RL** with tools and environment feedback;
- KL to a reference policy, entropy control, large batch.

### 6.4 Distillation

For smaller models, distill the flagship:

\[
\mathcal{L}_{\text{KD}} =
\operatorname{KL}\left(p_{\text{teacher}}\|p_{\text{student}}\right)
\]

on teacher or on-policy student samples. Qwen3 reports distillation at ~1/10 the GPU hours of the four-stage RL recipe for small models.

### 6.5 Safety

- red-team before release;
- refusal/scope data;
- PII and toxicity filters;
- evaluation cards;
- usage policy and license.

Post-training is 5–20% of compute but most of the product quality.

---

## 7. Team and timeline

A realistic 12B–14B from-scratch project:

| Role | Count |
|---|---:|
| Data engineering | 5–15 |
| Training/infra | 5–15 |
| Research/architecture | 3–8 |
| Evaluation | 3–8 |
| Safety/legal | 2–5 |
| Product | 2–4 |

Timeline: **6–12 months** for a first competitive model, longer for frontier quality. Data work starts first and never stops.

A single skilled engineer can reproduce the **architecture** and a **small** model; they cannot reproduce the data pipeline and compute of a 14B frontier run alone.

---

## 8. Alternatives, in order of cost

| Path | Cost | Quality | When |
|---|---|---|---|
| **LoRA/QLoRA on released weights** | $0–100 local | high for a domain | this handbook |
| **Continued pretraining** | $1k–100k | medium-high | new domain/language |
| **Distillation** | $1k–50k | good for narrow tasks | teacher available |
| **Model merging** | ~$0 | modest | two compatible adapters |
| **From scratch 1B** | $1k–10k | low general, good learning | research/education |
| **From scratch 14B** | $5M+ | frontier-ish | company-scale |

“Similar to Qwen/Gemma” almost always means **fine-tune the released model**, not retrain it. The architecture knowledge is useful for debugging, data design and small-scale research; it is not a shortcut around compute.

---

## 9. Legal and ethical constraints

- Dataset licenses: ODC-BY, CC-BY, MIT, Apache-2.0, `other`.
- Code licenses: per-file; keep provenance.
- Web data: copyright, robots, terms of service.
- PII and GDPR: lawful basis, deletion, minimization.
- Model licenses: Apache-2.0 (Qwen, Gemma 4) permits commercial use; dataset licenses may not.
- Safety: dual-use, cyber, bio, chemical; publish with care.
- Attribution and model cards.

A model is a derivative of its data. “Open weights” does not mean “any data is fine”.

---

## 10. Checklist

Before committing:

- [ ] Why from scratch? (new tokenizer/architecture/language/license)
- [ ] Compute budget and MFU measured.
- [ ] Data volume, licenses, dedup and decontamination.
- [ ] Tokenizer trained and compression measured.
- [ ] Architecture config and parameter count verified.
- [ ] Parallelism plan and interconnect.
- [ ] Checkpoint/restart plan.
- [ ] Evaluation suite and contamination checks.
- [ ] Post-training plan and safety.
- [ ] Team and 6–12 month timeline.
- [ ] Cheaper alternative considered (LoRA/continued pretraining/distillation).

If most boxes are empty, do not start a 14B run. Use [29 — Gemma fine-tuning](29-gemma4-finetuning.md) or [24 — React fine-tuning](24-react-finetuning.md) instead.

---

## 11. Sources

- Qwen3 technical report — <https://arxiv.org/abs/2505.09388>
- Qwen2.5 technical report — <https://arxiv.org/abs/2412.15115>
- Qwen2.5-Coder technical report — <https://arxiv.org/abs/2409.12186>
- Gemma 4 technical report — <https://arxiv.org/abs/2607.02770>
- Chinchilla — <https://arxiv.org/abs/2203.15556>
- Megatron-LM — <https://arxiv.org/abs/1909.08053>
- ZeRO — <https://arxiv.org/abs/1910.02054>
- FlashAttention — <https://arxiv.org/abs/2205.14135>
- Gated DeltaNet — <https://arxiv.org/abs/2412.06464>
- [32 — Mathematics](32-mathematics.md)
- [33 — How Qwen was built](33-qwen-creation.md)
- [34 — How Gemma 4 was built](34-gemma4-creation.md)
- [35 — From zero](35-from-zero.md)
