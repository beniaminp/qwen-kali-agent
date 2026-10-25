# 34 — How Gemma 4 was built

*Last verified: 25 October 2026 against the Gemma 4 technical report (arXiv:2607.02770), the Google model cards, and the Hugging Face config/chat template.*

Gemma 4 is Google DeepMind’s open-weight, natively multimodal family. The public report is unusually detailed about architecture, tokenizer, training infrastructure and post-training. This chapter reconstructs the design and the mathematics; exact data mixtures and hyperparameters remain undisclosed.

> **Read with:** [37 — Architecture anatomy](37-architecture.md) for the encoder-free 12B tensor shapes and [38 — Derivations and proofs](38-derivations.md) for the RoPE/GQA/quantization math.

---

## 1. The family

| Model | Total params | Effective / active | Modalities | Encoders |
|---|---:|---:|---|---|
| **Gemma 4 E2B** | 5.1B with embeddings | 2.3B effective | text, image, audio | 150M ViT + 305M audio |
| **Gemma 4 E4B** | 8B with embeddings | 4.5B effective | text, image, audio | 150M ViT + 305M audio |
| **Gemma 4 12B** | **11.95B** | dense | text, image, audio | **none** (encoder-free) |
| **Gemma 4 26B-A4B** | 25.2B | 3.8B active | text, image | 550M ViT |
| **Gemma 4 31B** | 30.7B | dense | text, image | 550M ViT |

All are Apache-2.0. The 12B is the model used in this handbook because it trains in 32 GB at 4-bit and has native vision/audio input without separate encoders.

The “E” in E2B/E4B means **effective** parameters: per-layer embeddings (PLE) store large lookup tables that are not all used per token, so the compute-relevant size is smaller than the stored size.

---

## 2. The decoder architecture

Gemma 4 is a decoder-only transformer with:

- **pre-norm and post-norm RMSNorm**;
- **QK-Norm**;
- **GeGLU** MLP;
- **RoPE** on local layers, **p-RoPE** on global layers;
- hybrid **local sliding-window / global** attention;
- global layers reuse **keys as values** (K=V) in models above E4B;
- vocabulary **262,144**, SentencePiece with split digits, preserved whitespace, byte-level fallback.

The block is:

\[
\begin{aligned}
u &= x + \operatorname{Attn}(\operatorname{RMSNorm}_1(x)),\\
v &= u + \operatorname{MLP}(\operatorname{RMSNorm}_2(u)),\\
y &= \operatorname{RMSNorm}_{\text{post}}(v) \quad\text{(Gemma-style post-norm)}.
\end{aligned}
\]

The exact placement of post-norm is implementation detail; the point is that Gemma uses both normalisations, unlike Qwen3’s pure pre-norm.

---

## 3. Hybrid attention and p-RoPE

Gemma 4 uses a **5:1 local-to-global** ratio (4:1 for E2B):

- five layers of sliding-window attention with window \(W=1024\);
- one layer of full global attention.

Local attention:

\[
A_{ij}=0 \quad\text{if } j<i-W+1.
\]

Global attention is standard causal attention. Global layers use:

- **K=V**: one projection is reused for keys and values;
- **p-RoPE**: only a fraction \(p=0.25\) of dimensions are rotated;
- RoPE base \(10^6\) global, \(10^4\) local.

### 3.1 Why K=V

Standard attention stores both \(K\) and \(V\) in the cache. If \(K=V\):

\[
\text{KV cache} = L G d_h T b
\]

instead of \(2LGd_hTb\) for those layers. Combined with p-RoPE and the local/global split, Gemma reports up to a **37.5% reduction** in global KV cache.

### 3.2 p-RoPE

For head dimension \(d_h\) and fraction \(p\):

\[
\tilde q = \big[R_{\theta}(q_{1:pd_h}),\; q_{pd_h+1:d_h}\big],
\]

where \(R_\theta\) is the standard RoPE rotation. Only the first \(p d_h\) dimensions carry position; the rest are position-invariant. The global layers already integrate information over the full sequence, so rotating every dimension is unnecessary and can hurt long-context stability.

### 3.3 Cache math

For 12B: \(L=48\), \(G=8\), local \(d_h=256\), window 1024, global \(d_h=512\). Roughly:

\[
\text{local cache} \approx 40 \cdot 2 \cdot 8 \cdot 256 \cdot 1024 \cdot 2 \approx 0.34\text{ GB},
\]

\[
\text{global cache} \approx 8 \cdot 8 \cdot 512 \cdot T \cdot 2
\]

at bf16, before K=V. At \(T=32{,}768\), the global part dominates. K=V and int8 KV are what make 256K serving practical.

---

## 4. The encoder-free 12B

This is the distinctive architectural choice.

### 4.1 Vision

Instead of a separate ViT, 12B takes **48×48×3 RGB patches** and applies a single large matmul:

\[
z_{\text{patch}} = W_{\text{patch}}\,x_{\text{patch}} + b,
\qquad
W_{\text{patch}}\in\mathbb{R}^{d\times(48\cdot48\cdot3)}.
\]

The projection has ~35M parameters. Spatial information is added as **2D coordinate positional embeddings** directly to patch representations, followed by LayerNorm:

\[
\tilde z = \operatorname{LayerNorm}(z + e_{\text{2D}}).
\]

### 4.2 Audio

Raw audio is segmented into **40ms chunks at 16kHz**, giving 640-dimensional vectors:

\[
640 = 16{,}000 \times 0.04.
\]

Each chunk is projected directly into the LLM embedding space:

\[
z_{\text{audio}} = W_{\text{audio}}\,x_{\text{audio}} + b.
\]

Audio is temporal, so no extra positional encoding is added. The 305M conformer encoder used by E2B/E4B is discarded entirely in 12B.

### 4.3 Why it matters

- fewer parameters and less memory fragmentation;
- one decoder trained on all modalities;
- the whole stack can be fine-tuned in one pass;
- but the runtime must implement the `gemma4_unified` architecture, not an older Gemma loader.

For a text-only MLX LoRA, the vision/audio weights are stripped; the language model still trains normally.

---

## 5. Vision and audio encoders (other sizes)

| Model | Vision | Audio |
|---|---|---|
| E2B/E4B | 150M ViT, patch 16, variable aspect ratio, axial 2D-RoPE, 2D absolute pos | 305M USM-based conformer, 40ms Mel chunks |
| 12B | none (35M matmul) | none (640-d projection) |
| 26B-A4B / 31B | 550M ViT | no audio |

The vision encoders support variable resolution with a token budget of 70, 140, 280, 560 or 1120 tokens. Higher budgets preserve detail for OCR/document parsing; lower budgets are faster for classification/video frames.

The encoders are **frozen** during pretraining in the non-unified models. Only the projection and the LLM are trained. In 12B, the projection is trained with the LLM from scratch.

---

## 6. Tokenizer

Gemma 4 uses the Gemini-style SentencePiece tokenizer:

- **262k vocabulary**;
- split digits;
- preserved whitespace;
- byte-level encoding fallback;
- 140+ languages;
- control tokens for turns, tools, thinking, images and audio.

The vocabulary is much larger than Qwen3’s 151,669. That improves compression for multilingual/code text, but increases the embedding/output cost:

\[
P_{\text{emb}} = Vd = 262{,}144 \times 3{,}840 \approx 1.01\text{B}
\]

for 12B (tied). The output softmax at every token is also over 262k classes, which is a real inference cost.

---

## 7. Pretraining

### 7.1 Data

- Large-scale web documents, code, images and audio.
- **Cutoff date: January 2025.**
- 140+ languages.
- Filtering for benchmark decontamination, unsafe content, recitation and personal information.
- Exact token count and mixture are not disclosed.

### 7.2 Objective

Standard next-token prediction:

\[
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
\]

For multimodal sequences, image/audio tokens are interleaved with text tokens and the same objective applies; the model learns to predict the next text token conditioned on the modality tokens. The encoders (when present) are frozen, so the gradient flows through the projection and the decoder.

### 7.3 Infrastructure

Gemma 4 is trained on **TPU v5p and TPU v6e**:

| Model | TPU | Chips | Data shards | Seq shards | Replicas |
|---|---|---:|---:|---:|---:|
| E2B | v6e | 4,096 | 16 | 8 | 32 |
| E4B | v6e | 6,144 | 16 | 16 | 24 |
| **12B** | **v5p** | **12,288** | **16** | **16** | **48** |
| 26B-A4B | v6e | 6,144 | 16 | 16 | 24 |
| 31B | v6e | 10,240 | 16 | 16 | 40 |

The stack uses JAX/Pathways, GSPMD partitioning, ZeRO-3 optimizer sharding, and slice-granularity elasticity to keep training alive when a TPU slice fails. This is the kind of infrastructure a from-scratch 12B requires; a single Mac cannot substitute.

### 7.4 Quantization-aware training

Gemma 4 ships quantized checkpoints trained with QAT:

- mobile: per-channel int2/int4 weights + int8 activations;
- Q4_0: blockwise 4-bit weights;
- a scalar scale per block to keep activations in fp16 range.

Memory for 12B:

| Format | Weights | KV at 32K |
|---|---:|---:|
| bf16 | 24.0 GB | — |
| Q4_0 | 7.65 GB | 0.28 GB (int8) |

QAT matters because post-training quantization of a 12B model can degrade reasoning; training with simulated quantization preserves quality.

---

## 8. Multi-token prediction drafter

Gemma 4 trains a small **autoregressive MTP drafter** for speculative decoding:

- input: last-layer activations from the main model + token embeddings;
- a 4-layer transformer block cross-attends to the main model’s KV cache;
- predicts future tokens sequentially;
- no separate MTP prefill;
- 12B drafter: ~400M parameters;
- 26B/31B drafter: model dimension 1024, three local + one global layer;
- E2B/E4B use top-k cluster projection to reduce the final matmul from \(d\times262{,}000\) to \(d\times4096\).

Speculative decoding accepts the longest correct prefix; the speedup depends on the acceptance rate \(\alpha\). The drafter is a separate checkpoint, not part of the main model’s parameter count in the usual sense.

---

## 9. Instruction tuning

Gemma 4 uses a Gemma-3-like post-training recipe with a new **thinking mode**.

### 9.1 Data filtering

- remove personal information;
- remove unsafe/toxic model outputs;
- remove mistaken self-identification;
- deduplicate;
- include attribution, hedging and refusal examples to reduce hallucination;
- preserve factuality without degrading other metrics.

### 9.2 Thinking

Thinking is enabled by `<|think|>` at the start of the system prompt. The model emits:

```text
<|channel>thought
...reasoning...
<channel|>final answer
```

When thinking is disabled, models other than E2B/E4B still emit an empty thought block. Multi-turn history should contain final answers only, except tool-call turns where thinking is preserved.

### 9.3 PT vs IT formatting

All models share the tokenizer, but:

- pretrained models end generation with `<eos>`;
- instruction-tuned models end with `<turn|>`;
- fine-tuning either type requires adding the respective end token.

This is a common source of broken Gemma fine-tunes: train with the wrong EOS and the model never learns to stop.

### 9.4 Function calling

Gemma 4 has native function calling with:

- OpenAI-style tool schemas;
- `<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>`;
- `<|tool_response>...<tool_response|>` results;
- tool-call ids resolved to function names.

The template rejects string `arguments`; they must be mappings. This is the single biggest difference from Qwen tool data.

---

## 10. Safety and responsibility

Gemma 4 is developed with Google’s safety policies:

- CSAM and exploitation filtering;
- dangerous content, self-harm, sexually explicit, hate, harassment;
- PII filtering and recitation risk reduction;
- safety evaluations and train-time mitigations;
- responsible open-model release with usage policies.

The technical report also discusses governance, assessment and residual risk. Open weights do not remove the need for your own scope/policy layer if you build an agent.

---

## 11. What is disclosed vs not

**Disclosed:**

- architecture, layer counts, heads, vocabulary, context;
- tokenizer design;
- training hardware and sharding;
- QAT and MTP;
- post-training filtering categories;
- benchmark results;
- safety approach.

**Not disclosed:**

- exact token count and data mixture;
- exact optimizer hyperparameters and LR schedule;
- exact SFT/RL dataset sizes;
- reward model details;
- full data sources and dedup thresholds;
- exact pretraining compute in FLOPs.

A reproduction can match the architecture and pipeline shape, not the exact model.

---

## 12. Reproducing Gemma’s ideas at small scale

| Gemma choice | Small-scale experiment |
|---|---|
| 5:1 local/global | 4-layer model with 3 local + 1 global |
| K=V global | share K/V projection, measure quality/cache |
| p-RoPE 0.25 | compare full RoPE vs 25% on a long-context task |
| GeGLU | swap SwiGLU for GeGLU in a 100M model |
| encoder-free 12B | small ViT-free patch projector on a toy image-caption task |
| QAT | quantize a 100M model to int8/int4 and measure perplexity |
| MTP drafter | train a 2-layer draft head, measure acceptance rate |
| thinking channel | SFT a small model on `thought` + answer traces |

These are research experiments, not a path to a 12B model on a laptop. The practical path is [29 — Gemma fine-tuning](29-gemma4-finetuning.md): adapt the released weights.

---

## 13. Sources

- Gemma 4 technical report — <https://arxiv.org/abs/2607.02770>
- Gemma 4 model card — <https://ai.google.dev/gemma/docs/core/model_card_4>
- Gemma 4 12B guide — <https://developers.googleblog.com/en/gemma-4-12b-the-developer-guide>
- Gemma 4 function calling — <https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4>
- Gemma 4 chat template — <https://huggingface.co/google/gemma-4-12B-it/blob/main/chat_template.jinja>
- Gemma 4 config — <https://huggingface.co/google/gemma-4-12B-it/blob/main/config.json>
- Gemma 4 12B model card — <https://huggingface.co/google/gemma-4-12B-it>
- Gemma 4 QAT / memory table — in the technical report, §2.5
- [32 — Mathematics](32-mathematics.md)
- [27 — Gemma 4 12B](27-gemma4-12b.md)
