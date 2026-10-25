# 37 — Architecture anatomy: from `config.json` to tensors

*Last verified: 25 October 2026.*

[32 — Mathematics](32-mathematics.md) gives the equations. This chapter connects them to the actual files, shapes and design choices in Qwen3-14B, Qwen2.5-Coder-14B and Gemma 4 12B. It is the page to read when you want to know **why a checkpoint uses so much memory**, **what a config key changes**, or **where a fine-tune actually attaches**.

---

## 1. One block, one tensor

![A pre-norm decoder block](assets/diagrams/decoder-block.svg)

A decoder is a stack of \(L\) blocks around a residual stream of shape

\[
X\in\mathbb{R}^{B\times T\times d},
\]

where \(B\) is batch, \(T\) sequence length and \(d\) hidden size. The block is

\[
\begin{aligned}
U &= X + \operatorname{Attn}(\operatorname{Norm}_1(X)),\\
Y &= U + \operatorname{MLP}(\operatorname{Norm}_2(U)).
\end{aligned}
\]

The residual stream is the only path that connects the first embedding to the last logits. Attention and MLP are *reads* from and *writes* to that stream. This is why:

- every block has the same input/output shape \(B\times T\times d\);
- you can stack 40 or 48 blocks without changing the interface;
- LoRA can be inserted into projections without changing the residual shape;
- activation memory scales with \(L\times B\times T\times d\), not with parameter count alone.

### 1.1 Shape trace for Qwen3-14B

With \(B=1\), \(T=2{,}048\), \(d=5{,}120\), \(H=40\), \(G=8\), \(d_h=128\), \(d_{\text{ff}}=17{,}408\):

| Tensor | Shape | Notes |
|---|---|---|
| token ids | `[1, 2048]` | int32/int64 |
| embedding | `[1, 2048, 5120]` | \(Vd\) table lookup |
| normed input | `[1, 2048, 5120]` | RMSNorm |
| Q | `[1, 40, 2048, 128]` | \(H d_h=5120\) |
| K, V | `[1, 8, 2048, 128]` | \(G d_h=1024\) |
| attention scores | `[1, 40, 2048, 2048]` | \(O(BHT^2)\) |
| attention output | `[1, 2048, 5120]` | concat heads, \(W_O\) |
| MLP gate/up | `[1, 2048, 17408]` | two projections |
| MLP output | `[1, 2048, 5120]` | down projection |
| logits | `[1, 2048, 151936]` | \(V\) per position |

The logits tensor is enormous: \(1\cdot2048\cdot151{,}936\approx311\text{M}\) values. In fp32 that is 1.24 GB; in bf16 0.62 GB. This is why training uses fused cross-entropy and why long-context training is memory-bound even before attention.

### 1.2 Shape trace for Gemma 4 12B

With \(B=1\), \(T=2{,}048\), \(d=3{,}840\), \(H=16\), \(G=8\), local \(d_h=256\), global \(d_h=512\), \(d_{\text{ff}}=15{,}360\):

| Tensor | Shape |
|---|---|
| token ids | `[1, 2048]` |
| embedding | `[1, 2048, 3840]` |
| Q | `[1, 16, 2048, 256]` |
| K, V (local) | `[1, 8, 2048, 256]` |
| K=V (global) | `[1, 8, 2048, 512]` |
| local scores | `[1, 16, 2048, 2048]` masked to window 1024 |
| global scores | `[1, 16, 2048, 2048]` every sixth layer |
| logits | `[1, 2048, 262144]` |

The output logits for Gemma are even larger: \(1\cdot2048\cdot262{,}144\approx537\text{M}\) values. The 262k vocabulary is a real inference cost, not a footnote.

### 1.3 Parameter budget

`scripts/inspect_architecture.py` decomposes the checkpoint:

| Component | Qwen3-14B | Gemma 4 12B |
|---|---:|---:|
| attention / layer | 62.91M | 47.19M |
| MLP / layer | 267.39M | 176.95M |
| total / layer | 330.32M | 224.15M |
| embedding | 777.91M | 1,006.63M |
| LM head | 777.91M (untied) | 0 (tied) |
| **total** | **14.77B** | **11.77B** |
| official total | 14.8B | 11.95B |
| 4-bit weights (approx.) | ~8.3 GB | ~6.6–7.6 GB |

The formula count for Gemma is 11.77B; the official figure is 11.95B because fused projections and implementation-specific parameters are not all represented in the simple per-layer formula. Treat the table as a decomposition, not a replacement for the model card.

Gemma’s larger vocabulary costs ~1.0B parameters in the embedding; Qwen’s untied head costs another ~0.78B. The MLP is the largest per-layer block in both models: 81% of Qwen3-14B’s layer parameters and 79% of Gemma 4 12B’s.

This is why LoRA often targets both attention **and** MLP projections: the MLP holds most of the capacity.

### 1.4 Activation memory

Activation memory is roughly

\[
M_{\text{act}}\approx L\,B\,T\,d\,(c_{\text{attn}}+c_{\text{mlp}})
\]

before checkpointing, where \(c_{\text{attn}}\) and \(c_{\text{mlp}}\) depend on the implementation (number of intermediates kept for backward). For Qwen3-14B at \(B=1,T=2048,d=5120,L=40\), the residual stream alone is

\[
40\cdot1\cdot2048\cdot5120\cdot2\text{ bytes}\approx0.84\text{ GB}
\]

in bf16; attention scores add \(BHT^2\) per layer, which is much larger if materialised. Gradient checkpointing trades compute for memory by recomputing activations; it does not change the weights or the KV cache.

The practical memory order is:

\[
\text{weights} < \text{optimizer states} < \text{activations} < \text{attention scores}.
\]

For LoRA, the base weights are quantized and frozen, the optimizer state is tiny (adapter only), but activations still scale with \(LBTd\). That is why sequence length is the first knob to reduce.

---

## 2. GQA shapes and the cache

![Multi-head vs grouped-query attention](assets/diagrams/gqa.svg)

For \(H\) query heads and \(G\) KV heads:

\[
Q\in\mathbb{R}^{B\times H\times T\times d_h},\qquad
K,V\in\mathbb{R}^{B\times G\times T\times d_h}.
\]

Each KV head is shared by \(H/G\) query heads. The attention score tensor is still \(B\times H\times T\times T\), but the cache is \(B\times G\times T\times d_h\).

### 2.1 Cache formula

\[
\text{KV bytes}=2\cdot L\cdot G\cdot d_h\cdot T\cdot b.
\]

Qwen3-14B at \(T=32{,}768\), bf16:

\[
2\cdot40\cdot8\cdot128\cdot32{,}768\cdot2=5.37\text{ GB}.
\]

Gemma 4 12B full-attention upper bound:

\[
2\cdot48\cdot8\cdot256\cdot32{,}768\cdot2=12.88\text{ GB}.
\]

Gemma’s real cache is lower because:

- 5 of 6 layers are local with window \(W=1024\), so their cache is bounded by \(W\);
- global layers reuse \(K=V\);
- int8 KV halves the bytes.

Approximate the hybrid cache as

\[
\text{KV}\approx
L_{\text{local}}\cdot 2Gd_hWb
+
L_{\text{global}}\cdot G d_h T b.
\]

With \(L_{\text{local}}\approx40\), \(L_{\text{global}}\approx8\), \(W=1024\), the local part is small and the global part dominates only at very long \(T\).

### 2.2 Why GQA matters

| Model | Q heads | KV heads | Ratio | Cache vs MHA |
|---|---:|---:|---:|---:|
| Qwen3-14B | 40 | 8 | 5:1 | 1/5 |
| Qwen2.5-Coder-14B | 40 | 8 | 5:1 | 1/5 |
| Gemma 4 12B | 16 | 8 | 2:1 | 1/2 |

A 5:1 GQA model has one fifth of the KV cache of an MHA model with the same \(d_h\). That is the difference between serving 32K and 128K on one machine.

---

## 3. Stack patterns: full vs hybrid

![Qwen3 and Gemma 4 stacks](assets/diagrams/hybrid-stacks.svg)

### 3.1 Qwen3 dense

All 40 layers are full causal GQA. Cost per layer is \(O(T^2)\) for scores and \(O(T)\) cache. This is simple, predictable and easy to train. Long context is handled by RoPE base/ABF/YaRN and by training a long-context stage, not by changing the attention pattern.

### 3.2 Gemma 4 12B

48 layers in a repeating **5 local + 1 global** pattern:

- local: sliding window \(W=1024\), RoPE;
- global: full attention, \(K=V\), p-RoPE \(p=0.25\), base \(10^6\).

The local layers do most of the work at \(O(TW)\); the global layers recover long-range retrieval. This is a **hybrid architecture**: not pure attention, not pure linear attention.

### 3.3 Qwen3.5 / 3.6

A different hybrid: **3 Gated DeltaNet + 1 gated attention** per group. The GDN layers are linear-time with a fixed state; the full-attention layer recovers precise retrieval. This is closer to Mamba/DeltaNet than to Gemma’s sliding window.

| Pattern | Per-layer cost | Cache | Strength |
|---|---|---|---|
| Full attention | \(O(T^2)\) | \(O(T)\) | exact retrieval, simple |
| Sliding window | \(O(TW)\) | \(O(W)\) | cheap local context |
| GDN / linear | \(O(T)\) | \(O(1)\) | very long context, streaming |
| Hybrid local/global | mixed | mixed | balance |
| Hybrid GDN/attention | mixed | mostly \(O(1)\) | long context + retrieval |

---

## 4. Reading `config.json`

The config is the contract. These keys determine shapes:

| Key | Meaning | Effect |
|---|---|---|
| `hidden_size` / `d_model` | \(d\) | residual width, attention/MLP input |
| `num_hidden_layers` | \(L\) | depth, activation memory, cache layers |
| `intermediate_size` | \(d_{\text{ff}}\) | MLP width, parameter count |
| `num_attention_heads` | \(H\) | query heads |
| `num_key_value_heads` | \(G\) | KV heads; absence means MHA |
| `head_dim` | \(d_h\) | per-head width; may differ from \(d/H\) |
| `vocab_size` | \(V\) | embedding and logits |
| `tie_word_embeddings` | tied | input/output share weights |
| `max_position_embeddings` | context | RoPE cache / training length |
| `rope_theta` | RoPE base | long-context extrapolation |
| `rms_norm_eps` | \(\epsilon\) | numerical stability |
| `sliding_window` | \(W\) | local attention window |
| `layer_types` | list | local/global/hybrid pattern |
| `num_experts` | \(E\) | MoE total experts |
| `num_experts_per_tok` | \(k\) | active experts |
| `moe_intermediate_size` | expert FFN | expert width |

### 4.1 Traps

- **`head_dim` may be absent.** Then \(d_h=d/H\). Qwen2.5-Coder-14B has `head_dim: null`; \(5120/40=128\).
- **Gemma has `global_head_dim`.** Local layers use 256, global use 512. A single \(d_h\) underestimates global attention.
- **Reported vocab vs padded vocab.** Qwen3 reports 151,669 but the checkpoint pads to 151,936; Qwen2.5-Coder reports 151,646 and pads to 152,064. The padded rows are unused but allocated.
- **`max_position_embeddings` is not the training length.** Qwen3 trains at 32K and extends to 128K with YaRN; Gemma 4 supports 256K but you train at 2K–3K.
- **`tie_word_embeddings` changes parameter count by \(Vd\).** For Gemma 4 12B that is ~1.0B.
- **MoE configs have two FFN sizes.** `intermediate_size` may be the dense/shared FFN; `moe_intermediate_size` is per expert.
- **`layer_types` is the architecture.** Ignoring it turns a hybrid model into the wrong compute model.

### 4.2 Inspect a checkpoint before downloading

```bash
python scripts/inspect_architecture.py --hf Qwen/Qwen3-14B --seq 32768
python scripts/inspect_architecture.py --preset gemma4-12b --seq 32768
python scripts/inspect_architecture.py --hf Qwen/Qwen3-30B-A3B --seq 4096
```

The script prints:

- parameter count per component and total;
- active parameters for MoE;
- GQA ratio;
- KV cache at a chosen context;
- training/inference FLOPs for a batch.

It reads only `config.json`; no weights, no PyTorch.

---

## 5. Design choices and why they exist

### 5.1 Depth vs width

Given a parameter budget, depth and width trade off:

- deeper: more sequential computation, better compositional reasoning, harder to train (residual scaling, QK-norm);
- wider: more parallel compute, larger matrices, better GPU utilisation, more KV cache per layer.

Qwen3-14B is 40 layers at \(d=5120\); Qwen2.5-Coder-14B is 48 layers at the same width but a smaller FFN (13,824 vs 17,408). Gemma 4 12B is 48 layers at \(d=3840\) with a wider FFN (15,360). There is no universal optimum; data and training stability decide.

### 5.2 GQA ratio

Higher \(H/G\) saves cache but can hurt quality. 5:1 is a common compromise; Gemma 4 12B uses 2:1 because its head dimension is larger and it relies on hybrid attention for long context. Do not “optimise” GQA after pretraining; it is an architecture choice.

### 5.3 Vocabulary

A larger vocabulary:

- compresses text better (fewer tokens);
- costs \(Vd\) embedding parameters and \(V\) softmax per token;
- can improve multilingual/code coverage.

Qwen’s 151k and Gemma’s 262k reflect different multilingual/code priorities. A 262k vocab is ~1B parameters at \(d=3840\) when tied; at \(d=5120\) it would be ~1.34B.

### 5.4 Activation

SwiGLU vs GeGLU:

\[
\operatorname{SwiGLU}=W_d(\operatorname{SiLU}(W_gx)\odot W_ux),
\]

\[
\operatorname{GeGLU}=W_d(\operatorname{GELU}(W_gx)\odot W_ux).
\]

Both have three matrices. The difference is the nonlinearity; empirical results are close, and the choice is usually inherited from the pretraining family. Do not swap activation in a fine-tune; it changes the architecture.

### 5.5 Normalization

- **Pre-norm** (Qwen3): stable deep training, residual path is identity.
- **Pre+post-norm** (Gemma 4): extra normalization after the residual add; can improve representation quality but needs careful init.
- **QK-norm** (Qwen3, Gemma 4): normalizes queries/keys per head, bounding attention logits and stabilising large LR.

### 5.6 Positional encoding

- **RoPE** (Qwen3, Qwen2.5, Gemma local): relative, no learned table.
- **p-RoPE** (Gemma global): rotate only \(p d_h\) dimensions; cheaper and more stable at long context.
- **ABF/YaRN/DCA** (Qwen3 long context): adjust base frequency and attention scaling to extend beyond training length.

### 5.7 MoE

MoE decouples total parameters from active compute:

\[
N_{\text{active}}\approx N_{\text{attn}}+N_{\text{router}}+\frac{k}{E}N_{\text{experts}}+N_{\text{embed}}.
\]

Qwen3-30B-A3B stores 30.5B and activates 3.3B per token. The cost is routing, load balancing, all-to-all communication and memory for all experts. For a single 32 GB machine, a 30B MoE at 4-bit is ~17 GB and inference-only; training is not a realistic local path.

---

## 6. What fine-tuning changes — and what it doesn’t

| Thing | Pretraining | LoRA fine-tune |
|---|---|---|
| Tokenizer | fixed | fixed |
| Architecture | fixed | fixed |
| Attention pattern | fixed | fixed |
| KV cache | fixed | fixed |
| Base weights | trained | frozen (usually 4-bit) |
| Adapters | — | low-rank \(BA\) in projections |
| Sequence length | 32K–256K | 2K–3K (memory budget) |
| Batch | millions of tokens | 8–16 examples effective |
| Objective | next-token | masked completion |
| Chat template | learned | must match |

Consequences:

- LoRA does **not** give the model a longer context; it only trains on longer sequences.
- `num_layers: 24` selects the last 24 blocks; it does not change the model’s depth at inference. The adapter is applied to those blocks.
- `max_seq_length` is a training memory setting, not the model’s native context.
- Quantizing the base changes the weights, not the architecture.
- A different chat template can break a good adapter; see [06 — Data format](06-data-format.md).

---

## 7. Architecture checklist

Before training or serving:

- [ ] `config.json` parsed and parameter count matches the model card (within padding).
- [ ] GQA ratio and head dimension confirmed.
- [ ] Vocabulary and tied embeddings confirmed.
- [ ] Attention pattern (full/local/hybrid) understood.
- [ ] KV cache at target context estimated.
- [ ] `max_position_embeddings` vs training length separated.
- [ ] MoE total vs active parameters separated.
- [ ] Chat template rendered once with tools.
- [ ] `num_layers` and `max_seq_length` chosen from memory, not habit.
- [ ] The adapter target modules checked with `scripts/check_lora.py`.

---

## 8. Sources

- Qwen3 config — <https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json>
- Qwen2.5-Coder-14B config — <https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct/blob/main/config.json>
- Gemma 4 12B config — <https://huggingface.co/google/gemma-4-12B-it/blob/main/config.json>
- Qwen3 technical report — <https://arxiv.org/abs/2505.09388>
- Gemma 4 technical report — <https://arxiv.org/abs/2607.02770>
- [32 — Mathematics](32-mathematics.md)
- [38 — Derivations and proofs](38-derivations.md)
