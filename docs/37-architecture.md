# 37 — Architecture anatomy: from `config.json` to tensors

*Last verified: 27 October 2026.*

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

!!! example "On paper"

    One token (\(B=T=1\)) with \(d=2\): \(X=(2,2)\), RMSNorm with gain 1 and \(\epsilon\approx0\). Suppose attention returns \((5,-1)\) and the MLP returns \((-1,1)\).

    \[
    \begin{aligned}
    \operatorname{Norm}_1(X)&=(2,2)/2=(1,1)\\
    U&=(2,2)+(5,-1)=(7,1)\\
    \operatorname{Norm}_2(U)&=(7,1)/5=(1.4,\,0.2)\\
    Y&=(7,1)+(-1,1)=(6,2)
    \end{aligned}
    \]

    The MLP reads \((1.4,0.2)\), a vector of RMS 1, even though \(U\) itself has RMS \(\sqrt{(49+1)/2}=5\). The sub-layers only add to the stream, and \(Y\) keeps the \(1\times1\times2\) shape of \(X\).

    Longer version: [43 §4](43-math-examples.md).

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

On the 32 GB Mac this is why `max_seq_length` is the first knob to turn. The fp32 logits grow linearly with \(T\): 2.49 GB at 4,096 tokens and 4.98 GB at 8,192, before a single attention score is stored.

### 1.2 Shape trace for Gemma 4 12B

With \(B=1\), \(T=2{,}048\), \(d=3{,}840\), \(H=16\), \(G=8\), local \(d_h=256\), global \(d_h=512\), \(d_{\text{ff}}=15{,}360\):

| Tensor | Shape | Notes |
|---|---|---|
| token ids | `[1, 2048]` | int32/int64 |
| embedding | `[1, 2048, 3840]` | tied with the LM head |
| Q (local) | `[1, 16, 2048, 256]` | \(H d_h=4096\ne d\) |
| K, V (local) | `[1, 8, 2048, 256]` | \(G=8\); cache capped at window 1024 |
| Q (global) | `[1, 16, 2048, 512]` | \(H d_h=8192\) |
| K=V (global) | `[1, 1, 2048, 512]` | **one** KV head; a single tensor serves as K and V |
| local scores | `[1, 16, 2048, 2048]` | masked to window 1024 |
| global scores | `[1, 16, 2048, 2048]` | every sixth layer |
| logits | `[1, 2048, 262144]` | \(V\) per position |

The output logits for Gemma are even larger: \(1\cdot2048\cdot262{,}144\approx537\text{M}\) values. The 262k vocabulary is a real inference cost, not a footnote.

Look at the global K=V row: one head of width 512 in bf16 is \(512\cdot2=1{,}024\) bytes, so each global layer caches exactly **1 KB per token**. Eight global layers at 32,768 tokens is \(8\cdot32{,}768\cdot1\text{ KB}=268.4\text{ MB}\), the global term in §2.1. Reading the shape table correctly is the KV-cache calculation.

### 1.3 Parameter budget

`scripts/inspect_architecture.py` decomposes the checkpoint:

| Component | Qwen3-14B | Gemma 4 12B |
|---|---:|---:|
| attention / local layer | 62.91M | 47.19M |
| attention / global layer | — | 64.88M |
| MLP / layer | 267.39M | 176.95M |
| total / local layer | 330.32M | 224.15M |
| total / global layer | — | 241.84M |
| layers | 40 full | 40 local + 8 global |
| embedding | 777.91M | 1,006.63M |
| LM head | 777.91M (untied) | 0 (tied) |
| **total (text decoder)** | **14.77B** | **11.91B** |
| checkpoint / model card | 14.8B | 11.96B / 11.95B |
| 4-bit weights (approx.) | ~8.3 GB | ~7.65 GB |

Gemma’s global layers use `num_global_key_value_heads = 1` and `global_head_dim = 512`, so their attention block is larger than a local layer (64.88M vs 47.19M). They are also **K=V**: the key projection doubles as the value, so a global layer has `q_proj`, `k_proj` and `o_proj` but **no `v_proj`** — the checkpoint confirms it. Counting a value projection there adds 15.7M weights that do not exist; counting all 48 layers as local gives 11.77B and undercounts by ~0.14B.

The `google/gemma-4-12B-it` checkpoint holds exactly 11,959,730,224 parameters: 11.907B of text decoder plus 52.4M of multimodal embedders (a 35M vision patch embedder and the vision/audio input projections). Norms are only 0.7M. The model card rounds the total to 11.95B.

Gemma’s larger vocabulary costs ~1.0B parameters in the embedding; Qwen’s untied head costs another ~0.78B. The MLP is the largest per-layer block in both models: 81% of Qwen3-14B’s layer parameters and 79% of a Gemma 4 12B local layer’s.

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

Which term dominates depends on the regime:

- **Full fine-tuning with AdamW:** the optimizer state. fp32 master weights plus two moments are 12 bytes per parameter — 177 GB for Qwen3-14B, against 29.5 GB of bf16 weights.
- **LoRA/QLoRA:** the base weights are quantized and frozen (~8.3 GB at 4-bit) and the optimizer state is tiny (adapter only), so activations, which scale with \(LBTd\), decide whether a run fits. That is why sequence length is the first knob to reduce.
- **Long context without a fused kernel:** the \(BHT^2\) attention scores. For Qwen3-14B they are 336 MB per layer at 2K but 86 GB per layer at 32K in bf16, which is why FlashAttention never materialises them.

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

Gemma 4 12B is hybrid, so the single-formula number is only an upper bound:

\[
2\cdot48\cdot8\cdot256\cdot32{,}768\cdot2=12.88\text{ GB}.
\]

The real cache is

\[
\underbrace{L_{\text{local}}\cdot2G_{\text{local}}d_{h,\text{local}}Wb}_{\text{local, capped at }W}
+
\underbrace{L_{\text{global}}\cdot G_{\text{global}}d_{h,\text{global}}Tb}_{\text{global, }K=V}.
\]

With \(L_{\text{local}}=40\), \(G_{\text{local}}=8\), \(d_{h,\text{local}}=256\), \(W=1024\), \(L_{\text{global}}=8\), \(G_{\text{global}}=1\), \(d_{h,\text{global}}=512\), bf16:

\[
2\cdot40\cdot8\cdot256\cdot1024\cdot2 = 335.5\text{ MB}
\]

\[
8\cdot1\cdot512\cdot32{,}768\cdot2 = 268.4\text{ MB}
\]

Total at 32K: **604 MB**, not 12.9 GB. At 256K the local term is unchanged and the global term is 2.15 GB, so the cache is **2.48 GB**. Int8 KV halves both. This is the actual reason a 12B model can serve 256K context; the sliding window and the single global KV head do the work.

### 2.2 Why GQA matters

| Model | Q heads | KV heads | Ratio | Cache vs MHA |
|---|---:|---:|---:|---:|
| Qwen3-14B | 40 | 8 | 5:1 | 1/5 |
| Qwen2.5-Coder-14B | 40 | 8 | 5:1 | 1/5 |
| Gemma 4 12B (local) | 16 | 8 | 2:1 | 1/2 |
| Gemma 4 12B (global) | 16 | 1 | 16:1 | 1/16 |
| Gemma 4 E4B (local and global) | 8 | 2 | 4:1 | 1/4 |
| Gemma 4 E2B (local and global) | 8 | 1 | 8:1 | 1/8 |

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

## 4. The edge models: Gemma 4 E2B and E4B

E2B and E4B are not shrunken 12Bs. Four mechanisms change how their parameters, compute and cache behave, and each one is a key in `config.json` and a visible shape in the checkpoint. Everything below was read from the configs and safetensors headers of `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it` and `google/gemma-4-12B-it`, and from the Hugging Face Transformers `gemma4` implementation, on 27 October 2026; `scripts/check_math_examples.py` reproduces all three checkpoints to the parameter.

### 4.1 The configs side by side

<div class="gb-keys" markdown>

| Key | E2B | E4B | 12B |
|---|---:|---:|---:|
| `hidden_size` \(d\) | 1,536 | 2,560 | 3,840 |
| `num_hidden_layers` \(L\) | 35 | 42 | 48 |
| local : global (counts) | **4 : 1** (28 + 7) | 5 : 1 (35 + 7) | 5 : 1 (40 + 8) |
| `sliding_window` \(W\) | **512** | **512** | 1,024 |
| query heads \(H\) | 8 | 8 | 16 |
| KV heads, local / global | 1 / 1 | 2 / 2 | 8 / 1 |
| head dim, local / global | 256 / 512 | 256 / 512 | 256 / 512 |
| `attention_k_eq_v` | false | false | **true** |
| `intermediate_size` | 6,144 (12,288 in shared layers) | 10,240 | 15,360 |
| `hidden_size_per_layer_input` | **256** | **256** | 0 |
| `num_kv_shared_layers` | **20** | **18** | 0 |
| `use_double_wide_mlp` | **true** | false | false |
| `max_position_embeddings` | 131,072 | 131,072 | 262,144 |

</div>

### 4.2 Per-layer embeddings: what “effective” means

Next to the ordinary token embedding (\(V\times d\)), the edge models store a second table, `embed_tokens_per_layer`, of shape \([V,\;L\cdot256]\): for every token, a separate 256-wide vector **for each layer**. Each layer \(\ell\) receives an input \(p_\ell\) that combines that table row with a projection of the token embedding, and after attention and the MLP it adds a small gated correction to the residual stream:

\[
p_\ell=\tfrac{1}{\sqrt2}\Big(E_{\text{PLE}}[x]_\ell+\operatorname{RMSNorm}\big((W_{\text{proj}}\,e_x)_\ell\big)\Big)\in\mathbb{R}^{256},
\]

\[
h\leftarrow h+\operatorname{RMSNorm}\Big(W_{\uparrow}\big(\operatorname{GELU}(W_{\downarrow}h)\odot p_\ell\big)\Big),
\]

with \(W_{\downarrow}\in\mathbb{R}^{256\times d}\) and \(W_{\uparrow}\in\mathbb{R}^{d\times256}\) per layer (fixed scale factors omitted). The table is a **lookup, not a matmul**: it costs storage, not compute. For E2B it is \(262{,}144\cdot35\cdot256=2.35\text{B}\) parameters, but a token reads only \(35\cdot256=8{,}960\) of them — 17.5 KB in bf16.

That is the whole meaning of the “E”. Subtract the two embedding tables from the checkpoint and you get the model-card size:

| Component (from the checkpoint) | E2B | E4B |
|---|---:|---:|
| attention, all layers | 278.4M | 587.2M |
| MLP, all layers | 1,557.1M | 3,303.0M |
| PLE gates, projections and norms | 41.3M | 82.7M |
| other norms and layer scalars | 0.2M | 0.4M |
| **text decoder compute** | **1.88B** | **3.97B** |
| token embedding (tied LM head) | 402.7M | 671.1M |
| per-layer embedding table | 2,348.8M | 2,818.6M |
| vision + audio encoders | 475.7M | 478.1M |
| unused K/V weights of shared layers | 18.9M | 55.1M |
| **total in the checkpoint** | **5.12B** | **8.00B** |
| total minus the two tables | 2.37B (card: 2.3B) | 4.51B (card: 4.5B) |

More than half of E2B’s file (53.7%) is embedding tables. For text-only work the per-token compute is \(\approx2\times(1.88+0.40)\text{B}=4.6\) GFLOPs for E2B and \(\approx2\times(3.97+0.67)\text{B}=9.3\) GFLOPs for E4B (the tied embedding still runs once as the LM head), against 23.8 GFLOPs for the 12B. As in the larger models, the MLP holds most of the capacity: 83% of the text decoder in both.

### 4.3 KV sharing: the last layers store nothing

The last `num_kv_shared_layers` layers do not compute keys or values. Each one reuses the K/V of the **last earlier layer of the same type**: in E4B, layers 24–41 reuse layer 22 (local) or 23 (global); in E2B, layers 15–34 reuse layer 13 or 14. Shared layers therefore have no `k_proj`/`v_proj` at run time — the checkpoints still ship those tensors (the “unused K/V” row above), and Transformers drops them on load.

Only the non-shared layers enter the cache: 20 local + 4 global for E4B, 12 local + 3 global for E2B. There is no K=V, so each cached global layer stores both K and V:

\[
\text{KV}=\sum_{\substack{\ell<L-S\\\text{local}}}2\,G\,d_h\,\min(T,W)\,b
\;+\sum_{\substack{\ell<L-S\\\text{global}}}2\,G_g\,d_{h,g}\,T\,b .
\]

| bf16 cache | global bytes per token | at 32K | at 128K |
|---|---:|---:|---:|
| Gemma 4 E2B | \(3\cdot2\cdot1\cdot512\cdot2=6\) KB | 208 MB | 812 MB |
| Gemma 4 E4B | \(4\cdot2\cdot2\cdot512\cdot2=16\) KB | 558 MB | **2.17 GB** |
| Gemma 4 12B | \(8\cdot1\cdot1\cdot512\cdot2=8\) KB (K=V) | 604 MB | 1.41 GB |

The local terms are small and constant (6.3 MB, 21 MB and 335.5 MB). The surprise is the last column: **at 128K the E4B cache is larger than the 12B’s**. The 12B stores one K=V head per global layer; E4B stores two heads, with separate K and V. A smaller model does not mean a smaller cache — the global-layer shape decides. Without KV sharing (a runtime that ignores `num_kv_shared_layers`) E4B would need 0.98 GB at 32K and E2B 0.48 GB; sharing saves 43% and 57%.

### 4.4 The double-wide MLP and the 4:1 pattern

With `use_double_wide_mlp`, E2B’s 20 KV-shared layers get twice the MLP width (12,288 instead of 6,144). The capacity saved by not projecting K/V goes back into the MLP: 1,132M of E2B’s 1,557M MLP parameters live in those 20 layers. E2B also uses a 4:1 local:global pattern (seven global layers in 35) instead of 5:1, and both edge models halve the sliding window to 512 tokens.

### 4.5 Shape trace for Gemma 4 E4B

With \(B=1\), \(T=2{,}048\):

| Tensor | Shape | Notes |
|---|---|---|
| token ids | `[1, 2048]` | int32/int64 |
| embedding | `[1, 2048, 2560]` | tied with the LM head |
| per-layer inputs | `[1, 2048, 42, 256]` | one 256-vector per layer, from the PLE table + projection |
| Q (local) | `[1, 8, 2048, 256]` | \(H d_h=2048\ne d\) |
| K, V (local, layers 0–23) | `[1, 2, 2048, 256]` | \(G=2\); cache keeps the last 512 |
| Q (global) | `[1, 8, 2048, 512]` | \(H d_h=4096\) |
| K, V (global, layers 0–23) | `[1, 2, 2048, 512]` | separate K and V |
| K, V (layers 24–41) | — | reused from layer 22 or 23 |
| MLP gate/up | `[1, 2048, 10240]` | GeGLU |
| logits | `[1, 2048, 262144]` | \(V\) per position |

### 4.6 What this means on the Mac

- **Size on disk is not size in compute.** A 4-bit E4B is ~4 GB of weights, but over 40% of it is lookup tables; per token it computes like a ~4B dense model, not an 8B one.
- **Adapters can only attach where weights run.** In the reference implementation the shared layers have no `k_proj`/`v_proj`, so LoRA there reaches `q_proj`, `o_proj` and the MLP. Run `scripts/check_lora.py` to see which modules your runtime actually adapts.
- **Long context is cheap on E2B, not on E4B.** E2B serves 128K in 0.8 GB of cache; E4B needs 2.2 GB, more than the 12B. Quantising the cache to int8 halves either.
- **Inspect before downloading:** `python scripts/inspect_architecture.py --hf google/gemma-4-E4B-it --seq 131072` models PLE, KV sharing and the double-wide MLP.

---

## 5. Reading `config.json`

The config is the contract. These keys determine shapes:

<div class="gb-keys" markdown>

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
| `attention_k_eq_v` | K=V | global layers have no `v_proj`; one cached tensor |
| `num_kv_shared_layers` | \(S\) | last \(S\) layers reuse earlier K/V: no K/V projections, no cache |
| `hidden_size_per_layer_input` | PLE width | per-layer embedding table \([V, L\cdot P]\) (edge models) |
| `use_double_wide_mlp` | — | KV-shared layers get a 2× MLP (E2B) |

</div>

### 5.1 Traps

- **`head_dim` may be absent.** Then \(d_h=d/H\). Qwen2.5-Coder-14B has `head_dim: null`; \(5120/40=128\).
- **Gemma has `global_head_dim`.** Local layers use 256, global use 512. A single \(d_h\) underestimates global attention.
- **Reported vocab vs padded vocab.** Qwen3 reports 151,669 but the checkpoint pads to 151,936; Qwen2.5-Coder reports 151,646 and pads to 152,064. The padded rows are unused but allocated.
- **`max_position_embeddings` is not the training length.** Qwen3 trains at 32K and extends to 128K with YaRN; Gemma 4 supports 256K but you train at 2K–3K.
- **`tie_word_embeddings` changes parameter count by \(Vd\).** For Gemma 4 12B that is ~1.0B.
- **MoE configs have two FFN sizes.** `intermediate_size` may be the dense/shared FFN; `moe_intermediate_size` is per expert.
- **`layer_types` is the architecture.** Ignoring it turns a hybrid model into the wrong compute model.
- **K=V means no `v_proj`.** Counting one in the 12B’s global layers adds 15.7M weights that do not exist.
- **An “E” size is not a compute size.** E2B’s 5.1B checkpoint is 54% embedding tables; its text decoder computes with 1.88B. See §4.2.
- **KV-shared layers ship unused weights.** The E2B/E4B files contain `k_proj`/`v_proj` for shared layers (18.9M / 55.1M) that are never used; do not count them, and do not expect a cache for those layers.

### 5.2 Inspect a checkpoint before downloading

```bash
python scripts/inspect_architecture.py --hf Qwen/Qwen3-14B --seq 32768
python scripts/inspect_architecture.py --preset gemma4-12b --seq 32768
python scripts/inspect_architecture.py --preset gemma4-e4b --seq 131072
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

## 6. Design choices and why they exist

### 6.1 Depth vs width

Given a parameter budget, depth and width trade off:

- deeper: more sequential computation, better compositional reasoning, harder to train (residual scaling, QK-norm);
- wider: more parallel compute, larger matrices, better GPU utilisation, more KV cache per layer.

Qwen3-14B is 40 layers at \(d=5120\); Qwen2.5-Coder-14B is 48 layers at the same width but a smaller FFN (13,824 vs 17,408). Gemma 4 12B is 48 layers at \(d=3840\) with a wider FFN (15,360). There is no universal optimum; data and training stability decide.

### 6.2 GQA ratio

Higher \(H/G\) saves cache but can hurt quality. 5:1 is a common compromise; Gemma 4 12B uses 2:1 because its head dimension is larger and it relies on hybrid attention for long context. Do not “optimise” GQA after pretraining; it is an architecture choice.

### 6.3 Vocabulary

A larger vocabulary:

- compresses text better (fewer tokens);
- costs \(Vd\) embedding parameters and \(V\) softmax per token;
- can improve multilingual/code coverage.

Qwen’s 151k and Gemma’s 262k reflect different multilingual/code priorities. A 262k vocab is ~1B parameters at \(d=3840\) when tied; at \(d=5120\) it would be ~1.34B.

### 6.4 Activation

SwiGLU vs GeGLU:

\[
\operatorname{SwiGLU}=W_d(\operatorname{SiLU}(W_gx)\odot W_ux),
\]

\[
\operatorname{GeGLU}=W_d(\operatorname{GELU}(W_gx)\odot W_ux).
\]

Both have three matrices. The difference is the nonlinearity; empirical results are close, and the choice is usually inherited from the pretraining family. Do not swap activation in a fine-tune; it changes the architecture.

!!! example "On paper"

    One hidden unit (\(d_{\text{ff}}=1\)) with gate \(W_gx=1\), up value \(W_ux=2\) and \(W_d=1\). Use \(\sigma(1)\approx0.7311\) (\(e\approx2.718\)) and the normal CDF \(\Phi(1)\approx0.8413\), so \(\operatorname{SiLU}(1)=\sigma(1)\) and \(\operatorname{GELU}(1)=\Phi(1)\).

    \[
    \begin{aligned}
    \operatorname{SwiGLU}&=0.7311\cdot2\approx1.462\\
    \operatorname{GeGLU}&=0.8413\cdot2\approx1.683
    \end{aligned}
    \]

    With a negative gate \(W_gx=-1\), \(\operatorname{SiLU}(-1)=-0.2689\) and \(\operatorname{GELU}(-1)=-0.1587\), giving \(-0.538\) and \(-0.317\). GELU lets more of a positive gate through and less of a negative one; Gemma’s tanh approximation of GELU agrees to three decimals.

    Longer version: [43 §9](43-math-examples.md).

### 6.5 Normalization

- **Pre-norm** (Qwen3): stable deep training, residual path is identity.
- **Pre+post-norm** (Gemma 4): extra normalization after the residual add; can improve representation quality but needs careful init.
- **QK-norm** (Qwen3, Gemma 4): normalizes queries/keys per head, bounding attention logits and stabilising large LR.

### 6.6 Positional encoding

- **RoPE** (Qwen3, Qwen2.5, Gemma local): relative, no learned table.
- **p-RoPE** (Gemma global): rotate only \(p d_h\) dimensions; cheaper and more stable at long context.
- **ABF/YaRN/DCA** (Qwen3 long context): adjust base frequency and attention scaling to extend beyond training length.

### 6.7 MoE

MoE decouples total parameters from active compute:

\[
N_{\text{active}}\approx N_{\text{attn}}+N_{\text{router}}+\frac{k}{E}N_{\text{experts}}+N_{\text{embed}}.
\]

Qwen3-30B-A3B stores 30.5B and activates 3.3B per token. The cost is routing, load balancing, all-to-all communication and memory for all experts. For a single 32 GB machine, a 30B MoE at 4-bit is ~17 GB and inference-only; training is not a realistic local path.

!!! example "On paper"

    Qwen3-30B-A3B: 48 layers, \(d=2{,}048\), 32 query and 4 KV heads of \(d_h=128\) (so \(Hd_h=4{,}096\ne d\)), \(E=128\) experts of width 768, \(k=8\), untied \(V=151{,}936\).

    | Term | Per layer | × 48 layers |
    |---|---:|---:|
    | \(N_{\text{attn}}\): \(2\cdot2{,}048\cdot4{,}096+2\cdot2{,}048\cdot4\cdot128\) | 18.87M | 0.906B |
    | \(N_{\text{router}}\): \(2{,}048\cdot128\) | 0.26M | 0.013B |
    | \(N_{\text{experts}}\): \(128\cdot3\cdot2{,}048\cdot768\) | 603.98M | 28.99B |
    | \(N_{\text{embed}}\): \(2Vd\), untied | — | 0.622B |

    \[
    \begin{aligned}
    \tfrac{k}{E}N_{\text{experts}}&=\tfrac{8}{128}\cdot28.99\text{B}=1.812\text{B}\\
    N_{\text{active}}&\approx0.906+0.013+1.812+0.622\\
    &\approx3.35\text{B}
    \end{aligned}
    \]

    Stored, all four rows add up to \(\approx30.5\)B; per token only 3.35B are used. Those are the 30.5B and 3.3B above.

    Longer version: [43 §10](43-math-examples.md).

---

## 7. What fine-tuning changes — and what it doesn’t

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

## 8. Architecture checklist

Before training or serving:

- [ ] `config.json` parsed and parameter count matches the model card (within padding).
- [ ] GQA ratio and head dimension confirmed.
- [ ] Vocabulary and tied embeddings confirmed.
- [ ] Attention pattern (full/local/hybrid) understood.
- [ ] KV cache at target context estimated.
- [ ] `max_position_embeddings` vs training length separated.
- [ ] MoE total vs active parameters separated.
- [ ] Edge models: per-layer embedding tables and KV-shared layers separated from compute and cache.
- [ ] Chat template rendered once with tools.
- [ ] `num_layers` and `max_seq_length` chosen from memory, not habit.
- [ ] The adapter target modules checked with `scripts/check_lora.py`.

---

## 9. Sources

- Qwen3 config — <https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json>
- Qwen2.5-Coder-14B config — <https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct/blob/main/config.json>
- Gemma 4 12B config — <https://huggingface.co/google/gemma-4-12B-it/blob/main/config.json>
- Gemma 4 E4B config — <https://huggingface.co/google/gemma-4-E4B-it/blob/main/config.json>
- Gemma 4 E2B config — <https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json>
- Transformers `gemma4` implementation — <https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4>
- Qwen3 technical report — <https://arxiv.org/abs/2505.09388>
- Gemma 4 technical report — <https://arxiv.org/abs/2607.02770>
- [32 — Mathematics](32-mathematics.md)
- [38 — Derivations and proofs](38-derivations.md)
