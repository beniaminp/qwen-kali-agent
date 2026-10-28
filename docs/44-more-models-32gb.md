# More 32 GB models — five other families, in depth

The handbook's main tracks use Qwen and Gemma. The local pipeline —
**download → check adapter → prepare data → QLoRA train → fuse → serve →
smoke-test → eval** — is not Qwen-specific. This chapter applies the *exact same
steps* to five other open-weight families that fit a 32 GB M1 Pro, and explains
**each one's architecture and the math behind it** down to the config field, so
you know what you are training and why it fits.

Read [`32-mathematics.md`](32-mathematics.md), [`37-architecture.md`](37-architecture.md)
and [`38-derivations.md`](38-derivations.md) first — this chapter reuses those
formulas and only fills in the per-family numbers.

> **Status.** Repo IDs, config shapes, sizes and licenses were read from Hugging
> Face `config.json` on 28 Oct 2026 with `scripts/inspect_architecture.py`. The
> pipeline *mechanics* were run end-to-end on the target M1 Pro against
> `mlx-community/Qwen3-14B-4bit`; the per-family commands are the same steps
> parameterised by `MODEL`, **verified as commands, not each hardware-tested to
> convergence**. The architecture numbers below are from the real configs.

> **Authorized testing only.** Same rules as the rest of the handbook. See
> [`12-safety-legal.md`](12-safety-legal.md).

---

## The five families (six models)

All are on the `mlx-community` org as pre-converted 4-bit MLX weights.

| Family | Model (repo id) | Params | 4-bit size | License | Tool-calling | 32 GB verdict |
|---|---|---:|---:|---|---|---|
| **Meta Llama** | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | 8.03B | ~4.5 GB | Llama 3.1 | native (chat template) | train **all layers**, lots of headroom |
| **Mistral** | `mlx-community/Mistral-Small-24B-Instruct-2501-4bit` | 23.57B | 13.3 GB | Apache-2.0 | native (`[TOOL_CALLS]`) | **serve** comfortably; train **last 8–16 layers** |
| **Microsoft Phi** | `mlx-community/phi-4-4bit` | 14.66B | 8.25 GB | MIT | weak native — prompt/train it in | train all layers (≈ Qwen3-14B) |
| **DeepSeek** | `mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit` | 14.77B | ~8 GB | MIT | none native — reasoning, emits `<think>` | train all layers; parse `<think>` |
| **IBM Granite** | `mlx-community/granite-3.3-8b-instruct-4bit` | 8.17B | ~4.6 GB | Apache-2.0 | native (`<tool_call>`) | train all layers, lots of headroom |
| **01.AI Yi** | `mlx-community/Yi-1.5-9B-Chat-4bit` | 8.83B | ~5 GB | Apache-2.0 | weak native — prompt/train it in | train all layers |

---

## The shared architecture

All six are the same creature as Qwen3: a **decoder-only, pre-norm transformer**
with **RMSNorm**, a **SwiGLU** MLP, **RoPE** positions and **grouped-query
attention (GQA)**. If you understand one, you understand all six; only the
*numbers* and a few scalars change. The pieces, with the formulas this chapter
reuses (notation as in [`32-mathematics.md`](32-mathematics.md): \(d\) hidden
size, \(f\) FFN width, \(V\) vocab, \(L\) layers, \(H_q/H_{kv}\) query/KV heads,
\(d_h\) head dimension, \(b\) bytes per value):

**RMSNorm** — no mean-subtraction, one gain vector \(g\):

\[
\operatorname{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \varepsilon}} \odot g
\]

**Attention**, one query group, with the usual \(1/\sqrt{d_h}\) score scale:

\[
\operatorname{Attn}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right)V
\]

**GQA**: \(H_q\) query heads share \(H_{kv}\) key/value heads, each KV head
serving \(r = H_q/H_{kv}\) queries. The KV cache — the thing that eats memory at
long context — is

\[
\text{KV bytes per token} = 2\,L\,H_{kv}\,d_h\,b \qquad (b=2 \text{ for bf16})
\]

so halving \(H_{kv}\) halves the cache. This one ratio explains the KV column two
sections down.

**RoPE**: rotate each pair of dims by angle \(\theta_i = \text{pos}\cdot \text{base}^{-2i/d_h}\).
The **base** (`rope_theta`) sets the longest wavelength; a larger base packs more
distinguishable positions, which is how these models reach 32K–128K context.

**SwiGLU MLP** — three matrices (gate, up, down), hence the \(3df\) term:

\[
\operatorname{FFN}(x) = \big(\operatorname{silu}(xW_{\text{gate}}) \odot xW_{\text{up}}\big)W_{\text{down}}
\]

**Parameter count** (what `inspect_architecture.py` sums), with \([\text{untied}]=1\)
when the LM head is not tied to the embedding:

\[
N \approx \underbrace{Vd}_{\text{embed}} + \underbrace{[\text{untied}]\,Vd}_{\text{LM head}} + L\Big(\underbrace{2d^2 + 2d\,H_{kv}d_h}_{\text{attn: }Q,O,K,V} + \underbrace{3df}_{\text{MLP}}\Big)
\]

**Training FLOPs** \(\approx 6ND\) over \(D\) tokens; **inference** \(\approx 2N\)
per token ([`32-mathematics.md`](32-mathematics.md)).

**QLoRA memory on 32 GB.** 4-bit weights are \(\approx 0.5\) GB per billion
params. QLoRA adds the frozen 4-bit base + small LoRA adapters (rank 16 ≈ tens of
MB) + AdamW state **for the adapters only** + activations (cut hard by
`grad_checkpoint`). With batch 1 and `max_seq_length: 2048`, every **dense model
≤ ~15B** here trains all-layers inside 32 GB — the Qwen3-14B envelope
([`02-hardware.md`](02-hardware.md)). **Mistral-Small-24B (13.3 GB resident) is
the one exception:** last 8–16 layers only, or serve it.

---

## Family by family: what's different, and the math

Every config below shares `hidden_act: silu` (SwiGLU), `attention_dropout: 0`,
head dim 128, and pre-norm RMSNorm. The rows that differ are what matter.

### Meta Llama 3.1 8B — the clean baseline

| field | value | field | value |
|---|---|---|---|
| layers \(L\) | 32 | hidden \(d\) | 4096 |
| heads \(H_q/H_{kv}\) | 32 / 8 (**GQA 4:1**) | FFN \(f\) | 14336 (3.5×) |
| vocab \(V\) | 128256 | tied | no |
| `rope_theta` | 500000 | ctx | 131072 |
| `rope_scaling` | **llama3** (factor 8) | `rms_norm_eps` | 1e-5 |

Textbook Llama: no attention bias, untied embeddings, SwiGLU, RMSNorm. The one
subtlety is **llama3 RoPE scaling**: frequencies with wavelength below
`low_freq_factor` of the original 8192 window stay as-is, those above
`high_freq_factor` are divided by `factor` (8), with a smooth ramp between — this
stretches a model trained at 8K out to 131K without retraining. **On paper**,
attention/layer \(= 2d^2 + 2dH_{kv}d_h = 2(4096^2) + 2(4096)(1024) = 41.94\)M;
MLP \(= 3df = 3(4096)(14336) = 176.16\)M; ×32 layers + two \(Vd = 525\)M
embeddings → **8.03B** ✓. KV \(= 2(32)(8)(128)(2) = 131{,}072\) B/token =
**128 KiB/token**. Native tool calls, biggest headroom → **the first model to
port the pipeline to.**

### Mistral-Small-24B-Instruct-2501 — wide MLP, giant RoPE base

| field | value | field | value |
|---|---|---|---|
| layers \(L\) | 40 | hidden \(d\) | 5120 |
| heads \(H_q/H_{kv}\) | 32 / 8 (**GQA 4:1**) | FFN \(f\) | **32768 (6.4×)** |
| vocab \(V\) | 131072 (Tekken) | tied | no |
| `rope_theta` | **1e8** | ctx | 32768 |
| `sliding_window` | none (full attn) | `rms_norm_eps` | 1e-5 |

What makes it 24B is the **unusually wide FFN**: \(f/d = 6.4\) vs the usual ~3.5,
so MLP/layer \(= 3(5120)(32768) = 503\)M — **90% of each layer's params live in
the MLP.** That is why QLoRA on 32 GB should target the **last 8–16 layers**: the
all-layer optimizer/activation footprint over 40 wide layers is what blows the
budget, not inference. The **enormous `rope_theta` (1e8)** is Mistral's way of
getting clean 32K context. Strong native `[TOOL_CALLS]`, Apache-2.0 → also an
excellent **teacher/judge** for synthetic data ([`07-synthetic-data.md`](07-synthetic-data.md)).

### Microsoft Phi-4 (~15B) — `phi3` class, fused projections

| field | value | field | value |
|---|---|---|---|
| layers \(L\) | 40 | hidden \(d\) | 5120 |
| heads \(H_q/H_{kv}\) | 40 / 10 (**GQA 4:1**) | FFN \(f\) | 17920 (3.5×) |
| vocab \(V\) | 100352 | tied | no |
| `rope_theta` | 250000 | ctx | 16384 |
| `model_type` | **phi3** | `rms_norm_eps` | 1e-5 |

Same memory envelope as Qwen3-14B (14.66B, 8.25 GB). The `phi3` implementation
**fuses** the Q/K/V matrices into one `qkv_proj` and gate/up into one
`gate_up_proj` — a kernel-efficiency choice, **mathematically identical** to the
split form, and `check_lora.py` still reports the standard projections after MLX
splits them. The real caveat is behavioural, not architectural: **Phi-4 has no
official tool-call template** (it was trained largely on synthetic textbook data
for reasoning), so it will not emit structured calls out of the box — that is
exactly what your fine-tune installs. Verify tool validity hard in step 8.

### DeepSeek-R1-Distill-Qwen-14B — Qwen2 shape, QKV bias, `<think>`

| field | value | field | value |
|---|---|---|---|
| layers \(L\) | **48** | hidden \(d\) | 5120 |
| heads \(H_q/H_{kv}\) | 40 / 8 (**GQA 5:1**) | FFN \(f\) | 13824 (2.7×) |
| vocab \(V\) | 152064 | tied | no |
| `rope_theta` | 1e6 | ctx | 131072 |
| `model_type` | **qwen2** | `rms_norm_eps` | 1e-5 |

Architecturally this **is Qwen2.5-14B** — deep and narrow (48 layers, FFN only
2.7×), and it carries Qwen2's one distinctive feature: a **bias term on the Q, K,
V projections** (Llama/Mistral/Granite/Yi have none). It is not a fresh
pre-train but a **distillation**: Qwen2.5-14B supervised-fine-tuned on
DeepSeek-R1's reasoning traces, so it **emits `<think> … </think>` before the
answer** and has no native tool calling. Use it where planning matters and make
the agent harness **strip or parse the think block** before reading tool calls.
Because the tokenizer is Qwen2.5's, try `--arguments-format dict` if `string`
renders oddly (the switch the Java/React tracks use).

### IBM Granite 3.3 8B — tied embeddings and four muP scalars

| field | value | field | value |
|---|---|---|---|
| layers \(L\) | 40 | hidden \(d\) | 4096 |
| heads \(H_q/H_{kv}\) | 32 / 8 (**GQA 4:1**) | FFN \(f\) | 12800 (3.1×) |
| vocab \(V\) | **49159** | tied | **yes** |
| `rope_theta` | 1e7 | ctx | 131072 |
| `embedding_multiplier` | **12.0** | `residual_multiplier` | **0.22** |
| `attention_multiplier` | **0.0078125** | `logits_scaling` | **16.0** |

Granite is the most *different* math here. **Tied embeddings** mean the input
table doubles as the output projection (LM head = 0 extra params), and with a
tiny 49k vocab the embeddings are only 201M of 8.17B — almost all capacity is in
the layers. Then four **µP-style scalars** rescale activations (a
depth/width-robust-training trick from IBM's Power-scheduler work):

\[
h_0 = 12.0\,E[\text{token}],\qquad
\text{scores} = (QK^\top)\cdot 0.0078125,\qquad
h \mathrel{+}= 0.22\,\operatorname{sublayer}(h),\qquad
\text{logits} = \tfrac{1}{16}\,hE^\top
\]

Note `attention_multiplier` \(= 0.0078125 = 1/128 = 1/d_h\): Granite replaces the
usual \(1/\sqrt{d_h}=1/\sqrt{128}\approx 0.088\) with \(1/d_h\), a **sharper**
score scale. These are constants baked into the weights, so training and serving
need no special handling — MLX applies them from the config. Native `<tool_call>`
markers, Apache-2.0, big headroom → **the best drop-in Qwen alternative for a
tool-calling agent on 32 GB.**

### 01.AI Yi-1.5-9B-Chat — deep-narrow, aggressive 8:1 GQA

| field | value | field | value |
|---|---|---|---|
| layers \(L\) | **48** | hidden \(d\) | 4096 |
| heads \(H_q/H_{kv}\) | 32 / **4** (**GQA 8:1**) | FFN \(f\) | 11008 (2.7×) |
| vocab \(V\) | 64000 | tied | no |
| `rope_theta` | 5e6 | ctx | **4096** |
| `rms_norm_eps` | **1e-6** | | |

Llama-architecture, but two things stand out. The **8:1 GQA** (only 4 KV heads)
gives the **smallest KV cache of the six** despite 48 layers:
\(2(48)(4)(128)(2) = 98{,}304\) B = **96 KiB/token**, vs Llama's 128 KiB on half
the depth — long-context serving is cheapest here. And the **short 4096 context**
of Yi-1.5 base: if you need long trajectories, pick a Yi 16K/32K variant or raise
RoPE base, and watch `token_stats.py` so training rows fit. Note too the tighter
`rms_norm_eps` of 1e-6. No strong native tool calling → train the format in, like
Phi-4.

### The numbers side by side

| model | \(L\) | \(d\) | \(H_q/H_{kv}\) | GQA | \(f/d\) | \(V\) | tied | KV / token | params |
|---|---:|---:|---:|---:|---:|---:|:--:|---:|---:|
| Llama-3.1-8B | 32 | 4096 | 32/8 | 4:1 | 3.5 | 128256 | no | 128 KiB | 8.03B |
| Mistral-Small-24B | 40 | 5120 | 32/8 | 4:1 | **6.4** | 131072 | no | 160 KiB | 23.57B |
| Phi-4 | 40 | 5120 | 40/10 | 4:1 | 3.5 | 100352 | no | 200 KiB | 14.66B |
| DeepSeek-R1-14B | **48** | 5120 | 40/8 | 5:1 | 2.7 | 152064 | no | 192 KiB | 14.77B |
| Granite-3.3-8B | 40 | 4096 | 32/8 | 4:1 | 3.1 | 49159 | **yes** | 160 KiB | 8.17B |
| Yi-1.5-9B | **48** | 4096 | 32/**4** | **8:1** | 2.7 | 64000 | no | **96 KiB** | 8.83B |

All KV figures reproduce `inspect_architecture.py`'s `@8192` output when
multiplied by 8192 (e.g. Yi 96 KiB × 8192 = 0.81 GB). Reproduce any row yourself:

```bash
python scripts/inspect_architecture.py --hf <repo-id> --seq 8192
```

---

## The same eight steps, parameterised

The workflow from [`appendix-c-runbook.md`](appendix-c-runbook.md), with the
model as a variable. Pick a row and set:

```bash
export MODEL="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
export TAG="llama31-8b"          # short name for the adapter/fused dirs
export NUM_LAYERS=-1             # -1 = all layers; use 16 (or 8) for Mistral-24B
export ARGS_FORMAT="string"      # "dict" for the Qwen2-based DeepSeek distill
```

```bash
# 1. download
hf download "$MODEL"

# 2. verify adapter targets (no training yet) — expect "OK: adapter covers ..."
python scripts/check_lora.py --model "$MODEL" --num-layers $NUM_LAYERS --rank 16 --scale 2

# 3. prepare data (normalises null tool-call content so every template renders)
python scripts/prepare_dataset.py data/interim/seed.jsonl --out data/processed --arguments-format $ARGS_FORMAT
python scripts/validate_dataset.py data/processed/train.jsonl --tools examples/tools/kali-tools.json

# 4. ONE-ITERATION TEMPLATE CHECK — do this before any long run
cp configs/qwen3-14b-all-layers.yaml configs/$TAG.yaml
#    edit: model: $MODEL · num_layers: $NUM_LAYERS · adapter_path: adapters/$TAG
mlx_lm.lora --config configs/$TAG.yaml --iters 1 --save-every 1

# 5. train
mlx_lm.lora --config configs/$TAG.yaml

# 6. fuse
mlx_lm.fuse --model "$MODEL" --adapter-path adapters/$TAG --save-path fused/$TAG

# 7. serve (loopback only)
mlx_lm.server --model fused/$TAG --host 127.0.0.1 --port 8080 --max-tokens 512 --temp 0.2

# 8. smoke-test + eval
python scripts/smoke_test.py   --base-url http://127.0.0.1:8080/v1 --model fused/$TAG \
  --tools-file examples/tools/kali-tools.json
python scripts/eval_toolcall.py --base-url http://127.0.0.1:8080/v1 --model fused/$TAG \
  --golden data/examples/golden.example.jsonl --tools-file examples/tools/kali-tools.json \
  --out reports/$TAG.json
```

**Step 4 is the one that catches family-specific breakage.** A null-content
tool-call turn crashes the Qwen3/Gemma chat template with
`TypeError: argument of type 'NoneType' is not iterable`; `prepare_dataset.py`
already normalises that to `""`, but run one iteration and confirm a finite loss
and a rendered prompt that carries the tool call before spending hours.

---

## What to expect

With base weights plus only a short adapter, tool-call pass rates are low — that
is the *untrained* baseline, and the harnesses returning a nonzero exit on
failures is them working, not breaking. A real adapter over 2–5M tokens
([`04-data-strategy.md`](04-data-strategy.md)) moves the numbers. Compare
families on your own 240-task golden set the way
[`31-gemma-vs-qwen.md`](31-gemma-vs-qwen.md) compares Gemma and Qwen: same data,
same gates, honest CIs.

**Bottom line for 32 GB:** **Granite 3.3 8B** and **Llama 3.1 8B** are the
easiest ports (native tool calls, big headroom); **DeepSeek-R1-Distill-14B** adds
reasoning at the cost of a `<think>` parser; **Phi-4** and **Yi-1.5-9B** need the
tool format trained in; **Mistral-Small-24B** is the strongest of the set but
trains only partially and shines as a serve/teacher model.

---

## Sources

- [mlx-community/Meta-Llama-3.1-8B-Instruct-4bit](https://huggingface.co/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit)
- [mlx-community/Mistral-Small-24B-Instruct-2501-4bit](https://huggingface.co/mlx-community/Mistral-Small-24B-Instruct-2501-4bit)
- [mlx-community/phi-4-4bit](https://huggingface.co/mlx-community/phi-4-4bit)
- [mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit](https://huggingface.co/mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit)
- [mlx-community/granite-3.3-8b-instruct-4bit](https://huggingface.co/mlx-community/granite-3.3-8b-instruct-4bit) · [IBM Granite docs](https://www.ibm.com/granite/docs/models/granite4-1)
- [mlx-community/Yi-1.5-9B-Chat-4bit](https://huggingface.co/mlx-community/Yi-1.5-9B-Chat-4bit)
- [mlx-lm](https://github.com/ml-explore/mlx-lm) · [vLLM-MLX tool-call parsers](https://vllm-mlx.is-a.dev/guides/tool-calling/)
