# 27 — Gemma 4 12B: the alternative track

*Last verified: 25 October 2026 against the Gemma 4 model card, the Hugging Face config and chat template, MLX-LM 0.32.2, and the MLX community 4-bit checkpoint.*

This is the start of a **separate Gemma 4 12B track**. It does not replace the Qwen tracks. The point is to run both:

- Qwen3-14B for the Kali/agent track;
- Qwen2.5-Coder-14B for the specialised Java and React adapters;
- **Gemma 4 12B** as a second base for comparison, and as the strongest choice for vision-heavy React work.

The Qwen files, configs, adapters and docs stay exactly as they are.

---

## 1. What Gemma 4 12B is

| Property | Gemma 4 12B |
|---|---|
| Release | 3 June 2026; model card current 30 July 2026 |
| Parameters | **11.95B** dense |
| Layers | **48** |
| Hidden / intermediate | 3,840 / 15,360 |
| Attention | 16 query heads / 8 KV heads; hybrid sliding-window + global attention |
| Context | **256K** native |
| Vocabulary | 262,144 |
| Modalities | Text, image, audio in; text out; encoder-free “Unified” architecture |
| Tool calling | Native function calling, `<|tool_call>` format |
| Thinking | `<|channel>thought … <channel|>`; enabled by `<|think|>` |
| License | **Apache-2.0** |
| MLX 4-bit | `mlx-community/gemma-4-12b-it-4bit`, ~6.6–7.6 GB |

“Unified” means Gemma 4 12B has **no separate vision or audio encoder**. Raw image patches and audio waveforms are projected directly into the decoder’s embedding space through linear layers. That is elegant for inference and fine-tuning, but it also means the text-only MLX loader strips the vision/audio weights; image training is a separate `mlx-vlm` path.

The architecture is a hybrid: most layers use a 1,024-token sliding window, with periodic full-attention layers; global layers use p-RoPE and unified K/V. For LoRA this matters because full-attention layers omit a separate `v_proj`. Do not assume every block has the same seven projections.

---

## 2. Gemma 4 12B vs the Qwen bases

| | Gemma 4 12B | Qwen3-14B | Qwen2.5-Coder-14B |
|---|---:|---:|---:|
| Parameters | 11.95B | 14.8B | 14.7B |
| Layers | 48 | 40 | 48 |
| Native context | **256K** | 32K (131K YaRN) | 128K |
| Vision | **Yes** | No | No |
| Tool calling | Native, custom syntax | `<tool_call>` JSON | `<tool_call>` JSON |
| Thinking | Channel-based | `<think>` | None |
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| Local 4-bit QLoRA | **Yes, 24/48** | Yes, 24/40 | Yes, 24/48 |
| Best role | React/vision, A/B Java | Kali/agent | Java/React code specialist |

**Do not read this table as “Gemma wins”.** It has newer training, more context and vision. Qwen2.5-Coder is code-specialised and its tool format is simpler. The only reliable comparison is your own golden set.

---

## 3. Why use it

### 3.1 React and design-to-code

Gemma 4 12B can see a screenshot. That is a genuine capability Qwen2.5-Coder lacks. It opens two options:

1. **Text-only LoRA** — train on `design.md → React/Tailwind`; use Gemma as a VLM to produce the design spec, then the adapter writes the code.
2. **Multimodal LoRA** — train image → code with `mlx-vlm` and a separate dataset. This is more complex and not covered by the text `mlx_lm.lora` configs here.

Even the text-only path benefits from the 256K context: a page, its components, tests and a design brief fit more comfortably.

### 3.2 Java/Quarkus

Gemma 4 12B is a newer, strong coding base. It may beat Qwen2.5-Coder on some tasks and lose on others. The Quarkus target is unchanged: **Java 25 LTS + Quarkus 3.40 LTS**, verified by Maven and ArchUnit.

### 3.3 Kali/agent

Gemma 4 has native function calling and a real improvement over Gemma 3. But Qwen3 has the stronger documented tool-calling record in this handbook, and Gemma’s tool format is unusual (`call:name{key:<|"|>value<|"|>}`). Treat Kali as an A/B experiment, not a default switch.

### 3.4 One base for all

Gemma 4 12B is a credible single base: Apache-2.0, 12B, 256K, vision, native tools. The trade-off is the same as with Qwen: **one base + three adapters** beats one mixed adapter for quality. See [31 — Running both and comparing](31-gemma-vs-qwen.md).

---

## 4. What fits on 32 GB

These are planning estimates. Measure with `mlx_lm.lora` and `mx.get_peak_memory()`.

| Setup | Weights | Adapter/optimizer | Peak estimate |
|---|---:|---:|---:|
| Gemma 4 12B 4-bit, inference | ~6.6–7.6 GB | — | 10–14 GB at 8K |
| 24/48 blocks, seq 2,048 | ~6.6–7.6 GB | ~0.3 GB | **16–22 GB** |
| 24/48 blocks, seq 3,072 | ~6.6–7.6 GB | ~0.3 GB | **20–26 GB** |
| 16/48 blocks, seq 4,096 | ~6.6–7.6 GB | ~0.2 GB | **20–26 GB** |
| all 48 blocks, seq 2,048 | ~6.6–7.6 GB | ~0.6 GB | **22–28 GB** |

Start at **24/48, seq 2,048**. If it is comfortable, raise sequence before layers. If it OOMs, drop to 16/48 or 2,048; do not start by shrinking the dataset.

`mlx_lm.lora` uses the last N blocks. All 48 is an experiment, not the first run.

### 4.1 Expected trainable parameters

The exact number depends on the MLX version and on `v_proj` being absent in full-attention layers:

- 24/48, rank 16, `scale: 2.0`: roughly **30–35M** trainable parameters.
- all 48: roughly **60–70M**.
- modules: about **5–7 per block**, not always 7, because of K=V in full-attention layers.

Run `scripts/check_lora.py` on the Mac. If it reports fewer than four modules per adapted block, stop.

---

## 5. Tool format: the part that breaks naive ports

Gemma 4 does **not** use Qwen’s JSON tool-call format.

Qwen3:

```text
<tool_call>{"name":"read_file","arguments":{"path":"src/App.tsx"}}</tool_call>
```

Gemma 4:

```text
<|tool_call>call:read_file{path:<|"|>src/App.tsx<|"|>}<tool_call|>
```

Consequences:

1. **Tool arguments must be JSON objects (mappings)** in the canonical messages. A string makes the Gemma template raise:
   `tool_calls[].function.arguments must be a JSON object (mapping), not a string`.
2. Use `prepare_dataset.py --arguments-format dict`.
3. Tool results use `role: "tool"` with a `tool_call_id`; Gemma resolves that id to the function name.
4. The template emits native `<|tool_call>` blocks; do not put Qwen-style `<tool_call>` JSON in assistant content.
5. Tool names should match `[A-Za-z_][A-Za-z0-9_-]*`.

Modern `mlx-lm` includes a Gemma 4 parser (`mlx_lm/tool_parsers/gemma4.py`). The old failure mode — tool calls landing in `content` with an empty `tool_calls` array — is fixed, but verify on your exact version:

```bash
curl -s http://127.0.0.1:8083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-12b-kali","messages":[{"role":"user","content":"Read src/App.tsx"}],"tools":[...]}' |
  python3 -m json.tool
```

If `tool_calls` is empty, use `llama-server --jinja` or update `mlx-lm`; do not train an agent on unparsed tool text.

---

## 6. Thinking mode

Gemma 4 uses a channel:

```text
<|channel>thought
...reasoning...
<channel|>final answer
```

- Thinking is enabled by `<|think|>` at the start of the system prompt; `enable_thinking=True` in the template adds it.
- The canonical MLX `mlx_lm.lora` path applies the tokenizer template with its default; for Gemma 4 that is **thinking off**.
- With thinking disabled, Gemma 4 still emits an **empty** thought block. Do not mix a no-think adapter with a template that injects a different block.
- `reasoning_content` in a message is rendered as the thinking channel.
- **Do not keep previous turns’ thinking in history**, except in tool-call turns. Gemma 4 expects historical assistant messages to contain final answers.

The practical rule: train and serve with the same thinking setting. The configs here target the default **no-think** path; if you want a thinking adapter, verify the template first and keep a separate adapter.

---

## 7. Vision and audio are separate

`mlx_lm` is the text path. Its Gemma 4 loader strips `vision_tower`, `multi_modal_projector`, `audio_tower` and related weights. That is fine for text-only LoRA.

For image or audio:

- use `mlx-vlm` and the multimodal Gemma checkpoint;
- train a separate adapter, with image content in the messages;
- do not put images into a `mlx_lm.lora` row;
- expect a different memory profile and a slower run.

The text-only React adapter can still use Gemma as a **captioner** or **design-spec writer**: render a screenshot, ask for a structured `design.md`, then train the coder on text → code.

---

## 8. The track layout

```text
docs/27-gemma4-12b.md          this chapter
docs/28-gemma4-datasets.md     datasets, licenses, Gemma-specific filters
docs/29-gemma4-finetuning.md   full MLX runbook
docs/30-gemma4-synthetic.md    generating verified Gemma data
docs/31-gemma-vs-qwen.md       running both and comparing
configs/gemma4-12b-*.yaml      4 configs
data/gemma4/                   per-track data layout
examples/gemma4/               prompts, modes, OpenCode config
scripts/validate_gemma4_dataset.py
scripts/compare_models.py
```

The Qwen directories are untouched. You can run the same task against both stacks and compare.

---

## 9. Recommendation

| Goal | Choice |
|---|---|
| **React/Tailwind with screenshots or long context** | **Gemma 4 12B** first |
| **Java/Quarkus** | A/B Gemma vs Qwen2.5-Coder on the same Maven/ArchUnit set |
| **Kali/agent** | Keep Qwen3-14B until Gemma proves better tool reliability |
| **One base for all three** | Gemma 4 12B is credible; Qwen3-14B is the safer agent base |
| **Best local training speed** | Gemma 4 12B is smaller than Qwen3-14B; measure tokens/sec |
| **Vision training** | Separate `mlx-vlm` project, not `mlx_lm.lora` |

Start with [28 — Gemma datasets](28-gemma4-datasets.md), then [29 — Gemma fine-tuning](29-gemma4-finetuning.md). Compare with [31 — Running both](31-gemma-vs-qwen.md).

---

## 10. Sources

- Gemma 4 model card — <https://ai.google.dev/gemma/docs/core/model_card_4>
- Gemma 4 12B developer guide — <https://developers.googleblog.com/en/gemma-4-12b-the-developer-guide>
- `google/gemma-4-12B-it` — <https://huggingface.co/google/gemma-4-12B-it>
- Gemma 4 chat template — <https://huggingface.co/google/gemma-4-12B-it/blob/main/chat_template.jinja>
- Gemma 4 config — <https://huggingface.co/google/gemma-4-12B-it/blob/main/config.json>
- Gemma 4 function calling — <https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4>
- MLX 4-bit checkpoint — <https://huggingface.co/mlx-community/gemma-4-12b-it-4bit>
- MLX Gemma 4 text loader — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4.py>
- MLX Gemma 4 tool parser — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/gemma4.py>
- MLX-LM LoRA — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md>
- [18 — Java landscape](18-java-quarkus.md)
- [22 — React & Tailwind](22-react-tailwind.md)
