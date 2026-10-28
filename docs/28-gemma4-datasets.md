# 28 — Gemma 4 12B datasets

*Last verified: 25 October 2026. Sizes and licenses were checked against the Hugging Face API and dataset cards; re-check before a large download.*

The raw material is largely the same as the Qwen tracks. What changes is the **template**, the **tool-call format**, and the option of using Gemma as a **vision model**.

This chapter is the Gemma-specific shopping list. The full catalogs are:

- [05 — Datasets](05-datasets.md) — general tool calling, agentic, cyber.
- [19 — Java datasets](19-java-datasets.md) — Java/Quarkus.
- [23 — React datasets](23-react-datasets.md) — React/Tailwind, design-to-code.
- [07 — Synthetic data](07-synthetic-data.md), [21 — Java synthetic](21-java-synthetic.md), [25 — React synthetic](25-react-synthetic.md).

---

## 1. What changes for Gemma

| Topic | Qwen | Gemma 4 12B |
|---|---|---|
| Tool arguments | Qwen3: JSON **string**; Qwen2.5-Coder: object | **Object/mapping** |
| Tool call syntax | `<tool_call>{"name":...}</tool_call>` | `<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>` |
| Tool result | `role: tool`, id ignored by Qwen3 template | `role: tool`, id resolved to name |
| Thinking | `<think>…</think>` | `<|channel>thought…<channel|>` |
| Thinking history | final assistant may include think block | **drop previous thoughts**, except tool-call turns |
| Vision | no (text model) | **yes**, but text `mlx_lm` strips vision weights |
| Context | 32K (Qwen3) / 128K (Coder) | **256K** |

The canonical data shape is still `messages` + optional `tools`. The template does the rendering. Do not store Gemma-rendered strings; they are brittle and impossible to re-template.

---

## 2. Text datasets that are worth using

### 2.1 Raw code

| Dataset | Size | License | Gemma use |
|---|---:|---|---|
| [HuggingFaceCode/stack-v3-train](https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train) | ~3.6T tokens / 15.9 TB | ODC-BY | seed repos; filter TS/TSX/Java |
| [bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) | 250B tokens / 86 languages | gated; per-file | optional; older |
| [code_search_net](https://huggingface.co/datasets/code_search_net) | 1.86M JS functions | repo MIT; data `other` | docstring→code; low priority |

Do not train raw files as SFT. Sample repositories and generate tasks with a verifier.

### 2.2 Code instructions

| Dataset | Size | License | Gemma use |
|---|---:|---|---|
| [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | 5M / ~19 GB | **CC-BY-4.0** | large instruction base; filter Java/TS/React |
| [ise-uiuc/Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | 75K | MIT (GPT-3.5 output; OpenAI policy) | diversity |
| [ise-uiuc/Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K) | 110K | Apache-2.0 | harder tasks |
| [cfahlgren1/react-code-instructions](https://huggingface.co/datasets/cfahlgren1/react-code-instructions) | 74,428 / ~907 MB | **MIT** | complete-app shape; React 18/Tailwind 3 era |

Rebuild and retag React examples to React 19/Tailwind 4; do not train the raw 2024-era apps as if they were current.

### 2.3 Design and visual

| Dataset | Size | License | Gemma use |
|---|---:|---|---|
| [HuggingFaceM4/WebSight](https://huggingface.co/datasets/HuggingFaceM4/WebSight) v0.2 | 1,922,671 / ~285 GB | **CC-BY-4.0** | screenshot + HTML/Tailwind; text subset for HTML/Tailwind style |
| [SALT-NLP/Design2Code](https://huggingface.co/datasets/SALT-NLP/Design2Code) | 484 / ~69 MB | **ODC-BY** | evaluation only; screenshot→HTML |
| [SALT-NLP/Design2Code-hf](https://huggingface.co/datasets/SALT-NLP/Design2Code-hf) | 484 | ODC-BY | HF-format version |
| [princeton-nlp/SWE-bench_Multimodal](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Multimodal) | 480 test + 100 dev | no dataset license tag; source repos | eval only; screenshots/mockups |
| [Roy-DOU/Webdev-Bench](https://huggingface.co/datasets/Roy-DOU/Webdev-Bench) | 40 tasks / 1,125 checks | **MIT** | eval only; multi-round front-end |

**Gemma’s vision advantage:** you can use WebSight/Design2Code/SWE-bench Multimodal as **image input** with `mlx-vlm`, not just text. But a text-only `mlx_lm.lora` run strips the vision tower; those images are then only references for a written spec.

### 2.4 Java/Quarkus

Reuse the sources in [19 — Java datasets](19-java-datasets.md):

- The Stack v3 Java subset.
- OpenCodeInstruct and Magicoder Java rows.
- Methods2Test for test generation.
- SWE-smith-java, Multi-SWE-bench Java, SWE-bench Multilingual Java for repair.
- Quarkus quickstarts and Quarkus repos for framework idioms.
- Maven/ArchUnit verification.

Gemma-specific filter: compile with **Java 25**, Quarkus **3.40 LTS**, `jakarta.*`, `quarkus-rest`; reject `javax.*`, Spring, legacy RESTEasy.

### 2.5 Kali/agent

Reuse [05 — Datasets](05-datasets.md):

- general tool calling: xLAM, ToolACE, Hermes, Toucan;
- agentic: SWE-smith, Nemotron Terminal, OpenThoughts;
- cyber: KaliBench train, Trendyol cyber QA, filtered security QA;
- refusal/scope: When2Call and your own scope dialogues.

Gemma’s tool parser accepts `\w-` names and native calls. Test tool-call validity on your exact tool schema before trusting a large Kali mix.

---

## 3. The Gemma row shape

```json
{
  "messages": [
    {"role": "system", "content": "MODE: REACT\nYou are ..."},
    {"role": "user", "content": "Add a StatCard ..."},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_1", "type": "function",
       "function": {"name": "read_file", "arguments": {"path": "src/components/ui/StatCard.tsx"}}}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "..."},
    {"role": "assistant", "content": "```tsx\n...\n```"}
  ],
  "tools": [
    {"type": "function", "function": {
      "name": "read_file", "description": "Read a file.",
      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    }}
  ],
  "meta": {"track": "react", "license": "MIT", "test_command": "npm run typecheck && npm test -- --run"}
}
```

Rules:

- `arguments` is an **object**; a string is a hard error in the Gemma template.
- Tool call `id` is required if a tool result follows.
- `tool_call_id` must match a preceding call.
- `content` can be `null` on a tool-call assistant turn.
- `reasoning_content` is optional; it renders as a thinking channel.
- Keep only the current turn’s thinking; drop previous thoughts except tool-call turns.
- The first message should be `system` or `developer`.
- Do not include Qwen’s `<tool_call>` JSON in content.

---

## 4. Data mix

### 4.1 Kali

| Bucket | Share | Source |
|---|---:|---|
| General tool calling | 35–40% | xLAM, ToolACE, Hermes, Toucan |
| Multi-turn/agentic | 15–20% | SWE-smith, Nemotron Terminal, synthetic |
| Cyber QA + Kali CLI | 20% | KaliBench train, Trendyol, filtered QA |
| Verified synthetic trajectories | 15% | DeepSeek/OpenCode + lab |
| Refusal/scope | 10% | When2Call, private scope dialogues |

Target 2–4M tokens for the first adapter. Do not train on BFCL, KaliBench test, Cybench, NYU CTF or CyberMetric.

### 4.2 Java

Reuse [19 — Java datasets](19-java-datasets.md) §8. Target 2–4M tokens:

- 30% repository edits/repair;
- 20% feature implementation;
- 15% tests;
- 15% review/architecture;
- 10% migration/refactor;
- 10% negative/security.

### 4.3 React

Reuse [23 — React datasets](23-react-datasets.md) §8. Target 2–4M tokens:

- 20% components/primitives;
- 15% pages/layouts;
- 15% data/forms;
- 10% design system;
- 10% accessibility;
- 10% testing;
- 10% bug fix;
- 5% migration;
- 5% negative/security.

### 4.4 Unified

40% Kali / 30% Java / 30% React, plus a mode token. Use [31 — Running both](31-gemma-vs-qwen.md) for the comparison method.

---

## 5. Gemma-specific quality gates

Run these in addition to the filters in [23 — React datasets](23-react-datasets.md) §9 and [19 — Java datasets](19-java-datasets.md) §9:

- [ ] `arguments` is a mapping on every tool call.
- [ ] Every tool call id is unique and every tool result resolves.
- [ ] Tool names are `[A-Za-z_][A-Za-z0-9_-]*`.
- [ ] No Qwen-style `<tool_call>` JSON in assistant content.
- [ ] No previous-turn thinking, except tool-call turns.
- [ ] Render one row with the Gemma tokenizer; confirm `<|tool_call>call:` appears.
- [ ] Render one row with tools and confirm the system/tool block is well formed.
- [ ] `mlx_lm.server` returns `tool_calls`, not raw text in `content`.
- [ ] Text-only rows contain no image placeholders.
- [ ] The source license is recorded; no unlicensed/branded seeds.
- [ ] Token p99 fits `max_seq_length` (2,048 or 3,072), not 256K.
- [ ] 13-gram overlap against Webdev-Bench, Design2Code, SWE-bench Multilingual/Multimodal and Multi-SWE-bench.

The static validator is:

```bash
python scripts/validate_gemma4_dataset.py data/gemma4/react/processed/train.jsonl \
  --tools examples/react/tools.json --strict
```

It catches string arguments, broken ids, undeclared tools, Qwen call syntax, secrets and missing metadata. It does not compile or render; the real verifier does.

---

## 6. Multimodal data (separate track)

If you want Gemma to consume screenshots directly:

1. Use the multimodal checkpoint (`google/gemma-4-12B-it` or an MLX-VLM build).
2. Build rows with image + text content, not text-only `messages`.
3. Use `mlx-vlm` training, not `mlx_lm.lora`.
4. Hold out Design2Code/WebSight and evaluate rendered similarity + human/VLM pairwise.
5. Do not mix multimodal rows into a text adapter; the weights and template differ.

A practical middle ground: use Gemma as a **VLM spec writer** to turn screenshots into `design.md`, then train the text adapter on spec → code. That keeps the training path simple and gives the text coder a fair input.

---

## 7. Licensing

Same rules as the Qwen tracks:

- The Stack v3: ODC-BY; per-file licenses.
- OpenCodeInstruct: CC-BY-4.0.
- Magicoder: MIT; GPT-3.5 output subject to OpenAI policy.
- react-code-instructions: MIT; generated apps may imitate brands.
- WebSight: CC-BY-4.0; synthetic with real images; check provenance.
- Design2Code: ODC-BY; evaluation only.
- Webdev-Bench: MIT; evaluation only.
- SWE-bench Multilingual: MIT; evaluation only.
- Multi-SWE-bench: `other`; source repos have their own licenses.
- SWE-bench Multimodal: no dataset license tag; source repos; evaluation only.
- shadcn/ui, Tailwind, React, Radix, TanStack, Storybook: MIT.
- Playwright: Apache-2.0.
- Gemma 4 weights: Apache-2.0.

Do not train on proprietary code, customer data, paid component libraries or real brand sites without permission. This is not legal advice.

---

## 8. Sources

- The Stack v3 — <https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train>
- OpenCodeInstruct — <https://huggingface.co/datasets/nvidia/OpenCodeInstruct>
- Magicoder — <https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K>
- React Code Instructions — <https://huggingface.co/datasets/cfahlgren1/react-code-instructions>
- WebSight — <https://huggingface.co/datasets/HuggingFaceM4/WebSight>
- Design2Code — <https://huggingface.co/datasets/SALT-NLP/Design2Code>
- Webdev-Bench — <https://huggingface.co/datasets/Roy-DOU/Webdev-Bench>
- SWE-bench Multilingual — <https://huggingface.co/datasets/SWE-bench/SWE-bench_Multilingual>
- Multi-SWE-bench — <https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench>
- SWE-bench Multimodal — <https://huggingface.co/datasets/princeton-nlp/SWE-bench_Multimodal>
- shadcn/ui — <https://github.com/shadcn-ui/ui>
- [05 — Datasets](05-datasets.md)
- [19 — Java datasets](19-java-datasets.md)
- [23 — React datasets](23-react-datasets.md)
