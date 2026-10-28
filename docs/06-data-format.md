# 06 — Data format: the contract between dataset, tokenizer and server

*Last verified: 25 October 2026 against `mlx-lm` 0.32.2, `transformers` 5.x, and the Qwen3/Qwen3.5 tokenizer templates.*

Most failed fine-tunes are not optimization failures. They are **format failures**: the model was trained on one tool-call syntax and served with another, the `arguments` field was a Python dict where the template expected a JSON string, or `--mask-prompt` silently trained on nothing because the example was one token too long.

This chapter defines the one format this repository uses, explains how each Qwen generation renders it, and lists the checks to run before a long training job.

---

## 1. The canonical format

One JSON object per line:

```json
{"messages": [...], "tools": [...]}
```

- `messages` is an OpenAI-style chat array: `system`, `user`, `assistant`, `tool`.
- `tools` is optional. When present, it is an array of OpenAI function schemas.
- Assistant tool calls use `tool_calls`; tool results use `role: "tool"`.
- Extra keys are ignored by MLX-LM but are preserved by our scripts for provenance (`id`, `source`, `license`, `split`).

This is exactly the format MLX-LM's `ChatDataset` passes to `tokenizer.apply_chat_template(messages, tools=tools)`. It is also the format the OpenAI-compatible servers accept, which minimizes the train/serve mismatch.

### A complete example

```json
{
  "messages": [
    {"role": "system", "content": "You are a careful security assistant. Only act inside the authorized scope."},
    {"role": "user", "content": "Enumerate the open TCP services on 10.10.10.5."},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_1", "type": "function", "function": {
        "name": "nmap_scan",
        "arguments": "{\"target\":\"10.10.10.5\",\"flags\":\"-sV\"}"
      }}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "22/tcp open ssh OpenSSH 9.6\n80/tcp open http nginx 1.26"},
    {"role": "assistant", "content": "Two TCP services are open: SSH on 22 and HTTP on 80. The HTTP banner is nginx 1.26; I would enumerate virtual hosts next, within scope."}
  ],
  "tools": [
    {"type": "function", "function": {
      "name": "nmap_scan",
      "description": "Run an nmap scan against a target that is explicitly in the authorized scope.",
      "parameters": {
        "type": "object",
        "properties": {
          "target": {"type": "string", "description": "IPv4 address, hostname, or CIDR inside scope."},
          "flags": {"type": "string", "description": "nmap flags, e.g. -sV or -sC."}
        },
        "required": ["target"]
      }
    }}
  ]
}
```

### Rules

1. **One assistant turn per example is the training unit.** A trajectory with five assistant turns becomes five examples (prefix expansion), because `--mask-prompt` only computes loss on the final assistant message. See §8.
2. **Every `tool` message must follow an assistant message with a matching tool call.** Qwen templates ignore `tool_call_id`, but the server and your validator should not.
3. **`arguments` is a JSON object encoded as a string** by default (OpenAI convention). **Qwen2.5-Coder is the exception:** its template applies `tojson` to `arguments`, so it must be a JSON **object**, not a string. Store the form your target template expects; `prepare_dataset.py --arguments-format dict` handles Qwen2.5-Coder.
4. **`content` may be `null`** on an assistant message that only calls tools. The Qwen3 template handles `None` by rendering an empty string.
5. **Do not put `<think>` tags in `content`.** Use `reasoning_content` (see §6). Inline tags cause the Qwen3 template to double-open the think block.
6. **Tool names must match `[A-Za-z0-9_-]+`.** Spaces and quotes break XML templates (Qwen3.5/3.6/Coder) and tool parsers.
7. **Every tool in a call must exist in `tools`.** Hallucinated tool names are the most common failure after format errors.

---

## 2. Tool schema rules

The schema is ordinary JSON Schema, but keep it small and explicit:

```json
{"type": "function", "function": {
  "name": "http_probe",
  "description": "Fetch headers from an HTTP(S) URL in scope. Use for banner and technology identification.",
  "parameters": {
    "type": "object",
    "properties": {
      "url": {"type": "string", "format": "uri"},
      "method": {"type": "string", "enum": ["GET", "HEAD"], "default": "HEAD"},
      "timeout_s": {"type": "integer", "minimum": 1, "maximum": 30, "default": 10}
    },
    "required": ["url"],
    "additionalProperties": false
  }
}}
```

Guidance:

- **Description is part of the prompt.** Say when to use the tool, when not to, and what the arguments mean. A 14B model relies on it more than on the name.
- **Prefer enums and bounded integers.** Free-form strings invite invalid flags.
- **Do not expose 50 tools in one request.** A 14B selects better among 6–10. Group capabilities behind a few typed tools (`kali_exec`, `http_probe`, `nmap_scan`, `report_finding`) rather than one tool per binary.
- **`additionalProperties: false`** where possible; it makes validation deterministic.
- **No secrets in defaults.** Never put credentials in a schema.

---

## 3. How Qwen3 renders chat + tools

Qwen3-8B and Qwen3-14B use the classic Qwen template (both tokenizer configs are byte-identical as of 25 Oct 2026). Key behaviour, verified from `tokenizer_config.json`:

1. If `tools` is present, the **first system message** is merged into a system block that begins with `# Tools`, then a `<tools>` XML wrapper containing one JSON object per tool, then a `<tool_call>` instruction block.
2. If there is no system message, the block is still created; the model gets the tool instructions without your persona. Always include a system message.
3. Assistant tool calls render as:

```text
<tool_call>
{"name": "nmap_scan", "arguments": {"target": "10.10.10.5", "flags": "-sV"}}
</tool_call>
```

4. Tool results render inside a **user** turn:

```text
<|im_start|>user
<tool_response>
22/tcp open ssh
</tool_response><|im_end|>
```

5. Consecutive `tool` messages are grouped into one user turn.
6. `tool_call_id` is ignored by the template.
7. The final assistant message is always prefixed with a think block:

```text
<|im_start|>assistant
<think>

</think>

Two TCP services are open...
```

Intermediate assistant messages with no `reasoning_content` do **not** get the think block. This asymmetry is important when you split trajectories.

### A rendered Qwen3 prompt (abridged)

```text
<|im_start|>system
You are a careful security assistant. Only act inside the authorized scope.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "nmap_scan", ...}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Enumerate the open TCP services on 10.10.10.5.<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "nmap_scan", "arguments": {"target": "10.10.10.5", "flags": "-sV"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
22/tcp open ssh
80/tcp open http
</tool_response><|im_end|>
<|im_start|>assistant
<think>

</think>

Two TCP services are open...
<|im_end|>
```

The tool schema is one line per tool in the real template (`tojson`), not pretty-printed. The example above is wrapped for readability.

---

## 4. How Qwen3.5 / 3.6 / Coder render tools

Qwen3.5-9B and Qwen3.6-35B-A3B use an **XML function-call** format:

```text
<tool_call>
<function=nmap_scan>
<parameter=target>
10.10.10.5
</parameter>
<parameter=flags>
-sV
</parameter>
</function>
</tool_call>
```

Qwen3-Coder-30B-A3B-Instruct and Qwen3-Coder-Next use the same XML call style; the Coder template also nests `<function>`, `<name>`, `<description>`, `<parameters>` blocks in the tool definitions. Qwen3.6 adds a `preserve_thinking` flag; Qwen3-Coder models do not use thinking.

Implications for your JSONL:

- `arguments` should be a **dict** when targeting these models, so the template can iterate key/value pairs. If your pipeline stores an OpenAI JSON string, parse it before rendering (our `prepare_dataset.py` does this).
- Tool names with spaces or punctuation break the XML tags. Normalize.
- `tool_call_id` is still ignored.
- Tool responses are still placed in a user message inside `<tool_response>`.

If you train Qwen3-14B (the local default), you do not need this syntax. If you later move to Qwen3.5/3.6, **re-render the entire dataset**; do not mix the two call styles in one adapter.

### 4.1 Qwen2.5-Coder: ChatML with JSON calls

Qwen2.5-Coder uses ChatML and the same JSON `<tool_call>` shape as Qwen3, but with one important difference: its template renders `tool_call.arguments | tojson`.

- **Store `arguments` as an object**, not a JSON string. A string becomes an escaped string in the prompt and teaches the wrong shape.
- There is no thinking block and no `reasoning_content`.
- Tool results still use `<tool_response>` inside a user turn; `tool_call_id` is ignored.
- The template supports `tools`, `tool_calls` and `tool` roles.

```bash
python scripts/prepare_dataset.py data/java/interim/all.jsonl \
  --out data/java/processed --arguments-format dict
```

### Model → call syntax

| Model family | Tool definitions | Tool call | Thinking |
|---|---|---|---|
| Qwen3-8B/14B/32B | JSON in `<tools>` | JSON in `<tool_call>` | `reasoning_content`; empty think block for final assistant |
| Qwen3.5-4B/9B | JSON in `<tools>` | XML `<function=...>` | `enable_thinking` supported |
| Qwen3.6-35B-A3B | JSON in `<tools>` | XML | adds `preserve_thinking` |
| Qwen3-Coder-30B-A3B | XML `<function><name>...` | XML | no thinking |
| Qwen3-Coder-Next | XML | XML | no thinking |
| Qwen2.5-Coder-7B/14B/32B | JSON in `<tools>` | JSON in `<tool_call>`; **arguments object** | none |

---

## 5. System prompt and scope

The system message should be short, stable, and identical between training and serving. A good template:

```text
You are a security operations assistant running inside an authorized penetration-testing lab.

Rules:
- Only act on targets in the scope provided by the user or the scope file.
- If the target is missing, ambiguous, public, or out of scope, ask for clarification or refuse.
- Prefer read-only reconnaissance. Never run destructive commands.
- Call at most one tool per turn unless the tools are independent and safe to parallelize.
- After each tool result, summarize what changed in your plan.
- If a tool fails, diagnose the failure before retrying; do not repeat the same call.
- Cite the command and the observed output in your final answer.
```

Do **not** train with a different system prompt for every row unless the variation is deliberate (e.g., scope/no-scope pairs). At serving time you will paste one prompt; if the model only ever saw another, behavior drifts.

---

## 6. Thinking and `reasoning_content`

Qwen3 supports a reasoning block. The template behaviour:

- If an assistant message has `reasoning_content`, it is rendered inside `<think>…</think>` before the response.
- If `content` contains `</think>`, the template splits it, but if the content *starts* with `<think>`, you get a double-open (`<think>\n thinking...`). This is a real, verified pitfall.
- The final assistant message always gets a think block, even when `reasoning_content` is empty.
- If you pass `enable_thinking: false` to `apply_chat_template` at serving time, the prompt ends with a **closed empty think block**. A model trained with the default template (which expects to generate `<think>`) may then emit a second think block. Train and serve with the same setting.

**Practical policy for this handbook:**

1. Keep thinking **enabled** during training and serving. The final assistant will emit `<think>\n\n</think>\n\n` before content; that is a few tokens of overhead, not a failure.
2. If you need a no-think adapter, patch the tokenizer's chat template to force the no-think branch *before training* and use the same template at serving. Do not simply pass `enable_thinking:false` at serving against a think-trained adapter.
3. Store reasoning traces in `reasoning_content`, never inline.
4. For tool-call-only assistant turns, you may omit reasoning. The template will not add a think block to intermediate turns.

`mlx_lm.lora` does not expose `enable_thinking`; it uses the tokenizer's default template. This is why option 1 is the path of least resistance.

---

## 7. Tool responses

- Put **real execution output** in `content`, including stderr and the exit code when relevant. A model that only sees success learns nothing about recovery.
- Prefix with a compact status when the raw output is ambiguous:

```text
exit_code=1
stderr: connect: Connection refused
stdout:
```

- Truncate long outputs to a fixed budget (e.g., 4,000 characters) and append `[truncated]`. Do not silently drop the middle; the model needs to know.
- Do not include ANSI escape codes, progress bars, or timestamps. Normalize.
- Never include real credentials, tokens, or customer data.

---

## 8. `--mask-prompt`, truncation and trajectory splitting

`mlx_lm.lora --mask-prompt` computes the completion offset by rendering `messages[:-1]` with `add_generation_prompt=True` and masking everything before that offset. For a `chat` dataset:

- Only the **final assistant message** is trained on.
- All earlier assistant turns — including earlier tool calls — are masked.
- If the final message is not an assistant message, the offset logic still runs; do not rely on it.

Therefore, a five-turn trajectory stored as one JSONL line teaches only the last turn. **Split it:**

```text
turn 1: system, user, assistant(tool_call 1)                  → train call 1
turn 2: system, user, assistant(call 1), tool(1), assistant(call 2) → train call 2
...
turn N: ... assistant(final answer)                            → train final answer
```

This is prefix expansion. It increases token count quadratically in the number of turns; cap trajectories at 3–5 assistant turns, or sample a subset of prefixes.

**Truncation.** MLX-LM truncates from the right at `max_seq_length`. If the prompt is already at the limit, the completion is empty and the loss becomes `nan`. Before training:

```bash
python scripts/token_stats.py data/processed/train.jsonl --model Qwen/Qwen3-14B
```

Choose `max_seq_length` at the 99th percentile, rounded up to a multiple of 256, and **drop or split** examples above it. Never let the truncator cut a `<tool_call>` JSON object in half.

**Tool schemas are part of the prompt length.** A realistic 10-tool request can add 1,500–3,000 tokens before the first user message. Include the real `tools` array when measuring, not just the messages.

---

## 9. Validation checklist

Run `scripts/validate_dataset.py` on every split. It checks:

- valid JSON per line;
- `messages` is a non-empty array;
- roles are in `{system, user, assistant, tool}`;
- the first message is `system` (warning if not);
- every `tool_calls[].function.name` exists in `tools`;
- every `arguments` string parses as a JSON object;
- required arguments are present;
- every `tool` message follows an assistant tool call and has non-empty content;
- no duplicate tool call IDs within an example;
- no `<think>` inside `content`;
- no empty assistant message;
- no obviously destructive command patterns (`rm -rf /`, `mkfs`, fork bomb, `dd of=/dev/`) without a lab marker;
- no secrets matching common token patterns.

It does not prove the data is *good*. It proves it is *well-formed*. Human spot checks remain mandatory.

---

## 10. Converting OpenAI-style messages

Most sources already use OpenAI roles. The work is usually in the tool schema and the arguments:

```python
import json

def to_canonical(row, tools, arguments_format="string"):
    """Minimal converter: parse arguments, normalize tool names.

    arguments_format="dict" for Qwen2.5-Coder; "string" for Qwen3/OpenAI.
    """
    messages = []
    for m in row["messages"]:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = c.get("function", c)
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                calls.append({
                    "id": c.get("id", f"call_{len(calls)}"),
                    "type": "function",
                    "function": {
                        "name": fn["name"].replace(" ", "_"),
                        "arguments": args if arguments_format == "dict" else json.dumps(args, ensure_ascii=False),
                    },
                })
            m["tool_calls"] = calls
            m["content"] = m.get("content") or None
        messages.append(m)
    return {"messages": messages, "tools": tools}
```

For source-specific mappings (xLAM, ToolACE, Hermes, APIGen-MT, terminus-2, KaliBench), see [`05-datasets.md`](05-datasets.md) §7.

---

## 11. Common failures

| Symptom | Likely cause | Fix |
|---|---|---|
| Model emits plain text instead of a tool call | Trained without `tools`, or tool block not rendered | Include `tools` in every tool-use example; inspect the rendered prompt. |
| Model emits JSON with single quotes / trailing commas | Trained on malformed calls | Validate every `arguments` value; reject bad rows. |
| Qwen2.5-Coder arguments are a quoted string | Stored as a string instead of an object | Re-render with `--arguments-format dict`. |
| Model calls a tool that does not exist | Hallucinated names in training data | Enforce name membership; add negative examples. |
| XML template produces empty parameters | Passed a JSON string where Qwen3.5 expects a mapping | Parse arguments to a dict before rendering. |
| Double `<think>` | Inline `<think>` in `content` | Use `reasoning_content`. |
| Loss is `nan` | Prompt ≥ `max_seq_length` with `--mask-prompt` | Measure lengths; drop/split long rows. |
| Model ignores the final answer style | Trajectory stored as one line; only last turn trained | Prefix-expand; or train without masking (not recommended). |
| Train/serve mismatch | Different chat template or tool schema | Freeze the tokenizer revision; reuse the exact system prompt and `tools`. |

---

## 12. Sources

- MLX-LM data formats — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md>
- MLX-LM `ChatDataset` — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/datasets.py>
- MLX-LM trainer truncation/masking — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py>
- Qwen3-14B tokenizer template — <https://huggingface.co/Qwen/Qwen3-14B/blob/main/tokenizer_config.json>
- Qwen3.5-9B tokenizer template — <https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/tokenizer_config.json>
- Qwen3-Coder-30B tokenizer template — <https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/main/tokenizer_config.json>
- Qwen2.5-Coder-14B tokenizer template — <https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct/blob/main/tokenizer_config.json>
- OpenAI chat/tool format — <https://platform.openai.com/docs/guides/function-calling>
