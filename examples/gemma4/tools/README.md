# Tool schemas for Gemma 4 12B

Do not concatenate all tools. Pass only the active mode's schema; Gemma 4's template renders native `<|tool_call>` blocks from the OpenAI-style `tools` array.

| Mode | Schema |
|---|---|
| Kali | `examples/tools/kali-tools.json` |
| Java | `examples/java/tools.json` |
| React | `examples/react/tools.json` |

## Rules

- Tool arguments must be **JSON objects** in the messages you pass to the Gemma template. A string triggers the template's explicit error:
  `tool_calls[].function.arguments must be a JSON object (mapping), not a string`.
- Tool names must match `[A-Za-z_][A-Za-z0-9_-]*`; Gemma's parser accepts `\w-`.
- Keep 6–10 tools per mode; more tools reduce selection accuracy.
- Tool results use `role: "tool"` with `tool_call_id`; Gemma resolves the id to the function name.
- Do not put Qwen-style `<tool_call>` JSON into assistant content. Let the template render it.
- At inference, verify `tool_calls` is populated; if it is empty and the text is in `content`, the server parser is not picking up Gemma 4's `<|tool_call>` format.

The Gemma 4 format is:

```text
<|tool_call>call:read_file{path:<|"|>src/App.tsx<|"|>}<tool_call|>
```

not Qwen's:

```text
<tool_call>{"name":"read_file","arguments":{"path":"src/App.tsx"}}</tool_call>
```
