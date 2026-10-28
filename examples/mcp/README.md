# MCP example: a support-tickets server, a client and a Qwen bridge

Runnable companion to [docs/58 — MCP](../../docs/58-ai-mcp.md). An in-memory,
read-mostly support-ticket server with typed tools, a resource template, a
prompt, structured output and actionable errors; a client; and a bridge that
exposes the server's tools to the handbook's local model.

## Install

Requires Python 3.13 and the 2.x MCP SDK (tested with `mcp` 2.3.0):

```bash
pip install "mcp>=2,<3" "openai>=1.40" "pydantic>=2"
```

## Files

| File | What it is |
|---|---|
| `tickets_server.py` | The MCP server: `search_tickets`, `get_ticket`, `add_comment`, the `ticket://{id}` resource template and the `triage` prompt. |
| `client_demo.py` | Connect over stdio; list and use tools, a resource and a prompt. |
| `qwen_bridge.py` | Map MCP tools to OpenAI tool definitions and run the tool-calling loop against the local model (Qwen3 thinking off per request, think blocks stripped, writes need a typed `y`). |
| `test_tickets_server.py` | Automated tests using the SDK client in-process (no sockets). |

## Run

```bash
# Serve over stdio (how a desktop host launches it)
python tickets_server.py

# Serve over Streamable HTTP on 127.0.0.1:8765 (loopback only: no authorization)
python tickets_server.py --http

# Talk to it over stdio
python client_demo.py

# Run the tests
python test_tickets_server.py          # or: python -m pytest test_tickets_server.py

# Inspect it (needs Node 22.19+; downloads the Inspector on first run, so pin it)
npx @modelcontextprotocol/inspector@2.12.0 python tickets_server.py
npx @modelcontextprotocol/inspector@2.12.0 --cli python tickets_server.py --method tools/list

# Bridge to the local model (mlx_lm.server from chapter 10 must be running)
python qwen_bridge.py "Which open tickets are urgent?"
```

The bridge points at `http://127.0.0.1:8080/v1` with model
`mlx-community/Qwen3-14B-4bit` by default; override with `--base-url` / `--model`.
It asks for Qwen3's non-thinking mode per request
(`chat_template_kwargs: {"enable_thinking": false}`); pass `--thinking` for an
adapter that was trained with thinking on (see docs/06-data-format.md §6).
The scripts find `tickets_server.py` next to themselves, so they run from any
directory.
