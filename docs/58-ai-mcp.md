# 58 — MCP: the Model Context Protocol

*Last verified: 28 October 2026 against the MCP specification revision `2026-07-28`, the Python SDK `mcp` 2.3.0, the TypeScript SDK v2 (`@modelcontextprotocol/server` 2.4.0) and the MCP Inspector 2.12.0.*

This chapter is the practical guide to the Model Context Protocol: what it is, how the wire protocol works, and how to build, test, secure and operate a server with the current SDK. Every JSON message here was captured from a server you can run; every Python block runs against the venv and prints what you see. Read [50 — Prompting and APIs](50-ai-prompting-apis.md) for tool/function calling inside one app first, then [52 — Agents](52-ai-agents.md) for the agent loop MCP plugs into. The companion runnable project is in `examples/mcp/`.

---

## 1. What MCP is, and the problem it solves

### 1.1 Plain definition

The Model Context Protocol (MCP) is an open, JSON-RPC-based protocol that lets an AI application discover and use **tools**, **resources** and **prompts** that live in a separate process, called a **server**. One protocol, spoken by many applications and many integrations, replaces a pile of one-off adapters.

An MCP server is not a model and not an agent. It is a small program that says "here are the things I can do" in a fixed shape, and does them when asked. The application holds the model; the server holds the capabilities.

### 1.2 Why an engineer cares: the N×M problem

You care because integration cost grows the wrong way without a shared protocol. Say you run \(H\) AI applications (a desktop assistant, an IDE, a CI bot, a support console) and you want each to reach \(S\) data sources (tickets, a wiki, the repo, a metrics store). Wire each pair by hand and you build one bespoke connector per pair:

\[
C_\text{bespoke} = H \times S
\]

!!! example "On paper"

    5 applications, 8 data sources, each pair wired by hand:

    \[
    C_\text{bespoke} = 5 \times 8 = 40
    \]

    Forty connectors to build, test and keep working.

With a shared protocol, each application learns MCP once and each source exposes an MCP server once. The connectors you maintain become a sum, not a product:

\[
C_\text{shared} = H + S
\]

!!! example "On paper"

    The same 5 applications and 8 sources, over one protocol:

    \[
    C_\text{shared} = 5 + 8 = 13
    \]

    13 pieces instead of 40, so you save:

    \[
    40 - 13 = 27
    \]

    And the gap widens as you grow. Add one more application:

    \[
    C_\text{bespoke} = 6 \times 8 = 48
    \]

    bespoke jumps by 8 (one new connector per source), while:

    \[
    C_\text{shared} = 6 + 8 = 14
    \]

    the protocol approach grows by 1.

This is the same shape as the editor/language problem that the **Language Server Protocol (LSP)** solved. Before LSP, every editor needed a custom plugin for every language: \(M\) editors × \(N\) languages. LSP defined one protocol between editor and language server, turning \(M \times N\) into \(M + N\). MCP is "LSP for AI applications and context."

### 1.3 What MCP is NOT

- **Not a model.** MCP carries no weights and does no inference. The model sits in the application.
- **Not an agent framework.** MCP does not plan, loop or decide. It is the wire between an application and a capability. The loop lives in the host; see [52 — Agents](52-ai-agents.md).
- **Not a replacement for function calling inside one app.** If your tools live in the same process as your model call, plain function calling (one JSON Schema per function, see [50](50-ai-prompting-apis.md)) is simpler and has no protocol overhead. MCP earns its keep when a capability must be **reused across applications** or **run in a separate process or machine** with its own dependencies, credentials or sandbox.

---

## 2. Architecture: host, client, server

### 2.1 The three roles

- **Host**: the application the user runs (a desktop assistant, an IDE, your agent). It holds the LLM, manages the conversation, and asks the user for consent.
- **Client**: a protocol connector *inside* the host. **One client talks to exactly one server.** A host that uses three servers runs three clients.
- **Server**: the separate program exposing tools, resources and prompts.

```text
┌─────────────────────────── Host application ───────────────────────────┐
│                                                                         │
│   the LLM lives here  ──►  conversation + tool-use loop                 │
│                                                                         │
│   ┌── client A ──┐   ┌── client B ──┐   ┌── client C ──┐                │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                │
└──────────┼──────────────────┼──────────────────┼───────────────────────┘
           │ stdio            │ stdio            │ Streamable HTTP
           │ (subprocess)     │ (subprocess)     │ (localhost or remote)
     ┌─────▼──────┐     ┌──────▼─────┐     ┌──────▼──────────┐
     │ tickets    │     │ filesystem │     │ company wiki    │
     │ server     │     │ server     │     │ server (remote) │
     └────────────┘     └────────────┘     └─────────────────┘
```

### 2.2 Where the LLM sits

The LLM is in the **host**, never in the server. A server does not see the model, the prompt, or the conversation. It receives a `tools/call` with arguments and returns a result. (The one exception is **sampling**, where a server asks the host to run a model turn on its behalf — but even then the host owns the model and the user's consent. Sampling is deprecated as of `2026-07-28`; see §5.)

This separation is the point: the server author writes capabilities without choosing or paying for a model, and the host author adds capabilities without reading the server's code.

### 2.3 Local vs remote

- **Local server (stdio transport):** the host launches the server as a child process and talks over its stdin/stdout. No network, no ports, credentials from the environment. This is how a desktop assistant runs a filesystem or git server. It is the default for a single user on one machine.
- **Remote server (Streamable HTTP transport):** the server is an HTTP service; the client connects over the network. This is how a team shares one server, and how a SaaS exposes MCP. It needs authorization (§7) and transport hardening (§6, §12).

---

## 3. The protocol: JSON-RPC 2.0 and the lifecycle

### 3.1 JSON-RPC 2.0

MCP messages are UTF-8 JSON-RPC 2.0. There are exactly three shapes:

- **Request** — has an `id`, expects a response: `{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{...}}`
- **Response** — carries the same `id`, with either `result` or `error`.
- **Notification** — no `id`, no reply: `{"jsonrpc":"2.0","method":"notifications/initialized"}`

Errors use the standard JSON-RPC codes, plus a few MCP codes from the server-error range. You will see these in practice:

| Code | Name | Means |
|---:|---|---|
| −32700 | Parse error | Invalid JSON on the wire |
| −32600 | Invalid request | Not a valid JSON-RPC object |
| −32601 | Method not found | Server does not implement that method |
| −32602 | Invalid params | Params do not fit the method, or name something that does not exist: an unknown tool (per the spec), a missing resource (SEP-2164) |
| −32603 | Internal error | Server-side crash |
| −32020 | HeaderMismatch | HTTP headers disagree with the body or are missing (`2026-07-28`, §6.2) |
| −32021 | MissingRequiredClientCapability | The request lacks a client capability the server needs (`2026-07-28`) |
| −32022 | UnsupportedProtocolVersion | The server does not speak the requested revision; `data.supported` lists the ones it does (`2026-07-28`) |

!!! warning "A protocol error is not a tool error"

    `tools/call` has two failure channels and they mean different things. A **protocol error** (a JSON-RPC `error` object) means the request itself was malformed or unroutable — the model never ran a tool. A **tool error** means the tool ran (or was asked to run) and failed in a way the model should read and react to: it comes back as a normal `result` with `isError: true` and a text explanation. Argument-validation failures and a tool's own `raise ToolError(...)` both land in the `isError` channel. An unknown tool name is a protocol error per the spec (`−32602`), although the Python SDK 2.3.0 reports it as an `isError` result (Exercise 17.5). §4.3 and §8 show both channels.

### 3.2 Two lifecycle eras

The SDK installed here speaks protocol revision `2026-07-28`, confirmed by `mcp.types.LATEST_PROTOCOL_VERSION`. That revision changed the connection model, so there are two eras you must understand. The SDK's `mcp_types.version` names them exactly:

| Set | Versions | Lifecycle |
|---|---|---|
| **Handshake era** | `2024-11-05`, `2025-03-26`, `2025-06-18`, `2025-11-25` | A connection-scoped session opened with an `initialize` handshake. |
| **Modern era** | `2026-07-28` | Stateless: every request carries its own version + capabilities envelope; `server/discover` replaces the handshake. |

The spec calls these eras *legacy* and *modern*, and an implementation that speaks both *dual-era*. The Python SDK is dual-era: its `Client` probes with `server/discover` and falls back to `initialize` for a legacy server, and `MCPServer` answers both. Two single-era peers of different eras cannot talk: a modern-only client fails against a legacy-only server, and the reverse. The modern revision is three months old, and many deployed servers and hosts still speak only the handshake era, so learn it first.

### 3.3 The handshake-era lifecycle (real wire capture)

The client opens with `initialize`, declaring `protocolVersion`, its `capabilities`, and `clientInfo`. The server replies with its own version and capabilities. The client then sends the `notifications/initialized` notification, and normal operation begins. Here is the real exchange, captured by piping newline-delimited JSON-RPC over stdio into a minimal `MCPServer(name="demo", version="1.0.0")` with one tool, `add(a: int, b: int) -> int`:

```json
→ {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"0.0.0"}}}
← {"jsonrpc":"2.0","id":1,"result":{"capabilities":{"prompts":{"listChanged":false},"resources":{"listChanged":false,"subscribe":false},"tools":{"listChanged":false}},"protocolVersion":"2025-06-18","serverInfo":{"name":"demo","version":"1.0.0"}}}
→ {"jsonrpc":"2.0","method":"notifications/initialized"}
→ {"jsonrpc":"2.0","id":2,"method":"tools/list"}
← {"jsonrpc":"2.0","id":2,"result":{"tools":[ ... ]}}
→ {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"add","arguments":{"a":2,"b":3}}}
← {"jsonrpc":"2.0","id":3,"result":{"content":[{"text":"5","type":"text"}],"isError":false,"structuredContent":{"result":5}}}
```

The two failure channels, also captured:

```json
→ {"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"add","arguments":{"a":"two","b":3}}}
← {"jsonrpc":"2.0","id":4,"result":{"content":[{"text":"Error executing tool add: 1 validation error ...","type":"text"}],"isError":true}}
→ {"jsonrpc":"2.0","id":6,"method":"resources/subscribe","params":{"uri":"x://y"}}
← {"jsonrpc":"2.0","id":6,"error":{"code":-32601,"message":"Method not found","data":"resources/subscribe"}}
```

A bad argument came back as a **tool error** (`isError: true`, a `result`). A call to a method the server does not serve came back as a **protocol error** (`-32601`).

### 3.4 Version negotiation

In the handshake, the client offers a `protocolVersion`; the server replies with the same version if it supports it, or with another version it supports, and the client disconnects if it cannot use that one. In the modern era, instead of a handshake the client may call `server/discover`, a request that returns the server's supported versions, capabilities and identity in one round trip, and then stamps a per-request envelope. Captured from the same server, driven in modern mode over stdio:

```json
→ {"jsonrpc":"2.0","id":1,"method":"server/discover","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}}
← {"jsonrpc":"2.0","id":1,"result":{"cacheScope":"private","capabilities":{"prompts":{"listChanged":true},"resources":{"listChanged":true,"subscribe":true},"tools":{"listChanged":true}},"resultType":"complete","supportedVersions":["2026-07-28"],"ttlMs":0,"_meta":{"io.modelcontextprotocol/serverInfo":{"name":"demo","version":"1.0.0"}}}}
→ {"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"add","arguments":{"a":2,"b":3},"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}}
← {"jsonrpc":"2.0","id":2,"result":{"content":[{"text":"5","type":"text"}],"isError":false,"resultType":"complete","structuredContent":{"result":5},"_meta":{"io.modelcontextprotocol/serverInfo":{"name":"demo","version":"1.0.0"}}}}
→ {"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"add","arguments":{"a":2,"b":3}}}
← {"jsonrpc":"2.0","id":4,"error":{"code":-32602,"message":"params._meta must be an object carrying the required 'io.modelcontextprotocol/protocolVersion' and 'io.modelcontextprotocol/clientCapabilities' envelope keys"}}
```

The modern request carries its protocol version and capabilities inside `params._meta`, under the keys `"io.modelcontextprotocol/protocolVersion"` and `"io.modelcontextprotocol/clientCapabilities"` (the dots and the slash are part of each key's name, not nesting), on **every** request, not once at the start. A client SHOULD also send `"io.modelcontextprotocol/clientInfo"`, and the server stamps its identity into each result's `_meta`. Every result carries `resultType` (`"complete"`, or `"input_required"` for the multi round-trip requests of §5), and the SDK adds the cache hints `ttlMs` and `cacheScope` that the spec requires on list results (§4.6). The last request above left out the envelope and was rejected; a version the server does not speak gets `−32022` with the supported versions in `data.supported`. You rarely write this by hand — the SDK client does it — but it explains why the modern era is stateless (§6, §13).

!!! tip "Capabilities declare features, not permission"

    In the handshake (in the modern era: in `server/discover` and in each request's `_meta`), each side lists what it *supports* (tools, resources with `subscribe`, prompts, `listChanged`). Clients check capabilities before using a feature so they do not call a method the peer never advertised. Capabilities are a feature list, not an access-control decision — consent is separate (§5).

---

## 4. Server features

A server exposes three primitives (tools, resources, prompts) plus a set of protocol utilities. Only build what you need.

### 4.1 Tools

A tool is a function the model can call. Each tool has:

- `name` — a stable identifier the model emits.
- `title` — a human label for UIs (optional).
- `description` — written *as an instruction to the model*: when to use it, when not to, what it returns.
- `inputSchema` — a JSON Schema for the arguments.
- `outputSchema` — an optional JSON Schema for a structured result.
- `annotations` — hints about behaviour (below).

A result carries `content` (a list of content blocks: `text`, `image`, `audio`, and others), and optionally `structuredContent` that conforms to `outputSchema`. The Python SDK derives both schemas from your type hints. This block runs:

```python
import asyncio, json
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from mcp.server.mcpserver import MCPServer
from mcp import Client
from mcp_types import ToolAnnotations

server = MCPServer(name="demo")
Status = Literal["open", "pending", "closed"]

class Summary(BaseModel):
    id: int
    status: Status

@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def find(
    status: Annotated[Status, Field(description="Status to match.")],
    limit: Annotated[int, Field(ge=1, le=50, description="Max rows (1-50).")] = 10,
) -> list[Summary]:
    """Return ticket summaries with the given status."""
    rows = [Summary(id=1, status="open"), Summary(id=2, status="open")]
    return [r for r in rows if r.status == status][:limit]

async def main():
    async with Client(server) as client:
        t = (await client.list_tools()).tools[0]
        print("inputSchema:", json.dumps(t.input_schema, separators=(",", ":")))
        print("readOnlyHint:", t.annotations.read_only_hint)
        res = await client.call_tool("find", {"status": "open", "limit": 1})
        print("structured:", json.dumps(res.structured_content, separators=(",", ":")))

asyncio.run(main())
```

```text
inputSchema: {"type":"object","properties":{"status":{"description":"Status to match.","enum":["open","pending","closed"],"title":"Status","type":"string"},"limit":{"default":10,"description":"Max rows (1-50).","maximum":50,"minimum":1,"title":"Limit","type":"integer"}},"required":["status"],"title":"findArguments"}
readOnlyHint: True
structured: {"result":[{"id":1,"status":"open"}]}
```

The `Literal` became an `enum`, the `Field(ge=1, le=50)` became `minimum`/`maximum`, and the scalar/list return was wrapped as `{"result": ...}` in `structuredContent`.

### 4.2 Tool annotations

Annotations are *hints* for the host and the model about a tool's behaviour. Their exact names in the current spec and SDK (the Python SDK spells them `read_only_hint` and so on), with the value a client must assume when one is absent:

| Annotation | Meaning | Default |
|---|---|---|
| `title` | A human-readable name for UIs. | none |
| `readOnlyHint` | The tool does not modify its environment. | `false` |
| `destructiveHint` | A write may destroy or overwrite rather than only add. Meaningful only when `readOnlyHint` is false. | `true` |
| `idempotentHint` | Calling again with the same arguments has no additional effect on the environment. Meaningful only when `readOnlyHint` is false. | `false` |
| `openWorldHint` | The tool reaches an open world of external entities (a web search) rather than a closed domain (a local store). | `true` |

The defaults are the pessimistic ones: a tool with no annotations counts as a destructive, non-idempotent write to the open world. A host can use the hints to decide when to ask — for example, let the user pre-approve the read-only tools of a server they trust, and prompt before every destructive call. The spec is explicit that clients **MUST** treat annotations as untrusted unless they come from a trusted server, so a hint never replaces the server's own checks or the host's consent. In `examples/mcp/tickets_server.py`, `search_tickets` and `get_ticket` are `readOnlyHint=True`; `add_comment` is `readOnlyHint=False, destructiveHint=False` (it appends, it does not destroy), `idempotentHint=False` (each call adds another comment).

### 4.3 Error handling in a tool

Return actionable errors. Raise `ToolError` for a failure the model should read and recover from; it becomes an `isError` result. Let an unexpected exception bubble only if you want it hidden (the model then sees a generic "error executing tool"). This runs:

```python
import asyncio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp import Client

server = MCPServer(name="demo")

@server.tool()
def divide(a: int, b: int) -> float:
    """Divide a by b."""
    if b == 0:
        raise ToolError("b must not be zero; pass a non-zero divisor.")
    return a / b

async def main():
    async with Client(server) as client:
        ok = await client.call_tool("divide", {"a": 6, "b": 3})
        print("ok         :", ok.is_error, "|", ok.content[0].text)
        # an anticipated failure -> is_error=True, message is for the model
        err = await client.call_tool("divide", {"a": 6, "b": 0})
        print("tool error :", err.is_error, "|", err.content[0].text)
        # a bad argument type is validated against inputSchema -> is_error=True
        bad = await client.call_tool("divide", {"a": "six", "b": 3})
        print("bad arg    :", bad.is_error, "|", bad.content[0].text.splitlines()[0])

asyncio.run(main())
```

```text
ok         : False | 2.0
tool error : True | Error executing tool divide: b must not be zero; pass a non-zero divisor.
bad arg    : True | Error executing tool divide: 1 validation error for divideArguments
```

### 4.4 Resources and resource templates

A resource is read-only context identified by a URI. A **resource template** is a URI pattern (RFC 6570) with variables, so one handler serves many instances. Resources carry a `mimeType`, can be listed and read, and — if the server advertises `subscribe` — watched for updates through the `notifications/resources/updated` notification. In the handshake era a client asks for them with `resources/subscribe`; `2026-07-28` removed that method and delivers the same notifications on one long-lived `subscriptions/listen` stream that the client opens and filters (§6.2). This runs:

```python
import asyncio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError
from mcp import Client

server = MCPServer(name="demo")
DB = {1: "Refund not received", 2: "Cannot log in"}

@server.resource("ticket://{tid}", mime_type="text/plain")
def ticket(tid: str) -> str:
    row = DB.get(int(tid))
    if row is None:
        raise ResourceNotFoundError(f"no ticket {tid}")
    return row

async def main():
    async with Client(server) as client:
        tmpls = await client.list_resource_templates()
        print("templates:", [t.uri_template for t in tmpls.resource_templates])
        r = await client.read_resource("ticket://1")
        print("ticket://1:", r.contents[0].text, "|", r.contents[0].mime_type)

asyncio.run(main())
```

```text
templates: ['ticket://{tid}']
ticket://1: Refund not received | text/plain
```

!!! note "Tools vs resources"

    A **tool** is a verb the model decides to call and that may have effects. A **resource** is a noun the host or user pulls into context, read-only. Reach for a resource when the model needs to *read* something addressable (a file, a record, a page); reach for a tool when it needs to *do* something.

### 4.5 Prompts

A prompt is a reusable, named message template the user invokes (often from a slash-command menu), with typed arguments. It is user-initiated, not model-initiated. This runs:

```python
import asyncio
from mcp.server.mcpserver import MCPServer
from mcp import Client

server = MCPServer(name="demo")

@server.prompt(title="Triage a ticket")
def triage(ticket_id: str) -> str:
    """Ask the model to triage one ticket."""
    return f"Read ticket://{ticket_id}, assign a severity, and draft a one-line reply."

async def main():
    async with Client(server) as client:
        prompts = await client.list_prompts()
        p = prompts.prompts[0]
        print("prompt:", p.name, "| args:", [a.name for a in p.arguments])
        got = await client.get_prompt("triage", {"ticket_id": "7"})
        print("role:", got.messages[0].role)
        print("text:", got.messages[0].content.text)

asyncio.run(main())
```

```text
prompt: triage | args: ['ticket_id']
role: user
text: Read ticket://7, assign a severity, and draft a one-line reply.
```

### 4.6 Protocol utilities

The current protocol and SDK also provide, only where you enable them:

- **`listChanged` notifications** — when your tool/resource/prompt list changes at runtime, emit `notifications/tools/list_changed` (and the resource/prompt variants) so clients re-list. The SDK exposes `ctx.notify_tools_changed()`, `ctx.notify_resources_changed()`, `ctx.notify_prompts_changed()`. In `2026-07-28` these reach only clients that opened a `subscriptions/listen` stream for that type.
- **Pagination** — list methods accept an opaque `cursor` and return a `nextCursor`; a client pages until `nextCursor` is absent. Never hand-parse a cursor; treat it as opaque.
- **Completion** — `completion/complete` offers argument autocompletion for prompts and resource templates (`@server.completion()`).
- **Progress** — a long call can report progress with `ctx.report_progress(done, total)`, delivered as `notifications/progress`.
- **Caching** (`2026-07-28`) — list results and resource reads carry `ttlMs` (how long a client may reuse them) and `cacheScope` (`"public"` or `"private"`, whether shared intermediaries may cache them). Servers SHOULD return `tools/list` in a deterministic order, so clients and prompt caches can reuse it.
- **Logging** — a tool can emit structured logs to the client with `ctx.info(...)`, `ctx.debug(...)`, `ctx.warning(...)`, `ctx.error(...)`. Deprecated as of `2026-07-28`, together with Roots and Sampling: `logging/setLevel` is gone, a client opts in per request with `io.modelcontextprotocol/logLevel` in `_meta`, and the spec's migration path is stderr for stdio servers or OpenTelemetry. For your own operations, log to stderr (§6.1).
- **Cancellation** — a client abandons an in-flight request with `notifications/cancelled` on stdio; on Streamable HTTP in `2026-07-28`, closing that request's response stream is the cancellation.
- **`ping`** existed in the handshake era and was removed in `2026-07-28` (a modern request for it gets `−32601`).

Build the three primitives first; add utilities only when a concrete need appears.

---

## 5. Client features: what the host offers the server

Capabilities flow both ways. A client may offer features a server can call back into. Verify each against the revision you target: `2026-07-28` deprecated two of the three, and changed how all three travel — instead of sending its own request to the client, the server returns an `InputRequiredResult` and the client retries (the Multi Round-Trip Requests pattern, below).

- **Elicitation** (*current*): a server can pause mid-request and ask the **user** for structured input. It has two modes: **form mode** (the server sends a JSON Schema, the client renders a form and validates the answer) and **URL mode** (the server hands a URL for the user to open out of band — used for credentials or OAuth, so the secret never passes through the client or the model). Elicitation uses the Multi Round-Trip Requests pattern: the server returns an `InputRequiredResult` carrying `elicitation/create`, the client collects the input, then retries the original request with a new JSON-RPC id, the answers in `inputResponses` and the server's opaque `requestState`. The SDK exposes `ctx.elicit(...)` and `ctx.elicit_url(...)`. Servers **must not** use form mode for passwords, API keys, access tokens or payment credentials — that is URL mode's job — and a client **must** show the full URL and get the user's consent before opening it.
- **Sampling** (*deprecated as of `2026-07-28`*): a server asks the host to run an LLM completion on its behalf, so the server needs no model of its own. New servers should call an LLM provider directly instead. Where still used, it runs through human-in-the-loop review in the host.
- **Roots** (*deprecated as of `2026-07-28`*): the client tells the server which filesystem directories (as `file://` URIs) are in scope. New servers should take directories via tool parameters, resource URIs or configuration. Roots were always advisory ("SHOULD respect"), never an OS-level boundary.

!!! warning "Consent is the host's job, and it is mandatory"

    Every client feature touches the user's trust boundary: elicitation asks the user for data, sampling spends the user's model budget and exposes context to a server, roots expose directories. The spec requires the host to get **explicit user consent**: show which server is asking and why, let the user review or decline, and never auto-open a URL or auto-approve a model call for an untrusted server. A server never gets to bypass this.

---

## 6. Transports

A transport frames and delivers the JSON-RPC messages. The protocol meaning is identical on every transport. There are two standard ones.

### 6.1 stdio (local)

The host launches the server as a subprocess and exchanges **newline-delimited JSON** over stdin/stdout: one JSON-RPC message per line, UTF-8. One rule dominates:

!!! warning "On stdio, stdout is the wire — never print to it"

    The server's stdout carries protocol frames only. A stray `print()`, a progress bar, or a library banner on stdout corrupts the stream and the client fails to parse it. **Log to stderr.** The handbook's `tickets_server.py` sets `logging.basicConfig(stream=sys.stderr)` for exactly this reason. A broken server story: a developer added `print("loaded config")` at import time; the host showed "failed to parse message" and the server "worked fine when I ran it myself" (because a terminal does not care). Moving the line to `logging.info` on stderr fixed it. Desktop hosts route a stdio server's stderr to a log file, which is where you look first.

stdio needs no ports and no auth; it takes credentials from the environment (§7).

### 6.2 Streamable HTTP (remote)

The HTTP transport uses a **single endpoint** (e.g. `/mcp`). The client **POSTs** each JSON-RPC message; the server replies either with one `application/json` object or with a `text/event-stream` (SSE) stream scoped to that request, when it needs to push progress or log messages before the result. The two eras differ here:

- **Handshake era (`2025-03-26` to `2025-11-25`):** the server may mint a session (`Mcp-Session-Id`), the client may open a **GET** stream for server-initiated messages, the server may send its own requests on SSE streams, and a dropped stream can resume with `Last-Event-ID`.
- **`2026-07-28`:** none of that. No sessions, no GET (a modern server answers it with `405`), no resumability: a dropped stream loses the in-flight request, and the client re-issues it with a new id. Long-lived change notifications arrive on the response stream of a `subscriptions/listen` POST.

The headers, with the spec's capitalization (HTTP header names are case-insensitive, and servers often print them in lowercase):

| Header | Carries | Era |
|---|---|---|
| `MCP-Protocol-Version` | the protocol version; in `2026-07-28` it must match the version in the body's `_meta` | required since `2025-06-18` |
| `Mcp-Method` | the request's `method`, mirrored so intermediaries can route without parsing the body | `2026-07-28`, every request |
| `Mcp-Name` | `params.name` or `params.uri`, for `tools/call`, `resources/read` and `prompts/get` | `2026-07-28` |
| `Mcp-Session-Id` | a server-assigned session id | handshake era only |
| `Last-Event-ID` | the last SSE event seen, so the server can replay what a dropped stream missed | handshake era only |

A header that disagrees with the body, or a required one that is missing, gets a `400` with `−32020 HeaderMismatch`. Captured from `tickets_server.py --http`, a `tools/call` sent without `Mcp-Name`:

```text
HTTP/1.1 400 Bad Request
content-type: application/json

{"jsonrpc":"2.0","id":2,"error":{"code":-32020,"message":"mcp-name header does not match the request body's 'name' parameter"}}
```

A modern `server/discover` (sent with `MCP-Protocol-Version: 2026-07-28` and `Mcp-Method: server/discover`) returns a plain JSON object, and **no** session id is minted (captured):

```text
HTTP/1.1 200 OK
content-type: application/json

{"jsonrpc":"2.0","id":1,"result":{"cacheScope":"private","capabilities":{...},"instructions":"Query and triage an in-memory support-ticket queue. ...","resultType":"complete","supportedVersions":["2026-07-28"],"ttlMs":0,"_meta":{"io.modelcontextprotocol/serverInfo":{"name":"support-tickets","title":"Support Tickets","version":"1.0.0"}}}}
```

A **handshake-era** `initialize` over the same endpoint *does* mint a session id and streams the reply as SSE (captured):

```text
HTTP/1.1 200 OK
content-type: text/event-stream
mcp-session-id: dc8718d97d4a4fe2b962212f03d140a5
```

### 6.3 The deprecated HTTP+SSE transport

Revision `2024-11-05` defined an older **HTTP+SSE** transport that used *two* endpoints: a long-lived `GET /sse` stream for server→client messages and a separate `POST` endpoint for client→server messages. It was replaced by the single-endpoint Streamable HTTP transport in `2025-03-26` and is deprecated. The SDK keeps an `sse` transport option for talking to old servers; do not build new servers on it.

---

## 7. Authorization for HTTP servers

Authorization is **optional** in MCP and applies to **HTTP-based** transports. A **stdio** server **should not** implement this flow; it takes credentials from its environment (an API key in an env var set by the host config). A remote server is where authorization matters, and the current spec builds it on OAuth 2.1 (still an IETF draft; the spec pins `draft-ietf-oauth-v2-1-13`). Every requirement below is from the `2026-07-28` authorization spec.

### 7.1 The flow

The MCP server is an OAuth 2.1 **resource server**; the client is an OAuth 2.1 **client**; a separate **authorization server** issues tokens.

1. The client calls the server with no token; the server returns **HTTP 401**. Its `WWW-Authenticate` header points at the server's **Protected Resource Metadata** (**RFC 9728**, which MCP servers **MUST** implement) in a `resource_metadata` parameter; without one, the client tries the well-known URI `/.well-known/oauth-protected-resource`.
2. The client fetches that metadata, learns which authorization server(s) to use, and discovers the authorization server via **Authorization Server Metadata** (**RFC 8414**) or OpenID Connect Discovery.
3. The client obtains a `client_id` (Client ID Metadata Documents, pre-registration, or Dynamic Client Registration **RFC 7591** — now deprecated, kept for backward compatibility).
4. The client runs the authorization-code flow with **PKCE** (the `code_challenge`/`code_verifier` proof that OAuth 2.1 mandates; defined in RFC 7636), carrying a **`resource` parameter** (**Resource Indicators, RFC 8707**) that names the exact MCP server the token is for, and records the issuer to validate the `iss` in the response (**RFC 9207**).
5. The client sends the token only in the `Authorization: Bearer ...` header (the Bearer scheme of RFC 6750; never in the URL), on every request.

### 7.2 Audience validation and no token passthrough

Two requirements prevent a whole class of attacks:

- **Audience binding:** the server **MUST** validate that each token was issued *for it* (RFC 8707 audience), and **MUST NOT** accept a token minted for a different resource.
- **No token passthrough:** the server **MUST NOT** forward a client's token to a downstream API. Passing a token through means the downstream cannot tell who really called, breaks audit and rate-limiting, and creates a confused-deputy opening (§12). The server authenticates to downstreams with its *own* credentials.

!!! tip "Least privilege through scopes"

    A server advertises the minimal `scopes_supported` for basic use and asks for more only when a privileged operation is first attempted (a step-up: `403` with `WWW-Authenticate: Bearer error="insufficient_scope", scope="..."`). Do not publish an omnibus `*`/`all` scope; a stolen broad token has a large blast radius.

---

## 8. Build a server, step by step

The runnable server is `examples/mcp/tickets_server.py`: a read-mostly support-ticket store with typed tools, a resource template, a prompt, validation, structured output, actionable errors and stderr logging. The pieces below are each self-contained and match that file; assemble them as shown there.

### 8.1 Define the server and a typed, read-only tool

Create the server with identity and `instructions` (a short note the host can show the model about how to use the server). Add a tool whose schema constrains the inputs with enums and ranges, whose return type produces `structuredContent`, and whose annotations mark it read-only. §4.1 is that pattern; the full server adds `search_tickets(status, priority, limit)`, `get_ticket(ticket_id)`, and the writing tool `add_comment(ticket_id, comment)` marked `readOnlyHint=False`.

### 8.2 Add a resource template and a prompt

`ticket://{ticket_id}` exposes each record as JSON (§4.4); the `triage` prompt (§4.5) gives the user a one-click triage action. Validate the template variable and raise `ResourceNotFoundError` for a missing id, so the client gets a clean `-32602` instead of a crash.

### 8.3 Run it over stdio and over Streamable HTTP

The same server object runs on either transport — only the `run(...)` call changes. `server.run()` serves stdio (a desktop host launches it as a subprocess); `server.run(transport="streamable-http", host="127.0.0.1", port=8765)` serves Streamable HTTP for a local or team service. `tickets_server.py` picks one from a flag. Its HTTP mode has no authorization, so it accepts only a loopback `--host` and spells out the Host/Origin checks of §12.2:

```bash
python tickets_server.py            # stdio
python tickets_server.py --http      # Streamable HTTP on 127.0.0.1:8765
```

### 8.4 Test with the SDK client, in-process

The fastest test needs no subprocess and no sockets: pass the server object straight to `Client`, which connects in-process. `examples/mcp/test_tickets_server.py` does this and asserts the schema the model will see, both error channels (an `isError` tool result, and `−32602` for a missing or malformed resource id), a write, a resource read and the prompt. Its real output:

```text
test_tickets_server: all assertions passed
```

This is a contract test, not an eval: its assertions are exact, and they pin the *schema and behaviour the model depends on* — tool names, required fields, enum values, `isError` on bad input. Run it in CI so a refactor that renames a field or loosens a range fails the build, not a user's agent. How well a model *uses* the tools is a separate, threshold-based eval ([54 — Evaluation](54-ai-evaluation.md)).

### 8.5 Inspect it with the MCP Inspector

The reference debugging tool is the MCP Inspector, published as `@modelcontextprotocol/inspector` and run with `npx` (needs Node 22.19+). The current line is v2 (2.12.0 at the time of writing; the v1 line gets security fixes only). It has a web UI (the default), a scriptable CLI (`--cli`) and a terminal UI (`--tui`); the mode flag comes first. Pin the version, as §12.4 asks of every `npx` tool:

```bash
# Web UI, launching your stdio server as the target
npx @modelcontextprotocol/inspector@2.12.0 python tickets_server.py

# Scriptable CLI: list tools, or call one
npx @modelcontextprotocol/inspector@2.12.0 --cli python tickets_server.py --method tools/list
npx @modelcontextprotocol/inspector@2.12.0 --cli python tickets_server.py \
    --method tools/call --tool-name get_ticket --tool-arg ticket_id=2
```

Run against `tickets_server.py`, the CLI returned the real tool list (`search_tickets`, `get_ticket`, `add_comment`) with each tool's full JSON Schema and annotations, and the call returned ticket 2 as both text and `structuredContent` — the quickest way to see exactly what a host will send the model. The Inspector keeps OAuth tokens and server `env` values in the OS keychain, or, on a machine without one, in `~/.mcp-inspector/secrets.json`, unencrypted unless you supply a key; keep production credentials out of it on a shared machine.

### 8.6 The same server in TypeScript

The TypeScript SDK is a separate implementation. Its **v2** line (2.0.0 shipped on 27 July 2026, for the `2026-07-28` revision) replaces the single `@modelcontextprotocol/sdk` package of v1 with separate packages, `@modelcontextprotocol/server` and `@modelcontextprotocol/client`; the old package still gets 1.x releases for existing code. The server class is `McpServer` (note the casing, unlike Python's `MCPServer`). This server was installed with `npm install @modelcontextprotocol/server@2.4.0 zod@4` and run with Node 22.22, which runs a `.ts` file directly by stripping its types:

```typescript
import { McpServer } from "@modelcontextprotocol/server";
import { serveStdio } from "@modelcontextprotocol/server/stdio";
import { z } from "zod";

const db: Record<number, string> = { 1: "Refund not received", 2: "Cannot log in" };

function createServer(): McpServer {
  const server = new McpServer({ name: "support-tickets-ts", version: "1.0.0" });
  server.registerTool(
    "get_ticket",
    {
      title: "Get ticket",
      description: "Return a ticket subject by numeric id.",
      inputSchema: z.object({ ticket_id: z.number().int().min(1) }),
      annotations: { readOnlyHint: true },
    },
    async ({ ticket_id }) => {
      const subject = db[ticket_id];
      if (!subject) {
        return { isError: true, content: [{ type: "text", text: `No ticket ${ticket_id}. Known ids: 1, 2.` }] };
      }
      return { content: [{ type: "text", text: subject }] };
    },
  );
  return server;
}

// serveStdio answers both eras (server/discover and the initialize handshake).
// stdout is the wire: log with console.error, never console.log.
serveStdio(createServer);
```

Driven with `npx @modelcontextprotocol/inspector@2.12.0 --cli node server.ts --method tools/list`, it reported one tool, `get_ticket`, with annotations `{readOnlyHint: true}` and an `inputSchema` whose `ticket_id` is an integer with `minimum: 1` (Zod's `.int()` also adds `maximum: 9007199254740991`, JavaScript's largest safe integer, \(2^{53}-1\)). A `tools/call` with `ticket_id=1` returned `Refund not received`, and the Python SDK's `Client` connected to it over stdio on `2026-07-28`. Wiring the same server by hand with `server.connect(new StdioServerTransport())` spoke only `2025-11-25` in that test; `serveStdio` is the dual-era entry point. Unlike the Python SDK, it answered an unknown tool name with a protocol error, as the spec says. The protocol is identical across SDKs; pick the language your capability already lives in.

---

## 9. Build a client and bridge MCP to the local model

### 9.1 A plain client

`examples/mcp/client_demo.py` launches the server over stdio (it finds `tickets_server.py` next to itself, so it runs from any directory), lets the SDK pick the era (it probes `server/discover` and would fall back to `initialize` for a legacy server), then lists and uses each feature. The same code connects to a remote server by passing a URL string such as `"http://127.0.0.1:8765/mcp"` to `Client(...)` instead of `StdioServerParameters`. Its real output:

```text
connected to: support-tickets 1.0.0
protocol: 2026-07-28
tools: ['search_tickets', 'get_ticket', 'add_comment']
search_tickets -> {"total": 3, "returned": 3, "tickets": [{"id": 2, "subject": "Cannot log in after reset", "status": "open", "priority": "urgent"}, {"id": 4, "subject": "Invoice total looks wrong", "status": "open", "priority": "normal"}, {"id": 1, "subject": "Refund not received", "status": "open", "priority": "high"}]}
get_ticket(2).subject -> Cannot log in after reset
resource ticket://2 mime -> application/json
prompts: ['triage']
```

### 9.2 Turning MCP tools into OpenAI tool definitions

The local model served by `mlx_lm.server` ([10 — Export and serving](10-export-serving.md)) speaks the OpenAI tool-calling shape, not MCP. A **bridge** converts each MCP tool into an OpenAI tool definition — the MCP `inputSchema` drops straight into `parameters`:

```python
import asyncio, json
from mcp.server.mcpserver import MCPServer
from mcp import Client

server = MCPServer(name="demo")

@server.tool()
def get_ticket(ticket_id: int) -> str:
    """Return a ticket's subject by id."""
    return {1: "Refund not received"}.get(ticket_id, "unknown")

def mcp_tools_to_openai(mcp_tools):
    return [{
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description or "",
            "parameters": t.input_schema,
        },
    } for t in mcp_tools]

async def main():
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
        print(json.dumps(mcp_tools_to_openai(tools)[0], indent=2))

asyncio.run(main())
```

```text
{
  "type": "function",
  "function": {
    "name": "get_ticket",
    "description": "Return a ticket's subject by id.",
    "parameters": {
      "type": "object",
      "properties": {
        "ticket_id": {
          "title": "Ticket Id",
          "type": "integer"
        }
      },
      "required": [
        "ticket_id"
      ],
      "title": "get_ticketArguments"
    }
  }
}
```

### 9.3 The tool-calling loop

`examples/mcp/qwen_bridge.py` runs the loop: send the user message and the converted tools to the model; if the reply has `tool_calls`, execute each through the MCP client, append the results as `role: tool` messages, and send again; stop when the model returns prose. The model is reached with `AsyncOpenAI` at `http://127.0.0.1:8080/v1`, model `mlx-community/Qwen3-14B-4bit`. The model never sees MCP — only OpenAI tool defs and tool results; the bridge is the MCP client. Four details make it survive a real model:

- **Qwen3's thinking mode.** Qwen3 thinks by default. `mlx_lm.server` puts the reasoning in a separate `reasoning` field and returns `content: null` when nothing else was generated (a tool-call turn, or a reply cut off by `max_tokens` inside the reasoning); a server that does not split the reasoning out returns content that starts with a `<think>…</think>` block. The bridge sends `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`, which `mlx_lm.server` accepts per request (as in [45](45-ai-engineer-path.md)), and still strips any think block and treats a `None` content as empty. An adapter trained with thinking on must be served as trained ([06 §6](06-data-format.md#6-thinking-and-reasoning_content)): `--thinking` keeps it on, with the model card's sampling (temperature 0.6, top-p 0.95) instead of greedy decoding and a larger token budget.
- **Bad calls become text, not crashes.** Malformed JSON arguments, an unknown tool name, an `isError` result and a protocol error (`MCPError`) all go back to the model as a tool message it can act on.
- **Writes need a human.** A tool the server does not mark `readOnlyHint` runs only after the user types `y`; with no terminal attached it is refused. The bridge trusts the hint only because it launches a server you can read (§4.2).
- **It finds the server next to itself**, so it runs from any directory.

You cannot run the real model here, so the bridge was tested against a stub chat-completions server (an `http.server` returning OpenAI-shaped JSON) that imitates Qwen3: thinking stays on unless the request carries `enable_thinking: false`, and while it is on, a tool-call turn has `content: null` plus a `reasoning` field and the answer turn starts with a `<think>` block. Turn one emits a `search_tickets` tool call; turn two answers from the tool result. Pointed at the stub, the bridge ran the real MCP tool and printed the same line in both modes (the default, and `--thinking`):

```text
1 urgent open ticket(s): #2 Cannot log in after reset.
```

A loop that returns `msg.content` as it comes prints the `<think>` block in front of that line against the same stub. Two more stub variants checked that a thinking turn cut off by `max_tokens` (`content: null`, `finish_reason: "length"`) ends with a clear `stopped:` message, and that malformed JSON arguments reach the model as a readable error instead of a crash.

The final line is grounded in the real tool output: #2 is the only open+urgent ticket in the store. **A real Qwen3-14B would differ**: it, not the stub, would decide whether and how to call `search_tickets` (it might pass different filters, call `get_ticket` as a follow-up, or answer without a tool if mis-prompted), and it would phrase the final sentence in its own words. The loop, the schema it is handed, and the tool execution are identical.

---

## 10. Connecting a server to hosts

Each host has its own config format; verify it on the host's own docs before writing it, because these move. As of October 2026 (use absolute paths, including the Python of the venv that has `mcp` installed: a desktop host does not start from your shell, so a bare `python` may be missing or lack the package):

**Claude Desktop** — `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`), top-level key `mcpServers`:

```json
{
  "mcpServers": {
    "tickets": {
      "command": "/abs/path/to/venv/bin/python",
      "args": ["/abs/path/to/examples/mcp/tickets_server.py"],
      "env": { "TICKETS_API_KEY": "..." }
    }
  }
}
```

The `env` block is where a real server's secret goes (the demo itself needs no key).

**Claude Code** — the `claude mcp add` command; `--` separates Claude's options from the server command:

```bash
# local stdio server
claude mcp add tickets -- /abs/path/to/venv/bin/python /abs/path/to/examples/mcp/tickets_server.py
# remote HTTP server
claude mcp add --transport http wiki https://mcp.example.com/mcp --header "Authorization: Bearer ..."
```

A project-scoped `.mcp.json` (committed to the repo, top-level key `mcpServers`) shares servers with your team. In an interactive session Claude Code asks for approval before using them; non-interactive runs such as `claude -p` load them without asking, so review a repo's `.mcp.json` before running it in CI.

**VS Code** — `.vscode/mcp.json`, top-level key `servers` (note: `servers`, not `mcpServers`). VS Code now also reads "portable" files that use `mcpServers` (such as a workspace-root `.mcp.json`) and labels the `.vscode/mcp.json` destination deprecated in its guided setup, kept for compatibility; check its MCP docs before you choose:

```json
{
  "servers": {
    "tickets": { "command": "/abs/path/to/venv/bin/python", "args": ["/abs/path/to/examples/mcp/tickets_server.py"] }
  }
}
```

**OpenCode** — `opencode.json`, an `mcp` block, as in [11 §3.2](11-kali-agent.md#32-opencode-provider-mcp) (`opencode mcp list` shows what it loaded):

```json
{
  "mcp": {
    "tickets": {
      "type": "local",
      "command": ["/abs/path/to/venv/bin/python", "/abs/path/to/examples/mcp/tickets_server.py"],
      "enabled": true
    }
  }
}
```

---

## 11. Tool-design best practices

The tool surface is a prompt, and the model pays for it on every request. Design it like an API you are teaching a junior to use blind.

### 11.1 Principles

- **Few, task-shaped tools, not one per endpoint.** `search_tickets(status, priority, limit)` beats `list`, `filter_by_status`, `filter_by_priority`, `sort`, `paginate`. Shape tools around the *task* the model does, not your REST routes.
- **Names and descriptions are instructions to the model.** Say when to use a tool, when *not* to, and what it returns. "Use this first to find ids; it returns summaries" guides sequencing.
- **Tight schemas.** Enums for closed sets, `minimum`/`maximum` for ranges, `required` for what you need, `additionalProperties: false` to reject junk. The schema is the model's guardrail and your validation.
- **Small, structured results with pagination.** Return `structuredContent` the host can parse, keep it short, and page large sets rather than dumping thousands of rows into context.
- **Actionable errors.** "No ticket 999. Known ids: 1, 2, 3, 4." lets the model recover; "error" does not.
- **Idempotency and dry-run for writes.** Mark reads `readOnlyHint`; for writes, support an idempotency key or a `dry_run` argument so a retried agent step does not double-charge.
- **Version and deprecate.** Changing a tool's schema after hosts depend on it breaks them. Add a new tool or a new argument with a default; deprecate the old one with a note in its description before removing it.

### 11.2 The token cost of tool definitions

Every tool definition is tokens in the prompt on **every** request. Model it:

\[
T_\text{req} = n_\text{tools} \times t_\text{def}
\]

!!! example "On paper"

    8 tools, roughly 125 tokens of schema each:

    \[
    T_\text{req} = 8 \times 125 = 1000
    \]

    1000 prompt tokens spent before the user's question.

Across a day of traffic:

\[
T_\text{day} = T_\text{req} \times R
\]

!!! example "On paper"

    At 4000 requests a day:

    \[
    T_\text{day} = 1000 \times 4000 = 4\,000\,000
    \]

    Assume \$3 per million input tokens:

    \[
    \text{cost} = 4 \times 3 = 12
    \]

    \$12 a day just to describe the tools. Trim to 4 tools and it halves to \$6.

A prompt cache (a hosted API's, or `mlx_lm.server`'s) can make a repeated tool prefix cheaper to process — one reason the spec asks servers to list tools in a deterministic order — but it does not free the context the definitions occupy.

### 11.3 Why a 14B local model needs a small surface

On a 32 GB M1 Pro, Qwen3-14B serves with a context window around 16K tokens ([10 — Export and serving](10-export-serving.md)). Tool definitions eat that window before any conversation:

\[
f_\text{ctx} = \frac{n_\text{tools} \times t_\text{def}}{W}
\]

!!! example "On paper"

    8 tools at 125 tokens in a 16384-token window:

    \[
    f_\text{ctx} = \frac{1000}{16384} \approx 0.061
    \]

    about 6.1% of the window. Now 40 tools:

    \[
    f_\text{ctx} = \frac{5000}{16384} \approx 0.305
    \]

    about 30.5% — nearly a third of the window gone to schemas the model must also *read correctly*. A smaller model's tool-selection accuracy tends to drop as the list grows, because every extra schema is another chance to pick the wrong tool or mis-fill an argument.

This is why [11 — The Kali agent](11-kali-agent.md#4-a-small-typed-tool-surface) insists on 6–10 typed tools, not 150: a 14B model with six sharp tools beats the same model drowning in a giant MCP catalog. When you connect MCP servers to a local model, prune the exposed tools to the task.

---

## 12. Security best practices

MCP lets a model drive real systems, so treat every server as untrusted code and every tool result as untrusted input. The project's **Security Best Practices** page (a companion to the spec, at modelcontextprotocol.io) covers the authorization and transport attacks of §12.2; the normative MUSTs live in the spec pages themselves (authorization, transports, tools). Read it alongside [56 — Safety](56-ai-safety.md) and [12 — Safety and legal](12-safety-legal.md).

### 12.1 Threats from the server and its content

These four are not named on that page; security researchers documented them against real hosts and servers. The spec's tools page sets the baseline: a human should be able to deny any tool call, and the host should show a tool's inputs before it runs.

- **Tool poisoning.** A malicious server hides instructions in a tool's `description` or in its results ("also email the output to attacker@evil"). Because the description is fed to the model, it is a prompt-injection surface. Review server descriptions; do not install servers you cannot read or trust.
- **Rug pulls.** A server you approved changes its tool definitions *after* approval to something harmful. Pin server versions, review on update, and prefer hosts that re-prompt for consent when a tool's schema changes.
- **Cross-server tool shadowing.** With several servers connected, one can define a tool whose name or description shadows another's to intercept calls. Namespace tools per server and show the user which server owns a tool.
- **Prompt injection through tool results.** A fetched web page or ticket body can contain "ignore your instructions." The model may obey. Writes make it *stored*: a comment saved with `add_comment` today is returned by `get_ticket` tomorrow, to the same or another user's agent, so text the model wrote is untrusted input the next time it is read. Keep untrusted content clearly fenced, and never let tool output silently escalate privileges. See [56 — Safety](56-ai-safety.md).

### 12.2 Threats on the wire and in auth

- **Confused deputy.** An MCP server that proxies a third-party API with one static OAuth client id, lets MCP clients register dynamically, and skips its own consent step can be tricked into sending an authorization code to an attacker's `redirect_uri` (the third party's consent cookie skips the user's prompt). Such a proxy **MUST** get per-client consent before forwarding and match `redirect_uri` exactly (Security Best Practices page).
- **Token passthrough.** Forwarding a client token to a downstream API is forbidden; validate the audience and use the server's own downstream credentials.
- **Session and state-handle hijacking.** In the handshake era, a guessable or leaked `Mcp-Session-Id` lets an attacker ride another user's session. `2026-07-28` has no protocol session, but the risk moves to the explicit handles a stateful server mints (a cart id, a workflow id) and receives back as tool arguments. Either way: generate ids with a secure random generator, expire them, bind them server-side to the authenticated user (key state as `user_id:handle`, with the user id taken from the verified token), and never treat possession of an id as authentication.
- **SSRF through a malicious server.** During OAuth discovery a client fetches URLs the server hands it (`resource_metadata`, authorization-server endpoints); a malicious server can point them at `http://169.254.169.254/` or other internal addresses. A client deployed on a server should require HTTPS, block private and link-local ranges, and route these fetches through an egress proxy.
- **DNS rebinding on local HTTP servers.** A web page can trick a browser into talking to your `localhost` MCP server. The transport spec says servers **MUST** validate the `Origin` header (answering `403` when it is present and invalid) and **SHOULD** bind to `127.0.0.1` when running locally; also check `Host`. A local server should prefer stdio, and if it must use HTTP, require a token even on loopback. The Python SDK turns Host/Origin checks on by itself when it binds to `127.0.0.1`, `localhost` or `::1` (not for any other address), and `tickets_server.py` passes them explicitly. Captured from `tickets_server.py --http`, a forged `Host` and then a foreign `Origin`:

```text
HTTP/1.1 421 Misdirected Request

Invalid Host header

HTTP/1.1 403 Forbidden

Invalid Origin header
```

### 12.3 Command and path injection inside a server

A tool that shells out is a command-injection hole if it builds a shell string from model-supplied text. Allowlist the binary (by absolute path, so `PATH` cannot swap it), validate the argument against a strict pattern, and pass a list without a shell so the argument stays one `argv` entry. The pattern must also refuse a leading `-`: a value such as `-fhosts.txt` contains no shell metacharacter and passes a letters-digits-dots-and-hyphens check, but the program parses it as an option, not a domain (`dig -f` reads queries from a file). That is argument injection. This runs:

```python
import re
import subprocess

TOOLS = {"dig": "/usr/bin/dig", "whois": "/usr/bin/whois"}  # allowlist
# dot-separated labels of letters, digits and "-"; no label starts with "-"
HOSTNAME = re.compile(r"(?!-)[A-Za-z0-9-]{1,63}(\.(?!-)[A-Za-z0-9-]{1,63})*")

def lookup_argv(tool: str, domain: str) -> list[str]:
    if tool not in TOOLS:
        raise ValueError(f"tool not allowed: {tool}")
    if len(domain) > 253 or not HOSTNAME.fullmatch(domain):
        raise ValueError(f"refusing suspicious argument: {domain!r}")
    return [TOOLS[tool], domain]

def lookup(tool: str, domain: str) -> str:
    # a list and no shell=True: the domain can never become a second command
    out = subprocess.run(lookup_argv(tool, domain), capture_output=True,
                         text=True, timeout=5)
    return out.stdout

for d in ["example.com", "example.com; rm -rf /", "-fhosts.txt"]:
    try:
        print(d, "->", lookup_argv("dig", d))
    except ValueError as e:
        print(d, "-> rejected:", e)
```

```text
example.com -> ['/usr/bin/dig', 'example.com']
example.com; rm -rf / -> rejected: refusing suspicious argument: 'example.com; rm -rf /'
-fhosts.txt -> rejected: refusing suspicious argument: '-fhosts.txt'
```

For paths, resolve against a fixed root and reject anything that escapes it (`..`, an absolute path, a symlink that points outside) before opening a file — the same discipline as [11 — The Kali agent](11-kali-agent.md)'s policy layer. This runs:

```python
from pathlib import Path

ROOT = Path("/srv/tickets/attachments")

def safe_path(user_path: str) -> Path:
    p = (ROOT / user_path).resolve()  # collapses "..", follows symlinks
    if not p.is_relative_to(ROOT.resolve()):
        raise ValueError(f"path escapes {ROOT}: {user_path!r}")
    return p

for name in ["invoice-4.pdf", "../../../etc/passwd", "/etc/passwd"]:
    try:
        print(name, "->", safe_path(name))
    except ValueError as e:
        print(name, "-> rejected:", e)
```

```text
invoice-4.pdf -> /srv/tickets/attachments/invoice-4.pdf
../../../etc/passwd -> rejected: path escapes /srv/tickets/attachments: '../../../etc/passwd'
/etc/passwd -> rejected: path escapes /srv/tickets/attachments: '/etc/passwd'
```

`ROOT / "/etc/passwd"` is just `/etc/passwd`: joining an absolute path discards the root, which is why the check runs after the join. A check-then-open still races with a symlink swapped in between, so run a server that handles untrusted paths in a sandbox that can see only its root (§12.4).

### 12.4 Operational discipline

- **Least privilege and per-tool scopes.** Grant each server only what it needs; do not run a filesystem server over your whole home directory.
- **Human approval for side effects.** Gate writes and destructive tools behind user confirmation; use annotations to drive the prompt.
- **Sandbox servers.** Run them in containers with a read-only filesystem and no network where possible; a local server should prefer stdio to limit who can reach it. A host that offers one-click install of a local server **MUST** show the exact command and get the user's approval before running it.
- **Supply chain.** Running servers via `npx`/`uvx` fetches and executes code from the internet. **Pin versions**, read what you run, and treat an unpinned `@latest` as an unaudited dependency.

---

## 13. Operating remote servers

- **Stateless vs stateful.** A stateless server holds no per-connection session; any replica can serve any request. In `2026-07-28` this is the protocol itself, not an option: there are no protocol sessions. It scales horizontally with no sticky routing. A stateful handshake-era server mints an `Mcp-Session-Id`; to scale it you either pin a session to a replica or share session state in an external store.
- **Horizontal scaling with sessions.** If you must keep sessions, key them in Redis or similar so any replica can resume one, and expire them. Prefer stateless plus explicit, user-bound state handles passed as tool arguments.
- **Rate limits.** Enforce per-client and per-tool limits; a model in a loop can hammer a tool. Return a clear, actionable error on limit.
- **Observability.** Log every `tools/call` with arguments **redacted** (never log secrets or PII), plus latency and result size. Trace a request end to end. This is your eval and incident data (see [55 — Production](55-ai-production.md)).
- **Versioning.** Version your tool contract; add, don't mutate; deprecate with notice.
- **The MCP Registry.** There is an official registry at `registry.modelcontextprotocol.io` for discovering and publishing servers. It launched in **preview in September 2025**, and its API has been frozen at `v0.1` since 24 October 2025; as of October 2026 servers are published to it, but its README still describes a preview with general availability to follow, so confirm the current status on the project's own site before relying on it as stable infrastructure. A registry entry says who published a server, not that its code is safe (§12.1).

---

## 14. MCP vs the alternatives

| Approach | When it fits | When it is overhead |
|---|---|---|
| **Plain function calling** ([50](50-ai-prompting-apis.md)) | Tools live in the same process as your model call; one app | Never worth a protocol if you don't need reuse or isolation |
| **OpenAPI-described tools** | You already have a REST API with an OpenAPI spec and a host that can ingest it | Hand-written per host; no standard discovery, prompts or resources across apps |
| **MCP** | A capability must be **reused across hosts** or **isolated** in its own process/machine/sandbox | A single script's three local functions — the protocol layer and a server process add cost for nothing |
| **Agent-to-agent protocols** | Autonomous agents coordinating as peers (negotiation, delegation) | Reaching *tools and data* — that is MCP's job, not A2A's |

Rule of thumb: plain function calling when it is one app's own tools; MCP when the same capability serves many hosts or must run apart from the model. For the broader landscape of agent frameworks that consume MCP servers, see [61 — Agent frameworks](61-ai-agent-frameworks.md).

---

## 15. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Host shows "failed to parse message" | Server wrote to **stdout** on stdio | Log to **stderr**; remove stray `print()`/banners (§6.1) |
| Client sees no tools | Not initialized, or wrong capability/era | Confirm `initialize`/`server/discover` succeeded; check `list_tools` directly with the Inspector |
| `-32601 Method not found` | Calling a method the server does not serve | Check capabilities before using a feature; in `2026-07-28`, `ping`, `logging/setLevel` and `resources/subscribe` no longer exist |
| `400` with `-32020 HeaderMismatch` | `Mcp-Method`/`Mcp-Name`/`MCP-Protocol-Version` missing or not matching the body | Use an SDK client, or mirror the body fields into the headers exactly (§6.2) |
| Tool "fails" but it is `isError`, not an exception | Confusing the two channels | A bad arg or `ToolError` is an `isError` result, not a protocol error (§3.1, §4.3) |
| Model picks the wrong tool | Too many/overlapping tools; vague descriptions | Prune to task-shaped tools; sharpen names and "when not to use" (§11) |
| 401 loop on an HTTP server | Audience/metadata misconfig | Serve RFC 9728 metadata; validate the token audience; check the `resource` parameter (§7) |
| `421 Invalid Host header` or `403 Invalid Origin header` on localhost | DNS-rebinding protection rejecting a forged `Host` or a foreign `Origin` | Expected for a hostile page; for a legitimate client, connect to `127.0.0.1`/`localhost` or add its origin to the allowed list (§12.2) |
| Remote server loses sessions under load | Stateful sessions not shared across replicas | Go stateless, or share session state in an external store (§13) |
| `npx` server behaves differently today | Unpinned `@latest` changed | Pin the version; review on update (§12.4) |
| `ModuleNotFoundError: mcp.server.fastmcp` | Code written for the 1.x SDK | This is mcp 2.x: `from mcp.server.mcpserver import MCPServer` |

---

## 16. Key takeaways

- MCP turns an \(H \times S\) pile of bespoke integrations into \(H + S\): one protocol, like LSP for AI applications. It is not a model, not an agent, and not a reason to drop in-process function calling.
- Host holds the LLM and asks for consent; a client talks to exactly one server; the server exposes tools, resources and prompts and never sees the model (except through sampling, deprecated in `2026-07-28`).
- The wire is JSON-RPC 2.0. A tool failure is an `isError` result for the model; a malformed/unroutable request is a JSON-RPC error. Keep them straight.
- The installed SDK speaks `2026-07-28`: stateless, a per-request version envelope in `_meta`, `server/discover`, no sessions, no GET stream, no resumability. The `initialize` handshake (`2025-11-25` and earlier) is the legacy path that many deployed servers still use, and dual-era SDKs speak both.
- Build with `MCPServer` (Python `mcp` 2.x) / `McpServer` (TypeScript v2, `@modelcontextprotocol/server`); test in-process with `Client`; debug with a pinned Inspector; bridge to the local model by mapping `inputSchema` into OpenAI tool definitions, turning Qwen3's thinking off per request and stripping any think block.
- Tool definitions cost tokens on every request and eat a 14B model's small window — keep the surface to a handful of sharp, task-shaped tools with tight schemas and actionable errors.
- Remote servers need OAuth 2.1 with audience validation and no token passthrough; local HTTP servers need Origin/Host validation and loopback binding; stdio servers must keep stdout clean and take secrets from the environment. Annotations are hints from a possibly untrusted server, not a security control.

---

## 17. Exercises

!!! question "Exercise 17.1"

    A company runs 4 AI applications and wants each to reach 6 data sources. How many connectors with bespoke pairwise integration, and how many over a shared protocol? How much do you save? Then they add a fifth application — how does each number change?

??? success "Answer"

    Bespoke \(= 4 \times 6 = 24\). Shared \(= 4 + 6 = 10\). Saving \(= 24 - 10 = 14\). Adding a fifth application: bespoke \(= 5 \times 6 = 30\) (up by 6, one new connector per source), shared \(= 5 + 6 = 11\) (up by 1). The product grows per-source; the sum grows by one.

!!! question "Exercise 17.2"

    Your MCP integration exposes 12 tools, each about 100 tokens of schema, to a model handling 3000 requests a day. At an assumed \$2 per million input tokens, what do the tool definitions cost per day?

??? success "Answer"

    Per request: \(12 \times 100 = 1200\) tokens. Per day: \(1200 \times 3000 = 3\,600\,000\) tokens. Cost: \(3.6 \times 2 = \$7.20\) per day — spent before any user text. Pruning unused tools is a direct cost and latency win.

!!! question "Exercise 17.3"

    You connect 20 tools, each about 150 tokens, to a local model with an 8192-token context window. What fraction of the window goes to tool definitions? Why is this worse on a 14B model than on a frontier model?

??? success "Answer"

    \(20 \times 150 = 3000\) tokens; \(3000 / 8192 \approx 0.366\), about 36.6% of the window — more than a third gone before the conversation. It is worse on a small model because it has less room to spare *and* lower tool-selection accuracy as the list grows, so each extra schema both crowds the window and raises the chance of a wrong tool or mis-filled argument.

!!! question "Exercise 17.4"

    A host gets two tools from a server: `search_tickets` with `readOnlyHint: true` and `delete_ticket` with `readOnlyHint: false, destructiveHint: true`. How should the host treat each, and are these hints a security control?

??? success "Answer"

    If the user trusts this server, the host may let them pre-approve `search_tickets` (no side effects claimed), but it should require explicit approval before every `delete_ticket` call (a destructive write). The hints are **advisory** for UX and the model, not a security boundary: the spec says clients **MUST** treat annotations as untrusted unless the server is trusted, the server must still enforce authorization itself, and a tool with no annotations at all gets the pessimistic defaults (destructive, open-world).

!!! question "Exercise 17.5"

    Classify each as a protocol error (JSON-RPC `error`) or a tool error (`isError` result): (a) the client calls `tools/call` with a tool name the server doesn't have; (b) the client sends malformed JSON; (c) a `search_tickets` call passes `limit: 999` when the schema caps it at 50; (d) the client calls `resources/subscribe` on a server that doesn't support subscriptions.

??? success "Answer"

    (a) Protocol error per the spec: `-32602`, which is what the TypeScript SDK answers ("Tool nope not found"). The Python SDK 2.3.0 instead returns an `isError` result "Unknown tool: nope", so a client should handle both. (b) Protocol error — `-32700 Parse error` (the Python SDK's HTTP transport answers it with a `400`; its stdio transport drops the line). (c) Tool error — argument validation fails and returns `isError: true`. (d) Protocol error — `-32601 Method not found`; in `2026-07-28` the method does not exist at all, and subscriptions go through `subscriptions/listen`. Rule: a request that is malformed, unroutable, or names a tool that does not exist is a protocol error; a call that reaches a real tool and fails, including argument validation, comes back as an `isError` result the model can read.

!!! question "Exercise 17.6"

    You are shipping a remote MCP server behind OAuth. Which RFC must your server implement so clients can discover its authorization server, and what must the server check about every access token before honoring it?

??? success "Answer"

    The server must implement **OAuth 2.0 Protected Resource Metadata (RFC 9728)** so clients discover the authorization server from the `resource_metadata` parameter of the `WWW-Authenticate` header on a 401 (or, without it, from `/.well-known/oauth-protected-resource`). For every token it must validate that the token was issued **for this server as the audience** (Resource Indicators, RFC 8707) and reject tokens minted for any other resource. It must **not** forward (pass through) the token to a downstream API.

!!! question "Exercise 17.7"

    A teammate's stdio MCP server "works when I run it by hand" but the desktop host reports a parse error on startup. The server prints `Loaded 3 plugins` at import time. What is wrong and how do you fix it?

??? success "Answer"

    On stdio the server's **stdout is the JSON-RPC wire**. `print("Loaded 3 plugins")` writes a non-JSON line to stdout and corrupts the stream, so the host cannot parse it; a terminal does not care, which is why it "works by hand." Fix: send all logging to **stderr** (`logging.basicConfig(stream=sys.stderr)` or `print(..., file=sys.stderr)`), and never write non-protocol output to stdout.

!!! question "Exercise 17.8"

    You install a popular MCP server with `npx some-mcp-server@latest`. A week later your agent starts exfiltrating data. Name two MCP-specific attack patterns that fit, and two concrete mitigations you should have had in place.

??? success "Answer"

    Patterns: a **rug pull** (the `@latest` server changed its tool definitions after you approved it) and **tool poisoning** (a malicious instruction embedded in a tool description or result). Mitigations: **pin the version** (not `@latest`) and review on update; **sandbox** the server (container, read-only FS, no/limited network) and require **human approval for side-effecting tools**; review server tool descriptions as untrusted content. See [56 — Safety](56-ai-safety.md).

---

Next: [61 — Agent frameworks](61-ai-agent-frameworks.md).
