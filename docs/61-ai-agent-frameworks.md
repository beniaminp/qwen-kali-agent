# 61 — Agent frameworks: what to use, when and why

*Last verified: 28 October 2026. Every Python example was run against a local OpenAI-compatible stub; package versions are named in [§15](#15-the-comparison-matrix). Framework status, ownership and licences were checked against official repos, PyPI and docs in October 2026.*

[52 — Agents and tool use](52-ai-agents.md) built an agent from scratch: a model, a tool registry, a dispatcher, a loop, a stop condition — about sixty lines of plain Python. This chapter answers the question that follows: *should you keep writing that yourself, or adopt a framework, and if so which one?* It is a decision guide, not a catalogue. You will see what a framework actually gives you and what it costs, a procedure for deciding whether you need one, the landscape sorted into categories, one honest section per framework (the same structure every time), the **same small task** implemented and run against a stub in ten-plus of them so the differences are concrete, a comparison matrix and a lines-of-code count from the real files, a selection guide, the cross-cutting concerns (observability, evaluation, testing, upgrades, security, cost), and migration paths in and out.

Read [52](52-ai-agents.md) first for the loop and the agentic-workflow patterns (chaining, routing, parallelization, orchestrator–workers, evaluator–optimizer), and [58 — MCP](58-ai-mcp.md) for the tool protocol every serious framework now speaks. The handbook's concrete agent — a local Qwen3-14B driving typed tools against an isolated Kali VM — is [11 — The Kali agent](11-kali-agent.md), whose [§3.4](11-kali-agent.md#34-framework-comparison) already compares *agent harnesses* (OpenCode, hackingBuddyGPT, HexStrike) for that security track; this chapter agrees with it and covers the general-purpose *application* frameworks instead.

The one idea to carry through: **a framework is leverage you rent, not capability you own.** It removes boilerplate and adds structure, and in exchange you debug through its abstractions and follow its upgrade treadmill. Rent it when the leverage beats the baseline you just saw — and measure, do not assume.

---

## 1. What a framework gives you — and what it costs

An "agent framework" is a library that owns some of the machinery around a model so you write less of it. The good ones give you most of this list; none is magic.

### 1.1 The ten things a framework can give you

1. **Model abstraction.** One interface over many providers (OpenAI, Anthropic, Gemini, local OpenAI-compatible, Bedrock). Swap the model without rewriting the app.
2. **Tool-schema generation.** Turn a typed Python function into the JSON Schema the model needs ([52](52-ai-agents.md)), from signatures and docstrings, instead of hand-writing it.
3. **The loop.** The `model → tool → observation → model` cycle, the stop condition, and the step budget ([52](52-ai-agents.md)) — built, tested and reused.
4. **State and memory.** Conversation history, scratchpads, summarisation of old turns, long-term memory stores.
5. **Multi-agent orchestration.** Patterns for several agents — orchestrator–workers, pipelines, critics ([52](52-ai-agents.md)) — with the message plumbing between them.
6. **Retrieval integrations.** Connectors to vector stores, rerankers and document loaders for RAG ([51 — RAG](51-ai-rag.md)).
7. **Streaming.** Token and event streaming to the client, including partial tool calls.
8. **Tracing / observability.** Spans for each model and tool call, token counts, latencies — locally or to a hosted backend ([§18.1](#181-observability)).
9. **Durable execution and human-in-the-loop.** Checkpoint a long run, pause for approval, resume after a crash ([52](52-ai-agents.md)).
10. **Deployment.** A server, a REST/streaming API, session storage, sometimes a UI.

### 1.2 What a framework costs

Nothing on that list is free. The costs are real and recurring:

- **Abstraction you must debug through.** When the agent misbehaves, the bug is in *your prompt, the model, or the framework's hidden glue* — and the framework's call stack is between you and the HTTP request. The hand-written baseline of [§2.1](#21-the-no-framework-baseline) has no such layer.
- **Churn and breaking changes.** These libraries move monthly. An API you learned in the spring is deprecated by the autumn (you will see one such rename in [§4](#4-graph-state-machine-orchestration-langgraph-langchain)). Pin versions ([§18.4](#184-managing-upgrades)).
- **Lock-in.** Your tools, prompts and state live inside the framework's types. Leaving means a rewrite — unless you wrapped it ([§19](#19-migration-paths)).
- **Hidden prompts.** Many frameworks inject their own system prompt, output-format instructions and few-shot examples. You did not write them, they count against your context and bill, and they change between versions. You will measure one at **3,544 characters** in [§9](#9-code-agent-libraries-smolagents).
- **Dependency weight.** A one-function feature can pull in hundreds of transitive packages, lengthening installs, container images and your security-review surface.
- **Performance.** Extra Python per step, extra tokens per request. The Python is usually negligible; the tokens are not always. A local model must prefill every injected prompt token, and at the roughly 120 prompt tokens/s that [50](50-ai-prompting-apis.md) estimates for Qwen3-14B 4-bit on an M1 Pro, an 886-token framework prompt adds about 7 s to a call the server cannot serve from its prompt cache ([§18.6](#186-cost-and-token-overhead)).

!!! warning

    The failure mode is adopting a multi-agent framework for a problem that is a single tool call in an existing service. You pay every cost above and use one item from the benefits list. [§2](#2-do-you-need-a-framework) is the guard against that.

### 1.3 A software analogy

A framework is a web framework for agents. Writing the loop yourself is writing an HTTP server with the standard library: fine, educational, and sometimes exactly right for a tiny service. A framework is Django or Rails: it gives you routing, ORM, auth and admin, and in return you build it *their* way and upgrade when they say. You would not hand-roll HTTP for a large app; you also would not pull in Django to serve one static file.

---

## 2. Do you need a framework?

Start from the baseline and justify every departure from it.

### 2.1 The no-framework baseline

This is the whole agent for the running example of this chapter — *answer a question about an order using two tools*, `get_order(order_id)` and `get_shipping_status(tracking_id)` — in plain Python with the `openai` client. It is the yardstick every framework is measured against. (`enable_thinking=False` and the `content or ""` guard are explained in [§14](#14-local-models-the-thinking-trap); keep them in mind.)

```python
import json
from openai import OpenAI

BASE = "http://127.0.0.1:8080/v1"          # the handbook's mlx_lm.server
MODEL = "mlx-community/Qwen3-14B-4bit"
client = OpenAI(base_url=BASE, api_key="local")


def get_order(order_id):
    return {"order_id": order_id, "status": "shipped", "amount": 129.99,
            "tracking_id": "TRK42", "_stub": "order_lookup_ok"}


def get_shipping_status(tracking_id):
    return {"tracking_id": tracking_id, "carrier": "UPS",
            "status": "in_transit", "eta": "2026-11-02",
            "_stub": "shipping_lookup_ok"}


TOOLS = [
    {"type": "function", "function": {
        "name": "get_order", "description": "Look up an order by id.",
        "parameters": {"type": "object",
                       "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "get_shipping_status",
        "description": "Look up the shipping status for a tracking id.",
        "parameters": {"type": "object",
                       "properties": {"tracking_id": {"type": "string"}},
                       "required": ["tracking_id"]}}},
]
FUNCS = {"get_order": get_order, "get_shipping_status": get_shipping_status}


def run(task, max_steps=6):
    messages = [{"role": "user", "content": task}]
    for _ in range(max_steps):
        r = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        msg = r.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        messages.append(msg.model_dump(exclude_none=True))
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            obs = FUNCS[tc.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(obs)})
    return None


print(run("Where is my order A1001?"))
```

```text
Order A1001 shipped via UPS (tracking TRK42) and is in transit, ETA 2026-11-02. Amount charged: $129.99.
```

That is **43 lines of code** (54 with blank lines). It generates no schema for you — you wrote `TOOLS` by hand — and it owns its own loop. Every framework below shrinks that wiring; the question is whether the shrink is worth the costs of [§1.2](#12-what-a-framework-costs).

!!! note

    The output above came from a local stub standing in for `mlx_lm.server` ([10 — Export and serving](10-export-serving.md)): it returns a `get_order` tool call, then a `get_shipping_status` call once the first tool's result is in the request, then a canned final sentence. The stub only moves on when it sees the tools' return values, so the code really ran both tools. Every Python framework run in this chapter was pointed at the same stub (DSPy at a variant that speaks its text format, [§10](#10-prompt-optimization-programming-dspy)) and printed the **same answer**, because the sentence is the stub's. A real Qwen3-14B should produce the same shape for this task (two tool calls, one answer), in its own words. The stub also imitates Qwen3's default *thinking* behaviour ([§14](#14-local-models-the-thinking-trap)).

### 2.2 The decision procedure

Work top to bottom; stop at the first "yes".

1. **Is it a single call or a fixed workflow?** Classify a ticket, extract invoice fields, a two-step extract-then-draft chain ([52](52-ai-agents.md)). → **No framework.** The `openai` client plus a `for` loop is less code than learning one, and you can read it.
2. **Is it one tool-calling feature inside an existing service?** → **No framework, or the thinnest possible one.** Add the tool round trip ([50](50-ai-prompting-apis.md)) to the code you have. A framework's server, state model and deployment story are things you already own.
3. **Does the path genuinely depend on intermediate results you cannot enumerate, over many steps?** → **A lightweight agent library** ([§7](#7-typed-lightweight-agent-libraries-pydantic-ai-agno)) earns the loop, retries and schema generation.
4. **Do you need durable, resumable, human-gated workflows** (hours-long runs, approvals, crash recovery)? → **A graph/state-machine or durable-execution framework** ([§4](#4-graph-state-machine-orchestration-langgraph-langchain), [§11](#11-stateful-and-durable-execution-platforms)).
5. **Is retrieval the core of the product** (RAG over documents)? → **A data/RAG framework** ([§6](#6-data-rag-centric-frameworks-llamaindex-haystack)).
6. **Do genuinely separate roles with different tools help, proven by an eval?** → **A multi-agent framework** ([§5](#5-role-based-multi-agent-crewai)) — but re-read [52](52-ai-agents.md) first; most "multi-agent" problems are one agent with good tools.
7. **Are non-engineers building the automations?** → **A low-code builder** ([§13](#13-low-code-and-visual-builders)).

The honest default for an engineer with a model endpoint and a product is **1 or 2**. Reach higher only when a concrete need on the list pulls you there.

---

## 3. The categories

Every framework in this chapter sits in one of these buckets. The bucket predicts its shape far better than its marketing does.

| Category | What it is for | Frameworks (this chapter) |
|---|---|---|
| **Graph / state-machine orchestration** | Explicit control flow, durable state, resumable graphs | LangGraph, LangChain |
| **Role-based multi-agent** | Crews of role-playing agents with tasks | CrewAI |
| **Data / RAG-centric** | Retrieval and document pipelines first, agents second | LlamaIndex, Haystack |
| **Typed / lightweight agent libraries** | A thin, typed loop over tools | Pydantic AI, Agno |
| **Vendor agent SDKs** | A model or cloud vendor's own agent layer | OpenAI Agents SDK, Microsoft Agent Framework, Google ADK, AWS Strands, Claude Agent SDK |
| **Code-agent libraries** | The model writes code that calls tools | smolagents |
| **Prompt-optimization / programming** | Compile and optimise prompts from typed signatures | DSPy |
| **Stateful / durable-execution platforms** | Persistent agent state; crash-safe long runs | Letta; Temporal, DBOS, Inngest, Restate |
| **TypeScript frameworks** | Agents inside a JS/TS product | Mastra, Vercel AI SDK |
| **Low-code / visual builders** | Drag-and-drop for non-engineers | n8n, Dify, Flowise, Langflow |

Every per-framework section below uses the **same ten-part structure**: what it is and who maintains it; the mental model in two or three sentences; a minimal example (run against the stub for the Python libraries that install cleanly); strengths; weaknesses ("minuses"); use it when; avoid it when; local OpenAI-compatible models (yes/no/how — see [§14](#14-local-models-the-thinking-trap)); MCP support ([58](58-ai-mcp.md)); licence.

---

## 4. Graph / state-machine orchestration: LangGraph & LangChain

### 4.1 LangGraph

**What and who.** LangGraph is a low-level orchestration runtime for stateful, multi-step agents, maintained by LangChain Inc. It reached **v1.0 in October 2025** (the version used here is 1.2.18) with a stability commitment: no breaking changes until 2.0.

**Mental model.** You describe the agent as a **graph**: nodes are functions (call the model, run a tool), edges are the control flow, and a shared typed **state** object flows through them. A **checkpointer** persists that state after every node, so a run can be interrupted for human approval and resumed later — even after a crash. For the common case, LangChain's `create_agent` ([§4.2](#42-langchain)) builds a ready-made ReAct-style graph for you.

**Minimal example** (run against the stub; `langchain` 1.4.7 + `langgraph` 1.2.18):

```python
import os
os.environ["OPENAI_API_KEY"] = "local"
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

BASE = "http://127.0.0.1:8080/v1"
MODEL = "mlx-community/Qwen3-14B-4bit"


def get_order(order_id: str) -> dict:
    """Look up an order by id; returns its status, amount and tracking_id."""
    return {"order_id": order_id, "status": "shipped", "amount": 129.99,
            "tracking_id": "TRK42", "_stub": "order_lookup_ok"}


def get_shipping_status(tracking_id: str) -> dict:
    """Look up the shipping status for a tracking id."""
    return {"tracking_id": tracking_id, "carrier": "UPS",
            "status": "in_transit", "eta": "2026-11-02",
            "_stub": "shipping_lookup_ok"}


llm = ChatOpenAI(
    model=MODEL, base_url=BASE, api_key="local", temperature=0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
agent = create_agent(llm, tools=[get_order, get_shipping_status])

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Where is my order A1001?"}]})
print(result["messages"][-1].content)
```

```text
Order A1001 shipped via UPS (tracking TRK42) and is in transit, ETA 2026-11-02. Amount charged: $129.99.
```

**Strengths.** Mature durable state and checkpointing; first-class human-in-the-loop (`interrupt`) and time-travel/replay; explicit control flow you can see and test; deep LangSmith tracing; a large ecosystem.
**Minuses.** The graph/state mental model is a real learning curve; the wider LangChain ecosystem has renamed and retired APIs often (`AgentExecutor` → legacy, `create_react_agent` → `create_agent`); easy to over-engineer a problem that was a single call.
**Use it when.** Long-running, resumable, human-gated workflows; branching control flow; you want checkpointing without building it.
**Avoid it when.** A single tool-calling feature — the graph is overhead you will debug through.
**Local models.** Yes — `ChatOpenAI(base_url=…, extra_body=…)` points at `mlx_lm.server` and passes `enable_thinking` ([§14](#14-local-models-the-thinking-trap)).
**MCP.** Yes, via the `langchain-mcp-adapters` package (load MCP servers' tools into the graph).
**Licence.** MIT.

### 4.2 LangChain

**What and who.** LangChain is the high-level library (chains, prompts, model wrappers, hundreds of integrations) from the same company; it reached **v1.0 in October 2025** alongside LangGraph. As of 1.x, **LangChain's agents run on the LangGraph runtime** — `create_agent` is a thin, opinionated layer over a LangGraph graph. The pre-1.0 `AgentExecutor` style is legacy; the 0.3 line is in maintenance until December 2026.

**Mental model.** LangChain is the "batteries-included" surface — model abstraction, prompt templates, retrievers, output parsers, and the `init_chat_model`/`create_agent` conveniences — while LangGraph is the engine underneath. Use LangChain's pieces to assemble; drop to LangGraph when you need custom control flow.

**Minimal example.** Identical to [§4.1](#41-langgraph): `create_agent` is imported from `langchain.agents` and is the same call shown above. (That the LangGraph example's import moved from `langgraph.prebuilt.create_react_agent` to `langchain.agents.create_agent` between versions is a concrete taste of the churn in [§1.2](#12-what-a-framework-costs).)

**Strengths.** One of the broadest integration catalogues (hundreds of provider, vector-store and tool packages); a quick way to wire a model, a retriever and tools together; shares LangGraph's runtime and LangSmith tracing.
**Minuses.** Breadth means surface area and version drift; the 1.x stability promise covers `langchain` and `langchain-core`, while `langchain-community` is still on 0.x (0.4.2 in October 2026) and outside it; the many layers can obscure the one HTTP request you care about.
**Use it when.** You want conveniences and integrations and are comfortable on the LangChain/LangGraph stack.
**Avoid it when.** You want a minimal dependency footprint, or you dislike debugging through layers.
**Local models / MCP / Licence.** Same as LangGraph: OpenAI-compatible via `ChatOpenAI`; MCP via `langchain-mcp-adapters`; MIT.

---

## 5. Role-based multi-agent: CrewAI

**What and who.** CrewAI is a standalone multi-agent framework (not built on LangChain) from CrewAI Inc., version 1.15.32 here.

**Mental model.** You define **agents** with a `role`, `goal` and `backstory`, give each some tools, and compose **tasks** into a **crew** that runs them sequentially or hierarchically. It leans on role-play: the personas shape behaviour. Under the hood, CrewAI 1.x calls providers through its own native integrations (the `openai/` prefix selects its OpenAI provider, which honours `base_url`), with LiteLLM as an optional extra for other providers, so any OpenAI-compatible endpoint works.

**Minimal example** (run against the stub; `crewai` 1.15.32). One agent, two tools — the same task, to keep the comparison honest:

```python
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

BASE = "http://127.0.0.1:8080/v1"
MODEL = "mlx-community/Qwen3-14B-4bit"


@tool("get_order")
def get_order(order_id: str) -> dict:
    """Look up an order by id; returns status, amount and tracking_id."""
    return {"order_id": order_id, "status": "shipped", "amount": 129.99,
            "tracking_id": "TRK42", "_stub": "order_lookup_ok"}


@tool("get_shipping_status")
def get_shipping_status(tracking_id: str) -> dict:
    """Look up the shipping status for a tracking id."""
    return {"tracking_id": tracking_id, "carrier": "UPS",
            "status": "in_transit", "eta": "2026-11-02",
            "_stub": "shipping_lookup_ok"}


llm = LLM(model="openai/" + MODEL, base_url=BASE, api_key="local",
          temperature=0,
          extra_body={"chat_template_kwargs": {"enable_thinking": False}})
agent = Agent(role="Support agent",
              goal="Answer order questions using the tools.",
              backstory="You help customers track their orders.",
              tools=[get_order, get_shipping_status], llm=llm, verbose=False)
task = Task(description="Where is my order A1001?",
            expected_output="A one-sentence status with carrier and ETA.",
            agent=agent)
crew = Crew(agents=[agent], tasks=[task], verbose=False)
print(crew.kickoff())
```

```text
Order A1001 shipped via UPS (tracking TRK42) and is in transit, ETA 2026-11-02. Amount charged: $129.99.
```

(CrewAI logs an "approximate input token count" warning to *stderr* for model names it does not know; stdout is the single line above.)

**Strengths.** The role/task/crew model is intuitive and reads like an org chart; fast to stand up a plausible multi-agent demo; a large templates-and-tools ecosystem; a separate `CrewAI Flows` API for deterministic control when you need it.
**Minuses.** Role prompts do not enforce structure: outside Flows, the model decides the control flow, so crews can loop and spend tokens ([52](52-ai-agents.md)); the persona abstraction hides the prompts (the run above sent a persona system message and task framing you did not write, [§18.6](#186-cost-and-token-overhead)).
**Use it when.** Genuinely separable roles with distinct tools, and a demo-to-prototype pace matters.
**Avoid it when.** The task is one agent's job (as above) — you pay the multi-agent tax for nothing; or you need tight, auditable control flow.
**Local models.** Yes — `LLM(model="openai/…", base_url=…, extra_body=…)` through CrewAI's native OpenAI provider.
**MCP.** Yes — natively through the agent's `mcps` field, or `MCPServerAdapter` from `crewai-tools`.
**Licence.** MIT.

---

## 6. Data / RAG-centric frameworks: LlamaIndex & Haystack

### 6.1 LlamaIndex

**What and who.** LlamaIndex (LlamaIndex Inc.) is a data framework for LLM applications — ingestion, indexing, retrieval — that grew an agent layer. Core version 0.14.25 here.

**Mental model.** Everything is organised around getting your data to the model: loaders → nodes → indexes → retrievers → query engines, with **Workflows** (an event-driven, step-based engine) as the orchestration layer and `FunctionAgent`/`AgentWorkflow` as agents built on it. If the product is "answer from my documents", LlamaIndex is shaped for it.

**Minimal example** (run against the stub; `llama-index` 0.14.25, `llama-index-llms-openai-like` 0.8.0). `FunctionAgent.run` is async:

```python
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai_like import OpenAILike

BASE = "http://127.0.0.1:8080/v1"
MODEL = "mlx-community/Qwen3-14B-4bit"


def get_order(order_id: str) -> dict:
    """Look up an order by id; returns status, amount and tracking_id."""
    return {"order_id": order_id, "status": "shipped", "amount": 129.99,
            "tracking_id": "TRK42", "_stub": "order_lookup_ok"}


def get_shipping_status(tracking_id: str) -> dict:
    """Look up the shipping status for a tracking id."""
    return {"tracking_id": tracking_id, "carrier": "UPS",
            "status": "in_transit", "eta": "2026-11-02",
            "_stub": "shipping_lookup_ok"}


llm = OpenAILike(
    model=MODEL, api_base=BASE, api_key="local",
    is_chat_model=True, is_function_calling_model=True, temperature=0,
    additional_kwargs={"extra_body": {
        "chat_template_kwargs": {"enable_thinking": False}}})
agent = FunctionAgent(tools=[get_order, get_shipping_status], llm=llm)


async def main():
    resp = await agent.run("Where is my order A1001?")
    print(str(resp))


asyncio.run(main())
```

```text
Order A1001 shipped via UPS (tracking TRK42) and is in transit, ETA 2026-11-02. Amount charged: $129.99.
```

**Strengths.** A very broad retrieval toolkit — chunking, many index types, rerankers, query transformations, hundreds of data connectors (LlamaHub); Workflows are a clean, typed orchestration primitive; strong for RAG and document agents.
**Minuses.** Large surface and fast-moving APIs (the `0.x` versioning tells you breaking changes are expected); the agent layer is younger than the retrieval layer; for a non-RAG tool loop it is more machinery than you need.
**Use it when.** Retrieval is central; you want batteries-included RAG with an agent on top.
**Avoid it when.** No documents are involved — a lighter library wires tools with less.
**Local models.** Yes — `OpenAILike(api_base=…, is_function_calling_model=True)` with `additional_kwargs={"extra_body": …}` for `enable_thinking`.
**MCP.** Yes, via `llama-index-tools-mcp` (`McpToolSpec`).
**Licence.** MIT.

### 6.2 Haystack

**What and who.** Haystack (deepset) is a mature, production-oriented framework for search and RAG pipelines that added agents and tools. Version 3.4.1 (`haystack-ai`) here.

**Mental model.** You build a **pipeline** of typed **components** connected by explicit edges — retrievers, generators, rankers, and an `Agent` component that runs a tool loop. The pipeline graph is serializable (to YAML), so the same definition runs in notebook and production.

**Minimal example** (run against the stub; `haystack-ai` 3.4.1):

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import tool
from haystack.utils import Secret

BASE = "http://127.0.0.1:8080/v1"
MODEL = "mlx-community/Qwen3-14B-4bit"


@tool
def get_order(order_id: str) -> dict:
    """Look up an order by id; returns status, amount and tracking_id."""
    return {"order_id": order_id, "status": "shipped", "amount": 129.99,
            "tracking_id": "TRK42", "_stub": "order_lookup_ok"}


@tool
def get_shipping_status(tracking_id: str) -> dict:
    """Look up the shipping status for a tracking id."""
    return {"tracking_id": tracking_id, "carrier": "UPS",
            "status": "in_transit", "eta": "2026-11-02",
            "_stub": "shipping_lookup_ok"}


generator = OpenAIChatGenerator(
    model=MODEL, api_base_url=BASE,
    api_key=Secret.from_token("local"),
    generation_kwargs={"temperature": 0,
                       "extra_body": {"chat_template_kwargs":
                                      {"enable_thinking": False}}})
agent = Agent(chat_generator=generator,
              tools=[get_order, get_shipping_status])
agent.warm_up()
result = agent.run(messages=[ChatMessage.from_user("Where is my order A1001?")])
print(result["messages"][-1].text)
```

```text
Order A1001 shipped via UPS (tracking TRK42) and is in transit, ETA 2026-11-02. Amount charged: $129.99.
```

**Strengths.** Explicit, typed, serializable pipelines that are genuinely production-minded; strong retrieval heritage; stable, predictable component contracts; clean local-model support.
**Minuses.** The pipeline/component model is more verbose for a simple agent; a smaller agent ecosystem than LangChain/LlamaIndex; fewer "magic" conveniences (which some teams consider a feature).
**Use it when.** Production RAG/search with a need for explicit, serializable, testable pipelines.
**Avoid it when.** You want a one-liner agent, or you are not doing retrieval.
**Local models.** Yes — `OpenAIChatGenerator(api_base_url=…, generation_kwargs={"extra_body": …})`.
**MCP.** Yes, via `mcp-haystack` (`MCPToolset`; stdio / SSE / Streamable HTTP).
**Licence.** Apache-2.0.

---

## 7. Typed / lightweight agent libraries: Pydantic AI & Agno

### 7.1 Pydantic AI

**What and who.** Pydantic AI, from the Pydantic team, brings the "FastAPI feel" to agents: typed, validated, minimal. It reached **1.0 in September 2025** and is on the 2.x line now (2.63.0 here).

**Mental model.** An `Agent` is typed on its dependencies and its output. Tools are plain decorated functions whose signatures become schemas; structured outputs are Pydantic models, validated (and retried) automatically. The library stays close to the request — no hidden orchestration graph.

**Minimal example** (run against the stub; `pydantic-ai` 2.63.0):

```python
import os
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

BASE = "http://127.0.0.1:8080/v1"
MODEL = "mlx-community/Qwen3-14B-4bit"

model = OpenAIChatModel(
    MODEL, provider=OpenAIProvider(base_url=BASE, api_key="local"))
settings = OpenAIChatModelSettings(
    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
agent = Agent(model, model_settings=settings)


@agent.tool_plain
def get_order(order_id: str) -> dict:
    """Look up an order by id; returns status, amount and tracking_id."""
    return {"order_id": order_id, "status": "shipped", "amount": 129.99,
            "tracking_id": "TRK42", "_stub": "order_lookup_ok"}


@agent.tool_plain
def get_shipping_status(tracking_id: str) -> dict:
    """Look up the shipping status for a tracking id."""
    return {"tracking_id": tracking_id, "carrier": "UPS",
            "status": "in_transit", "eta": "2026-11-02",
            "_stub": "shipping_lookup_ok"}


result = agent.run_sync("Where is my order A1001?")
print(result.output)
```

```text
Order A1001 shipped via UPS (tracking TRK42) and is in transit, ETA 2026-11-02. Amount charged: $129.99.
```

**Strengths.** Excellent typing and validation; small, legible surface; native structured output with validate-and-retry ([50](50-ai-prompting-apis.md)); first-class observability through Pydantic Logfire (OpenTelemetry); model-agnostic.
**Minuses.** Fewer batteries than the big ecosystems (you add retrieval yourself); durable execution comes through integrations with Temporal, DBOS and Prefect rather than a built-in checkpointer; multi-agent support is lighter than LangGraph's; younger than the incumbents.
**Use it when.** You want strict typing/validation and a thin, readable agent — the natural step up from the baseline for a Python service.
**Avoid it when.** You need heavy built-in RAG, visual tooling, or a built-in graph checkpointer with time travel.
**Local models.** Yes — `OpenAIChatModel` + `OpenAIProvider(base_url=…)`, with `OpenAIChatModelSettings(extra_body=…)` for `enable_thinking`.
**MCP.** Yes, native MCP client support.
**Licence.** MIT.

### 7.2 Agno

**What and who.** Agno (formerly **Phidata**, renamed in early 2025; repo `agno-agi/agno`) is a performance-focused agent library, version 3.1.11 here. The 2.x line added **AgentOS**, a FastAPI-based runtime for serving agents.

**Mental model.** A single `Agent` object bundles a model, tools, memory and knowledge (vector retrieval); Agno's own benchmarks stress low per-agent memory and fast instantiation. Scale out with teams; serve with AgentOS.

**Minimal example** (run against the stub; `agno` 3.1.11). Note the leanest wiring in this chapter — plain functions, no decorators:

```python
from agno.agent import Agent
from agno.models.openai.like import OpenAILike

BASE = "http://127.0.0.1:8080/v1"
MODEL = "mlx-community/Qwen3-14B-4bit"


def get_order(order_id: str) -> dict:
    """Look up an order by id; returns status, amount and tracking_id."""
    return {"order_id": order_id, "status": "shipped", "amount": 129.99,
            "tracking_id": "TRK42", "_stub": "order_lookup_ok"}


def get_shipping_status(tracking_id: str) -> dict:
    """Look up the shipping status for a tracking id."""
    return {"tracking_id": tracking_id, "carrier": "UPS",
            "status": "in_transit", "eta": "2026-11-02",
            "_stub": "shipping_lookup_ok"}


model = OpenAILike(id=MODEL, base_url=BASE, api_key="local",
                   extra_body={"chat_template_kwargs": {"enable_thinking": False}})
agent = Agent(model=model, tools=[get_order, get_shipping_status],
              telemetry=False)
resp = agent.run("Where is my order A1001?")
print(resp.content)
```

```text
Order A1001 shipped via UPS (tracking TRK42) and is in transit, ETA 2026-11-02. Amount charged: $129.99.
```

**Strengths.** Very low ceremony (the leanest wiring here, [§16](#16-lines-of-code-for-the-same-task)); built-in memory, knowledge/retrieval and session storage; AgentOS gives a serving runtime out of the box; broad model support including OpenAI-compatible.
**Minuses.** Smaller community than LangChain/LlamaIndex; the all-in-one `Agent` object can feel implicit; the Phidata→Agno rename means older tutorials reference the dead package.
**Use it when.** You want a batteries-included single-agent (or small-team) app with memory and a serving layer, with minimal code.
**Avoid it when.** You need the largest integration catalogue, or first-class graph checkpointing and time-travel.
**Local models.** Yes — `agno.models.openai.like.OpenAILike(base_url=…, extra_body=…)`.
**MCP.** Yes, via `agno.tools.mcp.MCPTools` (requires the `mcp` package).
**Licence.** Apache-2.0.

---

## 8. Vendor agent SDKs

A model or cloud vendor's own agent layer. Expect excellent support for *that* vendor's models and features, and variable support for everything else.

### 8.1 OpenAI Agents SDK

**What and who.** The OpenAI Agents SDK (`openai-agents`, 0.24.0 here) is OpenAI's lightweight agent framework; the experimental *Swarm* repository now points to it as Swarm's production-ready successor. It is provider-agnostic: besides OpenAI's Responses API it has an `OpenAIChatCompletionsModel` that points at any OpenAI-compatible endpoint.

**Mental model.** Primitives are **Agents** (a model plus instructions and tools), **handoffs** (one agent delegates to another), **guardrails** (input/output validation) and **sessions** (conversation state). Built-in tracing to the OpenAI dashboard (disable it for local work). Newer releases add *sandbox agents*, which work inside a container over long time horizons.

**Minimal example** (run against the stub; `openai-agents` 0.24.0). Use the chat-completions model so it targets `mlx_lm.server`, and disable tracing:

```python
from agents import (Agent, Runner, function_tool, ModelSettings,
                    OpenAIChatCompletionsModel, set_tracing_disabled)
from openai import AsyncOpenAI

BASE = "http://127.0.0.1:8080/v1"
MODEL = "mlx-community/Qwen3-14B-4bit"
set_tracing_disabled(True)


@function_tool
def get_order(order_id: str) -> dict:
    """Look up an order by id; returns status, amount and tracking_id."""
    return {"order_id": order_id, "status": "shipped", "amount": 129.99,
            "tracking_id": "TRK42", "_stub": "order_lookup_ok"}


@function_tool
def get_shipping_status(tracking_id: str) -> dict:
    """Look up the shipping status for a tracking id."""
    return {"tracking_id": tracking_id, "carrier": "UPS",
            "status": "in_transit", "eta": "2026-11-02",
            "_stub": "shipping_lookup_ok"}


client = AsyncOpenAI(base_url=BASE, api_key="local")
model = OpenAIChatCompletionsModel(model=MODEL, openai_client=client)
agent = Agent(name="support", model=model,
              tools=[get_order, get_shipping_status],
              model_settings=ModelSettings(extra_body={
                  "chat_template_kwargs": {"enable_thinking": False}}))

result = Runner.run_sync(agent, "Where is my order A1001?")
print(result.final_output)
```

```text
Order A1001 shipped via UPS (tracking TRK42) and is in transit, ETA 2026-11-02. Amount charged: $129.99.
```

**Strengths.** Small, clean primitives; handoffs and guardrails are well designed; strong tracing; native MCP; works with local endpoints via the chat-completions model.
**Minuses.** Best features (the Responses API, hosted tools, the trace dashboard) assume OpenAI's platform; local use means opting out of some of them; younger than the incumbents.
**Use it when.** You are on OpenAI, or want a minimal vendor-blessed loop and can point it at a local model.
**Avoid it when.** You want deep, vendor-neutral durable execution or the biggest integration catalogue.
**Local models.** Yes — `OpenAIChatCompletionsModel(openai_client=AsyncOpenAI(base_url=…))`; `ModelSettings(extra_body=…)` passes `enable_thinking`.
**MCP.** Yes, native (`MCPServerStdio`, `MCPServerStreamableHttp`).
**Licence.** MIT.

### 8.2 Microsoft Agent Framework (and the AutoGen / Semantic Kernel story)

**What and who.** The **Microsoft Agent Framework** (`agent-framework` on PyPI; also .NET) is Microsoft's unified agent stack, **generally available since April 2026** (1.0.0 reached PyPI on 2 April 2026; 1.23.0 by late October). The same teams built it as the **successor to both AutoGen and Semantic Kernel**:

- **AutoGen** (the research-oriented multi-agent framework) is **archived**. Its README says Microsoft no longer provides bug fixes, security fixes, new features or support, advises against production use, and points to the Agent Framework and a migration guide.
- **Semantic Kernel** (the enterprise SDK) is **still supported**: Microsoft committed to fixing critical bugs and security issues for at least one year after the Agent Framework's GA (so into at least April 2027), while most new features go to the Agent Framework. Its README now presents the Agent Framework as its successor and links a migration guide.

So for new projects the answer is the **Microsoft Agent Framework**; treat AutoGen and Semantic Kernel as legacy you migrate *from* — AutoGen urgently, since it no longer gets security fixes.

**Mental model.** It combines Semantic Kernel's enterprise plumbing (connectors, plugins, telemetry, .NET parity) with AutoGen's multi-agent patterns (group chat, hand-offs, workflows), under one API with OpenTelemetry built in.

**Minimal example.** Not shown here (this framework's sweet spot is Azure/.NET integration; it is documented, not benchmarked, in this chapter). In the 1.x Python API you wrap a chat client in an agent — `Agent(client=OpenAIChatCompletionClient(model=…, base_url=…), instructions=…, tools=[…])`, or `client.as_agent(...)` — and `await agent.run("…")`; see the official docs at <https://learn.microsoft.com/en-us/agent-framework/> for a current snippet.

**Strengths.** Enterprise-grade (observability, identity, Azure integration); .NET and Python parity; the consolidation ends the "AutoGen vs Semantic Kernel" confusion.
**Minuses.** Newest of the majors (GA only in 2026); heaviest when used outside the Microsoft/Azure ecosystem; migration work for existing AutoGen/SK code.
**Use it when.** A Microsoft/Azure shop, .NET in the mix, or you were on AutoGen/Semantic Kernel.
**Avoid it when.** A small Python-only service with no Azure footprint — lighter libraries fit better.
**Local models.** Yes — `OpenAIChatCompletionClient(base_url=…)`, with `default_options={"extra_body": {…}}` on the `Agent` for `enable_thinking`. A quick check against this chapter's stub (`agent-framework-core` 1.23.0) made both tool calls with thinking off and printed the same answer.
**MCP.** Yes, native MCP support.
**Licence.** MIT.

### 8.3 Google Agent Development Kit (ADK)

**What and who.** Google's **Agent Development Kit** (`google-adk`, 2.13.0 here; also TypeScript, Go, Java and Kotlin) is Google's open-source agent framework, model-agnostic through adapters such as **LiteLLM** for non-Gemini models.

**Mental model.** Agents (`LlmAgent`) are composed into workflows with explicit **runners**, **sessions** and **callbacks**; it ships an evaluation harness and deployment paths to Google Cloud (its managed agent runtime, Cloud Run, GKE). It is session- and runner-centric, so even a one-shot needs a session.

**Minimal example** (run against the stub; `google-adk` 2.13.0 with the `[extensions]` extra plus `mcp`, so `LiteLlm` is available). It is the most verbose in this chapter because of the runner/session ceremony:

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

BASE = "http://127.0.0.1:8080/v1"
MODEL = "mlx-community/Qwen3-14B-4bit"


def get_order(order_id: str) -> dict:
    """Look up an order by id; returns status, amount and tracking_id."""
    return {"order_id": order_id, "status": "shipped", "amount": 129.99,
            "tracking_id": "TRK42", "_stub": "order_lookup_ok"}


def get_shipping_status(tracking_id: str) -> dict:
    """Look up the shipping status for a tracking id."""
    return {"tracking_id": tracking_id, "carrier": "UPS",
            "status": "in_transit", "eta": "2026-11-02",
            "_stub": "shipping_lookup_ok"}


agent = LlmAgent(
    name="support",
    model=LiteLlm(model="openai/" + MODEL, api_base=BASE, api_key="local",
                  extra_body={"chat_template_kwargs": {"enable_thinking": False}}),
    tools=[get_order, get_shipping_status])
runner = InMemoryRunner(agent=agent, app_name="demo")


async def main():
    await runner.session_service.create_session(
        app_name="demo", user_id="u", session_id="s")
    msg = types.Content(role="user",
                        parts=[types.Part(text="Where is my order A1001?")])
    final = ""
    async for ev in runner.run_async(user_id="u", session_id="s",
                                     new_message=msg):
        if ev.is_final_response() and ev.content:
            final = ev.content.parts[0].text
    print(final)


asyncio.run(main())
```

```text
Order A1001 shipped via UPS (tracking TRK42) and is in transit, ETA 2026-11-02. Amount charged: $129.99.
```

**Strengths.** Production-minded (sessions, evaluation, deployment to Vertex); strong multi-agent and workflow primitives; model-agnostic via LiteLLM; native MCP.
**Minuses.** Verbose for simple cases (runner + session service even for one turn); best features point at Google Cloud; the 2.0 major release (May 2026) brought breaking changes recently (for example `MCPToolset` is now a deprecated alias of `McpToolset`).
**Use it when.** On Google Cloud/Gemini, or you want the built-in eval and deployment path.
**Avoid it when.** You want a two-line agent, or you have no Google-Cloud footprint.
**Local models.** Yes — `LiteLlm(model="openai/…", api_base=…, extra_body=…)` (install `google-adk[extensions]` and `mcp`).
**MCP.** Yes, native (`McpToolset`).
**Licence.** Apache-2.0.

### 8.4 AWS Strands Agents

**What and who.** **Strands Agents** (`strands-agents`, 1.62.0 here; also TypeScript) is AWS's open-source, model-driven agent SDK, **v1.0 since July 2025**, with multi-agent primitives (graph, swarm) and A2A (agent-to-agent) support.

**Mental model.** "Model-driven": you give the agent a model, tools and a prompt, and let the model drive the loop; the SDK stays thin. Deep Bedrock integration, but model-agnostic — it has a first-class `OpenAIModel` for OpenAI-compatible endpoints.

**Minimal example** (run against the stub; `strands-agents` 1.62.0):

```python
from strands import Agent, tool
from strands.models.openai import OpenAIModel

BASE = "http://127.0.0.1:8080/v1"
MODEL = "mlx-community/Qwen3-14B-4bit"


@tool
def get_order(order_id: str) -> dict:
    """Look up an order by id; returns status, amount and tracking_id."""
    return {"order_id": order_id, "status": "shipped", "amount": 129.99,
            "tracking_id": "TRK42", "_stub": "order_lookup_ok"}


@tool
def get_shipping_status(tracking_id: str) -> dict:
    """Look up the shipping status for a tracking id."""
    return {"tracking_id": tracking_id, "carrier": "UPS",
            "status": "in_transit", "eta": "2026-11-02",
            "_stub": "shipping_lookup_ok"}


model = OpenAIModel(
    client_args={"base_url": BASE, "api_key": "local"}, model_id=MODEL,
    params={"temperature": 0,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})
agent = Agent(model=model, tools=[get_order, get_shipping_status],
              callback_handler=None)
result = agent("Where is my order A1001?")
print(result.message["content"][0]["text"])
```

```text
Order A1001 shipped via UPS (tracking TRK42) and is in transit, ETA 2026-11-02. Amount charged: $129.99.
```

**Strengths.** Clean, thin, model-driven design; Apache-2.0; strong AWS/Bedrock integration and OpenTelemetry tracing; native MCP; A2A support.
**Minuses.** Newer ecosystem; best observability/deployment lean on AWS; "let the model drive" gives you less explicit control than a graph.
**Use it when.** On AWS/Bedrock, or you want a thin, open, model-driven SDK that still speaks OpenAI-compatible.
**Avoid it when.** You need explicit, auditable control flow or heavy built-in RAG.
**Local models.** Yes — `OpenAIModel(client_args={"base_url": …}, params={"extra_body": …})`.
**MCP.** Yes, native (`strands.tools.mcp.MCPClient`).
**Licence.** Apache-2.0.

### 8.5 Claude Agent SDK

**What and who.** Anthropic's **Claude Agent SDK** (`claude-agent-sdk` for Python, `@anthropic-ai/claude-agent-sdk` for TypeScript) is the agent harness that powers Claude Code, **renamed from the "Claude Code SDK"** in September 2025 (old packages `claude-code-sdk` and `@anthropic-ai/claude-code`; the rename also renamed `ClaudeCodeOptions` → `ClaudeAgentOptions`, stopped using Claude Code's system prompt by default, and moved the docs). It runs the Claude Code binary as a library — file tools, bash, sub-agents, MCP, permissions, sessions, hooks — behind a programmatic API.

**Mental model.** Rather than a bare model loop, you drive the *same agent harness* a coding assistant uses: a rich built-in toolset, a permission system, sub-agents and MCP, orchestrated for you. It is built around Claude models and their tool protocol.

**Minimal example.** Not run here: it is designed around Claude models and the Claude Code harness, not an arbitrary OpenAI-compatible local endpoint, so it falls outside this chapter's "point it at `mlx_lm.server`" test. The documented Python shape is `async for message in query(prompt=…, options=ClaudeAgentOptions(...))`; see <https://code.claude.com/docs/en/agent-sdk/overview> for a current snippet.

**Strengths.** The broadest built-in toolset of the SDKs here (file read/write/edit, shell, web search, sub-agents); a detailed permissions model; the same harness Claude Code runs on.
**Minuses.** Tied to Claude models — not a local-model framework; heavier and more opinionated than a generic loop; the recent rename means older material uses the old names.
**Use it when.** You are building on Claude and want coding-assistant-grade tooling and sub-agents without rebuilding the harness.
**Avoid it when.** You need to run on a local OpenAI-compatible model (use a vendor-neutral library instead).
**Local models.** Not meaningfully — it targets Claude; a local Qwen behind an OpenAI-compatible endpoint is not its path.
**MCP.** Yes, first-class (it is an MCP host).
**Licence.** Use is governed by Anthropic's Commercial Terms of Service. The Python package's own source is MIT-licensed, but it runs a bundled Claude Code binary, and the TypeScript package is not open source.

---

## 9. Code-agent libraries: smolagents

**What and who.** **smolagents** (Hugging Face, 1.26.0 here) is a deliberately small agent library whose headline idea is the **code agent**: the model writes Python that calls your tools, instead of emitting JSON tool calls.

**Mental model.** A `CodeAgent` prompts the model to express each action as a snippet of Python — `result = get_order(order_id="A1001")` — which runs in a sandbox (local, or E2B/Modal/Docker/Pyodide). This composes multiple calls, loops and data handling in one step, at the cost of a much larger reasoning prompt and the need to sandbox code execution ([56 — Safety](56-ai-safety.md)). A `ToolCallingAgent` variant asks for ordinary JSON tool calls, which is what maps onto an OpenAI-compatible endpoint.

**Minimal example** (run against the stub; `smolagents` 1.26.0). The `ToolCallingAgent` is used because it requests native tool calls (the `tools` field, with `tool_choice: "required"`) and finishes by calling a built-in `final_answer` tool. It replays the history to the model as plain text, though — an assistant turn `Calling tools: …` and a user turn `Observation: …` — not as `tool` messages. The `CodeAgent` headline mode is shown as a snippet afterward:

```python
from smolagents import ToolCallingAgent, OpenAIServerModel, tool

BASE = "http://127.0.0.1:8080/v1"
MODEL = "mlx-community/Qwen3-14B-4bit"


@tool
def get_order(order_id: str) -> dict:
    """Look up an order by id.

    Args:
        order_id: the order identifier, e.g. A1001.
    """
    return {"order_id": order_id, "status": "shipped", "amount": 129.99,
            "tracking_id": "TRK42", "_stub": "order_lookup_ok"}


@tool
def get_shipping_status(tracking_id: str) -> dict:
    """Look up the shipping status for a tracking id.

    Args:
        tracking_id: the carrier tracking id, e.g. TRK42.
    """
    return {"tracking_id": tracking_id, "carrier": "UPS",
            "status": "in_transit", "eta": "2026-11-02",
            "_stub": "shipping_lookup_ok"}


model = OpenAIServerModel(
    model_id=MODEL, api_base=BASE, api_key="local",
    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
agent = ToolCallingAgent(tools=[get_order, get_shipping_status], model=model,
                         verbosity_level=0)
print(agent.run("Where is my order A1001?"))
```

```text
Order A1001 shipped via UPS (tracking TRK42) and is in transit, ETA 2026-11-02. Amount charged: $129.99.
```

The `CodeAgent` is the same two lines with a different class — `CodeAgent(tools=[…], model=model)` — but it needs a model fluent in code-writing and a sandbox for execution, so it is not exercised against the simple stub here. (Its first request was logged, though: it describes the tools inside an 8,814-character system prompt instead of sending a `tools` field.)

**Strengths.** The code-agent idea is powerful for multi-step data manipulation (fewer round trips); genuinely small and readable core; Hugging Face Hub integration; `ToolCallingAgent` gives a conventional path when you want it.
**Minuses.** The injected system prompts are **large** — 3,544 characters for the `ToolCallingAgent` run above, 8,814 for a `CodeAgent` with the same two tools ([§18.6](#186-cost-and-token-overhead)) — which is real tokens and latency on a local model; executing model-written code *demands* a sandbox ([56](56-ai-safety.md)); smaller feature set than the big frameworks.
**Use it when.** Tasks that benefit from the model composing code (data wrangling, multi-tool steps), and you can sandbox execution.
**Avoid it when.** You cannot sandbox code, or the token/latency budget is tight on a local model, or you only need a simple JSON tool loop (the baseline is lighter).
**Local models.** Yes — `OpenAIServerModel(api_base=…, extra_body=…)`. The `CodeAgent` mode needs a model that writes good Python.
**MCP.** Yes, via `ToolCollection.from_mcp`.
**Licence.** Apache-2.0.

---

## 10. Prompt-optimization / programming: DSPy

**What and who.** **DSPy** (originated at Stanford NLP; MIT licence; a DSPy open-source team at Databricks and tight MLflow integration; 3.4.1 here) is a different animal: a framework for *programming* — not prompting — language models, where prompts are **compiled** from typed signatures and **optimised** against a metric.

**Mental model.** You declare **signatures** (`"question -> answer"` with typed fields) and compose **modules** (`Predict`, `ChainOfThought`, `ReAct`). An **optimiser** (e.g. MIPROv2, GEPA) then tunes the prompts and few-shot examples automatically against an eval set — you optimise a program the way you train a model, instead of hand-editing prompt strings. DSPy drives the model through its own structured text **adapters**, not native OpenAI tool calls.

**Minimal example** (run against a DSPy-format stub; `dspy` 3.4.1). The task is the same two-tool question via `dspy.ReAct`:

```python
import dspy

BASE = "http://127.0.0.1:8080/v1"
MODEL = "mlx-community/Qwen3-14B-4bit"


def get_order(order_id: str) -> dict:
    """Look up an order by id."""
    return {"order_id": order_id, "status": "shipped", "amount": 129.99,
            "tracking_id": "TRK42", "_stub": "order_lookup_ok"}


def get_shipping_status(tracking_id: str) -> dict:
    """Look up shipping status for a tracking id."""
    return {"tracking_id": tracking_id, "carrier": "UPS",
            "status": "in_transit", "eta": "2026-11-02",
            "_stub": "shipping_lookup_ok"}


dspy.configure(lm=dspy.LM(
    "openai/" + MODEL, api_base=BASE, api_key="local",
    model_type="chat", temperature=0, cache=False,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}))
react = dspy.ReAct("question -> answer",
                   tools=[get_order, get_shipping_status], max_iters=4)
print(react(question="Where is my order A1001?").answer)
```

```text
Order A1001 shipped via UPS (tracking TRK42) and is in transit, ETA 2026-11-02. Amount charged: $129.99.
```

!!! note

    DSPy does not send native tool calls; its `ChatAdapter` asks the model to emit `[[ ## field ## ]]` blocks (`next_thought`, `next_tool_name`, `next_tool_args`, then a separate `reasoning`/`answer` extract step). The run above therefore used a **DSPy-shaped stub** that replies in that format; DSPy itself parsed the fields, called both tools, and fed their results back in the next prompt's trajectory (four requests: three ReAct steps and the extract). A real Qwen3-14B is expected to produce those field blocks when prompted by DSPy, though a model can break the format, in which case `ChatAdapter` falls back to its `JSONAdapter`; the value of DSPy is not the loop but the **optimiser** that would tune this program against a labelled set ([54 — Evaluation](54-ai-evaluation.md)).

**Strengths.** Turns prompt engineering into measurable optimisation; portable across models (re-compile for a new model); excellent when you have an eval set and want to squeeze quality; MLflow-native observability.
**Minuses.** A genuinely different mental model with a learning curve; the structured-adapter format is particular; overkill if you do not have a metric to optimise against; not primarily a tool/agent-orchestration framework.
**Use it when.** You have an eval set and want to optimise a pipeline's quality systematically, or move prompts between models without rewriting them.
**Avoid it when.** You just need a tool loop, or you have no metric to compile against.
**Local models.** Yes — `dspy.LM("openai/…", api_base=…, extra_body=…)` (via LiteLLM).
**MCP.** Yes — `dspy.Tool.from_mcp_tool` wraps MCP tools.
**Licence.** MIT.

---

## 11. Stateful and durable-execution platforms

Two related needs: **persistent agent state** (the agent remembers across sessions) and **durable execution** (a long run survives crashes and restarts).

### 11.1 Letta (formerly MemGPT)

**What and who.** **Letta** (the company and framework that grew out of UC Berkeley's **MemGPT** research, renamed September 2024) is built around **stateful agents with self-editing memory**. Its current open-source code is **Letta Code** (`letta-ai/letta-code`, npm `@letta-ai/letta-code`, Apache-2.0): an agent harness with a terminal UI, an App Server you can self-host, channels and SDKs. The original Python API server ("Letta V1") is retired to an `archive` branch of `letta-ai/letta`. Letta Cloud, which stores agent state for you, is the default backend; a local backend is an option.

**Mental model.** The agent owns a structured, persistent memory (memory blocks plus searchable history), stored by the Letta backend and edited by the agent itself; you talk to that backend (Letta Cloud or your own App Server), which keeps the state, so agents live beyond a single process.

**Minimal example.** Not run here: Letta is server-centric (you run an App Server or use Letta Cloud and talk to it over its API/SDK), which is outside this chapter's single-process stub test. See <https://docs.letta.com/> for the current client snippet.

**Strengths.** Persistent, self-editing memory is the core of the design, not an add-on; agents retain state across sessions and restarts; self-hostable and open.
**Minuses.** The server/stateful model is heavier to operate than a library; the project's focus and APIs have moved quickly (MemGPT → Letta server → Letta Code); you adopt its memory abstraction.
**Use it when.** Long-lived assistants that must remember users and facts across sessions.
**Avoid it when.** Stateless request/response, or you do not want to run a stateful server.
**Local models.** Yes, with limits — a local backend can use the built-in Ollama or LM Studio providers, and an OpenAI-compatible custom provider can be added with `/connect` (for a Cloud-hosted agent the endpoint must be reachable from the internet, not `localhost`).
**MCP.** Yes on the Letta platform (register MCP servers; Streamable HTTP recommended), though for Letta Code its docs recommend skills instead of MCP.
**Licence.** Apache-2.0 (Letta Code; Letta Cloud is a hosted service).

### 11.2 Durable-execution platforms (Temporal, DBOS, Inngest, Restate)

These are not agent frameworks; they are **workflow engines** you wrap an agent in so a long, multi-step run is **crash-safe and resumable**. The pattern: each model call and tool call is a durably-recorded step, so if the process dies mid-run it replays deterministically from the last completed step — the durable-execution version of [52](52-ai-agents.md)'s pause-and-resume.

- **Temporal** — the mature, self-hostable/managed workflow engine; its SDKs now have agent-oriented helpers. Heavy but battle-tested.
- **DBOS** — durable execution as a library backed by Postgres; lighter to adopt.
- **Inngest** — event-driven durable functions, popular in the TypeScript world.
- **Restate** — a newer durable-execution runtime.

**Use them when** a run lasts minutes to hours, must survive restarts, or needs exactly-once side effects. LangGraph's checkpointer covers much of this inside one process; a durable-execution platform covers it across process and machine failures. Verify each project's current licence before adopting (several mix open-source cores with commercial cloud tiers).

---

## 12. TypeScript frameworks

If the product is a Next.js/Node app, do not shell out to Python — use a TS-native framework.

### 12.1 Mastra

**What and who.** **Mastra** is a TypeScript agent framework (agents, workflows, tools, memory, RAG, evals) built on top of the Vercel AI SDK's model layer. Core is Apache-2.0; enterprise features in the `ee/` directory are under a separate source-available licence.

**Mental model.** A batteries-included TS equivalent of the big Python frameworks: typed agents with tools and memory, graph-like workflows, RAG and evals, plus a local dev playground. Model access goes through the Vercel AI SDK, so it reaches any provider the AI SDK supports — including local OpenAI-compatible servers.

**Minimal example.** Not run here (this chapter's harness is Python). The documented TS shape is `new Agent({ model, tools })` with an AI SDK OpenAI-compatible provider pointed at a local `baseURL`; see <https://mastra.ai/docs> for a current snippet.

**Strengths.** Full-featured and TypeScript-native; good DX (playground, typed tools, workflows); rides the AI SDK's broad provider support; MCP support.
**Minuses.** Younger than the Python incumbents; enterprise `ee/` code is not Apache-2.0; smaller ecosystem.
**Use it when.** A TS/Next.js product that needs real agent structure (workflows, memory, RAG) rather than just model calls.
**Avoid it when.** You are in Python, or you only need a single model call in a web route (use the AI SDK directly).
**Local models.** Yes — via the Vercel AI SDK's OpenAI-compatible provider with a local `baseURL`; pass provider options for `enable_thinking`.
**MCP.** Yes.
**Licence.** Apache-2.0 (core); `ee/` under the Mastra Enterprise Licence.

### 12.2 Vercel AI SDK

**What and who.** The **Vercel AI SDK** (`ai` on npm, Apache-2.0; 7.x in October 2026) is a widely used TypeScript toolkit for building LLM UIs and tool-calling apps — the `useChat`/streaming layer for React and a provider-agnostic `generateText`/`streamText`/`tool` core.

**Mental model.** Not an agent-orchestration framework so much as the **model + tool-calling + streaming** layer for JS: `generateText({ model, tools, stopWhen: isStepCount(5) })` runs a tool loop (the older `maxSteps` option was replaced by `stopWhen` in AI SDK 5; `stepCountIs` is the earlier name of `isStepCount`), and a `ToolLoopAgent` class packages the same loop; the UI hooks stream it to the browser. For heavier orchestration you layer Mastra (or your own loop) on top.

**Minimal example.** Not run here (Python harness). The documented shape is `generateText({ model: local('mlx-community/Qwen3-14B-4bit'), tools: { getOrder, getShippingStatus }, stopWhen: isStepCount(5) })`, where `local = createOpenAICompatible({ name: 'local', baseURL: 'http://127.0.0.1:8080/v1' })` from `@ai-sdk/openai-compatible`; see <https://ai-sdk.dev/docs> for a current snippet.

**Strengths.** Widely used for streaming LLM UIs in React/Next.js; provider-agnostic; small and focused; tool-execution approval (`needsApproval`) for human-in-the-loop; MCP support (`createMCPClient` from `@ai-sdk/mcp`).
**Minuses.** Deliberately not a full agent framework (no durable state, limited multi-agent) — you add structure yourself; its agent loop is simpler than LangGraph's.
**Use it when.** Any TS/React app that calls models, streams responses, and does light tool calling.
**Avoid it when.** You need heavy multi-agent orchestration or durable workflows (add Mastra or a durable-execution engine).
**Local models.** Yes — `@ai-sdk/openai-compatible` with a local `baseURL`; keys you put under `providerOptions.<provider name>` (here `providerOptions.local`) are merged into the request body, which is how `chat_template_kwargs` gets through.
**MCP.** Yes (`createMCPClient` from `@ai-sdk/mcp`; `experimental_createMCPClient` remains as an alias).
**Licence.** Apache-2.0.

---

## 13. Low-code and visual builders

For non-engineers, or for fast internal automations, a drag-and-drop canvas beats code. The trade-off is less control, a runtime to operate, and licences that often restrict resale/multi-tenancy — read them before you build a product on one.

| Tool | What it is | Status & licence (Oct 2026) |
|---|---|---|
| **n8n** | Workflow automation with AI nodes; huge integration catalogue | Active. **Source-available** under the Sustainable Use License (fair-code) + a separate Enterprise License; internal use allowed, reselling/embedding restricted. |
| **Dify** | LLM-app platform (RAG, agents, prompt IDE) with a visual builder | Active. Dify's "Open Source License": Apache-2.0 **plus extra conditions** (no multi-tenant service without written permission; keep the console logo and copyright), so not an OSI-approved licence. |
| **Langflow** | Visual builder for LLM/agent flows (LangChain-lineage) | Active. **MIT**; owned by IBM (DataStax acquired Langflow in 2024, IBM acquired DataStax in May 2025). |
| **Flowise** | Visual LLM/agent builder | **Archived: the repository was archived (read-only) on 13 August 2026.** Apache-2.0, but an archived project gets no fixes, so do not start new projects on it. |

**Mental model.** You compose nodes on a canvas; each node is a model call, a tool, a condition or an integration; the platform runs and (usually) hosts it. Many can call local OpenAI-compatible endpoints by setting a custom base URL on the model node.

**Use them when.** Non-engineers own the automation, or you need a fast internal workflow with many SaaS integrations.
**Avoid them when.** You need version control, testing, and code review on the logic (export is often lossy), or the licence conflicts with shipping it in your product. For a permissive licence, **Langflow (MIT)** is the clearest; **n8n** and **Dify** carry commercial restrictions; **Flowise** is archived — avoid for new work.

---

## 14. Local models: the thinking trap

The handbook serves Qwen3-14B through `mlx_lm.server` ([10 — Export and serving](10-export-serving.md)), an OpenAI-compatible endpoint. Two things about a **local, thinking-capable** model cut across every framework, so "local-model friendliness" is a real selection criterion — not an afterthought.

### 14.1 Qwen3 thinks by default

Qwen3 emits a reasoning block before its answer. Through an OpenAI-compatible server, a request **without thinking control** can return `message.content = null`, or `content` that *starts with* `<think>…`, and that can confuse naive tool-call parsing and answer extraction. The handbook convention is to pass `chat_template_kwargs={"enable_thinking": False}` in the request body (with the `openai` client: `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`) when you want direct answers and tool calls — and to **guard `content or ""`** everywhere, because `content` can be `None`.

This demo (run against the stub, which imitates a server that leaves Qwen3's think block in `content`) shows the difference on the final answer:

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
MODEL = "mlx-community/Qwen3-14B-4bit"


def final_answer(extra):
    msgs = [
        {"role": "user", "content": "Where is order A1001?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "t1", "type": "function", "function": {
                "name": "get_shipping_status",
                "arguments": '{"tracking_id": "TRK42"}'}}]},
        {"role": "tool", "tool_call_id": "t1",
         "content": json.dumps({"carrier": "UPS", "status": "in_transit",
                                "_stub": "shipping_lookup_ok"})},
    ]
    r = client.chat.completions.create(model=MODEL, messages=msgs,
                                       temperature=0, **extra)
    return (r.choices[0].message.content or "")


on = final_answer({})
off = final_answer(
    {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})
print("default  :", repr(on[:22]))
print("no-think :", repr(off[:22]))
```

```text
default  : '<think>\nThe task needs'
no-think : 'Order A1001 shipped vi'
```

With thinking on, this answer begins with a `<think>` block you must strip; with `enable_thinking=False`, it is the clean answer (the sentence itself is the stub's canned text). Servers that leave the think block in `content` behave like this stub. `mlx_lm.server` 0.32.2 instead moves the reasoning into a separate `reasoning` field ([50 — Prompting and APIs](50-ai-prompting-apis.md)), so there `content` stays clean, or is `None` on a turn that only called a tool or ran out of `max_tokens` while thinking. The thinking tokens are still generated either way, so you wait for them and they can use up a small `max_tokens`. Turning thinking off for tool loops and guarding `None` is the fix that works on every server (see [10 §4](10-export-serving.md) for `--chat-template-args`, which sets it server-wide).

### 14.2 Can the framework pass extra body fields?

The practical question for each framework: **does it let you pass `chat_template_kwargs` (or arbitrary request-body fields) through to a local OpenAI-compatible endpoint?** Every Python framework run in this chapter can, and the exact call differs — this is the real local-model-friendliness criterion:

| Framework | How to pass `enable_thinking=False` |
|---|---|
| Baseline (`openai`) | `create(..., extra_body={"chat_template_kwargs": {"enable_thinking": False}})` |
| LangGraph / LangChain | `ChatOpenAI(..., extra_body={...})` |
| LlamaIndex | `OpenAILike(..., additional_kwargs={"extra_body": {...}})` |
| OpenAI Agents SDK | `ModelSettings(extra_body={...})` |
| Pydantic AI | `OpenAIChatModelSettings(extra_body={...})` |
| smolagents | `OpenAIServerModel(..., extra_body={...})` |
| CrewAI | `LLM(..., extra_body={...})` |
| Haystack | `OpenAIChatGenerator(generation_kwargs={"extra_body": {...}})` |
| Agno | `OpenAILike(..., extra_body={...})` |
| Strands | `OpenAIModel(params={"extra_body": {...}})` |
| DSPy | `dspy.LM(..., extra_body={...})` (LiteLLM) |
| Google ADK | `LiteLlm(..., extra_body={...})` (LiteLLM) |

!!! tip

    A framework that makes `extra_body` hard to reach, or that assumes non-null `content`, will fight a local thinking model. All twelve above (the baseline and eleven frameworks) sent `enable_thinking: false` on every request — the stub logged it — and returned the same answer; a framework where you cannot thread that field is a red flag for local-model work.

---

## 15. The comparison matrix

Facts only, verified in October 2026 from official repos/PyPI/docs. Versions are the ones actually installed and run for this chapter (the Microsoft Agent Framework, Claude Agent SDK, Letta and the TypeScript frameworks were not run in the chapter; their rows are from docs and PyPI/npm). "Durable state" means built-in checkpoint-and-resume; "partial" means session/message persistence without step-level resume.

**Table A — shape and control**

| Framework (version) | Category | Abstraction | Multi-agent | Durable state | Human-in-the-loop | Licence |
|---|---|---|---|---|---|---|
| Baseline (`openai` 3.40) | — | none (you write it) | no | no | manual | — |
| LangGraph 1.2.18 | graph | low (graph) | yes | **yes** (checkpointer) | **strong** (`interrupt`) | MIT |
| LangChain 1.4.7 | graph (on LangGraph) | medium | yes | yes (via LangGraph) | yes | MIT |
| CrewAI 1.15.32 | role multi-agent | medium | **yes** (crews) | partial (memory) | via Flows | MIT |
| LlamaIndex 0.14.25 | data/RAG | medium | yes (Workflows) | partial (Context) | yes | MIT |
| Haystack 3.4.1 | data/RAG | medium (pipeline) | yes | partial | yes | Apache-2.0 |
| Pydantic AI 2.63 | typed/light | low | yes | yes (via Temporal / DBOS / Prefect) | yes | MIT |
| Agno 3.1.11 | typed/light | low | yes (teams) | partial (DB sessions) | yes | Apache-2.0 |
| OpenAI Agents 0.24 | vendor SDK | low | yes (handoffs) | partial (sessions) | yes (approvals) | MIT |
| MS Agent Framework 1.23 | vendor SDK | medium | yes | yes (workflow checkpoints) | yes | MIT |
| Google ADK 2.13 | vendor SDK | medium | yes | partial (sessions) | yes (callbacks) | Apache-2.0 |
| AWS Strands 1.62 | vendor SDK | low | yes (graph, swarm, A2A) | partial (sessions) | yes | Apache-2.0 |
| Claude Agent SDK | vendor SDK | medium-high | yes (sub-agents) | partial (resumable sessions) | yes (permissions) | Anthropic commercial terms (Python wrapper MIT) |
| smolagents 1.26 | code-agent | low | yes (managed) | no | basic (user-input tool, step callbacks) | Apache-2.0 |
| DSPy 3.4.1 | prompt-opt | medium (different) | yes (modules) | no | no (not its job) | MIT |
| Letta | stateful | medium (server) | yes | **yes** (memory) | yes | Apache-2.0 |
| Mastra (TS) | TS | medium | yes | yes | yes | Apache-2.0 (+ee) |
| Vercel AI SDK 7 (TS) | TS | low | limited | no | yes (tool approval) | Apache-2.0 |

**Table B — local models, MCP, observability, maturity**

| Framework | Local OpenAI-compatible | MCP ([58](58-ai-mcp.md)) | Observability | Learning curve | Maturity / churn risk |
|---|---|---|---|---|---|
| Baseline | native (`extra_body`) | you wire it ([52](52-ai-agents.md)) | you add it | lowest | none (it's your code) |
| LangGraph | yes | `langchain-mcp-adapters` | LangSmith / OTel | medium-high | mature; ecosystem churn |
| LangChain | yes | `langchain-mcp-adapters` | LangSmith / OTel | medium | mature; higher churn |
| CrewAI | yes (native OpenAI provider) | native `mcps` / `crewai-tools` | AgentOps / Langfuse | low-medium | active; moving fast |
| LlamaIndex | yes | `llama-index-tools-mcp` | LlamaTrace / OTel | medium | mature; 0.x churn |
| Haystack | yes | `mcp-haystack` | OTel / Langfuse | medium | mature; stable |
| Pydantic AI | yes | native | **Logfire** (OTel) | low | young but stable (1.0+) |
| Agno | yes | `agno.tools.mcp` | AgentOS / native | low | active; renamed from Phidata |
| OpenAI Agents | yes (chat model) | native | built-in traces / OTel | low | young; vendor-led |
| MS Agent Framework | yes | native | OTel built-in | medium | GA 2026; new |
| Google ADK | yes (LiteLLM) | native (`McpToolset`) | Cloud Trace / OTel | medium | active; 2.0 in May 2026 |
| AWS Strands | yes | native (`MCPClient`) | OTel native | low-medium | active; 1.0+ |
| Claude Agent SDK | no (Claude-only) | first-class (host) | own tooling | medium | active; recently renamed |
| smolagents | yes | `ToolCollection.from_mcp` | OTel (OpenInference) | low | small; last release May 2026 |
| DSPy | yes (LiteLLM) | `dspy.Tool.from_mcp_tool` | **MLflow** native | medium (different) | active; 3.x |
| Letta | yes (Ollama, LM Studio, custom) | yes (platform) | own / OTel | medium | active; server replaced by Letta Code |
| Mastra | yes (AI SDK) | yes | own / OTel | medium | young |
| Vercel AI SDK | yes | `@ai-sdk/mcp` (`createMCPClient`) | OTel | low | mature (for its scope); 7.x |

---

## 16. Lines of code for the same task

Counted from the code blocks above, which are the real `run_*.py` files minus their module docstring (non-blank, non-comment lines; function docstrings count). Every row does the **identical** two-tool order task and returns the identical answer. The tool bodies are the same everywhere; the difference is the wiring.

| Implementation | SLOC | Notes |
|---|---:|---|
| Agno | 19 | leanest: plain functions, one `Agent` |
| DSPy | 19 | signature + `ReAct` (different protocol) |
| LangGraph | 22 | one `create_agent` call |
| Strands | 23 | `@tool` + `OpenAIModel` |
| LlamaIndex | 24 | async `FunctionAgent` |
| OpenAI Agents | 25 | `@function_tool` + `ModelSettings` |
| Pydantic AI | 25 | `@agent.tool_plain` |
| smolagents | 26 | `@tool` with `Args:` docstrings |
| CrewAI | 27 | role + goal + task + crew |
| Haystack | 29 | pipeline components + `Secret` |
| Google ADK | 34 | runner + session ceremony |
| **Baseline (no framework)** | **43** | you hand-write the tool schemas and the loop |

The frameworks need between 44% (Agno, DSPy: 19 lines) and 79% (ADK: 34 lines) of the baseline's 43 lines, about half to two-thirds for most of them, mostly because they generate tool schemas from function signatures and own the loop.

!!! example "On paper"

    The hand-rolled baseline is 43 lines of code; Agno does the same task in 19:

    \[
    43 / 19 \approx 2.3
    \]

    So the baseline is about 2.3 times as long as the leanest framework; against LangGraph (22 lines) the ratio is \(43/22 \approx 1.95\), so LangGraph needs roughly half the lines. That saved code is the benefit; the costs are the abstraction you debug through and the upgrade treadmill of [§1.2](#12-what-a-framework-costs). ADK at 34 lines shows a framework can also *add* ceremony (runners, sessions) for a simple task.

---

## 17. How to choose: decision tree and scenarios

### 17.1 Decision tree

```text
START: you have a model endpoint and a task.
│
├─ Is it a single call or a fixed, writable-in-advance workflow?
│     └─ YES → no framework (openai client + a function). [§2]
│
├─ One tool-calling feature inside an existing service?
│     └─ YES → no framework, or a typed/light library (Pydantic AI). [§7.1]
│
├─ TypeScript / Next.js product?
│     ├─ just model calls + streaming + light tools → Vercel AI SDK. [§12.2]
│     └─ real agent structure (workflows, memory, RAG) → Mastra. [§12.1]
│
├─ Retrieval over documents is the core?
│     └─ YES → LlamaIndex (RAG-first) or Haystack (prod pipelines). [§6]
│
├─ Long-running, resumable, human-approved workflow?
│     ├─ within one process → LangGraph (checkpointer + interrupt). [§4.1]
│     └─ across crashes/machines → + a durable-execution engine. [§11.2]
│
├─ Persistent memory across sessions is the point?
│     └─ YES → Letta. [§11.1]
│
├─ Genuinely separate roles with different tools (proven by an eval)?
│     └─ YES → CrewAI, or LangGraph for tighter control. [§5]
│
├─ You have an eval set and want to optimise prompts systematically?
│     └─ YES → DSPy. [§10]
│
├─ On a specific cloud/model vendor?
│     └─ OpenAI → OpenAI Agents; Google → ADK; AWS → Strands;
│        Azure/.NET → MS Agent Framework; Claude harness → Claude Agent SDK. [§8]
│
├─ Non-engineers building automations?
│     └─ YES → low-code (Langflow for MIT; n8n/Dify mind the licence). [§13]
│
└─ Else → start from the baseline; add a typed/light library when it hurts.
```

### 17.2 Scenario table

| Scenario | Recommended | Runner-up | Why |
|---|---|---|---|
| Single tool-calling feature in an existing service | **No framework** (openai client) | Pydantic AI | One call + a `for` loop beats learning a framework; Pydantic AI if you want typing. |
| RAG over company docs | **LlamaIndex** | Haystack | Richest retrieval toolkit; Haystack if you want explicit, serializable pipelines. |
| Long-running workflow with approvals | **LangGraph** | + Temporal/DBOS | Checkpointer + `interrupt` built in; a durable engine for cross-crash guarantees. |
| Multi-agent research assistant | **LangGraph** | CrewAI | Explicit orchestration and state; CrewAI if role personas fit and speed matters. |
| TypeScript / Next.js product | **Vercel AI SDK** | Mastra | Widely used for TS model calls + streaming; Mastra when you need agent structure. |
| Local model on a 32 GB Mac | **Pydantic AI** or baseline | Agno | Thin, typed, easy `extra_body` for `enable_thinking`; Agno for batteries + low overhead. |
| Strict typing / validation | **Pydantic AI** | baseline + pydantic | Native typed tools and validated structured output. |
| Prompt optimisation against an eval set | **DSPy** | — | The only framework here built for compiling/optimising prompts. |
| Non-engineers building automations | **Langflow** (MIT) | n8n / Dify | Permissive licence and a visual canvas; n8n/Dify if their integrations win — mind the licence. |
| Enterprise Azure/.NET shop | **MS Agent Framework** | — | .NET parity, Azure integration, OTel; the AutoGen/SK successor. |

---

## 18. Cross-cutting concerns

These apply whatever framework you pick (or none).

### 18.1 Observability

You cannot operate what you cannot see. Three neutral facts: the industry is converging on the **OpenTelemetry GenAI semantic conventions** (standard span/attribute names for model and tool calls), so prefer frameworks that emit them; **Langfuse**, **LangSmith** (LangChain's) and **Arize Phoenix** are widely used tracing/eval backends, and most frameworks have an integration for at least one. Pydantic AI leans on **Logfire**, DSPy on **MLflow**, OpenAI Agents and Strands emit traces natively. Whatever you choose, capture per-step spans with token counts and latency, and wire them into production observability ([55 — Production §observability](55-ai-production.md)).

### 18.2 Evaluation

A framework does not make your agent correct — an eval does. Treat evals as tests whose assertions are thresholds ([54 — Evaluation](54-ai-evaluation.md)): a golden set of tasks, reference or LLM-judge scoring, run in CI. Several frameworks ship eval helpers (ADK, Mastra, DSPy's optimisers, LangSmith datasets), but the discipline matters more than the tool. Evaluate the **whole system** — model + prompt + tools + framework glue — not the model in isolation, exactly as [11 §9](11-kali-agent.md) evaluates the agent loop rather than the endpoint.

### 18.3 Testing agents built on frameworks

The techniques are framework-independent:

- **Mock the model.** Replace the model with a scripted fake that returns fixed tool calls and answers ([52](52-ai-agents.md) uses exactly this), so the loop, dispatch and parsing are tested deterministically with no network.
- **Record/replay.** Capture real model responses once, then replay them in tests — fast, deterministic, and it catches regressions in your glue.
- **A stub endpoint.** The approach this whole chapter used: a tiny OpenAI-compatible server that returns canned tool calls and answers ([50](50-ai-prompting-apis.md) on the request shape). It let eleven frameworks and the baseline run the same task offline (DSPy against a variant that speaks its text format), and it is how you'd test a framework-based agent in CI without a GPU. Make the stub advance only when it sees your tools' real return values, as this one did, so a passing run proves the tools were called.
- **Test tools in isolation.** Your `get_order` is ordinary code — unit-test it without the model.

### 18.4 Managing upgrades

Given the churn ([§1.2](#12-what-a-framework-costs)): **pin exact versions** (you saw `create_react_agent` move to `create_agent` between LangGraph/LangChain versions, Pydantic AI already on 2.x, DSPy on 3.x, ADK and Haystack with new major versions in 2026). Read release notes before bumping. And **wrap the framework behind your own interface** — a thin `Agent` class of yours that the rest of the app calls — so a breaking change or a framework swap touches one module, not the whole codebase ([§19](#19-migration-paths)).

### 18.5 Security

Frameworks concentrate the risks of [56 — Safety and security](56-ai-safety.md), so they deserve extra care:

- **Hidden prompts are attack surface.** The framework's injected system prompt is text you did not audit; indirect prompt injection from tool outputs still applies ([56](56-ai-safety.md)).
- **Tool permissions.** A framework makes it easy to hand the model many tools; the fewer and more typed, the smaller the blast radius ([11 §4](11-kali-agent.md), [52](52-ai-agents.md)). Keep the policy layer in *your* code, not a prompt.
- **Code-execution agents need sandboxes.** A `CodeAgent` ([§9](#9-code-agent-libraries-smolagents)) runs model-written code; never run it unsandboxed. Use the framework's sandbox backends (E2B/Modal/Docker/Pyodide) and treat the output as untrusted.
- **Dependency risk.** Hundreds of transitive packages are a supply-chain surface; review them.

### 18.6 Cost and token overhead

Framework system prompts cost real tokens on every call. Measured here (the stub logged each request): LangGraph, LlamaIndex, Haystack, Pydantic AI, Agno, OpenAI Agents and Strands sent **no** system prompt of their own; CrewAI sent a 124-character persona system message (plus task framing in the user turn) and ADK a 50-character one; DSPy's adapter sent a 2,265-character format prompt on each ReAct step, because the tool descriptions and field format live in its prompt rather than in a `tools` field. **smolagents' `ToolCallingAgent` sent a 3,544-character system prompt**, the largest of the runs, and its first request was **4,872 bytes** versus the baseline's **602**. A `CodeAgent` with the same two tools sends 8,814 characters.


!!! example "On paper"

    smolagents' `ToolCallingAgent` sends a 3544-character system prompt. At about 4 characters per token, over the 3 model calls this task takes:

    \[
    \begin{aligned}
    \text{tokens} &\approx 3544 / 4 = 886\\
    \text{per task} &\approx 3 \times 886 = 2658
    \end{aligned}
    \]

    Each call carries about 886 tokens of scaffolding, about 5% of a 16K window, and the task makes the model read about 2,700 of them in total. On a local model that is prefill time: at roughly 120 prompt tokens/s (Qwen3-14B 4-bit on an M1 Pro, [50](50-ai-prompting-apis.md)), 886 tokens take about 7 s on any call that the server's prompt cache cannot serve.


!!! example "On paper"

    The stub logged smolagents' first request at 4872 bytes and the baseline's at 602:

    \[
    4872 / 602 \approx 8.1
    \]

    Eight times the bytes on the wire. Most of the 4,270 extra bytes are the 3,544-character system prompt; the rest is its `final_answer` tool and longer tool descriptions. Most frameworks add little; smolagents trades tokens for its reasoning scaffold. Measure yours — do not assume it is zero.

If you pay per token, that overhead has a price. Hosted-model prices move, so treat the number as an assumption, not a fact:


!!! example "On paper"

    Assume $0.50 per million input tokens. With 2658 scaffold tokens per task and 100,000 tasks a day:

    \[
    \begin{aligned}
    \text{per task} &\approx 2658/10^6 \times 0.50 \approx 0.0013\\
    \text{per day} &\approx 10^5 \times 0.001329 \approx 133
    \end{aligned}
    \]

    About $133 a day is the framework's system-prompt tax alone (before any prompt-cache discount), before your own prompt and the outputs. For a local model the cost is latency and context instead of dollars, but it is just as real.

---

## 19. Migration paths

### 19.1 From no framework to a framework

You start with the baseline ([§2.1](#21-the-no-framework-baseline)) and outgrow it (you need durable state, or multi-agent, or RAG). The migration is mechanical because the pieces map:

- Your tool functions become the framework's tools (usually just a decorator or passing the function).
- Your hand-written `TOOLS` schema is **deleted** — the framework generates it from signatures and docstrings. (So keep good type hints and docstrings even in the baseline.)
- Your `run()` loop becomes the framework's agent/runner.
- Your `extra_body` for `enable_thinking` moves to the framework's model-settings call ([§14.2](#142-can-the-framework-pass-extra-body-fields)).

Migrate behind your eval ([§18.2](#182-evaluation)): the answers should stay correct (against the stub every framework above printed the identical output; against a real model the wording can change, which is why you check with the eval rather than a string compare), so the eval catches any regression the migration introduces.

### 19.2 From a framework back to no framework (or to another)

This is the one teams underestimate. If the framework's abstractions fight you, or it churns too hard, you may want out. It is painful **only if you coupled to it everywhere**. The defence is an **adapter layer**: define your own small interface, make it the only module that imports the framework, and keep everything else depending on *your* type. Here the adapter wraps Pydantic AI ([§7.1](#71-pydantic-ai)); the shape is the same whatever you wrap:

```python
import os
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"
from typing import Protocol
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

BASE = "http://127.0.0.1:8080/v1"
MODEL = "mlx-community/Qwen3-14B-4bit"


class OrderAgent(Protocol):
    def answer(self, question: str) -> str: ...


def get_order(order_id: str) -> dict:
    """Look up an order by id; returns status, amount and tracking_id."""
    return {"order_id": order_id, "status": "shipped", "amount": 129.99,
            "tracking_id": "TRK42", "_stub": "order_lookup_ok"}


def get_shipping_status(tracking_id: str) -> dict:
    """Look up the shipping status for a tracking id."""
    return {"tracking_id": tracking_id, "carrier": "UPS",
            "status": "in_transit", "eta": "2026-11-02",
            "_stub": "shipping_lookup_ok"}


class PydanticAIOrderAgent:
    """The only class (in the only module) that touches the framework."""
    def __init__(self, tools):
        model = OpenAIChatModel(
            MODEL, provider=OpenAIProvider(base_url=BASE, api_key="local"))
        settings = OpenAIChatModelSettings(
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        self._agent = Agent(model, tools=tools, model_settings=settings)

    def answer(self, question: str) -> str:
        return self._agent.run_sync(question).output


agent: OrderAgent = PydanticAIOrderAgent([get_order, get_shipping_status])
print(agent.answer("Where is my order A1001?"))
```

```text
Order A1001 shipped via UPS (tracking TRK42) and is in transit, ETA 2026-11-02. Amount charged: $129.99.
```

The rest of your app depends on `OrderAgent` and your plain tool functions, not on Pydantic AI, LangGraph or CrewAI. Swapping frameworks — or dropping back to the baseline loop of [§2.1](#21-the-no-framework-baseline) — then means writing one new class with the same `answer` method, with your eval proving the behaviour held. This also neutralises the upgrade churn of [§18.4](#184-managing-upgrades): a breaking change lands in one file.

!!! tip

    Write the adapter on day one, even if you are sure of your framework choice. It costs about a dozen lines — the `Protocol` and the wrapper class above — and buys you an exit. The frameworks took 19 to 34 lines for this task; the wrapper adds only that dozen on top of whichever one you pick.

---

## 20. Key takeaways

- A framework is **leverage you rent**: it removes boilerplate (schema generation, the loop, state, streaming, tracing) and in exchange you debug through its abstractions and ride its upgrade treadmill. Justify every departure from the ~43-line baseline.
- **Most "agent" problems are a single call or a fixed workflow** ([52](52-ai-agents.md)). Use the least autonomy — and the least framework — that solves the task.
- Pick by **category**, then by fit: graph/state-machine (LangGraph), role multi-agent (CrewAI), data/RAG (LlamaIndex, Haystack), typed/light (Pydantic AI, Agno), vendor SDKs (OpenAI Agents, MS Agent Framework, Google ADK, AWS Strands, Claude Agent SDK), code-agent (smolagents), prompt-opt (DSPy), stateful (Letta), TypeScript (Mastra, Vercel AI SDK), low-code (Langflow/n8n/Dify; Flowise is archived).
- Status changed in 2025–2026: **AutoGen and Semantic Kernel → Microsoft Agent Framework** (AutoGen archived, Semantic Kernel on critical and security fixes); **LangChain 1.x runs on LangGraph**; the **Claude Code SDK → Claude Agent SDK**; **Flowise archived**; Letta's server gave way to **Letta Code**.
- **Local, thinking models are a real criterion.** Qwen3 thinks by default; pass `chat_template_kwargs={"enable_thinking": False}` and guard `content or ""`. Every Python framework here can thread that through `extra_body`; one that can't is a red flag ([§14](#14-local-models-the-thinking-trap)).
- **Measure the overhead.** System-prompt tokens are real (3,544 chars / ~886 tokens per call for smolagents here); so is lines-of-code; so is churn. Numbers beat vibes.
- **Wrap the framework behind your own interface** and pin versions, so a breaking change or a framework swap touches one module ([§19](#19-migration-paths)).
- Licences matter for shipping: most libraries are **MIT or Apache-2.0**; **n8n** and **Dify** add commercial restrictions (source-available, and modified Apache-2.0); the **Claude Agent SDK** is under Anthropic's commercial terms; **Flowise** is archived.

---

## 21. Exercises

!!! question "Exercise 21.1"

    You are adding "summarise this ticket and tag its priority" to an existing support service. Which framework do you reach for, and why?

??? success "Answer"

    None. That is a single model call (or a two-step extract-then-classify workflow), not an agent ([52](52-ai-agents.md)). The `openai` client plus the structured-output pattern ([50](50-ai-prompting-apis.md)) is fewer lines than learning a framework, has no hidden prompt, and you already own the service's deployment and state. Reach for a framework only when the decision procedure of [§2.2](#22-the-decision-procedure) pulls you past step 2.

!!! question "Exercise 21.2"

    The same two-tool order task is 43 lines in the no-framework baseline and 19 in Agno. Compute the ratio and say what the saved lines cost you.

??? success "Answer"

    \(43 / 19 \approx 2.3\). The baseline is about 2.3x larger, mostly because it hand-writes the `TOOLS` JSON schema and the loop, which Agno generates from signatures and owns internally. The saved lines cost you an abstraction layer to debug through, a hidden system prompt, a dependency tree, and an upgrade treadmill ([§1.2](#12-what-a-framework-costs)). Worth it when you need the batteries; not worth it for a single call.

!!! question "Exercise 21.3"

    Your agent, on a local Qwen3-14B, sometimes returns an empty or `<think>…`-prefixed answer through the framework. What are the two fixes, and how do you apply them in (a) the raw `openai` client and (b) LangGraph?

??? success "Answer"

    (1) Disable thinking by passing `chat_template_kwargs={"enable_thinking": False}`; (2) guard `content or ""` because `content` can be `None`. In the raw client: `create(..., extra_body={"chat_template_kwargs": {"enable_thinking": False}})`. In LangGraph: `ChatOpenAI(..., extra_body={"chat_template_kwargs": {"enable_thinking": False}})`. See [§14](#14-local-models-the-thinking-trap); every Python framework here exposes `extra_body` by some name.

!!! question "Exercise 21.4"

    smolagents' `ToolCallingAgent` system prompt measured 3,544 characters. Estimate the token overhead per task if the task takes 3 model calls, at ~4 characters per token.

??? success "Answer"

    \(3544 / 4 = 886\) tokens per call, and \(3 \times 886 = 2658\) tokens per task — about 2,700 tokens of scaffolding the model must read before it gets to the order data. Each call's 886 tokens are about 5% of a 16K window, and at roughly 120 prompt tokens/s of local prefill they add about 7 s to every call the prompt cache cannot serve ([§18.6](#186-cost-and-token-overhead)). A `CodeAgent` sends 8,814 characters, about 2,200 tokens per call; it can still be worth it when composing several tool calls in one code step saves round trips — measure both.

!!! question "Exercise 21.5"

    A teammate proposes a three-agent CrewAI "crew" (researcher, writer, reviewer) for a task a single agent handles today. Each agent step succeeds with probability 0.98. What happens to reliability and cost, and what do you ask for before agreeing?

??? success "Answer"

    Reliability compounds *down* with more steps ([52](52-ai-agents.md)): a single call is \(0.98\), while a naive 5-step three-agent flow is about \(0.98^5 \approx 0.904\) — and each agent re-sends its own context, so cost and latency multiply. Ask for an **eval** ([§18.2](#182-evaluation)) showing the crew beats the single agent on the real task. If it does not, the extra agents are cost and coordination risk for nothing ([52](52-ai-agents.md)).

!!! question "Exercise 21.6"

    You built on AutoGen a year ago. What is the current status, and what should you do?

??? success "Answer"

    AutoGen is **archived**: its README says Microsoft no longer provides bug fixes, security fixes, new features or support, and names the **Microsoft Agent Framework** (GA April 2026; `agent-framework` on PyPI) as its successor, with an AutoGen migration guide ([§8.2](#82-microsoft-agent-framework-and-the-autogen-semantic-kernel-story)). Without security fixes, staying put is a growing risk, so plan the migration as scheduled work: put an adapter around the AutoGen code first ([§19.2](#192-from-a-framework-back-to-no-framework-or-to-another)), port behind your eval, then switch. (Had you built on Semantic Kernel instead, you would have more time: it still gets critical and security fixes for at least a year after the Agent Framework's GA.) This is the churn risk of [§1.2](#12-what-a-framework-costs) made concrete.

!!! question "Exercise 21.7"

    Why is "wrap the framework behind your own interface" worth a dozen lines on day one?

??? success "Answer"

    Because it localises both risks of [§1.2](#12-what-a-framework-costs): a breaking upgrade or a decision to swap frameworks (or drop back to the baseline) touches one adapter module instead of the whole codebase ([§19.2](#192-from-a-framework-back-to-no-framework-or-to-another)). The frameworks here took 19 to 34 lines for the task (the baseline 43); the wrapper adds about a dozen lines on top of any of them, and your eval proves behaviour held across the swap.

!!! question "Exercise 21.8"

    You need RAG over company docs *and* a long-running, human-approved workflow around it. One framework, or two layers?

??? success "Answer"

    Two concerns, so consider two layers: a data/RAG framework for retrieval ([§6](#6-data-rag-centric-frameworks-llamaindex-haystack) — LlamaIndex or Haystack) and a graph/state-machine for the durable, human-gated workflow ([§4.1](#41-langgraph) — LangGraph's checkpointer + `interrupt`). LangGraph can call a LlamaIndex/Haystack retriever as a tool. If retrieval is simple, one framework with a retrieval integration may suffice; if the workflow is simple, add retrieval to a lighter agent. Let the dominant concern pick the primary framework, and keep both behind your adapter ([§19.2](#192-from-a-framework-back-to-no-framework-or-to-another)).

!!! question "Exercise 21.9"

    You want to ship a visual agent builder inside your commercial SaaS so customers can build flows. Which low-code tool's licence is safest, and which should you avoid?

??? success "Answer"

    **Langflow** is MIT, a permissive licence that allows embedding in a product. **n8n** (Sustainable Use License, source-available) and **Dify** (its "Open Source License": Apache-2.0 plus extra conditions) restrict reselling, multi-tenant use or embedding — read them before shipping, and get written permission or a commercial licence if needed. **Flowise** was archived on 13 August 2026 and gets no fixes — do not build new on it ([§13](#13-low-code-and-visual-builders)).

!!! question "Exercise 21.10"

    How did this chapter test eleven frameworks' agents (plus the baseline) without a GPU or a hosted model, and why does that generalise to testing your own framework-based agent?

??? success "Answer"

    With a **stub OpenAI-compatible server** that returns canned tool calls and a final answer, driven by sentinels — marker strings that only the tools' return values contain — so it advances only after each tool has really run ([§2.1 note](#21-the-no-framework-baseline), [§18.3](#183-testing-agents-built-on-frameworks)); DSPy used a variant that speaks its text format. Each framework was pointed at it and printed the identical answer (the stub's canned sentence), offline and deterministically. The same technique tests your production agent: mock the model (or stub the endpoint), record/replay real responses, and unit-test the tools as ordinary code — so your CI exercises the loop and the glue without a model at all.

Next: [62 — Integrating agents](62-ai-agent-integration.md) for how any of these is wired into a product and [63 — Agents in the real world](63-ai-agent-applications.md) for twelve applications; [52 — Agents and tool use](52-ai-agents.md) for the loop these frameworks wrap, [58 — MCP](58-ai-mcp.md) for the tool protocol they share, and [54 — Evaluation](54-ai-evaluation.md) to prove any choice here is the right one.
