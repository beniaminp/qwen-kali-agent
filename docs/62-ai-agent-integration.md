# 62 — Integrating agents into applications

*Last verified: 29 October 2026.*

You have an agent loop that works in a script: a model, a handful of tools, a `for` loop with a step budget ([52 §6](52-ai-agents.md#6-a-complete-agent-loop)). This chapter is about the distance between that script and a feature that many people use inside a real product. It covers the **shapes** an agent can take inside an application (an in-process call, a run service, a streaming chat UI, a queue worker, a batch job, a durable workflow, an agent used as a tool, a local harness), the **run contract** (request, record, events, states), **state** and sessions, **identity** and per-user tool scopes, how to **connect tools**, **approvals** inside a product UI, **streaming**, **reliability**, **observability**, **cost control**, **testing**, **rollout**, and a runnable minimal agent service built from the standard library. It does not re-teach the loop, the workflow patterns, autonomy levels, the reliability and cost math, or the idea of a budget and a human gate — those are [52 — Agents](52-ai-agents.md); the frameworks that package some of this are [61 — Agent frameworks](61-ai-agent-frameworks.md); the serving stack, gateway and observability basics are [55 — Production](55-ai-production.md). Every Python block here runs; where a model is needed, the code calls the handbook's local `mlx_lm.server` and was tested against a stub that imitates it, Qwen3 thinking mode included.

The one idea to carry through: **an agent in a product is a long-lived, resumable, auditable job with a budget — not a function call.** Design the job first; the model is one step inside it.

---

## 1. From a script to a feature

### 1.1 What changes

In the script, one person runs one task on their own machine, reads the output, and stops the process when it misbehaves. Inside a product every one of those assumptions breaks:

- **Many users at once.** Runs overlap. Two users' transcripts must never mix; one user's runaway run must not starve the others. The loop becomes a worker pool with a queue.
- **Identity.** The script used your credentials. In the product the agent acts *on behalf of* a user, with that user's permissions and nothing more (§5).
- **Latency budgets.** A request handler has a timeout. An agent that takes 40 seconds cannot live inside a 10-second request; it becomes a run you create, poll or stream (§2, §8).
- **Failures are routine.** The model times out, a tool returns 503, a worker restarts mid-run. A run must be resumable from its last step, and a retry must not refund twice (§4, §9).
- **Cost ceilings.** The script's bill was yours. The product's bill is the sum over every user and every step, and a prompt-injected or looping agent is an unbounded spend without a hard budget enforced by the runtime, not by the prompt (§9.5, §11).
- **Observability.** You can no longer read the terminal. Every run needs a trace, structured logs, and metrics that say whether the feature works (§10).
- **Change management.** A prompt edit is a deploy. You need versions, canaries, a kill switch, and a plan for runs that are in flight when you ship (§13).

A software-engineering analogy: the script is a shell one-liner; the feature is a job system with a scheduler, a job table, retries, idempotency, auth, metrics and a dashboard. The agent loop is the job's body. Most of this chapter is job-system engineering you already know, applied to a body that is non-deterministic and that can act.

### 1.2 Integration readiness

Before shipping, check each row. "Script" is where most agents start; "feature" is where they must be.

| Concern | Script | Feature |
|---|---|---|
| Who runs it | you, by hand | any user, concurrently, via a product surface |
| Credentials | your own | the user's, scoped per tool and per tenant (§5) |
| Lifetime | one process, one task | a run record with a state machine, resumable (§3, §4) |
| Failure | stack trace in the terminal | retries with idempotency; a `failed` state with a reason (§9) |
| Budget | `max_steps` | steps, tokens, wall time and money, enforced by the runtime (§9.5) |
| Side effects | happen | gated by approvals and an audit log (§5.4, §7) |
| Output | printed | streamed as events; stored as a record (§3, §8) |
| Cost | your API key | per-run and per-tenant ceilings, a monthly model (§11) |
| Visibility | you watch it | one trace per run; dashboards; alerts (§10) |
| Change | edit and rerun | versioned prompts and tools; shadow → canary → GA; kill switch (§13) |
| Tests | "it worked on my ticket" | record/replay, tool contracts, golden trajectories, eval gates in CI (§12) |

---

## 2. Integration shapes

An agent can be wired into a product in eight recognizable shapes. Each has a diagram, a case where it fits, an example product, and trade-offs. Pick by the requirements in §2.9, not by habit.

### 2.1 In-process call inside a request handler

The agent runs synchronously inside an existing HTTP handler, with a small step budget, and returns with the response.

```text
client ──POST /tickets/{id}/suggest──▶ handler
                                         │ loop ≤ 3 steps, ≤ 8 s
                                         ├─▶ model ─▶ tool ─▶ model
                                         ▼
client ◀────── 200 {suggestion} ───────┘
```

**When:** the task is short (1–3 steps), read-only or low-risk, and must fit the handler's timeout. **Example:** a "suggest a reply" button in a support console that looks up the order and drafts two sentences. **Trade-offs:** the simplest shape — no new infrastructure, no run table — but the handler's timeout is your hard budget, there is nothing to resume if the process dies, the user sees nothing until the end, and a slow model call ties up a request worker. Treat it as the single-call pattern of [52 §1](52-ai-agents.md#1-agent-vs-workflow-vs-single-call) with one or two tool round trips, and nothing more.

### 2.2 An agent service with run ids

The product creates a **run** and gets an id back immediately; a worker executes it; the client polls the record or streams its events.

```text
client ──POST /runs──▶ API ──▶ queue ──▶ worker pool ──▶ model / tools
   ▲        │ 202 {run_id}      │              │
   │        ▼                   ▼              ▼
   └──GET /runs/{id} ◀───── run store ◀── state, steps, events
   └──GET /runs/{id}/events (SSE)
```

**When:** runs take seconds to minutes, may need approvals, and must survive a client that disconnects. **Example:** "investigate this incident" in an on-call tool — the run takes two minutes, calls six tools, and the engineer comes back to the result. **Trade-offs:** you now own a run store, a queue, workers and a state machine (§3) — real engineering — but you gain resumability, budgets enforced centrally, auditing, cancellation, and one place to put every cross-cutting concern. This is the default shape for anything beyond §2.1, and §14 builds one.

### 2.3 A chat or assistant UI with streaming and visible tool calls

A conversation surface where each user turn starts a run, the run's events stream to the UI as they happen, and tool calls are shown as collapsible steps.

```text
user turn ──▶ run ──▶ events: tool.called ─▶ "Looking up order A1001…"
                              tool.result ─▶ "charged twice, $20"
                              approval.required ─▶ [Approve refund $20] [Deny]
                              message.delta ─▶ "Refunded $20.00 on…"
```

**When:** the user is present, wants to see progress, and may be asked to approve. **Example:** a billing assistant inside the account page. **Trade-offs:** the best experience for interactive work and the natural home for approvals; but you must design what each event looks like, keep the conversation state across turns (§4.1), handle disconnects (§8.4), and resist streaming raw model reasoning to end users (§8.1). It is §2.2 plus a UI contract.

### 2.4 An event-driven worker on a queue

No user starts the run; an event does — a webhook, a "ticket created" message, a file landing in a bucket.

```text
"ticket.created" ──▶ queue ──▶ worker: run(agent, ticket)
                                  │ writes: labels, a draft, a note
                                  ▼
                            ticket system ◀── (human reviews the draft)
```

**When:** the work is triggered by the system, not a person, and nobody is waiting on a screen. **Example:** every new ticket gets triaged — category, priority, a draft reply saved as an internal note for an agent to send or edit. **Trade-offs:** scales naturally with the queue and tolerates bursts; but with no user present, a bad action has no one to stop it — so writes are "suggest" or "act and report" level ([52 §8](52-ai-agents.md#8-autonomous-agents)), failures must page an operator, and you need dead-letter handling and idempotent consumption (the same event delivered twice must not run twice — §4.3).

### 2.5 Scheduled and batch agents

A scheduler fans out one run per item at a fixed time: nightly, hourly, weekly.

```text
cron 02:00 ──▶ select 5,000 stale tickets ──▶ N workers ──▶ run per ticket
                                                   │ rate-limited, budgeted
                                                   ▼
                                             report: 4,870 ok, 130 failed
```

**When:** throughput matters more than latency and the work is embarrassingly parallel. **Example:** re-categorize the backlog after a taxonomy change; summarize every long thread older than 30 days. **Trade-offs:** cheapest per item (pack batches, run off-peak — [55 §17](55-ai-production.md#17-streaming-and-async-jobs)); but a bug multiplies by N before anyone looks, so run a sample first, cap the batch's total budget, and make every item's run idempotent so a re-run after a crash skips what is done.

### 2.6 A durable workflow engine

For runs lasting minutes to days with approvals in the middle, wrap the loop in a workflow engine that records every step durably and replays after a crash: Temporal, DBOS, Inngest, Restate, or LangGraph's checkpointer inside one process ([61 §11.2](61-ai-agent-frameworks.md#112-durable-execution-platforms-temporal-dbos-inngest-restate)).

```text
workflow(run):
  step 1: model call      ──▶ recorded
  step 2: tool: get_order ──▶ recorded
  step 3: wait_for_signal("approval", timeout=24h)   ◀── human, hours later
  step 4: tool: issue_refund (exactly once)
  step 5: model call → done
       crash anywhere ──▶ restart replays 1..k from the log, continues at k+1
```

**When:** a run must survive restarts and deploys, wait hours for a human, and apply side effects exactly once. **Example:** a vendor-onboarding agent that gathers documents, waits for compliance sign-off, then creates accounts. **Trade-offs:** you get crash safety and long waits for free; you pay with determinism rules (the workflow body must replay identically, so model calls and tools live in "activities"), an engine to operate, and a learning curve. Use it when §2.2's "resume from the last checkpoint" is not enough.

### 2.7 Agent as a tool inside another agent or workflow

A whole agent is exposed as one tool with a typed input and output, and an orchestrator — another agent or a fixed workflow — calls it like any other function ([52 §11](52-ai-agents.md#11-multi-agent-systems)).

```text
orchestrator ──▶ tool: research_agent(question) ──▶ {answer, sources, cost}
             ──▶ tool: code_agent(spec)           ──▶ {diff, tests_passed}
             ──▶ synthesize
```

**When:** sub-problems are genuinely separable with different tools, and an eval shows the split helps. **Example:** a release-notes workflow that calls a "summarize this PR" agent per PR, then a fixed template. **Trade-offs:** clean boundaries and independent testing per sub-agent; but each nested agent has its own budget, its own failure modes and its own cost, so the outer run's budget must cap the sum (§9.5), and the inner agent's result must be structured and size-capped like any tool result (§6.1).

### 2.8 Local harnesses: IDE, CLI, the Kali agent

The agent runs on the user's machine or in their VM, with tools that touch local files, a shell, or a lab network. Coding assistants in an IDE, a CLI agent, and the handbook's Kali agent — a local Qwen3-14B driving typed tools behind a deterministic policy layer — are this shape ([11 — The Kali agent](11-kali-agent.md), [11 §1](11-kali-agent.md#1-architecture)).

```text
user ◀──▶ harness (IDE / CLI) ──▶ model (local or remote)
              │ policy: scope, allow-list, confirm, audit
              ▼
          tools: files, shell, nmap_scan … (sandboxed)
```

**When:** the data and the actions live on the user's machine, and the user is the approver. **Example:** "fix the failing test" in an editor; "enumerate services on the in-scope host" in the lab. **Trade-offs:** the richest tools and the tightest human loop; but the harness runs with the user's local privileges, so the policy layer, allow-lists and audit log ([11 §5](11-kali-agent.md#5-the-policy-layer)) carry the whole safety story, and nothing is observable centrally unless the harness reports home.

### 2.9 Choosing a shape

Map requirements to shapes. When two shapes fit, pick the simpler one on the left.

| Requirement | In-process (2.1) | Run service (2.2) | Chat UI (2.3) | Queue worker (2.4) | Batch (2.5) | Durable (2.6) | Agent-as-tool (2.7) | Local (2.8) |
|---|---|---|---|---|---|---|---|---|
| Latency: answer in < 10 s | yes | poll/stream | stream | — | — | — | — | yes |
| Duration: seconds–minutes | ≤ 3 steps | yes | yes | yes | yes | yes | inside parent | yes |
| Duration: hours–days, survives restarts | no | checkpoints | no | with retries | no | **yes** | no | no |
| Human approvals mid-run | no | yes | **yes** | no (nobody present) | no | yes | via parent | yes |
| Volume: thousands of items | no | with workers | no | **yes** | **yes** | costly | — | no |
| Side effects allowed | read-only | gated | gated | suggest / report | idempotent only | exactly-once | parent's gate | policy layer |
| User is present | yes | optional | yes | no | no | optional | — | yes |
| New infrastructure | none | store + queue + workers | + UI contract | queue | scheduler | engine | — | harness |

The advice agrees with [61 §17](61-ai-agent-frameworks.md#17-how-to-choose-decision-tree-and-scenarios): a single tool-calling feature needs no framework (§2.1 or §2.2 by hand); a long-running workflow with approvals wants LangGraph's checkpointer, plus a durable engine for cross-crash guarantees (§2.6).

---

## 3. The run contract

Whatever the shape, the product and the agent runtime agree on three documents — a **request**, a **record**, and an **event stream** — and on a **state machine**. Write them down before writing code; every section after this one refers back to them.

### 3.1 The run request

What the product sends to start a run. Shown as an example, not a formal schema — your validator (pydantic, JSON Schema) enforces it.

```json
{
  "input": "Refund the duplicate charge on order A1001.",
  "user": "u_42",
  "tenant": "acme",
  "context": {"ticket_id": "T-7781", "order_ids": ["A1001"]},
  "allowed_tools": ["get_order", "issue_refund"],
  "constraints": {"max_steps": 6, "max_tokens": 20000, "max_seconds": 60,
                  "max_cost_usd": 0.05, "approval_seconds": 900},
  "idempotency_key": "ticket-7781-refund-1",
  "versions": {"prompt": "refund-v12", "tools": "2026-10-20"}
}
```

- `input` is the user's task; `context` carries **references** (ids), not documents — the agent fetches what it needs through tools, so the request stays small and the access check happens at fetch time.
- `allowed_tools` is the intersection of what the tenant has, what the user may do, and what this feature needs (§5.2). The runtime never offers the model a tool outside this list.
- `constraints` are the budgets ([52 §14](52-ai-agents.md#14-budgets)), which the runtime enforces (§9.5). A request can tighten the defaults, never loosen them past a server-side ceiling.
- `idempotency_key` makes the create call safe to retry (§4.3).
- `versions` pins the prompt and tool set, so the run is reproducible and attributable ([55 §18.1](55-ai-production.md#181-version-everything)).

### 3.2 The run record

What `GET /runs/{id}` returns, and what the run store holds.

```json
{
  "id": "run_01HF3Q9",
  "state": "waiting_approval",
  "user": "u_42", "tenant": "acme",
  "steps": 2,
  "usage": {"tokens_in": 1840, "tokens_out": 96, "cost_usd": 0.0011},
  "pending": {"tool": "issue_refund", "args": {"order_id": "A1001", "amount": 20.0},
              "approval_id": "apr_7", "expires_at": "2026-10-29T10:15:09Z"},
  "result": null,
  "error": null,
  "versions": {"prompt": "refund-v12", "tools": "2026-10-20", "model": "Qwen3-14B-4bit"},
  "created_at": "2026-10-29T10:00:02Z", "updated_at": "2026-10-29T10:00:09Z"
}
```

The record answers the product's questions without the transcript: *is it done, what is it waiting for, what did it cost, what came out.* The transcript (every message) lives in the `steps` table (§4.4) and is not returned by default — it is large and it contains customer data.

### 3.3 The event stream

What `GET /runs/{id}/events` streams, one JSON object per event, with a monotonically increasing `id` so a client can resume (§8.4).

| Event | Payload | Emitted when |
|---|---|---|
| `run.started` | — | a worker picked the run up |
| `step.thought` | `text` | **omit for end users**; store a sanitized version for operators if your policy allows |
| `tool.called` | `step, tool, args` | the model asked for a tool (before it runs) |
| `tool.result` | `step, tool, result` (size-capped) | the tool returned |
| `approval.required` | `pending{tool, args, preview, expires_at}` | a write tool needs a human |
| `run.resumed` | `approved` | the decision arrived |
| `message.delta` | `text` | a piece of the final answer (many per run when streaming tokens) |
| `run.completed` | `result` | the model returned a final answer |
| `run.failed` | `error` | an exception, a budget, or an unrecoverable tool error |
| `run.cancelled` | — | the user or the kill switch stopped it |
| `run.timed_out` | `error` | wall time or approval deadline exceeded |

The last four are **terminal**: exactly one of them ends every stream. A client that sees none of them when the connection closes must reconnect (§8.4).

### 3.4 The run state machine

```text
          create                     worker picks up
 (none) ─────────▶ queued ───────────────────────────▶ running ◀─────────────┐
                     │                                   │                   │
                     │ cancel                 write tool │                   │ approve / deny
                     ▼                                   ▼                   │
                 cancelled ◀──── cancel ──────── waiting_approval ───────────┘
                                                         │
                     ┌─────────────┬───────────┬─────────┴──────── deadline
                     ▼             ▼           ▼                     ▼
                 completed      failed     cancelled             timed_out
```

Transition rules — the runtime enforces them; nothing else moves a run:

| From | To | Only when |
|---|---|---|
| queued | running | a worker took it from the queue |
| queued | cancelled | cancel arrived before any worker took it |
| running | waiting_approval | the model called a write tool; the proposed action is recorded first |
| running | completed | the model returned no tool call; `result` is set |
| running | failed | an exception, a step/token/cost budget, or an unrecoverable tool error; `error` is set |
| running | cancelled | the cancel flag was set, checked at a step boundary or inside a tool |
| running | timed_out | the wall-time budget expired |
| waiting_approval | running | a decision arrived (approved or denied — both resume; a denial becomes a tool result the model can read) |
| waiting_approval | cancelled | the user cancelled while waiting |
| waiting_approval | timed_out | the approval deadline passed with no decision |

Terminal states have no outgoing transitions. A late approval on a completed run is rejected, not applied. A worker that crashes leaves the run in `running`; a reaper marks stale `running` runs (no heartbeat for N seconds) `failed` with the reason `worker lost`, or — if you allow retries — re-enqueues them as a new attempt from the last checkpoint (§4.2). That `running → queued` move is reserved for the reaper and deliberately absent from the table above: no client can take it.

The table as code, with the two mistakes it must catch:

```python
NEXT = {
    "queued":           {"running", "cancelled"},
    "running":          {"waiting_approval", "completed", "failed", "cancelled", "timed_out"},
    "waiting_approval": {"running", "cancelled", "timed_out"},
}                                   # terminal states have no entry: nothing leaves them

def transition(run, to, reason):
    frm = run["state"]
    if to not in NEXT.get(frm, set()):
        raise ValueError(f"illegal transition {frm} -> {to} ({reason})")
    run["state"] = to
    run["history"].append((frm, to, reason))

run = {"id": "run_0001", "state": "queued", "history": []}
for to, why in [("running", "a worker picked it up"),
                ("waiting_approval", "issue_refund is a write tool"),
                ("running", "approved by u_42"),
                ("completed", "the model returned a final answer")]:
    transition(run, to, why)
for frm, to, why in run["history"]:
    print(f"{frm:>16} -> {to:<16} {why}")

for bad_run, to, why in [(run, "running", "a late approval arrived"),
                         ({"state": "queued", "history": []}, "completed",
                          "a worker skipped the running state")]:
    try:
        transition(bad_run, to, why)
    except ValueError as e:
        print("rejected:", e)
```

```text
          queued -> running          a worker picked it up
         running -> waiting_approval issue_refund is a write tool
waiting_approval -> running          approved by u_42
         running -> completed        the model returned a final answer
rejected: illegal transition completed -> running (a late approval arrived)
rejected: illegal transition queued -> completed (a worker skipped the running state)
```

The history is the audit trail of the run's life, separate from the audit log of its side effects (§5.4). In production `transition` is one `UPDATE … WHERE id = ? AND state = ?` whose row count tells you whether you won the race.

---

## 4. State and sessions

### 4.1 Three kinds of state

An agent feature juggles three states that are easy to confuse because they all look like "messages":

| State | Lives | Holds | Owned by |
|---|---|---|---|
| **Conversation state** | across turns of one chat | the user-visible exchange: user turns, final answers, maybe the tool calls that were shown | the product (a `conversations` table) |
| **Run state** | one run | the full transcript the model sees — system prompt, tool schemas, every tool call and result, budgets, the pending approval | the agent runtime (`runs` + `steps`) |
| **Long-term memory** | across conversations | facts about the user or account, past decisions, documents | a store you retrieve from, not stuff into context ([52 §10](52-ai-agents.md#10-memory), [51 — RAG](51-ai-rag.md)) |

A new user turn does not continue the old run; it starts a new run whose input is built from the conversation state (recent turns, summarized when long) plus retrieved memory. The transcript of the previous run — forty tool results — does not come along. This keeps every run's context bounded and makes runs independent units for retries, budgets and audit.

### 4.2 Checkpoints and resumability

Persist the run's transcript after **every step** — the explicit-state, checkpoint, resume discipline of [52 §8.1](52-ai-agents.md#81-long-running-agents), now with a database. A step is: one model call plus the tool calls it asked for. The checkpoint is the messages list after the tool results are appended, plus the token counters. On restart the worker loads the checkpoint, skips completed tool calls (their results are stored), and makes the next model call. A crash costs at most one step — and only if that step's tool calls were idempotent (§4.3), otherwise a repeated write is the real cost.

### 4.3 Idempotency keys: for runs and for tool calls

Two different keys, two different bugs prevented.

- **Run key.** The client sends `idempotency_key` with `POST /runs`. The server stores it with a unique constraint; a retry of the same create (a flaky network, a double-clicked button, a redelivered webhook) returns the *existing* run instead of starting a second one. The key is the product's natural identity for the task: `ticket-7781-refund-1`, the webhook's delivery id, the batch item id.
- **Tool-call key.** Each side-effecting tool call carries `run_id:step:tool_call_id`. The tool wrapper (or the downstream API, if it supports an idempotency key) stores the result under that key; a replay after a crash, or a model that asks for the same refund twice, returns the stored result instead of acting again. [52 §5](52-ai-agents.md#5-tool-calling-systems-the-architecture) shows the cache in the dispatcher.

Keys expire with the run's retention (§4.5). A key that lives forever would make a legitimately repeated task (next month's refund on the same order) look like a duplicate — include the task identity, not just the resource, in the key.

### 4.4 A minimal schema

Four tables carry everything above. Types are generic SQL; adjust to your database.

```sql
CREATE TABLE runs (
  id               TEXT PRIMARY KEY,         -- run_01HF3Q9 (ULID/UUID)
  tenant_id        TEXT NOT NULL,
  user_id          TEXT NOT NULL,
  idempotency_key  TEXT UNIQUE,              -- retry-safe create
  state            TEXT NOT NULL,            -- queued | running | waiting_approval | ...
  input            TEXT NOT NULL,            -- redacted before storage
  result           TEXT,
  error            TEXT,
  steps            INTEGER NOT NULL DEFAULT 0,
  tokens_in        INTEGER NOT NULL DEFAULT 0,
  tokens_out       INTEGER NOT NULL DEFAULT 0,
  cost_usd         REAL    NOT NULL DEFAULT 0,
  prompt_version   TEXT NOT NULL,
  tools_version    TEXT NOT NULL,
  model            TEXT NOT NULL,
  heartbeat_at     TIMESTAMP,                -- the reaper fails or requeues stale running runs
  created_at       TIMESTAMP NOT NULL,
  updated_at       TIMESTAMP NOT NULL,
  expires_at       TIMESTAMP NOT NULL        -- retention: deleted after this
);

CREATE TABLE steps (                          -- one row per model call: the checkpoint
  run_id           TEXT REFERENCES runs(id),
  n                INTEGER,                   -- 1-based step number
  messages_json    TEXT NOT NULL,             -- transcript after this step's tool results
  tokens_in        INTEGER, tokens_out INTEGER,
  started_at       TIMESTAMP, finished_at TIMESTAMP,
  PRIMARY KEY (run_id, n)
);

CREATE TABLE tool_calls (
  id               TEXT PRIMARY KEY,         -- run_id:step:tool_call_id: the idempotency key
  run_id           TEXT REFERENCES runs(id),
  step             INTEGER NOT NULL,
  tool_call_id     TEXT NOT NULL,            -- the model's id; unique only within one response
  tool             TEXT NOT NULL,
  args_json        TEXT NOT NULL,
  result_json      TEXT,                     -- size-capped
  error            TEXT,
  started_at       TIMESTAMP, finished_at TIMESTAMP
);

CREATE TABLE approvals (
  id               TEXT PRIMARY KEY,
  run_id           TEXT REFERENCES runs(id),
  tool_call_id     TEXT REFERENCES tool_calls(id),
  requested_at     TIMESTAMP NOT NULL,
  expires_at       TIMESTAMP NOT NULL,
  decided_by       TEXT,                     -- a user id, never "system"
  decision         TEXT,                     -- approved | denied | expired
  decided_at       TIMESTAMP
);
```

`steps.messages_json` is the big column; it is the checkpoint and it is where PII accumulates. `tool_calls` is the audit log of side effects (§5.4) joined to who approved them. A queue table (or a real queue) holds only run ids.

### 4.5 Retention and PII

Transcripts contain whatever the tools returned: names, emails, order histories. Decide retention **per table** before launch: the `runs` row (metadata, cost, state) can live for a year for billing and metrics; `steps.messages_json` should live for days — long enough to debug and to resume — and then be deleted or reduced to a redacted summary. Redact before storing (emails, card numbers, keys — [56 §8](56-ai-safety.md#8-pii-detection-and-redaction)), let tenants opt out of content logging while you keep the metadata, and restrict who can read transcripts ([55 §16.4](55-ai-production.md#164-privacy-in-logs), [56 §16](56-ai-safety.md#16-privacy-and-regulation)). The `expires_at` column makes the policy mechanical: a nightly job deletes what is past it.

The storage you commit to is runs per day times steps per run times bytes per checkpoint times the retention window:

\[
S = R \cdot n \cdot b \cdot D
\]

!!! example "On paper"

    1,000 runs a day, 8 steps each, a 3 KB checkpoint per step, kept 90 days:

    \[
    \begin{aligned}
    \text{per day} &= 1000 \cdot 8 \cdot 3\ \text{KB} = 24{,}000\ \text{KB}\\
    &= 24\ \text{MB}\\
    S &= 24\ \text{MB} \cdot 90 = 2{,}160\ \text{MB}
    \end{aligned}
    \]

    About 2.2 GB of transcripts on disk at any time — small, which is exactly why teams keep them forever "for debugging" and end up with a breach surface. Keep the 90 days for the metadata and cut the transcripts to 7.

---

## 5. Identity and permissions

### 5.1 On behalf of a user, never a shared superuser

The script called tools with one API key that could do everything. In the product, the agent acts **on behalf of the user who started the run**, and every tool call carries that identity. The failure this prevents is the classic confused deputy: user A asks the agent to "show me the latest orders", the agent's shared key can read every tenant's orders, and a prompt-injected ticket body ([56 §2](56-ai-safety.md#2-prompt-injection)) turns a read into a leak. With per-user credentials the tool returns A's orders because the *database* said so, not because the prompt asked nicely. The same rule governs MCP servers: the server authenticates callers and never passes a client's token downstream ([58 §7](58-ai-mcp.md#7-authorization-for-http-servers), [58 §7.2](58-ai-mcp.md#72-audience-validation-and-no-token-passthrough)).

### 5.2 Allow-lists per user and per tenant

The tool list the model sees is computed per run from three sets: what the feature needs, what the tenant has, what the user is allowed. The model never learns a tool exists if it cannot use it — a tool it cannot see is a tool it cannot be tricked into calling.

```python
import hashlib, json

TOOL_SCOPE = {"get_order": "orders:read", "issue_refund": "payments:write",
              "delete_account": "accounts:admin"}
TENANT_TOOLS = {"acme": {"get_order", "issue_refund"},        # what each tenant bought
                "globex": {"get_order"}}
USERS = {"u_42": {"tenant": "acme", "scopes": {"orders:read", "payments:write"}},
         "u_7":  {"tenant": "globex", "scopes": {"orders:read", "payments:write"}},
         "u_99": {"tenant": "acme", "scopes": {"orders:read"}}}

def tools_for(user_id):
    """The tool list this run may offer the model: tenant allow-list AND user scopes."""
    u = USERS[user_id]
    return sorted(t for t, scope in TOOL_SCOPE.items()
                  if t in TENANT_TOOLS[u["tenant"]] and scope in u["scopes"])

def mint_token(user_id, tool, now=1_700_000_000, ttl_s=60):
    """A scoped, short-lived credential for one tool call (your IdP/STS in production)."""
    claims = {"sub": user_id, "scope": TOOL_SCOPE[tool], "exp": now + ttl_s}
    sig = hashlib.sha256(json.dumps(claims, sort_keys=True).encode() + b"server-secret")
    return claims, sig.hexdigest()[:12]

AUDIT = []
def audit(user, run_id, step, tool, args, decision, token_sig):
    AUDIT.append({"user": user, "run": run_id, "step": step, "tool": tool,
                  "args": args, "decision": decision, "token": token_sig})

for uid in USERS:
    print(f"{uid}: {tools_for(uid)}")
claims, sig = mint_token("u_42", "issue_refund")
print("token claims:", claims, "sig:", sig)
audit("u_42", "run_0001", 2, "issue_refund", {"order_id": "A1001", "amount": 20.0},
      "approved", sig)
print("audit:", json.dumps(AUDIT[0]))
```

```text
u_42: ['get_order', 'issue_refund']
u_7: ['get_order']
u_99: ['get_order']
token claims: {'sub': 'u_42', 'scope': 'payments:write', 'exp': 1700000060} sig: dfd28df37225
audit: {"user": "u_42", "run": "run_0001", "step": 2, "tool": "issue_refund", "args": {"order_id": "A1001", "amount": 20.0}, "decision": "approved", "token": "dfd28df37225"}
```

`u_7` loses `issue_refund` because the tenant did not buy it; `u_99` loses it because the user lacks the scope; `delete_account` reaches nobody because no tenant list includes it. The computation runs in the runtime, before the first model call, and its result is stored on the run (`allowed_tools` in §3.1).

### 5.3 Scoped, short-lived tokens for tools

A tool wrapper should not hold a long-lived credential. Mint (or have your identity provider mint) a token scoped to one capability and one subject, valid for seconds to minutes, per tool call or per run — `mint_token` above is the shape, with a real signature and a real issuer in production. If a transcript, a log line or an MCP server leaks the token, it is scoped to `payments:write` for `u_42` and already expired. The MCP authorization spec has the same principle for remote servers: minimal `scopes_supported`, step-up for privileged operations, no omnibus scope ([58 §7](58-ai-mcp.md#7-authorization-for-http-servers)).

### 5.4 The audit log of side effects

Every side effect is logged **before** it executes and **after**, with `user`, `tenant`, `run`, `step`, `tool`, the exact arguments, the policy decision, who approved it, and a reference to the credential used — the Kali agent's audit record is the template ([11 §5.2](11-kali-agent.md#52-audit-log)). Store it append-only, outside the sandbox, with its own retention (longer than transcripts: it is the record that answers "who refunded this and why"). Read-only tool calls go to the trace (§10), not to the audit log, unless they touch regulated data. This is the "excessive agency" control of [56 §6](56-ai-safety.md#6-excessive-agency) made concrete: functionality (allow-lists), permissions (scoped tokens), autonomy (approvals, §7), each leaving a line in the audit log; the operational rules of [58 §12](58-ai-mcp.md#12-security-best-practices) apply when a tool arrives over MCP.

---

## 6. Connecting tools

### 6.1 Typed wrappers around internal APIs

A tool is a thin, typed wrapper around an API you already have. It validates arguments in (strict types, bounds, no extra fields), calls the API with the run's scoped credential, and returns a **structured, size-capped** result out, with actionable errors — the design rules of [52 §4](52-ai-agents.md#4-designing-tools), applied with pydantic.

```python
import json
from pydantic import BaseModel, Field, ValidationError

class RefundArgs(BaseModel):
    model_config = {"extra": "forbid"}          # hallucinated fields are rejected
    order_id: str = Field(pattern=r"^A\d{4}$")
    amount: float = Field(gt=0, le=500)

def billing_api_refund(order_id, amount):       # the internal API behind the tool
    return {"refund_id": "rf_9", "order_id": order_id, "amount": amount,
            "ledger": [{"entry": i} for i in range(40)]}

def cap(result, limit=120):
    blob = json.dumps(result)
    if len(blob) <= limit:
        return result
    return {"truncated": blob[:limit] + "...", "full_chars": len(blob)}

def issue_refund(raw_args, dry_run=False):
    """Validate in, call the API (or preview it), cap the result out."""
    try:
        args = RefundArgs(**raw_args)
    except ValidationError as e:                # actionable text for the model
        return {"error": "; ".join(f"{x['loc'][0]}: {x['msg']}" for x in e.errors())}
    if dry_run:
        return {"would_refund": args.amount, "order_id": args.order_id, "side_effect": False}
    return cap(billing_api_refund(args.order_id, args.amount))

print(issue_refund({"order_id": "A1001", "amount": 20.0}, dry_run=True))
print(issue_refund({"order_id": "A1001", "amount": 20.0}))
print(issue_refund({"order_id": "A1001", "amount": 20.0, "reason": "dup"}))
print(issue_refund({"order_id": "1001", "amount": 9000}))
```

```text
{'would_refund': 20.0, 'order_id': 'A1001', 'side_effect': False}
{'truncated': '{"refund_id": "rf_9", "order_id": "A1001", "amount": 20.0, "ledger": [{"entry": 0}, {"entry": 1}, {"entry": 2}, {"entry"...', 'full_chars': 660}
{'error': 'reason: Extra inputs are not permitted'}
{'error': "order_id: String should match pattern '^A\\d{4}$'; amount: Input should be less than or equal to 500"}
```

The second call shows the cap doing its job badly on purpose: a naive character cut truncates the useful fields along with the 40-entry ledger. A good wrapper **selects** fields before it caps (`refund_id`, `order_id`, `amount`) and drops the ledger entirely; the cap is the backstop, not the design. The last two calls show why `extra = "forbid"` and bounds matter: the error text names the field and the rule, which is what a model needs to correct itself.

### 6.2 MCP servers for shared tools

When several products or agents need the same tools — the ticket store, the order API, the lab's scanners — put them behind an MCP server once, and let each runtime connect as a client ([58 — MCP](58-ai-mcp.md)). The bridge that converts MCP tool definitions into the OpenAI tool shape and runs the loop against the local Qwen model is [58 §9](58-ai-mcp.md#9-build-a-client-and-bridge-mcp-to-the-local-model); inside a product the bridge lives in the worker. Three integration rules on top of the protocol: the runtime still applies the per-run allow-list (§5.2) *after* listing the server's tools; tool annotations such as `readOnlyHint` drive the read/write classification of §6.4 only for servers you control; and a remote server's results are untrusted input like any other ([58 §12.1](58-ai-mcp.md#121-threats-from-the-server-and-its-content)).

### 6.3 Long tools: webhooks and callbacks

A tool that takes minutes — a scan, a build, a report job — must not block the worker. Make it two tools: `start_scan(target) → {job_id}` and `get_scan(job_id) → {status, result?}`, and let the model poll, with the step budget bounding the polls. Better, make the long job call back: the tool returns `{job_id, status: "pending"}`, the run transitions to a waiting state (a `waiting_tool` sibling of `waiting_approval`), and the job's webhook `POST /runs/{id}/tool-results/{tool_call_id}` delivers the result and resumes the run — exactly the signal-and-resume mechanics of the durable engines in §2.6. Webhook deliveries are retried by the sender, so the result endpoint must be idempotent on `tool_call_id` (§4.3).

### 6.4 Read vs write, dry-run and confirm

Classify every tool as **read** (no side effect: lookups, searches, scans within scope) or **write** (money, messages, deletes, configuration, anything outside the sandbox). Reads run; writes go through the approval gate (§7) or, at the "act and report" level, run with an idempotency key and a loud audit entry. Two patterns make writes safe to propose:

- **Dry run.** A write tool accepts `dry_run=True` and returns what it *would* do (`would_refund: 20.0`) with no side effect. The agent calls it first; the preview becomes the approval card's content (§7.1).
- **Confirm.** The real call needs a `confirmation_token` that only the approval step mints. A model cannot fabricate it, so even a prompt-injected "refund now" produces a tool error, not a refund — the policy lives in code, not in the prompt ([52 §15](52-ai-agents.md#15-human-in-the-loop)).

### 6.5 Result truncation math

Tool results are the main thing that grows the context. With a window \(W\), a fixed prefix \(T_\text{sys}\) (system prompt plus tool schemas), \(n\) steps, and \(T_\text{turn}\) tokens of model output per step, the largest tool result you can afford per step is:

\[
r_\text{max} = \frac{W - T_\text{sys} - n\,T_\text{turn}}{n}
\]

!!! example "On paper"

    A 32,768-token window, a 2,000-token prefix, 10 steps, 200 tokens of model output per step:

    \[
    \begin{aligned}
    n\,T_\text{turn} &= 10 \cdot 200 = 2000\\
    W - T_\text{sys} - n\,T_\text{turn} &= 32768 - 4000 = 28768\\
    r_\text{max} &= 28768 / 10 \approx 2876
    \end{aligned}
    \]

    Any tool result over about 2,876 tokens, repeated for ten steps, overflows the window before the run can finish. A 1,000-token cap leaves room for the final answer and a summary; a raw 4,000-token result (a full order history, a whole log file) does not.

```python
W, T_SYS, N, T_TURN = 32768, 2000, 10, 200   # window, prefix, steps, model turn tokens

cap = (W - T_SYS - N * T_TURN) // N
print(f"largest safe tool result: {cap} tokens")
for raw in (4000, 1000):
    total = T_SYS + N * (raw + T_TURN)
    resent = (raw + T_TURN) * N * (N + 1) // 2   # 52 §13: the history is re-sent
    print(f"result={raw:4d} tokens: context at step {N} = {total:5d} "
          f"({'overflows' if total > W else 'fits'}), tokens billed = {resent}")
```

```text
largest safe tool result: 2876 tokens
result=4000 tokens: context at step 10 = 44000 (overflows), tokens billed = 231000
result=1000 tokens: context at step 10 = 14000 (fits), tokens billed = 66000
```

The second number is the bill: because the history is re-sent every step ([52 §13](52-ai-agents.md#13-cost-growth)), capping results at 1,000 tokens instead of 4,000 cuts the tokens processed by 3.5× as well as making the run possible at all. Cap at the wrapper, and compact further by selecting fields (§6.1) and summarizing old steps ([52 §10](52-ai-agents.md#10-memory)).

---

## 7. Approvals in a product

[52 §15](52-ai-agents.md#15-human-in-the-loop) gives the principle: gate the risky, show the exact action, default to deny, keep the gate in code. In a product the gate is a UI with a deadline, and the run is parked while a person decides.

### 7.1 What the approver sees

An approval card is a record, not a chat message. It shows the **proposed action** exactly as it will execute (tool and arguments, not a paraphrase), a **preview or diff** from a dry run (§6.4) — "Refund $20.00 of the $40.00 charged on A1001 (payment p2)", or a unified diff for a code change, or the rendered email — and the **evidence** the agent used: the tool results from earlier steps, linked to the trace. It states the deadline and the two outcomes. The approver must be able to **edit** the arguments (change $20 to $15) and **deny with a reason** that the model reads as a tool result, so it can finish the run sensibly ("The refund was not approved; no changes were made").

### 7.2 Approval timeouts and escalation

A run waiting for a human holds state, not a worker — the worker is released and the run sits in `waiting_approval` with an `expires_at` (`approval_seconds` in §3.1; minutes for an interactive user, hours or days for a back-office queue), and the approval endpoint re-enqueues it. (The minimal service of §14 blocks its worker on a queue instead, which is fine for a demo and wrong at scale: a hundred runs waiting on humans would be a hundred idle workers.) When it passes, the run transitions to `timed_out`, not silently to "approved": absence of a decision is a denial. Escalation is a second deadline: after a third of the window, notify a backup approver or a team queue; after the window, close the run and tell the requester. Never let an expired approval be applied later — the world moved on (the order may already be refunded by a human), which is why `timed_out` has no outgoing transition (§3.4): a decision that arrives after the deadline is rejected (the service of §14 answers `409`), never applied.

### 7.3 Batching, and "once for this run" versus per action

Four proposed writes can mean four interruptions, three, or one, depending on the policy:

```python
import json

WRITES = [("issue_refund", {"order_id": "A1001", "amount": 20.0}),
          ("add_note", {"order_id": "A1001", "text": "refunded duplicate charge"}),
          ("issue_refund", {"order_id": "A1002", "amount": 20.0}),
          ("send_email", {"to": "c_88", "template": "refund_done"})]

def prompts(policy):
    """How many times a human is asked, for the same four proposed writes."""
    shown, approved_tools = [], set()
    for tool, args in WRITES:
        if policy == "per action":
            shown.append([tool])
        elif policy == "once per tool per run" and tool not in approved_tools:
            shown.append([tool])
            approved_tools.add(tool)
    if policy == "batched per step":           # all writes of one step in one card
        shown.append([t for t, _ in WRITES])
    return shown

for policy in ("per action", "once per tool per run", "batched per step"):
    s = prompts(policy)
    print(f"{policy:22} -> {len(s)} prompt(s): {s}")

card = {"run_id": "run_0001", "step": 2, "tool": "issue_refund",
        "args": WRITES[0][1],
        "preview": "Refund $20.00 of the $40.00 charged on A1001 (payment p2)",
        "evidence": [{"step": 1, "tool": "get_order",
                      "result": {"status": "charged_twice", "amount": 20.0}}],
        "expires_in_s": 900, "escalate_to": "billing-leads"}
print(json.dumps(card, indent=1)[:200] + " ...")
```

```text
per action             -> 4 prompt(s): [['issue_refund'], ['add_note'], ['issue_refund'], ['send_email']]
once per tool per run  -> 3 prompt(s): [['issue_refund'], ['add_note'], ['send_email']]
batched per step       -> 1 prompt(s): [['issue_refund', 'add_note', 'issue_refund', 'send_email']]
{
 "run_id": "run_0001",
 "step": 2,
 "tool": "issue_refund",
 "args": {
  "order_id": "A1001",
  "amount": 20.0
 },
 "preview": "Refund $20.00 of the $40.00 charged on A1001 (payment p2)",
 "evidence ...
```

Which policy is right depends on reversibility, not convenience. "Once per tool per run" is fine for `add_note` (reversible, low stakes) and dangerous for `issue_refund` (the second refund goes to a different order the approver never saw). The safe rule: **per action for irreversible writes; once-per-run for reversible ones; batch only what the approver can read in one screen**, with the batch card listing every action individually and allowing per-item deny.

### 7.4 Resuming after approval

The decision is a `POST /runs/{id}/approve` with `{"approved": true|false, "args": {...edited...}, "reason": "..."}`. The runtime records it in `approvals` (who, when, what), transitions `waiting_approval → running`, executes the write with the approved (possibly edited) arguments and the idempotency key, appends the result — or the denial with its reason — as the tool message, and makes the next model call. The model never sees a difference between "the tool ran" and "a human ran the tool"; it sees a tool result and continues. §14 runs this end to end.

---

## 8. Streaming to the UI

### 8.1 SSE event shapes

Server-Sent Events is the simplest transport that fits: one long HTTP response, UTF-8 text, events separated by blank lines, three fields you use per event — `id`, `event`, `data` — and a client that reconnects by itself, sending the last `id` it saw in a `Last-Event-ID` header. The model's own token stream ([50 §12](50-ai-prompting-apis.md#12-streaming-with-sse)) is one SSE stream; the run's event stream to the UI is another, and the worker translates between them.

```text
id: 2
event: tool.called
data: {"tool": "get_order", "args": {"order_id": "A1001"}}

id: 3
event: tool.result
data: {"tool": "get_order", "result": {"status": "charged_twice"}}

id: 4
event: message.delta
data: {"text": "Refunded $20.00"}
```

What **not** to stream: `step.thought`. A thinking model's reasoning is unfiltered, may quote tool results verbatim (including other customers' data from a badly scoped tool), and is not an answer. Show a status line ("thinking…", "looking up the order…") derived from the event type, and keep the reasoning for operators in the trace, if your policy allows storing it at all. With the handbook's setup, thinking is off for tool loops (`enable_thinking: false`) and `content` can still be `None` on a tool-call turn — guard it with `content or ""` ([61 §14](61-ai-agent-frameworks.md#14-local-models-the-thinking-trap)).

### 8.2 Progressive rendering and waiting states

The UI renders each event type differently: a tool call becomes a collapsed step with a spinner that resolves when its result arrives; message deltas append to the answer; terminal events close the run. The same code, in memory, with a connection that drops and reconnects:

```python
import json

EVENTS = [
    (1, "run.started", {}),
    (2, "tool.called", {"tool": "get_order", "args": {"order_id": "A1001"}}),
    (3, "tool.result", {"tool": "get_order", "result": {"status": "charged_twice"}}),
    (4, "message.delta", {"text": "Refunded $20.00"}),
    (5, "message.delta", {"text": " on order A1001."}),
    (6, "run.completed", {"result": "Refunded $20.00 on order A1001."}),
]

def serve(after_id, drop_after=None):
    """The server side: SSE text for events after `after_id` (Last-Event-ID)."""
    out = ""
    for id_, event, data in EVENTS:
        if id_ <= after_id:
            continue
        out += f"id: {id_}\nevent: {event}\ndata: {json.dumps(data)}\n\n"
        if id_ == drop_after:               # simulate the connection dying here
            break
    return out

def parse(text):
    for block in text.strip().split("\n\n"):
        f = dict(line.split(": ", 1) for line in block.split("\n"))
        yield int(f["id"]), f["event"], json.loads(f["data"])

print("wire:", serve(0)[:34].replace("\n", "\\n"), "...")
ui = {"text": "", "last_id": 0}
for attempt, drop in ((1, 3), (2, None)):   # first connection drops after event 3
    print(f"[conn {attempt}] Last-Event-ID: {ui['last_id']}")
    for id_, event, data in parse(serve(ui["last_id"], drop)):
        ui["last_id"] = id_
        if event == "tool.called":
            print(f"  waiting: {data['tool']}({data['args']}) ...")
        elif event == "tool.result":
            print(f"  {data['tool']} -> {data['result']}")
        elif event == "message.delta":
            ui["text"] += data["text"]
            print(f"  render: {ui['text']!r}")
        elif event.startswith("run."):
            print(f"  {event}")
```

```text
wire: id: 1\nevent: run.started\ndata: {}\n ...
[conn 1] Last-Event-ID: 0
  run.started
  waiting: get_order({'order_id': 'A1001'}) ...
  get_order -> {'status': 'charged_twice'}
[conn 2] Last-Event-ID: 3
  render: 'Refunded $20.00'
  render: 'Refunded $20.00 on order A1001.'
  run.completed
```

The waiting state matters more than it looks: a tool call can take ten seconds, and a UI that shows nothing during it reads as broken. The `tool.called` event, emitted *before* the tool runs, is what lets the UI show "looking up order A1001…" immediately.

### 8.3 Partial failure mid-stream

A run that fails at step 4 has already streamed three tool steps and perhaps half an answer. The stream must end with `run.failed` (or `run.timed_out`, `run.cancelled`) carrying a reason the UI can show, and the UI must keep what was rendered — a visible "stopped: token budget exceeded" under the partial text is better than a blank. The record (§3.2) keeps the partial result and the error so a retry can be offered. What the UI must never do is treat a closed connection as success: only a terminal event is success or failure; a close without one is a disconnect (§8.4).

### 8.4 Reconnect with the last event id

Because every event has an increasing `id` and the server keeps the run's events (in memory while running, in the `steps` and `tool_calls` tables after), a client that lost the connection reconnects with `Last-Event-ID: 3` and receives events 4 onward — the second connection above. The browser's `EventSource` does this automatically; a mobile client does it by hand. The server side is a filter (`id > last`) plus a wait for new events; §14.2 implements it with a condition variable. For a run that finished while the client was away, the reconnect simply replays the tail and closes after the terminal event.

---

## 9. Reliability

[52 §12–14](52-ai-agents.md#12-reliability-math) gives the arithmetic — success compounds down with steps, cost compounds up, so budgets are mandatory — and [55 §14](55-ai-production.md#14-reliability-fallbacks-timeouts-circuit-breakers-degradation) the standard resilience kit for the model endpoint. Here is how the kit attaches to a run.

### 9.1 Timeouts per step and per run

Two layers, both enforced by the runtime. A **per-step** timeout bounds one model call (`timeout=` on the client) and one tool call (the wrapper's deadline); a **per-run** timeout bounds the whole run (`max_seconds`). They must be consistent: a run deadline shorter than the sum of the step deadlines will kill runs that are within every per-step limit.

\[
T_\text{run} \ge n_\text{max}\,(t_\text{model} + t_\text{tool})
\]

!!! example "On paper"

    A step budget of 6, a 20 s model timeout, a 10 s tool timeout:

    \[
    \begin{aligned}
    t_\text{model} + t_\text{tool} &= 30\ \text{s}\\
    n_\text{max}\cdot 30 &= 6 \cdot 30 = 180\ \text{s}
    \end{aligned}
    \]

    With `max_seconds = 60` the run can be killed after two slow steps that each stayed inside their limits. Either raise the run deadline to 180 s, or derive the step deadlines from it: \(60 / 6 = 10\) s per step, which for a 14B on an M1 Pro at ~16 tokens/s ([50 §14](50-ai-prompting-apis.md#14-the-latency-model)) means short tool results and short answers. Set the run deadline from the product (how long a user will wait) and the step deadlines from it, not the other way round.

### 9.2 Retries with idempotency

Retry the model call on timeouts, 429 and 5xx with backoff and jitter ([50 §15](50-ai-prompting-apis.md#15-timeouts-retries-backoff-and-idempotency)); never retry a 400. Retry a *tool* only if it is read-only or carries an idempotency key (§4.3). Retry a *run* (requeue after a worker crash) only from its last checkpoint, and cap the attempts: a run that crashed three workers is a bug, not bad luck — fail it and page. Every retry raises the per-step success rate \(p\) and so the run's \(p^n\) ([52 §12](52-ai-agents.md#12-reliability-math)), but only for failures you can detect; a wrong answer is not retried by any of this, which is what evals (§12.4) are for.

### 9.3 Circuit breakers to the model and to each tool

A breaker per dependency — one for the model endpoint, one per tool backend — stops a sick dependency from eating every worker's timeouts. Closed: calls pass. Open: calls fail fast and the run takes the fallback (§9.4). Half-open: after a cooldown, one probe call decides.

```python
class Breaker:
    """closed -> open after `threshold` straight failures; half-open after `cooldown` s."""
    def __init__(self, threshold, cooldown, clock):
        self.threshold, self.cooldown, self.clock = threshold, cooldown, clock
        self.failures, self.opened_at, self.state = 0, None, "closed"
    def allow(self):
        if self.state == "open" and self.clock() - self.opened_at >= self.cooldown:
            self.state = "half_open"              # let one probe call through
        return self.state != "open"
    def record(self, ok):
        if ok:
            self.failures, self.state = 0, "closed"
            return
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.threshold:
            self.state, self.opened_at = "open", self.clock()

NOW = [0.0]                                      # fake clock for a reproducible trace
br = Breaker(threshold=3, cooldown=30, clock=lambda: NOW[0])

def call_model(upstream_ok):
    if not br.allow():
        return "skipped -> fallback answer"       # fail fast, do not pile on
    br.record(upstream_ok)
    return "ok" if upstream_ok else "failed (timeout)"

for t, ok in [(0, False), (1, False), (2, False), (3, True), (20, True),
              (32, False), (40, True), (62, True), (63, True)]:
    NOW[0] = t
    print(f"t={t:>2}s upstream={'up  ' if ok else 'down'} -> {call_model(ok):26} state={br.state}")
```

```text
t= 0s upstream=down -> failed (timeout)           state=closed
t= 1s upstream=down -> failed (timeout)           state=closed
t= 2s upstream=down -> failed (timeout)           state=open
t= 3s upstream=up   -> skipped -> fallback answer state=open
t=20s upstream=up   -> skipped -> fallback answer state=open
t=32s upstream=down -> failed (timeout)           state=open
t=40s upstream=up   -> skipped -> fallback answer state=open
t=62s upstream=up   -> ok                         state=closed
t=63s upstream=up   -> ok                         state=closed
```

Three failures open the breaker at t = 2 s; the calls at 3 s and 20 s are skipped even though the upstream recovered, because the cooldown is 30 s; the probe at 32 s fails and re-opens it; the probe at 62 s succeeds and closes it. Tune `threshold` and `cooldown` per dependency: a tool backend that returns 503 for a minute during deploys wants a 60 s cooldown; the model endpoint wants a shorter one with a fallback model behind it.

### 9.4 Fallbacks: degrade, do not 500

When the breaker is open or the budget is spent, the run still has to end well. Decide the ladder per feature, in order of preference: **degrade to a single call** (no tools — answer from the context you already have, clearly labelled); **a canned answer** ("I could not check the order right now; here is how to request a refund manually"); **a human queue** (create a ticket with the transcript so far, tell the user who will follow up). Each rung is cheaper and less capable than the last; each is better than a spinner that ends in an error. The gateway of [55 §15](55-ai-production.md#15-a-minimal-gateway-in-python) shows the model-level fallback (a second model); the run-level ladder sits above it.

### 9.5 Budgets enforced by the runtime, not by the prompt

"You have at most 6 steps" in the system prompt is a suggestion the model will ignore under pressure. The budgets of [52 §14](52-ai-agents.md#14-budgets) — steps, tokens, wall time, money — are checked by the worker before every model call, from counters the worker keeps from `usage` and the clock. When one trips, the run transitions to `failed` (steps, tokens, cost) or `timed_out` (time) with the reason, the partial result is kept, and the fallback ladder (§9.4) decides what the user sees. A **per-tenant** budget sits above the per-run ones (§11.1): a tenant that has spent its monthly allowance gets runs refused at creation, not killed at step 5. §14.1 shows the step, token and approval budgets tripping.

### 9.6 Cancellation that reaches the tools

A user's "stop" must stop everything: no further model calls, and the tool that is running right now. A cancel flag on the run is checked at every step boundary — cheap and enough for model calls, which are short. Tools need cooperation: a long tool polls the flag, or it runs in a subprocess the worker kills, or its backend exposes a cancel endpoint. A cancelled write that had already started is the hard case: it must either complete and be recorded (the audit log says it happened), or roll back — never vanish half-done.

```python
import threading, time

def slow_tool(cancel, seconds):
    """A long tool that checks the run's cancel flag; a subprocess tool would be killed instead."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        if cancel.is_set():
            return {"error": "cancelled by the user", "partial": True}
        time.sleep(0.02)
    return {"ok": True}

def execute(cancel, log):
    for step in (1, 2, 3):
        if cancel.is_set():                      # checked at every step boundary...
            log.append(f"cancelled before step {step}")
            return
        out = slow_tool(cancel, seconds=0.5)     # ...and inside long tool calls
        log.append(f"step {step}: {out}")
        if "error" in out:
            return

cancel, log = threading.Event(), []
t0 = time.monotonic()
worker = threading.Thread(target=execute, args=(cancel, log))
worker.start()
threading.Timer(0.2, cancel.set).start()         # the user clicks "stop" 0.2 s in
worker.join()
elapsed = time.monotonic() - t0
for line in log:
    print(line)
print("stopped early:", elapsed < 0.45, "(3 steps would take 1.5 s)")
```

```text
step 1: {'error': 'cancelled by the user', 'partial': True}
stopped early: True (3 steps would take 1.5 s)
```

The tool returned within 20 ms of the flag being set, in the middle of its first 0.5 s step; without the check inside the tool, the cancel would have waited for the step boundary — 0.5 s here, minutes for a real scan. The kill switch (§13.3) sets the same flag on every running run.

---

## 10. Observability

### 10.1 One trace per run, a span per step and tool call

A run is a tree: the run span contains one span per step (the model call, with tokens and TTFT), and each step contains one span per tool call (name, duration, result size, error). [55 §16.2](55-ai-production.md#162-tracing-a-rag-or-agent-request) shows the shape for a RAG request; an agent run is the same tree, deeper.

```text
run run_0001  [9.4 s, 713 tokens, $0.0007, completed]
├─ step 1  model call      [2.1 s, 143 in / 21 out]
│  └─ tool get_order       [0.03 s, 58 B]
├─ step 2  model call      [2.4 s, 212 in / 27 out]
│  ├─ tool issue_refund    [waiting_approval 3.9 s → approved by u_42]
│  └─ tool issue_refund    [0.02 s, 41 B]
└─ step 3  model call      [0.9 s, 286 in / 24 out] → final answer
```

Propagate the trace id into every tool call (an HTTP header to your internal APIs, `_meta` for MCP) so a slow tool shows up as a slow span in the same tree, and put the trace id on the run record so support can jump from a user complaint to the trace.

### 10.2 Log fields

One structured log line per step and per tool call, with the fields of [55 §16.1](55-ai-production.md#161-what-to-log-per-request) plus the agent-specific ones:

| Field | Why |
|---|---|
| `run_id`, `step`, `tool_call_id`, `trace_id` | correlation across worker, tools, UI |
| `tenant`, `user` | budgets, rate limits, audit |
| `state_from`, `state_to`, `reason` | the state machine's history |
| `tool`, `args_hash`, `result_bytes`, `error` | tool behaviour without logging the arguments' content |
| `tokens_in`, `tokens_out`, `cost_usd` (cumulative) | budgets, cost per run |
| `prompt_version`, `tools_version`, `model` | attribution after a change |
| `approval_id`, `decided_by`, `decision`, `wait_s` | the human loop's health |
| `budget_remaining` (steps, tokens, seconds) | catching runs that nearly tripped |

Log argument *hashes* and result *sizes* by default; log content only in the trace store with the retention and redaction of §4.5.

### 10.3 The metrics that matter

Aggregate the run records, not the log lines. Six numbers tell you whether the feature works; compute them from a log of ten runs:

```python
import math

# One row per run: state, steps, tokens, wall seconds, tool calls, tool errors,
# approvals requested, approvals denied (a human override).
LOG = [
    ("completed", 3,  700,  4.0, 2, 0, 1, 0), ("completed", 4,  900,  5.0, 3, 0, 1, 0),
    ("failed",    6, 1800,  9.0, 6, 1, 0, 0), ("completed", 3,  650,  3.0, 2, 0, 1, 1),
    ("completed", 5, 1200,  6.0, 4, 1, 2, 0), ("cancelled", 2,  300,  2.0, 1, 0, 0, 0),
    ("completed", 3,  700,  4.0, 2, 0, 1, 0), ("completed", 4, 1000, 12.0, 3, 0, 1, 0),
    ("timed_out", 4, 1100, 60.0, 4, 0, 1, 0), ("completed", 3,  750,  4.0, 2, 0, 1, 0),
]
PRICE_PER_M = 1.00                      # assumption: blended $1 per million tokens

def percentile(values, p):              # nearest-rank, as in 55 §16.3
    s = sorted(values)
    return s[math.ceil(p / 100 * len(s)) - 1]

n = len(LOG)
col = lambda i: [row[i] for row in LOG]
success = sum(1 for row in LOG if row[0] == "completed") / n
print(f"task success     = {success:.2f}  ({int(success * n)}/{n} completed)")
print(f"steps per run    = {sum(col(1)) / n:.1f}")
print(f"tokens per run   = {sum(col(2)) / n:.0f}  -> ${sum(col(2)) / n * PRICE_PER_M / 1e6:.5f}/run")
print(f"latency p50/p95  = {percentile(col(3), 50):.0f} s / {percentile(col(3), 95):.0f} s")
print(f"tool error rate  = {sum(col(5)) / sum(col(4)):.3f}  ({sum(col(5))}/{sum(col(4))} calls)")
print(f"approvals/run    = {sum(col(6)) / n:.1f}")
print(f"override rate    = {sum(col(7)) / sum(col(6)):.3f}  ({sum(col(7))}/{sum(col(6))} denied)")
print(f"failure mix      = {sorted(set(col(0)) - {'completed'})}")
```

```text
task success     = 0.70  (7/10 completed)
steps per run    = 3.7
tokens per run   = 910  -> $0.00091/run
latency p50/p95  = 4 s / 60 s
tool error rate  = 0.069  (2/29 calls)
approvals/run    = 0.9
override rate    = 0.111  (1/9 denied)
failure mix      = ['cancelled', 'failed', 'timed_out']
```

The headline is the success rate, the share of runs that reached `completed`:

\[
\text{success} = \frac{\#\,\text{completed}}{\#\,\text{runs}}
\]

!!! example "On paper"

    Seven of the ten runs completed; the steps column sums to 37; the ten latencies sorted are 2, 3, 4, 4, 4, 5, 6, 9, 12, 60:

    \[
    \begin{aligned}
    \text{success} &= 7/10 = 0.70\\
    \text{steps per run} &= 37/10 = 3.7\\
    \text{p95 rank} &= \lceil 0.95 \cdot 10 \rceil = 10\\
    \text{p95} &= 60\ \text{s}\\
    \text{tool error rate} &= 2/29 \approx 0.069
    \end{aligned}
    \]

    A p50 of 4 s but a p95 of 60 s says the tail is the approval that nobody answered (run 9), not the model; the fix is an approval deadline and escalation (§7.2), not a faster model. "Completed" is not "correct" — pair this with the task-success eval of [52 §17](52-ai-agents.md#17-evaluating-agents) on a labelled sample.

Report the same numbers per tenant, per prompt version and per model, so a regression after a change is visible as a step in the series.

### 10.4 Alerts

Alert on rates over a window, never on single runs: success rate below the SLO; p95 latency over the budget; cost per run or per hour above a ceiling (a looping prompt shows up here first); tool error rate per tool above a threshold (a backend broke); approval wait p95 (the humans are the bottleneck); and the override rate — a jump means the agent started proposing actions the approvers reject, which is a prompt or tool regression caught by people, and belongs in the eval set ([55 §18.4](55-ai-production.md#184-the-data-flywheel)).

---

## 11. Cost control

### 11.1 Budgets per run and per tenant

The per-run budget (§9.5) caps the blast radius of one bad run; the **per-tenant** budget caps the sum. Keep a running monthly spend per tenant from the cost column of §4.4, refuse new runs at creation when it is reached (`429` with a reason, and an event to the account owner), and let the product sell more. A per-user daily budget on top stops one account from draining the tenant's allowance. All three are numbers in a table the runtime reads before `POST /runs` is accepted — not prompt text.

### 11.2 Cheap planning, capable execution

Not every step needs the capable model. Route by step: a small, cheap model classifies the request, picks the tools, and drafts the plan; the capable model takes the steps where the small one is unsure, and the final answer. This is the routing and cascade arithmetic of [55 §13](55-ai-production.md#13-routing-and-model-cascades) applied inside a run — the saving is the fraction of steps that never reach the big model, and it holds only if the escalation rule sends the hard steps up. Measure task success per route before trusting it ([54 — Evaluation](54-ai-evaluation.md)).

### 11.3 Cache the tool block, compact the results

Every step re-sends the system prompt and the tool schemas — identical bytes, often the largest part of the prompt ([58 §11.2](58-ai-mcp.md#112-the-token-cost-of-tool-definitions)). Prompt (prefix) caching ([55 §11.3](55-ai-production.md#113-prefix-prompt-caching), [50 §17](50-ai-prompting-apis.md#17-caching)) makes that prefix cheap on every step after the first; to benefit, keep the prefix byte-identical across steps and across runs (no per-run timestamp or user name at the top — put those at the end). Result compaction (§6.5) attacks the other term: the transcript that grows per step. Together they are the two levers in the model below.

### 11.4 A monthly cost model

State the assumptions, then compute. Prices are assumptions, not quotes: **$0.50 per million input tokens, $1.50 per million output tokens, $0.05 per million cached input tokens.** A run makes \(n = 4\) model calls; the fixed prefix \(P\) is 2,000 tokens; each step adds \(c = 1{,}500\) tokens of transcript (tool result plus model turn); each call produces \(O = 150\) output tokens. The history is re-sent every step, so the transcript term is the sum of [52 §13](52-ai-agents.md#13-cost-growth):

\[
C_\text{run} = \frac{p_\text{in}\left(c\,\tfrac{n(n+1)}{2}\right) + p_\text{prefix}\,P n + p_\text{out}\,O n}{10^{6}}
\]

!!! example "On paper"

    Uncached, the prefix is billed at the input price:

    \[
    \begin{aligned}
    c\,\tfrac{n(n+1)}{2} &= 1500 \cdot 10 = 15000\\
    P n &= 2000 \cdot 4 = 8000\\
    O n &= 150 \cdot 4 = 600\\
    0.50 \cdot 23000 + 1.50 \cdot 600 &= 11500 + 900 = 12400\\
    C_\text{run} &= 12400 / 10^{6} = \$0.0124
    \end{aligned}
    \]

    With the prefix cached at \$0.05 per million:

    \[
    \begin{aligned}
    0.50 \cdot 15000 + 0.05 \cdot 8000 + 900 &= 8800\\
    C_\text{run} &= \$0.0088
    \end{aligned}
    \]

    At 50,000 runs a month that is \$620 uncached and \$440 cached. Capping tool results so \(c\) drops to 900 brings it to \$290 — the transcript term, not the prefix, is where the money goes once caching is on.

```python
# Assumptions (state yours): prices per million tokens, and a run profile.
P_IN, P_OUT, P_CACHED = 0.50, 1.50, 0.05     # $/M input, output, cached input
RUNS = 50_000                                 # runs per month
PREFIX = 2_000                                # system prompt + tool schemas, identical every step
STEPS, OUT_PER_STEP = 4, 150                  # model calls per run, output tokens per call

def cost_per_run(growth, cached):
    """growth = transcript tokens added per step (tool result + model turn)."""
    resent = growth * STEPS * (STEPS + 1) // 2  # 52 §13: history is re-sent every step
    prefix = PREFIX * STEPS
    out = OUT_PER_STEP * STEPS
    dollars = resent * P_IN / 1e6 + out * P_OUT / 1e6
    dollars += prefix * (P_CACHED if cached else P_IN) / 1e6
    return resent + prefix, out, dollars

for label, growth, cached in [("baseline, no cache", 1500, False),
                              ("prefix cached", 1500, True),
                              ("cached + results capped", 900, True)]:
    inp, out, c = cost_per_run(growth, cached)
    print(f"{label:24} in={inp:6d} out={out:4d}  ${c:.4f}/run  ${c * RUNS:,.0f}/month")
```

```text
baseline, no cache       in= 23000 out= 600  $0.0124/run  $620/month
prefix cached            in= 23000 out= 600  $0.0088/run  $440/month
cached + results capped  in= 17000 out= 600  $0.0058/run  $290/month
```

Self-hosting changes the prices, not the shape: on the handbook's Mac the marginal token is free but the machine serves one run at a time, so the budget becomes seconds of machine time per run ([55 §12.2](55-ai-production.md#122-api-vs-self-host-break-even)). Re-run this model with your own numbers before every pricing or prompt change; the per-run figure is also the right default for `max_cost_usd` in §3.1 — set the ceiling at a few times the expected cost, so only a runaway trips it.

---

## 12. Testing and CI

### 12.1 Record and replay of model responses

Agent tests are slow and flaky if every test calls a model. Record the model's responses once, against the real endpoint, into a **cassette**; replay them in CI with no server. The recorder is the ordinary loop with a stubbed tool layer, writing each turn's content and tool calls:

```python
import json, os
from openai import OpenAI

client = OpenAI(base_url=os.environ.get("MODEL_URL", "http://127.0.0.1:8080/v1"),
                api_key="local")
MODEL = "mlx-community/Qwen3-14B-4bit"
TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters":
          {"type": "object", "properties": p, "required": list(p)}}} for n, d, p in [
    ("get_order", "Look up an order: status and amount.", {"order_id": {"type": "string"}}),
    ("issue_refund", "Refund an amount (USD) on an order.",
     {"order_id": {"type": "string"}, "amount": {"type": "number"}})]]
CANNED = {"get_order": {"status": "charged_twice", "amount": 20.0},
          "issue_refund": {"refunded": 20.0, "order_id": "A1001"}}   # tools stubbed while recording

messages = [{"role": "user", "content": "Refund the duplicate charge on order A1001."}]
cassette = []
for turn in range(6):
    r = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS, temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    msg = r.choices[0].message
    calls = [(tc.function.name, tc.function.arguments) for tc in msg.tool_calls or []]
    cassette.append({"content": msg.content or "", "tool_calls": calls,
                     "tokens": r.usage.total_tokens})
    if not msg.tool_calls:
        break
    messages.append({"role": "assistant", "content": msg.content,
                     "tool_calls": [tc.to_dict() for tc in msg.tool_calls]})
    for tc in msg.tool_calls:
        messages.append({"role": "tool", "tool_call_id": tc.id,
                         "content": json.dumps(CANNED[tc.function.name])})
json.dump(cassette, open("cassette.json", "w"), indent=1)
print(f"recorded {len(cassette)} turns, {sum(t['tokens'] for t in cassette)} tokens")
for t in cassette:
    print(" ", t["tool_calls"] or t["content"])
```

```text
recorded 3 turns, 701 tokens
  [('get_order', '{"order_id": "A1001"}')]
  [('issue_refund', '{"order_id": "A1001", "amount": 20.0}')]
  Refunded $20.00 on order A1001: the duplicate charge is reversed and one valid charge remains.
```

This recording came from the stub that imitates `mlx_lm.server` (set `MODEL_URL` to point the same code at a stub or a staging server; the default is the handbook's local server). A real Qwen3-14B would choose the calls itself — it may phrase the final sentence differently, call `get_order` with a different argument spelling, or ask a question instead of refunding — which is the point of recording: whatever it did becomes the fixture, and a *change* in what it does is what the next test detects. Record with `temperature=0` and the exact prompt and tool versions, and store the cassette next to the test with those versions in its name.

The replay side needs no server. It runs the same loop body against the cassette and checks the **trajectory** — the tool sequence — and the answer, which is the golden-trajectory test of §12.3:

```python
import json

CASSETTE = [                                   # pasted from the recording above
    {"content": "", "tool_calls": [("get_order", '{"order_id": "A1001"}')]},
    {"content": "", "tool_calls": [("issue_refund", '{"order_id": "A1001", "amount": 20.0}')]},
    {"content": "Refunded $20.00 on order A1001: the duplicate charge is reversed "
                "and one valid charge remains.", "tool_calls": []},
]
GOLDEN = {"tools": ["get_order", "issue_refund"], "answer_contains": "Refunded $20.00"}

def replay_model(cassette):
    turns = iter(cassette)
    return lambda messages: next(turns)         # no server: the i-th recorded turn

def run(model, max_steps=6):
    messages = [{"role": "user", "content": "Refund the duplicate charge on order A1001."}]
    trajectory = []
    for _ in range(max_steps):
        msg = model(messages)
        if not msg["tool_calls"]:
            return trajectory, msg["content"]
        for name, args in msg["tool_calls"]:
            trajectory.append(name)
            messages.append({"role": "tool", "content": json.dumps({"ok": True})})
    return trajectory, None

trajectory, answer = run(replay_model(CASSETTE))
print("trajectory:", trajectory, "| sequence ok:", trajectory == GOLDEN["tools"])
print("answer ok:", GOLDEN["answer_contains"] in (answer or ""))
# A regression cassette, as a new prompt version might produce: it refunds before looking.
regressed = [CASSETTE[1], CASSETTE[0], CASSETTE[2]]
t2, _ = run(replay_model(regressed))
print("regressed:", t2, "| sequence ok:", t2 == GOLDEN["tools"])
```

```text
trajectory: ['get_order', 'issue_refund'] | sequence ok: True
answer ok: True
regressed: ['issue_refund', 'get_order'] | sequence ok: False
```

A replayed cassette tests *your* code — the loop, the dispatcher, budgets, approvals, the state machine — deterministically and in milliseconds. It does not test the model: a new model or prompt needs a fresh recording, which is a deliberate, reviewed change to the fixture, exactly like updating a snapshot test.

### 12.2 Contract tests for every tool

Each tool wrapper gets the tests any API client gets, plus three agent-specific ones: **schema** (the JSON Schema you hand the model matches the pydantic model — generate one from the other so they cannot drift); **rejection** (extra fields, out-of-range values and wrong types return an actionable error, not an exception, as in §6.1); **cap** (a maximal real response stays under the result cap after field selection). Run them against the real backend in a staging environment nightly, and against a recorded backend on every commit.

### 12.3 Scenario tests with golden trajectories

A scenario is a task, a world (fixture data behind the tools), and an expected trajectory: the tool sequence (or a set of acceptable sequences), the arguments that matter, the approvals that must be requested, and a check on the final answer. The replay above is one scenario; a product has dozens — the happy path, a denied approval, an unknown order, a tool that errors once, a prompt-injected ticket body that must *not* produce a write. Grade trajectories, not just answers ([52 §17](52-ai-agents.md#17-evaluating-agents), [54 §13](54-ai-evaluation.md#13-evaluating-rag-and-agents)): the right answer by the wrong path fails on the next input.

### 12.4 Eval gates

Replay tests gate your code; evals gate the model and prompt. Keep a golden set of scenarios run against the live model on a schedule and before any prompt, tool or model change, with thresholds on task success, trajectory accuracy, zero out-of-scope writes, steps and cost per task, and with confidence intervals so a 1-point wobble does not block a release ([54 §10](54-ai-evaluation.md#10-evals-in-ci-as-a-release-gate), [54 §8.1](54-ai-evaluation.md#81-confidence-interval-for-a-pass-rate-wilson)). Failures found in production (§10.4) go into this set — the flywheel.

### 12.5 Load tests in tokens per second

An agent feature's load is not requests per second; it is **tokens per second**, in two kinds: prefill (the input the model must read, which the resend makes large) and decode (the output). Size the serving fleet from both ([55 §10](55-ai-production.md#10-sizing-hardware-for-a-target-load)) and load-test in those units:

\[
\text{tokens/s} = \frac{\text{runs per hour} \times \text{tokens per run}}{3600}
\]

!!! example "On paper"

    A peak of 1,800 runs an hour, each 600 output tokens and 23,000 input tokens (the uncached profile of §11.4):

    \[
    \begin{aligned}
    \text{decode} &= 1800 \cdot 600 / 3600 = 300\ \text{tok/s}\\
    \text{prefill} &= 1800 \cdot 23000 / 3600 = 11500\ \text{tok/s}
    \end{aligned}
    \]

    Decode at 300 tokens/s is twelve concurrent streams of 25 tokens/s on a batched GPU server, and far beyond one Mac, whose roofline for Qwen3-14B 4-bit is ~24 tokens/s and whose measured rate is ~16 ([55 §3](55-ai-production.md#3-the-decode-speed-bound)). Prefill at 11,500 tokens/s is the number that surprises teams: it is the resent transcript, and prefix caching removes 8,000 of every 23,000 tokens from it. Load-test the worst accepted run, not the average one.

---

## 13. Rollout and change management

### 13.1 Shadow → internal → canary → GA

Ship an agent change the way [55 §18.2](55-ai-production.md#182-rollouts) ships any model change, with two agent-specific additions: writes are disabled during shadow, and an internal stage sits before the canary. **Shadow:** run the new version on real inputs with **writes disabled** (every write tool in dry-run mode, §6.4), compare trajectories and proposed actions offline. **Internal:** your own team uses it with writes gated by approval; the override rate (§10.3) is the signal. **Canary:** 1–5% of tenants, chosen by a stable hash so a tenant sees one version consistently, with the success, cost and override dashboards split by version. **GA:** the flag goes to 100%, and the old version stays deployable for a week. Every stage has an exit criterion written down before it starts.

### 13.2 Versioning prompts and tools

A run records `prompt_version`, `tools_version` and `model` (§3.1, §4.4). Prompts live in files under version control with a changelog, not in a database field someone edits in production; the tool set — names, schemas, descriptions — is versioned as a unit, because changing a description changes behaviour. A model upgrade is a version too. Three versions on every run is what lets you say "success dropped 6 points on refund-v13" in one query.

### 13.3 The kill switch

A kill switch does two things, in this order: **refuse new runs** (the create endpoint returns a clear error and the UI hides the feature), and **cancel the running ones** (set the cancel flag on every non-terminal run, §9.6). It is a flag any on-call engineer can flip without a deploy, it is tested on a schedule like a fire alarm, and flipping it is a logged, reversible operation. A feature flag for the canary and the kill switch share the admission check:

```python
import threading, zlib

FLAGS = {"agent.enabled": True, "agent.canary_pct": 25, "agent.version": "v12"}
KILL = threading.Event()
RUNNING = {"run_0007": threading.Event(), "run_0008": threading.Event()}  # cancel flags

def admit(user_id):
    """Decide, before a run is created, whether and which version runs."""
    if KILL.is_set() or not FLAGS["agent.enabled"]:
        return "refused: agent disabled"
    bucket = zlib.crc32(user_id.encode()) % 100      # stable per user, not random
    return FLAGS["agent.version"] if bucket < FLAGS["agent.canary_pct"] else "v11"

def kill_switch():
    KILL.set()                                      # 1. no new runs
    for rid, cancel in RUNNING.items():             # 2. cancel the ones in flight
        cancel.set()
    return [rid for rid, c in RUNNING.items() if c.is_set()]

for uid in ("u_1", "u_2", "u_3", "u_4", "u_5"):
    print(f"{uid} -> {admit(uid)}")
print("kill switch pulled; cancelled:", kill_switch())
print("u_1 ->", admit("u_1"))
```

```text
u_1 -> v12
u_2 -> v11
u_3 -> v11
u_4 -> v11
u_5 -> v12
kill switch pulled; cancelled: ['run_0007', 'run_0008']
u_1 -> refused: agent disabled
```

The hash bucket keeps a user on one version across runs (two of these five users fall under 25); the switch refuses the next admission and cancels both in-flight runs. In production the flag lives in your feature-flag service and the cancel fans out through the run store, but the two steps are the same.

### 13.4 In-flight runs on deploy

A deploy restarts workers while runs are mid-step. Three rules make that safe. **Drain:** a worker that receives the shutdown signal finishes its current step, checkpoints, and stops taking new runs; the orchestrator waits for a bounded drain time. **Requeue:** runs still `running` after the drain (the worker died) are picked up by the reaper (§3.4) from the last checkpoint — safe because tool calls are idempotent (§4.3). **Pin versions:** a resumed run continues with the prompt and tool versions recorded on it, not with the new deploy's; if the old version is gone, the run fails with a clear reason rather than continuing with a transcript built under different instructions. Runs waiting for approval hold no worker and are unaffected.

---

## 14. A minimal agent service

Everything above, as a service you can run: `POST /runs` creates a run and returns its id; `GET /runs/{id}` returns the record; `GET /runs/{id}/events` streams events as SSE; `POST /runs/{id}/approve` resumes a run waiting on a write tool; `POST /runs/{id}/cancel` stops one. Budgets (steps, tokens, wall time, approval deadline) are enforced by the service. The model is the handbook's local `mlx_lm.server` through the `openai` client; the tests pointed the same code at a stub that imitates it (`MODEL_URL`), and the outputs below are from that stub. Only the standard library and the `openai` client are used.

### 14.1 The worker loop with budgets and approvals

The service's core, in isolation, against a scripted model, so the budgets can be seen tripping and both approval outcomes are visible. The loop is [52 §6](52-ai-agents.md#6-a-complete-agent-loop) with an exit per budget and a pause for approval.

```python
import queue, time

def get_order(order_id):
    return {"status": "charged_twice", "amount": 20.0}
def issue_refund(order_id, amount):
    return {"refunded": amount}
TOOLS = {"get_order": (get_order, "read"), "issue_refund": (issue_refund, "write")}

def execute(model, approvals, budget):
    """Loop until the model answers or a budget trips; a write tool pauses for approval."""
    run, t0 = {"obs": [], "tokens": 0, "steps": 0}, time.monotonic()
    for step in range(1, budget["max_steps"] + 1):
        if time.monotonic() - t0 > budget["max_seconds"]:
            return "timed_out", "wall time", run
        act = model(run["obs"])                   # a tool call or a final answer
        run["steps"], run["tokens"] = step, run["tokens"] + act["tokens"]
        if run["tokens"] > budget["max_tokens"]:
            return "failed", "token budget", run
        if "answer" in act:
            return "completed", act["answer"], run
        fn, kind = TOOLS[act["tool"]]
        if kind == "write":                       # pause: the run is now waiting_approval
            try:
                ok = approvals.get(timeout=budget["approval_seconds"])
            except queue.Empty:
                return "timed_out", f"no decision on {act['tool']}", run
            if not ok:
                run["obs"].append({"denied": act["tool"]})
                continue
        run["obs"].append(fn(**act["args"]))
    return "failed", "step budget", run

def refund_model(obs):                            # scripted stand-in for the model
    if len(obs) == 0:
        return {"tool": "get_order", "args": {"order_id": "A1001"}, "tokens": 300}
    if len(obs) == 1:
        return {"tool": "issue_refund", "args": {"order_id": "A1001", "amount": 20.0}, "tokens": 350}
    done = "denied" not in obs[-1]
    return {"answer": "Refunded $20.00 on A1001." if done else "Not approved; nothing changed.",
            "tokens": 400}
def looping_model(obs):
    return {"tool": "get_order", "args": {"order_id": "A1001"}, "tokens": 300}

BUDGET = {"max_steps": 5, "max_tokens": 10000, "max_seconds": 5, "approval_seconds": 0.2}
def decisions(*values):
    q = queue.Queue()
    for v in values:
        q.put(v)
    return q
for label, model, approvals in [("approved", refund_model, decisions(True)),
                                ("denied", refund_model, decisions(False)),
                                ("no decision", refund_model, decisions()),
                                ("looping", looping_model, decisions())]:
    state, detail, run = execute(model, approvals, BUDGET)
    print(f"{label:12} -> {state:10} steps={run['steps']} tokens={run['tokens']:5}  {detail}")
state, detail, run = execute(refund_model, decisions(True), {**BUDGET, "max_tokens": 1000})
print(f"{'over tokens':12} -> {state:10} steps={run['steps']} tokens={run['tokens']:5}  {detail}")
```

```text
approved     -> completed  steps=3 tokens= 1050  Refunded $20.00 on A1001.
denied       -> completed  steps=3 tokens= 1050  Not approved; nothing changed.
no decision  -> timed_out  steps=2 tokens=  650  no decision on issue_refund
looping      -> failed     steps=5 tokens= 1500  step budget
over tokens  -> failed     steps=3 tokens= 1050  token budget
```

The approval is a queue the loop blocks on with a deadline — in the service it is fed by an HTTP endpoint, here by a preloaded decision. A denial is not an exception: it becomes an observation, and the model finishes the run with a sensible sentence. The looping model exhausts the step budget and the run fails *with the reason*, which is the difference between a failed run and a bill. The token budget trips the same way: the third call takes the count to 1,050 against a ceiling of 1,000, and the run fails right after that call instead of making another.

### 14.2 The service

The complete file. Save it as `agent_service.py` and run it with the local model server up (or with `MODEL_URL` pointing elsewhere). It is longer than the other blocks because it is the whole thing: tools, the run store, the state machine, the worker pool, the model call, and the five endpoints. Read it top to bottom once; each part is the section it implements.

```python
import json, os, queue, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from openai import OpenAI

# agent_service.py -- a minimal agent service: POST /runs, GET /runs/{id},
# GET /runs/{id}/events (SSE), POST /runs/{id}/approve, POST /runs/{id}/cancel.
client = OpenAI(base_url=os.environ.get("MODEL_URL", "http://127.0.0.1:8080/v1"),
                api_key="local")
MODEL = "mlx-community/Qwen3-14B-4bit"
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
BUDGET = {"max_steps": 6, "max_tokens": 20000, "max_seconds": 60, "approval_seconds": 120}
PORT = 18741

ORDERS = {"A1001": {"status": "charged_twice", "amount": 20.0}}
def get_order(order_id):
    return ORDERS.get(order_id) or {"error": f"unknown order {order_id}"}
def issue_refund(order_id, amount):
    return {"refunded": amount, "order_id": order_id}
TOOLS = {"get_order": (get_order, "read"), "issue_refund": (issue_refund, "write")}
SCHEMAS = [{"type": "function", "function": {"name": n, "description": d, "parameters":
            {"type": "object", "properties": p, "required": list(p)}}} for n, d, p in [
    ("get_order", "Look up an order: status and amount.", {"order_id": {"type": "string"}}),
    ("issue_refund", "Refund an amount (USD) on an order. Needs approval.",
     {"order_id": {"type": "string"}, "amount": {"type": "number"}})]]

RUNS, QUEUE, COND = {}, queue.Queue(), threading.Condition()
NEXT = {"queued": {"running", "cancelled"},
        "running": {"waiting_approval", "completed", "failed", "cancelled", "timed_out"},
        "waiting_approval": {"running", "cancelled", "timed_out"}}
TERMINAL = {"completed", "failed", "cancelled", "timed_out"}
EVENT = {"running": "run.started", "waiting_approval": "approval.required",
         "completed": "run.completed", "failed": "run.failed",
         "cancelled": "run.cancelled", "timed_out": "run.timed_out"}

def emit(run, event, state=None, **data):
    with COND:                                  # state change + event are atomic
        if state:
            run["state"] = state
        run["events"].append({"id": len(run["events"]) + 1, "event": event, **data})
        COND.notify_all()                       # wakes every SSE streamer

def transition(run, state, event=None, **fields):
    assert state in NEXT[run["state"]], f"illegal {run['state']} -> {state}"
    run.update(fields)
    emit(run, event or EVENT[state], state=state, **fields)
    print(f"{run['id']} -> {state}", flush=True)

def call_model(run):
    tools = [s for s in SCHEMAS if s["function"]["name"] in run["allowed_tools"]]
    r = client.chat.completions.create(model=MODEL, messages=run["messages"], tools=tools,
                                       temperature=0, timeout=30, extra_body=NO_THINK)
    run["tokens"] += r.usage.total_tokens
    return r.choices[0].message

def run_tool(run, step, name, args):
    emit(run, "tool.called", step=step, tool=name, args=args)
    if name not in run["allowed_tools"]:
        return {"error": f"tool not allowed for this run: {name}"}
    fn, kind = TOOLS[name]
    if kind == "write":                         # pause until a human decides
        transition(run, "waiting_approval", pending={"tool": name, "args": args})
        try:
            ok = run["approval"].get(timeout=run["budget"]["approval_seconds"])
        except queue.Empty:
            return transition(run, "timed_out", error="approval timed out")
        if run["cancel"].is_set():
            return transition(run, "cancelled")
        transition(run, "running", event="run.resumed", pending=None, approved=ok)
        if not ok:
            return {"denied": "the write was not approved; do not retry it"}
    return fn(**args)

def execute(run):
    b = run["budget"]
    for step in range(1, b["max_steps"] + 1):
        if run["cancel"].is_set():
            return transition(run, "cancelled")
        if time.time() - run["t0"] > b["max_seconds"]:
            return transition(run, "timed_out", error="wall time exceeded")
        if run["tokens"] > b["max_tokens"]:
            return transition(run, "failed", error="token budget exceeded")
        msg = call_model(run)
        run["steps"] = step
        if not msg.tool_calls:                  # the model answered: done
            emit(run, "message.delta", text=msg.content or "")
            return transition(run, "completed", result=msg.content or "")
        run["messages"].append({"role": "assistant", "content": msg.content,
                                "tool_calls": [tc.to_dict() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            out = run_tool(run, step, tc.function.name, json.loads(tc.function.arguments))
            if run["state"] in TERMINAL:
                return
            emit(run, "tool.result", step=step, tool=tc.function.name, result=out)
            run["messages"].append({"role": "tool", "tool_call_id": tc.id,
                                    "content": json.dumps(out)})
    transition(run, "failed", error="step budget exceeded")

def worker():
    while True:
        run = RUNS[QUEUE.get()]
        if run["cancel"].is_set():
            transition(run, "cancelled")
            continue
        transition(run, "running")
        try:
            execute(run)
        except Exception as e:                  # a crash is a failed run, not a dead worker
            transition(run, "failed", error=f"{type(e).__name__}: {e}")

class Api(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
    def do_POST(self):
        parts = self.path.strip("/").split("/")
        if parts == ["runs"]:
            req = self.body()
            key = req.get("idempotency_key")
            with COND:
                dup = next((r for r in RUNS.values() if key and r["key"] == key), None)
                if dup:                         # same key: same run, no second execution
                    return self.send(200, {"run_id": dup["id"], "state": dup["state"]})
                rid = f"run_{len(RUNS) + 1:04d}"
                RUNS[rid] = {"id": rid, "state": "queued", "user": req["user"], "key": key,
                             "allowed_tools": req.get("allowed_tools", list(TOOLS)),
                             "budget": {**BUDGET, **req.get("constraints", {})},
                             "steps": 0, "tokens": 0, "pending": None, "result": None,
                             "error": None, "events": [], "t0": time.time(),
                             "messages": [{"role": "user", "content": req["input"]}],
                             "cancel": threading.Event(), "approval": queue.Queue()}
            QUEUE.put(rid)
            return self.send(202, {"run_id": rid, "state": "queued"})
        run = RUNS.get(parts[1]) if len(parts) == 3 and parts[0] == "runs" else None
        if run is None:
            return self.send(404, {"error": "no such run"})
        if parts[2] == "approve":
            decision = bool(self.body().get("approved"))
            with COND:                          # one decision per pending approval
                accepted = run["state"] == "waiting_approval" and run["approval"].empty()
                if accepted:
                    run["approval"].put(decision)
            if accepted:
                return self.send(202, {"run_id": run["id"], "state": run["state"]})
            if run["state"] == "waiting_approval":
                return self.send(409, {"error": "a decision for this approval is already queued"})
        if parts[2] == "cancel" and run["state"] not in TERMINAL:
            run["cancel"].set()
            run["approval"].put(False)          # unblock a run waiting on approval
            return self.send(202, {"run_id": run["id"], "state": run["state"]})
        self.send(409, {"error": f"cannot {parts[2]} a run in state {run['state']}"})
    def do_GET(self):
        parts = self.path.strip("/").split("/")
        run = RUNS.get(parts[1]) if len(parts) >= 2 and parts[0] == "runs" else None
        if run is None:
            return self.send(404, {"error": "no such run"})
        if len(parts) == 2:
            return self.send(200, {k: run[k] for k in ("id", "state", "user", "steps",
                                                       "tokens", "pending", "result", "error")})
        self.send_response(200)                 # /runs/{id}/events: Server-Sent Events
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        last = int(self.headers.get("Last-Event-ID", 0))
        while True:
            with COND:
                COND.wait_for(lambda: len(run["events"]) > last or run["state"] in TERMINAL,
                              timeout=5)
                new, done = run["events"][last:], run["state"] in TERMINAL
            for ev in new:
                self.wfile.write(f"id: {ev['id']}\nevent: {ev['event']}\n"
                                 f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()
            last += len(new)
            if done:
                return

for _ in range(2):
    threading.Thread(target=worker, daemon=True).start()
print(f"agent service on 127.0.0.1:{PORT}", flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), Api).serve_forever()
```

What it printed while the client session of §14.3 ran against it:

```text
agent service on 127.0.0.1:18741
run_0001 -> running
run_0001 -> waiting_approval
run_0001 -> running
run_0001 -> completed
```

Where each section of the chapter lives in the file: the run contract and state machine (§3) are `NEXT`, `EVENT`, `emit` and `transition` — the state change and its event are written under one lock so a streamer never sees one without the other; the per-run allow-list (§5.2) is `allowed_tools`, applied both to the schemas the model sees and to dispatch; the read/write split and the approval pause (§6.4, §7) are `run_tool` and the `approve` branch of `do_POST`, which takes one decision per pending approval under the same lock, so a double-clicked button cannot pre-approve the run's next write; the budgets (§9.5) are the four checks at the top of `execute`; cancellation (§9.6) is the flag checked at step boundaries and inside the approval wait; the idempotent create (§4.3) is the `key` lookup in `do_POST`; and reconnect (§8.4) is the `Last-Event-ID` filter in `do_GET`. What a production version adds is a database behind `RUNS` (with `transition` as a conditional `UPDATE`), a worker that is released during `waiting_approval` and a run that is re-enqueued by the approval (here the worker blocks on `run["approval"]`, §7.2), the trace and audit log, per-tenant budgets, scoped credentials in the tool wrappers, and real identity on the request instead of a trusted `user` field.

### 14.3 A client session

A client that creates a run, streams its events, approves the write when asked, reads the record, retries the create with the same idempotency key, and reconnects from the middle of the stream:

```python
import json, urllib.request

BASE = "http://127.0.0.1:18741"

def call(method, path, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req) as r:
        return json.load(r)

def stream(path, headers=None, on_event=lambda ev: None):
    req = urllib.request.Request(BASE + path, headers=headers or {})
    with urllib.request.urlopen(req) as r:
        for raw in r:                               # one SSE line at a time
            line = raw.decode().rstrip("\n")
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                extra = {k: v for k, v in ev.items() if k not in ("id", "event", "step")}
                print(f"  {ev['id']:>2} {ev['event']:<18} {extra or ''}".rstrip())
                on_event(ev)

req = {"user": "u_42", "input": "Refund the duplicate charge on order A1001.",
       "idempotency_key": "ticket-7781"}
created = call("POST", "/runs", req)
print("created:", created)
rid = created["run_id"]

def approve_writes(ev):
    if ev["event"] == "approval.required":
        print("approve:", call("POST", f"/runs/{rid}/approve", {"approved": True}))

stream(f"/runs/{rid}/events", on_event=approve_writes)
print("record:", call("GET", f"/runs/{rid}"))
print("same key again:", call("POST", "/runs", req))
print("reconnect from event 7:")
stream(f"/runs/{rid}/events", headers={"Last-Event-ID": "7"})
```

```text
created: {'run_id': 'run_0001', 'state': 'queued'}
   1 run.started
   2 tool.called        {'tool': 'get_order', 'args': {'order_id': 'A1001'}}
   3 tool.result        {'tool': 'get_order', 'result': {'status': 'charged_twice', 'amount': 20.0}}
   4 tool.called        {'tool': 'issue_refund', 'args': {'order_id': 'A1001', 'amount': 20.0}}
   5 approval.required  {'pending': {'tool': 'issue_refund', 'args': {'order_id': 'A1001', 'amount': 20.0}}}
approve: {'run_id': 'run_0001', 'state': 'waiting_approval'}
   6 run.resumed        {'pending': None, 'approved': True}
   7 tool.result        {'tool': 'issue_refund', 'result': {'refunded': 20.0, 'order_id': 'A1001'}}
   8 message.delta      {'text': 'Refunded $20.00 on order A1001: the duplicate charge is reversed and one valid charge remains.'}
   9 run.completed      {'result': 'Refunded $20.00 on order A1001: the duplicate charge is reversed and one valid charge remains.'}
record: {'id': 'run_0001', 'state': 'completed', 'user': 'u_42', 'steps': 3, 'tokens': 713, 'pending': None, 'result': 'Refunded $20.00 on order A1001: the duplicate charge is reversed and one valid charge remains.', 'error': None}
same key again: {'run_id': 'run_0001', 'state': 'completed'}
reconnect from event 7:
   8 message.delta      {'text': 'Refunded $20.00 on order A1001: the duplicate charge is reversed and one valid charge remains.'}
   9 run.completed      {'result': 'Refunded $20.00 on order A1001: the duplicate charge is reversed and one valid charge remains.'}
```

Read it against the state machine: the run started, read the order, proposed the refund (event 4 is emitted *before* anything runs), paused (event 5, the run is `waiting_approval` and the worker is blocked on the queue), resumed on the approval (event 6), executed the write, streamed the answer and completed. The record shows the three model calls and the 713 tokens they consumed, which the token budget was checked against. The second create with the same key returned the same run without running anything; the reconnect with `Last-Event-ID: 7` replayed only events 8 and 9.

### 14.4 Budgets, scopes and cancellation from the client

Three more runs against the same service, started right after the session above: one whose approval nobody answers within its one-second budget, one for a user whose scope has no write tool, and one the user cancels while it waits for approval.

```python
import json, time, urllib.request

BASE = "http://127.0.0.1:18741"
TASK = "Refund the duplicate charge on order A1001."

def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)

def wait_for(rid, states):
    while True:
        rec = call("GET", f"/runs/{rid}")
        if rec["state"] in states:
            return rec
        time.sleep(0.05)

DONE = {"completed", "failed", "cancelled", "timed_out"}
# 1. nobody answers the approval within this run's budget
a = call("POST", "/runs", {"user": "u_42", "input": TASK,
                           "constraints": {"approval_seconds": 1}})["run_id"]
# 2. a user whose scope has no write tool
b = call("POST", "/runs", {"user": "u_99", "input": TASK,
                           "allowed_tools": ["get_order"]})["run_id"]
# 3. the user cancels while the run waits for approval
c = call("POST", "/runs", {"user": "u_42", "input": TASK})["run_id"]
wait_for(c, {"waiting_approval"})
print("cancel:", call("POST", f"/runs/{c}/cancel"))
for rid in (a, b, c):
    rec = wait_for(rid, DONE)
    print(f"{rid}: {rec['state']:<10} steps={rec['steps']} "
          f"error={rec['error']!r} result={rec['result']!r}")
```

```text
cancel: {'run_id': 'run_0004', 'state': 'waiting_approval'}
run_0002: timed_out  steps=2 error='approval timed out' result=None
run_0003: completed  steps=2 error=None result='Order A1001 was charged twice, but this run has no refund tool; please escalate to billing.'
run_0004: cancelled  steps=2 error=None result=None
```

Run 2 timed out at its approval deadline with the reason on the record; run 3 never saw `issue_refund` — the model was offered one tool and answered accordingly (the stub's canned sentence; a real model would also have to stop at "no refund tool" because the schema is simply absent); run 4 went from `waiting_approval` to `cancelled` without the write executing. All three ended in a terminal state with the worker released, which is the property the whole chapter is about.

### 14.5 What a real model changes

Against a real Qwen3-14B on `mlx_lm.server` the service code is unchanged; the *behaviour* in the stream differs in three ways. The model chooses the calls — it may call `get_order` with the id spelled differently, add a second lookup, or answer a clarifying question instead of refunding (a completed run whose result is a question — the UI must handle it). Token counts are real: the tool schemas alone are a few hundred tokens of prefix on every step ([58 §11.2](58-ai-mcp.md#112-the-token-cost-of-tool-definitions)), so `tokens` on the record lands in the low thousands, and the per-step latency is seconds, not milliseconds — which is why `approval.required` and `tool.called` must stream immediately. And thinking: the service sends `enable_thinking: false` on every call (the stub logged it on each request), so `content` is clean; the `content or ""` guards remain because a tool-call turn carries `content: None` on every server, and because an adapter trained with thinking on is served with it on ([06 §6](06-data-format.md#6-thinking-and-reasoning_content)).

---

## 15. Integration checklist

- [ ] A shape chosen from §2.9 by latency, duration, approvals, volume and side effects — the simplest that fits.
- [ ] A written run contract: request, record, events, and the state machine with its transition rules.
- [ ] Runs, steps, tool calls and approvals in tables; checkpoint after every step; a reaper for stale runs.
- [ ] Idempotency keys on run creation and on every write tool call.
- [ ] The agent acts as the user: per-tenant and per-user tool allow-lists computed before the first model call; scoped, short-lived tool credentials; no shared superuser.
- [ ] An append-only audit log of side effects with user, run, step, tool, arguments, decision and approver.
- [ ] Tools are typed wrappers: strict validation in, selected and size-capped results out, actionable errors; reads and writes classified; dry-run for writes.
- [ ] Approval cards show the exact action, a preview, and the evidence; deadlines, escalation and per-action policy for irreversible writes.
- [ ] SSE events with ids; no raw reasoning to end users; waiting states; terminal events; reconnect with `Last-Event-ID`.
- [ ] Per-step and per-run timeouts that agree; retries only for detectable, idempotent failures; a breaker per dependency; a fallback ladder.
- [ ] Step, token, time and money budgets enforced by the worker; per-tenant budgets at admission; cancellation that reaches tools.
- [ ] One trace per run with spans per step and tool call; structured logs with hashes not content; success, steps, cost, p50/p95, approval and override rates, tool error rates — per version.
- [ ] A monthly cost model with your prices; prefix caching with a stable prefix; result compaction.
- [ ] Record/replay fixtures, tool contract tests, golden-trajectory scenarios, and an eval gate with thresholds and intervals in CI; load tests in tokens per second.
- [ ] Shadow with writes disabled, then internal, canary by stable hash, GA; prompt, tool and model versions on every run; a tested kill switch; a drain-and-requeue deploy.
- [ ] Transcript retention in days, metadata in months, audit log longest; PII redacted before storage; tenant opt-out.

---

## 16. Key takeaways

- An agent in a product is a **job**: a run with an id, a state machine, a budget, a transcript that is checkpointed, and an audit trail. Design the job before the prompt.
- Pick the integration **shape** from the requirements — in-process for short read-only steps, a run service for everything else, streaming on top when a user is present, a queue or scheduler when none is, a durable engine for hours-long runs with approvals, agent-as-tool only with an eval to justify it, a local harness when the data is local.
- The **run contract** (request, record, events, states) is the interface every other concern hangs on; the state machine's transitions are enforced by the runtime, and terminal states are final.
- Separate **conversation state, run state and long-term memory**; a new turn is a new run.
- Two **idempotency keys**: one for creating the run, one per side-effecting tool call. Retries and resumes are safe only with both.
- The agent acts **as the user**, with per-tenant and per-user tool lists computed before the first model call and short-lived scoped credentials per tool — never a shared superuser.
- Tools are **typed wrappers**: validate in, select and cap out, actionable errors, reads and writes classified, dry-run and confirmation for writes; the cap follows from \(r_\text{max} = (W - T_\text{sys} - nT_\text{turn})/n\).
- **Approvals** show the exact action with a preview and evidence, expire into `timed_out`, escalate, and resume as a tool result the model reads; per-action for irreversible writes.
- **Stream** events with ids, never raw reasoning; a close without a terminal event is a disconnect, and `Last-Event-ID` resumes it.
- **Reliability** is the standard kit attached to a run: consistent per-step and per-run timeouts, retries only for detectable idempotent failures, a breaker per dependency, a fallback ladder, budgets in the worker, cancellation that reaches tools.
- **Observe** one trace per run and six metrics per version — success, steps, cost, p50/p95, approval and override rates, tool errors — and alert on rates.
- **Cost** is the resent transcript: cache the prefix, cap the results, budget per run and per tenant, and re-run the monthly model with your own prices.
- **Test** your code with recorded cassettes and golden trajectories, your tools with contracts, your model and prompt with eval gates; load-test in tokens per second.
- **Ship** shadow → internal → canary → GA with versions on every run, a tested kill switch that refuses new runs and cancels running ones, and a deploy that drains, requeues from checkpoints and pins versions.

## 17. Exercises

!!! question "Exercise 17.1"

    Pick a shape from §2 for each: (a) every night, re-categorize 50,000 tickets after a taxonomy change; (b) a support agent chats with the assistant and must approve refunds; (c) a "pull request opened" webhook should trigger a review that posts comments as suggestions.

??? success "Answer"

    (a) Scheduled/batch (§2.5): throughput over latency, embarrassingly parallel, idempotent per item, run a sample first. (b) A chat UI with streaming and visible tool calls on top of a run service (§2.3 over §2.2): the user is present and approvals need a UI. (c) An event-driven queue worker (§2.4): no user is waiting, the event starts the run, and with nobody present the writes stay at the "suggest" level — comments, not merges — with idempotent consumption keyed on the webhook delivery id.

!!! question "Exercise 17.2"

    A 16,384-token window, a 1,500-token prefix, 8 steps, 150 tokens of model output per step. What is the largest tool result you can allow per step?

??? success "Answer"

    \(r_\text{max} = (16384 - 1500 - 8\cdot 150)/8 = (16384 - 2700)/8 = 13684/8 = 1710.5\), so 1,710 tokens — and in practice a cap around 600–800 to leave room for the final answer and a summary.

!!! question "Exercise 17.3"

    A client's network is flaky: it sends `POST /runs` with the same idempotency key three times. How many runs execute? A worker crashes right after a refund tool returned but before the checkpoint was written; on resume, what prevents a second refund?

??? success "Answer"

    One run executes; the second and third creates return the existing run's id and state (§4.3, and `same key again` in §14.3). The tool-call idempotency key `run_id:step:tool_call_id`: the wrapper (or the downstream API) stored the first result under it, so the replayed call returns the stored result instead of refunding again. Without that key, resuming from the previous checkpoint would re-run the refund.

!!! question "Exercise 17.4"

    5,000 runs a day, 6 steps each, 2 KB per checkpoint, kept 30 days. How much transcript storage is that?

??? success "Answer"

    Per day \(5000 \cdot 6 \cdot 2\ \text{KB} = 60{,}000\ \text{KB} = 60\ \text{MB}\); over 30 days \(60 \cdot 30 = 1{,}800\ \text{MB}\), about 1.8 GB. Small enough that nobody will prune it unless `expires_at` does it mechanically.

!!! question "Exercise 17.5"

    The step budget is 8, the model timeout 15 s and the tool timeout 5 s. What run deadline is consistent with them? The product can only wait 90 s; what per-step time does that leave?

??? success "Answer"

    \(T_\text{run} \ge 8 \cdot (15 + 5) = 160\) s. With a 90 s run deadline each step gets \(90/8 = 11.25\) s for model plus tool, so the per-step timeouts must shrink (say 8 s model, 3 s tool) or the step budget must drop — otherwise runs that respect every per-step limit still time out.

!!! question "Exercise 17.6"

    From a log of 20 runs: 16 completed, 12 approvals requested of which 3 were denied, 50 tool calls of which 4 errored. Compute the success rate, the override rate and the tool error rate. Which one points at a prompt regression caught by humans?

??? success "Answer"

    Success \(16/20 = 0.80\); override rate \(3/12 = 0.25\); tool error rate \(4/50 = 0.08\). The override rate: a quarter of proposed writes rejected by approvers means the agent is proposing actions people do not want — a prompt or tool-description regression, and each denial is a new eval case (§10.4).

!!! question "Exercise 17.7"

    10,000 runs a month; 5 model calls per run; 1,000 tokens of transcript growth per step; a 1,500-token prefix; 120 output tokens per call. Prices as assumed in §11.4 (\$0.50 / \$1.50 per million, cached input \$0.05). Monthly cost with and without prefix caching?

??? success "Answer"

    Resent transcript \(1000 \cdot \tfrac{5\cdot 6}{2} = 15{,}000\); prefix \(1500 \cdot 5 = 7{,}500\); output \(120 \cdot 5 = 600\). Uncached: \((0.50 \cdot 22500 + 1.50 \cdot 600)/10^6 = (11250 + 900)/10^6 = \$0.01215\) per run, \$121.50 a month. Cached: \((0.50 \cdot 15000 + 0.05 \cdot 7500 + 900)/10^6 = (7500 + 375 + 900)/10^6 = \$0.008775\) per run, \$87.75 a month. Caching saves about 28%; capping the transcript growth would save more.

!!! question "Exercise 17.8"

    A reviewer proposes letting an approval that arrives after the deadline still execute the write, "since the human did approve it". Which transition does that need, and why is it rejected?

??? success "Answer"

    It needs `timed_out → running` (or a write from a terminal state), which the state machine (§3.4) does not allow: terminal states are final. The run already ended with a reason the user saw; the world may have changed (a human may have refunded by hand); and an action executing minutes or hours after its context was assembled is exactly the stale-action bug approvals exist to prevent. The right path is a new run, with a fresh lookup and a fresh approval.

!!! question "Exercise 17.9"

    Peak load is 3,600 runs an hour; a run averages 800 output tokens and 20,000 input tokens (with the resend). What decode and prefill throughput must the load test sustain?

??? success "Answer"

    Decode \(3600 \cdot 800 / 3600 = 800\) tokens/s; prefill \(3600 \cdot 20000 / 3600 = 20{,}000\) tokens/s. One run per second sounds small; 20,000 prefill tokens per second is the number to size for — and the one prefix caching and result capping shrink.

Next: [63 — Agents in the real world: twelve applications](63-ai-agent-applications.md) applies these shapes end to end; [61 — Agent frameworks](61-ai-agent-frameworks.md) for the libraries that package the run store, checkpointer and streaming; [55 — Production](55-ai-production.md) for the serving stack underneath.
