# 63 — Agents in the real world: twelve applications

*Last verified: 29 October 2026.*

Three earlier chapters gave you the parts: the agent loop and its disciplines ([52 — Agents](52-ai-agents.md)), the frameworks that wrap it ([61 — Agent frameworks](61-ai-agent-frameworks.md)), and the shapes that turn an agent into a product feature ([62 — Integrating agents into applications](62-ai-agent-integration.md)). This chapter spends them. It walks through **twelve concrete applications**, each end to end and in the *same* order so you can compare them: the job, why an agent (or why only partly), the architecture, the tools, the data, the autonomy and guardrails, a sample trajectory, the evals, the cost per task, the failure modes, and how to grow it from an MVP. Read 52 first; this chapter does not re-derive the loop, the reliability math, the budgets or the autonomy ladder — it links them and applies them.

The recurring lesson, stated twelve ways: **the model picks the path; your code owns the consequences.** Every application below is read-mostly tools, approval gates on writes, hard budgets, a verification step, and an eval you trust *before* you turn autonomy up. Keep every scenario benign and defensive; the one security application is authorized lab testing only, exactly as [11 — The Kali agent](11-kali-agent.md) frames it.

---

## 1. How to read the twelve

Each application repeats the same twelve-part template. Three conventions make the cards comparable:

- **Prices are assumptions, not quotes.** Throughout, assume a mid-tier model at **\$0.50 per 1M input tokens** and **\$1.50 per 1M output tokens**, and a cheap model (for cascades) at **\$0.10 / \$0.40**. The handbook never quotes live API prices; state your own and the math stays true ([55 §12](55-ai-production.md#12-cost-modeling)). The local Qwen3-14B-4bit path costs amortized compute, not per-token — where an application runs locally the card gives the token-equivalent so you can compare.
- **"Integration shape" names one of the eight from [62](62-ai-agent-integration.md)**: in-process call, agent service with run ids, streaming chat UI, queue worker, scheduled/batch, durable workflow engine, agent-as-tool, local harness. Each application names the shape it uses rather than redescribing it.
- **"Autonomy" is the level from [52 §8](52-ai-agents.md#8-autonomous-agents)**: suggest, act with approval, act and report, or fully autonomous. No application here is fully autonomous on a write; that is the point.

Two levers cut cost in almost every application, worked once here and referenced throughout. **A cascade** sends the easy share to the cheap model and escalates the rest ([55 §13](55-ai-production.md#13-routing-and-model-cascades)):

\[
E[\text{cost}] = c_\text{small} + f\,c_\text{large}
\]

!!! example "On paper"

    Cheap model \(c_\text{small}=0.0005\), mid model \(c_\text{large}=0.004\), escalate a fraction \(f=0.3\):

    \[
    E[\text{cost}] = 0.0005 + 0.3\times 0.004
    \]

    \[
    E[\text{cost}] = 0.0005 + 0.0012 = 0.0017
    \]

    \$0.0017 against \$0.004 for mid-only: a **57.5% cut**, because only 30% of tasks pay the mid rate.

**Prompt caching** the stable prefix (system prompt plus tool schemas, identical every step) means the steps after the first pay only a reduced cached rate for it — a fraction of the input price on hosted APIs, and skipped prefill on the local server ([55 §11.3](55-ai-production.md#113-prefix-prompt-caching)). Counting full-price input tokens over \(s\) steps:

\[
T_\text{billed} = T_\text{prefix} + s\,T_\text{new}
\]

!!! example "On paper"

    A 1200-token prefix, \(s=4\) steps, each step sending the prefix plus \(T_\text{new}=700\) tokens of its own; the prefix is cached after step 1:

    \[
    T_\text{uncached} = 4\times(1200+700) = 7600
    \]

    \[
    T_\text{billed} = 1200 + 4\times 700 = 4000
    \]

    Caching bills 4000 full-price input tokens instead of 7600: the prefix is paid once instead of four times, saving \((s-1)\,T_\text{prefix} = 3600\) tokens, 47% of the input bill.

---

## 2. Customer-support resolution agent

### 2.1 The job

**Who:** a SaaS company's support team. **Input:** an inbound ticket in a helpdesk system (subject, body, customer id, order history). **Success:** the ticket is resolved correctly — answered, refunded within policy, or escalated — with the customer's next reply confirming it, and no wrong refund. **Volume (illustrative):** 2,000 tickets/day, of which ~40% are routine (where/how, status, small refunds) and the rest need a human.

### 2.2 Why an agent (or why only partly)

Apply the rule of [52 §1.1](52-ai-agents.md#11-the-decision-rule). Classifying the ticket and drafting a reply are a **fixed workflow** (route → retrieve policy → draft). What makes it agentic is the small, observation-driven tail: *look up the order, see it was charged twice, decide whether a refund is warranted and for how much.* The path there depends on what the lookups return, so the resolution step is an agent; everything around it is a workflow. Most of the work is not agentic, and that is good — keep the autonomous surface small.

### 2.3 Architecture

```text
integration shape: streaming chat UI (agent) + agent service with run ids
                   (helpdesk webhook -> run)

ticket ─▶ [router: small model] ─▶ class ─▶ [resolver agent: mid model]
                                               │   retrieval: policy + KB (RAG, 51)
                                               │   tools: lookups (read), refund (write)
                                               ▼
                                   [refund gate: amount<=limit?]──over──▶ human approve
                                               │ within
                                               ▼
                                     execute + [audit log] ─▶ reply (streamed)
guardrails: PII never leaves the tenant; refund tool scoped to one order; budgets per run
```

### 2.4 Tools

| Tool | R/W | Data it touches | Approval? | Size cap |
|---|---|---|---|---|
| `search_kb` | read | public help articles | no | 5 chunks |
| `get_customer` | read | one customer's profile | no | 1 record |
| `get_order` | read | one order + its payments | no | 1 record |
| `issue_refund` | write | payments ledger | if > \$50 | 1 refund/call |
| `escalate` | write | ticket queue | no (safe) | 1 ticket |

### 2.5 Context and data

Retrieved: the refund policy and relevant help articles, by hybrid search over the KB ([51 §7](51-ai-rag.md#7-hybrid-search-and-reciprocal-rank-fusion)). Remembered: the running ticket thread as working memory, summarized once it grows ([52 §10](52-ai-agents.md#10-memory)); past tickets for this customer are long-term memory, retrieved on demand, not stuffed in. Never sent to the model: full card numbers or other PII beyond the last four digits (redact first — [56 §8](56-ai-safety.md#8-pii-detection-and-redaction)), and no internal credentials ([56 §9](56-ai-safety.md#9-secrets-in-prompts-and-logs)).

### 2.6 Autonomy and guardrails

Autonomy: **act and report** at or under \$50, **act with approval** above it. Lookups and KB answers run automatically; a refund at or under \$50 posts automatically and is logged; a refund over \$50, or any second refund on the same order, pauses for a human who sees the exact amount and order ([52 §15](52-ai-agents.md#15-human-in-the-loop)). Budgets: 8 steps, 12k tokens, 30 s per ticket. Sandbox: the refund tool is scoped to the single order in context and cannot touch another customer.

The gate is the load-bearing part, and it is a few lines of your code, not a prompt:

```python
# Support agent: a refund tool behind a limit check and an approval gate.
LIMIT = 50.0                      # auto-refund ceiling in USD

def refund_gate(amount, order_total, approved=False):
    if amount <= 0 or amount > order_total:
        return "deny", "amount must be >0 and <= order total"
    if amount <= LIMIT:
        return "auto", f"within ${LIMIT:.0f} limit"
    if approved:
        return "execute", "human approved"
    return "needs_approval", f"over ${LIMIT:.0f}: human must confirm"

def model_decision(ticket):       # scripted fake model
    return {"t1": ("A1001", 20.0, 20.0),
            "t2": ("A1002", 80.0, 80.0)}[ticket]

for ticket in ("t1", "t2"):
    order, total, amt = model_decision(ticket)
    action, why = refund_gate(amt, total)
    print(f"{ticket}: refund ${amt:.0f} on {order} -> {action} ({why})")
    if action == "needs_approval":
        action, why = refund_gate(amt, total, approved=True)
        print(f"    after approval -> {action} ({why})")
```

```text
t1: refund $20 on A1001 -> auto (within $50 limit)
t2: refund $80 on A1002 -> needs_approval (over $50: human must confirm)
    after approval -> execute (human approved)
```

The \$20 refund posts; the \$80 one cannot execute until a human approves, and the amount the human sees is the concrete `$80 on A1002`, never a paraphrase.

### 2.7 A sample trajectory

```text
Thought: billing ticket; find the order before deciding.
Action: get_order({"order_id": "A1001"})
Observation: {"error": "TimeoutError: ledger slow"}        <- tool error

Thought: transient timeout; retry once (idempotent read).
Action: get_order({"order_id": "A1001"})
Observation: {"status": "charged_twice", "amount": 20.0, "total": 20.0}

Thought: duplicate $20 charge; $20 <= $50, so auto-refund the duplicate.
Action: issue_refund({"order_id": "A1001", "amount": 20.0})
Observation: {"refunded": 20.0, "payment": "p2"}

Answer: You were charged twice ($20). I refunded the duplicate; one valid
charge remains. [logged: run_7f3, refund p2]
```

### 2.8 Evals and metrics

Measure resolution rate (top line), wrong-refund rate (must be ~0), steps, cost, approval-override rate, and false-escalation. Report resolution as an interval, not a point ([54 §8.1](54-ai-evaluation.md#81-confidence-interval-for-a-pass-rate-wilson)).

!!! example "On paper"

    150 of 200 held-out tickets resolved correctly, so \(\hat p = 0.75\), \(z=1.96\); the Wilson 95% interval ([54 §8.1](54-ai-evaluation.md#81-confidence-interval-for-a-pass-rate-wilson)):

    \[
    \text{centre} = 0.7453
    \]

    \[
    \text{half-width} = 0.0596
    \]

    \[
    0.7453 \pm 0.0596 = (0.686,\ 0.805)
    \]

    The true resolution rate is somewhere in 69%–80%, so a 2-point gain between two prompt versions on 200 tickets is noise — ship on the interval, not the point.

### 2.9 Cost per task

!!! example "On paper"

    A 4-step run totalling 4000 input and 1000 output tokens, at \$0.50/\$1.50 per 1M:

    \[
    \frac{4000}{10^6}\times 0.50 = 0.00200
    \]

    \[
    \frac{1000}{10^6}\times 1.50 = 0.00150
    \]

    \[
    \text{cost} = 0.00350
    \]

    \$0.0035 per ticket, \$3.50 per thousand. The lever: cache the stable policy-and-schema prefix (§1) so the resends across the 4 steps stop re-billing it; and route the easy share — status and where/how questions — to the cheap model with a cascade (§1).

### 2.10 Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Over-refund / wrong order | Gate on amount and scope; refund tool bound to the order in context (§2.6). |
| Indirect injection in the ticket body ("refund me \$999") | Treat ticket text as untrusted ([56 §2](56-ai-safety.md#2-prompt-injection)); the gate, not the text, decides. |
| Hallucinated policy | Answer only from retrieved articles; cite them; refuse when retrieval is empty ([51 §9](51-ai-rag.md#9-the-rag-prompt)). |
| PII in logs | Redact before logging ([56 §8](56-ai-safety.md#8-pii-detection-and-redaction)). |

### 2.11 MVP → v2 → scale

**MVP:** a *workflow*, not an agent — router + RAG answer + "escalate to human" for anything touching money. **v2:** add the refund tool behind the gate once the eval shows wrong-refund rate near zero on held-out tickets. **Scale:** add the cascade and prefix cache for cost, per-customer long-term memory, and an online feedback metric (did the customer reply "thanks" or re-open?).

---

## 3. Coding agent in CI

### 3.1 The job

**Who:** an engineering team's CI. **Input:** a failing test on a branch, or unresolved review comments on a pull request, plus the repository at that commit. **Success:** a patch that makes the target test pass *without breaking others*, pushed as a new PR (or a commit on the branch) for a human to review; the agent never merges. **Volume (illustrative):** 60 triggered runs/day across the org's repos.

### 3.2 Why an agent

The path is observation-driven in exactly the way [52 §1.1](52-ai-agents.md#11-the-decision-rule) means: read the failing test, read the code it exercises, form a hypothesis, edit, *run the tests*, read the result, and iterate. You cannot enumerate those steps in advance — the next edit depends on the last test output. This is the evaluator–optimizer pattern ([52 §2.5](52-ai-agents.md#25-evaluatoroptimizer)) with an **external** check (the test runner), which is the trustworthy kind.

### 3.3 Architecture

```text
integration shape: queue worker (CI webhook -> job), sandboxed checkout

failing test ─▶ [job] ─▶ [coding agent: mid/large model]
                           │  retrieval: repo map + failing test + related files
                           │  tools: read_file, search_code, apply_patch, run_tests
                           ▼
                 loop: edit ─▶ run_tests ─▶ read result ─▶ edit  (budget: steps, time)
                           │ all green & no regressions
                           ▼
                 open_pr (never merge) ─▶ [audit log] ─▶ notify reviewer
guardrails: ephemeral container, no network egress, no prod creds, merge is human-only
```

### 3.4 Tools

| Tool | R/W | Data it touches | Approval? | Size cap |
|---|---|---|---|---|
| `read_file` | read | checkout files | no | 400 lines |
| `search_code` | read | the checkout | no | 30 hits |
| `apply_patch` | write | checkout only (sandbox) | no (sandboxed) | 200-line diff |
| `run_tests` | read | runs the suite in sandbox | no | summary + 50 lines |
| `open_pr` | write | the VCS host | human merges | 1 PR |

### 3.5 Context and data

Retrieved: a repository map and the files the failing test touches (code-aware chunking — [51 §3](51-ai-rag.md#3-chunking)). Remembered: the edit/test transcript as working memory, summarized when long ([52 §10](52-ai-agents.md#10-memory)). Never sent: `.env` files, deploy keys, or anything matching a secret pattern — scrub the checkout first ([56 §9](56-ai-safety.md#9-secrets-in-prompts-and-logs)).

### 3.6 Autonomy and guardrails

Autonomy: **act and report** *inside the sandbox* (it edits and runs tests freely there) and **suggest** at the boundary that matters (the PR is a proposal; a human merges). Budgets: 20 steps, 10 min wall-clock, a token ceiling. Sandbox: an ephemeral container with the checkout and no network egress and no production credentials — a bad patch cannot escape it ([52 §16](52-ai-agents.md#16-sandboxing-permissions-and-least-privilege)).

The gate is on the PR, and it is an external check — the tests — not the model's opinion:

```python
# Coding agent in CI: gate the PR on a green run that fixes the target and adds no regressions.
def run_tests(suite):             # fake runner: returns {test: passed?}
    return suite

BEFORE = {"test_login": False, "test_logout": True, "test_signup": True}

def pr_gate(before, after):
    newly_failing = [t for t in after if after[t] is False and before.get(t) is True]
    target_fixed = before.get("test_login") is False and after.get("test_login") is True
    all_green = all(after.values())
    ok = all_green and target_fixed and not newly_failing
    return ok, {"all_green": all_green, "target_fixed": target_fixed,
                "regressions": newly_failing}

after1 = run_tests({"test_login": True, "test_logout": True, "test_signup": False})
ok1, d1 = pr_gate(BEFORE, after1)
print(f"attempt 1: open_pr={ok1} {d1}")
after2 = run_tests({"test_login": True, "test_logout": True, "test_signup": True})
ok2, d2 = pr_gate(BEFORE, after2)
print(f"attempt 2: open_pr={ok2} {d2}")
```

```text
attempt 1: open_pr=False {'all_green': False, 'target_fixed': True, 'regressions': ['test_signup']}
attempt 2: open_pr=True {'all_green': True, 'target_fixed': True, 'regressions': []}
```

Attempt 1 fixes the target but breaks `test_signup`; the gate blocks the PR and feeds the regression back as the next observation. Attempt 2 is the recovery: green everywhere, target fixed, no regression — the PR opens.

### 3.7 A sample trajectory

```text
Thought: read the failing test to see what it expects.
Action: run_tests({"only": "test_login"})
Observation: FAIL test_login: AssertionError: expected 200, got 500

Thought: a 500; find the login handler.
Action: search_code({"q": "def login"})
Observation: auth/views.py:42 def login(req): ...

Thought: null session before auth; patch it.
Action: apply_patch({"file": "auth/views.py", "diff": "..."})
Observation: {"error": "patch did not apply: context mismatch"}   <- tool error

Thought: the file moved under me; re-read and re-target.
Action: read_file({"path": "auth/views.py", "start": 30, "end": 70})
Observation: <current lines>
Action: apply_patch({...corrected...}) -> run_tests({}) -> all green, no regressions
Action: open_pr({"title": "Fix null session in login", "branch": "fix-login"})
Answer: Opened PR #214 fixing test_login; 3/3 tests pass. Needs review.
```

### 3.8 Evals and metrics

Use **execution-based** evaluation — the tests are the grader ([54 §4](54-ai-evaluation.md#4-execution-based-evaluation)). Report `pass@1` for the deployment view and `pass@k` for capability ([09 §5](09-evaluation.md#passk-and-passk), [42 §2.3](42-distill-eval.md#23-passk)), plus steps, regression rate, and cost.

!!! example "On paper"

    The agent gets \(n=5\) independent attempts on one bug and \(c=3\) produce a green diff; the unbiased estimator ([42 §2.3](42-distill-eval.md#23-passk)):

    \[
    \text{pass@}1 = c/n = 3/5 = 0.6
    \]

    \[
    \text{pass@}2 = 1 - \frac{\binom{2}{2}}{\binom{5}{2}} = 1 - \frac{1}{10} = 0.9
    \]

    One attempt fixes the bug 60% of the time; allowing a second independent attempt lifts "at least one green diff" to 90%. CI can afford retries, so pass@2 is the number that matters here.

### 3.9 Cost per task

!!! example "On paper"

    A long run over a large checkout: 12000 input and 1500 output tokens, at \$0.50/\$1.50 per 1M:

    \[
    \frac{12000}{10^6}\times 0.50 = 0.00600
    \]

    \[
    \frac{1500}{10^6}\times 1.50 = 0.00225
    \]

    \[
    \text{cost} = 0.00825
    \]

    \$0.00825 per run. The lever: cache the repository-context prefix across the loop's many steps (§1); return only test *summaries*, not full logs, so each step adds less ([52 §13](52-ai-agents.md#13-cost-growth)).

### 3.10 Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Patch passes the target, breaks others | Run the *whole* suite; gate on no regressions (§3.6). |
| Agent weakens the test to make it pass | Diff the test files; block changes to the target test; human review. |
| Infinite edit/run loop | Step and time budgets; stuck-loop detector ([52 §18](52-ai-agents.md#18-failure-modes)). |
| Secret leakage from the checkout | Scrub secrets; no egress from the sandbox ([56 §9](56-ai-safety.md#9-secrets-in-prompts-and-logs)). |

### 3.11 MVP → v2 → scale

**MVP:** a single-shot "suggest a patch" that a human applies — no sandbox writes. **v2:** the sandboxed edit/run loop with the PR gate, once execution-based eval shows an acceptable pass@1 on a held-out bug set. **Scale:** parallel attempts (pass@k) with the best green diff chosen, repo-specific retrieval, and a regression-rate gate wired into CI ([54 §13](54-ai-evaluation.md#13-evaluating-rag-and-agents)).

---

## 4. Data-analyst agent (natural language → SQL)

### 4.1 The job

**Who:** non-technical staff asking questions of a data warehouse. **Input:** a plain-English question ("revenue by region last quarter"). **Success:** a correct, read-only query that returns the right rows, plus a short answer and a chart, with the SQL shown. **Volume (illustrative):** 500 questions/day.

### 4.2 Why an agent (or why only partly)

Mostly a **workflow**: retrieve the relevant schema, write SQL, verify it, run it read-only, chart the result. The agentic part is narrow — when the query errors or returns something implausible, the model reads the error and *revises* (the evaluator–optimizer loop again, [52 §2.5](52-ai-agents.md#25-evaluatoroptimizer)). One or two revise steps, not an open-ended loop.

### 4.3 Architecture

```text
integration shape: agent service with run ids (BI tool calls it)

question ─▶ [schema retrieval: which tables/columns] ─▶ [model writes SQL]
                                                           │
                                                           ▼
                                      [verifier: SELECT-only, LIMIT, no DDL]
                                                 │ ok              │ reject
                                                 ▼                 ▼
                                   run on READ-ONLY replica     revise (<=2)
                                                 │
                                                 ▼
                                   sanity check rows ─▶ chart + answer ─▶ [audit log]
```

### 4.4 Tools

| Tool | R/W | Data it touches | Approval? | Size cap |
|---|---|---|---|---|
| `get_schema` | read | table/column catalog | no | 40 tables |
| `run_sql` | read | **read-only** replica | no | 1000 rows |
| `make_chart` | read | the returned rows | no | 1 chart |

### 4.5 Context and data

Retrieved: only the schema fragments relevant to the question (full schemas blow the context and confuse the model) — a retrieval problem ([51 §2](51-ai-rag.md#2-embeddings-in-practice)). Remembered: the last few Q→SQL pairs in the session for follow-ups. Never sent: row-level PII; the model sees the *schema*, not the data, and the replica enforces row-level access ([51 §11](51-ai-rag.md#11-metadata-filters-and-access-control)). Credentials are read-only and never in the prompt ([56 §9](56-ai-safety.md#9-secrets-in-prompts-and-logs)).

### 4.6 Autonomy and guardrails

Autonomy: **act and report** — read-only queries run automatically because they cannot damage anything, and the result is reported with the SQL. The guardrail is the verifier plus a read-only credential: even a model that emitted `DROP TABLE` cannot run it, because the connection has no write grant and the verifier rejects it first.

The verifier is deterministic and sits between the model and the database:

```python
# Data-analyst agent: verify model-written SQL is safe and read-only before it runs.
import re

BANNED = re.compile(r"\b(insert|update|delete|drop|alter|truncate|create|grant|merge)\b", re.I)
MAX_ROWS = 1000

def verify_sql(sql):
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        return False, "multiple statements"
    if not re.match(r"(?is)^\s*select\b", s):
        return False, "not a SELECT"
    if BANNED.search(s):
        return False, "write/DDL keyword present"
    m = re.search(r"(?i)\blimit\s+(\d+)\b", s)
    if not m:
        return False, "no LIMIT clause"
    if int(m.group(1)) > MAX_ROWS:
        return False, f"LIMIT {m.group(1)} exceeds {MAX_ROWS}"
    return True, "ok"

cases = [
    "SELECT region, SUM(amount) FROM orders GROUP BY region LIMIT 100",
    "SELECT * FROM orders; DROP TABLE orders",
    "UPDATE orders SET amount=0",
    "SELECT * FROM orders",
]
for sql in cases:
    ok, why = verify_sql(sql)
    print(f"{'PASS' if ok else 'FAIL'}: {why:24} <- {sql[:38]}")
```

```text
PASS: ok                       <- SELECT region, SUM(amount) FROM orders
FAIL: multiple statements      <- SELECT * FROM orders; DROP TABLE order
FAIL: not a SELECT             <- UPDATE orders SET amount=0
FAIL: no LIMIT clause          <- SELECT * FROM orders
```

The verifier is defence in depth, not the only defence: the read-only replica is the one that holds even if the regex is fooled. Two independent controls, because one is never enough.

The same verifier guards real model output. Here the model runs against the handbook's local server ([10 §4](10-export-serving.md#4-serve-with-mlx)); thinking is turned off and `content` is guarded because it can be `None`:

```python
# The model writes SQL; your code verifies it before it ever runs.
import re
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
MODEL = "mlx-community/Qwen3-14B-4bit"
BANNED = re.compile(r"\b(insert|update|delete|drop|alter|truncate|create)\b", re.I)

def verify_sql(sql):
    s = sql.strip().rstrip(";")
    if not re.match(r"(?is)^\s*select\b", s): return False, "not a SELECT"
    if ";" in s: return False, "multiple statements"
    if BANNED.search(s): return False, "write/DDL keyword"
    if not re.search(r"(?i)\blimit\s+\d+\b", s): return False, "no LIMIT"
    return True, "ok"

r = client.chat.completions.create(
    model=MODEL, temperature=0.0,
    messages=[{"role": "system", "content": "Return one read-only SQL query, with a LIMIT."},
              {"role": "user", "content": "Revenue by region, top 10."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
sql = r.choices[0].message.content or ""     # content can be None; never crash
ok, why = verify_sql(sql)
print("sql:", sql)
print(f"verify: {'PASS' if ok else 'FAIL'} ({why}) -> {'run read-only' if ok else 'reject'}")
```

```text
sql: SELECT region, SUM(amount) AS revenue FROM orders GROUP BY region ORDER BY revenue DESC LIMIT 10
verify: PASS (ok) -> run read-only
```

The output above came from a local stub standing in for `mlx_lm.server` that reproduces Qwen3's thinking mode: with `enable_thinking` off it returns the SQL directly in `content`. A real Qwen3-14B returns the same SQL shape. With thinking *on*, `mlx_lm.server` puts the reasoning in a separate `reasoning` field ([50 §7](50-ai-prompting-apis.md#7-chain-of-thought-and-reasoning-models)) and this code would still read only `content` ([52 §6.1](52-ai-agents.md#61-the-same-loop-against-a-real-model)); a server that does not split it out leaves a `<think>…</think>` block at the start of `content`, which the verifier would reject as "not a SELECT" — one more reason to turn thinking off for this call. The `content or ""` guard matters because a cut-off or pure-reasoning turn can leave `content` as `None`; against the same stub with thinking on and a 20-token `max_tokens`, the run printed `sql: ` and `FAIL (not a SELECT) -> reject` instead of crashing.

### 4.7 A sample trajectory

```text
Thought: need the schema for revenue and region.
Action: get_schema({"q": "revenue region orders"})
Observation: orders(region, amount, created_at); ...

Thought: write the aggregate with a LIMIT.
Action: run_sql({"sql": "SELECT region, SUM(amont) ... LIMIT 10"})
Observation: {"error": "column \"amont\" does not exist"}     <- tool error

Thought: typo; the column is "amount". Fix and rerun.
Action: run_sql({"sql": "SELECT region, SUM(amount) AS revenue ... LIMIT 10"})
Observation: [{"region":"EU","revenue":48210.0}, ...]

Answer: EU leads at $48,210, then NA. [chart attached; SQL shown]
```

### 4.8 Evals and metrics

Measure **execution accuracy** (do the returned rows match the gold query's rows?) over a labelled question→SQL set, plus revise-count and cost. Report as a Wilson interval ([54 §8.1](54-ai-evaluation.md#81-confidence-interval-for-a-pass-rate-wilson)).

!!! example "On paper"

    72 of 80 questions return the correct rows, so \(\hat p = 0.90\), \(z=1.96\):

    \[
    \text{centre} = 0.8817
    \]

    \[
    \text{half-width} = 0.0668
    \]

    \[
    0.8817 \pm 0.0668 = (0.815,\ 0.948)
    \]

    "72 of 80" is consistent with a true execution accuracy from 82% to 95% — enough to ship read-only, far from enough to ever grant a write grant.

### 4.9 Cost per task

!!! example "On paper"

    Schema plus a short revise loop: 6000 input and 800 output tokens, at \$0.50/\$1.50 per 1M:

    \[
    \frac{6000}{10^6}\times 0.50 = 0.00300
    \]

    \[
    \frac{800}{10^6}\times 1.50 = 0.00120
    \]

    \[
    \text{cost} = 0.00420
    \]

    \$0.0042 per question. The lever: cache the schema prefix (§1) — it is the same for every question on a table — and route simple lookups to the cheap model, escalating only joins and window functions (§1).

### 4.10 Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Destructive SQL | Verifier rejects non-SELECT; read-only credential (§4.6). |
| Wrong but plausible numbers | Execution-accuracy eval; show the SQL so a human can audit. |
| Schema too big for context | Retrieve only relevant fragments ([51 §2](51-ai-rag.md#2-embeddings-in-practice)). |
| Row-level data leak | Row-level access on the replica ([51 §11](51-ai-rag.md#11-metadata-filters-and-access-control)). |

### 4.11 MVP → v2 → scale

**MVP:** schema → SQL → verify → run, no revise loop; show the SQL and let the user re-ask. **v2:** add the one- or two-step revise loop on errors once eval shows it lifts execution accuracy. **Scale:** per-team schema retrieval, cached query plans, a semantic cache of common questions ([55 §11.2](55-ai-production.md#112-semantic-cache)).

---

## 5. On-call incident responder

### 5.1 The job

**Who:** an SRE team. **Input:** a firing alert (service, metric, threshold) plus logs, dashboards, and runbooks. **Success:** a correct diagnosis and a *proposed* remediation; only allow-listed, approved actions execute; mean time to acknowledge drops. **Volume (illustrative):** 40 pages/week, most low-severity.

### 5.2 Why an agent

This is the canonical agent case from [52 §1.1](52-ai-agents.md#11-the-decision-rule): *investigate until you find the cause; the next probe depends on the last observation.* You cannot script "check CPU, then logs, then deploys" in a fixed order — which signal to pull next depends on what the alert and the previous query showed. The diagnosis is genuinely agentic; the *remediation* is not autonomous at all.

### 5.3 Architecture

```text
integration shape: queue worker (alert webhook -> run) + streaming updates to chat

alert ─▶ [responder agent: mid model]
            │  retrieval: runbooks (RAG), recent deploys, service graph
            │  tools (read): query_logs, query_metrics, list_deploys, get_runbook
            │  tools (write, allow-listed): restart_service, rollback
            ▼
     diagnose ─▶ propose remediation ─▶ [approval in chat: exact action]──no──▶ stop
            │ yes + action in allow-list
            ▼
     execute ─▶ verify recovery ─▶ [audit log] ─▶ post incident note
```

### 5.4 Tools

| Tool | R/W | Data it touches | Approval? | Size cap |
|---|---|---|---|---|
| `query_logs` | read | log store | no | 50 lines |
| `query_metrics` | read | metrics store | no | 20 points |
| `list_deploys` | read | deploy history | no | 10 entries |
| `get_runbook` | read | runbook docs | no | 3 chunks |
| `restart_service` | write | one named service | **yes** | 1 service |
| `rollback` | write | one deployment | **yes** | 1 rollback |

### 5.5 Context and data

Retrieved: the runbook for this alert and the recent deploy list (a frequent root cause) — RAG over internal docs ([51 §9](51-ai-rag.md#9-the-rag-prompt)). Remembered: the investigation transcript; prior incidents on this service are long-term memory, retrieved by similarity. Never sent: raw logs containing user PII or tokens (redact — [56 §8](56-ai-safety.md#8-pii-detection-and-redaction), [56 §9](56-ai-safety.md#9-secrets-in-prompts-and-logs)); the model gets aggregates and samples, not the firehose.

### 5.6 Autonomy and guardrails

Autonomy: **act with approval**. Read-only queries run automatically; every write (`restart_service`, `rollback`) is allow-listed *and* requires a human yes showing the exact action, because a wrong restart during an incident makes it worse. Budgets: 15 steps, 60k tokens (logs are large), 5 min. The allow-list is code: an action not on it is denied, not negotiated ([52 §16](52-ai-agents.md#16-sandboxing-permissions-and-least-privilege)).

### 5.7 A sample trajectory

```text
Thought: 5xx spike on checkout. Recent deploy is the first suspect.
Action: list_deploys({"service": "checkout", "hours": 2})
Observation: [{"id": "d_88", "at": "12:58", "by": "ci"}]

Thought: deploy 2 min before the spike. Check its error signature.
Action: query_logs({"service": "checkout", "since": "12:58"})
Observation: {"error": "RateLimitError from log backend"}       <- unexpected tool error

Thought: the log tool itself is rate-limited; back off and narrow the window.
Action: query_logs({"service": "checkout", "since": "13:00", "limit": 50})
Observation: NullPointerException in PaymentClient (new in d_88)

Thought: regression in d_88. Propose rollback; this is a write -> approval.
Action: rollback({"deploy": "d_88"})   [PENDING APPROVAL: "rollback d_88 on checkout"]
Observation: {"approved_by": "sre_oncall", "result": "rolled_back"}

Answer: 5xx caused by NPE in d_88 (PaymentClient). Rolled back after approval;
error rate recovering. [logged]
```

### 5.8 Evals and metrics

Measure diagnosis accuracy against labelled past incidents, the **approval-override rate** (how often the human rejects the proposal), and the **unsafe-action rate**, which must be zero ([54 §13](54-ai-evaluation.md#13-evaluating-rag-and-agents)).

!!! example "On paper"

    The agent proposes a remediation on 50 incidents; the human rejects 4:

    \[
    \text{override rate} = 4/50 = 0.08
    \]

    Across 120 runs, 0 out-of-allow-list actions were attempted; the Wilson 95% upper bound on the unsafe-action rate ([54 §8.1](54-ai-evaluation.md#81-confidence-interval-for-a-pass-rate-wilson)):

    \[
    \text{upper} = 0.031
    \]

    An 8% override rate means the proposals are usually right; "0 of 120" still only *bounds* the unsafe rate at ~3%, so the approval gate stays — zero observed is not zero possible.

### 5.9 Cost per task

!!! example "On paper"

    A log-heavy investigation: 20000 input and 1000 output tokens, at \$0.50/\$1.50 per 1M:

    \[
    \frac{20000}{10^6}\times 0.50 = 0.01000
    \]

    \[
    \frac{1000}{10^6}\times 1.50 = 0.00150
    \]

    \[
    \text{cost} = 0.01150
    \]

    \$0.0115 per incident — the costliest here, because logs dominate the input. The lever: summarize old observations so the resend stops growing ([52 §10](52-ai-agents.md#10-memory)); pre-aggregate logs so each tool result is small ([52 §13](52-ai-agents.md#13-cost-growth)).

### 5.10 Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Wrong remediation executed | Approval gate on every write; allow-list (§5.6). |
| Log flood blows the budget | Pre-aggregate; cap tool results; token budget ([52 §14](52-ai-agents.md#14-budgets)). |
| Injection via log contents | Treat logs as untrusted input ([56 §2](56-ai-safety.md#2-prompt-injection)). |
| Alert storm → many parallel runs | Dedup alerts; one run per incident; rate-limit. |

### 5.11 MVP → v2 → scale

**MVP:** read-only **suggest** — the agent triages and posts a diagnosis with evidence; a human does everything. **v2:** add a *single* allow-listed action (rollback) behind approval once override rate is low. **Scale:** more allow-listed actions one at a time, each gated; a durable workflow ([62](62-ai-agent-integration.md)) so a long investigation survives a restart.

---

## 6. Authorized security-testing agent (the Kali agent)

### 6.1 The job

**Who:** a security engineer running an **authorized** lab engagement. **Input:** an in-scope target inside an isolated lab network, under written authorization. **Success:** correct service enumeration and a structured, honest finding — in scope, logged, with evidence. **Volume (illustrative):** a handful of lab hosts per engagement. This application is fully specified in [11 — The Kali agent](11-kali-agent.md); this section only places it beside the other eleven and does not re-document it.

### 6.2 Why an agent

Observation-driven by nature: *enumerate services on an in-scope host, then decide what to probe next from what you found* ([52 §1.1](52-ai-agents.md#11-the-decision-rule)). The model plans; it is not trusted with the path to the internet.

### 6.3 Architecture

```text
integration shape: local harness (OpenCode / hackingBuddyGPT), air-gapped lab

task ─▶ [local Qwen3-14B-4bit] ─▶ typed MCP tool call
            │
            ▼
     [policy layer: scope + allow-list + confirm]   <- deterministic code, not an LLM
            │ allow                      │ deny
            ▼                            ▼
     MCP-Kali-Server (sandbox)         refuse + log
            │
            ▼
     lab target (no egress) ─▶ observation ─▶ model ─▶ report_finding ─▶ [audit log]
```

The three rules of [11 §1](11-kali-agent.md#1-architecture) are the whole design: the model never talks to the internet directly, the policy layer is code not a prompt, and every action is logged before it runs. Model: the handbook's local Qwen3-14B-4bit with a pentest LoRA, which is why the tool surface is small ([11 §4](11-kali-agent.md#4-a-small-typed-tool-surface)).

### 6.4 Tools

A small typed surface, summarised from [11 §4](11-kali-agent.md#4-a-small-typed-tool-surface) (see [16 §2](16-network-security.md#2-tool-surface-for-network-security) for the network-testing variants):

| Tool | R/W | Data it touches | Approval? | Size cap |
|---|---|---|---|---|
| `scope_check` | read | the scope file | no | 1 verdict |
| `nmap_scan` | read | in-scope host (flags enum) | no (recon) | stdout capped |
| `http_probe` | read | in-scope URL | no (recon) | headers + title |
| `kali_exec` | write-ish | allow-listed binaries only | **confirm** | timeout-bounded |
| `report_finding` | write | the report file | no (safe) | 1 finding |

### 6.5 Context and data

Retrieved: the scope file and tool docs; evidence notes under `/lab`. Remembered: the enumeration transcript, trimmed. Never sent off-box at all — the model is local and the lab has no egress; no real credentials, no customer data, no live CTF flags ([12 §3](12-safety-legal.md#3-data-handling)). This is the strongest version of "never send it to the model": the whole system is air-gapped.

### 6.6 Autonomy and guardrails

Autonomy: **act with approval**, split by the confirmation policy of [11 §5.1](11-kali-agent.md#51-confirmation-policy): read-only recon inside scope auto-approves; active enumeration confirms once per target; anything out of scope or off the allow-list is denied by the policy layer. The authorization itself is the precondition — missing authorization means the agent does not run at all ([12 §1](12-safety-legal.md#1-authorization-is-a-precondition-not-a-prompt)), and the legal frame is jurisdiction-specific ([12 §2](12-safety-legal.md#2-jurisdictions-summary)). Sandbox: an isolated Kali VM against disposable lab targets with no route to the public internet.

### 6.7 A sample trajectory

A safe, lab-only loop, condensed from [11 §7](11-kali-agent.md#7-a-worked-loop-safe-lab-only):

```text
Thought: 10.10.10.5 is inside 10.10.10.0/24; read-only recon is allowed.
Action: nmap_scan({"target": "10.10.10.5", "flags": "-sV"})
Observation: {"error": "host down / no route"}               <- tool error

Thought: maybe not up yet; scope_check first, then a ping sweep.
Action: scope_check({"target": "10.10.10.5"}) -> (allow, "in scope")
Action: nmap_scan({"target": "10.10.10.5", "flags": "-sn"})
Observation: host up
Action: nmap_scan({"target": "10.10.10.5", "flags": "-sV"})
Observation: 22/tcp ssh OpenSSH 9.6; 80/tcp http nginx 1.26
Action: http_probe({"url": "http://10.10.10.5", "method": "HEAD"})
Observation: 200 OK; Server: nginx/1.26; X-Powered-By: PHP/8.3

report_finding: "SSH + HTTP exposed on 10.10.10.5; nginx 1.26 / PHP 8.3.
Next (in scope): enumerate vhosts, check for a login page.
Evidence: nmap -sV; HEAD http://10.10.10.5."
```

### 6.8 Evals and metrics

Evaluate the **whole system** — model + prompt + tools + policy — not the endpoint ([11 §9](11-kali-agent.md#9-integration-with-evaluation), [54 §13](54-ai-evaluation.md#13-evaluating-rag-and-agents)). Report tool and argument accuracy ([09 §4.2](09-evaluation.md#42-toolargument-grader)), task completion, recovery rate, and — non-negotiable — a zero unsafe/out-of-scope action rate.

!!! example "On paper"

    On one lab task the agent gets \(n=5\) samples and \(c=2\) pass the verifier ([42 §2.3](42-distill-eval.md#23-passk)):

    \[
    \text{pass@}1 = c/n = 2/5 = 0.4
    \]

    \[
    \text{pass@}2 = 1 - \frac{\binom{3}{2}}{\binom{5}{2}} = 1 - \frac{3}{10} = 0.7
    \]

    A single run succeeds 40% of the time; a second try reaches 70%. Capability (pass@2) and reliability are different questions — but the unsafe-action rate must be **0** regardless of either.

### 6.9 Cost per task

Inference is local, so the real cost is amortized compute and wall-clock, not per-token. The token-equivalent, for comparison with the API applications:

!!! example "On paper"

    An enumeration run of 7000 input and 1000 output tokens, priced *as if* on the API at \$0.50/\$1.50 per 1M:

    \[
    \frac{7000}{10^6}\times 0.50 = 0.00350
    \]

    \[
    \frac{1000}{10^6}\times 1.50 = 0.00150
    \]

    \[
    \text{cost} = 0.00500
    \]

    \$0.005-equivalent per run; locally it is free at the margin once the Mac is on. The lever here is the **small typed tool surface** ([11 §4](11-kali-agent.md#4-a-small-typed-tool-surface)): 6–10 schemas instead of 150 keeps the per-step prompt small, which is cost *and* reliability on a 14B.

### 6.10 Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Out-of-scope scan | Policy-layer scope check (code, not prompt) — [11 §5](11-kali-agent.md#5-the-policy-layer). |
| Raw shell abuse | Typed tools; allow-list `kali_exec`; no bare shell ([11 §4](11-kali-agent.md#4-a-small-typed-tool-surface)). |
| Invented ports / findings | Report only what the tool output shows; evidence required ([11 §6](11-kali-agent.md#6-smoke-test)). |
| Running without authorization | Precondition check; no scope file, no run ([12 §1](12-safety-legal.md#1-authorization-is-a-precondition-not-a-prompt)). |

### 6.11 MVP → v2 → scale

**MVP:** planning-only — the model lists which tool it *would* use, calls nothing ([11 §3.2](11-kali-agent.md#32-opencode-provider-mcp)). **v2:** read-only recon tools behind the scope check; evaluate the loop. **Scale:** add active-enumeration tools one at a time, each confirmed, with the kill switch and audit log in place ([11 §5.3](11-kali-agent.md#53-kill-switch)). Never point it at anything you are not authorized to test.

---

## 7. Document-processing agent (invoices and contracts)

### 7.1 The job

**Who:** a finance or legal operations team. **Input:** a PDF invoice or contract. **Success:** a validated structured record (vendor, totals, dates, terms) posted to the system of record; low-confidence extractions go to a human review queue instead of being guessed. **Volume (illustrative):** 5,000 documents/day.

### 7.2 Why an agent (or why only partly)

Mostly a **workflow**: parse → extract fields → validate → route. The only agentic touch is handling messy documents — a scanned contract where the model must decide which clause is the term, or re-read a region. For clean invoices it is a single structured-output call, not an agent at all; reach for the loop only on the hard tail.

### 7.3 Architecture

```text
integration shape: queue worker + scheduled/batch (nightly backlog)

doc ─▶ [parse: OCR/PDF->text] ─▶ [extract: model -> JSON schema]
                                    │  validate: types, totals add up, dates sane
                                    ▼
                        [confidence router]
                         │>=0.90 auto     │0.70-0.90 review     │<0.70 reject
                         ▼                 ▼                     ▼
                    post record      human review queue      bounce back
                         └───────────────┬─────────────────────┘
                                         ▼
                                   [audit log]
```

### 7.4 Tools

| Tool | R/W | Data it touches | Approval? | Size cap |
|---|---|---|---|---|
| `ocr_page` | read | one document page | no | 1 page |
| `extract_fields` | read | parsed text | no | 1 record |
| `validate_record` | read | the extracted JSON | no | 1 record |
| `post_record` | write | system of record | if low-confidence | 1 record |
| `enqueue_review` | write | review queue | no (safe) | 1 item |

### 7.5 Context and data

Retrieved: the vendor's known format and prior records for that vendor (improves field mapping). Remembered: nothing across documents beyond vendor templates — each document is independent. Never sent: bank-account and tax-id numbers beyond what the record needs; redact and tokenize PII before it enters the model ([56 §8](56-ai-safety.md#8-pii-detection-and-redaction)).

### 7.6 Autonomy and guardrails

Autonomy: **act and report** for high-confidence records (auto-post and log), with a **human review queue** as the guardrail for everything uncertain. The confidence router is the whole safety story — a record only auto-posts when *every* field clears the bar.

```python
# Document agent: route each extracted invoice by its lowest field confidence.
AUTO, REVIEW = 0.90, 0.70         # >=0.90 auto-post; 0.70-0.90 human review; <0.70 reject

def route(fields):
    worst = min(fields.values())
    if worst >= AUTO:
        return "auto-post", worst
    if worst >= REVIEW:
        return "review-queue", worst
    return "reject", worst

invoices = {
    "INV-1": {"vendor": 0.99, "total": 0.97, "date": 0.95},
    "INV-2": {"vendor": 0.98, "total": 0.82, "date": 0.91},
    "INV-3": {"vendor": 0.66, "total": 0.80, "date": 0.72},
}
counts = {"auto-post": 0, "review-queue": 0, "reject": 0}
for name, f in invoices.items():
    dest, worst = route(f)
    counts[dest] += 1
    print(f"{name}: worst conf {worst:.2f} -> {dest}")
print("routing:", counts)
```

```text
INV-1: worst conf 0.95 -> auto-post
INV-2: worst conf 0.82 -> review-queue
INV-3: worst conf 0.66 -> reject
routing: {'auto-post': 1, 'review-queue': 1, 'reject': 1}
```

Routing on the *worst* field, not the average, is deliberate: a record is only as trustworthy as its least certain field, and a wrong total is worse than a blank one.

### 7.7 A sample trajectory

```text
Thought: parse the invoice, then extract into the schema.
Action: extract_fields({"doc": "INV-77"})
Observation: {"vendor": "...", "total": 1240.00, "conf_total": 0.98,
              "date": null, "conf_date": 0.0}

Thought: date missing; try OCR on the header region.
Action: ocr_page({"doc": "INV-77", "region": "header"})
Observation: {"error": "region out of bounds"}               <- tool error

Thought: bad region; OCR the whole first page instead.
Action: ocr_page({"doc": "INV-77", "page": 1})
Observation: "Invoice date: 2026-09-30"
Action: validate_record({...date filled, conf 0.88...}) -> route -> review-queue

Answer: INV-77 extracted; date confidence 0.88 -> sent to human review, not posted.
```

### 7.8 Evals and metrics

Measure per-field **precision and recall** against a labelled set, plus the auto-post share and the error rate *among auto-posted* records (the one that must stay tiny).

!!! example "On paper"

    For the `total` field: 90 correct extractions, 10 wrong, 20 missed. Precision and recall ([47 — Classical ML](47-ai-classical-ml.md) defines them; worked here):

    \[
    \text{precision} = \frac{90}{90+10} = 0.90
    \]

    \[
    \text{recall} = \frac{90}{90+20} = 0.818
    \]

    \[
    F_1 = \frac{2\cdot 0.90\cdot 0.818}{0.90+0.818} = 0.857
    \]

    90% of posted totals are right and 82% of true totals are caught; the 10% wrong is why the confidence router sends the shaky ones to a human instead of posting them.

### 7.9 Cost per task

!!! example "On paper"

    A single extraction call: 3000 input and 600 output tokens, at \$0.50/\$1.50 per 1M:

    \[
    \frac{3000}{10^6}\times 0.50 = 0.00150
    \]

    \[
    \frac{600}{10^6}\times 1.50 = 0.00090
    \]

    \[
    \text{cost} = 0.00240
    \]

    \$0.0024 per document. The lever: a cheap model does the clean majority and escalates only low-confidence documents to a bigger one (a cascade, §1) — most invoices are easy.

### 7.10 Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Confident wrong extraction | Route on worst field; validate totals/dates; sample auto-posts. |
| Silent field drop | Require every schema field; `null` with low confidence routes to review. |
| Injection in document text | Treat document text as untrusted ([56 §2](56-ai-safety.md#2-prompt-injection)). |
| PII in the record store | Tokenize/redact sensitive fields ([56 §8](56-ai-safety.md#8-pii-detection-and-redaction)). |

### 7.11 MVP → v2 → scale

**MVP:** a single structured-output extraction with validation, everything to a review queue — measure field accuracy first. **v2:** auto-post the high-confidence share once precision among auto-posts is proven. **Scale:** per-vendor templates, a batch pipeline for the backlog ([62](62-ai-agent-integration.md)), and active-learning on the reviewed queue to improve extraction ([60 §9.5](60-ai-data-preparation.md#95-active-learning-label-the-examples-that-teach-most)).

---

## 8. Research and report agent

### 8.1 The job

**Who:** analysts who need a sourced brief. **Input:** a research question and access to the web and internal documents. **Success:** a report that answers the question with every claim cited to a retrieved source, and a groundedness check passed before delivery. **Volume (illustrative):** 200 reports/day.

### 8.2 Why an agent

The orchestrator–workers and evaluator–optimizer shapes of [52 §2](52-ai-agents.md#2-agentic-workflow-patterns): the agent decides what to look up next from what it has found (research is not a fixed query list), then an evaluator checks the draft for groundedness and sends it back if a claim is unsupported. The decomposition cannot be written in advance — that is what makes it an agent.

### 8.3 Architecture

```text
integration shape: agent service with run ids (async, long-running)

question ─▶ [orchestrator: plan sub-questions]
               │  retrieval: web search + internal docs (RAG, 51)
               │  workers: search, read, extract-with-citation
               ▼
          draft ─▶ [groundedness evaluator] ──unsupported──▶ revise (<=N)
               │ every claim cited & supported
               ▼
          deliver report + citations ─▶ [audit log]
guardrails: untrusted web content; read-only; claims must trace to a source
```

### 8.4 Tools

| Tool | R/W | Data it touches | Approval? | Size cap |
|---|---|---|---|---|
| `web_search` | read | public web | no | 10 results |
| `fetch_page` | read | one URL (untrusted) | no | 4k chars |
| `search_internal` | read | internal doc index | no | 5 chunks |
| `cite` | read | retrieved spans | no | — |

### 8.5 Context and data

Retrieved: web results and internal chunks, each kept with its source id so a claim can be traced back ([51 §9](51-ai-rag.md#9-the-rag-prompt)). Remembered: the growing set of findings, summarized as it exceeds the budget ([52 §10](52-ai-agents.md#10-memory)). Never sent onward: internal documents the user is not cleared to see — enforce access control at retrieval, not in the prompt ([51 §11](51-ai-rag.md#11-metadata-filters-and-access-control)).

### 8.6 Autonomy and guardrails

Autonomy: **act and report** on the read-only research, then **suggest** — the report is delivered to a human, who decides what to do with it. The key guardrail is that fetched pages are untrusted input: a web page can carry an injection aimed at the agent, so the agent has no write tools and no outbound channel — removing a leg of the lethal trifecta ([56 §2.4](56-ai-safety.md#24-layered-defences)). The verification step is the groundedness evaluator ([54 §5](54-ai-evaluation.md#5-model-graded-evaluation-llm-as-judge)).

### 8.7 A sample trajectory

```text
Thought: break the question into three sub-questions; start with market size.
Action: web_search({"q": "EU heat-pump installs 2025"})
Observation: [{"url": "u1", "title": "..."}, ...]

Thought: read the first source.
Action: fetch_page({"url": "u1"})
Observation: {"error": "403 Forbidden"}                       <- tool error

Thought: that source is blocked; use the next result instead.
Action: fetch_page({"url": "u2"})
Observation: "EU installations reached 3.1M in 2025 ..."
Action: cite({"span": "3.1M in 2025", "source": "u2"})

[draft -> evaluator: "claim 'fastest-growing' has no citation" -> revise -> pass]
Answer: Report delivered; 7 claims, 7 citations, groundedness check passed.
```

### 8.8 Evals and metrics

Measure **groundedness** (fraction of claims supported by a cited source) and **citation accuracy** (does each marker point at a span that supports it?) — the RAG-generation metrics of [54 §13](54-ai-evaluation.md#13-evaluating-rag-and-agents) and [51 §13](51-ai-rag.md#13-end-to-end-rag-evaluation).

!!! example "On paper"

    A report makes 50 claims; 47 are supported by their cited source:

    \[
    \text{groundedness} = 47/50 = 0.94
    \]

    94% grounded means 3 unsupported claims slipped through — which is why the evaluator gate blocks delivery until groundedness clears its threshold, rather than trusting the first draft.

### 8.9 Cost per task

!!! example "On paper"

    A multi-source report: 10000 input and 2000 output tokens, at \$0.50/\$1.50 per 1M:

    \[
    \frac{10000}{10^6}\times 0.50 = 0.00500
    \]

    \[
    \frac{2000}{10^6}\times 1.50 = 0.00300
    \]

    \[
    \text{cost} = 0.00800
    \]

    \$0.008 per report. The lever: a cheap model for search-and-extract, the mid model only for synthesis and the groundedness check (a cascade, §1); cache the stable instruction prefix across the loop (§1).

### 8.10 Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Fabricated citation | Citation-accuracy eval; claims must trace to a retrieved span. |
| Injection from a fetched page | No write/outbound tools; untrusted content ([56 §2.4](56-ai-safety.md#24-layered-defences)). |
| Endless searching | Step/token budget; re-planning ([52 §9](52-ai-agents.md#9-planning-and-reflection)). |
| Leaking restricted internal docs | Access control at retrieval ([51 §11](51-ai-rag.md#11-metadata-filters-and-access-control)). |

### 8.11 MVP → v2 → scale

**MVP:** fixed RAG — retrieve once, answer with citations, no loop. **v2:** add the orchestrator and the groundedness evaluator once a citation-accuracy eval exists to prove they help. **Scale:** parallel workers for independent sub-questions ([52 §2.4](52-ai-agents.md#24-orchestratorworkers)), a semantic cache of common questions, and human spot-checks feeding the eval set.

---

## 9. Sales / CRM assistant

### 9.1 The job

**Who:** account executives. **Input:** a lead or account, plus CRM records, past emails, and public company data. **Success:** enriched account data, a meeting-prep brief, and *drafted* outreach emails the rep reviews and sends; CRM writes happen only with approval. **Volume (illustrative):** 300 account actions/day across a team.

### 9.2 Why an agent (or why only partly)

Enrichment and prep are a **workflow** (gather → summarize → draft). The agentic part is light: deciding which public sources to pull and which past threads matter for *this* account. Crucially, the assistant does not send email or write the CRM on its own — those are human or approval-gated, so autonomy stays low by design.

### 9.3 Architecture

```text
integration shape: agent-as-tool (invoked from the CRM UI) + agent service

account ─▶ [assistant: mid model]
             │  retrieval: CRM records + email history + public data
             │  tools (read): get_account, search_emails, enrich_company
             │  tools (write): draft_email (to a draft, not sent), update_crm (approval)
             ▼
     brief + draft email ─▶ rep reviews ─▶ rep sends / approves CRM write ─▶ [audit log]
```

### 9.4 Tools

| Tool | R/W | Data it touches | Approval? | Size cap |
|---|---|---|---|---|
| `get_account` | read | one CRM account | no | 1 record |
| `search_emails` | read | that account's threads | no | 10 messages |
| `enrich_company` | read | public company data | no | 1 profile |
| `draft_email` | write | drafts folder (not sent) | no (not sent) | 1 draft |
| `update_crm` | write | CRM fields | **yes** | 1 record |

### 9.5 Context and data

Retrieved: the account's CRM record, recent threads, and public firmographics. Remembered: a per-account summary as long-term memory, retrieved next time ([52 §10](52-ai-agents.md#10-memory)). Never sent: other accounts' data (scope each tool to the account in context), and no customer PII beyond what the draft needs ([56 §8](56-ai-safety.md#8-pii-detection-and-redaction)).

### 9.6 Autonomy and guardrails

Autonomy: **suggest** for email (the rep sends) and **act with approval** for CRM writes. The guardrail that keeps this safe is simply that the send button belongs to the human and `update_crm` is gated — a drafted email that is never auto-sent cannot embarrass anyone, and a CRM write the rep must confirm cannot corrupt the pipeline.

### 9.7 A sample trajectory

```text
Thought: prep for the Acme call; pull the account and recent threads.
Action: get_account({"id": "acc_42"})
Observation: {"name": "Acme", "stage": "negotiation", "owner": "rep_7"}

Thought: enrich with public data.
Action: enrich_company({"domain": "acme.example"})
Observation: {"error": "RateLimited: retry after 2s"}        <- tool error

Thought: transient; back off and retry once.
Action: enrich_company({"domain": "acme.example"})
Observation: {"headcount": 1200, "recent": "opened a London office"}

Action: draft_email({"to": "acc_42", "angle": "London expansion"})
Action: update_crm({"id": "acc_42", "stage": "proposal"})   [PENDING APPROVAL]
Answer: Brief ready; draft saved to your drafts (not sent). CRM stage change
to "proposal" is queued for your approval.
```

### 9.8 Evals and metrics

Measure the **draft acceptance rate** (fraction the rep sends with no or minor edits) and the CRM-write override rate. Report acceptance as a Wilson interval ([54 §8.1](54-ai-evaluation.md#81-confidence-interval-for-a-pass-rate-wilson)).

!!! example "On paper"

    Reps send 180 of 200 drafts with at most minor edits, so \(\hat p = 0.90\), \(z=1.96\):

    \[
    \text{centre} = 0.8925
    \]

    \[
    \text{half-width} = 0.0419
    \]

    \[
    0.8925 \pm 0.0419 = (0.851,\ 0.934)
    \]

    The true acceptance rate is 85%–93%; because nothing sends without a human, a bad draft costs a rep's minute, not a customer relationship — which is why "suggest" is the right level here.

### 9.9 Cost per task

!!! example "On paper"

    Enrichment plus a drafted email: 5000 input and 500 output tokens, at \$0.50/\$1.50 per 1M:

    \[
    \frac{5000}{10^6}\times 0.50 = 0.00250
    \]

    \[
    \frac{500}{10^6}\times 1.50 = 0.00075
    \]

    \[
    \text{cost} = 0.00325
    \]

    \$0.00325 per account action. The lever: cache the per-account context prefix across the session (§1); a cheap model drafts, the mid model only polishes high-value outreach (a cascade, §1).

### 9.10 Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Auto-sent wrong email | Draft only; the rep sends (§9.6). |
| CRM corruption | `update_crm` behind approval; scoped to one record. |
| Hallucinated company facts | Attribute enrichment to a source; mark unverified claims. |
| Cross-account data bleed | Scope every tool to the account in context ([56 §6](56-ai-safety.md#6-excessive-agency)). |

### 9.11 MVP → v2 → scale

**MVP:** a read-only prep brief — no writes at all. **v2:** add drafted emails to the drafts folder once acceptance rate is good. **Scale:** approval-gated CRM writes, per-account memory, and a cascade for cost; keep the send button human forever.

---

## 10. E-commerce order and returns agent

### 10.1 The job

**Who:** an online store's shoppers. **Input:** a chat message ("where is my order", "I want to return this"). **Success:** the order is looked up, policy applied, and a return or small refund processed under a limit — or escalated above it — with the shopper's issue resolved. **Volume (illustrative):** 10,000 conversations/day.

### 10.2 Why an agent (or why only partly)

A **workflow** with a narrow agentic middle: lookup → policy RAG → decide. The decision "is this return within the window and the refund under the limit?" is policy-driven, and the model's job is to apply the retrieved policy to the specific order, escalating when it does not fit. Autonomy is bounded hard by the limit.

### 10.3 Architecture

```text
integration shape: streaming chat UI (storefront) + agent service with run ids

message ─▶ [agent: mid model]
             │  retrieval: returns/refunds policy (RAG, 51)
             │  tools (read): get_order, get_policy
             │  tools (write): start_return, issue_refund (<= limit), escalate (> limit)
             ▼
     within limit ─▶ execute + [audit log] ─▶ reply (streamed)
     above  limit ─▶ escalate to human agent
```

### 10.4 Tools

| Tool | R/W | Data it touches | Approval? | Size cap |
|---|---|---|---|---|
| `get_order` | read | one shopper's order | no | 1 record |
| `get_policy` | read | returns policy docs | no | 3 chunks |
| `start_return` | write | returns system | no (within policy) | 1 return |
| `issue_refund` | write | payments | if > \$100 | 1 refund |
| `escalate` | write | human queue | no (safe) | 1 handoff |

### 10.5 Context and data

Retrieved: the returns/refunds policy relevant to this order (category, date, condition) — RAG with metadata filters ([51 §11](51-ai-rag.md#11-metadata-filters-and-access-control)). Remembered: the conversation; the shopper's order history on demand. Never sent: card numbers or full addresses beyond what the return needs ([56 §8](56-ai-safety.md#8-pii-detection-and-redaction)).

### 10.6 Autonomy and guardrails

Autonomy: **act with approval** above the limit, **act and report** below it. A refund at or under \$100 within the return window posts automatically; above \$100, or outside the window, it escalates to a human. The refund tool is scoped to the shopper's own order — the same excessive-agency discipline as the support agent ([56 §6](56-ai-safety.md#6-excessive-agency)).

### 10.7 A sample trajectory

```text
Thought: shopper wants a return; get the order and the policy.
Action: get_order({"id": "O-5521"})
Observation: {"item": "boots", "price": 60.0, "delivered": "19 days ago"}

Thought: policy window is 30 days; check refund limit.
Action: get_policy({"category": "footwear"})
Observation: {"window_days": 30, "refund_limit": 100.0}

Thought: 19 <= 30 and $60 <= $100 -> within policy; start the return.
Action: start_return({"order": "O-5521"})
Observation: {"return_id": "R-77", "error": "label service down"}   <- tool error

Thought: return R-77 exists but has no label; retry the label, else queue it.
Action: start_return({"order": "O-5521", "return_id": "R-77", "retry_label": true})
Observation: {"return_id": "R-77", "label": "emailed"}

Answer: Return started for your boots; a prepaid label is on its way. Refund
of $60 posts when we receive them.
```

### 10.8 Evals and metrics

Measure task success, the escalation rate (above-limit cases correctly handed off), and reliability over the multi-step flow ([52 §12](52-ai-agents.md#12-reliability-math)).

!!! example "On paper"

    A return flow is 4 steps, each succeeding independently with \(p=0.95\); end-to-end success ([52 §12](52-ai-agents.md#12-reliability-math)):

    \[
    S = 0.95^4 = 0.8145
    \]

    Of 100 conversations, 12 exceed the limit and are escalated:

    \[
    \text{escalation rate} = 12/100 = 0.12
    \]

    A 95% step gives only an 81% four-step flow — so keep the flow short — and a clean 12% escalation means the limit is doing its job, routing the expensive cases to a human.

### 10.9 Cost per task

!!! example "On paper"

    A short conversation: 2500 input and 500 output tokens, at \$0.50/\$1.50 per 1M:

    \[
    \frac{2500}{10^6}\times 0.50 = 0.00125
    \]

    \[
    \frac{500}{10^6}\times 1.50 = 0.00075
    \]

    \[
    \text{cost} = 0.00200
    \]

    \$0.002 per conversation. The lever: an exact-response cache for the many identical "where is my order" questions ([55 §11.1](55-ai-production.md#111-exact-response-cache)); a cheap model for status lookups, escalating only returns to the mid model (§1).

### 10.10 Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Refund outside policy | Apply retrieved policy; limit + window gate; escalate otherwise. |
| Over-limit auto-refund | Hard \$100 gate in code (§10.6). |
| Prompt injection in chat | Untrusted user text; gate decides, not the text ([56 §2](56-ai-safety.md#2-prompt-injection)). |
| Wrong shopper's order | Scope `get_order`/`issue_refund` to the authenticated shopper. |

### 10.11 MVP → v2 → scale

**MVP:** order-status answers + "escalate to human" for anything touching money. **v2:** returns and under-limit refunds behind the policy gate once eval shows a low wrong-refund rate. **Scale:** cascade and caching for cost, multilingual support, and a feedback metric from post-chat surveys ([55 §18.4](55-ai-production.md#184-the-data-flywheel)).

---

## 11. Internal IT / HR helpdesk agent

### 11.1 The job

**Who:** employees. **Input:** a request in a chat tool ("reset my password", "request access to repo X", "how much parental leave do I get?"). **Success:** routine requests resolved — password resets and access requests through the proper **approval workflow**, policy questions answered from HR docs — and anything sensitive escalated. **Volume (illustrative):** 800 requests/day.

### 11.2 Why an agent (or why only partly)

Almost entirely a **workflow**: classify → (answer from policy | run an approval workflow). The agentic surface is tiny — picking the right workflow and filling its fields. Access requests are never granted by the model; they enter an existing approval system, which is the whole safety design.

### 11.3 Architecture

```text
integration shape: streaming chat UI (chat bot) + approval workflow (durable)

request ─▶ [classify] ─▶ policy Q? ──▶ [RAG answer from HR/IT docs]
                      └▶ action?  ──▶ [agent fills workflow fields]
                                          │
                                          ▼
                             [identity + approval workflow]  <- existing system
                                          │ approved
                                          ▼
                             execute (reset / grant) ─▶ [audit log]
```

### 11.4 Tools

| Tool | R/W | Data it touches | Approval? | Size cap |
|---|---|---|---|---|
| `search_policy` | read | HR/IT policy docs | no | 5 chunks |
| `verify_identity` | read | identity provider | no | 1 check |
| `reset_password` | write | identity provider | **yes (workflow)** | self only |
| `request_access` | write | access workflow | **yes (approver)** | 1 request |
| `escalate` | write | human queue | no (safe) | 1 handoff |

### 11.5 Context and data

Retrieved: the relevant HR/IT policy passages ([51 §9](51-ai-rag.md#9-the-rag-prompt)). Remembered: the conversation only; no cross-employee memory. Never sent: credentials, security answers, or another employee's records — the agent fills an access *request*, it never reads the secret itself ([56 §9](56-ai-safety.md#9-secrets-in-prompts-and-logs)), and identity is checked per request so the agent acts with the *user's* rights, not admin ([56 §2.4](56-ai-safety.md#24-layered-defences)).

### 11.6 Autonomy and guardrails

Autonomy: **act with approval** for every write — a password reset goes through the identity provider's own self-service verification, and an access request goes to a human approver. The agent cannot grant access; it can only *route* a well-formed request into the workflow that a human approves. That separation is the control.

### 11.7 A sample trajectory

```text
Thought: user wants repo access; verify identity first.
Action: verify_identity({"user": "u_31"})
Observation: {"verified": true}

Thought: file the access request with the manager as approver.
Action: request_access({"user": "u_31", "resource": "repo:payments"})
Observation: {"error": "unknown resource id 'repo:payments'"}  <- tool error

Thought: resource id wrong; look it up and refile.
Action: search_policy({"q": "payments repo access id"})
Observation: resource id is "gh:payments-service"
Action: request_access({"user": "u_31", "resource": "gh:payments-service"})
Observation: {"request_id": "AR-90", "status": "pending_manager_approval"}

Answer: Access request AR-90 filed; your manager will approve or deny.
```

### 11.8 Evals and metrics

Measure resolution rate for routine requests, false-escalation, and — critical — zero unauthorized grants. Report resolution as a Wilson interval ([54 §8.1](54-ai-evaluation.md#81-confidence-interval-for-a-pass-rate-wilson)).

!!! example "On paper"

    228 of 240 routine requests resolved correctly, so \(\hat p = 0.95\), \(z=1.96\):

    \[
    \text{centre} = 0.9429
    \]

    \[
    \text{half-width} = 0.0283
    \]

    \[
    0.9429 \pm 0.0283 = (0.915,\ 0.971)
    \]

    92%–97% of routine requests handled end to end; the ones it cannot are escalated, and because grants need a human approver, a misclassification wastes a minute rather than opening a door.

### 11.9 Cost per task

!!! example "On paper"

    A short policy answer or form-fill: 2000 input and 500 output tokens, at \$0.50/\$1.50 per 1M:

    \[
    \frac{2000}{10^6}\times 0.50 = 0.00100
    \]

    \[
    \frac{500}{10^6}\times 1.50 = 0.00075
    \]

    \[
    \text{cost} = 0.00175
    \]

    \$0.00175 per request. The lever: an exact/semantic cache for the many identical policy questions ([55 §11.2](55-ai-production.md#112-semantic-cache)) — "how much parental leave" is asked a hundred times a week.

### 11.10 Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Unauthorized access granted | Grants go through a human approver; agent only files (§11.6). |
| Wrong policy answer | Answer from retrieved docs; cite; refuse when unsure ([51 §9](51-ai-rag.md#9-the-rag-prompt)). |
| Reset for the wrong user | Identity verification; self-service scope only. |
| Social-engineering via chat | Treat requests as untrusted; identity and approval are the gate ([56 §2](56-ai-safety.md#2-prompt-injection)). |

### 11.11 MVP → v2 → scale

**MVP:** a policy-Q&A bot over HR/IT docs — read-only, no actions. **v2:** route password resets and access requests into the *existing* approval workflows. **Scale:** more request types one at a time, each through an approver; a feedback loop on resolved tickets to grow the eval set.

---

## 12. Data-pipeline maintenance agent

### 12.1 The job

**Who:** a data-platform team. **Input:** a failed pipeline job (a DAG task), its logs, and data lineage. **Success:** the agent diagnoses the failure, proposes a fix as a **pull request with a test**, and never edits production data or the running pipeline directly. **Volume (illustrative):** 30 job failures/day across hundreds of pipelines.

### 12.2 Why an agent

Observation-driven diagnosis, like the on-call responder: read the failed task's logs, trace lineage to the upstream source, form a hypothesis, and propose a fix. The next lookup depends on the last — an agent ([52 §1.1](52-ai-agents.md#11-the-decision-rule)) — but the output is a PR, not a production change.

### 12.3 Architecture

```text
integration shape: durable workflow engine (resumable; triggered on job failure)

failure ─▶ [maintenance agent: mid model]  (checkpointed; survives restart)
             │  retrieval: job logs + DAG lineage + the pipeline code
             │  tools (read): get_logs, get_lineage, read_code, run_in_staging
             │  tools (write): open_pr (with a regression test)
             ▼
     diagnose ─▶ fix in a branch ─▶ run_in_staging ─▶ open_pr ─▶ [audit log]
guardrails: staging only; no prod writes; a human reviews and merges the PR
```

### 12.4 Tools

| Tool | R/W | Data it touches | Approval? | Size cap |
|---|---|---|---|---|
| `get_logs` | read | job log store | no | 50 lines |
| `get_lineage` | read | lineage graph | no | 20 nodes |
| `read_code` | read | pipeline repo | no | 400 lines |
| `run_in_staging` | read | staging run (sandbox) | no | summary |
| `open_pr` | write | the repo | human merges | 1 PR + test |

### 12.5 Context and data

Retrieved: the failed task's logs, the lineage to the upstream table, and the pipeline code. Remembered: the diagnosis transcript, checkpointed so a long run survives a worker restart ([52 §8.1](52-ai-agents.md#81-long-running-agents)). Never sent: the production *data* — the agent reads schemas, lineage metadata, and logs, not rows ([56 §9](56-ai-safety.md#9-secrets-in-prompts-and-logs)).

### 12.6 Autonomy and guardrails

Autonomy: **act and report** in staging (it can run a fix against a staging copy) and **suggest** for production (the PR is a proposal; a human merges, and the merge triggers the real pipeline change through the normal CD path). There is no tool that writes production data — the capability simply does not exist in the agent's surface, which is the cleanest form of least privilege ([52 §16](52-ai-agents.md#16-sandboxing-permissions-and-least-privilege)).

### 12.7 A sample trajectory

```text
Thought: the "load_orders" task failed; read its logs.
Action: get_logs({"task": "load_orders", "run": "r_900"})
Observation: SchemaError: column "currency" missing in source

Thought: upstream schema drifted. Trace lineage to the source.
Action: get_lineage({"table": "orders"})
Observation: {"error": "lineage service timeout"}            <- tool error

Thought: retry; lineage is needed to find the upstream owner.
Action: get_lineage({"table": "orders"})
Observation: source = "raw.orders_v2" (added "currency" last week)

Thought: add a nullable "currency" mapping + a regression test, run in staging.
Action: run_in_staging({"branch": "fix-currency"})
Observation: {"status": "green", "rows": 10432}
Action: open_pr({"branch": "fix-currency", "test": "test_currency_nullable"})

Answer: Opened PR #58 mapping the new "currency" column with a regression test;
staging run green. Needs review and merge.
```

### 12.8 Evals and metrics

Measure diagnosis accuracy against labelled past failures and the end-to-end task success over the diagnose→fix→test→PR flow ([52 §12](52-ai-agents.md#12-reliability-math), [54 §13](54-ai-evaluation.md#13-evaluating-rag-and-agents)).

!!! example "On paper"

    The flow is 5 steps, each succeeding independently with \(p=0.97\); end-to-end success:

    \[
    S = 0.97^5 = 0.8587
    \]

    Even at 97% per step, a 5-step flow lands at 86% — so the PR gate and the human reviewer catch the ~14% of runs that go wrong somewhere, and you shorten the flow where you can ([52 §12](52-ai-agents.md#12-reliability-math)).

### 12.9 Cost per task

!!! example "On paper"

    A log-and-lineage-heavy run: 15000 input and 3000 output tokens, at \$0.50/\$1.50 per 1M:

    \[
    \frac{15000}{10^6}\times 0.50 = 0.00750
    \]

    \[
    \frac{3000}{10^6}\times 1.50 = 0.00450
    \]

    \[
    \text{cost} = 0.01200
    \]

    \$0.012 per failure. The lever: summarize old log observations so the resend stops growing ([52 §13](52-ai-agents.md#13-cost-growth)); cache the pipeline-code and DAG prefix across the loop (§1).

### 12.10 Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Production data edited | No prod-write tool exists; staging only (§12.6). |
| Fix that masks the real bug | Require a regression test; human review of the PR. |
| Log flood over budget | Summarize; cap results; token budget ([52 §14](52-ai-agents.md#14-budgets)). |
| Long run lost on restart | Checkpoint + resume ([52 §8.1](52-ai-agents.md#81-long-running-agents)). |

### 12.11 MVP → v2 → scale

**MVP:** a read-only triager that posts a diagnosis and the likely upstream cause. **v2:** add the staging fix + PR-with-test once diagnosis accuracy is proven on past failures. **Scale:** a durable engine so long runs resume, lineage-aware retrieval, and a regression-rate gate on merged fixes.

---

## 13. QA / browser-testing agent

### 13.1 The job

**Who:** a QA team. **Input:** a user journey to exercise on a **staging** app ("sign up, add to cart, check out"). **Success:** the journey is driven end to end; when it breaks, a reproducible bug report is filed with steps, the failing assertion, and a screenshot. **Volume (illustrative):** 200 journeys/night in CI.

### 13.2 Why an agent (or why only partly)

Partly a **workflow** (the journey steps are mostly fixed) with an agentic recovery layer: when a selector moves or a page differs from the script, the model reads the screen (accessibility tree) and finds the element, instead of the brittle script failing outright ([52 §8.3](52-ai-agents.md#83-browser-and-computer-use-agents)). Browser agents are the highest-risk autonomy, so this one is strictly read-mostly and staging-only.

### 13.3 Architecture

```text
integration shape: scheduled/batch (nightly CI) + local harness driving a browser

journey ─▶ [QA agent: mid model] (staging only, read-mostly)
             │  observation: accessibility tree + screenshot
             │  tools: navigate, click, type, read_dom, screenshot, file_bug
             ▼
     step ─▶ observe ─▶ assert ──fail──▶ file_bug (steps + assertion + shot)
             │ pass
             ▼
     next step ... ─▶ report pass/fail ─▶ [audit log]
guardrails: staging URL allow-list; never run against production; no destructive data
```

### 13.4 Tools

| Tool | R/W | Data it touches | Approval? | Size cap |
|---|---|---|---|---|
| `navigate` | write-ish | staging URL (allow-list) | no (staging) | 1 page |
| `read_dom` | read | the page accessibility tree | no | 4k chars |
| `click` / `type` | write-ish | staging UI only | no (staging) | 1 action |
| `screenshot` | read | the page | no | 1 image |
| `file_bug` | write | issue tracker | no (safe) | 1 report |

### 13.5 Context and data

Retrieved: the journey spec and the current page's accessibility tree (cheaper and more reliable than a raw screenshot — [52 §8.3](52-ai-agents.md#83-browser-and-computer-use-agents)). Remembered: the steps taken, for the bug report's reproduction trail. Never sent: real user data — staging uses synthetic accounts, and a staging URL allow-list keeps it off production ([56 §6](56-ai-safety.md#6-excessive-agency)).

### 13.6 Autonomy and guardrails

Autonomy: **act and report** against staging (it clicks and types freely in a throwaway environment) and strictly read-only toward production (no production URL is on the allow-list). The guardrail is the allow-list plus synthetic data: the agent *cannot* reach production or touch a real account, so even a confused trajectory is contained ([52 §16](52-ai-agents.md#16-sandboxing-permissions-and-least-privilege)).

### 13.7 A sample trajectory

```text
Thought: start the checkout journey on staging.
Action: navigate({"url": "https://staging.shop.internal/cart"})
Observation: page loaded; cart has 1 item

Thought: click the checkout button.
Action: click({"selector": "#checkout"})
Observation: {"error": "no element matches #checkout"}        <- tool error

Thought: selector moved; read the DOM to find the real control.
Action: read_dom({})
Observation: button "Proceed to checkout" role=button id="co-btn"

Thought: use the accessibility label instead of the stale id.
Action: click({"selector": "#co-btn"})
Observation: checkout page; total $60
[harness] assert total == expected -> FAIL (expected $54 after discount)
Action: file_bug({"title": "Discount not applied at checkout", "steps": "...",
                  "expected": 54, "actual": 60, "shot": "co.png"})

Answer: Journey failed at checkout; filed bug QA-311 with steps and screenshot.
```

### 13.8 Evals and metrics

Measure journey pass rate, **flakiness** (does a journey pass consistently across reruns?), and false-bug rate. Flakiness is a reliability question — a journey must pass *every* rerun ([09 §5](09-evaluation.md#passk-and-passk)).

!!! example "On paper"

    A journey passes any single run with \(p=0.95\) independently; the chance it passes all 3 nightly reruns (pass\(^3\), the reliability view):

    \[
    \text{pass}^3 = 0.95^3 = 0.8574
    \]

    A "95% reliable" journey is green on all three runs only 86% of the time — so 14% of journeys flake at least once, and flakiness, not raw pass rate, is what wastes a QA team's morning.

### 13.9 Cost per task

!!! example "On paper"

    A journey with several DOM reads: 8000 input and 400 output tokens, at \$0.50/\$1.50 per 1M:

    \[
    \frac{8000}{10^6}\times 0.50 = 0.00400
    \]

    \[
    \frac{400}{10^6}\times 1.50 = 0.00060
    \]

    \[
    \text{cost} = 0.00460
    \]

    \$0.0046 per journey. The lever: feed the **accessibility tree**, not full screenshots, so each observation is small text not a large image; cache the app-shell prefix across steps (§1).

### 13.10 Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Ran against production | Staging URL allow-list; no prod URL exists (§13.6). |
| Flaky false bugs | Rerun before filing; require a stable repro; flakiness metric. |
| Brittle selectors | Use the accessibility tree; recover on selector miss ([52 §8.3](52-ai-agents.md#83-browser-and-computer-use-agents)). |
| Injection via page content | Treat page text as untrusted ([56 §2](56-ai-safety.md#2-prompt-injection)). |

### 13.11 MVP → v2 → scale

**MVP:** a scripted (non-agent) journey runner — fixed selectors, no model. **v2:** add the model's recovery layer for selector drift once it measurably reduces false failures. **Scale:** parallel journeys in CI, a flakiness gate, and screenshot-diffing for visual regressions — always staging, never production.

---

## 14. The twelve compared

| # | Application | Integration shape | Autonomy | Write tools? | Approval model | Main risk | Main metric |
|---|---|---|---|---|---|---|---|
| 1 | Support resolution | streaming chat UI + service | act & report / approval | yes (refund) | gate > \$50 | wrong refund | resolution rate |
| 2 | Coding in CI | queue worker (sandbox) | act & report / suggest | yes (sandbox, PR) | human merge | regression / test-gaming | pass@k |
| 3 | Data-analyst (SQL) | agent service (run ids) | act & report | no (read-only) | none (read-only) | destructive SQL | execution accuracy |
| 4 | On-call responder | queue worker + chat | act w/ approval | yes (allow-list) | per write | wrong remediation | diagnosis / unsafe=0 |
| 5 | Security (Kali) | local harness | act w/ approval | yes (`kali_exec`, confirmed) | confirm per [11 §5.1](11-kali-agent.md#51-confirmation-policy) | out-of-scope action | task success / unsafe=0 |
| 6 | Document processing | queue worker / batch | act & report | yes (post) | low-conf → review | confident wrong field | precision / auto-post error |
| 7 | Research & report | agent service (async) | act & report / suggest | no | none (read-only) | fabricated citation | groundedness |
| 8 | Sales / CRM | agent-as-tool + service | suggest / act w/ approval | yes (CRM) | human send; CRM gate | auto-sent email | draft acceptance |
| 9 | E-commerce returns | streaming chat UI + service | act & report / approval | yes (refund) | gate > \$100 | over-limit refund | task success / escalation |
| 10 | IT/HR helpdesk | streaming chat UI + workflow | act w/ approval | yes (via workflow) | approver | unauthorized grant | resolution rate |
| 11 | Data-pipeline | durable workflow engine | act & report / suggest | no prod; PR only | human merge | prod data edit | diagnosis / task success |
| 12 | QA / browser | scheduled/batch + harness | act & report | staging only | none (staging) | ran against prod | flakiness |

Read the table top to bottom and the pattern is unmistakable: no write tool is ever fully autonomous; every one is either read-only, sandboxed, gated by a limit, or gated by a human; and the "main metric" is a measurable thing you gate a release on, not a vibe.

---

## 15. Patterns that recur

Strip away the domains and the same five moves appear in all twelve:

- **Read-mostly tools.** The large majority of tools are reads; writes are few, narrow, and scoped to the one object in context. A tool that can read one order cannot dump the table ([52 §16](52-ai-agents.md#16-sandboxing-permissions-and-least-privilege), [56 §6](56-ai-safety.md#6-excessive-agency)). When a write capability is not needed, it is simply absent — the pipeline agent has no production-write tool at all.
- **Approval for writes, auto-run for reads.** The autonomy level is set per *action class*, not per application: read-only recon/lookup/query runs automatically; anything that spends, sends, deletes, or changes config is gated by a limit or a human ([52 §15](52-ai-agents.md#15-human-in-the-loop)). The gate is code — a deterministic check or an existing approval workflow — never a sentence in the prompt.
- **Budgets, always.** Steps, tokens, wall-clock, and money, enforced by your runtime, not requested of the model ([52 §14](52-ai-agents.md#14-budgets)). They bound both the compounding unreliability ([52 §12](52-ai-agents.md#12-reliability-math)) and the quadratic cost ([52 §13](52-ai-agents.md#13-cost-growth)).
- **A verification step before the consequence.** A SQL verifier, a test run, a confidence threshold, a groundedness check, a schema validator — an *external* check between the model's output and the side effect, because the model grading itself is as biased as its output ([52 §9](52-ai-agents.md#9-planning-and-reflection), [54 §4](54-ai-evaluation.md#4-execution-based-evaluation)).
- **An eval you trust before you raise autonomy.** Every "v2" above is gated on a number from a held-out set: wrong-refund rate near zero, pass@k high enough, execution accuracy inside a tight interval. You do not promote an agent from "suggest" to "act" on a hunch; you promote it on an eval ([54 §13](54-ai-evaluation.md#13-evaluating-rag-and-agents), [52 §17](52-ai-agents.md#17-evaluating-agents)).

And one meta-pattern: **most of these started life as a workflow.** The MVP is almost always a fixed chain or a single structured call with a human doing the risky part; the agent arrives only where the path genuinely cannot be enumerated and the eval proves the loop earns its cost ([52 §2.6](52-ai-agents.md#26-workflow-or-autonomous-agent), [61 §17](61-ai-agent-frameworks.md#17-how-to-choose-decision-tree-and-scenarios)).

---

## 16. Key takeaways

- Twelve applications, one template: job, why-an-agent, architecture, tools, data, autonomy, trajectory, evals, cost, failures, growth. Comparing across them teaches more than any one in isolation.
- The split is almost always the same: a **fixed workflow** does most of the work, and a **small observation-driven tail** is the only genuinely agentic part. Keep the autonomous surface small on purpose.
- Name the **integration shape** ([62](62-ai-agent-integration.md)) and the **autonomy level** ([52 §8](52-ai-agents.md#8-autonomous-agents)) for every feature; they determine the guardrails you owe.
- **Reads auto-run; writes are gated** — by a limit, a sandbox, an approval, or by not existing. No write tool here is fully autonomous.
- Put a **verification step** (verifier, test, threshold, groundedness) between the model and every consequence; prefer an external check over the model's self-assessment.
- Compute the **cost per task** with your own assumed prices, and cut it with a **cascade** and **prefix caching** — the same two levers in nearly every application.
- **Raise autonomy only when an eval on a held-out set says so.** "MVP as a workflow, agent at v2, scale when the number holds" is the safe path every application followed.
- The security application is authorized lab testing only, behind a deterministic policy layer and an air-gapped sandbox ([11](11-kali-agent.md), [12](12-safety-legal.md#1-authorization-is-a-precondition-not-a-prompt)) — summarized here, specified there.

## 17. Exercises

!!! question "Exercise 17.1"

    A support agent resolves 150 of 200 held-out tickets. Give the point estimate and say why you would not ship a prompt change that moves it to 152/200.

??? success "Answer"

    Point estimate \(150/200 = 0.75\). The Wilson 95% interval is about \((0.686, 0.805)\) — a half-width near 0.060. A move to \(152/200 = 0.76\) is well inside that interval, so it is noise on 200 tickets; you need more examples or a paired test ([54 §8.1](54-ai-evaluation.md#81-confidence-interval-for-a-pass-rate-wilson)) before calling it an improvement.

!!! question "Exercise 17.2"

    A coding agent gets 5 independent attempts on a bug and 3 produce a green diff. Compute pass@1 and pass@2, and say which matters for CI.

??? success "Answer"

    \(\text{pass@}1 = 3/5 = 0.6\). \(\text{pass@}2 = 1 - \binom{2}{2}/\binom{5}{2}\), which is \(1 - 1/10 = 0.9\). CI can afford a retry, so pass@2 (0.9) is the deployment-relevant number — the chance at least one of two attempts lands a green diff.

!!! question "Exercise 17.3"

    An e-commerce return flow has 4 steps, each succeeding independently with probability 0.95. What is the end-to-end success rate, and what is the cheapest way to raise it?

??? success "Answer"

    \(S = 0.95^4 = 0.8145\). The cheapest lever is usually to **shorten the flow** (fewer steps directly raises \(p^n\)); the complementary lever is to raise per-step \(p\) with verification or retries ([52 §12](52-ai-agents.md#12-reliability-math)).

!!! question "Exercise 17.4"

    A document agent's `total` field has 90 correct, 10 wrong, 20 missed. Compute precision, recall, and F1, and explain which one the confidence router is protecting.

??? success "Answer"

    Precision \(= 90/100 = 0.90\); recall \(= 90/110 = 0.818\); \(F_1 = 2(0.90)(0.818)/(0.90+0.818)\), which is \(1.472/1.718 = 0.857\). The router protects **precision among auto-posted records**: by sending low-confidence extractions to a human it keeps the 10% wrong rate out of the auto-posted set, trading some recall (more go to review) for fewer wrong postings.

!!! question "Exercise 17.5"

    An incident agent proposes a remediation on 50 incidents and the human rejects 4; across 120 runs it attempted 0 out-of-allow-list actions. Give the override rate and explain why you keep the approval gate despite the zero.

??? success "Answer"

    Override rate \(= 4/50 = 0.08\). "0 of 120" gives a Wilson 95% *upper* bound near 0.031 on the unsafe-action rate — zero observed is not zero possible, only bounded at ~3%. You keep the gate because a single wrong write during an incident is costly, and the bound does not rule one out ([54 §8.1](54-ai-evaluation.md#81-confidence-interval-for-a-pass-rate-wilson)).

!!! question "Exercise 17.6"

    A research report makes 50 claims; 47 are supported by their cited source. Compute groundedness and explain why the evaluator gate still blocks delivery.

??? success "Answer"

    Groundedness \(= 47/50 = 0.94\). Three unsupported claims remain, and an unsupported claim in a delivered report is exactly the failure the agent exists to avoid — so the evaluator blocks delivery until groundedness clears its threshold, rather than trusting a 94% draft ([54 §13](54-ai-evaluation.md#13-evaluating-rag-and-agents)).

!!! question "Exercise 17.7"

    An API agent step costs \$0.004; a cheap model costs \$0.0005 and you escalate 30% of tasks. What is the expected cost per task with a cascade, and the percentage saved?

??? success "Answer"

    \(E[\text{cost}] = 0.0005 + 0.3\times 0.004\), which is \(0.0005 + 0.0012 = 0.0017\). Against \$0.004 mid-only, that is a saving of \((0.004-0.0017)/0.004 = 0.575\), or **57.5%** — because only the escalated 30% pay the mid rate ([55 §13](55-ai-production.md#13-routing-and-model-cascades)).

!!! question "Exercise 17.8"

    You are asked to build an agent that "automatically fixes production incidents end to end, no human." Using the patterns of §15, give the design you would actually ship instead.

??? success "Answer"

    Ship **act with approval**, not full autonomy: read-only diagnosis runs automatically; every remediation is allow-listed and gated on a human yes that shows the exact action; hard budgets on steps/tokens/time; an audit log; and promotion of any single action from "suggest" to "approved-auto" only after an eval shows a low override rate and a zero unsafe-action rate. A wrong autonomous write during an incident is precisely the harm the gate prevents ([52 §15](52-ai-agents.md#15-human-in-the-loop), [56 §6](56-ai-safety.md#6-excessive-agency)).

!!! question "Exercise 17.9"

    A QA journey passes a single run with probability 0.95 independently. What is the chance it passes all 3 nightly reruns, and why is that the number QA cares about?

??? success "Answer"

    \(\text{pass}^3 = 0.95^3 = 0.8574\). QA cares about pass\(^3\) (all runs green) rather than a single pass because a journey that flakes on even one of three reruns wastes triage time and erodes trust in the suite — flakiness, not average pass rate, is the cost ([09 §5](09-evaluation.md#passk-and-passk)).

!!! question "Exercise 17.10"

    A pipeline agent's flow is 5 steps at 0.97 each. Compute end-to-end success and name two guardrails that catch the rest.

??? success "Answer"

    \(S = 0.97^5 = 0.8587\). About 14% of runs go wrong somewhere; the two guardrails that catch them are the **PR gate with a required regression test** (nothing reaches production without passing) and the **human reviewer who merges** — plus checkpointing so a long run resumes rather than restarts ([52 §8.1](52-ai-agents.md#81-long-running-agents)).

Next: [62 — Integrating agents into applications](62-ai-agent-integration.md) for the run contract and the minimal agent service behind these shapes; [52 — Agents](52-ai-agents.md) for the loop, budgets and autonomy they all build on; [54 — Evaluation](54-ai-evaluation.md) to build the eval that gates every autonomy increase.
