# 07 — Synthetic data: generating trajectories with DeepSeek and OpenCode

*Last verified: 25 October 2026. Prices and model IDs change; re-check the official pages before a large run.*

> This chapter uses the Kali/agent pipeline. For the **Java/Quarkus** variant — repo-grounded tasks, Maven tests and ArchUnit — see [21 — Java synthetic data](21-java-synthetic.md); the API, OpenCode and cost model carry over.

Public datasets teach general tool calling. They do not know **your** tool names, **your** scope policy, **your** lab, or the way a Kali command actually fails on your target. Synthetic data fills that gap — but only if it is generated and verified rigorously. Ungrounded synthetic data teaches a model to hallucinate confident command output, which is worse than no fine-tune.

This chapter describes a pipeline that generates candidate tasks and trajectories with a strong teacher (DeepSeek V4.1 Flash or V4-Pro), executes the commands in a disposable lab, keeps only verified trajectories, and converts them to the canonical JSONL from [`06-data-format.md`](06-data-format.md).

> **Safety.** Treat every generated command as hostile until it has run inside the sandbox. Never execute model output on the host, on a public network, or against a target you do not own. The pipeline below refuses to run without an explicit opt-in and a Docker-isolated lab network.

---

## 1. Why synthetic data, and where it fails

**It is good at:**

- teaching your exact tool schemas and argument conventions;
- producing multi-turn trajectories with realistic observations;
- covering the long tail of Kali tools and flags;
- generating negative examples (out of scope, ambiguous target, destructive request);
- creating enough volume to balance a small public corpus.

**It fails when:**

- the teacher invents tool output instead of executing;
- the teacher writes plausible commands that do not exist or do not run;
- the same five tasks are paraphrased 10,000 times;
- the data encodes the teacher's safety style rather than yours;
- there is no held-out evaluation, so you cannot tell improvement from memorization.

The pipeline is therefore **execution-first**: the teacher proposes, the sandbox executes, the teacher reflects, and a validator decides whether the trajectory is admissible.

---

## 2. DeepSeek API (October 2026)

### 2.1 Models and limits

| Model ID | Served by | Context | Max output | Concurrency |
|---|---|---|---:|---:|
| `deepseek-flash` | DeepSeek-V4.1-Flash-1017 | 1M | 384K | 2,500 |
| `deepseek-v4-pro` | DeepSeek-V4-Pro-0813 | 1M | 384K | 500 |

`deepseek-v4-flash` is a legacy alias billed at Flash price. The old `deepseek-chat` / `deepseek-reasoner` names are no longer listed; migrate old scripts.

Thinking is **on by default** (effort `high`). Reasoning tokens are billed as output. For high-volume generation, use `deepseek-flash` with thinking disabled or at low effort; reserve `deepseek-v4-pro` for hard trajectories and judging.

### 2.2 Pricing

USD per 1M tokens, off-peak / peak. Peak is 01:00–04:00 and 06:00–10:00 UTC, Monday–Friday, excluding Chinese public holidays; everything else is off-peak.

| | Flash off-peak | Flash peak | V4-Pro off-peak | V4-Pro peak |
|---|---:|---:|---:|---:|
| Input, cache hit | 0.003 | 0.006 | 0.022 | 0.044 |
| Input, cache miss | 0.15 | 0.30 | 0.66 | 1.32 |
| Output | 0.60 | 1.20 | 1.98 | 3.96 |

Source: <https://api-docs.deepseek.com/quick_start/pricing>.

### 2.3 JSON, tools and thinking

- **JSON output:** `response_format={"type": "json_object"}`. The model returns a JSON object, but you must still validate it against your schema. Arbitrary JSON Schema (`json_schema`) is not the supported mode; use the object mode plus a validator.
- **Tool calling:** OpenAI-compatible `tools` and `tool_choice`. Strict function schemas are available on the **beta** base URL (`https://api.deepseek.com/beta`) with `strict: true`; treat this as beta and always validate.
- **Thinking:** controlled per request; see the Thinking Mode guide. If you disable it, do not expect long-horizon plans to be as good.
- **OpenAI SDK compatibility:** use `base_url="https://api.deepseek.com"` and your API key.

### 2.4 Minimal client

```python
import json, os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

resp = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {"role": "system", "content": "You generate authorized-lab penetration-testing tasks as JSON."},
        {"role": "user", "content": "Propose 5 tasks against 10.10.10.0/24 using nmap, http_probe and kali_exec."},
    ],
    response_format={"type": "json_object"},
    temperature=0.7,
    max_tokens=4096,
)
data = json.loads(resp.choices[0].message.content)
print(json.dumps(data, indent=2))
```

### 2.5 Batch generation with retries and cost tracking

```python
import asyncio, json, os, random, time
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                     base_url="https://api.deepseek.com")

PRICE = {"in": 0.15, "out": 0.60}  # off-peak Flash, USD per 1M

async def generate(prompt, sem, stats, attempts=4):
    async with sem:
        for n in range(attempts):
            try:
                r = await client.chat.completions.create(
                    model="deepseek-flash",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.8,
                    max_tokens=8192,
                )
                u = r.usage
                stats["in"] += u.prompt_tokens
                stats["out"] += u.completion_tokens
                stats["usd"] += (u.prompt_tokens * PRICE["in"] +
                                 u.completion_tokens * PRICE["out"]) / 1e6
                return json.loads(r.choices[0].message.content)
            except Exception as e:
                if n == attempts - 1:
                    stats["failed"] += 1
                    return None
                await asyncio.sleep((2 ** n) + random.random())

async def main(prompts):
    sem = asyncio.Semaphore(32)
    stats = {"in": 0, "out": 0, "usd": 0.0, "failed": 0}
    results = await asyncio.gather(*(generate(p, sem, stats) for p in prompts))
    print(json.dumps(stats, indent=2))
    return results
```

Keep the concurrency at 16–64 until you know your account's limits. Log the request ID and the exact prompt hash for every row so a bad batch can be reproduced.

---

## 3. OpenCode as a headless generator

OpenCode is an agentic CLI that can drive a model, tools, MCP servers, and file edits. It is useful when you want **multi-step transcripts** rather than one-shot JSON.

### 3.1 Install and configure a DeepSeek provider

```bash
curl -fsSL https://opencode.ai/install | bash
opencode --version
export DEEPSEEK_API_KEY="sk-..."
```

`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "deepseek": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "https://api.deepseek.com",
        "apiKey": "{env:DEEPSEEK_API_KEY}"
      },
      "models": {
        "deepseek-flash": {"name": "DeepSeek V4.1 Flash"},
        "deepseek-v4-pro": {"name": "DeepSeek V4 Pro"}
      }
    }
  },
  "model": "deepseek/deepseek-flash",
  "permission": {
    "bash": "ask",
    "edit": "deny",
    "webfetch": "deny"
  }
}
```

### 3.2 Non-interactive runs

```bash
# One prompt, JSON events (good for parsing tool calls and messages)
opencode run -m deepseek/deepseek-flash --format json \
  "You are in an authorized lab. Propose a 4-step nmap → http_probe → report plan for 10.10.10.5. Do not execute anything."

# Continue a session
opencode run --session ses_abc123 --format json "Now turn that plan into tool calls."

# Use a custom agent with a restricted tool set
opencode run --agent lab-planner --format json "..."
```

`opencode run --help` (v2.1.4) confirms `--format json`, `--model`, `--agent`, `--session`, `--continue`, `--file`, `--thinking`, and `--auto`. Use `--format json` and parse the event stream; do not scrape the human UI.

### 3.3 Agents, skills, MCP and permissions

- **Agent:** a markdown file in `.opencode/agents/` with a model, prompt, and tool permissions. Create a `lab-planner` that can read files but not run bash, and a `lab-executor` that can only call the Kali MCP tool.
- **Skill:** a reusable instruction bundle in `.opencode/skills/`; useful for "how to write a finding" or "how to scope a request".
- **MCP:** configure the Kali MCP server under `mcp` in `opencode.json`; the model sees typed tools rather than a raw shell.
- **Permissions:** deny `edit` and `webfetch` for generation runs; require approval for `bash`; run inside the sandbox.

### 3.4 Exporting transcripts

`--format json` emits events; coalesce them into OpenAI-style messages:

- user prompt → `role:"user"`;
- assistant text → `role:"assistant"` with `content`;
- tool call → `role:"assistant"` with `tool_calls`;
- tool result → `role:"tool"`;
- reasoning blocks → `reasoning_content` (or drop them for a no-think adapter).

Write a small adapter rather than depending on an internal session format. Our `scripts/generate_synthetic.py` includes an OpenCode event normalizer skeleton.

**Limitation:** OpenCode is optimized for coding, not for generating thousands of independent JSON rows. Use it for a few hundred high-quality, tool-rich trajectories and for operating the lab; use the API for bulk generation.

---

## 4. Eight generation strategies

| # | Strategy | What it produces | Best for |
|---|---|---|---|
| 1 | **Self-instruct / persona** | Tasks from a role + constraints | breadth, easy volume |
| 2 | **Tool-graph traversal** | Tools → compatible tasks → calls | schema coverage |
| 3 | **Seed expansion from docs** | Tasks grounded in man pages, MITRE ATT&CK, CIS, Nmap/Nuclei docs | domain accuracy |
| 4 | **Execution-verified trajectories** | propose → run → observe → reflect | realism, recovery |
| 5 | **Rejection sampling / best-of-N** | keep the trace that passes a verifier | quality |
| 6 | **Distillation** | teacher plans, student format | hard reasoning |
| 7 | **Evol-Instruct** | increase constraints, ambiguity, depth | robustness |
| 8 | **Negative examples** | out-of-scope, destructive, missing tool, ambiguous | safety and no-call |

### 4.1 The task taxonomy

Generate against a matrix, not a single prompt:

- **Phase:** scoping, recon, enumeration, vulnerability validation, reporting.
- **Target type:** host, CIDR, web app, SMB/AD, database, wireless (lab only).
- **Tool family:** nmap, http, DNS, SMB, TLS, web fuzzing, credential audit (lab), packet capture.
- **Difficulty:** single call, 2–3 step chain, 5+ step with a failure, ambiguous request.
- **Safety:** in-scope, ambiguous scope, out-of-scope, destructive request, missing tool.

A balanced sample from this matrix is worth more than 10× the rows from one prompt.

### 4.2 Execution-verified loop

```text
seed task
   │
   ▼
teacher proposes next tool call (JSON)
   │
   ▼
validator: schema + tool exists + args in range + scope check
   │
   ▼
sandbox executor: docker exec, timeout, capture stdout/stderr/exit
   │
   ▼
teacher reflects on the observation and proposes the next step
   │
   ▼
stop at success / max turns / repeated failure
   │
   ▼
judge + human spot check → keep / repair / discard
```

Only the **executed** observation goes into the training row. Never let the teacher invent `22/tcp open ssh` because it “seems likely”.

### 4.3 Prompt templates

**Task generator (system):**

```text
You generate penetration-testing practice tasks for an isolated, authorized lab.
The lab network is 10.10.10.0/24. Tools available: nmap_scan, http_probe, kali_exec, report_finding.
Return JSON only:
{"tasks":[{"id":"...", "phase":"recon|enum|validate|report", "target":"...", "goal":"...",
           "constraints":["..."], "difficulty":"easy|medium|hard", "expected_tools":["..."]}]}
Do not propose actions against public IPs, domains, or anything outside 10.10.10.0/24.
```

**Trajectory generator (system):**

```text
You are the planner inside a lab agent. At each turn, either call one tool or finish.
Use only the tools provided. Arguments must match the schema.
If the observation shows an error, diagnose it before retrying.
If the request is out of scope, refuse and explain what authorization is needed.
Return a single JSON object:
{"thought":"...", "tool_call":{"name":"...","arguments":{...}}} or
{"thought":"...", "final":"..."}
```

**Judge (system):**

```text
Score the trajectory 1-5 on: tool correctness, argument correctness, use of observations,
recovery from failure, scope compliance, and report quality. Penalize invented output.
Return JSON: {"score":1-5,"pass":true|false,"reasons":["..."]}.
```

### 4.4 Negative examples are not optional

Generate at least four refusal families:

1. **Out of scope:** target is `8.8.8.8` or a public domain.
2. **Ambiguous authorization:** “scan my company’s network” with no scope file.
3. **Destructive:** “wipe the logs”, “delete the database”, “rm -rf”.
4. **Missing tool:** the request needs a capability not in the tool list; the model should say so, not invent a tool.

These examples teach the policy layer's expectations and give you an unsafe-action rate to measure.

---

## 5. The sandbox

The executor must be boring and locked down:

```yaml
# lab/docker-compose.yml
services:
  kali:
    image: kalilinux/kali-rolling
    command: ["sleep", "infinity"]
    networks: [labnet]
    cap_drop: ["ALL"]
    cap_add: ["NET_RAW"]          # nmap needs raw sockets; no NET_ADMIN
    security_opt: ["no-new-privileges:true"]
    read_only: true
    tmpfs: ["/tmp:size=64m", "/root:size=64m"]
    pids_limit: 128
    mem_limit: 1g
    cpus: 1.0

  dvwa:
    image: vulnerables/web-dvwa
    networks: [labnet]

  juice:
    image: bkimminich/juice-shop
    networks: [labnet]

networks:
  labnet:
    internal: true                # no route to the internet
```

Run with:

```bash
docker compose -f lab/docker-compose.yml up -d
docker compose -f lab/docker-compose.yml ps
```

Properties:

- **No egress:** `internal: true`; the lab cannot reach the internet, and the internet cannot reach it.
- **Disposable:** `docker compose down -v` resets; snapshot the VM for a stronger guarantee.
- **No host mounts:** do not mount `/`, the Docker socket, SSH keys, or the repo.
- **Resource limits:** CPU, memory, PID, and wall-clock timeout per command (e.g., 60 s).
- **Audit:** log every command, exit code, stdout/stderr hash, container ID, and timestamp outside the sandbox.
- **Allowlist:** only the tools you intend to teach; reject `curl | bash`, package installs, and writes outside `/tmp`.

For a stronger boundary, run the lab in a dedicated VM (UTM/Lima/Proxmox) with a host-only network and no shared folders.

---

## 6. Validation, dedup and quality gates

### 6.1 Hard gates (reject)

- invalid JSON / missing role / empty assistant;
- tool name not in the allowed set;
- arguments fail schema validation;
- command contains a destructive pattern or a public target;
- output contains a secret pattern (AWS key, private key, JWT, password assignment);
- execution timed out or the container crashed;
- more than N turns or more than M tokens;
- exact duplicate of an existing row.

### 6.2 Soft gates (score and rank)

- execution exit code 0 (or an expected, explained failure);
- the final answer references the observed output;
- the trajectory recovers from at least one failure (for hard examples);
- judge score ≥ 4/5;
- tool diversity and phase coverage;
- no repeated identical tool call more than twice.

### 6.3 Dedup

```python
from datasketch import MinHash, MinHashLSH

def minhash(text, n=5, num_perm=128):
    m = MinHash(num_perm=num_perm)
    words = text.lower().split()
    for i in range(len(words) - n + 1):
        m.update(" ".join(words[i:i+n]).encode())
    return m

lsh = MinHashLSH(threshold=0.8, num_perm=128)
for i, row in enumerate(rows):
    m = minhash(row["text"])
    if not lsh.query(m):
        lsh.insert(str(i), m)
        keep.append(row)
```

Also compute exact hashes of normalized messages and of the final assistant answer. Near-duplicate prompts with different tool outputs are still duplicates for training purposes; keep the best execution.

### 6.4 Splits and decontamination

- Split by **task family**, not by row, so paraphrases do not leak across train/test.
- Keep 10% of generated tasks as a held-out golden set; never train on them.
- Run 13-gram overlap against BFCL, KaliBench test, Cybench, NYU CTF and CyberMetric before training.
- Record the generator model, prompt hash, seed, temperature, and judge score in the dataset manifest.

---

## 7. Cost model

Assume a verified trajectory costs ~3,000 output tokens and ~1,500 input tokens (prompt + tool schemas), plus 30% retries/judging.

| Examples | Output tokens | Input tokens | Flash off-peak | V4-Pro off-peak |
|---:|---:|---:|---:|---:|
| 1,000 | 3.9M | 2.0M | ~$2.60 | ~$9.00 |
| 10,000 | 39M | 20M | ~$26 | ~$90 |
| 50,000 | 195M | 100M | ~$130 | ~$450 |
| 100,000 | 390M | 200M | ~$260 | ~$900 |

These are order-of-magnitude estimates; thinking tokens, retries and judge calls can double them. Use prompt caching for the stable system/tool block (cache-hit input is ~50× cheaper) and batch during off-peak hours. A 10k-example synthetic set is affordable; a 100k set is a real budget line.

**Do not generate 100k rows before you have trained on 2k and evaluated.** The first 2k will reveal schema and taxonomy mistakes that would otherwise be replicated 100k times.

---

## 8. Legal and terms

- **DeepSeek terms:** as of 25 Oct 2026, the terms permit using outputs to improve models, but review the current terms and your jurisdiction before commercial use. Do not assume “API output = no restrictions”.
- **OpenCode/provider terms:** the provider's terms apply to the model you configure. Log which provider generated which rows.
- **Dataset licenses:** synthetic rows generated by a model are not automatically free of the teacher's terms or of the source documents you seeded with. Keep seed provenance.
- **Personal data:** do not generate or retain real credentials, customer data, or live flags.
- **Attribution:** record the teacher model and version in the dataset card; it is good practice and often required.

This is not legal advice. For a commercial release, have counsel review the model and data licenses.

---

## 9. Failure modes

| Failure | Cause | Fix |
|---|---|---|
| Confident but wrong output | No execution; teacher invented observations | Execute every command; keep only real stdout/stderr. |
| Repetitive tasks | One prompt, high temperature | Use the taxonomy matrix; dedup; sample by phase/tool. |
| Model learns the teacher's refusals | Teacher policy leaked into data | Write your own scope policy; generate targeted negatives. |
| Commands fail in training but work in the lab | Different Kali version/tool flags | Pin the container image digest; record versions. |
| Context overflow | Tool schemas + 8 turns | Cap turns; trim tool list; 4K–8K max sequence. |
| Cost blow-up | Thinking + retries + judge | Disable thinking for easy splits; cache prompts; batch off-peak. |
| Evaluation looks great, lab fails | Contamination or same-family test set | Hold out by task family; execute the golden set. |

---

## 10. Sources

- DeepSeek pricing — <https://api-docs.deepseek.com/quick_start/pricing>
- DeepSeek JSON mode — <https://api-docs.deepseek.com/guides/json_mode>
- DeepSeek tool calls — <https://api-docs.deepseek.com/guides/tool_calls>
- DeepSeek thinking mode — <https://api-docs.deepseek.com/guides/thinking_mode>
- OpenCode docs — <https://opencode.ai/docs>
- OpenCode GitHub — <https://github.com/sst/opencode>
- APIGen (verifiable synthetic data) — <https://arxiv.org/abs/2406.18518>
- ToolACE (multi-agent synthesis) — <https://arxiv.org/abs/2409.00920>
- Self-Instruct — <https://arxiv.org/abs/2212.10560>
- MinHash/SimHash dedup — <https://arxiv.org/abs/2001.07219>
