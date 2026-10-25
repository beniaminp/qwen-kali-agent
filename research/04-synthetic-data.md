# 04 — Synthetic Agentic Data for a Kali Pentest Assistant

**Scope:** execution-verified trajectories and CLI QA for LoRA-tuning Qwen3-8B/14B on a 32 GB M1 Pro, generated with DeepSeek and OpenCode. Target format: MLX chat + tools JSONL.

**Safety:** every generated command is hostile until proven otherwise. Run it only in a disposable, isolated Kali lab against targets you own. Never on the host; never against third parties.

**Verified:** 25 Oct 2026; re-check prices/model IDs before scaling.

## 1. DeepSeek API (October 2026)

### Models, limits, pricing

| Model ID | Served by | Context | Max output | Concurrency |
|---|---|---|---:|---:|
| `deepseek-flash` | DeepSeek-V4.1-Flash-1017 | 1M | 384K | 2500 |
| `deepseek-v4-pro` | DeepSeek-V4-Pro-0813 | 1M | 384K | 500 |

Docs no longer list `deepseek-chat` or `deepseek-reasoner`; migrate old code. `deepseek-v4-flash` is a legacy alias billed at Flash price. Default output: 8K non-thinking, 64K thinking, 128K at max effort. Thinking is **on by default** (effort `high`); reasoning tokens bill as output.

USD per 1M tokens. Peak = 01:00–04:00 and 06:00–10:00 UTC Mon–Fri, excluding Chinese holidays; everything else is off-peak.

| | Flash off | Flash peak | V4-Pro off | V4-Pro peak |
|---|---:|---:|---:|---:|
| Input, cache hit | 0.003 | 0.006 | 0.022 | 0.044 |
| Input, cache miss | 0.15 | 0.30 | 0.66 | 1.32 |
| Output | 0.60 | 1.20 | 1.98 | 3.96 |

Off-peak is half price. Caching is automatic and prefix-based; read `usage.prompt_cache_hit_tokens` and `prompt_cache_miss_tokens`. Put stable content (system prompt, tool schemas, seed docs) first.

### JSON, schema, tools, thinking

- JSON Output is `response_format={"type":"json_object"}`. It guarantees a valid JSON *object*, not your schema. Include "json" plus an example in the prompt; occasionally returns empty content. There is **no `json_schema` response format**.
- Strict schema exists only in tool calling (Beta): base URL `https://api.deepseek.com/beta`, each tool `"strict": true`, every object property required, `additionalProperties: false`. Supported types include object/string/number/integer/boolean/array/enum/anyOf/`$ref`; `minLength`, `maxLength`, `minItems`, `maxItems` are unsupported.
- Thinking + tools: return `reasoning_content` on every later turn or the API returns 400. `tool_choice="required"` and named choice are unsupported in thinking mode. Temperature/penalties have no effect.
- Rate limits are account-level concurrency, not QPS: 429 above 2500/500. Retry 429/500/503 with jittered exponential backoff; never retry 400/401/402/422. Requests not started in 10 minutes are closed. OpenAI-compatible: the SDK above works; `/v1` is an alias.

Sources: <https://api-docs.deepseek.com/quick_start/pricing>, <https://api-docs.deepseek.com/api/create-chat-completion>, <https://api-docs.deepseek.com/guides/json_mode>, <https://api-docs.deepseek.com/guides/tool_calls>, <https://api-docs.deepseek.com/guides/thinking_mode>, <https://api-docs.deepseek.com/guides/kv_cache>, <https://api-docs.deepseek.com/quick_start/rate_limit>

### Minimal client: tools + JSON mode

```python
# pip install openai
import json, os
from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
# For strict tools use base_url="https://api.deepseek.com/beta" and "strict": True.

tools = [{"type": "function", "function": {
    "name": "run_kali",
    "description": "Run one allowlisted command in the isolated Kali lab.",
    "parameters": {"type": "object",
        "properties": {"command": {"type": "string"},
                       "timeout_s": {"type": "integer", "minimum": 1, "maximum": 120}},
        "required": ["command", "timeout_s"], "additionalProperties": False}}}]

msgs = [{"role": "system", "content": "Lab-only assistant. Call run_kali; never invent output."},
        {"role": "user", "content": "Enumerate HTTP on 10.10.0.5; answer as json."}]

r = client.chat.completions.create(
    model="deepseek-flash", messages=msgs, tools=tools, tool_choice="auto",
    max_tokens=4096, extra_body={"thinking": {"type": "disabled"}})
print(r.choices[0].message.tool_calls[0].function.arguments)

r2 = client.chat.completions.create(
    model="deepseek-flash",
    messages=[{"role": "system", "content": 'Return json only, e.g. {"ok":true,"summary":"..."}'},
              {"role": "user", "content": "22/tcp ssh; 80/tcp http."}],
    response_format={"type": "json_object"}, max_tokens=1024,
    extra_body={"thinking": {"type": "disabled"}})
record = json.loads(r2.choices[0].message.content)  # always validate
```

### Batch generation with retry, backoff, cost tracking

```python
import asyncio, os, random, time
from openai import AsyncOpenAI

PRICES = {"deepseek-flash": (0.003, 0.15, 0.60),   # off-peak: hit, miss, output
          "deepseek-v4-pro": (0.022, 0.66, 1.98)}
client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                     base_url="https://api.deepseek.com", max_retries=0)
USED = {"usd": 0.0, "calls": 0, "hit": 0, "miss": 0, "out": 0}

async def one(model, messages, tools=None, max_tokens=4096):
    delay = 1.0
    for attempt in range(8):
        try:
            kw = dict(model=model, messages=messages, max_tokens=max_tokens,
                      extra_body={"thinking": {"type": "disabled"}})
            if tools: kw["tools"] = tools
            r = await client.chat.completions.create(**kw)
            u = r.usage
            ch, cm, o = PRICES[model]
            USED["usd"] += (u.prompt_cache_hit_tokens*ch +
                            u.prompt_cache_miss_tokens*cm + u.completion_tokens*o)/1e6
            USED["calls"] += 1
            USED["hit"] += u.prompt_cache_hit_tokens
            USED["miss"] += u.prompt_cache_miss_tokens
            USED["out"] += u.completion_tokens
            return r
        except Exception as e:
            if getattr(e, "status_code", None) in (400, 401, 402, 422) or attempt == 7:
                raise
            await asyncio.sleep(delay + random.random()*delay)
            delay = min(delay*2, 60)

async def batch(prompts, model="deepseek-flash", concurrency=32):
    sem = asyncio.Semaphore(concurrency)
    async def run(p):
        async with sem:
            return await one(model, [{"role": "user", "content": p}])
    t0 = time.time()
    out = await asyncio.gather(*(run(p) for p in prompts), return_exceptions=True)
    print({"seconds": round(time.time()-t0, 1), **USED})
    return out
```

Use 32–64 concurrency, not the account maximum (the key is shared), and log `response.model` + `system_fingerprint` per record.

## 2. OpenCode as a headless generator

OpenCode is an MIT-licensed coding agent (terminal/desktop/IDE) built on the AI SDK and Models.dev. V1/V2 share the `opencode` command; DeepSeek recommends ≥1.18.30 (V1). Pin one major.

```bash
curl -fsSL https://opencode.ai/install | bash     # or: npm i -g opencode-ai
opencode auth login                                # provider: deepseek
opencode models deepseek                           # confirm IDs
```

`~/.config/opencode/opencode.json` or project `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "deepseek/deepseek-flash",
  "small_model": "deepseek/deepseek-flash",
  "provider": {"deepseek": {"options": {
    "baseURL": "https://api.deepseek.com",
    "apiKey": "{env:DEEPSEEK_API_KEY}", "timeout": 600000}}},
  "permission": {"bash": "deny", "edit": "deny", "webfetch": "deny",
                 "external_directory": "deny", "kali_*": "allow"}
}
```

### Headless runs and export

```bash
opencode run --model deepseek/deepseek-flash --agent pentest-generator \
  --format json --auto "Solve task.json using only kali_* tools." \
  | tee runs/0001.events.jsonl
opencode export ses_abc123 > runs/0001.session.json   # {"info":...,"messages":[...]}
opencode stats --days 7 --models 5
```

`--format json` emits one JSON object per line (`type`: `text`, `reasoning`, `tool_use`, `step_start`, `step_finish`, `error`; plus `timestamp`, `sessionID`). Without `--auto`, non-interactive runs auto-reject permission requests; with `--auto`, anything not explicitly denied is approved. For many runs, `opencode serve` once and `opencode run --attach http://localhost:4096` to avoid MCP cold starts. `opencode export --sanitize` redacts transcript text and paths.

### Agents, skills, MCP, permissions

Agents are markdown files in `.opencode/agents/` (or JSON under `agent`). `mode: primary|subagent|all`; `steps` caps iterations; `permission.task` limits which subagents can be called; the last matching permission rule wins. A critic uses `mode: subagent`, read-only permissions, and V4-Pro; the primary calls it via Task.

```markdown
---
description: Generates one execution-verified Kali trajectory
mode: primary
model: deepseek/deepseek-v4-pro
temperature: 0.2
permission: {bash: deny, edit: deny, webfetch: deny, "kali_*": allow}
---
Lab-only pentest instructor. Target 10.10.0.0/24 only. One kali_* call at a
time; after each result state what it proves and the next step. Never invent
output. Stop at the objective or 12 calls.
```

Skills are `SKILL.md` folders under `.opencode/skills/<name>/` with required `name` and `description`; the body loads on demand via the `skill` tool. A `kali-tool-schemas` skill with exact flags and output parsers improves validity cheaply.

MCP servers live under `mcp`; local tools are prefixed by server name:

```json
{"mcp": {"kali": {"type": "local",
  "command": ["ssh", "-i", "/lab/id_ed25519", "kali@10.10.0.10", "mcp-server"],
  "timeout": 120000, "enabled": true}}}
```

Kali ships `mcp-kali-server`: `apt install mcp-kali-server`, run `kali-server-mcp` (HTTP on 127.0.0.1:5000), then `mcp-server` (stdio bridge). It exposes raw command execution plus nmap/gobuster/nikto/sqlmap/hydra/john/metasploit/wpscan helpers. It is **not** an allowlist: keep it inside the disposable VM, bind to localhost, and put your own policy in front.

### Export to training JSONL

`opencode export` assistant parts are `text`, `reasoning`, or `tool`; a completed tool has `name`, `state.input`, and `state.content` blocks.

```python
import json, sys

def convert(path):
    data = json.load(open(path)); out = []
    for m in data["messages"]:
        if m["type"] == "user":
            out.append({"role": "user", "content": m["text"]})
        elif m["type"] == "assistant":
            text, calls, results = "", [], []
            for p in m.get("content", []):
                if p["type"] == "text": text += p["text"]
                elif p["type"] == "tool" and p["state"]["status"] == "completed":
                    calls.append({"id": p["id"], "type": "function",
                        "function": {"name": p["name"],
                                     "arguments": json.dumps(p["state"]["input"])}})
                    body = "\n".join(c.get("text", "") for c in p["state"].get("content", [])
                                     if isinstance(c, dict))
                    results.append({"role": "tool", "tool_call_id": p["id"],
                                    "content": body[:20000]})
            msg = {"role": "assistant", "content": text}
            if calls: msg["tool_calls"] = calls
            out.append(msg); out.extend(results)
    return {"messages": out}

if __name__ == "__main__":
    json.dump(convert(sys.argv[1]), open(sys.argv[2], "w"), ensure_ascii=False)
```

Qwen3 renders tool calls as JSON inside `<tool_call>...</tool_call>`. Let the tokenizer template do it: verify with `tokenizer.apply_chat_template(..., tools=...)` on your installed `transformers`/`mlx-lm`. Strip `reasoning` for non-thinking training; keep it only if the template supports `reasoning_content`. Attach the exact `tools` registry at the top level; exports do not reliably include it.

**Limits:** token growth (10 calls ≈ 20–60K input), nondeterminism (record fingerprint), auto-reject vs `--auto`, host `bash`/sandbox need, V1/V2 churn, secrets in exports.

Sources: <https://opencode.ai/docs/>, <https://opencode.ai/docs/cli/>, <https://opencode.ai/docs/config/>, <https://opencode.ai/docs/agents/>, <https://opencode.ai/docs/skills/>, <https://opencode.ai/docs/mcp-servers/>, <https://opencode.ai/docs/permissions/>, <https://www.kali.org/tools/mcp-kali-server/>

## 3. Synthetic data strategies

Use a portfolio: cheap generation for coverage, execution for truth, rejection sampling for quality, negatives for policy.

Trade-offs: self-instruct (cheap, unverifiable), tool-graph (valid order, needs graph), docs (real syntax, licensing), execution (truth, sandbox cost), rejection (quality, N×/verifier bias), distillation (reasoning, teacher errors), Evol-Instruct (complexity, drift), negatives (scope, over-refusal).

**(a) Self-Instruct / persona.** Bootstrap 100–300 human seeds; vary persona (SOC analyst, sysadmin, junior pentester). Prompt:

```text
Generate {n} requests a user would send a Kali assistant. Persona: {persona}.
Lab only: 10.10.0.0/24. Mix 40% single-command QA, 40% multi-step investigation,
10% ambiguous, 10% out-of-scope. No public IPs/domains; no DoS, malware,
persistence, or credential theft. Targets 10.10.0.x or *.lab.test. Return json:
{id, persona, request, difficulty, expected_tools, why_realistic}.
```

Filter invalid and too-similar items as in Self-Instruct (<https://arxiv.org/abs/2212.10560>).

**(b) Tool-graph traversal.** Nodes are tool schemas; edges mean output of A feeds B. Sample a path, then write a ticket whose solution is that path (ToolLLM/FireFly/Trajectory2Task). Example: `nmap -sT -sV → curl -I → gobuster dir → nikto → sqlmap --batch`. The ticket must not name the tools. Do not invent `exploit → root`.

**(c) Docs expansion.** Sources: Kali tool pages, man pages, Nmap guide, Nuclei docs, PortSwigger, MITRE ATT&CK, CIS Benchmarks. Extract facts, not prose:

```text
SOURCE: {short_excerpt}
Write 5 Q/A pairs a junior pentester would ask. Cover what the flag changes,
when NOT to use it, how to read output, and one common mistake. Paraphrase;
do not copy sentences. Return json.
```

**(d) Execution-verified loop.** Propose one command → policy validate → sandbox execute → capture stdout/stderr/exit/duration → model reflects → repeat. Prompt for each turn:

```text
RESULT: exit={exit} stdout={stdout[:4000]} stderr={stderr[:2000]}
PLAN: {plan}
Update the plan; name any assumption the result contradicted. Then emit exactly
one command as json {"command":"...","timeout_s":60,"why":"..."} or
{"done":true,"evidence":"..."}.
```

Accept only when a task-specific predicate holds (port open, file found, login succeeds, hash matches). Exit 0 alone is not success: `nikto` exits 0 with no findings.

**(e) Rejection sampling.** Sample N=4–8 at T=0.7, execute each fresh, score 1–5 on faithfulness, validity, execution, evidence, safety, teaching. Keep mean ≥4.0 and safety=5; 4× generation can still lower cost per accepted example.

**(f) Distillation.** Use local 4-bit Gemma 3 27B/Qwen3-32B or a frontier API for references, critiques, and hard variants; keep only execution-passing outputs. Phi-4/Nemotron-4 show filtering matters more than volume (<https://arxiv.org/abs/2412.08905>, <https://arxiv.org/abs/2406.11704>). Check teacher licenses.

**(g) Evol-Instruct.** Rewrite a passing seed along one axis:

```text
Apply ONE operation: deepen (one more evidence step), broaden (second service),
constrain (forbid the obvious tool), concretize (add a 403/timeout/noise), or
distract (irrelevant finding). Keep it solvable in the lab. No illegal targets
or destructive actions. Return the task, axis, and expected new tool sequence.
```

Re-execute every evolved task; complexity often creates impossible preconditions. Source: <https://arxiv.org/abs/2304.12244>.

**(h) Negatives.** Four balanced categories: out-of-scope target, destructive command, missing tool, ambiguous target. Desired behavior: short scope check → refusal or one clarifying question → safe lab alternative. Include benign adjacent requests that should be answered, or the model learns to over-refuse.

## 4. Production pipeline

```text
taxonomy → seeds → generator → sandbox executor → validator → dedup
                                                        ↓
dataset card ← split/decontam ← judge/rubric ←──────────┘
```

Each stage writes an immutable artifact + manifest (git SHA, config hashes, counts); never overwrite `raw/`.

```text
synthetic/
  configs/{taxonomy,tools,policies,generation}.yaml
  seeds/{handwritten,kali_tools,mitre_attack,cis_checks}.jsonl
  prompts/{generator,critic,evolve,negative,judge}.md
  src/{generate,execute,validate,dedup,judge,split,export_mlx}.py
  runs/<run_id>/{raw,candidates,executions,valid,dedup,scored}/
  data/{train,valid,test}.jsonl  data/dataset_card.md
  lab/{docker-compose.yml,Dockerfile.kali,policy.py}
```

Stages: taxonomy (success predicate per slot), seeds (provenance + license), generate (valid JSON, one command/step), execute (isolated, timeout, audit), validate (schema/allowlist/CIDR/predicate), dedup (exact + MinHash ≤0.8), judge (mean ≥4.0, safety 5), split (grouped, no family leakage), decontam (13-gram <0.5).

### Validator

```python
import ipaddress, re, shlex
from jsonschema import validate, ValidationError

ALLOWED = {"nmap": {"-sT","-sV","-Pn"}, "curl": {"-I","-s"}, "gobuster": {"dir"},
           "nikto": {"-h"}, "sqlmap": {"-u","--batch"}}
LAB = ipaddress.ip_network("10.10.0.0/24")
META = re.compile(r"[;&|`$<>]")

def check_step(s):
    cmd = s.get("command", "")
    if META.search(cmd): return ["shell metacharacter"]
    try: toks = shlex.split(cmd)
    except ValueError as e: return [f"unparseable: {e}"]
    if not toks or toks[0] not in ALLOWED: return ["binary not allowed"]
    if any(t.startswith("-") and t not in ALLOWED[toks[0]] for t in toks[1:]):
        return [f"flag not allowed for {toks[0]}"]
    try:
        if ipaddress.ip_address(s.get("target","")) not in LAB:
            return ["target outside lab"]
    except ValueError: return ["target not literal IP"]
    return []

def validate_example(ex, schema):
    try: validate(ex, schema)
    except ValidationError as e: return False, [str(e.message)]
    errs = [e for step in ex["trajectory"] for e in check_step(step)]
    return not errs, errs
```

Reject `sudo`, `ssh`, `docker`, installs, writes outside `/work`, and pipes/redirects. Use `bashlex` if pipes are required; never regex shell parsing.

### Dedup and splits

```python
from datasketch import MinHash, MinHashLSH

def mh(text, k=5):
    w = text.lower().split()
    m = MinHash(num_perm=128)
    for s in {" ".join(w[i:i+k]) for i in range(max(0, len(w)-k+1))}:
        m.update(s.encode())
    return m

lsh, kept = MinHashLSH(threshold=0.8, num_perm=128), []
for i, ex in enumerate(examples):
    key = ex["task"] + "\n" + "\n".join(s["command"] for s in ex["trajectory"])
    m = mh(key)
    if lsh.query(m): continue
    lsh.insert(f"ex{i}", m); kept.append(ex)
```

Also hash normalized `(tool, argv, target)` sequences exactly. Threshold 0.8 is a starting point; inspect 200 pairs at 0.7/0.8/0.9. Split by **seed family + target + tool set**, not randomly; 80/10/10. Ensure every tool and refusal category appears in valid/test. Decontaminate with 13-gram overlap and embeddings against CyberGym, SecBench, NYU CTF Bench, plus your own eval; >0.5% hit rate means leakage.

## 5. Sandbox design

Assume public-IP scans, `rm -rf`, reverse shells, fork bombs, socket escapes. Blast radius: one disposable container.

**Layers:** dedicated VM (UTM/Lima) or separate host → Docker Kali → `internal: true` network → per-run `--rm` container → `--cap-drop ALL`, `no-new-privileges`, non-root, read-only root, tmpfs, pids/memory/CPU limits, wall-clock timeout → no host network, no Docker socket, no `--privileged`. No `NET_RAW` means `nmap -sS` fails; use `-sT`. That also blocks spoofing and many DoS paths.

```yaml
# lab/docker-compose.yml
name: kali-lab
networks:
  labnet:
    driver: bridge
    internal: true
    ipam: {config: [{subnet: 10.10.0.0/24}]}
services:
  kali:
    build: {context: ., dockerfile: Dockerfile.kali}
    command: ["sleep", "infinity"]
    networks: {labnet: {ipv4_address: 10.10.0.10}}
    read_only: true
    tmpfs: ["/tmp:size=256m,mode=1777", "/run:size=64m", "/work:size=512m,mode=1777"]
    cap_drop: [ALL]
    security_opt: ["no-new-privileges:true"]
    pids_limit: 256
    mem_limit: 2g
    cpus: 2.0
    ulimits: {nofile: {soft: 1024, hard: 2048}}
    # no ports, no host network, no docker.sock
  dvwa:
    image: vulnerables/web-dvwa
    networks: {labnet: {ipv4_address: 10.10.0.20}}
    mem_limit: 512m
  juice:
    image: bkimminich/juice-shop
    networks: {labnet: {ipv4_address: 10.10.0.21}}
    mem_limit: 1g
```

Build Kali with only the taxonomy tools and pin the image digest. On Apple Silicon Kali is arm64; some targets/tools are x86-only and slow under emulation (`docker manifest inspect` first). Prefer DVWA/Juice Shop; use an x86 VM only for specific cases.

**Wrapper.** Run with `docker run --rm --network kali-lab_labnet --cap-drop ALL --security-opt no-new-privileges --read-only --tmpfs /tmp --tmpfs /work --pids-limit 256 --memory 2g --cpus 2 --user 1000:1000 <image> timeout 60 <argv>`. Prefer `subprocess.run(shlex.split(cmd), shell=False)` after validation; reserve `bash -lc` for a reviewed advanced split. Capture command, argv, target, exit, stdout/stderr (truncated), duration, `docker diff`, and hash.

**Reset and audit.** `docker compose down -v && docker compose up -d` between trajectories; wait for health checks. Snapshot the clean VM and roll back after each batch (UTM/Lima snapshot commands vary by version — UNVERIFIED). Before every batch, assert `curl -m 3 https://example.com` fails from Kali and `curl -m 3 http://10.10.0.20` succeeds; abort otherwise.

**Targets:** DVWA (GPL-3.0), Juice Shop (MIT), Metasploitable, VulnHub images. Only images you built or deliberately downloaded, only on `labnet`. Never point generated commands at public IPs, real domains, or third-party infrastructure; never run `ssh`, reverse shells, `curl | sh`, package installs, or `sudo` from generated text. The lab observes tool behavior, not reusable exploit capability.

## 6. Quality control

| Gate | Check | Reject if |
|---|---|---|
| Schema | JSON Schema on every record | any error |
| Tools | name ∈ registry; args match schema | unknown/missing/wrong type |
| Safety | allowlisted binary/flags/CIDR; no metacharacters | any violation |
| Execution | exit, timeout, task predicate | timeout or predicate false |
| Consistency | tool result equals captured output | fabricated/mismatched |
| Dedup | exact hash; MinHash Jaccard | duplicate |
| Judge | rubric mean, safety | mean < 4.0 or safety < 5 |

**Success is a predicate, not exit 0.** Define `port_open`, `http_status == 200`, `file_contains("flag{")`, `nmap_xml_has_service("http")`. Store the predicate and evidence. A zero exit with no evidence fails; a non-zero exit that yields the needed observation is labeled "observation", not "success".

**Dedup:** exact SHA-256 of normalized `(tool, argv, target)` + normalized task; MinHash 5-word shingles, 128 permutations, Jaccard ≥ 0.8; secondary embedding cosine ≥ 0.92. Dedup on intent + tool sequence, not wording alone. Over-dedup removes valuable flag variations.

**Diversity:** ≥80% taxonomy tools, ≥150 tool bigrams/5k, no intent family >25%, ≥5 services, p50 4–6 steps/p95 ≤12, each refusal category 10–20%, no binary >35%; track per split.

**Human rubric** (100–200 stratified samples, two blind reviewers, 1–5):

| Axis | 1 | 3 | 5 |
|---|---|---|---|
| Realism | artificial | plausible | exactly what a junior pentester asks |
| Command validity | wrong flag | runs with warning | idiomatic, minimal |
| Evidence | fabricated | partly grounded | every claim traceable |
| Safety/scope | unsafe | ambiguous | clearly lab-only |
| Teaching value | noisy | adequate | teaches diagnosis |

Accept mean ≥4.0, safety =5, no axis <3. κ <0.6 means rewrite the rubric.

**Contamination:** build 300–500 human eval tasks before generation; check 13-gram (>0.5 reject), embeddings (>0.9), exact commands; hold out entire tools/targets in test. Compare against CyberGym, SecBench, NYU CTF Bench and your own eval.

**How many examples?** Heuristics, not laws:

| Split | Examples | Tokens |
|---|---:|---:|
| Tool-call trajectories | 3,000–10,000 verified | 15–40M |
| Cyber CLI QA | 2,000–5,000 | 2–6M |
| Refusals/scope | 500–1,500 | 0.5–1.5M |
| Valid / test | 500–1,000 each | — |

FireAct used 500 GPT-4 trajectories, AgentTuning ~1.8k, LIMA 1k. Start with 3k, train 1–2 epochs, measure held-out execution, then add data where errors cluster.

## 7. Legal and terms

**DeepSeek.** The Open Platform ToS (22 Apr 2026), §4.2, assigns output rights and permits “training other models (such as model distillation).” The consumer Terms §3.1 requires AI-disclosure and accuracy checks for published outputs; apply that in the dataset card. Inputs may improve the service unless you opt out; never send PII or credentials. PRC law governs; enforceability abroad is **UNVERIFIED**.

Sources: <https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html>, <https://cdn.deepseek.com/policies/en-US/deepseek-terms-of-use.html>

**OpenCode.** The OSS is MIT; the MIT license exclusively governs the OSS build. Hosted Zen/Go are separate commercial terms. Local use with your own key = MIT + DeepSeek API terms.

Sources: <https://github.com/anomalyco/opencode/blob/dev/LICENSE>, <https://opencode.ai/legal/terms-of-service>

| Source | License/status | Rule |
|---|---|---|
| MITRE ATT&CK | royalty-free, commercial OK, notice required | include “© 2026 The MITRE Corporation…” |
| Nmap | NPSL; commercial redistribution restricted; guide CC BY 3.0 | facts/short excerpts; no binaries |
| Nuclei | MIT | attribution; check template licenses |
| DVWA / Juice Shop | GPL-3.0 / MIT | target use, not dataset text |
| Metasploit | GitHub NOASSERTION | verify LICENSE; no code redistribution |
| Kali docs / Burp / CIS | © OffSec / proprietary / restricted | paraphrase facts only; no bulk copying |
| VulnHub | varies per image | check each |

Assignment does not cure third-party copyright: paraphrase, record provenance, attribute. Cyber data is dual-use: add a dataset-card restriction to authorized lab/education use and compliance with local law.

**UNVERIFIED:** copyrightability of generated command text; exact CIS redistribution terms; Metasploit's current SPDX license; local enforceability of DeepSeek's assignment. Get counsel before commercial redistribution.

## 8. Cost model

Assumes 8K input + 3K output per example, 60% cache hit, 1.3× retries, 1.25× verifier. OpenCode’s resend behavior can double input; thinking mode can 2–5× output.

| Scenario | $/example | 10k | 50k | 100k |
|---|---:|---:|---:|---:|
| Flash off-peak | 0.0037 | $37 | $186 | $373 |
| Flash peak | 0.0075 | $75 | $373 | $746 |
| V4-Pro off-peak | 0.0133 | $133 | $663 | $1,326 |
| V4-Pro peak | 0.0265 | $265 | $1,326 | $2,651 |

At 70% hit, Flash off-peak base ≈ $0.0022/example. All-in 100k **accepted** examples with 50% rejection, OpenCode overhead, and a Pro verifier: **$800–$2,500**. Engineering dominates.

**Cut cost:** run off-peak (half price; weekends fully off-peak); keep the prefix cacheable (hit is ~50× cheaper than miss on Flash); route easy splits to Flash, hard trajectories/judging to Pro (10–20%); validate/execute before judging; dedup before judging; cap output at 2–4K; truncate tool output to 4–8K chars; stop on the first disallowed command. Track dollars per **accepted, verified** example.

## Key sources

**DeepSeek:** [pricing](https://api-docs.deepseek.com/quick_start/pricing) · [chat API](https://api-docs.deepseek.com/api/create-chat-completion) · [JSON](https://api-docs.deepseek.com/guides/json_mode) · [tools/strict](https://api-docs.deepseek.com/guides/tool_calls) · [thinking](https://api-docs.deepseek.com/guides/thinking_mode) · [cache](https://api-docs.deepseek.com/guides/kv_cache) · [rate limits](https://api-docs.deepseek.com/quick_start/rate_limit) · [errors](https://api-docs.deepseek.com/quick_start/error_codes) · [OpenCode integration](https://api-docs.deepseek.com/quick_start/agent_integrations/opencode)

**OpenCode:** [intro](https://opencode.ai/docs/) · [CLI](https://opencode.ai/docs/cli/) · [config](https://opencode.ai/docs/config/) · [agents](https://opencode.ai/docs/agents/) · [skills](https://opencode.ai/docs/skills/) · [MCP](https://opencode.ai/docs/mcp-servers/) · [permissions](https://opencode.ai/docs/permissions/) · [server](https://opencode.ai/docs/server/) · [source](https://github.com/anomalyco/opencode/tree/dev/packages/cli/src)

**Kali lab:** [mcp-kali-server](https://www.kali.org/tools/mcp-kali-server/) · [Kali docs](https://www.kali.org/docs/)

**Papers:** [Self-Instruct 2212.10560](https://arxiv.org/abs/2212.10560) · [WizardLM 2304.12244](https://arxiv.org/abs/2304.12244) · [ToolLLM 2307.16789](https://arxiv.org/abs/2307.16789) · [AgentTuning 2310.12823](https://arxiv.org/abs/2310.12823) · [FireAct 2310.05915](https://arxiv.org/abs/2310.05915) · [Nemotron-4 2406.11704](https://arxiv.org/abs/2406.11704) · [Phi-4 2412.08905](https://arxiv.org/abs/2412.08905) · [LIMA 2305.11206](https://arxiv.org/abs/2305.11206) · [Trajectory2Task 2601.20144](https://arxiv.org/abs/2601.20144) · [FireFly 2605.17558](https://arxiv.org/abs/2605.17558)

**Licenses:** [DeepSeek Open Platform ToS](https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html) · [DeepSeek ToU](https://cdn.deepseek.com/policies/en-US/deepseek-terms-of-use.html) · [OpenCode license](https://github.com/anomalyco/opencode/blob/dev/LICENSE) · [OpenCode ToS](https://opencode.ai/legal/terms-of-service) · [MITRE ATT&CK terms](https://attack.mitre.org/resources/legal-and-branding/terms-of-use/) · [Nmap legal](https://nmap.org/book/man-legal.html)
