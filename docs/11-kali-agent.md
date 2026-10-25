# 11 — The Kali agent: architecture, MCP, guardrails and smoke tests

*Last verified: 25 October 2026. This chapter is about wiring a local model to an isolated Kali VM. It deliberately keeps the tool surface small and the policy layer deterministic.*

The model is a planner. The **policy layer** decides what is allowed. The **sandbox** decides what is possible. The **audit log** decides what you can prove afterwards.

A 14B model with a good scaffold and six typed tools will outperform a 27B model with a raw shell and 150 tools. Build the scaffold first.

---

## 1. Architecture

```text
┌──────────────────────────── Mac (32 GB) ────────────────────────────┐
│  mlx_lm.server / llama-server / Ollama                              │
│  Qwen3-14B + LoRA, OpenAI-compatible, bound to 127.0.0.1:8080       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ SSH tunnel (preferred)
                                │ or host-only VMnet + firewall
┌───────────────────────────────▼─────────────────────────────────────┐
│  Kali VM (4–8 GB)                                                   │
│                                                                     │
│  OpenCode / hackingBuddyGPT / PentestGPT-legacy                     │
│        │                                                            │
│        ├── policy layer: scope, allowlist, confirm, audit            │
│        │                                                            │
│        └── MCP-Kali-Server (stdio client → HTTP bridge)             │
│                 │                                                   │
│                 ├── nmap_scan     ├── http_probe                    │
│                 ├── kali_exec     ├── report_finding                │
│                 └── ...                                             │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ Docker internal network (no egress)
┌───────────────────────────────▼─────────────────────────────────────┐
│  Lab targets: DVWA, Juice Shop, Metasploitable, VulnHub VMs         │
│  Disposable. Snapshotted. No route to the public internet.          │
└─────────────────────────────────────────────────────────────────────┘
```

Three rules:

1. **The model never talks to the internet directly.** It calls tools; the tools run in the lab.
2. **The policy layer is not an LLM.** It is code with explicit rules.
3. **Every action is logged before it executes**, with the model's stated reason, the exact arguments, and the authorizing scope.

---

## 2. Kali VM setup

### 2.1 VM configuration

| Setting | Recommendation |
|---|---|
| RAM | 4 GB for a 14B local model; 6–8 GB if the Mac is only serving |
| CPU | 2–4 vCPUs |
| Disk | 40–80 GB, thin-provisioned |
| Network | Host-only + NAT for updates; or bridged on a dedicated lab VLAN |
| Shared folders | **None** (or a read-only `~/lab` share) |
| Clipboard | Disabled if you handle sensitive data |
| Snapshot | Take a clean snapshot before installing tools |

Do not run the Kali VM and a 14B training job at the same time; both compete for unified memory.

### 2.2 Network modes

- **Host-only** (UTM `192.168.64.1`, VirtualBox `192.168.56.1`, VMware `vmnet1`): the VM sees the Mac and other lab VMs, not the internet. Best default.
- **Bridged:** the VM is a peer on the LAN. Use only on a trusted lab VLAN; never on public Wi-Fi.
- **NAT:** the VM reaches the internet through the host, but inbound is awkward. Use for updates, then switch back to host-only.

Verify from Kali:

```bash
ip -br addr
ip route
ping -c 1 192.168.64.1
```

### 2.3 Install the agent stack

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip jq curl nmap
```

Install OpenCode (headless agent):

```bash
curl -fsSL https://opencode.ai/install | bash
opencode --version
```

Clone MCP-Kali-Server:

```bash
sudo git clone https://github.com/Wh0am123/MCP-Kali-Server.git /opt/MCP-Kali-Server
cd /opt/MCP-Kali-Server
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Start the bridge on loopback only:

```bash
./server.py --ip 127.0.0.1 --port 5000
```

For a reproducible tool image, build `examples/Dockerfile.kali` (nmap, curl, dig, openssl, nikto, whatweb, gobuster, ffuf, smbclient, enum4linux) and start the lab with `docker compose -f examples/docker-compose.lab.yml up -d`.

---

## 3. Connect the model

### 3.1 SSH tunnel

On Kali:

```bash
ssh -N -f -L 8080:127.0.0.1:8080 macuser@192.168.64.1
curl -s http://127.0.0.1:8080/v1/models | jq .
```

Test a tool call:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @- <<'JSON' | jq '.choices[0].message.tool_calls'
{
  "model": "local",
  "messages": [
    {"role":"system","content":"Only scan 10.10.10.0/24."},
    {"role":"user","content":"Enumerate services on 10.10.10.5."}
  ],
  "tools": [{
    "type":"function",
    "function":{
      "name":"nmap_scan",
      "description":"Scan an in-scope target.",
      "parameters":{"type":"object","properties":{"target":{"type":"string"}},"required":["target"]}
    }
  }],
  "tool_choice":"auto",
  "max_tokens":256
}
JSON
```

If `tool_calls` is `null`, fix the template/parser before building the agent.

### 3.2 OpenCode provider + MCP

`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local Qwen",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "local"
      },
      "models": {
        "qwen3-14b-pentest": {"name": "Qwen3-14B Pentest"}
      }
    }
  },
  "model": "local/qwen3-14b-pentest",
  "mcp": {
    "kali": {
      "type": "local",
      "command": ["/opt/MCP-Kali-Server/.venv/bin/python", "/opt/MCP-Kali-Server/client.py"],
      "enabled": true
    }
  },
  "permission": {
    "bash": "ask",
    "edit": "deny",
    "webfetch": "deny"
  }
}
```

Run a planning-only session first:

```bash
opencode run -m local/qwen3-14b-pentest --format json \
  "List the tools you have. Do not call any. Explain which one you would use to find open ports on 10.10.10.5."
```

Then a real task with approvals:

```bash
opencode run -m local/qwen3-14b-pentest --format json \
  "Using only the authorized scope 10.10.10.0/24, enumerate services on 10.10.10.5 and summarize. Ask before any command that is not read-only."
```

### 3.3 hackingBuddyGPT (minimal baseline)

If you want a research baseline with a tiny tool loop, hackingBuddyGPT is a better fit than a mega-framework. Configure its OpenAI-compatible base URL to `http://127.0.0.1:8080/v1`, point it at the lab host, and keep its tool set read-only. It is intentionally minimal; do not expect a full pentest workflow.

### 3.4 Framework comparison

| Framework | Model interface | Tool surface | Fit for a local 14B |
|---|---|---|---|
| **OpenCode + MCP-Kali** | OpenAI-compatible | typed MCP tools + optional bash | **best UX; keep MCP tools ≤10** |
| **hackingBuddyGPT** | OpenAI-style base URL | small Python tool loop | **best research baseline** |
| **PentestGPT (legacy)** | OpenAI-compatible `--base-url` | shell/PTY + stage pipeline | good guided UX; autonomous path expects frontier models |
| **CAI** | LiteLLM → OpenAI-compatible | shell + typed tools | archived in 2026; use as reference only |
| **HexStrike AI** | MCP host supplies model | 150+ tools, 12+ agents | too many schemas for a 14B; context bloat |
| **MCP-Kali-Server** | none (MCP host) | 11 typed tools + `execute_command` | **best small tool surface** |

The winning combination for this hardware is **OpenCode (or hackingBuddyGPT) + MCP-Kali-Server**, with a policy wrapper around the MCP calls.

---

## 4. A small, typed tool surface

Expose 6–10 tools, not 150. Suggested set:

| Tool | Arguments | Purpose |
|---|---|---|
| `scope_check` | `target` | returns allowed/denied + reason |
| `nmap_scan` | `target`, `flags` (enum), `ports` | service discovery |
| `http_probe` | `url`, `method` | headers, status, title, tech |
| `dns_lookup` | `name`, `record_type` | A/AAAA/MX/TXT |
| `tls_info` | `host`, `port` | certificate, versions, ciphers |
| `kali_exec` | `command` (allowlisted), `timeout_s` | controlled CLI for approved tools |
| `read_file` | `path` (under `/lab`) | notes/evidence only |
| `report_finding` | `title`, `severity`, `evidence`, `remediation` | structured output |

Every tool schema should state:

- **when to use it** (and when not to),
- **the scope constraint**,
- **what it returns**,
- **the timeout**.

Example:

```json
{
  "type": "function",
  "function": {
    "name": "nmap_scan",
    "description": "Run a read-only nmap scan against a target that is explicitly inside the authorized scope. Use for host discovery and service/version detection. Never use against public IPs or domains. Returns stdout, stderr and exit code.",
    "parameters": {
      "type": "object",
      "properties": {
        "target": {"type": "string", "description": "IPv4, hostname, or CIDR inside scope."},
        "flags": {"type": "string", "enum": ["-sn", "-sV", "-sC", "-sV -sC", "-p-"], "default": "-sV"},
        "timeout_s": {"type": "integer", "minimum": 5, "maximum": 300, "default": 120}
      },
      "required": ["target"],
      "additionalProperties": false
    }
  }
}
```

**Do not expose `execute_command` without an allowlist.** MCP-Kali-Server includes it; wrap it or disable it.

---

## 5. The policy layer

The model's output is untrusted input. Before any tool executes:

```python
# policy.py (pseudocode, run in the MCP bridge or a wrapper)
import ipaddress, re, yaml, time

SCOPE = yaml.safe_load(open("/etc/agent/scope.yaml"))
DENY = [
    r"\brm\s+-rf\s+/\b", r"\bmkfs\b", r"\bdd\s+.*of=/dev/",
    r":\(\)\s*\{.*\};:", r"\bshutdown\b", r"\breboot\b",
    r"\bcurl\b.*\|\s*(ba)?sh", r"\bwget\b.*\|\s*(ba)?sh",
]
ALLOWED_BINARIES = {"nmap", "curl", "dig", "openssl", "nikto", "whatweb",
                    "gobuster", "ffuf", "smbclient", "enum4linux"}

def check_target(target: str) -> tuple[bool, str]:
    if not (SCOPE["window"]["start"] <= time.time() <= SCOPE["window"]["end"]):
        return False, "outside authorized time window"
    try:
        net = ipaddress.ip_network(target, strict=False)
    except ValueError:
        return False, "target is not an IP/CIDR"
    for allowed in SCOPE["targets"]:
        if net.subnet_of(ipaddress.ip_network(allowed)):
            return True, "in scope"
    return False, "target is outside the authorized scope"

def check_command(cmd: str) -> tuple[bool, str]:
    for pat in DENY:
        if re.search(pat, cmd, re.I):
            return False, f"denied pattern: {pat}"
    binary = cmd.strip().split()[0] if cmd.strip() else ""
    if binary not in ALLOWED_BINARIES:
        return False, f"binary not allowlisted: {binary}"
    return True, "allowed"

def audit(event: dict):
    with open("/var/log/agent/audit.jsonl", "a") as f:
        f.write(json.dumps({**event, "ts": time.time()}) + "\n")
```

`scope.yaml`:

```yaml
engagement_id: LAB-2026-001
authorized_by: "Beniamin Pantiru"
authorization_document: "/lab/authorization.pdf"
targets:
  - 10.10.10.0/24
domains: []
exclude:
  - 10.10.10.1          # hypervisor / management
allowed_tools:
  - nmap_scan
  - http_probe
  - dns_lookup
  - tls_info
  - kali_exec
  - report_finding
window:
  start: 2026-10-25T08:00:00Z
  end:   2026-10-26T08:00:00Z
```

**Scope check is not a prompt.** It is a function that parses the target and compares it to the file. The model cannot talk its way past it.

### 5.1 Confirmation policy

| Action class | Policy |
|---|---|
| Read-only recon (nmap `-sn`/`-sV`, HTTP GET, DNS, TLS) | auto-approve inside scope |
| Active enumeration (fuzzing, directory brute force) | confirm once per target |
| Authentication attempts / credential testing | confirm per action; lab only |
| Any write/delete/config change | deny |
| Any target outside scope | deny |
| Any command not on the allowlist | deny |

### 5.2 Audit log

Log **before** execution and **after**:

```json
{"ts":"...","engagement":"LAB-2026-001","actor":"qwen3-14b-pentest","tool":"nmap_scan","args":{"target":"10.10.10.5","flags":"-sV"},"scope_check":"pass","decision":"allow","reason":"in scope","stdout_sha256":"...","exit_code":0}
```

Keep the log outside the sandbox (write-only mount) so a compromised container cannot erase it.

### 5.3 Kill switch

- The agent process has a wall-clock budget per task (e.g., 10 minutes).
- The bridge has a maximum tool calls per task (e.g., 20).
- A `SIGINT` stops the agent and kills the bridge's child processes.
- The lab network can be torn down with `docker compose down -v`.

---

## 6. Smoke test

Do this before any real task.

```bash
# 1. Model endpoint
curl -s http://127.0.0.1:8080/v1/models | jq .

# 2. MCP bridge (inside Kali)
curl -s http://127.0.0.1:5000/health | jq .

# 3. Scope check
python3 - <<'PY'
from policy import check_target
print(check_target("10.10.10.5"))   # (True, 'in scope')
print(check_target("8.8.8.8"))      # (False, 'target is outside...')
PY

# 4. A safe tool call through the bridge
curl -s http://127.0.0.1:5000/nmap_scan \
  -H 'Content-Type: application/json' \
  -d '{"target":"10.10.10.5","flags":"-sV"}' | jq .

# 5. The model chooses the tool
opencode run -m local/qwen3-14b-pentest --format json \
  "Use the nmap_scan tool on 10.10.10.5 with -sV, then summarize the open ports."
```

Expected flow:

1. The model emits a `tool_calls` entry for `nmap_scan`.
2. The policy layer checks `10.10.10.5` against `scope.yaml` → allow.
3. The MCP bridge runs nmap in the lab.
4. The observation returns to the model.
5. The model summarizes only what the output shows, and does not invent ports.

If any step fails, fix it before adding tools or turns.

---

## 7. A worked loop (safe, lab-only)

```text
User: "Enumerate 10.10.10.5 and tell me what to look at next."

Model:
  thought: target is in 10.10.10.0/24; read-only recon is allowed.
  tool_call: nmap_scan {"target":"10.10.10.5","flags":"-sV"}

Tool:
  exit_code=0
  22/tcp open ssh OpenSSH 9.6
  80/tcp open http nginx 1.26

Model:
  thought: two services. HTTP is the broadest next step; probe headers.
  tool_call: http_probe {"url":"http://10.10.10.5","method":"HEAD"}

Tool:
  HTTP/1.1 200 OK
  Server: nginx/1.26
  X-Powered-By: PHP/8.3

Model:
  final: "SSH and HTTP are exposed. nginx 1.26 with PHP 8.3 suggests a web app;
  next I would enumerate virtual hosts and check for a login page, within scope.
  Evidence: nmap -sV; HEAD http://10.10.10.5."
```

The model does not need to be brilliant. It needs to be **consistent, in scope, and honest about what it observed**.

---

## 8. Lab targets

Use intentionally vulnerable applications in an isolated network:

| Target | Image / VM | What it exercises |
|---|---|---|
| DVWA | `vulnerables/web-dvwa` | web enumeration, auth, basic injection labs |
| OWASP Juice Shop | `bkimminich/juice-shop` | modern SPA, API discovery |
| Metasploitable 2/3 | VulnHub / Rapid7 VM | service enumeration, legacy protocols |
| VulnHub boxes | downloaded VMs | full multi-step lab |
| Windows/AD lab | evaluation VM + Samba AD | SMB/LDAP/Kerberos enumeration |

Do **not** point the agent at public bug bounty programs, company networks, or the open internet. The model cannot verify authorization; the policy layer can only enforce the scope you wrote.

---

## 9. Integration with evaluation

Run the same golden set through the full agent, not just the model endpoint:

```bash
python scripts/eval_toolcall.py \
  --base-url http://127.0.0.1:8080/v1 \
  --golden data/golden/golden.jsonl \
  --agent mcp \
  --sandbox http://127.0.0.1:5000 \
  --out reports/agent-14b.json
```

This measures the **system**: model + prompt + tools + policy. A model that scores 95% on isolated tool calls may drop to 60% in a 6-turn loop with noisy observations. Evaluate the loop.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Agent never calls a tool | Model not trained with `tools`, or template mismatch | Test the server directly; inspect rendered prompt. |
| Agent calls `execute_command` for everything | Tool descriptions too generic; `kali_exec` too permissive | Remove raw shell; use typed tools; add “when not to use”. |
| Agent scans `8.8.8.8` | No scope check; training data had public targets | Policy layer denies; add refusal examples; retrain. |
| MCP bridge unreachable | Bound to wrong interface; firewall | `127.0.0.1:5000`, SSH tunnel, check `curl /health`. |
| Tool call arguments invalid | Model undertrained on schema | Add schema examples; validate and retry with an error message. |
| Loop repeats the same call | No observation-awareness training | Add recovery trajectories; cap repeated calls in policy. |
| Latency grows every turn | Context not trimmed | Summarize old observations; cap turns; smaller tool schemas. |
| OpenCode cannot see MCP tools | Wrong command path/venv | Use absolute paths; run `opencode mcp list`. |

---

## 11. Sources

- OpenCode docs — <https://opencode.ai/docs>
- OpenCode MCP — <https://opencode.ai/docs/mcp-servers>
- MCP-Kali-Server — <https://github.com/Wh0am123/MCP-Kali-Server>
- PentestGPT — <https://github.com/GreyDGL/PentestGPT>
- hackingBuddyGPT — <https://github.com/ipa-lab/hackingBuddyGPT>
- HexStrike AI — <https://github.com/0x4m4/hexstrike-ai>
- CAI (archived) — <https://github.com/aliasrobotics/cai>
- Kali Linux — <https://www.kali.org/docs/>
- OWASP Juice Shop — <https://owasp.org/www-project-juice-shop/>
- DVWA — <https://github.com/digininja/DVWA>
- MCP specification — <https://modelcontextprotocol.io/>
