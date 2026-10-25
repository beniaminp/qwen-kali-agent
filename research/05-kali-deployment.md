# 05 — Kali Deployment & Integration for a Local Qwen3-14B LoRA

**Scope:** Qwen3-14B + LoRA served on an M1 Pro/32 GB Mac, consumed by an authorized-testing agent in a Kali VM on the same LAN. Verified against upstream docs/repos on **25 Oct 2026**. Star counts and push dates are from the GitHub API on that date. Performance figures marked "est." are estimates, not measured on this machine.

---

## 0. Recommended stack (short version)

**llama.cpp `llama-server` on the Mac (best tool-call fidelity) → SSH local-forward → OpenCode inside Kali → MCP-Kali-Server → isolated Docker lab network.** Use MLX for speed, Ollama for simplicity, llama.cpp for tool fidelity. Never expose any of these servers on `0.0.0.0` without a firewall — none authenticate requests.

---

## 1. Serving on macOS

### 1.1 `mlx_lm.server` (native MLX, fastest on M-series)

Install (`pip install mlx-lm`) — <https://github.com/ml-explore/mlx-lm>. Defaults: `127.0.0.1:8080`, `--max-tokens 512` — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md>.

```bash
pip install -U mlx-lm

mlx_lm.server \
  --model mlx-community/Qwen3-14B-4bit \
  --adapter-path ./adapters/qwen3-14b-pentest \
  --host 127.0.0.1 --port 8080 \
  --max-tokens 2048 \
  --kv-bits 4 --kv-group-size 64 --prefill-step-size 512 \
  --chat-template-args '{"enable_thinking": false}' \
  --log-level INFO
```

Notes, all verified in the current `SERVER.md` / `server.py`:

- `--adapter-path` loads PEFT/LoRA adapter weights; the adapter must match the base model's MLX conversion. You can also pass `"adapters"` per request.
- `--max-tokens` is only a **default**; clients can override per request. Agents should set 1024–2048.
- `--kv-bits 4` quantizes the KV cache but **disables batching** (one request at a time); acceptable for a single agent, drop it for concurrency. Reduce `--prefill-step-size` when using it, since the score matrix is `prefill_step_size × context`.
- `--pipeline` selects pipelining over tensor parallelism for distributed inference (`mx.distributed`); omit it on one Mac.
- `--chat-template` / `--chat-template-args` pass through to `apply_chat_template` (e.g. disable Qwen3 thinking for tool loops); `--use-default-chat-template` forces the model's built-in template.
- Tool calls: server code has a `ToolCallFormatter` and parses OpenAI-style `tools`, but it is thin — validate with a real call. **No API-key auth**, `--allowed-origins` defaults to `*`, and the docs warn it is "not recommended for production".

### 1.2 Ollama (simplest, GGUF)

Install the macOS app from <https://ollama.com/download> (Sonoma 14+, Apple silicon supported) — <https://docs.ollama.com/macos>. The app runs `ollama serve` on `127.0.0.1:11434`.

```bash
# Import a GGUF (Ollama does not quantize on import)
cat > Modelfile <<'EOF'
FROM ./qwen3-14b.Q4_K_M.gguf
PARAMETER num_ctx 16384
PARAMETER temperature 0.2
PARAMETER stop "<|im_end|>"
EOF
ollama create qwen3-pentest -f Modelfile
ollama run qwen3-pentest
```

Verified at <https://docs.ollama.com/import>. To expose on the LAN:

```bash
launchctl setenv OLLAMA_HOST "0.0.0.0:11434"   # then restart the Ollama app
# or for a foreground server:
OLLAMA_HOST=0.0.0.0:11434 OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

`OLLAMA_HOST`/`launchctl` are documented in <https://docs.ollama.com/faq>. **Default context is 4096**; set `OLLAMA_CONTEXT_LENGTH` or `num_ctx` per request. The OpenAI-compatible base is `http://<host>:11434/v1/`; the API key is required by clients but ignored (`api_key="ollama"`) — <https://docs.ollama.com/api/openai-compatibility>.

Caveats: Ollama cannot apply a PEFT LoRA directory directly — merge the adapter into the HF weights (`mlx_lm.fuse` or PEFT `merge_and_unload`), convert to GGUF, then `ollama create` (or use llama.cpp `--lora`, §1.3). The `/v1` surface is explicitly a **subset** of OpenAI; tool calling is documented for `/api/chat` (<https://docs.ollama.com/capabilities/tool-calling>) but not on the OpenAI-compat page. Prefer non-streaming or `/api/chat` for agent loops (UNVERIFIED for your build).

### 1.3 `llama.cpp` `llama-server` (best control and tool fidelity)

```bash
brew install llama.cpp        # official prebuilt path
```

<https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md>

```bash
llama-server \
  -m ./models/qwen3-14b-Q4_K_M.gguf \
  --lora ./adapters/qwen3-14b-pentest.gguf \
  --host 127.0.0.1 --port 8080 \
  -c 16384 -ngl 99 -fa auto \
  --jinja \
  --chat-template-file ./qwen3-tool-call.jinja \
  --reasoning-format none
```

- `--host` defaults to `127.0.0.1`; **`--port` now defaults to 9931**, so always pass `--port` explicitly — <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>.
- `--jinja` is **enabled by default on current master** (help text: `default: enabled`), but older packaged builds still need it. Tool calling is only active with Jinja templates; native handlers exist for Qwen 2.5/Hermes/Llama 3.x, otherwise a generic parser is used ("Chat format: Generic" in logs) — <https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md>. Qwen3 templates normally work; if parsing fails, supply `--chat-template-file`.
- `--lora` accepts a **GGUF-converted** adapter (use `convert_lora_to_gguf.py`; comma-separated for multiple) — the only one of the three servers that applies LoRA without merging. `--reasoning-format none` keeps Qwen3 thinking out of `message.content`.

**Memory (est.):** 4-bit weights ≈ 8–9 GB; 16k KV ≈ 2.5 GB (f16) → ~12 GB resident in 32 GB; 32k ≈ 15 GB, so use q8 KV. Decode ≈ 15–25 tok/s, prefill ≈ 150–300 tok/s (est., bandwidth-bound) — adequate if prompts stay short and output is capped.

---

## 2. Networking from the Kali VM

**VM mode choice**

**Bridged** gives the VM its own LAN IP and direct reach to the Mac — best on a trusted lab LAN. **Host-only** reaches the host's vmnet address (e.g. `192.168.56.1` VirtualBox, `192.168.64.1` UTM) — best for isolation. **NAT** reaches the host only via the gateway (`10.0.2.2` VirtualBox, `192.168.64.1` UTM) and needs a port-forward for inbound — avoid. For an autonomous pentest agent, use **host-only plus a NAT adapter for updates**, or bridged on a dedicated lab VLAN. Never bridge onto an untrusted network.

**Find the Mac's address**

```bash
ipconfig getifaddr en0                 # Wi-Fi
ipconfig getifaddr en1                 # Ethernet/Thunderbolt
ifconfig | grep "inet "                # all interfaces
scutil --get LocalHostName             # mDNS name: <name>.local
```

**macOS firewall:** if enabled, incoming connections to `python`/`ollama`/`llama-server` are blocked until allowed; allow the binary or (better) stay on loopback + SSH.

**SSH tunnel (preferred; no inbound port on the Mac)**

```bash
# On the Mac: System Settings > General > Sharing > Remote Login (on)
# On Kali:
ssh -N -f -L 8080:127.0.0.1:8080 macuser@192.168.1.50
curl -s http://127.0.0.1:8080/v1/models | jq .
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-14b","messages":[{"role":"user","content":"reply OK"}],"max_tokens":8}'
```

Keep the tunnel process supervised (`autossh -M 0 -N -L ...`). For MCP-Kali-Server in the other direction, forward port 5000 similarly. Binding `--host 0.0.0.0` is only acceptable behind a `pf` anchor that permits just the Kali VM's IP; mlx and llama-server have **no auth**, and Ollama has none either.

---

## 3. Agent frameworks for Kali

| Project | Stars / last push | Maintained | LLM interface | Tool interface | Local 14B fit |
|---|---|---|---|---|---|
| [PentestGPT](https://github.com/GreyDGL/PentestGPT) | 16.1k / 2026-07 | Yes | OpenAI-compatible in **legacy interactive mode** (`--base-url`), plus Ollama | Shell/PTY, stage pipeline | Good in `pentestgpt-legacy`; autonomous pipeline is Claude/Codex-only |
| [CAI](https://github.com/aliasrobotics/cai) | 9.8k / 2026-08 | **Archived** (read-only, Aug 2026) | LiteLLM → OpenAI-compatible | Shell + typed tools, agents | Do not start new deployments; `cai-framework 0.5.10` remains on PyPI |
| [hackingBuddyGPT](https://github.com/ipa-lab/hackingBuddyGPT) | 1.3k / 2026-10-24 | Active | OpenAI-style client, configurable base URL | Small set of typed/shell tools; "50 lines" philosophy | Excellent research baseline; minimal tool breadth |
| [HexStrike AI](https://github.com/0x4m4/hexstrike-ai) | 13.1k / 2026-08 | Yes | None built in — MCP host supplies the LLM | **MCP** server wrapping 150+ tools, 12+ agents | Powerful but huge tool schema; 14B context/tool-selection will struggle |
| [MCP-Kali-Server](https://github.com/Wh0am123/MCP-Kali-Server) | 862 / 2026-03 | Yes | None — MCP host supplies the LLM | **MCP** stdio client + HTTP bridge; 11 typed tools + raw `execute_command` | Best fit: small, explicit tool set, easy guardrails |
| [OpenCode](https://github.com/sst/opencode) | — / active | Yes | Any OpenAI-compatible provider (`@ai-sdk/openai-compatible`) | Built-in bash/edit + MCP | Best UX; 14B must be reliable at tool calls, keep tool count low |

**How they call the model**

- **PentestGPT:** `pentestgpt-legacy --reasoning-model ollama:qwen3 --base-url http://127.0.0.1:8080/v1`; OpenAI-compatible connectors share one registry entry. The newer autonomous pipeline shells out to Claude Code/Codex, so a local 14B cannot drive that path — <https://github.com/GreyDGL/PentestGPT>.
- **CAI:** OpenAI Agents SDK + LiteLLM, so any OpenAI-compatible endpoint worked; archived, use only as reference.
- **hackingBuddyGPT:** Python tool loop with a configurable OpenAI-style client; confirm the exact `base_url` key in `config.py` (UNVERIFIED key name).
- **HexStrike / MCP-Kali-Server:** no model endpoint of their own — the MCP host's LLM selects tools. MCP-Kali's `client.py` exposes `nmap_scan`, `gobuster_scan`, `dirb_scan`, `nikto_scan`, `sqlmap_scan`, `metasploit_run`, `hydra_attack`, `john_crack`, `wpscan_analyze`, `enum4linux_scan`, `server_health`, `execute_command` (names verified in `client.py`).
- **OpenCode:** custom provider block below; built-in shell can run commands directly, but the MCP path gives a typed, auditable choke point.

**Verdict:** MCP-Kali-Server + OpenCode (or hackingBuddyGPT) beats the heavyweight frameworks. HexStrike's 150+ schemas consume context and degrade 14B tool selection; PentestGPT legacy is the best guided UX. VulnBot/xOffense were not verifiable as maintained repos on 25 Oct 2026 (**UNVERIFIED**).

---

## 4. MCP integration

### Install and run (Kali)

```bash
sudo apt install mcp-kali-server       # Kali package, per upstream README
kali-server-mcp                        # or: ./server.py --ip 127.0.0.1 --port 5000
```

Bleeding edge:

```bash
git clone https://github.com/Wh0am123/MCP-Kali-Server.git /opt/MCP-Kali-Server
cd /opt/MCP-Kali-Server
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./server.py --ip 127.0.0.1 --port 5000
```

Keep the default loopback bind. If OpenCode runs on the Mac, tunnel 5000 and run `client.py` on the Mac; simpler is to run OpenCode inside Kali. The MCP client is stdio, so OpenCode launches it as a child process.

### OpenCode config (`~/.config/opencode/opencode.json`)

Local provider + local MCP:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local-qwen": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Qwen3-14B (Mac)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "dummy"
      },
      "models": {
        "qwen3-14b": {
          "name": "Qwen3-14B pentest LoRA",
          "tool_call": true,
          "limit": { "context": 16384, "output": 2048 }
        }
      }
    }
  },
  "mcp": {
    "kali": {
      "type": "local",
      "command": ["python3", "/opt/MCP-Kali-Server/client.py", "--server", "http://127.0.0.1:5000"],
      "enabled": true,
      "timeout": 120000
    }
  },
  "permission": { "bash": "ask", "edit": "ask" }
}
```

Provider syntax verified at <https://opencode.ai/docs/providers>; MCP syntax at <https://opencode.ai/docs/mcp-servers> (`type: local`, `command` array, `timeout`; remote uses `type: remote` + `url` + `headers`). Raise the 30 s default MCP timeout — nmap and sqlmap exceed it.

**Sandboxing:** `execute_command` runs arbitrary shell as the server's user. Use a dedicated unprivileged user, no Docker socket/cloud keys/SSH agent, read-only wordlists. Treat tool output as untrusted (prompt injection via banners, HTTP bodies, filenames): truncate and strip ANSI. Pin the repo commit — MCP tool descriptions are also attack surface.

---

## 5. Recommended architecture and smoke test

```
┌──────────────────────────── Mac M1 Pro 32GB ────────────────────────────┐
│  llama-server (Qwen3-14B Q4 + LoRA)  127.0.0.1:8080  --jinja            │
│         ▲                                                               │
│         │ SSH local forward (no inbound port)                           │
└─────────┼───────────────────────────────────────────────────────────────┘
          │  ssh -L 8080:127.0.0.1:8080 mac@mac.local
┌─────────┴──────────────── Kali VM (host-only / lab VLAN) ───────────────┐
│  OpenCode  ──stdio──▶  MCP-Kali-Server client  ──HTTP:5000──▶ server.py  │
│      │                                                        │         │
│      └── built-in shell (permission: ask)                     │ exec    │
│                                                               ▼         │
│                                        Docker lab net 172.28.0.0/24     │
│                                        target 172.28.0.10 (owned)       │
└─────────────────────────────────────────────────────────────────────────┘
```

**Smoke test (owned lab host only)**

```bash
# 1. Lab network with no route out
docker network create --internal --subnet 172.28.0.0/24 agentlab
docker run -d --name target --network agentlab --ip 172.28.0.10 \
  vulnerables/web-dvwa

# 2. Verify model + tunnel
curl -s http://127.0.0.1:8080/v1/models

# 3. Verify MCP bridge directly
curl -s -X POST http://127.0.0.1:5000/api/command \
  -H 'Content-Type: application/json' \
  -d '{"command":"nmap -sV -Pn 172.28.0.10"}'

# 4. Agent prompt
opencode run "Using the kali MCP tools, run nmap -sV -Pn 172.28.0.10, \
summarize open ports and services. Do not scan any other host."
```

Expected flow: `nmap_scan` tool call → POST `/api/command` → server executes → JSON `{stdout, stderr, return_code}` → model summarizes with reasoning logged on the Mac. **Pass criteria:** one target touched, command in the audit log, no injected metacharacters, <60 s.

---

## 6. Guardrails (concrete patterns)

1. **Scope file** — machine-readable, checked before every command:

```yaml
# /etc/agent/scope.yaml
authorization: "ENG-2026-0417"      # signed SOW / rules of engagement ID
expires: "2026-11-30T23:59:59Z"
networks: ["172.28.0.0/24"]
hosts: ["172.28.0.10"]
domains: ["dvwa.lab.internal"]
exclude: ["172.28.0.1", "172.28.0.254"]
allowed_tools: ["nmap", "curl", "nikto", "gobuster", "ffuf", "sqlmap"]
```

2. **Allowlist + target extraction** — tokenize with `shlex`, reject metacharacters (`; | & $() > < \``), resolve every hostname/IP with `ipaddress`, require scope membership. Denylist secondary: `rm`, `dd`, `mkfs`, `shutdown`, `reboot`, `iptables -F`, `useradd`, `passwd`, `chmod 777 /`, `curl … | sh`, `> /dev/sd*`.

3. **Choke point** — patch `server.py`'s `execute_command` to call `guard.check(command)` before `CommandExecutor`, or proxy port 5000 and validate the JSON body. Never rely on the system prompt.

4. **Confirmation** — OpenCode `"permission": {"bash": "ask"}` plus `--dry-run`/`--approve` in the guard for anything beyond read-only recon; prefer typed tools and consider removing `execute_command` from `client.py`.

5. **Audit** — append-only JSONL: timestamp, argv, target, scope decision, exit code, duration, model request ID; hash-chain (`sha256(prev + record)`) and ship off-box, including raw tool-call JSON.

6. **Rate limits** — token bucket in the guard (10 commands/min, 1 concurrent scan), `nmap --max-rate 500 --host-timeout`, and a 100 KB cap on tool output before it reaches the model.

7. **Kill switch** — guard checks `/etc/agent/STOP` on every call; `pkill -f mcp-kali`; tear down the tunnel; revoke VM networking at the hypervisor.

8. **Isolation** — unprivileged user, no Docker socket/cloud keys/SSH agent, `setcap cap_net_raw,cap_net_admin+eip /usr/bin/nmap` instead of passwordless sudo, lab on Docker `--internal`, no lab→LAN route, host-only vmnet, egress blocked except SSH to the Mac.

9. **Never public targets** — enforce in the guard and expire authorizations automatically; a stale scope file must fail closed.

---

## 7. Deployment options compared

| Option | Latency (est., 14B 4-bit) | Tool-call fidelity | LoRA | Complexity | Use when |
|---|---|---|---|---|---|
| `mlx_lm.server` | **Fastest** decode, ~18–25 tok/s | Medium — new `ToolCallFormatter`, thin OpenAI surface | Native `--adapter-path` | Low | Speed matters; single user; you can test tool loops |
| Ollama | ~12–20 tok/s | Medium-low through `/v1`; better on `/api/chat` | Requires merge → GGUF | **Lowest** | Quick demos, simple chat, non-critical tool use |
| `llama-server` (llama.cpp) | ~15–25 tok/s | **Highest** — `--jinja`, native/generic parsers, parallel calls opt-in | Native `--lora` (GGUF) | Medium | Production-ish agent use; this is the recommended default |

All are single-user-grade. Tool-call reliability dominates latency: 20 tok/s with valid JSON beats 30 tok/s with retries. Cap `max_tokens`, disable thinking, keep the tool list under ~12, and log every parse failure.
