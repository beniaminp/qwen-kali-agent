# 10 — Export and serving: from adapter to an OpenAI-compatible endpoint

*Last verified: 25 October 2026 against mlx-lm 0.32.2 and current llama.cpp/Ollama docs.*

You have three deployment paths:

| Path | Best for | Tool-call fidelity | Speed on M1 Pro |
|---|---|---|---|
| **MLX fused model** (`mlx_lm.server`) | fastest iteration, native Apple Silicon | good, but template-dependent | fastest |
| **GGUF + `llama-server`** | best control, portable, Jinja templates | **best** | fast |
| **GGUF + Ollama** | convenience, model management | good, but `/v1` is a subset | fast |

All three are **unauthenticated**. Bind to loopback and use an SSH tunnel from Kali, or put them behind a firewall that only admits the lab VM.

---

## 1. Fuse the adapter

An MLX adapter is a set of low-rank deltas. To serve it as a standalone model, merge it into the base.

```bash
mlx_lm.fuse \
  --model mlx-community/Qwen3-14B-4bit \
  --adapter-path adapters/qwen3-14b-pentest \
  --save-path fused/qwen3-14b-pentest-4bit
```

This keeps the base in 4-bit and writes a model you can serve directly with `mlx_lm.server`. The output is small (~8.5 GB).

For GGUF conversion, dequantize to bf16/f16 first:

```bash
mlx_lm.fuse \
  --model mlx-community/Qwen3-14B-4bit \
  --adapter-path adapters/qwen3-14b-pentest \
  --save-path fused/qwen3-14b-pentest-bf16 \
  --dequantize
```

That writes a ~30 GB bf16 model. You need disk headroom.

**Do not use `--export-gguf` for Qwen3.** In mlx-lm 0.32.2, `fuse.py` explicitly rejects model types other than `llama`, `mixtral`, and `mistral`:

```python
if model_type not in ["llama", "mixtral", "mistral"]:
    raise ValueError(f"Model type {model_type} not supported for GGUF conversion.")
```

Use llama.cpp's converter instead (§2).

---

## 2. Convert to GGUF with llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp ~/work/llama.cpp
cd ~/work/llama.cpp

# Python deps for the converter
python3 -m pip install -r requirements.txt

python3 convert_hf_to_gguf.py \
  ~/work/qwen-kali-agent/fused/qwen3-14b-pentest-bf16 \
  --outtype f16 \
  --outfile ~/work/qwen-kali-agent/models/qwen3-14b-pentest-f16.gguf

# Quantize (choose one)
./build/bin/llama-quantize \
  ~/work/qwen-kali-agent/models/qwen3-14b-pentest-f16.gguf \
  ~/work/qwen-kali-agent/models/qwen3-14b-pentest-Q4_K_M.gguf Q4_K_M
```

Build first if `build/bin/llama-quantize` does not exist:

```bash
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j
```

### Quantization choice

| Quant | Size (14B) | Quality | Use |
|---|---:|---|---|
| Q4_K_M | ~9 GB | very good | **default for 32 GB** |
| Q5_K_M | ~10.5 GB | better | if you have headroom and want max quality |
| Q6_K | ~12 GB | near-lossless | tight with 16K context |
| Q8_0 | ~15.5 GB | effectively lossless | only for 8B or short context |
| F16 | ~30 GB | reference | conversion intermediate; do not serve on 32 GB |

For a 14B on 32 GB, Q4_K_M + 16K context is the sweet spot. Q5_K_M is worth testing if you mostly run short contexts.

**Tool-call quality is more sensitive to the chat template than to 4-bit vs 5-bit quantization.** Verify the template before chasing quant gains.

---

## 3. Serve with `llama-server`

```bash
llama-server \
  -m models/qwen3-14b-pentest-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  -c 16384 \
  -ngl 99 \
  -fa auto \
  --jinja
```

Notes:

- `--port` defaults to 9931 in current builds; always pass it explicitly.
- `--jinja` is enabled by default on current master, but pass it to be safe; tool calling requires a Jinja template.
- `-c 16384` sets the context. More context costs KV memory; 16K is a good default.
- `-ngl 99` offloads all layers on Metal.
- `-fa auto` enables flash attention when supported.
- If tool calls are not parsed, inspect the server log for `Chat format: Generic` and supply `--chat-template-file` with the model's template. The Qwen3 template is in the model's `tokenizer_config.json`.
- `--reasoning-format none` hides thinking from `content`; use it only if you intentionally trained a no-think adapter. Otherwise leave the default.

Test:

```bash
curl -s http://127.0.0.1:8080/v1/models | jq .
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "messages": [{"role":"user","content":"Reply with OK."}],
    "max_tokens": 8
  }' | jq .
```

---

## 4. Serve with MLX

```bash
mlx_lm.server \
  --model fused/qwen3-14b-pentest-4bit \
  --host 127.0.0.1 --port 8080 \
  --max-tokens 2048 \
  --temp 0.2 \
  --top-p 0.95 \
  --prefill-step-size 512 \
  --log-level INFO
```

Useful flags (verified in `server.py`):

| Flag | Why |
|---|---|
| `--adapter-path` | Serve a base model + adapter without fusing. |
| `--max-tokens` | Default output cap; clients can override. |
| `--chat-template-args` | Pass `enable_thinking` etc.; **match training**. |
| `--kv-bits 4` | Quantize KV cache; reduces memory; disables batching. |
| `--kv-group-size 64` | KV quant group size. |
| `--quantized-kv-start` | Start quantizing after N tokens. |
| `--prefill-step-size` | Lower = less peak memory on long prompts. |
| `--prompt-cache-size` | Cache repeated system/tool prefixes. |
| `--pipeline` | Pipeline parallelism for distributed inference; not needed on one Mac. |
| `--draft-model` | Speculative decoding with a smaller draft model. |

**Thinking mismatch:** if you trained with the default Qwen3 template, do **not** serve with `--chat-template-args '{"enable_thinking": false}'`; the model was trained to generate the think block, and the no-think prompt already contains a closed one. Either serve with the default, or train a no-think adapter with a patched template. See [`06-data-format.md`](06-data-format.md) §6.

**Security:** `mlx_lm.server` has no authentication and `--allowed-origins` defaults to `*`. Bind to `127.0.0.1` and tunnel. Do not expose it on a café Wi-Fi.

---

## 5. Serve with Ollama

Ollama imports GGUF; it does not quantize on import.

```bash
cat > Modelfile <<'EOF'
FROM ./models/qwen3-14b-pentest-Q4_K_M.gguf
PARAMETER num_ctx 16384
PARAMETER temperature 0.2
PARAMETER stop "<|im_end|>"
EOF

ollama create qwen3-pentest -f Modelfile
ollama run qwen3-pentest
```

Expose on the LAN only if you accept the risk:

```bash
launchctl setenv OLLAMA_HOST "0.0.0.0:11434"   # restart the Ollama app
# or foreground:
OLLAMA_HOST=0.0.0.0:11434 OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

OpenAI-compatible endpoint: `http://<mac>:11434/v1/`; clients need an API key but Ollama ignores it (`api_key="ollama"`).

Caveats:

- Ollama cannot apply an MLX/PEFT adapter directory directly. Fuse and convert first.
- The `/v1` surface is a subset of OpenAI. Tool calling is documented for `/api/chat`; verify it on your build before relying on `/v1` tools. If tool calls are flaky, use `llama-server`.
- Default context is 4096 unless you set `num_ctx` or `OLLAMA_CONTEXT_LENGTH`.
- Model management is convenient, but you have less control over the template and sampler.

---

## 6. Networking from the Kali VM

### Option A — SSH tunnel (recommended)

On the Mac, enable Remote Login: **System Settings → General → Sharing → Remote Login**.

From Kali:

```bash
ssh -N -f -L 8080:127.0.0.1:8080 macuser@192.168.64.1
curl -s http://127.0.0.1:8080/v1/models | jq .
```

The model server stays on loopback; only SSH is exposed. Use `autossh` for a persistent tunnel:

```bash
autossh -M 0 -N -L 8080:127.0.0.1:8080 macuser@192.168.64.1
```

### Option B — Host-only / bridged LAN

- **Host-only** (UTM `192.168.64.1`, VirtualBox `192.168.56.1`): the VM can reach the Mac, but not the internet. Best for an isolated lab.
- **Bridged**: the VM gets a LAN IP; the Mac is reachable directly. Only on a trusted lab VLAN.
- **NAT**: the VM reaches the host via the gateway; inbound needs port forwarding. Avoid for serving.

Find the Mac's address:

```bash
ipconfig getifaddr en0
ipconfig getifaddr en1
ifconfig | grep "inet "
```

If the macOS firewall is on, allow the server binary or keep everything on loopback.

### Option C — reverse tunnel

If the Mac cannot accept inbound connections, run the tunnel from the Mac to Kali:

```bash
ssh -N -R 8080:127.0.0.1:8080 kali@192.168.64.10
```

Then the Kali agent uses `http://127.0.0.1:8080`.

---

## 7. OpenAI-compatible tool calling

Once the server is up, the client is ordinary OpenAI:

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")

tools = [{
    "type": "function",
    "function": {
        "name": "nmap_scan",
        "description": "Scan an in-scope target for open TCP services.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "flags": {"type": "string", "default": "-sV"}
            },
            "required": ["target"]
        }
    }
}]

r = client.chat.completions.create(
    model="local",
    messages=[
        {"role": "system", "content": "Only scan targets in the authorized scope: 10.10.10.0/24."},
        {"role": "user", "content": "Enumerate services on 10.10.10.5."},
    ],
    tools=tools,
    tool_choice="auto",
    temperature=0.0,
    max_tokens=512,
)
print(r.choices[0].message.tool_calls)
```

The response should contain a `tool_calls` array with `function.name` and a JSON string `function.arguments`. If it returns prose instead, the template or parser is wrong; go back to [`06-data-format.md`](06-data-format.md) and run the smoke test.

### Verifying tool-call fidelity

```bash
python scripts/smoke_test.py --base-url http://127.0.0.1:8080/v1 \
  --tools examples/tools/kali-tools.json
```

It sends 10 canonical prompts and checks:

- the response is a tool call, not prose;
- the tool name is in the schema;
- arguments parse and required fields exist;
- no forbidden target appears;
- the call is stable across 3 samples at temperature 0.

---

## 8. Performance expectations

Approximate 4-bit numbers on an M1 Pro 16-core, short context:

| Model | Decode tok/s | 8K context | Notes |
|---|---:|---:|---|
| Qwen3-8B | 30–33 | 25–28 | fastest useful |
| Qwen3-14B | 15–17 | 12–14 | **recommended balance** |
| Qwen3.5-9B | 28–30 | 22–25 | better base, but training caveat |
| Qwen3.8-27B | 8–10 | 6–8 | oracle only |
| Qwen3.6-35B-A3B | 40–50 | 30–40 | MoE, bulk generation |

Latency in an agent loop is dominated by **prefill of the system prompt + tools** and **output length**, not just decode speed. Cache the system/tool prefix where the server supports it (`--prompt-cache-size` in MLX), keep tool schemas small, and cap `max_tokens`.

---

## 9. Publishing the model (optional)

1. Fuse the adapter.
2. Write a model card: base model, dataset manifest, training config, evaluation, license, intended use, limitations, safety.
3. Upload the MLX 4-bit model:

```bash
hf upload <user>/qwen3-14b-pentest-4bit fused/qwen3-14b-pentest-4bit .
```

4. Upload the GGUF as a separate repo or a `gguf/` folder.
5. Include the adapter (small) if you want reproducibility:

```bash
hf upload <user>/qwen3-14b-pentest-adapter adapters/qwen3-14b-pentest .
```

Do **not** publish if the dataset includes CC-BY-NC, GPL, unknown-license, or leaked CTF content. See [`12-safety-legal.md`](12-safety-legal.md).

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Model type qwen3 not supported for GGUF conversion` | Used `mlx_lm.fuse --export-gguf` | Use llama.cpp `convert_hf_to_gguf.py`. |
| GGUF conversion fails on tokenizer | Missing `tokenizer.model`/config | Ensure the fused directory has tokenizer files; use `--outtype f16`. |
| Server returns prose, no tool calls | Template not applied / parser not detected | `--jinja` for llama.cpp; inspect the rendered prompt; use the Qwen template. |
| `tool_calls` arguments are a string, not an object | OpenAI convention | Parse `json.loads(arguments)` client-side. |
| Ollama ignores `num_ctx` | Parameter not set or app override | Set `PARAMETER num_ctx`; restart; check `ollama show`. |
| Out of memory at 16K | KV cache + weights | Reduce `-c`/`num_ctx`, use `--kv-bits 4` (MLX), Q4_K_M. |
| Very slow first token | Prefill of a long system/tool block | Shrink tools, raise `--prefill-step-size` carefully, cache prefix. |
| Tunnel dies | SSH idle timeout | `autossh -M 0 -N -L ...`. |
| LAN client cannot connect | Server bound to 127.0.0.1 | Use the tunnel, or bind to the VMnet address behind a firewall. |

---

## 11. Sources

- MLX-LM server guide — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md>
- MLX-LM `fuse.py` (GGUF restriction) — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/fuse.py>
- llama.cpp conversion — <https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py>
- llama.cpp server — <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- llama.cpp function calling — <https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md>
- Ollama import — <https://docs.ollama.com/import>
- Ollama OpenAI compatibility — <https://docs.ollama.com/api/openai-compatibility>
- Ollama macOS — <https://docs.ollama.com/macos>
- Qwen3-14B — <https://huggingface.co/Qwen/Qwen3-14B>
