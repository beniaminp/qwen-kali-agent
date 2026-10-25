# 13 — Troubleshooting: symptom → cause → fix

*Last verified: 25 October 2026. Commands assume the project venv is active.*

This is the chapter to open when something fails. It is organized by phase. When a fix involves a destructive action (delete cache, reboot), the safer alternative is listed first.

---

## 1. Environment and installation

| Symptom | Likely cause | Fix |
|---|---|---|
| `zsh: command not found: uv` | Homebrew path not in shell | `eval "$(/opt/homebrew/bin/brew shellenv)"`; add to `~/.zshrc`. |
| `ImportError: No module named 'mlx'` | Wrong venv / x86 Python | `uname -m` → `arm64`; `which python`; recreate with `uv venv --python 3.12`. |
| `mlx` imports but `default_device()` is CPU | Rosetta or bad wheel | Ensure arm64 Python; reinstall `mlx-lm`; check `pip debug --verbose`. |
| `xcrun: error: invalid active developer path` | CLT missing | `xcode-select --install`; `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`. |
| `transformers` version conflict | mlx-lm 0.32 requires transformers ≥5.7 | Let `uv` resolve; do not pin 4.x. |
| Hugging Face 401 | Gated repo or not logged in | Accept terms on the page; `hf auth login`. |
| `No space left on device` | HF cache on a small volume | `export HF_HOME=/Volumes/Big/hf`; delete unused `models--*` blobs. |
| Extremely slow first generation | Metal shader compilation | Normal on first run; subsequent runs are faster. |

---

## 2. Model download and load

| Symptom | Cause | Fix |
|---|---|---|
| Download stalls | Network / proxy | Retry; `HF_HUB_ENABLE_HF_TRANSFER=0`; use `hf download --resume`. |
| `Safetensors` shape mismatch | Wrong base model for the adapter | Fuse/serve with the exact base used for training. |
| `KeyError: model.layers` | Unsupported architecture | Check `mlx_lm/models/`; update mlx-lm. |
| Load succeeds, output is gibberish | Wrong tokenizer/template, or a corrupt quantized model | Re-download; test the base without the adapter; verify the chat template. |
| `trust_remote_code` prompt | Custom tokenizer | Pass `--trust-remote-code` only if you trust the repo. |

---

## 3. Dataset and tokenization

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: train.jsonl` | Wrong `--data` path | `--data` points to the **directory**, not the file. |
| `KeyError: messages` | Wrong schema | Each line needs `{"messages":[...]}`; see [`06-data-format.md`](06-data-format.md). |
| Tool call not rendered | `tools` missing | Add the `tools` array; inspect the rendered prompt. |
| `arguments` is a string but template expects dict | Qwen3.5/3.6 XML template | Parse to a dict before rendering; see §4 of chapter 06. |
| Truncated tool call | `max_seq_length` too small | Measure the 99th percentile; split or drop long rows. |
| `nan` loss at step 1 | Prompt ≥ max sequence with `--mask-prompt` | Find rows with `token_stats`; drop/split; rerun. |
| Duplicate rows | Multi-source merge | Hash normalized messages; dedup. |
| Model learns to refuse everything | Refusal data overrepresented or too uniform | Reduce refusal share; diversify scope examples; add in-scope positives. |

---

## 4. Training

| Symptom | Cause | Fix |
|---|---|---|
| `kIOGPUCommandBufferCallbackErrorOutOfMemory` | True GPU OOM | batch 1, grad-checkpoint, seq 1536, `--num-layers 8`, close apps. |
| `[metal::malloc] Resource limit (499000) exceeded` | Descriptor/handle leak | Restart process; update mlx-lm; reduce layers/seq; `--clear-cache-threshold 2GB`. |
| Process killed without traceback | Wired-memory pressure | Lower memory use; reboot; do not set the wired limit near 32 GB. |
| `RuntimeError: There is no Stream(gpu, 0) ...` | Mixed CPU/GPU arrays | Ensure the model and data are on the default device; reinstall MLX. |
| Tokens/sec collapses | Thermal throttling or swap | Cool the machine; close apps; check `memory_pressure`. |
| Loss stuck >1.0 | LR too low, data too diverse, mask wrong | Verify mask; raise LR to 1e-4; inspect a batch. |
| Loss spikes then recovers | LR too high / bad batch | Lower LR; clip? (MLX LoRA has no grad clip flag; reduce LR.) |
| Val loss rises while train falls | Overfitting | Stop earlier; add dropout; reduce epochs; more diverse data. |
| Loss <0.2 | Memorization/contamination | Check duplicates; evaluate on held-out tasks. |
| Adapter has no effect | Wrong `--adapter-path`, or fused the wrong model | Verify the path; test base vs adapter on the same prompt. |
| `--resume-adapter-file` starts from scratch | It loads weights only, not iteration/optimizer | Set `iters` for the remaining run; expect a schedule restart. |

**Diagnostic commands:**

```bash
memory_pressure | head -20
vm_stat | head -20
sysctl iogpu.wired_limit_mb
ps aux | sort -nrk 4 | head
```

---

## 5. Fusion and GGUF

| Symptom | Cause | Fix |
|---|---|---|
| `Model type qwen3 not supported for GGUF conversion` | Used `mlx_lm.fuse --export-gguf` | Use llama.cpp `convert_hf_to_gguf.py`; mlx fuse only supports Llama/Mistral/Mixtral. |
| GGUF conversion can't find tokenizer | Fused dir missing tokenizer files | Ensure `tokenizer.json`, `tokenizer_config.json`, `vocab.json`/`merges.txt` are present. |
| `llama-quantize` not found | llama.cpp not built | `cmake -B build -DGGML_METAL=ON && cmake --build build -j`. |
| GGUF loads but emits garbage | Wrong prompt template | Use `--jinja`; pass the Qwen template; verify with a simple prompt. |
| Fused model is 30 GB | `--dequantize` produced bf16 | Expected; quantize to GGUF, or skip dequantize for MLX serving. |
| Disk full during fusion | bf16 copy + original | Free 60+ GB; fuse to an external volume. |

---

## 6. Serving

| Symptom | Cause | Fix |
|---|---|---|
| `Address already in use` | Another server on 8080 | `lsof -i :8080`; kill or use another port. |
| Client gets 404 | Wrong base URL | Use `http://host:8080/v1`, not `/v1/chat/completions` as base. |
| `model` not found | Server expects a path/name it knows | Use the model name shown by `/v1/models`, or omit it. |
| No tool calls in response | Template/parser mismatch | Test with `curl`; check server logs; try `llama-server --jinja`. |
| Ollama ignores context | `num_ctx` not set | `PARAMETER num_ctx 16384`; restart; `ollama show`. |
| OOM at 16K | KV cache + weights | Reduce context; Q4_K_M; `--kv-bits 4` for MLX. |
| Very slow prefill | Long system/tool block | Shrink tools; cache prefix; raise prefill step cautiously. |
| LAN client cannot connect | Bound to 127.0.0.1 | SSH tunnel, or bind to the VMnet address behind a firewall. |
| Tunnel dies | SSH idle timeout | `autossh -M 0 -N -L ...`. |

---

## 7. Agent and MCP

| Symptom | Cause | Fix |
|---|---|---|
| OpenCode can't see MCP tools | Wrong command/venv path | Absolute paths; `opencode mcp list`; test the client manually. |
| MCP bridge 500 | Missing Python deps | Activate `/opt/MCP-Kali-Server/.venv`; `pip install -r requirements.txt`. |
| Agent loops the same call | No recovery training; no policy cap | Add recovery data; cap repeated calls; inject the error into the next prompt. |
| Agent invents tool output | Trained on synthetic observations | Retrain on executed output only; add “do not invent” examples. |
| Agent scans out of scope | No policy layer; bad training data | Deny at policy; add refusal examples; evaluate UAR. |
| Arguments invalid | Schema not in training data | Add schema examples; validate and return a corrective error. |
| Too many tools | 150-tool MCP server | Expose 6–10 typed tools; hide the rest. |

---

## 8. Evaluation

| Symptom | Cause | Fix |
|---|---|---|
| Scores too good | Contamination / train-test leak | 13-gram check; hold out by task family. |
| Scores vary wildly | Temperature >0 / small n | Use temperature 0 for pass/fail; report CIs. |
| Judge disagrees with humans | Bad rubric / no calibration | Label 30 examples; rewrite rubric; blind the judge. |
| Eval server OOM | Same Mac as training | Stop training; serve only; reduce context. |
| Tool calls parse in eval but not in agent | Different client/template | Use the same OpenAI client and schema in both. |

---

## 9. macOS and hardware

| Symptom | Cause | Fix |
|---|---|---|
| Fans loud, performance drops | Thermal throttling | Cool the machine; shorter runs; laptop stand. |
| Sleep interrupts training | Power settings | `sudo pmset -a sleep 0 disksleep 0`; restore later. |
| Kernel panic / reboot | Wired-memory overcommit | Lower `iogpu.wired_limit_mb`; reduce memory; reboot. |
| External display slows training | GPU/window server load | Close the lid? (thermal trade-off) or disconnect the display. |
| Battery drains while training | Not charging | Connect the charger; check the cable. |

---

## 10. Collecting a useful bug report

If you need to ask for help, include:

```bash
sw_vers
uname -m
python --version
python -c "import mlx.core as mx; print(mx.__version__)"
python -c "import mlx_lm; print(mlx_lm.__version__)"
sysctl hw.memsize iogpu.wired_limit_mb
```

Plus:

- the exact command and config (redact paths/secrets);
- the last 50 lines of the log;
- whether the base model works without the adapter;
- the dataset line that triggers it (redact content);
- whether it reproduces on the 8B.

Most “MLX is broken” reports turn out to be a template mismatch, a too-long sequence, or an adapter/base mismatch. Check those first.

---

## 11. Quick decision tree

```text
Does the base model work?
├─ No → installation/model/template problem (ch. 03/06)
└─ Yes
   ├─ Does the adapter load? 
   │  ├─ No → path/base mismatch
   │  └─ Yes
   │     ├─ Does it call tools?
   │     │  ├─ No → training data/template/mask
   │     │  └─ Yes
   │     │     ├─ Right tool/args?
   │     │     │  ├─ No → data quality/schema
   │     │     │  └─ Yes
   │     │     │     ├─ Completes tasks?
   │     │     │     │  ├─ No → recovery/agent scaffold
   │     │     │     │  └─ Yes → evaluate safety, ship
```

---

## 12. Sources

- MLX-LM issues — <https://github.com/ml-explore/mlx-lm/issues>
- llama.cpp server docs — <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- Ollama FAQ — <https://docs.ollama.com/faq>
- OpenCode troubleshooting — <https://opencode.ai/docs/troubleshooting>
- Apple Silicon memory pressure — <https://support.apple.com/guide/activity-monitor/>
