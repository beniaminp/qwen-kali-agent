# 03 — macOS setup from zero

*Target: a clean Apple Silicon Mac (M1 Pro, 32 GB) running macOS 26.x. Last verified: 25 October 2026.*

This chapter builds a reproducible environment with **uv**, a project-local virtual environment, and pinned MLX versions. It ends with a smoke test that proves Metal, the model download, and the chat template all work before you spend time on data.

---

## 1. What you need

| Requirement | Notes |
|---|---|
| Apple Silicon Mac | M1 or newer. Intel Macs cannot use MLX. |
| 32 GB unified memory | 16 GB can do 4B–8B with care; 32 GB is the minimum for a comfortable 14B QLoRA. |
| 100–150 GB free disk | Models, checkpoints, fused weights, datasets. |
| macOS 14+ | macOS 26.x recommended for the current MLX/Metal stack. |
| Xcode Command Line Tools | Metal compiler and headers. |
| Homebrew | For `uv`, `python@3.12`, `git-lfs`, `jq`, optional `llama.cpp`. |
| Python 3.11 or 3.12 | MLX-LM requires ≥3.11. 3.12 is the tested default here. |
| A Hugging Face account | Some datasets are gated; a token also avoids rate limits. |

You do **not** need Docker on the Mac for training. Docker is only for the synthetic-data sandbox and the lab, and it should not run during training.

---

## 2. Base tools

```bash
# 1. Command Line Tools (accept the GUI prompt)
xcode-select --install

# 2. Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# Follow the printed instructions to add brew to your PATH (Apple Silicon: /opt/homebrew/bin).

# 3. Core CLI tools
brew update
brew install uv python@3.12 git git-lfs jq htop
git lfs install

# 4. Optional: GGUF conversion and serving
brew install llama.cpp

# 5. Optional: Ollama app (GUI) — https://ollama.com/download
```

Verify:

```bash
sw_vers
uname -m                 # arm64
xcode-select -p
uv --version
python3.12 --version
llama-server --version 2>/dev/null || true
```

**Why `uv`?** It resolves and installs the MLX/transformers dependency tree in seconds, creates isolated environments, and makes the commands reproducible. `pip` works too, but you will wait longer and hit resolution conflicts more often.

---

## 3. Project layout

```bash
mkdir -p ~/work/qwen-kali-agent
cd ~/work/qwen-kali-agent
git clone https://github.com/beniaminp/qwen-kali-agent.git .
# or: unzip the handbook and cd into it

mkdir -p data/{raw,interim,processed,golden} \
         adapters fused mlx_models models logs reports
```

Set a dedicated Hugging Face cache so models do not fill your system drive unexpectedly:

```bash
# ~/.zshrc
export HF_HOME="$HOME/work/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
```

```bash
mkdir -p "$HF_HOME"
source ~/.zshrc
```

`HF_HUB_ENABLE_HF_TRANSFER=1` speeds up large downloads but requires `hf_transfer` (`uv pip install hf_transfer`). If it causes errors, unset it.

---

## 4. Python environment

```bash
cd ~/work/qwen-kali-agent

uv venv --python 3.12 .venv
source .venv/bin/activate

# Pin the stack. Check PyPI for the latest patch before a long run.
uv pip install \
  "mlx-lm[train]==0.32.2" \
  "huggingface-hub[cli]" \
  datasets \
  transformers \
  openai \
  pyyaml \
  jsonschema \
  rich \
  tqdm \
  matplotlib

python -c "import mlx.core as mx; print('mlx', mx.__version__ if hasattr(mx,'__version__') else 'ok')"
python -c "import mlx_lm; print('mlx-lm', mlx_lm.__version__)"
```

As of 25 Oct 2026, the tested pair is **mlx-lm 0.32.2** with **mlx 0.32.4** and **transformers 5.x**. MLX-LM 0.32.x requires `transformers>=5.7.0`; do not downgrade transformers to 4.x without checking the release notes. If you need a fully frozen environment:

```bash
uv pip freeze > requirements.lock.txt
```

**Do not install `mlx` separately from `mlx-lm`** unless you are debugging. `mlx-lm` pins a compatible MLX; a mismatched pair produces confusing Metal errors.

---

## 5. Hugging Face access

```bash
hf auth login
# paste a read token from https://huggingface.co/settings/tokens
```

For gated datasets (xLAM, some cyber sets), accept the terms on the dataset page first, then:

```bash
hf download Salesforce/xlam-function-calling-60k --repo-type dataset --local-dir data/raw/xlam
```

If a repo returns 401, it is gated or does not exist. The dataset research in [`05-datasets.md`](05-datasets.md) flags the known gated IDs.

---

## 6. Verify Metal and the memory budget

```bash
python - <<'PY'
import mlx.core as mx
info = mx.device_info()
for k, v in info.items():
    if isinstance(v, int) and v > 1024**2:
        print(f"{k}: {v/1024**3:.2f} GiB")
    else:
        print(f"{k}: {v}")
print("default device:", mx.default_device())
PY
```

Expected: `device_type: gpu`, `architecture: metal`, and `max_recommended_working_set_size` around 21–22 GiB on a 32 GB machine. If it says CPU, MLX did not build with Metal — reinstall the wheel for arm64 and check that you are not running under Rosetta (`uname -m` must be `arm64`).

Check the wired limit:

```bash
sysctl iogpu.wired_limit_mb
```

`0` means default. Do not change it yet; the training chapter explains when raising it helps.

Run the environment checker:

```bash
python scripts/check_env.py
```

It reports macOS version, arch, Python, MLX, free disk, RAM, default working set, and whether the expected model/tooling imports succeed.

---

## 7. Smoke test: template before training

Download a tiny model and verify the chat template renders tools correctly. This catches template/tokenizer breakage in minutes instead of after a 20-hour run.

```bash
python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
messages = [
    {"role": "system", "content": "You are a careful security assistant."},
    {"role": "user", "content": "Scan 10.10.10.5 for open services."},
    {"role": "assistant", "tool_calls": [{
        "id": "call_1", "type": "function",
        "function": {"name": "nmap_scan",
                     "arguments": "{\"target\":\"10.10.10.5\",\"flags\":\"-sV\"}"}
    }]},
    {"role": "tool", "tool_call_id": "call_1", "content": "22/tcp open ssh\n80/tcp open http"},
    {"role": "assistant", "content": "Ports 22 and 80 are open."},
]
tools = [{"type": "function", "function": {
    "name": "nmap_scan",
    "description": "Run an nmap scan against an authorized target.",
    "parameters": {"type": "object",
                   "properties": {"target": {"type": "string"},
                                  "flags": {"type": "string"}},
                   "required": ["target"]}}}]
print(tok.apply_chat_template(messages, tools=tools, tokenize=False))
PY
```

You should see `<|im_start|>system`, a `<tools>` block with JSON, `<tool_call>` with `{"name": ..., "arguments": ...}`, and a `<tool_response>` inside a user turn. If the output is truncated, the assistant content is missing, or the tool block is malformed, stop and fix the template before training. Chapter [`06-data-format.md`](06-data-format.md) explains every line.

---

## 8. Optional: serving stack

### MLX server

Already installed with `mlx-lm`:

```bash
mlx_lm.server --help | head -40
```

### llama.cpp

```bash
brew install llama.cpp
llama-server --version
# Conversion scripts live in the Homebrew share or the repo:
# brew --prefix llama.cpp
```

If the Homebrew build lacks the Python conversion scripts, clone the repo:

```bash
git clone https://github.com/ggml-org/llama.cpp ~/work/llama.cpp
cd ~/work/llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j
```

### Ollama

Install the app, then verify:

```bash
ollama --version
ollama list
```

Ollama runs a background service on `127.0.0.1:11434`. It is convenient for GGUF inference, but it cannot apply an MLX/PEFT adapter directory directly; you must fuse and convert first.

---

### Java / Quarkus toolchain (optional track)

The MLX environment does not need a JDK, but the Java/Quarkus track does:

```bash
# Temurin 25 LTS (or install the JDK 25 package from Adoptium)
brew install --cask temurin@25
brew install maven

java -version      # 25.x
mvn -version       # 3.9+
```

Quarkus projects should use their Maven wrapper (`./mvnw`), so the system Maven is only a fallback. The Quarkus CLI is optional:

```bash
brew install quarkusio/tap/quarkus
quarkus --version
```

You do not need the CLI to build or test; `./mvnw quarkus:dev` and `./mvnw test` are enough.

### Documentation site

Material for MkDocs is separate from the MLX environment, but installs cleanly into the same venv:

```bash
uv pip install -r requirements-docs.txt
mkdocs serve --dev-addr 127.0.0.1:8000
mkdocs build --strict
```

The site is styled as a GitBook (sidebar, search, grouped parts, edit links) and lives in `site/`. The live copy is <https://beniaminp.github.io/qwen-kali-agent/>; `.github/workflows/docs.yml` publishes to `gh-pages`.

## 9. Reproducibility notes

- **Pin versions** before a long run. Write them into `requirements.lock.txt` and record the git commit of this repo.
- **Record the model revision.** `from_pretrained(..., revision="<sha>")` or `hf download ... --revision <sha>`. Qwen occasionally updates tokenizer/config files.
- **Record the dataset manifest.** A JSON file with each source ID, revision, row count, hash, license.
- **Do not upgrade mid-run.** MLX and transformers both move; a resumed checkpoint may not load.
- **Keep the raw data immutable.** All cleaning goes to `data/interim` and `data/processed`.

---

## 10. Common setup failures

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: No module named 'mlx'` | Wrong Python/venv, or x86 Python under Rosetta | `uname -m` → `arm64`; recreate the venv with `uv venv --python 3.12`; ensure the venv is activated. |
| `mlx` installs but `default_device()` is CPU | Rosetta or an old wheel | Reinstall on arm64; check `pip debug --verbose` platform tags. |
| `xcrun: error: invalid active developer path` | CLT missing/broken | `xcode-select --install`, then `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer` if needed. |
| `OSError: [Errno 28] No space left on device` | HF cache on a small volume | Set `HF_HOME` to an external/large volume; delete old `models--*` blobs. |
| 401 from Hugging Face | Gated repo or not logged in | Accept terms on the website; `hf auth login`. |
| `transformers` version conflict | mlx-lm 0.32 requires transformers 5.x | Let `uv` resolve; do not force an old pin. |
| `metal::malloc` on first model load | Wired limit too low or another app hogging GPU | Close apps; check `mx.device_info()`; reboot if a leaked process persists. |
| Extremely slow first run | Model download + Metal shader compilation | The first generation compiles kernels; subsequent runs are faster. |

---

## 11. Sources

- MLX-LM installation and training — <https://github.com/ml-explore/mlx-lm>
- MLX-LM LoRA guide — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md>
- MLX device info / wired limit — <https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html>
- uv — <https://docs.astral.sh/uv/>
- Hugging Face CLI — <https://huggingface.co/docs/huggingface_hub/guides/cli>
- llama.cpp build — <https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md>
- Ollama macOS — <https://docs.ollama.com/macos>
