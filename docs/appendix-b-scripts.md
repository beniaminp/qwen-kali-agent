# Appendix B — Scripts

All scripts live in `scripts/` and are run from the repository root with the project venv active. They are deliberately small, dependency-light, and safe by default.

| Script | Purpose | Example |
|---|---|---|
| `check_env.py` | Verify macOS, arch, Python, MLX, memory, disk. | `python scripts/check_env.py` |
| `validate_dataset.py` | Schema, tool-name, argument, secret and destructive-pattern checks. | `python scripts/validate_dataset.py data/processed/train.jsonl --tools examples/tools/kali-tools.json` |
| `token_stats.py` | Token-length distribution; recommends `max_seq_length`. | `python scripts/token_stats.py data/processed/train.jsonl --model Qwen/Qwen3-14B` |
| `check_lora.py` | Print adapted blocks, projection counts and trainable params before training. | `python scripts/check_lora.py --model mlx-community/Qwen3-14B-4bit --num-layers -1 --rank 16 --scale 2` |
| `make_sweep.py` | Generate rank/LR/layer configs and a `commands.sh`. | `python scripts/make_sweep.py --base configs/qwen3-14b-all-layers.yaml --out configs/sweeps/r1 --rank 16,32 --lr 5e-5,1e-4` |
| `prepare_dataset.py` | Normalize a canonical JSONL, split train/valid/test, write a manifest. | `python scripts/prepare_dataset.py data/interim/all.jsonl --out data/processed --split 0.9 0.05 0.05` |
| `split_dataset.py` | Deterministic split with exact/near-dup filtering. | `python scripts/split_dataset.py data/interim/all.jsonl --out data/processed` |
| `generate_synthetic.py` | DeepSeek/OpenAI-compatible generation with retries, cost tracking and JSON validation. | `python scripts/generate_synthetic.py --tasks 50 --out data/interim/gen.jsonl` |
| `execute_verified.py` | Execute proposed commands in a disposable Docker Kali container; capture observations; build trajectories. | `python scripts/execute_verified.py --input data/interim/plans.jsonl --out data/interim/verified.jsonl --i-understand` |
| `eval_toolcall.py` | Run a golden set against an OpenAI-compatible endpoint; schema/tool/refusal graders; JSON report. | `python scripts/eval_toolcall.py --base-url http://127.0.0.1:8080/v1 --golden data/golden/golden.jsonl --out reports/eval.json` |
| `smoke_test.py` | 10 canonical prompts to verify tool-call fidelity. | `python scripts/smoke_test.py --base-url http://127.0.0.1:8080/v1` |
| `make_dataset_card.py` | Generate a dataset card skeleton from a manifest. | `python scripts/make_dataset_card.py data/processed/manifest.json --out data/cards/dataset-card.md` |
| `export_gguf.sh` | Fuse, dequantize, convert, quantize. | `bash scripts/export_gguf.sh` |
| `serve_mlx.sh` | Launch `mlx_lm.server` with safe defaults. | `bash scripts/serve_mlx.sh` |

---

## B.1 `check_env.py`

Checks:

- macOS and `arm64`;
- Python ≥3.11;
- `mlx`, `mlx_lm`, `transformers` versions;
- `mx.device_info()` and the recommended working set;
- free disk in the repo volume and `HF_HOME`;
- whether the expected model/tool files exist.

It exits non-zero on a hard failure (wrong arch, missing MLX).

---

## B.2 `validate_dataset.py`

Validates one JSONL file. Exit code 0 = no hard errors. Warnings do not fail the run.

Hard checks:

- JSON parse;
- `messages` array, valid roles, non-empty;
- tool name membership;
- `arguments` parses as a JSON object;
- required fields;
- tool message follows an assistant tool call;
- no ` thinking` in `content`;
- no empty assistant message.

Soft checks (warnings):

- first message is not `system`;
- destructive patterns;
- secret patterns;
- suspiciously short/long rows;
- duplicate IDs.

Use `--strict` to make warnings fail in CI.

---

## B.3 `token_stats.py`

Loads the tokenizer (default `Qwen/Qwen3-14B`), renders each row with its `tools`, and reports:

- count, min, p50, p90, p95, p99, max;
- how many rows exceed 2048/4096/8192;
- a recommended `max_seq_length` at the 99th percentile rounded to 256.

It does not require MLX; it uses `transformers` only. It downloads the tokenizer once into `HF_HOME`.

---

## B.4 `prepare_dataset.py`

Input: canonical JSONL with `messages` and optional `tools`.
Output:

```text
data/processed/
├── train.jsonl
├── valid.jsonl
├── test.jsonl
└── manifest.json
```

It:

- normalizes tool names to `[A-Za-z0-9_-]`;
- parses string `arguments` to objects and re-serializes them;
- drops rows missing a tool definition;
- exact-dedups on normalized messages;
- splits deterministically by hash (no RNG);
- writes counts, token estimates, and a SHA-256 for each split.

It does **not** near-dedup by default; use `split_dataset.py --minhash` for that.

---

## B.5 `split_dataset.py`

Deterministic split with optional MinHash near-duplicate removal. Use it when merging multiple sources with paraphrases.

```bash
python scripts/split_dataset.py data/interim/all.jsonl \
  --out data/processed --minhash --threshold 0.8
```

Requires `datasketch` (`uv pip install datasketch`). If not installed, it falls back to exact dedup and warns.

---

## B.6 `generate_synthetic.py`

A **scaffold**, not a turnkey data factory. It:

- reads a task taxonomy or a list of prompts;
- calls an OpenAI-compatible endpoint (DeepSeek by default);
- requests JSON, validates it, retries with backoff;
- tracks input/output tokens and estimated USD;
- writes one JSON object per line with provenance (`teacher`, `prompt_hash`, `created`).

It does **not** execute commands. Pair it with `execute_verified.py`.

```bash
export DEEPSEEK_API_KEY=...
python scripts/generate_synthetic.py \
  --tasks 25 --model deepseek-flash --concurrency 8 \
  --out data/interim/tasks.jsonl --dry-run
```

`--dry-run` prints the prompts and cost estimate without calling the API.

---

## B.7 `execute_verified.py`

The dangerous one. Safety properties:

- refuses to run without `--i-understand`;
- requires Docker and a container named `kali-lab` (or `--container`);
- refuses if the container has `--network host`;
- default `--dry-run` prints the commands and exits;
- per-command timeout (default 60 s);
- allowlist of binaries;
- destructive-pattern denylist;
- captures stdout/stderr/exit code, truncates to 4,000 chars;
- writes a JSONL trajectory per plan.

```bash
# 1. Start the lab (no egress)
docker compose -f examples/docker-compose.lab.yml up -d

# 2. Inspect what would run
python scripts/execute_verified.py --input data/interim/plans.jsonl --out /tmp/x.jsonl

# 3. Execute (only after reviewing)
python scripts/execute_verified.py --input data/interim/plans.jsonl \
  --out data/interim/verified.jsonl --container kali-lab --i-understand
```

It never executes on the host. If Docker is unavailable, it exits with an error.

---

## B.8 `eval_toolcall.py`

Runs a golden set against an OpenAI-compatible endpoint. For each task:

1. builds the system prompt + scope + tools;
2. sends the user task;
3. if the model returns tool calls, grades names/arguments (and optionally executes them via a sandbox URL);
4. if the model returns text, grades refusal/no-call;
5. records turns, latency, pass/fail, reason.

Output: a JSON report with per-category metrics, Wilson intervals, and a failure list.

```bash
python scripts/eval_toolcall.py \
  --base-url http://127.0.0.1:8080/v1 \
  --model local \
  --golden data/golden/golden.jsonl \
  --out reports/14b-r1.json
```

`--sandbox http://127.0.0.1:5000` enables execution grading (optional).

---

## B.9 `smoke_test.py`

Ten prompts covering:

- simple tool call;
- multiple tools;
- tool + argument enum;
- no-call question;
- out-of-scope refusal;
- ambiguous target;
- malicious/destructive request;
- malformed tool response;
- long system prompt;
- parallel-safe calls.

It checks tool name membership, JSON validity, and forbidden targets. Exit non-zero if any hard check fails.

---

## B.10 `make_dataset_card.py`

Reads `manifest.json`, samples rows, and writes a Markdown card with:

- intended use / out-of-scope use;
- sources and licenses;
- preprocessing and dedup;
- counts and token estimates;
- known issues;
- evaluation and maintenance.

It does not invent license text; it copies the manifest fields.

---

## B.11 `check_lora.py`

Loads the base model, applies the same `linear_to_lora_layers` conversion MLX-LM uses, and reports:

- transformer block count;
- selected vs adapted blocks;
- number of LoRA modules and counts by projection (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`);
- trainable parameters and percentage;
- sample module names.

Use it to catch attention-only adapters, stale MLX versions, or a custom `lora_parameters.keys` that silently excludes MLP. Exit code 2 means the module count is unusually low.

```bash
python scripts/check_lora.py --model mlx-community/Qwen3-14B-4bit \
  --num-layers -1 --rank 16 --scale 2
```

Requires MLX; it will not run on Linux.

---

## B.12 `make_sweep.py`

Generates a Cartesian product of rank, learning rate and layer count from a base YAML. It sets `scale = 32 / rank` by default, writes one YAML per run, a `commands.sh`, and a `sweep.md` table.

```bash
python scripts/make_sweep.py \
  --base configs/qwen3-14b-all-layers.yaml \
  --out configs/sweeps/r1 \
  --rank 16,32 --lr 5e-5,1e-4 --layers all,16
bash configs/sweeps/r1/commands.sh
```

Each run gets a unique `adapter_path` under `adapters/sweeps/`. It does not launch training; review the configs and the time budget first.

---

## B.13 Shell helpers

`export_gguf.sh`:

1. checks the adapter and base exist;
2. fuses with `--dequantize`;
3. runs `convert_hf_to_gguf.py --outtype f16`;
4. runs `llama-quantize Q4_K_M`;
5. prints sizes and checksums.

`serve_mlx.sh`:

1. activates the venv if present;
2. binds `127.0.0.1:8080`;
3. sets `--max-tokens 2048`, `--temp 0.2`, `--prefill-step-size 512`;
4. does **not** set `enable_thinking:false` (train/serve mismatch);
5. prints the SSH tunnel command for Kali.

---

## B.14 Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | validation/hard failure |
| 2 | usage error |
| 3 | unsafe conditions (Docker/host network/scope) |
| 4 | upstream API failure after retries |
