# Appendix B — Scripts

All scripts live in `scripts/` and are run from the repository root with the project venv active. They are deliberately small, dependency-light, and safe by default.

| Script | Purpose | Example |
|---|---|---|
| `check_env.py` | Verify macOS, arch, Python, MLX, memory, disk. | `python scripts/check_env.py` |
| `validate_dataset.py` | Schema, tool-name, argument, secret and destructive-pattern checks. | `python scripts/validate_dataset.py data/processed/train.jsonl --tools examples/tools/kali-tools.json` |
| `token_stats.py` | Token-length distribution; recommends `max_seq_length`. | `python scripts/token_stats.py data/processed/train.jsonl --model Qwen/Qwen3-14B` |
| `check_lora.py` | Print adapted blocks, projection counts and trainable params before training. | `python scripts/check_lora.py --model mlx-community/Qwen3-14B-4bit --num-layers -1 --rank 16 --scale 2` |
| `make_sweep.py` | Generate rank/LR/layer configs and a `commands.sh`. | `python scripts/make_sweep.py --base configs/qwen3-14b-all-layers.yaml --out configs/sweeps/r1 --rank 16,32 --lr 5e-5,1e-4` |
| `validate_java_dataset.py` | Java/Quarkus-specific lint: javax, Spring, version tags, secrets, argument shape. | `python scripts/validate_java_dataset.py data/java/processed/train.jsonl --tools examples/java/tools.json --require-dict-arguments --strict` |
| `mine_java_commits.py` | Mine Java commit seeds (base commit, task, patch, test command) from a git repo. | `python scripts/mine_java_commits.py --repo ~/src/quarkus-quickstarts --out data/java/interim/seeds.jsonl --limit 300 --only-with-tests` |
| `validate_react_dataset.py` | React/Tailwind lint: v3 directives, legacy React, dynamic classes, XSS, a11y smells, argument shape. | `python scripts/validate_react_dataset.py data/react/processed/train.jsonl --tools examples/react/tools.json --require-dict-arguments --strict` |
| `mine_react_commits.py` | Mine React/TS commit seeds from a git repo. | `python scripts/mine_react_commits.py --repo ~/src/shadcn-ui/ui --out data/react/interim/seeds.jsonl --limit 300 --only-with-tests` |
| `mix_datasets.py` | Mix track JSONL with weights, convert tool arguments, split deterministically. | `python scripts/mix_datasets.py --source kali=... --source java=... --source react=... --out data/unified/processed --arguments-format string` |
| `validate_gemma4_dataset.py` | Gemma 4 lint: object arguments, tool ids, native call syntax, thinking history. | `python scripts/validate_gemma4_dataset.py data/gemma4/react/processed/train.jsonl --tools examples/react/tools.json --strict` |
| `compare_models.py` | Send the same prompts/tools to two OpenAI-compatible endpoints and record both. | `python scripts/compare_models.py --a-base-url ... --b-base-url ... --prompts ... --tools ... --out reports/compare.jsonl` |
| `estimate_pretraining.py` | Parameters, FLOPs, GPU-hours, cost and KV cache from an architecture config. | `python scripts/estimate_pretraining.py --preset qwen3-14b --tokens 36e12 --gpus 10000 --mfu 0.4` |
| `inspect_architecture.py` | Read a local/HF `config.json`, print per-component params, shapes, GQA, cache and FLOPs. | `python scripts/inspect_architecture.py --hf Qwen/Qwen3-14B --seq 32768` |
| `train_tiny_transformer.py` | Runnable tiny Llama-style model with RMSNorm, RoPE, GQA and SwiGLU. | `python scripts/train_tiny_transformer.py --data data/tiny/corpus.txt --out runs/tiny --dim 256 --layers 6` |
| `prepare_dataset.py` | Normalize a canonical JSONL, split train/valid/test, write a manifest. | `python scripts/prepare_dataset.py data/interim/all.jsonl --out data/processed --split 0.9 0.05 0.05` |
| `split_dataset.py` | Deterministic split with exact/near-dup filtering. | `python scripts/split_dataset.py data/interim/all.jsonl --out data/processed` |
| `generate_synthetic.py` | DeepSeek/OpenAI-compatible generation with retries, cost tracking and JSON validation. | `python scripts/generate_synthetic.py --tasks 50 --out data/interim/gen.jsonl` |
| `execute_verified.py` | Execute proposed commands in a disposable Docker Kali container; capture observations; build trajectories. | `python scripts/execute_verified.py --input data/interim/plans.jsonl --out data/interim/verified.jsonl --i-understand` |
| `eval_toolcall.py` | Run a golden set against an OpenAI-compatible endpoint; schema/tool/refusal graders; JSON report. | `python scripts/eval_toolcall.py --base-url http://127.0.0.1:8080/v1 --golden data/golden/golden.jsonl --out reports/eval.json` |
| `smoke_test.py` | 10 canonical prompts to verify tool-call fidelity. | `python scripts/smoke_test.py --base-url http://127.0.0.1:8080/v1` |
| `make_dataset_card.py` | Generate a dataset card skeleton from a manifest. | `python scripts/make_dataset_card.py data/processed/manifest.json --out data/cards/dataset-card.md` |
| `generate_distill_data.py` | Teacher traces → canonical JSONL; preserves reasoning/tool calls; resume and dry-run. | `python scripts/generate_distill_data.py --prompts examples/distill/prompts.jsonl --out data/distill/raw/deepseek.jsonl` |
| `distill_logits.py` | GPU reference for CE + top-k KL logit distillation with LoRA. | `python scripts/distill_logits.py --teacher google/gemma-4-12B-it --student google/gemma-4-E4B-it --data ... --out ... --dry-run` |
| `check_math_examples.py` | Assert the arithmetic in the worked-examples chapter. | `python scripts/check_math_examples.py` |
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
- parses `arguments` and writes them in the target shape: JSON string by default,
  or a JSON object with `--arguments-format dict` for Qwen2.5-Coder;
- drops rows missing a tool definition;
- exact-dedups on normalized messages;
- splits deterministically by hash (no RNG);
- writes counts, token estimates, and a SHA-256 for each split.

```bash
python scripts/prepare_dataset.py data/java/interim/all.jsonl \
  --out data/java/processed --arguments-format dict
```

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

`--system-file` and `--prompt-file` override the built-in Kali prompts. The prompt file may use `{n}` for the batch size; this is how the Java/Quarkus task generator is wired:

```bash
python scripts/generate_synthetic.py --mode tasks --tasks 300 --batch 10 \
  --model deepseek-flash \
  --system-file examples/java/teacher-system.md \
  --prompt-file examples/java/task-prompt.md \
  --out data/java/interim/tasks.jsonl
```

For React/Tailwind:

```bash
python scripts/generate_synthetic.py --mode tasks --tasks 300 --batch 10 \
  --model deepseek-flash \
  --system-file examples/react/teacher-system.md \
  --prompt-file examples/react/task-prompt.md \
  --out data/react/interim/tasks.jsonl
```

It does **not** execute commands. Pair it with `execute_verified.py` for Kali, the Maven sandbox in [21 — Java synthetic data](21-java-synthetic.md) for Java, or the Playwright sandbox in [25 — React synthetic data](25-react-synthetic.md) for React.

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

## B.13 `validate_java_dataset.py`

A Java/Quarkus linter for canonical chat JSONL. It checks assistant text and tool arguments for:

- legacy `javax.persistence/ws.rs/inject/enterprise/transaction/validation`;
- Spring idioms (`org.springframework`, `@Autowired`);
- legacy `quarkus-resteasy-reactive`;
- Quarkus/Java version tags other than the target;
- `System.out`, `printStackTrace`, `TODO`/`FIXME`, Lombok;
- probable secrets;
- unbalanced code fences;
- tool arguments stored as strings when `--require-dict-arguments` is set (Qwen2.5-Coder).

It does not compile Java. Pair it with `./mvnw test`.

---

## B.14 `mine_java_commits.py`

Reads a git repository and writes Java commit seeds:

```json
{"schema":"java-commit-task-v1","repo":"...","base_commit":"...","commit":"...",
 "task":"...","files":["..."],"patch":"diff --git ...","test_command":"./mvnw -q -DskipITs test"}
```

Options:

- `--limit` — number of commits to inspect;
- `--since` — git date filter;
- `--only-with-tests` — require a change under `src/test`;
- `--max-patch-bytes`, `--max-files` — skip huge commits;
- `--license` — record the source license.

It is read-only and does not execute code. The seeds are inputs for the LLM + sandbox pipeline in [21 — Java synthetic data](21-java-synthetic.md), not training rows by themselves.

---

## B.15 `validate_react_dataset.py`

A React/Tailwind linter for canonical chat JSONL. It scans code blocks for:

- `@tailwind base/components/utilities` (allowed only in a migration task);
- `tailwind.config.*` in a v4 task;
- removed v4 utilities (`bg-opacity-*`, `flex-shrink-*`, `overflow-ellipsis`, …);
- `outline-none` on focusable elements;
- `ReactDOM.render`, `useFormState`, `React.FC`, class components, `propTypes`, `defaultProps`;
- dynamic Tailwind class construction and arbitrary hex;
- unsanitized `dangerouslySetInnerHTML`;
- `<img>` without `alt`, `<button>` without `type`, clickable `div` without role;
- `target="_blank"` without `rel`;
- `key={index}`, `any`, `@ts-ignore`, `console.log`, `TODO`;
- probable secrets;
- tool arguments stored as strings when `--require-dict-arguments` is set.

It does not compile, render or test. Pair it with `npm run typecheck`, Vitest, Playwright and axe.

---

## B.16 `mine_react_commits.py`

Reads a git repository and writes React/TS commit seeds:

```json
{"schema":"react-commit-task-v1","repo":"...","base_commit":"...","commit":"...",
 "task":"...","files":["..."],"patch":"diff --git ...","setup_command":"npm ci",
 "test_command":"npm run typecheck && npm test -- --run && npm run build"}
```

Options:

- `--limit`, `--since` — commit range;
- `--only-with-tests` — require a changed test/E2E file;
- `--max-patch-bytes`, `--max-files` — skip huge commits;
- `--license` — record the source license.

It detects npm/pnpm/yarn from `package.json` and builds a check command from the scripts that exist. It is read-only and does not execute code. The seeds are inputs for the LLM + browser sandbox in [25 — React synthetic data](25-react-synthetic.md), not training rows by themselves.

---

## B.17 `mix_datasets.py`

Builds one train/valid/test split from several track files:

```bash
python scripts/mix_datasets.py \
  --source kali=data/processed/train.jsonl \
  --source java=data/java/processed/train.jsonl \
  --source react=data/react/processed/train.jsonl \
  --weights kali=0.40,java=0.30,react=0.30 \
  --out data/unified/processed --arguments-format string
```

It:

- converts tool `arguments` to JSON strings (`string`) or objects (`dict`);
- adds `meta.track` when missing;
- exact-dedups rows;
- samples each source to its weight relative to the total available;
- splits each source, then shuffles the combined splits;
- writes a manifest with per-track counts and SHA-256 per file.

Use `--arguments-format string` for Qwen3 and `dict` for Qwen2.5-Coder. `--dry-run` prints counts without writing. It does not near-dedup; use `split_dataset.py --minhash` before mixing if needed.

---

## B.18 `validate_gemma4_dataset.py`

Gemma 4's template is stricter than Qwen's:

- `tool_calls[].function.arguments` must be a **mapping**; a string is a hard error.
- `tool_call_id` must resolve to a preceding tool call.
- Tool names must match `[A-Za-z_][A-Za-z0-9_-]*`.
- Qwen-style `<tool_call>` JSON in content is a warning; let the Gemma template render.
- Previous-turn `reasoning_content` is a warning; Gemma expects final text in history except tool-call turns.
- Secrets and missing `meta.track`/`meta.license` are flagged.

```bash
python scripts/validate_gemma4_dataset.py data/gemma4/react/processed/train.jsonl \
  --tools examples/react/tools.json --strict
```

It is a static linter. Render a row with `AutoProcessor.apply_chat_template` and check `<|tool_call>call:` before training.

---

## B.19 `compare_models.py`

Calls two OpenAI-compatible endpoints with the same messages, tools and sampling, and writes one JSONL record per prompt plus a summary:

```bash
python scripts/compare_models.py \
  --a-base-url http://127.0.0.1:8082/v1 --a-model qwen25-coder-14b-react \
  --b-base-url http://127.0.0.1:8085/v1 --b-model gemma4-12b-react \
  --prompts data/golden/react-prompts.jsonl \
  --tools examples/react/tools.json \
  --temperature 0.2 --max-tokens 4096 --seed 42 \
  --out reports/compare-react.jsonl
```

It records content, tool calls, latency, usage and errors. It does not judge; apply each patch and run the real verifier. See [31 — Gemma vs Qwen](31-gemma-vs-qwen.md).

---

## B.20 `estimate_pretraining.py`

A planning calculator, not a benchmark. It computes:

- parameters from \(V,d,L,d_{\text{ff}},H,G,d_h\) and tied/untied embeddings;
- training FLOPs \(C\approx6ND\);
- inference FLOPs per token \(2N\);
- GPU-hours, wall time and cost from peak TFLOP/s and MFU;
- KV-cache size at a given context.

Presets: `qwen3-14b`, `qwen25-coder-14b`, `gemma4-12b`, `tiny-100m`, `small-1b`.

```bash
python scripts/estimate_pretraining.py --preset gemma4-12b --tokens 10e12 \
  --gpus 1000 --peak-tflops 495 --mfu 0.4 --cost-per-gpu-hour 2.5
```

Replace default peak/MFU with measured numbers. Real runs add 20–50% for failures, checkpointing, data loading and evaluation.

---

## B.21 `train_tiny_transformer.py`

A dependency-light PyTorch reference implementation of a modern decoder:

- byte-level tokens (vocab 256), so no tokenizer training is needed;
- RMSNorm;
- RoPE;
- GQA;
- SwiGLU;
- causal cross-entropy;
- AdamW, cosine schedule, warmup, gradient clipping;
- validation and sampling.

```bash
python scripts/train_tiny_transformer.py --data data/tiny/corpus.txt \
  --out runs/tiny --dim 256 --layers 6 --heads 8 --kv-heads 2 \
  --ffn 768 --seq 256 --batch 16 --steps 2000
```

For a GPU: `--device cuda --dtype bf16`. It is a teaching implementation, not a production trainer; use FSDP/Megatron for real runs. See [35 — From zero](35-from-zero.md).

---

## B.22 `inspect_architecture.py`

Reads only `config.json` — no weights, no PyTorch — and prints:

- hidden size, layers, FFN, heads, KV heads, head dimension, vocabulary;
- parameter count for attention, MLP, embedding, LM head and total;
- active parameters for MoE;
- KV cache at a chosen sequence length;
- training/inference FLOPs for a batch.

```bash
python scripts/inspect_architecture.py --hf Qwen/Qwen3-14B --seq 32768
python scripts/inspect_architecture.py --hf Qwen/Qwen3-30B-A3B --seq 4096
python scripts/inspect_architecture.py --preset gemma4-12b --seq 32768
python scripts/inspect_architecture.py --config config.json --json
```

Presets: `qwen3-14b`, `qwen25-coder-14b`, `gemma4-12b`, `qwen3-30b-a3b`.

The KV estimate models local sliding windows and global K=V sharing when the config provides `layer_types`, `sliding_window`, `num_global_key_value_heads`, `global_head_dim` and `attention_k_eq_v`. Otherwise it is a full-attention upper bound. See [37 — Architecture anatomy](37-architecture.md) and [43 — Worked examples](43-math-examples.md).

---

## B.23 `generate_distill_data.py`

Generate sequence-level distillation data from any OpenAI-compatible teacher:

```bash
python scripts/generate_distill_data.py \
  --prompts examples/distill/prompts.jsonl \
  --system-file examples/distill/teacher-system.md \
  --model deepseek-flash \
  --out data/distill/raw/deepseek.jsonl \
  --n 2 --temperature 0.8 --max-tokens 2048
```

Use `--base-url http://127.0.0.1:8080/v1 --allow-empty-key` for a local
`mlx_lm.server`. It writes canonical `messages` rows, preserves
`reasoning_content` and `tool_calls`, supports `--resume` and `--dry-run`, and
never executes tools. See [40 — Teachers, students and data](40-distill-data.md).

---

## B.24 `distill_logits.py`

A GPU reference for logit KD with top-k teacher logits and LoRA:

```bash
python scripts/distill_logits.py \
  --teacher google/gemma-4-12B-it \
  --student google/gemma-4-E4B-it \
  --data data/distill/e4b/train.jsonl \
  --out adapters/gemma4-e4b-distill \
  --device cuda:0 --dtype bf16 --max-seq-length 1024 \
  --top-k 64 --temperature 2.0 --alpha 0.6 --lora-r 16
```

It checks that teacher and student vocabularies match, masks the prompt, computes
CE + top-k KL over the response span, supports separate teacher/student devices,
and saves the adapter plus `distill_config.json`. It is not an MLX trainer; on
32 GB use sequence KD. See [41 — Distillation training](41-distill-training.md).

---

## B.25 `check_math_examples.py`

Pure-stdlib assertions for the worked examples in [43 — Worked examples](43-math-examples.md):

```bash
python scripts/check_math_examples.py
# all 38 checks passed
```

It checks 38 values: softmax/CE, the Jacobian, RMSNorm, attention, GQA, KV cache (including the Gemma hybrid), RoPE, SwiGLU, MoE, GDN, speculative decoding, parameter counts, FLOPs, quantization, AdamW, LoRA and KD. No model or network is loaded.

---

## B.26 Shell helpers

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

## B.27 Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | validation/hard failure |
| 2 | usage error |
| 3 | unsafe conditions (Docker/host network/scope) |
| 4 | upstream API failure after retries |
