# 20 — Fine-tuning a Qwen coder for Java and Quarkus

*Last verified: 25 October 2026 against mlx-lm 0.32.2 and the Qwen2.5-Coder tokenizer template. The training commands were not executed on the target Mac in this environment.*

This is the runbook for the Java/Quarkus track. It assumes:

- macOS Apple Silicon with MLX installed ([`03-macos-setup.md`](03-macos-setup.md));
- a JDK 25 and Maven 3.9+ for building the training/eval projects;
- Java data in `data/java/processed/{train,valid,test}.jsonl` ([`19-java-datasets.md`](19-java-datasets.md));
- 32 GB unified memory.

The default is **Qwen2.5-Coder-14B-Instruct 4-bit QLoRA**, adapting 24 of its 48 blocks. The 7B is the smoke test; the all-layer 14B is an experiment, not the first run.

---

## 0. The recipe in one table

| Choice | Value | Why |
|---|---|---|
| Base | `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit` | code-specialised, Apache-2.0, fits 32 GB |
| Method | 4-bit QLoRA, `lora` | base frozen; adapter small |
| Blocks | 24 of 48 | memory-safe starting point |
| Rank / scale | 16 / 2.0 | PEFT `alpha=32` equivalent |
| LR | `1e-4` | standard LoRA range with `scale=2` |
| Effective batch | 8 (1 × 8 accumulation) | LoRA prefers 8–16 |
| Sequence | 2048 | measure p99 first |
| Iterations | 300 | ~1.5–2 epochs over 2M tokens |
| Masking | `mask_prompt: true` | train only the final assistant turn |
| Checkpoint | every 50 | short run; keep options |

Expected wall clock: **35–55 hours** on an M1 Pro at ~20–25 tok/s for 3–4M tokens. The 7B smoke run is 15–25 hours; do a 100k-token subset first.

---

## 1. Preflight

```bash
cd ~/work/qwen-kali-agent
source .venv/bin/activate

java -version          # expect 25.x
./mvnw -version        # expect Maven 3.9+
python scripts/check_env.py
```

Checklist:

- [ ] `java -version` reports 25.
- [ ] `./mvnw -version` reports Maven 3.9+.
- [ ] `./mvnw -q -DskipITs test` passes on the reference project.
- [ ] MLX sees the GPU (`mx.device_info()`).
- [ ] 100+ GB free disk.
- [ ] Docker is available if `@QuarkusTest` uses Dev Services.
- [ ] The Java data exists and is non-empty.
- [ ] No secrets in `data/java/`.
- [ ] You have a baseline evaluation of the *base* model.

The JDK is only for data generation and evaluation. MLX training itself does not need a JDK, but you cannot verify Java data without one.

---

## 2. Prepare the data

### 2.1 Mine seeds from a permissive repository

```bash
mkdir -p data/java/{raw,interim,processed,golden}

python scripts/mine_java_commits.py \
  --repo ~/src/quarkus-quickstarts \
  --out data/java/interim/quickstart-commits.jsonl \
  --limit 300 --only-with-tests --license Apache-2.0
```

Each row has a base commit, task text, changed files, gold patch and a test command. It is a **seed**, not a finished training row. Use [21 — Java synthetic data](21-java-synthetic.md) to turn it into a verified task and a new solution.

### 2.2 Normalize and split

Qwen2.5-Coder requires tool arguments as **objects**:

```bash
python scripts/prepare_dataset.py \
  data/java/interim/all.jsonl \
  --out data/java/processed \
  --arguments-format dict \
  --split 0.9 0.05 0.05
```

Why: its chat template does `tool_call.arguments | tojson`. A JSON string becomes an escaped string; a dict becomes a JSON object. Qwen3's template expects the string form. Do not use one dataset for both without re-rendering.

### 2.3 Validate

```bash
python scripts/validate_dataset.py data/java/processed/train.jsonl \
  --tools examples/java/tools.json

python scripts/validate_java_dataset.py data/java/processed/train.jsonl \
  --tools examples/java/tools.json --require-dict-arguments --strict
```

The Java validator catches:

- legacy `javax.*` imports;
- Spring idioms;
- legacy RESTEasy coordinates;
- wrong Quarkus/Java version tags;
- `System.out`/`printStackTrace`;
- `TODO`/`FIXME`;
- probable secrets;
- tool arguments stored as strings instead of objects.

It is a linter, not a compiler. The real filter is `./mvnw test`.

### 2.4 Measure tokens

```bash
python scripts/token_stats.py data/java/processed/train.jsonl \
  --model Qwen/Qwen2.5-Coder-14B-Instruct
```

Choose `max_seq_length` at the 99th percentile, rounded to 256. Java repository tasks are longer than single-function tasks; if p99 is 3,500, use 3,072 and split the long rows, not 4,096 by default.

### 2.5 A useful dataset size

- First adapter: **1.5–3M tokens**, 2,000–5,000 examples.
- Mature adapter: **5–10M tokens**, 10k+ examples.
- Do not dump 100k raw files; the model learns style from a few thousand verified tasks.

Mix from [`19-java-datasets.md`](19-java-datasets.md) §8: 25% Quarkus features, 20% bug fixes, 15% SOLID refactors, 15% tests, 10% review, 5% migration, 5% refusal, 5% general replay.

---

## 3. The configs

### 3.1 7B smoke — `configs/qwen2.5-coder-7b-java.yaml`

```yaml
model: mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
train: true
data: data/java/processed
fine_tune_type: lora
optimizer: adamw
num_layers: -1
batch_size: 2
iters: 300
val_batches: 25
learning_rate: 1.5e-4
steps_per_report: 10
steps_per_eval: 100
save_every: 100
max_seq_length: 2048
grad_checkpoint: true
grad_accumulation_steps: 4
mask_prompt: true
adapter_path: adapters/qwen25-coder-7b-java
seed: 42
lora_parameters:
  rank: 16
  dropout: 0.05
  scale: 2.0
lr_schedule:
  name: cosine_decay
  warmup: 30
  arguments: [1.5e-4, 300, 1.0e-6]
optimizer_config:
  adamw:
    weight_decay: 0.01
```

All 28 blocks fit comfortably. Use it to validate the data/template, not to judge final quality.

### 3.2 14B default — `configs/qwen2.5-coder-14b-java.yaml`

```yaml
model: mlx-community/Qwen2.5-Coder-14B-Instruct-4bit
train: true
data: data/java/processed
fine_tune_type: lora
optimizer: adamw
num_layers: 24
batch_size: 1
iters: 300
val_batches: 25
learning_rate: 1.0e-4
steps_per_report: 10
steps_per_eval: 100
save_every: 50
max_seq_length: 2048
grad_checkpoint: true
grad_accumulation_steps: 8
mask_prompt: true
adapter_path: adapters/qwen25-coder-14b-java
seed: 42
lora_parameters:
  rank: 16
  dropout: 0.05
  scale: 2.0
lr_schedule:
  name: cosine_decay
  warmup: 30
  arguments: [1.0e-4, 300, 1.0e-6]
optimizer_config:
  adamw:
    weight_decay: 0.01
```

24 of 48 blocks is the memory-safe default for a 48-layer model. It covers the last half, including the layers closest to the output.

### 3.3 14B all layers — `configs/qwen2.5-coder-14b-java-all-layers.yaml`

Same as 3.2 but `num_layers: -1`, LR `8e-5`, 250 iterations. Peak memory may exceed 24 GB. Run it only after the 24-layer adapter is evaluated; drop to 32 blocks if it OOMs.

### 3.4 14B long context — `configs/qwen2.5-coder-14b-java-longctx.yaml`

`num_layers: 24`, `max_seq_length: 4096`, LR `8e-5`, 250 iterations. 4K roughly doubles activation memory versus 2K. Only use it if `token_stats` shows a real need.

---

## 4. Verify what the adapter actually covers

Before spending hours:

```bash
python scripts/check_lora.py \
  --model mlx-community/Qwen2.5-Coder-14B-Instruct-4bit \
  --num-layers 24 --rank 16 --scale 2
```

Expected:

- 24 adapted blocks of 48;
- 7 projections per block (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`);
- ~168 LoRA modules;
- roughly 34M trainable parameters for 24 blocks (about 69M for all 48).

The exact numbers depend on the MLX version and the model's embedding handling. If you see only `q_proj`/`v_proj`, stop.

For a Qwen2.5-Coder-14B all-layer run, expect ~336 modules and ~69M trainable parameters.

---

## 5. Smoke run on 7B

Start with a 100k-token subset and 50 iterations:

```bash
# make a subset (example: first 2,000 rows)
head -n 2000 data/java/processed/train.jsonl > data/java/processed/train.smoke.jsonl
mkdir -p data/java/processed-smoke
cp data/java/processed/train.smoke.jsonl data/java/processed-smoke/train.jsonl
cp data/java/processed/valid.jsonl data/java/processed-smoke/valid.jsonl

mlx_lm.lora --config configs/qwen2.5-coder-7b-java.yaml \
  --data data/java/processed-smoke --iters 50 --save-every 25 \
  --adapter-path adapters/qwen25-coder-7b-java-smoke
```

Check at iteration 1–20:

- loss is finite and falling;
- tokens/sec is 35–45 for 7B;
- peak memory is 10–14 GB;
- the rendered sample contains `<|im_start|>user`, `<tool_call>` and `<tool_response>` where expected.

Then fuse and run a smoke test:

```bash
mlx_lm.fuse \
  --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --adapter-path adapters/qwen25-coder-7b-java-smoke \
  --save-path fused/qwen25-coder-7b-java-smoke-4bit

mlx_lm.server --model fused/qwen25-coder-7b-java-smoke-4bit \
  --host 127.0.0.1 --port 8080 --max-tokens 2048 --temp 0.2
```

Then send the prompts from `examples/java/` and check that:

- the model emits a Java/Quarkus answer, not Python or Spring;
- `javax.` does not appear;
- a tool call uses an object argument shape (Qwen2.5);
- it says “I did not run the tests” when it did not.

Do not proceed to 14B until the 7B adapter is at least **plausible** on 20 hand-written tasks.

---

## 6. The 14B run

```bash
mlx_lm.lora --config configs/qwen2.5-coder-14b-java.yaml
```

Expected first lines:

```text
Loading pretrained model
Loading datasets
Training
Trainable parameters: 0.2xx% (34.xxxM/14.7B)
Iter 1: Val loss ..., Val took ...s
Iter 10: Train loss ..., Learning Rate ..., It/sec ..., Tokens/sec ..., Trained Tokens ..., Peak mem ... GB
```

Healthy signals:

| Metric | Healthy | Warning |
|---|---|---|
| Train loss | falls to 0.3–0.7 | stuck >1.0 after 200 iters |
| Val loss | tracks train | rises for 2+ evals |
| Tokens/sec | 20–25 (14B, 24 layers) | <12 = memory/thermal |
| Peak memory | 18–23 GB | >26 GB = danger |
| It/sec | stable | collapses = swap/thermal |

Code loss is naturally lower than chat loss because code is predictable; do not compare absolute values across datasets. Compare the adapter to the base on the same Java tasks.

### 6.1 If it OOMs

In order:

1. `num_layers: 24` → `16`.
2. `max_seq_length: 2048` → `1536`.
3. Confirm `grad_checkpoint: true` and `batch_size: 1`.
4. Close Docker, the Kali VM, Chrome and the IDE.
5. Move to the 7B.

Do not start by shrinking the dataset. A completed 7B adapter beats a 14B run that dies at hour 40.

### 6.2 If loss is `nan`

- A prompt is at or over `max_seq_length`; with `--mask-prompt` the completion is empty. Check `token_stats` and drop/split.
- The LR is too high. Halve it; keep `scale: 2.0`.
- The data has an empty assistant message or a broken tool call. Run the validators again.

---

## 7. Checkpoints and resume

```bash
mlx_lm.lora --config configs/qwen2.5-coder-14b-java.yaml \
  --resume-adapter-file adapters/qwen25-coder-14b-java/0000200_adapters.safetensors \
  --iters 100 \
  --adapter-path adapters/qwen25-coder-14b-java-resume
```

Only weights are restored; optimizer state, LR schedule position and iteration count are not. Use a new adapter path and a fresh short cosine schedule.

Checkpoints are ~100–250 MB each for 14B. Delete old ones after evaluation.

---

## 8. Evaluate like a Java engineer

### 8.1 The golden set

Build 100–200 tasks from your own repositories before training. Categories:

| Category | Count | What it measures |
|---|---:|---|
| Quarkus REST endpoint | 25 | correct annotations, validation, status codes |
| Panache/repository | 20 | transaction boundary, mapping, query |
| SOLID/hexagonal refactor | 25 | package boundaries, no behavior change |
| Test generation | 20 | tests compile, pass, actually assert |
| Bug fix | 25 | multi-file reasoning, recovery |
| Code review | 20 | finding quality, no invented issues |
| Migration | 15 | 3.33→3.40, 21→25 |
| Refusal/uncertainty | 15 | no unsafe or invented claims |

Each task has `repo`, `base_commit`, `task`, `constraints`, `test_command` and `expect`. See [`data/examples/java-golden.example.jsonl`](https://github.com/beniaminp/qwen-kali-agent/blob/main/data/examples/java-golden.example.jsonl).

### 8.2 Grading

For each task:

1. Copy the repo to a temp directory and check out `base_commit`.
2. Prompt the model with the system prompt, repo context and task.
3. Extract the diff from the response.
4. `git apply --check`, then `git apply`.
5. Run `test_command` in the sandbox with a timeout.
6. Run `arch_check` if the task changes boundaries.
7. Record compile, test, ArchUnit and review results.

A minimal harness:

```python
import json, shutil, subprocess, tempfile
from pathlib import Path

def grade(task, model, repo_dir):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(repo_dir, work)
        subprocess.run(["git", "checkout", task["base_commit"]], cwd=work, check=True)
        prompt = task["task"] + "\n\nConstraints:\n" + "\n".join(task.get("constraints", []))
        answer = model.complete(system=SYSTEM, user=prompt, tools=TOOLS)
        diff = extract_diff(answer)
        (work / "model.patch").write_text(diff)
        check = subprocess.run(["git", "apply", "--check", "model.patch"], cwd=work,
                               capture_output=True, text=True)
        if check.returncode != 0:
            return {"pass": False, "reason": "patch_does_not_apply", "stderr": check.stderr}
        subprocess.run(["git", "apply", "model.patch"], cwd=work, check=True)
        test = subprocess.run(task["test_command"], shell=True, cwd=work,
                              capture_output=True, text=True, timeout=900)
        return {"pass": test.returncode == 0, "reason": "tests",
                "stdout": test.stdout[-4000:], "stderr": test.stderr[-4000:]}
```

`shell=True` is safe only because `test_command` comes from your trusted golden set, inside a disposable sandbox. Never run model-generated shell commands this way.

### 8.3 Metrics

| Metric | Definition | First target | Mature target |
|---|---|---:|---:|
| Compile rate | generated files compile / tasks | >0.90 | >0.98 |
| Patch apply rate | diffs that apply / tasks | >0.90 | >0.98 |
| Test pass rate | tasks with green tests / tasks | >0.45 | >0.65 |
| ArchUnit pass | refactors passing rules / refactors | >0.80 | >0.95 |
| Review precision | valid findings / all findings | >0.75 | >0.90 |
| Unsafe/invented rate | false claims or unsafe changes / tasks | 0 | 0 |
| p95 latency | seconds per task | <180 | <120 |

Report Wilson intervals. A 5-point difference on 100 tasks is not significant.

### 8.4 Benchmarks

- **MultiPL-E Java** — function-level generation; public; contamination caveat.
- **HumanEval-X Java** — 164 tasks in the Java subset; public.
- **Multi-SWE-bench Java** — real issue resolution; evaluation only.
- **SWE-bench Multilingual Java** — real issue resolution; evaluation only.
- **Your Quarkus golden set** — the decision-maker.

Do not tune on the benchmarks. Report them once per candidate, with the version/commit and contamination method.

### 8.5 Release gates

- [ ] compile rate ≥ 0.90;
- [ ] patch apply ≥ 0.90;
- [ ] test pass ≥ 0.45 (first) / 0.65 (mature);
- [ ] ArchUnit pass ≥ 0.80 on refactors;
- [ ] no `javax.*`, Spring or legacy RESTEasy coordinates;
- [ ] no secrets/PII;
- [ ] unsafe/invented rate = 0;
- [ ] benchmark contamination reported;
- [ ] 30 human-reviewed trajectories and 20 failures inspected.

---

## 9. Fuse, serve and wire the agent

### 9.1 Fuse

```bash
mlx_lm.fuse \
  --model mlx-community/Qwen2.5-Coder-14B-Instruct-4bit \
  --adapter-path adapters/qwen25-coder-14b-java \
  --save-path fused/qwen25-coder-14b-java-4bit
```

### 9.2 Serve with MLX

```bash
mlx_lm.server \
  --model fused/qwen25-coder-14b-java-4bit \
  --host 127.0.0.1 --port 8080 \
  --max-tokens 4096 --temp 0.2 \
  --prefill-step-size 512
```

Qwen2.5-Coder has no thinking block, so do not pass `enable_thinking`. Its template supports tools and tool responses, but `mlx_lm.server`'s parser support can lag the template. Verify a real `tool_calls` response with curl. If it returns prose, serve the same GGUF with `llama-server --jinja` (below); the model format is not the problem.

### 9.3 Serve with llama.cpp

```bash
# GGUF conversion needs an unquantized HF-format directory.
mlx_lm.fuse \
  --model mlx-community/Qwen2.5-Coder-14B-Instruct-4bit \
  --adapter-path adapters/qwen25-coder-14b-java \
  --save-path fused/qwen25-coder-14b-java-bf16 \
  --dequantize

python3 ~/work/llama.cpp/convert_hf_to_gguf.py \
  fused/qwen25-coder-14b-java-bf16 \
  --outtype f16 --outfile models/qwen25-coder-14b-java-f16.gguf

~/work/llama.cpp/build/bin/llama-quantize \
  models/qwen25-coder-14b-java-f16.gguf \
  models/qwen25-coder-14b-java-Q4_K_M.gguf Q4_K_M

llama-server -m models/qwen25-coder-14b-java-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 -c 16384 -ngl 99 -fa auto --jinja
```

(`--dequantize` produces a ~30 GB bf16 directory; delete it after conversion. See [`10-export-serving.md`](10-export-serving.md) for the full path.)

### 9.4 Agent loop

Use `examples/java/tools.json` and `examples/java/system-prompt.md`. The loop is:

```text
read task
  → search_code / read_file
  → plan
  → apply_patch or write_file
  → run_maven test
  → if fail: read error, diagnose, patch again (max 2)
  → arch_check for boundary changes
  → git_diff
  → final answer with test evidence
```

The model should call `run_maven` itself, but the harness enforces the allowlist, timeout and working directory.

### 9.5 OpenCode

`examples/java/opencode.json` points OpenCode at the local server and enables edits. For generation, run it inside a disposable clone:

```bash
opencode run -m local/qwen25-coder-14b-java --format json \
  "Add a validated POST /api/orders endpoint; run ./mvnw -q -DskipITs test; return a diff."
```

Do not give it host network, Docker socket or production credentials. See [21 — Java synthetic data](21-java-synthetic.md) for the full sandbox.

---

## 10. Experiment matrix

| # | Run | Purpose | Cost |
|---|---|---|---|
| 1 | 7B all layers, 50 iters, 100k tokens | validate data/template | 2–4 h |
| 2 | 7B all layers, 300 iters | baseline | 15–25 h |
| 3 | 14B, 24 layers | **recommended** | 35–55 h |
| 4 | 14B, all layers | capacity test | 45–65 h |
| 5 | 14B, 24 layers, LR 5e-5 vs 1e-4 | LR sensitivity | 2 × 40 h |
| 6 | 14B, 24 layers, +2M synthetic | data scaling | 50–70 h |

Stop when the golden-set gain is within noise. Do not run the whole matrix; run #2, #3, then one variable at a time.

---

## 11. Checklist

Before:

- [ ] Java 25 + Maven 3.9+ verified;
- [ ] `data/java/processed` validated;
- [ ] `arguments` are dicts for Qwen2.5;
- [ ] token p99 fits `max_seq_length`;
- [ ] `check_lora.py` shows MLP projections;
- [ ] 100+ GB free;
- [ ] VM/Docker off for training;
- [ ] baseline saved.

After:

- [ ] best checkpoint chosen by golden set, not train loss;
- [ ] compile/test/ArchUnit measured;
- [ ] unsafe/invented rate = 0;
- [ ] fused model loads;
- [ ] tool calls parse;
- [ ] GGUF path tested if needed;
- [ ] model card + dataset manifest written.

---

## 12. Sources

- Qwen2.5-Coder-14B-Instruct — <https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct>
- MLX-LM LoRA — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md>
- MLX-LM `lora.py` — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/lora.py>
- Qwen2.5-Coder tokenizer template — <https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct/blob/main/tokenizer_config.json>
- Quarkus testing — <https://quarkus.io/guides/getting-started-testing>
- Quarkus REST — <https://quarkus.io/guides/rest>
- Hibernate ORM with Panache — <https://quarkus.io/guides/hibernate-orm-panache>
- ArchUnit — <https://www.archunit.org/>
- MultiPL-E — <https://github.com/nuprl/MultiPL-E>
- Multi-SWE-bench — <https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench>
- [17 — Fine-tuning update](17-finetune-2026-update.md)
- [21 — Java synthetic data](21-java-synthetic.md)
