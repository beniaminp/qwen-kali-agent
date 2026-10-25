# 24 — Fine-tuning a Qwen coder for React and Tailwind

*Last verified: 25 October 2026 against mlx-lm 0.32.2 and the Qwen2.5-Coder tokenizer template. The commands were not executed on the target Mac in this environment.*

This is the runbook for the front-end track. It assumes:

- macOS Apple Silicon with MLX installed ([`03-macos-setup.md`](03-macos-setup.md));
- Node 24 LTS and npm for building/evaluating the React projects;
- React/Tailwind data in `data/react/processed/{train,valid,test}.jsonl` ([`23-react-datasets.md`](23-react-datasets.md));
- 32 GB unified memory.

The default is **Qwen2.5-Coder-14B-Instruct 4-bit QLoRA**, adapting 24 of 48 blocks at 3,072 tokens. The 7B all-layer run is the smoke test and the long-context fallback.

---

## 0. The recipe in one table

| Choice | Value | Why |
|---|---|---|
| Base | `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit` | code-specialised, Apache-2.0, fits 32 GB |
| Method | 4-bit QLoRA, `lora` | base frozen; adapter small |
| Blocks | 24 of 48 | memory-safe for 3K context |
| Rank / scale | 16 / 2.0 | PEFT `alpha=32` equivalent |
| LR | `1e-4` | standard LoRA range with `scale=2` |
| Effective batch | 8 (1 × 8 accumulation) | LoRA prefers 8–16 |
| Sequence | 3072 | component + styles + tests |
| Iterations | 300 | ~4–5M tokens; adjust to your data |
| Masking | `mask_prompt: true` | train only the final assistant turn |
| Checkpoint | every 50 | short run; keep options |

Front-end tasks are longer than single-function tasks. At 12–18 tok/s, 300 iterations × 8 × ~1,800 tokens (~4.3M) is roughly **65–100 hours**. For a two-day run, use 200 iterations or a 1.5M-token dataset. Do not guess; measure with `token_stats.py`.

---

## 1. Preflight

```bash
cd ~/work/qwen-kali-agent
source .venv/bin/activate

node -v            # expect 24.x LTS
npm -v
npx playwright --version

python scripts/check_env.py
```

Checklist:

- [ ] Node 24 LTS; not Node 26 Current.
- [ ] `npm ci` passes on the reference React project.
- [ ] `npm run typecheck` passes.
- [ ] `npm run lint` passes.
- [ ] `npm run build` passes.
- [ ] `npx playwright test` passes (Chromium installed).
- [ ] MLX sees the GPU (`mx.device_info()`).
- [ ] 100+ GB free disk.
- [ ] The React data exists and is non-empty.
- [ ] No secrets in `data/react/`.
- [ ] You have a baseline evaluation of the *base* model.

Node is only for data generation and evaluation. MLX training does not need Node, but you cannot verify React data without it.

---

## 2. Prepare the data

### 2.1 Mine seeds from a permissive repository

```bash
mkdir -p data/react/{raw,interim,processed,processed-long,golden}

python scripts/mine_react_commits.py \
  --repo ~/src/shadcn-ui/ui \
  --out data/react/interim/shadcn-commits.jsonl \
  --limit 300 --only-with-tests --license MIT
```

Each row has a base commit, task text, changed files, gold patch, package manager and test command. It is a **seed**, not a finished training row. Use [25 — React synthetic data](25-react-synthetic.md) to turn it into a verified task and a new solution.

### 2.2 Normalize and split

Qwen2.5-Coder requires tool arguments as **objects**:

```bash
python scripts/prepare_dataset.py \
  data/react/interim/all.jsonl \
  --out data/react/processed \
  --arguments-format dict \
  --split 0.9 0.05 0.05
```

For the 4K run, build a separate split from rows whose rendered length is between 2,049 and 4,096 tokens. Inspect the distribution with `token_stats.py`, then filter before `prepare_dataset.py`; do not include 8K examples in a 4K run. If a row is longer than 4,096, split the trajectory or drop it rather than truncating a tool call.

### 2.3 Validate

```bash
python scripts/validate_dataset.py data/react/processed/train.jsonl \
  --tools examples/react/tools.json

python scripts/validate_react_dataset.py data/react/processed/train.jsonl \
  --tools examples/react/tools.json --require-dict-arguments --strict
```

The React validator catches:

- legacy Tailwind v3 directives and v4-removed utilities;
- `ReactDOM.render`, `useFormState`, class components, `React.FC`;
- dynamic class construction;
- unsanitized `dangerouslySetInnerHTML`;
- missing `alt`, `button type`, `rel` on `_blank`;
- `div` click handlers without roles;
- `any`, `@ts-ignore`, `console.log`, `TODO`;
- probable secrets;
- tool arguments stored as strings instead of objects.

It is a linter, not a compiler or a browser. The real filter is `tsc`, ESLint, Vitest, Vite, Playwright and axe.

### 2.4 Measure tokens

```bash
python scripts/token_stats.py data/react/processed/train.jsonl \
  --model Qwen/Qwen2.5-Coder-14B-Instruct
```

Choose `max_seq_length` at the 99th percentile, rounded to 256. If p99 is 3,400, use 3,072 and split long rows; do not default to 4,096 because a few examples are long.

### 2.5 A useful dataset size

- First adapter: **2–3M tokens**, 2,000–5,000 examples.
- Mature adapter: **5–10M tokens**, 8k+ examples.
- Do not dump 100k raw components; the model learns style from a few thousand verified tasks.

Mix from [`23-react-datasets.md`](23-react-datasets.md) §8: 20% components, 15% pages, 15% data/forms, 10% design system, 10% a11y, 10% testing, 10% bug fix, 5% migration, 5% refusal.

---

## 3. The configs

### 3.1 7B smoke / long context — `configs/qwen2.5-coder-7b-react.yaml`

```yaml
model: mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
train: true
data: data/react/processed
fine_tune_type: lora
optimizer: adamw
num_layers: -1
batch_size: 1
iters: 300
val_batches: 25
learning_rate: 1.5e-4
steps_per_report: 10
steps_per_eval: 100
save_every: 100
max_seq_length: 4096
grad_checkpoint: true
grad_accumulation_steps: 8
mask_prompt: true
adapter_path: adapters/qwen25-coder-7b-react
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

All 28 blocks fit comfortably. Use it to validate the data/template and to handle 4K multi-file examples quickly.

### 3.2 14B default — `configs/qwen2.5-coder-14b-react.yaml`

```yaml
model: mlx-community/Qwen2.5-Coder-14B-Instruct-4bit
train: true
data: data/react/processed
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
max_seq_length: 3072
grad_checkpoint: true
grad_accumulation_steps: 8
mask_prompt: true
adapter_path: adapters/qwen25-coder-14b-react
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

Peak is the risk: 21–25 GB at 3,072. If it OOMs, reduce sequence to 2,048 first; then blocks to 16.

### 3.3 14B all layers — `configs/qwen2.5-coder-14b-react-all-layers.yaml`

All 48 blocks, seq 2,048, LR `8e-5`, 250 iterations. Higher capacity, higher activation memory. Run after the 24-layer adapter is evaluated; drop to 32 blocks if it OOMs.

### 3.4 14B long context — `configs/qwen2.5-coder-14b-react-longctx.yaml`

Last 16 blocks, seq 4,096, LR `8e-5`, 250 iterations. For page + component + tests examples that exceed 3K. 4K roughly doubles activation memory versus 2K; fewer blocks compensate.

### 3.5 How many iterations?

```text
tokens_seen ≈ iters × batch_size × grad_accumulation_steps × avg_tokens_per_sample
iters ≈ epochs × dataset_tokens / (batch_size × grad_accumulation_steps × avg_tokens_per_sample)
```

Example: 2.5M-token dataset, average 1,800 tokens, effective batch 8, **1 epoch**:

```text
iters = 1 × 2,500,000 / (8 × 1,800) ≈ 174
```

So 175–200 iterations is one epoch; 300 is ~1.5–1.7 epochs. Use validation loss **and** the golden set; do not train to a fixed number.

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
- roughly 34M trainable parameters.

All-layer 14B: ~336 modules, ~69M trainable parameters.
7B all 28 blocks: ~196 modules, roughly 40M trainable parameters.

Exact numbers depend on the MLX version and embedding handling. If you see only `q_proj`/`v_proj`, stop.

---

## 5. Smoke run on 7B

Start with a 100k-token subset and 50 iterations:

```bash
head -n 1500 data/react/processed/train.jsonl > data/react/processed/train.smoke.jsonl
mkdir -p data/react/processed-smoke
cp data/react/processed/train.smoke.jsonl data/react/processed-smoke/train.jsonl
cp data/react/processed/valid.jsonl data/react/processed-smoke/valid.jsonl

mlx_lm.lora --config configs/qwen2.5-coder-7b-react.yaml \
  --data data/react/processed-smoke --iters 50 --save-every 25 \
  --adapter-path adapters/qwen25-coder-7b-react-smoke
```

Check at iteration 1–20:

- loss is finite and falling;
- tokens/sec is 25–40 for 7B at 4K (varies with memory pressure);
- peak memory is 12–16 GB;
- the rendered sample contains `<|im_start|>user`, `<tool_call>` and `<tool_response>` where expected.

Fuse and test:

```bash
mlx_lm.fuse \
  --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --adapter-path adapters/qwen25-coder-7b-react-smoke \
  --save-path fused/qwen25-coder-7b-react-smoke-4bit

mlx_lm.server --model fused/qwen25-coder-7b-react-smoke-4bit \
  --host 127.0.0.1 --port 8080 --max-tokens 4096 --temp 0.2
```

Send 20 hand-written prompts. Check:

- it emits React 19 function components, not class components or jQuery;
- it uses `@import "tailwindcss"` and v4 utilities, not `@tailwind base`;
- it uses tokens (`bg-surface`, `text-ink`) rather than raw hex;
- it includes loading/empty/error states;
- it says “I did not run the tests” when it did not;
- tool arguments are objects.

Do not proceed to 14B until the 7B output is at least **plausible** on 20 tasks.

---

## 6. The 14B run

```bash
mlx_lm.lora --config configs/qwen2.5-coder-14b-react.yaml
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
| Tokens/sec | 12–18 (14B, 24 layers, 3K) | <8 = memory/thermal |
| Peak memory | 21–25 GB | >26 GB = danger |
| It/sec | stable | collapses = swap/thermal |

Code loss is lower than chat loss because code is predictable; do not compare absolute values across datasets. Compare the adapter to the base on the same React tasks.

### 6.1 If it OOMs

In order:

1. `max_seq_length: 3072` → `2048`.
2. `num_layers: 24` → `16`.
3. Confirm `grad_checkpoint: true` and `batch_size: 1`.
4. Close Docker, Chrome, the IDE and the Kali VM.
5. Move to the 7B.

Do not start by shrinking the dataset. A completed 7B adapter beats a 14B run that dies at hour 60.

### 6.2 If loss is `nan`

- A prompt is at or over `max_seq_length`; with `--mask-prompt` the completion is empty. Check `token_stats` and drop/split.
- The LR is too high. Halve it; keep `scale: 2.0`.
- The data has an empty assistant message or a broken tool call. Run the validators again.

---

## 7. Checkpoints and resume

```bash
mlx_lm.lora --config configs/qwen2.5-coder-14b-react.yaml \
  --resume-adapter-file adapters/qwen25-coder-14b-react/0000200_adapters.safetensors \
  --iters 100 \
  --adapter-path adapters/qwen25-coder-14b-react-resume
```

Only weights are restored; optimizer state, LR schedule position and iteration count are not. Use a new adapter path and a fresh short cosine schedule.

Checkpoints are ~100–250 MB each for 14B. Delete old ones after evaluation.

---

## 8. Evaluate like a front-end engineer

### 8.1 The golden set

Build 100–200 tasks from your own product before training:

| Category | Count | What it measures |
|---|---:|---|
| Component/primitive | 30 | props, variants, states, tokens, a11y |
| Page/layout | 25 | hierarchy, responsive, composition |
| Data/forms | 20 | Query states, Actions, validation, optimistic |
| Design-system refactor | 20 | tokens, CVA, dedupe, no visual drift |
| Accessibility | 20 | keyboard, focus, labels, axe |
| Bug fix | 20 | hooks, race, controlled input, layout shift |
| Migration | 15 | React 18→19, Tailwind 3→4, Router 7→8, TS 6→7 |
| Review | 15 | real findings, no invented issues |
| Refusal/security | 15 | secrets, XSS, client-only auth |

Each task has `repo`, `base_commit`, `task`, `constraints`, `test_command`, `expect`. See [`data/examples/react-golden.example.jsonl`](https://github.com/beniaminp/qwen-kali-agent/blob/main/data/examples/react-golden.example.jsonl).

### 8.2 Grading

For each task:

1. Copy the repo to a temp directory and check out `base_commit`.
2. Prompt the model with the system prompt, repo context and task.
3. Extract the diff (or the single file).
4. `git apply --check`, then `git apply`.
5. Run `npm ci` (cached), `npm run typecheck`, `npm run lint`, `npm test -- --run`, `npm run build`.
6. Start the preview server; run Playwright flows and axe at 320/768/1440.
7. For refactors, compare before/after screenshots.
8. Record build, tests, axe, responsive, visual and requirement results.

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
        commands = [
            ["npm", "run", "typecheck"],
            ["npm", "run", "lint"],
            ["npm", "test", "--", "--run"],
            ["npm", "run", "build"],
        ]
        results = []
        for cmd in commands:
            r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=900)
            results.append({"cmd": " ".join(cmd), "code": r.returncode,
                            "stdout": r.stdout[-2000:], "stderr": r.stderr[-2000:]})
        return {"pass": all(r["code"] == 0 for r in results), "results": results}
```

`npm test -- --run` assumes a Vitest `test` script. If the repo uses a different script, read `package.json`; do not guess. Never run model-generated shell commands outside a disposable sandbox.

### 8.3 Browser checks

```bash
npx playwright test
npx playwright test --project=mobile
npx playwright test --update-snapshots   # only when the design intentionally changes
```

The Playwright config should include three projects (mobile/tablet/desktop), `reducedMotion: "reduce"`, `animations: "disabled"` for screenshots, and a fixed `baseURL`.

Axe:

```ts
import AxeBuilder from "@axe-core/playwright";

const results = await new AxeBuilder({ page })
  .withTags(["wcag2a", "wcag2aa", "wcag22aa"])
  .analyze();

expect(results.violations.filter(v => v.impact === "serious" || v.impact === "critical"))
  .toEqual([]);
```

Responsive overflow:

```ts
const overflow = await page.evaluate(
  () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
);
expect(overflow).toBeLessThanOrEqual(1);
```

Screenshot:

```ts
await expect(page).toHaveScreenshot("dashboard.png", {
  fullPage: true,
  animations: "disabled",
  maxDiffPixelRatio: 0.01,
});
```

### 8.4 Metrics

| Metric | Definition | First target | Mature target |
|---|---|---:|---:|
| Typecheck | `tsc --noEmit` passes / tasks | >0.90 | >0.98 |
| Build | `vite build` passes / tasks | >0.90 | >0.98 |
| Unit tests | Vitest green / tasks | >0.45 | >0.65 |
| E2E | Playwright flows green / tasks | >0.45 | >0.65 |
| Axe | zero serious/critical / tasks | >0.80 | >0.95 |
| Responsive | no overflow at 320/768/1440 | >0.90 | >0.98 |
| Visual | rubric ≥4/5 average | >0.60 | >0.80 |
| Requirement satisfaction | checklist items average | >0.55 | >0.75 |
| Unsafe/invented | false claims or unsafe code / tasks | 0 | 0 |

Report Wilson intervals. A 5-point difference on 100 tasks is not significant.

### 8.5 Benchmarks

| Benchmark | What it measures | Caveat |
|---|---|---|
| **Webdev-Bench** | 40 multi-round Vite + React tasks | MIT; evaluation only; 40 tasks is noisy |
| **Design2Code** | 484 screenshot→HTML pages | ODC-BY; image input; text coder cannot run directly |
| **SWE-bench Multilingual** | 43 JS/TS issue tasks | MIT; not React UI-specific |
| **Multi-SWE-bench** | 580 TS/JS issue tasks | `other` license; evaluation only |
| **SWE-bench Multimodal** | 480 JS/TS issues with images | no license tag; visual input |
| **Your golden set** | your product, your design system | the decision-maker |

Do not tune on the benchmarks. Report them once per candidate, with version/commit and contamination method.

### 8.6 Release gates

- [ ] typecheck ≥ 0.90;
- [ ] build ≥ 0.90;
- [ ] unit ≥ 0.45 (first) / 0.65 (mature);
- [ ] E2E ≥ 0.45 / 0.65;
- [ ] axe serious/critical = 0;
- [ ] no horizontal overflow at 320/768/1440;
- [ ] visual rubric ≥ 4/5;
- [ ] no `@tailwind base`, `bg-opacity-*`, dynamic classes;
- [ ] no `ReactDOM.render`, `useFormState`;
- [ ] no secrets, no unsanitized HTML, no client-only auth;
- [ ] unsafe/invented rate = 0;
- [ ] benchmark contamination reported;
- [ ] 30 human-reviewed screens and 20 failures inspected.

---

## 9. Fuse, serve and wire the agent

### 9.1 Fuse

```bash
mlx_lm.fuse \
  --model mlx-community/Qwen2.5-Coder-14B-Instruct-4bit \
  --adapter-path adapters/qwen25-coder-14b-react \
  --save-path fused/qwen25-coder-14b-react-4bit
```

### 9.2 Serve with MLX

```bash
mlx_lm.server \
  --model fused/qwen25-coder-14b-react-4bit \
  --host 127.0.0.1 --port 8080 \
  --max-tokens 4096 --temp 0.2 \
  --prefill-step-size 512
```

Qwen2.5-Coder has no thinking block, so do not pass `enable_thinking`. Its template supports tools and tool responses, but `mlx_lm.server`'s parser support can lag the template. Verify a real `tool_calls` response with curl. If it returns prose, serve the same GGUF with `llama-server --jinja`; the model format is not the problem.

### 9.3 Serve with llama.cpp

```bash
mlx_lm.fuse \
  --model mlx-community/Qwen2.5-Coder-14B-Instruct-4bit \
  --adapter-path adapters/qwen25-coder-14b-react \
  --save-path fused/qwen25-coder-14b-react-bf16 \
  --dequantize

python3 ~/work/llama.cpp/convert_hf_to_gguf.py \
  fused/qwen25-coder-14b-react-bf16 \
  --outtype f16 --outfile models/qwen25-coder-14b-react-f16.gguf

~/work/llama.cpp/build/bin/llama-quantize \
  models/qwen25-coder-14b-react-f16.gguf \
  models/qwen25-coder-14b-react-Q4_K_M.gguf Q4_K_M

llama-server -m models/qwen25-coder-14b-react-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 -c 16384 -ngl 99 -fa auto --jinja
```

(`--dequantize` produces a ~30 GB bf16 directory; delete it after conversion. See [`10-export-serving.md`](10-export-serving.md).)

### 9.4 Agent loop

Use `examples/react/tools.json` and `examples/react/system-prompt.md`:

```text
read task
  → search_code / read_file
  → plan
  → write_file or apply_patch
  → run_npm typecheck
  → run_npm lint
  → run_npm test
  → run_npm build
  → run_playwright / screenshot / axe_check
  → if fail: read the error, fix, retry (max 2)
  → git_diff
  → final answer with evidence
```

The model may call `run_npm` and `run_playwright`, but the harness enforces the allowlist, timeout and working directory. Never give it a raw unrestricted shell or the host filesystem.

### 9.5 OpenCode

`examples/react/opencode.json` points OpenCode at the local server and denies webfetch. For generation, run it inside a disposable clone:

```bash
opencode run -m local/qwen25-coder-14b-react --format json \
  "Build a responsive settings page with a sidebar, form validation and an empty state. Use the existing design tokens. Run typecheck, lint, test and build. Return a diff."
```

Do not enable `--auto` globally; use an agent whose bash permission allows only `npm run`, `npx playwright` and `git diff`, rooted at the temporary clone.

---

## 10. Experiment matrix

| # | Run | Purpose | Cost |
|---|---|---|---|
| 1 | 7B all, 50 iters, 100k tokens | validate data/template | 2–4 h |
| 2 | 7B all, 300 iters, 4K | baseline | 35–55 h |
| 3 | 14B, 24/48, 3K | **recommended** | 65–100 h |
| 4 | 14B, all/48, 2K | capacity test | 45–65 h |
| 5 | 14B, 24/48, LR 5e-5 vs 1e-4 | LR sensitivity | 2 × 60 h |
| 6 | 14B, 16/48, 4K | long-context check | 55–85 h |

Stop when the golden-set gain is within noise. Do not run the whole matrix; run #2, #3, then one variable at a time.

---

## 11. Checklist

Before:

- [ ] Node 24 + npm verified;
- [ ] Playwright Chromium installed;
- [ ] `data/react/processed` validated;
- [ ] `arguments` are dicts for Qwen2.5;
- [ ] token p99 fits `max_seq_length`;
- [ ] `check_lora.py` shows MLP projections;
- [ ] 100+ GB free;
- [ ] Docker/VM/Chrome off for training;
- [ ] baseline saved.

After:

- [ ] best checkpoint chosen by golden set, not train loss;
- [ ] typecheck/build/unit/E2E/axe measured;
- [ ] visual rubric and requirement satisfaction measured;
- [ ] no horizontal overflow at 320/768/1440;
- [ ] unsafe/invented rate = 0;
- [ ] fused model loads;
- [ ] tool calls parse;
- [ ] GGUF path tested if needed;
- [ ] model card + dataset manifest written.

---

## 12. Sources

- Qwen2.5-Coder-14B-Instruct — <https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct>
- MLX-LM LoRA — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md>
- Qwen2.5-Coder template — <https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct/blob/main/tokenizer_config.json>
- React 19.3 — <https://react.dev/blog/2026/09/09/react-19-3>
- Tailwind 4.3 — <https://tailwindcss.com/blog/tailwindcss-v4-3>
- Vite 8 — <https://vite.dev/blog/announcing-vite8>
- TypeScript 7 — <https://devblogs.microsoft.com/typescript/announcing-typescript-7-0>
- React Router 8 — <https://remix.run/blog/react-router-v8>
- Playwright — <https://playwright.dev/docs/test-snapshots>
- axe-core/playwright — <https://github.com/dequelabs/axe-core-npm/tree/develop/packages/playwright>
- Webdev-Bench — <https://huggingface.co/datasets/Roy-DOU/Webdev-Bench>
- Design2Code — <https://huggingface.co/datasets/SALT-NLP/Design2Code>
- [17 — Fine-tuning update](17-finetune-2026-update.md)
- [25 — React synthetic data](25-react-synthetic.md)
