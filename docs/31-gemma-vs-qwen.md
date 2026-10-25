# 31 — Running Gemma and Qwen side by side

*Last verified: 25 October 2026.*

The point of the Gemma track is not to replace Qwen. It is to **keep both** and compare them on the same tasks, with the same tools and the same verifier.

This chapter defines the separation, the serving layout, the comparison harness and the decision rule.

---

## 1. Keep the stacks separate

| Path | Qwen | Gemma |
|---|---|---|
| Docs | `docs/01…26` | `docs/27…31` |
| Configs | `configs/qwen3-*`, `configs/qwen2.5-coder-*` | `configs/gemma4-12b-*` |
| Data | `data/processed`, `data/java`, `data/react`, `data/unified` | `data/gemma4/{kali,java,react,unified}` |
| Adapters | `adapters/qwen*` | `adapters/gemma4-*` |
| Fused | `fused/qwen*` | `fused/gemma4-*` |
| Examples | `examples/{tools,java,react}` | `examples/gemma4` |
| Ports | 8080–8082 | 8083–8086 |

Rules:

1. Do not overwrite a Qwen config, adapter or data split to “try Gemma”.
2. Do not mix Qwen and Gemma rows in one file. Their templates are incompatible.
3. Do not compare a fine-tuned Gemma to a base Qwen. Compare base-to-base and adapter-to-adapter.
4. Stop one server before starting another on 32 GB.

The Qwen files stay exactly as they are. The Gemma track adds new files.

---

## 2. Serving layout

```text
Qwen3-14B        → 8080  Kali/agent
Qwen2.5-Coder    → 8081  Java
Qwen2.5-Coder    → 8082  React
Gemma 4 12B      → 8083  Kali
Gemma 4 12B      → 8084  Java
Gemma 4 12B      → 8085  React
Gemma 4 12B      → 8086  Unified
```

Run only one or two at a time:

```bash
# Qwen Kali
mlx_lm.server --model fused/qwen3-14b-kali-4bit --port 8080

# Gemma Kali
mlx_lm.server --model fused/gemma4-12b-kali-4bit --port 8083

# compare
python scripts/compare_models.py \
  --a-base-url http://127.0.0.1:8080/v1 --a-model qwen3-14b-kali \
  --b-base-url http://127.0.0.1:8083/v1 --b-model gemma4-12b-kali \
  --prompts data/golden/kali-prompts.jsonl \
  --tools examples/tools/kali-tools.json \
  --out reports/compare-kali.jsonl
```

Do not run three 12B/14B servers plus a browser on 32 GB. One base at a time is the practical limit.

---

## 3. The comparison protocol

### 3.1 Same inputs

For each task, both models get:

- the same system prompt (mode + constraints);
- the same user task;
- the same tool schemas, in the same order;
- the same temperature, top-p, max tokens and seed;
- the same context budget;
- the same repository commit and environment.

If one model gets a better prompt, the comparison is meaningless.

### 3.2 Same verifier

Do not judge by prose. Apply the patch and run the track verifier:

| Track | Verifier |
|---|---|
| Kali | scope + tool-call validity + evidence + unsafe-action rate |
| Java | `./mvnw -q -DskipITs test` + ArchUnit |
| React | `tsc`, Vitest, Vite build, Playwright, axe |
| Review | rubric + human spot check |

A model that writes more confidently is not better. A model that compiles and passes is better.

### 3.3 Same split

Use the **same golden tasks** for both. Do not tune one model on the test set. Keep a private holdout and a public benchmark section.

### 3.4 Report per track

Never report a single average across Kali, Java and React. Report:

- tool-call validity;
- task success;
- compile/build/test pass;
- axe/visual/requirement;
- unsafe/invented rate;
- latency and peak memory.

---

## 4. `scripts/compare_models.py`

The collector calls both OpenAI-compatible endpoints and writes one JSONL record per prompt:

```json
{
  "id": "r-001",
  "track": "react",
  "messages": [...],
  "a": {"ok": true, "latency_ms": 4200, "content": "...", "tool_calls": [...], "usage": {...}},
  "b": {"ok": true, "latency_ms": 3900, "content": "...", "tool_calls": [...], "usage": {...}}
}
```

It also writes `*.summary.json` with error counts, tool-call counts and average latency.

```bash
python scripts/compare_models.py \
  --a-base-url http://127.0.0.1:8082/v1 --a-model qwen25-coder-14b-react \
  --b-base-url http://127.0.0.1:8085/v1 --b-model gemma4-12b-react \
  --prompts data/golden/react-prompts.jsonl \
  --tools examples/react/tools.json \
  --system-file examples/react/system-prompt.md \
  --temperature 0.2 --max-tokens 4096 --seed 42 \
  --out reports/compare-react.jsonl
```

Then grade each side:

1. Extract the patch or file.
2. `git apply --check`.
3. Run `npm run typecheck`, lint, test, build, Playwright, axe.
4. Record pass/fail and the first failure.
5. For refactors, compare screenshots.
6. For Kali, run the policy/golden checks.

The script does **not** judge. The verifier does.

### 4.1 Prompt file shape

```json
{"id":"r-001","track":"react","messages":[{"role":"system","content":"MODE: REACT\n..."},{"role":"user","content":"Build /orders ..."}]}
{"id":"k-001","track":"kali","messages":[{"role":"user","content":"Check 10.10.10.5:443 ..."}]}
{"id":"j-001","track":"java","messages":[{"role":"user","content":"Add POST /orders ..."}]}
```

If a row has no `messages`, the script uses `prompt`/`task` plus `--system-file`.

### 4.2 Tool schemas

Pass only the active track's schema:

- `examples/tools/kali-tools.json`
- `examples/java/tools.json`
- `examples/react/tools.json`

A union of 25 tools will make both models worse and the comparison noisy.

---

## 5. Metrics and analysis

### 5.1 Primary

| Metric | Definition | Decision use |
|---|---|---|
| Tool-call validity | parseable call / attempts | reject a model below 0.95 for agents |
| Task success | verifier pass / tasks | primary |
| Compile/build | build pass / tasks | primary for code |
| Tests | test pass / tasks | primary for code |
| Axe | zero serious/critical / pages | gate for UI |
| Unsafe rate | unsafe actions / tasks | gate (must be 0) |
| Invented evidence | false claims / tasks | gate (must be 0) |

### 5.2 Secondary

- latency p50/p95;
- tokens/sec and total tokens;
- peak memory;
- diff size and files touched;
- human preference (pairwise, blinded);
- visual rubric (1–5).

### 5.3 Statistics

Report Wilson intervals, not bare percentages. A 5-point difference on 100 tasks is not significant. Use at least 100 tasks per track for a release decision, 30 for a smoke comparison.

### 5.4 What not to compare

- Training loss across different tokenizers/datasets.
- Raw parameter count.
- Vendor benchmarks run on different harnesses.
- A fine-tuned model against a base model.
- One model with tools and one without.
- One model with a 32K prompt and one with 2K.

---

## 6. The decision rule

### 6.1 Per track

Keep the challenger only if:

1. it passes the hard gates (unsafe = 0, invented = 0);
2. it is within 3–5 points of the incumbent on task success;
3. it is better on at least one **material** dimension (tool reliability, compile rate, visual quality, latency, memory, vision);
4. the operational cost is acceptable.

Otherwise, keep the incumbent and record the result.

### 6.2 Default recommendation

| Track | Default | Challenger condition |
|---|---|---|
| Kali | **Qwen3-14B** | Gemma matches tool validity and unsafe rate |
| Java | **Qwen2.5-Coder-14B** | Gemma beats compile/test by >5 points |
| React | **Gemma 4 12B** | Qwen matches visual/axe and is faster |
| Unified | Qwen3-14B (safer) | Gemma matches safety and wins code/vision |

The defaults are not permanent. The report decides.

---

## 7. A minimal report

```md
# Gemma 4 12B vs Qwen — React track

Date: 2026-10-25
Tasks: 120 (private golden set)
Base commits: fixed per task
Temperature: 0.2, seed 42, max 4096
Tools: examples/react/tools.json
Verifier: tsc + Vitest + Vite build + Playwright + axe

| Metric | Qwen2.5-Coder-14B | Gemma 4 12B |
|---|---:|---:|
| Typecheck | 0.96 | 0.94 |
| Build | 0.95 | 0.93 |
| Unit tests | 0.62 | 0.58 |
| Playwright | 0.57 | 0.55 |
| Axe zero | 0.91 | 0.93 |
| Visual rubric ≥4 | 0.74 | 0.81 |
| p50 latency | 4.1 s | 3.6 s |
| Peak memory | 21 GB | 18 GB |

Decision: keep Gemma for visual/React, keep Qwen for code-only React; route by
whether the task includes a design reference. Both pass the axe and unsafe gates.
```

Use the real numbers from your run; do not copy this table.

---

## 8. Keeping both healthy

- Pin the MLX version per stack; record it in the report.
- Keep separate adapters and separate fused models.
- Keep a `reports/` directory per date and track.
- Do not overwrite a previous adapter when retraining; use a new path.
- Re-run the base model on the same prompts when you change MLX or quantization.
- Archive the prompt set, tool schema and commit for every comparison.
- If a model regresses after a server update, test the parser before blaming the adapter.

---

## 9. What “both” costs

| Item | Disk | Time |
|---|---:|---|
| Qwen3-14B 4-bit | ~8.3 GB | — |
| Qwen2.5-Coder-14B 4-bit | ~8.7 GB | — |
| Gemma 4 12B 4-bit | ~6.6–7.6 GB | — |
| Adapters (3 Qwen + 3 Gemma) | ~1–2 GB | — |
| Fused copies (optional) | 3 × ~8 GB + 3 × ~7 GB | — |
| One 2–4M-token adapter | — | 40–80 h |
| One comparison (100 tasks) | — | 4–12 h incl. verification |

You do not need all fused copies. Fuse the winner, keep the adapter.

---

## 10. Sources

- MLX-LM server — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/server.py>
- OpenAI Python client — <https://github.com/openai/openai-python>
- Wilson interval — <https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval>
- [09 — Evaluation](09-evaluation.md)
- [27 — Gemma 4 12B](27-gemma4-12b.md)
- [29 — Gemma fine-tuning](29-gemma4-finetuning.md)
