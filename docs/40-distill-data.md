# 40 — Teachers, students and distillation data

*Last verified: 25 October 2026.*

This chapter defines the data contract for Gemma distillation: what a prompt is, what a teacher response is, how to render it, how to filter it, and how much of it you need. It applies to all three directions in [39 — Gemma 12B distillation](39-gemma-distillation.md).

---

## 1. The data contract

Every row is canonical JSONL, exactly like the fine-tuning chapters:

```json
{
  "id": "kali-scope-01-s0",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "tools": [],
  "meta": {
    "teacher": "deepseek-flash",
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 2048,
    "prompt_tokens": 412,
    "completion_tokens": 638,
    "category": "kali",
    "source": "synthetic",
    "created": "2026-10-25T18:00:00Z"
  }
}
```

Rules:

- one JSON object per line;
- UTF-8, no BOM;
- `messages` alternates roles; the last message is `assistant`;
- `tools` is optional but must be a JSON array of OpenAI-style schemas;
- tool `arguments` in **Gemma 4** data must be JSON **objects**, not strings;
- `reasoning_content` is optional and only when the student template supports it;
- `meta` is ignored by training but kept for lineage;
- no secrets, customer data, public IPs or destructive commands.

The format is defined in [06 — Data format](06-data-format.md). Gemma-specific validation is in [28 — Gemma datasets](28-gemma4-datasets.md).

---

## 2. Teacher selection

A teacher is not automatically better because it is larger. Choose by:

1. **task quality** on your domain;
2. **tokenizer compatibility** (logit KD) or API availability (sequence KD);
3. **license/terms** for using outputs as training data;
4. **cost** per 1M tokens;
5. **latency** and rate limits;
6. **safety behaviour** you are willing to reproduce.

| Teacher | Params | Same Gemma tokenizer? | Local on 32 GB? | Best use |
|---|---:|---:|---|---|
| **Gemma 4 31B** | 31.3B | **Yes** | 4-bit inference only (~17 GB) | Best same-family logit teacher on a GPU |
| **Gemma 4 12B** | 12.0B | **Yes** | Yes (~7.65 GB 4-bit) | Self-distillation, cheap data, offline |
| **Qwen3.8-27B** | 27B class | No | 4-bit inference only | Strong general/code sequence teacher |
| **Qwen3-Coder-30B-A3B** | 30.5B / 3.3B active | No | 4-bit inference only | Agentic code and tool use |
| **DeepSeek V4.1 Flash** | API | No | No | Cheap bulk sequence data |
| **DeepSeek V4-Pro** | API | No | No | Hard reasoning/code examples |
| **Claude / GPT** | API | No | No | High-quality gold traces; check terms |

### 2.1 Which teacher for which direction

- **Teacher → Gemma 12B:** Gemma 4 31B if you can afford the GPU and want logit KD; Qwen3.8-27B or DeepSeek for sequence KD.
- **Gemma 12B → E4B/E2B:** Gemma 4 12B is the natural teacher; 31B is better but harder to fit.
- **Gemma 12B as teacher:** use it when privacy/offline/cost dominates; accept lower peak quality.

### 2.2 Do not mix incompatible logits

A teacher with a different tokenizer cannot provide logits for a Gemma student. The vocabularies are different sizes and token meanings. Use sequence KD instead. See [39 — Gemma 12B distillation](39-gemma-distillation.md) §4.

---

## 3. Student selection

| Student | Effective params | 4-bit weights | Context | Best for |
|---|---:|---:|---:|---|
| **Gemma 4 12B** | 12.0B | ~7.65 GB | 256K | Best local quality; 24/48-block QLoRA |
| **Gemma 4 E4B** | 4.5B | ~4.0 GB | 128K | Laptop/edge deployment; all 42 blocks |
| **Gemma 4 E2B** | 2.3B | ~2.6 GB | 128K | Phone/edge; all 35 blocks |
| Qwen3-8B | 8B | ~4.5 GB | 32K | Cross-family sequence KD |
| Qwen2.5-Coder-7B | 7.6B | ~4.3 GB | 128K | Code sequence KD |

The student decides the tokenizer and template. Never train a student on the teacher’s template.

---

## 4. Prompt distribution

The prompt set is the **syllabus**. If it contains only easy questions, the student learns only easy behaviour.

A balanced first run for a mixed agent/code model:

| Bucket | Share | Examples |
|---|---:|---|
| Domain hard tasks | 35% | multi-step Kali scope decisions, Quarkus refactors, React a11y fixes |
| Multi-turn / recovery | 20% | tool error, retry, partial evidence, failed test |
| Tool use | 15% | valid calls, invalid arguments, refusal to call out of scope |
| Reasoning | 10% | diagnosis, trade-offs, “what would you measure?” |
| Refusal / scope | 10% | unauthorized target, ambiguous authorization, unsafe request |
| General retention | 10% | short general code, explanation, formatting |

Do not let one track dominate unless you are intentionally building a single-track adapter. A “Kali” adapter can be 70% Kali; a unified adapter needs the mix above.

### 4.1 Prompt sources

- your private golden set prompts (responses removed);
- real issues/tickets, redacted;
- commit messages and failing tests (Java/React);
- lab tasks and post-mortems (Kali);
- public instruction datasets, license-checked;
- synthetic prompt generators (teacher or a small model), then filtered.

Keep a **held-out prompt set** that the teacher never sees during data generation. Evaluating on prompts used to generate training data measures memorization, not distillation.

### 4.2 How many prompts

| Goal | Prompts | Samples/prompt | Traces | Approx tokens |
|---|---:|---:|---:|---:|
| Smoke test | 50–100 | 1 | 50–100 | 50k–200k |
| First adapter | 1k–3k | 1–2 | 2k–6k | 2M–6M |
| Serious domain run | 5k–15k | 1–3 | 10k–30k | 10M–30M |
| Broad unified run | 15k–50k | 1–2 | 30k–80k | 30M–80M |

A 300-iteration MLX run at effective batch 8 and sequence 2,048 sees roughly

\[
300\times8\times2048\approx4.9\text{M tokens}.
\]

So a 5M-token dataset is about one epoch; 10M is about two. More data only helps if quality holds.

---

## 5. Teacher generation

### 5.1 The generation contract

For each prompt, the teacher returns:

- `content` (required);
- `reasoning_content` if the API exposes it;
- `tool_calls` if tools are enabled;
- `finish_reason` (`stop`, `length`, `tool_calls`);
- usage tokens.

The script `scripts/generate_distill_data.py` preserves these in canonical JSONL. It supports:

```bash
python scripts/generate_distill_data.py \
  --prompts data/distill/prompts.jsonl \
  --system-file examples/distill/teacher-system.md \
  --model deepseek-flash \
  --base-url https://api.deepseek.com \
  --out data/distill/raw/deepseek.jsonl \
  --n 2 --temperature 0.8 --top-p 0.95 --max-tokens 2048 \
  --concurrency 4 --seed 42
```

Local Gemma teacher:

```bash
mlx_lm.server --model mlx-community/gemma-4-12b-it-4bit \
  --host 127.0.0.1 --port 8080 --max-tokens 4096

python scripts/generate_distill_data.py \
  --prompts data/distill/prompts.jsonl \
  --system-file examples/distill/teacher-system.md \
  --model gemma-4-12b \
  --base-url http://127.0.0.1:8080/v1 \
  --allow-empty-key \
  --out data/distill/raw/gemma-12b.jsonl \
  --n 2 --temperature 0.8 --max-tokens 4096
```

Use `--dry-run` first: it prints the prompt count, request count and the first rendered prompt without making an API call.

### 5.2 Decoding settings

| Setting | Sequence KD | Logit KD data | Why |
|---|---|---|---|
| Temperature | 0.7–1.0 | 0.7–1.0 | diversity without incoherence |
| Top-p | 0.95 | 0.95 | keep the tail but avoid nonsense |
| Top-k | default | default | not critical for data |
| Max tokens | 2K–4K | 2K–4K | long enough for code/traces |
| Samples/prompt | 1–3 | 1–2 | more samples = more diversity, more cost |
| Seed | fixed per shard | fixed | reproducibility |

Do **not** use temperature 0 for all data: you get one deterministic path and the student overfits that style. A small amount of diversity helps, but the teacher must stay correct.

### 5.3 Best-of-\(n\) and rejection sampling

For reasoning/code, generate \(n\) samples and keep the best according to a verifier:

1. sample \(n=4\)–\(8\) at temperature 0.8;
2. execute code tests, typecheck, Playwright, or validate tool schema;
3. keep the shortest passing trace, or the one with the best rubric;
4. record the rejected count in `meta`.

This is **rejection-sampling distillation**. It costs more teacher calls but raises target quality. It is often better than a larger dataset of unverified samples.

### 5.4 What to do with tool calls

- Keep the teacher’s tool call only if the tool name is in the row’s `tools` array.
- Parse `arguments`; reject malformed JSON.
- For Gemma, convert arguments to a **mapping** before writing the row.
- Reject calls to tools that are not in scope.
- For agentic data, pair the call with the tool response and the recovery turn; a single call is not a trajectory.

`scripts/validate_gemma4_dataset.py --strict` checks these rules.

---

## 6. Sequence data vs logit data

### 6.1 Sequence data

One row = one prompt + one teacher response. The student is trained with masked cross-entropy on the assistant span.

Advantages:

- works with any teacher API;
- works across tokenizers;
- supported by MLX-LM today;
- easy to inspect and deduplicate.

Disadvantages:

- no soft labels;
- no teacher uncertainty;
- exposure bias: the student trains on teacher prefixes, not its own;
- a single sampled response may be one of many valid paths.

### 6.2 Logit data

Two ways to use teacher logits:

**Online:** run teacher and student together; compute KL on the fly. Needs both models in memory.

**Offline:** cache the teacher’s top-\(k\) logits/probabilities per token, then train the student later. The storage is

\[
\text{bytes}\approx N_{\text{tokens}}\times k\times(4+4)
\]

for int32 indices + fp32 probabilities. For 5M tokens and \(k=64\):

\[
5\text{M}\times64\times8\approx2.56\text{ GB}.
\]

For \(k=256\), ~10 GB. Full 262k logits per token would be ~5.2 TB for 5M tokens; never cache full logits.

If you cache logits, record:

- teacher checkpoint hash;
- tokenizer hash;
- top-\(k\);
- temperature used to produce probabilities (or store raw top-k logits);
- prompt/template hash;
- the exact input token IDs, not just text.

Without the input IDs, the logits are not reusable.

### 6.3 Which to choose

| Constraint | Choice |
|---|---|
| 32 GB Mac, API teacher | Sequence KD |
| 32 GB Mac, Gemma 12B teacher → E4B/E2B | Sequence KD first; logit KD on GPU |
| 80 GB GPU, same tokenizer | Logit KD, top-k 64 |
| 2×80 GB, 31B → 12B | Logit KD |
| Teacher has no logits | Sequence KD |
| Teacher tokenizer differs | Sequence KD |

---

## 7. Rendering and tokenization

The student template is the contract. For Gemma 4:

```python
prompt_text = tokenizer.apply_chat_template(
    messages[:-1], tools=tools, tokenize=False, add_generation_prompt=True
)
full_text = tokenizer.apply_chat_template(
    messages, tools=tools, tokenize=False, add_generation_prompt=False
)
```

Then tokenize **without adding special tokens again**:

```python
prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
```

The response span is `full_ids[len(prompt_ids):]` if the prefix matches. If it does not, find the longest common prefix and log the mismatch; do not silently train on a wrong mask.

### 7.1 Gemma thinking

Gemma 4 supports a thinking channel:

```text
<|channel>thought
...reasoning...
<channel|>final answer
```

If the teacher produces `reasoning_content`, decide whether the student should learn it:

- **learn it** if the student supports thinking and you want reasoning traces;
- **drop it** if the student is used in no-think mode or the trace is noisy;
- **never** put teacher thinking into a student that was trained without it.

For E2B/E4B vs 12B/31B, the templates differ on empty thought blocks. Validate the rendered prompt with the actual student tokenizer.

### 7.2 Tool schemas

Tool definitions are part of the prompt. A 10-tool schema can be 1–3k tokens. Check:

- every `tool_call.function.name` exists in `tools`;
- `arguments` is an object (Gemma) or string (Qwen) as required;
- no tool response for a call that was never made;
- the final assistant turn matches the tool state.

---

## 8. Filtering and verification

A teacher trace is a **proposal**. Keep it only if it passes the cheapest verifier that can catch its failure.

| Data type | Verifier | Reject if |
|---|---|---|
| Java | `./mvnw test`, compile, ArchUnit | does not compile/test; uses Spring/javax; wrong Quarkus version |
| React | `tsc`, Vitest, Playwright, axe | type error; test fail; missing alt/label; contrast fail |
| Kali | tool schema + scope policy + lab run | out-of-scope target; invalid args; destructive command; invented output |
| Reasoning | exact answer / judge / self-consistency | wrong answer; unsupported claim; no final answer |
| General | format check + judge | refusal for a benign request; truncation; repetition |

Additional filters:

- exact dedup by SHA-256 of normalized messages;
- MinHash near-dedup (threshold 0.8–0.9);
- language ID;
- length percentiles (drop the bottom/top 2%);
- secret/PII regex;
- unsafe-content filter;
- tool-call validity;
- code syntax;
- test execution (if available);
- refusal consistency (do not train both “yes” and “no” on the same scope);
- style: no “As an AI…”, no meta-commentary, no fake citations.

### 8.1 Verification budget

Execution is the most expensive filter, so tier it:

1. **cheap:** schema, JSON, length, dedup, regex;
2. **medium:** compile/typecheck, unit tests;
3. **expensive:** full Playwright, integration tests, lab execution;
4. **human/judge:** only for a sample and for disagreements.

Run the expensive tier on a stratified sample first; if the pass rate is low, fix the teacher prompt before scaling.

### 8.2 Rejected data is still useful

Keep a `rejected.jsonl` with the reason. It tells you:

- which prompt categories the teacher fails;
- whether the teacher prompt is ambiguous;
- whether the verifier is too strict;
- what the student should **not** learn.

Do not train on rejected rows, but do not delete the evidence.

---

## 9. Mixing with human and existing SFT data

Pure teacher data can make the student sound like the teacher and forget general behaviour. A safe mix:

| Source | Share | Why |
|---|---:|---|
| Teacher domain traces | 50–70% | the new behaviour |
| Human/curated SFT | 15–25% | correctness and style anchor |
| General instruction data | 10–20% | retention |
| Refusal/scope | 5–10% | safety |

For a narrow adapter, 80–90% domain is fine. For a unified adapter, keep at least 30% non-domain data.

Do not mix a teacher trace and a human answer to the **same prompt** with contradictory facts. Deduplicate by prompt and keep the better answer, or keep both only if they are genuinely different valid paths.

---

## 10. Data card

Write a data card before training:

```yaml
name: gemma4-12b-distill-kali-v1
teacher: deepseek-flash
teacher_date: 2026-10-25
student: gemma-4-12b-it
prompts: 3200
traces: 5100
tokens: 7.4M
categories: {kali: 0.62, refusal: 0.12, reasoning: 0.10, general: 0.16}
filters: [schema, scope, dedup, secret, length]
verification: {schema: 100%, lab_exec: 18%, human_sample: 2%}
licenses: {prompts: internal, outputs: provider terms}
known_gaps: [no Windows targets, no cloud IAM, limited IPv6]
```

The card is part of the model release. It tells future you what the adapter actually learned.

---

## 11. Commands

```bash
# 1. generate teacher traces
python scripts/generate_distill_data.py \
  --prompts data/distill/prompts.jsonl \
  --system-file examples/distill/teacher-system.md \
  --model deepseek-flash \
  --out data/distill/raw/deepseek.jsonl \
  --n 2 --temperature 0.8 --max-tokens 2048

# 2. validate as Gemma 4 data
python scripts/validate_gemma4_dataset.py \
  data/distill/raw/deepseek.jsonl \
  --tools examples/distill/tools.json --strict

# 3. token statistics
python scripts/token_stats.py \
  data/distill/raw/deepseek.jsonl \
  --model google/gemma-4-12B-it

# 4. normalize, convert tool arguments, split
python scripts/prepare_dataset.py \
  data/distill/raw/deepseek.jsonl \
  --out data/distill/12b/processed \
  --arguments-format dict \
  --split 0.9 0.05 0.05

# 5. near-dedup and contamination checks (separate step)
python scripts/split_dataset.py \
  data/distill/12b/interim/all.jsonl \
  --out data/distill/12b/processed --minhash

# 6. inspect a rendered row before training
python scripts/smoke_test.py --data data/distill/12b/processed/train.jsonl \
  --model google/gemma-4-12B-it
```

The exact split flags depend on your script version; `prepare_dataset.py --help` is authoritative.

---

## 12. Sources

- Distilling the Knowledge in a Neural Network — <https://arxiv.org/abs/1503.02531>
- Sequence-Level Knowledge Distillation — <https://arxiv.org/abs/1606.07947>
- Gemma 4 12B model card — <https://huggingface.co/google/gemma-4-12B-it>
- Gemma 4 E4B model card — <https://huggingface.co/google/gemma-4-E4B-it>
- Gemma 4 E2B model card — <https://huggingface.co/google/gemma-4-E2B-it>
- Gemma 4 31B model card — <https://huggingface.co/google/gemma-4-31B-it>
- DeepSeek API docs — <https://api-docs.deepseek.com/>
- [06 — Data format](06-data-format.md)
- [28 — Gemma datasets](28-gemma4-datasets.md)
- [39 — Gemma 12B distillation](39-gemma-distillation.md)
- [41 — Distillation training](41-distill-training.md)
