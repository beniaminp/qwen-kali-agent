# 42 — Distillation evaluation and deployment

*Last verified: 25 October 2026.*

A distillation run is not done when the loss goes down. It is done when the student is measurably better than the base on your tasks, no worse on the behaviours you refuse to lose, and cheap enough to deploy. This chapter is the protocol.

---

## 1. Why distillation evaluation is different

Ordinary fine-tuning asks: “did the adapter learn the data?” Distillation asks four questions:

1. **Did the student gain the teacher’s behaviour** on the target distribution?
2. **Did it keep its own general behaviour** outside that distribution?
3. **Did it close the teacher gap**, or only imitate the teacher’s style?
4. **Is it worth the deployment cost** compared with the base and the teacher?

A student can look excellent on the training distribution and be worse than the base everywhere else. Always evaluate **four systems** on the same prompts:

| System | What it tells you |
|---|---|
| **Teacher** | the ceiling on this distribution |
| **Base student** | what you started with |
| **SFT baseline** (hard labels, no teacher) | the value of distillation vs ordinary data |
| **Distilled student** | the result |

If you only compare distilled vs base, you cannot tell whether the gain came from distillation, more data, or a lucky prompt set.

---

## 2. The evaluation matrix

### 2.1 Task metrics

| Track | Metrics | Verifier |
|---|---|---|
| **Kali/agent** | tool validity, argument accuracy, scope compliance, recovery, unsafe-action rate, task success | tool schema + policy + lab execution |
| **Java/Quarkus** | compile rate, test pass rate, ArchUnit violations, review score, patch size | Maven + ArchUnit + judge |
| **React/Tailwind** | typecheck, build, Vitest, Playwright, axe, visual rubric | `tsc`, Vitest, Playwright, axe |
| **Reasoning** | exact match, verifier pass, consistency, length | exact/test/verifier |
| **General** | MMLU/ARC subset, instruction following, refusal quality | harness/judge |

### 2.2 Distillation metrics

| Metric | Definition | Use |
|---|---|---|
| **Agreement** | \(\frac{1}{N}\sum_i \mathbb{1}[\arg\max p_S=\arg\max p_T]\) | how often the student picks the teacher’s top token |
| **Top-k overlap** | \(\frac{1}{N}\sum_i |\operatorname{TopK}_S\cap\operatorname{TopK}_T|/k\) | ranking similarity |
| **KL** | \(\frac{1}{N}\sum_i \operatorname{KL}(p_T\|p_S)\) | distribution gap |
| **Perplexity on teacher traces** | \(\exp(-\frac{1}{T}\sum\log p_S(y_t))\) | imitation quality |
| **Pass@k** | task success in \(k\) samples | capability, not just imitation |
| **Calibration (ECE)** | \(|\text{accuracy}-\text{confidence}|\) | whether confidence is meaningful |
| **Length ratio** | student tokens / teacher tokens | verbosity drift |
| **Diversity** | distinct-\(n\), entropy of outputs | mode collapse |
| **Safety** | unsafe-action rate, refusal correctness | release gate |

Agreement and KL are **diagnostics**, not goals. A student can agree with the teacher’s top token and still fail the task; a student can disagree and be correct.

### 2.3 Pass@k

For a task with \(n\) samples and \(c\) correct:

\[
\text{pass@}k
=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}.
\]

Use \(k=1\) for deployment, \(k=5\) or \(k=10\) for capability. Report the estimator and the number of samples; pass@k is biased with small \(n\).

### 2.4 Wilson interval

For a success rate \(p=k/n\), the 95% Wilson interval is

\[
\frac{\hat p+\frac{z^2}{2n}\pm z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}
{1+\frac{z^2}{n}},
\qquad z=1.96.
\]

Use it for tool validity, scope compliance and task success. A 2-point difference on 100 tasks is noise.

---

## 3. The golden-set protocol

### 3.1 Freeze it before training

- 200–500 prompts for a first evaluation; 50–100 per track.
- Include easy, medium and hard; include failure/recovery and refusal.
- Keep a private held-out set the teacher never saw.
- Store prompts, tools, seeds and expected checks in version control.
- Never train on the golden set or its paraphrases.

### 3.2 Run the same conditions

For all four systems:

- same system prompt;
- same tool schemas;
- same temperature (0 for deterministic checks, 0.2 for realism);
- same max tokens;
- same stop sequences;
- same seed if supported;
- same number of samples.

Record the endpoint, model ID, adapter hash, quantization and server version.

### 3.3 Grade with the real verifier

- Java: apply the patch and run `./mvnw test`; run ArchUnit.
- React: write the files and run `tsc`, `npm test`, Playwright, axe.
- Kali: validate tool schema, scope, then execute in the lab only if safe.
- Reasoning: exact answer or a verifier; judge only for style.

Do not use an LLM judge as the only grader for code or security. Use it for review quality after the executable checks.

---

## 4. Running the comparison

`scripts/compare_models.py` sends the same prompts/tools to two OpenAI-compatible endpoints and records both outputs:

```bash
python scripts/compare_models.py \
  --a-base-url http://127.0.0.1:8082/v1 --a-model gemma4-12b-base \
  --b-base-url http://127.0.0.1:8083/v1 --b-model gemma4-12b-distill \
  --prompts data/golden/kali-prompts.jsonl \
  --tools examples/distill/tools.json \
  --out reports/distill-kali.jsonl
```

For tool-call evaluation:

```bash
python scripts/eval_toolcall.py \
  --base-url http://127.0.0.1:8083/v1 \
  --model gemma4-12b-distill \
  --prompts data/golden/kali-prompts.jsonl \
  --tools examples/distill/tools.json \
  --out reports/distill-kali-graded.jsonl
```

The output is JSONL, one row per prompt, with both responses, latency, usage and tool calls. Grade it with the track verifier; do not eyeball 200 rows.

### 4.1 A/B/C/D report

```jsonl
{"id":"kali-001","teacher":0.86,"base":0.54,"sft":0.61,"distilled":0.79,
 "tool_valid":true,"scope_ok":true,"unsafe":false,"latency_ms":1840}
{"id":"java-014","teacher":0.91,"base":0.48,"sft":0.58,"distilled":0.74,
 "compiled":true,"tests_pass":true,"archunit":0,"latency_ms":2210}
```

Then aggregate:

| System | Kali success | Tool valid | Scope safe | Java tests | React a11y | Latency p50 |
|---|---:|---:|---:|---:|---:|---:|
| Teacher | | | | | | |
| Base | | | | | | |
| SFT | | | | | | |
| Distilled | | | | | | |

---

## 5. Reading the results

### 5.1 The four outcomes

| Pattern | Interpretation | Action |
|---|---|---|
| Distilled > base, teacher gap small | success | ship, monitor |
| Distilled > base, teacher gap large | partial transfer | more/better data, logit KD, on-policy |
| Distilled ≈ base | no transfer | check template, mask, LR, data relevance |
| Distilled < base | harmful | stop; inspect data, LR, overfitting, safety drift |

### 5.2 If the student beats the teacher

Suspect:

- evaluation leakage (teacher saw the golden prompts);
- a judge biased toward the student’s style;
- the teacher was run with a bad prompt/decoding setting;
- the student memorized the answers;
- the task is easier than the teacher’s training distribution.

A student can beat a teacher on a narrow task, but it should be explained, not assumed.

### 5.3 General regression

Run a small general suite before and after:

- instruction following;
- refusal of clearly harmful requests;
- a few MMLU/ARC/HellaSwag items;
- code syntax in another language;
- multilingual if relevant.

A domain adapter that loses 10 points of general ability may still be worth it; a safety regression is not.

### 5.4 Safety and scope

For Kali, the release gate is not “did it solve more tasks?” It is:

- **zero** out-of-scope tool calls;
- **zero** destructive commands outside the allowlist;
- correct refusal when authorization is missing;
- no invented scan results;
- audit logs for every tool call.

A model that is 5% better and 1% unsafe is not shippable.

---

## 6. Statistical hygiene

- Report \(n\), success count and a Wilson interval, not just a percentage.
- Use the same prompts for all systems; paired comparisons are more sensitive.
- Bootstrap the difference for confidence intervals:

\[
\Delta = \frac{1}{B}\sum_{b=1}^{B}
\left(\text{score}_S^{(b)}-\text{score}_T^{(b)}\right),
\]

resampling prompts with replacement.
- Do not compare runs with different decoding settings.
- Do not tune on the test set; use a validation split.
- Keep a changelog of prompts and graders.

---

## 7. Release gates

A practical gate table:

| Gate | Threshold | Blocks release? |
|---|---|---|
| Domain task success | ≥ base + 5 points, CI excludes 0 | Yes |
| Tool validity | ≥ base, ≥ 98% | Yes |
| Scope compliance | 100% | Yes |
| Unsafe-action rate | 0 on the golden set | Yes |
| General regression | ≤ 5 points | Review |
| Refusal correctness | ≥ base | Yes |
| Latency | within deployment budget | Review |
| Memory | fits target device with 20% headroom | Yes |
| License/provenance | clean | Yes |

These are starting points. Set thresholds before the run, not after seeing the numbers.

---

## 8. Deployment matrix

| Student | 4-bit weights | Context | Target | Serving |
|---|---:|---:|---|---|
| Gemma 4 12B | ~7.65 GB | 256K | 32 GB Mac / 16 GB GPU | MLX, llama.cpp |
| Gemma 4 E4B | ~4.0 GB | 128K | laptop / edge | MLX, llama.cpp, Ollama |
| Gemma 4 E2B | ~2.6 GB | 128K | phone / edge | MLX, llama.cpp |
| Qwen3-8B | ~4.5 GB | 32K | 16 GB Mac | MLX, llama.cpp |

### 8.1 Memory budget

At inference:

\[
\text{memory}\approx\text{weights}+\text{KV cache}+\text{runtime}.
\]

For Gemma 4 12B at 32K, the KV cache is several GB even with GQA and K=V; see [37 — Architecture anatomy](37-architecture.md). A 16 GB device should use E4B/E2B or a shorter context.

### 8.2 Serve and compare

```bash
# student
mlx_lm.server --model fused/gemma4-e4b-distill-4bit \
  --host 127.0.0.1 --port 8084 --max-tokens 4096

# teacher
mlx_lm.server --model mlx-community/gemma-4-12b-it-4bit \
  --host 127.0.0.1 --port 8080 --max-tokens 4096
```

Use the same OpenAI API for both; run `compare_models.py` with the two ports.

### 8.3 Agent wiring

- Keep the same tool schemas and system prompt as training.
- Validate `tool_calls` before execution.
- Keep a scope policy between the model and the lab.
- Log prompts, tool calls, results and refusals.
- Provide a kill switch.

A distilled model is still a model; it is not authorization.

---

## 9. Report template

```markdown
# Distillation report — gemma4-12b-distill-kali-v1

## Setup
- Teacher: deepseek-flash, 2026-10-25
- Student: Gemma 4 12B IT, 4-bit QLoRA, 24/48 blocks
- Data: 5,120 traces, 7.4M tokens, 62% Kali
- Training: 300 iters, effective batch 8, seq 2048, LR 8e-5
- Hardware: M1 Pro 32 GB, 61 hours

## Results (n=240)
| System | Success | Tool valid | Scope safe | p50 latency |
|---|---:|---:|---:|---:|
| Teacher | 0.81 | 0.99 | 1.00 | 2.1s |
| Base | 0.52 | 0.93 | 0.99 | 1.8s |
| SFT | 0.60 | 0.96 | 1.00 | 1.8s |
| Distilled | 0.74 | 0.99 | 1.00 | 1.8s |

## Decision
Ship the adapter for the Kali track. Keep the base for general tasks.
Known gaps: no IPv6, limited Windows, no cloud IAM.

## Reproduction
- dataset hash: ...
- adapter hash: ...
- config: configs/gemma4-12b-distill.yaml
```

---

## 10. When to stop

Stop and do not ship when:

- the gain is within noise;
- tool validity or scope compliance regresses;
- the student only repeats the teacher’s style but fails the verifier;
- the data card is incomplete;
- the license/terms are unclear;
- you cannot reproduce the run from the config and data hash.

Distillation is a compression technique, not a substitute for a product decision.

---

## 11. Sources

- Distilling the Knowledge in a Neural Network — <https://arxiv.org/abs/1503.02531>
- Sequence-Level Knowledge Distillation — <https://arxiv.org/abs/1606.07947>
- Pass@k estimator — <https://arxiv.org/abs/2107.03374>
- Wilson interval — <https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval>
- Gemma 4 technical report — <https://arxiv.org/abs/2607.02770>
- [09 — Evaluation](09-evaluation.md)
- [37 — Architecture anatomy](37-architecture.md)
- [39 — Gemma 12B distillation](39-gemma-distillation.md)
- [40 — Teachers, students and data](40-distill-data.md)
- [41 — Distillation training](41-distill-training.md)
