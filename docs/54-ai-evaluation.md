# 54 — Evaluation: evals are your test suite

*Last verified: 28 October 2026.*

You cannot ship what you cannot measure, and with a probabilistic model you cannot measure it by looking. An **eval** is an automated check on a model's output; an **eval set** is a frozen collection of inputs with graders and thresholds — a regression suite whose assertions are not `==` but "pass rate ≥ 0.9". This chapter builds that suite: deterministic checks, reference metrics, execution tests, LLM judges, human agreement, the statistics to tell a real improvement from noise, a runnable harness, and the CI gate that blocks a bad release. It is the engineering counterpart to the model-specific protocols in [09 — Evaluation](09-evaluation.md) (tool-call graders, golden sets, release gates) and [42 — Distillation evaluation](42-distill-eval.md); the Wilson-interval arithmetic agrees with [43 — Worked examples](43-math-examples.md). For evaluating retrieval, see [51 — RAG](51-ai-rag.md) §§12–13; for agents, [52 — Agents](52-ai-agents.md).

---

## 1. Why evals are the core skill

A software test asserts an exact result: `assert add(2, 2) == 4`. A model test cannot, because the model is non-deterministic and natural language has many right answers. So the unit of measurement shifts from "this output is correct" to "the pass *rate* over a set is high enough." Everything else follows from that shift:

- An **eval set** is a regression suite. You run it on every candidate (new prompt, new model, new retriever) and compare to the last known-good run.
- A **grader** (scorer) is an assertion that returns pass/fail — but written to tolerate paraphrase, formatting and the other legitimate variation of language.
- A **threshold** is the assertion's bar: "exact-match ≥ 0.85", "zero unsafe outputs". Setting it *before* you see the numbers is what keeps the suite honest.

The engineer's instinct transfers directly: you would not refactor without tests, and you should not change a prompt without an eval. The discipline that is new is *statistics* — because a pass rate is an estimate with error bars, "88% beats 85%" might be pure noise (§8).

Treat evals as the deliverable, not the afterthought. The teams that ship reliable LLM products are the ones with the best eval sets, not the cleverest prompts.

---

## 2. Deterministic checks

The cheapest, most reliable evals need no reference answer and no model — they check structure. Run them first and on every output, because a response that is not even parseable fails before any question of quality.

- **Format / schema**: is it valid JSON? does it match the expected schema (keys, types)?
- **Regex**: does it contain the required pattern (an order ID, a citation marker) and none of the forbidden ones (an email address in a redacted answer)?
- **Length / bounds**: within the token budget? not empty? not a 2,000-word essay where one line was asked for?

```python
import json
import re
from collections import Counter

def exact_match(pred, gold):
    return pred == gold

def normalize(s):          # lowercase, drop punctuation, squash whitespace
    s = re.sub(r"[^\w\s]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()

def normalized_em(pred, gold):
    return normalize(pred) == normalize(gold)

def token_f1(pred, gold):
    p, g = pred.split(), gold.split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)

def valid_json(pred):
    try:
        json.loads(pred)
        return True
    except json.JSONDecodeError:
        return False

print("exact_match:", exact_match("Yes", "yes"))
print("normalized_em:", normalized_em("  Yes ", "yes"), normalized_em("Paris.", "paris"))
print("token_f1:", round(token_f1("the total is 42",
                                   "the invoice total is 42 dollars"), 4))
print("valid_json:", valid_json('{"label": "billing"}'), valid_json("billing"))
```

```text
exact_match: False
normalized_em: True True
token_f1: 0.8
valid_json: True False
```

Notice `exact_match("Yes", "yes")` is `False` while `normalized_em` is `True` (and "Paris." matches "paris"): raw exact match is brutal about case, whitespace and punctuation, which is why reference metrics (§3) normalise first. Deterministic checks catch the failures that make everything else moot — a tool call that is not valid JSON cannot be executed no matter how good its intent.

---

## 3. Reference-based metrics

When each input has a known gold answer, you compare the output to it. The art is tolerating correct variation.

### 3.1 Exact and normalized exact match

**Exact match** (EM) is 1 if the output equals the gold string. Useful for short, closed answers (a label, a number, a yes/no). **Normalized EM** lowercases, trims whitespace and strips punctuation first, so "Paris." and "paris" count as equal. Normalise everything that is not semantically meaningful — but no more, or you will mark wrong answers right.

### 3.2 Token-level F1

For short free-text answers, token F1 gives partial credit by treating the answer as a bag of tokens. Precision is the fraction of predicted tokens that appear in the gold answer; recall is the fraction of gold tokens the prediction covers; F1 is their harmonic mean:

\[
F_1=\frac{2\,P\,R}{P+R}
\]

!!! example "On paper"

    Prediction "the total is 42" (4 tokens), gold "the invoice total is 42 dollars" (6 tokens). The shared tokens are the, total, is, 42 — 4 of them.

    \[
    P=4/4=1.0
    \]

    \[
    R=4/6\approx 0.667
    \]

    \[
    F_1=\frac{2\cdot 1.0\cdot 0.667}{1.0+0.667}=0.8
    \]

    Precision is perfect (every word the model said is in the gold answer) but recall is 0.667 (it dropped "invoice" and "dollars"), so F1 lands at 0.8 — partial credit for a mostly-right answer that exact match would have scored 0.

### 3.3 BLEU and ROUGE, and why they fail for open answers

BLEU (precision-oriented, from machine translation) and ROUGE (recall-oriented, from summarisation) score n-gram overlap with one or more references. ROUGE-1 overlap on unigrams:

\[
\text{ROUGE-1}_{\text{recall}}=\frac{\text{overlapping unigrams}}{\text{reference unigrams}}
\]

!!! example "On paper"

    Reference summary "the refund was issued today" (5 words), candidate "refund issued today" (3 words). The overlap is refund, issued, today — 3 unigrams.

    \[
    \text{recall}=3/5=0.6
    \]

    \[
    \text{precision}=3/3=1.0
    \]

    \[
    F_1=\frac{2\cdot 0.6\cdot 1.0}{0.6+1.0}=0.75
    \]

    The candidate is a fine shorter paraphrase, yet ROUGE recall is only 0.6 because it dropped two reference words. That is the whole problem: a *correct* answer phrased differently scores low.

These metrics are useful when there is a tight reference (translation, extractive summary) and worthless when the space of good answers is open (an explanation, a chat reply). They reward surface overlap, not meaning: against the reference "the refund was approved", the candidate "the refund was not approved" contains all 4 reference words, so its ROUGE-1 recall is a perfect 4/4 = 1.0 (precision 4/5 = 0.8) although it says the opposite; a perfect paraphrase with no shared words scores zero. For open answers, use an LLM judge (§5) or human eval (§6), not BLEU/ROUGE.

---

## 4. Execution-based evaluation

For code and anything with a verifiable outcome, do not score the text — **run it**. Generate a function, run the unit tests, pass if they pass. This is the gold standard because it checks behaviour, not resemblance. The Java and React tracks do exactly this (compile, `tsc`, Vitest); see [42](42-distill-eval.md) §3.3.

Because generation is stochastic, you sample \(n\) attempts and report **pass@k** — the probability that at least one of \(k\) sampled attempts passes. With \(c\) of \(n\) samples correct, the unbiased estimator is

\[
\text{pass@}k=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}
\]

!!! example "On paper"

    A coding task sampled \(n=6\) times, \(c=2\) passing the tests.

    \[
    \text{pass@}1=1-\frac{\binom{4}{1}}{\binom{6}{1}}=1-\frac46
    \]

    \[
    \text{pass@}1\approx 0.333
    \]

    \[
    \text{pass@}3=1-\frac{\binom{4}{3}}{\binom{6}{3}}=1-\frac{4}{20}=0.8
    \]

    pass@1 (0.333) is just the per-sample success rate \(c/n\) — the number for deployment, where the user sees one answer. pass@3 (0.8) is the chance one of three tries works — a capability number, and relevant only if your product actually retries. Report which \(k\) you mean; see [42](42-distill-eval.md) §2.3 for the same estimator on an agent task.

---

## 5. Model-graded evaluation (LLM-as-judge)

When there is no reference and no executable check — "is this summary faithful?", "which of two answers is more helpful?" — you can ask a strong model to grade. It scales human judgement to thousands of examples, but it is a *measurement instrument with known biases*, so calibrate it.

### 5.1 Rubric design

A judge is only as good as its rubric. Give it: the task, the input, the output(s), any reference, and an explicit scale with described anchors ("1 = does not answer; 3 = answers but omits a caveat; 5 = complete and correct"). Ask for a short reason *before* the score (so the score is justified, not reflexive), and output a parseable field. Vague rubrics ("rate 1–10") produce vague, unstable scores.

### 5.2 Pointwise, pairwise, reference-guided

- **Pointwise**: score one output on the rubric. Simple, but absolute scores drift and cluster.
- **Pairwise**: show two outputs, ask which is better. More reliable — relative judgements are easier and more stable than absolute ones — and the natural fit for comparing two prompts or models.
- **Reference-guided**: give the judge a gold answer to compare against. Raises agreement with humans when a reference exists.

### 5.3 Known biases and mitigations

LLM judges have systematic, measured biases:

- **Position bias**: a tendency to prefer the first (or a fixed) option in a pairwise test. *Mitigation*: run each pair both ways (A then B, B then A) and only count a win if it holds in both orders; ties otherwise.
- **Verbosity bias**: longer answers are scored higher regardless of quality. *Mitigation*: instruct the rubric to ignore length; cap or normalise length; penalise padding explicitly.
- **Self-preference**: a model tends to prefer text written by itself or its own family. *Mitigation*: use a different model family as judge than the one under test; anonymise which system produced which output.

Always validate the judge against human labels before trusting it, and never let it overturn a deterministic failure — a judge that "likes" invalid JSON is wrong by construction (see [09](09-evaluation.md) §4.5).

---

## 6. Human evaluation

Humans remain the ground truth that calibrates every automated metric. You use them sparingly — on a sample, to validate judges and to grade what nothing else can.

### 6.1 Guidelines

Write a labelling guide as precise as a rubric: the exact question, the scale, worked examples of each level, and tie-break rules. Ambiguity in the guide shows up as disagreement between annotators. Collect at least two labels per item on a sample so you can *measure* that disagreement.

### 6.2 Inter-annotator agreement: Cohen's kappa

Raw agreement overstates reliability because two annotators agree sometimes by chance. **Cohen's kappa** corrects for chance, where \(p_o\) is observed agreement and \(p_e\) is the agreement expected if both labelled at random with their own base rates:

\[
\kappa=\frac{p_o-p_e}{1-p_e}
\]

!!! example "On paper"

    Two annotators label 100 answers pass/fail. Both say pass on 45, both fail on 40, and they disagree on 15 (A-pass/B-fail 10, A-fail/B-pass 5). Observed agreement:

    \[
    p_o=(45+40)/100=0.85
    \]

    A says pass 55/100, B says pass 50/100. Chance agreement:

    \[
    p_e=0.55\cdot 0.50+0.45\cdot 0.50=0.5
    \]

    \[
    \kappa=\frac{0.85-0.5}{1-0.5}=0.70
    \]

    Raw agreement looks strong at 0.85, but half of it was expected by chance, so kappa is 0.70 — "substantial" by the usual rule of thumb (0.6–0.8), good enough to trust the labels and to calibrate a judge against them. A kappa near 0 would mean your guide is too vague to label consistently; fix the guide, not the annotators.

---

## 7. Building a golden set

The eval set is the asset; the model is replaceable. Build it deliberately.

- **Sources**: real production traffic (sampled and labelled), real support tickets, real documents — not invented toy inputs. The distribution must match what you ship against.
- **Size**: enough for the precision you need (§8). A few hundred is a usable start; detecting small differences needs thousands.
- **Stratification**: label each case with tags (topic, difficulty, language, length) so you can read per-tag pass rates, not just one average that hides a failing slice.
- **Edge and adversarial cases**: include the empty input, the multi-part question, the out-of-scope request, the prompt-injection attempt (see [56](56-ai-safety.md)). Average accuracy on easy cases is comfortable and useless.
- **Held-out set**: keep a slice you never look at during development, so you can tell whether you improved the system or just overfit the eval set — the same lesson as a test set in ML.
- **Versioning**: give the set a version (`golden-2026-10-28`); never silently edit it after seeing results. A changed eval set invalidates comparisons to older runs.

---

## 8. Statistics: is the difference real?

A pass rate is an estimate from a sample, so it has error bars. Skipping this step is how teams ship "improvements" that are noise.

### 8.1 Confidence interval for a pass rate (Wilson)

For \(k\) passes out of \(n\), the 95% Wilson score interval (better than the textbook normal interval at small \(n\) and extreme rates) is

\[
\frac{\hat p+\frac{z^2}{2n}\pm z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}},
\quad z=1.96
\]

!!! example "On paper"

    40 of 100 cases pass, so \(\hat p=0.4\), \(z^2=3.8416\), denominator \(1+z^2/n=1.0384\).

    \[
    \text{centre}=\frac{0.4+0.0192}{1.0384}=0.4037
    \]

    \[
    \text{half}=1.96\cdot 0.04996/1.0384=0.0943
    \]

    \[
    0.4037\pm 0.0943=(0.309,\;0.498)
    \]

    "40 of 100" is consistent with a true rate from 31% to 50% — so a 2-point difference between two models on 100 cases is noise. This matches [43 — Worked examples](43-math-examples.md#wilson-interval) to the digit; report the interval, not just the point.

### 8.2 How many examples to detect a 5-point difference

To detect a difference \(\Delta\) between two independent pass rates at significance \(\alpha\) and power \(1-\beta\), the per-arm sample size is

\[
n=\frac{(z_{\alpha/2}+z_{\beta})^2\,[p_1(1-p_1)+p_2(1-p_2)]}{\Delta^2}
\]

!!! example "On paper"

    Detect a 5-point lift from 50% to 55% (\(\Delta=0.05\)), at \(\alpha=0.05\) two-sided (\(z=1.96\)) and 80% power (\(z=0.84\)). The variance term is \(0.25+0.2475=0.4975\):

    \[
    (1.96+0.84)^2=7.84
    \]

    \[
    n=7.84\cdot 0.4975/0.0025
    \]

    \[
    n\approx 1561
    \]

    You need about 1,560 examples *per model* to see a 5-point difference. The rule of thumb \(n\approx 16\,p(1-p)/\Delta^2 = 16\cdot 0.25/0.0025 = 1600\) lands in the same place. Small differences need large sets — which is why a 200-case suite can only resolve large changes.

### 8.3 Paired comparison: McNemar's test

Running both systems on the *same* items is far more powerful than two independent samples, because each item's difficulty cancels. For paired pass/fail data, only the **discordant** pairs matter — the items where the two systems disagree. With \(b\) = "A right, B wrong" and \(c\) = "A wrong, B right", McNemar's statistic is

\[
\chi^2=\frac{(b-c)^2}{b+c}
\]

!!! example "On paper"

    Two prompts on 100 items: both right 60, both wrong 20, A-only right 15, B-only right 5.

    \[
    \chi^2=\frac{(15-5)^2}{15+5}=\frac{100}{20}=5.0
    \]

    With the continuity correction \((|b-c|-1)^2/(b+c)=81/20=4.05\). Both exceed the \(\chi^2_{1,0.05}=3.841\) cutoff, so prompt A is significantly better.

### 8.4 Why unpaired comparisons waste data

The same 100 items, scored unpaired, give A = 75/100 and B = 65/100. The two-proportion test:

\[
z=\frac{p_A-p_B}{\sqrt{\bar p(1-\bar p)(2/n)}}
\]

!!! example "On paper"

    Pooled rate \(\bar p=0.70\), so the standard error is \(\sqrt{0.21\cdot 0.02}=0.0648\):

    \[
    z=0.10/0.0648=1.54
    \]

    1.54 is below 1.96: **not significant.** The very same results are significant when paired (§8.3) and not significant when unpaired, because the unpaired test's error bars include item-to-item difficulty that the paired test cancels. Always run both systems on the same items and compare pairwise.

### 8.5 Multiple comparisons

If you test 20 prompt variants against a baseline at \(\alpha=0.05\), you expect about one false positive by chance alone (\(20\times 0.05=1\)) — with a two-sided test, half of those look like wins and half like losses. Guard against it: pre-register the comparison you care about, or correct the threshold (Bonferroni: divide \(\alpha\) by the number of tests), or validate the apparent winner on the held-out set before believing it.

---

## 9. A runnable eval harness

The pieces above assemble into a small harness: a dataset of cases, a function under test, scorers, an aggregate report with a confidence interval, a per-tag breakdown, and JSONL results you can diff between runs. Here the "model" is a keyword stub so the block runs anywhere; §9.1 swaps in a served model unchanged.

```python
import json
import math

# 1. the eval set: inputs, gold labels, tags (a frozen regression suite)
CASES = [
    {"id": "t1", "text": "I was charged twice this month", "gold": "billing", "tags": ["easy"]},
    {"id": "t2", "text": "the app crashes when I open settings", "gold": "bug", "tags": ["easy"]},
    {"id": "t3", "text": "how do I export my data?", "gold": "howto", "tags": ["easy"]},
    {"id": "t4", "text": "refund the duplicate charge please", "gold": "billing", "tags": ["hard"]},
    {"id": "t5", "text": "login button does nothing on mobile", "gold": "bug", "tags": ["hard"]},
    {"id": "t6", "text": "where is the invoice for March?", "gold": "billing", "tags": ["hard"]},
]

# 2. the system under test (a keyword stub; swap for a real model later)
def model(text):
    t = text.lower()
    if any(w in t for w in ("charge", "refund", "billing")):
        return "billing"
    if any(w in t for w in ("crash", "bug", "error")):
        return "bug"
    return "howto"

# 3. a scorer turns one case into pass/fail (the assertion)
def score(case):
    return model(case["text"]) == case["gold"]

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

results = [{"id": c["id"], "pass": score(c), "tags": c["tags"]} for c in CASES]
k = sum(r["pass"] for r in results)
n = len(results)
lo, hi = wilson(k, n)
print(f"overall: {k}/{n} = {k/n:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
for tag in ("easy", "hard"):
    sub = [r for r in results if tag in r["tags"]]
    kk = sum(r["pass"] for r in sub)
    print(f"  {tag:5s}: {kk}/{len(sub)} = {kk/len(sub):.2f}")
print("failures:", [r["id"] for r in results if not r["pass"]])
with open("results.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
```

```text
overall: 4/6 = 0.67  95% CI [0.30, 0.90]
  easy : 3/3 = 1.00
  hard : 1/3 = 0.33
failures: ['t5', 't6']
```

The per-tag breakdown earns its keep: the overall 0.67 hides that easy cases are perfect and hard cases fail two in three. The CI [0.30, 0.90] on six cases is enormous — a reminder that a tiny eval set can tell you almost nothing (§8). The failures list (`t5`, `t6`) feeds the error analysis of §12.

### 9.1 The same harness against a served model

Swap the keyword stub for the OpenAI client pointed at the local server (see [50 — Prompting and APIs](50-ai-prompting-apis.md) and [10 — Serving](10-export-serving.md)). The dataset, scorers and aggregation are unchanged — only `model` becomes a network call.

```python
from openai import OpenAI

CASES = [
    {"id": "t1", "text": "I was charged twice this month", "gold": "billing", "tags": ["easy"]},
    {"id": "t2", "text": "the app crashes when I open settings", "gold": "bug", "tags": ["easy"]},
    {"id": "t3", "text": "how do I export my data?", "gold": "howto", "tags": ["easy"]},
    {"id": "t4", "text": "refund the duplicate charge please", "gold": "billing", "tags": ["hard"]},
    {"id": "t5", "text": "login button does nothing on mobile", "gold": "bug", "tags": ["hard"]},
    {"id": "t6", "text": "where is the invoice for March?", "gold": "billing", "tags": ["hard"]},
]
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
MODEL = "mlx-community/Qwen3-14B-4bit"
SYSTEM = "Classify the ticket as exactly one of: billing, bug, howto. Reply with the label only."

def model(text):
    r = client.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=4,
        # Qwen3 thinks first by default; 4 tokens would all go to the think block
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": text}])
    return (r.choices[0].message.content or "").strip().lower()

results = [{"id": c["id"], "pass": model(c["text"]) == c["gold"]} for c in CASES]
k = sum(r["pass"] for r in results)
print(f"served model: {k}/{len(CASES)} = {k/len(CASES):.2f}")
print("failures:", [r["id"] for r in results if not r["pass"]])
```

Run against a local **stub** that classifies by richer keywords (no model needed to test the harness):

```text
served model: 5/6 = 0.83
failures: ['t6']
```

The stub recognises "login button does nothing" as a bug (which the §9 keyword baseline missed) but still trips on "invoice for March", so the harness reports 5/6 and points at `t6`. A real Qwen3-14B would classify from meaning rather than keywords and likely clear both hard cases — but the only way to *know* is to run this harness against it with `temperature=0` and read the number, not to assume. Apart from the `model` function, the harness is unchanged; that is the point.

Two lines exist because the model under test is Qwen3. Its default chat template makes it think before it answers ([50](50-ai-prompting-apis.md)), so without `enable_thinking: False` the 4-token budget is spent inside the think block and the answer never arrives: `mlx_lm.server` then returns `content: null`. Without the `or ""` guard the script crashes on `None.strip()`; with the guard but without the switch, every case fails, and the cause is the harness, not the model. Both `mlx_lm.server` and `llama-server` accept `chat_template_kwargs` per request. That switch is right for the base model; for an adapter trained with thinking on, serve it as trained ([06 §6](06-data-format.md#6-thinking-and-reasoning_content)) and raise `max_tokens` instead.

---

## 10. Evals in CI as a release gate

Wire the harness into CI so a prompt or model change cannot merge without passing. Treat it exactly like a test suite, with two adjustments for the probabilistic world.

- **Thresholds and tolerance bands.** Gate on absolute bars ("exact-match ≥ 0.85; unsafe-output rate = 0") and on *no regression* against the last green run, but allow a **tolerance band** so a 1-point wobble within the confidence interval does not block a release. Block only when the drop is larger than noise (§8).
- **Flaky-eval handling.** Non-determinism makes evals flaky. Pin it down: `temperature=0`, a fixed `seed` where the server supports it, and fixed decoding params, so the same input gives the same output run to run. For anything still stochastic (sampling at `temperature>0`, an LLM judge), run it \(k\) times and gate on the mean, and store the per-run numbers so you can see the variance. A gate that flaps red/green on identical code is worse than no gate — nobody trusts it.

Record, with every run, the model ID, prompt version, eval-set version, and decoding params, so a red build is reproducible and a green history is comparable. This is the same hygiene as [09](09-evaluation.md) §9's release gates.

---

## 11. Online evaluation

Offline evals measure a frozen set; production measures reality. You need both, because the offline set never perfectly matches live traffic.

- **A/B tests.** Route a fraction of traffic to the new version, compare outcome metrics (resolution rate, thumbs-up rate, conversion) with the statistics of §8. The ground truth for "is it better" — but slow and needing volume.
- **Interleaving.** For ranked outputs (search, retrieval), mix results from both systems in one list and see which the user clicks. More sensitive than A/B for ranking because each user compares both directly.
- **Explicit feedback.** Thumbs up/down, star ratings, "was this helpful?" — sparse (few users click) and biased (unhappy users click more), but a direct signal; log it against the request so you can replay failures.
- **Implicit feedback.** Did the user copy the answer, retry, rephrase, escalate to a human, abandon the session? Plentiful and noisy; good for trends, weak per-example.
- **Guardrail metrics.** Watch latency (p50/p95/p99), cost per request, and error rate alongside quality. A new prompt that lifts quality 2 points but doubles latency or cost may be a net loss; guardrails stop you from "winning" on quality while losing the product. See [55 — Production](55-ai-production.md).

---

## 12. Error analysis workflow

The highest-value hour in the whole discipline: read your failures. Metrics tell you *that* the system fails; only reading the outputs tells you *why*.

1. Pull ~50 failing cases from the latest run.
2. Read each and assign a short cause label, inventing the taxonomy as you go.
3. Count the buckets.
4. Fix the biggest bucket.
5. Add those cases (and near-variants) to the eval set so the fix is protected.

!!! example "On paper"

    Tally of 50 failures from the support-bot eval:

    | Cause | Count | Share |
    |---|---:|---:|
    | Retrieval missed the doc | 18 | 0.36 |
    | Hallucinated a fact | 12 | 0.24 |
    | Wrong output format | 8 | 0.16 |
    | Over-refused (said "I can't") | 7 | 0.14 |
    | Other | 5 | 0.10 |
    | **Total** | **50** | **1.00** |

    Retrieval misses are 18/50 = 0.36 of failures — more than a third. Fixing retrieval (better chunking or hybrid search, [51](51-ai-rag.md)) removes the largest bucket; chasing the "wrong format" bucket first would fix 16% and feel productive while the real problem stayed. Count before you fix.

Error analysis turns a vague "the model is bad" into a ranked, fixable backlog — exactly like triaging a bug tracker by frequency and impact.

---

## 13. Evaluating RAG and agents

The frameworks specialise for multi-step systems: evaluate each **stage** so a failure localises, rather than scoring only the final text.

**RAG pipelines** are two subsystems, graded separately (if the right chunk was never retrieved, no prompt can save the answer):

- *Retrieval*, against queries labelled with their relevant chunks: **recall@k** (did we find the relevant chunks?), **precision@k**, **MRR** (how high was the first good hit?), **nDCG@k** (ranking quality). Worked by hand in [51 — RAG](51-ai-rag.md) §12.
- *Generation*, against the retrieved context: **groundedness/faithfulness** (is every claim supported by the context?), **answer relevance** (does it address the question?), and **citation accuracy** (does each marker point at a chunk that supports its claim?). Worked in [51](51-ai-rag.md) §13.

Decompose a wrong answer along the pipeline — was the chunk *retrieved*, *in the prompt* (context budget), *used* by the model? Each "no" is a different fix. A grounded answer that cites the wrong chunk is still a failure; measure citation accuracy explicitly.

**Agentic workflows** add **trajectory** evaluation — the final answer is not enough:

- **Tool-selection accuracy** (right tool for the step) and **argument accuracy** (right arguments), gradeable deterministically against expected calls — see the tool-call graders in [09 — Evaluation](09-evaluation.md) §4.
- **Task success / completion rate** (did the whole task finish correctly), **recovery rate** (did it recover from a failed call), **steps per task** and **cost/latency per task** as guardrails, and **scope/safety** (zero out-of-scope or destructive actions).
- **Per-step reliability compounds**: a 95%-reliable step over 10 steps is \(0.95^{10}\approx 0.60\) — a 60% task. This is why agents need very high per-step reliability; report pass@k and pass^k (all-k-succeed) as in [09](09-evaluation.md) §5. Full agent evaluation is in [52 — Agents](52-ai-agents.md).

In both, keep the deterministic and execution-based checks (§§2, 4) at the base of the stack and reserve the LLM judge (§5) for what only judgement can score.

---

## 14. Contamination and leaderboard caveats

- **Contamination**: if a public benchmark's test set leaked into a model's training data, the model's score measures memorisation, not skill. Assume popular benchmarks are partly contaminated; your private, recent, held-out golden set is the only clean measurement. Decontaminate your own training data against your eval set (n-gram overlap, near-duplicate detection); see [09](09-evaluation.md) §7.
- **Leaderboards versus your eval**: a leaderboard ranks models on *its* distribution, which is not yours. A model that tops a benchmark can still lose on your support tickets, your documents, your formats. Use leaderboards to shortlist candidates; decide with your own eval on your own data. The benchmark is someone else's regression suite — informative, not authoritative.

A number without its provenance (which set, which version, contaminated?) is not evidence. Report the set, its date, its size, and how you know it was not trained on.

---

## 15. Key takeaways

- An eval set is a regression suite whose assertions are thresholds on pass *rates*, not `==`. It is the deliverable; the model is replaceable.
- Layer graders cheapest-first: deterministic checks (format, JSON, regex, length), then reference metrics (EM, token F1), then execution tests (pass@k), then LLM judges, then humans. Never let a judge overturn a deterministic failure.
- BLEU/ROUGE reward surface overlap and fail on open answers; use a judge or humans there, and calibrate the judge against human labels (Cohen's kappa) while mitigating position, verbosity and self-preference bias.
- A pass rate is an estimate with error bars. Report Wilson intervals; detecting a 5-point difference needs about 1,560 examples per arm; compare two systems *paired* (McNemar) on the same items, because unpaired tests waste data.
- Gate releases in CI on absolute bars and no-regression bands, with `temperature=0`, fixed seeds and repeated runs to kill flakiness. Watch guardrail metrics (latency, cost) beside quality.
- Measure production with A/B, interleaving and feedback; nothing offline replaces live outcomes.
- Read 50 failures, build a taxonomy, fix the biggest bucket, add the cases back. Trust your own held-out eval over any leaderboard, and never report a number without its provenance.

---

## 16. Exercises

!!! question "Exercise 1"

    A model answers "The total is 42 dollars" to a question whose gold answer is "42 dollars". Compute token F1. Why is this a fairer score than exact match here?

??? success "Answer"

    Prediction tokens: the, total, is, 42, dollars (5). Gold tokens: 42, dollars (2). Common: 42, dollars (2). \(P=2/5=0.4\), \(R=2/2=1.0\), \(F_1=2\cdot0.4\cdot1.0/(1.4)=0.571\). Exact match scores 0 because the strings differ, yet the answer contains the full correct answer plus filler — token F1's 0.571 reflects the perfect recall and the padding, which is fairer for a short free-text answer.

!!! question "Exercise 2"

    A coding task is sampled 5 times; 1 sample passes the unit tests. Compute pass@1 and pass@5.

??? success "Answer"

    \(n=5\), \(c=1\). pass@1 \(=1-\binom{4}{1}/\binom{5}{1}=1-4/5=0.2\) (= \(c/n\)). pass@5 \(=1-\binom{4}{5}/\binom{5}{5}\); \(\binom{4}{5}=0\), so pass@5 \(=1-0=1.0\) — with all 5 samples drawn, at least one of 5 is certain to be the one that passes.

!!! question "Exercise 3"

    Two annotators agree on 80 of 100 items. Annotator A says "pass" 70 times, B says "pass" 60 times. Compute Cohen's kappa.

??? success "Answer"

    \(p_o=0.80\). \(p_e=0.70\cdot0.60+0.30\cdot0.40=0.42+0.12=0.54\). \(\kappa=(0.80-0.54)/(1-0.54)=0.26/0.46\approx0.565\) — "moderate". The guide probably needs tightening before these labels are trusted as ground truth.

!!! question "Exercise 4"

    Your eval of 200 cases gives 150 passes. Compute the point estimate and, using the Wilson numbers, say roughly how wide the 95% interval is. Can you distinguish this from a model scoring 73%?

??? success "Answer"

    Point estimate \(150/200=0.75\). Wilson at \(n=200\): \(z^2/n=0.0192\), so the denominator is 1.0192 and the centre is \((0.75+0.0096)/1.0192=0.745\). The half-width is \(1.96\sqrt{0.0009375+0.000024}/1.0192\), which is \(1.96\cdot0.0310/1.0192\approx0.060\), so the interval is about (0.686, 0.805), roughly ±6 points. (The plain normal shortcut \(1.96\sqrt{0.75\cdot0.25/200}\approx0.060\) gives nearly the same width here, because \(n\) is large and \(p\) is not extreme; Wilson also shifts the centre toward 0.5.) A 73% model sits well inside that, so no — 200 cases cannot distinguish 75% from 73%; you would need far more (§8.2).

!!! question "Exercise 5"

    Prompt A and prompt B are run on the same 50 items. They agree on 44 (both right 38, both wrong 6); on the 6 they disagree, A is right 5 times and B once. Apply McNemar's test (no continuity correction).

??? success "Answer"

    \(b=5\), \(c=1\). \(\chi^2=(5-1)^2/(5+1)=16/6\approx2.67\). This is below the 3.841 cutoff, so the difference is *not* significant despite A winning 5 of 6 disagreements — only 6 discordant pairs is too little evidence. Collect more items.

!!! question "Exercise 6"

    You run an eval at `temperature=0.7` and the pass rate jumps between 0.80 and 0.88 across identical-code runs in CI. What two changes make the gate trustworthy?

??? success "Answer"

    (1) Set `temperature=0` (and a fixed seed where supported) so the output is deterministic for format/reference checks. (2) For anything that must stay stochastic, run it \(k\) times and gate on the mean with a tolerance band sized by the observed variance, storing per-run numbers. A gate that flaps on identical code is ignored.

!!! question "Exercise 7"

    You test 10 prompt variants against a baseline, each at \(\alpha=0.05\), and one comes out "significantly better". Why be skeptical, and what do you do?

??? success "Answer"

    With 10 tests at \(\alpha=0.05\) you expect \(10\times0.05=0.5\) false positives by chance, so a lone "winner" is plausibly noise. Apply a multiple-comparison correction (Bonferroni: use \(\alpha=0.005\)), or validate the apparent winner on a held-out set before shipping it (§8.5).

!!! question "Exercise 8"

    An error-analysis tally of 40 failures finds: hallucination 20, retrieval miss 8, format 7, other 5. Which bucket do you fix first, what share of failures does it represent, and what do you do after fixing it?

??? success "Answer"

    Hallucination, 20/40 = 0.50 of failures — fix it first (tighten the "answer only from context" instruction, add groundedness checks). After fixing, add those 20 cases (and variants) to the eval set so the fix is protected against regression, then re-run and re-tally — the next-biggest bucket is now the target.

---

Next: [55 — Production: serving, cost and observability](55-ai-production.md).
