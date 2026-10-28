# 57 — Roadmap, projects and interviews

*Last verified: 28 October 2026.*

This is the chapter that turns the rest of the AI-engineer menu into a plan. It gives you a week-by-week study schedule mapped to the handbook, a portfolio of twelve projects with real acceptance criteria, how to read the jobs you are applying for, interview preparation (concept questions with model answers, coding tasks with runnable solutions and tests, and a system-design walkthrough with real numbers), the tool landscape, how to read a paper, and a rubric to grade yourself. Read [45 — From software engineer to AI engineer](45-ai-engineer-path.md) first for the mindset; this chapter assumes you have decided to do the work.

---

## 1. A week-by-week learning plan

A suggestion, not a law: **24 weeks at roughly 10 hours a week**, about 240 hours in all. Adjust the pace to your life, but keep two rules. **Build something every week** — a deliverable you could show someone — and **map every week to a chapter** so you are never studying in the abstract. The deliverables are the portfolio projects of §3; the plan threads them in as you gain the skills.

| Wk | Focus | Chapters | Deliverable |
|---:|---|---|---|
| 1 | What the job is; the mindset shift | [45](45-ai-engineer-path.md) | Write your "before/after" project plan (a feature you will rebuild the AI way) |
| 2 | The math you actually need | [46](46-ai-math.md), [32](32-mathematics.md) | numpy: softmax, cosine, gradient descent by hand |
| 3–4 | Classical ML and metrics | [47](47-ai-classical-ml.md), [09](09-evaluation.md) | **Project 1** — ticket classifier, classical vs LLM |
| 5–6 | Deep learning | [48](48-ai-deep-learning.md) | Train an MLP; read a loss curve; explain overfit |
| 7–8 | How LLMs work | [49](49-ai-llm-internals.md), [37](37-architecture.md) | BPE merge + attention in numpy (coding tasks §6) |
| 9 | Data preparation | [60](60-ai-data-preparation.md), [04](04-data-strategy.md), [06](06-data-format.md) | **Project 2** — clean, dedup, version a messy corpus + data card |
| 10–11 | Prompting and LLM APIs | [50](50-ai-prompting-apis.md) | **Project 3** — structured invoice extractor with validation |
| 12–13 | Embeddings and RAG | [51](51-ai-rag.md) | **Project 4** — RAG over a docs site with an eval |
| 14 | Agents and tool use | [52](52-ai-agents.md) | **Project 5** — tool-using agent with budgets + approvals |
| 15 | MCP | [58](58-ai-mcp.md), [11](11-kali-agent.md) | **Project 6** — build, test and publish an MCP server |
| 16 | Agent frameworks | [61](61-ai-agent-frameworks.md) | Port Project 5 to one framework; write the trade-off note |
| 17–18 | Fine-tuning | [53](53-ai-fine-tuning.md), [08](08-finetuning.md), [15](15-agentic-methods.md), [17](17-finetune-2026-update.md) | **Project 7** — LoRA fine-tune on the M1 Pro, before/after eval |
| 19 | Training from zero | [35](35-from-zero.md) | **Project 8** — a tiny model trained from scratch |
| 20–21 | Evaluation | [54](54-ai-evaluation.md), [42](42-distill-eval.md) | **Project 9** — an eval harness that runs in CI |
| 22 | Production and cost | [55](55-ai-production.md), [10](10-export-serving.md), [02](02-hardware.md) | **Project 10** cost cascade + **Project 11** gateway with tracing |
| 23 | Safety and security | [56](56-ai-safety.md), [12](12-safety-legal.md) | **Project 12** — a red-team suite + a threat model |
| 24 | System design + applications | this chapter | The 2M-ticket design doc (§7); polish READMEs; the §12 checklist |

The analogy: this is a training plan, not a textbook. A runner does not read about running for 24 weeks; they run a little more each week and race at the end. Your "race" is a portfolio and a set of interviews you can pass.

---

## 2. Reading an AI engineer job description

Job posts list buzzwords; your job is to translate each into a skill you can prove and a question you can answer. Here are three phrasings that recur in real posts, mapped to where you build the skill, the project that is your evidence, and the questions an interviewer asks about it.

### 2.1 "Pipeline & Agent Building"

> *"developing RAG architectures, vector databases, tool-calling systems, and autonomous AI agents"*

| Translate to | Learn in | Your evidence | They will ask (§5) |
|---|---|---|---|
| RAG architecture: chunking, retrieval, reranking, citations | [51](51-ai-rag.md) | Project 4 (RAG + eval) | 5.6: when to rerank; naive vs hybrid retrieval |
| Vector databases: exact vs ANN, memory, filters, tenancy | [51](51-ai-rag.md), [60](60-ai-data-preparation.md) | Project 4 | 5.6: choosing a vector DB; index memory math |
| Tool-calling systems | [50](50-ai-prompting-apis.md), [52](52-ai-agents.md), [58](58-ai-mcp.md) | Projects 5, 6 | 5.7: tool schema design; MCP (5.8) |
| Autonomous agents | [52](52-ai-agents.md), [61](61-ai-agent-frameworks.md) | Project 5 | 5.7: workflow vs autonomous; reliability compounding |

### 2.2 "AI & LLMs"

> *"prompt engineering, embeddings, RAG, agentic workflows, and AI evaluations"*

| Translate to | Learn in | Your evidence | They will ask (§5) |
|---|---|---|---|
| Prompt engineering (structure, few-shot, structured output) | [50](50-ai-prompting-apis.md) | Project 3 | 5.5: few-shot vs fine-tune; JSON-output reliability |
| Embeddings | [46](46-ai-math.md), [51](51-ai-rag.md) | Project 4 | 5.6: what an embedding is; cosine (coding §6) |
| RAG + agentic workflows | [51](51-ai-rag.md), [52](52-ai-agents.md), [61](61-ai-agent-frameworks.md) | Projects 4, 5 | 5.7: workflow patterns vs a free-running agent |
| AI evaluations | [54](54-ai-evaluation.md), [09](09-evaluation.md) | Project 9 | 5.10: golden sets; LLM-as-judge biases; CIs |

### 2.3 "Data preparation"

> *"data preparation"* — small words, large job. It usually means: turn a messy source into a clean, deduplicated, leakage-free, labeled, versioned dataset, and be able to defend its quality.

| Translate to | Learn in | Your evidence | They will ask (§5) |
|---|---|---|---|
| Cleaning, dedup, provenance, versioning | [60](60-ai-data-preparation.md), [04](04-data-strategy.md) | Project 2 | 5.13: near-duplicate dedup; a data card |
| Train/test discipline, leakage | [47](47-ai-classical-ml.md), [60](60-ai-data-preparation.md) | Projects 2, 9 | 5.13: where leakage hides |
| Labeling quality, agreement | [54](54-ai-evaluation.md), [60](60-ai-data-preparation.md) | Project 2 | 5.13: measuring label quality |

If a post has a skill with no matching project, build the project before you apply. The interview is mostly "walk me through something you built"; you want a true answer for every line of the post.

---

## 3. A portfolio of projects

Twelve projects in increasing difficulty. Each lists a **spec**, the **skills** it demonstrates, **acceptance criteria with numbers** (so "done" is objective), **stretch goals**, and what the **README** must contain. Put every project in its own repo with the README, the eval numbers, and a short "what I'd do next" note. A hiring manager reads the README and the eval table first.

!!! note "README template (every project)"

    1. One-paragraph problem statement and who it is for.
    2. Architecture diagram (ASCII is fine) and the key decisions, with the trade-off.
    3. How to run it (one command) and how to run the eval.
    4. The **eval table**: the metric, the number, and the baseline it beats.
    5. Known limitations and failure modes (honesty reads as seniority).
    6. What you would do next with more time.

### Project 1 — Ticket classifier: classical ML vs LLM

- **Spec.** Classify support tickets into 5 categories. Build two solutions — a scikit-learn baseline (TF-IDF + logistic regression) and a zero/few-shot LLM — and compare them head to head.
- **Skills.** Classical ML, metrics, baselines, prompt engineering, cost/latency thinking.
- **Acceptance.** Baseline macro-F1 ≥ 0.75 on a held-out set; a comparison table with macro-F1, p95 latency and cost-per-1k for both; a written recommendation.
- **Stretch.** Add a confidence threshold that routes low-confidence tickets to the LLM only (a cascade).
- **README.** The table showing where classical ML wins (cheap, fast, good enough) and where the LLM wins (few labels, new categories).

### Project 2 — Data preparation: messy dump to clean corpus

- **Spec.** Take a messy document dump (mixed formats, duplicates, boilerplate, PII) and produce a clean, **deduplicated**, **versioned** corpus with a **data card**.
- **Skills.** Data preparation, dedup (exact + near-duplicate), PII redaction, provenance, versioning, documentation.
- **Acceptance.** Reproducible pipeline (one command); a report with rows in/out, duplicates removed (exact and near), PII spans redacted; a data card ([60](60-ai-data-preparation.md)) listing source, license, collection method and known biases; the corpus hashed and version-tagged.
- **Stretch.** Add a leakage check that proves no near-duplicate spans cross the train/test split.
- **README.** The before/after row counts and the dedup method; why versioning and the data card matter for a later fine-tune ([53](53-ai-fine-tuning.md)) and for GDPR ([12](12-safety-legal.md#3-data-handling)).

### Project 3 — Structured invoice extractor with validation

- **Spec.** Extract `{invoice_id, total, currency, line_items}` from invoice text into schema-valid JSON, with validate-and-retry on failure.
- **Skills.** Structured output, pydantic/JSON Schema, retries, prompting.
- **Acceptance.** ≥ 98% schema-valid output on a 100-document set after at most 2 retries; totals numerically correct on ≥ 95%; a clear error for unparseable input (no silent guess).
- **Stretch.** Cross-field validation (line items sum to the total) and a human-review queue for mismatches.
- **README.** The schema, the retry loop, and the measured valid-rate before and after adding retries (the coding task in §6 is the core of this).

### Project 4 — RAG over a docs site with an eval

- **Spec.** Index a documentation site; answer questions with citations; ship an offline retrieval + answer eval.
- **Skills.** Chunking, embeddings, vector search, reranking, citations, retrieval metrics.
- **Acceptance.** recall@5 ≥ 0.80 on a 50-question gold set; groundedness ≥ 0.85 (judged); every answer cites a chunk; abstention ≥ 0.80 on at least 10 extra questions whose answer is not in the docs.
- **Stretch.** Hybrid search (BM25 + vectors) with reciprocal-rank fusion and a reranker; show the recall@5 lift.
- **README.** The retrieval metrics, the chunking choice and why, and the failure cases you found by error analysis.

### Project 5 — Tool-using agent with budgets and approvals

- **Spec.** An agent that uses 3–5 typed tools to complete a multi-step task, with a per-run tool-call budget and human approval for any side effect.
- **Skills.** Agent loop, tool-schema design, reliability math, budgets, human-in-the-loop, sandboxing.
- **Acceptance.** Task-completion ≥ 0.70 on a 20-task set; a hard cap on tool calls per run; zero unapproved side effects; a trace per run.
- **Stretch.** A recovery path when a tool errors; report pass^k reliability over repeated runs.
- **README.** The tool schemas, the budget and approval design, and the per-step vs per-run reliability numbers ([52](52-ai-agents.md)).

### Project 6 — Build, test and publish an MCP server

- **Spec.** A read-only MCP server (e.g. a docs or tickets server) exposing typed tools, with tests via an in-memory client and a short security review.
- **Skills.** MCP tool design, typed schemas, testing, the security mindset from [56](56-ai-safety.md) and [58](58-ai-mcp.md).
- **Acceptance.** 3+ typed tools with input validation; a test suite that drives the server through an in-memory client (no network) and covers the error paths; a published package or container; a security review covering tool-description trust, least privilege, and consent.
- **Stretch.** Wire the server into the local Qwen model ([58](58-ai-mcp.md)) and show an end-to-end tool call.
- **README.** The tool schemas, the in-memory test approach, and the security review checklist with your findings.

### Project 7 — LoRA fine-tune on the M1 Pro, before/after eval

- **Spec.** Fine-tune a small model with LoRA on a task where prompting plateaus (e.g. a strict output format), on a 32 GB M1 Pro, and prove the gain with an eval.
- **Skills.** PEFT/LoRA, data prep with loss masking, hyperparameters, overfitting, evaluation.
- **Acceptance.** A before/after eval on a held-out set where the 95% confidence interval of the gain on the target metric excludes zero (a paired comparison on the same examples); a training curve; the exact config and data manifest.
- **Stretch.** A DPO or preference pass; compare against a RAG-only solution for the same task.
- **README.** The decision table (why fine-tune over prompt/RAG here), the config, the before/after numbers with intervals, and the cost in wall-clock hours ([08](08-finetuning.md), [53](53-ai-fine-tuning.md)).

### Project 8 — A small model trained from zero

- **Spec.** Train a small language model from scratch on a tiny corpus to understand the full pipeline (tokenizer → data → training loop → eval). See [35 — From zero](35-from-zero.md).
- **Skills.** Tokenizer training, data pipeline, training loop, loss curves, memory math.
- **Acceptance.** On a tiny set (one batch) the training loss falls close to zero, which proves the loop can learn; on the larger corpus the validation loss ends well below the uniform-guess baseline \(\ln V\) for vocabulary size \(V\) and tracks the training loss; peak memory within 20% of your written estimate.
- **Stretch.** A small post-training (SFT) pass on top of the pretrained base.
- **README.** The architecture choices, the loss curve, and the honest statement that this is for understanding, not for a product.

### Project 9 — An eval harness that runs in CI

- **Spec.** A golden-set eval harness that runs on every change and gates a release, with statistics (not just a mean).
- **Skills.** Evals as tests, golden sets, reference metrics, LLM-as-judge, confidence intervals, CI.
- **Acceptance.** Runs in CI; prints per-metric scores with Wilson intervals; fails the build when a metric drops below a threshold; a contamination note.
- **Stretch.** A paired comparison (McNemar) between two model versions; an LLM-as-judge with measured agreement against human labels (Cohen's kappa).
- **README.** The metrics, the thresholds, and a screenshot of a failing run that caught a regression ([54](54-ai-evaluation.md), [09](09-evaluation.md)).

### Project 10 — A cost-optimized cascade

- **Spec.** Route easy requests to a small/cheap model and hard ones to a large model, holding quality while cutting cost.
- **Skills.** Routing, cascades, cost modeling, calibration/confidence, evaluation.
- **Acceptance.** Task quality no worse than the large-model-only baseline by more than a margin you fix in advance (e.g. 2 points: the lower end of the quality delta's 95% interval stays above −2), at a measured cost reduction (report the % and the quality delta with its interval).
- **Stretch.** Add exact and semantic caching; report the hit rate and the extra savings.
- **README.** The routing rule, the cost-vs-quality curve, and the break-even analysis ([55](55-ai-production.md)).

### Project 11 — A production gateway with tracing

- **Spec.** A thin gateway in front of your model(s): auth, rate/cost limits, retries with backoff, timeouts, and full request tracing with token usage and latency percentiles.
- **Skills.** Production LLMOps, observability, reliability patterns, privacy in logs.
- **Acceptance.** Per-key rate and spend caps enforced; p50/p95/p99 latency and token usage per request in traces; PII redacted from logs ([56](56-ai-safety.md)); a load test showing the limits hold.
- **Stretch.** Prompt/model versioning and a canary rollout with automatic rollback on a metric drop.
- **README.** The architecture, the SLOs, and the dashboard of latency and cost.

### Project 12 — A red-team suite

- **Spec.** An automated red-team suite of benign injection/exfiltration canary cases that runs against your app and reports an attack-success rate, plus a written threat model.
- **Skills.** Threat modeling, prompt-injection defenses, output handling, red-teaming as a process.
- **Acceptance.** 15+ versioned test cases; an ASR reported with an interval; a demonstrated drop in ASR after adding a defense; a threat model and a system card.
- **Stretch.** Wire it into CI so a release is blocked if ASR rises.
- **README.** The threat model, the case list (benign canaries only), and the before/after ASR ([56](56-ai-safety.md)).

---

## 4. How to transition into the role

You rarely need to quit to become an AI engineer. The fastest paths, roughly in order of leverage:

- **Inside your current company.** You already have a codebase, data, and users — the three things a bootcamp cannot give you. Find a task that is a genuine fit for an LLM (unstructured text in, a decision or structured data out) and build it the right way, with an eval and a cost budget. Nothing on a résumé beats "I shipped the first AI feature at my company, here are the eval numbers."
- **Picking the first AI feature.** Choose one that is **valuable, bounded, and measurable**, and where a wrong answer is cheap (a draft a human reviews, not an irreversible action). Good first features: ticket triage, draft replies, internal search over docs, log summarization. Bad first features: anything that moves money or sends messages without a human in the loop. Write the eval before the prompt.
- **Open source.** Contribute to the tools you use — a bug fix, a doc, an eval, an MCP server. It builds a public track record and teaches you the internals faster than reading.
- **Writing about your work.** A short, honest write-up of a project (what you tried, what failed, the numbers) is the single highest-leverage thing you can do for visibility. It doubles as interview prep: you cannot write it clearly until you understand it.

The analogy: this is the same move as adopting any new technology at work — start with a low-stakes internal service, measure it, and expand once it earns trust. The difference is that "measure it" is harder and more important, because the output is probabilistic.

---

## 5. Interview preparation: concept questions

Short questions, concise model answers. Say the definition, then the decision it drives, in a sentence or two — that is what an interviewer wants.

### 5.1 Math and statistics

**Q. What does cosine similarity measure and why is it used for embeddings?** The cosine of the angle between two vectors; it ignores magnitude and measures direction, so two texts with the same meaning but different lengths still score as similar. It is the default similarity for normalized embeddings.

**Q. Why subtract the max before exponentiating in softmax?** Numerical stability: `exp` of a large logit overflows to infinity. Subtracting the max shifts the logits so the largest is 0, which cannot overflow, and does not change the result because softmax is shift-invariant.

**Q. A model beats the baseline by 2 points on 100 examples. Is it better?** Not necessarily. Compute a confidence interval; on 100 examples a 2-point gap is usually inside the noise. Report intervals and use a paired test before claiming an improvement.

### 5.2 Classical ML

**Q. When does classical ML beat an LLM?** When you have labels and tabular or short-text features, need low latency and cost, and want calibrated probabilities. A TF-IDF + logistic-regression classifier can match an LLM on a fixed-category task at a fraction of the cost and latency.

**Q. What is data leakage and give an example.** Information from the test set (or the future) sneaking into training, inflating offline scores that collapse in production. Example: fitting the TF-IDF vocabulary on the full dataset before splitting, or a near-duplicate document appearing in both train and test.

**Q. Precision vs recall — when do you optimize which?** Precision when a false positive is expensive (flagging a good transaction as fraud); recall when a false negative is expensive (missing a cancer). You trade them with a threshold and summarize with F1 or the PR curve.

### 5.3 Deep learning

**Q. What does backpropagation compute?** The gradient of the loss with respect to every parameter, by applying the chain rule backward through the computation graph, so an optimizer can take a step that lowers the loss.

**Q. Why use Adam over plain SGD?** Adam adapts the step size per parameter using running averages of the gradient (its mean) and of the squared gradient (an uncentered variance), which makes it less sensitive to the learning-rate choice and fast to converge — the sensible default for transformers. It still needs a tuned learning rate and usually a warmup.

**Q. Your training loss drops but validation loss rises. What is happening and what do you do?** Overfitting. Add regularization (weight decay, dropout), get more or cleaner data, reduce model/adapter capacity, or stop earlier. The gap between the curves is the diagnosis.

### 5.4 LLMs

**Q. What is a token and why does it matter for cost?** A sub-word unit from the tokenizer; the model reads and writes tokens, and APIs bill per token. Token count, not character count, drives your cost and your context budget.

**Q. Explain temperature, top-k and top-p.** They shape sampling from the next-token distribution. Temperature flattens (high) or sharpens (low) it; top-k keeps the k most likely tokens; top-p (nucleus) keeps the smallest set whose probability mass reaches p. Low temperature and small p give more deterministic output.

**Q. What is the KV cache and why does context length cost memory?** During generation the model caches the keys and values of past tokens so it does not recompute them; that cache grows linearly with context length, so a long context costs memory per concurrent request, not just compute.

**Q. Pretraining vs SFT vs preference tuning — one line each.** Pretraining learns language from raw text (next-token prediction); SFT teaches it to follow instructions from example conversations; preference tuning (e.g. DPO) nudges it toward answers humans prefer over others.

### 5.5 Prompting

**Q. When do you few-shot prompt instead of fine-tuning?** When you have few examples, the task changes often, or you want to iterate in minutes. Fine-tune when the behavior is stable, you have enough data, and you need lower latency/cost or a format prompting cannot reliably hit.

**Q. How do you get reliable JSON out of a model?** Ask for a schema, constrain decoding or use the provider's structured-output mode if available, then **validate and retry**: parse against a schema (pydantic), and on failure send the error back and ask again. Never trust the first string (the coding task in §6.7).

### 5.6 RAG, embeddings and vector databases

**Q. Walk me through a RAG architecture and the choices at each stage.** Chunk the corpus (size/overlap chosen by content), embed the chunks, index them, retrieve top-k for a query (optionally hybrid BM25 + vector with rank fusion), rerank with a cross-encoder if precision matters, build a prompt with the chunks and ask for a cited answer, and abstain when nothing relevant is retrieved. Each stage is a dial you tune against a retrieval eval.

**Q. How do you choose a vector database?** By the constraints, not the brand. Corpus size and growth, latency target, whether you need metadata filters and per-tenant access control, exact vs approximate search, and whether it must be embedded (in-process) or a service. At a few hundred thousand chunks an in-process index fits in RAM on one box (see the memory math in §7); you reach for a distributed store when the index no longer fits or you need managed ops, not before.

**Q. Exact vs approximate nearest-neighbor search — the trade-off?** Exact search is correct but scans everything (slow at scale); approximate (HNSW, IVF, PQ) trades a little recall for a large speed and memory win. You pick based on your recall@k budget and corpus size.

**Q. When does reranking earn its cost?** When the right chunk is in the retrieved shortlist (say the top 50) but not among the few chunks you pass to the model, so the answer misses it. A cross-encoder reranker reorders a shortlist for precision at the cost of latency; measure the recall@k lift at the k you actually pass to the model before keeping it.

### 5.7 Agents, workflows and frameworks

**Q. Agentic workflow vs an autonomous agent — what is the difference and which do you prefer?** A workflow is a fixed graph of steps you designed (retrieve → draft → check → send), with the model filling in each node; an autonomous agent decides its own next step in a loop. Prefer the workflow whenever the steps are known: it is cheaper, more reliable, and easier to evaluate. Reach for autonomy only when the path genuinely cannot be predetermined, and even then bound it with budgets and approvals.

**Q. Why does per-step reliability matter so much for agents?** Errors compound. If each step succeeds with probability p over n steps, the run succeeds with roughly p^n — 95% per step over 10 steps is about 60% per run. Fewer steps, verified steps, and human gates on side effects beat chasing the per-step rate.

**Q. How do you design a tool schema for an agent?** Narrow and typed: the smallest set of tools the task needs, each with a strict input schema, least privilege, a clear description of when (and when not) to use it, and validation of arguments before execution. Avoid open-ended tools like "run shell"; build a specific tool instead.

**Q. When do you adopt an agent framework, and when not?** Adopt one when you need its abstractions — a graph runtime, state/memory, retries, observability — and the team will maintain it; a framework saves boilerplate and encodes patterns. Skip it when your flow is a short, fixed pipeline you can write in plain code and test easily; a framework then adds a dependency and hidden control flow for little gain. Decide against your actual flow, and see [61 — Agent frameworks](61-ai-agent-frameworks.md) for the specific trade-offs.

### 5.8 MCP

**Q. What is MCP and what problem does it solve?** The Model Context Protocol is an open standard for connecting models to tools, data and prompts through a uniform client/server interface, so a capability built once (an MCP server) works across any MCP-aware host instead of being re-wired per app. It is the "USB-C port" for model context. See [58 — MCP](58-ai-mcp.md).

**Q. Name two MCP-specific security risks and how you mitigate them.** Tool-description poisoning (a malicious server's tool descriptions are untrusted text in your context that can steer the model) — mitigate by allow-listing servers, treating descriptions as untrusted, and least privilege. Token passthrough (the server accepts a token that was not issued to it and forwards it to a downstream API, bypassing that API's audience checks and your audit trail) — the MCP spec forbids it: accept only tokens whose audience is your server and call downstream APIs with their own credentials. A third, the confused deputy, hits MCP proxy servers that use one static OAuth client ID for every user — mitigate with per-client consent before forwarding to the third-party authorization server. [58](58-ai-mcp.md) and [56](56-ai-safety.md) cover these.

**Q. How do you test an MCP server without a network or a live model?** Drive it through an in-memory client in the test process: call each tool with valid and invalid inputs, assert the typed outputs and the error paths, and check that out-of-scope requests are refused. That keeps tests fast and deterministic (Project 6).

### 5.9 Fine-tuning

**Q. Prompt vs RAG vs fine-tune — how do you decide?** Prompt first (fastest to change). Add RAG when the model needs facts it does not have or that change. Fine-tune when you need a stable behavior or format, lower latency/cost, or a style prompting cannot hit reliably — and you have the data to do it. They compose; RAG + a light fine-tune is common.

**Q. What does LoRA change, and why is it cheap?** It freezes the base weights and learns small low-rank update matrices for chosen layers, so you train and store millions to tens of millions of parameters (well under 1% of the model) instead of billions — a rank-16 all-layer adapter for Qwen3-14B is about 60M. Combined with a 4-bit base (QLoRA), that is what makes fine-tuning a 14B model feasible on a 32 GB Mac ([08](08-finetuning.md)).

### 5.10 Evaluation

**Q. What is a golden set and how is it like a test suite?** A fixed, versioned set of inputs with expected outputs or graders; it is a regression suite whose assertions are thresholds. You run it on every change and block releases when a metric regresses.

**Q. What are the biases of LLM-as-judge and how do you control them?** Position bias (favoring the first answer), verbosity bias (favoring longer answers), and self-preference. Control them by randomizing order, using a rubric, and measuring the judge's agreement with human labels (Cohen's kappa) before trusting it.

### 5.11 Production

**Q. What are TTFT and TPOT and why report both?** Time-to-first-token (how long until output starts) and time-per-output-token (how fast it streams after). They separate prefill latency from decode speed; a user feels TTFT immediately, so streaming with a low TTFT beats a lower total latency without streaming.

**Q. When is self-hosting cheaper than an API?** At high, steady volume where a reserved GPU (or a Mac for small models) is busy enough to beat per-token pricing; below that break-even, or for bursty traffic, an API wins. Compute the break-even from your real token volume rather than guessing.

### 5.12 Safety

**Q. Why can't prompt injection be fixed with a better prompt?** The model makes no architectural distinction between instructions and data — a guard sentence is just more tokens an attacker can out-weigh. Defense is architectural: least-privilege tools, treating output as untrusted, human approval, and breaking the lethal trifecta ([56](56-ai-safety.md)).

**Q. The model outputs text you render in a web page. What is the risk and the fix?** Insecure output handling → XSS, because the output is attacker-influenceable. Treat it like user input: HTML-escape before rendering, use a strict CSP, and do not auto-render images or links that could exfiltrate data.

### 5.13 Data preparation

**Q. How do you deduplicate a corpus, and why beyond exact duplicates?** Exact dedup (hash each document) is the first pass; then near-duplicate dedup (shingling + MinHash/LSH, or embedding similarity) because paraphrases and reformatted copies inflate the data and can cross your train/test split. Report both counts.

**Q. Where does leakage hide in a dataset, and how do you catch it?** In near-duplicates across the split, in features computed over the whole set before splitting, in timestamps that let the model "see the future", and in labels derived from the target. Catch it by splitting first, deduping across the split, and checking that no near-duplicate spans appear on both sides.

**Q. How do you measure labeling quality?** Multiple annotators on a sample and an inter-annotator agreement statistic (e.g. Cohen's kappa); adjudicate disagreements; track the error rate on a gold re-labeling. Low agreement means the task is ambiguous — fix the guidelines before scaling the labeling.

---

## 6. Interview preparation: coding tasks

Eight tasks with full, runnable solutions and tests. Each runs on the handbook's Python (numpy/scikit-learn/pydantic/openai). The model-facing ones use the local server ([10 — Export & serving](10-export-serving.md)); the output shown here was produced against a local stub, and a real model would wrap the same values in prose.

### 6.1 Numerically stable softmax

```python
import numpy as np


def softmax(x):
    x = np.asarray(x, dtype=float)
    z = x - x.max()                    # subtract the max for stability
    e = np.exp(z)
    return e / e.sum()


p = softmax([2.0, 1.0, 0.0])
print("probs:", np.round(p, 4))
print("sum:  ", round(float(p.sum()), 6))

# Shift invariance: adding a constant to every logit must not change the result.
big = softmax([1002.0, 1001.0, 1000.0])
print("shifted equal:", bool(np.allclose(p, big)))

# The naive version overflows at large logits; the stable one does not.
with np.errstate(over="ignore"):
    print("naive exp(1000):", np.exp(1000.0))

assert abs(float(p.sum()) - 1.0) < 1e-12
assert np.allclose(p, big)
print("tests: ok")
```

```text
probs: [0.6652 0.2447 0.09  ]
sum:   1.0
shifted equal: True
naive exp(1000): inf
tests: ok
```

### 6.2 Top-p (nucleus) sampling

```python
import numpy as np


def top_p_sample(probs, p, rng):
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(probs)[::-1]            # high to low
    cum = np.cumsum(probs[order])
    cutoff = int(np.searchsorted(cum, p)) + 1  # smallest nucleus with mass >= p
    keep = order[:cutoff]
    renorm = probs[keep] / probs[keep].sum()
    return int(rng.choice(keep, p=renorm)), keep.tolist()


vocab = ["refund", "return", "cancel", "ship", "track"]
probs = [0.5, 0.25, 0.15, 0.07, 0.03]
rng = np.random.default_rng(0)

token, nucleus = top_p_sample(probs, 0.9, rng)
print("nucleus:", [vocab[i] for i in nucleus])
counts = {}
for _ in range(10000):
    t, _ = top_p_sample(probs, 0.9, rng)
    counts[vocab[t]] = counts.get(vocab[t], 0) + 1
print("sampled:", vocab[token])
print("empirical:", {k: round(v / 10000, 3) for k, v in sorted(counts.items())})

assert [vocab[i] for i in nucleus] == ["refund", "return", "cancel"]
print("tests: ok")
```

```text
nucleus: ['refund', 'return', 'cancel']
sampled: return
empirical: {'cancel': 0.165, 'refund': 0.554, 'return': 0.281}
tests: ok
```

The nucleus at p=0.9 is the three tokens whose mass reaches 0.9 (0.5 + 0.25 + 0.15); `ship` and `track` are cut, and the empirical frequencies match the renormalized nucleus.

### 6.3 Cosine top-k search

\[
\cos(\mathbf a,\mathbf b)=\frac{\mathbf a\cdot\mathbf b}{\lVert\mathbf a\rVert\,\lVert\mathbf b\rVert}
\]

!!! example "On paper"

    Query a = (1, 2, 2), document b = (2, 1, 2):

    \[
    \begin{aligned}
    \mathbf a\cdot\mathbf b &= 2+2+4 = 8\\
    \lVert\mathbf a\rVert &= \sqrt{1+4+4} = 3\\
    \lVert\mathbf b\rVert &= 3\\
    \cos &= 8/9 \approx 0.889
    \end{aligned}
    \]

    0.889 is near 1, so the two point almost the same way. The code ranks the exactly-parallel vector (3, 6, 6) first with cosine 1.0, then this pair.

```python
import numpy as np


def normalize(m):
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def top_k(query, docs, k):
    q = query / np.linalg.norm(query)
    scores = normalize(docs) @ q               # cosine similarity
    idx = np.argsort(scores)[::-1][:k]
    return idx, scores[idx]


docs = np.array([
    [2.0, 1.0, 2.0],   # 0: "money back please"
    [0.0, 1.0, 0.0],   # 1: "update my address"
    [1.0, 0.0, 0.0],   # 2: "track my parcel"
    [3.0, 6.0, 6.0],   # 3: "refund request" (same direction as query)
])
query = np.array([1.0, 2.0, 2.0])             # "refund my order"

idx, scores = top_k(query, docs, 2)
print("top-2 indices:", idx.tolist())
print("top-2 scores: ", np.round(scores, 4).tolist())

assert idx.tolist() == [3, 0]
print("tests: ok")
```

```text
top-2 indices: [3, 0]
top-2 scores:  [1.0, 0.8889]
tests: ok
```

### 6.4 One BPE merge step

```python
import re
from collections import Counter


def best_pair(corpus):
    pairs = Counter()
    for word, freq in corpus.items():
        syms = word.split()
        for a, b in zip(syms, syms[1:]):
            pairs[(a, b)] += freq
    return pairs.most_common(1)[0]


def merge(corpus, pair):
    # match the pair only as two whole symbols, never inside a longer one
    bigram = re.compile(r"(?<!\S)" + re.escape(" ".join(pair)) + r"(?!\S)")
    joined = "".join(pair)
    return {bigram.sub(joined, w): f for w, f in corpus.items()}


# words are space-separated symbols; </w> marks a word end
corpus = {"l o w </w>": 5, "l o w e r </w>": 2,
          "n e w e s t </w>": 6, "w i d e s t </w>": 3}

pair, count = best_pair(corpus)
print("most frequent pair:", pair, "count:", count)
corpus = merge(corpus, pair)
print("after one merge:", corpus)

assert pair == ("e", "s") and count == 9
print("tests: ok")
```

```text
most frequent pair: ('e', 's') count: 9
after one merge: {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w es t </w>': 6, 'w i d es t </w>': 3}
tests: ok
```

The pair `('e', 's')` appears in `newest` (6) and `widest` (3), total 9, so it is merged into the new symbol `es`. It is tied: `('s', 't')` and `('t', '</w>')` also count 9, and `most_common` breaks the tie by first occurrence. Real tokenizers fix a tie-break rule too, or two runs on the same data give different vocabularies. The regex in `merge` matters once symbols grow: a plain `str.replace("e s", "es")` would also fire inside `"ne s"`, gluing a symbol `ne` to `s`. Repeat the two steps to grow the vocabulary — that is BPE training.

### 6.5 Attention in numpy

```python
import numpy as np


def softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)            # scaled dot product
    if mask is not None:
        scores = np.where(mask, scores, -1e9)  # causal: block the future
    weights = softmax(scores)
    return weights @ V, weights


np.random.seed(0)
T, d = 3, 4                                     # 3 tokens, head dim 4
Q = np.random.randn(T, d)
K = np.random.randn(T, d)
V = np.random.randn(T, d)
causal = np.tril(np.ones((T, T), dtype=bool))   # token t sees 0..t only

out, w = attention(Q, K, V, causal)
print("weights row sums:", np.round(w.sum(axis=1), 6).tolist())
print("token 0 attends to:", np.round(w[0], 4).tolist())
print("output shape:", out.shape)

assert np.allclose(w.sum(axis=1), 1.0)
assert np.allclose(w[0], [1.0, 0.0, 0.0])
print("tests: ok")
```

```text
weights row sums: [1.0, 1.0, 1.0]
token 0 attends to: [1.0, 0.0, 0.0]
output shape: (3, 4)
tests: ok
```

Each row of the attention matrix sums to 1 (it is a distribution). With a causal mask, token 0 can only attend to itself, so its row is `[1, 0, 0]`. The deep version is in [32 — Mathematics](32-mathematics.md).

### 6.6 Retry with exponential backoff

```python
import functools


def retry(max_attempts=4, base=0.5, factor=2.0, sleep=None):
    sleep = sleep or (lambda s: None)          # injectable for tests

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kw):
            delay = base
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kw)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    print(f"attempt {attempt} failed ({e}); sleep {delay}")
                    sleep(delay)
                    delay *= factor
        return wrapper
    return deco


slept = []
calls = {"n": 0}


@retry(sleep=slept.append)
def flaky():
    calls["n"] += 1
    if calls["n"] < 3:
        raise ConnectionError("503")
    return "ok"


print("result:", flaky())
print("calls:", calls["n"])
print("delays:", slept)

assert slept == [0.5, 1.0] and calls["n"] == 3
print("tests: ok")
```

```text
attempt 1 failed (503); sleep 0.5
attempt 2 failed (503); sleep 1.0
result: ok
calls: 3
delays: [0.5, 1.0]
tests: ok
```

The delays double (0.5, 1.0, 2.0, …); injecting `sleep` makes the test instant and deterministic. In production add jitter and a cap, and only retry idempotent calls on 429/5xx.

### 6.7 A JSON-output validator with retry

```python
import json
import re
from openai import OpenAI
from pydantic import BaseModel, ValidationError

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
MODEL = "mlx-community/Qwen3-14B-4bit"


class Invoice(BaseModel):
    invoice_id: str
    total: float
    currency: str


def extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group()) if m else None


def get_invoice(doc, max_tries=3):
    messages = [{"role": "user",
                 "content": f"Extract the invoice as JSON. Document:\n{doc}"}]
    for attempt in range(1, max_tries + 1):
        reply = client.chat.completions.create(
            model=MODEL, messages=messages,
            # Qwen3 thinks first by default; ask for the JSON directly
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ).choices[0].message.content or ""        # content may be None
        try:
            return Invoice.model_validate(extract_json(reply)), attempt
        except (ValidationError, json.JSONDecodeError) as e:
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user",
                 "content": f"That did not validate: {str(e)[:80]}. "
                            "Return only valid JSON with total as a number."}]
    raise ValueError("no valid output after retries")


invoice, attempt = get_invoice("Invoice INV-1042, total 129,50 EUR")
print("validated on attempt:", attempt)
print(invoice.model_dump())

assert invoice.total == 129.5 and attempt <= 3   # the stub needs 2 tries
print("tests: ok")
```

```text
validated on attempt: 2
{'invoice_id': 'INV-1042', 'total': 129.5, 'currency': 'EUR'}
tests: ok
```

The stub first returns a preamble plus `"total": "129,50"` (a string with a comma), which fails pydantic validation; the loop feeds the error back and the second reply validates. A real model often gets it right the first time; when it does not, the repair message is what makes structured output reliable. Two lines exist because the model is Qwen3, which thinks before it answers by default: without `enable_thinking: False` the reply can start with a think block whose braces confuse `extract_json`, or, under a tight `max_tokens`, come back with `content: null`. The `or ""` turns a null into an ordinary validation failure and a retry instead of a crash. If you want the reasoning, keep thinking on, raise `max_tokens` and strip the think block before parsing ([06 §6](06-data-format.md#6-thinking-and-reasoning_content)).

### 6.8 Precision and recall from predictions

```python
def precision_recall_f1(y_true, y_pred, positive=1):
    tp = sum(t == positive and p == positive for t, p in zip(y_true, y_pred))
    fp = sum(t != positive and p == positive for t, p in zip(y_true, y_pred))
    fn = sum(t == positive and p != positive for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, (tp, fp, fn)


# 50 tickets flagged urgent, 40 of them truly urgent; 60 urgent exist overall.
y_pred = [1] * 50 + [0] * 50            # 50 flagged urgent
y_true = [1] * 40 + [0] * 10 + [1] * 20 + [0] * 30
p, r, f1, (tp, fp, fn) = precision_recall_f1(y_true, y_pred)
print(f"TP={tp} FP={fp} FN={fn}")
print(f"precision={p:.2f} recall={r:.3f} f1={f1:.3f}")

# Cross-check against scikit-learn.
from sklearn.metrics import precision_score, recall_score
print("sklearn:", round(precision_score(y_true, y_pred), 2),
      round(recall_score(y_true, y_pred), 3))

assert (tp, fp, fn) == (40, 10, 20)
assert round(p, 2) == 0.80 and round(r, 3) == 0.667
print("tests: ok")
```

```text
TP=40 FP=10 FN=20
precision=0.80 recall=0.667 f1=0.727
sklearn: 0.8 0.667
tests: ok
```

---

## 7. An LLM system-design walkthrough with numbers

The interview favorite: **"design a support assistant for 2,000,000 tickets/month."** Work it in the order below; the numbers are what separate a hand-wave from a design.

### 7.1 Requirements

Functional: draft a reply grounded in the knowledge base, with citations; escalate to a human when unsure; never take an irreversible action. Non-functional: p95 TTFT under ~1.5 s (streamed); grounded answers; a cost budget; PII kept out of logs; an eval gate on releases.

### 7.2 Throughput

\[
\text{rps}_{\text{avg}} = \frac{2{,}000{,}000}{30 \cdot 24 \cdot 3600}
\]

!!! example "On paper"

    2,000,000 tickets over a 30-day month of 2,592,000 seconds:

    \[
    \begin{aligned}
    30 \cdot 24 \cdot 3600 &= 2{,}592{,}000\\
    \text{avg} &= 2{,}000{,}000 / 2{,}592{,}000 \approx 0.77\\
    \text{peak} &= 5 \times 0.77 \approx 3.9
    \end{aligned}
    \]

    About 0.77 requests/second on average, ~3.9/s at a 5x daytime peak. Size the service for the peak, not the average, and in tokens, not requests: at 300 output tokens per ticket the peak (3.86/s unrounded) is 3.86 × 300 ≈ 1,160 output tokens/s, and at 6.5 s per reply (§7.6) about 3.86 × 6.5 ≈ 25 replies are in flight at once. That is one GPU replica with continuous batching plus headroom if it sustains ~2,000 output tokens/s ([55](55-ai-production.md) uses that figure as its sizing assumption), or a hosted API. It is not a laptop: one 32 GB M1 Pro decodes a single Qwen3-14B stream at 15–17 tokens/s ([10 §8](10-export-serving.md#8-performance-expectations)).

### 7.3 Retrieval sizing

\[
\text{bytes} = N_{\text{chunks}} \times d \times \text{bytes/number}
\]

!!! example "On paper"

    200,000 chunks, each a 1024-dim float32 vector (4 bytes per number):

    \[
    \begin{aligned}
    200{,}000 \times 1024 \times 4 &= 819{,}200{,}000\\
    &\approx 819\ \text{MB}\\
    \text{int8 (1 byte)} &\approx 205\ \text{MB}
    \end{aligned}
    \]

    ~0.82 GB in float32, ~0.20 GB quantized to int8. The index fits in RAM on one box, so an in-process ANN index beats a distributed vector service at this size (answers the vector-DB question in §5.6).

### 7.4 Model choice

A mid-size instruct model (self-hosted on a GPU with a batching serving engine — prototype and evaluate it with the handbook's Qwen3-14B via `mlx_lm.server` — or a hosted equivalent) for the drafting, with retrieval doing the heavy lifting on facts. Reserve a larger model only for the hard fraction via a cascade (Project 10). Pick by eval on your golden set, not by leaderboard.

### 7.5 Cost budget

\[
\text{\$/ticket} = \frac{T_{\text{in}}}{10^6}\,c_{\text{in}} + \frac{T_{\text{out}}}{10^6}\,c_{\text{out}}
\]

!!! example "On paper"

    Assume \$3 per 1M input tokens and \$15 per 1M output tokens; a ticket uses 1500 input (prompt + retrieved chunks) and 300 output tokens:

    \[
    \begin{aligned}
    \text{input} &= 1500/10^6 \times 3 = 0.0045\\
    \text{output} &= 300/10^6 \times 15 = 0.0045\\
    \text{per ticket} &= 0.009\\
    \text{per month} &= 2{,}000{,}000 \times 0.009 = 18{,}000
    \end{aligned}
    \]

    About \$0.009 per ticket, ~\$18,000/month in model spend. The prices are assumptions — plug in your own. Note the split: input and output cost exactly the same here, because 5× more input tokens at one fifth of the price is a tie. So trimming retrieved chunks and capping reply length are equal levers; add more context and input takes over, write longer replies and output does.

### 7.6 Latency budget

\[
T_{\text{total}} = \text{TTFT} + \frac{N_{\text{out}}}{\text{tok/s}}
\]

!!! example "On paper"

    TTFT 0.5 s, then 300 output tokens at an assumed 50 tokens/second:

    \[
    \begin{aligned}
    300 / 50 &= 6.0\ \text{s}\\
    T_{\text{total}} &= 0.5 + 6.0 = 6.5\ \text{s}
    \end{aligned}
    \]

    6.5 s end to end — but if you stream, the agent sees the first token at 0.5 s and reads as it generates. Report TTFT and p95, not the total, and keep output short.

### 7.7 Evals, rollout, risks

Evals: a golden set with retrieval metrics (recall@k) and answer metrics (groundedness, abstention), run in CI (Project 9). Rollout: shadow, then a small canary, then ramp, with automatic rollback on a metric drop. Risks: prompt injection from ticket content and retrieved docs (treat both as untrusted, [56](56-ai-safety.md)); hallucinated policy claims (grounding + abstention); PII in logs (redact, [56](56-ai-safety.md) §8–9); cost blow-ups (per-key caps). Name the risks and your mitigation — that is the senior move.

---

## 8. The tool landscape, grouped by job

Learn the **job**, then pick a tool for it; tools churn, jobs do not. Neutral categories with a few well-known, stable examples. Do not memorize brands for an interview — explain what the category does.

| Job | What it does | Examples (non-exhaustive) |
|---|---|---|
| Numerics / classical ML | arrays, models, metrics | numpy, scikit-learn, PyTorch |
| Local serving | run models on your hardware, OpenAI-compatible | `mlx_lm.server`, llama.cpp, Ollama, vLLM ([10](10-export-serving.md)) |
| Prompt / API clients | talk to models, structured output | the `openai` client, pydantic |
| Agent frameworks | graphs, state, tool loops | **see [61 — Agent frameworks](61-ai-agent-frameworks.md)** for the review and trade-offs |
| Tool/data connectivity | standard tool & context interface | MCP ([58](58-ai-mcp.md)) |
| Retrieval / vector search | embed, index, nearest-neighbor | an embedded ANN index (e.g. FAISS) or a vector service ([51](51-ai-rag.md)) |
| Data preparation | clean, dedup, label, version | [60](60-ai-data-preparation.md) |
| Fine-tuning | PEFT/LoRA on your hardware | MLX-LM ([08](08-finetuning.md), [53](53-ai-fine-tuning.md)) |
| Evaluation | golden sets, judges, CI gates | your own harness ([54](54-ai-evaluation.md)) |
| Observability / gateway | traces, token usage, limits | a gateway you build (Project 11) |
| Safety / guardrails | injection defense, moderation, red-team | your layers ([56](56-ai-safety.md)) |

Agent frameworks specifically (LangGraph, LlamaIndex, CrewAI, the OpenAI Agents SDK, and others) are reviewed with strengths and weaknesses in [61 — Agent frameworks](61-ai-agent-frameworks.md); this chapter only tells you *when* to reach for one (§5.7).

---

## 9. How to read an ML paper

You cannot read every paper front to back. Use S. Keshav's **three-pass method** ([Keshav, 2016](https://www.read.seas.harvard.edu/cs161/2019/pdf/keshav16how.pdf)).

- **First pass (5–10 min).** Read the title, abstract, and introduction; the section headings; and the conclusions; glance at the references. Then answer the **five Cs**: Category, Context, Correctness (do the assumptions hold?), Contributions, Clarity. Decide whether to read on.
- **Second pass (up to an hour).** Read the body carefully but skip proofs; study the figures and tables — are axes labeled, are results shown with error bars? Mark references to follow. You should now be able to summarize the paper to someone else.
- **Third pass (hours).** Virtually re-implement it: make the authors' assumptions and re-derive the work, challenging every step. You should be able to reconstruct it from memory and name its hidden assumptions and weaknesses.

### 9.1 A worked first pass: "Attention Is All You Need"

Applying pass one to the Transformer paper ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) — stating only what the paper's own abstract claims:

- **Category.** A new model architecture (and machine-translation results).
- **Context.** It replaces the dominant recurrent/convolutional encoder–decoder models; the best of those already connected encoder and decoder through an attention mechanism.
- **Correctness (assumptions).** The claim is that attention alone, without recurrence or convolution, suffices for sequence transduction; the evidence is translation quality and training cost.
- **Contributions (from the abstract).** A network "based solely on attention mechanisms, dispensing with recurrence and convolutions entirely"; it is more parallelizable and faster to train; it reaches 28.4 BLEU on WMT 2014 English-to-German and a then-new single-model 41.8 BLEU on English-to-French after training 3.5 days on eight GPUs; and it generalizes to English constituency parsing.
- **Clarity.** A short, well-structured paper — the title itself states the thesis.

That is enough to decide this one earns a second and third pass (it is the basis of every model in this handbook — [32](32-mathematics.md), [37](37-architecture.md)). The habit matters more than the example: five Cs in ten minutes before you commit an hour.

---

## 10. Staying current without drowning

The field produces more than anyone can read. A sustainable routine:

- **Depth beats feed.** One paper read in three passes teaches more than fifty abstracts skimmed. Pick a few per month that matter to your work.
- **Follow primary sources, not hot takes.** Model cards, official docs, and the arXiv papers behind the headlines. A benchmark number without a method is noise.
- **Build to learn.** You remember what you implement. A 50-line reproduction beats a saved link.
- **Keep a lab notebook.** A running file of what you tried and the numbers you got; it becomes your write-ups (§4) and your interview stories.
- **Timebox it.** A fixed weekly slot for reading and one for building. The goal is a compounding skill, not an anxious scroll.

---

## 11. A self-assessment rubric

Grade yourself per skill. "Working" is roughly the bar for a first AI-engineering role; "Proficient" for a confident hire.

| Skill | Novice | Working | Proficient |
|---|---|---|---|
| Math & stats | recognizes the terms | computes cosine/softmax; reads a CI | derives gradients; designs an A/B test |
| Classical ML | knows the metrics | builds a baseline with a clean split | picks the right model; avoids leakage |
| Deep learning | knows the parts | trains an MLP; reads a loss curve | debugs training; reasons about memory |
| LLM internals | knows tokens exist | explains attention, decoding, KV cache | reasons about size/quantization/cost |
| Prompting & APIs | writes prompts | gets reliable structured output | versions and tests prompts |
| RAG | has done a tutorial | ships RAG with a retrieval eval | tunes chunking/hybrid/rerank by metric |
| Agents & MCP | has run an agent | builds a tool loop with budgets | designs for reliability; writes an MCP server |
| Fine-tuning | knows LoRA exists | runs a LoRA with a before/after eval | decides prompt vs RAG vs fine-tune |
| Evaluation | runs a model by hand | has a golden set in CI | uses stats; controls judge bias |
| Production | can call an API | adds limits, retries, tracing | models cost; runs canaries and SLOs |
| Safety | has heard of injection | treats output as untrusted; redacts PII | threat-models; runs a red-team suite |
| Data preparation | can load a CSV | dedups and splits without leakage | versions data; writes a data card |

Score honestly. The gaps are your next projects.

---

## 12. The "ready to apply" checklist

- [ ] Three to five portfolio projects, each with a README, an architecture note, and an **eval table with real numbers**.
- [ ] At least one project with a measured before/after (fine-tune, cascade, or added defense).
- [ ] A data-preparation project with a dedup report and a data card.
- [ ] An MCP server you built, tested with an in-memory client, and reviewed for security.
- [ ] A red-team suite and a threat model for one of your apps.
- [ ] You can do the §6 coding tasks from scratch and explain each line.
- [ ] You can run the §7 system-design walkthrough with your own numbers.
- [ ] You can answer every §5 question in two sentences, and map every line of a target job post (§2) to a project.
- [ ] A short written piece about one project (what failed, the numbers).
- [ ] Your self-assessment (§11) is at "Working" or above on the skills the role needs.

When most boxes are ticked, apply. The remaining gaps close faster inside a role than outside one.

---

## 13. Key takeaways

- Follow a plan with a weekly deliverable mapped to a chapter; the deliverables are your portfolio.
- Translate each line of a job post into a skill, a project that proves it, and a question you can answer.
- Build projects with objective acceptance criteria and honest READMEs; three to five polished ones are enough to apply (§12), and a data-prep project and an MCP server are strong evidence for the roles in §2.
- Interviews are "walk me through what you built" plus concept questions and a few coding tasks — practice all three, with numbers.
- For system design, compute throughput, index memory, cost and latency; the arithmetic is what makes it a design.
- Learn jobs, not brands; for agent frameworks, reach for one only when your flow needs it, and see [61](61-ai-agent-frameworks.md).
- Read papers in three passes, stay current with depth over feed, and grade yourself honestly against the rubric.

## 14. Exercises

!!! question "Exercise 14.1"

    Your target job lists "developing RAG architectures, vector databases, tool-calling systems, and autonomous AI agents." Which portfolio projects are your evidence, and which two §5 questions should you rehearse first?

??? success "Answer"

    Evidence: Project 4 (RAG + eval) for RAG architectures and vector databases, and Projects 5 and 6 (tool-using agent, MCP server) for tool-calling and agents. Rehearse 5.6 (RAG architecture and vector-DB choice) and 5.7 (agentic workflow vs autonomous agent; when to use a framework). See the mapping in §2.1.

!!! question "Exercise 14.2"

    Redo the support-assistant cost at 500 input and 400 output tokens per ticket, same assumed prices (\$3/\$15 per 1M). What is the per-ticket cost, and did input or output dominate?

??? success "Answer"

    Input = 500/1e6 × 3 = 0.0015; output = 400/1e6 × 15 = 0.006; per ticket = 0.0075, so about \$0.0075 and ~\$15,000/month. Output dominates here (0.006 vs 0.0015) because output tokens are priced 5× higher and the counts are closer. In §7.5 the two halves tied at 0.0045, because 5× more input tokens exactly offset the 5× price gap.

!!! question "Exercise 14.3"

    In the top-p task, what is the nucleus at p=0.7 for the distribution [0.5, 0.25, 0.15, 0.07, 0.03]?

??? success "Answer"

    Cumulative mass: 0.5, then 0.75. The smallest set reaching 0.7 is the first two tokens (`refund`, `return`), since 0.5 < 0.7 ≤ 0.75. So the nucleus is `['refund', 'return']` — tighter than at p=0.9, which also included `cancel`.

!!! question "Exercise 14.4"

    An interviewer asks: "Should this feature be an autonomous agent?" The task is: read a ticket, retrieve policy, draft a reply, and send it after a human clicks approve. What do you answer?

??? success "Answer"

    No — it is a fixed workflow (retrieve → draft → human approval → send), so build it as an explicit graph, not a free-running agent. The steps are known, which makes a workflow cheaper, more reliable and easier to evaluate; the human approval on the only side effect (send) is the right gate. Reserve autonomy for tasks whose path you genuinely cannot predetermine (§5.7).

!!! question "Exercise 14.5"

    You find that 8% of your train documents have a near-duplicate in the test set. Why is that a problem, and what do you do?

??? success "Answer"

    It is leakage: the model can memorize a train document and "recognize" its near-duplicate at test time, inflating your offline score; it will not transfer to production. Deduplicate across the split (shingling + MinHash/LSH or embedding similarity), remove the overlaps, and re-split. Report the near-duplicate count in the data card (§5.13, Project 2).

!!! question "Exercise 14.6"

    Do the first pass of Keshav's method on any paper you have not read, and write the five Cs in under ten minutes. What decision did the five Cs lead to?

??? success "Answer"

    Any honest answer that names Category, Context, Correctness, Contributions and Clarity, and ends in a keep/skip decision, is correct. The point is the habit: you decided whether the paper earns a second pass in ten minutes, instead of grinding through a paper that was never relevant (§9).

!!! question "Exercise 14.7"

    Your RAG eval shows recall@5 of 0.80 but groundedness of only 0.60. Which stage is the problem, and what do you try?

??? success "Answer"

    Retrieval is mostly fine (the right chunk is in the top 5 for 80% of questions), so most of the gap is generation: the model is not sticking to the chunks it was given. Try: a stricter grounding prompt that forbids outside knowledge, require citations and reject uncited claims, and add abstention when chunks are weak. Better retrieval (a reranker so the best chunk comes first) can only help on the 20% of questions where the chunk was missed, so it is the second lever, not the first.

!!! question "Exercise 14.8"

    You are asked to add an agent framework to a three-step fixed pipeline that already works in 120 lines of tested code. Argue for or against.

??? success "Answer"

    Against, by default: the flow is a short, fixed pipeline you can already read and test, so a framework adds a dependency and hidden control flow for little gain. Adopt one only if you will soon need its abstractions (a graph runtime, shared state/memory, built-in retries and tracing) and the team will maintain it. Decide against the actual flow, not the hype (§5.7, [61](61-ai-agent-frameworks.md)).

---

Next: [59 — AI engineering glossary](59-ai-glossary.md).
