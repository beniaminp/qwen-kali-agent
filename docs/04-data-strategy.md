# 04 — Data strategy: what the model must learn

*Last verified: 25 October 2026.*

A tool-calling model is not trained by showing it “cybersecurity text”. It is trained by showing it **decisions inside an environment**: which tool to call, with which arguments, in what order, what to do with the observation, and when to stop. This chapter defines the capability taxonomy, the token budget, the mix, the curriculum, and the quality gates.

The companion chapter [`05-datasets.md`](05-datasets.md) is the catalog of where to get the data. This chapter is the architecture of the mix.

---

## 1. The six capabilities

Every training example should serve at least one of these. If you cannot name which one, delete it.

| # | Capability | What it looks like | Failure if missing |
|---|---|---|---|
| 1 | **Tool selection** | “Given these 8 tools and this request, call `nmap_scan`, not `http_request`.” | Wrong tool, or a plausible but useless call. |
| 2 | **Argument construction** | Valid JSON, correct types, enums, required fields, target from scope. | Schema errors, scans against the wrong host. |
| 3 | **Sequencing** | recon → enumerate → verify → report; one call at a time or parallel where safe. | Random tool spam, repeated calls, no progress. |
| 4 | **Observation handling** | Parse stdout/stderr/exit code, notice “connection refused”, adapt. | Ignores errors, repeats the same failing command. |
| 5 | **Recovery** | Timeout → smaller scan; auth failure → check credentials/scope; no result → try a different service. | Gives up, hallucinates results, or escalates destructively. |
| 6 | **Scope and refusal** | “That target is outside the authorized scope; I need written authorization.” | Scans third parties, runs destructive commands, leaks data. |

A dataset that only teaches #1 and #2 produces a model that can call a tool but cannot run a task. A dataset that only teaches #6 produces a model that refuses everything. The mix must cover all six.

---

## 2. What “agentic” means here

The target loop is:

```text
system prompt + tools + scope
        │
        ▼
   model plans / decides
        │
        ├── no tool needed → answer
        └── tool call(s) ──► policy layer ──► lab target
                                  │
                                  ▼
                           observation (stdout/stderr/exit)
                                  │
                                  ▼
                         model reflects, repeats
```

Training data must mirror this loop, not a single-turn Q&A. Concretely, each trajectory should contain:

- a **system prompt** that states the role, the authorized scope, and the tool rules;
- a **user task** with a target, a goal, and constraints;
- zero or more **assistant tool calls** with realistic arguments;
- matching **tool responses** (real execution output, not invented);
- a **final assistant answer** that summarizes evidence and next steps;
- optionally, a **refusal or clarification** when the request is out of scope or ambiguous.

If your data has no `role: "tool"` messages, it is tool-calling data, not agentic data. Both are useful, but they teach different things.

---

## 3. Token budget, not row count

Rows are a bad unit because a row can be 200 tokens or 20,000. Plan in tokens.

| Stage | Training tokens | Purpose | Local wall-clock (14B @ 25 tok/s) |
|---|---:|---|---:|
| Format smoke test | 50K–200K | prove the template, masking, and tool JSON | 1–2 h |
| First adapter | 1–2M | learn the tool surface and domain vocabulary | 11–22 h |
| Serious adapter | 3–5M | add recovery, multi-turn, refusal behavior | 33–56 h |
| Scale-up | 10M+ | only if eval shows underfitting | 4–5 days |

**Build a larger pool than you train on.** A 45–90M-token pool lets you sample different mixes and re-run after an evaluation failure. You do not need to train on all of it. Sampling 2–5M high-quality tokens from a 50M pool is normal and desirable.

**The 90M-token reference mix** in [`05-datasets.md`](05-datasets.md) is a pool design, not a single local run. Training it end-to-end on this Mac would take weeks. Use it for cloud runs or multi-stage local experiments.

---

## 4. The mix

Start with this target, then adjust based on the evaluation failure taxonomy.

| Bucket | Share | What it buys | Candidate sources |
|---|---:|---|---|
| **General tool calling** | 35–40% | schema adherence, argument JSON, parallel calls, tool choice | xLAM, ToolACE, Hermes, Glaive (filtered), Toucan |
| **Multi-turn / agentic** | 15–20% | observe → act, error recovery, long horizons | ToolACE, SWE-smith, Nemotron Terminal, OpenThoughts, APIGen-MT (internal only) |
| **Cyber QA + Kali CLI** | 20% | vocabulary, flags, procedures, reporting | KaliBench train, Trendyol, filtered cyber QA |
| **Synthetic verified trajectories** | 15% | *your* tools, *your* scope policy, *your* lab | generated with DeepSeek/OpenCode, executed in the sandbox |
| **Refusal / scope / no-call** | 10% | authorization checks, ask vs act, abstain | When2Call, curated scope dialogues, negative examples |

### Why these proportions?

- **General tool calling is the grammar.** If the model cannot emit valid JSON or choose among tools, domain knowledge is irrelevant. It gets the largest share because it is the prerequisite.
- **Agentic data is the behavior.** A model trained only on single-turn calls will answer “run nmap” with a call and then stop. Multi-turn trajectories teach it to look at the result and continue.
- **Cyber QA is the vocabulary.** Kali has 1,600+ tools and thousands of flags. A general model knows some; targeted data fixes the long tail.
- **Synthetic trajectories are the alignment.** Public data does not know your tool names, your scope file, or your lab topology. Generate that.
- **Refusal is a capability, not a tax.** A pentest agent that scans an out-of-scope host is a liability. Train the refusal explicitly, and evaluate it separately.

### The minimal viable mix (~2–5M tokens)

| Bucket | Rows (approx.) | Tokens (approx.) |
|---|---:|---:|
| General tool calling | 8–15k | 1.2–2M |
| Agentic | 1–3k | 0.5–1M |
| Cyber QA + Kali CLI | 5–10k | 0.5–1M |
| Synthetic trajectories | 1–3k | 0.5–1M |
| Refusal/scope | 2–4k | 0.3–0.6M |

Do not obsess over exact percentages at this scale. Cover every bucket, keep the quality high, and iterate.

---

## 5. Curriculum

A strict multi-stage curriculum is hard to tune at 2–5M tokens. Use a **mild easy→hard ramp with a refusal-heavy anneal**:

1. **Warm-up (10%).** Short, single-turn examples with simple schemas: KaliBench easy rows, Hermes single-turn, a few xLAM rows. Goal: learn `<tool_call>` syntax and JSON validity.
2. **Bulk (60%).** Shuffled multi-turn tool+observation data, terminal trajectories, cyber QA. Shuffle **by bucket**, not by row, so each batch sees a mix of formats. This prevents catastrophic forgetting of one format.
3. **Cyber-heavy (20%).** KaliBench ×2–3, Trendyol, synthetic Kali trajectories, converted explanations. Goal: domain depth.
4. **Anneal (10%).** Refusal/scope/no-call examples at a lower learning rate, plus 5% general-tool replay. Goal: end on the behavior you want, without making the model refuse everything.

Sprinkle ~30% of the refusal examples through stages 1–3 as well. If all refusals are at the end, the model may overfit to “refuse” as a global style.

**Do not** train a “cyber-only” model and then bolt on tool calling. The tool format is the fragile part; it needs constant rehearsal.

---

## 6. Licensing strategy

Decide **before** you download whether the adapter will be published.

| Intended release | Allowed data | Avoid |
|---|---|---|
| **Private / internal only** | Almost anything you have lawful access to; still respect terms | Leaked data, live credentials, unlicensed writeups |
| **Public adapter (Apache/MIT)** | Apache-2.0, MIT, CC-BY-4.0 (with attribution) | CC-BY-NC, CC-BY-NC-SA, GPL, “unknown” |
| **Commercial product** | Apache-2.0, MIT, CC-BY-4.0, commercially licensed | NC, SA, GPL, unlicensed scrapes |

The big traps:

- **APIGen-MT-5k is CC-BY-NC-4.0.** It is one of the best multi-turn sets, and it cannot go into a published/commercial model without a separate license. Keep it in an internal ablation.
- **NYU CTF Bench is GPL-2.0.** Treat it as evaluation-only; do not redistribute derived data.
- **SecEval's mirror is CC-BY-NC-SA-4.0.** The share-alike clause compounds.
- **Mirrors relicense xLAM as MIT/Apache.** That does not override Salesforce's CC-BY-4.0. Trace to the origin.
- **“Unknown” means all rights reserved.** Do not publish a model trained on it.

Record the license of every source in the dataset manifest. If you cannot identify the license, do not use the source in a release mix.

---

## 7. Decontamination

Public benchmarks are contaminated by public training data. If you train on BFCL, KaliBench test, Cybench, NYU CTF or CyberMetric, your evaluation is meaningless.

**Minimum procedure:**

1. **Hold out the benchmark files physically.** Keep them in `data/benchmarks/`, never in `data/processed/`.
2. **13-gram overlap.** Normalize whitespace/case, tokenize into 13-grams, and remove any training example with ≥1 overlapping 13-gram with a benchmark item. This is conservative but catches verbatim leakage.
3. **Near-duplicate check.** MinHash/SimHash within the training set (threshold ~0.8 Jaccard) and against the benchmark.
4. **Canary strings.** Insert a unique canary into each benchmark item and check the model never emits it. Useful for detecting leakage in a published checkpoint.
5. **Report the method and the overlap rate.** “0.7% of training rows removed by 13-gram overlap” is a result; silence is not.

KaliBench's public train/test splits have **zero exact query overlap**, but both are public. A model trained after release may still have seen the test split through other corpora. Use it as a **reported benchmark**, not as your only holdout.

---

## 8. Splits and the golden set

| Split | Purpose | Size | Rule |
|---|---|---:|---|
| `train.jsonl` | optimization | 90% of the mix | may overlap public training sources |
| `valid.jsonl` | early stopping / model selection | 5% | same distribution as train, no exact duplicates |
| `test.jsonl` | final internal check | 5% | held out before any tuning; touch once |
| `golden.jsonl` | behavioral evaluation | 200–300 tasks | hand-written or lab-generated; **never** in train |
| `benchmarks/` | external comparison | official sets | never in train |

The golden set is the most valuable asset in the project. Write it **before** training, from real lab tasks, with expected tool names/arguments and pass criteria. If you write it after seeing the model's outputs, you will unconsciously make it easy.

A good golden set covers:

- 60 single-tool tasks (selection + arguments),
- 60 multi-step tasks (2–6 calls, one recovery),
- 40 refusal/scope tasks (out-of-scope target, ambiguous authorization, destructive request),
- 20 no-call tasks (answer from knowledge, no tool needed),
- 20 malformed-input tasks (bad JSON, missing target, tool error),
- 20 reporting tasks (summarize evidence, cite the command/output).

---

## 9. Dataset manifest and card

Every dataset version gets a manifest:

```json
{
  "version": "2026-10-25.v1",
  "created": "2026-10-25T12:00:00Z",
  "sources": [
    {"id": "Team-ACE/ToolACE", "revision": "abc123", "rows": 11300, "license": "Apache-2.0", "hash": "sha256:..."},
    {"id": "RISys-Lab/KaliBench", "revision": "def456", "rows": 3504, "license": "CC-BY-4.0", "hash": "sha256:..."}
  ],
  "counts": {"train": 48000, "valid": 2500, "test": 2500},
  "tokens": {"train": 4200000, "valid": 210000, "test": 210000},
  "decontamination": {"method": "13-gram", "removed": 312},
  "known_issues": ["APIGen-MT excluded from release mix (CC-BY-NC)"]
}
```

And a dataset card covering: intended use, out-of-scope use, sources/licenses, collection method, preprocessing, personal/sensitive data, biases, evaluation, maintenance. The `scripts/make_dataset_card.py` helper generates the skeleton.

---

## 10. Common mistakes

| Mistake | Why it hurts | Fix |
|---|---|---|
| Training on 500k rows because “more data is better” | Weeks of compute, diluted quality, overfitting to noise | Start with 1–2M curated tokens; measure. |
| Mixing tool-call syntaxes in one run | Model learns neither format reliably | Normalize everything to the target model's template. |
| Using the test split for early stopping | Inflates results | Three splits + a separate golden set. |
| No refusal examples | Agent scans anything it is asked to | 10% refusal/scope/no-call, plus a policy layer. |
| Synthetic data without execution | Model learns plausible but wrong command output | Execute in a sandbox; keep only verified trajectories. |
| Truncating mid-tool-call | Teaches malformed JSON | Split per assistant turn; filter by token length. |
| Ignoring licenses until release day | Cannot publish the adapter | License audit before download. |
| No dataset version | Cannot reproduce a good run | Manifest + git tag + hash. |

---

## 11. Sources

- LIMA: Less Is More for Alignment — <https://arxiv.org/abs/2305.11206>
- ToolACE — <https://arxiv.org/abs/2409.00920>
- APIGen — <https://arxiv.org/abs/2406.18518>
- When2Call — <https://arxiv.org/abs/2504.18851>
- APIGen-MT — <https://arxiv.org/abs/2504.03601>
- Toucan — <https://arxiv.org/abs/2510.01179>
- SWE-smith — <https://arxiv.org/abs/2504.21798>
- KaliBench — <https://arxiv.org/abs/2610.02206>
- BFCL — <https://gorilla.cs.berkeley.edu/leaderboard.html>
- Dataset licensing details — [`05-datasets.md`](05-datasets.md) §4
