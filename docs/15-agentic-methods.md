# 15 — Beyond SFT: preference optimization and agentic RL

*Last verified: 25 October 2026. This chapter is a map of the methods, not a promise that they fit 32 GB.*

Supervised fine-tuning (SFT) on trajectories is the right first step. It teaches the format, the tool surface, and the basic policy. It does not directly optimize for **task success**, and it cannot distinguish a trajectory that merely looks plausible from one that actually works.

This chapter explains the methods that come after SFT, what they need, and which ones are realistic on an M1 Pro.

---

## 1. The ladder

| Method | Signal | Data | Compute | Fits 32 GB? |
|---|---|---|---|---|
| **SFT** | next-token loss on chosen trajectories | 1–5M tokens | low | **yes** |
| **Rejection sampling (RFT)** | keep only successful samples | generator + verifier | medium | yes (generation is the cost) |
| **DPO / ORPO** | chosen vs rejected pairs | 5–50k pairs | low–medium | 14B QLoRA yes |
| **GRPO / PPO** | scalar reward from execution | environment + rollouts | high | not for 14B locally |
| **Process reward** | step-level labels | expensive annotation | medium | research |
| **Distillation** | teacher logits/trajectories | teacher access | low | yes |

The practical order is **SFT → rejection sampling → DPO → (cloud) RL**.

---

## 2. SFT: what it can and cannot do

SFT maximizes the likelihood of the demonstrated action at each step. It is excellent for:

- tool-call syntax and schema adherence;
- domain vocabulary and CLI flags;
- imitation of a competent trajectory;
- refusal/scope style.

It is weak at:

- choosing between two plausible actions when only one succeeds;
- recovering from an error it has not seen;
- optimizing long-horizon success (one bad early call ruins the task);
- calibrating confidence.

A model trained only on successful trajectories often learns “what a good answer looks like” rather than “what to do when the tool fails”.

---

## 3. Rejection sampling fine-tuning (RFT)

**Idea:** sample N trajectories per task from the current model (or a teacher), execute them, keep the successful ones, and SFT on those. Repeat.

```text
task set
  │
  ├── sample k trajectories (temperature 0.7–1.0)
  │
  ├── execute in the sandbox
  │
  ├── verifier: success? in scope? no destructive action?
  │
  └── keep successes → SFT → next round
```

**Strengths:** simple, uses the same SFT infrastructure, directly improves success rate, works with a verifiable sandbox.

**Weaknesses:** needs many rollouts (cost), can collapse diversity, and the verifier must be trustworthy. If the verifier is an LLM judge, it can be gamed.

**On 32 GB:** generation is the bottleneck, not training. A 14B at 15–20 tok/s can produce a few hundred short trajectories per day; a 27B oracle is slower. Use a smaller model for rollouts and a larger one for judging.

---

## 4. DPO and ORPO

**DPO** (Direct Preference Optimization) trains on pairs `(prompt, chosen, rejected)` and increases the margin between them without a separate reward model. **ORPO** combines SFT and preference optimization in one objective.

Data sources:

- successful vs failed trajectories from the sandbox;
- correct vs incorrect tool call (same task, one valid JSON, one malformed);
- in-scope vs out-of-scope response;
- concise evidence-based answer vs invented output.

A pair looks like:

```json
{"prompt": [...], "chosen": [...], "rejected": [...]}
```

**Strengths:** cheap compared with RL; directly targets preferences; can fix refusal/verbosity/format issues.

**Weaknesses:** pair quality dominates; DPO can over-optimize and reduce diversity; it does not teach new capabilities, only re-weights existing ones.

**On 32 GB:** a 14B QLoRA DPO run is feasible (similar memory to SFT, plus a reference model or adapter). MLX-LM does not ship a DPO trainer as of Oct 2026; use PEFT/TRL on a rented GPU, or implement a small preference loss on top of MLX. Treat this as an advanced path.

---

## 5. RL with verifiable rewards (GRPO/PPO)

**Idea:** let the model roll out trajectories, score them with a reward function (task success, scope compliance, efficiency, evidence quality), and update the policy.

GRPO (Group Relative Policy Optimization) is popular for reasoning/tool-use because it avoids a value model: sample a group of outputs, normalize rewards within the group, and update.

**What you need:**

- a fast, deterministic environment (the lab);
- a reward function that cannot be gamed;
- many rollouts per task;
- a training framework that supports RL (TRL, verl, OpenRLHF, etc.);
- substantially more compute than SFT.

### Reward design for multi-turn tool use

A single outcome reward is a weak signal when a task takes 8–12 turns. A practical 2026 recipe combines:

| Term | Suggested weight | Purpose |
|---|---:|---|
| Final outcome | +1.0 / −1.0 | Anchors the agent to the real goal. |
| Per-turn progress | small positive | Rewards useful intermediate state changes. |
| Step penalty | −0.05 per step | Discourages stalling and infinite loops. |
| Invalid tool call | −0.2 per error | Penalizes malformed names/arguments. |
| Scope/safety violation | large negative / terminate | Must dominate any task reward. |

The dominant term must remain the final outcome; otherwise the model learns to farm intermediate rewards. For a local 14B, use this table to design the **verifier** and the evaluation harness even if you never run RL — it clarifies what “good” means for SFT data.

A caveat from the LoRA study: RL needs very little adapter capacity (even rank 1 can match full fine-tuning), but it needs many rollouts. On 32 GB, the environment and generation cost dominate, not the adapter. Rent a GPU for RL; keep the local Mac for SFT and evaluation.

**On 32 GB:** not realistic for a 14B. Even 4-bit inference plus rollouts plus training does not fit, and the wall clock is prohibitive. Rent an L40S/A100 or use a smaller model (1–4B) for RL research.

**Safety:** an RL agent will find reward hacks. If the reward is “open ports found”, it may scan out of scope. Put the policy layer and scope check **inside the reward**, and make violations a hard negative, not a small penalty.

---

## 6. Process supervision

Instead of scoring only the final answer, score each step:

- Was the tool choice correct?
- Were the arguments valid and in scope?
- Was the observation interpreted correctly?
- Was the next step justified?

Process labels are expensive. A practical compromise:

- use execution success as the outcome reward;
- use a rubric LLM to score intermediate steps;
- keep a small human-labeled set to calibrate the rubric.

For a 14B SFT pipeline, process labels are most useful as **data quality filters**, not as an RL signal.

---

## 7. Distillation

A larger teacher (Qwen3.8-27B, DeepSeek V4-Pro, a frontier API) generates:

- plans and reflections;
- hard negative examples;
- rationales for tool choice;
- judge scores.

The student trains on the teacher's **trajectories**, not necessarily its logits. This is “trajectory distillation” and is compatible with SFT/QLoRA.

Risks:

- the student inherits the teacher's hallucinations;
- the teacher's tool schema differs from yours;
- the teacher's safety policy is not yours;
- license/terms may restrict distillation.

Always execute/validate teacher output before training on it.

---

## 8. Curriculum and environment design

The biggest gains often come from the environment, not the algorithm:

- **Task difficulty ramp:** single call → 2–3 calls → 5+ calls with a failure.
- **Tool surface:** 6–10 typed tools, clear descriptions, enums.
- **Observation realism:** real stdout/stderr, timeouts, partial failures.
- **Scope variation:** in-scope, ambiguous, out-of-scope, destructive.
- **Target variation:** Linux host, web app, SMB, TLS, DNS.
- **Adversarial cases:** tool returns garbage, target is down, credentials expire.
- **Reward/verifier:** execution success + policy compliance + evidence.

A model trained on a narrow environment will fail the moment the tool output changes. Randomize ports, banners, latency, and error messages in the lab.

### 8.1 Front-end agent RL

The 2026 **WebCoder** work applies artifact-grounded agentic RL to front-end development: it grades the running application, distributes credit across multi-round requirements, compacts context, and trains on long trajectories. On **Webdev-Bench**, WebCoder-27B reports a requirement-satisfaction score of 72.8 versus 64.8 for its Qwen3.8-27B base and 65.1 for Claude-Opus-4.8.

That is a 27B model trained with RL on web artifacts, not a 14B QLoRA on a 32 GB Mac. Use it as a roadmap:

- local: SFT on verified React/Tailwind tasks, then rejection sampling with Playwright/axe;
- rented GPU: DPO/GRPO on Webdev-Bench-style milestones;
- always: grade the **running app**, not the code text.

The benchmark and task format are in [23 — React datasets](23-react-datasets.md); the verifier design is in [24 — React fine-tuning](24-react-finetuning.md) §8 and [25 — React synthetic data](25-react-synthetic.md).

---

## 9. A realistic roadmap

| Stage | What | Compute | Gate |
|---|---|---|---|
| 1 | SFT on 1–2M curated tokens | local 8B/14B | TVR >0.95, TCR >0.40 |
| 2 | Add recovery/scope data; SFT again | local | TCR >0.55, UAR 0 |
| 3 | Rejection sampling on 200–500 tasks | local generation | TCR +5–10 points |
| 4 | DPO on success/failure pairs | rented GPU | preference accuracy >0.75 |
| 5 | GRPO on a verifiable lab | rented GPU | TCR >0.75, UAR 0 |
| 6 | Distill the best policy into a 14B | local/cloud | close the gap |

Do not skip stage 1. RL on a model that cannot emit valid JSON is wasted compute.

---

## 10. What to measure for each method

- **SFT:** loss, TVR, TCR, UAR.
- **RFT:** success rate before/after, diversity (unique trajectories), pass@k.
- **DPO:** preference accuracy on a held-out pair set, TCR, verbosity, refusal precision.
- **GRPO:** reward curve, TCR, UAR, reward-hack audit (manual).
- **Distillation:** student vs teacher on the same golden set, cost/latency.

Always include a **safety eval**. Preference/RL methods can increase capability and decrease caution at the same time.

---

## 11. Sources

- InstructGPT / RLHF — <https://arxiv.org/abs/2203.02155>
- DPO — <https://arxiv.org/abs/2305.18290>
- ORPO — <https://arxiv.org/abs/2403.07691>
- GRPO (DeepSeekMath) — <https://arxiv.org/abs/2402.03300>
- Rejection sampling fine-tuning — <https://arxiv.org/abs/2308.01825>
- Process supervision — <https://arxiv.org/abs/2305.20050>
- ToolRL / tool-use RL — search current 2025–2026 work; the field is moving.
- Multi-turn RL reward design — <https://fireworks.ai/blog/best-practices-for-multi-turn-RL>
- Multi-turn GRPO experiment — <https://builder.aws.com/content/30atJ2tkb88UPTQXFOu7OoxpCDS/fine-tuning-an-llm-to-improve-complicated-agentic-tool-calling-workflows>
- WebCoder (front-end agentic RL) — <https://arxiv.org/abs/2610.23104>
- TRL — <https://github.com/huggingface/trl>
- verl — <https://github.com/volcengine/verl>
