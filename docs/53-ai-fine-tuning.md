# 53 — Fine-tuning and model adaptation

*Last verified: 28 October 2026.*

This chapter is about the fourth rung of the adaptation ladder: changing the model's weights instead of changing its prompt. You will learn when fine-tuning is the right tool and when it is a waste of two days, the math of LoRA and QLoRA (counted by hand so you can predict memory before you launch), how to prepare data and mask the loss, how to read the hyperparameters and the loss curve, how to detect overfitting and catastrophic forgetting, and how to evaluate the result against the base. There is a tiny LoRA that runs on your CPU in a few seconds, a DPO loss computed with numbers, and a cost model for a local run versus a rented GPU. Read [50 — Prompting and APIs](50-ai-prompting-apis.md) and [51 — RAG](51-ai-rag.md) first; they are the rungs below this one, and most problems are solved there. The runbook for this handbook's actual training is [08 — Fine-tuning with MLX](08-finetuning.md) and [17 — Fine-tuning update](17-finetune-2026-update.md); the math behind the model is in [32 — Mathematics](32-mathematics.md) and [37 — Architecture anatomy](37-architecture.md).

---

## 1. The adaptation ladder

### 1.1 Six rungs, cheapest first

You rarely need to fine-tune. There is a ladder of ways to make a model do what you want, and each rung costs more than the one below it. Climb only as far as the problem forces you.

| Rung | What you change | Cost to try | Changes the weights? |
|---|---|---|---|
| 1. Prompting | the instruction | minutes | no |
| 2. Few-shot | examples in the prompt | minutes | no |
| 3. RAG | retrieved context | days | no |
| 4. Fine-tuning (PEFT/LoRA) | a small adapter | hours–days | yes (adapter) |
| 5. Continued pretraining | all weights, on raw text | weeks, a cluster | yes (all) |
| 6. Training from scratch | all weights, from zero | months, millions of dollars | yes (all) |

- **Prompting** is editing the system/user message. [50](50-ai-prompting-apis.md) is the whole rung.
- **Few-shot** is prompting with worked examples (`input → output` pairs) so the model copies the pattern. Still just a prompt.
- **RAG** retrieves documents at request time and pastes them into the prompt. It is how you add *facts* and *freshness*. [51](51-ai-rag.md) is the rung.
- **Fine-tuning** adjusts weights on your examples. With LoRA you adjust a small adapter, not the whole model. This chapter.
- **Continued pretraining** keeps training the base model on a large corpus of raw text (e.g. all of your company's code) to shift its *distribution*, not just its behaviour on one task. It needs a GPU cluster and is out of scope for a 32 GB laptop.
- **Training from scratch** is [35 — From zero](35-from-zero.md). You almost never do this.

!!! tip

    The ladder is a cost gradient, not a quality gradient. A good prompt beats a bad fine-tune every time. Fine-tuning does not *replace* a good prompt; it lets you *shorten* one and make a behaviour reliable.

### 1.2 A decision procedure

Ask these in order and stop at the first "yes that's the problem":

```text
1. Is the output format/style wrong, or the instruction ignored?
      → Fix the prompt first (rung 1–2). Add 2–5 examples.

2. Does the model lack a FACT, or need fresh/private data?
      → RAG (rung 3). Fine-tuning does NOT reliably add facts.

3. Is the behaviour right but UNRELIABLE, or the prompt huge
   and slow, or do you need a smaller/cheaper model to match
   a big one on ONE narrow task?
      → Fine-tune (rung 4). You have a repeatable skill to bake in.

4. Does the model lack a broad CAPABILITY (a language, a domain
   it has never seen) across many tasks?
      → Continued pretraining (rung 5). Rare; needs a cluster.

5. Do you have a novel architecture or a cluster and a reason?
      → Train from scratch (rung 6). Almost never.
```

Before fine-tuning, you must be able to answer: *what prompt am I trying to replace, and on what eval does the fine-tune have to beat it?* If you cannot write that eval, you are not ready (see [§13](#13-evaluating-a-fine-tune) and [54](54-ai-evaluation.md)).

### 1.3 Three worked product cases

**Case A — the support-ticket classifier.** You must tag incoming tickets as `billing`, `bug`, `feature`, or `other`. A prompt with the four definitions and three examples gets 88% accuracy but the prompt is 600 tokens and you classify 200k tickets a day.

- Rung 1–2 solves the *accuracy*. The problem is *cost and latency*: 600 tokens of prompt on every one of 200k calls.
- Fine-tune a small model on 5k labelled tickets so the prompt becomes `Classify: <ticket>`. The baked-in task lets you drop to a 20-token prompt and a smaller model. **Verdict: fine-tune, for cost and latency, not capability.**

**Case B — the "answer from our docs" bot.** Users ask about your product; the model must answer from your current documentation and cite it.

- The model does not *know* your docs, and they change weekly. Fine-tuning would memorise a snapshot and still hallucinate the parts it did not memorise.
- **Verdict: RAG, not fine-tune.** Retrieve the relevant doc chunk and put it in the prompt. Fine-tune only if the *format* of the answers (citation style, refusal when unsupported) is unreliable — and even then, fine-tune the format on top of RAG, do not replace it.

**Case C — the Kali tool-call agent (this handbook).** The model must emit valid tool calls against your schema, stay in scope, and recover from tool errors.

- The behaviour is narrow, repeatable, and has a hard-to-prompt syntax (the exact JSON tool-call format). Few-shot helps but is not reliable enough at temperature 0 across 240 golden tasks.
- **Verdict: fine-tune** (SFT on trajectories; [08](08-finetuning.md)), then maybe preference tuning ([§10](#10-preference-tuning-dpo)) to fix the "what to do when the tool fails" cases that SFT cannot teach. This is the whole reason the handbook exists.

!!! note "Software-engineering analogy"

    The ladder is the same instinct as the performance ladder: tune the query before you add a cache, add a cache before you shard the database, shard before you rewrite in a faster language. Each rung is cheaper to try and easier to undo than the one above it. Fine-tuning is a database migration: powerful, slow to run, and painful to roll back — so you reach for it only when the cheaper moves are exhausted.

---

## 2. What fine-tuning changes — and what it does not

### 2.1 Reliably changes

Fine-tuning (even a small LoRA on a few thousand examples) reliably changes **behaviour the base model already has the parts for**:

- **Output format.** JSON shape, Markdown vs plain, always-cite, always-end-with-a-summary.
- **Style and tone.** Terse, formal, your house voice; refusal language and scope.
- **Tool-call syntax.** The exact function-call format your server parses ([06](06-data-format.md)).
- **Narrow skills.** One classification task, one extraction schema, one translation direction — baked in so the prompt shrinks.
- **Latency and cost, indirectly.** By making a *smaller* model reliable on your task, so you can retire a bigger one or a long prompt.

### 2.2 Does not reliably change

Fine-tuning on a modest dataset does **not** reliably:

- **Add facts.** Teaching "our API key rotation policy is 90 days" by putting it in 50 training examples is unreliable: the model may learn it, may half-learn it (right policy, wrong number), or may forget it under a slightly different question. Facts that must be *correct* belong in RAG or the prompt, where they are looked up, not recalled. (A large continued-pretraining run *can* add knowledge; a LoRA on a few thousand rows cannot be trusted to.)
- **Add a broad new capability.** You cannot LoRA a model into a language it never saw, or into solving a class of problems it cannot do at all. LoRA *re-weights and sharpens* existing ability; it does not create it (this is the [15](15-agentic-methods.md) point: SFT teaches "what a good answer looks like", not new reasoning).
- **Fix a bad base.** If the base model is wrong about something in every sample, fine-tuning on 1k examples teaches it to be wrong confidently.

!!! warning

    "The model keeps getting the price wrong, I'll fine-tune the price in" is the single most common fine-tuning mistake. The model will sometimes emit the trained price and sometimes a plausible neighbour, with no warning which. Prices, policies, inventory, anything that must be *exactly* right and *changes* goes in the prompt via retrieval, not in the weights.

---

## 3. Full fine-tuning vs parameter-efficient methods

### 3.1 Full fine-tuning

Full fine-tuning updates **every** weight. For Qwen3-14B that is 14.77B parameters, and with AdamW you also store, per parameter, an fp32 master copy and two moment estimates. That is 12 bytes per parameter of optimizer state on top of the weights:

\[
M_{\text{opt}} = 12 \times 14.77\times10^{9}\ \text{bytes}
\]

!!! example "On paper"

    AdamW keeps, per parameter, 4 bytes of fp32 master weight plus 4+4 bytes of first and second moment:

    \[
    12 \times 14.77\times10^{9} = 1.77\times10^{11}
    \]

    That is **177 GB** of optimizer state alone, on top of ~29.5 GB of bf16 weights and another ~29.5 GB of bf16 gradients: about 236 GB before activations. It does not fit a 32 GB laptop or a single 80 GB datacenter GPU. This is why full fine-tuning of a 14B model needs a multi-GPU node.

Full fine-tuning is the most flexible and the most dangerous: with that much capacity on a small dataset you overfit and forget fast (see [§8](#8-overfitting-and-catastrophic-forgetting)).

### 3.2 Parameter-efficient fine-tuning (PEFT)

PEFT freezes the base weights and trains a small number of new parameters. The dominant method is **LoRA** (and its 4-bit cousin **QLoRA**). Others you will meet: prefix/prompt tuning (train a few virtual tokens), adapters (small bottleneck MLPs between layers), and DoRA (LoRA split into magnitude and direction; [08 §9](08-finetuning.md)). LoRA is the default because it adds nothing at inference once merged and gives most of full fine-tuning's quality on post-training tasks ([17 §2](17-finetune-2026-update.md)).

| | Full | LoRA | QLoRA |
|---|---|---|---|
| Base weights | trained, fp32/bf16 | frozen, bf16 | frozen, **4-bit** |
| Trained params (14B) | 14.77B | ~64M (r=16, all layers) | ~64M |
| Optimizer state | ~177 GB | tiny (adapter only) | tiny |
| Fits 32 GB? | no | no (29.5 GB bf16 base) | **yes** |
| Quality ceiling | highest | ~full, post-training | ~LoRA |

---

## 4. LoRA: the low-rank update

### 4.1 The idea

A weight matrix \(W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}\) is frozen. LoRA adds a trainable update that is forced to be low rank: two thin matrices \(B\in\mathbb{R}^{d_{\text{out}}\times r}\) and \(A\in\mathbb{R}^{r\times d_{\text{in}}}\), with \(r\ll d\). The layer computes

\[
y = Wx + \tfrac{\alpha}{r}\,B A\,x,
\]

where \(\alpha\) is a scaling constant. \(A\) is initialised random and \(B\) is initialised to **zero**, so at step 0 the update is exactly \(0\) and the model behaves like the base. Only \(A\) and \(B\) are trained; \(W\) never moves.

!!! example "On paper"

    One token \(x=(1,0,1)\) through a frozen \(2\times3\) layer, LoRA rank \(r=1\), \(\alpha=2\), \(A=(1,0,1)\), \(B=(1,2)^\top\). First the base, with \(W\) the all-ones matrix:

    \[
    Wx = (1{+}0{+}1,\;1{+}0{+}1) = (2,2)
    \]

    Now the update. \(Ax = 1\cdot1+0\cdot0+1\cdot1 = 2\):

    \[
    \tfrac{\alpha}{r}BA x = \tfrac{2}{1}(1,2)\cdot 2 = (4,8)
    \]

    \[
    y = (2,2) + (4,8) = (6,10)
    \]

    At step 0, \(B=0\), so the second term is \((0,0)\) and \(y=Wx=(2,2)\): the adapter starts invisible and only the trained \(A,B\) move it.

MLX writes the same thing as `y = Wx + scale · BA x` with `scale = α / r`. This is the trap from [17 §3](17-finetune-2026-update.md): MLX's default `scale: 20` is *not* PEFT's `α`, so you must set `scale = 32/r` to match a standard `α=32` recipe.

### 4.2 Parameter count for one projection

Take one square projection of Qwen3-14B, say `o_proj`, which is \(5120\times5120\) ([37 §1.1](37-architecture.md)). Full fine-tuning trains every entry. LoRA at \(r=16\) trains only \(A\) (\(16\times5120\)) and \(B\) (\(5120\times16\)).

\[
\text{full} = 5120\times5120 = 26{,}214{,}400
\]

!!! example "On paper"

    LoRA adds \(r(d_{\text{in}}+d_{\text{out}})\) parameters:

    \[
    16\times(5120+5120) = 163{,}840
    \]

    \[
    \frac{163{,}840}{26{,}214{,}400} = 0.00625
    \]

    The adapter is **0.625%** of the full projection — about 1 parameter trained for every 160. That ratio is why the optimizer state is tiny and why LoRA fits where full fine-tuning does not.

### 4.3 Parameter count for the whole Qwen3-14B adapter

An all-layer adapter puts LoRA on all seven projections in all 40 blocks. The shapes (from [37 §1.1](37-architecture.md): \(d=5120\), \(H d_h=5120\), \(G d_h=1024\), \(d_{\text{ff}}=17408\)) are:

| Projection | \(d_{\text{in}}\) | \(d_{\text{out}}\) | LoRA params at \(r=16\) |
|---|---:|---:|---:|
| `q_proj` | 5120 | 5120 | 163,840 |
| `k_proj` | 5120 | 1024 | 98,304 |
| `v_proj` | 5120 | 1024 | 98,304 |
| `o_proj` | 5120 | 5120 | 163,840 |
| `gate_proj` | 5120 | 17408 | 360,448 |
| `up_proj` | 5120 | 17408 | 360,448 |
| `down_proj` | 17408 | 5120 | 360,448 |
| **per block** | | | **1,605,632** |

!!! example "On paper"

    Sum one block, \(r(d_{\text{in}}+d_{\text{out}})\) each:

    \[
    16(10240+6144+6144+10240) = 524{,}288
    \]

    \[
    16(22528+22528+22528) = 1{,}081{,}344
    \]

    \[
    524{,}288 + 1{,}081{,}344 = 1{,}605{,}632
    \]

    Times 40 blocks:

    \[
    40\times1{,}605{,}632 = 64{,}225{,}280
    \]

    About **64M** trainable parameters, 0.43% of 14.77B. This is the exact count behind the rounder "on the order of 60M, not 13M" in [17 §4](17-finetune-2026-update.md) and "≈ 63M" in [02 §3.2](02-hardware.md). If `scripts/check_lora.py` shows only `q_proj`/`v_proj` modules (at \(r=16\) that is \(40\times262{,}144\approx10.5\)M parameters) or any total far below 64M, the adapter scope is narrower than you asked for — stop and fix it.

### 4.4 Why low rank works

The claim behind LoRA is that the *change* a fine-tune needs to make has low intrinsic rank, even though \(W\) itself is full rank. You are not re-learning the projection; you are nudging it toward one narrow behaviour, and that nudge lives in a small subspace. Empirically ([17 §2](17-finetune-2026-update.md)), rank 16–32 matches full fine-tuning on post-training tasks. The MLP projections hold most of the capacity ([37 §1.3](37-architecture.md): the MLP is ~81% of a Qwen3-14B block's parameters), which is why all-layer LoRA that includes `gate/up/down` beats attention-only.

### 4.5 Rank and alpha

- **Rank \(r\)** is the capacity of the adapter. Low (8) for format/style; 16 is the handbook default; 32 for a capacity experiment; 64+ only with a large, clean dataset. More rank = more parameters = more overfitting risk, not automatically better.
- **Alpha \(\alpha\)** scales the update. Because the update is \(\tfrac{\alpha}{r}BA\), the *effective* learning rate scales with \(\alpha/r\). Keep \(\alpha/r\) fixed (the handbook uses \(\alpha=32\), so `scale = 32/r`) and the learning rate is comparable across ranks. This is the single most important thing to get right in MLX ([17 §3](17-finetune-2026-update.md)).

### 4.6 Which modules

Adapt all seven projections (`q,k,v,o,gate,up,down`) in every block — MLX's `num_layers: -1`. Attention-only leaves quality on the table; MLP-only is nearly as good as both ([17 §2.1](17-finetune-2026-update.md)). Fall back to the last 24 then 16 blocks only if memory forces it ([02 §6](02-hardware.md)).

---

## 5. QLoRA: 4-bit base, bf16 adapter

QLoRA is LoRA with the frozen base **quantized to 4-bit**. The base weights read at ~4.5 bits/parameter; the adapter \(A,B\) and its optimizer state stay in bf16/fp32. This cuts the largest term — the weights — by ~3.5× (16 bits down to 4.5, 29.5 GB down to 8.3 GB) and is what lets a 14B model train on 32 GB.

### 5.1 Memory budget for 14B on 32 GB

The terms (from [02 §3.1–3.2](02-hardware.md)):

| Term | Size | Why |
|---|---:|---|
| 4-bit base weights | ~8.3 GB | 14.77B at ~4.5 bits |
| LoRA params (r=16, all layers) | ~0.25 GB | 64M in fp32 |
| Optimizer (AdamW, adapter only) | ~0.7 GB | 2–3× adapter |
| Logits, B=1, L=2048, fp32 | ~1.24 GB | \(2048\times151936\times4\) |
| Activations (grad checkpoint) | ~4–8 GB | scales with \(L\) and adapted layers |
| Runtime (Metal, Python, tokenizer) | ~2–3 GB | overhead |

!!! example "On paper"

    The logits dominate the "fixed" part. Vocabulary \(V=151936\), one 2048-token sequence, fp32:

    \[
    2048\times151936\times4 = 1.24\times10^{9}
    \]

    That is **1.24 GB** before a single activation, and it grows linearly with sequence length: 2.49 GB at 4096 tokens. The column sums to about 16.5–21.5 GB. Add the attention and dequantization temporaries of [02 §3.1](02-hardware.md) (10–30% of activations) and fragmentation, and [17 §5](17-finetune-2026-update.md) estimates a peak of roughly **19–24 GB** for the all-layer run — under the ~24 GB ceiling only with the Kali VM and Chrome shut down.

The practical consequences (all in [02 §6](02-hardware.md)): cut `max_seq_length` before cutting `num_layers`; use `--grad-checkpoint`; keep the effective batch at 8 with `batch_size: 1` and `grad_accumulation_steps: 8`.

---

## 6. Data preparation and loss masking

This section covers the steps that are *specific* to fine-tuning: the chat JSONL shape, the train/valid/test split, loss masking, packing and sequence length. The general data work that comes first — sourcing, parsing, cleaning, deduplication, PII removal, labeling and data cards — is in [60 — Data preparation](60-ai-data-preparation.md); do that before you get here, because a fine-tune is only as good as the cleaned, deduped data you feed it.

### 6.1 Chat-format JSONL

Training data is one JSON object per line, each a full conversation in the chat format the server will use ([06](06-data-format.md)):

```json
{"messages": [
  {"role": "system", "content": "You scan only authorized scope."},
  {"role": "user", "content": "Enumerate services on 10.10.10.5."},
  {"role": "assistant", "content": "", "tool_calls": [
    {"type": "function", "function": {"name": "nmap_scan",
     "arguments": "{\"target\": \"10.10.10.5\"}"}}]}
]}
```

The trainer renders each row through the model's chat template (the same one the server uses) and tokenizes the result. **If the training template and the serving template differ, a good adapter produces garbage** ([10 §4](10-export-serving.md), [37 §7](37-architecture.md)).

### 6.2 Train / valid / test split

Split *before* you do anything else, and split by *source task*, not by row, so near-duplicate variations of one task do not straddle the boundary (that leaks the answer into validation). The handbook uses `train` to fit, `valid` to watch for overfitting during the run, and a frozen `test`/golden set to decide release ([09 §3](09-evaluation.md), [54](54-ai-evaluation.md)). A rule of thumb: 80/10/10, but the valid and test sets matter more for their *coverage* than their size — 200 well-chosen tasks beat 2000 random ones.

### 6.3 Loss masking: train only on assistant tokens

The model sees the whole conversation, but you only want it to *learn to produce* the assistant's turn, not to parrot the prompt. So you mask the loss: prompt and tool-result tokens contribute nothing; only the final assistant completion does. This is `--mask-prompt` in MLX and exactly the completion loss \(\mathcal{L}_{\text{SFT}}\) of [32 §3.1](32-mathematics.md).

```python
import numpy as np

# A tiny rendered example: 3 prompt tokens, then a 2-token answer.
# mask=0 on prompt, mask=1 on the assistant completion.
tokens = ["<user>", "scan", "10.10.10.5", "<assistant>", "nmap", "10.10.10.5"]
mask   = [0,        0,      0,            0,             1,      1]

# Per-token negative log-likelihood the model assigned (toy numbers).
nll = np.array([2.0, 1.5, 3.0, 0.1, 0.22, 0.51])  # natural log units
m = np.array(mask, dtype=float)

masked_loss = float((nll * m).sum() / m.sum())
unmasked_loss = float(nll.mean())
print(f"tokens in completion: {int(m.sum())}")
print(f"masked loss   (assistant only): {masked_loss:.4f}")
print(f"unmasked loss (all tokens):     {unmasked_loss:.4f}")
```

```text
tokens in completion: 2
masked loss   (assistant only): 0.3650
unmasked loss (all tokens):     1.2217
```

The masked loss averages only the two assistant tokens, \((0.22+0.51)/2=0.365\). The unmasked number is polluted by how badly the model "predicts" the user's own words, which you do not care about and cannot change. A `nan` masked loss means a row has *zero* completion tokens (the prompt filled the whole `max_seq_length`); that is the classic [08 §5](08-finetuning.md) crash.

!!! note "Software-engineering analogy"

    Loss masking is `assert` on the outputs only. You do not assert that the test *fixture* is correct — it is given. You assert on what the function *returns*. The prompt is the fixture; the assistant turn is the return value.

### 6.4 Packing and sequence length

- **Packing** concatenates several short examples into one `max_seq_length` sequence (with attention masks that stop one example attending to the next) so you do not waste compute padding. It raises tokens/sec on short data.
- **Sequence length vs memory** is the main lever: activations and the logit tensor grow linearly with \(L\) ([02 §3.4](02-hardware.md)). Measure your 99th-percentile token length with `scripts/token_stats.py` and set `max_seq_length` just above it; do not pay for 4096 when 2048 covers 99% of rows.

---

## 7. Hyperparameters and the loss curve

### 7.1 Rules of thumb

| Knob | Start | Notes |
|---|---|---|
| Learning rate | 1e-4 (with `scale=32/r`) | 10× the full-FT LR ([17 §2.2](17-finetune-2026-update.md)); halve on `nan`/spikes. |
| Epochs | 1–2 | >1M tokens: 1 epoch, add a 2nd only if val still falls. |
| Effective batch | 8 | `batch_size 1` × `grad_accum 8`; keep 8–32, avoid >32 ([17 §2.3](17-finetune-2026-update.md)). |
| Warmup | ~50 steps | ramp LR from 0 so the first steps do not blow up. |
| Rank | 16 | 8 for format, 32 for capacity. |
| Dropout | 0.05 | 0.1 for <500k tokens. |

### 7.2 Reading the loss curve

| Symptom | Likely cause | Action |
|---|---|---|
| Train 1.5 → 0.6 over 200 steps, val tracks it | healthy | continue |
| Train keeps falling, **val rises** for 2+ evals | overfitting | stop, keep the earlier checkpoint, more data/dropout |
| Train stuck >1.0 after 200 steps | LR too low, data too diverse, or bad mask | raise LR, check masking |
| Loss spikes then `nan` | LR too high, or empty completion | halve LR, check `token_stats` |
| Train <0.2 very fast | memorisation/duplication/leak | dedup, check contamination |

!!! warning

    Loss is not the goal; the eval is. A model can reach 0.3 loss by memorising tool *names* and still fail held-out tasks. Always evaluate mid-run against a held-out set, not just watch the curve ([08 §8](08-finetuning.md)).

---

## 8. Overfitting and catastrophic forgetting

### 8.1 Two different failures

- **Overfitting** is learning the training set, including its noise, so held-out performance drops. You detect it with a **validation set**: when val loss turns up while train loss falls, you have passed the useful point.
- **Catastrophic forgetting** is subtler: the model gets better at your task and *worse at everything else*. Fine-tune hard on tool calls and the model may lose some general chat ability, reasoning, or even safety behaviour. The training and validation loss (both on *your* distribution) look great while a capability you did not measure quietly degrades.

### 8.2 Detecting forgetting: a general-skills eval

The only way to see forgetting is to measure a skill you are *not* training. Keep a small **general-skills eval** — a handful of tasks from the base model's competence (simple reasoning, a non-domain Q&A, an instruction-following check) — and run it on the base and on every adapter. If the adapter wins on your golden set but drops more than a point or two on the general eval, you are trading away breadth.

```python
# Toy: base vs fine-tune on the target task AND on a held-out general set.
base = {"target": 0.55, "general": 0.90}
tune = {"target": 0.78, "general": 0.83}

for k in ("target", "general"):
    delta = tune[k] - base[k]
    print(f"{k:8s}: base {base[k]:.2f} -> tune {tune[k]:.2f}  ({delta:+.2f})")

target_gain = tune["target"] - base["target"]
general_drop = base["general"] - tune["general"]
print(f"traded {general_drop:.2f} general for {target_gain:.2f} target")
```

```text
target  : base 0.55 -> tune 0.78  (+0.23)
general : base 0.90 -> tune 0.83  (-0.07)
traded 0.07 general for 0.23 target
```

Whether a 7-point general drop is acceptable for a 23-point target gain is a product decision — but you can only make it because you *measured* the general skill. Mitigations: fewer epochs, lower rank/LR, mix a little general data into the training set, or keep adapters separate and swap them in only for the task.

---

## 9. A tiny LoRA that runs on your CPU

This is LoRA in ~35 lines of PyTorch on the CPU: a frozen `Linear`, a trainable low-rank adapter, trained on a toy task. It prints trainable vs total parameters and a falling loss. Run it with the handbook venv; it finishes in a few seconds, most of them spent importing torch.

```python
import torch, torch.nn as nn

torch.manual_seed(0)
D_IN, D_OUT, R, ALPHA = 16, 16, 4, 8.0

class LoRALinear(nn.Module):
    def __init__(self, d_in, d_out, r, alpha):
        super().__init__()
        self.base = nn.Linear(d_in, d_out, bias=False)
        self.base.weight.requires_grad_(False)       # freeze the base
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r))  # B=0 => adapter invisible at step 0
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.t()) @ self.B.t()

model = LoRALinear(D_IN, D_OUT, R, ALPHA)

# The target is the frozen base plus a rank-4 update: exactly the kind of
# change a rank-4 adapter can represent, so LoRA can fit it.
tA, tB = torch.randn(R, D_IN), torch.randn(D_OUT, R)
delta = tB @ tA                                       # a rank-4 matrix
X = torch.randn(256, D_IN)
Y = model.base(X) + X @ delta.t()
opt = torch.optim.Adam([model.A, model.B], lr=0.05)   # ONLY A and B

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable={trainable}  total={total}  ({100*trainable/total:.1f}%)")

for step in range(201):
    opt.zero_grad()
    loss = ((model(X) - Y) ** 2).mean()
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")
```

```text
trainable=128  total=384  (33.3%)
step   0  loss 109.2847
step  50  loss 0.8949
step 100  loss 0.0061
step 150  loss 0.0000
step 200  loss 0.0000
```

The 128 trainable parameters are \(A\) (\(4\times16=64\)) plus \(B\) (\(16\times4=64\)); the 256 frozen base weights never move. The loss starts high (the untrained adapter is \(0\), so the model is just the base and misses the whole update) and falls to zero as \(A,B\) learn it — it can, because the update was *built* to be rank 4, exactly LoRA's assumption from [§4.4](#44-why-low-rank-works). On a real model the base is a 4-bit Qwen3-14B and the "task" is your trajectories, but the mechanism — train a thin \(BA\), leave \(W\) alone — is exactly this.

---

## 10. Preference tuning (DPO)

### 10.1 When to use it

SFT teaches the model to imitate good trajectories. It cannot tell a trajectory that *looks* right from one that *works*, and it cannot learn "when the tool fails, do X instead of Y" from demonstrations alone ([15 §2](15-agentic-methods.md)). Preference tuning fixes that: you give pairs `(prompt, chosen, rejected)` and push the model to prefer `chosen`. Use it *after* SFT, to fix verbosity, refusal calibration, format slips, and the "pick the action that succeeds" cases — not to teach a new capability.

**Direct Preference Optimization (DPO)** does this without a separate reward model. It keeps a frozen reference model (your SFT checkpoint) and trains the policy to raise the chosen response's likelihood and lower the rejected one's, relative to that reference.

### 10.2 The loss

\[
\mathcal{L}_{\text{DPO}} = -\log\sigma\!\big(\beta(\Delta_w - \Delta_l)\big),
\]

where \(\Delta_w=\log\pi_\theta(y_w\mid x)-\log\pi_{\text{ref}}(y_w\mid x)\) is the chosen response's log-prob shift from the reference, \(\Delta_l\) is the same for the rejected response, \(\beta\) controls how far the policy may stray from the reference, and \(\sigma\) is the logistic function.

!!! example "On paper"

    One pair, \(\beta=0.1\). The policy raised the chosen log-prob by 1.0 over the reference and lowered the rejected by 1.0, so \(\Delta_w=1.0\), \(\Delta_l=-1.0\):

    \[
    \beta(\Delta_w-\Delta_l) = 0.1(1.0-(-1.0)) = 0.2
    \]

    \[
    \sigma(0.2) = \frac{1}{1+e^{-0.2}} \approx 0.550
    \]

    \[
    \mathcal{L}_{\text{DPO}} = -\ln 0.550 \approx 0.598
    \]

    The margin is positive, so the loss (0.598) is already below \(-\ln 0.5=0.693\), the value at zero margin. A gradient step widens the margin further, driving the loss toward 0; if the model instead preferred the *rejected* answer the margin would be negative and the loss would climb above 0.693.

MLX-LM ships no DPO trainer as of October 2026; use TRL/PEFT on a rented GPU, or a small preference loss on top of MLX ([15 §4](15-agentic-methods.md)). DPO's weakness: pair quality dominates, and it can reduce output diversity if over-trained.

---

## 11. Distillation

Distillation trains a small **student** to copy a larger **teacher**. It is a form of fine-tuning where the targets come from another model. Two kinds (full treatment in [39](39-gemma-distillation.md)–[42](42-distill-eval.md)):

- **Sequence (behaviour) distillation.** The teacher *generates* answers; the student is trained on them with ordinary SFT ([39 §2.3](39-gemma-distillation.md)). Cheap, robust, works through an API, and is the default on a 32 GB Mac. It transfers style, format and the teacher's solution path — not its uncertainty.
- **Logit distillation.** The student matches the teacher's full next-token *distribution* with a KL term ([39 §2.1](39-gemma-distillation.md)). Richer signal (the "dark knowledge" in the tail), but you need both models' logits at once and matching tokenizers, so it is heavier.

Use distillation when you have access to a stronger model and want to compress its behaviour on your tasks into something you can serve. The evaluation is the same golden-set comparison as any fine-tune ([42](42-distill-eval.md)).

---

## 12. Quantizing the result and shipping the adapter

### 12.1 Quantization error on a small row

You will quantize the final model to serve it in 4-bit (smaller, faster decode; [55](55-ai-production.md)). Quantization maps each weight to a small integer grid with a per-group scale ([32 §15](32-mathematics.md)). The error is bounded by half a step.

\[
s = \frac{\max_i |w_i|}{2^{k-1}-1}, \qquad \hat w_i = s\cdot\operatorname{round}(w_i/s).
\]

!!! example "On paper"

    One group of 4 weights \(w=(1.0,-0.6,0.3,-0.2)\), symmetric int4, grid \(-7\ldots7\):

    \[
    s = \frac{1.0}{7} \approx 0.142857
    \]

    \[
    \operatorname{round}(w/s) = (7,\,-4,\,2,\,-1)
    \]

    \[
    \hat w = (1.0,\,-0.571,\,0.286,\,-0.143)
    \]

    The errors are \((0,\,0.029,\,0.014,\,0.057)\), all within \(s/2\approx0.071\). At int8 the grid is \(-127\ldots127\), so \(s\approx0.0079\) and the worst error is ~18× smaller: that is the int4-vs-int8 quality/size trade.

Group-wise scales (one \(s\) per block of 32–128 weights) keep the error small for each block instead of letting one large weight set a coarse scale for the whole row.

### 12.2 Merge vs serve adapters separately

Two ways to ship a LoRA:

- **Merge (fuse) the adapter** into the base: compute \(W' = W + \tfrac{\alpha}{r}BA\) once and save a standalone model ([10 §1](10-export-serving.md), `mlx_lm.fuse`). Zero inference overhead, one artifact to serve, but you have baked in one adapter and lost the ability to swap.
- **Serve the base + adapter separately**: keep the base in memory and apply the small adapter at run time, for a small per-token cost. `mlx_lm.server --adapter-path` serves one adapter this way; a request that names a different adapter makes it reload the whole model. Multi-adapter servers such as vLLM (`--enable-lora`) keep many adapters next to one base and pick one per request, so one server can host many fine-tunes (one per customer or task) cheaply. Choose this when you have many adapters and one base; merge when you ship one model.

---

## 13. Evaluating a fine-tune

A fine-tune is a change; the only question that matters is whether it is an *improvement*. Evaluate the base and the adapter on the **same frozen golden set**, under the **same decoding settings**, with the **same grader** ([09 §8](09-evaluation.md), [42 §3](42-distill-eval.md), [54](54-ai-evaluation.md)). Report the metric that matters for the product (tool-call validity, task success), not the loss. An adapter that does not beat the base on the golden set is not a release candidate, no matter how pretty its loss curve.

!!! example "On paper"

    Golden set of 240 tasks. Base solves 132, adapter solves 158:

    \[
    p_{\text{base}} = 132/240 = 0.550, \quad p_{\text{tune}} = 158/240 = 0.658
    \]

    \[
    \Delta = 0.658 - 0.550 = 0.108
    \]

    An 11-point gain on 240 tasks is outside the noise. Treating the two runs as independent, the 95% margin on the difference is \(1.96\sqrt{(0.55\cdot0.45+0.658\cdot0.342)/240}\approx0.087\), and a paired test on the same tasks is usually tighter. A 1–2 point gap would be noise; use the Wilson/paired tests in [42 §2.4](42-distill-eval.md) and [54](54-ai-evaluation.md) to be sure. Only now is the adapter a candidate.

Also run the **general-skills eval** of [§8](#8-overfitting-and-catastrophic-forgetting) so a gain on the target task is not hiding a regression everywhere else.

---

## 14. Cost and time: local vs rented GPU

### 14.1 A local LoRA run

Use the token arithmetic from [08 §3.6](08-finetuning.md). A 2M-token dataset, average 1,500 tokens/sample, effective batch 8, two epochs:

\[
\text{iters} = 2\times\frac{2{,}000{,}000}{8\times1500} \approx 334,
\]

!!! example "On paper"

    Each iteration sees \(8\times1500\) tokens, so the run processes:

    \[
    334\times8\times1500 \approx 4.0\times10^{6}\ \text{tokens}
    \]

    At the handbook's 22 tok/s for a 14B all-layer run ([02 §4](02-hardware.md)):

    \[
    \frac{4.0\times10^{6}}{22} \approx 1.82\times10^{5}\ \text{s}
    \]

    \[
    1.82\times10^{5}/3600 \approx 50\ \text{hours}
    \]

    About **50 hours of pure compute**, ~58 hours of wall time with the ~15% overhead for data loading, validation, checkpoints and thermal throttling. In dollars the laptop run is "free", but the machine is occupied for two and a half days.

### 14.2 Renting a GPU

Assume a rented GPU at **\$2.00 per hour** (state your real price; prices move). Estimate its speed with the same FLOPs rule that gave the laptop's 22 tok/s ([02 §4](02-hardware.md), [32 §14.4](32-mathematics.md)): tokens per second are the effective FLOP/s divided by \(6N\). An A100 has 312 TFLOP/s of dense BF16 peak and an H100 SXM 989. Assume a modest 100 TFLOP/s effective (about 32% of an A100's peak, 10% of an H100's), because QLoRA dequantization and small batches waste compute:

\[
r = \frac{\text{effective FLOP/s}}{6N}
\]

!!! example "On paper"

    \(N=14.77\times10^{9}\), so \(6N\approx8.86\times10^{10}\) FLOPs per token:

    \[
    \begin{aligned}
    r &= 10^{14}/(8.86\times10^{10})\\
    &\approx 1130\ \text{tok/s}\\
    t &= 4.0\times10^{6}/1130\\
    &\approx 3540\ \text{s}\approx 1\ \text{h}\\
    \text{cost} &\approx 1\ \text{h}\times\$2.00 = \$2
    \end{aligned}
    \]

    About 50 times the laptop's 22 tok/s: the run that occupies the Mac for two and a half days takes about an hour, for about \$2 plus setup (downloading ~30 GB of weights), storage and egress.

The trade is clear: for a one-off adapter the laptop is fine if you can wait; for a sweep of 5 configs, or anything over ~24 hours, renting is the rational choice ([02 §6](02-hardware.md), item 9). It also frees your laptop and avoids the thermal penalty.

---

## 15. A checklist before you fine-tune

- [ ] You tried prompting, few-shot and RAG first, and named why they are not enough.
- [ ] The thing you want is a *behaviour/format/skill*, not a *fact* (facts go in RAG).
- [ ] You have a frozen eval that the fine-tune must beat, and a base-model score on it.
- [ ] You have a general-skills eval to catch forgetting.
- [ ] Data is in the correct chat format, deduped, split by task, loss-masked.
- [ ] The training template equals the serving template.
- [ ] You checked the adapter scope (`check_lora.py`): ~64M params, all 7 projections.
- [ ] `scale = 32/r` and the learning rate match (not MLX's `scale: 20` trap).
- [ ] Memory budget fits under ~24 GB with grad checkpointing.
- [ ] You know how you will serve it: merged, or base + adapter.

---

## 16. Key takeaways

- The adaptation ladder is a cost gradient: prompting → few-shot → RAG → fine-tune → continued pretraining → from scratch. Climb only as far as the problem forces you.
- Fine-tuning reliably changes format, style, tool-call syntax and narrow skills, and buys latency by making a smaller model reliable. It does **not** reliably add facts — those go in RAG.
- LoRA trains a thin \(W+\tfrac{\alpha}{r}BA\); at \(r=16\) an all-layer Qwen3-14B adapter is ~64M parameters, 0.43% of the model. QLoRA's 4-bit base is what fits it on 32 GB.
- Mask the loss to the assistant tokens; keep `scale = 32/r`; watch validation loss *and* a general-skills eval, not just train loss.
- Evaluate the adapter against the base on the same frozen golden set. A pretty loss curve is not a release gate.

---

## 17. Exercises

!!! question "Exercise 17.1"

    Your model returns valid JSON only 70% of the time at temperature 0, and the system prompt is already 500 tokens of format rules. Which rung of the ladder, and why?

??? success "Answer"

    Fine-tune (rung 4). The behaviour (JSON format) is one the model has the parts for but produces unreliably, and the prompt is already large. This is the "right but unreliable, prompt huge" case: SFT on a few thousand correctly-formatted examples bakes the format in so you can shorten the prompt. RAG would not help (no facts missing); more prompting has diminishing returns at 500 tokens.

!!! question "Exercise 17.2"

    A `4096×1024` projection (`k_proj`-shaped) gets a LoRA adapter at rank 8. How many trainable parameters, and what fraction of the full projection?

??? success "Answer"

    LoRA adds \(r(d_{\text{in}}+d_{\text{out}}) = 8\times(4096+1024) = 8\times5120 = 40{,}960\). The full projection is \(4096\times1024 = 4{,}194{,}304\). The fraction is \(40{,}960/4{,}194{,}304 \approx 0.00977\), about **0.98%**.

!!! question "Exercise 17.3"

    You train on 100 examples that each contain the sentence "our refund window is 30 days". In production the model sometimes says 30 days, sometimes 14. Why, and what should you have done?

??? success "Answer"

    Fine-tuning does not reliably store facts. The model learned a *tendency* toward "30 days", but a slightly different question surfaces a plausible neighbour (14) with no warning. Facts that must be exact and can change belong in the prompt via retrieval (RAG), where they are looked up, not recalled. The fine-tune should have targeted the *answer format* ("state the window, then cite the policy doc"), with the number supplied by RAG.

!!! question "Exercise 17.4"

    With `β=0.1`, a DPO pair has `Δ_w = 0.5` and `Δ_l = 0.5` (the policy shifted chosen and rejected by the same amount). Compute the loss and say what it means.

??? success "Answer"

    \(\beta(\Delta_w-\Delta_l)=0.1(0.5-0.5)=0\), so \(\sigma(0)=0.5\) and \(\mathcal{L}=-\ln 0.5\approx0.693\). Zero margin means the policy has learned *nothing* about this pair relative to the reference — it moved chosen and rejected equally. 0.693 is the DPO loss at no preference; training must push \(\Delta_w\) above \(\Delta_l\).

!!! question "Exercise 17.5"

    Quantize the weight row \(w=(0.8, -0.4, 0.4, -0.8)\) to symmetric int4 (grid \(-7\ldots7\)). Give the scale, the integer codes, the dequantized values and the max error.

??? success "Answer"

    \(s=\max|w|/7=0.8/7\approx0.11429\). Then \(w/s=(7,-3.5,3.5,-7)\); the halves round to even, so the codes are \((7,-4,4,-7)\). Dequantized \(\hat w=(0.8,-0.457,0.457,-0.8)\), errors \((0,0.057,0.057,0)\), max \(0.057\le s/2\approx0.057\). The extreme values \(\pm0.8\) are exact (they set the scale); the mid values sit exactly half a step from the grid, the worst case.

!!! question "Exercise 17.6"

    Your 14B LoRA run processes 6M tokens. At 25 tok/s, how long is the pure compute, and roughly the wall time?

??? success "Answer"

    \(6\times10^{6}/25 = 240{,}000\) s \(= 66.7\) hours of compute. With ~15% overhead, wall time is about \(66.7\times1.15\approx77\) hours, a little over three days. At the ~1,130 tok/s assumed for a rented GPU in §14.2, the same tokens take \(6\times10^{6}/1130\approx5{,}310\) s, about 1.5 hours, or about \$3 at \$2/hour — the point at which renting clearly wins.

!!! question "Exercise 17.7"

    After fine-tuning, your golden-set success rises from 0.55 to 0.72 but a held-out general reasoning eval drops from 0.88 to 0.70. What happened and name two fixes.

??? success "Answer"

    Catastrophic forgetting: the model gained 17 points on the target task but lost 18 on general reasoning — a bad trade you could only see because you kept a general-skills eval. Fixes: fewer epochs or lower LR/rank (less aggressive adaptation); mix some general-purpose data into the training set; or keep the adapter separate and swap it in only for the target task.

!!! question "Exercise 17.8"

    You have one base model and 20 customers who each want a slightly different fine-tune. Merge the adapters or serve them separately? Why?

??? success "Answer"

    Serve them separately (base + adapter), on a server that picks the adapter per request (§12.2). Merging would mean 20 full standalone models in memory/disk; serving one shared base with 20 small adapters costs ~64M extra parameters each instead of 14.77B, and lets one server host all customers. Merge only when you ship a single model and want zero adapter overhead at inference.

---

Next: [54 — Evaluation](54-ai-evaluation.md) turns "is the fine-tune better?" into a real test suite with statistics.
