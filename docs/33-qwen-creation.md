# 33 — How Qwen was built

*Last verified: 25 October 2026 against the Qwen2.5, Qwen2.5-Coder and Qwen3 technical reports, the Qwen3-Coder and Qwen3.5 releases, and the model cards.*

This chapter reconstructs the Qwen pipeline from public information. It separates what is disclosed (architecture, token counts, training stages, some RL details) from what is not (exact data mixture, exact hyperparameters, full RL recipe).

> **Read with:** [37 — Architecture anatomy](37-architecture.md) for the shape-level view and [38 — Derivations and proofs](38-derivations.md) for the math behind GRPO, distillation and attention.

---

## 1. The family tree

| Generation | Models used here | What changed |
|---|---|---|
| **Qwen2.5** | 0.5B–72B dense, 30B-A3B MoE | 18T tokens, 1M+ SFT samples, strong multilingual/coding |
| **Qwen2.5-Coder** | 0.5B–32B | continued pretraining on 5.5T code tokens; file + repo level; FIM |
| **Qwen3** | 0.6B–32B dense, 30B-A3B / 235B-A22B MoE | 36T tokens, 119 languages, thinking/non-thinking fusion, strong RL |
| **Qwen3-Coder** | 30B-A3B, 480B-A35B, Coder-Next | 7.5T tokens 70% code, 256K, agentic RL |
| **Qwen3.5 / 3.6** | dense 0.8B–27B, MoE 35B-A3B / 122B-A10B / 397B-A17B | hybrid GDN + gated attention, native multimodal, 256K–1M |
| **Qwen3.8** | 27B dense, larger MoE | newer generation; not the local training target here |

The local handbook uses **Qwen3-14B** for agents and **Qwen2.5-Coder-14B** for specialised code adapters because both train reliably in 32 GB. The newer Qwen3.5/3.6/3.8 models are stronger but their GDN training path is immature in MLX on this machine.

---

## 2. Qwen2.5: the base recipe

The Qwen2.5 report describes:

- **18 trillion pretraining tokens**, up from 7T in Qwen2.
- Dense and MoE variants, 0.5B–72B.
- **1M+ supervised fine-tuning samples**.
- Strong multilingual, coding, math and structured-output data.
- A base architecture that Qwen3 and Qwen2.5-Coder inherit: GQA, SwiGLU, RoPE, RMSNorm with pre-norm, QKV bias in Qwen2 (removed in Qwen3).

Qwen2.5 is important because it establishes the data and post-training philosophy: large-scale pretraining, then a large SFT mixture, then preference optimisation. The exact mixture is not public.

---

## 3. Qwen2.5-Coder: continued pretraining on code

Qwen2.5-Coder is not trained from random weights. It starts from Qwen2.5 and **continues pretraining** on a code-heavy corpus.

### 3.1 Data

- **5.5 trillion tokens** total.
- **5.2T file-level tokens** at sequence length 8,192, with next-token prediction and **FIM** (fill-in-the-middle).
- **300B repo-level tokens** extending context to 32,768 for repository understanding.
- Final mixture **70% code, 20% text, 10% math**.
- Sources: public GitHub repositories, web-crawled code text, synthetic code data, code-related documents.
- Cleaning: dedup, quality filtering, language balancing, license filtering.

### 3.2 FIM objective

Fill-in-the-middle trains the model to complete a span given a prefix and suffix. With special tokens `<fim_prefix>`, `<fim_suffix>`, `<fim_middle>`:

\[
\mathcal{L}_{\text{FIM}} =
-\sum_{t\in\text{middle}}
\log p_\theta(x_t\mid \text{prefix},\text{suffix},x_{<t}).
\]

!!! example "On paper"

    A tiny Python infill: the prefix is `def sq(x):`, the suffix is `print(sq(3))`, and the middle `return x*x` is three tokens that the model predicts with probabilities \(\tfrac12\), \(\tfrac12\) and \(\tfrac14\). Use \(\ln2\approx0.693\).

    \[
    \begin{aligned}
    \mathcal{L}_{\text{FIM}}
    &=-\big(\ln\tfrac12+\ln\tfrac12+\ln\tfrac14\big)\\
    &=4\ln2\approx2.77
    \end{aligned}
    \]

    The prefix and suffix tokens are context only and add nothing to the sum; a perfect infill (all three probabilities 1) would score 0.

    Longer version: [43 §1](43-math-examples.md).

This is why code models are good at editing inside a file, not just continuing at the end.

### 3.3 Architecture

| Model | Params | Layers | Heads (Q/KV) | Context | Vocab |
|---|---:|---:|---|---:|---:|
| Qwen2.5-Coder-7B | 7.6B | 28 | 28 / 4 | 128K | 151,646 |
| **Qwen2.5-Coder-14B** | **14.7B** | **48** | **40 / 8** | **128K** | **151,646** |
| Qwen2.5-Coder-32B | 32.5B | 64 | 40 / 8 | 128K | 151,646 |

The 14B has \(d=5{,}120\), \(d_{\text{ff}}=13{,}824\), 48 layers, GQA 40/8, untied embeddings. Apache-2.0.

### 3.4 Post-training

- SFT on code instructions, repo tasks, code repair, tests, documentation.
- DPO/preference optimisation for helpfulness and correctness.
- Instruction data includes file-level and repo-level tasks, not just isolated functions.

The result is a code-specialised model with strong FIM, repair and repository reasoning, but a narrower general-agent profile than Qwen3.

---

## 4. Qwen3: the general agent base

### 4.1 Architecture

Qwen3 uses a Llama-style decoder with:

- **GQA**;
- **SwiGLU**;
- **RoPE**;
- **RMSNorm** with pre-norm;
- **QK-Norm** for stability;
- **no QKV bias** (removed from Qwen2);
- **BBPE tokenizer**, vocabulary **151,669**.

Dense family:

| Model | Layers | Heads (Q/KV) | Tie embedding | Context |
|---|---:|---|---:|---:|
| Qwen3-0.6B | 28 | 16 / 8 | Yes | 32K |
| Qwen3-1.7B | 28 | 16 / 8 | Yes | 32K |
| Qwen3-4B | 36 | 32 / 8 | Yes | 128K |
| Qwen3-8B | 36 | 32 / 8 | No | 128K |
| **Qwen3-14B** | **40** | **40 / 8** | **No** | **128K** |
| Qwen3-32B | 64 | 64 / 8 | No | 128K |

MoE family:

| Model | Layers | Heads (Q/KV) | Experts (total/active) | Context |
|---|---:|---|---:|---:|
| Qwen3-30B-A3B | 48 | 32 / 4 | 128 / 8 | 128K |
| Qwen3-235B-A22B | 94 | 64 / 4 | 128 / 8 | 128K |

The report’s “128K” is the architecture’s extended context; pretraining’s long-context stage trains to **32,768**, and YaRN/DCA extend inference to 128K. The model card’s “32K native / 131K YaRN” is the practical statement.

Qwen3-14B has \(d=5{,}120\), \(d_{\text{ff}}=17{,}408\), 40 layers, 14.8B total, 13.2B non-embedding.

### 4.2 Pretraining data: 36T tokens, 119 languages

Qwen3 is trained on **36 trillion tokens** covering **119 languages and dialects**:

- web documents, books, code, STEM, reasoning, multilingual text;
- synthetic data generated by Qwen2.5, Qwen2.5-Math and Qwen2.5-Coder;
- PDF text extracted with Qwen2.5-VL, then refined with Qwen2.5;
- additional multilingual data.

### 4.3 Three pretraining stages

1. **General stage** — about **30T tokens** at 4,096 context, building broad knowledge.
2. **Reasoning/STEM stage** — knowledge-intensive data, more STEM and code, faster LR decay.
3. **Long-context stage** — hundreds of billions of tokens at **32,768**; 75% of samples 16K–32K, 25% 4K–16K. RoPE base raised from \(10^4\) to \(10^6\) with **ABF**; **YaRN** and **Dual Chunk Attention** give 4× inference extension.

The three-stage structure is the template to copy if you build a small model: general → reasoning/code → long context.

### 4.4 Post-training: four stages

Qwen3’s post-training is the most instructive part.

#### Stage 1 — Long-CoT cold start

Curate math, code, logic and STEM problems with **verified answers or test cases**. Filter:

- queries that are not easily verifiable;
- queries the model can answer without CoT;
- responses with wrong answers or bad format.

The report uses **3,995 query–verifier pairs** for the reasoning RL stage. Cold-start SFT teaches the format of long reasoning before RL.

#### Stage 2 — Reasoning RL

Qwen3 uses **GRPO** (Group Relative Policy Optimisation). For a query \(q\), sample \(G\) outputs \(\{o_i\}\), score them with a verifier \(R_i\), and compute:

\[
\hat A_i = \frac{R_i - \operatorname{mean}(R)}{\operatorname{std}(R)},
\]

\[
r_i(\theta) = \frac{\pi_\theta(o_i\mid q)}{\pi_{\text{old}}(o_i\mid q)},
\]

\[
\mathcal{J}_{\text{GRPO}}(\theta) =
\mathbb{E}\left[
\frac{1}{G}\sum_{i=1}^{G}
\min\left(
r_i(\theta)\hat A_i,\;
\operatorname{clip}(r_i(\theta),1-\epsilon,1+\epsilon)\hat A_i
\right)
-\beta\,\operatorname{KL}(\pi_\theta\|\pi_{\text{ref}})
\right].
\]

!!! example "On paper"

    One maths query, \(G=4\) sampled answers. The verifier scores them \(R=(1,0,1,0)\), so \(\operatorname{mean}(R)=0.5\) and, with the population standard deviation, \(\operatorname{std}(R)=0.5\). Take \(\epsilon=0.2\), \(\beta=0.04\) and a KL estimate of 0.25.

    | \(i\) | \(R_i\) | \(\hat A_i\) | \(r_i(\theta)\) | \(\operatorname{clip}(r_i)\) | \(\min(\ldots)\) |
    |---|---:|---:|---:|---:|---:|
    | 1 | 1 | +1 | 1.5 | 1.2 | 1.2 |
    | 2 | 0 | −1 | 0.5 | 0.8 | −0.8 |
    | 3 | 1 | +1 | 1.0 | 1.0 | 1.0 |
    | 4 | 0 | −1 | 1.0 | 1.0 | −1.0 |

    \[
    \begin{aligned}
    \tfrac14(1.2-0.8+1.0-1.0)&=0.1\\
    \mathcal{J}_{\text{GRPO}}&=0.1-0.04\cdot0.25=0.09
    \end{aligned}
    \]

    Answer 1 is already 50% more likely than under \(\pi_{\text{old}}\) and answer 2 has halved. In both rows the min picks the clipped value, which is a constant, so this step pushes neither of them further; answers 3 and 4 get the full update.

The group baseline replaces a learned value function. The KL term keeps the policy near the reference. Qwen3 reports large-batch, many-rollout, off-policy training and entropy control as important for stability.

#### Stage 3 — Thinking mode fusion

The goal is one model that can think or answer directly. Qwen3:

- builds SFT data with and without reasoning paths;
- uses chat templates with `/think` and `/no_think`;
- adds a **thinking budget** to cap reasoning tokens;
- trains the model to follow the mode token.

This is why Qwen3 can serve both “reasoning” and “chat” from one checkpoint, unlike separate QwQ/Qwen chat models.

#### Stage 4 — General RL

A reward system over **20+ tasks**:

- instruction following, format following, preference alignment;
- tool use, agentic tasks, long-horizon decision making;
- RAG factuality, specialized scenarios;
- rule-based, model-based and human feedback rewards.

#### Strong-to-weak distillation

Smaller Qwen3 models are trained by **distilling** the flagship:

\[
\mathcal{L}_{\text{KD}} =
\operatorname{KL}\left(
p_{\text{teacher}}(\cdot\mid x)
\;\|\;
p_{\text{student}}(\cdot\mid x)
\right),
\]

often on **on-policy** student samples. The report says distillation is more efficient than RL for small models and uses about **1/10 of the GPU hours** of the four-stage recipe. The student learns both thinking and non-thinking behaviour from the teacher.

!!! example "On paper"

    Three candidate next tokens. The teacher predicts \(p_{\text{teacher}}=(\tfrac12,\tfrac14,\tfrac14)\) and the student \(p_{\text{student}}=(\tfrac14,\tfrac12,\tfrac14)\). Use \(\ln2\approx0.693\).

    \[
    \begin{aligned}
    \operatorname{KL}
    &=\tfrac12\ln2+\tfrac14\ln\tfrac12+\tfrac14\ln1\\
    &=\tfrac14\ln2\approx0.173
    \end{aligned}
    \]

    Swapping the student's first two probabilities would match the teacher and make the loss 0. Unlike a one-hot label, the target also tells the student how much weight the teacher puts on the alternatives.

    Longer version: [43 §19](43-math-examples.md).

### 4.5 What is not public

- Exact data mixture and dedup thresholds.
- Exact SFT/RL dataset sizes beyond the disclosed 3,995 verifier pairs.
- Full reward model details.
- Exact hyperparameters for the 14B run.
- The precise filtering classifiers.

You can reproduce the *recipe shape*, not the exact model.

---

## 5. Qwen3-Coder: agentic code RL

Qwen3-Coder is built on Qwen3:

- **7.5T tokens**, **70% code**.
- 30B-A3B and 480B-A35B; 256K native, 1M with YaRN.
- Code RL on verifiable tasks; synthetic test-case scaling.
- **Agent RL** for multi-turn tool use: the model interacts with a repository, runs commands, observes failures and repairs.
- Qwen2.5-Coder is used to clean and rewrite noisy code data.
- Qwen3-Coder-30B-A3B: 48 layers, 32/4 heads, 128/8 experts, non-thinking.
- Qwen3-Coder-Next: 80B total / 3B active, 256K, trained with executable, interaction-rich tasks.

The lesson: code capability at the frontier is **execution-driven**. Static next-token prediction is not enough; the model needs a verifier and multi-turn recovery.

---

## 6. Qwen3.5 / 3.6: hybrid linear attention

Qwen3.5 changes the architecture:

- **3:1 hybrid**: three **Gated DeltaNet** layers per one **gated attention** layer.
- **Sparse MoE**: 512 experts, 17B active out of 397B (4.28%) in the flagship; dense 9B/27B variants.
- **Native multimodal**: trained from scratch on interleaved text, image and video tokens.
- **256K input**, extendable to **1M**.
- **Multi-token prediction**.
- **RL at scale**: million-agent environments, progressively harder tasks.

The GDN recurrence (see [32 — Mathematics](32-mathematics.md) §10):

\[
S_t = \alpha_t S_{t-1}(I-\beta_t k_t k_t^\top) + \beta_t v_t k_t^\top,
\qquad
o_t=S_t\,q_t,
\]

with \(S_t\in\mathbb{R}^{d_v\times d_k}\).

!!! example "On paper"

    Take \(d_k=d_v=2\) and a state that already stores two associations, key \(e_1\) to \(4e_1\) and key \(e_2\) to \(2e_2\): \(S_{t-1}=\operatorname{diag}(4,2)\). Write the new pair \(k_t=e_1=(1,0)\), \(v_t=(6,0)\) with \(\alpha_t=0.5\) and \(\beta_t=0.5\).

    \[
    \begin{aligned}
    I-\beta_tk_tk_t^\top&=\operatorname{diag}(0.5,\,1)\\
    \alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)&=\operatorname{diag}(1,\,1)\\
    \beta_tv_tk_t^\top&=\operatorname{diag}(3,\,0)\\
    S_t&=\operatorname{diag}(4,\,1)
    \end{aligned}
    \]

    Reading with \(q_t=e_1\) gives \(o_t=(4,0)\): the \(e_1\) slot decayed from 4 to 2, then moved halfway to the new value 6. The \(e_2\) slot was not written, so it only decayed, from 2 to 1.

    Longer version: [43 §11](43-math-examples.md).

The 3:1 stack gives cheap long-context memory while keeping periodic full attention for retrieval. Qwen3.6 shares the architecture. Qwen3.8 is a newer generation with stronger coding/agentic results; it is not a 32 GB local training target.

**Why this matters for you:** Qwen3.5/3.6 are better models but harder to train in MLX due to the GDN kernels and backward pass. Qwen3-14B remains the reliable local base. If you rent a GPU, Qwen3.5-9B is the better quality/time trade.

---

## 7. Reproducing the recipe at small scale

You cannot reproduce 36T tokens or 14.8B parameters on a Mac. You can reproduce the **shape**:

| Qwen stage | Small-scale analogue |
|---|---|
| 36T general pretraining | 20–100B tokens on a 0.1–1B model |
| STEM/code stage | 10–30% code/math, higher-quality subset |
| 32K long-context stage | 4K–8K sequences on a small model |
| Cold-start SFT | 10k–100k verified reasoning traces |
| Reasoning RL | GRPO on math/code with exact verifiers |
| Thinking fusion | mixed think/no-think SFT |
| General RL | preference pairs + rule rewards |
| Distillation | teacher logits or on-policy samples |

The exact numbers are in [35 — From zero](35-from-zero.md); the 12B/14B budget is in [36 — Reproducing 12B–14B](36-reproducing-12b-14b.md).

---

## 8. What to copy, what not to copy

**Copy:**

- three-stage pretraining (general → reasoning → long context);
- GQA + SwiGLU + RoPE + RMSNorm + QK-norm;
- a large, deduplicated, decontaminated corpus;
- verified cold-start data before RL;
- GRPO with a real verifier;
- thinking-mode fusion via chat template;
- distillation for small models;
- execution-driven agentic training.

**Do not copy blindly:**

- the 36T token budget at small scale (Chinchilla-optimal is far less);
- the exact LR/batch from a 14B run on a 100M model;
- the full MoE routing without load-balancing and expert parallelism;
- RL before the model can follow format and emit valid tools;
- vendor benchmark claims as your target.

---

## 9. Sources

- Qwen2.5 technical report — <https://arxiv.org/abs/2412.15115>
- Qwen2.5-Coder technical report — <https://arxiv.org/abs/2409.12186>
- Qwen3 technical report — <https://arxiv.org/abs/2505.09388>
- Qwen3-14B model card — <https://huggingface.co/Qwen/Qwen3-14B>
- Qwen3-30B-A3B model card — <https://huggingface.co/Qwen/Qwen3-30B-A3B>
- Qwen3-Coder blog — <https://qwenlm.github.io/blog/qwen3-coder>
- Qwen3-Coder-Next technical report — <https://arxiv.org/abs/2603.00729>
- Qwen3.5 blog — <https://qwen.ai/blog?id=qwen3.5>
- Qwen3.5 model card — <https://huggingface.co/Qwen/Qwen3.5-122B-A10B>
- Qwen3.5-Omni technical report — <https://arxiv.org/abs/2604.15804>
- GRPO — <https://arxiv.org/abs/2402.03300>
- Gated DeltaNet — <https://arxiv.org/abs/2412.06464>
- [32 — Mathematics](32-mathematics.md)
- [35 — From zero](35-from-zero.md)
