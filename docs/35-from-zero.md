# 35 — From zero: training a small model you can actually build

*Last verified: 25 October 2026. The arithmetic is standard; the hardware numbers are estimates and must be replaced with measurements.*

You cannot train a Qwen3-14B or Gemma 4 12B from scratch on one Mac. What you *can* do is build a small model that uses the same architecture and training recipe, learn the whole pipeline, and then decide whether to rent a GPU for a larger run.

This chapter is the honest path:

1. compute the budget;
2. choose a small architecture;
3. train a tokenizer;
4. build a clean corpus;
5. run a stable pretraining job;
6. evaluate it;
7. post-train it;
8. scale only if the small run works.

---

## 1. What “from zero” costs

Training compute is approximately

\[
C \approx 6ND,
\]

where \(N\) is parameters and \(D\) is training tokens. The factor 6 is forward (2) plus backward (4) FLOPs per parameter per token.

!!! example "On paper"

    Step 1 of the learning path in §8: a 10M-parameter model on 1B tokens.

    \[
    C\approx6\cdot10^{7}\cdot10^{9}=6\times10^{16}\ \text{FLOPs}
    \]

    At the M1 Pro’s 1.5 TFLOP/s effective (the table below), that is \(6\times10^{16}/(1.5\times10^{12})=40{,}000\) s, about 11 hours: an overnight run. The first table row works the same way: \(6\cdot125\text{M}\cdot20\text{B}=1.5\times10^{19}\) FLOPs, 250 times more.

    Longer version: [43 §15](43-math-examples.md).

### 1.1 Worked examples

| Model | Tokens | FLOPs | One M1 Pro (1.5 TFLOP/s effective) | 8×A100 (35% MFU) | Cost at $2/GPU-h |
|---|---:|---:|---:|---:|---:|
| 125M | 20B | \(1.5\times10^{19}\) | ~116 days | ~0.2 days | ~$76 |
| 336M | 20B | \(4.0\times10^{19}\) | ~309 days | ~0.5 days | ~$205 |
| 1.2B | 20B | \(1.5\times10^{20}\) | ~1,124 days | ~1.9 days | ~$741 |
| 1.2B | 100B | \(7.3\times10^{20}\) | ~15 years | ~9.6 days | ~$3,700 |
| 14B | 36T | \(3.2\times10^{24}\) | ~67,000 years | ~9.3 days on 10,000 H100s (40% MFU) | ~$4.5M |

The last row is Qwen3-14B’s disclosed token count and uses 10,000 H100s, not 8×A100s. It is not a laptop project. Even the 1.2B/20B run needs a serious GPU and a clean data pipeline.

> The M1 Pro number is intentionally brutal: local training is for **LoRA adapters**, not pretraining. The rest of this handbook is about adapters for a reason.

### 1.2 Chinchilla is a lower bound, not a target

Chinchilla-style compute-optimal training uses roughly

\[
D \approx 20N.
\]

A 1.2B model would need ~24B tokens to be compute-optimal. But models that will be served many times are often **overtrained** far beyond 20 tokens/parameter because inference cost dominates. Qwen3-14B at 36T is ~2,438 tokens/parameter; that is a deliberate inference-quality trade, not a mistake. For your first small model, 20–100 tokens/parameter is a reasonable range; more tokens usually help if data quality holds.

---

## 2. Choose an architecture

For a first model, use a Llama-style decoder:

- RMSNorm (pre-norm);
- RoPE;
- GQA;
- SwiGLU;
- no biases;
- tied embeddings for small models;
- vocabulary 32k–64k.

Why this shape: it is simple, well-supported, and close enough to Qwen to port lessons. Gemma’s GeGLU/local-global/p-RoPE stack is a second experiment, not a first build.

### 2.1 Three sizes

| Name | Layers | \(d\) | Heads | KV heads | FFN | Params | Tokens | Use |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tiny-100m | 12 | 768 | 12 | 4 | 2,048 | ~125M | 20B | CPU/1 GPU learning run |
| small-300m | 24 | 1,024 | 16 | 4 | 2,816 | ~336M | 20–40B | one or two GPUs |
| small-1b | 24 | 2,048 | 16 | 4 | 5,632 | ~1.2B | 20–100B | 8×A100-class |

The parameter column assumes a tied **64,000-token** vocabulary. With 32,000 tokens the same stacks are ~100M, ~303M and ~1.15B — at this size the vocabulary choice moves the total by up to 20%.

Parameter formula:

\[
P \approx L\left(2d^2 + 2dG d_h + 3d d_{\text{ff}}\right) + Vd
\]

for tied embeddings and \(H d_h=d\).

!!! example "On paper"

    The `tiny-100m` row: \(L=12\), \(d=768\), \(G=4\), \(d_h=64\), \(d_{\text{ff}}=2{,}048\), tied, with a 64,000-token vocabulary. Per layer:

    | Projection | Count |
    |---|---:|
    | Q and O, \(d\times d\) each | \(2\cdot768^2=1{,}179{,}648\) |
    | K and V | \(2\cdot768\cdot4\cdot64=393{,}216\) |
    | MLP | \(3\cdot768\cdot2{,}048=4{,}718{,}592\) |
    | **one layer** | **6,291,456** |

    \[
    \begin{aligned}
    12\cdot6{,}291{,}456&=75{,}497{,}472\\
    Vd=64{,}000\cdot768&=49{,}152{,}000\\
    P&\approx124.6\text{M}
    \end{aligned}
    \]

    That is the table’s ~125M. With a 32,000-token vocabulary the same stack is ≈100M: at this size the embedding is 39% of the model.

    Longer version: [43 §14](43-math-examples.md).

Use:

```bash
python scripts/estimate_pretraining.py --preset small-1b --vocab 64000 --tokens 20e9 \
  --gpus 8 --peak-tflops 312 --mfu 0.35 --cost-per-gpu-hour 2
```

This prints 1.21B parameters, 1.9 days and $741 — the §1.1 row. The `small-1b` and `tiny-100m` presets default to a 32,000-token vocabulary; without `--vocab 64000` the same command prints 1.15B and $701. Replace the defaults with your GPU’s peak and a measured MFU.

---

## 3. Train a tokenizer

You need a tokenizer before you can tokenize the corpus. A practical recipe:

1. Sample 1–10 GB of clean, deduplicated text from the same distribution as the corpus.
2. Train SentencePiece BPE or Unigram with vocabulary 32k–64k.
3. Enable byte fallback so no character is unknown.
4. Split digits (`split_digits=true`) for math/code.
5. Preserve whitespace if code is important.
6. Add special tokens: BOS, EOS, PAD, UNK, turn separators, tool tokens.
7. Measure compression on held-out text and code.

```bash
spm_train \
  --input=tok_sample.txt \
  --model_prefix=tok32k \
  --vocab_size=32000 \
  --model_type=bpe \
  --byte_fallback=true \
  --split_digits=true \
  --character_coverage=0.9995
```

Do not reuse a tokenizer from a different model family unless you are doing continued pretraining. A mismatched tokenizer breaks the embedding and the compression assumptions.

---

## 4. Build the corpus

A small model is dominated by data quality, not architecture.

### 4.1 Sources

| Source | Use | License |
|---|---|---|
| FineWeb / FineWeb-Edu | general web | ODC-BY / CC-BY |
| The Stack v3 | code | ODC-BY, per-file licenses |
| OpenWebMath / proof-pile | math | check each |
| Wikipedia | facts | CC-BY-SA |
| Books (public domain) | long-form | public domain |
| arXiv (where permitted) | STEM | metadata/licenses vary |
| Your own domain corpus | specialization | your rights |

### 4.2 Pipeline

```text
raw → language ID → text extraction → normalization
    → exact dedup → MinHash near-dedup
    → quality filter (classifier or heuristics)
    → toxicity/PII filter
    → benchmark decontamination (13-gram overlap)
    → mixture → tokenize → shard
```

Rules:

- Deduplicate before tokenizing; duplicated tokens waste compute.
- Decontaminate against every benchmark you will report.
- Keep a held-out split by source/domain, not random rows.
- Record provenance and licenses per shard.
- Mix code/math/general deliberately; do not let one source dominate by volume.
- Curriculum: general first, then higher-quality/STEM/code, then long-context annealing.

A reasonable first mixture for a general small model: 70% web, 15% code, 10% math, 5% books/reference. For a code model, invert toward code; for a domain model, add 10–30% domain data.

---

## 5. The training recipe

### 5.1 Starting hyperparameters

| Setting | 125M | 336M | 1.2B |
|---|---:|---:|---:|
| Tokens | 20B | 20–40B | 20–100B |
| Sequence | 2,048 | 2,048–4,096 | 4,096–8,192 |
| Batch tokens | 0.5M | 1M | 2M |
| Peak LR | 6e-4 | 3e-4 | 2e-4 |
| Min LR | 10% of peak | 10% | 10% |
| Warmup | 2% | 2% | 1–2% |
| Weight decay | 0.1 | 0.1 | 0.1 |
| Grad clip | 1.0 | 1.0 | 1.0 |
| AdamW β | 0.9 / 0.95 | 0.9 / 0.95 | 0.9 / 0.95 |
| Precision | bf16 | bf16 | bf16 |

These are starting points, not laws. At small scale, the LR can be higher; at large scale, lower. Use a short LR sweep on a 1B-token subset before the full run.

### 5.2 Expected loss

At initialisation, a uniform model has

\[
\mathcal{L}_0 \approx \ln V.
\]

For \(V=32{,}000\), that is ~10.4. A healthy run drops quickly toward 3–4 and then slowly. A loss stuck near \(\ln V\) means the model is not learning (LR, data, mask, or bug). A sudden spike means instability (LR, bad batch, numerical issue).

### 5.3 Training loop

Install the optional dependencies first:

```bash
uv pip install -r requirements-pretrain.txt
```

The reference implementation is:

```bash
python scripts/train_tiny_transformer.py \
  --data data/tiny/corpus.txt \
  --out runs/tiny \
  --dim 256 --layers 6 --heads 8 --kv-heads 2 --ffn 768 \
  --seq 256 --batch 16 --steps 2000 \
  --lr 3e-4 --eval-every 100
```

It implements byte-level tokens, RMSNorm, RoPE, GQA, SwiGLU, causal cross-entropy, AdamW, cosine schedule, gradient clipping, validation and sampling. It is a teaching implementation, not a substitute for a production trainer.

For a real run, use:

- **PyTorch FSDP** or **Megatron-LM/NeMo** for multi-GPU;
- **FlashAttention-2/3** for long sequences;
- **activation checkpointing** to trade compute for memory;
- **bf16** with fp32 master weights;
- **gradient accumulation** to reach the target batch;
- **distributed checkpointing** and resumable data loaders;
- **deterministic data order** per run for reproducibility.

### 5.4 Memory math

Training memory is roughly:

\[
\text{bytes} \approx
\underbrace{2N}_{\text{bf16 weights}}
+ \underbrace{12N}_{\text{AdamW fp32 states}}
+ \underbrace{2N}_{\text{gradients}}
+ \text{activations}.
\]

That is ~16 bytes/parameter before activations for full fine-tuning. A 1.2B model needs ~19 GB for weights/optimizer alone; activations add more. FSDP/ZeRO shards these states across GPUs. LoRA avoids most of it by freezing the base.

---

## 6. Evaluate before scaling

Minimum evaluation:

| Test | What it catches |
|---|---|
| Held-out perplexity | basic learning, data leak |
| MMLU / ARC / HellaSwag | general knowledge |
| GSM8K / MATH subset | reasoning |
| HumanEval / MBPP | code |
| Long-context needle | context extension |
| Toxicity / safety prompts | unsafe behaviour |
| Memorisation probes | recitation risk |

Report the tokenizer, data cutoff, contamination check and exact prompts. A small model’s MMLU will be low; that is expected. The question is whether the curve is healthy and whether the data pipeline works.

Plot loss vs tokens and fit a scaling law:

\[
L(N,D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}.
\]

If the curve is flat, more tokens will not save you; fix the data or the architecture.

!!! example "On paper"

    Suppose your small runs fit \(E=1.7\), \(A=5{,}000\), \(B=50{,}000\) and \(\alpha=\beta=0.5\). These are made-up round constants so the square roots come out exact; Chinchilla’s fit has \(\alpha\approx0.34\), \(\beta\approx0.28\). Start at 100M parameters and 2.5B tokens:

    \[
    L=1.7+\frac{5{,}000}{10^{4}}+\frac{50{,}000}{5\times10^{4}}=3.2
    \]

    Now spend 4× the compute (\(6ND\)) two ways:

    | Run | \(A/\sqrt N\) | \(B/\sqrt D\) | \(L\) |
    |---|---:|---:|---:|
    | 100M, 2.5B tokens | 0.5 | 1.0 | 3.20 |
    | 400M, 2.5B tokens | 0.25 | 1.0 | 2.95 |
    | 100M, 10B tokens | 0.5 | 0.5 | 2.70 |

    Here the data term is the larger one, so 4× the tokens beats 4× the parameters. No run gets below \(E=1.7\), the irreducible loss.

---

## 7. Post-train the small model

A base model is not an assistant. The minimum post-training:

1. **SFT** on 10k–100k high-quality instruction/response pairs.
2. **Preference optimisation** (DPO/ORPO) on chosen/rejected pairs.
3. **Verifiable RL** (GRPO) on math/code with exact answers.
4. **Safety tuning** and refusal/scope examples.
5. **Distillation** from a larger teacher if you have one.

Losses:

\[
\mathcal{L}_{\text{SFT}} =
-\sum_{t\in\text{completion}}\log p_\theta(x_t\mid x_{<t}),
\]

!!! example "On paper"

    Prompt `What is 2+2?`, completion `4<eos>`. The prompt tokens are masked; the model gives the two completion tokens probabilities \(\tfrac14\) and \(\tfrac12\). Use \(\ln2\approx0.693\).

    \[
    \mathcal{L}_{\text{SFT}}=-\big(\ln\tfrac14+\ln\tfrac12\big)=3\ln2\approx2.08
    \]

    However badly the model predicts the user’s words, they add nothing: the gradient only teaches the answer.

\[
\mathcal{L}_{\text{DPO}} =
-\log\sigma\left(
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
\right).
\]

!!! example "On paper"

    Take \(\beta=0.5\) and use \(\sigma(\ln a)=a/(1+a)\). At the start \(\pi_\theta=\pi_{\text{ref}}\), both log-ratios are 0 and \(\mathcal{L}_{\text{DPO}}=-\ln\sigma(0)=\ln2\approx0.693\). After some steps the policy makes the chosen answer twice as likely as the reference does, and the rejected one half as likely:

    \[
    \begin{aligned}
    z&=0.5\big(\ln2-\ln\tfrac12\big)=\ln2\\
    \sigma(\ln2)&=\tfrac{2}{1+2}=\tfrac23\\
    \mathcal{L}_{\text{DPO}}&=-\ln\tfrac23=\ln1.5\approx0.405
    \end{aligned}
    \]

    The loss falls only as the gap between chosen and rejected grows, and both sides are measured relative to the frozen reference.

GRPO is in [33 — How Qwen was built](33-qwen-creation.md) §4.4.

For a small model, SFT + DPO is usually enough; GRPO only makes sense when you have a verifier and a base that can already emit valid formats.

---

## 8. A realistic learning path

| Step | Model | Data | Hardware | Goal |
|---|---|---|---|---|
| 1 | 10M | 1B tokens | CPU/1 GPU | pipeline works |
| 2 | 125M | 10–20B | 1–8 GPUs | loss curve, tokenizer |
| 3 | 336M | 20–40B | 8 GPUs | scaling, eval |
| 4 | 1.2B | 20–100B | 8–64 GPUs | real base model |
| 5 | 3B | 100B+ | 64–256 GPUs | serious model |
| 6 | 12–14B | 10–36T | 1k–10k GPUs | frontier replication |

Do not skip steps. Each step exposes a different failure: data pipeline, tokenizer, stability, parallelism, evaluation.

---

## 9. What not to do

- **Do not train 12B from scratch on a Mac.** It is not a matter of patience; the memory and compute do not fit.
- **Do not train on raw web text without dedup/decontamination.** You will waste compute and overfit.
- **Do not compare loss across tokenizers.** Compare task metrics.
- **Do not use a 14B LR on a 100M model.** Scale the LR and batch.
- **Do not run RL before SFT.** Format errors make reward meaningless.
- **Do not report benchmark numbers without contamination checks.**
- **Do not ignore data licensing.** A model is a derivative of its data.
- **Do not skip small-scale ablations.** They are cheaper than a failed 1B run.

---

## 10. If you want “similar to Qwen/Gemma” without pretraining

The practical alternatives, in order of cost:

1. **LoRA/QLoRA on the released model** — this handbook. Best quality per dollar.
2. **Continued pretraining** on domain data — moderate cost, needs GPUs.
3. **Distillation** from a larger teacher into a smaller student — good for narrow domains.
4. **Model merging** — cheap, limited.
5. **From scratch** — only if you need a new tokenizer, architecture, language or license, and you have a real budget.

“Similar” usually means similar **behaviour**, not similar **initialisation**. Fine-tuning gets you behaviour; from-scratch gets you architecture control.

---

## 11. Sources

- Chinchilla — <https://arxiv.org/abs/2203.15556>
- Kaplan scaling laws — <https://arxiv.org/abs/2001.08361>
- Llama 2 — <https://arxiv.org/abs/2307.09288>
- GPT-3 — <https://arxiv.org/abs/2005.14165>
- The Stack v3 — <https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train>
- FineWeb — <https://huggingface.co/datasets/HuggingFaceFW/fineweb>
- SentencePiece — <https://github.com/google/sentencepiece>
- FlashAttention — <https://arxiv.org/abs/2205.14135>
- ZeRO — <https://arxiv.org/abs/1910.02054>
- Megatron-LM — <https://arxiv.org/abs/1909.08053>
- DPO — <https://arxiv.org/abs/2305.18290>
- [32 — Mathematics](32-mathematics.md)
- [36 — Reproducing 12B–14B](36-reproducing-12b-14b.md)
- [08 — Fine-tuning](08-finetuning.md)
