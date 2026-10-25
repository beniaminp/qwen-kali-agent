# 41 — Distillation training: what happens behind the loss

*Last verified: 25 October 2026. The MLX flags are checked against mlx-lm 0.32.2; the PyTorch reference is a GPU-oriented implementation, not a Mac default.*

This chapter is the mechanics. It follows one prompt from teacher generation to a saved adapter, explains every tensor and gradient, and gives the runnable paths:

1. **Sequence KD** with MLX-LM — the 32 GB Mac path.
2. **Logit KD** with `scripts/distill_logits.py` — the GPU path.

Read [39 — Gemma 12B distillation](39-gemma-distillation.md) for the direction and [40 — Teachers, students and data](40-distill-data.md) for the data contract.

---

## 1. The pipeline end to end

```text
prompts
  → teacher generation          (API or local model)
  → validation / filtering      (schema, compile, tests, scope)
  → canonical JSONL             (messages + tools)
  → chat template               (student template)
  → tokenization + mask         (prompt tokens = -100)
  → student forward             (base + LoRA)
  → loss                        (CE, or CE + KD)
  → backward                    (adapter gradients)
  → AdamW update                (adapter only)
  → checkpoint / fuse
  → evaluation                  (same golden set for teacher and student)
```

Each stage has a different failure mode. Most disappointing distillation runs fail at **data** or **evaluation**, not at the loss function.

---

## 2. The losses, precisely

### 2.1 Cross-entropy

For a response token \(y_t\) with student logits \(z_S\):

\[
\mathcal{L}_{\text{CE}}
=-\frac{1}{|\mathcal{C}|}\sum_{t\in\mathcal{C}}
\log \operatorname{softmax}(z_{S,t})_{y_t},
\]

where \(\mathcal{C}\) is the completion span. The gradient is

\[
\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_{S,t}}
=p_{S,t}-e_{y_t}.
\]

This is the same loss as ordinary SFT. Sequence KD is therefore **SFT on teacher-generated completions**, with all of SFT’s strengths and weaknesses.

### 2.2 Logit KD

With temperature \(T\) (call it \(T_{\text{temp}}\) to avoid confusion with sequence length):

\[
p_T^{T}=\operatorname{softmax}\left(\frac{z_T}{T_{\text{temp}}}\right),
\qquad
p_S^{T}=\operatorname{softmax}\left(\frac{z_S}{T_{\text{temp}}}\right),
\]

\[
\mathcal{L}_{\text{KD}}
=T_{\text{temp}}^2\,
\operatorname{KL}\left(p_T^{T}\;\|\;p_S^{T}\right).
\]

The total loss is

\[
\boxed{
\mathcal{L}=(1-\alpha)\mathcal{L}_{\text{CE}}
+\alpha\mathcal{L}_{\text{KD}}
}
\]

with \(\alpha\in[0,1]\).

Why \(T_{\text{temp}}^2\): the gradient of the KL with respect to \(z_S\) scales as \(1/T_{\text{temp}}^2\), so multiplying by \(T_{\text{temp}}^2\) keeps the KD gradient comparable to CE when you change the temperature. Without it, higher temperatures silently reduce the KD learning rate.

### 2.3 Top-k KD

A full Gemma vocabulary is 262,144 tokens. The tail is usually irrelevant and expensive. Keep the teacher’s top-\(k\) tokens:

\[
\mathcal{T}=\operatorname{TopK}(z_T,k),
\qquad
q_i=\frac{\exp(z_{T,i}/T_{\text{temp}})}
{\sum_{j\in\mathcal{T}}\exp(z_{T,j}/T_{\text{temp}})},
\]

\[
\mathcal{L}_{\text{top-k}}
=-\sum_{i\in\mathcal{T}}q_i\log p_{S,i}^{T}.
\]

Notes:

- renormalizing over the top-\(k\) makes \(q\) a proper distribution;
- \(k=32\)–\(128\) is typical;
- \(k=0\) means full-vocabulary KL (memory-heavy);
- top-k is an approximation: the student is not penalized for putting mass on a token outside the teacher’s top-\(k\), which is usually fine but can matter for safety-critical near-misses.

### 2.4 Sequence KD is a special case

If you set \(\alpha=0\) and replace the hard label \(y_t\) with a teacher-sampled token \(\tilde y_t\), you get

\[
\mathcal{L}_{\text{seq}}
=-\sum_t \log p_S(\tilde y_t\mid \tilde y_{<t},x),
\qquad
\tilde y\sim p_T.
\]

That is sequence KD. It uses only hard labels and no teacher logits. It is what `mlx_lm.lora` can do today.

### 2.5 On-policy KD

Sequence KD trains on teacher prefixes. At inference, the student sees its own prefixes, including mistakes. On-policy KD (DAgger-style) fixes the state distribution:

1. student rolls out \(\hat y\sim p_S\);
2. teacher labels each visited state (correct token, continuation, or score);
3. student trains on \(\hat y_{<t}\rightarrow y_t^*\);
4. repeat with a fresh rollout.

The loss is the same CE/KD, but the prefixes come from the student. This is more expensive and can be unstable if the teacher corrections are noisy; use a verifier.

### 2.6 Feature KD (optional)

For a white-box pair with hidden states \(h_T^l,h_S^l\):

\[
\mathcal{L}_{\text{feat}}
=\sum_{l\in\mathcal{L}}
\left\|h_S^l-W_l h_T^l\right\|_2^2.
\]

You need a projection \(W_l\), aligned layers, and both models in memory. This is a research direction, not the first run.

---

## 3. Inside one training step

Take a single row:

```json
{"messages":[{"role":"user","content":"..."},
             {"role":"assistant","content":"..."}]}
```

### Step 1 — render

The student tokenizer applies its chat template:

```text
<|turn>user
...
<turn|>
<|turn>model
...
<turn|>
```

The exact tokens depend on the template. For Gemma 4 12B/31B, thinking-off still includes an empty thought block; for E4B/E2B it does not.

### Step 2 — tokenize and mask

The prompt is tokenized separately:

\[
\text{prompt\_ids}=[x_1,\dots,x_P],
\qquad
\text{full\_ids}=[x_1,\dots,x_P,y_1,\dots,y_R].
\]

The labels are

\[
\ell_t=
\begin{cases}
-100, & t\le P,\\
y_t, & t>P.
\end{cases}
\]

`-100` is PyTorch/MLX’s ignore index; those positions do not contribute to the loss. This is exactly `mask_prompt: true` in MLX.

### Step 3 — forward

The student embeds the tokens, runs \(L\) blocks, and produces logits:

\[
z_S\in\mathbb{R}^{B\times T\times V}.
\]

For Gemma 4 12B, \(V=262{,}144\); at \(B=1,T=2048\), that is 537M values. In bf16, ~1.07 GB just for the logits tensor. This is why long-context training is memory-bound even before attention.

### Step 4 — loss

For each response position:

- CE compares \(z_S\) with the hard token;
- KD compares \(z_S/T_{\text{temp}}\) with the teacher’s \(z_T/T_{\text{temp}}\).

The loss is averaged over response positions, not over the whole sequence.

### Step 5 — backward

Autograd computes

\[
\frac{\partial\mathcal{L}}{\partial\theta}
\]

for every trainable parameter. With LoRA, the base weights are frozen and only \(A,B\) adapters receive gradients. With QLoRA, the base is 4-bit and frozen; the adapters are higher precision.

### Step 6 — optimizer

AdamW updates the adapters:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\]

\[
\theta_t=\theta_{t-1}-\eta_t
\left(
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
+\lambda\theta_{t-1}
\right).
\]

Gradient accumulation sums \(G\) micro-batches before one optimizer step. The effective batch is

\[
B_{\text{eff}}=B_{\text{micro}}\times G.
\]

### Step 7 — checkpoint

MLX saves the adapter tensors (`adapters.safetensors`) plus the config. The base model is unchanged. At inference, the adapter is applied to the frozen base.

---

## 4. What LoRA is doing mathematically

A frozen weight \(W\in\mathbb{R}^{d\times k}\) is augmented with a low-rank update:

\[
W'=W+\frac{\alpha}{r}BA,
\qquad
B\in\mathbb{R}^{d\times r},\quad
A\in\mathbb{R}^{r\times k}.
\]

Only \(A,B\) train. The rank \(r\) controls capacity; \(\alpha/r\) controls the update scale.

In MLX-LM, the config field is `scale`, and the effective update is

\[
W'=W+\text{scale}\cdot BA.
\]

The PEFT convention is

\[
W'=W+\frac{\alpha_{\text{PEFT}}}{r}BA.
\]

So

\[
\text{scale}=\frac{\alpha_{\text{PEFT}}}{r}.
\]

For rank 16 and PEFT alpha 32, `scale: 2.0`. This is the trap in the older Qwen configs: `scale: 20` was not “alpha 20”; it was a 10× larger update than a standard rank-16 recipe. See [17 — Fine-tuning update](17-finetune-2026-update.md).

### 4.1 Which modules

For Gemma 4, adapt both attention and MLP projections:

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

Attention-only adapters are cheaper but often underfit a new domain. The MLP holds most of the parameters; adapting it matters.

### 4.2 QLoRA

The base is quantized to 4-bit and frozen; adapters are trained in bf16/fp16. This cuts weight memory ~4× at the cost of some quality and throughput. For a 12B on 32 GB, it is the only practical full-model option.

---

## 5. Sequence KD with MLX-LM

This is the recommended 32 GB path.

### 5.1 Prepare the data

```bash
python scripts/generate_distill_data.py \
  --prompts data/distill/prompts.jsonl \
  --system-file examples/distill/teacher-system.md \
  --model deepseek-flash \
  --out data/distill/raw/deepseek.jsonl \
  --n 2 --temperature 0.8 --max-tokens 2048

python scripts/validate_gemma4_dataset.py \
  data/distill/raw/deepseek.jsonl \
  --tools examples/distill/tools.json --strict

python scripts/prepare_dataset.py \
  data/distill/raw/deepseek.jsonl \
  --out data/distill/12b/processed \
  --arguments-format dict \
  --split 0.9 0.05 0.05
```

### 5.2 Check the token budget

```bash
python scripts/token_stats.py \
  data/distill/12b/processed/train.jsonl \
  --model google/gemma-4-12B-it
```

The 95th percentile should fit `max_seq_length`. If it does not, either truncate long prompts or raise the sequence length at the cost of memory.

### 5.3 Render one row

```bash
python scripts/smoke_test.py \
  --data data/distill/12b/processed/train.jsonl \
  --model google/gemma-4-12B-it
```

Look for:

- the prompt/response boundary;
- the assistant end token;
- tool-call syntax;
- a response that is not empty;
- no duplicated `<bos>`.

### 5.4 Choose a config

| Student | Config | Layers | Seq | Effective batch |
|---|---|---:|---:|---:|
| Gemma 4 12B | `configs/gemma4-12b-distill.yaml` | 24/48 | 2048 | 8 |
| Gemma 4 12B long | `configs/gemma4-12b-distill-longctx.yaml` | 16/48 | 4096 | 8 |
| Gemma 4 E4B | `configs/gemma4-e4b-distill.yaml` | all 42 | 2048 | 8 |
| Gemma 4 E2B | `configs/gemma4-e2b-distill.yaml` | all 35 | 2048 | 8 |

### 5.5 Verify LoRA scope

```bash
python scripts/check_lora.py \
  --model mlx-community/gemma-4-12b-it-4bit \
  --num-layers 24 --rank 16 --scale 2
```

You want attention **and** MLP projections, and a trainable parameter count consistent with the rank.

### 5.6 Train

```bash
mlx_lm.lora --config configs/gemma4-12b-distill.yaml
```

What MLX does per iteration:

1. samples a row;
2. applies the tokenizer’s chat template;
3. tokenizes to `max_seq_length`;
4. masks the prompt if `mask_prompt: true`;
5. forwards the frozen 4-bit base + adapters;
6. computes cross-entropy on the completion;
7. backpropagates into the adapters;
8. accumulates gradients;
9. steps AdamW;
10. reports loss;
11. validates every `steps_per_eval`;
12. saves the adapter every `save_every`.

It does **not** load a teacher. The “teacher” is already baked into the assistant tokens. That is why sequence KD is cheap.

### 5.7 What the loss means

- Starts around \(\ln V\) if the model is random; for a pretrained Gemma, much lower.
- Drops as the adapter learns the teacher’s style.
- A rising validation loss means overfitting or a bad split.
- `nan` means zero-length completion, truncation, bad mask or LR too high.
- A flat loss means the adapter is not attached, the LR is too low, or the data is already in-distribution.

Do not compare this loss to a logit-KD loss; the objectives are different.

### 5.8 Fuse and serve

```bash
mlx_lm.fuse \
  --model mlx-community/gemma-4-12b-it-4bit \
  --adapter-path adapters/gemma4-12b-distill \
  --save-path fused/gemma4-12b-distill-4bit

mlx_lm.server --model fused/gemma4-12b-distill-4bit \
  --host 127.0.0.1 --port 8080 --max-tokens 4096
```

Verify tool calls with curl before wiring an agent. If MLX’s parser lags the template, serve the GGUF with `llama-server --jinja`.

---

## 6. Logit KD with the reference trainer

`scripts/distill_logits.py` is a **GPU reference**. It is not the MLX path and it is not tested on the target Mac; treat it as a readable implementation of the equation.

Install the optional dependencies on the GPU host:

```bash
uv pip install -r requirements-distill.txt
```

### 6.1 What it does

For each batch:

1. render `messages` with the **student** tokenizer;
2. build `input_ids` and `labels` with prompt masking;
3. run the teacher with `torch.no_grad()`;
4. run the student with gradients;
5. compute CE on hard labels;
6. compute top-\(k\) KL on the response positions;
7. combine with \(\alpha\);
8. backprop into LoRA adapters;
9. save the adapter + `distill_config.json`.

### 6.2 Why it checks the vocabulary

Before training, it compares `teacher_tokenizer.get_vocab()` with the student’s. If they differ, logit KD is invalid and the script exits. This is deliberate: silently comparing logits over different token IDs is a common and expensive bug.

### 6.3 Run it

Single GPU, 12B teacher → E4B student:

```bash
python scripts/distill_logits.py \
  --teacher google/gemma-4-12B-it \
  --student google/gemma-4-E4B-it \
  --data data/distill/e4b/train.jsonl \
  --out adapters/gemma4-e4b-distill \
  --device cuda:0 --dtype bf16 \
  --max-seq-length 1024 \
  --top-k 64 --temperature 2.0 --alpha 0.6 \
  --lora-r 16 --grad-checkpoint
```

Two GPUs, 31B teacher → 12B student:

```bash
python scripts/distill_logits.py \
  --teacher google/gemma-4-31B-it \
  --teacher-device cuda:1 \
  --student google/gemma-4-12B-it \
  --device cuda:0 \
  --data data/distill/12b/train.jsonl \
  --out adapters/gemma4-12b-distill \
  --top-k 64 --temperature 2.0 --alpha 0.5 \
  --max-seq-length 1024 --lora-r 16 --grad-checkpoint
```

Dry run (no models loaded):

```bash
python scripts/distill_logits.py \
  --teacher google/gemma-4-31B-it \
  --student google/gemma-4-12B-it \
  --data data/distill/12b/train.jsonl \
  --out /tmp/distill-dry --dry-run
```

### 6.4 Memory rules

- `--load-in-4bit` uses bitsandbytes, CUDA only.
- A 4-bit teacher produces approximate logits; if you can afford bf16, use it.
- `--top-k 64` is a good default; `0` uses full-vocab KL and is much heavier.
- `--max-seq-length 1024` is the first knob for OOM; do not start at 4K.
- The teacher is frozen and `no_grad`; only the student adapter trains.
- The teacher and student can be on different GPUs; the script moves only the top-k indices/probabilities.

### 6.5 What the script does not do

- It is not distributed; for multi-node, use FSDP/DeepSpeed.
- It does not stream a huge dataset; it loads JSONL into memory.
- It does not implement feature KD.
- It does not implement on-policy rollouts.
- It does not support MLX.

Those are deliberate boundaries. For 32 GB, use sequence KD.

---

## 7. On-policy distillation

### 7.1 The loop

```text
student checkpoint
  → sample prompts
  → student rollout (temperature 0.7)
  → teacher scores / corrects / continues
  → verifier (compile, tests, scope, tool schema)
  → keep accepted corrections
  → SFT / logit KD on student prefixes
  → new student checkpoint
```

### 7.2 What the teacher does

- **correction:** given the student’s prefix, produce the next correct token/segment;
- **continuation:** let the teacher finish from the student’s state;
- **scoring:** assign a scalar to the rollout; use it for rejection or preference pairs;
- **critique:** produce a natural-language correction, then a corrected answer.

### 7.3 Why it helps

The student learns to recover from its own errors. This is exactly what tool-using agents need: a bad tool call, a failed test, a wrong assumption, then a safe recovery.

### 7.4 Cost

Every iteration needs student generation + teacher inference + verification. It is 3–10× the cost of sequence KD. Use it only after sequence KD has plateaued and you have a reliable verifier.

---

## 8. Hyperparameters

### 8.1 Sequence KD

| Setting | 12B | E4B | E2B |
|---|---:|---:|---:|
| Rank | 16 | 16 | 16 |
| `scale` | 2.0 | 2.0 | 2.0 |
| LR | 8e-5 | 1e-4 | 1.5e-4 |
| Effective batch | 8 | 8 | 8 |
| Sequence | 2048 | 2048 | 2048 |
| Iterations | 300 | 300 | 300 |
| Epochs (approx.) | 1–2 | 1–2 | 1–2 |
| Warmup | 30 | 30 | 30 |
| Weight decay | 0.01 | 0.01 | 0.01 |

### 8.2 Logit KD

| Setting | Start | Range |
|---|---:|---:|
| Temperature | 2.0 | 1.5–4.0 |
| \(\alpha\) | 0.5 | 0.3–0.7 |
| Top-k | 64 | 32–128 |
| LR | 5e-5 | 2e-5–1e-4 |
| Effective batch | 8 | 4–16 |
| Sequence | 1024 | 512–2048 |
| Epochs | 1 | 1–2 |

Logit KD is more sensitive to LR than sequence KD because the soft-label gradient is dense. Start lower.

### 8.3 How to tune

1. Fix the data and the student.
2. Run 50–100 steps at 3 LRs (e.g. 5e-5, 1e-4, 2e-4 for sequence KD).
3. Evaluate on 100 held-out prompts with the task verifier.
4. Pick the best, then run the full job.
5. Do not tune \(\alpha\) and LR at the same time.

---

## 9. Throughput and wall clock

Use measured tokens/sec, not theory.

\[
\text{wall time}=\frac{\text{tokens}}{\text{tokens/sec}}.
\]

Rough M1 Pro anchors from [02 — Hardware](02-hardware.md):

| Student | Tokens/sec (EST) | 5M tokens | 10M tokens |
|---|---:|---:|---:|
| Gemma 4 12B QLoRA | 20–30 | 46–69 h | 92–139 h |
| Gemma 4 E4B QLoRA | 50–80 | 17–28 h | 35–56 h |
| Gemma 4 E2B QLoRA | 80–120 | 12–17 h | 23–35 h |

Add 10–20% for data loading, validation, checkpoints and thermal throttling. A 12B sequence-KD run is a multi-day job; do a 40-iteration smoke run first.

---

## 10. Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss `nan` | zero-length completion; truncation; LR too high | filter long rows; check mask; lower LR |
| Loss flat near \(\ln V\) | adapter not attached; wrong data path; template mismatch | `check_lora.py`; render one row; inspect dataset |
| Train loss down, val up | overfitting; duplicated rows | more data; dedup; fewer epochs |
| Student copies teacher quirks | low temperature; only one sample; no filter | diversify; verify; mix human data |
| Student refuses benign requests | refusal over-representation | rebalance; add benign examples |
| Tool calls malformed | arguments string/object mismatch | `--arguments-format dict`; validate |
| Student ignores tools | tool data too rare; schema too long | raise tool share; shorten schemas |
| Output too long | teacher verbosity; max tokens | length filter; style examples |
| E4B worse than base | data too narrow or LR too high | mix general data; lower LR; evaluate base |
| KD loss dominates | \(\alpha\) too high; teacher noisy | lower \(\alpha\); use top-k; verify teacher |
| Teacher/student mismatch | different tokenizer/template | sequence KD; align template |

---

## 11. Cost and time

### 11.1 Teacher cost

At DeepSeek Flash off-peak pricing (USD per 1M cache-hit/miss/output tokens; check current pricing):

| Tokens | Approx cost |
|---|---:|
| 1M in / 1M out | ~$0.75 |
| 10M in / 10M out | ~$7.50 |
| 50M in / 50M out | ~$37.50 |

For 10k prompts × 1k output tokens ≈ 10M output tokens, plus prompt tokens. Add retries and rejected samples. Use [40 — Teachers, students and data](40-distill-data.md) for volume.

### 11.2 GPU cost

- 12B → E4B logit KD on one 80 GB GPU: roughly 20–60 GPU-hours for 5M tokens, depending on MFU.
- 31B → 12B logit KD on two 80 GB GPUs: roughly 40–120 GPU-hours.
- Rent for the shortest job that answers the question; do not rent for a multi-day run before a smoke test.

### 11.3 Mac cost

Electricity and time. A 12B sequence-KD run is 2–3 days; E4B is under a day. Run E4B first to validate the data, then decide whether 12B is worth it.

---

## 12. Sources

- Distilling the Knowledge in a Neural Network — <https://arxiv.org/abs/1503.02531>
- Sequence-Level Knowledge Distillation — <https://arxiv.org/abs/1606.07947>
- DAgger — <https://arxiv.org/abs/1011.0686>
- LoRA — <https://arxiv.org/abs/2106.09685>
- QLoRA — <https://arxiv.org/abs/2305.14314>
- AdamW — <https://arxiv.org/abs/1711.05101>
- Gemma 4 technical report — <https://arxiv.org/abs/2607.02770>
- MLX-LM LoRA docs — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md>
- [02 — Hardware](02-hardware.md)
- [17 — Fine-tuning update](17-finetune-2026-update.md)
- [39 — Gemma 12B distillation](39-gemma-distillation.md)
- [40 — Teachers, students and data](40-distill-data.md)
- [42 — Distillation evaluation](42-distill-eval.md)
