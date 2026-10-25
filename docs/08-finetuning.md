# 08 — Fine-tuning with MLX: the full runbook

*Last verified: 25 October 2026 against mlx-lm 0.32.2. Commands were checked against `mlx_lm/lora.py` and `LORA.md`; they were not executed on the target Mac in this environment.*

This is the chapter you run. It assumes:

- macOS with MLX installed ([`03-macos-setup.md`](03-macos-setup.md));
- a validated dataset in `data/processed/{train,valid,test}.jsonl` ([`06-data-format.md`](06-data-format.md));
- 32 GB unified memory and a base model that fits ([`02-hardware.md`](02-hardware.md)).

The default path is **Qwen3-14B 4-bit QLoRA**. The 8B path is for debugging. Do not start with the 14B.

---

## 0. Preflight (10 minutes)

```bash
cd ~/work/qwen-kali-agent
source .venv/bin/activate

python scripts/check_env.py
python scripts/validate_dataset.py data/processed/train.jsonl --tools examples/tools/kali-tools.json
python scripts/validate_dataset.py data/processed/valid.jsonl --tools examples/tools/kali-tools.json
python scripts/token_stats.py data/processed/train.jsonl --model Qwen/Qwen3-14B
```

Checklist:

- [ ] `mx.device_info()` reports `gpu`/`metal`.
- [ ] `max_recommended_working_set_size` is ~21–22 GiB (32 GB machine).
- [ ] train/valid/test exist and are non-empty.
- [ ] 99th-percentile token length is ≤ your planned `max_seq_length`.
- [ ] every tool call names a tool in `tools`;
- [ ] no `nan`-triggering rows (prompt ≥ max sequence).
- [ ] 100–150 GB free disk.
- [ ] Kali VM and Docker Desktop are **shut down**.
- [ ] charger connected; sleep disabled.
- [ ] you have a baseline evaluation of the *base* model (see §8).

---

## 1. What MLX-LM actually optimizes

- **LoRA** freezes the base weights and injects low-rank matrices into selected linear layers.
- **QLoRA** is the same, but the base is a 4-bit quantized MLX model. `mlx_lm.lora` detects a quantized model and does this automatically.
- By default, LoRA is applied to the **last `num_layers` transformer blocks**, to every linear layer in those blocks (q, k, v, o, gate, up, down). `--num-layers -1` applies to all blocks.
- Only adapter weights and optimizer state are trained. The base model is read-only.
- `--mask-prompt` computes loss only on the final assistant completion. Everything before it is context.

The default adapter is tiny: rank 8, 16 layers on a 14B model ≈ 13M trainable parameters (~50 MB fp32). Rank 16 doubles it. The adapter is not the memory problem; **activations and logits are**.

---

## 2. Every flag that matters

From `mlx_lm.lora --help` and `mlx_lm/lora.py`:

| Flag | Default | What to set | Notes |
|---|---|---|---|
| `--model` | `Qwen/Qwen3-0.6b` | `mlx-community/Qwen3-14B-4bit` | A 4-bit MLX repo triggers QLoRA. |
| `--train` | false | `true` | Required to train. |
| `--data` | WikiSQL | `data/processed` | Directory with `train.jsonl`, `valid.jsonl`, `test.jsonl`. |
| `--fine-tune-type` | `lora` | `lora` | `dora`, `full` also exist; full does not fit 14B. |
| `--optimizer` | `adam` | `adamw` | `adam`, `adamw`, `muon`, `sgd`, `adafactor`. |
| `--num-layers` | 16 | 8–16 | Last N blocks. `-1` = all; more layers = more memory/time. |
| `--batch-size` | 4 | 1–2 | Per-step micro-batch. Start at 1. |
| `--iters` | 1000 | computed | One iteration = one batch, not one epoch. |
| `--val-batches` | 25 | 25–50 | `-1` uses the whole validation set (slow). |
| `--learning-rate` | 1e-5 | 5e-5–1e-4 | LoRA tolerates higher LR than full FT. |
| `--steps-per-report` | 10 | 10 | Console cadence. |
| `--steps-per-eval` | 200 | 100 | Validation cadence. |
| `--grad-accumulation-steps` | 1 | 4–8 | Effective batch without activation memory. |
| `--resume-adapter-file` | — | path | Loads weights only, not optimizer/iteration state. |
| `--adapter-path` | `adapters` | `adapters/qwen3-14b-pentest` | Output directory. |
| `--save-every` | 100 | 100–200 | Saves `NNNNNNN_adapters.safetensors` plus `adapters.safetensors`. |
| `--test` | false | after training | Evaluates test loss after training. |
| `--test-batches` | 500 | -1 or 100 | Cap test evaluation. |
| `--max-seq-length` | 2048 | 2048–4096 | Truncates from the right. |
| `--grad-checkpoint` | false | **true** | Biggest memory win; ~20–30% slower. |
| `--clear-cache-threshold` | 0 | e.g. `2GB` | Clears allocator cache between steps if it grows. |
| `--mask-prompt` | false | **true** | Train only on the final assistant turn. |
| `--report-to` | — | `wandb`/`swanlab` | Optional experiment tracking. |
| `--seed` | 0 | 42 | Reproducibility (not bit-exact on GPU). |
| `-c`, `--config` | — | YAML path | CLI flags override YAML. |

**YAML-only settings** (not exposed as CLI flags): `lora_parameters` (`rank`, `dropout`, `scale`), `lr_schedule`, `optimizer_config`.

`scale` in MLX is **not** `alpha / rank`. The layer computes `y + scale * (x @ A @ B)`. The default `scale: 20.0` at rank 8 is roughly PEFT `alpha=160`. Do not copy a PEFT config (`lora_alpha: 16`) into MLX without converting.

---

## 3. The three configs

The repository ships them in `configs/`. Use them as starting points, not sacred numbers.

### 3.1 8B fast iteration (`configs/qwen3-8b-fast.yaml`)

```yaml
model: mlx-community/Qwen3-8B-4bit
train: true
data: data/processed
fine_tune_type: lora
optimizer: adamw
num_layers: 12
batch_size: 2
iters: 400
val_batches: 25
learning_rate: 1.0e-4
steps_per_report: 10
steps_per_eval: 100
save_every: 100
max_seq_length: 2048
grad_checkpoint: true
grad_accumulation_steps: 4
mask_prompt: true
adapter_path: adapters/qwen3-8b-pentest
seed: 42
lora_parameters:
  rank: 16
  dropout: 0.05
  scale: 20.0
lr_schedule:
  name: cosine_decay
  warmup: 50
  arguments: [1.0e-4, 400, 1.0e-6]
optimizer_config:
  adamw:
    weight_decay: 0.01
```

Effective batch = 2 × 4 = 8. At ~1,500 tokens/sample that is ~12k tokens per optimizer step; 400 steps ≈ 4.8M tokens seen (about 2–3 epochs over a 2M-token set). At 40–55 tok/s, expect **25–35 hours**. For a 100k-token smoke test, set `iters: 50` and use a 100k-token subset.

### 3.2 14B balanced (`configs/qwen3-14b-balanced.yaml`)

```yaml
model: mlx-community/Qwen3-14B-4bit
train: true
data: data/processed
fine_tune_type: lora
optimizer: adamw
num_layers: 16
batch_size: 1
iters: 600
val_batches: 25
learning_rate: 5.0e-5
steps_per_report: 10
steps_per_eval: 100
save_every: 100
max_seq_length: 2048
grad_checkpoint: true
grad_accumulation_steps: 8
mask_prompt: true
adapter_path: adapters/qwen3-14b-pentest
seed: 42
lora_parameters:
  rank: 16
  dropout: 0.05
  scale: 20.0
lr_schedule:
  name: cosine_decay
  warmup: 80
  arguments: [5.0e-5, 600, 1.0e-6]
optimizer_config:
  adamw:
    weight_decay: 0.01
```

Effective batch 8. At ~1,500 tokens/sample, 600 steps ≈ 7.2M tokens seen. At 20–30 tok/s, **65–100 hours**. If that is too long, cut `iters` to 300 and evaluate; more data is not automatically better.

### 3.3 14B conservative (`configs/qwen3-14b-conservative.yaml`)

```yaml
model: mlx-community/Qwen3-14B-4bit
train: true
data: data/processed
fine_tune_type: lora
optimizer: adamw
num_layers: 8
batch_size: 1
iters: 400
val_batches: 25
learning_rate: 5.0e-5
steps_per_report: 10
steps_per_eval: 100
save_every: 100
max_seq_length: 1536
grad_checkpoint: true
grad_accumulation_steps: 8
mask_prompt: true
adapter_path: adapters/qwen3-14b-conservative
seed: 42
lora_parameters:
  rank: 8
  dropout: 0.05
  scale: 20.0
lr_schedule:
  name: cosine_decay
  warmup: 50
  arguments: [5.0e-5, 400, 1.0e-6]
```

Use this when the balanced config OOMs or the machine is shared. It trains less capacity (last 8 blocks) and shorter sequences; quality will be lower, but it finishes.

### 3.4 How many iterations?

```text
tokens_seen ≈ iters × batch_size × grad_accumulation_steps × avg_tokens_per_sample
iters ≈ epochs × dataset_tokens / (batch_size × grad_accumulation_steps × avg_tokens_per_sample)
```

Example: 2M-token dataset, avg 1,200 tokens, effective batch 8, 3 epochs:

```text
iters = 3 × 2,000,000 / (8 × 1,200) ≈ 625
```

Round to 600. If you change `max_seq_length`, re-measure `avg_tokens_per_sample`.

---

## 4. Hyperparameter guidance

### Rank and scale

| Rank | Trainable params (14B, 16 layers) | Use |
|---|---:|---|
| 8 | ~13M | format/domain adaptation; least memory |
| 16 | ~27M | **default** for this handbook |
| 32 | ~53M | more capacity; more overfit risk; 14B still fits |
| 64 | ~107M | only with a large, clean dataset |

Scale 20 is the MLX default and works with LR 5e-5–1e-4. If you increase rank, you do **not** automatically increase scale (unlike PEFT alpha); keep 16–20 unless you see underfitting.

### Learning rate

- 8B: `1e-4` is a good start.
- 14B: `5e-5` is safer; `1e-4` can work with warmup and cosine decay.
- If loss spikes or becomes `nan`: halve the LR, verify there are no empty completions, then retry.
- If loss plateaus high (>0.9) after 200 steps: the data may be too diverse, the LR too low, or the mask is wrong.

### Layers

More adapted layers = more capacity and more activation memory. 8 layers is a memory compromise; 16 is the default; 24–40 is only for a large, clean dataset on a 64 GB machine.

### Dropout

`0.05` is a reasonable regularizer for 2–5M tokens. For a very small dataset (<500k tokens), use `0.1`; for >10M tokens, `0.0–0.05`.

### Batch and accumulation

Start `batch_size: 1`, `grad_accumulation_steps: 8`. Increase batch only after measuring peak memory. Effective batch 8–16 is usually enough for LoRA SFT.

### Sequence length

2048 is the default. Agentic trajectories with tool schemas often need 3–4K. Measure the 99th percentile. Going to 4096 roughly doubles activation memory; use gradient checkpointing and batch 1.

---

## 5. Launch and read the logs

```bash
mlx_lm.lora --config configs/qwen3-8b-fast.yaml
```

Expected first lines:

```text
Loading pretrained model
Loading datasets
Training
Trainable parameters: 0.xxx% (xx.xxxM/8.xxxB)
Iter 1: Val loss ..., Val took ...s
Iter 10: Train loss ..., Learning Rate ..., It/sec ..., Tokens/sec ..., Trained Tokens ..., Peak mem ... GB
```

What matters:

| Metric | Healthy | Warning |
|---|---|---|
| Train loss | falls to 0.3–0.7 | stuck >1.0 after 200 iters |
| Val loss | tracks train, falls | rises for 2+ evals = overfit |
| Tokens/sec | 20–30 (14B), 40–55 (8B) | <10 = memory pressure/thermal |
| Peak mem | 16–22 GB (14B) | >26 GB = danger |
| It/sec | stable | collapses = swap/thermal |

**Do not judge by loss alone.** A model can reach 0.3 loss by memorizing tool names and still fail on held-out tasks. Evaluate at least once mid-run.

### Interpreting loss

- **1.5 → 0.6** over 200 steps: normal.
- **0.6 → 0.3** and val **0.7 → 0.6**: good.
- **Train 0.3, val 0.9**: overfitting or distribution mismatch.
- **<0.2**: suspicious; check for duplicates/contamination and whether the model is copying the prompt.
- **`nan`**: prompt length ≥ max sequence with `--mask-prompt` (empty completion), or LR too high. Check `token_stats` first.

---

## 6. Checkpoints and resume

Every `save_every` iterations, MLX writes:

```text
adapters/qwen3-14b-pentest/
├── adapter_config.json
├── adapters.safetensors
├── 0000100_adapters.safetensors
├── 0000200_adapters.safetensors
...
```

Resume:

```bash
mlx_lm.lora --config configs/qwen3-14b-balanced.yaml \
  --resume-adapter-file adapters/qwen3-14b-pentest/0000300_adapters.safetensors \
  --iters 300 \
  --adapter-path adapters/qwen3-14b-pentest-resume
```

**Caveats:**

- Only weights are loaded. Optimizer state, LR schedule position, and iteration count are **not** restored.
- `--iters` is the number of iterations for this invocation, not the global target. The schedule restarts; use a fresh cosine schedule for the remaining run.
- Save to a new adapter path to avoid overwriting the checkpoint you resumed from.
- Clean old checkpoints (`find adapters -name '*_adapters.safetensors'`) — each is ~50–110 MB, and a long run can accumulate several GB.

---

## 7. Memory troubleshooting

| Symptom | Fix |
|---|---|
| `kIOGPUCommandBufferCallbackErrorOutOfMemory` | batch 1, `--grad-checkpoint`, seq 1536, `--num-layers 8`, close apps. |
| `[metal::malloc] Resource limit (499000) exceeded` | descriptor/handle leak; restart the process, update MLX, reduce layers/seq, `--clear-cache-threshold 2GB`. |
| Process killed, no traceback | wired-memory pressure; lower `iogpu.wired_limit_mb` or memory use; reboot. |
| Swapouts climbing | close Chrome/Docker/VM; reduce batch/seq; do not continue. |
| Very slow after 30 min | thermal throttling; improve cooling, lower sustained load. |
| Loss `nan` at step 1 | a prompt is ≥ max sequence; check `token_stats`; drop/split rows. |

For a deeper explanation, see [`02-hardware.md`](02-hardware.md) §6.

---

## 8. Baseline and mid-run evaluation

Before training, save the base model's outputs on 50 golden tasks:

```bash
python scripts/eval_toolcall.py \
  --base-url http://127.0.0.1:8080/v1 \
  --model mlx-community/Qwen3-14B-4bit \
  --golden data/golden/golden.jsonl \
  --out reports/base.json
```

Serve the base model with `mlx_lm.server` in another terminal, or point the script at a running server. Then evaluate the adapter mid-run and at the end. A adapter that does not beat the base on **tool validity** and **task success** is not ready, regardless of loss.

---

## 9. DoRA, full fine-tuning, and newer architectures

- **DoRA** (`--fine-tune-type dora`) decomposes the update into magnitude and direction. It can improve quality at low rank, but is slower and less tested in MLX. Try it only after a LoRA baseline.
- **Full fine-tuning** (`--fine-tune-type full`) unfreezes the last `num_layers` blocks. A 14B full fine-tune needs tens of GB for optimizer state; it does **not** fit 32 GB. Do not attempt it.
- **Qwen3.5/3.6/3.8 LoRA** is not production-safe on this hardware as of Oct 2026 (GDN backward path). See [`01-landscape.md`](01-landscape.md). If you must, train on a rented CUDA GPU and convert the merged weights.
- **MoE models** (Qwen3-Coder-30B-A3B, Qwen3.6-35B-A3B) need special handling; memory is dominated by weights, and MLX's MoE LoRA support has had routing bugs. Treat them as inference-only on 32 GB.

---

## 10. Experiment tracking

Minimum viable record for every run:

```yaml
run_id: 2026-10-25-qwen3-14b-r1
base_model: mlx-community/Qwen3-14B-4bit
dataset_version: 2026-10-24.v1
dataset_tokens: 2.1M
config: configs/qwen3-14b-balanced.yaml
iters: 600
effective_batch: 8
max_seq_length: 2048
peak_mem_gb: 19.4
tokens_per_sec: 24.1
wall_clock_h: 71
final_train_loss: 0.41
final_val_loss: 0.52
eval_tool_validity: 0.94
eval_task_success: 0.61
notes: "First 14B run; recovery examples underrepresented."
```

Use `--report-to wandb` if you want curves, but keep a local YAML/JSON record as well. W&B is optional; reproducibility is not.

---

## 11. The full runbook

```bash
# 1. Environment
source .venv/bin/activate
python scripts/check_env.py

# 2. Data
python scripts/validate_dataset.py data/processed/train.jsonl --tools examples/tools/kali-tools.json
python scripts/token_stats.py data/processed/train.jsonl --model Qwen/Qwen3-14B

# 3. Smoke test on 8B (hours)
mlx_lm.lora --config configs/qwen3-8b-fast.yaml

# 4. Evaluate the 8B adapter
mlx_lm.fuse --model mlx-community/Qwen3-8B-4bit \
  --adapter-path adapters/qwen3-8b-pentest \
  --save-path fused/qwen3-8b-pentest-4bit
mlx_lm.server --model fused/qwen3-8b-pentest-4bit --port 8080 &
python scripts/eval_toolcall.py --base-url http://127.0.0.1:8080/v1 \
  --golden data/golden/golden.jsonl --out reports/8b.json

# 5. Main 14B run
mlx_lm.lora --config configs/qwen3-14b-balanced.yaml

# 6. Fuse and evaluate
mlx_lm.fuse --model mlx-community/Qwen3-14B-4bit \
  --adapter-path adapters/qwen3-14b-pentest \
  --save-path fused/qwen3-14b-pentest-4bit
mlx_lm.server --model fused/qwen3-14b-pentest-4bit --port 8080 &
python scripts/eval_toolcall.py --base-url http://127.0.0.1:8080/v1 \
  --golden data/golden/golden.jsonl --out reports/14b.json
```

Export to GGUF/Ollama is in [`10-export-serving.md`](10-export-serving.md).

---

## 12. Checklist before you walk away

- [ ] run is logged (config + dataset hash);
- [ ] first 20 iterations look sane (loss not `nan`, tokens/sec normal);
- [ ] `save_every` is set and the first checkpoint exists;
- [ ] disk has room for checkpoints;
- [ ] sleep is disabled, charger connected;
- [ ] the Kali VM/Docker are off;
- [ ] you know how to stop it cleanly (`Ctrl-C`; the last checkpoint is intact);
- [ ] you have a plan for evaluating the result, not just watching loss.

---

## 13. Sources

- MLX-LM LoRA guide — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md>
- MLX-LM `lora.py` (flags/defaults) — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/lora.py>
- MLX-LM `datasets.py` (chat/tools/masking) — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/datasets.py>
- MLX-LM `trainer.py` (truncation/eval) — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py>
- MLX-LM issue #1206 (GDN training OOM) — <https://github.com/ml-explore/mlx-lm/issues/1206>
- MLX-LM issue #1185 (Metal descriptor leak) — <https://github.com/ml-explore/mlx-lm/issues/1185>
- Qwen3-14B model card — <https://huggingface.co/Qwen/Qwen3-14B>
- LoRA paper — <https://arxiv.org/abs/2106.09685>
- QLoRA paper — <https://arxiv.org/abs/2305.14314>
