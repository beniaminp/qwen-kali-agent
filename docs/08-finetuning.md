# 08 — Fine-tuning with MLX: the full runbook

*Last verified: 25 October 2026 against mlx-lm 0.32.2. Commands were checked against `mlx_lm/lora.py` and `LORA.md`; they were not executed on the target Mac in this environment. The hyperparameter defaults were revised after the research pass in [17 — Fine-tuning update](17-finetune-2026-update.md): use all-layer LoRA, `scale = 32/rank`, and one to two epochs.*

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
- MLX applies LoRA to **every linear layer in the selected blocks** (q, k, v, o, gate, up, down), not just attention. `--num-layers -1` selects all blocks; a positive N selects the last N.
- **The revised default is all layers** (`num_layers: -1`). The 2026 LoRA evidence shows attention-only and last-block-only adapters leave quality on the table, especially on MLP layers. Fall back to 24 or 16 only if memory forces it.
- Only adapter weights and optimizer state are trained. The base model is read-only.
- `--mask-prompt` computes loss only on the final assistant completion. Everything before it is context.

Adapter size is not the memory problem; **activations and logits are**. A rank-16 all-layer Qwen3-14B adapter is on the order of 60M parameters (~240 MB fp32), versus the ~8.3 GB 4-bit base. The extra activation memory from adapting 40 blocks is what you have to watch.

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
| `--num-layers` | 16 | **-1 (all)** | Last N blocks. `-1` = all; 24/16 only if memory forces it. |
| `--batch-size` | 4 | 1–2 | Per-step micro-batch. Start at 1. |
| `--iters` | 1000 | computed | One iteration = one batch, not one epoch. |
| `--val-batches` | 25 | 25–50 | `-1` uses the whole validation set (slow). |
| `--learning-rate` | 1e-5 | 1e-4 (scale 2) | Pair with `scale`; see §3. |
| `--steps-per-report` | 10 | 10 | Console cadence. |
| `--steps-per-eval` | 200 | 100 | Validation cadence. |
| `--grad-accumulation-steps` | 1 | 4–8 | Effective batch without activation memory. |
| `--resume-adapter-file` | — | path | Loads weights only, not optimizer/iteration state. |
| `--adapter-path` | `adapters` | `adapters/qwen3-14b-all-layers` | Output directory. |
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

`scale` in MLX **is** the multiplier on `B A`:

```text
MLX:  y = W x + scale * B A x
PEFT: y = W x + (alpha / r) * B A x
=>    scale = alpha / r
```

The revised configs use `scale = 32 / rank`, so they are directly comparable to standard PEFT recipes:

| Rank | MLX `scale` | PEFT `alpha` | LR |
|---:|---:|---:|---:|
| 8 | 4.0 | 32 | 1e-4 |
| 16 | 2.0 | 32 | 1e-4 |
| 32 | 1.0 | 32 | 1e-4 |

MLX's default `scale: 20.0` at rank 8 is equivalent to PEFT `alpha=160`; that is why MLX examples use `1e-5`. If you keep `scale: 20`, divide these LRs by 10.

---

## 3. The configs

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
  scale: 2.0
lr_schedule:
  name: cosine_decay
  warmup: 50
  arguments: [1.0e-4, 400, 1.0e-6]
optimizer_config:
  adamw:
    weight_decay: 0.01
```

Effective batch = 2 × 4 = 8. 400 steps × 8 × ~1,500 tokens ≈ 4.8M tokens. At 40–55 tok/s, expect **25–35 hours**. For a 100k-token smoke test, set `iters: 50` and use a subset. `qwen3-8b-all-layers.yaml` is the all-layer 8B variant.

### 3.2 14B all layers — recommended (`configs/qwen3-14b-all-layers.yaml`)

```yaml
model: mlx-community/Qwen3-14B-4bit
train: true
data: data/processed
fine_tune_type: lora
optimizer: adamw
num_layers: -1
batch_size: 1
iters: 300
val_batches: 25
learning_rate: 1.0e-4
steps_per_report: 10
steps_per_eval: 100
save_every: 50
max_seq_length: 2048
grad_checkpoint: true
grad_accumulation_steps: 8
mask_prompt: true
adapter_path: adapters/qwen3-14b-all-layers
seed: 42
lora_parameters:
  rank: 16
  dropout: 0.05
  scale: 2.0
lr_schedule:
  name: cosine_decay
  warmup: 50
  arguments: [1.0e-4, 300, 1.0e-6]
optimizer_config:
  adamw:
    weight_decay: 0.01
```

Effective batch 8. 300 steps × 8 × ~1,500 tokens ≈ 3.6M tokens (~1.5–2 epochs over 2M tokens). At 20–30 tok/s, **40–55 hours**. Peak memory is the risk: estimates are 19–24 GB. If it OOMs, set `num_layers: 24`, then `16`; do not start by shortening the data.

### 3.3 14B last-16 baseline (`configs/qwen3-14b-balanced.yaml`)

Same as 3.2 but `num_layers: 16`, `adapter_path: adapters/qwen3-14b-last16`. Keep it as the comparison run; it is the previous default and uses less memory (16–20 GB). Do not ship it without checking it against the all-layer adapter on the golden set.

### 3.4 14B conservative (`configs/qwen3-14b-conservative.yaml`)

`num_layers: 8`, rank 8, `scale: 4.0`, seq 1536, 250 iterations. Peak ~12–15 GB. Use it when the machine is shared or the all-layer run OOMs. Quality is lower; evaluate before committing.

### 3.5 Capacity and long-context variants

- `configs/qwen3-14b-rank32.yaml` — all layers, rank 32, `scale: 1.0`, 300 iterations. Tests whether the adapter is capacity-limited.
- `configs/qwen3-14b-longctx.yaml` — last 16 blocks, seq 4096, rank 16, `scale: 2.0`, 250 iterations. Only if your 99th-percentile length truly needs it.
- `configs/qwen3-8b-all-layers.yaml` — all blocks, rank 16, `scale: 2.0`, 300 iterations.

### 3.6 How many iterations?

```text
tokens_seen ≈ iters × batch_size × grad_accumulation_steps × avg_tokens_per_sample
iters ≈ epochs × dataset_tokens / (batch_size × grad_accumulation_steps × avg_tokens_per_sample)
```

Example: 2M-token dataset, avg 1,500 tokens, effective batch 8, **1 epoch**:

```text
iters = 1 × 2,000,000 / (8 × 1,500) ≈ 167
```

So 150–200 iterations is one epoch; 300 is ~1.8 epochs. The old 600-iteration recipe was ~3.6 epochs and is no longer the default. If you change `max_seq_length`, re-measure `avg_tokens_per_sample`.

## 4. Hyperparameter guidance

### Rank and scale

| Rank | Approx. trainable params (14B, all 40 layers) | Use |
|---|---:|---|
| 8 | ~31M | memory-constrained; format adaptation |
| 16 | ~63M | **default** for this handbook |
| 32 | ~125M | capacity experiment; more overfit risk |
| 64 | ~250M | only with a large, clean dataset and a good reason |

Use `scale = 32 / rank` (4.0, 2.0, 1.0, 0.5). If you keep MLX's `scale: 20`, divide the LR by 10. Verify the actual count with `python scripts/check_lora.py`; do not trust the table.

### Learning rate

With `scale = 32 / rank`:

- 8B: `1e-4` is a good start.
- 14B: `1e-4` is the default; `5e-5` is the conservative choice.
- If you keep `scale: 20`, use `1e-5` instead.
- If loss spikes or becomes `nan`: halve the LR, verify there are no empty completions, then retry.
- If loss plateaus high (>0.9) after 200 steps: the data may be too diverse, the LR too low, or the mask is wrong.

### Layers

More adapted layers = more capacity and more activation memory. **All 40 is the default now.** 24 is the first fallback, 16 the second, 8 the last. The earlier “16 is enough” advice came from attention-only intuition; the evidence favors MLP + all layers. On 32 GB, all layers usually fit with gradient checkpointing and seq 2048; measure it.

### Dropout

`0.05` is a reasonable regularizer for 2–5M tokens. For a very small dataset (<500k tokens), use `0.1`; for >10M tokens, `0.0–0.05`.

### Batch and accumulation

Start `batch_size: 1`, `grad_accumulation_steps: 8`. Increase only after measuring peak memory. Keep the effective batch **8–16**, and never above 32: LoRA is less tolerant of large batches than full fine-tuning, and raising rank does not fix it.

### Sequence length

2048 is the default. Agentic trajectories with tool schemas often need 3–4K. Measure the 99th percentile. Going to 4096 roughly doubles activation memory; use gradient checkpointing, batch 1, and preferably last 16 blocks rather than all 40.

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
| Peak mem | 19–24 GB (14B all layers) | >26 GB = danger |
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
adapters/qwen3-14b-all-layers/
├── adapter_config.json
├── adapters.safetensors
├── 0000100_adapters.safetensors
├── 0000200_adapters.safetensors
...
```

Resume:

```bash
mlx_lm.lora --config configs/qwen3-14b-all-layers.yaml \
  --resume-adapter-file adapters/qwen3-14b-all-layers/0000200_adapters.safetensors \
  --iters 100 \
  --adapter-path adapters/qwen3-14b-all-layers-resume
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
| `kIOGPUCommandBufferCallbackErrorOutOfMemory` | batch 1, `--grad-checkpoint`, seq 1536, `--num-layers 24` then `16`, close apps. |
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
config: configs/qwen3-14b-all-layers.yaml
iters: 300
effective_batch: 8
max_seq_length: 2048
peak_mem_gb: 22.0
tokens_per_sec: 22.0
wall_clock_h: 48
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

# 3. Verify the adapter scope before spending hours
python scripts/check_lora.py --model mlx-community/Qwen3-8B-4bit \
  --num-layers -1 --rank 16 --scale 2

# 4. Smoke test on 8B (hours)
mlx_lm.lora --config configs/qwen3-8b-all-layers.yaml

# 5. Evaluate the 8B adapter
mlx_lm.fuse --model mlx-community/Qwen3-8B-4bit \
  --adapter-path adapters/qwen3-8b-all-layers \
  --save-path fused/qwen3-8b-pentest-4bit
mlx_lm.server --model fused/qwen3-8b-pentest-4bit --port 8080 &
python scripts/eval_toolcall.py --base-url http://127.0.0.1:8080/v1 \
  --golden data/golden/golden.jsonl --out reports/8b.json

# 6. Main 14B run (all layers)
mlx_lm.lora --config configs/qwen3-14b-all-layers.yaml

# 7. Fuse and evaluate
mlx_lm.fuse --model mlx-community/Qwen3-14B-4bit \
  --adapter-path adapters/qwen3-14b-all-layers \
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
- MLX-LM issue #1956 (GDN backward step scaling) — <https://github.com/ml-explore/mlx-lm/issues/1956>
- LoRA Without Regret — <https://thinkingmachines.ai/blog/lora>
- [17 — Fine-tuning update](17-finetune-2026-update.md)
- Qwen3-14B model card — <https://huggingface.co/Qwen/Qwen3-14B>
- LoRA paper — <https://arxiv.org/abs/2106.09685>
- QLoRA paper — <https://arxiv.org/abs/2305.14314>
