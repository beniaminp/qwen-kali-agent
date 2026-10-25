# Appendix A — Configs, explained

The files in `configs/` are starting points. Every value here has a reason; change one variable at a time and record the result.

---

## A.1 `qwen3-8b-fast.yaml` — debug and iterate

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

| Value | Why |
|---|---|
| 8B 4-bit | ~4.6 GB weights; 10–12 GB peak; fast iteration. |
| 12 layers | Enough capacity to learn the format; less memory than 16. |
| batch 2 × accum 4 | Effective batch 8; fits comfortably. |
| 400 iters | ~4.8M tokens at 1.5k/sample; adjust to your dataset. |
| LR 1e-4 + cosine | Standard LoRA range for 8B. |
| rank 16, scale 20 | MLX defaults; ~27M trainable params. |
| seq 2048 | Cheap; split longer trajectories. |
| grad-checkpoint | Buys headroom; ~20–30% slower. |

Use this for a **100k-token smoke test** first: subset the data and set `iters: 50`. Confirm loss falls and tool calls render before running the full config.

---

## A.2 `qwen3-14b-balanced.yaml` — the main run

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

| Value | Why |
|---|---|
| 14B 4-bit | ~8.3 GB weights; 16–22 GB peak. |
| 16 layers | Default capacity/memory balance. |
| batch 1 × accum 8 | Effective batch 8; the only safe starting point. |
| 600 iters | ~7.2M tokens at 1.5k/sample; 2–3 epochs over 2–3M tokens. |
| LR 5e-5 + warmup | Conservative for 14B; less likely to spike. |
| seq 2048 | Most examples fit; split long ones. |
| save_every 100 | ~100 MB/checkpoint; 6 checkpoints. |

If you have a 2M-token dataset and average 1,200 tokens/sample, 600 iters is ~2.5 epochs. If your average is 2,000, it is ~4 epochs; reduce to 400.

---

## A.3 `qwen3-14b-conservative.yaml` — when memory is tight

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

Differences: last 8 layers, rank 8, 1536 context. Peak ~12–15 GB. Use it if the balanced config OOMs, the machine is shared, or you need to keep the Kali VM running. Quality is lower; evaluate before committing.

---

## A.4 `qwen3-14b-longctx.yaml` — 4K agent trajectories

```yaml
model: mlx-community/Qwen3-14B-4bit
train: true
data: data/processed_long
fine_tune_type: lora
optimizer: adamw
num_layers: 12
batch_size: 1
iters: 500
val_batches: 20
learning_rate: 4.0e-5
steps_per_report: 10
steps_per_eval: 100
save_every: 100
max_seq_length: 4096
grad_checkpoint: true
grad_accumulation_steps: 8
mask_prompt: true
adapter_path: adapters/qwen3-14b-longctx
seed: 42
lora_parameters:
  rank: 16
  dropout: 0.05
  scale: 20.0
lr_schedule:
  name: cosine_decay
  warmup: 80
  arguments: [4.0e-5, 500, 1.0e-6]
```

Long context is expensive: 4K roughly doubles activation memory versus 2K. If it OOMs, drop to 3072 or 8 layers. Do not raise `max_seq_length` just because a few examples are long; split them instead.

---

## A.5 Tuning rules of thumb

| If… | Change |
|---|---|
| OOM at step 1 | `batch_size: 1`, `grad_checkpoint: true`, `max_seq_length: 1536`, `num_layers: 8`. |
| Loss `nan` | Check token lengths; drop rows where prompt ≥ max sequence. Halve LR. |
| Loss stuck >1.0 | Verify mask/template; raise LR to 1e-4 (8B) or 1e-4 (14B with warmup). |
| Val loss rises | Stop earlier; dropout 0.1; fewer epochs; more diverse data. |
| Underfitting | More layers (16→24), rank 16→32, more epochs, cleaner data. |
| Too slow | 8B instead; fewer layers; shorter sequences; close apps. |
| Adapter has no effect | Wrong path/base; verify with a fixed prompt. |

---

## A.6 What not to change casually

- **Model family** mid-run: the adapter is tied to the base architecture.
- **Chat template** mid-run: the tokenizer is part of the model.
- **`scale`** without understanding MLX semantics: it is not PEFT `alpha`.
- **`--fine-tune-type full`** on 14B: it does not fit 32 GB.
- **`max_seq_length`** without re-measuring: truncation can cut a tool call.

---

## A.7 Config validation

The YAML must parse and the keys must be recognized. MLX-LM prints unknown keys? It merges into `CONFIG_DEFAULTS`; unknown keys may be ignored. Check:

```bash
python - <<'PY'
import yaml, json
from pathlib import Path
for p in Path("configs").glob("*.yaml"):
    d = yaml.safe_load(p.read_text())
    print(p.name, sorted(d.keys()))
PY
```

Run a **1-iteration** test before a long run:

```bash
mlx_lm.lora --config configs/qwen3-8b-fast.yaml --iters 1 --save-every 1
```

This catches path, schema, and template errors in minutes.
