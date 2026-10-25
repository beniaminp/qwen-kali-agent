# Appendix A — Configs, explained

The files in `configs/` are starting points. Every value here has a reason; change one variable at a time and record the result.

The defaults below follow [17 — Fine-tuning update](17-finetune-2026-update.md): all-layer LoRA, `scale = 32 / rank`, and one to two epochs.

| Config | Role | Layers | Rank | Scale | LR |
|---|---|---|---:|---:|---:|
| `qwen3-8b-fast.yaml` | smoke / format | 12 | 16 | 2.0 | 1e-4 |
| `qwen3-8b-all-layers.yaml` | 8B all-layer | -1 | 16 | 2.0 | 1.5e-4 |
| `qwen3-14b-all-layers.yaml` | **recommended 14B** | -1 | 16 | 2.0 | 1e-4 |
| `qwen3-14b-balanced.yaml` | last-16 baseline | 16 | 16 | 2.0 | 1e-4 |
| `qwen3-14b-rank32.yaml` | capacity test | -1 | 32 | 1.0 | 1e-4 |
| `qwen3-14b-conservative.yaml` | memory fallback | 8 | 8 | 4.0 | 1e-4 |
| `qwen3-14b-longctx.yaml` | 4K context | 16 | 16 | 2.0 | 8e-5 |

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
  scale: 2.0
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
| 12 layers | Enough to learn the format; lower memory than all-layer. |
| batch 2 × accum 4 | Effective batch 8; fits comfortably. |
| 400 iters | ~4.8M tokens at 1.5k/sample; adjust to your dataset. |
| LR 1e-4 + scale 2.0 | Standard alpha=32 parameterization. |
| rank 16 | ~19M trainable params over 12 layers. |
| seq 2048 | Cheap; split longer trajectories. |
| grad-checkpoint | Buys headroom; ~20–30% slower. |

Use this for a **100k-token smoke test** first: subset the data and set `iters: 50`. Confirm loss falls and tool calls render before running the full config.

---

## A.2 `qwen3-14b-all-layers.yaml` — recommended

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

| Value | Why |
|---|---|
| all 40 layers | The 2026 evidence favors MLP + attention everywhere. |
| rank 16, scale 2.0 | ~63M trainable params; alpha=32 equivalence. |
| batch 1 × accum 8 | Effective batch 8; LoRA dislikes large batches. |
| 300 iters | ~3.6M tokens; ~1.5–2 epochs over 2M. |
| seq 2048 | Most agentic examples fit; split long ones. |
| grad-checkpoint | Required for all-layer LoRA on 32 GB. |
| save_every 50 | Shorter runs; more checkpoints. |

Peak estimate: **19–24 GB**. If it OOMs, change `num_layers` to 24, then 16 — not the data first.

---

## A.3 `qwen3-14b-balanced.yaml` — last-16 baseline

Same as A.2 but `num_layers: 16`, `adapter_path: adapters/qwen3-14b-last16`, `iters: 300`. Peak 16–20 GB. Keep it as a comparison, not the default. If it beats the all-layer adapter on the golden set, use it; otherwise the all-layer result is the candidate.

---

## A.4 `qwen3-14b-conservative.yaml` — memory fallback

```yaml
model: mlx-community/Qwen3-14B-4bit
train: true
data: data/processed
fine_tune_type: lora
optimizer: adamw
num_layers: 8
batch_size: 1
iters: 250
val_batches: 25
learning_rate: 1.0e-4
steps_per_report: 10
steps_per_eval: 100
save_every: 50
max_seq_length: 1536
grad_checkpoint: true
grad_accumulation_steps: 8
mask_prompt: true
adapter_path: adapters/qwen3-14b-conservative
seed: 42
lora_parameters:
  rank: 8
  dropout: 0.05
  scale: 4.0
lr_schedule:
  name: cosine_decay
  warmup: 50
  arguments: [1.0e-4, 250, 1.0e-6]
```

Last 8 layers, rank 8, 1536 context. Peak ~12–15 GB. Use it if the machine is shared or the all-layer config OOMs. Quality is lower; evaluate before committing.

---

## A.5 `qwen3-14b-rank32.yaml` — capacity test

All layers, rank 32, `scale: 1.0`, 300 iters. ~125M trainable params. Run only if the rank-16 all-layer adapter underfits and you have a clean, larger dataset. The extra capacity is not free: more overfitting risk and a larger adapter.

---

## A.6 `qwen3-14b-longctx.yaml` — 4K context

Last 16 layers, rank 16, `scale: 2.0`, seq 4096, 250 iters. 4K roughly doubles activation memory versus 2K. If it OOMs, drop to 3072 or 12 layers. Do not raise `max_seq_length` just because a few examples are long; split them instead.

---

## A.7 Tuning rules of thumb

| If… | Change |
|---|---|
| OOM at step 1 | `num_layers: 24`, then `16`; keep seq 2048; `batch_size: 1`. |
| Still OOM | seq 1536; conservative config; `--clear-cache-threshold 2GB`. |
| Loss `nan` | Check token lengths; drop rows where prompt ≥ max sequence. Halve LR. |
| Loss stuck >1.0 | Verify mask/template; check `scale`; raise LR only after `check_lora.py`. |
| Val loss rises | Stop earlier; dropout 0.1; fewer epochs; more diverse data. |
| Underfitting | rank 16→32, more data, cleaner labels; layers already all. |
| Too slow | 8B; last 16; shorter sequences; close apps. |
| Adapter has no effect | Wrong base/path; verify with a fixed prompt. |

---

## A.8 What not to change casually

- **Model family** mid-run: the adapter is tied to the base architecture.
- **Chat template** mid-run: the tokenizer is part of the model.
- **`scale`** without adjusting LR: `scale` multiplies the update; `scale = alpha / rank`.
- **`--fine-tune-type full`** on 14B: it does not fit 32 GB.
- **`max_seq_length`** without re-measuring: truncation can cut a tool call.

---

## A.9 Java/Quarkus configs

| Config | Role | Layers | Rank | Scale | LR |
|---|---|---|---:|---:|---:|
| `qwen2.5-coder-7b-java.yaml` | 7B smoke | -1 (28) | 16 | 2.0 | 1.5e-4 |
| `qwen2.5-coder-14b-java.yaml` | **recommended 14B** | 24/48 | 16 | 2.0 | 1e-4 |
| `qwen2.5-coder-14b-java-all-layers.yaml` | capacity test | -1 (48) | 16 | 2.0 | 8e-5 |
| `qwen2.5-coder-14b-java-longctx.yaml` | 4K context | 24/48 | 16 | 2.0 | 8e-5 |

The Java configs point at `data/java/processed` and use `arguments` objects for the Qwen2.5-Coder template. Details: [18 — Java landscape](18-java-quarkus.md) and [20 — Java fine-tuning](20-java-finetuning.md).

---

## A.10 React/Tailwind configs

| Config | Role | Layers | Rank | Scale | Seq | LR |
|---|---|---|---:|---:|---:|---:|
| `qwen2.5-coder-7b-react.yaml` | 7B long-context smoke | -1 (28) | 16 | 2.0 | 4096 | 1.5e-4 |
| `qwen2.5-coder-14b-react.yaml` | **recommended 14B** | 24/48 | 16 | 2.0 | 3072 | 1e-4 |
| `qwen2.5-coder-14b-react-all-layers.yaml` | capacity test | -1 (48) | 16 | 2.0 | 2048 | 8e-5 |
| `qwen2.5-coder-14b-react-longctx.yaml` | 4K page context | 16/48 | 16 | 2.0 | 4096 | 8e-5 |

The React configs point at `data/react/processed` (or `processed-long`) and use `arguments` objects for the Qwen2.5-Coder template. Details: [22 — React & Tailwind](22-react-tailwind.md) and [24 — React fine-tuning](24-react-finetuning.md).

---

## A.11 Unified Qwen3 config

| Config | Base | Role | Layers | Rank | Scale | Seq | LR |
|---|---|---|---|---:|---:|---:|---:|
| `qwen3-14b-unified.yaml` | Qwen3-14B 4-bit | one adapter, Kali + Java + React | 24/40 | 16 | 2.0 | 2048 | 1e-4 |

Data: `data/unified/processed`, built with `scripts/mix_datasets.py --arguments-format string`. Details and trade-offs: [26 — One Qwen3 for all](26-one-model-for-all.md).

---

## A.12 Gemma 4 12B configs

| Config | Role | Layers | Rank | Scale | Seq | LR |
|---|---|---|---:|---:|---:|---:|
| `gemma4-12b-kali.yaml` | Kali/agent | 24/48 | 16 | 2.0 | 2048 | 1e-4 |
| `gemma4-12b-java.yaml` | Java/Quarkus | 24/48 | 16 | 2.0 | 2048 | 1e-4 |
| `gemma4-12b-react.yaml` | React/Tailwind | 24/48 | 16 | 2.0 | 3072 | 1e-4 |
| `gemma4-12b-unified.yaml` | one mixed adapter | 24/48 | 16 | 2.0 | 2048 | 1e-4 |

Data lives under `data/gemma4/<track>/processed`; tool arguments must be **objects** (`--arguments-format dict`). Details: [27 — Gemma 4 12B](27-gemma4-12b.md), [29 — Gemma fine-tuning](29-gemma4-finetuning.md).

---

## A.13 Config validation

```bash
python - <<'PY'
import yaml
from pathlib import Path
for p in Path("configs").glob("*.yaml"):
    d = yaml.safe_load(p.read_text())
    print(p.name, "layers=", d.get("num_layers"), "rank=", d["lora_parameters"]["rank"],
          "scale=", d["lora_parameters"]["scale"], "lr=", d.get("learning_rate"))
PY
```

Run a **1-iteration** test before a long run:

```bash
mlx_lm.lora --config configs/qwen3-8b-fast.yaml --iters 1 --save-every 1
```

And verify the adapter actually covers MLP layers:

```bash
python scripts/check_lora.py --model mlx-community/Qwen3-14B-4bit \
  --num-layers -1 --rank 16 --scale 2
```

Generate a small sweep:

```bash
python scripts/make_sweep.py --base configs/qwen3-14b-all-layers.yaml \
  --out configs/sweeps/r1 --rank 16,32 --lr 5e-5,1e-4 --layers all,16
```
