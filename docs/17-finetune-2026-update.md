# 17 — Fine-tuning update: what the 2026 evidence changes

*Research pass: 25 October 2026. This chapter supersedes the hyperparameter defaults in chapter 08 where they disagree.*

The first edition of this handbook used the common MLX recipe: last 16 blocks, rank 16, `scale: 20`, learning rate `5e-5–1e-4`, and 400–600 iterations. That recipe works, but it is not the best-supported starting point in late 2026. Three findings matter:

1. **LoRA should cover all layers, especially the MLP layers — not just the last N blocks.**
2. **MLX's `scale` is not PEFT's `alpha`; `scale: 20` silently multiplies the effective learning rate by 10× versus a standard `alpha=32, r=16` recipe.** This is the most common cause of “my MLX LoRA diverges or forgets”.
3. **One epoch is often enough** for a 1M+ token post-training set; more epochs increase forgetting without improving held-out tool use.

Qwen3.5/3.6 training on Apple Silicon is still not a safe local path: the relevant MLX issues remain open (see §7).

---

## 1. TL;DR: the revised recipe

For **Qwen3-14B 4-bit QLoRA** on a 32 GB M1 Pro:

```yaml
num_layers: -1          # all 40 blocks; fall back to 24 or 16 if OOM
batch_size: 1
grad_accumulation_steps: 8
max_seq_length: 2048
grad_checkpoint: true
lora_parameters:
  rank: 16
  scale: 2.0            # = PEFT alpha 32 at rank 16
  dropout: 0.05
learning_rate: 1.0e-4
iters: 300              # ~1.5–2 epochs over 2M tokens at 1.5k/sample
```

If all layers do not fit, use the last 24 blocks before falling back to 16. If you keep `scale: 20`, divide the learning rate by 10 (`1e-5`), because the update is `scale × BA`.

The rest of this chapter explains why, and how to test it on your machine.

---

## 2. The evidence: “LoRA Without Regret”

Thinking Machines' *LoRA Without Regret* (Sep 2025) is the cleanest controlled study of LoRA versus full fine-tuning for post-training. It trained Llama and Qwen models on Tulu3 and OpenThoughts3, sweeping rank, learning rate, batch size, and which layers receive adapters.

### 2.1 All layers, especially MLP

The headline result:

- **Attention-only LoRA underperforms**, even when you match trainable parameter counts by raising the attention rank.
- **MLP-only LoRA performs about as well as MLP + attention.**
- **All-layer LoRA is the best default.**

Practical consequence: adapters on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` in every block. In MLX, `num_layers: -1` does this; `num_layers: 16` only adapts the last 16 blocks.

### 2.2 Learning rate: ~10× full fine-tuning

Across 14 Llama/Qwen models, the optimal LoRA LR was consistently about **10× the full fine-tuning LR** (about 15× for very short runs). If full FT uses `2e-5`, LoRA wants roughly `2e-4`, not `2e-5`.

The optimal LR is approximately **rank-independent** because of LoRA's `1/r` scaling. Rank 1 needed `1.2e-4`, rank 16 `2.2e-4`, rank 256 `2.5e-4` in their Llama-3.1-8B sweep; full FT needed `2.5e-5`.

### 2.3 Batch size: keep it modest

LoRA is **less tolerant of large batches** than full FT. The gap grows beyond an effective batch of ~32 and cannot be fixed by increasing rank. Effective batch **8–32** is the safe zone; 8–16 is a good target on one Mac.

Our configs use `batch_size: 1` with `grad_accumulation_steps: 8` → effective batch 8. That is deliberate.

### 2.4 Rank: 16–32 for post-training

For small-to-medium post-training sets, **rank 16–32 is enough**. Increase to 64–128 only if validation loss plateaus well above full FT *and* you have >5M tokens. Rank 256 is for research-scale sweeps, not a 32 GB laptop.

### 2.5 One epoch

The study trained for **one epoch** on Tulu3/OpenThoughts3 and matched full FT. For a 2–5M token agentic set, one to two epochs is usually right. Three or more epochs on a narrow dataset teaches style and memorization, not tool use.

Use validation loss **and** a held-out tool-call eval to decide; do not train to a fixed iteration count.

---

## 3. The MLX `scale` trap

MLX's LoRA layer computes:

```text
y = W x + scale · B A x
```

PEFT computes:

```text
y = W x + (alpha / r) · B A x
```

Therefore:

```text
scale = alpha / r
```

MLX's default is `scale: 20.0`, which at rank 8 corresponds to PEFT `alpha = 160`. That is why MLX examples often use a tiny LR (`1e-5`). If you copy a standard PEFT recipe (`alpha=32, r=16, lr=2e-4`) but keep `scale: 20`, your effective update is **10× too large**.

| MLX `scale` | Rank | Equivalent PEFT `alpha` | Sensible LR |
|---:|---:|---:|---:|
| 20.0 | 8 | 160 | 1e-5–2e-5 |
| 20.0 | 16 | 320 | 1e-5 |
| 2.0 | 16 | 32 | **1e-4–2e-4** |
| 4.0 | 8 | 32 | **1e-4–2e-4** |
| 1.0 | 32 | 32 | **1e-4–2e-4** |

The revised configs in this repository use **`scale` = 32 / rank**, so the LR is directly comparable to the published LoRA literature:

- rank 8 → `scale: 4.0`
- rank 16 → `scale: 2.0`
- rank 32 → `scale: 1.0`

If you prefer to keep MLX's default `scale: 20`, that is fine — just divide the LRs in this chapter by 10 and sweep around `1e-5`.

### Reproducing the old configs

| Old | New (equivalent-ish) |
|---|---|
| `scale: 20`, `lr: 5e-5`, rank 16 | `scale: 2`, `lr: 5e-5` is **10× weaker**; use `scale: 2, lr: 5e-4` to match |
| `scale: 20`, `lr: 1e-4`, rank 16 | `scale: 2, lr: 1e-3` to match — too high; do not |

This is why the old `5e-5–1e-4` range is not directly comparable to PEFT recommendations. The new configs deliberately use a standard parameterization.

---

## 4. Does MLX adapt the MLP layers?

Yes, in current MLX-LM (0.32.2). `linear_to_lora_layers` collects every `nn.Linear`, `QuantizedLinear`, `SwitchLinear`, and embedding in the selected blocks — including `gate_proj`, `up_proj`, and `down_proj`. An older report ([mlx#2616](https://github.com/ml-explore/mlx/issues/2616), Sep 2025) claimed only `q_proj`/`v_proj` were adapted; it was closed the same day and the current code does not have that restriction.

Still, **verify on your installed version**, because a stale environment or a custom `keys` list can silently narrow the adapter:

```bash
python scripts/check_lora.py --model mlx-community/Qwen3-14B-4bit --num-layers -1 --rank 16 --scale 2
```

The script prints:

- number of transformer blocks;
- number of LoRA modules;
- counts by projection (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`);
- trainable parameters and percentage of the model.

Expected for all-layer rank 16 on Qwen3-14B: roughly 40 blocks × 7 projections = **280 LoRA modules** (plus any embedding adapters), and on the order of **60M trainable parameters**, not 13M. If you see only `q_proj`/`v_proj` or ~13M, stop and fix the adapter before training.

> Exact counts depend on the model's layer structure and MLX version; use the script's output as the source of truth, not this estimate.

---

## 5. All-layer LoRA on 32 GB: memory

All-layer LoRA increases activation memory because every adapted block stores more intermediates for the backward pass. With gradient checkpointing, the increase is manageable but not free.

Approximate peak for Qwen3-14B 4-bit, batch 1, seq 2048, rank 16, grad checkpoint:

| Adapted layers | Peak estimate | Comment |
|---|---:|---|
| 8 | 12–15 GB | conservative fallback |
| 16 | 16–20 GB | previous default |
| 24 | 18–22 GB | good compromise |
| **40 (all)** | **19–24 GB** | **try first; measure** |

These are estimates, not measurements on your machine. Start with all layers; if you see `kIOGPUCommandBufferCallbackErrorOutOfMemory` or swap, drop to 24, then 16. On a 32 GB Mac, keep peak below ~24 GB so macOS and the display server have room.

`max_seq_length` is the bigger lever: 2048 → 1536 saves more than 16 → 8 layers. Measure with `scripts/token_stats.py` and cut long trajectories before cutting capacity.

---

## 6. Epochs, iterations, and early stopping

One epoch means:

```text
iters = dataset_tokens / (batch_size × grad_accumulation_steps × avg_tokens_per_sample)
```

Example: 2M tokens, effective batch 8, average 1,500 tokens/sample:

```text
iters = 2,000,000 / (8 × 1,500) ≈ 167
```

So 150–200 iterations is one epoch; 300 is ~1.8 epochs. The old 600-iteration recipe was ~3.6 epochs on that dataset — likely too much.

Rules:

- **>1M tokens:** 1 epoch; evaluate; add a second only if val loss is still falling.
- **<500k tokens:** 2–3 epochs, dropout 0.05–0.1, early stop.
- **Any size:** stop when validation loss has not improved for 2–3 evals; keep the best checkpoint.
- **Never** choose iterations by wall-clock alone.

Because `--mask-prompt` trains only on the final assistant message, a multi-turn trajectory split into N prefixes increases the token count roughly quadratically. Cap at 3–5 assistant turns; one epoch over prefix-expanded data is often enough.

---

## 7. Qwen3.5/3.6: still not a local training target

The MLX issues have not been resolved:

- [#1206](https://github.com/ml-explore/mlx-lm/issues/1206) — Qwen3.5-9B crashes on the first backward pass (M5 Max, and confirmed by an M1 Pro user); open since 26 Apr 2026.
- [#1956](https://github.com/ml-explore/mlx-lm/issues/1956) — step time scales with the number of trained GatedDeltaNet layers because backward runs the per-token path; open since 8 Oct 2026.
- [#1217](https://github.com/ml-explore/mlx-lm/pull/1217) — the Metal VJP kernel for `gated_delta_update` was **closed unmerged** on 21 Aug 2026.

In addition, Unsloth's Qwen3.5 guide recommends **against 4-bit QLoRA** for Qwen3.5 dense and MoE models because of higher-than-normal quantization differences. If you want Qwen3.5, use bf16 LoRA on a rented CUDA GPU, or full fine-tuning if you have the memory.

For this 32 GB M1 Pro handbook, **Qwen3-14B remains the training target** and **Qwen3.5-9B remains a serving/teacher option** (if you can run it), not a local fine-tune.

---

## 8. The revised experiment matrix

Do not run all of these. Run them in order, stopping when the gain is within noise.

| # | Config | Purpose | Cost |
|---|---|---|---|
| 1 | `qwen3-8b-fast.yaml` | smoke test / format learning | 6–12 h |
| 2 | `qwen3-14b-last16.yaml` | previous baseline | 35–50 h |
| 3 | `qwen3-14b-all-layers.yaml` | test the all-layer hypothesis | 45–65 h |
| 4 | `qwen3-14b-rank32.yaml` | capacity check | 45–60 h |
| 5 | all-layers, `lr: 5e-5` vs `1e-4` | LR sensitivity | 2 × 45 h |
| 6 | all-layers, `dora` | optional quality check | 50–70 h |

At ~20–25 tok/s, a 300-iteration 14B run over ~3.6M tokens is ~40–50 hours. That is why the matrix starts with the 8B and stops early.

**Decision rule:** the all-layer adapter replaces the last-16 adapter only if it improves **TCR by ≥5 points** on the golden set, with no regression in tool validity or refusal precision. A 1–2 point difference on 240 tasks is within noise.

---

## 9. What changed in the repo

| Item | Before | After |
|---|---|---|
| Adapter scope | last 12–16 blocks | all blocks (`-1`) preferred; 24/16 fallback |
| `scale` | 20.0 | 4.0 / 2.0 / 1.0 = alpha 32 |
| LR (14B) | 5e-5 | 1e-4 (standard parameterization) |
| Iterations (14B) | 600 | 300 (~1.5–2 epochs) |
| Rank | 16 | 16 default; 32 experiment |
| Epochs | 2–3 implicit | 1–2, early stop |
| Qwen3.5 | “conditional” | **not local**; rent a GPU |

The existing configs were renamed/kept for reproducibility:

- `qwen3-14b-balanced.yaml` is now the **last-16 baseline** (renamed conceptually; file retained).
- `qwen3-14b-all-layers.yaml` is the **new recommended default**.
- `qwen3-14b-rank32.yaml` tests capacity.
- `qwen3-8b-all-layers.yaml` is the fast all-layer smoke run.

---

## 10. Sources

- LoRA Without Regret — <https://thinkingmachines.ai/blog/lora>
- Reference implementation — <https://github.com/michaelbzhu/lora-without-regret>
- TRL summary — <https://huggingface.co/docs/trl/en/lora_without_regret>
- MLX-LM LoRA guide — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md>
- MLX LoRA layer (`scale` semantics) — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/lora.py>
- MLX LoRA target selection — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/utils.py>
- MLX issue #2616 (closed) — <https://github.com/ml-explore/mlx/issues/2616>
- MLX-LM issue #1206 — <https://github.com/ml-explore/mlx-lm/issues/1206>
- MLX-LM issue #1956 — <https://github.com/ml-explore/mlx-lm/issues/1956>
- MLX-LM PR #1217 — <https://github.com/ml-explore/mlx-lm/pull/1217>
- Unsloth Qwen3.5 guide — <https://unsloth.ai/docs/models/qwen3.5/fine-tune>
