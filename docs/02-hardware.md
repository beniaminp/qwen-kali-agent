# 02 — Hardware reality on a 32 GB M1 Pro

*Last verified: 25 October 2026. Numbers marked **EST** are estimates derived from architecture and published benchmarks, not measurements on this exact machine.*

This chapter is the memory accountant's view of fine-tuning. It answers three questions:

1. **Will it fit?**
2. **How long will it take?**
3. **What will macOS do when it doesn't fit?**

If you only read one section, read [§3](#3-the-memory-budget) and [§6](#6-what-to-do-when-it-doesnt-fit).

---

## 1. The machine

| Component | M1 Pro (binned 10-core CPU / 16-core GPU) |
|---|---|
| CPU | 10 cores (8 performance + 2 efficiency) |
| GPU | 16 cores, ~5.2 TFLOPS FP32; FP16/BF16 at the same peak rate (no double-rate half precision) |
| Neural Engine | 16 cores (not used by MLX training) |
| Memory | 32 GB unified, ~200 GB/s bandwidth |
| Media engine | ProRes/ProRes RAW hardware (irrelevant here) |
| OS | macOS 26.x (Tahoe or later in 2026) |
| Disk | Typically 512 GB–1 TB; keep 100+ GB free |

The critical property is **unified memory**: the CPU and GPU share the same physical RAM. There is no separate VRAM. A model that “fits in 24 GB of VRAM” on a PC does not automatically fit here, because macOS, the window server, the terminal, and any VM are using the same pool.

MLX allocates GPU buffers from this pool and can set a **wired limit** — the maximum the GPU is allowed to wire. Apple's default is conservative: on a 32 GB machine, `mx.device_info()["max_recommended_working_set_size"]` typically reports **~21.3 GiB**, not 32 GB. You can raise it deliberately, but every gigabyte you wire is a gigabyte macOS cannot compress or swap.

---

## 2. The three budgets

There are three different limits, and confusing them is the most common source of “it worked in the demo, then OOM'd”:

| Budget | Typical value | What it governs |
|---|---:|---|
| **Physical RAM** | 32 GB | Everything: OS, apps, model, VM. |
| **Default GPU working set** | ~21.3 GiB | What MLX will allocate without intervention. |
| **Wired limit** | 0 = default; can be set | The hard ceiling for GPU-wired memory. |

Check them:

```bash
sysctl hw.memsize
sysctl iogpu.wired_limit_mb
python - <<'PY'
import mlx.core as mx
print(mx.device_info())
PY
```

`iogpu.wired_limit_mb=0` means “use the system default”, not “unlimited”. `mx.device_info()` includes `max_recommended_working_set_size` in bytes.

**Do not set the wired limit to 30 GB on a 32 GB machine.** You will starve the OS. The practical ceiling is ~24–26 GB, and only when the Kali VM is shut down and Chrome/Docker/Slack are closed. A kernel panic or a forced reboot in the middle of a 3-day run is worse than a slightly smaller batch.

---

## 3. The memory budget

### 3.1 The equation

For QLoRA training, peak memory is approximately:

```text
M_peak ≈ M_weights
       + M_lora_params + M_optimizer
       + M_activations
       + M_logits
       + M_temporaries
       + M_runtime (Metal, tokenizer, Python, OS)
```

| Term | Approximate size | Notes |
|---|---|---|
| `M_weights` | 4.5 bits/param for MLX 4-bit | Includes quantization scales/biases. |
| `M_lora_params` | `rank × Σ(in+out)` over adapted layers | Rank 16 on 16 layers of 14B ≈ 27M params ≈ 107 MB fp32; all 40 layers ≈ 63M ≈ 252 MB. |
| `M_optimizer` | 2–3× adapter params | Adam/AdamW moments; still small because the base is frozen. |
| `M_activations` | `B × L × N_lora × (c₁H + c₂I) × bytes` | Dominated by sequence length and adapted layers. |
| `M_logits` | `B × L × V × 4 bytes` | V = 151,936 for Qwen3; ~1.24 GB per 2K sequence in fp32. |
| `M_temporaries` | 10–30% of activations | Attention buffers, dequantization, reductions. |
| `M_runtime` | 2–4 GB | Metal, Python, tokenizer, fragmentation. |

`B` batch size, `L` sequence length, `N_lora` number of adapted layers, `H` hidden size, `I` FFN intermediate size, `V` vocabulary.

### 3.2 Worked example: Qwen3-14B at 2K

Qwen3-14B: `H=5120`, `I=17408`, 40 layers, `V=151936`.

- 4-bit weights: **~8.3 GB**.
- LoRA rank 16 on the last 16 layers: **~0.11 GB** params, ~0.3 GB optimizer.
- LoRA rank 16 on all 40 layers: **~0.25 GB** params, ~0.7 GB optimizer; the activation term roughly doubles, so peak is ~21–24 GB even with checkpointing.
- Logits at `B=1, L=2048`: `1 × 2048 × 151936 × 4 ≈ 1.24 GB` (fp32 cross-entropy).
- Activations without checkpointing: roughly `1 × 2048 × 16 × (10×5120 + 4×17408) × 2 ≈ 7.9 GB`.
- Runtime: ~2–3 GB.

Peak ≈ **19–21 GB** — fits under the default 21.3 GiB only if nothing else is running, which is not a safe assumption.

**Java/Quarkus note.** Qwen2.5-Coder-14B has 48 layers (hidden 5120, intermediate 13824). A 24-block rank-16 adapter is roughly 34M parameters; peak is in the same 18–23 GB range as Qwen3-14B. Adapting all 48 blocks raises activations and can exceed 24 GB, so the Java runbook starts at 24/48.

**React/Tailwind note.** The same Qwen2.5-Coder-14B at 3,072 tokens peaks around 21–25 GB with 24/48 blocks; the 7B at 4,096 with all 28 blocks is the safer long-context run at 12–16 GB. Front-end examples include component code, styles and tests, so context is the pressure point.

**Gemma 4 12B note.** 11.95B, 48 layers, 4-bit weights ~6.6–7.6 GB. 24/48 at 2,048 peaks around 16–22 GB; 24/48 at 3,072 around 20–26 GB. Start at 24/48, not all 48. See [27 — Gemma 4 12B](27-gemma4-12b.md).

With `--grad-checkpoint`:

- Activations ≈ `B × L × N_lora × 2H × 2` ≈ **0.67 GB**.
- Peak ≈ **12–14 GB**, plus whatever the recomputation temporaries need. In practice, budget **16–18 GB**.

That is why every config in this repo enables gradient checkpointing. It costs ~20–30% throughput and buys 5–7 GB of headroom.

### 3.3 Worked example: Qwen3-8B at 2K

Qwen3-8B: `H=4096`, `I=12288`, 36 layers.

- 4-bit weights: **~4.6 GB**.
- Logits: **1.24 GB** (same vocabulary).
- Activations without checkpointing: `1 × 2048 × 16 × (10×4096 + 4×12288) × 2 ≈ 5.9 GB`.
- Peak: **12–14 GB** without checkpointing, **8–10 GB** with it.

The 8B is the “it always fits” option. If a training bug appears, reproduce it here before blaming the 14B.

### 3.4 Batch size and sequence length

Memory scales **linearly with batch** and **linearly with sequence length**, but logits and attention temporaries make long sequences disproportionately expensive. On 32 GB:

| Model | Seq 1K | Seq 2K | Seq 4K | Seq 8K |
|---|---|---|---|---|
| Qwen3-8B QLoRA, grad-ckpt, B=1 | ~7 GB | ~9 GB | ~13 GB | ~21 GB |
| Qwen3-8B QLoRA, grad-ckpt, B=2 | ~9 GB | ~13 GB | ~21 GB | OOM |
| Qwen3-14B QLoRA, grad-ckpt, B=1 | ~11 GB | ~16 GB | ~24 GB | OOM |
| Qwen3-14B QLoRA, grad-ckpt, B=2 | ~16 GB | ~24 GB | OOM | OOM |

These are planning estimates, not guarantees. Measure with `mx.get_peak_memory()` or the trainer's printed peak. Start one row lower than you think you need.

**Gradient accumulation** is the escape hatch: `batch_size=1` with `grad_accumulation_steps=8` gives an effective batch of 8 without the activation memory of 8 samples. It is slower (more optimizer steps per token? no — fewer, but each micro-batch is serial), but it fits.

### 3.5 KV cache is inference-only

Do not add KV cache to the training budget. It matters when serving:

```text
KV bytes/token = 2 (K,V) × layers × kv_heads × head_dim × bytes
```

Qwen3-14B: `2 × 40 × 8 × 128 × 2 = 163,840 bytes/token = 160 KiB/token`.

| Context | KV cache (fp16) | With 4-bit KV |
|---:|---:|---:|
| 4K | 0.67 GB | ~0.17 GB |
| 8K | 1.34 GB | ~0.34 GB |
| 16K | 2.68 GB | ~0.67 GB |
| 32K | 5.37 GB | ~1.34 GB |

At 32 GB, a 14B 4-bit model (~8.3 GB) + 32K fp16 KV (~5.4 GB) + runtime (~2 GB) ≈ 15.7 GB. That fits. A 27B 4-bit (~15.6 GB) + 32K (~5.4 GB) + runtime ≈ 23 GB — at the edge, especially with a VM running.

---

## 4. Training throughput and wall-clock

Training FLOPs are roughly `6 × N × tokens` (forward + backward), where `N` is the number of **active** parameters. Throughput is `effective_TFLOPS / (6N)`.

M1 Pro peak is ~5.2 TFLOPS, and FP16/BF16 run at the same peak rate as FP32: Apple's GPUs have no double-rate half precision ([metal-benchmarks](https://github.com/philipturner/metal-benchmarks) measures ~5.3 TFLOPS for both). Real transformer training with 4-bit dequantization, Python overhead and memory stalls achieves maybe 25–45% of peak, about 1.3–2.3 TFLOPS. For Qwen3-14B (`N=14.8B`, `6N ≈ 88.8 GFLOP/token`):

- At 1.3 TFLOPS effective: **14.6 tok/s**.
- At 2 TFLOPS: **22.5 tok/s**.
- At 2.3 TFLOPS: **25.9 tok/s**.

LoRA skips the weight gradients of the frozen base, so the real cost is closer to `4N` per token and these figures are conservative by up to 1.5×. Published anchors are consistent with this range: an M1 Max 32 GB trains Qwen3-8B 4-bit at **92–98 tok/s** at 2K with a small number of adapted layers, and the M1 Pro has roughly half the GPU and bandwidth. Budget **20–30 tok/s for 14B** and **40–55 tok/s for 8B** **(EST)**. If you see 60+ tok/s on a 14B, verify that you are actually training the layers you think you are.

| Run | Tokens | 14B @ 25 tok/s | 8B @ 45 tok/s |
|---|---:|---:|---:|
| Quick format test | 100K | 1.1 h | 0.6 h |
| Minimal adapter | 2M | 22 h | 12 h |
| Serious first adapter | 5M | 56 h | 31 h |
| Scaled run | 10M | 111 h | 62 h |
| “Just 50M tokens” | 50M | 23 days | 13 days |

**Interpretation:** 50M tokens is not a realistic local run. The research on data-efficient fine-tuning (LIMA and successors) shows that a small, curated set can outperform a much larger noisy one. Plan **1–2M tokens for the first adapter, 3–5M for the second**, and only scale after the evaluation says the model is underfitting rather than under-specified.

**Tokens/sec is not the only clock.** Data loading, tokenization, validation, checkpoint saves and thermal throttling add 10–20%. A 56-hour estimate is realistically 60–70 hours of wall time.

---

## 5. Disk, heat and power

- **Disk:** Qwen3-14B bf16 is ~29.5 GB; 4-bit MLX ~8.3 GB; a fused bf16 model ~30 GB; GGUF Q4_K_M ~9 GB; adapters ~0.1 GB each; checkpoints can add several GB. Reserve **100–150 GB**.
- **Heat:** a sustained GPU load on a MacBook Pro will throttle after 30+ minutes. A laptop stand and a cool room help; a closed lid with an external display is worse. Expect 10–20% throughput loss on long runs **(EST)**.
- **Power:** keep the charger connected. Training on battery will throttle aggressively and may sleep.
- **Sleep:** disable sleep for the duration:

```bash
sudo pmset -a sleep 0 disksleep 0
# restore later: sudo pmset -a sleep 1 disksleep 10
```

- **Swap:** if `vm_stat` shows swapouts climbing during training, you are over budget. Reduce batch/sequence length; do not “wait it out”.

---

## 6. What to do when it doesn't fit

In order of effectiveness:

1. **Enable gradient checkpointing.** Biggest win per line of config.
2. **Reduce `max_seq_length`.** 2048 → 1536 or 1024. Truncation is better than OOM, but fix the data if many examples are long.
3. **Reduce `num_layers`.** All 40 → 24 → 16 → 8. You lose adaptation capacity; tool-format learning usually survives, but evaluate.
4. **Batch size 1 + gradient accumulation.** Effective batch without activation memory.
5. **Quantize the base to 4-bit.** You should already be doing this. 6-bit/8-bit doubles or quadruples weight memory.
6. **Close everything.** Chrome, Docker Desktop, Slack, the Kali VM, photo apps. Check `memory_pressure`.
7. **Raise the wired limit carefully** to 24–25 GB, only after the above. `sudo sysctl iogpu.wired_limit_mb=24576`.
8. **Move to the 8B.** A completed 8B run beats a 14B run that dies at hour 40.
9. **Rent a GPU.** For runs over ~24 hours, this is usually the rational choice.

**Symptom → likely cause:**

| Symptom | Likely cause |
|---|---|
| `[metal::malloc] Resource limit (499000) exceeded` | Descriptor/handle leak or too many wired buffers; reduce layers/seq, restart, update MLX. |
| `kIOGPUCommandBufferCallbackErrorOutOfMemory` | True GPU OOM; reduce memory. |
| Process killed by macOS, no Python traceback | Wired-memory pressure / kernel watchdog. Lower the limit or memory use. |
| Swapouts climbing, UI stutters | Overcommitted unified memory; close apps or reduce batch. |
| Loss becomes `nan` | Prompt longer than `max_seq_length` with `--mask-prompt` (zero completion tokens), or LR too high. Check the data length distribution. |
| Throughput collapses after 30 min | Thermal throttling; improve cooling, reduce sustained load. |

---

## 7. Recommended machine settings

```bash
# Inspect
sysctl hw.memsize
sysctl iogpu.wired_limit_mb
python -c "import mlx.core as mx; print(mx.device_info())"
memory_pressure | head
vm_stat | head

# Optional: give the GPU up to 24 GiB (only for a dedicated training session)
sudo sysctl iogpu.wired_limit_mb=24576

# Restore after training / before normal use
sudo sysctl iogpu.wired_limit_mb=0
```

**A note on the Kali VM.** An 8 GB Kali VM plus a 20 GB training process plus macOS does not fit in 32 GB. During training, shut the VM down. During evaluation/serving, the training process is gone, so a 4–8 GB VM is fine. Do not run both.

---

## 8. What the numbers mean for the rest of the handbook

- Use **Qwen3-8B** for the first end-to-end run. It will finish in hours and expose format bugs.
- Use **Qwen3-14B** for the adapter you keep. Budget 2–5M tokens and 2–4 days.
- Keep sequences at **2048** unless the data demands more. Agentic trajectories with 5+ turns can exceed 4K; split them per assistant turn instead of raising the limit.
- Evaluate on a **held-out** set before scaling the run. More tokens cannot fix a schema mismatch.
- If you need Qwen3.5/3.8 quality, train in the cloud. The local inference path for those models is fine; the local training path is not.

---

## 9. Sources and measurements

- MLX memory and wired limit — <https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html>
- MLX-LM LoRA memory guidance — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md>
- Qwen3-8B QLoRA on M1 Max, 92–98 tok/s and 7,936/23,040-token limits — <https://pypi.org/project/mlx-train-perf/0.8.0/>
- Qwen3.5-9B first-backward OOM at 36 GB — <https://github.com/ml-explore/mlx-lm/issues/1206>
- Qwen3.5-27B / 3.8-27B descriptor leak and 60–117 GB peaks — <https://github.com/ml-explore/mlx-lm/issues/1185>, <https://github.com/ml-explore/mlx-lm/pull/1217>
- M1 Pro inference measurements (9B/14B/27B/35B-A3B) — <https://omlx.ai/benchmarks/performance>
- Apple Silicon unified-memory sizing — <https://llms-explorer.com/tree/apple-silicon-unified-memory-and-llm-sizing/>
- LIMA (data efficiency) — <https://arxiv.org/abs/2305.11206>
