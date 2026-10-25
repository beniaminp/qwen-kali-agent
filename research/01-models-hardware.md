# Qwen agentic fine-tuning on a 32 GB M1 Pro

**25 Oct 2026.** Target: MacBook Pro M1 Pro, 32 GB, 16-core GPU (~200 GB/s), macOS. Goal: LoRA/QLoRA a Qwen model into a tool-calling agent for *authorized* Kali/network-security work, served to a local Kali VM. All repos checked 25 Oct 2026. **UNVERIFIED** = estimate/single report.

## 1. Qwen open-weight landscape (Oct 2026)

All **Apache-2.0** and tool-capable. Qwen3.5/3.6/3.8 are image/video VLMs; Qwen3/Coder are text-only. Qwen3 (2025) is plain attention; Qwen3.5+ use hybrid **Gated DeltaNet (GDN) + gated full attention**, with stronger agentic post-training in 3.6/3.8.

| Model | Released | Total / active | Architecture | Native ctx | Notable benchmarks (vendor) |
|---|---|---:|---|---:|---|
| [`Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | 29 Apr 2025 | 8.19B / 8.19B | Dense, GQA 32Q/8KV, 36 layers | 32K (131K YaRN) | MMLU-Redux 87.5, GPQA 62.0, BFCL-v3 68.1 |
| [`Qwen3-14B`](https://huggingface.co/Qwen/Qwen3-14B) | 29 Apr 2025 | 14.77B / 14.77B | Dense, GQA 40Q/8KV, 40 layers | 32K (131K YaRN) | MMLU-Redux 88.6, GPQA 64.0, BFCL-v3 70.4 |
| [`Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B) | 29 Apr 2025 | 32.76B / 32.76B | Dense, GQA 64Q/8KV, 64 layers | 32K (131K YaRN) | MMLU-Redux 90.9, GPQA 68.4, BFCL-v3 70.3 |
| [`Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) | 27 Feb 2026 | 4.66B / 4.66B | Hybrid, 32 layers: 24 GDN + 8 attn; dense FFN | 262K (~1.01M ext.) | MMLU-Pro 79.1, GPQA 76.2, BFCL-v4 50.3, TAU2 79.9 |
| [`Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) | 27 Feb 2026 | 9.65B / 9.65B | Hybrid, 32 layers: 24 GDN + 8 attn; dense FFN | 262K (~1.01M ext.) | MMLU-Pro 82.5, GPQA 81.7, **BFCL-v4 66.1**, **TAU2 79.1** |
| [`Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 15 Apr 2026 | 35.95B / ~3B | Hybrid, 40 layers: 30 GDN + 10 attn; MoE 256 experts, 8 active + shared | 262K (~1.01M ext.) | SWE-bench Verified 73.4, Terminal-Bench 2.0 51.5, TAU3 67.2, MCPMark 37.0 |
| [`Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | 5 Aug 2026 | 27.78B / 27.78B | Hybrid, 64 layers: 48 GDN + 16 attn; dense FFN | 262K (~1M ext.) | Terminal-Bench 2.1 73.0, SWE-bench Pro 61.7, GPQA 89.2, LCB-v6 90.3 |
| [`Qwen3-Coder-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | 31 Jul 2025 | 30.53B / 3.3B | MoE 128 experts, 8 active; 48 layers; non-thinking | 262K (1M YaRN) | SWE-bench Verified 51.6 (OpenHands, vendor) |
| [`Qwen3-Coder-Next`](https://huggingface.co/Qwen/Qwen3-Coder-Next) | 30 Jan 2026 | 79.67B / ~3B | Hybrid, 48 layers: 12×(3 GDN + 1 attn); MoE 512 experts, 10 active + shared | 262K | SWE-bench Verified 70.6, SWE-bench Pro 44.3, Terminal-Bench 2.0 34.2 |

Qwen3 numbers are thinking-mode results from [arXiv:2505.09388](https://arxiv.org/abs/2505.09388); later numbers are from the linked cards and the [Coder-Next report](https://github.com/QwenLM/Qwen3-Coder/blob/main/qwen3_coder_next_tech_report.pdf).

Qwen3.5-9B beats Qwen3-30B-A3B on agent tasks (BFCL-v4 66.1 vs 42.4; TAU2 79.1 vs 41.9). Coder-Next (~44.8 GB at 4-bit) cannot fit, so the 30B-A3B Coder is the coding specialist.

## 2. Training feasibility on 32 GB: TRAIN vs INFERENCE

Training needs weights + adapters + optimizer state + activations + logits. The 248,320-token Qwen3.5/3.8 vocabulary costs `248320 × seq × 4 B` just for logits — ~2.0 GB at 2K, ~8.1 GB at 8K. KV cache is inference-only.

**4-bit weights** (MLX ≈4.5 bits/param incl. scales): Qwen3-8B 4.6 GB, Qwen3-14B 8.3 GB, Qwen3.5-4B 2.6 GB, Qwen3.5-9B 5.4 GB, Qwen3.8-27B 15.6 GB, Qwen3.6-35B-A3B 20.2 GB, Coder-30B 17.2 GB, Coder-Next 44.8 GB. bf16 doubles these (14B 29.5 GB, 27B 55.6 GB).

| Training target (rank 16, batch 1, grad-checkpoint, seq 2K) | Weights | Est. peak | Max seq | Verdict |
|---|---:|---:|---:|---|
| Qwen3-8B, 4-bit QLoRA | 4.6 GB | 10–12 GB | ~8K stock; ~23K with flash patch | **Yes — safest** |
| Qwen3-8B, bf16 LoRA | 16.4 GB | 22–25 GB | 2K, maybe 4K | **Yes, tight** |
| Qwen3-14B, 4-bit QLoRA | 8.3 GB | 18–22 GB | 2K comfortable; 4K borderline | **Yes — local primary** |
| Qwen3-14B, bf16 LoRA | 29.5 GB | >34 GB | — | **No** |
| Qwen3.5-4B, 4-bit QLoRA | 2.6 GB | 12.6–14.8 GB measured ([#1956](https://github.com/ml-explore/mlx-lm/issues/1956)) | 1K; 2K likely | **Conditional** |
| Qwen3.5-9B, 4-bit QLoRA | 5.4 GB | not reliably measurable | 2K OOM reported at 36 GB | **No / high risk** |
| Qwen3.8-27B, 4-bit QLoRA | 15.6 GB | 32–60 GB reported | — | **No** |
| Qwen3.6-35B-A3B, 4-bit QLoRA | 20.2 GB | 30–35 GB | 512–1K at best | **No** |

Peaks assume 8–16 LoRA layers; adapters/optimizer are <1 GB.

**Anchors.** Qwen3-8B-4bit QLoRA on an M1 Max 32 GB trains to **7,936** tokens stock / **23,040** with a flash-attention patch; 8K peaks at **25.68 vs 12.75 GiB**, and 2K runs at **92–98 tok/s** ([mlx-train-perf 0.8.0](https://pypi.org/project/mlx-train-perf/0.8.0/)). An M1 Pro has half the GPU/bandwidth, so budget ~45–55 tok/s at 2K (**ESTIMATE**).

Qwen3.5-9B-4bit LoRA crashes on the **first backward pass** on a 36 GB M5 Max (`kIOGPUCommandBufferCallbackErrorOutOfMemory`) regardless of batch, seq 2K–8K, 4–16 layers, or `--grad-checkpoint`; Qwen3-8B-4bit trains with identical settings ([#1206](https://github.com/ml-explore/mlx-lm/issues/1206), open 26 Apr 2026). The thread includes an M1 Pro report, so it is a real risk here. Qwen3.5-27B-4bit LoRA peaks ~**60 GB** on a 128 GB M4 Max before a Metal descriptor-limit failure ([#1185](https://github.com/ml-explore/mlx-lm/issues/1185)); an independent report puts 27B QLoRA at **117 GB peak on main** ([PR #1217](https://github.com/ml-explore/mlx-lm/pull/1217)).

**Wall clock** at 2K: Qwen3-8B 40–55 tok/s → 10M tokens in 50–69 h, 50M in 253–347 h. Qwen3-14B 20–30 tok/s → 93–139 h / 463–694 h. Most LoRA SFT needs **2–5M high-quality tokens**, not 50M; LIMA showed 1K curated examples can beat much larger noisy sets ([arXiv:2305.11206](https://arxiv.org/abs/2305.11206)). Start with 1–2M, evaluate, then scale.

Start: rank 16, seq 2K, 8 layers, batch 1 × grad-accum 8, `--mask-prompt`, lr 1e-4. `--mask-prompt` trains on assistant/tool-call tokens only. Checkpoint every 100–200 iters; the #1185 leak can appear on long runs, so restart + `clear_cache`.

## 3. MLX-LM support: mature for Qwen3, risky for GDN models

MLX-LM `v0.32.2` shipped 16 Oct 2026; `main` was pushed 24 Oct 2026. Files exist for [`qwen3.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py), `qwen3_moe.py`, `qwen3_5.py`, `qwen3_5_moe.py`, `qwen3_next.py`. LoRA is applied by `linear_to_lora_layers()` over `model.layers`; VLM wrappers point at the language model, so the vision tower is not trained. MoE switch layers use `LoRASwitchLinear`.

The blocker is the **GDN training path**. `qwen3_5.py` calls the recurrence with `use_kernel=not self.training` ([line 193](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py#L193)); during training it falls back to `gated_delta_ops`, a Python loop over sequence length, because the Metal kernel has no VJP. Step time scales with trained GDN layers and memory blows up.

| Model | MLX file | LoRA status today | Evidence |
|---|---|---|---|
| Qwen3-8B/14B/32B | `qwen3.py` | **Mature**; 8B/14B fit | Standard attention; 8B is the working control in #1206; 14B uses the same path; 32B won't fit 32 GB |
| Qwen3.5-4B | `qwen3_5.py` | **Conditional** | 12.6–14.8 GB peak on M4 Pro (#1956); also OOM on M3 Pro 18 GB (#1206); slow GDN backward |
| Qwen3.5-9B | `qwen3_5.py` | **Risky / not production-safe on 32 GB** | First-backward OOM at 36 GB (#1206), M1 Pro report, no VJP kernel merged |
| Qwen3.5-27B / 3.8-27B | `qwen3_5.py` | **Not feasible on 32 GB** | ~60–117 GB peak (#1185, PR #1217); same GDN path |
| Qwen3.6-35B-A3B | `qwen3_5_moe.py` | **Memory-infeasible** | Qwen3-Next MoE block already stop-gradients routing; 20 GB weights + activations > 32 GB |
| Coder-30B-A3B | `qwen3_moe.py` | **Inference-only on 32 GB** | Qwen3 MoE routing fix [PR #1787](https://github.com/ml-explore/mlx-lm/pull/1787) merged 28 Aug 2026; 17 GB weights + overhead marginal |
| Coder-Next | `qwen3_next.py` | **Inference-only; doesn't fit** | 44.8 GB at 4-bit; GDN path |

**Timeline (open unless noted).** [#1185](https://github.com/ml-explore/mlx-lm/issues/1185) (23 Apr 2026): 27B LoRA dies with a Metal descriptor/handle leak (`Resource limit (499000) exceeded`), also seen on Qwen3.6-35B and Gemma; workarounds (`clear_cache`, `MLX_DISABLE_COMPILE`) are not fixes. [#1206](https://github.com/ml-explore/mlx-lm/issues/1206) (26 Apr): 9B first-backward OOM at 36 GB; 8B works. [PR #1217](https://github.com/ml-explore/mlx-lm/pull/1217) (27 Apr): Metal VJP kernel; 27B QLoRA 117 GB → 39 GB peak, ~3× throughput; **closed unmerged 21 Aug 2026** during a PR-backlog cleanup, not for technical reasons. [PR #1870](https://github.com/ml-explore/mlx-lm/pull/1870) (10 Sep, open): chunkwise fix.

**Tool parsing.** `mlx_lm.server` auto-infers `qwen3_coder` from the `<tool_call>\n<function=` template ([tokenizer_utils.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tokenizer_utils.py)); Qwen3.5/3.8 templates match, but [#1293](https://github.com/ml-explore/mlx-lm/issues/1293) (open 21 May 2026) reports 3.5/3.6 non-Coder models returning empty content with tools. Smoke-test; fall back to Ollama/llama.cpp or Qwen-Agent if it fails.

**Verdict:** Qwen3-8B/14B LoRA is safe; Qwen3.5/3.6/3.8 LoRA is not production-safe on 32 GB today.

## 4. Local Mac vs rented GPU

Prices are 25 Oct 2026 listings and move daily. A100 40GB is legacy but still the cheapest 40 GB.

| GPU | VRAM | 2026 price | Fits for training | 10M tokens @1,000 tok/s | 50M tokens |
|---|---|---:|---|---:|---:|
| RTX 4090 | 24 GB | $0.34–0.89/h (RunPod) | 8B/14B QLoRA, 9B QLoRA | 2.8 h → $1–2.5 | 14 h → $5–12 |
| L40S | 48 GB | $0.79–1.09/h (RunPod) | 9B bf16 LoRA, 27B QLoRA | 2.8 h → $2.2–3.1 | 14 h → $11–15 |
| A100 40GB | 40 GB | $0.42–1.00/h (Vast/RunPod); $1.99 Lambda | 9B/14B bf16, 27B QLoRA tight | 2.8 h → $1.2–2.8 | 14 h → $6–14 |
| A100 80GB | 80 GB | $1.19–1.79/h (RunPod); $2.79 Lambda | 27B bf16 LoRA, 35B QLoRA | 2.8 h → $3.3–5.0 | 14 h → $17–25 |

Sources: [RunPod](https://www.runpod.io/pricing), [Vast A100-40](https://gpueconomy.com/providers/vast-ai/a100-40), [Lambda](https://lambda.ai/instances). 1,000 tok/s is a conservative 14B QLoRA assumption; a tuned 4090 can exceed it, a bad config can be 3–5× slower (**UNVERIFIED**).

**Renting wins** for ≥5M tokens, multiple sweeps, or time >$5/h; 10M local tokens take 4–6 days. Local wins for privacy/air-gap, <1M-token iteration, and inference.

## 5. Recommendation

**Primary strictly local: `Qwen/Qwen3-14B` + 4-bit QLoRA.** Largest model that trains reliably here with today's MLX, mature tool calling, 32K native context, well-understood failure surface. Rank 16, seq 2K, 8 layers, batch 1 × grad-accum 8, 2–5M tokens.

**Primary if you will rent (recommended for quality): `Qwen/Qwen3.5-9B`.** Much stronger base than Qwen3-14B on agent/tool benchmarks, 262K context, trains comfortably on an L40S 48 GB or A100 40 GB. Serve the merged model as 4-bit MLX at ~28–30 tok/s; a 5–10M-token run costs **$5–30**.

**Fallbacks:** `Qwen/Qwen3-8B` — the “it always trains” option; bf16 LoRA at 2K or 4-bit QLoRA at 4–8K. `Qwen/Qwen3.5-4B` — modern architecture and strong agent scores (TAU2 79.9) at ~13–15 GB, but GDN training is slow and has machine-specific OOM reports; good for dataset/format iteration.

**Inference-only teacher/oracle: `mlx-community/Qwen3.8-27B-4bit`.** Best quality that fits (16–23 GB at 1–32K), 8.4–10 tok/s; use it to generate/validate hard trajectories and as a judge. For bulk synthetic data, `mlx-community/Qwen3.6-35B-A3B-4bit` is the throughput choice: 50 tok/s at 1K, 40 at 8K, 19–24 GB. Do not train either on 32 GB.

**Why not 27B locally:** the memory math fails before quality matters (60–117 GB peaks on main), and a well-curated 9B/14B run beats a truncated, unstable 27B. Dataset quality dominates: 2–5k clean multi-turn trajectories with correct tool schemas, realistic output, error recovery, and scope checks beat 10× noisy synthetic data. Gate on 100 held-out tasks; require the adapter to beat the base on ≥80%.

## 6. Serving on the Mac

**Runtimes.** **MLX (`mlx_lm.server`)** is fastest on M1 Pro, supports 4/6/8-bit and adapter hot-load, but has no auth and its docs say “not recommended for production; only basic security checks” ([SERVER.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md)). **Ollama** is easiest (`qwen3.5:9b`, `qwen3.6:35b-a3b`, `qwen3.8:27b`, `qwen3-coder-next`), with less quant/adapter control. **llama.cpp/LM Studio** give GGUF portability and Qwen3.5/3.8 LoRA conversion ([PR #28324](https://github.com/ggml-org/llama.cpp/pull/28324)).

```bash
mlx_lm.server --model mlx-community/Qwen3.5-9B-4bit --adapter-path ./adapters --host 0.0.0.0 --port 8080
```

OpenAI: `http://<mac-ip>:8080/v1`; no auth, so bind to VMnet/host-only or use `pf`/SSH tunnel. From Kali: `curl http://192.168.64.1:8080/v1/models`.

**Measured/derived M1 Pro throughput (16-core):**

| Model | 4-bit tok/s | 6-bit | 8-bit | 4-bit peak | Recommendation |
|---|---:|---:|---:|---:|---|
| Qwen3-8B | ~30–33 **EST** | ~20–24 **EST** | ~10–18 est. | 4.6 GB | 8-bit or bf16 |
| Qwen3.5-9B | **28–30** measured | ~20–22 **EST** | ~12–15 **EST** | 5.8–7.0 GB | **6-bit sweet spot** (7.2 GB) |
| Qwen3-14B | **15–17** @1–4K; 6.9 @32K | ~11–12 **EST** | **6.2–7.2** measured | 8.1–12.9 GB | 4-bit for speed; 6-bit for quality |
| Qwen3.8-27B | **8.4–10** measured | ~6 **EST** | no (27.8 GB) | 16.4–23.2 GB | 4-bit only; prefer DWQ/OptiQ |
| Qwen3.6-35B-A3B | **50 @1K, 46 @4K, 40 @8K, 27 @64K** | ~35–40 **EST** (27 GB) | no (36 GB) | 19.3–24.0 GB | 4-bit only |
| Coder-30B-A3B | **41 @1K, 32 @4K, 24 @8K, 18 @16K** | ~28 **EST** | no (30.5 GB) | 16.5–18.6 GB | 4-bit coding specialist |

Sources: oMLX M1 Pro measurements for [9B](https://omlx.ai/benchmarks/n1uqd0aw), [14B 4-bit](https://omlx.ai/benchmarks/performance/x9j9xdki), [14B 8-bit](https://omlx.ai/benchmarks/performance/5ygoq6ze), [27B](https://omlx.ai/benchmarks/performance/a8a6mjl2), [35B-A3B](https://omlx.ai/benchmarks/performance/cnsfbf7j), [Coder-30B](https://omlx.ai/benchmarks/performance/68z4pjbe). 6-bit figures are bandwidth-derived, not measured.

**Quantization.** 4-bit MLX (group 64) is the default and usually fine for tool calling. 6-bit is worth it for a 9B/14B agent because tool-call formatting is brittle (9B 7.2 GB, 14B 11.1 GB). 8-bit only makes sense for 4B/8B, or 14B if you accept ~7 tok/s. 27B/35B must be 4-bit; prefer DWQ/OptiQ. Cap context (~16K); 9B at 200K peaks at 19.5 GB/5.8 tok/s ([oMLX](https://omlx.ai/benchmarks/c2m3z3pd)), and 27B is at 23.2 GB by 32K.

## 7. Decision matrix and hardware settings

| Option | Fits 32 GB train? | MLX maturity | Overall |
|---|---|---|---|
| **Local Qwen3-14B QLoRA** | Yes (~18–22 GB) | High | **Best local default** |
| Local Qwen3-8B QLoRA/bf16 | Yes | Highest | **Safest fallback** |
| Cloud Qwen3.5-9B LoRA | N/A (L40S/A100) | N/A (CUDA) | **Best quality/time** |
| Inference Qwen3.8-27B-4bit | Inference only | High | **Oracle/teacher** |
| Inference Qwen3.6-35B-A3B-4bit | Inference only | High | **Bulk generator** |

1. **Check the default GPU budget.** On 32 GB M1 Pro macOS exposes ~**21.33 GiB** (`recommendedMaxWorkingSetSize` ≈ 22,906 MB), not 24 GB ([logged values](https://llms-explorer.com/tree/apple-silicon-unified-memory-and-llm-sizing/)). Verify with `sysctl iogpu.wired_limit_mb` (0 = default) and `python3 -c "import mlx.core as mx; print(mx.device_info())"`.
2. **Raise deliberately.** For 27B/35B at 16–32K, or 14B training: `sudo sysctl iogpu.wired_limit_mb=24576` (24 GiB → ~25.7 GB working set). For 35B-A3B at 64K, 26 GiB (`26624`) may be needed; that is the ceiling. Never 30 GB on 32 GB. Revert with `=0`; reboot resets. Set it **before** launching, since processes cache the limit at startup.
3. **Leave 6–8 GB for macOS and the Kali VM; close Chrome/Docker/Slack.** An 8 GB VM + a 24 GB model + window server is over budget: give the VM 4 GB for 27B/35B, 6 GB for 14B, and shut it down during training. Watch `memory_pressure`/`vm_stat` swapouts; wired memory cannot be swapped, and overcommit can freeze or panic macOS ([reported panics](https://llms-explorer.com/tree/macos-wired-memory-limit-and-gpu-kernel-panics/)). Keep the laptop plugged in and 30–60 GB disk free; sustained training throttles ~10–20% after 30+ min (**UNVERIFIED**).

**Kali.** Host-only (`vmnet1`) or private NAT; no auth, so use `pf` or `ssh -L 8080:127.0.0.1:8080`.

## Key sources

- Qwen cards: [3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) · [3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) · [3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) · [3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) · [Coder-30B](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) · [Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next)
- MLX-LM — https://github.com/ml-explore/mlx-lm · [LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) · [server guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md) · [GDN fallback](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py#L193)
- Issues: [#1185](https://github.com/ml-explore/mlx-lm/issues/1185) · [#1206](https://github.com/ml-explore/mlx-lm/issues/1206) · [#1956](https://github.com/ml-explore/mlx-lm/issues/1956) · [#1293](https://github.com/ml-explore/mlx-lm/issues/1293)
- PRs: [#1217](https://github.com/ml-explore/mlx-lm/pull/1217) · [#1389](https://github.com/ml-explore/mlx-lm/pull/1389) · [#1870](https://github.com/ml-explore/mlx-lm/pull/1870) · [#1787](https://github.com/ml-explore/mlx-lm/pull/1787)
- M1 Max training — https://pypi.org/project/mlx-train-perf/0.8.0/ · oMLX — https://omlx.ai/benchmarks/performance
- Memory limits — https://llms-explorer.com/tree/apple-silicon-unified-memory-and-llm-sizing/ · [MLX `set_wired_limit`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html)
- GPU prices — https://www.runpod.io/pricing · https://gpueconomy.com/providers/vast-ai/a100-40 · https://lambda.ai/instances
