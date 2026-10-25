# Appendix C — The runbook

A realistic execution plan for one person with a 32 GB M1 Pro. Adjust dates, not order.

---

## Day 0 — Decide and prepare (2–3 hours)

- [ ] Confirm the target Mac: M1 Pro, 32 GB, 100+ GB free.
- [ ] Read [`01-landscape.md`](01-landscape.md) and choose the base: **Qwen3-14B local**, 8B fallback.
- [ ] Decide whether you will rent a GPU for a larger run. If yes, choose Qwen3.5-9B.
- [ ] Write the **authorization/scope** for the lab (even if you own it).
- [ ] Install the environment ([`03-macos-setup.md`](03-macos-setup.md)).
- [ ] Run `python scripts/check_env.py`.

**Exit criteria:** `mx.device_info()` shows GPU/Metal, `mlx_lm` imports, 100+ GB free.

---

## Day 1 — Data skeleton and smoke test (4–6 hours)

- [ ] Create the directory layout:

```bash
mkdir -p data/{raw,interim,processed,golden} adapters fused models reports
```

- [ ] Download a small, clean seed set (ToolACE + Hermes + When2Call + KaliBench train).
- [ ] Convert 2,000 rows to the canonical format.
- [ ] Validate and measure:

```bash
python scripts/validate_dataset.py data/interim/seed.jsonl --tools examples/tools/kali-tools.json
python scripts/token_stats.py data/interim/seed.jsonl --model Qwen/Qwen3-14B
```

- [ ] Build the **golden set** (start with 30 tasks; grow to 240).
- [ ] Run a 1-iteration training test on 8B:

```bash
mlx_lm.lora --config configs/qwen3-8b-fast.yaml --iters 1 --save-every 1
```

**Exit criteria:** a rendered prompt shows `<tools>`, `<tool_call>`, `<tool_response>`, and a valid `arguments` object; the 1-iteration run completes without `nan`.

---

## Day 2 — 8B smoke run (6–10 hours)

- [ ] Subset to ~100k tokens.
- [ ] Set `iters: 50`, `save_every: 25`.
- [ ] Train.
- [ ] Fuse and serve.
- [ ] Run `smoke_test.py` and 30 golden tasks.
- [ ] Inspect 10 outputs manually.

**Exit criteria:** tool validity >0.95, no `nan`, the model calls the right tool on simple prompts. If not, fix data/template before continuing.

---

## Day 3 — Build the real mix (full day)

- [ ] Implement the mix from [`04-data-strategy.md`](04-data-strategy.md): 35–40% general tool calling, 15–20% agentic, 20% cyber/CLI, 15% synthetic, 10% refusal.
- [ ] Generate 200–500 synthetic trajectories with DeepSeek/OpenCode; execute in the lab; keep only verified.
- [ ] Dedup, decontaminate, split.
- [ ] Write the manifest and dataset card.
- [ ] Token stats; choose `max_seq_length`.

**Exit criteria:** 1.5–3M tokens in `train.jsonl`, 5% valid, 5% test, a separate golden set, and a manifest with licenses.

---

## Day 4–6 — 14B run (2–4 days)

- [ ] Preflight: VM/Docker off, charger connected, sleep disabled.
- [ ] Run `python scripts/check_lora.py --model mlx-community/Qwen3-14B-4bit --num-layers -1 --rank 16 --scale 2`; confirm MLP projections are present.
- [ ] Start `configs/qwen3-14b-all-layers.yaml`.
- [ ] Check at iteration 20: loss not `nan`, tokens/sec 18–28, peak mem <24 GB.
- [ ] Check at the first eval: val loss tracking train.
- [ ] Evaluate the first checkpoint at ~50% (optional, but useful).
- [ ] Stop when val loss has not improved for 2–3 evals.

**Exit criteria:** a complete adapter, a training log, at least 3 checkpoints, and no OOM.

---

## Day 7 — Evaluate and decide (4–8 hours)

- [ ] Fuse the best checkpoint.
- [ ] Serve with `mlx_lm.server` or `llama-server`.
- [ ] Run the full golden set (240 tasks).
- [ ] Run a 5–10 task Cybench subset (optional).
- [ ] Run the unsafe-action suite.
- [ ] Compare base vs adapter, blind.
- [ ] Write the report.

**Decision:**

- **Ship** if TCR ≥0.60, TVR ≥0.98, UAR = 0, and a human review is positive.
- **Iterate** if TCR is low but TVR is high: add recovery/agentic data.
- **Stop** if UAR >0 or the adapter is worse than the base: fix data/policy before more compute.

---

## Day 8 — Package and serve

- [ ] Fuse the 4-bit MLX model.
- [ ] Convert to GGUF Q4_K_M.
- [ ] Import into Ollama or run `llama-server`.
- [ ] SSH tunnel from Kali.
- [ ] Run the end-to-end smoke test through MCP.
- [ ] Write the model card.
- [ ] Archive the adapter, config, manifest, and report.

**Exit criteria:** the Kali agent completes a 3–5 step lab task, stays in scope, and logs every action.

---

## Ongoing — Iterate

| Signal | Next experiment |
|---|---|
| Wrong tool | More tool-selection data; fewer tools; clearer descriptions. |
| Bad arguments | Schema-focused examples; enums; validation retries. |
| No recovery | More executed failure trajectories; cap repeated calls. |
| Over-refusal | Reduce refusal share; add in-scope positives. |
| Unsafe action | Stop; fix policy; add refusal/scope data; re-evaluate. |
| Too slow | Smaller model; shorter context; prefix cache. |

---

## Checklists

### Before training

- [ ] data validated;
- [ ] token stats measured;
- [ ] config reviewed;
- [ ] disk ≥100 GB;
- [ ] VM/Docker off;
- [ ] charger/sleep set;
- [ ] baseline evaluation saved;
- [ ] run ID and manifest recorded.

### Before serving

- [ ] adapter fused with the correct base;
- [ ] template smoke test passes;
- [ ] tool calls parse;
- [ ] context/KV budget fits;
- [ ] server bound to loopback;
- [ ] SSH tunnel tested;
- [ ] audit log writable;
- [ ] scope file loaded;
- [ ] kill switch tested.

### Before release

- [ ] golden set + benchmarks run;
- [ ] UAR = 0;
- [ ] licenses audited;
- [ ] secrets/PII scanned;
- [ ] model card written;
- [ ] dataset card written;
- [ ] limitations documented;
- [ ] owner/maintenance named.

---

## Time and cost estimate

| Phase | Wall clock | Cash |
|---|---:|---:|
| Setup | 2–3 h | $0 |
| Data skeleton + golden set | 1–2 days | $0 |
| 8B smoke | 6–10 h | $0 |
| Synthetic generation (5k rows) | 1 day | $10–30 |
| 14B run (3–5M tokens) | 2–4 days | $0 (electricity) |
| Evaluation | 1 day | $0–20 (judge API) |
| Packaging | half day | $0 |
| **Total** | **~1.5–2 weeks part-time** | **$10–50** |

Renting an L40S for the 14B run would cut the training wall clock to hours for ~$5–15, but add setup time. The local path is slower and cheaper; the cloud path is faster and metered.

---

## Specialised tracks

The day-by-day plan above is the Kali/agent track. The two coding tracks reuse the same MLX setup but have their own runbooks:

| Track | Base | Smoke | Main run | Verifier |
|---|---|---|---|---|
| **Java/Quarkus** | Qwen2.5-Coder-14B 4-bit | 7B all layers | 24/48, seq 2K | Maven tests + ArchUnit |
| **React/Tailwind** | Qwen2.5-Coder-14B 4-bit | 7B all layers, seq 4K | 24/48, seq 3K | TypeScript + Vitest + Playwright + axe |

Start with [18 — Java landscape](18-java-quarkus.md) or [22 — React & Tailwind](22-react-tailwind.md). Expect 2–4M tokens of verified data and a 2–4 day 14B run on the M1 Pro.

---

## What “done” looks like


- A reproducible dataset with a manifest and card.
- An adapter that beats the base on a held-out golden set.
- Zero unsafe actions on the forbidden suite.
- A served model reachable from Kali over a tunnel.
- An agent that completes a lab task and logs every action.
- Documentation that lets you do it again in six months.
