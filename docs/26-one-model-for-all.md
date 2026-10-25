# 26 — One Qwen3 model for all three tracks

*Last verified: 25 October 2026 against the Qwen model cards, the Qwen3 technical report, the Qwen3-Coder repository and MLX-LM 0.32.2.*

> Considering **Gemma 4 12B** as a second base? See [27 — Gemma 4 12B](27-gemma4-12b.md) and [31 — Gemma vs Qwen](31-gemma-vs-qwen.md). The Qwen recommendation here is unchanged.

The handbook now has three tracks:

- **Kali/agent** — tool calling, scope, recovery, refusal.
- **Java/Quarkus** — Java 25, Quarkus 3.40, architecture, Maven.
- **React/Tailwind** — React 19.3, Tailwind 4.3, design systems, Playwright.

They currently use two bases: **Qwen3-14B** for Kali and **Qwen2.5-Coder-14B** for Java/React. The obvious question:

> Is there one Qwen3 model that can do all three?

The short answer:

- **One local base you can fine-tune for all three: Qwen3-14B.** Use **one base and three LoRA adapters**, not one mixed adapter, unless you deliberately want the simplicity trade-off.
- **One model you can serve for all three without training: Qwen3-Coder-30B-A3B-Instruct** for code-heavy work, or **Qwen3-30B-A3B-Instruct-2507** if you need more general/security breadth and 262K context. Both fit 4-bit inference on 32 GB; neither is a reliable local LoRA target.
- **Do not pick Qwen3.5/3.6/3.8 for local 32 GB training.** They are stronger models, but the Gated DeltaNet training path in MLX is still immature. Rent a GPU if you want them.

---

## 1. What “for all” can mean

| Goal | Best answer | Why |
|---|---|---|
| **One base to fine-tune locally for all three** | **Qwen3-14B**, three adapters | Only Qwen3 dense model that is both strong enough and reliably trainable in 32 GB |
| **One adapter for all three** | Qwen3-14B, mixed data | Simpler serving; expect per-track regressions |
| **One model to serve, no training** | **Qwen3-Coder-30B-A3B** or **Qwen3-30B-A3B-2507** | 30B-class quality, 3.3B active, 4-bit fits 32 GB; MoE is inference-only here |
| **Best quality, training on a rented GPU** | Qwen3.5-9B or Qwen3.8-27B | 262K context, vision, stronger agentic/coding; not local 32 GB |
| **Best code-only model** | Qwen3-Coder-30B-A3B | Code-specialised, 256K, agentic tool use |
| **Best general/security breadth** | Qwen3-30B-A3B-Instruct-2507 | 262K, non-thinking, general instruction and tool use |

“For all” is therefore a **base-model** question, not a **single-adapter** question. A 14B adapter has limited capacity; three small adapters on one frozen base give you specialization without three copies of the weights.

---

## 2. The Qwen3 candidates

All are Apache-2.0 except where noted. Vendor benchmarks are useful for relative ordering, not for predicting your task.

| Model | Params | Layers | Native context | Local training on 32 GB | Local inference | Role |
|---|---:|---:|---:|---|---|---|
| **Qwen3-8B** | 8.2B | 36 | 32K / 131K YaRN | **Yes**, all layers | Fast | Unified fallback |
| **Qwen3-14B** | 14.8B | 40 | 32K / 131K YaRN | **Yes**, 24/40 or all 40 | 15–17 tok/s | **Recommended base for all** |
| Qwen3-32B | 32.8B | 64 | 32K / 131K | No | Slow | Inference/teacher |
| Qwen3-30B-A3B | 30.5B / 3.3B | 48 | 32K | Marginal / not recommended | Yes, ~17–18 GB 4-bit | General MoE |
| **Qwen3-30B-A3B-Instruct-2507** | 30.5B / 3.3B | 48 | **262K** | No | Yes, ~17–18 GB 4-bit | General/long-context MoE |
| **Qwen3-Coder-30B-A3B-Instruct** | 30.5B / 3.3B | 48 | **262K** | No | Yes, ~17–18 GB 4-bit | Code/agentic specialist |
| Qwen3.5-4B | 4.7B | 32 | 262K | Conditional | Yes | Small multimodal agent |
| Qwen3.5-9B | 9.7B | 32 | 262K | **No / high risk** | Yes | Rent a GPU to train |
| Qwen3.6-35B-A3B | 36.0B / ~3B | 48 | 262K | No | Yes, 19–24 GB | Agentic coding MoE |
| Qwen3.8-27B | 27.8B | 64 | 262K | No | Yes, ~15–17 GB 4-bit | Teacher/oracle |
| Qwen3-Coder-Next | 79.7B / ~3B | — | 262K | No | No (~45 GB 4-bit) | API/rented GPU |

The Qwen3-14B is the only row that is simultaneously:

- dense and standard-attention (no GDN/MoE training surprises);
- strong at tool calling and instruction following;
- small enough for 4-bit QLoRA on 32 GB;
- large enough to be useful for Java and React after fine-tuning;
- Apache-2.0 with an official MLX 4-bit conversion.

### 2.1 Why not Qwen3-8B as the “one for all”?

It fits more comfortably and trains all 36 layers, but it has less capacity for three domains. It is the right **debug** model and the right **fallback** if the 14B will not fit your context needs. It is not the best final base.

### 2.2 Why not Qwen3-Coder-30B-A3B as the local training base?

Because it is a 30.5B MoE, not a 14B dense model:

- 4-bit weights alone are ~17–18 GB; gradients, activations and optimizer state push past 32 GB.
- MoE routing and expert gradients are not a reliable MLX LoRA path on this machine.
- It is excellent **served** at 4-bit with 8K–32K context, and excellent as a teacher/judge.

Use it for inference and data generation, not for a local adapter.

### 2.3 Why not Qwen3-32B?

32.8B dense at 4-bit is ~18.5 GB of weights; LoRA training does not fit in 32 GB. It is an inference/teacher model.

### 2.4 Why not Qwen3.5/3.6/3.8?

They are newer and stronger, but they use hybrid Gated DeltaNet (GDN) plus gated attention. The MLX training path is still immature: first-backward OOMs, Metal descriptor leaks and step time scaling with trained GDN layers have been reported on 32–36 GB machines, and the Metal VJP PR was closed unmerged. Unsloth also advises against 4-bit QLoRA for Qwen3.5.

If you rent a 48–80 GB GPU, **Qwen3.5-9B** is the best quality/time balance and **Qwen3.8-27B** is the strongest dense option. That is a separate training track, not a 32 GB local one.

---

## 3. Recommendation: one base, three adapters

```text
Qwen3-14B-4bit (frozen)
├── adapters/qwen3-14b-kali     # tool calling, scope, recovery, refusal
├── adapters/qwen3-14b-java     # Java 25, Quarkus 3.40, architecture
└── adapters/qwen3-14b-react    # React 19.3, Tailwind 4.3, a11y
```

Why this is better than one mixed adapter:

| | Three adapters | One mixed adapter |
|---|---|---|
| Per-track quality | **Highest** | Diluted |
| Interference | None | Kali refusal vs coding style; Java vs React syntax |
| Disk | ~0.3–0.8 GB total | ~0.2–0.4 GB |
| Serving | Three ports or per-request adapter | One model, one port |
| Evaluation | Per-track clean | Must check every track |
| Data mixing | None | Ratios, mode tokens, caps |
| Updating one track | Retrain one adapter | Retrain everything |

LoRA adapters are cheap. Three adapters on one base is the pragmatic “one model for all” answer: **one base model to download, quantize, and serve; three small specializations.**

### 3.1 Training order

1. **Kali adapter** first. It teaches the tool-call contract, recovery and scope/refusal behaviour.
2. **Java adapter** second. It teaches repository editing, Maven and architecture.
3. **React adapter** third. It teaches component/design-system/browser verification.

They do not share an adapter; the order only affects when you have a usable model. If you want one base checkpoint and three adapters, train each from the **same** `Qwen3-14B-4bit`, not adapter-on-adapter.

### 3.2 Serving

**Option A — three fused models on three ports** (simplest, most robust):

```bash
mlx_lm.fuse --model mlx-community/Qwen3-14B-4bit \
  --adapter-path adapters/qwen3-14b-kali --save-path fused/qwen3-14b-kali-4bit
mlx_lm.fuse --model mlx-community/Qwen3-14B-4bit \
  --adapter-path adapters/qwen3-14b-java --save-path fused/qwen3-14b-java-4bit
mlx_lm.fuse --model mlx-community/Qwen3-14B-4bit \
  --adapter-path adapters/qwen3-14b-react --save-path fused/qwen3-14b-react-4bit

mlx_lm.server --model fused/qwen3-14b-kali-4bit  --port 8080
mlx_lm.server --model fused/qwen3-14b-java-4bit  --port 8081
mlx_lm.server --model fused/qwen3-14b-react-4bit --port 8082
```

Run only the adapter you need at a time on 32 GB; one 14B 4-bit server plus a browser is already a lot of memory. Three servers at once is a bad idea.

**Option B — one server, per-request adapter**:

`mlx_lm.server` accepts `--adapter-path` and a per-request `adapters` field. This is convenient for a router, but client support varies. If your OpenAI client cannot send `adapters`, use Option A or restart the server with the right adapter.

**Option C — llama.cpp** for GGUF and better tool-call fidelity:

Convert each fused model to GGUF, quantize to Q4_K_M, and serve with `llama-server --jinja` on separate ports. See [`10-export-serving.md`](10-export-serving.md).

### 3.3 Routing

A small router is better than asking one model to infer the domain from a vague prompt:

```text
if request touches authorized hosts / scan / exploit / report:
    port 8080, Kali system prompt + Kali tools + scope file
elif repository contains pom.xml / build.gradle / .java:
    port 8081, Java system prompt + repo tools + Maven
elif repository contains package.json / .tsx / tailwind:
    port 8082, React system prompt + repo/browser tools
else:
    ask the user which mode, or use the base model without an adapter
```

Do not rely on the model to choose between three tool schemas. Pass **only** the relevant tools.

---

## 4. If you insist on one mixed adapter

Sometimes one adapter is genuinely better: one endpoint, one deployment, one model card, simpler ops. It can work if you accept a quality trade-off and design the data carefully.

### 4.1 Mixed data layout

```text
data/unified/
├── raw/                 # optional snapshots
├── interim/             # per-track rows before mixing
├── processed/
│   ├── train.jsonl
│   ├── valid.jsonl
│   ├── test.jsonl
│   └── manifest.json
└── golden/              # per-track golden sets, never trained on
```

Build it from the three track splits:

```bash
python scripts/mix_datasets.py \
  --source kali=data/processed/train.jsonl \
  --source java=data/java/processed/train.jsonl \
  --source react=data/react/processed/train.jsonl \
  --weights kali=0.40,java=0.30,react=0.30 \
  --out data/unified/processed \
  --arguments-format string
```

`--arguments-format string` is mandatory: **Qwen3 expects OpenAI-style JSON strings**, while the Java/React rows were prepared for Qwen2.5-Coder with **objects**. The mixer converts them and records the choice in `manifest.json`.

### 4.2 Ratios

| Bucket | Share | Why |
|---|---:|---|
| Kali/agent | 40% | Tool contract, scope/refusal, recovery; smallest public data, highest risk |
| Java/Quarkus | 30% | Repository edits, Maven, architecture |
| React/Tailwind | 30% | Components, design tokens, browser checks |
| Cross-domain | included | Repo navigation, test-failure recovery, refusal, instruction following |

A 1/3 split is also reasonable. Do not let Java/React dominate because their rows are longer; weight by **examples**, then check **tokens per track** in `manifest.json` and `token_stats.py`.

### 4.3 Mode token

Put an explicit mode at the top of the system prompt:

```text
MODE: KALI
```

or

```text
MODE: JAVA
```

or

```text
MODE: REACT
```

Then the task. The model learns to switch behaviour without a separate router. At inference, the harness sets the mode and supplies the matching tools.

Do not encode the mode only in the user text (“this is a Java task”) and then expect it to generalize; make it a stable system-level field.

### 4.4 Tools per row, not a mega-schema

A unified adapter does **not** need one tool schema containing Kali, Maven and Playwright tools. Each training row carries only its track's tools:

- Kali: 6–10 typed tools (`scope_check`, `nmap_scan`, `http_probe`, …).
- Java: `read_file`, `search_code`, `apply_patch`, `run_maven`, `arch_check`, `git_diff`.
- React: `read_file`, `search_code`, `apply_patch`, `run_npm`, `run_playwright`, `screenshot`, `axe_check`, `git_diff`.

At inference, pass only the relevant set. A 14B model with 30 tools will choose worse than the same model with 8.

### 4.5 Training

```bash
mlx_lm.lora --config configs/qwen3-14b-unified.yaml
```

The config uses:

- Qwen3-14B 4-bit;
- 24 of 40 blocks;
- rank 16, `scale: 2.0`;
- LR `1e-4`, effective batch 8;
- sequence 2,048;
- `mask_prompt: true`.

Start with 1–2 epochs (~200–300 iterations for 2–3M tokens), not a fixed 600. Evaluate every track before scaling.

Verify the adapter before the run:

```bash
python scripts/check_lora.py \
  --model mlx-community/Qwen3-14B-4bit \
  --num-layers 24 --rank 16 --scale 2
```

Expect 24 adapted blocks of 40, seven projections per block (168 modules) and roughly 36M trainable parameters. All 40 blocks is ~60M and ~280 modules; test it only after the 24-block adapter is evaluated.

### 4.6 The quality trade-off

Expect one or more of:

- Java/React code style drifts toward the base Qwen3 style; fewer Quarkus/Tailwind idioms.
- Kali refusal becomes less crisp if coding rows dominate.
- The adapter spends capacity on three vocabularies; per-track loss is higher than a dedicated adapter.
- A single sequence budget must fit three very different row shapes.

A reasonable acceptance bar:

| Track | Dedicated adapter | Unified adapter tolerance |
|---|---:|---:|
| Kali tool-call validity | >0.95 | ≥0.90 |
| Kali unsafe-action rate | 0 | 0 |
| Java compile rate | >0.90 | ≥0.85 |
| Java test pass | >0.45 | ≥0.40 |
| React typecheck/build | >0.90 | ≥0.85 |
| React axe serious/critical | 0 | 0 |

If the unified adapter misses these, keep the three-adapter design.

---

## 5. Thinking mode

Qwen3-14B has **hybrid thinking**:

- default `enable_thinking=True`;
- `/think` and `/no_think` soft switches;
- `enable_thinking=False` for a hard non-thinking mode.

With MLX `--mask-prompt`, the completion starts at `assistant\n` and includes the think block, so the adapter learns to emit ` thinking…`. Train and serve with the **same** setting:

- **Thinking on** (default): better reasoning, more tokens, more latency; use sequence 2,048+ and check truncation.
- **Thinking off**: shorter, faster; requires patching the local `tokenizer_config.json` to default `enable_thinking=False` before training, because `mlx_lm.lora` has no chat-template flag. Do not train thinking-on and serve thinking-off; you get a duplicated or empty think block.

For a unified adapter, **thinking on** is the simpler, safer default. If you need no-think for code latency, train a separate no-think adapter from a patched tokenizer config; do not mix the two.

---

## 6. What to evaluate

A unified adapter must be evaluated **per track**, not on an average:

### Kali

- tool-call validity;
- tool/argument accuracy;
- multi-step task success;
- unsafe-action rate (must be 0);
- scope/refusal precision.

Use [`09-evaluation.md`](09-evaluation.md) and the private 240-task golden set.

### Java

- `./mvnw -q -DskipITs test`;
- compile rate;
- ArchUnit pass rate;
- review precision;
- no `javax.*`/Spring/legacy RESTEasy.

Use [`20-java-finetuning.md`](20-java-finetuning.md) §8.

### React

- `tsc --noEmit`, `vite build`;
- Vitest/Playwright;
- axe serious/critical = 0;
- no horizontal overflow at 320/768/1440;
- visual rubric and requirement satisfaction.

Use [`24-react-finetuning.md`](24-react-finetuning.md) §8.

### The decision rule

Keep the unified adapter only if:

1. no track loses more than ~5 points versus its dedicated adapter;
2. the unsafe/security gates still pass;
3. the operational simplicity is worth the loss.

Otherwise, **one base + three adapters** is the better “one model for all”.

---

## 7. What not to do

- **Do not train Qwen3-Coder-30B-A3B locally** on 32 GB. It is a 30.5B MoE; use it as teacher/judge/serving.
- **Do not train Qwen3-32B** locally. 32.8B dense does not fit.
- **Do not mix Qwen2.5-Coder `arguments` objects with Qwen3 strings.** Re-render the rows with `mix_datasets.py --arguments-format string`.
- **Do not put all domain tools in one schema.** 6–10 tools per request.
- **Do not train Qwen3.5/3.6/3.8 LoRA on this Mac.** GDN training is not reliable in MLX yet.
- **Do not evaluate only the average.** A unified model can look fine on average and fail safety or Java compilation.
- **Do not serve three 14B models at once on 32 GB.** One adapter at a time.

---

## 8. Cost and time

| Plan | Training | Wall clock (est.) | Disk |
|---|---|---:|---:|
| Three adapters, 2M tokens each | 3 × ~1 epoch | 45–75 h | 0.5 GB adapters + 3 × 8.5 GB fused (optional) |
| One mixed adapter, 4M tokens | ~1.5 epochs | 50–70 h | 0.3 GB adapter + one fused model |
| Qwen3-8B unified, 3M tokens | all 36 layers | 20–30 h | 0.3 GB |
| Qwen3-Coder-30B inference only | none | minutes to start | ~18 GB 4-bit |

The three-adapter plan costs more wall clock but gives you a usable model earlier (after the Kali adapter) and avoids a full retrain when one track changes.

---

## 9. The answer in one table

| If you want… | Use | Train locally? |
|---|---|---|
| **One Qwen3 base for all three, best local quality** | **Qwen3-14B + 3 LoRA adapters** | **Yes** |
| One adapter, simplest ops | Qwen3-14B + mixed data | Yes, with trade-offs |
| One model to serve all three, no training | Qwen3-Coder-30B-A3B (code-heavy) or Qwen3-30B-A3B-2507 (general) | No |
| Best code quality | Qwen3-Coder-30B-A3B | No |
| Best Java/React adapter quality | Qwen2.5-Coder-14B track adapters | Yes |
| Best overall with a rented GPU | Qwen3.5-9B or Qwen3.8-27B | Rented GPU |

**Bottom line:** **Qwen3-14B is the “one Qwen3 for all” on this machine.** It is the only Qwen3 dense model that trains reliably in 32 GB and is strong enough for all three domains. Keep one base and three adapters; use a mixed adapter only when operational simplicity matters more than per-track quality.

---

## 10. Sources

- Qwen3-14B — <https://huggingface.co/Qwen/Qwen3-14B>
- Qwen3-8B — <https://huggingface.co/Qwen/Qwen3-8B>
- Qwen3-32B — <https://huggingface.co/Qwen/Qwen3-32B>
- Qwen3-30B-A3B — <https://huggingface.co/Qwen/Qwen3-30B-A3B>
- Qwen3-30B-A3B-Instruct-2507 — <https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507>
- Qwen3-Coder-30B-A3B-Instruct — <https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct>
- Qwen3-Coder repository — <https://github.com/QwenLM/Qwen3-Coder>
- Qwen3 technical report — <https://arxiv.org/abs/2505.09388>
- Qwen3.5 release — <https://github.com/QwenLM/Qwen3.8>
- Qwen3.8-27B — <https://huggingface.co/Qwen/Qwen3.8-27B>
- MLX-LM LoRA — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md>
- MLX-LM server — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/server.py>
- [17 — Fine-tuning update](17-finetune-2026-update.md)
- [18 — Java landscape](18-java-quarkus.md)
- [22 — React & Tailwind](22-react-tailwind.md)
