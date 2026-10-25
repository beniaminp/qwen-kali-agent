# 29 — Fine-tuning Gemma 4 12B

*Last verified: 25 October 2026 against MLX-LM 0.32.2 and the Gemma 4 12B config/chat template. The commands were not executed on the target Mac in this environment.*

This is the runbook for the Gemma track. It does not touch the Qwen configs or adapters.

The default is **24 of 48 blocks**, rank 16, `scale: 2.0`, sequence 2,048, effective batch 8. The React config raises sequence to 3,072; the all-layer path is an experiment.

---

## 0. The recipe in one table

| Choice | Value | Why |
|---|---|---|
| Base | `mlx-community/gemma-4-12b-it-4bit` | 11.95B, 48 layers, 4-bit, text-only loader |
| Method | 4-bit QLoRA, `lora` | base frozen; adapter small |
| Blocks | 24 of 48 | memory-safe default |
| Rank / scale | 16 / 2.0 | PEFT `alpha=32` equivalent |
| LR | `1e-4` | standard LoRA range with `scale=2` |
| Effective batch | 8 (1 × 8 accumulation) | LoRA prefers 8–16 |
| Sequence | 2,048 (React 3,072) | component/code rows fit; 256K is a serving feature, not a training budget |
| Iterations | 300 | ~1–2 epochs; adjust to tokens |
| Masking | `mask_prompt: true` | train only the final assistant turn |
| Checkpoint | every 50 | short run; keep options |

Gemma 4 12B is smaller than Qwen3-14B, but it has **48 layers** and a **262K vocabulary**. Do not assume it is faster just because it has fewer parameters; measure.

---

## 1. Preflight

```bash
cd ~/work/qwen-kali-agent
source .venv/bin/activate

python -c "import mlx_lm, mlx.core as mx; print(mlx_lm.__version__); print(mx.device_info())"
python scripts/check_env.py
```

Checklist:

- [ ] `mlx-lm>=0.32.2` (the Gemma 4 tool parser and `gemma4_unified` mapping).
- [ ] `mlx-community/gemma-4-12b-it-4bit` downloads and loads.
- [ ] `mx.get_peak_memory()` works.
- [ ] 100+ GB free disk.
- [ ] The Gemma data exists and is non-empty.
- [ ] No secrets in `data/gemma4/`.
- [ ] Qwen adapters and configs are untouched.
- [ ] You have a baseline evaluation of the base Gemma model.

```bash
python - <<'PY'
from mlx_lm import load
model, tok = load("mlx-community/gemma-4-12b-it-4bit")
print(type(model).__name__, len(model.layers))
PY
```

If this fails, fix the MLX/model version before touching the data.

---

## 2. Render the template before training

This is the step that catches most Gemma mistakes. Use the **multimodal processor** for the canonical template:

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google/gemma-4-12B-it")

messages = [
    {"role": "system", "content": "MODE: REACT\nYou are a senior React/Tailwind engineer."},
    {"role": "user", "content": "Read src/App.tsx."},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "read_file", "arguments": {"path": "src/App.tsx"}}}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "export default function App() {}"},
    {"role": "assistant", "content": "I read the file."},
]

tools = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a repository file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}]

print(processor.apply_chat_template(
    messages, tools=tools, tokenize=False,
    add_generation_prompt=True, enable_thinking=False,
))
```

Expect:

- a `<|turn>system` block containing `<|tool>…<tool|>`;
- `<|tool_call>call:read_file{path:<|"|>src/App.tsx<|"|>}<tool_call|>`;
- `<|tool_response>…<tool_response|>`;
- an empty `<|channel>thought\n<channel|>` when thinking is disabled.

If you see an error about `arguments must be a JSON object`, your row used a string. Convert with:

```bash
python scripts/prepare_dataset.py ... --arguments-format dict
```

### 2.1 Thinking

`mlx_lm.lora` has no `enable_thinking` switch. The Gemma template defaults to **thinking off** (`enable_thinking=false`). The no-think path still emits an empty thought block. Train and serve with the same setting.

If you want a thinking adapter, patch the local model's `tokenizer_config.json` to default `enable_thinking=true` (or use a wrapper that passes it) and verify the rendered sample. Do not train thinking-on and serve thinking-off; you get a mismatched channel.

---

## 3. Prepare the data

### 3.1 Build a track

```bash
mkdir -p data/gemma4/{kali,java,react,unified}

# Normalize for Gemma: arguments are mappings
python scripts/prepare_dataset.py data/gemma4/kali/interim/all.jsonl \
  --out data/gemma4/kali/processed --arguments-format dict \
  --split 0.9 0.05 0.05

python scripts/prepare_dataset.py data/gemma4/java/interim/all.jsonl \
  --out data/gemma4/java/processed --arguments-format dict

python scripts/prepare_dataset.py data/gemma4/react/interim/all.jsonl \
  --out data/gemma4/react/processed --arguments-format dict
```

### 3.2 Validate

```bash
python scripts/validate_dataset.py data/gemma4/react/processed/train.jsonl \
  --tools examples/react/tools.json

python scripts/validate_gemma4_dataset.py data/gemma4/react/processed/train.jsonl \
  --tools examples/react/tools.json --strict
```

The Gemma validator catches:

- string `arguments` (the template will reject them);
- broken/unknown `tool_call_id`;
- undeclared tool names;
- Qwen-style `<tool_call>` JSON in content;
- previous-turn thinking in history;
- secrets and missing metadata.

### 3.3 Token stats

```bash
python scripts/token_stats.py data/gemma4/react/processed/train.jsonl \
  --model google/gemma-4-12B-it
```

Use the 99th percentile, rounded to 256. Do not confuse the **256K serving context** with the **2K–3K training budget**; long rows cost memory and time.

### 3.4 Unified mix

```bash
python scripts/mix_datasets.py \
  --source kali=data/gemma4/kali/processed/train.jsonl \
  --source java=data/gemma4/java/processed/train.jsonl \
  --source react=data/gemma4/react/processed/train.jsonl \
  --weights kali=0.40,java=0.30,react=0.30 \
  --out data/gemma4/unified/processed --arguments-format dict
```

`dict` is mandatory for Gemma. `string` is for Qwen3; mixing them is the most common template error.

---

## 4. The configs

### 4.1 Kali — `configs/gemma4-12b-kali.yaml`

```yaml
model: mlx-community/gemma-4-12b-it-4bit
train: true
data: data/gemma4/kali/processed
fine_tune_type: lora
optimizer: adamw
num_layers: 24
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
adapter_path: adapters/gemma4-12b-kali
seed: 42
lora_parameters:
  rank: 16
  dropout: 0.05
  scale: 2.0
lr_schedule:
  name: cosine_decay
  warmup: 30
  arguments: [1.0e-4, 300, 1.0e-6]
optimizer_config:
  adamw:
    weight_decay: 0.01
```

### 4.2 Java — `configs/gemma4-12b-java.yaml`

Same shape, `data/gemma4/java/processed`, adapter `adapters/gemma4-12b-java`, sequence 2,048.

### 4.3 React — `configs/gemma4-12b-react.yaml`

Same shape, `data/gemma4/react/processed`, adapter `adapters/gemma4-12b-react`, **sequence 3,072**. If it OOMs, drop to 2,048 before dropping layers.

### 4.4 Unified — `configs/gemma4-12b-unified.yaml`

Same shape, `data/gemma4/unified/processed`, adapter `adapters/gemma4-12b-unified`.

### 4.5 How many iterations?

```text
iters ≈ epochs × dataset_tokens / (batch_size × grad_accumulation_steps × avg_tokens_per_sample)
```

A 2.5M-token dataset, average 1,800 tokens, effective batch 8, one epoch:

```text
iters = 2,500,000 / (8 × 1,800) ≈ 174
```

So 200 is one epoch; 300 is ~1.5 epochs. Use validation loss and the golden set, not a fixed 600.

---

## 5. Verify the adapter

```bash
python scripts/check_lora.py \
  --model mlx-community/gemma-4-12b-it-4bit \
  --num-layers 24 --rank 16 --scale 2
```

Expected:

- 48 blocks total; **24 adapted**.
- `q_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` at minimum; `k_proj` and sometimes `v_proj` depending on full vs sliding attention.
- Roughly **30–35M** trainable parameters for 24/48 at rank 16.
- ~60–70M for all 48.

Do not expect exactly 7 modules per block: Gemma 4 uses K=V in full-attention layers, so `v_proj` is absent there. `check_lora.py` only warns if fewer than four modules per adapted block.

If you see only `q_proj`/`v_proj`, stop.

---

## 6. Smoke run

Create a 100k-token subset and 40 iterations:

```bash
head -n 1500 data/gemma4/react/processed/train.jsonl > data/gemma4/react/processed/train.smoke.jsonl
mkdir -p data/gemma4/react/processed-smoke
cp data/gemma4/react/processed/train.smoke.jsonl data/gemma4/react/processed-smoke/train.jsonl
cp data/gemma4/react/processed/valid.jsonl data/gemma4/react/processed-smoke/valid.jsonl

mlx_lm.lora --config configs/gemma4-12b-react.yaml \
  --data data/gemma4/react/processed-smoke --iters 40 --save-every 20 \
  --adapter-path adapters/gemma4-12b-react-smoke
```

Check:

- finite loss, falling;
- **tokens/sec** — budget 18–28 for 12B at 2K, lower at 3K;
- peak memory 16–26 GB;
- rendered sample contains `<|tool_call>call:` and `<|tool_response>`;
- no `arguments must be a JSON object` error.

Fuse and test:

```bash
mlx_lm.fuse \
  --model mlx-community/gemma-4-12b-it-4bit \
  --adapter-path adapters/gemma4-12b-react-smoke \
  --save-path fused/gemma4-12b-react-smoke-4bit

mlx_lm.server --model fused/gemma4-12b-react-smoke-4bit \
  --host 127.0.0.1 --port 8083 --max-tokens 4096 \
  --temp 0.2 --top-p 0.95
```

Send 20 tasks. Check:

- React 19 function components, no class components or jQuery;
- Tailwind 4 (`@import "tailwindcss"`, `@theme`), not v3 directives;
- semantic tokens, not raw hex;
- loading/empty/error states;
- actual `tool_calls`, not raw text;
- honest “I did not run the tests” when it did not.

Do not launch the 14B/12B full run until the smoke output is plausible.

---

## 7. The main runs

```bash
# Kali
mlx_lm.lora --config configs/gemma4-12b-kali.yaml

# Java
mlx_lm.lora --config configs/gemma4-12b-java.yaml

# React
mlx_lm.lora --config configs/gemma4-12b-react.yaml

# One mixed adapter
mlx_lm.lora --config configs/gemma4-12b-unified.yaml
```

Healthy signals:

| Metric | Healthy | Warning |
|---|---|---|
| Train loss | falls to 0.3–0.7 | stuck >1.0 after 200 iters |
| Val loss | tracks train | rises for 2+ evals |
| Tokens/sec | 18–28 (2K) | <10 = memory/thermal |
| Peak memory | 16–26 GB | >27 GB = danger |
| It/sec | stable | collapses = swap/thermal |

Code loss is lower than chat loss; compare the adapter to the base on the same tasks, not across datasets.

### 7.1 If it OOMs

1. `max_seq_length: 3072` → `2048`.
2. `num_layers: 24` → `16`.
3. Confirm `grad_checkpoint: true`, `batch_size: 1`.
4. Close Docker, Chrome, the IDE and the Kali VM.
5. Move to a smaller model or a rented GPU.

Do not start by shrinking the dataset.

### 7.2 If loss is `nan`

- A prompt is at or over `max_seq_length`; with `mask_prompt` the completion is empty.
- The LR is too high; halve it.
- The data has an empty assistant message or a string `arguments` (the template would usually error first).
- Run the validator and the template render again.

---

## 8. Checkpoints and resume

```bash
mlx_lm.lora --config configs/gemma4-12b-react.yaml \
  --resume-adapter-file adapters/gemma4-12b-react/0000200_adapters.safetensors \
  --iters 100 \
  --adapter-path adapters/gemma4-12b-react-resume
```

Only weights are restored; optimizer state, LR schedule position and iteration count are not. Use a new adapter path and a fresh short cosine schedule.

Checkpoints are ~100–250 MB each. Delete old ones after evaluation.

---

## 9. Evaluate

### 9.1 Kali

Use the Kali golden set and the 09/31 harness:

- tool-call validity;
- tool/argument accuracy;
- multi-step success;
- unsafe-action rate = 0;
- scope/refusal precision.

### 9.2 Java

Use the 20 harness:

- `./mvnw -q -DskipITs test`;
- compile rate;
- ArchUnit pass;
- review precision;
- no `javax.*`/Spring/legacy RESTEasy.

### 9.3 React

Use the 24 harness:

- `tsc --noEmit`, `vite build`;
- Vitest/Playwright;
- axe serious/critical = 0;
- no horizontal overflow at 320/768/1440;
- visual rubric and requirement satisfaction.

### 9.4 Release gates

Use the same gates as the Qwen tracks. For Gemma, add:

- [ ] `tool_calls` populated by `mlx_lm.server`;
- [ ] no `arguments` string errors in a 100-row render;
- [ ] no previous-turn thinking leakage in multi-turn rows;
- [ ] if multimodal, a separate `mlx-vlm` evaluation.

---

## 10. Fuse and serve

```bash
mlx_lm.fuse \
  --model mlx-community/gemma-4-12b-it-4bit \
  --adapter-path adapters/gemma4-12b-react \
  --save-path fused/gemma4-12b-react-4bit

mlx_lm.server \
  --model fused/gemma4-12b-react-4bit \
  --host 127.0.0.1 --port 8083 \
  --max-tokens 4096 --temp 0.2 --top-p 0.95
```

Gemma’s recommended sampling for general use is temperature 1.0, top-p 0.95, top-k 64. For deterministic code and security work, a lower temperature (0.2) is reasonable; keep it fixed across the comparison.

### 10.1 Tool-call check

```bash
curl -s http://127.0.0.1:8083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma4-12b-react",
    "messages": [{"role": "user", "content": "Read src/App.tsx"}],
    "tools": [{"type": "function", "function": {
      "name": "read_file", "description": "Read a file.",
      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    }}]
  }' | python3 -m json.tool
```

Check `choices[0].message.tool_calls`; if empty and the content contains `<|tool_call>`, update MLX or use `llama-server --jinja`.

### 10.2 GGUF path

Gemma 4 12B uses the new `gemma4_unified` architecture. Verify that your `llama.cpp` build and `convert_hf_to_gguf.py` support it before relying on GGUF; if conversion fails, use the MLX path. A GGUF build is not required for the comparison.

```bash
mlx_lm.fuse \
  --model mlx-community/gemma-4-12b-it-4bit \
  --adapter-path adapters/gemma4-12b-react \
  --save-path fused/gemma4-12b-react-bf16 \
  --dequantize

python3 ~/work/llama.cpp/convert_hf_to_gguf.py \
  fused/gemma4-12b-react-bf16 \
  --outtype f16 --outfile models/gemma4-12b-react-f16.gguf

~/work/llama.cpp/build/bin/llama-quantize \
  models/gemma4-12b-react-f16.gguf \
  models/gemma4-12b-react-Q4_K_M.gguf Q4_K_M

llama-server -m models/gemma4-12b-react-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8083 -c 16384 -ngl 99 -fa auto --jinja
```

`--dequantize` creates a ~24 GB bf16 directory; delete it after conversion. See [10 — Export & serving](10-export-serving.md).

---

## 11. Run both stacks

| Stack | Base | Ports |
|---|---|---:|
| Qwen Kali | Qwen3-14B | 8080 |
| Qwen Java | Qwen2.5-Coder-14B | 8081 |
| Qwen React | Qwen2.5-Coder-14B | 8082 |
| **Gemma Kali/Java/React** | **Gemma 4 12B** | **8083–8085** |
| Gemma unified | Gemma 4 12B | 8086 |

Do not run all of them at once on 32 GB. Run one adapter per stack, compare, then stop it. Use [31 — Running both](31-gemma-vs-qwen.md) and `scripts/compare_models.py`.

---

## 12. Checklist

Before:

- [ ] `mlx-lm>=0.32.2`;
- [ ] Gemma template renders;
- [ ] data uses `--arguments-format dict`;
- [ ] validator strict passes;
- [ ] token p99 fits;
- [ ] `check_lora.py` shows MLP projections;
- [ ] 100+ GB free;
- [ ] Docker/VM/Chrome off;
- [ ] baseline saved;
- [ ] Qwen stack untouched.

After:

- [ ] best checkpoint by golden set;
- [ ] tool calls parse;
- [ ] typecheck/build/tests/axe on React;
- [ ] Maven/ArchUnit on Java;
- [ ] Kali unsafe rate = 0;
- [ ] fused model loads;
- [ ] GGUF path tested;
- [ ] model card + dataset manifest written;
- [ ] comparison report vs Qwen.

---

## 13. Sources

- Gemma 4 model card — <https://ai.google.dev/gemma/docs/core/model_card_4>
- Gemma 4 12B guide — <https://developers.googleblog.com/en/gemma-4-12b-the-developer-guide>
- Gemma 4 chat template — <https://huggingface.co/google/gemma-4-12B-it/blob/main/chat_template.jinja>
- Gemma 4 function calling — <https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4>
- MLX 4-bit checkpoint — <https://huggingface.co/mlx-community/gemma-4-12b-it-4bit>
- MLX-LM LoRA — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md>
- MLX Gemma 4 tool parser — <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/gemma4.py>
- [09 — Evaluation](09-evaluation.md)
- [10 — Export & serving](10-export-serving.md)
- [20 — Java fine-tuning](20-java-finetuning.md)
- [24 — React fine-tuning](24-react-finetuning.md)
