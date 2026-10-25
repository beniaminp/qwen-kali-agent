# Gemma 4 12B data layout

Keep Gemma data separate from Qwen. The chat template and tool-call format differ,
even when the source examples are the same.

```text
data/gemma4/
├── kali/
│   ├── raw/
│   ├── interim/
│   ├── processed/{train,valid,test}.jsonl
│   └── golden/golden.jsonl
├── java/
├── react/
└── unified/
    ├── interim/
    ├── processed/
    └── golden/
```

## Rules

1. **Gemma tool arguments are JSON objects** (`mapping`), not JSON strings. The
   template raises an error on a string:
   `tool_calls[].function.arguments must be a JSON object (mapping), not a string`.
   Use `prepare_dataset.py --arguments-format dict`.
2. `reasoning_content` is rendered as `<|channel>thought … <channel|>`. Do not
   include previous turns' thinking in history except tool-call turns; Gemma 4
   expects history to contain final answers.
3. Tool results use `role: "tool"` with `tool_call_id`; Gemma resolves the id to
   the function name for `<|tool_response>`.
4. Do not reuse Qwen-rendered strings. Store canonical `messages` + `tools` and
   let Gemma's template render.
5. Images belong in a separate multimodal dataset and a `mlx-vlm` run. Do not put
   image content into a text-only `mlx_lm.lora` row.
6. Split by repository/commit, not random row. Keep a private golden set.
7. Never train on `golden/`.

## Build a track

```bash
mkdir -p data/gemma4/{kali,java,react,unified}

# Normalize for Gemma (dict arguments)
python scripts/prepare_dataset.py data/gemma4/react/interim/all.jsonl \
  --out data/gemma4/react/processed --arguments-format dict

python scripts/validate_gemma4_dataset.py data/gemma4/react/processed/train.jsonl \
  --tools examples/react/tools.json --strict
```

Unified mix:

```bash
python scripts/mix_datasets.py \
  --source kali=data/gemma4/kali/processed/train.jsonl \
  --source java=data/gemma4/java/processed/train.jsonl \
  --source react=data/gemma4/react/processed/train.jsonl \
  --weights kali=0.40,java=0.30,react=0.30 \
  --out data/gemma4/unified/processed --arguments-format dict
```
