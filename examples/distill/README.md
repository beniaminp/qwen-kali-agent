# Distillation examples

| File | Purpose |
|---|---|
| `teacher-system.md` | Teacher system prompt: concise, verifiable, scope-safe answers |
| `prompts.jsonl` | Seven seed prompts across Kali, Java, React and reasoning |
| `tools.json` | Optional tool schemas for tool-aware teacher generation |

Generate sequence-level teacher data:

```bash
export DEEPSEEK_API_KEY=...
python scripts/generate_distill_data.py \
  --prompts examples/distill/prompts.jsonl \
  --system-file examples/distill/teacher-system.md \
  --model deepseek-flash \
  --out data/distill/raw/deepseek.jsonl \
  --n 2 --temperature 0.8 --max-tokens 2048
```

Use a local Gemma 4 12B teacher instead:

```bash
mlx_lm.server --model mlx-community/gemma-4-12b-it-4bit \
  --host 127.0.0.1 --port 8080 --max-tokens 4096

python scripts/generate_distill_data.py \
  --prompts examples/distill/prompts.jsonl \
  --system-file examples/distill/teacher-system.md \
  --model gemma-4-12b \
  --base-url http://127.0.0.1:8080/v1 \
  --allow-empty-key \
  --out data/distill/raw/gemma-12b.jsonl
```

The teacher output is a **proposal**, not ground truth. Verify code with Maven,
TypeScript/Playwright or the Kali golden set before it enters the training mix.

See [`docs/39-gemma-distillation.md`](../../docs/39-gemma-distillation.md).
