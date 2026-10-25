# Distillation track

The distillation track covers three directions:

1. a stronger teacher → **Gemma 4 12B** (specialize/compress behaviour);
2. **Gemma 4 12B** → **E4B / E2B** (cheap deployment, same tokenizer);
3. **Gemma 4 12B** as a data teacher for any student.

Start with [`docs/39-gemma-distillation.md`](../../docs/39-gemma-distillation.md).

| File | Purpose |
|---|---|
| `configs/gemma4-12b-distill.yaml` | 12B sequence-KD adapter, seq 2048 |
| `configs/gemma4-12b-distill-longctx.yaml` | 12B long-context variant, seq 4096 |
| `configs/gemma4-e4b-distill.yaml` | E4B student, all 42 blocks |
| `configs/gemma4-e2b-distill.yaml` | E2B student, all 35 blocks |
| `scripts/generate_distill_data.py` | teacher traces → canonical JSONL |
| `scripts/distill_logits.py` | GPU reference logit-KD trainer |
| `scripts/validate_gemma4_dataset.py` | Gemma 4 data validation |
| `examples/distill/` | teacher prompt, seed prompts, tools |

32 GB Mac: use sequence KD with MLX. Logit KD needs a custom trainer and a GPU.
