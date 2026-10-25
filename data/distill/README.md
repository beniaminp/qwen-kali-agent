# Distillation data layout

```text
data/distill/
├── prompts.jsonl              prompt set (messages or prompt)
├── raw/                       teacher outputs, one JSONL per teacher
├── interim/                   normalized, filtered, deduplicated
├── 12b/
│   ├── interim/
│   └── processed/             train.jsonl, valid.jsonl, test.jsonl
├── 12b-long/                  sequence 4096 variant
├── e4b/
│   └── processed/
├── e2b/
│   └── processed/
├── logits/                    optional cached top-k logits
└── rejected/                  failed verification, with reasons
```

Rules:

- Raw teacher output is immutable. Never edit it in place.
- `interim/` is where normalization and filtering happen.
- `processed/` is the only directory `mlx_lm.lora` should read.
- Keep a manifest per split: counts, token estimates, teacher, date, filters.
- Keep `rejected.jsonl` so you can inspect teacher failures.
- Never put the golden set in any training split.

See [`docs/40-distill-data.md`](../../docs/40-distill-data.md).
