# Data layout

Datasets are **not** committed. This directory documents the expected structure and holds tiny examples only.

```text
data/
├── raw/                 # downloaded datasets, immutable
├── interim/             # converted/normalized JSONL, still being cleaned
├── processed/
│   ├── train.jsonl
│   ├── valid.jsonl
│   ├── test.jsonl
│   └── manifest.json
├── golden/
│   └── golden.jsonl     # private eval set, never trained on
├── benchmarks/          # BFCL/KaliBench/Cybench, never trained on
├── cards/
│   └── dataset-card.md
└── examples/
    ├── train.example.jsonl
    └── golden.example.jsonl
```

## Rules

1. **Raw is immutable.** Never edit a file in `raw/`; write a new file in `interim/`.
2. **One manifest per processed version.** Include source IDs, revisions, licenses, row/token counts, hashes and known issues.
3. **Never train on `golden/` or `benchmarks/`.** Keep them physically separate; do not symlink.
4. **No secrets or real customer data.** Run `validate_dataset.py` and a secret scan before training.
5. **Version the splits.** A good run must be reproducible from the manifest + git tag.

## Manifest skeleton

```json
{
  "version": "2026-10-25.v1",
  "created": "2026-10-25T12:00:00Z",
  "sources": [
    {"id": "Team-ACE/ToolACE", "revision": "main", "rows": 11300, "license": "Apache-2.0"}
  ],
  "counts": {"train": 0, "valid": 0, "test": 0},
  "tokens": {"train": 0, "valid": 0, "test": 0},
  "decontamination": {"method": "13-gram", "removed": 0},
  "known_issues": []
}
```

## Token budget

Plan in tokens, not rows. A first adapter is 1–2M tokens; a serious one is 3–5M. The 90M-token reference mix in the handbook is a pool design, not a single local run.
