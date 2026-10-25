# React / Tailwind data layout

Keep the front-end track separate from Kali and Java.

```text
data/react/
├── raw/                 # cloned MIT repos, sampled WebSight/Stack, design refs
├── interim/             # mined commits, generated tasks, candidate patches
├── processed/
│   ├── train.jsonl
│   ├── valid.jsonl
│   ├── test.jsonl
│   └── manifest.json
├── processed-long/      # 2,049–4,096-token rows for the 4K config
├── golden/
│   └── golden.jsonl     # private evaluation tasks, never trained on
└── benchmarks/          # Webdev-Bench, Design2Code, SWE-bench JS/TS
```

Rules:

1. **Never train on `golden/` or `benchmarks/`.**
2. Split by **repository and base commit**, not by row; otherwise paraphrases leak.
3. Keep provenance: repo, commit, files, license, teacher model, test command, viewports.
4. Use `prepare_dataset.py --arguments-format dict` for Qwen2.5-Coder tool data.
5. Run `validate_react_dataset.py` before tokenization.
6. Do not store screenshots of proprietary sites or real customer data.

The processed format is the canonical `messages` + optional `tools` JSONL described in [`docs/06-data-format.md`](../../docs/06-data-format.md), with the Qwen2.5-Coder `arguments` exception.
