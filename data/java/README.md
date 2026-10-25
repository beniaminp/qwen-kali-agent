# Java / Quarkus data layout

Keep the Java track separate from the Kali track.

```text
data/java/
├── raw/                 # cloned permissive repos, dataset snapshots, docs
├── interim/             # mined commits, generated tasks, candidate patches
├── processed/
│   ├── train.jsonl
│   ├── valid.jsonl
│   ├── test.jsonl
│   └── manifest.json
├── golden/
│   └── golden.jsonl     # private evaluation tasks, never trained on
└── benchmarks/          # MultiPL-E, Multi-SWE-bench, SWE-bench Multilingual
```

Rules:

1. **Never train on `golden/` or `benchmarks/`.**
2. Split by **repository and base commit**, not by row; otherwise paraphrases leak.
3. Keep every row's provenance: repo, commit, file paths, license, teacher model, test command.
4. Use `prepare_dataset.py --arguments-format dict` for Qwen2.5-Coder tool data.
5. Run `validate_java_dataset.py` before tokenization.

The processed format is the canonical `messages` + optional `tools` JSONL described in [`docs/06-data-format.md`](../../docs/06-data-format.md), with the Qwen2.5-Coder `arguments` exception.
