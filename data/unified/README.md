# Unified data layout (one Qwen3 base, mixed tracks)

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

**Qwen3 uses OpenAI-style JSON strings for tool arguments**, unlike Qwen2.5-Coder. The mixer converts Java/React `arguments` objects back to strings. Do not skip this step.

Rules:

1. Never train on `golden/`.
2. Evaluate every track separately; do not report only an average.
3. Check token share per track, not just row share: Java and React rows are longer.
4. Keep provenance: `meta.track`, source file, original repo/commit, license.
5. Use a mode token (`MODE: KALI|JAVA|REACT`) in the system prompt and pass only that track's tools.

See [`docs/26-one-model-for-all.md`](../../docs/26-one-model-for-all.md).
