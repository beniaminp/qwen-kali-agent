# Unified examples

| File | Purpose |
|---|---|
| `mode-prompts.md` | Three system-prompt modes and their tool sets for one mixed adapter |

The unified adapter uses the track tool schemas in `examples/tools/`, `examples/java/tools.json` and `examples/react/tools.json` — never all at once.

**Data:** `data/unified/processed`, built with `scripts/mix_datasets.py --arguments-format string`.
**Config:** `configs/qwen3-14b-unified.yaml`.
**Docs:** [`docs/26-one-model-for-all.md`](../../docs/26-one-model-for-all.md).
