# Gemma 4 12B track

Everything needed to fine-tune and serve **Gemma 4 12B** alongside the Qwen tracks.

| File | Purpose |
|---|---|
| `system-prompt.md` | Mode-based system prompt: Kali, Java, React |
| `AGENTS.md` | Short rules OpenCode reads from the repo |
| `opencode.json` | Local Gemma provider and permissions |
| `tools/README.md` | Which tool schema to pass for each mode |
| `prompts/solver.md` | Plan + patch + checks |
| `prompts/component.md` | React component + test |
| `prompts/page.md` | React page + states + Playwright |
| `prompts/review.md` | Code/UI/a11y review |
| `prompts/migration.md` | React/Tailwind/Router/TS migration |
| `prompts/a11y.md` | WCAG 2.2 audit/fix |
| `prompts/test.md` | Vitest/Playwright tests |
| `prompts/design.md` | Screenshot → design spec |

The model files, configs and scripts live outside this directory:

- configs: `configs/gemma4-12b-*.yaml`
- data: `data/gemma4/`
- validator: `scripts/validate_gemma4_dataset.py`
- comparison: `scripts/compare_models.py`
- docs: `docs/27-gemma4-12b.md` … `docs/31-gemma-vs-qwen.md`
