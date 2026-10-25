# React / Tailwind examples

| File | Purpose |
|---|---|
| `tools.json` | Typed agent tools: read/search/write/patch, npm, Playwright, screenshot, axe, git diff |
| `system-prompt.md` | React 19.3 / Tailwind 4.3 / TypeScript 7 / accessibility / design system prompt |
| `AGENTS.md` | Short rules OpenCode reads from the repo |
| `opencode.json` | Local Qwen2.5-Coder provider and permissions |
| `teacher-system.md` | System prompt for synthetic task generation |
| `task-prompt.md` | Task taxonomy for synthetic generation |
| `prompts/solver.md` | Plan + patch + files + checks |
| `prompts/component.md` | Component + variants + story + test |
| `prompts/page.md` | Page/route + states + responsive |
| `prompts/test.md` | Vitest/RTL and Playwright tests |
| `prompts/refactor.md` | Design-system refactor |
| `prompts/review.md` | UI/architecture/a11y review |
| `prompts/migration.md` | React 18→19, Tailwind 3→4, Router 7→8, TS 6→7 |
| `prompts/a11y.md` | WCAG 2.2 AA audit/fix |
| `prompts/design.md` | Screenshot → design spec / visual critique |
| `Dockerfile.web-lab` | Playwright + Node verification container |

The canonical training rows are in `data/examples/react-train.example.jsonl`; the golden-set example is `data/examples/react-golden.example.jsonl`.

**Qwen2.5-Coder tool arguments are objects, not JSON strings.** Use `prepare_dataset.py --arguments-format dict`.
