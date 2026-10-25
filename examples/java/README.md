# Java / Quarkus examples

| File | Purpose |
|---|---|
| `tools.json` | Typed agent tools: read/search/write/apply patch, Maven, ArchUnit, git diff |
| `system-prompt.md` | Java 25 / Quarkus 3.40 / SOLID / test-first system prompt |
| `AGENTS.md` | Short rules OpenCode reads from the repo |
| `opencode.json` | Local Qwen2.5-Coder provider and permissions |
| `teacher-system.md` | System prompt for synthetic task generation |
| `task-prompt.md` | Task taxonomy for synthetic generation |
| `prompts/solver.md` | Plan + patch + tests |
| `prompts/test.md` | JUnit/Quarkus test generation |
| `prompts/refactor.md` | SOLID/hexagonal refactor |
| `prompts/review.md` | Architecture/code review |
| `prompts/migration.md` | Quarkus/Java upgrade |
| `Dockerfile.java-lab` | JDK 25 + Maven 3.9 verification container |

The canonical training rows are in `data/examples/java-train.example.jsonl`; the golden-set example is `data/examples/java-golden.example.jsonl`.

**Qwen2.5-Coder tool arguments are objects, not JSON strings.** Use `prepare_dataset.py --arguments-format dict`.
