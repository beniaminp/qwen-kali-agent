# 30 — Generating Gemma 4 12B training data

*Last verified: 25 October 2026. The teacher/API mechanics are shared with [07 — Synthetic data](07-synthetic-data.md); the Gemma-specific template, tool format and VLM spec path are new.*

The public data problem is the same as for Qwen: no dataset reliably teaches **your** tool schema, **your** design tokens, **your** Quarkus version, or **your** scope policy. The answer is a repo-grounded pipeline with a real verifier.

```text
permissive repo + design tokens + commits
        │
        ▼
teacher proposes a task (JSON)
        │
        ▼
teacher proposes a patch + tests (JSON)
        │
        ▼
static validator: Gemma argument shape, names, ids, versions, secrets
        │
        ▼
sandbox: Maven / npm / Playwright / Kali policy
        │
        ▼
keep only verified trajectories → canonical JSONL (arguments as mappings)
```

Use Gemma as the **student**; use DeepSeek, a larger Qwen, or a frontier API as the **teacher**. Use Gemma-as-VLM only to write design specs or to judge screenshots.

---

## 1. What to generate

| Gap | Task family | Verifier |
|---|---|---|
| React 19 idioms | Actions, `useActionState`, `ref` as prop, `use` | Vitest + typecheck |
| Tailwind 4 tokens | `@theme`, semantic classes, dark mode | build + screenshot |
| Design systems | primitive/pattern/feature split, CVA | tests + review |
| Accessibility | keyboard, focus, labels, target size | axe + Playwright |
| Repository edits | multi-file feature, repair | Maven / npm |
| Recovery | inject type/test/visual failure, fix | failing then green |
| Vision | screenshot → design spec → code | VLM spec + rendered diff |
| Migration | React 18→19, Tailwind 3→4, Router 7→8, TS 6→7 | build + screenshot |
| Safety | secrets, XSS, client auth, scope | rubric + refusal |
| Kali | tool selection, scope, evidence, refusal | policy + golden set |

Do not generate 10,000 paraphrases of “build a dashboard”. Generate a matrix: track × layer × difficulty × failure mode × viewport.

---

## 2. Teachers

| Teacher | Best for | Caveats |
|---|---|---|
| DeepSeek V4.1 Flash | bulk tasks/components | cheapest; validate JSON |
| DeepSeek V4-Pro | hard pages/refactors | ~3–4× price |
| Qwen3-Coder-30B-A3B 4-bit | private code, offline | ~18–20 GB; slower |
| Qwen3-Coder-Next | hardest repo tasks | 80B; API/rented GPU |
| Gemma 4 12B (VLM) | screenshot → design spec, visual review | not the text student |
| Frontier API | gold examples, rubric calibration | cost/terms |

For a first run, Flash for volume and Pro for the hard 10–20% is enough. Use a local model for private repositories.

---

## 3. Gemma-specific output shape

The teacher should return JSON that can be converted to canonical messages. Do **not** ask it to return Gemma-rendered `<|tool_call>` text; the template does that. Ask for OpenAI-style `tool_calls` with **object arguments**:

```json
{
  "task": "...",
  "plan": ["..."],
  "patch": "diff --git ...",
  "files": ["..."],
  "tests": ["..."],
  "commands": ["npm run typecheck", "npm test -- --run"],
  "tool_calls": [
    {"id": "call_1", "type": "function",
     "function": {"name": "read_file", "arguments": {"path": "src/App.tsx"}}}
  ],
  "evidence": ["..."],
  "risks": ["..."]
}
```

Then normalize with:

```bash
python scripts/prepare_dataset.py data/gemma4/react/interim/all.jsonl \
  --out data/gemma4/react/processed --arguments-format dict
```

A string `arguments` value is a hard error in Gemma’s template, not a warning.

---

## 4. Prompts

Reusable files live in `examples/gemma4/`:

- `system-prompt.md` — mode-based Kali/Java/React prompt.
- `prompts/solver.md` — plan + patch + checks.
- `prompts/component.md` — component + variants + test.
- `prompts/page.md` — page + states + responsive.
- `prompts/test.md` — Vitest/Playwright.
- `prompts/review.md` — findings by severity.
- `prompts/migration.md` — React/Tailwind/Router/TS.
- `prompts/a11y.md` — WCAG 2.2 audit/fix.
- `prompts/design.md` — screenshot → design spec.

### 4.1 Task generation

```bash
export DEEPSEEK_API_KEY="..."
python scripts/generate_synthetic.py \
  --mode tasks --tasks 300 --batch 10 \
  --model deepseek-flash \
  --system-file examples/react/teacher-system.md \
  --prompt-file examples/react/task-prompt.md \
  --out data/gemma4/react/interim/tasks.jsonl
```

The React task prompt is reusable; the output rows are track-agnostic. For Java/Kali use the corresponding prompt from `examples/java/` or `examples/tools/`.

### 4.2 Solving

```python
import json
import os
from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
template = Path("examples/gemma4/prompts/solver.md").read_text()
system = Path("examples/gemma4/system-prompt.md").read_text()

for task in tasks:
    prompt = (template
              .replace("{mode}", task["track"].upper())
              .replace("{task}", task["task"])
              .replace("{repo_context}", json.dumps(task["repo_context"]))
              .replace("{constraints}", json.dumps(task["constraints"]))
              .replace("{acceptance}", json.dumps(task["acceptance"]))
              .replace("{test_command}", task["test_command"]))
    r = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=12000,
    )
    candidate = json.loads(r.choices[0].message.content)
    # validate schema, apply patch, run checks, keep only verified
```

Use `.replace`, not `.format`; the prompt contains JSON braces. A full page often needs 12k output tokens; a component 6–8k is enough.

### 4.3 Repair loop

If the build fails:

1. Capture the first 80 lines of the compiler/test error.
2. Send task + patch + error: “Here is the real output; produce a minimal fix.”
3. Max 2 attempts.
4. Keep the final verified trajectory; keep the failed attempt only if it teaches recovery.

Do not let the teacher “fix” by deleting the failing test or adding `@ts-ignore`.

---

## 5. Execution harness

### 5.1 React

```bash
docker build -t web-lab -f examples/react/Dockerfile.web-lab .
```

Run a candidate in a disposable clone:

```bash
rm -rf /tmp/gemma-react && cp -a ~/src/my-app /tmp/gemma-react
cd /tmp/gemma-react
git checkout <base_commit>
git apply --check model.patch && git apply model.patch

docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /tmp/gemma-react:/workspace -v "$HOME/.npm":/tmp/.npm \
  -w /workspace web-lab \
  bash -lc 'npm ci --prefer-offline && npm run typecheck && npm run lint && npm test -- --run && npm run build'
```

Then Playwright/axe:

```bash
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /tmp/gemma-react:/workspace -v "$HOME/.npm":/tmp/.npm \
  -w /workspace web-lab npx playwright test
```

Use the `reducedMotion: "reduce"` and `animations: "disabled"` settings from [25 — React synthetic data](25-react-synthetic.md) §6.

### 5.2 Java

```bash
docker build -t java-lab -f examples/java/Dockerfile.java-lab .
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /tmp/gemma-java:/workspace -v "$HOME/.m2":/tmp/.m2 \
  -w /workspace java-lab \
  bash -lc './mvnw -q -DskipITs test'
```

Use JDK 25 and Quarkus 3.40. Verify ArchUnit and reject `javax.*`.

### 5.3 Kali

Use the isolated lab from [11 — Kali agent](11-kali-agent.md) and the policy layer from [12 — Safety & legal](12-safety-legal.md). The verifier is not “the model said it worked”; it is scope + evidence + no out-of-scope action.

### 5.4 Gemma tool-call validation

Before a row enters training:

```bash
python scripts/validate_gemma4_dataset.py data/gemma4/react/processed/train.jsonl \
  --tools examples/react/tools.json --strict
```

Then render one row with `AutoProcessor.apply_chat_template` and confirm `<|tool_call>call:` and `<|tool_response>`.

At serving time, verify the server returns `tool_calls`, not raw text. A row that only looks right as text can still fail the agent loop.

---

## 6. Vision: screenshot → design spec

A text-only Gemma adapter cannot consume a screenshot. Use a VLM (Gemma 4 12B multimodal or another) to produce a **design.md**:

```md
# Design spec — Billing settings

## Layout
Two columns: 240px settings nav + max 720px content. Mobile stacked.

## Hierarchy
H1 "Billing"; plan card primary; invoice table secondary; danger zone last.

## Tokens
Canvas #FAFAFB; surface #FFFFFF; ink #1F2430; muted #6B7280; brand #6D28D9;
success #15803D; danger #B91C1C; radius 12; shadow card.

## Components
PlanCard, PaymentMethodRow, InvoiceTable, ConfirmDialog, Button, Badge.

## States
loading skeleton; empty invoices; error retry; success toast; disabled pending.

## Responsive
>=1024 two columns; 768 one; 320 no horizontal scroll; 44px targets.

## Accessibility
Landmarks; one H1; labels; focus-visible; 4.5:1; dialog focus trap; reduced motion.

## Acceptance
typecheck, lint, test, build; axe zero serious/critical; screenshots at 320/768/1440.
```

Then train/evaluate `design.md → React + Tailwind`. This is the highest-value use of Gemma’s vision without building a full multimodal adapter.

If you do want image → code end-to-end:

- use `mlx-vlm`, not `mlx_lm.lora`;
- build image+text rows, not text-only messages;
- hold out Design2Code/WebSight;
- evaluate rendered similarity + human/VLM pairwise;
- keep it a separate adapter and a separate report.

---

## 7. Quality gates

### 7.1 Hard gates (reject)

- Invalid JSON / missing fields.
- Patch does not apply.
- Typecheck/build/test fails (unless “diagnose only”).
- Test deleted or weakened.
- Axe serious/critical remains.
- Horizontal overflow at 320px.
- Tailwind 3 directives in a v4 task.
- `ReactDOM.render`, `useFormState`.
- Unsanitized `dangerouslySetInnerHTML`, secrets, client-only auth.
- **String tool arguments** (Gemma template hard error).
- Broken `tool_call_id` / undeclared tool.
- Qwen-style `<tool_call>` in content.
- Previous-turn thinking leakage.
- Diff > 400 lines or > 10 files without reason.
- Claimed checks with no output.

### 7.2 Soft rubric

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| Correctness | no typecheck | build/tests pass | edge cases covered |
| Architecture | one giant component | some separation | primitives/features/pages |
| Design system | raw palette | partial tokens | tokens + CVA + dark |
| Visual | broken hierarchy | acceptable | clear rhythm |
| A11y | keyboard traps | labels/focus | axe zero + semantics |
| Responsive | overflow | one width | 320/768/1440 |
| Tests | none/weak | happy path | failure + E2E |
| Evidence | invented | partial | real output |

Keep average ≥4 and no hard-gate failure. Human-review 10%.

### 7.3 Dedup and splits

- MinHash near-dedup 0.8; exact patch hash.
- Perceptual hash for near-identical screenshots.
- Split by repository/base commit.
- Decontaminate against Webdev-Bench, Design2Code, SWE-bench Multilingual/Multimodal, Multi-SWE-bench.
- Time-based holdout for commits.

---

## 8. Cost

Assume ~2k input + ~8k output tokens per task, 30% retries, one judge call.

| Examples | Output | Input | Flash off-peak | V4-Pro off-peak |
|---:|---:|---:|---:|---:|
| 1,000 | 10.4M | 2.6M | ~$6 | ~$22 |
| 5,000 | 52M | 13M | ~$30 | ~$110 |
| 20,000 | 208M | 52M | ~$125 | ~$440 |

Browser/CPU time: 1–5 minutes per task; 5,000 tasks = 80–400 CPU-hours. Verify 200 before generating 20k.

A local Qwen3-Coder-30B teacher has no API bill but needs ~18–20 GB and is slower.

---

## 9. Legal

- The Stack v3: ODC-BY; per-file licenses.
- OpenCodeInstruct: CC-BY-4.0.
- Magicoder: MIT; GPT-3.5 output; OpenAI policy.
- react-code-instructions: MIT; brand imitation risk.
- WebSight: CC-BY-4.0; real images; check provenance.
- Design2Code: ODC-BY; evaluation only.
- Webdev-Bench: MIT; evaluation only.
- SWE-bench Multilingual: MIT; evaluation only.
- Multi-SWE-bench: `other`; source repos.
- SWE-bench Multimodal: no dataset license tag; eval only.
- Gemma 4: Apache-2.0.
- shadcn/ui, Tailwind, React, Radix, TanStack, Storybook: MIT.
- Playwright: Apache-2.0.

Do not train on proprietary screenshots, customer data or paid component libraries. Get counsel for a commercial release.

---

## 10. Sources

- DeepSeek pricing — <https://api-docs.deepseek.com/quick_start/pricing>
- OpenCode — <https://opencode.ai/docs>
- Gemma 4 model card — <https://ai.google.dev/gemma/docs/core/model_card_4>
- Gemma 4 function calling — <https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4>
- MLX-VLM — <https://github.com/Blaizzy/mlx-vlm>
- Playwright screenshots — <https://playwright.dev/docs/test-snapshots>
- axe-core — <https://github.com/dequelabs/axe-core>
- [07 — Synthetic data](07-synthetic-data.md)
- [25 — React synthetic data](25-react-synthetic.md)
- [31 — Running both](31-gemma-vs-qwen.md)
