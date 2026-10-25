# 25 — Generating React/Tailwind training data with another LLM

*Last verified: 25 October 2026. The API and OpenCode mechanics are shared with [07 — Synthetic data](07-synthetic-data.md); the front-end prompts, browser verification and visual gates are new.*

Public React datasets teach syntax and toy apps. They do not teach **your** design tokens, **your** component architecture, **your** responsive breakpoints, **your** accessibility bar, or the difference between a plausible screenshot and a working interface.

This chapter describes a repo-grounded pipeline:

```text
permissive React repo + design tokens + commits
        │
        ▼
teacher proposes a task (JSON)
        │
        ▼
teacher proposes a patch + tests (JSON)
        │
        ▼
static validator: schema, versions, a11y, security
        │
        ▼
sandbox: npm ci → typecheck → lint → test → build
        │
        ▼
browser: Playwright flows → screenshots → axe → responsive
        │
        ▼
human/VLM visual review → keep / repair / discard
        │
        ▼
canonical JSONL (Qwen2.5-Coder arguments as objects)
```

The teacher can be DeepSeek V4.1 Flash/Pro, a local Qwen3-Coder-30B-A3B, or a larger API model. The verifier is not an LLM: it is TypeScript, ESLint, Vitest, Vite, Playwright and axe.

---

## 1. What synthetic data must add

| Gap in public data | Synthetic task |
|---|---|
| React 19 idioms | Actions, `useActionState`, `ref` as prop, `use`, View Transitions |
| Tailwind 4 tokens | `@theme`, semantic classes, dark mode, variants |
| Component architecture | primitives/patterns/features/pages, composition |
| Repository context | multi-file changes, read before edit, patch format |
| Recovery | inject a type/test/visual failure, ask for a fix, verify |
| Accessibility | keyboard, focus, labels, axe, target size |
| Design quality | hierarchy, spacing, typography, states, responsive |
| Migration | React 18→19, Tailwind 3→4, Router 7→8, TS 6→7 |
| Safety | refuse secrets, XSS, client-only auth, destructive changes |

Do not generate 10,000 paraphrases of “build a dashboard”. Generate a **matrix**: layer × task family × difficulty × failure mode × viewport.

---

## 2. Teacher choices

| Teacher | Where | Best for | Caveats |
|---|---|---|---|
| **DeepSeek V4.1 Flash** | API | bulk tasks, components, tests | cheap; validate JSON; thinking tokens cost output |
| **DeepSeek V4-Pro** | API | hard pages, refactors, reviews | ~3–4× price; reserve for hard slices |
| **Qwen3-Coder-30B-A3B-Instruct 4-bit** | local MLX/llama.cpp | private code, offline generation | ~18–20 GB; 8K–16K practical; slower |
| **Qwen3-Coder-Next** | API/rented GPU | hardest repo tasks | 80B; not local |
| **Qwen3.8-27B 4-bit** | local | judge/reviewer | slow; good quality |
| **A VLM (Qwen3-VL class)** | local/API | screenshot → design spec | not the student; text-only coder cannot see images |
| **A frontier API** | API | gold examples, rubric calibration | cost/terms; no proprietary code without permission |

Use **Flash for volume**, **Pro for the hard 10–20%**, and a local model for private repositories. Use a VLM only to turn images into **text specs** or to judge visual output.

---

## 3. Seed sources

Use only code you have the right to use:

- `shadcn-ui/ui` (MIT) — components, tokens, Radix patterns.
- `tailwindlabs/tailwindcss` (MIT) — v4 idioms and tests.
- `facebook/react` (MIT) — React 19 patterns and tests.
- `radix-ui/primitives` (MIT) — accessibility primitives.
- `TanStack/query` (MIT) — server-state patterns.
- `storybookjs/storybook` (MIT) — stories and a11y.
- `microsoft/playwright` (Apache-2.0) — test patterns.
- `cfahlgren1/react-code-instructions` (MIT) — complete-app shape.
- The Stack v3 JS/TS/TSX with permissive `license_type` (ODC-BY).
- Your own product, with permission.

Do **not** seed from proprietary code, leaked data, unlicensed repositories, paid component libraries (Tailwind Plus), or real brand sites you do not have rights to replicate.

---

## 4. The ten generation strategies

| # | Strategy | Output | Verification |
|---|---|---|---|
| 1 | **Component spec** | component + variants + stories + tests | typecheck, Vitest, axe |
| 2 | **Page/route brief** | page + layout + states | build, Playwright, responsive |
| 3 | **Design-system refactor** | tokens/CVA extraction, no visual drift | tests + screenshot diff |
| 4 | **Tailwind 3 → 4 migration** | CSS-first config, renamed utilities | build + visual diff |
| 5 | **React 18 → 19 migration** | Actions, ref as prop, remove legacy | tests + build |
| 6 | **Accessibility audit/fix** | semantic HTML, focus, labels, keyboard | axe + keyboard test |
| 7 | **Test generation** | Vitest/RTL/Playwright | tests run and assert |
| 8 | **Bug injection/recovery** | broken commit → diagnosis → fix | failing test then green |
| 9 | **Design-to-code via VLM spec** | design.md → React + Tailwind | screenshot similarity + rubric |
| 10 | **Review/critique** | findings with file/line and fixes | rubric + human spot check |

Plus a negative slice: unsafe, under-specified, out-of-scope, invented API, false claim.

### 4.1 Component spec

Input: design tokens, 2–5 existing components, a test file, and a task.
Output: a component with typed props, variants, states, accessible markup and tests.

This is the highest-value bucket. It teaches the component shape, tokens, CVA, focus styles, and test style.

### 4.2 Page/route brief

Input: routes, layout primitives, a data hook, a design brief.
Output: a page with hierarchy, responsive layout, loading/empty/error/content states.

Require all four states. A page with only a happy path is a failed example.

### 4.3 Design-system refactor

Start from a working screen with raw palette classes, duplicated buttons, inconsistent radii. Ask for:

- semantic tokens;
- a CVA primitive;
- direct imports, no barrel;
- no visual change.

Verify with tests + before/after screenshots. A refactor that changes pixels without a reason is not behavior-preserving.

### 4.4 Tailwind 3 → 4 migration

Take a v3 page:

- `@tailwind base/components/utilities` → `@import "tailwindcss"`;
- `tailwind.config.js` → `@theme`;
- `shadow-sm` → `shadow-xs`;
- `rounded` → `rounded-sm`;
- `outline-none` → `outline-hidden` (or a real focus ring);
- `bg-opacity-50` → `bg-black/50`;
- `flex-shrink-*` → `shrink-*`.

Verify with build + screenshot diff. This is a high-value migration task because the model otherwise blends v3 and v4.

### 4.5 React 18 → 19 migration

- `ReactDOM.render` → `createRoot`;
- `forwardRef` → `ref` as prop where appropriate;
- `useFormState` → `useActionState`;
- `propTypes`/`defaultProps` on function components → TypeScript props;
- class components → function components only when the task asks;
- `Context.Provider` → `<Context>`.

Verify with typecheck, tests and build. Do not mix Next.js/RSC into a Vite SPA migration.

### 4.6 Accessibility audit/fix

Take a component with real issues:

- `div` with `onClick`;
- icon-only button without a label;
- missing `alt`;
- focus ring removed;
- low contrast;
- 16px target;
- dialog without focus trap;
- form error not linked.

Ask for a minimal fix. Verify with axe, keyboard navigation and a screen-reader spot check.

### 4.7 Test generation

Generate:

- Vitest + Testing Library for behavior via roles;
- Playwright for critical flows and responsive;
- axe assertions;
- visual snapshots for stable pages.

Reject tests that pass on the broken baseline (they assert nothing), use `Thread.sleep`/arbitrary timeouts, or assert implementation details.

### 4.8 Bug injection/recovery

Inject a realistic bug:

- stale closure in a form;
- missing `key`;
- controlled/uncontrolled input switch;
- race between two queries;
- layout shift from an unsized image;
- missing error state;
- focus lost after a dialog closes;
- Tailwind class conflict from `cn` misuse.

The training row should include the **failing output** and the **diagnosis**, not just the final patch.

### 4.9 Design-to-code via a VLM spec

A text coder cannot see a screenshot. Use a VLM to produce a structured `design.md`:

```md
# Design spec — Billing settings

## Product and audience
B2B SaaS admin; needs to change plan and payment method quickly.

## Layout
Two-column: settings nav (240px) + content (max 720px). Mobile: stacked.

## Hierarchy
H1 "Billing"; plan card primary; invoice table secondary; danger zone last.

## Palette
Canvas #FAFAFB; surface #FFFFFF; ink #1F2430; muted #6B7280; brand #6D28D9;
success #15803D; danger #B91C1C.

## Typography
Inter; H1 28/36 semibold; H2 18/28 semibold; body 14/22; caption 12/16.

## Components
PlanCard, PaymentMethodRow, InvoiceTable, ConfirmDialog, Button, Badge.

## States
loading skeleton; empty invoices; error retry; success toast; disabled while pending.

## Responsive
>=1024 two columns; 768 one column; 320 no horizontal scroll; 44px targets.

## Accessibility
Landmarks; one H1; labels; focus-visible; 4.5:1; dialog focus trap; reduced motion.

## Copy
Real plan names and prices; no lorem ipsum.

## Acceptance
typecheck, lint, test, build; axe zero serious/critical; screenshots at 320/768/1440.
```

Then ask the coder for React + Tailwind. Verify by rendering and comparing to the reference screenshot (SSIM/CLIP + human/VLM pairwise), not by asking the same teacher “does it look good?”.

If you need image input end-to-end, train a **VLM** (Qwen3-VL class or Design2Code-style) separately. Do not put images into a Qwen2.5-Coder row.

### 4.10 Review/critique

Ask for findings by severity with file/line, principle, issue and minimal fix. The judge checks:

- Is the finding real?
- Is the line plausible?
- Is the fix minimal?
- Does it avoid invented APIs?
- Does it distinguish “taste” from “bug”?

Review data is weaker than executed patches; keep it a small slice and label it.

---

## 5. Prompt templates

Reusable files live in `examples/react/`:

- `teacher-system.md` — global rules.
- `task-prompt.md` — task taxonomy.
- `prompts/solver.md` — plan + patch + tests.
- `prompts/component.md` — component + variants + stories.
- `prompts/page.md` — page + states + responsive.
- `prompts/test.md` — Vitest/Playwright.
- `prompts/refactor.md` — design system.
- `prompts/review.md` — findings.
- `prompts/migration.md` — React/Tailwind/Router/TS upgrades.
- `prompts/a11y.md` — audit/fix.
- `prompts/design.md` — design spec / visual critique.

### 5.1 Task generation with DeepSeek

```bash
export DEEPSEEK_API_KEY="..."
python scripts/generate_synthetic.py \
  --mode tasks --tasks 300 --batch 10 \
  --model deepseek-flash \
  --system-file examples/react/teacher-system.md \
  --prompt-file examples/react/task-prompt.md \
  --out data/react/interim/tasks.jsonl
```

The script uses JSON-object mode, retries and token accounting. It does **not** execute anything and does not solve the tasks.

### 5.2 Solving a task

Use the OpenAI-compatible API with `prompts/solver.md` and repo context:

```python
import json
import os
from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
template = Path("examples/react/prompts/solver.md").read_text()

for task in tasks:
    prompt = (template
              .replace("{task}", task["task"])
              .replace("{repo_context}", json.dumps(task["repo_context"]))
              .replace("{constraints}", json.dumps(task["constraints"]))
              .replace("{acceptance}", json.dumps(task["acceptance"]))
              .replace("{test_command}", task["test_command"]))
    r = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "system", "content": Path("examples/react/teacher-system.md").read_text()},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=12000,
    )
    candidate = json.loads(r.choices[0].message.content)
    # validate schema, apply patch, run checks, keep only verified
```

Use `.replace`, not `.format`: the prompt contains JSON braces. For a full page, 12k output tokens is often needed; for a component, 6–8k is enough.

### 5.3 A solver response

```json
{
  "plan": ["read tokens", "read Button", "add SettingsPage", "add states", "add tests"],
  "patch": "diff --git a/src/routes/settings.tsx b/src/routes/settings.tsx\n...",
  "files": ["src/routes/settings.tsx", "src/features/settings/components/ProfileForm.tsx"],
  "tests": ["src/features/settings/components/ProfileForm.test.tsx"],
  "commands": ["npm run typecheck", "npm run lint", "npm test -- --run", "npm run build"],
  "risks": ["focus management after save"],
  "notes": "uses design tokens; no new dependencies"
}
```

Validate the patch with `git apply --check` before applying. Reject responses with prose outside JSON.

### 5.4 Repair loop

If the build fails:

1. Capture the first 80 lines of the TypeScript/ESLint/Vitest/Playwright error.
2. Send the task, patch and error back: “Here is the real compiler/test output; produce a minimal fix.”
3. Max 2 attempts.
4. Keep the final **verified** trajectory, including the failed attempt if it teaches recovery.

Do not let the teacher “fix” by deleting the failing test or adding `@ts-ignore`.

---

## 6. Execution harness

### 6.1 Build the lab image

```bash
docker build -t web-lab -f examples/react/Dockerfile.web-lab .
```

The image is based on the official Playwright image (Chromium installed). Check `node -v` once; Vite 8 needs Node 20.19+ or 22.12+. It has no application dependencies; `npm ci` installs them.

### 6.2 Run a candidate

```bash
rm -rf /tmp/react-work && cp -a ~/src/my-app /tmp/react-work
cd /tmp/react-work
git checkout <base_commit>

git apply --check model.patch && git apply model.patch

docker run --rm \
  -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /tmp/react-work:/workspace \
  -v "$HOME/.npm":/tmp/.npm \
  -w /workspace web-lab \
  bash -lc 'npm ci --prefer-offline && npm run typecheck && npm run lint && npm test -- --run && npm run build'
```

Playwright:

```bash
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /tmp/react-work:/workspace -v "$HOME/.npm":/tmp/.npm \
  -w /workspace web-lab \
  npx playwright test
```

Axe and screenshots are part of the Playwright suite.

### 6.3 Offline builds

1. Run one online `npm ci` to populate the cache.
2. Use `npm ci --prefer-offline` or `--offline` with a complete cache.
3. Pin `package-lock.json`; no `latest` ranges in test repos.
4. Do not let the model add dependencies; if it must, require a justification and record it.

### 6.4 Playwright config

```ts
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  use: {
    baseURL: "http://127.0.0.1:4173",
    trace: "on-first-retry",
    reducedMotion: "reduce",
  },
  webServer: {
    command: "npm run preview -- --port 4173",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: !process.env.CI,
  },
  projects: [
    { name: "mobile", use: { ...devices["iPhone 13"] } },
    { name: "tablet", use: { ...devices["iPad Mini"] } },
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } },
  ],
});
```

`reducedMotion: "reduce"` makes screenshots and motion tests deterministic. Disable animations in `toHaveScreenshot`.

### 6.5 Axe and responsive checks

```ts
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("settings page is accessible and does not overflow", async ({ page }) => {
  await page.goto("/settings");

  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag22aa"])
    .analyze();

  const severe = results.violations.filter(
    (v) => v.impact === "serious" || v.impact === "critical",
  );
  expect(severe).toEqual([]);

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
});
```

### 6.6 Keyboard and focus

```ts
test("dialog traps focus and returns it", async ({ page }) => {
  await page.goto("/settings");
  await page.getByRole("button", { name: "Delete workspace" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("button", { name: "Delete workspace" })).toBeFocused();
});
```

Manual keyboard checks are still required; Playwright can prove focus order, not screen-reader quality.

### 6.7 Visual regression

```ts
await expect(page).toHaveScreenshot("settings-desktop.png", {
  fullPage: true,
  animations: "disabled",
  maxDiffPixelRatio: 0.01,
});
```

For refactors, capture the baseline **before** the patch, apply the patch, capture after, and compare. For new pages, compare against a reference design with a rubric and a human/VLM pairwise check; pixel diff is not meaningful without a baseline.

---

## 7. OpenCode as the agent in the loop

`examples/react/opencode.json` points OpenCode at the local MLX/llama.cpp server.

### 7.1 Project rules

OpenCode reads `AGENTS.md` from the repository. Put the React/Tailwind rules there: React 19, Tailwind 4.3, TypeScript 7-compatible source, semantic tokens, accessible primitives, `npm run typecheck`, no new dependencies without asking.

### 7.2 Run headless

```bash
cd /tmp/react-work
opencode run -m local/qwen25-coder-14b-react --format json \
  "Build a responsive billing settings page. Read the design tokens and existing form primitives first. Include loading, empty and error states. Run typecheck, lint, test and build. Return a diff."
```

Parse the event stream into OpenAI-style messages:

- assistant text → `content`;
- tool call → `tool_calls`;
- tool result → `role: "tool"`;
- reasoning → drop for Qwen2.5-Coder (no thinking template).

Normalize tool arguments to objects. Reject transcripts that claim checks passed without a `run_npm` or `run_playwright` result.

### 7.3 What OpenCode is good and bad at

**Good:** multi-step repo navigation, edits, browser/test loops, generating a transcript.
**Bad:** bulk throughput, strict JSON schemas, deterministic cost. Use the API for volume; use OpenCode for a few hundred high-quality trajectories and for operating the sandbox.

### 7.4 Playwright MCP

If you use a browser MCP server, give it only `localhost` URLs in the disposable container. Do not expose the host browser, cookies, corporate SSO or the open internet. The agent needs the rendered app, not your session.

---

## 8. Quality gates

### 8.1 Hard gates (reject)

- JSON invalid or missing fields.
- Patch does not apply with `git apply --check`.
- Typecheck fails.
- Lint fails on errors.
- Unit or E2E tests fail (unless the task is explicitly “diagnose only”).
- Test deleted or weakened to pass.
- `vite build` fails.
- axe serious/critical violations remain.
- Horizontal overflow at 320px.
- `@tailwind base`, `bg-opacity-*`, `flex-shrink-*`, dynamic class construction.
- `ReactDOM.render`, `useFormState` in a React 19 task.
- `dangerouslySetInnerHTML` unsanitized; secrets; client-only auth.
- New dependency without justification.
- Diff > 400 lines or > 10 files without a clear reason.
- Model claims checks passed but no output exists.

### 8.2 Soft gates (score)

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| Correctness | does not typecheck | typecheck/build pass | tests cover edge cases |
| Architecture | one giant component | some separation | primitives/features/pages + composition |
| Design system | raw palette, inconsistent | partial tokens | semantic tokens + CVA + dark mode |
| Visual quality | broken hierarchy | acceptable | clear hierarchy, rhythm, restraint |
| Accessibility | keyboard traps | labels/focus | axe zero + keyboard + semantics |
| Responsive | overflow | works at one width | 320/768/1440 + container queries |
| Tests | none/weak | happy path | failure paths + roles + E2E |
| Evidence | invented output | partial | real commands/screenshots |

Keep rows with average ≥ 4 and no hard-gate failure. Keep a 10% human-reviewed sample.

### 8.3 Dedup and splits

- MinHash near-dedup at 0.8 on normalized code + task.
- Exact dedup on patch hash.
- Perceptual hash for near-identical screenshots.
- Split by **repository and base commit**, not row.
- Decontaminate against Webdev-Bench, Design2Code, SWE-bench Multilingual, Multi-SWE-bench and SWE-bench Multimodal with 13-gram overlap.
- Keep a time-based holdout: train on commits before a date, test on later commits.

---

## 9. Cost

Assume a task is ~2k input tokens and ~8k output tokens, plus 30% retries and a judge call.

| Examples | Output | Input | Flash off-peak | V4-Pro off-peak |
|---:|---:|---:|---:|---:|
| 1,000 | 10.4M | 2.6M | ~$6 | ~$22 |
| 5,000 | 52M | 13M | ~$30 | ~$110 |
| 20,000 | 208M | 52M | ~$125 | ~$440 |

Add browser/CPU time: 1–5 minutes per task; 5,000 tasks is 80–400 CPU-hours. Run Playwright in parallel, but cap workers so a single Mac or CI runner does not thrash.

**Do not generate 20k before verifying 200.** The first 200 expose the schema, sandbox, design-token and licensing mistakes that would otherwise be replicated.

A local Qwen3-Coder-30B teacher has no API bill but is slower and needs 18–20 GB of memory. Use it for private code, not for bulk volume.

---

## 10. Legal and licensing

- **The Stack v3:** ODC-BY; attribution; per-file licenses apply.
- **OpenCodeInstruct:** CC-BY-4.0; attribution.
- **Magicoder OSS-Instruct:** MIT, but generated by GPT-3.5; OpenAI usage policy applies.
- **react-code-instructions:** MIT; generated apps may imitate brands; do not ship clones.
- **WebSight:** CC-BY-4.0; synthetic, but contains real images; check image provenance.
- **Design2Code:** ODC-BY; evaluation only; real webpages and brands.
- **Webdev-Bench:** MIT; evaluation only.
- **SWE-bench Multilingual:** MIT; evaluation only.
- **Multi-SWE-bench:** `other`; source repositories have their own licenses.
- **SWE-bench Multimodal:** no dataset license tag; source repositories; evaluation only.
- **shadcn/ui, Tailwind, React, Radix, TanStack, Storybook:** MIT.
- **Playwright:** Apache-2.0.
- **Tailwind Plus / paid component libraries:** do not train on them.
- **DeepSeek/OpenCode terms:** review the current terms; record the teacher model and version.

Do not train on proprietary screenshots, customer data, or a real brand's site without permission. Replicating a trademarked interface can raise copyright/trade-dress issues even if the code is generated.

This is not legal advice. For a commercial release, have counsel review the dataset and model licenses.

---

## 11. Failure modes

| Failure | Cause | Fix |
|---|---|---|
| Confident but unbuildable UI | no typecheck/build loop | execute every candidate; keep only passes |
| “Looks fine” but broken mobile | no viewport tests | Playwright at 320/768/1440; overflow assertion |
| Tailwind classes do nothing | v3 directives in a v4 project, dynamic classes | version tags; validator; build + screenshot |
| Generic AI look | no design spec/reference | design.md, tokens, rubric, human review |
| Accessibility regressions | axe not run | axe gate + keyboard tests |
| Tests pass but assert nothing | weak tests | mutation/coverage; review assertions |
| Visual diff flaky | animations, fonts, dynamic data | reduced motion, disable animations, seed data, mask |
| Model learns `div` buttons | low-quality seed data | validator + a11y examples |
| Over-refusal | too many negatives | rebalance; add in-scope positives |
| Data leakage | random split | split by repo/commit; 13-gram decontamination |
| Cost blow-up | long pages + retries + judge | Flash for bulk, Pro for hard, cap tokens |
| License problem | scraped/branded seeds | provenance + license audit |

---

## 12. Sources

- DeepSeek pricing — <https://api-docs.deepseek.com/quick_start/pricing>
- DeepSeek JSON mode — <https://api-docs.deepseek.com/guides/json_mode>
- DeepSeek tool calls — <https://api-docs.deepseek.com/guides/tool_calls>
- OpenCode — <https://opencode.ai/docs>
- React 19.3 — <https://react.dev/blog/2026/09/09/react-19-3>
- Tailwind 4.3 — <https://tailwindcss.com/blog/tailwindcss-v4-3>
- Tailwind upgrade guide — <https://tailwindcss.com/docs/upgrade-guide>
- Playwright screenshots — <https://playwright.dev/docs/test-snapshots>
- Playwright accessibility — <https://playwright.dev/docs/accessibility-testing>
- axe-core — <https://github.com/dequelabs/axe-core>
- Webdev-Bench — <https://huggingface.co/datasets/Roy-DOU/Webdev-Bench>
- Design2Code — <https://huggingface.co/datasets/SALT-NLP/Design2Code>
- WebSight — <https://huggingface.co/datasets/HuggingFaceM4/WebSight>
- SWE-bench Multilingual — <https://huggingface.co/datasets/SWE-bench/SWE-bench_Multilingual>
- Multi-SWE-bench — <https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench>
- SWE-bench Multimodal — <https://huggingface.co/datasets/SWE-bench/SWE-bench_Multimodal>
- shadcn/ui — <https://ui.shadcn.com/>
- WCAG 2.2 — <https://www.w3.org/TR/WCAG22/>
- [07 — Synthetic data](07-synthetic-data.md)
- [24 — React fine-tuning](24-react-finetuning.md)
