Refactor the front-end code to use the design system without changing observable
behaviour. Return JSON only.

Rules:
- Replace raw palette classes and inline colors with semantic tokens.
- Extract repeated markup into a primitive or pattern; use CVA for variants.
- Keep the DOM semantics and accessible names stable unless the task says otherwise.
- Do not change spacing, type or color visually unless the task asks.
- Remove duplication; do not add abstraction without a second use.
- No new dependencies.
- Add or update tests; add Playwright screenshots for the before/after.
- Return a unified diff and the verification commands.

Task:
{task}

Current code:
{code}

Design tokens:
{tokens}

Constraints:
{constraints}

Return:
{
  "plan": ["..."],
  "patch": "unified diff",
  "components": ["extracted primitives"],
  "visual_risk": ["what could shift"],
  "commands": ["npm run typecheck", "npm test -- --run", "npx playwright test"],
  "screenshots": ["routes and viewports"]
}
