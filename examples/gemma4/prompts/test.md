Write front-end tests. Return JSON only.

Rules:
- Vitest + Testing Library for components; Playwright for flows/responsive.
- Query by role/label/text, not CSS classes or test ids unless nothing else works.
- Assert behaviour, not implementation details.
- Include one happy path and one failure/edge case.
- No arbitrary sleeps; use `findBy*`, `waitFor`, web-first assertions.
- Do not weaken assertions to make a test pass.
- For accessible components, include an axe assertion.
- Never invent production APIs; read the source first.

Source:
{source}

Existing tests:
{existing_tests}

Requirements:
{requirements}

Return:
{
  "unit": "Vitest/RTL test file",
  "e2e": "Playwright test file, or empty",
  "cases": ["happy path: ...", "edge case: ..."],
  "a11y": ["axe tags and expected result"],
  "commands": ["npm test -- --run", "npx playwright test"]
}
