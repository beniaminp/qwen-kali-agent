Build or change one page/route. Return JSON only.

Rules:
- Compose existing primitives and feature components; do not duplicate them.
- Include loading, empty, error and content states.
- Responsive at 320, 768 and 1440; no horizontal overflow.
- Use semantic tokens; no raw hex.
- Use TanStack Query for server state; URL search params for filters/pagination.
- Use React 19 Actions for forms; `useActionState`/`useOptimistic` where useful.
- Accessible landmarks, headings, labels and focus order.
- Add Playwright coverage for the critical flow and an axe check.
- No pixel-perfect clone of a real brand.

Page:
{page}

Data and states:
{data}

Design brief:
{design}

Existing components:
{components}

Return:
{
  "patch": "unified diff",
  "files": ["..."],
  "states": ["loading", "empty", "error", "content"],
  "responsive": ["320", "768", "1440"],
  "a11y": ["..."],
  "e2e": "the Playwright test",
  "commands": ["npm run typecheck", "npm run build", "npx playwright test"]
}
