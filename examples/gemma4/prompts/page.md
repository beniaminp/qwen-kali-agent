Build or change one React page/route. Return JSON only.

Target: React 19.3, Vite 8, Tailwind 4.3, React Router 8.5 data mode, TanStack Query 5.

Rules:
- Compose existing primitives; do not duplicate them.
- Include loading, empty, error and content states.
- Responsive at 320, 768 and 1440; no horizontal overflow.
- Semantic tokens, accessible landmarks, one H1, labels and focus order.
- TanStack Query for server state; URL search params for filters/pagination.
- React 19 Actions/`useActionState`/`useOptimistic` for forms.
- Playwright test for the critical flow + axe check.
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
  "e2e": "Playwright test",
  "a11y": ["..."],
  "commands": ["npm run typecheck", "npm run build", "npx playwright test"]
}
