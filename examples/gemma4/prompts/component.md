Build one React component. Return JSON only.

Target: React 19.3, TypeScript 7-compatible source, Vite 8, Tailwind CSS 4.3.

Rules:
- Typed props; no `any`; no `React.FC`.
- Semantic design tokens; `cn` for conditional classes; CVA for variants.
- Semantic HTML; Radix primitives for dialogs/menus/tabs/tooltips.
- Include hover, focus-visible, disabled and loading states where relevant.
- Add a Vitest + Testing Library test that asserts behaviour by role.
- No dynamic class construction, raw hex, `@tailwind base` or `outline-none` on focus.
- Add a Storybook story only if the repository already uses Storybook.

Component:
{component}

Props and behaviour:
{behaviour}

Design tokens:
{tokens}

Existing primitives:
{primitives}

Return:
{
  "code": "complete component file",
  "test": "test file",
  "story": "optional story",
  "states": ["default", "hover", "focus", "disabled", "loading", "error"],
  "a11y": ["what makes it accessible"],
  "commands": ["npm run typecheck", "npm test -- --run"]
}
