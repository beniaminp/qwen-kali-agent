Migrate the project. Return JSON only.

Targets:
- React 19.3
- TypeScript 7-compatible source/tsconfig
- Vite 8
- Tailwind CSS 4.3
- React Router 8.5 data mode
- Node 24 LTS

Rules:
- React 18 → 19: `createRoot`, `ref` as prop, `useActionState`; remove
  `ReactDOM.render`, `useFormState`, `propTypes`, `defaultProps` on functions.
- Tailwind 3 → 4: `@import "tailwindcss"`, `@theme`, Vite plugin, renamed utilities
  (`shadow-sm` → `shadow-xs`, `rounded` → `rounded-sm`, `outline-none` →
  `outline-hidden`, `bg-opacity-*` → `/opacity`).
- React Router 7 → 8: `react-router-dom` is removed; import core from
  `react-router`, browser utilities from `react-router/dom`.
- TypeScript 6 → 7: remove `baseUrl`, `target: es5`, `moduleResolution: node`,
  `module: amd/umd/systemjs/none`; set `strict`, explicit `types`.
- Do not jump to Next.js or Server Components.
- Keep public UI behaviour stable.
- Run all checks; report the real result.
- Do not invent migration steps.

Current versions:
{versions}

Task:
{task}

Relevant files:
{files}

Return:
{
  "breaking_changes": ["..."],
  "patch": "unified diff",
  "commands": ["npm run typecheck", "npm run lint", "npm test -- --run", "npm run build", "npx playwright test"],
  "manual_steps": ["..."],
  "risks": ["..."]
}
