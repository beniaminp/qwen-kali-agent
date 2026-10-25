# React / Tailwind agent rules

## Platform

- React 19.3, React DOM 19.3.
- TypeScript 7-compatible source and tsconfig; no `baseUrl`, no `target: es5`.
- Vite 8, Tailwind CSS 4.3, React Router 8.5 data mode.
- Node 24 LTS, npm.

## Non-negotiable

- Function components only; `ref` as a prop; no `React.FC`.
- Tailwind 4 CSS-first: `@import "tailwindcss"`, `@theme`; never `@tailwind base`.
- Semantic tokens, not raw hex; no dynamic class names.
- `cn` + CVA for variants.
- Accessible markup: labels, alt, button type, focus-visible, keyboard, 24x24 targets.
- Loading, empty, error and content states on every page.
- Responsive at 320/768/1440; no horizontal overflow.
- No `dangerouslySetInnerHTML` without sanitization; no secrets in `VITE_*`.
- No client-only authorization.
- Run `npm run typecheck`, `npm run lint`, `npm test -- --run`, `npm run build`.
- Never claim checks passed without output.
- No new dependencies without asking.

## Architecture

- `components/ui` primitives; `components/layout` shell; `features/*` domain; `routes/*` pages.
- TanStack Query for server state; URL for filters; Actions for forms; local state for UI.
- Composition over configuration; no barrel files in hot paths.

## Output

- Single file: `tsx` block.
- Multi-file: unified diff.
- Review: findings with severity, file/line, principle and minimal fix.
