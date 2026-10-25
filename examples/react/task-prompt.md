Propose {n} diverse React/Tailwind training tasks for a repository-grounded
front-end coding agent. Cover this mix, in roughly equal parts:

- component: primitive with typed props, variants, states, tests
- page: route with layout, loading, empty, error and content states
- data: TanStack Query, mutations, optimistic updates, URL state
- form: React 19 Actions, validation, errors, pending state
- design system: replace raw palette/duplicated classes with tokens + CVA
- accessibility: keyboard, focus, labels, contrast, target size, axe
- testing: Vitest/Testing Library or Playwright, meaningful assertions
- bug fix: hooks, controlled inputs, race conditions, layout shift
- migration: React 18->19, Tailwind 3->4, Router 7->8, TS 6->7
- review: find real UI/architecture/a11y issues without changing code
- negative: unsafe, under-specified or out-of-scope request

Each task must include a concrete acceptance test, a test command and the
viewports to check.

Return:
{
  "tasks": [
    {
      "id": "short-stable-id",
      "category": "component|page|data|form|design-system|a11y|test|bug|migration|review|negative",
      "difficulty": "easy|medium|hard",
      "task": "what the agent must do",
      "repo_context": ["files or modules to read"],
      "constraints": ["React 19.3", "Tailwind 4.3", "semantic tokens", "..."],
      "acceptance": ["what must pass", "which checks"],
      "viewports": ["320", "768", "1440"],
      "test_command": "npm run typecheck && npm run lint && npm test -- --run && npm run build",
      "why": "what capability this trains"
    }
  ]
}
