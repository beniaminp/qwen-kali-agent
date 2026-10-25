You implement the front-end task below in the repository. Target React 19.3,
TypeScript 7-compatible source, Vite 8, Tailwind CSS 4.3 and React Router 8.5.

Rules:
- Read the design tokens, existing primitives and tests before editing.
- Function components only; `ref` as a prop; no `React.FC`.
- Use semantic tokens, `cn` and CVA; no raw hex or dynamic class names.
- Include loading, empty, error and content states where relevant.
- Accessible markup: labels, alt, button type, focus-visible, keyboard.
- No new dependencies unless the task requires them; if you add one, explain why.
- Do not weaken or delete tests to make them pass.
- Run the checks; never invent results.
- Return JSON only.

Task:
{task}

Repository context:
{repo_context}

Constraints:
{constraints}

Acceptance:
{acceptance}

Test command:
{test_command}

Return:
{
  "plan": ["short step", "..."],
  "patch": "a unified diff starting with diff --git, or an empty string for a review-only task",
  "files": ["new or changed files"],
  "tests": ["new or changed test files"],
  "commands": ["commands you ran or would run"],
  "screenshots": ["routes and viewports to inspect"],
  "risks": ["what could break"],
  "notes": "anything the reviewer must know"
}
