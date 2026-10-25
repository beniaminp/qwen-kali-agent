Build or change one React component. Return JSON only.

Rules:
- Typed props; no `any`; no `React.FC`.
- Use design tokens and `cn`; variants with CVA if there is more than one variant.
- Include hover, focus-visible, active, disabled and loading states where relevant.
- Use semantic HTML; Radix primitives for complex widgets.
- Add a Vitest + Testing Library test that asserts behaviour by role.
- Add a Storybook story if the repository has Storybook.
- No raw hex, no dynamic class construction.
- Do not add dependencies without asking.

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
  "code": "the complete component file",
  "styles": "Tailwind classes and any CSS",
  "test": "the test file",
  "story": "optional story file",
  "states": ["default", "hover", "focus", "disabled", "loading", "error"],
  "a11y": ["what makes it accessible"],
  "commands": ["npm run typecheck", "npm test -- --run"]
}
