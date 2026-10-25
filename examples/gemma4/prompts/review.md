Review the code or diff. Do not change code. Return JSON only.

Focus:
- Correctness: React 19 hooks/effects/keys/controlled inputs; Java 25/Quarkus 3.40; Kali scope.
- Architecture: primitives vs features vs pages; prop explosion; duplication; hexagonal boundaries.
- Tailwind 4: semantic tokens, variants, dark mode, dynamic classes, v3 leftovers.
- Accessibility: semantics, labels, focus, keyboard, target size, contrast, reduced motion.
- Responsive: 320/768/1440, overflow, touch targets.
- Tests: behaviour, meaningful assertions, flakiness.
- Security: XSS, secrets, client-only auth, unsafe links, out-of-scope tool calls.

Return:
{
  "findings": [
    {
      "severity": "blocker|major|minor|nit",
      "file": "path",
      "line": 0,
      "principle": "correctness|a11y|design|architecture|test|security|performance",
      "issue": "what is wrong",
      "fix": "smallest concrete fix"
    }
  ],
  "summary": "one paragraph",
  "tests_missing": ["..."],
  "design_notes": ["..."]
}

Do not invent files, lines or APIs. If context is missing, say so.

Code or diff:
{code}
