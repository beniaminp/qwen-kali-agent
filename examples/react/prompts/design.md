Turn a visual reference into a written design spec, or critique a rendered page
against a spec. Return JSON only.

Write the spec as if a text-only engineer must implement it without seeing the
image. Be concrete: layout, hierarchy, spacing, type, color, components, states,
responsive behaviour, accessibility and copy.

Rules:
- Do not identify or clone a real brand. Describe the design language.
- Use semantic tokens, not raw hex, except when recording an observed color.
- No lorem ipsum; write real short copy.
- Call out uncertainty; do not invent details you cannot see.
- For a critique, separate bugs from taste.

Mode:
{mode}

Reference or spec:
{reference}

Product context:
{context}

Return:
{
  "design_md": "# Design spec\n...",
  "layout": ["..."],
  "tokens": {"canvas": "...", "surface": "...", "ink": "...", "brand": "..."},
  "components": ["..."],
  "states": ["loading", "empty", "error", "success"],
  "responsive": {"320": "...", "768": "...", "1440": "..."},
  "a11y": ["..."],
  "acceptance": ["..."],
  "findings": [{"severity": "...", "issue": "...", "fix": "..."}]
}
