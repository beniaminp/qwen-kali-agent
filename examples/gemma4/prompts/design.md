Turn a visual reference into a written design spec, or critique a rendered page
against a spec. Return JSON only.

Write the spec so a text-only engineer can implement it without seeing the image.
Be concrete: layout, hierarchy, spacing, type, color, components, states,
responsive behaviour, accessibility and copy.

Rules:
- Do not identify or clone a real brand.
- Use semantic tokens; record observed colors separately.
- No lorem ipsum; write real short copy.
- Call out uncertainty; do not invent details.
- For a critique, separate bugs from taste.
- Gemma 4 can see images, but the implementation may be text-only; do not rely on
  the coder having the screenshot unless the harness passes it.

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
