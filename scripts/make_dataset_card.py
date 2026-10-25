#!/usr/bin/env python3
"""Generate a dataset card skeleton from a manifest.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as fh:
        return sum(1 for line in fh if line.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", default="data/cards/dataset-card.md")
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    m = json.loads(Path(args.manifest).read_text())
    base = Path(args.manifest).parent
    counts = m.get("counts", {})
    if not counts:
        counts = {s: count_lines(base / f"{s}.jsonl") for s in ("train", "valid", "test")}

    sources = m.get("sources", {})
    licenses = m.get("licenses", {})
    lines = [
        "# Dataset Card",
        "",
        f"- **Version:** {m.get('version', 'unknown')}",
        f"- **Created:** {m.get('created', 'unknown')}",
        f"- **Seed:** {m.get('seed', 'unknown')}",
        "",
        "## Intended use",
        "",
        "Supervised fine-tuning of a Qwen-family model for **authorized** lab penetration testing,",
        "network-security checks, tool calling and scope-aware refusal behaviour.",
        "",
        "**Out of scope:** unauthorized testing, public targets, destructive actions, credential theft,",
        "or any use without written authorization.",
        "",
        "## Counts",
        "",
        "| Split | Rows |",
        "|---|---:|",
    ]
    for k, v in counts.items():
        lines.append(f"| {k} | {v} |")

    lines += ["", "## Sources", "", "| Source | Rows |", "|---|---:|"]
    for k, v in sorted(sources.items(), key=lambda x: -x[1])[:50]:
        lines.append(f"| {k} | {v} |")

    lines += ["", "## Licenses", "", "| License | Rows |", "|---|---:|"]
    for k, v in sorted(licenses.items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v} |")
    if not licenses:
        lines.append("| **UNKNOWN — audit before release** | — |")

    lines += [
        "",
        "## Preprocessing",
        "",
        "- normalized to OpenAI-style `messages` + `tools`;",
        "- tool names normalized to `[A-Za-z0-9_-]`;",
        "- `arguments` parsed and re-serialized as JSON objects;",
        "- exact dedup; optional MinHash near-dedup;",
        "- deterministic hash split;",
        "- 13-gram decontamination against held-out benchmarks.",
        "",
        "## Known issues",
        "",
    ]
    issues = m.get("known_issues") or ["None recorded; review before release."]
    lines += [f"- {i}" for i in issues]

    lines += [
        "",
        "## Evaluation",
        "",
        "See the repository evaluation report. Do not release without a held-out golden set,",
        "a benchmark comparison, and an unsafe-action rate of zero.",
        "",
        "## Maintenance",
        "",
        "Owner: _fill in_. Report issues via the repository issue tracker.",
        "",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
