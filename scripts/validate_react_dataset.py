#!/usr/bin/env python3
"""React/Tailwind-specific checks for a canonical chat JSONL file.

Complements validate_dataset.py. It scans code blocks for legacy React,
Tailwind 3 leftovers, unsafe HTML, accessibility smells and bad tool shapes.
It is a linter, not a compiler or browser.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TAILWIND_V3 = re.compile(r"@tailwind\s+(?:base|components|utilities)\b")
TAILWIND_V3_CONFIG = re.compile(r"tailwind\.config\.(?:js|ts|cjs|mjs)")
V4_REMOVED = [
    (r"\bbg-opacity-\d+\b", "bg-opacity-* was removed in v4; use bg-black/50"),
    (r"\btext-opacity-\d+\b", "text-opacity-* was removed in v4; use text-black/50"),
    (r"\bborder-opacity-\d+\b", "border-opacity-* was removed in v4; use border-black/50"),
    (r"\bdivide-opacity-\d+\b", "divide-opacity-* was removed in v4; use divide-black/50"),
    (r"\bflex-shrink-\S+", "flex-shrink-* was renamed to shrink-*"),
    (r"\bflex-grow-\S+", "flex-grow-* was renamed to grow-*"),
    (r"\boverflow-ellipsis\b", "overflow-ellipsis was renamed to text-ellipsis"),
    (r"\bdecoration-slice\b", "use box-decoration-slice"),
    (r"\bdecoration-clone\b", "use box-decoration-clone"),
]
LEGACY_REACT = [
    (r"ReactDOM\.render\s*\(", "ReactDOM.render is removed in React 19; use createRoot"),
    (r"\buseFormState\s*\(", "useFormState is the old name; use useActionState"),
    (r"\bReact\.FC\b", "React.FC is unnecessary; type props explicitly"),
    (r"\bextends\s+React\.Component\b", "class components are not the target"),
    (r"\bpropTypes\b", "propTypes are legacy; use TypeScript props"),
    (r"\bdefaultProps\b", "defaultProps on function components are legacy; use defaults"),
    (r"\bforwardRef\s*\(", "prefer ref as a prop in React 19"),
]
DYNAMIC_CLASS = re.compile(r"\b(?:bg|text|border|ring|from|to|via)-\$\{")
RAW_HEX_CLASS = re.compile(r"\b(?:bg|text|border|ring|from|to|via)-\[#")
SECRETS = [
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"\bsk-[A-Za-z0-9]{20,}\b",
    r"\bghp_[A-Za-z0-9]{30,}\b",
    r"\bAIza[0-9A-Za-z_-]{30,}\b",
    r"(?i)\b(?:password|passwd|secret|token|api[_-]?key)\s*[=:]\s*['\"]?[^\s'\"<]{8,}",
]
CODE_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)


def code_of(text: str) -> str:
    blocks = CODE_FENCE.findall(text)
    return "\n".join(blocks) if blocks else text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tools", help="Optional tool schema file for name membership")
    ap.add_argument("--require-dict-arguments", action="store_true",
                    help="Require tool arguments to be objects (Qwen2.5-Coder template)")
    ap.add_argument("--strict", action="store_true", help="Fail on warnings too")
    args = ap.parse_args()

    allowed = None
    if args.tools:
        data = json.loads(Path(args.tools).read_text())
        if isinstance(data, dict) and "tools" in data:
            data = data["tools"]
        allowed = {t.get("function", t)["name"] for t in data}

    errors: list[str] = []
    warnings: list[str] = []
    rows = 0

    for lineno, line in enumerate(Path(args.path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        rows += 1
        where = f"{args.path}:{lineno}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{where}: invalid JSON: {exc}")
            continue

        meta = row.get("meta") or {}
        category = meta.get("category")
        if not meta.get("license"):
            warnings.append(f"{where}: meta.license is missing")
        if not meta.get("test_command") and meta.get("category") not in {"review", "design", "refusal"}:
            warnings.append(f"{where}: meta.test_command is missing")
        versions = meta.get("versions") or {}
        if versions.get("react") and not str(versions["react"]).startswith("19"):
            warnings.append(f"{where}: React {versions['react']}, target is 19.x")
        if versions.get("tailwind") and not str(versions["tailwind"]).startswith("4"):
            warnings.append(f"{where}: Tailwind {versions['tailwind']}, target is 4.x")

        row_tools = {t.get("function", t).get("name") for t in (row.get("tools") or [])}

        for i, m in enumerate(row.get("messages", [])):
            content = m.get("content")
            if isinstance(content, str):
                for pat in SECRETS:
                    if re.search(pat, content):
                        errors.append(f"{where}: message {i} contains a probable secret")
                code = code_of(content)
                if TAILWIND_V3.search(code) and category != "migration":
                    errors.append(f"{where}: message {i} uses v3 @tailwind directives; use @import \"tailwindcss\"")
                if TAILWIND_V3_CONFIG.search(code):
                    warnings.append(f"{where}: message {i} references tailwind.config; v4 is CSS-first")
                for pat, msg in V4_REMOVED:
                    if re.search(pat, code):
                        warnings.append(f"{where}: message {i} {msg}")
                if re.search(r"\boutline-none\b", code):
                    warnings.append(f"{where}: message {i} uses outline-none; use outline-hidden or a real focus ring")
                for pat, msg in LEGACY_REACT:
                    if re.search(pat, code):
                        warnings.append(f"{where}: message {i} {msg}")
                if DYNAMIC_CLASS.search(code):
                    errors.append(f"{where}: message {i} constructs Tailwind classes dynamically")
                if RAW_HEX_CLASS.search(code):
                    warnings.append(f"{where}: message {i} uses arbitrary hex; prefer a design token")
                if re.search(r"style=\{\{", code):
                    warnings.append(f"{where}: message {i} uses inline styles; prefer tokens/utilities")
                if re.search(r"dangerouslySetInnerHTML", code) and not re.search(r"sanitize|DOMPurify", code, re.I):
                    errors.append(f"{where}: message {i} uses unsanitized dangerouslySetInnerHTML")
                if re.search(r"<img\b(?![^>]*\balt=)", code):
                    warnings.append(f"{where}: message {i} has an img without alt")
                if re.search(r"<button\b(?![^>]*\btype=)", code):
                    warnings.append(f"{where}: message {i} has a button without type")
                if re.search(r"<div[^>]*\bonClick=", code) and not re.search(r"role=|tabIndex=", code):
                    errors.append(f"{where}: message {i} has a clickable div without role/tabIndex")
                if re.search(r"target=[\"']_blank[\"']", code) and not re.search(r"rel=[\"'][^\"']*(?:noreferrer|noopener)", code):
                    warnings.append(f"{where}: message {i} has target=_blank without rel noopener/noreferrer")
                if re.search(r"key=\{index\}", code):
                    warnings.append(f"{where}: message {i} uses key={index}; prefer a stable id")
                if re.search(r"\bany\b", code):
                    warnings.append(f"{where}: message {i} uses any")
                if re.search(r"@ts-ignore|@ts-expect-error", code):
                    warnings.append(f"{where}: message {i} suppresses TypeScript errors")
                if re.search(r"\bconsole\.(?:log|debug|warn)\b", code):
                    warnings.append(f"{where}: message {i} uses console logging")
                if re.search(r"\bTODO\b|\bFIXME\b", code):
                    warnings.append(f"{where}: message {i} contains TODO/FIXME")
                if content.count("```") % 2:
                    warnings.append(f"{where}: message {i} has unbalanced code fences")

            for call in m.get("tool_calls") or []:
                fn = call.get("function", call)
                name = fn.get("name")
                if allowed is not None and name not in allowed:
                    errors.append(f"{where}: tool {name!r} is not in the tools file")
                if row_tools and name not in row_tools:
                    errors.append(f"{where}: tool {name!r} is not declared in this row")
                raw = fn.get("arguments")
                if args.require_dict_arguments and not isinstance(raw, dict):
                    errors.append(f"{where}: tool {name!r} arguments must be an object for Qwen2.5-Coder")

    print(f"rows: {rows}")
    print(f"errors: {len(errors)}  warnings: {len(warnings)}")
    for e in errors[:60]:
        print("ERROR  ", e)
    for w in warnings[:60]:
        print("WARN   ", w)
    if len(errors) > 60 or len(warnings) > 60:
        print("... (truncated)")
    return 1 if errors or (args.strict and warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
