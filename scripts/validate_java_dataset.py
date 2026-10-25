#!/usr/bin/env python3
"""Java/Quarkus-specific checks for a canonical chat JSONL file.

Complements validate_dataset.py: it looks for legacy Jakarta/Quarkus idioms,
missing test commands, unsafe output and bad tool-argument shapes.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LEGACY_JAVAX = re.compile(
    r"\bjavax\.(persistence|ws\.rs|inject|enterprise|transaction|validation|annotation|servlet)\b"
)
SECRETS = [
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"\bsk-[A-Za-z0-9]{20,}\b",
    r"\bghp_[A-Za-z0-9]{30,}\b",
    r"(?i)\b(password|passwd|secret|token)\s*[=:]\s*['\"]?[^\s'\"<]{8,}",
]
QUARKUS_VERSION = re.compile(r"quarkus\.platform\.version\s*>\s*([^<\s]+)")
JAVA_RELEASE = re.compile(r"<(?:maven\.compiler\.release|java\.version)>\s*([^<\s]+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tools", help="Optional tool schema file for name membership")
    ap.add_argument("--target-java", default="25")
    ap.add_argument("--target-quarkus", default="3.40")
    ap.add_argument("--require-dict-arguments", action="store_true",
                    help="Require tool arguments to be objects (Qwen2.5-Coder template)")
    ap.add_argument("--strict", action="store_true")
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

        if not row.get("meta", {}).get("license"):
            warnings.append(f"{where}: meta.license is missing")
        if not row.get("meta", {}).get("test_command") and row.get("meta", {}).get("category") not in {"review", "refusal"}:
            warnings.append(f"{where}: meta.test_command is missing")

        row_tools = set()
        for t in row.get("tools") or []:
            row_tools.add(t.get("function", t).get("name"))

        for i, m in enumerate(row.get("messages", [])):
            content = m.get("content")
            if isinstance(content, str):
                if LEGACY_JAVAX.search(content):
                    errors.append(f"{where}: message {i} uses legacy javax.* imports; use jakarta.*")
                if re.search(r"\borg\.springframework\b|@Autowired\b", content):
                    warnings.append(f"{where}: message {i} contains Spring idioms")
                if re.search(r"quarkus-resteasy-reactive|quarkus-resteasy\b", content):
                    warnings.append(f"{where}: message {i} uses legacy RESTEasy coordinates; prefer quarkus-rest")
                for v in QUARKUS_VERSION.findall(content):
                    if not v.startswith(args.target_quarkus):
                        warnings.append(f"{where}: message {i} targets Quarkus {v}, not {args.target_quarkus}")
                for v in JAVA_RELEASE.findall(content):
                    if v not in {args.target_java, "${java.version}"}:
                        warnings.append(f"{where}: message {i} targets Java {v}, not {args.target_java}")
                if re.search(r"\bSystem\.out\.print|\bprintStackTrace\b", content):
                    warnings.append(f"{where}: message {i} uses console printing; prefer structured logging")
                if re.search(r"\bTODO\b|\bFIXME\b", content):
                    warnings.append(f"{where}: message {i} contains TODO/FIXME")
                if re.search(r"\blombok\b", content, re.I):
                    warnings.append(f"{where}: message {i} references Lombok; prefer records/plain Java")
                if content.count("```") % 2:
                    warnings.append(f"{where}: message {i} has unbalanced code fences")
                for pat in SECRETS:
                    if re.search(pat, content):
                        errors.append(f"{where}: message {i} contains a probable secret")

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
    for e in errors[:50]:
        print("ERROR  ", e)
    for w in warnings[:50]:
        print("WARN   ", w)
    if len(errors) > 50 or len(warnings) > 50:
        print("... (truncated)")
    return 1 if errors or (args.strict and warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
