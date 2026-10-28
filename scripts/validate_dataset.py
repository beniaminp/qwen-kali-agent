#!/usr/bin/env python3
"""Validate a canonical chat+tools JSONL file.

Hard errors exit 1. Warnings exit 1 only with --strict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROLES = {"system", "user", "assistant", "tool"}
DESTRUCTIVE = [
    r"\brm\s+-rf\s+/(?:\s|$)",
    r"\bmkfs(\.\w+)?\b",
    r"\bdd\s+.*of=/dev/",
    r":\(\)\s*\{.*\};:",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bcurl\b[^\n]*\|\s*(?:ba|z|da)?sh\b",
    r"\bwget\b[^\n]*\|\s*(?:ba|z|da)?sh\b",
]
SECRETS = [
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"\bsk-[A-Za-z0-9]{20,}\b",
    r"\bghp_[A-Za-z0-9]{30,}\b",
    r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
]
TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def load_tools(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "tools" in data:
        data = data["tools"]
    out = {}
    for item in data:
        fn = item.get("function", item)
        if fn.get("name"):
            out[fn["name"]] = fn
    return out


def norm_messages(messages) -> str:
    return json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tools", help="JSON file with tool schemas (list or {tools: [...]})")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    ap.add_argument("--max-chars", type=int, default=8000, help="Warn above this message size")
    args = ap.parse_args()

    tools = load_tools(args.tools)
    allowed = set(tools)
    errors: list[str] = []
    warnings: list[str] = []
    seen: dict[str, int] = {}
    n = 0

    path = Path(args.path)
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            n += 1
            where = f"{path}:{lineno}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{where}: invalid JSON: {exc}")
                continue

            msgs = row.get("messages")
            if not isinstance(msgs, list) or not msgs:
                errors.append(f"{where}: missing/empty messages")
                continue

            digest = hashlib.sha256(norm_messages(msgs).encode()).hexdigest()
            if digest in seen:
                warnings.append(f"{where}: duplicate of line {seen[digest]}")
            else:
                seen[digest] = lineno

            row_tools = row.get("tools") or []
            row_names = set()
            for t in row_tools:
                fn = t.get("function", t)
                name = fn.get("name", "")
                row_names.add(name)
                if not TOOL_NAME.match(name):
                    errors.append(f"{where}: invalid tool name {name!r}")
                params = fn.get("parameters", {})
                if params.get("type") != "object":
                    warnings.append(f"{where}: tool {name!r} parameters.type != object")

            if msgs[0].get("role") != "system":
                warnings.append(f"{where}: first message is not system")

            pending: set[str] = set()
            for i, m in enumerate(msgs):
                role = m.get("role")
                if role not in ROLES:
                    errors.append(f"{where}: message {i} has invalid role {role!r}")
                    continue
                content = m.get("content")
                if content is not None and not isinstance(content, str):
                    errors.append(f"{where}: message {i} content is not a string/null")
                if role == "assistant" and not content and not m.get("tool_calls"):
                    errors.append(f"{where}: empty assistant message at {i}")
                if role == "assistant" and isinstance(content, str) and "<think>" in content:
                    errors.append(f"{where}: assistant content contains <think> (use reasoning_content)")

                calls = m.get("tool_calls") or []
                if role == "assistant" and calls:
                    for c in calls:
                        fn = c.get("function", c)
                        name = fn.get("name")
                        if name not in row_names:
                            errors.append(f"{where}: call to unknown tool {name!r}")
                        if name not in allowed and allowed:
                            errors.append(f"{where}: tool {name!r} not in --tools file")
                        raw = fn.get("arguments", {})
                        if isinstance(raw, str):
                            try:
                                raw = json.loads(raw)
                            except json.JSONDecodeError:
                                errors.append(f"{where}: arguments for {name!r} are not valid JSON")
                                raw = {}
                        if not isinstance(raw, dict):
                            errors.append(f"{where}: arguments for {name!r} are not an object")
                            raw = {}
                        schema = tools.get(name, {}).get("parameters", {})
                        for req in schema.get("required", []):
                            if req not in raw:
                                warnings.append(f"{where}: {name!r} missing required arg {req!r}")
                        if "target" in raw and isinstance(raw["target"], str):
                            if re.match(r"^(?:8\.8\.8\.8|1\.1\.1\.1)$", raw["target"]):
                                warnings.append(f"{where}: public target in training row: {raw['target']}")
                        cid = c.get("id")
                        if cid:
                            pending.add(cid)

                if role == "tool":
                    content = content or ""
                    cid = m.get("tool_call_id")
                    if not pending:
                        errors.append(f"{where}: tool message at {i} has no preceding assistant tool call")
                    elif cid and cid not in pending:
                        warnings.append(f"{where}: tool_call_id {cid!r} not in pending calls")
                    if not content.strip():
                        errors.append(f"{where}: empty tool response at {i}")
                    if cid:
                        pending.discard(cid)

                if isinstance(content, str) and len(content) > args.max_chars:
                    warnings.append(f"{where}: message {i} is {len(content)} chars")

            blob = line
            for pat in DESTRUCTIVE:
                if re.search(pat, blob, re.I):
                    warnings.append(f"{where}: destructive pattern {pat!r}")
            for pat in SECRETS:
                if re.search(pat, blob):
                    warnings.append(f"{where}: possible secret matching {pat!r}")

    print(f"rows: {n}")
    print(f"errors: {len(errors)}  warnings: {len(warnings)}")
    for e in errors[:50]:
        print("ERROR  ", e)
    for w in warnings[:50]:
        print("WARN   ", w)
    if len(errors) > 50 or len(warnings) > 50:
        print("... (truncated)")

    if errors or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
