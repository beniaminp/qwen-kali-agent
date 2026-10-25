#!/usr/bin/env python3
"""Gemma 4-specific checks for a canonical chat JSONL file.

The important difference from Qwen: Gemma 4's template requires
`tool_calls[].function.arguments` to be a JSON object (mapping), not a string.
It also emits `<|tool_call>call:name{...}<tool_call|>` and resolves tool results
by `tool_call_id`, so ids and ordering matter.

This is a static linter, not a renderer. Always render one row with the Gemma
tokenizer before a long training run.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
QWEN_CALL = re.compile(r"<tool_call>\s*\{")
SECRETS = [
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"\bsk-[A-Za-z0-9]{20,}\b",
    r"\bghp_[A-Za-z0-9]{30,}\b",
    r"\bAIza[0-9A-Za-z_-]{30,}\b",
]


def load_tools(path: str | None) -> set[str] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "tools" in data:
        data = data["tools"]
    return {t.get("function", t)["name"] for t in data}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tools", help="Optional tool schema file for name membership")
    ap.add_argument("--strict", action="store_true", help="Fail on warnings too")
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    allowed = load_tools(args.tools)
    errors: list[str] = []
    warnings: list[str] = []
    rows = 0

    for lineno, line in enumerate(Path(args.path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        if args.max_rows and rows >= args.max_rows:
            break
        rows += 1
        where = f"{args.path}:{lineno}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{where}: invalid JSON: {exc}")
            continue

        msgs = row.get("messages")
        if not isinstance(msgs, list) or not msgs:
            errors.append(f"{where}: messages must be a non-empty list")
            continue
        if msgs[0].get("role") not in {"system", "developer"}:
            warnings.append(f"{where}: first message is not system/developer")
        if msgs[0].get("role") in {"system", "developer"} and not isinstance(msgs[0].get("content"), str):
            errors.append(f"{where}: system content must be a string")

        meta = row.get("meta") or {}
        if not meta.get("license"):
            warnings.append(f"{where}: meta.license is missing")
        if not meta.get("track"):
            warnings.append(f"{where}: meta.track is missing")

        row_tools = {t.get("function", t).get("name") for t in (row.get("tools") or [])}
        if row_tools:
            for name in row_tools:
                if not name or not NAME_RE.match(name):
                    errors.append(f"{where}: invalid tool name {name!r}")
            if allowed is not None:
                unknown = row_tools - allowed
                if unknown:
                    errors.append(f"{where}: tools not in schema file: {sorted(unknown)}")

        pending: dict[str, str] = {}
        saw_tool_result = False

        for i, m in enumerate(msgs):
            role = m.get("role")
            content = m.get("content")
            text = content if isinstance(content, str) else ""

            if role == "tool":
                saw_tool_result = True
                cid = m.get("tool_call_id")
                if not cid:
                    errors.append(f"{where}: message {i} tool role has no tool_call_id")
                elif cid not in pending:
                    errors.append(f"{where}: message {i} tool_call_id {cid!r} has no preceding tool call")
                else:
                    pending.pop(cid)
                if not isinstance(content, str):
                    errors.append(f"{where}: message {i} tool content must be a string")
                continue

            if role == "assistant":
                rc = m.get("reasoning_content")
                if rc is not None and not isinstance(rc, str):
                    errors.append(f"{where}: message {i} reasoning_content must be a string")
                # Gemma expects previous thinking to be dropped, except tool-call turns.
                if rc and i < len(msgs) - 1 and not (m.get("tool_calls") or (i + 1 < len(msgs) and msgs[i + 1].get("role") == "tool")):
                    warnings.append(f"{where}: message {i} keeps reasoning_content in history; Gemma expects final text only")

                calls = m.get("tool_calls") or []
                if calls and not row_tools:
                    errors.append(f"{where}: message {i} has tool_calls but the row declares no tools")
                for call in calls:
                    fn = call.get("function", call)
                    name = fn.get("name")
                    if not name or not NAME_RE.match(name):
                        errors.append(f"{where}: message {i} invalid tool name {name!r}")
                    if row_tools and name not in row_tools:
                        errors.append(f"{where}: message {i} calls undeclared tool {name!r}")
                    if allowed is not None and name not in allowed:
                        errors.append(f"{where}: message {i} calls tool not in schema file {name!r}")
                    call_id = call.get("id")
                    if not call_id:
                        warnings.append(f"{where}: message {i} tool call {name!r} has no id; tool results cannot bind")
                    else:
                        pending[call_id] = name
                    raw = fn.get("arguments")
                    if raw is not None and not isinstance(raw, dict):
                        errors.append(f"{where}: message {i} tool call {name!r} arguments must be an object (mapping), not {type(raw).__name__}")
                    elif isinstance(raw, dict) and not all(isinstance(k, str) for k in raw):
                        errors.append(f"{where}: message {i} tool call {name!r} arguments keys must be strings")

                if QWEN_CALL.search(text):
                    warnings.append(f"{where}: message {i} contains Qwen-style <tool_call> JSON; let the Gemma template render tool calls")
                for pat in SECRETS:
                    if re.search(pat, text):
                        errors.append(f"{where}: message {i} contains a probable secret")

            elif role == "user":
                if not isinstance(content, str):
                    warnings.append(f"{where}: message {i} user content is not a string (multimodal?)")

            elif role not in {"system", "developer"}:
                errors.append(f"{where}: message {i} has unsupported role {role!r}")

        if pending:
            warnings.append(f"{where}: {len(pending)} tool call(s) have no matching tool result")
        if saw_tool_result and not row_tools:
            warnings.append(f"{where}: tool results present but no tools declared")

    print(f"rows: {rows}")
    print(f"errors: {len(errors)}  warnings: {len(warnings)}")
    for e in errors[:80]:
        print("ERROR  ", e)
    for w in warnings[:80]:
        print("WARN   ", w)
    if len(errors) > 80 or len(warnings) > 80:
        print("... (truncated)")
    return 1 if errors or (args.strict and warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
