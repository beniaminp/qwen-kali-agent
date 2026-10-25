#!/usr/bin/env python3
"""Normalize a canonical chat+tools JSONL and split it deterministically.

Output: train.jsonl, valid.jsonl, test.jsonl, manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

TOOL_NAME = re.compile(r"[^A-Za-z0-9_-]+")


def clean_name(name: str) -> str:
    return TOOL_NAME.sub("_", str(name)).strip("_") or "tool"


def norm_row(row: dict, arguments_format: str = "string") -> dict | None:
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None

    tools = row.get("tools") or []
    fixed_tools = []
    seen_tools = set()
    for t in tools:
        fn = dict(t.get("function", t))
        fn["name"] = clean_name(fn.get("name", "tool"))
        if fn["name"] in seen_tools:
            continue
        seen_tools.add(fn["name"])
        fixed_tools.append({"type": "function", "function": fn})

    fixed_msgs = []
    for m in msgs:
        m = dict(m)
        if not isinstance(m.get("content"), (str, type(None))):
            m["content"] = json.dumps(m["content"], ensure_ascii=False)
        calls = m.get("tool_calls")
        if calls:
            new_calls = []
            for c in calls:
                c = dict(c)
                fn = dict(c.get("function", c))
                fn["name"] = clean_name(fn.get("name", "tool"))
                raw = fn.get("arguments", {})
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except json.JSONDecodeError:
                        return None
                if not isinstance(raw, dict):
                    return None
                if arguments_format == "dict":
                    fn["arguments"] = raw
                else:
                    fn["arguments"] = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
                c["function"] = fn
                c.setdefault("type", "function")
                new_calls.append(c)
            m["tool_calls"] = new_calls
        fixed_msgs.append(m)

    out = dict(row)
    out["messages"] = fixed_msgs
    if fixed_tools:
        out["tools"] = fixed_tools
    else:
        out.pop("tools", None)
    return out


def digest(row: dict, seed: int) -> str:
    blob = json.dumps(row["messages"], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(f"{seed}:{blob}".encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--split", nargs=3, type=float, default=[0.9, 0.05, 0.05],
                    metavar=("TRAIN", "VALID", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arguments-format", choices=["string", "dict"], default="string",
                    help="string for Qwen3/OpenAI JSONL; dict for Qwen2.5-Coder's template")
    args = ap.parse_args()

    if abs(sum(args.split) - 1.0) > 1e-6:
        print("split fractions must sum to 1.0")
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, seen = [], set()
    dropped = Counter()
    licenses = Counter()
    sources = Counter()

    with Path(args.input).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped["invalid_json"] += 1
                continue
            row = norm_row(row, args.arguments_format)
            if not row:
                dropped["normalize_failed"] += 1
                continue
            d = digest(row, args.seed)
            if d in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(d)
            rows.append((d, row))
            if row.get("license"):
                licenses[row["license"]] += 1
            if row.get("source"):
                sources[row["source"]] += 1

    if not rows:
        print("no valid rows")
        return 1

    rows.sort(key=lambda x: x[0])
    n = len(rows)
    n_train = int(n * args.split[0])
    n_valid = int(n * args.split[1])
    splits = {
        "train": rows[:n_train],
        "valid": rows[n_train:n_train + n_valid],
        "test": rows[n_train + n_valid:],
    }

    manifest = {
        "version": datetime.now(timezone.utc).strftime("%Y-%m-%d.v1"),
        "created": datetime.now(timezone.utc).isoformat(),
        "input": str(Path(args.input)),
        "seed": args.seed,
        "arguments_format": args.arguments_format,
        "split": args.split,
        "counts": {k: len(v) for k, v in splits.items()},
        "dropped": dict(dropped),
        "licenses": dict(licenses),
        "sources": dict(sources.most_common(50)),
        "files": {},
    }

    for name, part in splits.items():
        p = out / f"{name}.jsonl"
        with p.open("w") as fh:
            for _, row in part:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["files"][name] = {
            "rows": len(part),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: len(v) for k, v in splits.items()}, indent=2))
    print("dropped:", dict(dropped))
    print("wrote:", out / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
