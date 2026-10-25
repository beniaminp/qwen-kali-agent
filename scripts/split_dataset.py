#!/usr/bin/env python3
"""Deterministic split with exact and optional MinHash near-duplicate removal.

Use this when merging multiple sources that contain paraphrases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def text_of(row: dict) -> str:
    parts = []
    for m in row.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        for call in m.get("tool_calls", []) or []:
            parts.append(json.dumps(call, ensure_ascii=False))
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--split", nargs=3, type=float, default=[0.9, 0.05, 0.05])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--minhash", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    dropped = Counter()

    # exact dedup on normalized messages
    exact, keep = set(), []
    for row in rows:
        blob = json.dumps(row.get("messages"), sort_keys=True, ensure_ascii=False)
        h = hashlib.sha256(blob.encode()).hexdigest()
        if h in exact:
            dropped["exact"] += 1
            continue
        exact.add(h)
        keep.append(row)

    if args.minhash:
        try:
            from datasketch import MinHash, MinHashLSH

            lsh = MinHashLSH(threshold=args.threshold, num_perm=128)
            near = []
            for i, row in enumerate(keep):
                m = MinHash(num_perm=128)
                words = text_of(row).lower().split()
                for j in range(max(0, len(words) - 4)):
                    m.update(" ".join(words[j:j + 5]).encode())
                if lsh.query(m):
                    dropped["near"] += 1
                    continue
                lsh.insert(str(i), m)
                near.append(row)
            keep = near
        except ImportError:
            print("datasketch not installed; skipped MinHash (pip install datasketch)")

    if not keep:
        print("no rows left")
        return 1

    keep.sort(key=lambda r: hashlib.sha256(
        f"{args.seed}:{json.dumps(r.get('messages'), sort_keys=True)}".encode()
    ).hexdigest())
    n = len(keep)
    a, b = int(n * args.split[0]), int((args.split[0] + args.split[1]) * n)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", keep[:a]), ("valid", keep[a:b]), ("test", keep[b:])):
        with (out / f"{name}.jsonl").open("w") as fh:
            for row in part:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(part)}")
    print("dropped:", dict(dropped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
