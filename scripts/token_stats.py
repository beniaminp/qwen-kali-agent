#!/usr/bin/env python3
"""Token-length distribution for a chat+tools JSONL file.

Uses the tokenizer's chat template, including the tools array, so the numbers
match what mlx-lm will see.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def pct(values: list[int], p: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(math.ceil(p / 100 * len(values))) - 1))
    return values[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        print("transformers is required:", exc)
        return 2

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    lengths: list[int] = []
    failures = 0

    with Path(args.path).open() as fh:
        for i, line in enumerate(fh):
            if args.limit and i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                ids = tok.apply_chat_template(
                    row["messages"],
                    tools=row.get("tools"),
                    tokenize=True,
                    return_dict=False,
                )
                lengths.append(len(ids))
            except Exception as exc:
                failures += 1
                if failures <= 5:
                    print(f"row {i}: {exc}")
    if not lengths:
        print("no rows tokenized")
        return 1

    print(f"rows: {len(lengths)}  failures: {failures}")
    print(f"min: {min(lengths)}  max: {max(lengths)}")
    for p in (50, 75, 90, 95, 99, 99.9):
        print(f"p{p:>4}: {pct(lengths, p)}")
    for limit in (1024, 1536, 2048, 3072, 4096, 6144, 8192):
        n = sum(1 for x in lengths if x > limit)
        print(f"> {limit:5}: {n:6} rows ({100*n/len(lengths):.1f}%)")

    recommended = max(512, int(math.ceil(pct(lengths, 99) / 256) * 256))
    print(f"\nrecommended max_seq_length (p99, rounded to 256): {recommended}")
    print("If many rows exceed it, split trajectories per assistant turn before raising the limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
