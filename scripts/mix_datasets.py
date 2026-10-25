#!/usr/bin/env python3
"""Mix track-specific chat JSONL into one deterministic train/valid/test split.

Use this when you want one adapter to cover several domains. It:
  - reads one JSONL per track;
  - converts tool arguments to the target shape (Qwen3 wants JSON strings);
  - adds meta.track when missing;
  - dedups exact rows;
  - samples each track to the requested weight;
  - splits each track, then shuffles the combined splits.

Example:
  python scripts/mix_datasets.py \
    --source kali=data/processed/train.jsonl \
    --source java=data/java/processed/train.jsonl \
    --source react=data/react/processed/train.jsonl \
    --weights kali=0.4,java=0.3,react=0.3 \
    --out data/unified/processed --arguments-format string
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path


def canonical(row: dict) -> str:
    return json.dumps(row.get("messages", []), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def normalize(row: dict, arguments_format: str) -> dict | None:
    row = copy.deepcopy(row)
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None

    for m in msgs:
        for call in m.get("tool_calls") or []:
            fn = call.get("function", call)
            raw = fn.get("arguments", {})
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    return None
            if not isinstance(raw, dict):
                return None
            fn["arguments"] = raw if arguments_format == "dict" else json.dumps(
                raw, ensure_ascii=False, separators=(",", ":")
            )
            call["function"] = fn

    meta = row.setdefault("meta", {})
    if not isinstance(meta, dict):
        return None
    return row


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_weights(raw: str | None, names: list[str]) -> dict[str, float]:
    if not raw:
        return {n: 1.0 for n in names}
    out: dict[str, float] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        name, sep, value = part.partition("=")
        if not sep:
            raise SystemExit(f"bad weight {part!r}; use name=0.4")
        out[name.strip()] = float(value)
    missing = [n for n in names if n not in out]
    if missing:
        raise SystemExit(f"missing weights for: {', '.join(missing)}")
    if any(v <= 0 for v in out.values()):
        raise SystemExit("weights must be positive")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True,
                    help="name=path, repeatable")
    ap.add_argument("--weights", help="name=0.4,name2=0.3; default equal")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", nargs=3, type=float, default=[0.9, 0.05, 0.05],
                    metavar=("TRAIN", "VALID", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arguments-format", choices=["string", "dict"], default="string",
                    help="string for Qwen3/OpenAI; dict for Qwen2.5-Coder")
    ap.add_argument("--max-per-source", type=int, default=0,
                    help="Optional cap before weighting; 0 = no cap")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if abs(sum(args.split) - 1.0) > 1e-6:
        raise SystemExit("--split must sum to 1.0")

    sources: dict[str, Path] = {}
    for item in args.source:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"bad --source {item!r}; use name=path")
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"missing source {p}")
        sources[name] = p

    weights = parse_weights(args.weights, list(sources))
    total_weight = sum(weights.values())

    loaded: dict[str, list[dict]] = {}
    for name, path in sources.items():
        rows: list[dict] = []
        seen: set[str] = set()
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = normalize(row, args.arguments_format)
            if row is None:
                continue
            key = canonical(row)
            if key in seen:
                continue
            seen.add(key)
            row.setdefault("meta", {}).setdefault("track", name)
            rows.append(row)
            if args.max_per_source and len(rows) >= args.max_per_source:
                break
        loaded[name] = rows

    available = {n: len(r) for n, r in loaded.items()}
    if not any(available.values()):
        raise SystemExit("no usable rows")

    # Weighted target: sample down large sources so the mix matches the ratios.
    total_available = sum(available.values())
    counts = {
        n: min(available[n], max(1, round(weights[n] / total_weight * total_available)))
        for n in loaded
    }
    # If a small source cannot fill its quota, do not silently over-sample it.
    rng = random.Random(args.seed)
    selected: dict[str, list[dict]] = {}
    for name, rows in loaded.items():
        rng.shuffle(rows)
        selected[name] = rows[: counts[name]]

    splits: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    track_stats: dict[str, dict[str, int]] = {}
    for name, rows in selected.items():
        n = len(rows)
        n_train = int(n * args.split[0])
        n_valid = int(n * args.split[1])
        if n >= 3:
            n_train = max(1, n_train)
            n_valid = max(1, n_valid)
        train = rows[:n_train]
        valid = rows[n_train:n_train + n_valid]
        test = rows[n_train + n_valid:]
        splits["train"].extend(train)
        splits["valid"].extend(valid)
        splits["test"].extend(test)
        track_stats[name] = {
            "available": available[name],
            "selected": n,
            "train": len(train),
            "valid": len(valid),
            "test": len(test),
        }

    for split in splits.values():
        rng.shuffle(split)

    print(f"arguments: {args.arguments_format}")
    for name, stat in track_stats.items():
        print(f"{name:12} available={stat['available']:>6} selected={stat['selected']:>6} "
              f"train={stat['train']:>5} valid={stat['valid']:>4} test={stat['test']:>4}")
    print(f"totals       train={len(splits['train'])} valid={len(splits['valid'])} test={len(splits['test'])}")

    if args.dry_run:
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        path = out / f"{split}.jsonl"
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "seed": args.seed,
        "arguments_format": args.arguments_format,
        "split": args.split,
        "weights": weights,
        "sources": {n: str(p) for n, p in sources.items()},
        "tracks": track_stats,
        "files": {
            split: {"rows": len(rows), "sha256": sha256_file(out / f"{split}.jsonl")}
            for split, rows in splits.items()
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}/{{train,valid,test}}.jsonl and manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
