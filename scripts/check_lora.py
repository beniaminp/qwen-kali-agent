#!/usr/bin/env python3
"""Inspect which modules MLX-LM actually adapts, and how many parameters train.

Run this on the Mac before a long job. It catches stale MLX versions, overly
narrow `keys` lists, and accidental attention-only adapters.

Requires MLX (macOS / Apple Silicon).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

def count_params(tree) -> int:
    from mlx.utils import tree_flatten
    return sum(p.size for _, p in tree_flatten(tree))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-14B-4bit")
    ap.add_argument("--num-layers", type=int, default=-1,
                    help="-1 = all blocks; otherwise the last N blocks")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--dora", action="store_true")
    ap.add_argument("--keys", help="Optional comma-separated module keys; normally omit")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, _ = load(args.model)
    blocks = len(model.layers)
    base_params = count_params(model.parameters())

    config = {"rank": args.rank, "scale": args.scale, "dropout": args.dropout}
    if args.keys:
        config["keys"] = {k.strip() for k in args.keys.split(",") if k.strip()}

    linear_to_lora_layers(model, args.num_layers, config, use_dora=args.dora)

    trainable = count_params(model.trainable_parameters())
    modules = [(n, m) for n, m in model.named_modules() if hasattr(m, "lora_a")]
    suffixes = Counter(n.rsplit(".", 1)[-1] for n, _ in modules)
    adapted_blocks = len({n.split(".layers.")[1].split(".")[0]
                          for n, _ in modules if ".layers." in n})

    report = {
        "model": args.model,
        "dora": args.dora,
        "blocks": blocks,
        "num_layers_requested": args.num_layers,
        "adapted_blocks": adapted_blocks,
        "lora_modules": len(modules),
        "by_projection": dict(sorted(suffixes.items())),
        "base_params": base_params,
        "trainable_params": trainable,
        "trainable_pct": round(100 * trainable / base_params, 3) if base_params else None,
        "sample_modules": [n for n, _ in modules[:12]],
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"model            : {args.model}")
        print(f"dora             : {args.dora}")
        print(f"blocks           : {blocks}")
        print(f"requested layers : {args.num_layers} (adapted {adapted_blocks})")
        print(f"lora modules     : {len(modules)}")
        print("by projection    : " + ", ".join(f"{k}={v}" for k, v in sorted(suffixes.items())))
        print(f"base params      : {base_params:,}")
        print(f"trainable params : {trainable:,} ({report['trainable_pct']}%)")
        print("sample modules   :")
        for n in report["sample_modules"]:
            print("  " + n)

    expected_min = adapted_blocks * 4
    if len(modules) < expected_min:
        print(f"\nWARNING: {len(modules)} modules for {adapted_blocks} blocks is unusually low; "
              "expected attention + MLP projections. Check MLX version and `keys`.")
        return 2
    if adapted_blocks == 0:
        print("\nERROR: no adapted blocks found.")
        return 1
    print("\nOK: adapter covers attention and MLP projections in the selected blocks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
