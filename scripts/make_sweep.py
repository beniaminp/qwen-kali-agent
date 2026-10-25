#!/usr/bin/env python3
"""Generate a small, reproducible MLX LoRA sweep from a base YAML config.

Example:
  python scripts/make_sweep.py --base configs/qwen3-14b-all-layers.yaml \
    --out configs/sweeps/r1 --rank 16,32 --lr 5e-5,1e-4 --layers all,16
"""
from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path

import yaml


def set_dotted(cfg: dict, key: str, value) -> None:
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="configs/sweeps")
    ap.add_argument("--rank", default="16")
    ap.add_argument("--lr", default="1e-4")
    ap.add_argument("--layers", default="all",
                    help="comma list; use 'all' or a negative number with '--layers=-1,16'")
    ap.add_argument("--scale", default="auto",
                    help="'auto' = 32/rank, or a comma-separated list")
    ap.add_argument("--iters", type=int, default=0, help="Override iterations if >0")
    ap.add_argument("--prefix", default="sweep")
    args = ap.parse_args()

    base = yaml.safe_load(Path(args.base).read_text())
    ranks = [int(x) for x in args.rank.split(",") if x.strip()]
    lrs = [float(x) for x in args.lr.split(",") if x.strip()]
    layers = []
    for x in args.layers.split(","):
        x = x.strip()
        if not x:
            continue
        layers.append(-1 if x.lower() == "all" else int(x))
    if not layers:
        layers = [-1]
    if args.scale == "auto":
        scales = {r: 32.0 / r for r in ranks}
    else:
        vals = [float(x) for x in args.scale.split(",") if x.strip()]
        scales = {r: vals[i % len(vals)] for i, r in enumerate(ranks)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for rank, lr, nlayers in itertools.product(ranks, lrs, layers):
        cfg = copy.deepcopy(base)
        name = f"{args.prefix}-r{rank}-lr{lr:g}-l{nlayers}"
        set_dotted(cfg, "lora_parameters.rank", rank)
        set_dotted(cfg, "lora_parameters.scale", scales[rank])
        set_dotted(cfg, "learning_rate", lr)
        set_dotted(cfg, "num_layers", nlayers)
        set_dotted(cfg, "adapter_path", f"adapters/sweeps/{name}")
        if args.iters:
            set_dotted(cfg, "iters", args.iters)
            if cfg.get("lr_schedule", {}).get("arguments"):
                cfg["lr_schedule"]["arguments"][1] = args.iters
        path = out / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        rows.append((name, path, rank, scales[rank], lr, nlayers, cfg.get("iters")))

    with (out / "commands.sh").open("w") as fh:
        fh.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for _, path, *_ in rows:
            fh.write(f"mlx_lm.lora --config {path}\n")
    (out / "commands.sh").chmod(0o755)

    with (out / "sweep.md").open("w") as fh:
        fh.write("# Sweep\n\n")
        fh.write("| run | rank | scale | lr | layers | iters |\n")
        fh.write("|---|---:|---:|---:|---:|---:|\n")
        for name, _, rank, scale, lr, nlayers, iters in rows:
            fh.write(f"| {name} | {rank} | {scale:g} | {lr:g} | {nlayers} | {iters or ''} |\n")
        fh.write("\nRun: `bash " + str(out / "commands.sh") + "`\n")

    print(f"wrote {len(rows)} configs to {out}")
    print(f"run: bash {out / 'commands.sh'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
