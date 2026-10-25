#!/usr/bin/env python3
"""Preflight for the Qwen Kali Agent handbook.

Checks the host, Python, MLX, memory headroom, disk and optional tooling.
Exits non-zero on a hard failure.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-free-gb", type=float, default=100.0)
    args = ap.parse_args()

    hard_fail = False
    warn = []

    print("== host ==")
    print("platform:", platform.platform())
    print("machine :", platform.machine())
    if sys.platform != "darwin":
        warn.append("not macOS; MLX/Metal checks will be skipped")
    if platform.machine() != "arm64":
        warn.append("not arm64; MLX requires Apple Silicon")

    print("\n== python ==")
    print("version:", sys.version.split()[0])
    if sys.version_info < (3, 11):
        hard_fail = True
        print("ERROR: Python >= 3.11 is required")

    print("\n== packages ==")
    try:
        import mlx.core as mx  # type: ignore

        print("mlx     :", getattr(mx, "__version__", "ok"))
        if sys.platform == "darwin":
            info = mx.device_info()
            for k, v in info.items():
                if isinstance(v, int) and v > 1024**2:
                    print(f"  {k}: {human(v)}")
                else:
                    print(f"  {k}: {v}")
            device = str(mx.default_device()).lower()
            if "gpu" not in device:
                hard_fail = True
                print("ERROR: default MLX device is not GPU")
    except Exception as exc:
        warn.append(f"mlx import failed: {exc}")

    for name in ("mlx_lm", "transformers", "huggingface_hub"):
        try:
            mod = __import__(name)
            print(f"{name:9}:", getattr(mod, "__version__", "ok"))
        except Exception as exc:
            warn.append(f"{name} import failed: {exc}")

    print("\n== memory ==")
    if sys.platform == "darwin":
        mem = run(["sysctl", "-n", "hw.memsize"])
        if mem:
            print("physical:", human(int(mem)))
        wired = run(["sysctl", "-n", "iogpu.wired_limit_mb"])
        print("wired limit (MB, 0=default):", wired or "unknown")
    else:
        try:
            print("physical:", human(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")))
        except Exception:
            pass

    print("\n== disk ==")
    for label, path in (("repo", Path.cwd()), ("HF_HOME", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")))):
        try:
            free = shutil.disk_usage(path).free
            print(f"{label:8}: {human(free)} free at {path}")
            if label == "repo" and free < args.min_free_gb * 1024**3:
                warn.append(f"only {human(free)} free; recommended >= {args.min_free_gb:.0f} GB")
        except Exception:
            print(f"{label:8}: unavailable ({path})")

    print("\n== optional tooling ==")
    for name in ("git", "hf", "llama-server", "ollama", "docker"):
        path = shutil.which(name)
        print(f"{name:12}: {path or 'not found'}")

    print("\n== model/tool files ==")
    for p in (
        Path("examples/tools/kali-tools.json"),
        Path("configs/qwen3-14b-balanced.yaml"),
        Path("data/processed/train.jsonl"),
    ):
        print(f"{str(p):42}: {'ok' if p.exists() else 'missing'}")

    print("\n== result ==")
    if warn:
        print("WARNINGS:")
        for w in warn:
            print(" -", w)
    if hard_fail:
        print("FAIL")
        return 1
    print("OK" if not warn else "OK (with warnings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
