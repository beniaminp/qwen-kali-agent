#!/usr/bin/env python3
"""Check the arithmetic in every "On paper" example card outside chapter 43.

Each chapter's cards are asserted by one module in scripts/paper_examples/
(stdlib only; tolerances match the rounding shown in the text). Chapter 43 has
its own checker, scripts/check_math_examples.py. Run both after editing a card:

    python scripts/check_paper_examples.py
    python scripts/check_math_examples.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

MODULES = [
    "ch32_mathematics.py",
    "ch33_37_creation.py",
    "ch38_derivations.py",
    "ch39_42_distillation.py",
]


def main() -> int:
    here = Path(__file__).resolve().parent / "paper_examples"
    failed = []
    for name in MODULES:
        try:
            runpy.run_path(str(here / name), run_name="__main__")
        except AssertionError as exc:
            failed.append(name)
            print(f"FAIL {name}: {exc}", file=sys.stderr)
    if failed:
        print(f"{len(failed)} of {len(MODULES)} chapter checks failed", file=sys.stderr)
        return 1
    print(f"all {len(MODULES)} chapter checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
