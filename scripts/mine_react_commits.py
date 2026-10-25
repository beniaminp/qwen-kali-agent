#!/usr/bin/env python3
"""Mine React/TypeScript commits from a git repository into seed task records.

This does not call an LLM and does not execute code. Each record is a candidate
for a repo-grounded synthetic task: the parent commit is the base, the commit
message is the problem statement, and the patch is a gold change.

Example:
  python scripts/mine_react_commits.py --repo ~/src/shadcn-ui/ui \
    --out data/react/interim/shadcn-seeds.jsonl --limit 300 --only-with-tests --license MIT
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

FRONTEND_EXT = (".tsx", ".jsx", ".ts", ".js", ".css", ".scss", ".json")
COMPONENT_EXT = (".tsx", ".jsx", ".css", ".scss")
TEST_MARKERS = (".test.", ".spec.", "/__tests__/", "/e2e/", "/tests/")


def git(repo: Path, *args: str, timeout: int = 60) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL, timeout=timeout
    )


def npm_commands(repo: Path) -> tuple[str, str]:
    """Return (setup_command, test_command) based on package.json scripts."""
    pkg = repo / "package.json"
    if not pkg.exists():
        return "", ""
    try:
        data = json.loads(pkg.read_text())
    except json.JSONDecodeError:
        return "", ""
    scripts = data.get("scripts") or {}
    manager = "npm"
    if data.get("packageManager", "").startswith("pnpm"):
        manager = "pnpm"
    elif data.get("packageManager", "").startswith("yarn"):
        manager = "yarn"

    setup = {
        "npm": "npm ci",
        "pnpm": "pnpm install --frozen-lockfile",
        "yarn": "yarn install --immutable",
    }[manager]

    parts = []
    if "typecheck" in scripts:
        parts.append(f"{manager} run typecheck")
    if "lint" in scripts:
        parts.append(f"{manager} run lint")
    if "test" in scripts:
        parts.append(f"{manager} test" + (" -- --run" if manager == "npm" else ""))
    if "build" in scripts:
        parts.append(f"{manager} run build")
    return setup, " && ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--out", default="data/react/interim/commit-seeds.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--since", help="Optional git --since value, e.g. 2025-01-01")
    ap.add_argument("--max-patch-bytes", type=int, default=24000)
    ap.add_argument("--max-files", type=int, default=25)
    ap.add_argument("--only-with-tests", action="store_true",
                    help="Require a changed test/E2E file")
    ap.add_argument("--license", default="unknown")
    args = ap.parse_args()

    repo = args.repo.expanduser().resolve()
    if not (repo / ".git").exists():
        print(f"not a git repository: {repo}")
        return 2

    setup, test_cmd = npm_commands(repo)
    fmt = "%H%x1f%ct%x1f%s%x1f%b%x1e"
    log_args = ["log", "--no-merges", f"--format={fmt}", f"-n{args.limit}"]
    if args.since:
        log_args.append(f"--since={args.since}")
    raw = git(repo, *log_args, timeout=120)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0

    with out.open("w") as fh:
        for record in raw.split("\x1e"):
            record = record.strip("\n")
            if not record:
                continue
            parts = record.split("\x1f")
            if len(parts) < 4:
                continue
            sha, ts, subject, body = parts[0], parts[1], parts[2], parts[3]
            try:
                files_blob = git(repo, "show", "--name-only", "--format=", "--diff-filter=AMR", sha, timeout=60)
                patch = git(repo, "show", "--format=", "--patch", "--no-color", sha, timeout=120)
                parent = git(repo, "rev-parse", "--verify", f"{sha}^", timeout=30).strip()
            except subprocess.CalledProcessError:
                skipped += 1
                continue

            files = [f for f in files_blob.splitlines() if f.strip()]
            frontend_files = [f for f in files if f.endswith(FRONTEND_EXT)]
            component_files = [f for f in frontend_files if f.endswith(COMPONENT_EXT)]
            if not component_files:
                skipped += 1
                continue
            if len(files) > args.max_files:
                skipped += 1
                continue
            if args.only_with_tests and not any(any(m in f for m in TEST_MARKERS) for f in files):
                skipped += 1
                continue
            if len(patch.encode("utf-8")) > args.max_patch_bytes:
                skipped += 1
                continue
            if "diff --git" not in patch:
                skipped += 1
                continue

            row = {
                "schema": "react-commit-task-v1",
                "repo": repo.name,
                "repo_path": str(repo),
                "base_commit": parent,
                "commit": sha,
                "commit_time": int(ts),
                "task": subject.strip(),
                "description": body.strip()[:4000],
                "files": files,
                "frontend_files": frontend_files,
                "component_files": component_files,
                "patch": patch,
                "setup_command": setup,
                "test_command": test_cmd,
                "license": args.license,
                "source": "git-commit",
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} seeds, skipped {skipped} commits -> {out}")
    print("Next: generate candidate tasks/patches with an LLM, apply them to base_commit, run typecheck/lint/test/build/Playwright, keep passes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
