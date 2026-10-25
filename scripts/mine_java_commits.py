#!/usr/bin/env python3
"""Mine Java commits from a git repository into seed task records.

This does not call an LLM and does not execute code. Each record is a candidate
for a repo-grounded synthetic task: the parent commit is the base, the commit
message is the problem statement, and the patch is a gold change. Pair it with
scripts/generate_synthetic.py and a Maven sandbox.

Example:
  python scripts/mine_java_commits.py --repo ~/src/quarkus-quickstarts \
    --out data/java/interim/commit-seeds.jsonl --limit 200 --only-with-tests
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def git(repo: Path, *args: str, timeout: int = 60) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL, timeout=timeout
    )


def test_command(repo: Path) -> str:
    if (repo / "mvnw").exists():
        return "./mvnw -q -DskipITs test"
    if (repo / "gradlew").exists():
        return "./gradlew test"
    if (repo / "pom.xml").exists():
        return "mvn -q -DskipITs test"
    if (repo / "build.gradle").exists() or (repo / "build.gradle.kts").exists():
        return "gradle test"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--out", default="data/java/interim/commit-seeds.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--since", help="Optional git --since value, e.g. 2025-01-01")
    ap.add_argument("--max-patch-bytes", type=int, default=20000)
    ap.add_argument("--max-files", type=int, default=25)
    ap.add_argument("--only-with-tests", action="store_true",
                    help="Require at least one changed file under src/test")
    ap.add_argument("--license", default="unknown")
    args = ap.parse_args()

    repo = args.repo.expanduser().resolve()
    if not (repo / ".git").exists():
        print(f"not a git repository: {repo}")
        return 2

    fmt = "%H%x1f%ct%x1f%s%x1f%b%x1e"
    log_args = ["log", "--no-merges", f"--format={fmt}", f"-n{args.limit}"]
    if args.since:
        log_args.append(f"--since={args.since}")
    raw = git(repo, *log_args, timeout=120)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    cmd = test_command(repo)

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
            java_files = [f for f in files if f.endswith(".java")]
            if not java_files:
                skipped += 1
                continue
            if len(files) > args.max_files:
                skipped += 1
                continue
            if args.only_with_tests and not any("/src/test/" in f or f.startswith("src/test/") for f in files):
                skipped += 1
                continue
            if len(patch.encode("utf-8")) > args.max_patch_bytes:
                skipped += 1
                continue
            if "diff --git" not in patch:
                skipped += 1
                continue

            row = {
                "schema": "java-commit-task-v1",
                "repo": repo.name,
                "repo_path": str(repo),
                "base_commit": parent,
                "commit": sha,
                "commit_time": int(ts),
                "task": subject.strip(),
                "description": body.strip()[:4000],
                "files": files,
                "java_files": java_files,
                "patch": patch,
                "test_command": cmd,
                "license": args.license,
                "source": "git-commit",
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} seeds, skipped {skipped} commits -> {out}")
    print("Next: generate candidate tasks/patches with an LLM, apply them to base_commit, run the test command, keep passes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
