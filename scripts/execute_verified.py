#!/usr/bin/env python3
"""Execute proposed commands only inside a disposable, network-isolated
Docker Kali container, then optionally build training trajectories.

Safety defaults:
  * dry-run unless --i-understand is passed;
  * refuses if Docker is missing;
  * refuses host networking or any non-internal network;
  * binary allowlist + destructive-pattern denylist;
  * no shell metacharacters, no host execution.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ALLOWED = {
    "nmap", "curl", "dig", "openssl", "nikto", "whatweb",
    "gobuster", "ffuf", "smbclient", "enum4linux",
}
DENY = [
    r"\brm\s+-rf\s+/(?:\s|$)", r"\bmkfs(\.\w+)?\b", r"\bdd\s+.*of=/dev/",
    r":\(\)\s*\{.*\};:", r"\bshutdown\b", r"\breboot\b",
    r"\b(curl|wget)\b[^\n]*\|\s*(?:ba|z|da)?sh\b",
]
META = re.compile(r"[;&|`$()<>]")
MAX_OUT = 4000


def sh(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def docker_ok() -> bool:
    try:
        return sh(["docker", "info"], 10).returncode == 0
    except Exception:
        return False


def container_networks(container: str) -> list[str]:
    r = sh(["docker", "inspect", "-f", "{{json .NetworkSettings.Networks}}", container])
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return list(json.loads(r.stdout).keys())


def network_internal(name: str) -> bool:
    r = sh(["docker", "network", "inspect", "-f", "{{.Internal}}", name])
    return r.returncode == 0 and r.stdout.strip().lower() == "true"


def check_safety(container: str) -> tuple[bool, str]:
    r = sh(["docker", "inspect", "-f", "{{.HostConfig.NetworkMode}}", container])
    if r.returncode != 0:
        return False, f"container not found: {r.stderr.strip()}"
    if r.stdout.strip() == "host":
        return False, "refusing host networking"
    nets = container_networks(container)
    if not nets:
        return False, "container has no networks"
    bad = [n for n in nets if not network_internal(n)]
    if bad:
        return False, f"refusing non-internal networks: {bad}"
    return True, "ok"


def validate_command(command: str) -> tuple[bool, str, list[str]]:
    if "\n" in command or "\r" in command:
        return False, "newlines are not allowed", []
    if META.search(command):
        return False, "shell metacharacters are not allowed", []
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return False, f"cannot parse command: {exc}", []
    if not argv:
        return False, "empty command", []
    binary = Path(argv[0]).name
    if binary not in ALLOWED:
        return False, f"binary not allowlisted: {binary}", argv
    for pat in DENY:
        if re.search(pat, command, re.I):
            return False, f"denied pattern: {pat}", argv
    return True, "ok", argv


def trajectory(row: dict, command: str, stdout: str, stderr: str, code: int) -> dict:
    tools = [{
        "type": "function",
        "function": {
            "name": "kali_exec",
            "description": "Run an allowlisted Kali command in the isolated lab.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_s": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    }]
    obs = f"exit_code={code}\nstdout:\n{stdout}\nstderr:\n{stderr}"[:MAX_OUT]
    messages = [
        {"role": "system", "content": "You are a security assistant in an authorized lab. Scope: 10.10.10.0/24."},
        {"role": "user", "content": row.get("task") or command},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "kali_exec",
                         "arguments": json.dumps({"command": command}, ensure_ascii=False)},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": obs},
    ]
    return {"messages": messages, "tools": tools}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL with {command} or {commands:[...]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--container", default="kali-lab")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--trajectory", action="store_true", help="also emit chat+tools messages")
    ap.add_argument("--dry-run", action="store_true", help="print what would run")
    ap.add_argument("--i-understand", action="store_true", help="required to execute")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]

    if not args.i_understand:
        args.dry_run = True

    if not args.dry_run:
        if not docker_ok():
            print("Docker is not available; refusing to execute.", file=sys.stderr)
            return 3
        ok, why = check_safety(args.container)
        if not ok:
            print(f"unsafe sandbox: {why}", file=sys.stderr)
            return 3

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    executed = kept = 0

    with out.open("w") as fh:
        for row in rows:
            commands = row.get("commands") or ([row["command"]] if row.get("command") else [])
            for cmd in commands:
                ok, why, argv = validate_command(cmd)
                if not ok:
                    print(f"REJECT {why}: {cmd!r}", file=sys.stderr)
                    continue
                if args.dry_run:
                    print(f"DRY-RUN {args.container}: {cmd}")
                    continue
                try:
                    r = subprocess.run(
                        ["docker", "exec", "-w", "/tmp", "-e", "HOME=/tmp", args.container, *argv],
                        capture_output=True, text=True, timeout=args.timeout,
                    )
                    stdout = (r.stdout or "")[:MAX_OUT]
                    stderr = (r.stderr or "")[:MAX_OUT]
                    code = r.returncode
                except subprocess.TimeoutExpired:
                    stdout, stderr, code = "", f"timeout after {args.timeout}s", 124
                executed += 1
                record = {
                    "id": row.get("id"),
                    "task": row.get("task"),
                    "target": row.get("target"),
                    "command": cmd,
                    "argv": argv,
                    "exit_code": code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "verified": code == 0,
                    "container": args.container,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                if args.trajectory:
                    record["trajectory"] = trajectory(row, cmd, stdout, stderr, code)
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
                print(f"{'OK ' if code == 0 else 'ERR'} ({code}) {cmd}")

    print(f"executed={executed} written={kept} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
