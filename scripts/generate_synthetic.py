#!/usr/bin/env python3
"""Generate synthetic *task proposals* (or single-command plans) with an
OpenAI-compatible teacher such as DeepSeek.

This script does not execute anything. Pair it with execute_verified.py.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

PRICES = {  # USD per 1M tokens, off-peak Flash defaults
    "deepseek-flash": (0.15, 0.60),
    "deepseek-v4-pro": (0.66, 1.98),
}

SYSTEM = """You generate penetration-testing practice tasks for an isolated,
authorized lab. The lab network is 10.10.10.0/24. Never propose public IPs,
public domains, destructive actions, or real credentials. Tools: scope_check,
nmap_scan, http_probe, dns_lookup, tls_info, kali_exec, report_finding.
Return JSON only."""

TAXONOMY = [
    ("recon", "host discovery and service identification"),
    ("enum", "service enumeration and version detection"),
    ("web", "HTTP headers, technologies and exposed paths"),
    ("tls", "certificate and protocol inspection"),
    ("report", "summarize evidence into a finding"),
    ("refusal", "out-of-scope or ambiguous authorization"),
]


def task_prompt(n: int) -> str:
    rows = "\n".join(f"- {p}: {d}" for p, d in TAXONOMY)
    return f"""Propose {n} diverse lab tasks. Cover these phases:
{rows}

Return:
{{"tasks":[{{"id":"...","phase":"...","target":"10.10.10.x","goal":"...",
"constraints":["..."],"suggested_tools":["..."],"difficulty":"easy|medium|hard"}}]}}
"""


def plan_prompt(task: dict) -> str:
    return f"""Task: {task.get('goal')}
Target: {task.get('target')}
Allowed tools: {", ".join(task.get('suggested_tools') or ['nmap_scan','http_probe','kali_exec'])}
Scope: 10.10.10.0/24 only.

Return one JSON object:
{{"tool":"nmap_scan|http_probe|kali_exec","command":"a single allowlisted shell command, no pipes or redirects","arguments":{{...}},"reason":"..."}}
Prefer nmap_scan/http_probe over kali_exec. Never use a public target."""


def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for opener, closer in (("{", "}"), ("[", "]")):
            i, j = text.find(opener), text.rfind(closer)
            if i >= 0 and j > i:
                try:
                    return json.loads(text[i:j + 1])
                except json.JSONDecodeError:
                    pass
    return None


async def call(client, model, prompt, sem, stats, attempts=4):
    async with sem:
        for n in range(attempts):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.8,
                    max_tokens=4096,
                )
                u = r.usage
                pin, pout = PRICES.get(model, (0.30, 1.20))
                stats["in"] += u.prompt_tokens
                stats["out"] += u.completion_tokens
                stats["usd"] += (u.prompt_tokens * pin + u.completion_tokens * pout) / 1e6
                return extract_json(r.choices[0].message.content or "")
            except Exception as exc:
                if n == attempts - 1:
                    stats["failed"] += 1
                    print("generation failed:", exc, file=sys.stderr)
                    return None
                await asyncio.sleep((2 ** n) + random.random())


async def run(args, stats):
    from openai import AsyncOpenAI

    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("set DEEPSEEK_API_KEY or OPENAI_API_KEY", file=sys.stderr)
        return 2
    client = AsyncOpenAI(api_key=key, base_url=args.base_url)
    sem = asyncio.Semaphore(args.concurrency)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "tasks":
        batches = max(1, (args.tasks + args.batch - 1) // args.batch)
        results = await asyncio.gather(*[
            call(client, args.model, task_prompt(args.batch), sem, stats)
            for _ in range(batches)
        ])
        tasks = []
        for r in results:
            if isinstance(r, dict):
                tasks.extend(r.get("tasks") or [])
            elif isinstance(r, list):
                tasks.extend(r)
        tasks = tasks[:args.tasks]
        with out.open("w") as fh:
            for t in tasks:
                t.setdefault("id", hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest()[:12])
                t["teacher"] = args.model
                t["created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                fh.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"wrote {len(tasks)} tasks to {out}")
        return 0

    # plans: one task per line in --input
    tasks = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    tasks = tasks[:args.tasks]
    results = await asyncio.gather(*[
        call(client, args.model, plan_prompt(t), sem, stats) for t in tasks
    ])
    kept = 0
    with out.open("w") as fh:
        for task, plan in zip(tasks, results):
            if not isinstance(plan, dict):
                continue
            plan["task_id"] = task.get("id")
            plan["task"] = task.get("goal")
            plan["target"] = task.get("target")
            plan["teacher"] = args.model
            fh.write(json.dumps(plan, ensure_ascii=False) + "\n")
            kept += 1
    print(f"wrote {kept} plans to {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tasks", "plans"], default="tasks")
    ap.add_argument("--tasks", type=int, default=25)
    ap.add_argument("--batch", type=int, default=10, help="tasks per API call in tasks mode")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="data/interim/generated.jsonl")
    ap.add_argument("--input", help="tasks JSONL for --mode plans")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.mode == "plans" and not args.input:
        print("--input is required for --mode plans", file=sys.stderr)
        return 2

    if args.dry_run:
        p = task_prompt(args.batch) if args.mode == "tasks" else plan_prompt({"goal": "example", "target": "10.10.10.5"})
        print("=== prompt preview ===\n" + p)
        print(f"\napprox. requests: {max(1, (args.tasks + args.batch - 1) // args.batch)}")
        print("no API call was made (--dry-run)")
        return 0

    stats = {"in": 0, "out": 0, "usd": 0.0, "failed": 0}
    code = asyncio.run(run(args, stats))
    print(json.dumps(stats, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
