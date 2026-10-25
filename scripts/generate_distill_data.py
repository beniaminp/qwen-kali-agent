#!/usr/bin/env python3
"""Generate sequence-level distillation data with an OpenAI-compatible teacher.

A "teacher" can be a local MLX/llama.cpp server, DeepSeek, or any provider that
speaks the OpenAI chat-completions API. The output is canonical training JSONL:
`messages` plus optional `tools`, one row per line, ready for
`scripts/prepare_dataset.py`.

Example:
  python scripts/generate_distill_data.py \
    --prompts data/distill/prompts.jsonl \
    --model deepseek-flash \
    --base-url https://api.deepseek.com \
    --out data/distill/raw/kali-teacher.jsonl \
    --n 2 --temperature 0.8 --max-tokens 2048

  # local Gemma 4 12B teacher
  python scripts/generate_distill_data.py \
    --prompts data/distill/prompts.jsonl \
    --model gemma-4-12b \
    --base-url http://127.0.0.1:8080/v1 \
    --api-key-env LOCAL_API_KEY \
    --out data/distill/raw/gemma-teacher.jsonl

The script does not execute tools. For agentic data, pair it with a verifier
(`scripts/execute_verified.py`) or keep teacher tool calls as proposals only.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def prompt_id(row: dict, index: int) -> str:
    if row.get("id"):
        return str(row["id"])
    payload = json.dumps(row.get("messages") or row.get("prompt"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def normalize_prompt(row: dict, default_system: str | None) -> list[dict]:
    if row.get("messages"):
        messages = [dict(m) for m in row["messages"]]
        if default_system and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": default_system})
        return messages
    text = row.get("prompt") or row.get("instruction") or row.get("question")
    if not text:
        raise ValueError("prompt row needs 'messages' or 'prompt'")
    messages = []
    if default_system:
        messages.append({"role": "system", "content": default_system})
    messages.append({"role": "user", "content": text})
    return messages


def strip_unsupported(message: dict) -> dict:
    """Keep only fields the OpenAI API accepts for an input message."""
    allowed = {"role", "content", "name", "tool_calls", "tool_call_id"}
    return {k: v for k, v in message.items() if k in allowed and v is not None}


def response_message(choice) -> dict:
    msg = choice.message
    out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        out["reasoning_content"] = reasoning
    if getattr(msg, "tool_calls", None):
        out["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in msg.tool_calls
        ]
    return out


async def call_one(client, args, messages: list[dict], tools: list[dict] | None, sem):
    import openai

    async with sem:
        for attempt in range(args.retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": args.model,
                    "messages": [strip_unsupported(m) for m in messages],
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "seed": args.seed,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                response = await client.chat.completions.create(**kwargs)
                usage = response.usage
                return {
                    "message": response_message(response.choices[0]),
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(usage, "completion_tokens", 0),
                    },
                }
            except Exception as exc:  # noqa: BLE001 - provider errors are broad
                if attempt == args.retries - 1:
                    print(f"generation failed after {args.retries} attempts: {exc}", file=sys.stderr)
                    return None
                await asyncio.sleep((2 ** attempt) + random.random())
    return None


async def run(args) -> int:
    from openai import AsyncOpenAI

    key_env = args.api_key_env
    key = os.environ.get(key_env) or ""
    if not key and not args.allow_empty_key:
        print(f"set {key_env} (or pass --allow-empty-key for a local server)", file=sys.stderr)
        return 2

    prompts = read_jsonl(Path(args.prompts))
    if args.limit:
        prompts = prompts[: args.limit]
    tools = json.loads(Path(args.tool_file).read_text()) if args.tool_file else None
    system = Path(args.system_file).read_text().strip() if args.system_file else args.system

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.resume and out.exists():
        for line in out.read_text().splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass

    client = AsyncOpenAI(api_key=key or "local", base_url=args.base_url)
    sem = asyncio.Semaphore(args.concurrency)
    started = time.time()
    stats = {"requests": 0, "failed": 0, "prompt_tokens": 0, "completion_tokens": 0}

    tasks = []
    metadata = []
    for i, row in enumerate(prompts):
        pid = prompt_id(row, i)
        if pid in done:
            continue
        messages = normalize_prompt(row, system)
        meta = {
            "id": pid,
            "source": row.get("source", "prompt"),
            "category": row.get("category", "general"),
            "track": row.get("track", args.track),
            "license": row.get("license", args.license),
            "teacher": args.model,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        for sample in range(args.n):
            row_id = f"{pid}-s{sample}" if args.n > 1 else pid
            sample_meta = {**meta, "sample": sample}
            tasks.append(call_one(client, args, messages, tools, sem))
            metadata.append((row_id, messages, sample_meta))

    results = await asyncio.gather(*tasks)
    mode = "a" if args.resume else "w"
    kept = 0
    with out.open(mode) as fh:
        for (pid, messages, meta), result in zip(metadata, results):
            if result is None:
                stats["failed"] += 1
                continue
            row = {
                "id": pid,
                "messages": messages + [result["message"]],
                "meta": {
                    **meta,
                    "finish_reason": result["finish_reason"],
                    **result["usage"],
                },
            }
            if tools:
                row["tools"] = tools
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            stats["requests"] += 1
            stats["prompt_tokens"] += result["usage"]["prompt_tokens"]
            stats["completion_tokens"] += result["usage"]["completion_tokens"]

    elapsed = time.time() - started
    print(f"wrote {kept} rows to {out} in {elapsed:.1f}s")
    print(json.dumps(stats, indent=2))
    return 0 if kept else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="JSONL with messages/prompt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--allow-empty-key", action="store_true", help="local server without auth")
    ap.add_argument("--system-file", help="File containing the teacher system prompt")
    ap.add_argument("--system", help="Inline teacher system prompt")
    ap.add_argument("--tool-file", help="JSON array of tool schemas")
    ap.add_argument("--n", type=int, default=1, help="samples per prompt")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--track", default="distill", help="meta.track recorded on every row")
    ap.add_argument("--license", default="unverified", help="meta.license recorded on every row")
    ap.add_argument("--limit", type=int, help="only the first N prompts")
    ap.add_argument("--resume", action="store_true", help="skip ids already in --out")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        prompts = read_jsonl(Path(args.prompts))[: args.limit or None]
        system = Path(args.system_file).read_text().strip() if args.system_file else args.system
        print(f"prompts: {len(prompts)}")
        print(f"requests: {len(prompts) * args.n} (n={args.n})")
        print(f"model: {args.model} @ {args.base_url}")
        if prompts:
            print("first prompt:")
            print(json.dumps(normalize_prompt(prompts[0], system), indent=2)[:1200])
        print("no API call was made (--dry-run)")
        return 0

    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
