#!/usr/bin/env python3
"""Run the same prompts against two OpenAI-compatible endpoints and record both.

This is a data collector, not an LLM judge. Use it to compare Gemma 4 12B and
Qwen3-14B on the same golden tasks, then grade the outputs with the track's real
verifier (Maven, Vitest/Playwright, or the Kali golden-set checks).

Example:
  python scripts/compare_models.py \
    --a-base-url http://127.0.0.1:8080/v1 --a-model qwen3-14b-kali \
    --b-base-url http://127.0.0.1:8083/v1 --b-model gemma4-12b-kali \
    --prompts data/golden/kali-prompts.jsonl \
    --tools examples/tools/kali-tools.json \
    --out reports/compare-kali.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from openai import OpenAI


def load_tools(path: str | None):
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "tools" in data:
        data = data["tools"]
    return data


def read_prompts(path: Path, system_file: str | None) -> list[dict]:
    system = Path(system_file).read_text() if system_file else None
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "messages" in row:
            msgs = row["messages"]
        else:
            text = row.get("prompt") or row.get("task") or row.get("input")
            if not text:
                continue
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": text})
        rows.append({
            "id": row.get("id", f"p{len(rows) + 1:04d}"),
            "track": row.get("track"),
            "messages": msgs,
            "tools": row.get("tools", []),
        })
    return rows


def serialize_calls(calls) -> list[dict]:
    out = []
    for call in calls or []:
        fn = call.function
        raw = fn.arguments
        try:
            arguments = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            arguments = raw
        out.append({"id": call.id, "name": fn.name, "arguments": arguments})
    return out


def call(client: OpenAI, model: str, messages: list[dict], tools, args) -> dict:
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
    }
    if args.seed is not None:
        kwargs["seed"] = args.seed
    if tools:
        kwargs["tools"] = tools
    started = time.time()
    try:
        response = client.chat.completions.create(**kwargs)
        elapsed = int((time.time() - started) * 1000)
        msg = response.choices[0].message
        usage = response.usage
        return {
            "ok": True,
            "latency_ms": elapsed,
            "content": msg.content or "",
            "tool_calls": serialize_calls(msg.tool_calls),
            "finish_reason": response.choices[0].finish_reason,
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            },
        }
    except Exception as exc:  # keep the run going; record the failure
        return {
            "ok": False,
            "latency_ms": int((time.time() - started) * 1000),
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-base-url", required=True)
    ap.add_argument("--b-base-url", required=True)
    ap.add_argument("--a-model", default="default")
    ap.add_argument("--b-model", default="default")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--tools")
    ap.add_argument("--system-file")
    ap.add_argument("--out", default="reports/compare.jsonl")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tools = load_tools(args.tools)
    prompts = read_prompts(Path(args.prompts), args.system_file)
    if args.limit:
        prompts = prompts[: args.limit]

    a = OpenAI(api_key="not-needed", base_url=args.a_base_url, timeout=args.timeout)
    b = OpenAI(api_key="not-needed", base_url=args.b_base_url, timeout=args.timeout)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "a_base_url": args.a_base_url,
        "b_base_url": args.b_base_url,
        "a_model": args.a_model,
        "b_model": args.b_model,
        "prompts": len(prompts),
        "tools": bool(tools),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "rows": 0,
        "a_errors": 0,
        "b_errors": 0,
        "a_tool_calls": 0,
        "b_tool_calls": 0,
        "a_latency_ms": 0,
        "b_latency_ms": 0,
        "a_tokens": 0,
        "b_tokens": 0,
    }

    with out.open("w") as fh:
        for i, item in enumerate(prompts, 1):
            row_tools = item.get("tools") or tools
            ra = call(a, args.a_model, item["messages"], row_tools, args)
            rb = call(b, args.b_model, item["messages"], row_tools, args)
            record = {
                "id": item["id"],
                "track": item.get("track"),
                "messages": item["messages"],
                "a": ra,
                "b": rb,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary["rows"] += 1
            for side, r in (("a", ra), ("b", rb)):
                if not r["ok"]:
                    summary[f"{side}_errors"] += 1
                    continue
                summary[f"{side}_tool_calls"] += len(r["tool_calls"])
                summary[f"{side}_latency_ms"] += r["latency_ms"]
                summary[f"{side}_tokens"] += (r.get("usage") or {}).get("total_tokens") or 0
            print(f"[{i}/{len(prompts)}] {item['id']}  A={'ok' if ra['ok'] else 'ERR'}  B={'ok' if rb['ok'] else 'ERR'}")

    for side in ("a", "b"):
        ok = summary["rows"] - summary[f"{side}_errors"]
        summary[f"{side}_avg_latency_ms"] = round(summary[f"{side}_latency_ms"] / ok) if ok else None
    (out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    # Grade the CSV/JSONL with the real verifiers; this script only collects.
    print("\nNext: apply each patch, run the track verifier, and compare pass rates. "
          "Do not judge by prose alone.")
    return 0 if summary["a_errors"] == 0 and summary["b_errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
