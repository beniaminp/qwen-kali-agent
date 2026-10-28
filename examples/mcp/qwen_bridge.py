"""Bridge MCP tools to an OpenAI-compatible chat model and run the tool loop.

This turns every tool on an MCP server into an OpenAI-style tool definition,
then runs the usual tool-calling loop against the handbook's local model
(mlx_lm.server at http://127.0.0.1:8080/v1, see 10-export-serving.md). The
model never speaks MCP; the bridge is the MCP client, and the model only sees
plain OpenAI tool definitions and tool results.

    python qwen_bridge.py "Which open tickets are urgent?"
    python qwen_bridge.py --thinking "..."   # keep Qwen3's thinking mode on

Qwen3 thinks by default: the reply's content is None (the reasoning sits in a
separate field) or starts with a <think>...</think> block. The bridge asks for
non-thinking mode per request (chat_template_kwargs), and still strips a think
block and guards a None content in case the server ignores the switch.
Serve an adapter trained with thinking on as trained (06-data-format.md §6):
use --thinking for it.

Tools that the server does not mark read-only run only after a human types y.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI

from mcp import Client, StdioServerParameters
from mcp.shared.exceptions import MCPError

SERVER = Path(__file__).with_name("tickets_server.py")  # works from any cwd
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)


def mcp_tools_to_openai(mcp_tools) -> list[dict]:
    """Map MCP tool definitions to the OpenAI chat-completions `tools` shape."""
    return [{
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description or "",
            "parameters": t.input_schema,
        },
    } for t in mcp_tools]


def visible_text(content: str | None) -> str:
    """The reply without Qwen3's reasoning; '' when content is None."""
    text = THINK_BLOCK.sub("", content or "")
    return text.split("<think>", 1)[0].strip()  # drop an unterminated block too


async def approved(tool, args: dict) -> bool:
    """Run read-only tools freely; ask a human before anything that writes.

    Annotations are only hints: trust read_only_hint only from a server you
    wrote or reviewed, as here.
    """
    if tool.annotations is not None and tool.annotations.read_only_hint:
        return True
    if not sys.stdin.isatty():
        return False  # nobody to ask: deny
    answer = await asyncio.to_thread(input, f"Allow {tool.name}({json.dumps(args)})? [y/N] ")
    return answer.strip().lower() == "y"


async def call_tool(mcp: Client, tools: dict, name: str, raw_args: str | None) -> str:
    """Execute one model tool call; every failure becomes text the model can read."""
    if name not in tools:
        return f"Unknown tool {name!r}. Available tools: {', '.join(tools)}."
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError as exc:
        return f"Arguments for {name} are not valid JSON ({exc.msg}). Send one JSON object."
    if not await approved(tools[name], args):
        return f"The user did not approve {name}. Do not call it again; answer without it."
    try:
        result = await mcp.call_tool(name, args)
    except MCPError as exc:  # protocol error, e.g. an unknown tool on another SDK
        return f"Tool {name} failed: {exc}"
    return "".join(c.text for c in result.content if getattr(c, "text", None))


async def run(question: str, base_url: str, model: str, thinking: bool = False,
              max_steps: int = 5) -> str:
    oai = AsyncOpenAI(base_url=base_url, api_key="local")
    # Non-thinking mode for the base model, temperature 0 for repeatable tool
    # calls (the Qwen3 card suggests 0.7 for chat). Thinking mode: the card
    # warns against greedy decoding, and the think block needs a bigger budget.
    extra = {} if thinking else {"chat_template_kwargs": {"enable_thinking": False}}
    sampling = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 4096} if thinking \
        else {"temperature": 0.0, "max_tokens": 1024}
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with Client(params) as mcp:
        listed = (await mcp.list_tools()).tools
        tools = {t.name: t for t in listed}
        messages = [
            {"role": "system", "content": "You answer support questions using the tools. "
                                          "Call a tool before stating facts about tickets."},
            {"role": "user", "content": question},
        ]
        for _ in range(max_steps):
            resp = await oai.chat.completions.create(
                model=model, messages=messages, tools=mcp_tools_to_openai(listed),
                tool_choice="auto", extra_body=extra, **sampling,
            )
            choice = resp.choices[0]
            msg = choice.message
            if not msg.tool_calls:
                text = visible_text(msg.content)
                if not text and choice.finish_reason == "length":
                    return "stopped: the model used up max_tokens before answering"
                return text
            # Echo the assistant turn (its tool calls, no reasoning) into the history.
            messages.append({
                "role": "assistant", "content": visible_text(msg.content),
                "tool_calls": [tc.model_dump(exclude_none=True) for tc in msg.tool_calls],
            })
            for tc in msg.tool_calls:
                text = await call_tool(mcp, tools, tc.function.name, tc.function.arguments)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})
        return "stopped: hit max_steps without a final answer"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="mlx-community/Qwen3-14B-4bit")
    ap.add_argument("--thinking", action="store_true",
                    help="Keep Qwen3 thinking mode on (for an adapter trained with it).")
    args = ap.parse_args()
    print(asyncio.run(run(args.question, args.base_url, args.model, args.thinking)))


if __name__ == "__main__":
    main()
