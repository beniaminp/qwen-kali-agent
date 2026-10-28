"""Connect to tickets_server over stdio, then list and use its features.

    python client_demo.py

Launches `python tickets_server.py` as a subprocess and talks MCP over its
stdin/stdout. The same code connects to a remote server if you pass a URL to
Client() instead of StdioServerParameters.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

SERVER = Path(__file__).with_name("tickets_server.py")  # works from any cwd


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with Client(params) as client:
        print("connected to:", client.server_info.name, client.server_info.version)
        print("protocol:", client.protocol_version)

        tools = await client.list_tools()
        print("tools:", [t.name for t in tools.tools])

        res = await client.call_tool("search_tickets", {"status": "open", "limit": 3})
        print("search_tickets ->", json.dumps(res.structured_content))

        one = await client.call_tool("get_ticket", {"ticket_id": 2})
        print("get_ticket(2).subject ->", one.structured_content["subject"])

        doc = await client.read_resource("ticket://2")
        print("resource ticket://2 mime ->", doc.contents[0].mime_type)

        prompts = await client.list_prompts()
        print("prompts:", [p.name for p in prompts.prompts])


if __name__ == "__main__":
    asyncio.run(main())
