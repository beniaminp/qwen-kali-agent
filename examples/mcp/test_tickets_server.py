"""Automated tests for tickets_server, using the SDK client in-process.

    python -m pytest test_tickets_server.py      # or:
    python test_tickets_server.py                 # runs the checks directly
"""
from __future__ import annotations

import asyncio
import json

from mcp import Client
from mcp.shared.exceptions import MCPError

import tickets_server


async def _run() -> None:
    async with Client(tickets_server.server) as client:
        # initialize happened on __aenter__; the server identified itself
        assert client.server_info.name == "support-tickets"

        # list tools and check the schema the model will see
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert set(tools) == {"search_tickets", "get_ticket", "add_comment"}
        st = tools["search_tickets"]
        assert st.annotations.read_only_hint is True
        assert tools["add_comment"].annotations.read_only_hint is False
        assert st.input_schema["properties"]["limit"]["maximum"] == 50

        # a successful call returns structuredContent matching outputSchema
        res = await client.call_tool("search_tickets", {"status": "open", "limit": 2})
        assert res.is_error is False
        data = res.structured_content
        assert data["total"] == 3 and data["returned"] == 2

        # a bad argument comes back as a tool error (is_error=True), not a crash
        bad = await client.call_tool("get_ticket", {"ticket_id": 999})
        assert bad.is_error is True
        assert "No ticket with id 999" in bad.content[0].text

        # schema validation: limit out of range is rejected as a tool error
        oor = await client.call_tool("search_tickets", {"limit": 999})
        assert oor.is_error is True

        # a write tool: add a comment, then confirm it landed
        before = (await client.call_tool("get_ticket", {"ticket_id": 2})).structured_content
        await client.call_tool("add_comment", {"ticket_id": 2, "comment": "Paged on-call."})
        after = (await client.call_tool("get_ticket", {"ticket_id": 2})).structured_content
        assert len(after["comments"]) == len(before["comments"]) + 1

        # resource template: read ticket://1
        tmpls = await client.list_resource_templates()
        assert any(t.uri_template == "ticket://{ticket_id}" for t in tmpls.resource_templates)
        read = await client.read_resource("ticket://1")
        payload = json.loads(read.contents[0].text)
        assert payload["id"] == 1 and payload["priority"] == "high"

        # a missing or malformed resource id is a protocol error (-32602), not a crash
        for uri in ("ticket://99", "ticket://1_0"):
            try:
                await client.read_resource(uri)
            except MCPError as exc:
                assert exc.error.code == -32602
            else:
                raise AssertionError(f"{uri} should not resolve")

        # prompt
        prompts = {p.name for p in (await client.list_prompts()).prompts}
        assert "triage" in prompts
        got = await client.get_prompt("triage", {"ticket_id": "1"})
        assert "ticket://1" in got.messages[0].content.text

    print("test_tickets_server: all assertions passed")


def test_tickets_server() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    test_tickets_server()
