"""A small, read-mostly MCP server over an in-memory support-ticket store.

Run it over stdio (default) or Streamable HTTP:

    python tickets_server.py                 # stdio, for a desktop host
    python tickets_server.py --http           # Streamable HTTP on 127.0.0.1:8765

The HTTP mode has no authorization, so it binds to loopback only and checks
the Host and Origin headers (DNS-rebinding defence). A shared or remote
deployment needs OAuth first (docs/58-ai-mcp.md, section 7).

Requires mcp>=2,<3 (the 2.x SDK: MCPServer, not the 1.x FastMCP).
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from mcp.server.mcpserver import MCPServer, Context
from mcp.server.mcpserver.exceptions import ToolError, ResourceNotFoundError
from mcp.server.transport_security import TransportSecuritySettings
from mcp_types import ToolAnnotations

# Log to stderr, NEVER stdout: on stdio, stdout carries protocol frames only.
logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("tickets")

Status = Literal["open", "pending", "closed"]
Priority = Literal["low", "normal", "high", "urgent"]
MAX_COMMENTS = 100  # bound what a looping agent can append to one ticket

# ---- the "database": a dict, seeded once --------------------------------
_TICKETS: dict[int, dict] = {
    1: {"id": 1, "subject": "Refund not received", "status": "open",
        "priority": "high", "customer": "ava@example.com",
        "opened": "2026-10-20", "comments": ["Customer says refund is 9 days late."]},
    2: {"id": 2, "subject": "Cannot log in after reset", "status": "open",
        "priority": "urgent", "customer": "liam@example.com",
        "opened": "2026-10-26", "comments": []},
    3: {"id": 3, "subject": "Feature request: dark mode", "status": "pending",
        "priority": "low", "customer": "noa@example.com",
        "opened": "2026-10-11", "comments": ["Logged for triage."]},
    4: {"id": 4, "subject": "Invoice total looks wrong", "status": "open",
        "priority": "normal", "customer": "mia@example.com",
        "opened": "2026-10-25", "comments": []},
}

# ---- typed result models (become outputSchema + structuredContent) ------
class TicketSummary(BaseModel):
    id: int
    subject: str
    status: Status
    priority: Priority

class SearchResult(BaseModel):
    total: int = Field(description="How many tickets matched before the limit.")
    returned: int = Field(description="How many are in this response.")
    tickets: list[TicketSummary]

class Ticket(BaseModel):
    id: int
    subject: str
    status: Status
    priority: Priority
    customer: str
    opened: str
    comments: list[str]

server = MCPServer(
    name="support-tickets",
    title="Support Tickets",
    version="1.0.0",
    instructions=(
        "Query and triage an in-memory support-ticket queue. "
        "Prefer search_tickets to narrow the queue, then get_ticket for detail. "
        "add_comment writes; call it only when the user asked to."
    ),
)

@server.tool(
    annotations=ToolAnnotations(
        title="Search tickets",
        read_only_hint=True,     # no side effects (so idempotent/destructive do not apply)
        open_world_hint=False,   # closed, local dataset
    ),
)
def search_tickets(
    status: Annotated[Status | None, Field(description="Filter by status; omit for any.")] = None,
    priority: Annotated[Priority | None, Field(description="Filter by priority; omit for any.")] = None,
    limit: Annotated[int, Field(ge=1, le=50, description="Max tickets to return (1-50).")] = 10,
) -> SearchResult:
    """Search the support queue by status and/or priority.

    Use this first to find ticket IDs; it returns short summaries, newest first.
    Then call get_ticket for the full record of one ticket.
    """
    rows = list(_TICKETS.values())
    if status is not None:
        rows = [t for t in rows if t["status"] == status]
    if priority is not None:
        rows = [t for t in rows if t["priority"] == priority]
    rows.sort(key=lambda t: t["opened"], reverse=True)
    picked = rows[:limit]
    log.info("search_tickets status=%s priority=%s -> %d/%d", status, priority, len(picked), len(rows))
    return SearchResult(
        total=len(rows),
        returned=len(picked),
        tickets=[TicketSummary(**{k: t[k] for k in ("id", "subject", "status", "priority")}) for t in picked],
    )

@server.tool(
    annotations=ToolAnnotations(title="Get ticket", read_only_hint=True, open_world_hint=False),
)
def get_ticket(ticket_id: Annotated[int, Field(ge=1, description="Numeric ticket id.")]) -> Ticket:
    """Return the full record of one ticket by its numeric id."""
    t = _TICKETS.get(ticket_id)
    if t is None:
        known = ", ".join(str(k) for k in sorted(_TICKETS))
        raise ToolError(f"No ticket with id {ticket_id}. Known ids: {known}.")
    return Ticket(**t)

@server.tool(
    annotations=ToolAnnotations(
        title="Add comment",
        read_only_hint=False,     # it writes
        destructive_hint=False,   # append-only, not a destroy/overwrite
        idempotent_hint=False,    # each call adds another comment
    ),
)
def add_comment(
    ticket_id: Annotated[int, Field(ge=1, description="Ticket to comment on.")],
    comment: Annotated[str, Field(min_length=1, max_length=2000, description="Comment text.")],
) -> Ticket:
    """Append a comment to a ticket. This is a write; use only when asked."""
    t = _TICKETS.get(ticket_id)
    if t is None:
        raise ToolError(f"No ticket with id {ticket_id}; nothing was written.")
    if len(t["comments"]) >= MAX_COMMENTS:
        raise ToolError(f"Ticket {ticket_id} already has {MAX_COMMENTS} comments; nothing was written.")
    t["comments"].append(comment)
    log.info("add_comment ticket=%d len=%d", ticket_id, len(comment))
    return Ticket(**t)

@server.resource(
    "ticket://{ticket_id}",
    title="Ticket record",
    mime_type="application/json",
)
def ticket_resource(ticket_id: str) -> str:
    """Expose each ticket as a JSON resource at ticket://<id>."""
    # Validate the URI variable strictly: int() alone would also accept
    # " 7", "+7" and "1_0".
    if not re.fullmatch(r"[1-9][0-9]{0,8}", ticket_id):
        raise ResourceNotFoundError(f"ticket id must be a positive integer, got {ticket_id!r}")
    tid = int(ticket_id)
    t = _TICKETS.get(tid)
    if t is None:
        raise ResourceNotFoundError(f"no ticket {tid}")
    return Ticket(**t).model_dump_json(indent=2)

@server.prompt(title="Triage a ticket")
def triage(ticket_id: str) -> str:
    """A prompt that asks the model to triage one ticket."""
    return (
        f"Read the resource ticket://{ticket_id}. Decide a severity (S1-S4), "
        f"name the single next action, and draft a one-line reply to the customer. "
        f"Be concise and do not invent facts that are not in the ticket."
    )

LOOPBACK = ("127.0.0.1", "localhost", "::1")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="Serve Streamable HTTP instead of stdio.")
    ap.add_argument("--host", default="127.0.0.1", choices=LOOPBACK,
                    help="Loopback only: this demo has no authorization.")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.http:
        # The SDK turns this on by itself for loopback hosts; spelled out here
        # so the defence is visible: a forged Host gets 421, a foreign Origin 403.
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
            allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"],
        )
        log.info("serving Streamable HTTP on http://%s:%d/mcp", args.host, args.port)
        server.run(transport="streamable-http", host=args.host, port=args.port,
                   transport_security=security)
    else:
        log.info("serving stdio")
        server.run()  # stdio

if __name__ == "__main__":
    main()
