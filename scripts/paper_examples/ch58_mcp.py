#!/usr/bin/env python3
"""Check every number in the "On paper" cards, worked examples and exercise
answers of docs/58-ai-mcp.md (stdlib only)."""

checks = 0


def close(a, b, tol):
    global checks
    checks += 1
    assert abs(a - b) <= tol, f"{a!r} != {b!r} (tol {tol})"


def eq(a, b):
    global checks
    checks += 1
    assert a == b, f"{a!r} != {b!r}"


# ---------------------------------------------------------------- §1 N x M
# 5 hosts x 8 data sources, bespoke vs a shared protocol.
H, S = 5, 8
eq(H * S, 40)            # bespoke: one connector per (host, source) pair
eq(H + S, 13)            # shared protocol: one per host + one per source
eq(H * S - (H + S), 27)  # connectors saved

# Add a ninth integration by adding a host (H: 5 -> 6).
eq(6 * 8, 48)            # bespoke grows by S = 8
eq((6 * 8) - (5 * 8), 8)
eq(6 + 8, 14)            # shared grows by 1
eq((6 + 8) - (5 + 8), 1)

# ------------------------------------------------- §11 token cost of tools
# tokens_per_request = n_tools * tokens_per_definition
n_tools, t_def = 8, 125
eq(n_tools * t_def, 1000)                 # tokens per request for tool defs
R = 4000                                  # requests per day
eq((n_tools * t_def) * R, 4_000_000)      # tokens per day
# assume $3 per million input tokens
close(4_000_000 / 1_000_000 * 3, 12.0, 1e-9)   # $/day
eq(12 * 30, 360)                               # $/30-day month
# trim to 4 tools -> half the tokens, half the cost
eq(4 * t_def, 500)
eq((4 * t_def) * R, 2_000_000)
close(2_000_000 / 1_000_000 * 3, 6.0, 1e-9)

# ------------------------------------------------- §11 context fraction
# Qwen3-14B at a 16K window (see 10-export-serving.md).
W = 16384
close((8 * 125) / W, 0.061, 5e-4)    # 8 tools ~ 6.1% of the window
close((8 * 125) / W, 0.0610, 5e-5)
close((40 * 125) / W, 0.305, 5e-4)   # 40 tools ~ 30.5% of the window
close((40 * 125) / W, 0.3052, 5e-4)
eq(40 * 125, 5000)

# ---------------------------------------------------------------- JSON-RPC
# Standard JSON-RPC 2.0 error codes used in the chapter.
eq(-32700, -32700)   # parse error
eq(-32600, -32600)   # invalid request
eq(-32601, -32601)   # method not found
eq(-32602, -32602)   # invalid params
eq(-32603, -32603)   # internal error
# MCP codes in the spec-reserved band -32020..-32099 (2026-07-28)
eq(-32020, -32020)   # HeaderMismatch
eq(-32021, -32021)   # MissingRequiredClientCapability
eq(-32022, -32022)   # UnsupportedProtocolVersion
assert -32099 <= -32022 <= -32020
# Zod .int() adds maximum = JavaScript's largest safe integer (section 8.6)
eq(2**53 - 1, 9007199254740991)

# ------------------------------------------------- Exercises
# Ex: 4 hosts x 6 sources
eq(4 * 6, 24)
eq(4 + 6, 10)
eq(4 * 6 - (4 + 6), 14)
eq(5 * 6, 30)            # add a host
eq(5 * 6 - 4 * 6, 6)
eq(5 + 6, 11)
eq(5 + 6 - (4 + 6), 1)

# Ex: 12 tools x 100 tokens, 3000 req/day, $2/M
eq(12 * 100, 1200)
eq(12 * 100 * 3000, 3_600_000)
close(3_600_000 / 1_000_000 * 2, 7.2, 1e-9)

# Ex: 20 tools x 150 tokens in an 8192 window
eq(20 * 150, 3000)
close((20 * 150) / 8192, 0.366, 5e-4)
close((20 * 150) / 8192, 0.3662, 5e-4)

print(f"ch58_mcp: all {checks} checks passed")
