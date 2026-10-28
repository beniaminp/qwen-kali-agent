#!/usr/bin/env python3
"""Check every number in the "On paper" cards, worked examples and exercise
answers of docs/61-ai-agent-frameworks.md (stdlib only).

Numbers come from real measurements of the chapter's framework scripts run
against a local OpenAI-compatible stub: lines of code per implementation,
per-request byte sizes the stub logged, and the system-prompt lengths the
frameworks actually sent (text characters, not the repr of a content-part
list). Tolerances match the rounding shown.
"""

checks = 0


def close(a, b, tol):
    global checks
    checks += 1
    assert abs(a - b) <= tol, f"{a!r} != {b!r} (tol {tol})"


def eq(a, b):
    global checks
    checks += 1
    assert a == b, f"{a!r} != {b!r}"


# ---------------------------------------------------------------- measured data
# Source-lines-of-code for the identical two-tool order task (non-blank,
# non-comment lines; function docstrings count), counted from the doc blocks.
SLOC = {
    "baseline": 43, "langgraph": 22, "llamaindex": 24, "openai_agents": 25,
    "pydantic_ai": 25, "smolagents": 26, "crewai": 27, "haystack": 29,
    "agno": 19, "strands": 23, "dspy": 19, "adk": 34,
}
BASELINE_TOTAL_LINES = 54        # §2.1: 43 code lines + 11 blank lines
# First-request body size in bytes, as the stub logged it (thinking disabled).
FIRST_BYTES = {
    "baseline": 602, "langgraph": 663, "llamaindex": 916, "openai_agents": 813,
    "pydantic_ai": 748, "smolagents": 4872, "crewai": 1104, "haystack": 663,
    "agno": 626, "strands": 794, "adk": 812,
}
# System prompt the framework itself injected (characters of text sent).
SYSTEM_CHARS = {
    "baseline": 0, "langgraph": 0, "llamaindex": 0, "haystack": 0,
    "pydantic_ai": 0, "agno": 0, "openai_agents": 0, "strands": 0,
    "crewai": 124, "adk": 50, "dspy_react_step": 2265,
    "smolagents_toolcalling": 3544, "smolagents_code": 8814,
}
REQUESTS_PER_TASK = 3             # tool call, tool call, final answer
PREFILL_TOK_S = 120               # ch50 estimate, Qwen3-14B 4-bit, M1 Pro
CONTEXT_16K = 16384

eq(SLOC["baseline"], 43)
eq(BASELINE_TOTAL_LINES - SLOC["baseline"], 11)
eq(min(SLOC.values()), 19)        # Agno / DSPy are the leanest
eq(sorted(k for k, v in SLOC.items() if v == 19), ["agno", "dspy"])
eq(max(v for k, v in SLOC.items() if k != "baseline"), 34)   # ADK
eq(max(FIRST_BYTES.values()), 4872)   # smolagents sends the most
eq(min(FIRST_BYTES.values()), 602)    # the hand-rolled baseline the least
eq(max(v for k, v in SYSTEM_CHARS.items() if k != "smolagents_code"), 3544)
eq(sorted(k for k, v in SYSTEM_CHARS.items() if v == 0 and k != "baseline"),
   sorted(["langgraph", "llamaindex", "haystack", "pydantic_ai", "agno",
           "openai_agents", "strands"]))

# ---------------------------------------------------------------- §16 LOC
# Table order (ascending) and the prose ranges.
order = ["agno", "dspy", "langgraph", "strands", "llamaindex", "openai_agents",
         "pydantic_ai", "smolagents", "crewai", "haystack", "adk", "baseline"]
eq([SLOC[k] for k in order], sorted(SLOC[k] for k in order))
eq(round(100 * 19 / 43), 44)
eq(round(100 * 34 / 43), 79)
mid = [SLOC[k] / 43 for k in order if 22 <= SLOC[k] <= 29]
assert all(0.5 <= r <= 2 / 3 + 0.01 for r in mid)   # "half to two-thirds"
checks += 1
# Card: 43 / 19 and 43 / 22.
close(43 / 19, 2.3, 5e-2)
close(43 / 19, 2.263, 5e-4)
close(43 / 22, 1.95, 5e-3)
eq(43 - 19, 24)

# ---------------------------------------------------------------- §18.6 tokens
# Card: the ToolCallingAgent's 3544-char system prompt at ~4 chars/token,
# re-sent on each of the 3 calls.
tok = SYSTEM_CHARS["smolagents_toolcalling"] / 4
eq(tok, 886.0)
eq(REQUESTS_PER_TASK * 886, 2658)
close(886 / CONTEXT_16K, 0.05, 5e-3)          # "about 5% of a 16K window"
close(886 / PREFILL_TOK_S, 7.0, 0.5)          # "about 7 s" of prefill
close(SYSTEM_CHARS["smolagents_code"] / 4, 2200, 10)   # CodeAgent per call
# §1.2 repeats the prefill number.
eq(round(886 / PREFILL_TOK_S), 7)

# ---------------------------------------------------------------- §18.6 bytes
close(4872 / 602, 8.1, 5e-2)
close(4872 / 602, 8.093, 5e-3)
eq(4872 - 602, 4270)
assert SYSTEM_CHARS["smolagents_toolcalling"] > 4270 / 2   # "most of" it
checks += 1

# ---------------------------------------------------------------- §18.6 cost
# Card: assume $0.50 per million tokens; 2658 scaffold tokens per task.
close(2658 / 1e6 * 0.50, 0.001329, 1e-12)
close(2658 / 1e6 * 0.50, 0.0013, 5e-5)
close(1e5 * 0.001329, 132.9, 1e-9)
eq(round(1e5 * 2658 / 1e6 * 0.50), 133)

# ---------------------------------------------------------------- §14.1 demo
eq(len("<think>\nThe task needs"), 22)
eq(len("Order A1001 shipped vi"), 22)

# ---------------------------------------------------------------- Exercise 21.5
# A naive 5-step multi-agent flow vs a single call, each step 0.98 reliable.
close(0.98 ** 5, 0.904, 5e-4)
close(0.98 ** 1, 0.98, 1e-12)

# ---------------------------------------------------------------- Exercise 21.4
eq(3544 / 4, 886)
eq(3 * 886, 2658)

# ---------------------------------------------------------------- §19.2 adapter
# Protocol (2 lines) + wrapper class (10 lines) = "about a dozen lines".
eq(2 + 10, 12)

# ---------------------------------------------------------------- frameworks run
# Eleven frameworks plus the baseline ran the task ("all twelve above").
ran = [k for k in SLOC if k != "baseline"]
eq(len(ran), 11)
eq(len(ran) + 1, 12)

if __name__ == "__main__":
    print(f"ch61_frameworks: all {checks} checks passed")
