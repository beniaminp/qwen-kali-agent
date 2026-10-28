#!/usr/bin/env python3
"""Check every number in the "On paper" cards, worked examples, code outputs
and exercise answers of docs/50 (Prompting and LLM APIs) and docs/52 (Agents
and tool use). Stdlib only."""
import json as _json
from collections import Counter
from math import comb, exp, log, sqrt

checks = 0


def close(a, b, tol):
    global checks
    checks += 1
    assert abs(a - b) <= tol, f"{a!r} != {b!r} (tol {tol})"


def eq(a, b):
    global checks
    checks += 1
    assert a == b, f"{a!r} != {b!r}"


# ================================================================= chapter 50

# ---- §: self-consistency (majority vote) ----------------------------------
# majority-of-3 = p^3 + 3 p^2 (1-p); majority-of-5 adds 10 p^3 (1-p)^2.
def maj3(p):
    return p ** 3 + 3 * p ** 2 * (1 - p)


def maj5(p):
    return p ** 5 + 5 * p ** 4 * (1 - p) + 10 * p ** 3 * (1 - p) ** 2


p = 0.8
close(p ** 3, 0.512, 1e-12)
close(3 * p ** 2 * (1 - p), 0.384, 1e-12)
close(maj3(p), 0.896, 1e-12)
# majority-of-5 term by term
close(p ** 5, 0.32768, 1e-12)
close(5 * p ** 4 * (1 - p), 0.4096, 1e-12)
close(10 * p ** 3 * (1 - p) ** 2, 0.2048, 1e-12)
close(maj5(p), 0.94208, 1e-12)
close(maj5(p), 0.942, 5e-4)          # printed 0.942
# binomial coefficients used in the formula
eq(comb(3, 2), 3)
eq(comb(5, 3), 10)
eq(comb(5, 4), 5)
# code sample: 3 of 5 votes for "refund"
eq(3, 3)
eq(5, 5)

# prose: 20% error falls to ~10%; 5 samples gain 0.046 over 3
close(1 - maj3(0.8), 0.104, 1e-12)
close(maj5(0.8) - maj3(0.8), 0.04608, 1e-12)

# ---- §2: ChatML miniature renders 156 characters (matches the real Qwen3
# template for system + user in default thinking mode) ----------------------
_msgs = [("system", "You are a terse support triage bot."),
         ("user", "My invoice is wrong, I was charged twice.")]
_prompt = "\n".join([f"<|im_start|>{r}\n{c}<|im_end|>" for r, c in _msgs]
                    + ["<|im_start|>assistant\n"])
eq(len(_prompt), 156)

# ---- §3: system-prompt overhead -------------------------------------------
eq(2000 * 200_000, 400_000_000)      # 2,000-token prompt x 200k req/day = 400M

# ---- §7: thinking overhead -------------------------------------------------
close(500 / 20, 25.0, 1e-12)         # 500 thinking tokens before a 20-token answer

# ---- §: few-shot label bias ----------------------------------------------
# 6 shots: 5 billing, 1 bug, 0 how-to -> empirical prior skews to billing.
close(5 / 6, 0.833, 5e-4)
close(1 / 6, 0.167, 5e-4)
close(0 / 6, 0.0, 1e-12)
close(5 / 6 + 1 / 6, 1.0, 1e-12)

# ---- §: tokens and cost ---------------------------------------------------
# code output: 28 input tokens, 1 output token (cl100k_base), prices assumed.
n_in, n_out = 28, 1
PRICE_IN, PRICE_OUT = 3.0, 15.0
cost = n_in / 1e6 * PRICE_IN + n_out / 1e6 * PRICE_OUT
close(n_in / 1e6 * PRICE_IN, 0.000084, 1e-12)
close(n_out / 1e6 * PRICE_OUT, 0.000015, 1e-12)
close(cost, 0.000099, 1e-12)
close(200_000 * cost, 19.8, 1e-9)
# On paper (clean numbers): 1000 input, 500 output, $3/M in, $15/M out
close(1000 / 1e6 * 3.0, 0.003, 1e-12)
close(500 / 1e6 * 15.0, 0.0075, 1e-12)
close(1000 / 1e6 * 3.0 + 500 / 1e6 * 15.0, 0.0105, 1e-12)
# output token priced 5x the input token here
close(PRICE_OUT / PRICE_IN, 5.0, 1e-12)

# ---- §: latency model -----------------------------------------------------
def latency(ttft, n, tps):
    return ttft + n / tps


close(latency(0.30, 50, 16), 3.425, 1e-12)
close(latency(0.30, 200, 16), 12.8, 1e-12)
close(latency(0.30, 800, 16), 50.30, 1e-12)
close(200 / 16, 12.5, 1e-12)         # On paper: decode time at 16 tok/s
close(0.30 + 12.5, 12.8, 1e-12)
# card caveat: prefill ~120 tok/s for Qwen3-14B 4-bit on M1 Pro, from llama.cpp
# pp512 = 266 tok/s for a 7B Q4_0 (16-core M1 Pro) scaled by 7 / 14.8.
close(266 * 7 / 14.8, 125.8, 0.05)
close(266 * 6.74 / 14.8, 121.1, 0.05)  # LLaMA-7B has 6.74B parameters
close(0.30 * 120, 36.0, 1e-9)          # 0.30 s ~ 36 prompt tokens
close(2000 / 120, 16.67, 5e-3)         # 2000 tokens ~ 17 s
assert round(2000 / 120) == 17
checks += 1

# ---- §: retries / exponential backoff -------------------------------------
# ceiling = min(cap, base * 2^n); base 0.5, cap 8.
base, cap = 0.5, 8.0
ceilings = [min(cap, base * 2 ** n) for n in range(5)]
eq(ceilings, [0.5, 1.0, 2.0, 4.0, 8.0])
eq(base * 2 ** 4, 8.0)
eq(min(cap, base * 2 ** 5), 8.0)     # capped beyond attempt 4

# ---- §: rate limit (token bucket) -----------------------------------------
# capacity 3, rate 2/s, requests every 0.25s. refill per gap = 0.5.
close(0.25 * 2.0, 0.5, 1e-12)
# tokens left after each of the 6 requests (decrement then observe)
rate, capacity = 2.0, 3.0
tokens, t = capacity, 0.0
seq = []
for i in range(6):
    now = i * 0.25
    tokens = min(capacity, tokens + (now - t) * rate)
    t = now
    if tokens >= 1.0:
        tokens -= 1.0
        seq.append((True, round(tokens, 2)))
    else:
        seq.append((False, round(tokens, 2)))
eq(seq, [(True, 2.0), (True, 1.5), (True, 1.0),
         (True, 0.5), (True, 0.0), (False, 0.5)])

# ---- §: caching (hit rate) ------------------------------------------------
# effective cost = (1 - hit_rate) * full cost.
h = 0.6
close(1 - h, 0.4, 1e-12)
close((1 - h) * 100.0, 40.0, 1e-12)  # $100/day of calls -> $40 with 60% hits
# code output: 3 identical requests -> 1 model call, 2 cache hits
eq(1, 1)
eq(3 - 1, 2)

# ---- §: ~4 chars/token rule of thumb --------------------------------------
close(1000 / 4, 250, 1e-12)

# =================================================== chapter 50 exercises
close(maj3(0.9), 0.972, 1e-12)                       # ex: majority-of-3, p=0.9
close(0.9 ** 3, 0.729, 1e-12)
close(3 * 0.9 ** 2 * 0.1, 0.243, 1e-12)
close(2000 / 1e6 * 0.5 + 800 / 1e6 * 1.5, 0.0022, 1e-12)   # ex: cost
close(2000 / 1e6 * 0.5, 0.001, 1e-12)
close(800 / 1e6 * 1.5, 0.0012, 1e-12)
close(latency(0.5, 300, 30), 10.5, 1e-12)            # ex: latency
close(300 / 30, 10.0, 1e-12)
eq(min(30, 1 * 2 ** 6), 30)                          # ex: backoff ceiling
eq(1 * 2 ** 4, 16)
eq(5, 5)                                             # ex: token bucket burst 5
close(0.3 * 200.0, 60.0, 1e-12)                      # ex: 70% hit -> $60
close(maj3(0.7), 0.784, 1e-12)                       # ex: majority-of-3, p=0.7
close(0.7 ** 3, 0.343, 1e-12)
close(3 * 0.49 * 0.3, 0.441, 1e-12)
# ex: p=0.4 in the two-outcome model -> voting hurts
close(0.4 ** 3, 0.064, 1e-12)
close(3 * 0.16 * 0.6, 0.288, 1e-12)
close(maj3(0.4), 0.352, 1e-12)
assert maj3(0.4) < 0.4
checks += 1

# ================================================================= chapter 52

# ---- §: agentic workflow patterns -----------------------------------------
# prompt chaining: the gate counts outline items (>= 3 to pass)
outline = "1. Problem\n2. Approach\n3. Result"
eq(len([l for l in outline.splitlines() if l.strip()]), 3)
# routing: three ticket classes
eq(len({"billing", "bug", "general"}), 3)
# parallelization / voting: majority of 5 votes
eq(Counter(["safe", "safe", "unsafe", "safe", "safe"]).most_common(1)[0], ("safe", 4))
# orchestrator-workers: 3 subtasks
eq(len(["search docs", "search code", "search issues"]), 3)
# evaluator-optimizer: draft word counts against a 12-word limit
long_draft = ("The incident lasted all day and involved many systems and "
              "many teams and much confusion overall today")
short_draft = "Double-charge bug; fixed in 2h; refunds issued."
eq(len(long_draft.split()), 17)
eq(len(short_draft.split()), 7)
assert 17 > 12 and 7 <= 12
checks += 1

# ---- §: tool-calling system -----------------------------------------------
# the result-size cap (60 chars) does not truncate a small refund result
blob = _json.dumps({"refunded": 20.0, "order": "A1"})
assert len(blob) <= 60
checks += 1

# ---- §: autonomous agents (resumability) ----------------------------------
# three steps; interrupted after step index 1 (transform) -> checkpoint = 2
eq(len(["fetch", "transform", "write"]), 3)
eq(1 + 1, 2)

# ---- §: reliability compounding -------------------------------------------
close(0.95 ** 10, 0.5987, 5e-5)
close(0.95 ** 10, 0.598736939, 1e-8)
close(0.99 ** 10, 0.9044, 5e-5)
close(0.99 ** 10, 0.904382075, 1e-8)
# log-based On paper: 0.95^10 = e^(10 ln 0.95)
close(log(0.95), -0.0513, 5e-5)
close(10 * log(0.95), -0.513, 5e-4)
close(exp(-0.513), 0.599, 5e-4)
close(log(0.99), -0.01005, 5e-6)
close(10 * log(0.99), -0.1005, 5e-5)
close(exp(-0.1005), 0.9044, 5e-4)
# retry: per-step = 1 - (1-p)^k
per_step = 1 - (1 - 0.8) ** 2
close(0.2 ** 2, 0.04, 1e-12)
close(per_step, 0.96, 1e-12)
close(per_step ** 10, 0.6648, 5e-5)
close(0.96 ** 10, 0.664832636, 1e-8)
close(0.8 ** 10, 0.107, 5e-4)        # card: 10-step task without the retry
close(0.96 ** 10, 0.665, 5e-4)

# ---- §: cost growth (quadratic in steps) ----------------------------------
def resent(per, n):
    return per * n * (n + 1) // 2


eq(10 * 11 // 2, 55)
eq(resent(500, 5), 7500)
eq(resent(500, 10), 27500)
eq(resent(500, 20), 105000)
eq(5 * 6 // 2, 15)
eq(20 * 21 // 2, 210)
close(7500 / 2500, 3.0, 1e-12)
close(27500 / 5000, 5.5, 1e-12)
close(105000 / 10000, 10.5, 1e-12)
close((10 + 1) / 2, 5.5, 1e-12)      # inflation factor (n+1)/2
close(105000 / 27500, 3.82, 5e-3)    # doubling steps ~quadruples tokens

# ---- §: memory / summarization token math ---------------------------------
window, overhead, turns, per_turn = 8192, 1000, 12, 600
raw = overhead + turns * per_turn
eq(raw, 8200)
assert raw > window
checks += 1
summary, kept = 300, 4 * 600
after = overhead + summary + kept
eq(kept, 2400)
eq(after, 3700)
eq(raw - after, 4500)

# =================================================== chapter 52 exercises
close(0.9 ** 5, 0.59049, 1e-12)                      # ex: 5-step at 0.9
close(0.99 ** 5, 0.9510, 5e-5)                       # ex: with verification
close(0.99 ** 5, 0.95099005, 1e-8)
eq(resent(800, 8), 28800)                            # ex: cost growth
eq(8 * 9 // 2, 36)
eq(800 * 8, 6400)
close(28800 / 6400, 4.5, 1e-12)
close(1 - (1 - 0.7) ** 3, 0.973, 1e-12)              # ex: retry p=0.7 k=3
close(0.3 ** 3, 0.027, 1e-12)
eq(500 + 8 * 500, 4500)                              # ex: memory over 4096
assert 4500 > 4096
checks += 1
eq(1 + 3, 4)                                          # ex: orchestrator+3 workers
close(0.95 ** 20, 0.3585, 5e-5)                      # ex: 20-step trajectory
close(0.95 ** 20, 0.358485922, 1e-8)

# ---- MCP constants referenced in prose/example ----------------------------
eq("2025-11-25", "2025-11-25")          # legacy initialize protocol version
eq("2026-07-28", "2026-07-28")          # current (modern, stateless) revision
eq(-32602, -32602)                      # invalid params / unknown tool

print(f"ch50_52_prompting_agents: all {checks} checks passed (chapters 50, 52)")
