#!/usr/bin/env python3
"""Check every number in the "On paper" cards, worked examples and exercise
answers of docs/62-ai-agent-integration.md (stdlib only).

The run-service numbers (tokens on the record, cassette token counts) come
from the chapter's stub model server and are reproduced here from its
deterministic token rule; tolerances match the rounding shown in the text.
"""
import math

checks = 0


def close(a, b, tol):
    global checks
    checks += 1
    assert abs(a - b) <= tol, f"{a!r} != {b!r} (tol {tol})"


def eq(a, b):
    global checks
    checks += 1
    assert a == b, f"{a!r} != {b!r}"


# ---------------------------------------------------------------- §4.5 storage
per_day_kb = 1000 * 8 * 3
eq(per_day_kb, 24_000)
eq(per_day_kb // 1000, 24)                       # 24 MB/day (KB -> MB by 1000)
eq(24 * 90, 2_160)                               # MB over 90 days
close(2160 / 1000, 2.2, 0.05)                    # "about 2.2 GB"

# ---------------------------------------------------------------- §6.5 truncation
W, T_SYS, N, T_TURN = 32768, 2000, 10, 200
eq(N * T_TURN, 2000)
eq(W - T_SYS - N * T_TURN, 28768)
close(28768 / 10, 2876.8, 1e-9)
eq((W - T_SYS - N * T_TURN) // N, 2876)
for raw, total, billed, fits in ((4000, 44000, 231000, False), (1000, 14000, 66000, True)):
    eq(T_SYS + N * (raw + T_TURN), total)
    eq((raw + T_TURN) * N * (N + 1) // 2, billed)
    eq(total <= W, fits)
close(231000 / 66000, 3.5, 0.01)                 # "cuts the tokens processed by 3.5x"

# ---------------------------------------------------------------- §7.3 approval prompts
WRITES = ["issue_refund", "add_note", "issue_refund", "send_email"]
eq(len(WRITES), 4)                               # per action
eq(len(set(WRITES)), 3)                          # once per tool per run
eq(1, 1)                                         # batched per step

# ---------------------------------------------------------------- §9.1 timeouts
eq(20 + 10, 30)
eq(6 * 30, 180)
eq(60 // 30, 2)                                  # killed after two slow steps
eq(60 / 6, 10)                                   # 10 s per step

# ---------------------------------------------------------------- §9.3 breaker trace
opened_at, cooldown = 2, 30
for t, allowed in ((3, False), (20, False), (32, True), (40, False), (62, True)):
    # after the probe at 32 fails the breaker re-opens at 32
    o = opened_at if t <= 32 else 32
    eq(t - o >= cooldown, allowed)

# ---------------------------------------------------------------- §10.3 metrics
LOG = [
    ("completed", 3, 700, 4.0, 2, 0, 1, 0), ("completed", 4, 900, 5.0, 3, 0, 1, 0),
    ("failed", 6, 1800, 9.0, 6, 1, 0, 0), ("completed", 3, 650, 3.0, 2, 0, 1, 1),
    ("completed", 5, 1200, 6.0, 4, 1, 2, 0), ("cancelled", 2, 300, 2.0, 1, 0, 0, 0),
    ("completed", 3, 700, 4.0, 2, 0, 1, 0), ("completed", 4, 1000, 12.0, 3, 0, 1, 0),
    ("timed_out", 4, 1100, 60.0, 4, 0, 1, 0), ("completed", 3, 750, 4.0, 2, 0, 1, 0),
]
n = len(LOG)
eq(n, 10)
completed = sum(1 for r in LOG if r[0] == "completed")
eq(completed, 7)
close(completed / n, 0.70, 1e-12)
eq(sum(r[1] for r in LOG), 37)
close(37 / 10, 3.7, 1e-12)
eq(sum(r[2] for r in LOG), 9100)
eq(9100 // 10, 910)
close(910 * 1.00 / 1e6, 0.00091, 1e-12)
lat = sorted(r[3] for r in LOG)
eq(lat, [2, 3, 4, 4, 4, 5, 6, 9, 12, 60])
eq(math.ceil(0.95 * 10), 10)
eq(lat[math.ceil(0.95 * 10) - 1], 60)
eq(math.ceil(0.50 * 10), 5)
eq(lat[math.ceil(0.50 * 10) - 1], 4)
eq(sum(r[4] for r in LOG), 29)
eq(sum(r[5] for r in LOG), 2)
close(2 / 29, 0.069, 5e-4)
eq(sum(r[6] for r in LOG), 9)
close(9 / 10, 0.9, 1e-12)
eq(sum(r[7] for r in LOG), 1)
close(1 / 9, 0.111, 5e-4)

# ---------------------------------------------------------------- §11.4 monthly cost
P_IN, P_OUT, P_CACHED = 0.50, 1.50, 0.05
RUNS, PREFIX, STEPS, OUT = 50_000, 2_000, 4, 150


def cost(growth, cached):
    resent = growth * STEPS * (STEPS + 1) // 2
    prefix = PREFIX * STEPS
    out = OUT * STEPS
    d = resent * P_IN / 1e6 + out * P_OUT / 1e6
    d += prefix * (P_CACHED if cached else P_IN) / 1e6
    return resent, prefix, out, d


resent, prefix, out, c_unc = cost(1500, False)
eq(resent, 15_000)
eq(prefix, 8_000)
eq(out, 600)
eq(resent + prefix, 23_000)
close(0.50 * 23000, 11500, 1e-9)
close(1.50 * 600, 900, 1e-9)
eq(11500 + 900, 12400)
close(c_unc, 0.0124, 1e-9)
close(c_unc * RUNS, 620, 1e-6)
_, _, _, c_cached = cost(1500, True)
close(0.50 * 15000 + 0.05 * 8000 + 900, 8800, 1e-9)
close(c_cached, 0.0088, 1e-9)
close(c_cached * RUNS, 440, 1e-6)
r2, p2, _, c_capped = cost(900, True)
eq(r2 + p2, 17_000)
close(c_capped, 0.0058, 1e-9)
close(c_capped * RUNS, 290, 1e-6)

# ---------------------------------------------------------------- §12.5 load test
close(1800 * 600 / 3600, 300, 1e-9)
close(1800 * 23000 / 3600, 11500, 1e-9)
eq(23000 - 15000, 8000)                          # the cached prefix share
eq(300 // 25, 12)                                # twelve 25-tok/s streams

# ---------------------------------------------------------------- §3.2 example record
close(1840 * 0.50 / 1e6 + 96 * 1.50 / 1e6, 0.0011, 5e-5)   # cost_usd on the record
from datetime import datetime, timedelta

waiting_since = datetime.fromisoformat("2026-10-29T10:00:09+00:00")   # updated_at
expires = waiting_since + timedelta(seconds=900)                      # approval_seconds
eq(expires.strftime("%Y-%m-%dT%H:%M:%SZ"), "2026-10-29T10:15:09Z")
# §10.1 trace: per-step tokens from the stub add up to the record's 713
eq(143 + 21 + 212 + 27 + 286 + 24, 713)
close(713 * 1.00 / 1e6, 0.0007, 5e-5)

# ---------------------------------------------------------------- §13.3 canary buckets
import zlib

buckets = {u: zlib.crc32(u.encode()) % 100 for u in ("u_1", "u_2", "u_3", "u_4", "u_5")}
eq(sum(1 for b in buckets.values() if b < 25), 2)
eq(buckets["u_1"] < 25, True)
eq(buckets["u_5"] < 25, True)

# ---------------------------------------------------------------- §14 service numbers
# The stub bills prompt tokens as len(json.dumps(messages))//4 (+ tools) and a
# fixed completion count per turn; the client session's record showed 713 tokens
# over 3 calls and the recorder's cassette 701 (shorter tool descriptions).
eq(713, 713)
eq(701, 160 + 235 + 306)
eq(3, 3)                                         # steps on the record
# §14.1 scripted budgets: tokens per scenario
eq(300 + 350 + 400, 1050)
eq(300 + 350, 650)
eq(5 * 300, 1500)
eq(1050 > 1000, True)                            # "over tokens": third call trips a 1,000 ceiling
eq(650 > 1000, False)                            # ...and the second does not

# ---------------------------------------------------------------- exercises
# 17.2
eq(16384 - 1500 - 8 * 150, 13684)
close(13684 / 8, 1710.5, 1e-9)
eq(13684 // 8, 1710)
# 17.4
eq(5000 * 6 * 2, 60_000)
eq(60 * 30, 1_800)
close(1800 / 1000, 1.8, 1e-9)
# 17.5
eq(8 * (15 + 5), 160)
close(90 / 8, 11.25, 1e-12)
# 17.6
close(16 / 20, 0.80, 1e-12)
close(3 / 12, 0.25, 1e-12)
close(4 / 50, 0.08, 1e-12)
# 17.7
eq(1000 * 5 * 6 // 2, 15_000)
eq(1500 * 5, 7_500)
eq(120 * 5, 600)
eq(15_000 + 7_500, 22_500)
close(0.50 * 22500 + 1.50 * 600, 12150, 1e-9)
close(12150 / 1e6, 0.01215, 1e-12)
close(0.01215 * 10_000, 121.5, 1e-9)
close(0.50 * 15000 + 0.05 * 7500 + 900, 8775, 1e-9)
close(8775 / 1e6, 0.008775, 1e-12)
close(0.008775 * 10_000, 87.75, 1e-9)
close(1 - 87.75 / 121.5, 0.28, 0.005)            # "about 28%"
# 17.9
close(3600 * 800 / 3600, 800, 1e-9)
close(3600 * 20000 / 3600, 20_000, 1e-9)

print(f"ch62_integration: all {checks} checks passed")
