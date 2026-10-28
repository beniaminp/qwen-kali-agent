#!/usr/bin/env python3
"""Check every number in docs/45-ai-engineer-path.md (stdlib only).

Covers the On paper cards, the worked triage project, the tables built from
counts, the time plan and the exercise answers.
"""
from fractions import Fraction as F
from math import exp, log, sqrt

checks = 0


def close(a, b, tol):
    global checks
    checks += 1
    assert abs(a - b) <= tol, f"{a!r} != {b!r} (tol {tol})"


def eq(a, b):
    global checks
    checks += 1
    assert a == b, f"{a!r} != {b!r}"


def ok(cond, msg="condition failed"):
    global checks
    checks += 1
    assert cond, msg


def softmax(z, t=1.0):
    w = [exp(v / t) for v in z]
    s = sum(w)
    return [x / s for x in w]


def cost(n_in, n_out, p_in, p_out):
    """Dollars for one request; prices in dollars per million tokens."""
    return (n_in * p_in + n_out * p_out) / 1e6


def cost_per_1000(n_in, n_out, p_in, p_out):
    return 1000 * cost(n_in, n_out, p_in, p_out)


def se(p, n):
    return sqrt(p * (1 - p) / n)


# ------------------------------------------------ §1.2 Tuesday: prompt caching
eq(1600 + 400, 2000)
eq(2000 * 3, 6000)
eq(300 * 15, 4500)
eq(6000 + 4500, 10500)
close(10500 / 1e6, 0.0105, 1e-15)
close(cost(2000, 300, 3, 15), 0.0105, 1e-12)
close(0.1 * 3, 0.3, 1e-12)                       # cached price = c * p_in
eq(400 * 3, 1200)
close(1600 * 0.3, 480, 1e-9)
close(1200 + 480 + 4500, 6180, 1e-9)
after = (400 * 3 + 1600 * 0.1 * 3 + 300 * 15) / 1e6
close(after, 0.00618, 1e-12)
saving = 1 - after / 0.0105
close(saving, 0.411, 5e-4)                       # "≈ 0.411", "41%"
close(saving * 100, 41, 0.5)
close(4500 / 6180 * 100, 73, 0.5)                # output share after caching

# ------------------------------------------------ §2.1 / §2.7 latency
close(48 / 16, 3, 1e-12)                         # table: 48 tokens at 16 tok/s = 3 s
eq(1 + 240 / 16, 16)
eq(1 + 48 / 16, 4)
eq(1000 * 4, 4000)
close(4000 / 60, 67, 0.5)                        # ≈ 67 min
eq(16 / 4, 4)                                    # "four times faster"
ok(15 <= 16 <= 17)                               # 16 tok/s inside 10 §8's 15–17

# §2.4 example trace record: ttft 2100 ms + 47 tokens at 16 tok/s ≈ 5040 ms
close(2100 + 47 / 16 * 1000, 5040, 5)
ok(1200 <= 1512)                                 # cached part of the input
close(6.96, 7, 0.05)                             # p99 "about 7 s"
ok(6.96 > 5)                                     # beyond a 5 s timeout

# §2.6 response cache
close(1 - 0.30, 0.70, 1e-12)

# ------------------------------------------------ §2.8 eval diff
eq(12 - 7, 5)
close(5 / 200 * 100, 2.5, 1e-12)

# ------------------------------------------------ §2.9 failure taxonomy (50 chats)
buckets = [21, 11, 9, 6, 3]
eq(sum(buckets), 50)
for c, pct in zip(buckets, (42, 22, 18, 12, 6)):
    close(c / 50 * 100, pct, 1e-9)

# ------------------------------------------------ §3.1 temperature
z = [2 * log(2), log(2), 0.0]
for got, want in zip([exp(v) for v in z], (4, 2, 1)):
    close(got, want, 1e-12)
p1 = softmax(z, 1.0)
for got, frac, shown in zip(p1, (F(4, 7), F(2, 7), F(1, 7)), (0.571, 0.286, 0.143)):
    close(got, float(frac), 1e-12)
    close(got, shown, 5e-4)
for got, want in zip([exp(2 * v) for v in z], (16, 4, 1)):
    close(got, want, 1e-9)
p05 = softmax(z, 0.5)
for got, frac, shown in zip(p05, (F(16, 21), F(4, 21), F(1, 21)), (0.762, 0.190, 0.048)):
    close(got, float(frac), 1e-12)
    close(got, shown, 5e-4)
eq(1 - F(4, 7), F(3, 7))                         # not billing 3 times in 7
eq(1 - F(16, 21), F(5, 21))                      # 5 in 21
close(21 / 5, 4, 0.25)                           # "about 1 in 4"
draws_T1 = "account billing billing billing account other account account".split()
eq(draws_T1.count("billing"), 3)                 # "won only 3 of 8 calls"

# ------------------------------------------------ §3.2 standard error
close(se(0.9, 100), 0.03, 1e-12)
close(0.9 * 0.1 / 100, 0.0009, 1e-15)
close(1.96 * 0.03, 0.059, 5e-4)
close(0.9 - 1.96 * 0.03, 0.841, 5e-4)
close(0.9 + 1.96 * 0.03, 0.959, 5e-4)
ok(0.841 < 0.92 < 0.959)                         # 92/100 is inside the interval
close(se(0.9, 400) / se(0.9, 100), 0.5, 1e-12)   # 4x items halves the interval

# ------------------------------------------------ §3.3 labels and mix
close(460 / 500, 0.92, 1e-12)
eq(500 - 460, 40)
close(0.95 * 0.92, 0.874, 1e-12)
close(0.05 * 0.70, 0.035, 1e-12)
gold_mix = 0.95 * 0.92 + 0.05 * 0.70
close(gold_mix, 0.909, 1e-12)
close(0.70 * 0.92, 0.644, 1e-12)
close(0.30 * 0.70, 0.210, 1e-12)
prod_mix = 0.70 * 0.92 + 0.30 * 0.70
close(prod_mix, 0.854, 1e-12)
close((gold_mix - prod_mix) * 100, 5.5, 1e-9)

# ------------------------------------------------ §3.5 model upgrade by slice
slices = {  # n, old correct, new correct, old acc, new acc (as printed)
    "billing": (80, 72, 76, 0.900, 0.950),
    "refunds": (45, 40, 36, 0.889, 0.800),
    "technical": (120, 102, 108, 0.850, 0.900),
    "account": (55, 48, 48, 0.873, 0.873),
}
for n, old, new, a_old, a_new in slices.values():
    close(old / n, a_old, 5e-4)
    close(new / n, a_new, 5e-4)
N = sum(v[0] for v in slices.values())
OLD = sum(v[1] for v in slices.values())
NEW = sum(v[2] for v in slices.values())
eq((N, OLD, NEW), (300, 262, 268))
close(OLD / N, 0.873, 5e-4)
close(NEW / N, 0.893, 5e-4)
close((NEW - OLD) / N * 100, 2, 1e-9)            # total up 2 points
close((40 / 45 - 36 / 45) * 100, 9, 0.2)         # refunds down 9 points

# ------------------------------------------------ §4 triage project
eq(200 + 300, 500)
# §4.4 baseline on test (300 tickets, 60 urgent)
tp, fp, fn, tn = 30, 20, 30, 220
eq(tp + fp + fn + tn, 300)
eq(tp + fn, 60)
eq(tp + fp, 50)
close(186 / 300, 0.62, 1e-12)
close(tp / (tp + fp), 0.60, 1e-12)
close(tp / (tp + fn), 0.50, 1e-12)
close(fp / (tp + fp), 0.4, 1e-12)                # four pages in ten are false alarms
f1_rules = 2 * 0.6 * 0.5 / (0.6 + 0.5)
close(0.6 + 0.5, 1.1, 1e-12)
close(f1_rules, 0.55, 5e-3)                      # 0.5454... prints as 0.55

# §4.6 prompt v1 on dev (200 tickets, 40 urgent)
close(160 / 200, 0.80, 1e-12)
eq(34 + 16, 50)
eq(34 + 6, 40)
close(34 / 50, 0.68, 1e-12)
close(34 / 40, 0.85, 1e-12)
eq(14 + 9 + 7 + 4 + 6, 40)
eq(200 - 160, 40)
eq(11 + 5, 16)
eq(160 + 6, 166)
close(166 / 200, 0.83, 1e-12)
close(6 / 200 * 100, 3, 1e-9)                    # label fix worth 3 points

# §4.7 prompt v2 on dev
close(182 / 200, 0.91, 1e-12)
close((182 - 166) / 200 * 100, 8, 1e-9)          # the prompt is worth 8 points
eq(37 + 5, 42)
eq(37 + 3, 40)
close(37 / 42, 0.881, 5e-4)
close(37 / 40, 0.925, 1e-12)
ok(182 / 200 >= 0.85 and 37 / 40 >= 0.85 and 37 / 42 >= 0.80)  # dev passes

# §4.7 prompt v2 on test, run once
tp, fp, fn, tn = 51, 9, 9, 231
eq(tp + fp + fn + tn, 300)
eq(tp + fn, 60)
eq(tp + fp, 60)
close(tp / (tp + fp), 0.85, 1e-12)
close(tp / (tp + fn), 0.85, 1e-12)
close(270 / 300, 0.90, 1e-12)
close(se(0.9, 300), 0.017, 5e-4)
close(1.96 * se(0.9, 300), 0.034, 5e-4)
ok(0.90 >= 0.85 and 0.85 >= 0.85 and 0.85 >= 0.80)  # test passes all three
close(37 / 42, 0.88, 5e-3)                       # "0.85 against 0.88"

# §4.8 cost per 1,000 tickets (assumed prices)
eq(1200 + 300, 1500)
eq(1500 * 3, 4500)
eq(50 * 15, 750)
eq(4500 + 750, 5250)
large = cost_per_1000(1500, 50, 3, 15)
close(large, 5.25, 1e-12)
close(4500 / 5250 * 100, 86, 0.5)
close(30 * large, 157.50, 1e-9)

# §4.9 cascade
close(1500 * 0.2, 300, 1e-9)
close(50 * 0.8, 40, 1e-9)
small = cost_per_1000(1500, 50, 0.2, 0.8)
close(small, 0.34, 1e-12)
q = 0.2
close(q * large, 1.05, 1e-12)
cascade = small + q * large
close(cascade, 1.39, 1e-12)
close(1 - cascade / large, 0.735, 5e-4)
close(round((1 - cascade / large) * 100), 74, 0)  # code prints 74%
close(cascade / large * 100, 26, 0.5)             # "26% of the cost", "about a quarter"
close(1 - small / large, 0.935, 5e-4)             # break-even escalation rate
acc = (1 - q) * 0.92 + q * 0.80
close(0.8 * 0.92, 0.736, 1e-12)
close(0.2 * 0.80, 0.160, 1e-12)
close(acc, 0.896, 1e-12)
ok(abs(0.90 - acc) < 1.96 * se(0.9, 300))         # inside the ±0.034 interval

# ------------------------------------------------ §7 job descriptions (illustrative)
# §7.2 RAG pipeline: recall@5 after each change on 200 questions
for hits, shown in ((156, 0.78), (172, 0.86), (184, 0.92), (190, 0.95)):
    close(hits / 200, shown, 1e-12)
eq(184 - 172, 12)                                 # reranking moved 12 more into the top 5
eq(200 - 184, 16)                                 # 16 questions still missed
ok(9 <= 16 and 7 <= 9)                            # 9 of the misses, 7 of those fixed
close(376 / 400, 0.94, 1e-12)                     # groundedness
close(46 / 50, 0.92, 1e-12)                       # judge agreement on calibration set
# §7.2 agent: 120 tasks x 3 runs
eq(120 * 3, 360)
close(300 / 360, 0.833, 5e-4)
close(85 / 120, 0.71, 5e-3)
close((300 / 360) ** 3, 0.58, 5e-3)
close(0.833 ** 3, 0.58, 5e-3)
ok(85 / 120 > (300 / 360) ** 3)                   # failures cluster
eq(120 - 85, 35)
per_call = (2000 * 3 + 150 * 15) / 1e6
close(per_call, 0.00825, 1e-12)
close(3 * per_call, 0.025, 5e-4)                  # about $0.025 per task
# §7.3 embeddings comparison on 200 queries
close(168 / 200, 0.84, 1e-12)
close(178 / 200, 0.89, 1e-12)
eq(14 - 4, 10)
eq(178 - 168, 10)
close(se(0.9, 300) * 1.96, 0.034, 5e-4)          # 0.90 ± 0.034 quoted again
# §7.4 data preparation funnel
eq(48000 - 6300, 41700)
eq(41700 - 2100, 39600)
eq(39600 - 1600, 38000)
close(8360 / 38000, 0.22, 1e-12)

# ------------------------------------------------ §8 self-assessment, §9 time plan
eq(2 + 5 + 4 + 8 + 5 + 7, 31)                     # items per group
weeks = [1, 4, 4, 5, 3, 3, 3, 3, 1]               # phases 0..8 (phase 8: week 27)
eq(sum(weeks), 27)
eq(1 + 4, 5)                                      # phase 1 ends in week 5
eq(1 + 4 + 4 + 5, 14)                             # phase 3 ends in week 14
eq(sum(weeks[:8]), 26)                            # phase 7 ends in week 26
eq(27 * 8, 216)
eq(27 * 10, 270)
ok(6 < 27 / (52 / 12) < 6.5)                      # a little over six months
ok(7 <= 216 / 30 <= 7.5 and 8.5 <= 270 / 30 <= 9)  # 7–9 weeks at 30 h/week

# ------------------------------------------------ §10 traps
close(0.8 ** 2, 0.64, 1e-12)
close(0.8 ** 4, 0.4096, 1e-12)
close(0.8 ** 5, 0.32768, 1e-12)
close(0.8 ** 5, 0.33, 5e-3)
close(0.55 ** 5, 0.050, 5e-4)
close(0.55 ** 5, 1 / 20, 1e-3)                    # one time in twenty
close(1 - 0.55, 0.45, 1e-12)
close(0.95 ** 2, 0.9025, 1e-12)
close(0.95 ** 4, 0.8145, 5e-5)
close(0.95 ** 8, 0.6634, 5e-5)
close(0.95 ** 8 * 100, 66, 0.5)
m, n = 400, 10
history = sum(k * m for k in range(1, n + 1))
eq(history, m * n * (n + 1) // 2)
eq(history, 22000)
eq(n * m, 4000)
close(history / (n * m), 5.5, 1e-12)
close(history * 3 / 1e6, 0.066, 1e-12)
close(n * m * 3 / 1e6, 0.012, 1e-12)
eq(20 * 21 // 2, 210)
eq(400 * 210, 84000)

# ------------------------------------------------ exercises
# 1
P, R = 30 / 40, 30 / 50
close(P, 0.75, 1e-12)
close(R, 0.60, 1e-12)
close(2 * P * R, 0.9, 1e-12)
close(P + R, 1.35, 1e-12)
close(2 * P * R / (P + R), 0.667, 5e-4)
close(10 / 40, 0.25, 1e-12)                        # one page in four
close(20 / 50, 0.40, 1e-12)                        # 40% missed
# 2
close(cost(1000, 100, 2, 8), 0.0028, 1e-12)
eq(1000 * 2 + 100 * 8, 2800)
close(cost_per_1000(1000, 100, 2, 8), 2.80, 1e-12)
close(50 * 2.80, 140, 1e-9)
close(800 / 2800 * 100, 29, 0.5)
close(100 / 1100 * 100, 9, 0.5)
eq(8 / 2, 4)
# 3
close(0.40 + 0.25 * 4.00, 1.40, 1e-12)
close((4.00 - 0.40) / 4.00, 0.9, 1e-12)
# 4
z = [log(3), 0.0]
for got, want in zip(softmax(z, 1.0), (0.75, 0.25)):
    close(got, want, 1e-12)
for got, want in zip([exp(2 * v) for v in z], (9, 1)):
    close(got, want, 1e-9)
for got, want in zip(softmax(z, 0.5), (0.9, 0.1)):
    close(got, want, 1e-12)
# 5
close(0.8 ** 10, 0.32768 ** 2, 1e-12)
close(0.8 ** 10, 0.107, 5e-4)
close(1 / 0.8 ** 10, 9, 0.5)                       # about one demo in nine
# 6
close(0.97 ** 2, 0.9409, 1e-12)
close(0.97 ** 3, 0.9127, 5e-5)
close(0.97 ** 6, 0.833, 5e-4)
ok(0.97 ** 6 < 0.93)
# 7
close(se(0.8, 400), 0.02, 1e-12)
close(0.8 * 0.2 / 400, 0.0004, 1e-15)
close(1.96 * 0.02, 0.039, 5e-4)
close(0.8 - 1.96 * 0.02, 0.761, 5e-4)
close(0.8 + 1.96 * 0.02, 0.839, 5e-4)
ok(0.761 < 0.82 < 0.839)
# 8
eq(round(250 * (1 - 0.80)), 50)
eq(20 + 12 + 10 + 8, 50)
close(0.80 + 10 / 250, 0.84, 1e-12)
close(12 / 250, 0.048, 1e-12)
close(20 / 250 * 100, 8, 1e-9)

print(f"ch45_path: all {checks} checks passed")
