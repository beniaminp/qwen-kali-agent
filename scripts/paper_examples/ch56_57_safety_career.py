#!/usr/bin/env python3
"""Check every number in the "On paper" cards, worked examples and exercise
answers of chapters 56 (safety) and 57 (career). Stdlib only.

Same pattern as scripts/paper_examples/ch39_42_distillation.py:
close(a, b, tol) / eq(a, b) count checks; a final line prints the total.
"""
from math import sqrt

checks = 0


def close(a, b, tol):
    global checks
    checks += 1
    assert abs(a - b) <= tol, f"{a!r} != {b!r} (tol {tol})"


def eq(a, b):
    global checks
    checks += 1
    assert a == b, f"{a!r} != {b!r}"


def luhn_ok(number: str) -> bool:
    total, parity = 0, len(number) % 2
    for i, ch in enumerate(number):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ================================================================ chapter 56
# ---- §8 PII: Luhn check on a test card number, computed by hand
# 4242 4242 4242 4242: from the right the eight 4s are doubled to 8 each,
# the eight 2s are left alone.
eq(len("4242424242424242"), 16)
eq("4242424242424242".count("2"), 8)
eq("4242424242424242".count("4"), 8)
eq(4 * 2, 8)                      # doubling a 4 gives 8 (<= 9, no subtraction)
eq(8 * 8, 64)                     # eight doubled 4s
eq(8 * 2, 16)                     # eight untouched 2s
eq(64 + 16, 80)                   # Luhn total
eq(80 % 10, 0)                    # divisible by 10 -> passes Luhn
eq(luhn_ok("4242424242424242"), True)

# near miss: flip the last digit 2 -> 1. The last digit is not doubled, so the
# untouched sum drops by 1 to 15; the total is 79 and Luhn fails.
eq(16 - 2 + 1, 15)
eq(64 + 15, 79)
eq(79 % 10, 9)
eq(luhn_ok("4242424242424241"), False)

# ---- §11 bias and fairness on two 100-person confusion matrices
# Group A: Y=1 -> 60, Y=0 -> 40.  TP=48 FN=12 FP=2 TN=38.
TPa, FNa, FPa, TNa = 48, 12, 2, 38
eq(TPa + FNa, 60)                 # positives in group A
eq(FPa + TNa, 40)                 # negatives in group A
eq(TPa + FNa + FPa + TNa, 100)
selA = (TPa + FPa) / 100
close(selA, 0.5, 1e-12)           # selection (positive-prediction) rate A
close(TPa / (TPa + FNa), 0.8, 1e-12)   # TPR_A (recall on Y=1)
close(FPa / (FPa + TNa), 0.05, 1e-12)  # FPR_A
close((TPa + TNa) / 100, 0.86, 1e-12)  # accuracy A

# Group B: Y=1 -> 40, Y=0 -> 60.  TP=24 FN=16 FP=6 TN=54.
TPb, FNb, FPb, TNb = 24, 16, 6, 54
eq(TPb + FNb, 40)
eq(FPb + TNb, 60)
eq(TPb + FNb + FPb + TNb, 100)
selB = (TPb + FPb) / 100
close(selB, 0.3, 1e-12)
close(TPb / (TPb + FNb), 0.6, 1e-12)   # TPR_B
close(FPb / (FPb + TNb), 0.1, 1e-12)   # FPR_B
close((TPb + TNb) / 100, 0.78, 1e-12)  # accuracy B

# Demographic parity difference and equal opportunity difference.
close(selA - selB, 0.2, 1e-12)                     # DPD
close(TPa / (TPa + FNa) - TPb / (TPb + FNb), 0.2, 1e-12)  # EOD

# §11.3: selection rate = TPR * base + FPR * (1 - base); base rates 0.6 vs 0.4.
close(0.8 * 0.6 + 0.05 * 0.4, selA, 1e-12)
close(0.6 * 0.4 + 0.1 * 0.6, selB, 1e-12)
# With unequal base rates both gaps CAN be zero, but only with FPRs apart:
# TPR 0.8 for both, FPR_A 0.05, FPR_B 0.3 -> both select 50%.
close(0.8 * 0.6 + 0.05 * 0.4, 0.5, 1e-12)
close(0.8 * 0.4 + 0.3 * 0.6, 0.5, 1e-12)
# Equal TPR *and* equal FPR with equal selection needs (TPR - FPR)(bA - bB) = 0,
# i.e. TPR = FPR (a classifier no better than chance) when bA != bB.
tpr_, fpr_ = 0.8, 0.05
close((tpr_ * 0.6 + fpr_ * 0.4) - (tpr_ * 0.4 + fpr_ * 0.6),
      (tpr_ - fpr_) * (0.6 - 0.4), 1e-12)

# ---- §6 excessive agency: compounding chance of a wrong autonomous write
# 10 independent write actions, each 2% chance of a wrong action.
close(0.98 ** 2, 0.9604, 1e-12)
close(0.98 ** 10, 0.8171, 5e-4)
close(1 - 0.98 ** 10, 0.1829, 5e-4)    # ~18% chance of at least one wrong write

# ---- §10 hallucination: groundedness and abstention, measured
close(34 / 40, 0.85, 1e-12)            # supported claims / total claims
close(20 / 25, 0.8, 1e-12)             # abstentions on unanswerable / unanswerable

# ---- §13 red-team harness: attack-success rate before/after defenses
close(4 / 4, 1.0, 1e-12)               # naive app: all 4 canary attacks land
close(0 / 4, 0.0, 1e-12)               # defended app: none land
close(4 / 4 - 0 / 4, 1.0, 1e-12)       # absolute risk reduction
# 95% Wilson upper bound for 0 successes in n trials: z^2 / (n + z^2)
z = 1.96
close(z ** 2, 3.84, 5e-3)
close(4 + z ** 2, 7.84, 5e-3)
close(z ** 2 / (4 + z ** 2), 0.49, 5e-3)    # 0 of 4 is compatible with ASR 0.49

# ================================================================ chapter 57
# ---- §7 system design: support assistant for 2,000,000 tickets/month
SEC_PER_MONTH = 30 * 24 * 3600
eq(SEC_PER_MONTH, 2_592_000)
close(2_000_000 / SEC_PER_MONTH, 0.772, 5e-4)       # average requests/second
close(5 * 2_000_000 / SEC_PER_MONTH, 3.86, 5e-3)    # 5x peak factor
peak = 5 * 2_000_000 / SEC_PER_MONTH
close(peak * 300, 1160, 5)                          # output tokens/s at peak
close(peak * 6.5, 25, 0.5)                          # Little's law: in flight
eq(peak * 300 < 2000, True)                         # fits one 2,000 tok/s GPU
eq(peak * 300 / 17 > 60, True)                      # vs 15-17 tok/s M1 Pro stream

# Cost per ticket at assumed $3 / 1M input, $15 / 1M output tokens.
in_cost = 1500 * 3 / 1_000_000
out_cost = 300 * 15 / 1_000_000
close(in_cost, 0.0045, 1e-12)
close(out_cost, 0.0045, 1e-12)
close(in_cost, out_cost, 1e-12)                     # input and output tie
eq(1500 / 300, 5.0)                                 # 5x the input tokens ...
eq(15 / 3, 5.0)                                     # ... at 1/5 the price
close(in_cost + out_cost, 0.009, 1e-12)             # $ per ticket
close(2_000_000 * (in_cost + out_cost), 18_000, 1e-9)  # $ per month

# Vector index memory: 200,000 chunks x 1024 dims x 4 bytes (float32).
eq(200_000 * 1024 * 4, 819_200_000)
close(200_000 * 1024 * 4 / 1e6, 819.2, 1e-9)        # MB
close(200_000 * 1024 * 4 / 1e9, 0.8192, 1e-12)      # GB
close(200_000 * 1024 * 1 / 1e6, 204.8, 1e-9)        # int8: 1 byte/dim

# Latency budget: TTFT 0.5 s + 300 output tokens at 50 tok/s.
close(300 / 50, 6.0, 1e-12)
close(0.5 + 300 / 50, 6.5, 1e-12)                   # total non-streaming seconds

# ---- interview exercise: precision / recall / F1
# 50 flagged urgent, 40 truly urgent among them, 60 urgent tickets total.
P = 40 / 50
R = 40 / 60
close(P, 0.8, 1e-12)
close(R, 0.667, 5e-4)
close(2 * P * R / (P + R), 0.727, 5e-4)             # F1

# ---- coding task: cosine top-k search, the worked pair
# a = (1,2,2) "refund my order", b = (2,1,2) "money back please"
a = (1, 2, 2)
b = (2, 1, 2)
dot = sum(x * y for x, y in zip(a, b))
eq(dot, 8)
close(sqrt(1 + 4 + 4), 3.0, 1e-12)
close(sqrt(4 + 1 + 4), 3.0, 1e-12)
close(dot / 9, 0.889, 5e-4)                         # cosine ~ 0.889

# ---- §5.7 interview answer: per-step reliability compounds
close(0.95 ** 10, 0.60, 5e-3)

# ---- §5.9 interview answer: LoRA rank 16 on all 7 projections of Qwen3-14B
# (d=5120, 40 layers, 8 KV heads x 128 = 1024, d_ff = 17408; see 02/32)
d, kv, ff, L, r = 5120, 1024, 17408, 40, 16
per_layer = r * ((d + d) + (d + kv) * 2 + (d + d) + (d + ff) * 2 + (ff + d))
close(per_layer * L / 1e6, 64.2, 0.1)               # "about 60M"
eq(per_layer * L / 14.8e9 < 0.01, True)             # well under 1%

# ---- §6.1 softmax([2, 1, 0])
from math import exp
Z = exp(2) + exp(1) + 1
close(exp(2) / Z, 0.6652, 5e-5)
close(exp(1) / Z, 0.2447, 5e-5)
close(1 / Z, 0.0900, 5e-5)

# ---- §6.2 top-p nucleus at p = 0.9, renormalized
close(0.5 + 0.25 + 0.15, 0.9, 1e-12)
close(0.5 / 0.9, 0.556, 5e-4)
close(0.25 / 0.9, 0.278, 5e-4)
close(0.15 / 0.9, 0.167, 5e-4)

# ---- §6.4 BPE: (e, s) ties with (s, t) and (t, </w>) at 9
from collections import Counter
bpe = {"l o w </w>": 5, "l o w e r </w>": 2,
       "n e w e s t </w>": 6, "w i d e s t </w>": 3}
pairs = Counter()
for w, f in bpe.items():
    s = w.split()
    for x, y in zip(s, s[1:]):
        pairs[(x, y)] += f
eq(pairs[("e", "s")], 9)
eq(pairs[("s", "t")], 9)
eq(pairs[("t", "</w>")], 9)
eq(max(pairs.values()), 9)
eq(pairs.most_common(1)[0][0], ("e", "s"))      # first occurrence wins the tie

# ---- §1 learning plan: rows cover 24 weeks; ~10 h/week
weeks = [1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1]
eq(sum(weeks), 24)
eq(24 * 10, 240)

# ================================================================ exercises
# ---- ch56 Exercise 20.2: Luhn on the 8-digit 4242 4242
eq("42424242".count("2"), 4)
eq("42424242".count("4"), 4)
eq(4 * 8, 32)                          # four doubled 4s
eq(4 * 2, 8)                           # four untouched 2s
eq(32 + 8, 40)
eq(40 % 10, 0)
eq(luhn_ok("42424242"), True)

# ---- ch56 Exercise 20.3: DPD for 60/100 vs 45/100
close(60 / 100, 0.6, 1e-12)
close(45 / 100, 0.45, 1e-12)
close(0.6 - 0.45, 0.15, 1e-12)

# ---- ch56 Exercise 20.4: 5 writes at 4% each
close(0.96 ** 5, 0.8154, 5e-4)
close(1 - 0.96 ** 5, 0.1846, 5e-4)

# ---- ch57 Exercise 14.2: cost at 500 in / 400 out tokens
close(500 * 3 / 1_000_000, 0.0015, 1e-12)
close(400 * 15 / 1_000_000, 0.006, 1e-12)
close(500 * 3 / 1_000_000 + 400 * 15 / 1_000_000, 0.0075, 1e-12)
close(2_000_000 * 0.0075, 15_000, 1e-9)

# ---- ch57 Exercise 14.3: nucleus at p = 0.7
cum = [0.5, 0.75, 0.90, 0.97, 1.0]
close(cum[0], 0.5, 1e-12)
close(cum[1], 0.75, 1e-12)
eq(cum[0] < 0.7 <= cum[1], True)       # smallest nucleus reaching 0.7 is first two

print(f"ch56_57_safety_career: all {checks} checks passed (chapters 56, 57)")
