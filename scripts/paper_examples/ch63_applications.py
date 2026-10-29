#!/usr/bin/env python3
"""Check every number in the "On paper" cards, worked examples and exercise
answers of docs/63-ai-agent-applications.md (stdlib only)."""
from math import sqrt, comb

checks = 0


def close(a, b, tol):
    global checks
    checks += 1
    assert abs(a - b) <= tol, f"{a!r} != {b!r} (tol {tol})"


def eq(a, b):
    global checks
    checks += 1
    assert a == b, f"{a!r} != {b!r}"


def cost(ti, to, pin=0.5, pout=1.5):
    return ti / 1e6 * pin, to / 1e6 * pout


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c, m, c - m, c + m


def passk(n, c, k):
    return 1 - comb(n - c, k) / comb(n, k)


# ---------------------------------------------------------------- §1 levers
# cascade: E = c_small + f*c_large
eq(0.3 * 0.004, 0.0012)
close(0.0005 + 0.3 * 0.004, 0.0017, 1e-12)
close((0.004 - 0.0017) / 0.004, 0.575, 1e-9)        # 57% cut (also Exercise 17.7)
# caching: uncached vs full-price billed (prefix paid once, T_new every step)
eq(4 * (1200 + 700), 7600)
eq(1200 + 4 * 700, 4000)
eq(7600 - 4000, 3600)
eq((4 - 1) * 1200, 3600)                            # saving = (s-1) * T_prefix
close(3600 / 7600, 0.47, 5e-3)                      # 47% of the input bill

# ---------------------------------------------------------------- §2 support
c, m, lo, hi = wilson(150, 200)
close(150 / 200, 0.75, 1e-12)
close(c, 0.7453, 5e-5)
close(m, 0.0596, 5e-5)
close(lo, 0.686, 5e-4)
close(hi, 0.805, 5e-4)
close(0.7453 - 0.0596, 0.686, 5e-4)                 # as displayed (pencil check)
close(0.7453 + 0.0596, 0.805, 5e-4)
close(m, 0.060, 1e-3)                               # Exercise 17.1 "near 0.060"
ci, co = cost(4000, 1000)
close(ci, 0.00200, 1e-12)
close(co, 0.00150, 1e-12)
close(ci + co, 0.00350, 1e-12)
# refund gate limit
close(50.0, 50.0, 0)

# ---------------------------------------------------------------- §3 coding CI
close(passk(5, 3, 1), 0.6, 1e-12)
eq(3 / 5, 0.6)
close(passk(5, 3, 2), 0.9, 1e-12)
eq(comb(5, 2), 10)
eq(comb(2, 2), 1)
close(1 - 1 / 10, 0.9, 1e-12)
ci, co = cost(12000, 1500)
close(ci, 0.00600, 1e-12)
close(co, 0.00225, 1e-12)
close(ci + co, 0.00825, 1e-12)

# ---------------------------------------------------------------- §4 data-analyst
c, m, lo, hi = wilson(72, 80)
close(72 / 80, 0.90, 1e-12)
close(c, 0.8817, 5e-5)
close(m, 0.0668, 5e-5)
close(lo, 0.815, 5e-4)
close(hi, 0.948, 5e-4)
close(0.8817 - 0.0668, 0.815, 5e-4)                 # as displayed
ci, co = cost(6000, 800)
close(ci, 0.00300, 1e-12)
close(co, 0.00120, 1e-12)
close(ci + co, 0.00420, 1e-12)

# ---------------------------------------------------------------- §5 on-call
close(4 / 50, 0.08, 1e-12)
c, m, lo, hi = wilson(0, 120)
close(hi, 0.031, 1e-3)                              # upper bound ~0.031
ci, co = cost(20000, 1000)
close(ci, 0.01000, 1e-12)
close(co, 0.00150, 1e-12)
close(ci + co, 0.01150, 1e-12)

# ---------------------------------------------------------------- §6 security
close(passk(5, 2, 1), 0.4, 1e-12)
eq(2 / 5, 0.4)
close(passk(5, 2, 2), 0.7, 1e-12)
eq(comb(3, 2), 3)
close(1 - 3 / 10, 0.7, 1e-12)
ci, co = cost(7000, 1000)
close(ci, 0.00350, 1e-12)
close(co, 0.00150, 1e-12)
close(ci + co, 0.00500, 1e-12)

# ---------------------------------------------------------------- §7 document
prec = 90 / (90 + 10)
rec = 90 / (90 + 20)
f1 = 2 * prec * rec / (prec + rec)
close(prec, 0.90, 1e-12)
close(rec, 0.818, 5e-4)
close(f1, 0.857, 5e-4)
close(2 * 0.90 * 0.818 / (0.90 + 0.818), 0.857, 5e-4)   # as displayed, rounded inputs
close(2 * 0.90 * 0.818, 1.472, 5e-4)                    # Exercise 17.4 intermediate
close(0.90 + 0.818, 1.718, 1e-12)
close(1.472 / 1.718, 0.857, 5e-4)
# confidence router (slice C): worst-field routing
eq(min(0.99, 0.97, 0.95) >= 0.90, True)
eq(0.90 > min(0.98, 0.82, 0.91) >= 0.70, True)
eq(min(0.66, 0.80, 0.72) < 0.70, True)
ci, co = cost(3000, 600)
close(ci, 0.00150, 1e-12)
close(co, 0.00090, 1e-12)
close(ci + co, 0.00240, 1e-12)

# ---------------------------------------------------------------- §8 research
close(47 / 50, 0.94, 1e-12)
ci, co = cost(10000, 2000)
close(ci, 0.00500, 1e-12)
close(co, 0.00300, 1e-12)
close(ci + co, 0.00800, 1e-12)

# ---------------------------------------------------------------- §9 sales/CRM
c, m, lo, hi = wilson(180, 200)
close(180 / 200, 0.90, 1e-12)
close(c, 0.8925, 5e-5)
close(m, 0.0419, 5e-5)
close(lo, 0.851, 5e-4)
close(hi, 0.934, 5e-4)
close(0.8925 - 0.0419, 0.851, 5e-4)                 # as displayed
close(0.8925 + 0.0419, 0.934, 5e-4)
ci, co = cost(5000, 500)
close(ci, 0.00250, 1e-12)
close(co, 0.00075, 1e-12)
close(ci + co, 0.00325, 1e-12)

# ---------------------------------------------------------------- §10 e-commerce
close(0.95 ** 4, 0.8145, 5e-5)
close(12 / 100, 0.12, 1e-12)
ci, co = cost(2500, 500)
close(ci, 0.00125, 1e-12)
close(co, 0.00075, 1e-12)
close(ci + co, 0.00200, 1e-12)

# ---------------------------------------------------------------- §11 IT/HR
c, m, lo, hi = wilson(228, 240)
close(228 / 240, 0.95, 1e-12)
close(c, 0.9429, 5e-5)
close(m, 0.0283, 5e-5)
close(lo, 0.915, 5e-4)
close(hi, 0.971, 5e-4)
close(0.9429 - 0.0283, 0.915, 5e-4)                 # as displayed
close(0.9429 + 0.0283, 0.971, 5e-4)
ci, co = cost(2000, 500)
close(ci, 0.00100, 1e-12)
close(co, 0.00075, 1e-12)
close(ci + co, 0.00175, 1e-12)

# ---------------------------------------------------------------- §12 data-pipeline
close(0.97 ** 5, 0.8587, 5e-5)
ci, co = cost(15000, 3000)
close(ci, 0.00750, 1e-12)
close(co, 0.00450, 1e-12)
close(ci + co, 0.01200, 1e-12)

# ---------------------------------------------------------------- §13 QA/browser
close(0.95 ** 3, 0.8574, 5e-5)
ci, co = cost(8000, 400)
close(ci, 0.00400, 1e-12)
close(co, 0.00060, 1e-12)
close(ci + co, 0.00460, 1e-12)

print(f"ch63_applications: all {checks} checks passed")
