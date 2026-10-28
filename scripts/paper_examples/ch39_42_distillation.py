#!/usr/bin/env python3
"""Check every number in the "On paper" cards of docs/39-42 (stdlib only)."""
from fractions import Fraction as F
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


def softmax(z, t=1.0):
    w = [exp(v / t) for v in z]
    s = sum(w)
    return [x / s for x in w]


def kl(p, q):
    return sum(a * log(a / b) for a, b in zip(p, q) if a > 0)


LN2, LN3, LN6, LN15 = log(2), log(3), log(6), log(1.5)
close(LN2, 0.693, 5e-4)
close(LN2, 0.6931, 5e-5)
close(LN15, 0.405, 5e-4)
close(LN15, 0.4055, 5e-5)

# ---------------------------------------------------------------- chapter 39
# §2 temperature softmax: z_T = (2 ln 6, 2 ln 2, 0), T = 2
zT = [2 * LN6, 2 * LN2, 0.0]
close(zT[0], 3.58, 5e-3)
close(zT[1], 1.39, 5e-3)
close(exp(zT[0] / 2), 6, 1e-12)
close(exp(zT[1] / 2), 2, 1e-12)
eq(6 + 2 + 1, 9)
p2 = softmax(zT, 2)
for got, frac, shown in zip(p2, (F(6, 9), F(2, 9), F(1, 9)), (0.667, 0.222, 0.111)):
    close(got, float(frac), 1e-12)
    close(got, shown, 5e-4)

# §2.2 same logits at T = 1, 2, 4 (weights squared / square-rooted)
eq([6 ** 2, 2 ** 2, 1], [36, 4, 1])
eq(36 + 4 + 1, 41)
close(sqrt(6), 2.449, 5e-4)
close(sqrt(2), 1.414, 5e-4)
close(2.449 + 1.414 + 1, 4.863, 1e-9)
close(sqrt(6) + sqrt(2) + 1, 4.863, 1e-3)  # table sums the rounded weights shown
for got, s in zip((2.449 / 4.863, 1.414 / 4.863, 1 / 4.863), (0.504, 0.291, 0.206)):
    close(got, s, 5e-4)  # same 3-decimal result from the rounded weights
for t, shown in ((1, (0.878, 0.098, 0.024)), (2, (0.667, 0.222, 0.111)), (4, (0.504, 0.291, 0.206))):
    for got, s in zip(softmax(zT, t), shown):
        close(got, s, 5e-4)
p1, p4 = softmax(zT, 1), softmax(zT, 4)
close(p1[2] * 100, 2.4, 0.05)
close(p4[2] * 100, 20.6, 0.05)
assert p4[2] / p1[2] > 8  # "more than eightfold"
checks += 1

# §2.1 Hinton objective: z_T = (2 ln 2, 0, 0), z_S = (0, 2 ln 2, 0), T = 2, alpha = 0.5, label = token 1
zT, zS, T, alpha = [2 * LN2, 0.0, 0.0], [0.0, 2 * LN2, 0.0], 2, 0.5
close(zT[0], 1.39, 5e-3)
pT, pS = softmax(zT, T), softmax(zS, T)
for a, b in zip(pT, (0.5, 0.25, 0.25)):
    close(a, b, 1e-12)
for a, b in zip(pS, (0.25, 0.5, 0.25)):
    close(a, b, 1e-12)
KL = kl(pT, pS)
close(KL, 0.5 * LN2 + 0.25 * log(0.5) + 0.25 * log(1), 1e-12)
close(KL, LN2 / 4, 1e-12)
close(KL, 0.1733, 5e-5)
close(KL, 0.173, 5e-4)
pS1 = softmax(zS, 1)
for a, b in zip(pS1, (1 / 6, 2 / 3, 1 / 6)):
    close(a, b, 1e-12)
CE = -log(pS1[0])
close(CE, LN6, 1e-12)
close(CE, 1.792, 5e-4)
close(0.5 * 1.792, 0.896, 1e-12)
close(0.5 * T ** 2 * 0.1733, 0.347, 5e-4)
close(0.896 + 0.347, 1.24, 5e-3)
L = (1 - alpha) * CE + alpha * T ** 2 * KL
close(L, 1.24, 5e-3)
close(T ** 2, 4, 0)

# §2.1 gradient: alpha*T*(pS^T - pT^T) + (1-alpha)*(pS - e_y)
eq(alpha * T, 1)
kd = [alpha * T * (s - t) for s, t in zip(pS, pT)]
for a, b in zip(kd, (-0.25, 0.25, 0.0)):
    close(a, b, 1e-12)
ce = [(1 - alpha) * (s - e) for s, e in zip(pS1, (1, 0, 0))]
for a, b in zip(ce, (-5 / 12, 1 / 3, 1 / 12)):
    close(a, b, 1e-12)
g = [a + b for a, b in zip(kd, ce)]
for a, frac, shown in zip(g, (F(-2, 3), F(7, 12), F(1, 12)), (-0.667, 0.583, 0.083)):
    close(a, float(frac), 1e-12)
    close(a, shown, 5e-4)
close(sum(g), 0, 1e-12)
eq(F(-1, 4) + F(-5, 12), F(-2, 3))
eq(F(1, 4) + F(1, 3), F(7, 12))

# §2.3 sequence KD: p_S = 1/2, 1/4, 1/2, 1/2
Lseq = -sum(log(p) for p in (0.5, 0.25, 0.5, 0.5))
eq(1 + 2 + 1 + 1, 5)
close(Lseq, 5 * LN2, 1e-12)
close(Lseq, 3.466, 5e-4)

# §2.4 on-policy: p_S = 1/2, 1/16, 1/2
Lon = -sum(log(p) for p in (0.5, 1 / 16, 0.5))
close(Lon, 6 * LN2, 1e-12)
close(Lon, 4.159, 5e-4)
close(log(16), 2.773, 5e-4)
close(log(16) / Lon, 2 / 3, 1e-12)

# §2.5 feature KD: W = [[1,0,1],[0,1,0]]
W = [[1, 0, 1], [0, 1, 0]]


def mv(m, v):
    return [sum(a * b for a, b in zip(row, v)) for row in m]


def sq(v):
    return sum(x * x for x in v)


hTa, hSa, hTb, hSb = [1, 2, 1], [1, 2], [0, 1, 3], [2, 3]
eq(mv(W, hTa), [2, 2])
eq(mv(W, hTb), [3, 1])
da = [s - t for s, t in zip(hSa, mv(W, hTa))]
db = [s - t for s, t in zip(hSb, mv(W, hTb))]
eq(da, [-1, 0])
eq(db, [-1, 2])
eq(sq(da), 1)
eq(sq(db), 1 + 4)
eq(sq(da) + sq(db), 6)
eq(F(sq(db), sq(da) + sq(db)), F(5, 6))

# ---------------------------------------------------------------- chapter 40
# No cards added (all display blocks are already numeric or followed by one).
# Existing numbers re-checked for the report:
eq(300 * 8 * 2048, 4_915_200)
close(5e6 * 64 * 8 / 1e9, 2.56, 1e-12)
close(5e6 * 256 * 8 / 1e9, 10.24, 1e-12)
close(5e6 * 262_144 * 4 / 1e12, 5.24, 5e-3)

# ---------------------------------------------------------------- chapter 41
# §2.1 CE + gradient: |C| = 2, correct-token probs 0.5 and 0.25
CE = -(log(0.5) + log(0.25)) / 2
close(0.693 + 1.386, 2.079, 1e-12)
close(CE, 1.5 * LN2, 1e-12)
close(CE, 1.04, 5e-3)
grad = [p - e for p, e in zip((0.5, 0.25, 0.25), (0, 1, 0))]
eq(grad, [0.5, -0.75, 0.25])
eq(sum(grad), 0)

# §2.2 KD + total: z_T = (2 ln 3, 0), z_S = (0, 0), T = 2, alpha = 0.6, label = token 1
zT, zS, T, alpha = [2 * LN3, 0.0], [0.0, 0.0], 2, 0.6
close(zT[0], 2.197, 5e-4)
close(exp(zT[0] / T), 3, 1e-12)
pT, pS = softmax(zT, T), softmax(zS, T)
close(pT[0], 0.75, 1e-12)
close(pS[0], 0.5, 1e-12)
KL = kl(pT, pS)
close(KL, 0.75 * LN15 + 0.25 * log(0.5), 1e-12)
close(0.75 * 0.4055, 0.3041, 5e-5)
close(0.25 * 0.6931, 0.1733, 5e-5)
close(0.3041 - 0.1733, 0.1308, 1e-12)
close(KL, 0.1308, 5e-5)
LKD = T ** 2 * KL
close(4 * 0.1308, 0.523, 5e-4)
close(LKD, 0.523, 5e-4)
CE = -log(softmax(zS, 1)[0])
close(CE, LN2, 1e-12)
close(0.4 * 0.693, 0.277, 5e-4)
close(0.6 * 0.523, 0.314, 5e-4)
close(0.277 + 0.314, 0.591, 1e-12)
L = (1 - alpha) * CE + alpha * LKD
close(L, 0.591, 5e-4)
close(alpha * LKD / L * 100, 53, 0.5)
close(alpha * KL, 0.078, 5e-4)  # without T^2
close(alpha * KL / ((1 - alpha) * CE + alpha * KL), 0.22, 5e-3)  # "about a fifth"

# §2.3 top-k: weights (5,3,1,1), k = 2, p_S^T = (0.5, 0.25, 0.125, 0.125)
w = [5, 3, 1, 1]
q = [F(x, w[0] + w[1]) for x in w[:2]]
eq(q, [F(5, 8), F(3, 8)])
close(float(q[0]), 0.625, 0)
close(float(q[1]), 0.375, 0)
pS = [0.5, 0.25, 0.125, 0.125]
close(sum(pS), 1, 1e-12)
Ltop = -sum(float(qi) * log(pi) for qi, pi in zip(q, pS[:2]))
close(Ltop, 1.375 * LN2, 1e-12)
close(Ltop, 0.953, 5e-4)
close(pS[2] + pS[3], 0.25, 0)

# §2.4 sequence KD: 1/2, 1/4, 1/8
Lseq = -sum(log(p) for p in (0.5, 0.25, 0.125))
eq(1 + 2 + 3, 6)
close(Lseq, 6 * LN2, 1e-12)
close(Lseq, 4.159, 5e-4)
close(Lseq / 3, 2 * LN2, 1e-12)
close(Lseq / 3, 1.386, 5e-4)

# §2.6 feature KD: W = 1/2 [[1,1,0,0],[0,0,1,1]]
W = [[F(1, 2), F(1, 2), 0, 0], [0, 0, F(1, 2), F(1, 2)]]
hTa, hSa, hTb, hSb = [2, 4, 1, 3], [3, 0], [0, 2, 2, 2], [2, 2]
eq(mv(W, hTa), [3, 2])
eq(mv(W, hTb), [1, 2])
da = [s - t for s, t in zip(hSa, mv(W, hTa))]
db = [s - t for s, t in zip(hSb, mv(W, hTb))]
eq(da, [0, -2])
eq(db, [1, 0])
eq(sq(da), 4)
eq(sq(db), 1)
eq(sq(da) + sq(db), 5)

# §3 step 2 labels: P = 412, R = 638
P, R = 412, 638
eq(P + R, 1050)
close(R / (P + R), 0.608, 5e-4)
close(R / (P + R) * 100, 60.8, 0.05)

# §3 step 6 AdamW, first step
b1, b2, g1, th0, eta, lam = 0.9, 0.999, 0.2, 0.1, 8e-5, 0.01
m1 = b1 * 0 + (1 - b1) * g1
v1 = b2 * 0 + (1 - b2) * g1 ** 2
close(m1, 0.02, 1e-15)
close(v1, 0.00004, 1e-15)
close(g1 ** 2, 0.04, 1e-15)
mh, vh = m1 / (1 - b1), v1 / (1 - b2)
close(mh, 0.2, 1e-12)
close(vh, 0.04, 1e-12)
close(mh / sqrt(vh), 1, 1e-12)
eps = 1e-8
th1 = th0 - eta * (mh / (sqrt(vh) + eps) + lam * th0)
close(lam * th0, 0.001, 1e-15)
close(eta * (1 + 0.001), 0.00008008, 1e-15)
close(0.1 - 0.00008008, 0.09991992, 1e-15)
close(th1, 0.0999199, 5e-8)
close(lam * th0 / 1 * 100, 0.1, 1e-12)  # "adds only 0.1%"

# §3 effective batch
eq(1 * 8, 8)
eq(2 * 4, 8)
eq(8 * 2048, 16_384)

# §4 LoRA: W = I, r = 1, alpha = 2, B = (1, 2)^T, A = (0.5, -1)
Bv, Av, s = [1, 2], [F(1, 2), -1], F(2, 1)
BA = [[b * a for a in Av] for b in Bv]
eq(BA, [[F(1, 2), -1], [1, -2]])
Wp = [[(1 if i == j else 0) + s * BA[i][j] for j in range(2)] for i in range(2)]
eq(Wp, [[2, -2], [2, -3]])
eq(16 * (3840 + 3840), 122_880)
eq(16 * 7680, 122_880)
eq(3840 * 3840, 14_745_600)
close(122_880 / 14_745_600 * 100, 0.83, 5e-3)

# §5.7 ln V
eq(2 ** 18, 262_144)
close(log(262_144), 18 * LN2, 1e-12)
close(18 * 0.6931, 12.48, 5e-3)
close(log(262_144), 12.48, 5e-3)

# §9 wall time
tok = 300 * 8 * 2048
eq(tok, 4_915_200)
eq(F(tok, 25), 196_608)
close(196_608 / 3600, 54.6, 0.05)
close(54.6 * 1.15, 62.8, 0.05)
close(tok / 25 / 3600 * 1.15, 62.8, 0.05)
close(tok / 25 / 3600 * 1.15 / 24, 2.6, 0.05)
# existing table rows (report check)
for rate_lo, rate_hi, lo5, hi5, lo10, hi10 in (
    (20, 30, 46, 69, 92, 139), (50, 80, 17, 28, 35, 56), (80, 120, 12, 17, 23, 35)):
    close(5e6 / rate_hi / 3600, lo5, 0.6)
    close(5e6 / rate_lo / 3600, hi5, 0.6)
    close(10e6 / rate_hi / 3600, lo10, 0.6)
    close(10e6 / rate_lo / 3600, hi10, 0.6)

# ---------------------------------------------------------------- chapter 42
# §2.2 agreement: 6 of 8
eq(F(1 + 1 + 0 + 1 + 1 + 0 + 1 + 1, 8), F(3, 4))
# top-k overlap: k = 4, overlaps 3 and 2
eq(F(1, 2) * (F(3, 4) + F(2, 4)), F(5, 8))
close(float(F(5, 8)), 0.625, 0)
# KL: position 2, p_T = (0.5, 0.5), p_S = (0.25, 0.75)
k2 = kl([0.5, 0.5], [0.25, 0.75])
eq((F(1, 2) / F(1, 4), F(1, 2) / F(3, 4)), (2, F(2, 3)))  # ratios p_T/p_S
close(log(2 / 3), -LN15, 1e-12)
close(0.5 * 0.693, 0.3465, 1e-12)
close(0.5 * 0.405, 0.2025, 1e-12)
close(0.3465 - 0.2025, 0.144, 1e-12)
close(0.5 * 0.693 - 0.5 * 0.405, 0.144, 5e-4)
close(k2, 0.144, 5e-4)
close((0 + k2) / 2, 0.072, 5e-4)
close((0 + 0.144) / 2, 0.072, 1e-12)
# perplexity: 1/2, 1/2, 1/2, 1/8
nll = -sum(log(p) for p in (0.5, 0.5, 0.5, 0.125)) / 4
eq(F(1 + 1 + 1 + 3, 4), F(3, 2))
close(nll, 1.5 * LN2, 1e-12)
close(nll, 1.04, 5e-3)
close(exp(nll), 2 * sqrt(2), 1e-12)
close(exp(nll), 2.83, 5e-3)
# ECE: 60 @ 0.90 with 48 correct, 40 @ 0.60 with 26 correct
eq(F(48, 60), F(4, 5))
eq(F(26, 40), F(13, 20))
close(abs(0.80 - 0.90), 0.10, 1e-12)
close(abs(0.65 - 0.60), 0.05, 1e-12)
ece = 60 / 100 * abs(48 / 60 - 0.90) + 40 / 100 * abs(26 / 40 - 0.60)
close(ece, 0.08, 1e-12)

# §2.3 pass@k: n = 5, c = 2


def pass_at(n, c, k):
    return 1 - comb(n - c, k) / comb(n, k)


eq((comb(3, 1), comb(5, 1)), (3, 5))
eq((comb(3, 2), comb(5, 2)), (3, 10))
close(pass_at(5, 2, 1), 0.4, 1e-12)
close(pass_at(5, 2, 2), 0.7, 1e-12)
close(pass_at(5, 2, 1), 2 / 5, 1e-12)  # = c/n

# §2.4 Wilson: 98 of 100
n, ph, z = 100, 0.98, 1.96
z2 = z * z
close(z2, 3.8416, 1e-12)
den = 1 + z2 / n
close(den, 1.0384, 5e-5)
close(z2 / (2 * n), 0.0192, 5e-5)
centre = (ph + z2 / (2 * n)) / den
close(centre, 0.962, 5e-4)
close((0.98 + 0.0192) / 1.0384, 0.962, 5e-4)
close(ph * (1 - ph) / n, 0.000196, 1e-12)
close(z2 / (4 * n * n), 0.000096, 5e-7)
root = sqrt(ph * (1 - ph) / n + z2 / (4 * n * n))
close(root, 0.01709, 5e-6)
close(sqrt(0.000196 + 0.000096), 0.01709, 5e-6)
half = z * root / den
close(half, 0.032, 5e-4)
close(1.96 * 0.01709 / 1.0384, 0.032, 5e-4)
close(centre - half, 0.930, 5e-4)
close(centre + half, 0.994, 5e-4)
close(0.962 - 0.032, 0.930, 1e-12)
close(0.962 + 0.032, 0.994, 1e-12)
close(round(centre - half, 2) * 100, 93, 1e-9)

# §6 bootstrap: B = 4
S = [0.72, 0.77, 0.73, 0.74]
Tt = [0.80, 0.82, 0.81, 0.81]
diffs = [round(s - t, 10) for s, t in zip(S, Tt)]
eq(diffs, [-0.08, -0.05, -0.08, -0.07])
close(sum(diffs), -0.28, 1e-12)
close(sum(diffs) / 4, -0.07, 1e-12)
close(sum(S) / 4, 0.74, 1e-12)   # matches the report's distilled 0.74
close(sum(Tt) / 4, 0.81, 1e-12)  # matches the report's teacher 0.81

# §8.1 memory: E4B at 128K on 16 GB
mem = 4.0 + 2.17 + 1.0
close(mem, 7.17, 1e-12)
close(0.8 * 16, 12.8, 1e-12)
close(12.8 - mem, 5.6, 0.05)
close(2.17 / 2, 1.1, 0.05)

print(f"ch39_42_distillation: all {checks} checks passed (chapters 39, 40, 41, 42)")
