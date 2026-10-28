#!/usr/bin/env python3
"""Verify every number in the "On paper" cards of docs/38-derivations.md.

Stdlib only. Each check uses a tolerance that matches the rounding shown in
the text (half a unit in the last displayed digit, unless noted).
"""
import math
import struct

N_CHECKS = 0


def close(a, b, tol):
    global N_CHECKS
    N_CHECKS += 1
    assert abs(a - b) <= tol, f"{a!r} != {b!r} (tol {tol})"


def eq(a, b):
    global N_CHECKS
    N_CHECKS += 1
    assert a == b, f"{a!r} != {b!r}"


def sigma(x):
    return 1.0 / (1.0 + math.exp(-x))


def softmax(z):
    m = max(z)
    ex = [math.exp(v - m) for v in z]
    s = sum(ex)
    return [v / s for v in ex]


def lse(z):
    m = max(z)
    return m + math.log(sum(math.exp(v - m) for v in z))


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
            for i in range(len(A))]


def outer(u, v):
    return [[a * b for b in v] for a in u]


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


H3, H4 = 5e-4, 5e-5   # half a unit in the 3rd / 4th decimal

# ---------------------------------------------------------------- §1 softmax + CE
e = math.e
close(e, 2.718, H3)
close(1 / e, 0.368, H3)
z = [1.0, 0.0, -1.0]
Z = sum(math.exp(v) for v in z)
close(2.718 + 1 + 0.368, 4.086, 1e-12)
close(Z, 4.086, H3)
p = softmax(z)
for got, shown in zip(p, [0.665, 0.245, 0.090]):
    close(got, shown, H3)
# from the rounded intermediate values a reader would use
for got, shown in zip([2.718 / 4.086, 1 / 4.086, 0.368 / 4.086], [0.665, 0.245, 0.090]):
    close(got, shown, H3)
L = -math.log(p[0])
close(L, 0.408, H3)
close(-math.log(0.665), 0.408, H3)
close(math.log(3), 1.099, H3)

# ---------------------------------------------------------------- §1.1 gradient
grad = [p[0] - 1, p[1], p[2]]
for got, shown in zip(grad, [-0.335, 0.245, 0.090]):
    close(got, shown, H3)
close(-0.335 + 0.245 + 0.090, 0.0, 1e-12)
close(math.exp(0.1), 1.10517, 5e-6)
close(math.exp(-0.1), 0.90484, 5e-6)
Zp = Z + math.exp(0.1) - 1
Zm = Z + math.exp(-0.1) - 1
close(Zp, 4.1913, H4)
close(Zm, 3.9910, H4)
close(4.1913 / 3.9910, 1.0502, H4)
close(math.log(1.0502) / 0.2, 0.2449, H4)
close((math.log(Zp) - math.log(Zm)) / 0.2, 0.2449, H4)   # exact central difference
close(1 / 4.0862, 0.2447, H4)
close(p[1], 0.2447, H4)

# ---------------------------------------------------------------- §1.2 Jacobian
p = [0.5, 0.25, 0.25]
close(softmax([math.log(2), 0, 0])[0], 0.5, 1e-12)
J = [[(p[i] if i == j else 0) - p[i] * p[j] for j in range(3)] for i in range(3)]
J16 = [[4, -2, -2], [-2, 3, -1], [-2, -1, 3]]
for i in range(3):
    for j in range(3):
        close(16 * J[i][j], J16[i][j], 1e-12)
    close(sum(J[i]), 0.0, 1e-12)                                  # J 1 = 0
for j in range(3):                                                # row1 = -(row2+row3)
    close(J16[0][j], -(J16[1][j] + J16[2][j]), 0)
close(J16[1][1] * J16[2][2] - J16[1][2] * J16[2][1], 8, 0)        # rank 2
close(p[0] * (1 - p[0]), 0.25, 1e-12)
pp = softmax([math.log(2) + 0.1, 0, 0])[0]
pm = softmax([math.log(2) - 0.1, 0, 0])[0]
close(pp, 0.52498, 5e-6)
close(pm, 0.47502, 5e-6)
close((pp - pm) / 0.2, 0.2498, H4)
close((0.52498 - 0.47502) / 0.2, 0.2498, H4)

# ---------------------------------------------------------------- §1.3 stable softmax + LSE
close(math.log(struct.unpack('>f', bytes.fromhex('7f7fffff'))[0]), 88.7, 0.05)  # fp32 max
close(struct.unpack('>f', bytes.fromhex('7f7fffff'))[0], 3.4e38, 0.05e38)
import sys
close(math.log(sys.float_info.max), 709.8, 0.05)                 # fp64 max ~ e^709.8
bf16_max = struct.unpack('>f', bytes.fromhex('7f7f0000'))[0]
close(bf16_max, 3.4e38, 0.05e38)                                 # bf16 max ~ 3.39e38
close(math.log(bf16_max), 88.7, 0.05)
try:
    math.exp(1000)                     # e^1000 overflows even float64
    raise AssertionError("expected overflow")
except OverflowError:
    pass
try:
    struct.pack('>f', math.exp(89))    # fp32 overflows past e^88.7
    raise AssertionError("expected overflow")
except OverflowError:
    pass
zb = [1000.0, 999.0, 998.0]
close(math.exp(-1), 0.3679, H4)
close(math.exp(-2), 0.1353, H4)
close(1 + 0.3679 + 0.1353, 1.5032, 1e-12)
close(sum(math.exp(v - 1000) for v in zb), 1.5032, H4)
pb = softmax(zb)
for got, shown in zip(pb, [0.665, 0.245, 0.090]):
    close(got, shown, H3)
for got, shown in zip([1 / 1.5032, 0.3679 / 1.5032, 0.1353 / 1.5032], [0.665, 0.245, 0.090]):
    close(got, shown, H3)
for a, b in zip(pb, softmax(z)):
    close(a, b, 1e-12)                                            # shift invariance
close(lse(zb), 1000.408, H3)
close(1000 + math.log(1.5032), 1000.408, H3)
close(lse(zb) - 1000, 0.408, H3)

# ---------------------------------------------------------------- z-loss
lam = 1e-4
close(lse(z), 1.408, H3)
close(lam * lse(z) ** 2, 0.000198, 5e-7)
close(lam * 1.408 ** 2, 0.000198, 5e-7)
close(lam * lse(zb) ** 2, 100.08, 5e-3)
close(lam * 1000.408 ** 2, 100.08, 5e-3)


def to_bf16(x):
    """Round a float to bfloat16 (round-to-nearest-even on the fp32 bits)."""
    b = struct.unpack('>I', struct.pack('>f', x))[0]
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return struct.unpack('>f', struct.pack('>I', b))[0]


eq(to_bf16(1004.0) - to_bf16(1000.0), 4.0)                        # spacing 4 near 1000
eq([to_bf16(v) for v in zb], [1000.0, 1000.0, 1000.0])

# ---------------------------------------------------------------- §2 attention forward
q2 = [1, 1, 1, 1]
k1, k2 = [2, 2, 0, 0], [0, 0, 1, -1]
v1, v2 = [1, 0, 0, 0], [0, 1, 0, 0]
S2 = [dot(q2, k1) / 2, dot(q2, k2) / 2]
eq(S2, [2.0, 0.0])
close(math.exp(2), 7.389, H3)
P2 = softmax(S2)
close(P2[0], 0.881, H3)
close(P2[1], 0.119, H3)
close(7.389 / 8.389, 0.881, H3)
close(1 / 8.389, 0.119, H3)
O2 = [P2[0] * a + P2[1] * b for a, b in zip(v1, v2)]
for got, shown in zip(O2, [0.881, 0.119, 0, 0]):
    close(got, shown, H3)
P1 = softmax([dot(q2, k1) / 2, -math.inf])   # any row-1 score; masked entry is -inf
eq(P1, [1.0, 0.0])

# ---------------------------------------------------------------- §2.1 complexity (Qwen3-14B)
eq(128, 2 ** 7)
eq(4096, 2 ** 12)
eq(4096 ** 2 * 128, 2 ** 31)
close(2 ** 31, 2.15e9, 0.005e9)
close(40 * 2 ** 31, 8.6e10, 0.05e10)
eq(4096 * 128, 2 ** 19)
eq(2 ** 19, 524288)
eq(2 ** 31 // 2 ** 19, 4096)

# ---------------------------------------------------------------- §2.2 attention backward
dO2 = [1, -1, 0, 0]
dP2 = [dot(dO2, v1), dot(dO2, v2)]
eq(dP2, [1, -1])
s2 = dot(dP2, P2)
close(s2, 0.762, H3)
close(1 * 0.881 - 1 * 0.119, 0.762, 1e-12)
dS2 = [P2[j] * (dP2[j] - s2) for j in range(2)]
close(dS2[0], 0.210, H3)
close(dS2[1], -0.210, H3)
close(1 - 0.762, 0.238, 1e-12)
close(-1 - 0.762, -1.762, 1e-12)
close(0.881 * 0.238, 0.210, H3)
close(0.119 * -1.762, -0.210, H3)
close(sum(dS2), 0.0, 1e-12)
# softmax-Jacobian check: J dP = 2 p1 p2 (1, -1)
Jr = [[P2[0] * (1 - P2[0]), -P2[0] * P2[1]], [-P2[0] * P2[1], P2[1] * (1 - P2[1])]]
JdP = [dot(Jr[0], dP2), dot(Jr[1], dP2)]
close(JdP[0], dS2[0], 1e-12)
close(JdP[1], dS2[1], 1e-12)
close(2 * 0.881 * 0.119, 0.210, H3)
dq2 = [(dS2[0] * a + dS2[1] * b) / 2 for a, b in zip(k1, k2)]
for got, shown in zip(dq2, [0.210, 0.210, -0.105, 0.105]):
    close(got, shown, H3)
for got, shown in zip([0.105 * (a - b) for a, b in zip(k1, k2)], [0.210, 0.210, -0.105, 0.105]):
    close(got, shown, 1e-12)
# finite-difference check of dq2 (loss = dO2 . O2 as a function of q2)


def row2_out(q):
    sc = [dot(q, k1) / 2, dot(q, k2) / 2]
    pr = softmax(sc)
    return [pr[0] * a + pr[1] * b for a, b in zip(v1, v2)]


for i in range(4):
    qp = list(q2); qp[i] += 1e-5
    qm = list(q2); qm[i] -= 1e-5
    fd = (dot(dO2, row2_out(qp)) - dot(dO2, row2_out(qm))) / 2e-5
    close(fd, dq2[i], 1e-8)
# Evidence for the reported matrix-form error: dP - 1 s^T is not dP - s 1^T
Pm = [[1.0, 0.0], [P2[0], P2[1]]]
dPm = [[0.0, 0.0], [1.0, -1.0]]
s = [dot(dPm[i], Pm[i]) for i in range(2)]
right = [[Pm[i][j] * (dPm[i][j] - s[i]) for j in range(2)] for i in range(2)]
wrong = [[Pm[i][j] * (dPm[i][j] - s[j]) for j in range(2)] for i in range(2)]  # (1 s^T)_ij = s_j
close(right[1][0], 0.210, H3)
close(wrong[1][0], 0.881, H3)
close(sum(wrong[1]), 0.671, H3)   # row of dS no longer sums to 0

# ---------------------------------------------------------------- §2.3 variance
sums = [a + b for a in (1, -1) for b in (1, -1)]
eq(sorted(sums), [-2, 0, 0, 2])
close(sum(x * x for x in sums) / 4, 2.0, 0)
close(math.sqrt(128), 11.3, 0.05)
w_hi = sigma(math.sqrt(128))
close(w_hi, 0.99999, 5e-6)
close(1 - w_hi, 0.00001, 5e-6)
close(sigma(1), 0.731, H3)
close(1 - sigma(1), 0.269, H3)

# ---------------------------------------------------------------- §3 RMSNorm forward
x = [1, -1, 3, 5]
eq(sum(v * v for v in x), 36)
r = math.sqrt(sum(v * v for v in x) / 4 + 1e-6)
close(r, 3.0, 1e-6)
y = [v / 3 for v in x]
for got, shown in zip(y, [0.333, -0.333, 1, 1.667]):
    close(got, shown, H3)
close(sum(v * v for v in y) / 4, 1.0, 1e-12)
close(sum(x) / 4, 2.0, 0)

# ---------------------------------------------------------------- §3 RMSNorm Jacobian
x = [1.0, 7.0]
d = 2
r = math.sqrt(sum(v * v for v in x) / d)
close(r, 5.0, 1e-12)
close(d * r * r, 50.0, 1e-9)
Jn = [[(1 / r) * ((1 if i == j else 0) - x[i] * x[j] / (d * r * r)) for j in range(2)] for i in range(2)]
close(1 - 1 / 50, 0.98, 1e-12)
close(7 / 50, 0.14, 1e-12)
close(1 - 49 / 50, 0.02, 1e-12)
for got, shown in zip([Jn[0][0], Jn[0][1], Jn[1][0], Jn[1][1]], [0.196, -0.028, -0.028, 0.004]):
    close(got, shown, 1e-12)
close(0.196 - 7 * 0.028, 0.0, 1e-12)
close(-0.028 + 7 * 0.004, 0.0, 1e-12)
close(math.sqrt(25.105), 5.0105, H4)
close(math.sqrt(24.905), 4.9905, H4)
close((1.1 ** 2 + 49) / 2, 25.105, 1e-12)
close((0.9 ** 2 + 49) / 2, 24.905, 1e-12)
yp = 1.1 / math.sqrt(25.105)
ym = 0.9 / math.sqrt(24.905)
close(yp, 0.21954, 5e-6)
close(ym, 0.18034, 5e-6)
close((yp - ym) / 0.2, 0.1960, H4)
close((0.21954 - 0.18034) / 0.2, 0.1960, H4)

# ---------------------------------------------------------------- §4 RoPE relative property


def R(theta):
    c, s_ = round(math.cos(theta), 12), round(math.sin(theta), 12)
    return [[c, -s_], [s_, c]]


def rot(theta, v):
    M = R(theta)
    return [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]]


w = math.pi / 2
cq, ck = [1, 0], [2, 1]
q1, k3 = rot(1 * w, cq), rot(3 * w, ck)
eq([round(v, 9) + 0.0 for v in q1], [0.0, 1.0])
eq([round(v, 9) + 0.0 for v in k3], [1.0, -2.0])
close(dot(q1, k3), -2, 1e-9)
q2r, k4 = rot(2 * w, cq), rot(4 * w, ck)
eq([round(v, 9) + 0.0 for v in q2r], [-1.0, 0.0])
eq([round(v, 9) + 0.0 for v in k4], [2.0, 1.0])
close(dot(q2r, k4), -2, 1e-9)
eq([round(v, 9) + 0.0 for v in rot(math.pi, ck)], [-2.0, -1.0])
close(dot(cq, rot(math.pi, ck)), -2, 1e-9)
eq([round(v, 9) + 0.0 for v in rot(math.pi / 2, ck)], [-1.0, 2.0])
close(dot(cq, rot(math.pi / 2, ck)), -1, 1e-9)
# R_a^T R_b = R_{b-a}
a_, b_ = 0.7, 1.9
Ra, Rb = R(a_), R(b_)
RaT = [[Ra[0][0], Ra[1][0]], [Ra[0][1], Ra[1][1]]]
lhs = matmul(RaT, Rb)
rhs = R(b_ - a_)
for i in range(2):
    for j in range(2):
        close(lhs[i][j], rhs[i][j], 1e-9)

# ---------------------------------------------------------------- §4.1 base frequency
eq(2 * 32 / 128, 0.5)
close((1e4) ** -0.5, 0.01, 1e-15)
close((1e6) ** -0.5, 0.001, 1e-15)
close(2 * math.pi / 0.01, 628, 0.5)
close(2 * math.pi / 0.001, 6283, 0.5)
close(2 * math.pi, 6.28, 0.005)
close((1e6 / 1e4) ** (2 * 32 / 128), 10, 1e-9)         # middle pair stretched 10x
close((1e6 / 1e4) ** (2 * 63 / 128), 93, 0.5)          # slowest pair about 93x

# ---------------------------------------------------------------- §4.2 p-RoPE
qv = [1, 0, 2, -1, 0, 3, 1, 1]
pd = int(0.25 * 8)
eq(pd, 2)
qt = [round(v, 9) + 0.0 for v in rot(math.pi / 2, qv[:2])] + qv[2:]
eq(qt, [0.0, 1.0, 2, -1, 0, 3, 1, 1])
eq(int(0.25 * 512), 128)
eq(512 - 128, 384)
eq(128 // 2, 64)

# ---------------------------------------------------------------- §5 GQA heads
kk1, kk2 = [1, 1, 0, 0], [0, 0, 1, 1]
qa, qb = [2, 2, 0, 0], [0, 0, 2, 2]
eq([dot(qa, kk1) / 2, dot(qa, kk2) / 2], [2.0, 0.0])
eq([dot(qb, kk1) / 2, dot(qb, kk2) / 2], [0.0, 2.0])
Pa = softmax([2.0, 0.0])
Pb = softmax([0.0, 2.0])
close(Pa[0], 0.881, H3); close(Pa[1], 0.119, H3)
close(Pb[0], 0.119, H3); close(Pb[1], 0.881, H3)
qbar = [(a + b) / 2 for a, b in zip(qa, qb)]
eq(qbar, [1, 1, 1, 1])
eq([dot(qbar, kk1) / 2, dot(qbar, kk2) / 2], [1.0, 1.0])
eq(softmax([1.0, 1.0]), [0.5, 0.5])

# ---------------------------------------------------------------- §5 GQA cache (Qwen3-14B)
eq(2 * 40 * 128 * 2, 20480)
eq(2 * 8 * 128 * 2, 4096)
close(8 / 40, 0.20, 0)
close(4096 / 20480, 0.20, 1e-12)
eq(4096 * 40, 160 * 1024)
close(4096 * 40 * 16384 / 2 ** 30, 2.5, 0)
close(20480 * 40 * 16384 / 2 ** 30, 12.5, 0)

# ---------------------------------------------------------------- §6 SwiGLU forward/backward
close(sigma(2), 0.881, H3)
close(1 - sigma(2), 0.119, H3)
a = [0.0, 2.0]
b = [3.0, 1.0]
Wd = [1.0, 1.0]
g = [v * sigma(v) for v in a]
close(g[0], 0.0, 0)
close(g[1], 1.762, H3)
close(2 * 0.881, 1.762, 1e-12)
yv = dot(Wd, [gi * bi for gi, bi in zip(g, b)])
close(yv, 1.762, H3)


def gprime(v):
    return sigma(v) + v * sigma(v) * (1 - sigma(v))


close(gprime(0), 0.5, 1e-12)
close(gprime(2), 1.091, H3)
close(0.881 + 2 * 0.881 * 0.119, 1.091, H3)
dLa = [Wd[i] * b[i] * gprime(a[i]) for i in range(2)]
close(dLa[0], 1.5, 1e-12)
close(dLa[1], 1.091, H3)
dLb = [Wd[i] * g[i] for i in range(2)]
close(dLb[0], 0.0, 0)
close(dLb[1], 1.762, H3)
close(sigma(2.1), 0.89090, 5e-6)
close(sigma(1.9), 0.86989, 5e-6)
close(2.1 * sigma(2.1), 1.8709, H4)
close(1.9 * sigma(1.9), 1.6528, H4)
close(2.1 * 0.89090, 1.8709, H4)
close(1.9 * 0.86989, 1.6528, H4)
close((2.1 * 0.89090 - 1.9 * 0.86989) / 0.2, 1.0905, H4)
close((1.8709 - 1.6528) / 0.2, 1.0905, H4)
close((2.1 * sigma(2.1) - 1.9 * sigma(1.9)) / 0.2, 1.0905, H4)
close(gprime(2), 1.0908, H4)


def y_of(av):
    return dot(Wd, [ai * sigma(ai) * bi for ai, bi in zip(av, b)])


for i in range(2):
    ap = list(a); ap[i] += 1e-6
    am = list(a); am[i] -= 1e-6
    close((y_of(ap) - y_of(am)) / 2e-6, dLa[i], 1e-7)

# ---------------------------------------------------------------- §6 GELU
Phi = lambda v: 0.5 * (1 + math.erf(v / math.sqrt(2)))
close(math.erf(1 / math.sqrt(2)), 0.6827, H4)
close(1 * Phi(1), 0.841, H3)
close(0.5 * (1 + 0.6827), 0.841, H3)
close(-1 * Phi(-1), -0.159, H3)
close(-0.5 * (1 - 0.6827), -0.159, H3)
close(1 * sigma(1), 0.731, H3)
close(-1 * sigma(-1), -0.269, H3)

# ---------------------------------------------------------------- §7 MoE routing
rr = [1.0, 0.5, 0.0, -1.0]
close(math.exp(0.5), 1.6487, H4)
close(math.exp(1), 2.7183, H4)
close(2.7183 + 1.6487 + 1 + 0.3679, 5.7349, 1e-12)
close(sum(math.exp(v) for v in rr), 5.7349, H4)
gg = softmax(rr)
for got, shown in zip(gg, [0.474, 0.287, 0.174, 0.064]):
    close(got, shown, H3)
for got, shown in zip([2.7183 / 5.7349, 1.6487 / 5.7349, 1 / 5.7349, 0.3679 / 5.7349],
                      [0.474, 0.287, 0.174, 0.064]):
    close(got, shown, H3)
top = sorted(range(4), key=lambda i: -gg[i])[:2]
eq(sorted(top), [0, 1])
ren = [gg[i] / (gg[0] + gg[1]) for i in (0, 1)]
close(ren[0], 0.622, H3)
close(ren[1], 0.378, H3)
close(2.7183 + 1.6487, 4.3670, 1e-12)
close(2.7183 / 4.3670, 0.622, H3)
close(1.6487 / 4.3670, 0.378, H3)
close(gg[2] + gg[3], 0.24, 0.005)

# ---------------------------------------------------------------- §7.1 load balance
alpha, E = 0.001, 4
f = [7 / 8, 1 / 8, 0, 0]
P = [0.6, 0.2, 0.1, 0.1]
eq(f[:2], [0.875, 0.125])
close(dot(f, P), 0.55, 1e-12)
close(0.875 * 0.6 + 0.125 * 0.2, 0.55, 1e-12)
close(alpha * E * dot(f, P), 0.0022, 1e-12)
close(alpha * E * dot(f, P) / alpha, 2.2, 1e-9)
close(alpha * E * sum((1 / E) * (1 / E) for _ in range(E)), alpha, 1e-15)
close(alpha * E * f[0], 0.0035, 1e-12)
close(alpha * E * f[2], 0.0, 0)

# ---------------------------------------------------------------- §7.2 active parameters (Qwen3-30B-A3B config)
L_, d_, nexp, ffe, hq, hkv, dh, vocab = 48, 2048, 128, 768, 32, 4, 128, 151936
per_expert = 3 * d_ * ffe
close(per_expert / 1e6, 4.72, 0.005)
experts = per_expert * nexp * L_
close(experts / 1e9, 29.0, 0.05)
close(4.72e6 * 128 * 48 / 1e9, 29.0, 0.05)
attn = L_ * (d_ * hq * dh + 2 * d_ * hkv * dh + hq * dh * d_)
close(attn / 1e9, 0.91, 0.005)
emb = 2 * vocab * d_                 # untied embedding + LM head
close(emb / 1e9, 0.62, 0.005)
router = L_ * d_ * nexp
close(router / 1e9, 0.01, 0.005)
active = attn + router + experts * 8 / 128 + emb
close(active / 1e9, 3.35, 0.005)
close(29.0 * 8 / 128, 1.81, 0.005)
close(0.91 + 0.01 + 1.81 + 0.62, 3.35, 1e-12)
total = attn + router + experts + emb
close(total / 1e9, 30.5, 0.05)
close(0.91 + 0.01 + 29.0 + 0.62, 30.54, 1e-9)
close(8 / 128, 0.0625, 0)

# ---------------------------------------------------------------- §8 Gated DeltaNet (d_k=2, d_v=1)
S0 = [2.0, 0.0]
kt = [0.6, 0.8]
close(dot(kt, kt), 1.0, 1e-12)                                   # unit key
vt, at, bt = 1.0, 0.5, 0.5
pred = at * dot(S0, kt)
close(pred, 0.6, 1e-12)
close(0.5 * (2 * 0.6 + 0 * 0.8), 0.6, 1e-12)
St = [at * s_ + bt * (vt - pred) * k_ for s_, k_ in zip(S0, kt)]
close(St[0], 1.12, 1e-12); close(St[1], 0.16, 1e-12)
close(1.12 * 0.6, 0.672, 1e-12); close(0.16 * 0.8, 0.128, 1e-12)
ot = dot(St, kt)
close(ot, 0.8, 1e-12)
close(ot, pred + 0.5 * (1 - pred), 1e-12)                        # halfway from 0.6 to 1
Sp = [at * s_ for s_ in S0]
eq(Sp, [1.0, 0.0])
S2step = [sp - bt * (dot(Sp, kt) - vt) * k_ for sp, k_ in zip(Sp, kt)]
close(S2step[0], 1.12, 1e-12); close(S2step[1], 0.16, 1e-12)
# gradient of 1/2 (S^T k - v)^2 by finite differences equals (S^T k - v) k
for i in range(2):
    Sa = list(S0); Sa[i] += 1e-6
    Sb = list(S0); Sb[i] -= 1e-6
    fd = (0.5 * (dot(Sa, kt) - vt) ** 2 - 0.5 * (dot(Sb, kt) - vt) ** 2) / 2e-6
    close(fd, (dot(S0, kt) - vt) * kt[i], 1e-7)

# ---------------------------------------------------------------- §8.1 linear attention vs delta rule
close(dot(S0, kt), 1.2, 1e-12)
Slin = [s_ + vt * k_ for s_, k_ in zip(S0, kt)]
close(Slin[0], 2.6, 1e-12); close(Slin[1], 0.8, 1e-12)
close(dot(Slin, kt), 2.2, 1e-12)
close(2.6 * 0.6 + 0.8 * 0.8, 2.2, 1e-12)
close(1 - 1.2, -0.2, 1e-12)
Sdel = [s_ + (vt - dot(S0, kt)) * k_ for s_, k_ in zip(S0, kt)]
close(Sdel[0], 1.88, 1e-12); close(Sdel[1], -0.16, 1e-12)
close(dot(Sdel, kt), 1.0, 1e-12)
close(1.88 * 0.6, 1.128, 1e-12); close(-0.16 * 0.8, -0.128, 1e-12)

# ---------------------------------------------------------------- §8.2 transition form
kkT = outer(kt, kt)
for got, shown in zip([kkT[0][0], kkT[0][1], kkT[1][0], kkT[1][1]], [0.36, 0.48, 0.48, 0.64]):
    close(got, shown, 1e-12)
M = [[(1 if i == j else 0) - bt * kkT[i][j] for j in range(2)] for i in range(2)]
for got, shown in zip([M[0][0], M[0][1], M[1][0], M[1][1]], [0.82, -0.24, -0.24, 0.68]):
    close(got, shown, 1e-12)
row = matmul([S0], M)[0]
close(row[0], 1.64, 1e-12); close(row[1], -0.48, 1e-12)
row = [at * v for v in row]
close(row[0], 0.82, 1e-12); close(row[1], -0.24, 1e-12)
bvk = [bt * vt * k_ for k_ in kt]
close(bvk[0], 0.3, 1e-12); close(bvk[1], 0.4, 1e-12)
Strans = [a_ + b_ for a_, b_ in zip(row, bvk)]
close(Strans[0], 1.12, 1e-12); close(Strans[1], 0.16, 1e-12)

# ---------------------------------------------------------------- §8.2 chunk product (C = 2)
kc1, kc2 = [1.0, 0.0], [0.6, 0.8]
A1 = [[(1 if i == j else 0) - 0.5 * outer(kc1, kc1)[i][j] for j in range(2)] for i in range(2)]
A2 = [[(1 if i == j else 0) - 0.5 * outer(kc2, kc2)[i][j] for j in range(2)] for i in range(2)]
eq(A1, [[0.5, 0.0], [0.0, 1.0]])
prod = matmul(A1, A2)
for got, shown in zip([prod[0][0], prod[0][1], prod[1][0], prod[1][1]], [0.41, -0.12, -0.24, 0.68]):
    close(got, shown, 1e-12)
close(dot(kc1, kc2), 0.6, 1e-12)
close(-0.25 * 0.6, -0.15, 1e-12)
part = [[(1 if i == j else 0) - 0.5 * outer(kc1, kc1)[i][j] - 0.5 * outer(kc2, kc2)[i][j]
         for j in range(2)] for i in range(2)]
for got, shown in zip([part[0][0], part[0][1], part[1][0], part[1][1]], [0.32, -0.24, -0.24, 0.68]):
    close(got, shown, 1e-12)
cross = [[0.15 * v for v in rr_] for rr_ in outer(kc1, kc2)]
for got, shown in zip([cross[0][0], cross[0][1], cross[1][0], cross[1][1]], [0.09, 0.12, 0, 0]):
    close(got, shown, 1e-12)
for i in range(2):
    for j in range(2):
        close(part[i][j] + cross[i][j], prod[i][j], 1e-12)
# I - K^T W K with K rows k1, k2 and W = [[0.5, -0.15], [0, 0.5]] reproduces the product
K = [kc1, kc2]
W = [[0.5, -0.15], [0.0, 0.5]]
KtWK = matmul(matmul([[K[0][0], K[1][0]], [K[0][1], K[1][1]]], W), K)
for i in range(2):
    for j in range(2):
        close((1 if i == j else 0) - KtWK[i][j], prod[i][j], 1e-12)

# ---------------------------------------------------------------- §11 quantization
k = 4
eq(2 ** (k - 1) - 1, 7)
wq = [0.46, -1.4, 0.13, 0.95]
s = max(abs(v) for v in wq) / 7
close(s, 0.2, 1e-12)
ratios = [v / s for v in wq]
for got, shown in zip(ratios, [2.3, -7, 0.65, 4.75]):
    close(got, shown, 1e-9)
qint = [max(-7, min(7, round(v))) for v in ratios]
eq(qint, [2, -7, 1, 5])
what = [s * qi for qi in qint]
for got, shown in zip(what, [0.4, -1.4, 0.2, 1.0]):
    close(got, shown, 1e-12)
err = [w_ - wh for w_, wh in zip(wq, what)]
for got, shown in zip(err, [0.06, 0, -0.07, -0.05]):
    close(got, shown, 1e-12)
assert all(abs(v) <= s / 2 + 1e-12 for v in err)
close(s / 2, 0.1, 1e-12)

# ---------------------------------------------------------------- §11 block scale
wb = [0.05, -0.07, 0.02, 0.03]
eq([round(v / 0.2) for v in wb], [0, 0, 0, 0])
assert all(abs(v / 0.2) < 0.5 for v in wb)
close(max(abs(v) for v in wb), 0.07, 1e-12)
sb = 0.07 / 7
close(sb, 0.01, 1e-12)
qb = [round(v / sb) for v in wb]
eq(qb, [5, -7, 2, 3])
for v, qi in zip(wb, qb):
    close(v, sb * qi, 1e-12)
close(sb / 2, 0.005, 1e-12)
close((s / 2) / (sb / 2), 20, 1e-9)

# ---------------------------------------------------------------- §11.1 STE
w0 = 0.46
close(s * round(w0 / s), 0.4, 1e-12)
w1 = w0 - 0.1 * 2
close(w1, 0.26, 1e-12)
close(w1 / s, 1.3, 1e-9)
close(s * round(w1 / s), 0.2, 1e-12)
close(1.5 * s, 0.3, 1e-12)        # rounding boundary between q=1 and q=2

# ---------------------------------------------------------------- §12 scaling law (toy constants)
Lf = lambda N, D: 1.7 + 400 / N ** (1 / 3) + 400 / D ** (1 / 3)
close((1e9) ** (1 / 3), 1000, 1e-6)
close((64e9) ** (1 / 3), 4000, 1e-6)
close((8e9) ** (1 / 3), 2000, 1e-6)
close((512e9) ** (1 / 3), 8000, 1e-6)
close(Lf(1e9, 64e9), 2.2, 1e-9)
close(Lf(8e9, 64e9), 2.0, 1e-9)
close(Lf(1e9, 512e9), 2.15, 1e-9)

# ---------------------------------------------------------------- §12 Chinchilla allocation
C = 1.2e22
N = math.sqrt(C / 120)
close(C / 120, 1e20, 1e6)
close(N, 1e10, 1)
close(20 * N, 2e11, 10)
close(6 * N * 20 * N, C, 1e10)
N4 = math.sqrt(4 * C / 120)
close(N4, 2e10, 1)
close(20 * N4, 4e11, 10)
close(36e12 / 14.8e9, 2400, 50)
close(36e12 / 14.8e9 / 20, 120, 5)

# ---------------------------------------------------------------- §13 attention share (Qwen3-14B, GQA)
d, Hq, dhh, dff, Gkv = 5120, 40, 128, 17408, 8
eq(Hq * dhh, d)
eq(Gkv * dhh, 1024)
proj = 6 * (2 * d * Hq * dhh + 2 * d * Gkv * dhh)      # 6 (2 d H d_h + 2 d G d_h)
close(proj / (d * d), 14.4, 1e-9)
eq(12 * d + 12 * 1024 + 18 * dff, 387072)
eq((12 * d + 12 * 1024 + 18 * dff) / 12, 32256)


def ratio(T):
    return 12 * Hq * dhh * T * T / ((proj + 18 * d * dff) * T)


close(ratio(2048), 2048 / 32256, 1e-12)
close(ratio(2048), 0.063, H3)
close(ratio(2048) / (1 + ratio(2048)), 0.06, 0.005)
close(ratio(32768), 1.02, 0.005)
close(ratio(32768) / (1 + ratio(32768)), 0.50, 0.005)

# ---------------------------------------------------------------- §14 AdamW
g1, b1, b2 = -0.2, 0.9, 0.999
m1 = b1 * 0 + (1 - b1) * g1
v1_ = b2 * 0 + (1 - b2) * g1 ** 2
close(m1, -0.02, 1e-12)
close(v1_, 0.00004, 1e-15)
close(g1 ** 2, 0.04, 1e-12)
mh = m1 / (1 - b1 ** 1)
vh = v1_ / (1 - b2 ** 1)
close(mh, -0.2, 1e-12)
close(vh, 0.04, 1e-12)
theta0, eta, lam_wd, eps = 0.5, 0.01, 0.1, 1e-8
close(mh / (math.sqrt(vh) + eps), -1.0, 1e-6)
theta1 = theta0 - eta * (mh / (math.sqrt(vh) + eps) + lam_wd * theta0)
close(theta1, 0.5095, 5e-9)
close(0.5 - 0.01 * (-1 + 0.1 * 0.5), 0.5095, 1e-12)
close(math.sqrt(v1_), 0.00632, 5e-6)
close(m1 / math.sqrt(v1_), -3.16, 0.005)
close(-0.02 / 0.00632, -3.16, 0.005)

# ---------------------------------------------------------------- §8 convention
# S in R^{d_v x d_k} maps a key to a value, o = S q; the delta update
# S + beta (v - S k) k^T then recalls the new value exactly (alpha = beta = 1).
Sm = [[1.0, 0.0], [0.0, 0.0]]            # key e1 -> value (1, 0)
ke, ve = [1.0, 0.0], [0.0, 1.0]           # overwrite key e1 with value (0, 1)
read = lambda S_, q: [sum(S_[i][j] * q[j] for j in range(2)) for i in range(2)]   # o = S q
err_ = [ve[i] - read(Sm, ke)[i] for i in range(2)]
upd = [[Sm[i][j] + err_[i] * ke[j] for j in range(2)] for i in range(2)]         # (v - S k) k^T
eq(read(upd, ke), [0.0, 1.0])
# gradient of 1/2 ||S k - v||^2 w.r.t. S_ij by finite differences = (S k - v)_i k_j
for i in range(2):
    for j in range(2):
        Sa = [row_[:] for row_ in Sm]; Sa[i][j] += 1e-6
        Sb = [row_[:] for row_ in Sm]; Sb[i][j] -= 1e-6
        La = 0.5 * sum((read(Sa, ke)[c] - ve[c]) ** 2 for c in range(2))
        Lb = 0.5 * sum((read(Sb, ke)[c] - ve[c]) ** 2 for c in range(2))
        close((La - Lb) / 2e-6, (read(Sm, ke)[i] - ve[i]) * ke[j], 1e-7)
# §8.1: alpha = beta = 1 in the boxed recurrence is the delta rule, not linear attention
assert Sdel != Slin

print(f"ch38_derivations: all {N_CHECKS} checks passed")
