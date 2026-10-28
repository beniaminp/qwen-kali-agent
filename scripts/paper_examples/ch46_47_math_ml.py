#!/usr/bin/env python3
"""Check every number in the "On paper" cards, worked examples and exercise
answers of docs/46-ai-math.md and docs/47-ai-classical-ml.md (stdlib only)."""
from fractions import Fraction as F
from math import ceil, comb, cos, e, exp, log, log2, pi, sin, sqrt, tanh
from statistics import NormalDist, median

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


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return sqrt(dot(a, a))


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def softmax(z, t=1.0):
    w = [exp(v / t) for v in z]
    s = sum(w)
    return [x / s for x in w]


N = NormalDist()
LN2, LN3, LN6 = log(2), log(3), log(6)

# ======================================================================
# chapter 46 — the math you actually need
# ======================================================================

# §1.1 table
close(LN2, 0.693, 5e-4)
close(320 / 400, 0.8, 1e-12)

# §2.2 embeddings memory
eq(8 * 512 * 5120, 20_971_520)
eq(20_971_520 * 2, 41_943_040)
eq(41_943_040, 40 * 2**20)
close(8 * 512 * 151_936 * 4 / 1e9, 2.49, 5e-3)
close(8 * 512 * 151_936 * 4 / 2**30, 2.32, 5e-3)

# §2.3 toy lookup: bytes of x = 2*4*8 float32
eq(2 * 4 * 8 * 4, 256)

# §3.1 addition and scaling
a, b = (1, 2, 2), (2, 1, 2)
eq(tuple(x + y for x, y in zip(a, b)), (3, 3, 4))
eq(tuple((x + y) / 2 for x, y in zip(a, b)), (1.5, 1.5, 2))

# §3.2 pricing dot product
n_tok, price = (1000, 4000, 250), (2, 0.5, 8)
eq([x * y for x, y in zip(n_tok, price)], [2000, 2000, 2000])
eq(dot(n_tok, price), 6000)
close(6000 / 1e6, 0.006, 1e-15)
close(0.006 * 1e6, 6000, 1e-9)

# §3.3 geometry
eq(dot((1, 0), (1, 1)), 1)
close(cos(pi / 4), 0.707, 5e-4)
close(1 * sqrt(2) * 0.707, 1, 5e-4)
close(1 * sqrt(2) * cos(pi / 4), 1, 1e-12)

# §3.4 norms
eq(sqrt(9 + 16), 5)
eq(abs(3) + abs(-4), 7)

# §3.5 unit vector
eq((3 / 5, 4 / 5), (0.6, 0.8))
close(0.36 + 0.64, 1, 1e-12)

# §3.6 cosine
c_vec = (-2, 1, 0)
eq(dot(a, b), 8)
eq(norm(a), 3)
eq(norm(b), 3)
close(8 / 9, 0.889, 5e-4)
eq(dot(a, c_vec), 0)

# §3.7 distance on unit vectors
ua, ub = [x / 3 for x in a], [x / 3 for x in b]
diff = [x - y for x, y in zip(ua, ub)]
close(diff[0], -1 / 3, 1e-12)
close(diff[1], 1 / 3, 1e-12)
close(diff[2], 0, 1e-12)
close(dot(diff, diff), 2 / 9, 1e-12)
close(2 / 9, 0.222, 5e-4)
close(2 - 2 * 8 / 9, 2 / 9, 1e-12)

# §3.8 semantic search: query is the average of the two refund docs
D = [(1, 2, 2), (2, 1, 2), (-2, 1, 0), (0, -1, 3)]
q = (1.5, 1.5, 2.0)
cs = [dot(d, q) / (norm(d) * norm(q)) for d in D]
close(cs[0], 0.972, 5e-4)
close(cs[1], 0.972, 5e-4)
close(cs[2], -0.230, 5e-4)
close(cs[3], 0.488, 5e-4)

# §4.1 matrix-vector
W = [[1, 2], [3, 4], [0, -1]]
x = (5, 6)
eq([dot(r, x) for r in W], [17, 39, -6])

# §4.2 matrix-matrix
A = [[1, 0, 2], [0, 1, 1]]
B = [[1, 2], [3, 4], [5, 6]]
eq(matmul(A, B), [[11, 14], [8, 10]])
eq(2 * 3 * 2, 12)
eq(len(matmul(B, A)), 3)
macs = 2048 * 5120 * 5120
close(macs / 1e10, 5.4, 0.05)
close(2 * macs / 1e9, 107, 0.5)

# §4.3 transpose
AB = matmul(A, B)
ABt = [list(r) for r in zip(*AB)]
eq(ABt, [[11, 8], [14, 10]])
Bt, At = [list(r) for r in zip(*B)], [list(r) for r in zip(*A)]
eq(matmul(Bt, At), ABt)
eq(1 * 1 + 3 * 0 + 5 * 2, 11)

# §4.5 broadcasting bug
yt = [1.0, 2.0, 3.0, 4.0]
yp = [1.1, 1.9, 3.2, 3.9]
close(sum((t - p) ** 2 for t, p in zip(yt, yp)) / 4, 0.0175, 1e-12)
close(sum((t - p) ** 2 for t in yt for p in yp) / 16, 2.4425, 1e-9)

# §4.6 inverse
a11, a12, a21, a22 = 2, 1, 1, 1
det = a11 * a22 - a12 * a21
eq(det, 1)
Ainv = [[a22 / det, -a12 / det], [-a21 / det, a11 / det]]
eq(Ainv, [[1, -1], [-1, 2]])
xs = [dot(r, (3, 2)) for r in Ainv]
eq(xs, [1, 1])
eq(2 * 1 + 1 * 1, 3)
eq(1 * 1 + 1 * 1, 2)
eq(matmul([[2, 1], [1, 1]], Ainv), [[1, 0], [0, 1]])

# §4.7 linear layer
W = [[1, 0, -1], [2, 1, 0]]
bias = (0.5, -1)
Wx = [dot(r, (1, 2, 3)) for r in W]
eq(Wx, [-2, 4])
eq([u + v for u, v in zip(Wx, bias)], [-1.5, 3])
eq(2 * 3 + 2, 8)
eq(5120 * 5120, 26_214_400)
eq(5120 * 1024, 5_242_880)
eq(8 * 128, 1024)
X = [(1, 2, 3), (0, 0, 0), (1, 1, 1), (-1, 0, 2)]
eq([[dot(r, xi) + bb for r, bb in zip(W, bias)] for xi in X], [[-1.5, 3], [0.5, -1], [0.5, 2], [-2.5, -3]])

# §5.1 slope table for f(x) = x^2 at 3
for h, f3h, df, ratio in ((1, 16, 7, 7), (0.1, 9.61, 0.61, 6.1), (0.01, 9.0601, 0.0601, 6.01),
                          (0.001, 9.006001, 0.006001, 6.001)):
    close((3 + h) ** 2, f3h, 1e-9)
    close((3 + h) ** 2 - 9, df, 1e-9)
    close(((3 + h) ** 2 - 9) / h, ratio, 1e-6)

# §5.2 definition
close(((3 + 0.01) ** 2 - 9) / 0.01, 2 * 3 + 0.01, 1e-9)
eq(2 * 3, 6)


def sig(z):
    return 1 / (1 + exp(-z))


# §5.3 derivative table and sigmoid card
eq(3 * 2**2, 12)
eq(exp(0), 1)
eq(1 / 2, 0.5)
eq(sig(0) * (1 - sig(0)), 0.25)
eq(1 - tanh(0) ** 2, 1)
close(LN3, 1.099, 5e-4)
close(sig(LN3), 0.75, 1e-12)
close(sig(LN3) * (1 - sig(LN3)), 0.1875, 1e-12)

# §5.4 chain rule
w, xx, yy = 2, 3, 4
u = w * xx - yy
eq(u, 2)
eq(u**2, 4)
eq(2 * u, 4)
eq(2 * u * xx, 12)
u2 = 2.001 * 3 - 4
close(u2, 2.003, 1e-12)
close(u2**2, 4.012009, 1e-12)
close(u2**2 - 4, 0.012009, 1e-12)
close((u2**2 - 4) / 0.001, 12.009, 1e-6)


def f55(w1, w2):
    return w1**2 + 3 * w1 * w2


# §5.5 partial derivatives
eq(f55(1, 2), 7)
g = (2 * 1 + 3 * 2, 3 * 1)
eq(g, (8, 3))
close(f55(1.01, 2) - 7, 0.08, 0.002)
close(f55(1, 2.01) - 7, 0.03, 1e-9)

# §5.6 step against the gradient
eq(8**2 + 3**2, 73)
w1, w2 = 1 - 0.01 * 8, 2 - 0.01 * 3
close(w1, 0.92, 1e-12)
close(w2, 1.97, 1e-12)
close(w1**2, 0.8464, 1e-12)
close(3 * w1 * w2, 5.4372, 1e-12)
close(f55(w1, w2), 6.2836, 1e-12)
close(7 - 0.01 * 73, 6.27, 1e-12)
close(7 - f55(w1, w2), 0.7164, 1e-12)
close(0.01 * 73, 0.73, 1e-12)

# §6.1 / §6.2 1-D gradient descent, L = (w-3)^2, lr 0.1
eq(2 * (0 - 3), -6)
close(0 - 0.1 * (-6), 0.6, 1e-12)
close(2 * (0.6 - 3), -4.8, 1e-12)
close(0.6 + 0.48, 1.08, 1e-12)


def gd1(lr, steps, w=0.0):
    path = [w]
    for _ in range(steps):
        w = w - lr * 2 * (w - 3)
        path.append(w)
    return path


path = gd1(0.1, 5)
for got, shown in zip(path, (0, 0.6, 1.08, 1.464, 1.7712, 2.01696)):
    close(got, shown, 1e-9)
for got, shown in zip(path, (3, 2.4, 1.92, 1.536, 1.2288, 0.98304)):
    close(3 - got, shown, 1e-9)
for got, shown in zip(path, (9, 5.76, 3.6864, 2.3593, 1.5099, 0.9664)):
    close((got - 3) ** 2, shown, 5e-5)
close(0.98304 / 1.2288, 0.8, 1e-12)
close(0.9664 / 1.5099, 0.64, 5e-4)

# §6.3 factors and table
for lr, fac in ((0.01, 0.98), (0.1, 0.8), (0.5, 0), (1.0, -1), (1.1, -1.2)):
    close(1 - 2 * lr, fac, 1e-12)
for lr, shown in ((0.01, (0, 0.06, 0.1188, 0.1764)), (0.1, (0, 0.6, 1.08, 1.464)), (0.5, (0, 3, 3, 3)),
                  (1.0, (0, 6, 0, 6)), (1.1, (0, 6.6, -1.32, 8.184))):
    for got, s in zip(gd1(lr, 3), shown):
        close(got, s, 5e-5)
eq((6 - 3) ** 2, 9)

# §6.4 2-D valley L = w1^2 + 4 w2^2


def L64(w):
    return w[0] ** 2 + 4 * w[1] ** 2


def step64(w, lr):
    return (w[0] - lr * 2 * w[0], w[1] - lr * 8 * w[1])


eq(L64((4, 2)), 32)
close(1 - 2 * 0.1, 0.8, 1e-12)
close(1 - 8 * 0.1, 0.2, 1e-12)
s1 = step64((4, 2), 0.1)
close(s1[0], 3.2, 1e-12)
close(s1[1], 0.4, 1e-12)
close(L64(s1), 10.88, 1e-9)
s2 = step64(s1, 0.1)
close(s2[0], 2.56, 1e-12)
close(s2[1], 0.08, 1e-12)
close(L64(s2), 6.5792, 1e-9)
close(1 - 2 * 0.3, 0.4, 1e-12)
close(1 - 8 * 0.3, -1.4, 1e-12)
s1 = step64((4, 2), 0.3)
close(s1[0], 1.6, 1e-12)
close(s1[1], -2.8, 1e-12)
close(L64(s1), 33.92, 1e-9)
eq(2 / 8, 0.25)
close(1 - 2 * 0.2499, 0.5, 1e-3)  # just under the cap the gentle axis roughly halves
close(2 / 200, 0.01, 1e-15)  # L = w1^2 + 100 w2^2: steep gradient 200 w2, cap 0.01
close(1 - 2 * 0.01, 0.98, 1e-12)  # gentle-axis factor at that cap

# §6.6 central differences on x^3 at 2
close(2.01**3, 8.120601, 1e-9)
close(1.99**3, 7.880599, 1e-9)
close(2.01**3 - 1.99**3, 0.240002, 1e-9)
close((2.01**3 - 1.99**3) / 0.02, 12.0001, 1e-6)
close((2.01**3 - 8) / 0.01, 12.0601, 1e-6)
close(12.0001 - 12, 0.0001, 1e-9)
close(12.0601 - 12, 0.06, 2e-3)
close(0.5 / 1.5, 1 / 3, 1e-12)

# §7.1 log rules
close(e, 2.71828, 5e-6)
close(exp(LN2 + LN3), 6, 1e-12)
close(LN6, 0.693 + 1.099, 1e-3)
close(LN6, 1.792, 5e-4)
close(log(0.5), -0.693, 5e-4)
close(log(8), 3 * LN2, 1e-12)
close(log(8), 2.079, 5e-4)
close(2.079 / 0.693, 3, 1e-12)
close(log2(8), 3, 1e-12)
close(log(0.5 * 0.25), log(0.125), 1e-12)
close(log(0.125), -2.079, 5e-4)
close(log(0.5) + log(0.25), -2.079, 5e-4)
close(-0.693 - 1.386, -2.079, 1e-12)

# §7.2 log space
close(200 * log(0.01), -921.034, 5e-4)
close(log(0.01), -4.60517, 5e-6)
ok(0.01**200 == 0.0, "0.01**200 underflows")
close(0.6**200 / 1e-45, 4, 0.5)  # "about 4e-45" at 0.6 per token
close(2.0**-126, 1.2e-38, 5e-40)  # smallest normal float32
ok(0.6**200 < 2.0**-126, "0.6**200 is below the smallest normal float32")
close(200 * log(0.01) / log(10), -400, 1e-9)  # 0.01**200 = 1e-400

# §7.3 log-sum-exp
close(exp(-1), 0.3679, 5e-5)
close(exp(-2), 0.1353, 5e-5)
close(0.1353 + 0.3679 + 1, 1.5032, 1e-12)
close(log(exp(-2) + exp(-1) + 1), 0.4076, 5e-5)
lse = 1002 + log(exp(-2) + exp(-1) + 1)
close(lse, 1002.4076, 5e-5)
for got, s in zip(softmax([0, 1, 2]), (0.0900, 0.2447, 0.6652)):
    close(got, s, 5e-5)

# §7.4 nats and bits
close(1 / LN2, 1.443, 5e-4)

# §8.1 / §8.2 tickets
close(200 / 1000, 0.2, 1e-12)
close(300 / 1000, 0.3, 1e-12)
close(120 / 1000, 0.12, 1e-12)
close(0.12 / 0.3, 0.4, 1e-12)
close(0.2 + 0.3 - 0.12, 0.38, 1e-12)

# §8.3 independence
close(0.2 * 0.3, 0.06, 1e-12)
close(0.1 * 0.1, 0.01, 1e-12)

# §8.4 Bayes fraud
tp, fp = 0.9 * 0.01, 0.05 * 0.99
close(tp, 0.009, 1e-12)
close(fp, 0.0495, 1e-12)
close(tp + fp, 0.0585, 1e-12)
close(tp / (tp + fp), 0.154, 5e-4)
eq((90, 10, 100), (int(round(0.9 * 100)), 100 - 90, 10_000 // 100))
eq(int(round(0.05 * 9900)), 495)
eq(9900 - 495, 9405)
eq(90 + 495, 585)
eq(10 + 9405, 9415)
eq(585 + 9415, 10_000)
close(90 / 585, 0.154, 5e-4)
close(90 / 585, 0.1533, 1e-3)  # the simulation printed 0.1533

# §8.5 expectation
vals, probs = (100, 200, 600), (0.5, 0.3, 0.2)
parts = [v * p for v, p in zip(vals, probs)]
for got, s in zip(parts, (50, 60, 120)):
    close(got, s, 1e-9)
EX = sum(parts)
close(EX, 230, 1e-9)
ok(120 / 230 > 0.5, "long answers carry more than half")
close(1000 * EX, 230_000, 1e-6)

# §8.6 variance
parts2 = [v * v * p for v, p in zip(vals, probs)]
for got, s in zip(parts2, (5000, 12000, 72000)):
    close(got, s, 1e-6)
EX2 = sum(parts2)
close(EX2, 89000, 1e-6)
eq(230**2, 52900)
close(EX2 - EX**2, 36100, 1e-6)
close(sqrt(EX2 - EX**2), 190, 1e-9)

# §8.7 sqrt(n)
close(190 * sqrt(100), 1900, 1e-9)
close(190 / sqrt(100), 19, 1e-12)
close(100 * 230, 23_000, 1e-9)
close(1900 / 23000 * 100, 8, 0.5)
close(190 / 230 * 100, 83, 0.5)

# §9.1 Bernoulli
close(0.8 * 0.2, 0.16, 1e-12)
close(sqrt(0.16), 0.4, 1e-12)
close(0.5 * 0.5, 0.25, 1e-12)

# §9.2 binomial
eq(comb(10, 8), 45)
close(0.8**8, 0.1678, 5e-5)
close(45 * 0.1678 * 0.04, 0.302, 5e-4)
close(comb(10, 8) * 0.8**8 * 0.2**2, 0.302, 5e-4)
close(sum(comb(10, k) * 0.8**k * 0.2 ** (10 - k) for k in range(7)), 0.12, 5e-3)
eq(10 * 0.8, 8)
close(sqrt(10 * 0.8 * 0.2), 1.26, 5e-3)

# §9.4 uniform
eq((0 + 2) / 2, 1)
close(4 / 12, 0.333, 5e-4)
close(sqrt(4 / 12), 0.577, 5e-4)

# §9.5 normal
close(N.cdf(1) - N.cdf(-1), 0.68, 5e-3)
close(N.cdf(1.96) - N.cdf(-1.96), 0.95, 5e-4)
close(N.cdf(3) - N.cdf(-3), 0.997, 5e-4)
eq((1000 - 800) / 100, 2)
close(1 - N.cdf(2), 0.023, 5e-4)
close(1 / (1 - N.cdf(2)) / 7, 6, 0.5)  # "roughly once every six weeks"
close((1 - N.cdf(3)) * 100, 0.13, 5e-3)

# §9.7 lognormal theory behind the printed sample values
close(300 * exp(0.8**2 / 2), 413, 1)
close(300 * exp(0.8 * N.inv_cdf(0.99)) / 300, 6.4, 0.1)

# §10.1 softmax
close(LN6, 1.792, 5e-4)
close(LN3, 1.099, 5e-4)
for got, s in zip(softmax([LN6, LN3, 0]), (0.6, 0.3, 0.1)):
    close(got, s, 1e-12)
for got, s in zip(softmax([LN6 + 10, LN3 + 10, 10]), (0.6, 0.3, 0.1)):
    close(got, s, 1e-12)

# §10.2 temperature
eq((6**2, 3**2, 1), (36, 9, 1))
eq(36 + 9 + 1, 46)
for got, s in zip(softmax([LN6, LN3, 0], 0.5), (0.783, 0.196, 0.022)):
    close(got, s, 5e-4)
close(sqrt(6), 2.449, 5e-4)
close(sqrt(3), 1.732, 5e-4)
for got, s in zip(softmax([LN6, LN3, 0], 2), (0.473, 0.334, 0.193)):
    close(got, s, 5e-4)

# §10.3 sigmoid
close(sig(LN3), 0.75, 1e-12)
close(sig(-LN3), 0.25, 1e-12)
close(log(0.75 / 0.25), LN3, 1e-12)
close(sig(LN2) / (1 - sig(LN2)), 2, 1e-12)  # +0.693 doubles the odds (from 1 to 2)

# §11.1 likelihood
close(0.7**7, 0.0824, 5e-5)
eq(round(0.3**3, 10), 0.027)
close(0.5**10, 0.000977, 5e-7)
close(0.0824 * 0.027, 0.00222, 5e-6)
close(0.7**7 * 0.3**3, 0.00222, 5e-6)
close(0.7**7 * 0.3**3 / 0.5**10, 2.3, 0.05)

# §11.2 log-likelihood
close(log(0.7), -0.3567, 5e-5)
close(log(0.3), -1.2040, 5e-5)
close(7 * (-0.3567) + 3 * (-1.2040), -6.109, 5e-4)
close(7 * log(0.7) + 3 * log(0.3), -6.109, 5e-4)
close(10 * (-0.6931), -6.931, 5e-4)
close(10 * log(0.5), -6.931, 5e-4)
close(7 / 0.7 - 3 / 0.3, 0, 1e-12)

# §12.1 surprise
eq(-log2(1 / 2), 1)
eq(-log2(1 / 8), 3)
eq(-log2(1), 0)

# §12.2 / §12.3 / §12.4 entropy, cross-entropy, KL in bits
p3, q3 = (0.5, 0.25, 0.25), (0.25, 0.5, 0.25)
H = -sum(v * log2(v) for v in p3)
CE = -sum(a_ * log2(b_) for a_, b_ in zip(p3, q3))
KL = sum(a_ * log2(a_ / b_) for a_, b_ in zip(p3, q3))
eq(H, 1.5)
eq(-sum(0.25 * log2(0.25) for _ in range(4)), 2)
eq(CE, 1.75)
eq(KL, 0.25)
eq(0.5 * log2(2) + 0.25 * log2(0.5) + 0.25 * log2(1), 0.25)
eq(CE - H, 0.25)
pa_, pb_ = (0.9, 0.1), (0.5, 0.5)
close(sum(x_ * log2(x_ / y_) for x_, y_ in zip(pa_, pb_)), 0.531, 5e-4)
close(sum(x_ * log2(x_ / y_) for x_, y_ in zip(pb_, pa_)), 0.737, 5e-4)

# §12.5 language-model loss and perplexity
pc = (1 / 2, 1 / 4, 1 / 2, 1 / 8)
eq([-log2(v) for v in pc], [1, 2, 1, 3])
eq((1 + 2 + 1 + 3) / 4, 1.75)
close(1.75 * LN2, 1.213, 5e-4)
close(-sum(log(v) for v in pc) / 4, 1.213, 5e-4)
close(2**1.75, 3.364, 5e-4)
close(exp(1.75 * LN2), 3.3636, 5e-5)
close(exp(2), 7.39, 5e-3)

# §13.1 latency
ms = [110, 120, 120, 130, 140, 150, 160, 180, 250, 2400]
eq(sum(ms), 3760)
eq(sum(ms) / 10, 376)
eq(median(ms), 145)
eq(sorted(ms)[ceil(0.9 * 10) - 1], 250)
ok(376 / 145 > 2, "mean more than twice the median")
eq(sum(v <= 250 for v in ms), 9)

# §13.2 sample sd
t8 = [2, 4, 4, 4, 5, 5, 7, 9]
eq(sum(t8) / 8, 5)
eq([(v - 5) ** 2 for v in t8], [9, 1, 1, 1, 0, 0, 4, 16])
eq(sum((v - 5) ** 2 for v in t8), 32)
close(32 / 7, 4.571, 5e-4)
close(sqrt(32 / 7), 2.138, 5e-4)
eq(sqrt(32 / 8), 2)

# §13.3 standard error
close(0.8 * 0.2, 0.16, 1e-12)
close(0.16 / 400, 0.0004, 1e-15)
close(sqrt(0.16 / 400), 0.02, 1e-12)
close(sqrt(0.16 / 1600), 0.01, 1e-12)

# §13.4 normal CI
close(1.96 * 0.02, 0.0392, 1e-12)
close(0.8 - 0.0392, 0.7608, 1e-12)
close(0.8 + 0.0392, 0.8392, 1e-12)
ok(0.7608 < 0.82 < 0.8392, "82% inside the interval")


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c, h


# §13.5 Wilson 0/20
z = 1.96
close(z * z, 3.8416, 1e-12)
close(1 + z * z / 20, 1.19208, 1e-12)
close(z * z / 40, 0.09604, 1e-12)
c0, h0 = wilson(0, 20)
close(0.09604 / 1.19208, 0.0806, 5e-5)
close(c0, 0.0806, 5e-5)
close(sqrt(z * z / (4 * 400)), 0.049, 1e-12)
close(1.96 * 0.049 / 1.19208, 0.0806, 5e-5)
close(h0, 0.0806, 5e-5)
close(c0 - h0, 0, 1e-12)
close(c0 + h0, 0.161, 5e-4)
eq(3 / 20, 0.15)
for k_, n_, lo, hi in ((320, 400, 0.758, 0.836), (18, 20, 0.699, 0.972), (20, 20, 0.839, 1.0)):
    cc, hh = wilson(k_, n_)
    close(cc - hh, lo, 5e-4)
    close(min(1.0, cc + hh), hi, 5e-4)

# §13.6 A/B test
pbar = 632 / 800
close(pbar, 0.79, 1e-12)
close(pbar * (1 - pbar), 0.1659, 1e-12)
se = sqrt(0.1659 * 2 / 400)
close(se, 0.0288, 5e-5)
zab = 0.08 / se
close(zab, 2.78, 5e-3)
close(0.08 / 0.0288, 2.78, 5e-3)
pv = 2 * (1 - N.cdf(zab))
close(pv, 0.0055, 5e-5)
close(300 / 400, 0.75, 1e-12)
close(332 / 400, 0.83, 1e-12)
close(0.75 * 0.25, 0.1875, 1e-12)
close(0.83 * 0.17, 0.1411, 1e-12)
sed = sqrt((0.1875 + 0.1411) / 400)
close(sed, 0.0287, 5e-5)
close(1.96 * 0.0287, 0.056, 5e-4)
close(0.08 - 1.96 * sed, 0.024, 5e-4)
close(0.08 + 1.96 * sed, 0.136, 5e-4)
se100 = sqrt((158 / 200) * (42 / 200) * 2 / 100)
close(0.08 / se100, 2.778 / 2, 1e-3)

# §13.8 sample size
close((1.96 + 0.8416) ** 2, 7.849, 5e-4)
close(0.75 * 0.25 + 0.8 * 0.2, 0.3475, 1e-12)
close((0.80 - 0.75) ** 2, 0.0025, 1e-15)
nss = 7.849 * 0.3475 / 0.0025
close(nss, 1091, 0.5)
za, zb = N.inv_cdf(0.975), N.inv_cdf(0.8)
close(za, 1.96, 5e-4)
close(zb, 0.8416, 5e-5)
n_exact = (za + zb) ** 2 * 0.3475 / 0.0025
eq(ceil(n_exact), 1091)
n_quarter = (za + zb) ** 2 * (0.75 * 0.25 + 0.775 * 0.225) / 0.025**2
close(n_quarter / n_exact, 4, 0.25)

# §13.9 multiple comparisons
close(0.95**20, 0.358, 5e-4)
close(1 - 0.95**20, 0.642, 5e-4)
eq(20 * 0.05, 1)
eq(0.05 / 20, 0.0025)
close(1 - 0.9975**20, 0.049, 5e-4)

# §16 exercises
# 16.1
eq(dot((3, 4), (4, 3)), 24)
eq(norm((3, 4)), 5)
eq(24 / 25, 0.96)
close(2 - 2 * 0.96, 0.08, 1e-12)
close((0.6 - 0.8) ** 2 + (0.8 - 0.6) ** 2, 0.08, 1e-12)
# 16.2
eq(768 * 3072, 2_359_296)
eq(768 * 3072 + 3072, 2_362_368)
# 16.3


def gd5(lr, steps, w=1.0):
    path = [w]
    for _ in range(steps):
        w = w - lr * 2 * (w - 5)
        path.append(w)
    return path


eq(gd5(0.25, 3), [1, 3, 4, 4.5])
eq(2 * (1 - 5), -8)
eq(1 - 2 * 0.25, 0.5)
eq(gd5(1.0, 2), [1, 9, 1])
# 16.4
eq(sig(0) - 1, -0.5)
eq((sig(0) - 1) * 2, -1)
# 16.5
close(0.5 * 0.2, 0.10, 1e-12)
close(0.05 * 0.8, 0.04, 1e-12)
close(0.10 / 0.14, 0.714, 5e-4)
eq((round(1000 * 0.2 * 0.5), round(1000 * 0.8 * 0.05)), (100, 40))
# 16.6
close(0.9 * 1 + 0.1 * 3, 1.2, 1e-12)
close(0.9 * 1 + 0.1 * 9, 1.8, 1e-12)
close(1.8 - 1.2**2, 0.36, 1e-12)
close(sqrt(0.36), 0.6, 1e-12)
close(1000 * 1.2, 1200, 1e-9)
close(0.6 * sqrt(1000), 19, 0.05)
# 16.7
close(3 / 4, 0.75, 1e-12)
close(-log(0.75), 0.288, 5e-4)
close(exp(-log(0.75)), 1.333, 5e-4)
eq(log2(8), 3)
eq(2**3, 8)
# 16.8
close(160 / 200, 0.80, 1e-12)
close(176 / 200, 0.88, 1e-12)
close(336 / 400, 0.84, 1e-12)
close(0.84 * 0.16 * (2 / 200), 0.001344, 1e-12)
se8 = sqrt(0.001344)
close(se8, 0.0367, 5e-5)
close(0.08 / se8, 2.18, 5e-3)
close(2 * (1 - N.cdf(0.08 / se8)), 0.029, 5e-4)
eq(0.05 / 5, 0.01)
# 16.9
eq(3 / 300, 0.01)
close(3 / 0.001, 3000, 1e-9)
# 16.10
ms10 = [100, 100, 110, 120, 120, 130, 140, 150, 160, 1000]
eq(sum(ms10), 2130)
eq(sum(ms10) / 10, 213)
eq(median(ms10), 125)
eq(sorted(ms10)[ceil(0.9 * 10) - 1], 160)

ch46_checks = checks

# ======================================================================
# chapter 47 — machine learning fundamentals
# ======================================================================

# §4.1 test-set size
eq((7000, 1500, 1500), (int(10_000 * 0.70), int(10_000 * 0.15), int(10_000 * 0.15)))
close(0.9 * 0.1, 0.09, 1e-12)
close(sqrt(0.09 / 1500), 0.0077, 5e-5)
close(1.96 * sqrt(0.09 / 1500), 0.015, 5e-4)
close(sqrt(0.09 / 200), 0.0212, 5e-5)
close(1.96 * sqrt(0.09 / 200), 0.042, 5e-4)


# §4.2 expected maximum of 50 standard normals (numeric integration)
def expected_max(m, lo=-9.0, hi=9.0, steps=36_000):
    h = (hi - lo) / steps
    tot = 0.0
    for i in range(steps + 1):
        x = lo + i * h
        f = x * m * N.cdf(x) ** (m - 1) * N.pdf(x)
        tot += f * (1 if i in (0, steps) else (4 if i % 2 else 2))
    return tot * h / 3


emax = expected_max(50)
close(emax, 2.25, 5e-3)
close(2.25 * 0.0077, 0.017, 5e-4)

# §4.3 stratified counts
eq(200 * 0.05, 10)
close(sqrt(200 * 0.05 * 0.95), 3.08, 5e-3)
ok(abs(4 - 10) / 3.08 < 2 and abs(16 - 10) / 3.08 < 2, "4 and 16 are within 2 sd of 10")

# §6.1 majority baseline, §6.2 keyword rule
eq(900 / 1000, 0.9)
eq(0 / 100, 0)
eq(50 / 80, 0.625)
eq(50 / 100, 0.5)
# §6.3 stratified dummy: expected accuracy
close(0.9**2 + 0.1**2, 0.82, 1e-12)

# §7.1 guess w=2, b=0
xs, ys = (1, 2, 3), (2, 3, 7)
eq([2 * x_ for x_ in xs], [2, 4, 6])
eq([2 * x_ - y_ for x_, y_ in zip(xs, ys)], [0, 1, -1])
close((0 + 1 + 1) / 3, 0.667, 5e-4)

# §7.2 closed form
xb, yb = sum(xs) / 3, sum(ys) / 3
eq((xb, yb), (2, 4))
eq([x_ - xb for x_ in xs], [-1, 0, 1])
eq([y_ - yb for y_ in ys], [-2, -1, 3])
sxy = sum((x_ - xb) * (y_ - yb) for x_, y_ in zip(xs, ys))
sxx = sum((x_ - xb) ** 2 for x_ in xs)
eq((sxy, sxx), (5, 2))
wl, bl = sxy / sxx, yb - sxy / sxx * xb
eq((wl, bl), (2.5, -1))
eq([wl * x_ + bl for x_ in xs], [1.5, 4, 6.5])
close(sum((wl * x_ + bl - y_) ** 2 for x_, y_ in zip(xs, ys)) / 3, 0.5, 1e-12)
eq(2.5 * 4 - 1, 9)

# §7.3 normal equations
XtX = [[3, sum(xs)], [sum(xs), sum(x_ * x_ for x_ in xs)]]
eq(XtX, [[3, 6], [6, 14]])
Xty = [sum(ys), sum(x_ * y_ for x_, y_ in zip(xs, ys))]
eq(Xty, [12, 29])
eq(3 * 14 - 6 * 6, 6)
eq((14 * 12 - 6 * 29, -6 * 12 + 3 * 29), (168 - 174, -72 + 87))
eq(((168 - 174) / 6, (-72 + 87) / 6), (-1, 2.5))

# §8.1 sigmoid
close(log(9), 2.197, 5e-4)
eq(sig(0), 0.5)
close(sig(log(9)), 0.9, 1e-12)
close(sig(-log(9)), 0.1, 1e-12)

# §8.2 log-loss on four tickets
losses = [-log(0.9), -log(1 - 0.2), -log(0.6), -log(1 - 0.7)]
for got, s in zip(losses, (0.105, 0.223, 0.511, 1.204)):
    close(got, s, 5e-4)
close(sum(losses), 2.043, 5e-4)
close(sum(losses) / 4, 0.511, 5e-4)
ok(losses[3] > sum(losses[:3]), "confident mistake costs more than the other three")
close(-log(0.01), 4.6, 0.01)

# §8.3 gradient (p - y) x
close(0.9 - 1, -0.1, 1e-12)
close(-0.1 * 2, -0.2, 1e-12)
close(0.7 - 0, 0.7, 1e-12)

# §8.4 truth for the ticket (outage, enterprise, two "!")
close(2 + 1 + 0.5 * 2 - 3, 1, 1e-12)
close(sig(1), 0.731, 5e-4)

# §8.5 cost threshold
close(5 / (5 + 50), 0.091, 5e-4)
eq(1 / (1 + 1), 0.5)


def gini(*counts):
    n = sum(counts)
    return 1 - sum((c / n) ** 2 for c in counts)


# §9.2 Gini
close(gini(6, 4), 0.48, 1e-12)
close(0.6**2, 0.36, 1e-12)
close(0.4**2, 0.16, 1e-12)
eq(gini(5), 0)
close(gini(1, 1), 0.5, 1e-12)

# §9.3 the two candidate splits
close(gini(4, 1), 0.32, 1e-12)
close(gini(2, 3), 0.48, 1e-12)
close(0.5 * 0.32 + 0.5 * 0.48, 0.40, 1e-12)
close(0.48 - 0.40, 0.08, 1e-12)
eq(gini(4, 0), 0)
close(gini(2, 4), 4 / 9, 1e-12)
close(4 / 9, 0.444, 5e-4)
close(0.6 * gini(2, 4), 0.267, 5e-4)
close(0.48 - 0.6 * gini(2, 4), 0.213, 5e-4)

# §9.4 feature importances of the depth-2 tree
d_ent = gini(6, 4) - 0.6 * gini(2, 4)
d_out = 0.6 * (gini(2, 4) - (2 / 6) * gini(1, 1) - (4 / 6) * gini(1, 3))
close(d_ent, 0.2133, 5e-5)
close(gini(2, 4), 0.4444, 5e-5)
close((2 / 6) * gini(1, 1) + (4 / 6) * gini(1, 3), 0.4167, 5e-5)
close(d_out, 0.0167, 5e-5)
close(d_ent + d_out, 0.23, 1e-12)
close(d_ent / 0.23, 0.9275, 5e-5)
close(d_out / 0.23, 0.0725, 5e-5)

# §10.1 variance of an average of correlated trees
for rho, shown in ((0, 0.01), (0.3, 0.307), (1, 1)):
    close(rho * 1 + (1 - rho) / 100 * 1, shown, 1e-12)

# §10.2 one boosting step
F0 = sum(ys) / 3
eq(F0, 4)
res = [y_ - F0 for y_ in ys]
eq(res, [-2, -1, 3])
h1 = [(-2 + -1) / 2, (-2 + -1) / 2, 3]
eq(h1, [-1.5, -1.5, 3])
F1 = [F0 + 0.5 * v for v in h1]
eq(F1, [3.25, 3.25, 5.5])
res1 = [y_ - f_ for y_, f_ in zip(ys, F1)]
eq(res1, [-1.25, -0.25, 1.5])
close(sum(r * r for r in res) / 3, 4.667, 5e-4)
eq(sum(r * r for r in res), 14)
close(sum(r * r for r in res1), 3.875, 1e-12)
close(3.875 / 3, 1.292, 5e-4)

# §11.1 k-NN distances from q = (1, 1)
pts = {"P1": ((1, 2), "B"), "P2": ((3, 1), "T"), "P3": ((4, 5), "T"),
       "P4": ((1, 4), "B"), "P5": ((2, 2), "T"), "P6": ((5, 1), "B")}
dist = {k: sqrt((p[0] - 1) ** 2 + (p[1] - 1) ** 2) for k, (p, _) in pts.items()}
eq(dist["P1"], 1)
close(dist["P5"], sqrt(2), 1e-12)
close(sqrt(2), 1.414, 5e-4)
eq(dist["P2"], 2)
eq(dist["P4"], 3)
eq(dist["P6"], 4)
eq(dist["P3"], sqrt(9 + 16))
eq(dist["P3"], 5)
order = sorted(dist, key=dist.get)
eq(order, ["P1", "P5", "P2", "P4", "P6", "P3"])


def vote(k):
    labs = [pts[p][1] for p in order[:k]]
    return max(set(labs), key=labs.count), labs.count("T"), labs.count("B")


eq(vote(1)[0], "B")
eq(vote(3), ("T", 2, 1))
eq(vote(5), ("B", 2, 3))

# §12 k-means by hand
P6 = [(1, 1), (1, 2), (2, 1), (8, 8), (8, 9), (9, 8)]


def d2(a_, b_):
    return (a_[0] - b_[0]) ** 2 + (a_[1] - b_[1]) ** 2


def mean(ps):
    return (sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps))


c1, c2 = (1, 1), (2, 1)
eq((d2((1, 2), c1), d2((1, 2), c2)), (1, 2))
eq((d2((8, 8), c1), d2((8, 8), c2)), (98, 85))
ok(d2((8, 9), c2) < d2((8, 9), c1) and d2((9, 8), c2) < d2((9, 8), c1), "(8,9),(9,8) closer to c2")
A1 = [p for p in P6 if d2(p, c1) <= d2(p, c2)]
B1 = [p for p in P6 if d2(p, c1) > d2(p, c2)]
eq(A1, [(1, 1), (1, 2)])
c1, c2 = mean(A1), mean(B1)
eq(c1, (1, 1.5))
eq(c2, (6.75, 6.5))
in1 = sum(d2(p, c1) for p in A1)
in2 = sum(d2(p, c2) for p in B1)
eq(in1, 0.5)
eq(in2, 71.75)
eq(in1 + in2, 72.25)
eq(d2((2, 1), c1), 1.25)
close(d2((2, 1), c2), 52.8, 0.05)
A2 = [p for p in P6 if d2(p, c1) <= d2(p, c2)]
B2 = [p for p in P6 if d2(p, c1) > d2(p, c2)]
eq(A2, [(1, 1), (1, 2), (2, 1)])
c1, c2 = mean(A2), mean(B2)
close(c1[0], 4 / 3, 1e-12)
close(c2[0], 25 / 3, 1e-12)
close(sum(d2(p, c1) for p in A2), 12 / 9, 1e-12)
close(sum(d2(p, c2) for p in B2), 12 / 9, 1e-12)
close(24 / 9, 2.667, 5e-4)
A3 = [p for p in P6 if d2(p, c1) <= d2(p, c2)]
eq(A3, A2)  # converged
close(sum(d2(p, mean(P6)) for p in P6), 149.667, 5e-4)  # sklearn k=1 inertia

# §13.1 / §13.2 naive Bayes
eq(F(2 + 1, 5 + 7), F(3, 12))
close(3 / 12, 0.25, 1e-12)
close(1 / 12, 0.083, 5e-4)
spam_c = {"free": 2, "refund": 1, "now": 1, "money": 1}
ham_c = {"refund": 1, "my": 1, "order": 2, "status": 1}
eq(sum(spam_c.values()), 5)
eq(sum(ham_c.values()), 5)
eq(len(set(spam_c) | set(ham_c)), 7)


def pw(c, w):
    return F(c.get(w, 0) + 1, 12)


s_spam = F(1, 2) * pw(spam_c, "free") * pw(spam_c, "refund")
s_ham = F(1, 2) * pw(ham_c, "free") * pw(ham_c, "refund")
eq(s_spam, F(1, 2) * F(6, 144))
eq(s_ham, F(1, 2) * F(2, 144))
eq(s_spam / (s_spam + s_ham), F(3, 4))
eq(pw(spam_c, "free") * pw(spam_c, "order"), pw(ham_c, "free") * pw(ham_c, "order"))  # 'free order' ties

# §14.3 bias-variance
for preds, bias2, var, tot in (((11, 12, 13), 4, F(2, 3), 5.667), ((6, 10, 14), 0, F(32, 3), 11.667)):
    mu = F(sum(preds), 3)
    eq((mu - 10) ** 2, bias2)
    eq(sum((F(p_) - mu) ** 2 for p_ in preds) / 3, var)
    close(float(bias2 + var + 1), tot, 5e-4)
close(2 / 3, 0.667, 5e-4)
close(32 / 3, 10.667, 5e-4)

# §14.2 noise floor
close(0.3**2, 0.09, 1e-12)

# §15.1 ridge and lasso in one dimension
eq(sum(x_ * y_ for x_, y_ in zip(xs, ys)), 29)
eq(2 + 6 + 21, 29)
eq(sum(x_ * x_ for x_ in xs), 14)
close(29 / 14, 2.071, 5e-4)
close(29 / 28, 1.036, 5e-4)
close((29 - 14 / 2) / 14, 1.571, 5e-4)
eq(29 - 7, 22)
eq(2 * 29, 58)
close((29 / 28) / (29 / 14), 0.5, 1e-12)

# §15.2 ridge shrinks the weight vector as a whole (printed, rounded coefficients)
w_none = (3.00, -2.22, 1.52, 0.19, 0.19, -0.03, 0.15, -0.16)
w_r10 = (2.39, -1.80, 1.22, 0.03, 0.22, 0.01, 0.14, -0.02)
w_r100 = (0.87, -0.69, 0.48, -0.15, 0.17, 0.07, 0.10, 0.12)
close(norm(w_none), 4.0, 0.05)
close(norm(w_r10), 3.2, 0.05)
close(norm(w_r100), 1.2, 0.05)
ok(w_r10[4] > w_none[4], "a useless weight grows under ridge 10")
ok(w_none[3] * w_r100[3] < 0 and w_none[7] * w_r100[7] < 0, "useless weights flip sign under ridge 100")

# §16.1 five-fold CV
folds = [0.82, 0.85, 0.80, 0.84, 0.84]
close(sum(folds), 4.15, 1e-12)
close(sum(folds) / 5, 0.83, 1e-12)
devs = [round(f_ - 0.83, 10) for f_ in folds]
eq(devs, [-0.01, 0.02, -0.03, 0.01, 0.01])
close(sum(d_ * d_ for d_ in devs), 0.0016, 1e-12)
close(0.0016 / 4, 0.0004, 1e-15)
close(sqrt(0.0004), 0.02, 1e-12)
eq(1000 // 5, 200)
eq(1000 - 200, 800)

# §17.1 number of fits
eq(4 * 3 * 5, 60)
eq(60 * 5 + 1, 301)
eq(20 * 5 + 1, 101)
eq(4 * 3, 12)  # the GridSearchCV example tries 12 combinations

# §18.1-18.4 confusion matrix metrics
TP, FN, FP, TN = 80, 20, 40, 860
eq(TP + FN, 100)
eq(FP + TN, 900)
eq(TP + FP, 120)
eq(FN + TN, 880)
eq((TP + TN) / 1000, 0.94)
Pm, Rm = TP / (TP + FP), TP / (TP + FN)
close(Pm, 0.667, 5e-4)
eq(Rm, 0.8)
close(FP / 900, 0.044, 5e-4)
close(2 * Pm * Rm, 1.0667, 5e-5)
close(Pm + Rm, 1.4667, 5e-5)
close(2 * Pm * Rm / (Pm + Rm), 0.727, 5e-4)
eq(F(2 * TP, 2 * TP + FP + FN), F(160, 220))
close(5 * Pm * Rm, 2.667, 5e-4)
close(4 * Pm + Rm, 3.467, 5e-4)
close(5 * Pm * Rm / (4 * Pm + Rm), 0.769, 5e-4)
close(1.25 * Pm * Rm, 0.667, 5e-4)
close(0.25 * Pm + Rm, 0.967, 5e-4)
close(1.25 * Pm * Rm / (0.25 * Pm + Rm), 0.690, 5e-4)
eq(99 / 100, 0.99)  # "with 1% positives, 'never' scores 0.99"

# §18.5 threshold table
sc = [0.9, 0.8, 0.7, 0.6, 0.4, 0.2]
lab = [1, 1, 0, 1, 0, 0]
for t, flagged, tp_, fp_, prec, rec in ((0.85, 1, 1, 0, 1.00, 0.33), (0.75, 2, 2, 0, 1.00, 0.67),
                                        (0.65, 3, 2, 1, 0.67, 0.67), (0.50, 4, 3, 1, 0.75, 1.00),
                                        (0.30, 5, 3, 2, 0.60, 1.00), (0.10, 6, 3, 3, 0.50, 1.00)):
    sel = [l_ for s_, l_ in zip(sc, lab) if s_ >= t]
    eq(len(sel), flagged)
    eq(sum(sel), tp_)
    eq(len(sel) - sum(sel), fp_)
    close(sum(sel) / len(sel), prec, 5e-3)
    close(sum(sel) / 3, rec, 5e-3)

# §18.6 AUC by pairs and by trapezoids
pos = [s_ for s_, l_ in zip(sc, lab) if l_]
neg = [s_ for s_, l_ in zip(sc, lab) if not l_]
eq(len(pos) * len(neg), 9)
eq([sum(p_ > n_ for n_ in neg) for p_ in pos], [3, 3, 2])
eq(F(8, 9), F(sum(p_ > n_ for p_ in pos for n_ in neg), 9))
close(8 / 9, 0.889, 5e-4)
eq(F(1, 3) * F(2, 3) + F(1, 3) * 1 + F(1, 3) * 1, F(8, 9))

# §18.7 average precision
close((1 / 3) * (1 + 1 + 0.75), 0.917, 5e-4)
eq(3 / 4, 0.75)
eq(3 / 6, 0.5)


# expected AP of a random order of 3 positives among 6 (all 20 placements equally likely)
def ap_ranks(ranks):
    return sum(F(i + 1, r) for i, r in enumerate(sorted(ranks))) / len(ranks)


from itertools import combinations  # noqa: E402

eq(ap_ranks((1, 2, 4)), F(11, 12))  # the card's AP, 0.917
rand_ap = [ap_ranks(c) for c in combinations(range(1, 7), 3)]
eq(len(rand_ap), 20)
eq(sum(rand_ap) / 20, F(129, 200))
close(float(sum(rand_ap) / 20), 0.645, 1e-12)

# §18.9 Brier
br = [(0.9 - 1) ** 2, (0.8 - 1) ** 2, (0.3 - 0) ** 2, (0.6 - 0) ** 2]
for got, s in zip(br, (0.01, 0.04, 0.09, 0.36)):
    close(got, s, 1e-12)
close(sum(br), 0.5, 1e-12)
close(sum(br) / 4, 0.125, 1e-12)
eq((0.5 - 1) ** 2, 0.25)

# §19 regression metrics
yr, yh = (2, 4, 6, 8), (3, 4, 4, 9)
er = [h_ - y_ for y_, h_ in zip(yr, yh)]
eq(er, [1, 0, -2, 1])
eq(sum(abs(v) for v in er) / 4, 1.0)
eq(sum(v * v for v in er) / 4, 1.5)
close(sqrt(1.5), 1.225, 5e-4)
eq(sum(yr) / 4, 5)
eq(sum((y_ - 5) ** 2 for y_ in yr), 20)
eq(sum(v * v for v in er), 6)
close(1 - 6 / 20, 0.7, 1e-12)

# §20.2 class weights
close(1000 / (2 * 900), 0.556, 5e-4)
eq(1000 / (2 * 100), 5)
close(900 * 1000 / (2 * 900), 500, 1e-9)
eq(100 * 5, 500)
close(5 / (1000 / 1800), 9, 1e-9)
close(900 * (1000 / 1800), 500, 1e-9)


# §20.3 prior correction
def correct(p_, pi, pis):
    o = p_ / (1 - p_) * (pi / (1 - pi)) / (pis / (1 - pis))
    return o / (1 + o)


close((0.1 / 0.9) / (0.5 / 0.5), 1 / 9, 1e-12)
close(correct(0.5, 0.1, 0.5), 0.1, 1e-12)
close(correct(0.9, 0.1, 0.5), 0.5, 1e-12)
close(0.9 / 0.1, 9, 1e-12)

# §21.1 scaling
age = (100, 200, 300)
sd_pop = sqrt(sum((a_ - 200) ** 2 for a_ in age) / 3)
close(sd_pop, 81.65, 5e-3)
for got, s in zip([(a_ - 200) / sd_pop for a_ in age], (-1.225, 0, 1.225)):
    close(got, s, 5e-4)
eq([(a_ - 100) / 200 for a_ in age], [0, 0.5, 1])

# §21.3 TF-IDF
idf_r = log(4 / 3) + 1
idf_o = log(4 / 2) + 1
close(idf_r, 1.2877, 5e-5)
close(idf_o, 1.6931, 5e-5)
close(idf_r**2, 1.6581, 5e-5)
close(idf_o**2, 2.8667, 5e-5)
nrm = sqrt(idf_r**2 + idf_o**2)
close(nrm, 2.1272, 5e-5)
close(idf_r / nrm, 0.6053, 5e-5)
close(idf_o / nrm, 0.7960, 5e-5)

# §21.4 cyclical hours
close(sin(pi / 12), 0.2588, 5e-5)
close(cos(pi / 12), 0.9659, 5e-5)
close(sin(2 * pi * 23 / 24), -0.2588, 5e-5)
close(cos(2 * pi * 23 / 24), 0.9659, 5e-5)
close(sin(2 * pi * 1 / 24), 0.2588, 5e-5)
close(2 * 0.2588, 0.518, 5e-4)
close(sqrt((sin(2 * pi * 23 / 24) - sin(2 * pi / 24)) ** 2 + (cos(2 * pi * 23 / 24) - cos(2 * pi / 24)) ** 2), 0.518, 5e-4)
close(sin(2 * pi * 6 / 24), 1, 1e-12)
close(cos(2 * pi * 6 / 24), 0, 1e-12)
close(sin(2 * pi * 18 / 24), -1, 1e-12)
eq(23 - 1, 22)
close(sin(2 * pi * 23.5 / 24), -0.1305, 5e-5)  # the 23:30 example in code
close(cos(2 * pi * 23.5 / 24), 0.9914, 5e-5)

# §22.3 accuracy ceiling with 5% random labels over 4 classes
close(0.95 + 0.05 / 4, 0.9625, 1e-12)

# §24.2 cost comparison
eq(3_000_000 * 600, 1_800_000_000)
close(1800 * 0.50, 900, 1e-9)
eq(3_000_000 * 10, 30_000_000)
eq(30 * 2, 60)
eq(900 + 60, 960)
close(3_000_000 * 0.2 / 1000, 600, 1e-9)
close(266 * 6.74 / 14.8, 120, 2)  # llama.cpp pp512, 7B (6.74B) Q4_0 on a 16-core M1 Pro, scaled
eq(600 / 120, 5)
eq(3_000_000 * 5, 15_000_000)
eq(30 * 24 * 3600, 2_592_000)
close(15_000_000 / 2_592_000, 5.8, 0.05)

# §25.1 router
close(0.7 * 0.10 + 0.3 * 1.00, 0.37, 1e-12)
close(1 - 0.37, 0.63, 1e-12)

# §27 exercises
# 27.1
eq(40 / 50, 0.80)
close(40 / 60, 0.667, 5e-4)
close(2 * 0.8 * (40 / 60) / (0.8 + 40 / 60), 0.727, 5e-4)
eq(F(80, 80 + 10 + 20), F(80, 110))
# 27.2
pos2, neg2 = (0.8, 0.4), (0.6, 0.3, 0.1)
eq([sum(p_ > n_ for n_ in neg2) for p_ in pos2], [3, 2])
close(5 / 6, 0.833, 5e-4)
close(0.5 * (1 + 2 / 3), 0.833, 5e-4)
# 27.3
close(gini(8, 2), 0.32, 1e-12)
close(gini(3, 2), 0.48, 1e-12)
close(0.5 * 0 + 0.5 * 0.48, 0.24, 1e-12)
close(0.32 - 0.24, 0.08, 1e-12)
# 27.4
x4, y4 = (1, 2, 3, 4), (1, 3, 3, 5)
xb4, yb4 = sum(x4) / 4, sum(y4) / 4
eq((xb4, yb4), (2.5, 3))
sxy4 = sum((a_ - xb4) * (b_ - yb4) for a_, b_ in zip(x4, y4))
sxx4 = sum((a_ - xb4) ** 2 for a_ in x4)
eq((sxy4, sxx4), (6, 5))
w4 = sxy4 / sxx4
eq(w4, 1.2)
b4 = yb4 - w4 * xb4
close(b4, 0, 1e-12)
pr4 = [w4 * a_ + b4 for a_ in x4]
for got, s in zip(pr4, (1.2, 2.4, 3.6, 4.8)):
    close(got, s, 1e-12)
r4 = [b_ - p_ for b_, p_ in zip(y4, pr4)]
for got, s in zip(r4, (-0.2, 0.6, -0.6, 0.2)):
    close(got, s, 1e-12)
close(sum(v * v for v in r4), 0.8, 1e-12)
close(sum(v * v for v in r4) / 4, 0.2, 1e-12)
eq(sum((b_ - yb4) ** 2 for b_ in y4), 8)
close(1 - 0.8 / 8, 0.9, 1e-12)
# 27.5
close(-log(0.8), 0.223, 5e-4)
close(-log(0.6), 0.511, 5e-4)
close((-log(0.8) - log(0.6)) / 2, 0.367, 5e-4)
close(0.8 - 1, -0.2, 1e-12)
eq(0.4 - 0, 0.4)
# 27.6
close(2 / 42, 0.048, 5e-4)
# 27.7
eq(2000 // 5, 400)
eq(2000 - 400, 1600)
eq(4 * 5, 20)
eq(20 * 5 + 1, 101)
# 27.8
eq(10_000 / (2 * 200), 25)
close(10_000 / (2 * 9800), 0.510, 5e-4)
close(25 / (10_000 / (2 * 9800)), 49, 1e-9)
eq(9800 / 200, 49)
# 27.10
eq(1_000_000 * 500, 500_000_000)
eq(500 * 1, 500)
eq(1_000_000 * 5, 5_000_000)
eq(5 * 4, 20)
eq(500 + 20, 520)
close(0.2 * 520, 104, 1e-9)
close(520 - 104, 416, 1e-9)

print(f"ch46_47_math_ml: all {checks} checks passed "
      f"(chapter 46: {ch46_checks}, chapter 47: {checks - ch46_checks})")
