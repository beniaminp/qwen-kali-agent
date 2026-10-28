#!/usr/bin/env python3
"""Check every number in the "On paper" cards, worked examples and exercise
answers of docs/48-ai-deep-learning.md and docs/49-ai-llm-internals.md (stdlib only)."""
from collections import Counter
from math import ceil, cos, erf, exp, log, pi, radians, sin, sqrt, tanh

checks = 0


def close(a, b, tol):
    global checks
    checks += 1
    assert abs(a - b) <= tol, f"{a!r} != {b!r} (tol {tol})"


def eq(a, b):
    global checks
    checks += 1
    assert a == b, f"{a!r} != {b!r}"


def ok(cond, msg=""):
    global checks
    checks += 1
    assert cond, msg


def sig(z):
    return 1 / (1 + exp(-z))


def Phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def softmax(z, t=1.0):
    w = [exp(v / t) for v in z]
    s = sum(w)
    return [x / s for x in w]


def relu(v):
    return max(0.0, v)


E = exp(1)
LN2 = log(2)
close(LN2, 0.693, 5e-4)
close(E, 2.718, 5e-4)
close(exp(2), 7.389, 5e-4)

# ============================================================== chapter 48
# §1.1 neuron: x = (2, 1), w = (1.5, 1), b = -2
z = 1.5 * 2 + 1 * 1 - 2
eq(z, 2.0)
close(exp(-2), 0.135, 5e-4)
close(1 + exp(-2), 1.135, 5e-4)
close(sig(2), 0.881, 5e-4)
close(sig(-2), 0.119, 5e-4)
for (x1, x2), shown in (((2, 1), 0.881), ((0, 1), 0.269), ((1, 0), 0.378)):
    close(sig(1.5 * x1 + 1.0 * x2 - 2), shown, 5e-4)   # code output table

# §1.2 linear collapse and the XOR network
eq(2 * 3, 6)
def xor_net(x1, x2):
    h1, h2 = relu(x1 + x2), relu(x1 + x2 - 1)
    return h1, h2, h1 - 2 * h2
eq(xor_net(1, 1), (2.0, 1.0, 0.0))
eq(xor_net(0, 1), (1.0, 0.0, 1.0))
eq(xor_net(0, 0)[2], 0.0)
eq(xor_net(1, 0)[2], 1.0)

# §2.1 sigmoid and tanh
eq(sig(0), 0.5)
close(sig(0) * (1 - sig(0)), 0.25, 1e-12)
close(sig(2) * (1 - sig(2)), 0.105, 5e-4)
close(0.881 * 0.119, 0.105, 5e-4)
close(exp(-1), 0.368, 5e-4)
close(2.718 - 0.368, 2.350, 1e-9)
close(2.718 + 0.368, 3.086, 1e-9)
close(2.350 / 3.086, 0.7615, 5e-5)
close(tanh(1), 0.762, 5e-4)
close(1 - tanh(1) ** 2, 0.420, 5e-4)
close(1 - 0.7615 ** 2, 0.420, 5e-4)
eq(1 / 0.25, 4.0)  # tanh slope at 0 is four times the sigmoid's

# §2.2 ReLU on (-2, 3)
eq([relu(-2), relu(3)], [0.0, 3.0])

# §2.3 GELU and SiLU
close(Phi(1), 0.841, 5e-4)
close(Phi(-1), 0.159, 5e-4)
close(sig(1), 0.731, 5e-4)
close(sig(-1), 0.269, 5e-4)
close(1 * Phi(1), 0.841, 5e-4)
close(-1 * Phi(-1), -0.159, 5e-4)
close(1 * sig(1), 0.731, 5e-4)
silu_d = lambda x: sig(x) * (1 + x * (1 - sig(x)))
close(silu_d(1), 0.928, 5e-4)
close(0.731 * (1 + 1 * 0.269), 0.928, 5e-4)
# the code table (selected entries)
close(-2 * Phi(-2), -0.046, 5e-4)
close(2 * Phi(2), 1.954, 5e-4)
close(-2 * sig(-2), -0.238, 5e-4)
close(2 * sig(2), 1.762, 5e-4)
gelu_d = lambda x: Phi(x) + x * exp(-x * x / 2) / sqrt(2 * pi)
close(gelu_d(-2), -0.085, 5e-4)
close(gelu_d(1), 1.083, 5e-4)
close(silu_d(-2), -0.091, 5e-4)
close(silu_d(-1), 0.072, 5e-4)

# §3 forward pass: x = (1, 2), W1 = [[0.5, 0.25], [-0.5, 1]], b1 = (0, 0.5), w2 = (1, -0.5), b2 = 0
x = (1.0, 2.0)
W1 = [[0.5, 0.25], [-0.5, 1.0]]
b1 = [0.0, 0.5]
w2 = [1.0, -0.5]
b2 = 0.0
y = 1.0


def forward(W1, b1, w2, b2):
    z1 = [W1[j][0] * x[0] + W1[j][1] * x[1] + b1[j] for j in range(2)]
    h = [relu(v) for v in z1]
    z2 = w2[0] * h[0] + w2[1] * h[1] + b2
    p = sig(z2)
    L = -(y * log(p) + (1 - y) * log(1 - p))
    return z1, h, z2, p, L


z1, h, z2, p, L = forward(W1, b1, w2, b2)
eq(z1, [1.0, 2.0])
eq(h, [1.0, 2.0])
eq(z2, 0.0)
eq(p, 0.5)
close(L, 0.6931, 5e-5)
eq(4 + 2 + 2 + 1, 9)

# §4.1 MSE
errs = [3 - 2, 5 - 7]
eq(errs, [1, -2])
mse = sum(e * e for e in errs) / 2
eq(mse, 2.5)
close(sqrt(mse), 1.58, 5e-3)
close(sqrt(mse) * 100, 158, 0.5)

# §4.2 BCE
close(-log(0.8), 0.223, 5e-4)
close(-log(0.8), 0.2231, 5e-5)
close(-log(0.2), 1.609, 5e-4)
close(-log(0.2), 1.6094, 5e-5)
close(-log(0.001), 6.91, 5e-3)
close(log(0.8 / 0.2), log(4), 1e-12)

# §4.3 categorical CE, logits (ln 2, 0, 0)
sm = softmax([LN2, 0, 0])
for a, b in zip(sm, (0.5, 0.25, 0.25)):
    close(a, b, 1e-12)
close(-log(sm[0]), 0.693, 5e-4)
close(-log(sm[1]), 1.386, 5e-4)
close(-log(sm[1]), 1.3863, 5e-5)
# §4.4 CE on probabilities (the bug)
sm2 = softmax(sm)
close(-log(sm2[0]), 0.9391, 5e-5)
close(-log(sm2[0]), 0.939, 5e-4)

# §5.1 dL/dz2
close(-1 / p, -2.0, 1e-12)
close(p * (1 - p), 0.25, 1e-12)
dz2 = p - y
eq(dz2, -0.5)
eq(-2 * 0.25, dz2)

# §5.2 every partial derivative
g_w2 = [dz2 * hi for hi in h]
eq(g_w2, [-0.5, -1.0])
g_b2 = dz2
eq(g_b2, -0.5)
g_h = [dz2 * wi for wi in w2]
eq(g_h, [-0.5, 0.25])
g_z1 = [g_h[j] * (1.0 if z1[j] > 0 else 0.0) for j in range(2)]
eq(g_z1, [-0.5, 0.25])
g_W1 = [[g_z1[j] * x[i] for i in range(2)] for j in range(2)]
eq(g_W1, [[-0.5, -1.0], [0.25, 0.5]])
g_b1 = g_z1
eq(g_b1, [-0.5, 0.25])

# finite differences confirm all nine (central, eps = 1e-6)
params = {"W1": W1, "b1": b1, "w2": w2}
def loss_with(W1_, b1_, w2_, b2_):
    return forward(W1_, b1_, w2_, b2_)[4]
eps = 1e-6
for j in range(2):
    for i in range(2):
        Wp = [r[:] for r in W1]; Wm = [r[:] for r in W1]
        Wp[j][i] += eps; Wm[j][i] -= eps
        close((loss_with(Wp, b1, w2, b2) - loss_with(Wm, b1, w2, b2)) / (2 * eps), g_W1[j][i], 1e-8)
    bp, bm = b1[:], b1[:]
    bp[j] += eps; bm[j] -= eps
    close((loss_with(W1, bp, w2, b2) - loss_with(W1, bm, w2, b2)) / (2 * eps), g_b1[j], 1e-8)
    wp, wm = w2[:], w2[:]
    wp[j] += eps; wm[j] -= eps
    close((loss_with(W1, b1, wp, b2) - loss_with(W1, b1, wm, b2)) / (2 * eps), g_w2[j], 1e-8)
close((loss_with(W1, b1, w2, eps) - loss_with(W1, b1, w2, -eps)) / (2 * eps), g_b2, 1e-8)

# §5.3 one gradient step, lr = 0.1
lr = 0.1
w2n = [w - lr * g for w, g in zip(w2, g_w2)]
for a, b in zip(w2n, (1.05, -0.4)):
    close(a, b, 1e-12)
b2n = b2 - lr * g_b2
close(b2n, 0.05, 1e-12)
W1n = [[W1[j][i] - lr * g_W1[j][i] for i in range(2)] for j in range(2)]
for a, b in zip(sum(W1n, []), (0.55, 0.35, -0.525, 0.95)):
    close(a, b, 1e-12)
b1n = [b - lr * g for b, g in zip(b1, g_b1)]
for a, b in zip(b1n, (0.05, 0.475)):
    close(a, b, 1e-12)
z1n, hn, z2n, pn, Ln = forward(W1n, b1n, w2n, b2n)
close(z1n[0], 1.30, 1e-12)
close(z1n[1], 1.85, 1e-12)
close(1.05 * 1.3 - 0.4 * 1.85 + 0.05, 0.675, 1e-12)
close(z2n, 0.675, 1e-12)
close(pn, 0.6626, 5e-5)
close(Ln, 0.412, 5e-4)
close(Ln, 0.4116, 5e-5)
close(-log(0.6626), 0.412, 5e-4)

# §5.4 finite difference on b2 with eps = 0.01
Lp, Lm = log(1 + exp(-0.01)), log(1 + exp(0.01))
close(Lp, 0.68816, 5e-6)
close(Lm, 0.69816, 5e-6)
close((Lp - Lm) / 0.02, -0.5, 1e-6)
close((0.68816 - 0.69816) / 0.02, -0.5, 1e-9)

# §6.1 graph: L = d^2, d = ab + c, a = 2, b = -3, c = 10
a_, b_, c_ = 2.0, -3.0, 10.0
d_ = a_ * b_ + c_
eq(d_, 4.0)
eq(d_ * d_, 16.0)
dLd = 2 * d_
eq(dLd, 8.0)
eq(dLd * b_, -24.0)
eq(dLd * a_, 16.0)
eq(dLd * 1, 8.0)

# §8.1 SGD
close(1 - 0.1 * 0.5, 0.95, 1e-12)
# §8.2 momentum, mu = 0.9, lr = 0.1, g = 0.5 twice
v1 = 0.5
th1 = 1 - 0.1 * v1
v2 = 0.9 * v1 + 0.5
th2 = th1 - 0.1 * v2
close(th1, 0.95, 1e-12)
close(v2, 0.95, 1e-12)
close(0.1 * v2, 0.095, 1e-12)
close(th2, 0.855, 1e-12)
close(1 / (1 - 0.9), 10, 1e-9)
# §8.3 Adam, one step
m1 = 0.1 * 0.5
vv1 = 0.001 * 0.25
close(m1, 0.05, 1e-12)
close(vv1, 0.00025, 1e-15)
mh = m1 / (1 - 0.9)
vh = vv1 / (1 - 0.999)
close(mh, 0.5, 1e-12)
close(vh, 0.25, 1e-9)
close(sqrt(vh), 0.5, 1e-9)
close(1 - 0.001 * mh / sqrt(vh), 0.999, 1e-9)
# §8.4 AdamW
close(0.001 * 0.01 * 1, 0.00001, 1e-15)
close(1 - 0.001 - 0.00001, 0.99899, 1e-12)
close((1 - 0.001 * 0.01) * 1 - 0.001, 0.99899, 1e-12)   # PyTorch's order gives the same
close((1 - 1e-5) ** 1e4, exp(-0.1), 1e-5)
close(exp(-0.1), 0.905, 5e-4)
# §8.5 / §9.1 the divergence cliff
close(2 / 100, 0.02, 1e-12)
for lr_ in (0.019, 0.021):
    ok((abs(1 - lr_ * 100) < 1) == (lr_ < 0.02))

# §9.1 gradient descent on w^2
def gd(lr_, n):
    w, out = 1.0, []
    for _ in range(n):
        w = (1 - 2 * lr_) * w
        out.append(w)
    return out
for a, b in zip(gd(0.1, 2), (0.8, 0.64)):
    close(a, b, 1e-12)
for a, b in zip(gd(1.1, 2), (-1.2, 1.44)):
    close(a, b, 1e-12)
for a, b in zip(gd(0.9, 5), (-0.8, 0.64, -0.512, 0.4096, -0.32768)):
    close(a, b, 1e-12)
ok(abs(1 - 2 * 1.0) >= 1)            # eta = 1 is already the cliff
eq(gd(1.0, 4), [-1.0, 1.0, -1.0, 1.0])   # eta = 1 bounces between -1 and +1 forever
ok(abs(1 - 2 * 1.01) > 1)            # above 1 it grows: divergence
close(854_000, 8.54e5, 1e3)           # "about 854,000" quoted from the run

# §9.2 warmup + cosine
eta_max, eta_min, Tw, T = 3e-4, 3e-5, 100, 1000
def sched(t):
    if t < Tw:
        return eta_max * t / Tw
    s = (t - Tw) / (T - Tw)
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * s))
close(sched(50), 1.5e-4, 1e-15)
close((550 - Tw) / (T - Tw), 0.5, 1e-12)
close(0.5 * (eta_max - eta_min), 1.35e-4, 1e-15)
close(sched(550), 1.65e-4, 1e-12)
close(sched(1000), 3e-5, 1e-15)
close(sched(325), 2.605e-4, 5e-8)
close(sched(775), 6.954e-5, 5e-9)

# §10.1 steps arithmetic
eq(ceil(50_000 / 32), 1563)
close(50_000 / 32, 1562.5, 1e-12)
eq(1563 * 3, 4689)
eq(4 * 8, 32)
eq(50_000 // 4, 12_500)
eq(8 * 2048, 16_384)
# §10.2 gradient noise
close(1 / sqrt(4), 0.5, 1e-12)
close(0.960 / 1.922, 0.4995, 5e-5)
for a, b in ((1.922, 0.960), (0.960, 0.478), (0.478, 0.232)):
    close(b / a, 0.5, 0.02)

# §11.2 Xavier and He
close(sqrt(2 / 1024), 0.0442, 5e-5)
close(sqrt(1 / 512), sqrt(2 / 1024), 1e-15)
eq(sqrt(2 / 512), 1 / 16)
close(1 / 16, 0.0625, 1e-15)
eq(512 / 2, 256)
eq(sqrt(256), 16)
close(16 ** 5, 1.05e6, 1e4)
# std-1 stack in fp16: layer-4 outputs have std about 8.56e5 / 16 = 53,500, so values
# two standard deviations out already pass fp16's max: overflow at layer 4 (measured)
close(8.56e5 / 16, 53_500, 1)
ok(2 * 8.56e5 / 16 > 65504, "std-1 stack passes fp16's max at layer 4")
ok(8.56e5 / 16 ** 2 * 2 < 65504, "but not yet at layer 3")

# §12.1 BatchNorm on (-1, 3, 1)
col = [-1.0, 3.0, 1.0]
mu = sum(col) / 3
eq(mu, 1.0)
var = sum((v - mu) ** 2 for v in col) / 3
close(var, 8 / 3, 1e-12)
close(var, 2.667, 5e-4)
close(sqrt(var), 1.633, 5e-4)
for a, b in zip([(v - mu) / sqrt(var) for v in col], (-1.225, 1.225, 0.0)):
    close(a, b, 5e-4)
# §12.2 LayerNorm and RMSNorm on (-1, -1, 7, 7)
xv = [-1.0, -1.0, 7.0, 7.0]
m = sum(xv) / 4
eq(m, 3.0)
dev = [v - m for v in xv]
eq(dev, [-4.0, -4.0, 4.0, 4.0])
var = sum(d * d for d in dev) / 4
eq(var, 16.0)
eq([d / sqrt(var) for d in dev], [-1.0, -1.0, 1.0, 1.0])
ms = sum(v * v for v in xv) / 4
eq(ms, 25.0)
for a, b in zip([v / sqrt(ms) for v in xv], (-0.2, -0.2, 1.4, 1.4)):
    close(a, b, 1e-12)
# code row 2 (3, 5, 3, 5): RMS sqrt(17)
close(3 / sqrt(17), 0.728, 5e-4)
close(5 / sqrt(17), 1.213, 5e-4)

# §13.1 dropout
hv = [2, 4, 6, 8]
mask = [1, 0, 1, 0]
eq([mi * hi / 0.5 for mi, hi in zip(mask, hv)], [4.0, 0.0, 12.0, 0.0])
eq(sum(hv) / 4, 5.0)
eq((4 + 0 + 12 + 0) / 4, 4.0)
# §13.2 weight decay
close(1 - 0.1 * 0.5, 0.95, 1e-12)
close(0.95 * 2, 1.9, 1e-12)
close(2 * 0.95 ** 10, 1.197, 5e-4)
close(1e-4 * 0.1, 1e-5, 1e-18)
# §13.3 early stopping numbers quoted from the run
close(1.536 / 0.564, 2.72, 5e-3)      # "almost tripled"
eq(300 - 49, 251)                     # "about 250 epochs" saved
eq(49 - 29, 20)                       # patience

# §14.1 products of many factors
close(0.25 ** 10, 9.5e-7, 5e-9)
close(1.5 ** 20, 3325, 0.5)
# §14.3 clipping
nrm = sqrt(3 ** 2 + 4 ** 2)
eq(nrm, 5.0)
for a, b in zip((3 * 1 / nrm, 4 * 1 / nrm), (0.6, 0.8)):
    close(a, b, 1e-12)
close(sqrt(0.3 ** 2 + 0.4 ** 2), 0.5, 1e-12)
close(1.0 / (5.0 + 1e-6) * 3, 0.6, 5e-5)   # PyTorch's 1e-6 still prints 0.6000

# §15.1 floating point
eq(2 ** (30 - 15) * (1 + 1023 / 1024), 65504.0)
eq(32_768 * 2047 / 1024, 32 * 2047)
eq(32 * 2047, 65_504)
close(2 ** -10, 0.000977, 5e-7)
close(2 ** -7, 0.0078, 5e-5)
close(2 ** 127 * (2 - 2 ** -7), 3.4e38, 0.05e38)
eq(2 ** (16 - 7), 512)
close(70000 / 512, 136.7, 0.05)
eq(round(70000 / 512) * 512, 70144)
eq(137 * 512, 70144)
eq(2 ** 16, 65536)
ok(65536 < 70000 < 2 ** 17)
close(2 ** -24, 6e-8, 0.05e-7)   # smallest fp16 subnormal, "about 6e-8"
close(2 ** -14, 6.1e-5, 0.05e-5)  # smallest fp16 normal, "6e-5"
close(1 + 2 ** -10, 1.00098, 5e-6)
close(1.0011717677116394e-08, 1.00117e-08, 5e-14)

# §16.1 training memory
eq(2 + 2 + 4 + 4 + 4, 16)
eq(16 * 1e9 / 1e9, 16.0)
close(16 * 14.77, 236.3, 0.05)
close(2 * 14.77, 29.5, 0.05)
close(236.3 / 29.5, 8, 0.02)
eq(ceil(236.3 / 80), 3)
close(12 * 14.77, 177, 0.5)     # 37 §1.4
close(4 * 14.77, 59.1, 0.05)
# §16.2 activations per example
per = 1024 + 4096 + 4096 + 1024
eq(per, 10_240)
eq(per * 4, 40 * 1024)
eq(512 * 40 * 1024, 20 * 2 ** 20)
close(8 * 40 * 1024 / 2 ** 20, 0.31, 0.005)
close(64 * 40 * 1024 / 2 ** 20, 2.50, 0.005)
eq((1024 * 4096 + 4096 + 4096 * 1024 + 1024) * 4 / 2 ** 20 > 32.0, True)  # 32.0 MiB weights
close((1024 * 4096 + 4096 + 4096 * 1024 + 1024) * 4 / 2 ** 20, 32.0, 0.05)
# §17 what fits
gib_budget_gb = 21.3 * 2 ** 30 / 1e9
close(gib_budget_gb, 22.9, 0.05)
ok(29.5 > gib_budget_gb and 29.5 > 24 and 8.3 < 24)

# §18.1 starting loss
close(log(3), 1.099, 5e-4)
close(log(151_936), 11.93, 5e-3)
close(log(10), 2.303, 5e-4)

# §19.1 convolution
sig_ = [1, 2, 4, 7, 7]
ker = [-1, 1]
eq([sum(ker[j] * sig_[i + j] for j in range(2)) for i in range(4)], [1, 2, 3, 0])
eq(3 * 3 * 64 * 64 + 64, 36_928)
eq(64 * 32 * 32 * 64 * 30 * 30, 3_774_873_600)
close(3_774_873_600 / 1e9, 3.8, 0.05)
# §19.2 RNN
h1 = tanh(1)
close(h1, 0.762, 5e-4)
close(0.5 * h1, 0.381, 5e-4)
h2 = tanh(0.5 * h1)
close(h2, 0.363, 5e-4)
close(0.5 * h2, 0.182, 5e-4)
h3 = tanh(0.5 * h2)
close(h3, 0.180, 5e-4)
close(tanh(0.5 * h3), 0.090, 5e-4)
eq(4 * (64 * 16 + 64 * 64 + 64 + 64), 20_992)
# §20 embedding lookup
table = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
onehot = (0, 0, 1)
eq(tuple(sum(onehot[r] * table[r][c] for r in range(3)) for c in range(2)), (0.5, 0.6))
eq(151_936 * 5_120, 777_912_320)
close(777_912_320 * 2 / 1e9, 1.56, 0.005)

# chapter 48 exercises
close(1 * 2 - 2 * 0.75 + 0.5, 1.0, 1e-12)                       # 1
close(1 + exp(-1), 1.368, 5e-4)
close(sig(1), 0.731, 5e-4)
close(-log(1 - 0.5), 0.693, 5e-4)                                 # 2
eq([0.5 * hi for hi in h], [0.5, 1.0])
close(-0.5 + 2 - 3, -1.5, 1e-12)                                  # 3
eq(1e-3 * 1000, 1.0)                                              # 4
close(1e-3 * 0.001, 1e-6, 1e-18)
eq(120_000 / 64, 1875.0)                                          # 5
eq(1875 * 2, 3750)
close(0.03 * 3750, 112.5, 1e-9)
eq(ceil(0.03 * 3750), 113)
eq(16 * 3, 48)                                                    # 6
eq(2 * 3, 6)
close(80_000 / 512, 156.25, 1e-12)                                # 7
eq(156 * 512, 79_872)
eq(80_000 - 79_872, 128)
close(128 / 80_000 * 100, 0.16, 1e-9)
ok(80_000 > 65_504)
close(sqrt(36 + 64 + 0), 10.0, 1e-12)                             # 8
for a, b in zip([6 * 0.2, 8 * 0.2, 0.0], (1.2, 1.6, 0.0)):
    close(a, b, 1e-12)
close(sqrt(1.2 ** 2 + 1.6 ** 2), 2.0, 1e-12)
close(23.0 / log(10), 10, 0.02)                                   # 9: ten times ln 10

# ============================================================== chapter 49
# §1.2 sentence probability
close(0.5 * 0.4 * 0.9, 0.18, 1e-12)

# §1.3 bigram probabilities quoted in the output
close((3 + 0.1) / (3 + 0.8), 0.82, 5e-3)
close((2 + 0.1) / (3 + 0.8), 0.55, 5e-3)
close((1 + 0.1) / (2 + 0.8), 0.39, 5e-3)


# §2.2 BPE: replay the whole training run
def bpe(corpus, n_merges, end="_"):
    words = {tuple(w) + ((end,) if end else ()): n for w, n in corpus.items()}
    merges = []
    for _ in range(n_merges):
        counts = Counter()
        for s, n in words.items():
            for pair in zip(s, s[1:]):
                counts[pair] += n
        best = max(counts, key=counts.get)
        merges.append((best, counts[best], dict(counts)))
        new = {}
        for s, n in words.items():
            out, i = [], 0
            while i < len(s):
                if s[i:i + 2] == best:
                    out.append(s[i] + s[i + 1]); i += 2
                else:
                    out.append(s[i]); i += 1
            new[tuple(out)] = n
        words = new
    return merges, words


corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
merges, words = bpe(corpus, 8)
first = merges[0][2]
eq(first[("e", "s")], 9)
eq(first[("w", "e")], 8)
eq(first[("l", "o")], 7)
eq(sorted(p for p, c in first.items() if c == 9), [("e", "s"), ("s", "t"), ("t", "_")])
eq([(a + b, c) for (a, b), c, _ in merges],
   [("es", 9), ("est", 9), ("est_", 9), ("lo", 7), ("low", 7), ("ne", 6), ("new", 6), ("newest_", 6)])
sixth = merges[5][2]
eq(sorted(p for p, c in sixth.items() if c == 6), sorted([("n", "e"), ("e", "w"), ("w", "est_")]))
eq(5 * 4 + 2 * 6 + 6 * 7 + 3 * 7, 95)
eq(sum(len(s) * n for s, n in words.items()), 36)
eq(5 * 2 + 2 * 4 + 6 * 1 + 3 * 4, 36)
eq(len(set("".join(corpus)) | {"_"}), 11)
eq(11 + 8, 19)

# §2.3 tiktoken table ratios quoted in the prose
close(45 / 11, 4.09, 5e-3)
close(23 / 11, 2.09, 5e-3)

# §2.4 cost example (assumed prices $3 / $15 per million)
tin, tout = 100_000 * 2_000, 100_000 * 300
eq(tin, 200_000_000)
eq(tout, 30_000_000)
eq(tin / 1e6 * 3, 600.0)
eq(tout / 1e6 * 15, 450.0)
eq(600 + 450, 1050)
eq(100_000 * 500 / 1e6 * 3, 150.0)
close(1050 * 23 / 11, 2200, 10)

# §4.2 / §4.3 attention by hand
Xa = [(1, 0, 1, 0), (0, 1, 0, 1), (1, 1, 1, 1)]
swap = lambda v: (v[1], v[0], v[3], v[2])
Qa = [swap(v) for v in Xa]
eq(Qa, [(0, 1, 0, 1), (1, 0, 1, 0), (1, 1, 1, 1)])
Ka = Xa
Va = [(v[0], v[1]) for v in Xa]
eq(Va, [(1, 0), (0, 1), (1, 1)])
raw = [[sum(q[i] * k[i] for i in range(4)) for k in Ka] for q in Qa]
eq(raw, [[0, 2, 2], [2, 0, 2], [2, 2, 4]])
scaled = [[s / sqrt(4) for s in r] for r in raw]
eq(scaled, [[0, 1, 1], [1, 0, 1], [1, 1, 2]])
Wt = [softmax(scaled[t][:t + 1]) + [0.0] * (2 - t) for t in range(3)]
close(Wt[0][0], 1.0, 1e-12)
close(Wt[1][0], E / (E + 1), 1e-12)
close(Wt[1][0], 0.731, 5e-4)
close(Wt[1][1], 0.269, 5e-4)
close(Wt[2][0], 1 / (2 + E), 1e-12)
close(Wt[2][0], 0.212, 5e-4)
close(Wt[2][2], E / (2 + E), 1e-12)
close(Wt[2][2], 0.576, 5e-4)
close(2.718 + 2.718 + 7.389, 12.825, 1e-9)
close(2.718 / 12.825, 0.212, 5e-4)
close(7.389 / 12.825, 0.576, 5e-4)
outs = [tuple(sum(Wt[t][j] * Va[j][c] for j in range(3)) for c in range(2)) for t in range(3)]
close(outs[0][0], 1.0, 1e-12)
close(outs[1][0], 0.731, 5e-4)
close(outs[1][1], 0.269, 5e-4)
close(outs[2][0], 0.788, 5e-4)
close(outs[2][1], 0.788, 5e-4)
close(0.212 + 0.576, 0.788, 1e-12)
close(softmax([0, 1, 1])[1], 0.422, 5e-4)   # unmasked "the" would split ~equally over later words
close(softmax([0, 1, 1])[1], softmax([0, 1, 1])[2], 1e-12)

# §4.5 scaling
close(sqrt(128), 11.3, 0.05)
close(2 * sqrt(128), 22.6, 0.05)
close(1 / (1 + exp(-11.3)), 0.99999, 5e-6)
close(E / (E + 1), 0.731, 5e-4)
# §4.6 T^2 cost
close(40 * 2 ** 22 * 2 / 1e6, 336, 0.5)
close(40 * 2 ** 30 * 2 / 1e9, 85.9, 0.05)
eq(2 ** 11, 2048)
eq(2 ** 15, 32_768)
eq(16 ** 2, 256)

# §5.2 KV per token
kv_tok = 2 * 40 * 8 * 128 * 2
eq(kv_tok, 163_840)
eq(kv_tok, 160 * 1024)
eq(2 * 40 * 40 * 128 * 2, 819_200)
eq(819_200, 800 * 1024)
eq(819_200 // kv_tok, 5)

# §6.2 RoPE with omega = 30 degrees
def rot(v, deg):
    c, s = cos(radians(deg)), sin(radians(deg))
    return (c * v[0] - s * v[1], s * v[0] + c * v[1])
q90, k30 = rot((1, 0), 90), rot((1, 0), 30)
close(q90[0], 0, 1e-12); close(q90[1], 1, 1e-12)
close(k30[0], 0.866, 5e-4); close(k30[1], 0.5, 1e-12)
q150, k90 = rot((1, 0), 150), rot((1, 0), 90)
close(q150[0], -0.866, 5e-4); close(q150[1], 0.5, 1e-12)
dot = lambda u, v: u[0] * v[0] + u[1] * v[1]
close(dot(q90, k30), 0.5, 1e-12)
close(dot(q150, k90), 0.5, 1e-12)
close(cos(radians(60)), 0.5, 1e-12)
close(dot(rot((1, 0), 105 * 30), rot((1, 0), 103 * 30)), 0.5, 1e-9)
close(dot(rot((1, 0), 150), rot((1, 0), 30)), -0.5, 1e-12)   # offset 4
close(cos(radians(120)), -0.5, 1e-12)
gq, gk = (1, 2), (3, -1)
close(dot(rot(gq, 90), rot(gk, 30)), -5.562, 5e-4)
close(dot(rot(gq, 150), rot(gk, 30)), -6.562, 5e-4)

# §7 decoder block parameter count (d=8, H=4, G=2, d_h=2, d_ff=16)
attn = 8 * 4 * 2 + 2 * 8 * 2 * 2 + 4 * 2 * 8
eq(attn, 192)
eq(3 * 8 * 16, 384)
eq(192 + 384 + 16, 592)
close(384 / 592, 2 / 3, 0.02)
close(267.39 / 330.32, 0.81, 5e-3)

# §8.2 / §8.3 loss and perplexity
Lm = (log(2) + log(4) + log(8)) / 3
close(Lm, 2 * LN2, 1e-12)
close(Lm, 1.386, 5e-4)
close(exp(Lm), 4.0, 1e-12)
close(log(151_936), 11.93, 5e-3)
# §8.4 quoted bigram results
close(exp(0.556), 1.74, 5e-3)
close(exp(1.207), 3.34, 5e-3)
ok(32.62 > 8)   # worse than a uniform guess over the 8-word vocabulary
eq(len(set("<s> your order shipped . <s> your refund arrived . <s> your order arrived late .".split())), 8)

# §9.2 temperature
P5 = [0.5, 0.2, 0.15, 0.1, 0.05]
sq = [v * v for v in P5]
close(sum(sq), 0.325, 1e-12)
for a, b in zip([v / sum(sq) for v in sq], (0.769, 0.123, 0.069, 0.031, 0.008)):
    close(a, b, 5e-4)
for a, b in zip(softmax([log(v) for v in P5], 0.5), (0.769, 0.123, 0.069, 0.031, 0.008)):
    close(a, b, 5e-4)
rt = [sqrt(v) for v in P5]
close(sum(rt), 2.081, 5e-4)
for a, b in zip(softmax([log(v) for v in P5], 2.0), (0.340, 0.215, 0.186, 0.152, 0.107)):
    close(a, b, 5e-4)
# §9.3 top-p, top-k, min-p
cum = [sum(P5[:i + 1]) for i in range(5)]
for a, b in zip(cum[:3], (0.5, 0.7, 0.85)):
    close(a, b, 1e-12)
kept = [v for v, c in zip(P5, cum) if c - v < 0.8]
eq(len(kept), 3)
close(sum(kept), 0.85, 1e-12)
for a, b in zip([v / 0.85 for v in kept], (0.588, 0.235, 0.176)):
    close(a, b, 5e-4)
for a, b in zip([0.5 / 0.7, 0.2 / 0.7], (0.714, 0.286)):
    close(a, b, 5e-4)
close(0.25 * 0.5, 0.125, 1e-12)
eq([v for v in P5 if v >= 0.125], [0.5, 0.2, 0.15])
close(0.1 * 0.9, 0.09, 1e-12)
eq([v for v in (0.9, 0.05, 0.03, 0.02) if v >= 0.09], [0.9])
# §9.4 penalties
close(3.0 - 0.5 * 2 - 0.5 * 1, 1.5, 1e-12)
close(exp(1.5), 4.5, 0.02)
close(2.6 / 1.3, 2.0, 1e-12)
close(-1.0 * 1.3, -1.3, 1e-12)
# §9.5 frequency penalty flips greedy
close(3 * 0.4, 1.2, 1e-12)
ok(log(0.5) - 1.2 < log(0.2) < log(0.5))
ok(log(0.5) - 3 * 0.3 > log(0.2))  # 0.3 would not have been enough
# §9.7 logprobs
close(exp(-0.094), 0.910, 5e-4)
close(exp(-2.813), 0.060, 5e-4)
close(exp(-3.912), 0.020, 5e-4)
close(log(0.91), -0.094, 5e-4)
close(log(0.06), -2.813, 5e-4)
close(log(0.02), -3.912, 5e-4)

# §10.1 context budget
prompt = 1_500 + 8 * 600 + 6_000 + 200
eq(8 * 600, 4_800)
eq(prompt, 12_500)
eq(32_768 - prompt, 20_268)
# §10.2 KV cache
eq(kv_tok * 32_768, 5_368_709_120)
close(5_368_709_120 / 1e9, 5.37, 0.005)
eq(5_368_709_120, 5 * 2 ** 30)
eq(4 * kv_tok * 8_192, 5_368_709_120)
# KV-cache demo: projections with and without the cache
eq(2 * 6, 12)
eq(sum(2 * (t + 1) for t in range(6)), 42)
# §10.3 prefill and decode
Cp = 2 * 14.77e9 * 8_000
close(Cp, 2.36e14, 0.005e14)
# llama.cpp discussion 4167: 16-core M1 Pro, LLaMA-2 7B (6.74B params) Q4_0, pp512 = 266 tok/s
close(2 * 6.74e9 * 266, 3.6e12, 0.05e12)
close(3.6e12 / (2 * 14.77e9), 120, 3)
close(3.5e12 / (2 * 14.77e9), 120, 3)
close(Cp / 3.5e12, 68, 0.5)
ok(Cp / 3.5e12 > 60 and 8_000 / 120 > 60, "more than a minute")
close(200 / 8.3, 24, 0.2)
close(8.3 + 5.37, 13.67, 1e-9)
close(200 / 13.67, 14.6, 0.05)
close(300 / 24, 12.5, 1e-12)

# §11.1 parameters times bytes
close(14.77 * 2, 29.5, 0.05)
close(14.77 * 1, 14.8, 0.05)
close(14.77 * 0.5, 7.4, 0.05)
close(4.5 / 8, 0.5625, 1e-12)
close(14.77 * 0.5625, 8.3, 0.05)
eq((64 * 4 + 32) / 64, 4.5)
# the size table
for n, bf16, b8, b45 in ((8.2, 16.4, 8.2, 4.6), (14.8, 29.5, 14.8, 8.3), (32.8, 65.6, 32.8, 18.5), (30.5, 61, 30.5, 17.2)):
    close(n * 2, bf16, 0.11)
    close(n * 1, b8, 1e-9)
    close(n * 0.5625, b45, 0.06)
ok(16.4 < gib_budget_gb and 14.8 < gib_budget_gb and 29.5 > gib_budget_gb)
# §11.2 absmax int8
w = [0.523, -1.27, 0.336, 0.0414]
s = max(abs(v) for v in w) / 127
close(s, 0.01, 1e-12)
qv = [round(v / s) for v in w]
eq(qv, [52, -127, 34, 4])
wh = [q * s for q in qv]
for a, b in zip(wh, (0.52, -1.27, 0.34, 0.04)):
    close(a, b, 1e-12)
er = [abs(a - b) for a, b in zip(w, wh)]
for a, b in zip(er, (0.003, 0.0, 0.004, 0.0014)):
    close(a, b, 1e-9)
ok(max(er) <= s / 2)
close(s / 2, 0.005, 1e-12)
# outlier case
w_out = [0.523, -12.7, 0.336, 0.0414]
s_out = 12.7 / 127
close(s_out, 0.1, 1e-12)
eq([round(v / s_out) for v in w_out], [5, -127, 3, 0])
close(0.01492 / 0.00202, 7.4, 0.05)   # "cuts the error sevenfold"

# §12.1 pretraining compute
close(6 * 14.77e9 * 36e12, 3.19e24, 0.005e24)
# §12.4 test-time compute latency
close(2 + 2_200 / 24, 93.7, 0.05)
close(2 + 200 / 24, 10.3, 0.05)
close((2 + 2_200 / 24) / (2 + 200 / 24), 9.1, 0.05)
# §13 Chinchilla
close(20 * 14.77e9, 295e9, 0.5e9)
close(36e12 / 14.77e9, 2_437, 0.5)
close(36e12 / 14.77e9 / 20, 120, 2.5)
# §15 MoE
close(30.5 * 0.5625, 17.2, 0.05)
close(2 * 3.3e9, 6.6e9, 1e-3)
close(2 * 14.77e9, 2.95e10, 0.005e10)
close(17.2 / 8.3, 2, 0.1)
close(6.6 / 29.54, 0.25, 0.03)
close(3.3 * 0.5625, 1.86, 0.005)
close(200 / (3.3 * 0.5625), 108, 0.5)
# §17.2 config arithmetic (Qwen3-14B)
d, Lq, ff, H, G, dh, V = 5120, 40, 17408, 40, 8, 128, 151_936
attn14 = d * H * dh + 2 * d * G * dh + H * dh * d
mlp14 = 3 * d * ff
close(attn14 / 1e6, 62.91, 0.005)
close(mlp14 / 1e6, 267.39, 0.005)
total14 = Lq * (attn14 + mlp14 + 2 * d + 2 * dh) + 2 * V * d + d
close(total14 / 1e9, 14.77, 0.005)
close(2 * V * d / 1e9, 1.56, 0.005)
eq(H // G, 5)
close(total14 * 16 / 8 / 1e9, 29.5, 0.05)
close(total14 * 8 / 8 / 1e9, 14.8, 0.05)
close(total14 * 4 / 8 / 1e9, 7.4, 0.05)
close(total14 * 4.5 / 8 / 1e9, 8.3, 0.05)
close(V * d / 1e9, 0.78, 0.005)

# chapter 49 exercises
ex_merges, _ = bpe({"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}, 3, end=None)   # 1
c0 = ex_merges[0][2]
eq((c0[("u", "g")], c0[("u", "n")], c0[("p", "u")], c0[("h", "u")]), (20, 16, 17, 15))
eq([(a + b, c) for (a, b), c, _ in ex_merges], [("ug", 20), ("un", 16), ("hug", 15)])
c1 = ex_merges[1][2]
eq((c1[("h", "ug")], c1[("u", "n")], c1[("p", "u")], c1[("p", "ug")]), (15, 16, 12, 5))
c2 = ex_merges[2][2]
eq((c2[("h", "ug")], c2[("p", "un")]), (15, 12))
wts = softmax([0, log(3)])                                                     # 2
for a, b in zip(wts, (0.25, 0.75)):
    close(a, b, 1e-12)
close(0.25 * 4 + 0.75 * 0, 1.0, 1e-12)
close(0.25 * 0 + 0.75 * 8, 6.0, 1e-12)
P4 = [0.4, 0.3, 0.2, 0.1]                                                      # 3
cum4 = [sum(P4[:i + 1]) for i in range(4)]
kept4 = [v for v, c in zip(P4, cum4) if c - v < 0.65]
eq(kept4, [0.4, 0.3])
close(0.4 / 0.7, 0.571, 5e-4)
close(0.3 / 0.7, 0.429, 5e-4)
close(0.4 * 0.4, 0.16, 1e-12)
eq([v for v in P4 if v >= 0.16], [0.4, 0.3, 0.2])
for a, b in zip([v / 0.9 for v in (0.4, 0.3, 0.2)], (0.444, 0.333, 0.222)):
    close(a, b, 5e-4)
sq4 = [v * v for v in P4]
close(sum(sq4), 0.30, 1e-12)
for a, b in zip([v / 0.30 for v in sq4], (0.533, 0.3, 0.133, 0.033)):
    close(a, b, 5e-4)
close(-log(0.25), 1.386, 5e-4)                                                 # 4
close(exp(log(4)), 4, 1e-12)
close(exp(1.0), 2.718, 5e-4)
eq(kv_tok * 8_192, 1_342_177_280)                                              # 5
close(1_342_177_280 / 1e9, 1.34, 0.005)
close(4 * 1_342_177_280 / 1e9, 5.37, 0.005)
close(4 * 1_342_177_280 / 2 / 1e9, 2.68, 0.005)
close(1_342_177_280 / 2 / 1e9, 0.67, 0.005)
close(4 * 1.342, 5.37, 0.005)
close(32.8 * 4.5 / 8, 18.45, 1e-9)                                             # 6
kv32 = 2 * 64 * 8 * 128 * 2
eq(kv32, 262_144)
close(kv32 * 32_768 / 1e9, 8.59, 0.005)
close(18.45 + 8.59, 27.0, 0.05)
close(kv32 * 8_192 / 1e9, 2.15, 0.005)
close(18.45 + 2.15, 20.6, 1e-9)
ok(18.45 + 8.59 > gib_budget_gb)
w7 = [2.54, -0.8, 0.013, 1.0]                                                  # 7
s7 = 2.54 / 127
close(s7, 0.02, 1e-12)
close(0.013 / s7, 0.65, 1e-9)
eq([round(v / s7) for v in w7], [127, -40, 1, 50])
for a, b in zip([round(v / s7) * s7 for v in w7], (2.54, -0.8, 0.02, 1.0)):
    close(a, b, 1e-12)
close(0.02 - 0.013, 0.007, 1e-12)
ok(0.007 < s7 / 2)
close(0.007 / 0.013, 0.54, 5e-3)
close(20 * 7e9, 1.4e11, 1)                                                     # 8
close(6 * 7e9 * 1.4e11, 5.88e21, 1e18)
close(2 + 20 * 4 / 64, 3.25, 1e-12)                                            # 9
eq(2 + 20, 22)
close(22e9 * 4.5 / 8 / 1e9, 12.4, 0.05)
close(exp(-0.22), 0.80, 5e-3)                                                  # 10
close(exp(-1.83), 0.16, 5e-3)

print(f"ch48_49_dl_llm: all {checks} checks passed (chapters 48, 49)")
