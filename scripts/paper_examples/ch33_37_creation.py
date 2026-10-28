#!/usr/bin/env python3
"""Check every number in the "On paper" cards added to chapters 33-37.

Stdlib only. Each close() tolerance matches the rounding shown in the text.
"""
import math

LN2 = math.log(2)


def close(a, b, tol):
    assert abs(a - b) <= tol, f"{a!r} != {b!r} (tol {tol})"


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def phi(x):  # standard normal CDF
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def gelu_tanh(x):
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def rms(v):
    return math.sqrt(sum(x * x for x in v) / len(v))


n = 0  # number of checks


def check(a, b, tol=0.0):
    global n
    close(a, b, tol)
    n += 1


# ---------------------------------------------------------------- 33
# FIM: middle tokens p = 1/2, 1/2, 1/4
fim = -(math.log(0.5) + math.log(0.5) + math.log(0.25))
check(fim, 4 * LN2, 1e-12)
check(fim, 2.77, 0.005)
check(4 * 0.693, 2.77, 0.005)  # reader using ln2 ~ 0.693 gets the same 3 s.f.

# GRPO: R = (1,0,1,0), population std, eps = 0.2, beta = 0.04, KL = 0.25
R = [1, 0, 1, 0]
mean = sum(R) / 4
std = math.sqrt(sum((x - mean) ** 2 for x in R) / 4)
check(mean, 0.5)
check(std, 0.5)
A = [(x - mean) / std for x in R]
assert A == [1, -1, 1, -1]
r = [1.5, 0.5, 1.0, 1.0]
eps, beta, kl = 0.2, 0.04, 0.25
clipped = [min(max(x, 1 - eps), 1 + eps) for x in r]
for got, want in zip(clipped, [1.2, 0.8, 1.0, 1.0]):
    check(got, want, 1e-12)
terms = [min(ri * ai, ci * ai) for ri, ci, ai in zip(r, clipped, A)]
for got, want in zip(terms, [1.2, -0.8, 1.0, -1.0]):
    check(got, want, 1e-12)
# rows 1 and 2 pick the clipped (constant) value
assert terms[0] == clipped[0] * A[0] and terms[1] == clipped[1] * A[1]
avg = sum(terms) / 4
check(avg, 0.1, 1e-12)
J = avg - beta * kl
check(J, 0.09, 1e-12)
check(r[0] - 1, 0.5)  # "50% more likely"

# KD: teacher (1/2,1/4,1/4), student (1/4,1/2,1/4)
pt = [0.5, 0.25, 0.25]
ps = [0.25, 0.5, 0.25]
kd = sum(p * math.log(p / q) for p, q in zip(pt, ps))
check(kd, 0.25 * LN2, 1e-12)
check(kd, 0.173, 0.0005)
check(0.25 * 0.693, 0.173, 0.0005)
swapped = [ps[1], ps[0], ps[2]]
check(sum(p * math.log(p / q) for p, q in zip(pt, swapped)), 0.0, 1e-12)


# GDN: S_t = a S (I - b k k^T) + b v k^T, o = S^T q (2x2, all diagonal)
def matmul(X, Y):
    return [[sum(X[i][k] * Y[k][j] for k in range(2)) for j in range(2)] for i in range(2)]


S0 = [[4.0, 0.0], [0.0, 2.0]]
k = [1.0, 0.0]
v = [6.0, 0.0]
a_t, b_t = 0.5, 0.5
I_bkk = [[(1.0 if i == j else 0.0) - b_t * k[i] * k[j] for j in range(2)] for i in range(2)]
assert I_bkk == [[0.5, 0.0], [0.0, 1.0]]
decayed = [[a_t * x for x in row] for row in matmul(S0, I_bkk)]
assert decayed == [[1.0, 0.0], [0.0, 1.0]]
write = [[b_t * v[i] * k[j] for j in range(2)] for i in range(2)]
assert write == [[3.0, 0.0], [0.0, 0.0]]
S1 = [[decayed[i][j] + write[i][j] for j in range(2)] for i in range(2)]
assert S1 == [[4.0, 0.0], [0.0, 1.0]]
q = [1.0, 0.0]
o_T = [sum(S1[i][j] * q[i] for i in range(2)) for j in range(2)]  # S^T q
o_N = [sum(S1[i][j] * q[j] for j in range(2)) for i in range(2)]  # S q
assert o_T == o_N == [4.0, 0.0]
# "decayed 4 -> 2, then moved halfway to 6"; e2 slot decayed 2 -> 1
check(a_t * 4, 2.0)
check(2 + b_t * (6 - 2), 4.0)
check(a_t * 2, 1.0)
n += 8

# ---------------------------------------------------------------- 34
# Block with post-norm: x=(1,1), attn=(3,-1), mlp=(2,6)
x = [1.0, 1.0]
check(rms(x), 1.0, 1e-12)
u = [x[0] + 3, x[1] - 1]
assert u == [4.0, 0.0]
vv = [u[0] + 2, u[1] + 6]
assert vv == [6.0, 6.0]
check(rms(vv), 6.0, 1e-12)
y = [c / rms(vv) for c in vv]
assert y == [1.0, 1.0]
n += 3

# Local mask: W=4, i=6 -> visible j = 3..6 ; W=1024, i=5000 -> 3977..5000
W, i = 4, 6
assert i - W + 1 == 3
vis = [j for j in range(1, 7) if not (j < i - W + 1)]
assert vis == [3, 4, 5, 6]
W, i = 1024, 5000
assert i - W + 1 == 3977
assert len(range(3977, 5001)) == 1024
n += 4

# K=V: L=1, G=2, d_h=64, T=4096, b=2
kv = 1 * 2 * 64 * 4096 * 2
assert kv == 1_048_576 == 2 ** 20  # 1 MiB
assert 2 * kv == 2 * 2 ** 20  # 2 MiB without K=V
n += 2

# p-RoPE: d_h=8, p=0.25, q=(1,0,2,...), angle 1 rad
assert 0.25 * 8 == 2
c1, s1 = math.cos(1.0), math.sin(1.0)
check(c1, 0.540, 0.0005)
check(s1, 0.841, 0.0005)
qt = [1 * c1 - 0 * s1, 1 * s1 + 0 * c1] + [2] * 6
check(qt[0], 0.540, 0.0005)
check(qt[1], 0.841, 0.0005)
assert 0.25 * 512 == 128 and 512 - 128 == 384
n += 1

# Patch embedder
assert 48 * 48 * 3 == 6912
w_patch = 3840 * 6912
assert w_patch == 26_542_080
e2d = 1120 * 2 * 3840
assert e2d == 8_601_600
check((w_patch + e2d) / 1e6, 35.1, 0.05)
check(w_patch / (w_patch + e2d), 0.75, 0.01)   # "about three quarters"
check(e2d / (w_patch + e2d), 0.25, 0.01)       # "one quarter"
small = 3840 + 2 * 6912 + 2 * 3840 + 2 * 3840  # patch bias, ln1, ln2, pos_norm
check(small / 1e6, 0.03, 0.005)
n += 3

# Audio
assert 16_000 / 640 == 25
assert 30 * 25 == 750
w_audio = 3840 * 640
assert w_audio == 2_457_600
check(w_audio / 1e6, 2.46, 0.005)
assert w_audio / 305e6 < 0.01
n += 4

# Next-token loss: p = 1/2, 1/8, 1/2, 1/8
ps4 = [0.5, 0.125, 0.5, 0.125]
L4 = -sum(math.log(p) for p in ps4) / 4
check(L4, 2 * LN2, 1e-12)
check(L4, 1.386, 0.0005)
check(2 * 0.693, 1.386, 0.0005)
check(math.exp(L4), 4.0, 1e-9)


# ---------------------------------------------------------------- 35
# 6ND: 10M on 1B tokens, M1 Pro 1.5 TFLOP/s
C10 = 6 * 1e7 * 1e9
check(C10, 6e16, 1)
sec = C10 / 1.5e12
check(sec, 40_000, 1e-6)
check(sec / 3600, 11.1, 0.05)  # "about 11 hours"
C125 = 6 * 125e6 * 20e9
check(C125, 1.5e19, 1)
check(C125 / C10, 250, 1e-9)


# Parameter count, tiny-100m with V=64,000 (Q and O are d x d each)
def dense_params(L, d, G, dh, dff, V, untied):
    layer = 2 * d * d + 2 * d * G * dh + 3 * d * dff
    return layer, L * layer + (2 if untied else 1) * V * d


assert 2 * 768 ** 2 == 1_179_648
assert 2 * 768 * 4 * 64 == 393_216
assert 3 * 768 * 2048 == 4_718_592
layer, P = dense_params(12, 768, 4, 64, 2048, 64_000, False)
assert layer == 6_291_456
assert 12 * layer == 75_497_472
assert 64_000 * 768 == 49_152_000
assert P == 124_649_472
check(P / 1e6, 124.6, 0.05)
check(P / 1e6, 125, 0.5)  # the table's ~125M
_, P32 = dense_params(12, 768, 4, 64, 2048, 32_000, False)
check(P32 / 1e6, 100, 0.5)
check(49_152_000 / P, 0.39, 0.005)
# The same convention reproduces the other two rows of the 35 table (V=64,000).
check(dense_params(24, 1024, 4, 64, 2816, 64_000, False)[1] / 1e6, 336, 0.5)
check(dense_params(24, 2048, 4, 128, 5632, 64_000, False)[1] / 1e9, 1.2, 0.05)
n += 7

# Scaling law, toy constants
E, A_, B_, al, be = 1.7, 5000, 50_000, 0.5, 0.5


def Lnd(N, D):
    return E + A_ / N ** al + B_ / D ** be


check(5000 / 1e4, 0.5, 1e-12)
check(50_000 / 5e4, 1.0, 1e-12)
check(Lnd(1e8, 2.5e9), 3.20, 1e-9)
check(A_ / 4e8 ** al, 0.25, 1e-12)
check(Lnd(4e8, 2.5e9), 2.95, 1e-9)
check(B_ / 1e10 ** be, 0.5, 1e-12)
check(Lnd(1e8, 1e10), 2.70, 1e-9)
check(6 * 4e8 * 2.5e9 / (6 * 1e8 * 2.5e9), 4, 1e-12)
check(6 * 1e8 * 1e10 / (6 * 1e8 * 2.5e9), 4, 1e-12)

# SFT: completion p = 1/4, 1/2
sft = -(math.log(0.25) + math.log(0.5))
check(sft, 3 * LN2, 1e-12)
check(sft, 2.08, 0.005)
check(3 * 0.693, 2.08, 0.005)

# DPO: beta = 0.5, ratios 2 and 1/2
check(-math.log(sigmoid(0)), LN2, 1e-12)
check(LN2, 0.693, 0.0005)
z = 0.5 * (math.log(2) - math.log(0.5))
check(z, LN2, 1e-12)
check(sigmoid(z), 2 / 3, 1e-12)
dpo = -math.log(sigmoid(z))
check(dpo, math.log(1.5), 1e-12)
check(dpo, 0.405, 0.0005)

# ---------------------------------------------------------------- 36
# Qwen3-14B parameter count
d, G, dh, dff, L, V = 5120, 8, 128, 17_408, 40, 151_936
check(2 * d * d / 1e6, 52.43, 0.005)
check(2 * d * G * dh / 1e6, 10.49, 0.005)
check(3 * d * dff / 1e6, 267.39, 0.005)
layer, P14 = dense_params(L, d, G, dh, dff, V, True)
check(layer / 1e6, 330.30, 0.005)
check(L * layer / 1e9, 13.21, 0.005)
check(2 * V * d / 1e9, 1.56, 0.005)
check(P14 / 1e9, 14.77, 0.005)
check(2 * V * d / P14, 0.105, 0.0005)        # "about 10.5%"
check(L * 3 * d * dff / P14, 0.72, 0.005)    # "72%"

# GPU-hours on A100: 1B model, 20B tokens, 312 TFLOP/s dense, 35% MFU, $2/h
C1 = 6 * 1e9 * 2e10
check(C1, 1.2e20, 1)
per_h = 312e12 * 0.35 * 3600
check(per_h / 1e17, 3.93, 0.005)
gpu_h = C1 / per_h
check(gpu_h, 305, 0.5)
check(C1 / 3.93e17, 305, 0.5)  # with the rounded per-hour figure
check(gpu_h * 2, 610, 1)
check(gpu_h / 8, 38, 0.5)
C14 = 6 * 14.77e9 * 36e12
check(C14 / C1 / 1e3, 27, 0.5)  # "about 27,000 times"

# Wall time from measured throughput: 4,000 tok/s/GPU x 1,000 GPUs
rate = 4000 * 1000
check(rate, 4e6, 1e-9)
t_ch = 280e9 / rate
check(t_ch, 70_000, 1e-6)
check(t_ch / 3600, 19.4, 0.05)
t_q = 36e12 / rate
check(t_q, 9e6, 1e-3)
check(t_q / 86400, 104, 0.5)
check(t_q / 86400 / (365.25 / 12), 3.5, 0.1)  # "about three and a half months"
check((1 / 0.8) - 1, 0.25, 1e-12)              # 20% lower throughput -> 25% longer
assert 20 * 14e9 == 280e9                      # Chinchilla tokens from section 2.3

# KD with zeros: teacher (1/2,1/2,0,0), student uniform
pt4 = [0.5, 0.5, 0.0, 0.0]
ps4u = [0.25] * 4
kd4 = sum(p * math.log(p / q) for p, q in zip(pt4, ps4u) if p > 0)
check(kd4, LN2, 1e-12)
check(kd4, 0.693, 0.0005)
check(sum(p * math.log(p / q) for p, q in zip(pt4, [0.5, 0.5, 1e-300, 1e-300]) if p > 0), 0.0, 1e-12)

# ---------------------------------------------------------------- 37
# Pre-norm block: X=(2,2), attn=(5,-1), mlp=(-1,1)
X = [2.0, 2.0]
check(rms(X), 2.0, 1e-12)
n1 = [c / rms(X) for c in X]
assert n1 == [1.0, 1.0]
U = [X[0] + 5, X[1] - 1]
assert U == [7.0, 1.0]
check(rms(U), 5.0, 1e-12)
n2 = [c / rms(U) for c in U]
check(n2[0], 1.4, 1e-12)
check(n2[1], 0.2, 1e-12)
check(rms(n2), 1.0, 1e-12)
Y = [U[0] - 1, U[1] + 1]
assert Y == [6.0, 2.0]
n += 3

# SwiGLU / GeGLU, one hidden unit: gate +-1, up 2, W_d 1
check(math.e, 2.718, 0.0005)
check(sigmoid(1), 0.7311, 0.00005)
check(phi(1), 0.8413, 0.00005)
check(1 * sigmoid(1) * 2, 1.462, 0.0005)
check(0.7311 * 2, 1.462, 0.0005)
check(1 * phi(1) * 2, 1.683, 0.0005)
check(0.8413 * 2, 1.683, 0.0005)
check(-1 * sigmoid(-1), -0.2689, 0.00005)
check(-1 * phi(-1), -0.1587, 0.00005)
check(-1 * sigmoid(-1) * 2, -0.538, 0.0005)
check(-0.2689 * 2, -0.538, 0.0005)
check(-1 * phi(-1) * 2, -0.317, 0.0005)
check(-0.1587 * 2, -0.317, 0.0005)
# GELU passes more of a positive gate and less of a negative one
assert phi(1) > sigmoid(1) and abs(phi(-1)) < abs(sigmoid(-1))
# tanh approximation agrees to three decimals at +-1
check(round(gelu_tanh(1), 3), round(phi(1), 3), 0)
check(round(gelu_tanh(-1), 3), round(-phi(-1), 3), 0)

# MoE: Qwen3-30B-A3B
d, H, G, dh, Lm, Ex, k_act, w, V = 2048, 32, 4, 128, 48, 128, 8, 768, 151_936
assert H * dh == 4096
attn = 2 * d * H * dh + 2 * d * G * dh
check(attn / 1e6, 18.87, 0.005)
check(Lm * attn / 1e9, 0.906, 0.0005)
router = d * Ex
check(router / 1e6, 0.26, 0.005)
check(Lm * router / 1e9, 0.013, 0.0005)
experts = Ex * 3 * d * w
check(experts / 1e6, 603.98, 0.005)
check(Lm * experts / 1e9, 28.99, 0.005)
embed = 2 * V * d
check(embed / 1e9, 0.622, 0.0005)
active_exp = k_act / Ex * Lm * experts
check(active_exp / 1e9, 1.812, 0.0005)
check(8 / 128 * 28.99, 1.812, 0.0005)
active = Lm * attn + Lm * router + active_exp + embed
check(active / 1e9, 3.35, 0.005)
check(0.906 + 0.013 + 1.812 + 0.622, 3.35, 0.005)
total = Lm * attn + Lm * router + Lm * experts + embed
check(total / 1e9, 30.5, 0.05)

print(f"ch33_37_creation: all {n} checks passed")
