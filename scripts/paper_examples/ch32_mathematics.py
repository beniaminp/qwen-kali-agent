#!/usr/bin/env python3
"""Check every number in the "On paper" cards of docs/32-mathematics.md (stdlib only)."""
import math

checks = 0


def close(a, b, tol):
    """Assert |a-b| <= tol and count the check."""
    global checks
    assert abs(a - b) <= tol, f"{a!r} != {b!r} (tol {tol})"
    checks += 1


def eq(a, b):
    global checks
    assert a == b, f"{a!r} != {b!r}"
    checks += 1


def softmax(z):
    m = max(z)
    ex = [math.exp(v - m) for v in z]
    s = sum(ex)
    return [v / s for v in ex]


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
            for i in range(len(A))]


def rot(theta, v):
    c, s = math.cos(theta), math.sin(theta)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1])


# ---- 2.1 BPE: "ababcab"
text = "ababcab"
pairs = {}
for a, b in zip(text, text[1:]):
    pairs[a + b] = pairs.get(a + b, 0) + 1
eq(pairs, {"ab": 3, "ba": 1, "bc": 1, "ca": 1})
eq(max(pairs, key=pairs.get), "ab")
merged = text.replace("ab", "Z")
eq(merged, "ZZcZ")
eq(len(text), 7)
eq(len(merged), 4)
eq(256 + 1, 257)

# ---- 2.1 compression ratio
close(7 / 4, 1.75, 1e-12)
close(7 / 7, 1.0, 1e-12)
eq(2000 // 4, 500)

# ---- 2.2 softmax over V=4, z=(2,0,0,0)
close(math.exp(2), 7.389, 5e-4)
close(math.exp(2) + 3, 10.389, 5e-4)
p = softmax([2, 0, 0, 0])
close(p[0], 0.711, 5e-4)
close(p[1], 0.096, 5e-4)
close(1 / 10.389, 0.096, 5e-4)

# ---- 3 factorisation
close(0.5 * 0.8 * 0.25, 0.1, 1e-12)

# ---- 3 NLL
nll = -(math.log(0.5) + math.log(0.8) + math.log(0.25)) / 3
close(-math.log(0.1), 2.303, 5e-4)
close(math.log(10), 2.303, 5e-4)
close(2.303 / 3, 0.768, 5e-4)
close(nll, 0.768, 5e-4)
close(-math.log(0.5), 0.693, 5e-4)
close(-math.log(0.8), 0.223, 5e-4)
close(-math.log(0.25), 1.386, 5e-4)
close(math.exp(nll), 2.15, 5e-3)

# ---- 3 cross-entropy
lse = math.log(math.exp(2) + 3)
close(math.log(10.389), 2.341, 5e-4)
close(lse, 2.341, 5e-4)
close(-2 + lse, 0.341, 5e-4)
close(-2 + 2.341, 0.341, 1e-9)
close(-math.log(0.711), 0.341, 5e-4)
close(0 + lse, 2.341, 5e-4)

# ---- 3.1 SFT masked loss
sft = -(math.log(0.5) + math.log(0.25)) / 2
close(sft, 1.5 * math.log(2), 1e-12)
close(sft, 1.04, 5e-3)
close(-(math.log(0.5) + math.log(0.25)), 2.079, 5e-4)
close(2.079 / 5, 0.42, 5e-3)

# ---- 4 residual block
x = (2, -1)
u = (x[0] + 0.5, x[1] + 0.5)
eq(u, (2.5, -0.5))
y = (u[0] - 1, u[1] + 1)
eq(y, (1.5, 0.5))

# ---- 5 LayerNorm on (1,3,3,9)
x = [1, 3, 3, 9]
mu = sum(x) / 4
eq(mu, 4)
dev = [v - mu for v in x]
eq(dev, [-3, -1, -1, 5])
var = sum(d * d for d in dev) / 4
eq(var, 9)
ln = [d / math.sqrt(var) for d in dev]
for a, b in zip(ln, [-1, -0.333, -0.333, 1.667]):
    close(a, b, 5e-4)
close(sum(ln) / 4, 0, 1e-12)
close(sum(v * v for v in ln) / 4, 1, 1e-12)
for xs in ([v + 10 for v in x], [2 * v for v in x]):
    m2 = sum(xs) / 4
    s2 = math.sqrt(sum((v - m2) ** 2 for v in xs) / 4)
    for a, b in zip([(v - m2) / s2 for v in xs], ln):
        close(a, b, 1e-12)

# ---- 5 RMSNorm on (1,3,3,9)
ms = sum(v * v for v in x) / 4
eq(ms, 25)
rms_out = [v / math.sqrt(ms) for v in x]
for a, b in zip(rms_out, [0.2, 0.6, 0.6, 1.8]):
    close(a, b, 1e-12)
close(math.sqrt(sum(v * v for v in rms_out) / 4), 1, 1e-12)

# ---- 5 QK-norm
q = [60, 0, 0, 80]
k = [30, 0, 0, 40]
eq(sum(a * b for a, b in zip(q, k)), 5000)
close(5000 / math.sqrt(4), 2500, 1e-12)
rq = math.sqrt(sum(v * v for v in q) / 4)
rk = math.sqrt(sum(v * v for v in k) / 4)
eq((3600 + 6400) / 4, 2500)
close(rq, 50, 1e-12)
close(rk, 25, 1e-12)
qt = [v / rq for v in q]
kt = [v / rk for v in k]
for a, b in zip(qt, [1.2, 0, 0, 1.6]):
    close(a, b, 1e-12)
for a, b in zip(kt, [1.2, 0, 0, 1.6]):
    close(a, b, 1e-12)
dot = sum(a * b for a, b in zip(qt, kt))
close(dot, 4, 1e-12)
close(dot / 2, 2, 1e-12)
close(math.sqrt(sum(v * v for v in qt)), 2, 1e-12)  # length sqrt(d_h)

# ---- 6 single-query attention
q = [1, 1, 0, 0]
K = [[1, 1, 0, 0], [0, 0, 1, 1]]
Vv = [[10, 0], [0, 10]]
scores = [sum(a * b for a, b in zip(q, kr)) for kr in K]
eq(scores, [2, 0])
scaled = [s / math.sqrt(4) for s in scores]
eq(scaled, [1, 0])
w = softmax(scaled)
close(w[0], 0.731, 5e-4)
close(w[1], 0.269, 5e-4)
close(math.e / (math.e + 1), 0.731, 5e-4)
out = [w[0] * Vv[0][j] + w[1] * Vv[1][j] for j in range(2)]
close(out[0], 7.31, 5e-3)
close(out[1], 2.69, 5e-3)
close(w[1], 0.27, 5e-3)  # "27% of the other"

# ---- 6 causal mask
S = [[1, 5], [2, 2]]
A = []
for i, row in enumerate(S):
    masked = [v if j <= i else -math.inf for j, v in enumerate(row)]
    A.append(softmax(masked))
close(A[0][0], 1, 1e-12)
close(A[0][1], 0, 1e-12)
close(A[1][0], 0.5, 1e-12)
close(A[1][1], 0.5, 1e-12)
O = matmul(A, [[4, 0], [0, 4]])
for a, b in zip(sum(O, []), [4, 0, 2, 2]):
    close(a, b, 1e-12)

# ---- 6.1 multi-head
eq(5120 // 40, 128)
WO = [[1, 1], [0, 2]]
eq(matmul([[3, -1]], WO), [[3, 1]])
eq(3 * 1 - 1 * 0, 3)
eq(3 * 1 - 1 * 2, 1)

# ---- 7 RoPE rotation, d_h=8, base 10,000, t=5, i=1
dh = 8
omega = [10000 ** (-2 * i / dh) for i in range(dh // 2)]
close(omega[1], 0.1, 1e-12)
theta = 5 * omega[1]
close(theta, 0.5, 1e-12)
close(math.cos(0.5), 0.8776, 5e-5)
close(math.sin(0.5), 0.4794, 5e-5)
qr = rot(theta, (1, 1))
close(qr[0], 0.3982, 5e-5)
close(qr[1], 1.3570, 5e-5)
close(0.8776 - 0.4794, 0.3982, 1e-9)
close(0.4794 + 0.8776, 1.3570, 1e-9)
close(math.hypot(*qr), math.sqrt(2), 1e-12)
close(math.sqrt(2), 1.414, 5e-4)
for a, b in zip([5 * o for o in (omega[0], omega[2], omega[3])], [5, 0.05, 0.005]):
    close(a, b, 1e-12)

# ---- 7 relative property with omega = pi/2, q=(1,0), k=(2,1)
w_ = math.pi / 2
qv, kv = (1, 0), (2, 1)
close(rot(w_, (3, 7))[0], -7, 1e-12)  # R_1 (x,y) = (-y, x)
close(rot(w_, (3, 7))[1], 3, 1e-12)
rows = [(1, 0, (0, 1), (2, 1), 1), (3, 2, (0, -1), (-2, -1), 1),
        (2, 0, (-1, 0), (2, 1), -2), (4, 2, (1, 0), (-2, -1), -2)]
for t, s, Rq, Rk, d in rows:
    a = rot(t * w_, qv)
    b = rot(s * w_, kv)
    close(a[0], Rq[0], 1e-12)
    close(a[1], Rq[1], 1e-12)
    close(b[0], Rk[0], 1e-12)
    close(b[1], Rk[1], 1e-12)
    close(a[0] * b[0] + a[1] * b[1], d, 1e-12)

# ---- 7.1 base frequency, d_h = 128, i = 32
close(2 * 32 / 128, 0.5, 1e-12)
o4 = (10 ** 4) ** (-2 * 32 / 128)
o6 = (10 ** 6) ** (-2 * 32 / 128)
close(o4, 0.01, 1e-15)
close(o6, 0.001, 1e-15)
close(2 * math.pi / o4, 628, 0.5)
close(2 * math.pi / o6, 6283, 0.5)
close((2 * math.pi / o6) / (2 * math.pi / o4), 10, 1e-9)

# ---- 7.2 p-RoPE
eq(0.25 * 512, 128)
eq(512 - 128, 384)
eq(128 // 2, 64)
eq(0.25 * 8, 2)
qq = [1, 0, 3, 1, 4, 1, 5, 9]
r = rot(math.pi / 2, qq[:2])
qtil = [round(r[0], 12), round(r[1], 12)] + qq[2:]
eq(qtil, [0, 1, 3, 1, 4, 1, 5, 9])

# ---- 8 MLP / GLU / SwiGLU / GeGLU, x=(1,-1)
W1 = [[1, -1], [1, 1]]
xv = [[1], [-1]]
a = [r_[0] for r_ in matmul(W1, xv)]
eq(a, [2, 0])
close(math.exp(-2), 0.135, 5e-4)
close(1 + math.exp(-2), 1.135, 5e-4)
close(1 / 1.135, 0.881, 5e-4)
close(sigmoid(2), 0.881, 5e-4)
close(sigmoid(0), 0.5, 1e-12)
mlp = sigmoid(2) + sigmoid(0)
close(mlp, 1.381, 5e-4)
close(0.881 + 0.5, 1.381, 1e-9)
eq(matmul([[1, 1]], W1), [[2, 0]])  # W2 W1 without the nonlinearity

b = [1, 3]
glu = [sigmoid(a[0]) * b[0], sigmoid(a[1]) * b[1]]
close(glu[0], 0.881, 5e-4)
close(glu[1], 1.5, 1e-12)
close(sum(glu), 2.381, 5e-4)
close(glu[0] / b[0], 0.88, 5e-3)  # 88% passes

silu = lambda z: z * sigmoid(z)
close(silu(2), 1.762, 5e-4)
close(2 * 0.881, 1.762, 1e-9)
close(silu(0), 0, 1e-12)
sw = silu(a[0]) * b[0] + silu(a[1]) * b[1]
close(sw, 1.762, 5e-4)
close(sigmoid(-1), 0.269, 5e-4)
close(silu(-1), -0.269, 5e-4)

gelu = lambda z: z * Phi(z)
gelu_tanh = lambda z: 0.5 * z * (1 + math.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z ** 3)))
close(Phi(2), 0.977, 5e-4)
close(2 * 0.977, 1.954, 1e-9)
close(gelu(2), 1.95, 5e-3)
close(gelu_tanh(2), 1.95, 5e-3)
gg = gelu(a[0]) * b[0] + gelu(a[1]) * b[1]
close(gg, 1.95, 5e-3)
close(silu(2), 1.76, 5e-3)

# ---- 9 MoE routing, logits (1,1,0,-1), k=2
close(math.exp(-1), 0.368, 5e-4)
close(2 * 2.718 + 1 + 0.368, 6.804, 1e-9)
close(2 * math.e + 1 + math.exp(-1), 6.804, 5e-4)
g = softmax([1, 1, 0, -1])
for a_, b_ in zip(g, [0.399, 0.399, 0.147, 0.054]):
    close(a_, b_, 5e-4)
close(g[0] / (g[0] + g[1]), 0.5, 1e-12)
close(g[1] / (g[0] + g[1]), 0.5, 1e-12)
yv = [0.5 * 2 + 0.5 * 0, 0.5 * 0 + 0.5 * 4]
eq(yv, [1.0, 2.0])

# ---- 9.1 load balancing
f = [3 / 4, 1 / 4]
P = [0.7, 0.3]
sfp = sum(a_ * b_ for a_, b_ in zip(f, P))
close(sfp, 0.6, 1e-12)
close(2 * sfp, 1.2, 1e-12)
close(2 * 0.5, 1.0, 1e-12)
close(0.01 * 2 * sfp, 0.012, 1e-12)
close(0.01 * 2 * (0.5 * 0.5 + 0.5 * 0.5), 0.010, 1e-12)

# ---- 9.2 active params, Qwen3-30B-A3B config
Lq, dq, Hq, Gq, dhq, ffe, Eq, kq, Vq = 48, 2048, 32, 4, 128, 768, 128, 8, 151936
attn = Lq * (dq * Hq * dhq + 2 * dq * Gq * dhq + Hq * dhq * dq)
router = Lq * dq * Eq
experts = Lq * Eq * 3 * dq * ffe
emb = 2 * Vq * dq  # untied
close(attn / 1e9, 0.9, 0.05)
close(router / 1e9, 0.01, 0.005)
close(experts / 1e9, 29, 0.05)
close(emb / 1e9, 0.6, 0.05)
active_r = 0.9 + 0.01 + 8 / 128 * 29 + 0.6
total_r = 0.9 + 0.01 + 29 + 0.6
close(active_r, 3.3, 0.05)
close(total_r, 30.5, 0.05)
close((attn + router + kq / Eq * experts + emb) / 1e9, 3.3, 0.06)
close((attn + router + experts + emb) / 1e9, 30.5, 0.05)
close(8 / 128, 1 / 16, 1e-15)
close(active_r / total_r, 0.11, 5e-3)

# ---- 10 linear attention, d_k=d_v=2
def outer(u_, v_):
    return [[a_ * b_ for b_ in v_] for a_ in u_]


def madd(A_, B_):
    return [[a_ + b_ for a_, b_ in zip(ra, rb)] for ra, rb in zip(A_, B_)]


def read(S_, q_):  # o = S q, with S in R^{d_v x d_k} (Gated DeltaNet convention)
    return [sum(S_[i][j] * q_[j] for j in range(len(q_))) for i in range(len(S_))]


# S_t = S_{t-1} + v_t k_t^T: outer(v, k)
S2 = madd(outer([3, 5], [1, 0]), outer([2, 4], [0, 1]))
eq(S2, [[3, 2], [5, 4]])
eq(read(S2, [1, 0]), [3, 5])
S3 = madd(S2, outer([1, 1], [1, 0]))
eq([S3[0][0], S3[1][0]], [4, 6])          # the first column becomes (4, 6)
eq(read(S3, [1, 0]), [3 + 1, 5 + 1])

# ---- 10 DeltaNet scalar
S_prev, kk, vv, beta = 3, 1, 1, 0.5
S_new = S_prev + beta * (vv - S_prev * kk) * kk
close(S_new, 2, 1e-12)
eq((S_prev * kk - vv) ** 2, 4)
close((S_new * kk - vv) ** 2, 1, 1e-12)
close(S_prev + 1.0 * (vv - S_prev * kk) * kk, 1, 1e-12)
eq(S_prev + kk * vv, 4)  # linear attention would add

# ---- 10 Gated DeltaNet scalar, both forms
alpha = 0.5
f1 = alpha * S_prev * (1 - beta * kk * kk) + beta * vv * kk
f2 = alpha * S_prev + beta * (vv - alpha * S_prev * kk) * kk
close(f1, 1.25, 1e-12)
close(f2, 1.25, 1e-12)
close(0.5 * 3 * (1 - 0.5) + 0.5 * 1, 1.25, 1e-12)
close(1.5 + 0.5 * (1 - 1.5), 1.25, 1e-12)
close(alpha * S_prev, 1.5, 1e-12)
close(f1 * 1, 1.25, 1e-12)  # o_t with q_t = 1

# ---- 11 sliding window
i_, W_ = 2000, 1024
eq(i_ - W_ + 1, 977)
visible = [j for j in range(0, 5000) if not (j < i_ - W_ + 1) and j <= i_]
eq(len(visible), 1024)
eq(visible[0], 977)
eq(visible[-1], 2000)
eq(2000 - 977 + 1, 1024)

# ---- 11 hybrid cache, Gemma 4 12B at T = 8,192
local = 40 * 2 * 8 * 256 * 1024 * 2
glob = 8 * 1 * 512 * 8192 * 2
eq(local, 40 * 2 ** 23)
eq(glob, 2 ** 26)
close(local / 1e6, 335.5, 0.05)
close(glob / 1e6, 67.1, 0.05)
close((local + glob) / 1e6, 402.7, 0.05)
close(glob / (local + glob), 1 / 6, 1e-12)
glob32 = 8 * 1 * 512 * 32768 * 2
close(glob32 / 1e6, 268.4, 0.05)
close((local + glob32) / 1e6, 604, 0.5)

# ---- 12 MTP loss
close(math.log(2), 0.6931, 5e-5)
mtp = -(math.log(0.5) + math.log(0.25) + math.log(0.125))
close(mtp, 6 * math.log(2), 1e-12)
close(mtp, 4.159, 5e-4)
close(6 * 0.6931, 4.159, 5e-4)
close(3 * math.log(2) / mtp, 0.5, 1e-12)

# ---- 12 speculative speedup
close(3 / 12, 0.25, 1e-12)
num = 1 + 0.5 + 0.5 ** 2 + 0.5 ** 3
close(num, 1.875, 1e-12)
close(num / 1.25, 1.5, 1e-12)
num9 = 1 + 0.9 + 0.81 + 0.729
close(num9, 1 + 0.9 + 0.9 ** 2 + 0.9 ** 3, 1e-12)
close(num9 / 1.25, 2.75, 5e-3)

# ---- 13 attention params, Qwen3-14B
d, H, G, dh = 5120, 40, 8, 128
eq(d, 5 * 2 ** 10)
eq(H * dh, 5120)
eq(G * dh, 1024)
eq(d * H * dh, 25 * 2 ** 20)
eq(H * dh * d, 25 * 2 ** 20)
eq(2 * d * G * dh, 10 * 2 ** 20)
p_attn = d * H * dh + 2 * d * G * dh + H * dh * d
eq(p_attn, 60 * 2 ** 20)
eq(2 ** 20, 1048576)
close(p_attn / 1e6, 62.9, 0.05)
close((2 * d * G * dh) / p_attn, 1 / 6, 1e-12)
close(G / H, 1 / 5, 1e-12)

# ---- 13 toy total
dt, Ht, Gt, dht, fft, Lt, Vt = 8, 4, 2, 2, 16, 2, 10
eq(dt * Ht * dht, 64)
eq(2 * dt * Gt * dht, 64)
eq(Ht * dht * dt, 64)
pa = dt * Ht * dht + 2 * dt * Gt * dht + Ht * dht * dt
eq(pa, 192)
pm = 3 * dt * fft
eq(pm, 384)
eq(Lt * (pa + pm), 1152)
eq(Vt * dt, 80)
eq(Lt * (pa + pm) + (1 + 0) * Vt * dt, 1232)
eq(Lt * (pa + pm) + (1 + 1) * Vt * dt, 1312)

# ---- 14.1 training compute
C = 6 * 1e9 * 20e9
close(C, 1.2e20, 1e8)
close(2 * 1e9 * 20e9, 4e19, 1e8)
close(4 * 1e9 * 20e9, 8e19, 1e8)

# ---- 14.1 attention FLOPs, Qwen3-14B
att2k = 12 * 40 * 2048 ** 2 * 5120
eq(att2k, 2400 * 2 ** 32)
eq(12 * 40 * 5, 2400)
close(att2k / 1e13, 1.03, 5e-3)
dense2k = 6 * 14.77e9 * 2048
close(dense2k / 1e14, 1.81, 5e-3)
close(att2k / dense2k, 0.06, 5e-3)
att32k = 12 * 40 * 32768 ** 2 * 5120
dense32k = 6 * 14.77e9 * 32768
close(att32k / att2k, 256, 1e-9)
close(dense32k / dense2k, 16, 1e-9)
close(att32k / 1e15, 2.64, 5e-3)
close(dense32k / 1e15, 2.90, 5e-3)
close(att32k / dense32k, 0.91, 5e-3)

# ---- 14.2 inference compute
ci = 2 * 14.77e9
close(ci / 1e10, 2.95, 5e-3)
close(ci / 1e9, 30, 0.5)
close(500 * ci / 1e13, 1.5, 0.05)
close(2 * 3.3e9, 6.6e9, 1)

# ---- 15 affine quantization, s=0.1, z=8, codes 0..15
def quant(xv_, s_, z_, lo, hi):
    return max(lo, min(hi, round(xv_ / s_) + z_))


qa = quant(0.33, 0.1, 8, 0, 15)
eq(round(0.33 / 0.1), 3)
eq(qa, 11)
close(0.1 * (qa - 8), 0.3, 1e-12)
close(abs(0.33 - 0.1 * (qa - 8)), 0.03, 1e-12)
close(0.1 / 2, 0.05, 1e-12)
eq(round(1.0 / 0.1) + 8, 18)
qb = quant(1.0, 0.1, 8, 0, 15)
eq(qb, 15)
close(0.1 * (qb - 8), 0.7, 1e-12)
close(1.0 - 0.1 * (qb - 8), 0.3, 1e-12)

# ---- 15 blockwise scale, k=4
wb = [0.7, -0.33, 0.12, -0.06]
eq(2 ** (4 - 1) - 1, 7)
sb = max(abs(v) for v in wb) / 7
close(sb, 0.1, 1e-12)
qs = [round(v / sb) for v in wb]
eq(qs, [7, -3, 1, -1])
what = [sb * v for v in qs]
for a_, b_ in zip(what, [0.7, -0.3, 0.1, -0.1]):
    close(a_, b_, 1e-12)
errs = [abs(a_ - b_) for a_, b_ in zip(wb, what)]
for a_, b_ in zip(errs, [0, 0.03, 0.02, 0.04]):
    close(a_, b_, 1e-12)
assert max(errs) <= sb / 2 + 1e-12
checks += 1

# ---- 15 QAT / STE
eq(round(0.33 / 0.1), 3)
close(2 * 1, 2, 0)
close(0.33 - 0.01 * 2, 0.31, 1e-12)
eq(round(0.31 / 0.1), 3)  # forward still sees 0.3
eq(round(0.24 / 0.1), 2)  # below 0.25 -> 0.2

# ---- 16 AdamW, one step
b1, b2, eta, lam, th0, g1 = 0.9, 0.95, 0.01, 0.1, 2.0, -0.2
m1 = (1 - b1) * g1
v1 = (1 - b2) * g1 ** 2
close(m1, -0.02, 1e-12)
close(v1, 0.002, 1e-12)
close(g1 ** 2, 0.04, 1e-12)
mh = m1 / (1 - b1 ** 1)
vh = v1 / (1 - b2 ** 1)
close(mh, -0.2, 1e-12)
close(math.sqrt(vh), 0.2, 1e-12)
close(mh / math.sqrt(vh), -1, 1e-12)
th1 = th0 - eta * (mh / math.sqrt(vh) + lam * th0)
close(th1, 2.008, 1e-12)
close(eta * lam * th0, 0.002, 1e-12)

# ---- 16 LR schedule
emax, emin, Tw, Tt = 1e-4, 1e-5, 100, 1000


def lr(t):
    if t < Tw:
        return emax * t / Tw
    return emin + 0.5 * (emax - emin) * (1 + math.cos(math.pi * (t - Tw) / (Tt - Tw)))


close(lr(50), 5e-5, 1e-15)
close((400 - Tw) / (Tt - Tw), 1 / 3, 1e-12)
close(math.cos(math.pi / 3), 0.5, 1e-12)
close(0.5 * 9e-5 * 1.5, 6.75e-5, 1e-15)
close(lr(400), 7.75e-5, 1e-15)
close((lr(400) - emin) / (emax - emin), 0.75, 1e-9)

# ---- 16.1 init
close(math.sqrt(80), 8.944, 5e-4)
close(0.02 / math.sqrt(2 * 40), 0.0022, 5e-5)
close(2 * 0.02, 0.04, 1e-12)
close(80 * (0.02 ** 2 / 80), 0.02 ** 2, 1e-15)
close(Phi(2) - Phi(-2), 0.95, 5e-3)  # about 95% within 2 sigma

print(f"ch32_mathematics: all {checks} checks passed")
