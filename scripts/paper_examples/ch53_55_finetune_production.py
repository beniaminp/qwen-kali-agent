#!/usr/bin/env python3
"""Check every number in the "On paper" cards, worked examples and exercise
answers of docs/53 (fine-tuning) and docs/55 (production). Stdlib only.

Pattern matches scripts/paper_examples/ch39_42_distillation.py:
close(a, b, tol) / eq(a, b) count checks; a final line prints the total.
"""
from math import ceil, exp, log

checks = 0


def close(a, b, tol):
    global checks
    checks += 1
    assert abs(a - b) <= tol, f"{a!r} != {b!r} (tol {tol})"


def eq(a, b):
    global checks
    checks += 1
    assert a == b, f"{a!r} != {b!r}"


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


LN2 = log(2)
close(LN2, 0.693, 5e-4)

# ============================================================ chapter 53
# ---- §3.1 full fine-tuning optimizer state
N14 = 14.77e9
close(12 * N14, 1.77e11, 5e8)          # 177 GB of AdamW state
close(12 * N14 / 1e9, 177.24, 0.5)     # ~177 GB
close(2 * N14 / 1e9, 29.5, 0.05)       # bf16 weights, and as much again for bf16 grads
close((12 + 2 + 2) * N14 / 1e9, 236, 0.5)  # ~236 GB before activations
ok = (12 + 2 + 2) * N14 / 1e9 > 80
eq(ok, True)                           # does not fit one 80 GB GPU

# ---- §4.1 LoRA forward on paper: x=(1,0,1), W all-ones 2x3, r=1, alpha=2
x = [1, 0, 1]
Wx = [sum(x), sum(x)]                   # all-ones rows
eq(Wx, [2, 2])
A = [1, 0, 1]
Ax = sum(a * xi for a, xi in zip(A, x))
eq(Ax, 2)
scale = 2 / 1
B = [1, 2]
upd = [scale * b * Ax for b in B]
eq(upd, [4, 8])
eq([Wx[0] + upd[0], Wx[1] + upd[1]], [6, 10])
# step 0: B = 0 -> update is (0,0), y = Wx
eq([scale * 0 * Ax, scale * 0 * Ax], [0, 0])

# ---- §4.2 one 5120x5120 projection
full_proj = 5120 * 5120
eq(full_proj, 26_214_400)
lora_o = 16 * (5120 + 5120)
eq(lora_o, 163_840)
close(lora_o / full_proj, 0.00625, 1e-9)   # 0.625%

# ---- §4.3 all-layer Qwen3-14B adapter at r=16
def lora(din, dout, r=16):
    return r * (din + dout)


q = lora(5120, 5120)
k = lora(5120, 1024)
v = lora(5120, 1024)
o = lora(5120, 5120)
gate = lora(5120, 17408)
up = lora(5120, 17408)
down = lora(17408, 5120)
eq([q, k, v, o], [163_840, 98_304, 98_304, 163_840])
eq([gate, up, down], [360_448, 360_448, 360_448])
per_block = q + k + v + o + gate + up + down
eq(per_block, 1_605_632)
# card groupings
eq(16 * (10240 + 6144 + 6144 + 10240), 524_288)
eq(16 * (22528 + 22528 + 22528), 1_081_344)
eq(524_288 + 1_081_344, 1_605_632)
total_adapter = 40 * per_block
eq(total_adapter, 64_225_280)              # ~64M
close(total_adapter / N14, 0.00435, 5e-5)  # 0.43%
eq(q + v, 262_144)                         # q_proj + v_proj per block
eq(40 * (q + v), 10_485_760)               # ~10.5M if only q/v are adapted
close(40 * (q + v) / 1e6, 10.5, 0.05)

# ---- §5.1 QLoRA logits memory
V = 151936
logits_2k = 2048 * V * 4
eq(logits_2k, 1_244_659_712)
close(logits_2k / 1e9, 1.24, 0.01)
close(2 * logits_2k / 1e9, 2.49, 0.01)     # 4096 tokens
close(16 / 4.5, 3.56, 5e-3)                # 4-bit base: ~3.5x smaller, not 4x
close(29.5 / 8.3, 3.55, 5e-3)
close(N14 * 4.5 / 8 / 1e9, 8.3, 0.05)      # ~8.3 GB of 4-bit weights
close(total_adapter * 4 / 1e9, 0.257, 5e-3)  # ~0.25 GB adapter in fp32
lo = 8.3 + 0.25 + 0.7 + 1.24 + 4 + 2
hi = 8.3 + 0.25 + 0.7 + 1.24 + 8 + 3
close(lo, 16.5, 0.05)                      # the 5.1 column sums to ~16.5-21.5 GB
close(hi, 21.5, 0.05)
close(lo + 0.10 * 4, 16.9, 0.05)           # + temporaries (10-30% of activations)
close(hi + 0.30 * 8, 23.9, 0.05)           # stays inside 17 §5's 19-24 GB estimate

# ---- §6.3 loss masking code output
nll = [2.0, 1.5, 3.0, 0.1, 0.22, 0.51]
mask = [0, 0, 0, 0, 1, 1]
masked = sum(n * m for n, m in zip(nll, mask)) / sum(mask)
close(masked, 0.3650, 5e-5)
close(sum(nll) / len(nll), 1.2217, 5e-5)
eq(sum(mask), 2)

# ---- §9 LoRA demo param counts
A_params, B_params = 4 * 16, 16 * 4
eq(A_params + B_params, 128)               # trainable
eq(16 * 16, 256)                           # frozen base
eq(256 + 128, 384)                         # total
close(100 * 128 / 384, 33.3, 0.05)

# ---- §10.2 DPO loss on paper
beta = 0.1
dw, dl = 1.0, -1.0
margin = beta * (dw - dl)
close(margin, 0.2, 1e-12)
close(sigmoid(0.2), 0.550, 5e-4)
close(-log(sigmoid(0.2)), 0.598, 5e-4)
close(-log(0.5), 0.693, 5e-4)              # zero-margin reference

# ---- §12.1 int4 quantization error on a small row
def quant(w, k):
    s = max(abs(x) for x in w) / (2 ** (k - 1) - 1)
    q = [round(x / s) for x in w]
    wh = [s * qi for qi in q]
    err = [abs(a - b) for a, b in zip(w, wh)]
    return s, q, wh, err


w = [1.0, -0.6, 0.3, -0.2]
s, q, wh, err = quant(w, 4)
close(s, 1 / 7, 1e-9)
close(s, 0.142857, 5e-6)
eq(q, [7, -4, 2, -1])
for got, shown in zip(wh, [1.0, -0.571, 0.286, -0.143]):
    close(got, shown, 5e-4)
for got, shown in zip(err, [0.0, 0.029, 0.014, 0.057]):
    close(got, shown, 5e-4)
close(max(err), s / 2, 0.02)
s8, _, _, err8 = quant(w, 8)
close(s8, 1 / 127, 1e-9)
close(s8, 0.007874, 5e-6)
close(127 / 7, 18.14, 5e-3)                 # ~18x smaller step

# ---- §13 evaluating a fine-tune
close(132 / 240, 0.550, 5e-4)
close(158 / 240, 0.658, 5e-4)
close(158 / 240 - 132 / 240, 0.108, 5e-4)
margin_diff = 1.96 * ((0.55 * 0.45 + 0.658 * 0.342) / 240) ** 0.5
close(margin_diff, 0.087, 5e-4)            # unpaired 95% margin on the difference
eq(0.108 > margin_diff, True)

# ---- §14 cost and time of a local run
iters = 2 * 2_000_000 / (8 * 1500)
close(iters, 333.3, 0.1)                    # ~334
close(ceil(iters), 334, 0)
toks = 334 * 8 * 1500
eq(toks, 4_008_000)
close(toks / 1e6, 4.0, 0.01)
secs = 4.0e6 / 22
close(secs, 1.82e5, 1e3)
close(secs / 3600, 50.5, 0.3)               # ~50 h
close(secs / 3600 * 1.15, 58, 0.2)          # ~58 h wall time
# §14.2 rented GPU, FLOPs rule of 02 §4: tok/s = effective FLOP/s / 6N
six_n = 6 * N14
close(six_n, 8.86e10, 5e7)
close(2e12 / (6 * 14.8e9), 22.5, 0.05)  # 02 §4: 2 TFLOP/s -> 22.5 tok/s
close(100 / 312, 0.32, 5e-3)               # 100 TFLOP/s = 32% of an A100
close(100 / 989, 0.10, 5e-3)               # = 10% of an H100 SXM
r_gpu = 1e14 / six_n
close(r_gpu, 1130, 5)                       # ~1,130 tok/s
close(4.0e6 / 1130, 3540, 1)                # ~3,540 s
close(4.0e6 / r_gpu / 3600, 1.0, 0.05)      # ~1 h
close(1.0 * 2.00, 2.0, 0)                   # ~$2
close(r_gpu / 22, 51, 0.5)                  # ~50x the laptop

# ---- §17 exercises
# 17.2
close(8 * (4096 + 1024), 40_960, 0)
eq(8 * (4096 + 1024), 40_960)
eq(4096 * 1024, 4_194_304)
close(40_960 / 4_194_304, 0.00977, 5e-5)    # 0.98%
# 17.4
close(0.1 * (0.5 - 0.5), 0.0, 1e-12)
close(sigmoid(0.0), 0.5, 1e-12)
close(-log(0.5), 0.693, 5e-4)
# 17.5 int4 on (0.8,-0.4,0.4,-0.8)
w5 = [0.8, -0.4, 0.4, -0.8]
s5, q5, wh5, err5 = quant(w5, 4)
close(s5, 0.8 / 7, 1e-9)
close(s5, 0.114286, 5e-6)
eq(q5, [7, -4, 4, -7])                       # 3.5 -> 4, -3.5 -> -4 (banker's)
for got, shown in zip(wh5, [0.8, -0.457, 0.457, -0.8]):
    close(got, shown, 5e-4)
close(max(err5), 0.057, 5e-4)
close(s5 / 2, 0.057143, 5e-6)
# 17.6
close(6e6 / 25, 240_000, 1)
close(240_000 / 3600, 66.7, 0.1)
close(66.7 * 1.15, 76.7, 0.1)               # ~77 h
close(6e6 / 1130, 5310, 1)                  # rented GPU at ~1,130 tok/s
close(6e6 / 1130 / 3600, 1.5, 0.03)         # ~1.5 h
close(6e6 / 1130 / 3600 * 2, 2.95, 0.01)    # ~$3
# 17.7
close(0.72 - 0.55, 0.17, 1e-9)
close(0.88 - 0.70, 0.18, 1e-9)
# 17.8 adapter overhead vs full
eq(64_225_280 < 14.77e9, True)

# ============================================================ chapter 55
# ---- §3 decode bound M1 Pro Qwen3-14B 4-bit
close(200 / 8.3, 24.096, 5e-3)
close(200 / 8.3, 24, 0.1)
close(15 / (200 / 8.3), 0.62, 5e-3)        # observed 15-17 tok/s = 62-71% of it
close(17 / (200 / 8.3), 0.71, 5e-3)
# ---- §3 decode bound 8B fp16 on H100 SXM
close(3350 / 16, 209.375, 1e-3)
close(3350 / 16, 209, 0.5)
close(0.6 * 3350 / 16, 125.6, 0.1)          # ~125-145 tok/s real
close(0.7 * 3350 / 16, 146.6, 0.1)
close(3350 / 200, 16.75, 1e-9)              # ~17x the M1 Pro's bandwidth
# ---- §4.1 batching
eq(8 * 24, 192)
close(2 * 14.8, 29.6, 1e-9)                        # 2N GFLOP per decoded token
close(192 * 2 * 14.8e9 / 1e12, 5.7, 0.05)          # batch-8 roofline needs ~5.7 TFLOPS
close(117 * 2 * 14.8e9 / 1e12, 3.5, 0.05)          # KV-aware bound needs ~3.5 TFLOPS
close(117 * 2 * 14.8e9 / 1e12 / 5.2, 2 / 3, 0.01)  # about two-thirds of the 5.2 TFLOPS peak
close(8.3 + 8 * 0.67, 13.7, 0.05)           # weights + 8 per-request 4K KV caches
close(8 * 200 / 13.7, 117, 0.5)             # KV-aware bound at 4K context
# ---- §5 KV cache per request and concurrency
kv_tok = 2 * 40 * 8 * 128 * 2
eq(kv_tok, 163_840)
close(kv_tok / 1024, 160, 1e-9)             # 160 KiB/token
close(kv_tok * 4096 / 1e9, 0.67, 5e-3)     # ~0.67 GB at 4K (02 §3.5)
close((24 - 8.3 - 2) / 0.67, 20.45, 0.05)   # ~20 concurrent
close((80 - 29.5 - 2) / 0.67, 72, 0.5)      # 80 GB GPU, bf16 weights: ~70
close((80 - 8.3 - 2) / 0.67, 104, 0.5)      # 80 GB GPU, 4-bit weights: ~100
# ---- §7 quantization in serving: decode speedup
close(200 / 29.5, 6.78, 5e-3)               # fp16
close(200 / 8.3, 24.1, 0.05)                # 4-bit
close(24.1 / 6.78, 3.55, 5e-2)
close(29.5 / 8.3, 3.55, 5e-3)
# ---- §8 speculative decoding
close(0.8 ** 5, 0.32768, 1e-9)
E = (1 - 0.8 ** 5) / (1 - 0.8)
close(1 - 0.8 ** 5, 0.67232, 1e-9)
close(E, 3.3616, 5e-4)                       # ~3.36 tok/step
# ---- §10 sizing hardware
eq(50 * 200, 10_000)
eq(ceil(10_000 / 2_000), 5)
eq(5 + 1, 6)
# ---- §11.2 cache saving
close(0.6 * 0.0007, 0.00042, 1e-9)
close(0.4 * 0.0007, 0.00028, 1e-9)
close(0.00028 * 1e6, 280, 1e-6)             # $280/day at 1M req
# ---- §12.1 cost per request
close(800 / 1e6 * 0.50, 0.00040, 1e-12)
close(200 / 1e6 * 1.50, 0.00030, 1e-12)
close(800 / 1e6 * 0.50 + 200 / 1e6 * 1.50, 0.00070, 1e-12)
# ---- §12.2 break-even volume
fixed = 2.0 * 24 * 30
eq(fixed, 1440)
close(1440 / 0.0007, 2.057e6, 1e3)          # ~2.06M
cap = 10 * 86400 * 30
eq(cap, 25_920_000)
close(cap / 1e6, 25.92, 1e-6)
# ---- §12.3 break-even utilization
close(cap * 0.0007, 18_144, 1)
close(1440 / 18_144, 0.0794, 5e-4)          # ~8%
# ---- §13 routing / cascade
close(0.0001 + 0.3 * 0.0010, 0.0004, 1e-12)
close(0.7 * 0.90 + 0.3 * 0.95, 0.915, 1e-12)
eq(round(0.7 * 0.90, 2), 0.63)
close(0.3 * 0.95, 0.285, 1e-12)
# ---- §16.3 percentiles nearest-rank
lat = [120, 135, 140, 150, 155, 160, 165, 170, 175, 180,
       185, 190, 200, 210, 220, 240, 260, 300, 420, 850]
eq(len(lat), 20)


def pctile(vals, p):
    s = sorted(vals)
    return s[ceil(p / 100 * len(s)) - 1]


eq(pctile(lat, 50), 180)
eq(pctile(lat, 95), 420)
eq(pctile(lat, 99), 850)
eq(ceil(0.95 * 20), 19)
close(sum(lat) / len(lat), 231.25, 1e-6)    # mean ~231
# ---- §18.3 SLO / error budget
close((1 - 0.995) * 2e6, 10_000, 1e-6)
close((1 - 0.999) * 30 * 24 * 60, 43.2, 1e-6)

# ---- chapter 55 exercises
# 21.1
close(2000 / 24, 83.33, 5e-2)
# 21.2
close(8192 * 160 / 1024 / 1024, 1.25, 1e-6)  # 1.25 GiB
close(kv_tok * 8192 / 1e9, 1.34, 5e-3)       # 02's 1.34 GB
close(40 / 1.34, 29.9, 5e-2)                 # 29 full requests, ~30
eq(int(40 / (kv_tok * 8192 / 1e9)), 29)
# 21.3
close(0.9 ** 4, 0.6561, 1e-9)
close((1 - 0.9 ** 4) / (1 - 0.9), 3.439, 5e-4)
# 21.4
close(2000 / 1e6 * 0.50, 0.0010, 1e-12)
close(500 / 1e6 * 1.50, 0.00075, 1e-12)
close(0.0010 + 0.00075, 0.00175, 1e-12)
# 21.5
fx = 3.0 * 24 * 30
eq(fx, 2160)
close(2160 / 0.0007, 3.086e6, 1e3)           # ~3.09M
close(2160 / (20e6 * 0.0007), 0.154, 5e-4)   # ~15.4%
# 21.6
close(0.8 * 0.85 + 0.2 * 0.96, 0.872, 1e-12)
close(0.0001 + 0.2 * 0.0012, 0.00034, 1e-12)
# 21.7
ex = [90, 95, 100, 110, 120, 130, 150, 180, 250, 600]
eq(pctile(ex, 50), 120)
eq(pctile(ex, 90), 250)
eq(pctile(ex, 99), 600)
close(sum(ex) / len(ex), 182.5, 1e-9)
# 21.8
close((1 - 0.999) * 5e6, 5000, 1e-6)
close(0.003 * 167_000, 501, 0.5)
close(501 / 5000, 0.10, 5e-3)               # ~10% of the budget per day
eq(3 * 501 < 5000, True)                    # three such days do not exhaust it
eq(10 * 501, 5010)                          # about ten do
eq(10 * 501 >= 5000, True)

print(f"ch53_55_finetune_production: all {checks} checks passed (chapters 53, 55)")
