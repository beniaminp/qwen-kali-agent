#!/usr/bin/env python3
"""Check the arithmetic in docs/43-math-examples.md.

Pure stdlib; no model, no torch, no network. Run it after editing the worked
examples so the documentation cannot silently drift from the formulas.
"""
from __future__ import annotations

import math


def close(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    checks = 0

    # 1. Softmax and CE
    z = [2.0, 1.0, 0.0, -1.0]
    e = [math.exp(v) for v in z]
    Z = sum(e)
    p = [v / Z for v in e]
    assert all(close(a, b) for a, b in zip(p, [0.643914, 0.236883, 0.087144, 0.032059]))
    ce = -math.log(p[0])
    assert close(ce, 0.440190)
    checks += 2

    # 2. CE gradient
    grad = [p[i] - (1.0 if i == 0 else 0.0) for i in range(4)]
    assert close(grad[0], -0.356086)
    assert close(sum(grad), 0.0, 1e-12)
    checks += 2

    # 3. Softmax Jacobian
    q = [0.5, 0.3, 0.2]
    J = [[(q[i] if i == j else 0.0) - q[i] * q[j] for j in range(3)] for i in range(3)]
    assert all(close(sum(row), 0.0, 1e-12) for row in J)
    assert close(J[0][0], 0.25) and close(J[0][1], -0.15)
    checks += 2

    # 4. RMSNorm
    x = [1.0, 2.0, 3.0, 4.0]
    rms = math.sqrt(sum(v * v for v in x) / 4)
    y = [v / rms for v in x]
    assert close(rms, 2.738613)
    assert close(math.sqrt(sum(v * v for v in y) / 4), 1.0)
    dy0dx0 = 1 / rms - x[0] ** 2 / (4 * rms ** 3)
    assert close(dy0dx0, 0.352977)
    checks += 3

    # 5. Attention
    qv = [1.0, 0.0]
    K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    scores = [sum(qv[i] * k[i] for i in range(2)) / math.sqrt(2) for k in K]
    ex = [math.exp(s) for s in scores]
    Z = sum(ex)
    ap = [v / Z for v in ex]
    V = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    out = [sum(ap[j] * V[j][i] for j in range(3)) for i in range(2)]
    assert close(out[0], 0.802224) and close(out[1], 0.598888)
    checks += 2

    # 6. GQA params (d=4,H=2,dh=2,G=1)
    d, H, dh, G = 4, 2, 2, 1
    gqa = d * H * dh + 2 * d * G * dh + H * dh * d
    mha = d * H * dh + 2 * d * H * dh + H * dh * d
    assert gqa == 48 and mha == 64
    checks += 2

    # 7. KV cache
    assert 2 * 2 * 1 * 2 * 3 * 2 == 48
    gemma_local = 2 * 40 * 8 * 256 * 1024 * 2
    gemma_global = 8 * 1 * 512 * 32768 * 2
    assert close((gemma_local + gemma_global) / 1e6, 603.98, 0.05)
    assert close((gemma_local + 8 * 1 * 512 * 262144 * 2) / 1e9, 2.483, 0.01)
    checks += 3

    # 8. RoPE
    assert close(math.cos(1.0), 0.540302)
    assert close(math.cos(2.0), -0.416147)
    checks += 2

    # 9. SwiGLU
    a = [1.0, -1.0, 0.0]
    b = [1.0, -1.0, 2.0]
    silu = [v / (1 + math.exp(-v)) for v in a]
    gate = [silu[i] * b[i] for i in range(3)]
    assert close(sum(gate), 1.0)
    gprime = [1 / (1 + math.exp(-v)) + v * math.exp(-v) / (1 + math.exp(-v)) ** 2 for v in a]
    assert close(gprime[0], 0.927671) and close(gprime[1], 0.072329)
    checks += 3

    # 10. MoE
    r = [2.0, 1.0, 0.0, -1.0]
    er = [math.exp(v) for v in r]
    Z = sum(er)
    g = [v / Z for v in er]
    top2 = [g[0] / (g[0] + g[1]), g[1] / (g[0] + g[1])]
    assert close(top2[0], 0.731059) and close(top2[1], 0.268941)
    assert close(g[0], 0.643914)
    checks += 2

    # 11. GDN
    S = 0.0
    S = S + 0.5 * (2.0 - S) * 1.0
    assert close(S, 1.0)
    S = S + 0.5 * (4.0 - S) * 1.0
    assert close(S, 2.5)
    S2 = 0.5 * 1.0 + 0.5 * (4.0 - 0.5 * 1.0) * 1.0
    assert close(S2, 2.25)
    checks += 3

    # 12. Sliding window cache
    assert 2 * 1 * 2 * 4 * 2 == 32
    checks += 1

    # 13. Speculative decoding
    alpha, gamma, c = 0.8, 3, 0.1
    exp_acc = sum(alpha ** i for i in range(1, gamma + 1))
    speed = (1 + exp_acc) / (1 + gamma * c)
    assert close(exp_acc, 1.952) and close(speed, 2.27077)
    checks += 2

    # 14. Parameter count
    toy = 2 * (48 + 96) + 2 * 5 * 4
    assert toy == 328
    assert 6 * toy * 1000 == 1_968_000
    checks += 2

    # 15. Quantization
    w = [-1.0, -0.4, 0.0, 0.4, 1.0]
    s = 1 / 7
    q = [max(-7, min(7, round(v / s))) for v in w]
    dq = [v * s for v in q]
    err = [abs(w[i] - dq[i]) for i in range(5)]
    assert max(err) < s / 2
    checks += 1

    # 16. AdamW
    b1, b2, eta = 0.9, 0.95, 0.1
    m = v = 0.0
    theta = 1.0
    for t in (1, 2):
        m = b1 * m + (1 - b1) * 0.5
        v = b2 * v + (1 - b2) * 0.25
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        theta -= eta * mh / (math.sqrt(vh) + 1e-8)
    assert close(theta, 0.8)
    checks += 1

    # 17. LoRA scale
    assert close(8 / 4, 2.0)
    checks += 1

    # 18. KD
    zt, zs, temp = [2.0, 1.0, 0.0], [1.0, 2.0, 0.0], 2.0
    pt0 = [math.exp(v / temp) for v in zt]
    ps0 = [math.exp(v / temp) for v in zs]
    Zt, Zs = sum(pt0), sum(ps0)
    pt = [v / Zt for v in pt0]
    ps = [v / Zs for v in ps0]
    kl = sum(pt[i] * math.log(pt[i] / ps[i]) for i in range(3))
    assert close(kl, 0.099642)
    assert close(temp ** 2 * kl, 0.398569)
    checks += 2

    # 19. Pass@k
    from math import comb
    pass5 = 1 - comb(6, 5) / comb(10, 5)
    assert close(pass5, 0.976190)
    checks += 1

    # 20. ECE
    ece = 0.5 * abs(0.8 - 0.9) + 0.5 * abs(0.4 - 0.6)
    assert close(ece, 0.15)
    checks += 1

    # 21. Wilson score interval (43 §20): 40/100 at 95%
    p_hat, n_obs, zc = 0.4, 100, 1.96
    z2 = zc * zc
    denom = 1 + z2 / n_obs
    center = (p_hat + z2 / (2 * n_obs)) / denom
    half = zc * math.sqrt(p_hat * (1 - p_hat) / n_obs + z2 / (4 * n_obs ** 2)) / denom
    lo, hi = center - half, center + half
    assert close(center, 0.403699, 1e-4)
    assert close(lo, 0.309389, 1e-4) and close(hi, 0.498009, 1e-4)
    checks += 2

    # 22. Real-model parameter counts (43 §14, 37 §1.3).
    # One formula, two configs: attention = Q + K + V + O projections.
    def attn_params(dm: int, h: int, g: int, dh: int) -> int:
        return dm * h * dh + 2 * dm * g * dh + h * dh * dm

    # Qwen3-14B: d=5120, L=40, ffn=17408, H=40, G=8, dh=128, V=151936, untied.
    q_attn = attn_params(5120, 40, 8, 128)
    q_mlp = 3 * 5120 * 17408
    q_layer = q_attn + q_mlp + 2 * 5120  # + two RMSNorm vectors
    q_total = 40 * q_layer + 2 * 151936 * 5120 + 5120  # untied head + final norm
    assert q_attn == 62_914_560 and q_mlp == 267_386_880
    assert q_total == 14_768_296_960
    checks += 2

    # Gemma 4 12B: d=3840, 40 local + 8 global, ffn=15360, V=262144, tied.
    # Global layers are K=V: the key projection doubles as the value, so
    # there is no v_proj (the google/gemma-4-12B-it checkpoint has none).
    g_local = attn_params(3840, 16, 8, 256)
    g_global = attn_params(3840, 16, 1, 512) - 3840 * 1 * 512  # no v_proj
    g_mlp = 3 * 3840 * 15360
    g_embed = 3840 * 262144
    g_local_layer = g_local + g_mlp + 2 * 3840
    g_global_layer = g_global + g_mlp + 2 * 3840
    g_total = 40 * g_local_layer + 8 * g_global_layer + g_embed + 3840
    g_all_local = 48 * g_local_layer + g_embed + 3840  # the common miscount
    assert g_local == 47_185_920 and g_global == 64_880_640 and g_mlp == 176_947_200
    assert g_total == 11_906_952_960
    assert g_all_local == 11_765_395_200
    assert g_total - g_all_local == 141_557_760  # ~0.14B the naive count misses
    checks += 4

    # 23. Gemma 4 family, reproduced to the parameter from the safetensors
    # headers of google/gemma-4-{12B,E4B,E2B}-it (37 §4, 43 §7 and §14).
    # Counts every RMSNorm, q/k norm and layer scalar.
    def gemma4(d, L, ffn, H, G, dh, Gg, dhg, pattern, kv_shared=0,
               double_wide=False, k_eq_v=False, ple=0, V=262144):
        types = (pattern * L)[:L]
        first_shared = L - kv_shared
        attn = mlp = rest = 0
        for i, t in enumerate(types):
            g, h = (Gg, dhg) if t == "G" else (G, dh)
            shared = i >= first_shared
            attn += 2 * d * H * h + h                    # q, o, q_norm
            if not shared:                               # shared layers reuse K/V
                attn += d * g * h + h                    # k, k_norm
                if not (k_eq_v and t == "G"):
                    attn += d * g * h                    # v (v_norm has no weight)
            mlp += 3 * d * ffn * (2 if double_wide and shared else 1)
            rest += 4 * d + 1                            # four RMSNorms + layer scalar
            if ple:
                rest += 2 * d * ple + d                  # PLE gate, projection, norm
        rest += d + ((L * ple * d + ple) if ple else 0)  # final norm, PLE projection
        tables = V * d + (V * L * ple if ple else 0)     # token + per-layer embeddings
        return attn, mlp, attn + mlp + rest, tables

    def file_params(size_bytes, header_bytes):           # bf16 safetensors
        return (size_bytes - header_bytes - 8) // 2

    # 12B: text decoder + 52.4M multimodal embedders = the 23.9 GB file.
    a12, m12, c12, t12 = gemma4(3840, 48, 15360, 16, 8, 256, 1, 512, "LLLLLG", k_eq_v=True)
    assert a12 == 2_406_510_592 and m12 == 8_493_465_600
    assert c12 + t12 == 11_907_350_320
    assert c12 + t12 + 52_379_904 == file_params(23_919_549_408, 88_952) == 11_959_730_224
    checks += 3

    # E4B: last 18 layers share KV; PLE 256 per layer; no K=V.
    a4, m4, c4, t4 = gemma4(2560, 42, 10240, 8, 2, 256, 2, 512, "LLLLLG",
                            kv_shared=18, ple=256)
    unused_kv4, vision4, audio4 = 55_055_616, 169_331_136, 308_757_248
    assert a4 == 587_222_272 and m4 == 3_303_014_400
    assert c4 == 3_973_352_490 and t4 == 3_489_660_928
    assert c4 + t4 + unused_kv4 + vision4 + audio4 == file_params(15_992_595_884, 281_040)
    effective4 = file_params(15_992_595_884, 281_040) - t4
    assert effective4 == 4_506_496_490                  # model card: "4.5B effective"
    checks += 4

    # E2B: last 20 layers share KV and have a double-wide MLP; 4:1 pattern.
    a2, m2, c2, t2 = gemma4(1536, 35, 6144, 8, 1, 256, 1, 512, "LLLLG",
                            kv_shared=20, double_wide=True, ple=256)
    unused_kv2, vision2, audio2 = 18_880_512, 168_544_704, 307_184_384
    assert a2 == 278_412_288 and m2 == 1_557_135_360
    assert c2 == 1_877_105_955 and t2 == 2_751_463_424
    assert c2 + t2 + unused_kv2 + vision2 + audio2 == file_params(10_246_621_918, 263_952)
    effective2 = file_params(10_246_621_918, 263_952) - t2
    assert effective2 == 2_371_715_555                  # model card: "2.3B effective"
    assert t2 * 100 // file_params(10_246_621_918, 263_952) == 53  # tables are 53% of E2B
    checks += 5

    # 24. Gemma 4 KV cache with sliding windows, K=V and KV sharing (bf16).
    def gemma4_kv(L, pattern, kv_shared, G, dh, Gg, dhg, W, T, k_eq_v=False):
        types = (pattern * L)[:L]
        total = 0
        for t in types[:L - kv_shared]:                  # shared layers store nothing
            if t == "L":
                total += 2 * G * dh * min(T, W)
            else:
                total += (1 if k_eq_v else 2) * Gg * dhg * T
        return 2 * total

    T32, T128 = 32_768, 131_072
    assert gemma4_kv(48, "LLLLLG", 0, 8, 256, 1, 512, 1024, T32, k_eq_v=True) == 603_979_776
    kv12_128 = gemma4_kv(48, "LLLLLG", 0, 8, 256, 1, 512, 1024, T128, k_eq_v=True)
    kv4_32 = gemma4_kv(42, "LLLLLG", 18, 2, 256, 2, 512, 512, T32)
    kv4_128 = gemma4_kv(42, "LLLLLG", 18, 2, 256, 2, 512, 512, T128)
    kv2_32 = gemma4_kv(35, "LLLLG", 20, 1, 256, 1, 512, 512, T32)
    kv2_128 = gemma4_kv(35, "LLLLG", 20, 1, 256, 1, 512, 512, T128)
    assert kv12_128 == 1_409_286_144
    assert kv4_32 == 557_842_432 and kv4_128 == 2_168_455_168
    assert kv2_32 == 207_618_048 and kv2_128 == 811_597_824
    assert kv4_128 > kv12_128                            # the 4B-class cache outgrows the 12B
    # Without KV sharing every layer would cache: E4B 0.98 GB, E2B 0.48 GB at 32K.
    assert gemma4_kv(42, "LLLLLG", 0, 2, 256, 2, 512, 512, T32) == 976_224_256
    assert gemma4_kv(35, "LLLLG", 0, 1, 256, 1, 512, 512, T32) == 484_442_112
    checks += 6

    # 25. "On paper" cards: GQA does not average queries (§6), the QAT
    # straight-through estimator (§16) and sequence KD (§19).
    head = math.e / (math.e + 1)
    assert close(head, 0.731, 5e-4) and close(1 - head, 0.269, 5e-4)
    s_q, w_lat = 0.25, 0.30
    assert round(w_lat / s_q) == 1 and close(round(w_lat / s_q) * s_q, 0.25)
    w_new = w_lat - 0.5 * 0.4
    assert close(w_new, 0.10) and round(w_new / s_q) == 0 and close(0.5 * s_q, 0.125)
    seq = -(math.log(0.5) + math.log(0.25) + math.log(0.8))
    assert close(seq, 2.303, 5e-4) and close(seq, -math.log(0.1)) and close(seq / 3, 0.768, 5e-4)
    checks += 3

    # 26. Full-pretraining budget (43 §15): dense BF16 H100 peak is ~989
    # TFLOP/s; NVIDIA's 1,979 is with sparsity and 494.5 is dense TF32.
    gpu_hours = 6 * 14.77e9 * 36e12 / (989e12 * 0.4 * 3600)
    assert close(gpu_hours / 1e6, 2.24, 0.005)
    assert close(gpu_hours / 10_000 / 24, 9.3, 0.05) and close(gpu_hours * 2.5 / 1e6, 5.6, 0.01)
    checks += 2

    print(f"all {checks} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
