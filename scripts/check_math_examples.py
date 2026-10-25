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

    print(f"all {checks} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
