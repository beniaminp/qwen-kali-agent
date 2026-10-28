#!/usr/bin/env python3
"""Check every number in the "On paper" cards, worked examples and exercise
answers of docs/51-ai-rag.md and docs/54-ai-evaluation.md (stdlib only)."""
from collections import Counter
from math import ceil, comb, log, log2, sqrt

checks = 0


def close(a, b, tol):
    global checks
    checks += 1
    assert abs(a - b) <= tol, f"{a!r} != {b!r} (tol {tol})"


def eq(a, b):
    global checks
    checks += 1
    assert a == b, f"{a!r} != {b!r}"


# ================================================================ chapter 51
# ---- §2.2 / §2.4 embedding memory, cosine, L2 ----
close(10 ** 6 * 768 * 4, 3.072e9, 1)
close(3.072e9 / 1e9, 3.072, 1e-9)

a = (1, 2, 2)
b = (2, 1, 2)
c = (3, 0, 0)
dot_ab = sum(x * y for x, y in zip(a, b))
eq(dot_ab, 8)
eq(2 + 2 + 4, 8)
na = sqrt(sum(x * x for x in a))
nb = sqrt(sum(x * x for x in b))
close(na, 3.0, 1e-12)
close(nb, 3.0, 1e-12)
eq(1 + 4 + 4, 9)
close(dot_ab / (na * nb), 8 / 9, 1e-12)
close(8 / 9, 0.889, 5e-4)
# L2 on unit vectors: 2 - 2 cos = 2/9
close(2 - 2 * (8 / 9), 2 / 9, 1e-12)
close(2 / 9, 0.2222, 5e-4)
close(sqrt(2 / 9), 0.471, 5e-4)
# code block outputs (similarity.py)
close(dot_ab, 8.0, 1e-9)
close(8 / 9, 0.889, 5e-4)
l2_ab = sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
close(l2_ab, 1.414, 5e-4)
dot_ac = sum(x * y for x, y in zip(a, c))
eq(dot_ac, 3)
close(dot_ac / (na * sqrt(sum(x * x for x in c))), 0.333, 5e-4)
l2_ac = sqrt(sum((x - y) ** 2 for x, y in zip(a, c)))
close(l2_ac, 3.464, 5e-4)
# unit vectors
ah = [x / na for x in a]
ub = [x / nb for x in b]
uc = [x / sqrt(sum(x * x for x in c)) for x in c]
close(sum(x * y for x, y in zip(ah, ub)), 0.889, 5e-4)
close(sum((x - y) ** 2 for x, y in zip(ah, ub)), 0.222, 5e-4)
close(2 - 2 * sum(x * y for x, y in zip(ah, ub)), 0.222, 5e-4)
close(sum(x * y for x, y in zip(ah, uc)), 0.333, 5e-4)
close(sum((x - y) ** 2 for x, y in zip(ah, uc)), 1.333, 5e-4)
close(2 - 2 * sum(x * y for x, y in zip(ah, uc)), 1.333, 5e-4)

# ---- §3.1 chunking ----
def n_chunks(T, C, o):
    return ceil((T - C) / (C - o)) + 1

eq(512 - 64, 448)
close((12000 - 512) / 448, 25.64, 0.01)
eq(ceil(25.64), 26)
eq(n_chunks(12000, 512, 64), 27)
eq(n_chunks(12000, 512, 0), 24)  # no overlap: ceil(11488/512)+1 = 23+1
eq(ceil((12000 - 512) / 512) + 1, 24)

# ---- §5.1 exact search cost ----
eq(10 ** 6 * 768, 768_000_000)
close(10 ** 6 * 768, 7.68e8, 1)
eq(2 * 10 ** 6 * 768, 1_536_000_000)          # 1.536 billion FLOPs (mul + add)
close(3.072 / 100, 0.03, 5e-3)                # 3.072 GB at 100 GB/s
close(3.072 / 10, 0.3, 1e-2)                  # 3.072 GB at 10 GB/s
eq(300_000 * 768 * 4, 921_600_000)            # 300k vectors = 0.92 GB
close(300_000 * 768 * 4 / 1e9, 0.92, 5e-3)
assert 300_000 * 768 * 4 / 10e9 < 0.1         # under 0.1 s at 10 GB/s
checks += 1

# ---- §5.2 IVF ----
close(10 * (10 ** 6 / 1000), 10_000, 1e-9)
close(10 ** 6 / (10 * (10 ** 6 / 1000)), 100.0, 1e-9)

# ---- §5.4 PQ ----
close(10 ** 6 * 768 * 4, 3.072e9, 1)
close(10 ** 6 * 96, 96e6, 1)
close(96e6 / 1e6, 96.0, 1e-9)
close((10 ** 6 * 768 * 4) / (10 ** 6 * 96), 32.0, 1e-9)
eq(768 // 96, 8)  # 8 dims per sub-vector

# ---- §5.5 recall@k ----
close(4 / 5, 0.8, 1e-12)

# ---- §6.2 BM25 ----
k1, bparam, N3 = 1.5, 0.75, 3
avgdl = (6 + 5 + 4) / 3
close(avgdl, 5.0, 1e-12)


def idf(n):
    return log(1 + (N3 - n + 0.5) / (n + 0.5))


IR = idf(1)   # "refund" in 1 doc
IP = idf(3)   # "policy" in all 3
close(IR, log(1 + 2.5 / 1.5), 1e-12)
close(IR, log(2.6666666666666665), 1e-12)
close(IR, 0.981, 5e-4)
close(IP, log(1 + 0.5 / 3.5), 1e-12)
close(IP, log(1.142857142857143), 1e-12)
close(IP, 0.134, 5e-4)
close(IR / IP, 7.3, 0.05)   # "a factor of 7.3" / "7x"


def K(dl):
    return k1 * (1 - bparam + bparam * dl / avgdl)


close(K(6), 1.725, 1e-12)
close(1.5 * (0.25 + 0.75 * 6 / 5), 1.725, 1e-12)
close(1.5 * 1.15, 1.725, 1e-12)
close(K(5), 1.5, 1e-12)
close(K(4), 1.275, 1e-12)


def comp(idfv, tf, dl):
    return idfv * (tf * (k1 + 1)) / (tf + K(dl))


eq(round(comp(IR, 2, 6), 3), 1.317)   # doc shows "≈ 1.317"
close(comp(IR, 2, 6), 1.317, 5e-4)
close(0.981 * 1.342, 1.317, 1e-3)
close(comp(IP, 1, 6), 0.123, 5e-4)
d1 = comp(IR, 2, 6) + comp(IP, 1, 6)
close(d1, 1.44, 5e-3)
close(comp(IP, 1, 5), 0.13, 5e-3)       # Doc2
close(comp(IP, 1, 4), 0.15, 5e-3)       # Doc3
# saturation: f=1 vs f=2 length factor in Doc1
close((1 * (k1 + 1)) / (1 + K(6)), 0.917, 5e-4)
close((2 * (k1 + 1)) / (2 + K(6)), 1.342, 5e-4)
# Doc3 edges Doc2 (shorter doc, length normalisation)
assert comp(IP, 1, 4) > comp(IP, 1, 5)
checks += 1

# ---- §7 RRF ----
kk = 60
rrf_A = 1 / (kk + 1) + 1 / (kk + 3)
rrf_B = 1 / (kk + 2) + 1 / (kk + 1)
rrf_C = 1 / (kk + 3) + 1 / (kk + 2)
close(rrf_A, 0.03227, 5e-5)
close(rrf_B, 0.03252, 5e-5)
close(rrf_C, 0.03200, 5e-5)
assert rrf_B > rrf_A > rrf_C
checks += 1

# ---- §8 reranking cost ----
eq(20 * 512, 10_240)

# ---- §9.2 context budget ----
eq(8192 - 512 - 1024, 6656)
close(6656 / 512, 13.0, 1e-12)

# ---- §12 retrieval metrics ----
close(3 / 4, 0.75, 1e-12)   # recall@5
close(3 / 5, 0.60, 1e-12)   # precision@5
close(1 / 1, 1.0, 1e-12)    # MRR
close(log2(6), 2.585, 5e-4)
close(log2(3), 1.585, 5e-4)
close(log2(5), 2.322, 5e-4)
DCG = 1 / log2(2) + 1 / log2(4) + 1 / log2(6)
close(1 / log2(2), 1.0, 1e-12)
close(1 / log2(4), 0.5, 1e-12)
close(1 / log2(6), 0.387, 5e-4)
close(DCG, 1.887, 5e-4)
IDCG = 1 / log2(2) + 1 / log2(3) + 1 / log2(4) + 1 / log2(5)
close(1 / log2(3), 0.631, 5e-4)
close(1 / log2(5), 0.431, 5e-4)
close(IDCG, 2.562, 5e-4)
close(DCG / IDCG, 0.737, 5e-4)

# ---- §13 citation accuracy ----
close(3 / 4, 0.75, 1e-12)

# ---- §16 cost per query ----
close(6656 / 1e6 * 3 + 500 / 1e6 * 15, 0.0275, 5e-5)
close(13 * 512, 6656, 1e-9)

# ---- §17 RAG architectures: long-context vs RAG ($3 / M input) ----
close(10240 / 1e6 * 3, 0.031, 5e-4)       # reranker tokens at the chat rate
assert 10240 / 1e6 * 3 > 4000 / 1e6 * 3   # more than naive RAG's $0.012
checks += 1
eq(100 * 512, 51_200)                     # middle path: ~51K tokens
close(4000 / 1e6 * 3, 0.012, 1e-9)       # naive RAG / retrieved 4K
close(200000 / 1e6 * 3, 0.60, 1e-9)      # stuffing 200K
close(0.60 / 0.012, 50.0, 1e-9)
close(200000 / 4000, 50.0, 1e-9)

# ---- §18 sizing example: 5M x 1024 float32 vs PQ 128 ----
close(5e6 * 1024 * 4, 20.48e9, 1)
close(5e6 * 1024 * 4 / 1e9, 20.48, 1e-6)
close(5e6 * 128, 640e6, 1)
close(5e6 * 128 / 1e6, 640.0, 1e-6)
close((5e6 * 1024 * 4) / (5e6 * 128), 32.0, 1e-12)

# ---- mini-RAG code output (minirag.py cosine scores) ----
import re


def _tok(t):
    return re.findall(r"[a-z0-9]+", t.lower())


DOCS = [
    "Reset your password from the account settings page.",
    "To change your password, open Settings and click Security.",
    "Secure your account with a strong password and a backup email.",
    "Billing invoices are emailed on the first of each month.",
    "Upgrade or downgrade your plan under Billing settings.",
    "Two-factor authentication adds a one-time code from your phone.",
    "Contact support by email for a refund within thirty days.",
    "The mobile app supports fingerprint and face unlock.",
]
_vocab = sorted({t for d in DOCS for t in _tok(d)})
_col = {t: i for i, t in enumerate(_vocab)}
_N = len(DOCS)
_df = [0.0] * len(_vocab)
for _d in DOCS:
    for _t in set(_tok(_d)):
        _df[_col[_t]] += 1
_idf = [log((_N + 1) / (x + 1)) + 1.0 for x in _df]


def _embed(text):
    v = [0.0] * len(_vocab)
    for t in _tok(text):
        if t in _col:
            v[_col[t]] += 1.0
    v = [x * i for x, i in zip(v, _idf)]
    n = sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


_M = [_embed(d) for d in DOCS]
_q = _embed("how do I reset my account password")
_scores = [sum(x * y for x, y in zip(row, _q)) for row in _M]
_order = sorted(range(_N), key=lambda i: -_scores[i])[:3]
eq(_order, [0, 2, 1])
close(_scores[0], 0.649, 5e-4)
close(_scores[2], 0.276, 5e-4)
close(_scores[1], 0.134, 5e-4)

# ---- §15 TF-IDF ranking remark: doc2 shares one more query word than doc1 ----
_qset = set(_tok("how do I reset my account password"))
eq(sorted(_qset & set(_tok(DOCS[2]))), ["account", "password"])
eq(sorted(_qset & set(_tok(DOCS[1]))), ["password"])

# ---- §15.1 embedding-server stub output = plain bag-of-words cosine ----
def _bow_cos(x, y):
    cx, cy = Counter(_tok(x)), Counter(_tok(y))
    dot = sum(cx[t] * cy[t] for t in cx)
    return dot / (sqrt(sum(v * v for v in cx.values())) *
                  sqrt(sum(v * v for v in cy.values())))


_qq = "how do I reset my account password"
eq(len(set(_tok(_qq))), 7)
eq(len(set(_tok(DOCS[0]))), 8)
eq(len(_qset & set(_tok(DOCS[0]))), 3)
close(3 / sqrt(8 * 7), 0.401, 5e-4)
close(_bow_cos(DOCS[0], _qq), 0.401, 5e-4)
close(_bow_cos(DOCS[2], _qq), 0.210, 5e-4)
close(_bow_cos(DOCS[1], _qq), 0.126, 5e-4)
_bow_order = sorted(range(8), key=lambda i: -_bow_cos(DOCS[i], _qq))[:3]
eq(_bow_order, [0, 2, 1])
close(639_150_592 / 1e9, 0.64, 5e-3)          # Qwen3-Embedding-0.6B-Q8_0.gguf

# ---- exercises ch51 ----
# Ex1: 2M x 1024 float32 vs PQ 128 codes
close(2e6 * 1024 * 4, 8.192e9, 1)
close(8.192e9 / 1e9, 8.192, 1e-6)
close(2e6 * 128, 2.56e8, 1)
close(2.56e8 / 1e6, 256.0, 1e-6)
close((1024 * 4) / 128, 32.0, 1e-12)
# Ex2: 30000 @ 1000/200
eq(1000 - 200, 800)
close((30000 - 1000) / 800, 36.25, 1e-9)
eq(ceil(36.25), 37)
eq(n_chunks(30000, 1000, 200), 38)
# Ex3: IVF 4000 cells / 4M, probe 20
close(4e6 / 4000, 1000.0, 1e-9)
close(20 * 1000, 20_000, 1e-9)
close(4e6 / 20000, 200.0, 1e-9)
# Ex4: recall@10 9/10
close(9 / 10, 0.9, 1e-12)
# Ex5: RRF ranks (1,5) vs (4,2), k=60
e5_1 = 1 / 61 + 1 / 65
e5_2 = 1 / 64 + 1 / 62
close(e5_1, 0.031778, 5e-6)
close(e5_2, 0.031754, 5e-6)
assert e5_1 > e5_2
checks += 1
# Ex6: retrieved [d1..d5], relevant {d2,d3}
close(2 / 2, 1.0, 1e-12)   # recall@5
close(2 / 5, 0.4, 1e-12)   # precision@5
close(1 / 2, 0.5, 1e-12)   # MRR (first relevant at rank 2)
# Ex8: groundedness 4/5
close(4 / 5, 0.8, 1e-12)

# ================================================================ chapter 54
# ---- §2/§3.2 token F1 ----
pred = "the total is 42".split()
gold = "the invoice total is 42 dollars".split()
common = sum((Counter(pred) & Counter(gold)).values())
eq(common, 4)
eq(len(pred), 4)
eq(len(gold), 6)
P = common / len(pred)
R = common / len(gold)
close(P, 1.0, 1e-12)
close(R, 2 / 3, 1e-12)
close(R, 0.667, 5e-4)
close(2 * P * R / (P + R), 0.8, 1e-12)

# ---- §3.3 ROUGE-1 ----
cand = "refund issued today".split()
rref = "the refund was issued today".split()
ov = sum((Counter(cand) & Counter(rref)).values())
eq(ov, 3)
eq(len(rref), 5)
eq(len(cand), 3)
close(ov / len(rref), 0.6, 1e-12)
# opposite-meaning candidate still gets perfect ROUGE-1 recall
_ref2 = "the refund was approved".split()
_cand2 = "the refund was not approved".split()
_ov2 = sum((Counter(_cand2) & Counter(_ref2)).values())
eq(_ov2, 4)
eq(len(_ref2), 4)
close(_ov2 / len(_ref2), 1.0, 1e-12)
close(_ov2 / len(_cand2), 0.8, 1e-12)
close(ov / len(cand), 1.0, 1e-12)
rr, rp = ov / len(rref), ov / len(cand)
close(2 * rr * rp / (rr + rp), 0.75, 1e-9)

# ---- §4 pass@k ----
n6, c2 = 6, 2
eq(comb(4, 1), 4)
eq(comb(6, 1), 6)
close(1 - comb(n6 - c2, 1) / comb(n6, 1), 1 / 3, 1e-12)
close(1 - 4 / 6, 0.333, 5e-4)
eq(comb(4, 3), 4)
eq(comb(6, 3), 20)
close(1 - comb(n6 - c2, 3) / comb(n6, 3), 0.8, 1e-12)
close(1 - 4 / 20, 0.8, 1e-12)

# ---- §6.2 Cohen's kappa ----
both_p, ab, ba, both_f, tot = 45, 10, 5, 40, 100
po = (both_p + both_f) / tot
close(po, 0.85, 1e-12)
Ap = (both_p + ab) / tot
Bp = (both_p + ba) / tot
close(Ap, 0.55, 1e-12)
close(Bp, 0.50, 1e-12)
pe = Ap * Bp + (1 - Ap) * (1 - Bp)
close(pe, 0.5, 1e-12)
close(0.55 * 0.50 + 0.45 * 0.50, 0.5, 1e-12)
close((po - pe) / (1 - pe), 0.70, 1e-12)

# ---- §8.1 Wilson 40/100 (agrees with ch43) ----
z = 1.96
nn, kcnt = 100, 40
ph = kcnt / nn
z2 = z * z
close(z2, 3.8416, 1e-9)
den = 1 + z2 / nn
close(den, 1.0384, 5e-5)
centre = (ph + z2 / (2 * nn)) / den
close(ph + z2 / (2 * nn), 0.4192, 5e-5)
close(0.4 + 0.0192, 0.4192, 1e-9)
close(centre, 0.4037, 5e-5)
root = sqrt(ph * (1 - ph) / nn + z2 / (4 * nn * nn))
close(root, 0.04996, 5e-6)
half = z * root / den
close(1.96 * 0.04996 / 1.0384, 0.0943, 5e-4)
close(half, 0.0943, 5e-5)
close(centre - half, 0.309, 5e-4)
close(centre + half, 0.498, 5e-4)

# ---- §8.2 sample size for 5-point difference ----
za, zb = 1.96, 0.84
p1s, p2s, delta = 0.5, 0.55, 0.05
close((za + zb) ** 2, 7.84, 1e-9)
var = p1s * (1 - p1s) + p2s * (1 - p2s)
close(var, 0.4975, 1e-12)
close(0.25 + 0.2475, 0.4975, 1e-12)
nsz = (za + zb) ** 2 * var / delta ** 2
close(nsz, 1560.16, 0.01)
eq(ceil(nsz), 1561)
close(16 * 0.25 / delta ** 2, 1600.0, 1e-6)

# ---- §8.3 McNemar ----
bdisc, cdisc = 15, 5
close((bdisc - cdisc) ** 2 / (bdisc + cdisc), 5.0, 1e-12)
eq((15 - 5) ** 2, 100)
eq(15 + 5, 20)
close((abs(bdisc - cdisc) - 1) ** 2 / (bdisc + cdisc), 4.05, 1e-12)
eq((10 - 1) ** 2, 81)
close(81 / 20, 4.05, 1e-12)

# ---- §8.4 unpaired two-proportion ----
pA, pB = 75 / 100, 65 / 100
pooled = (75 + 65) / 200
close(pooled, 0.70, 1e-12)
se = sqrt(pooled * (1 - pooled) * (2 / 100))
close(0.21 * 0.02, 0.0042, 1e-12)
close(se, 0.0648, 5e-5)
close((pA - pB) / se, 1.54, 5e-3)
assert (pA - pB) / se < 1.96   # not significant unpaired
assert 5.0 > 3.841             # significant paired
checks += 2

# ---- §13 agentic per-step reliability compounding ----
close(0.95 ** 10, 0.60, 5e-3)

# ---- §12 error analysis tally ----
tally = {"retrieval": 18, "halluc": 12, "format": 8, "refusal": 7, "other": 5}
eq(sum(tally.values()), 50)
close(18 / 50, 0.36, 1e-12)
close(12 / 50, 0.24, 1e-12)
close(8 / 50, 0.16, 1e-12)
close(7 / 50, 0.14, 1e-12)
close(5 / 50, 0.10, 1e-12)

# ---- §9 harness output (fake model) ----
CASES = [
    ("I was charged twice this month", "billing", "easy"),
    ("the app crashes when I open settings", "bug", "easy"),
    ("how do I export my data?", "howto", "easy"),
    ("refund the duplicate charge please", "billing", "hard"),
    ("login button does nothing on mobile", "bug", "hard"),
    ("where is the invoice for March?", "billing", "hard"),
]


def _model(text):
    t = text.lower()
    if any(w in t for w in ("charge", "refund", "billing")):
        return "billing"
    if any(w in t for w in ("crash", "bug", "error")):
        return "bug"
    return "howto"


res = [(_model(t) == g, tag, i) for i, (t, g, tag) in
       zip("t1 t2 t3 t4 t5 t6".split(), CASES)]
kpass = sum(1 for ok, _, _ in res if ok)
eq(kpass, 4)
eq(len(res), 6)
close(kpass / 6, 0.67, 5e-3)
eq(sum(1 for ok, tag, _ in res if tag == "easy" and ok), 3)
eq(sum(1 for ok, tag, _ in res if tag == "hard" and ok), 1)
eq([i for ok, _, i in res if not ok], ["t5", "t6"])
# harness Wilson CI on 4/6
ph6 = 4 / 6
den6 = 1 + z2 / 6
c6 = (ph6 + z2 / (2 * 6)) / den6
h6 = z * sqrt(ph6 * (1 - ph6) / 6 + z2 / (4 * 36)) / den6
close(c6 - h6, 0.30, 5e-3)
close(c6 + h6, 0.90, 5e-3)

# ---- §9.1 served-model stub output: 5/6, only t6 fails ----
def _stub(text):
    t = text.lower()
    if any(w in t for w in ("charge", "refund", "bill")):
        return "billing"
    if any(w in t for w in ("crash", "bug", "error", "button", "does nothing")):
        return "bug"
    return "howto"


sres = [(_stub(t) == g, i) for i, (t, g, _) in
        zip("t1 t2 t3 t4 t5 t6".split(), CASES)]
eq(sum(1 for ok, _ in sres if ok), 5)
close(5 / 6, 0.83, 5e-3)
eq([i for ok, i in sres if not ok], ["t6"])

# ---- exercises ch54 ----
# Ex1 token F1
ep = "the total is 42 dollars".split()
eg = "42 dollars".split()
ec = sum((Counter(ep) & Counter(eg)).values())
eq(ec, 2)
close(ec / len(ep), 0.4, 1e-12)
close(ec / len(eg), 1.0, 1e-12)
close(2 * 0.4 * 1.0 / 1.4, 0.571, 5e-4)
# Ex2 pass@k n=5 c=1
eq(comb(4, 1), 4)
eq(comb(5, 1), 5)
close(1 - comb(4, 1) / comb(5, 1), 0.2, 1e-12)
eq(comb(4, 5), 0)
close(1 - comb(4, 5) / comb(5, 5), 1.0, 1e-12)
# Ex3 kappa
close(0.70 * 0.60 + 0.30 * 0.40, 0.54, 1e-12)
close((0.80 - 0.54) / (1 - 0.54), 0.565, 5e-4)
# Ex4 Wilson 150/200
ph2 = 150 / 200
close(ph2, 0.75, 1e-12)
close(1.96 * sqrt(0.75 * 0.25 / 200), 0.060, 5e-4)   # normal shortcut
close(z2 / 200, 0.0192, 5e-5)
close(1 + z2 / 200, 1.0192, 5e-5)
close(z2 / 400, 0.0096, 5e-5)
c200 = (ph2 + z2 / 400) / (1 + z2 / 200)
close(c200, 0.745, 5e-4)
close(0.75 * 0.25 / 200, 0.0009375, 1e-12)
close(z2 / (4 * 200 ** 2), 0.000024, 5e-7)
r200 = sqrt(0.75 * 0.25 / 200 + z2 / (4 * 200 ** 2))
close(r200, 0.0310, 5e-5)
h200 = 1.96 * r200 / (1 + z2 / 200)
close(h200, 0.060, 5e-4)
close(c200 - h200, 0.686, 5e-4)
close(c200 + h200, 0.805, 5e-4)
assert c200 - h200 < 0.73 < c200 + h200
checks += 1
# Ex5 McNemar b=5 c=1
close((5 - 1) ** 2 / (5 + 1), 2.67, 5e-3)
assert (5 - 1) ** 2 / (5 + 1) < 3.841
checks += 1
# Ex7 multiple comparisons
close(10 * 0.05, 0.5, 1e-12)
close(0.05 / 10, 0.005, 1e-12)
# Ex8 tally 40 failures, hallucination 20
eq(20 + 8 + 7 + 5, 40)
close(20 / 40, 0.5, 1e-12)

print(f"ch51_54_rag_eval: all {checks} checks passed (chapters 51, 54)")
