#!/usr/bin/env python3
"""Check every number in the "On paper" cards, worked examples and exercise
answers of docs/60-ai-data-preparation.md (stdlib only)."""
from math import log

checks = 0


def close(a, b, tol):
    global checks
    checks += 1
    assert abs(a - b) <= tol, f"{a!r} != {b!r} (tol {tol})"


def eq(a, b):
    global checks
    checks += 1
    assert a == b, f"{a!r} != {b!r}"


# ---------------------------------------------------------------- §5 Unicode
import unicodedata

_raw = "Invoice １２３ due; ﬁnal notice"
eq(len(_raw), 28)
eq(unicodedata.normalize("NFKC", _raw), "Invoice 123 due; final notice")
eq(len(unicodedata.normalize("NFKC", _raw)), 29)
eq(unicodedata.normalize("NFC", "é"), "é")          # NFC composes é
eq(unicodedata.normalize("NFKC", "a b"), "a b")           # no-break space folded
eq(unicodedata.normalize("NFKC", "x²"), "x2")             # lossy: superscript
eq(unicodedata.normalize("NFKC", "½"), "1⁄2")        # lossy: 1/2 fraction
eq(unicodedata.normalize("NFKC", "™"), "TM")              # lossy: trademark
for _c in "’“–":                                 # smart quotes/dash survive
    eq(unicodedata.normalize("NFC", _c), _c)
    eq(unicodedata.normalize("NFKC", _c), _c)
eq(unicodedata.category("‍"), "Cf")                       # ZWJ is in category C

# ---------------------------------------------------------------- §6 dedup
# Jaccard of 2-word shingles of two short sentences.
s1 = {"reset your", "your password", "password now"}
s2 = {"reset your", "your password", "password today"}
eq(len(s1), 3)
eq(len(s2), 3)
eq(len(s1 & s2), 2)          # {"reset your", "your password"}
eq(len(s1 | s2), 4)
close(len(s1 & s2) / len(s1 | s2), 0.5, 1e-12)

# MinHash runnable estimate (deterministic SHA-1 salted hashes, num_perm=200).
import hashlib


def _shingles(text, k=2):
    w = text.lower().split()
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}


def _sig(s, num_perm=200):
    out = []
    for p in range(num_perm):
        out.append(min(int.from_bytes(
            hashlib.sha1(f"{p}:{sh}".encode()).digest()[:8], "big") for sh in s))
    return out


def _est(a, b):
    return sum(x == y for x, y in zip(a, b)) / len(a)


a1 = _shingles("reset your password now")
a2 = _shingles("reset your password today")
close(_est(_sig(a1), _sig(a2)), 0.525, 1e-9)   # matches the doc's printed 0.525
from math import sqrt
_se = sqrt(0.5 * 0.5 / 200)                      # standard error of the estimate
close(_se, 0.0353553, 1e-6)
close(_se, 0.035, 5e-4)
assert abs(0.525 - 0.5) < _se                    # within one standard error
close(sqrt(0.5 * 0.5 / 800), _se / 2, 1e-12)     # 4x permutations halves it
close(3 / 4, 0.75, 1e-12)                        # three shared of four distinct

# LSH band-collision probability P = 1 - (1 - s^r)^b.
def lsh(s, r, b):
    return 1 - (1 - s ** r) ** b


close(0.5 ** 2, 0.25, 1e-12)
close(1 - 0.25, 0.75, 1e-12)
close(0.75 ** 4, 0.31640625, 1e-12)
close(lsh(0.5, 2, 4), 0.68359375, 1e-9)
close(lsh(0.5, 2, 4), 0.684, 5e-4)
close(0.8 ** 2, 0.64, 1e-12)
close(1 - 0.64, 0.36, 1e-12)
close(0.36 ** 4, 0.01679616, 1e-12)
close(lsh(0.8, 2, 4), 0.98320384, 1e-9)
close(lsh(0.8, 2, 4), 0.983, 5e-4)
close((1 / 4) ** (1 / 2), 0.5, 1e-12)            # threshold ~ (1/b)^(1/r)
# Same threshold 0.5, longer signature (r=4, b=16): steeper step.
close((1 / 16) ** (1 / 4), 0.5, 1e-12)
assert lsh(0.3, 4, 16) < lsh(0.3, 2, 4) and lsh(0.7, 4, 16) > lsh(0.7, 2, 4)

# ---------------------------------------------------------------- §7 quality
# Precision/recall of a low-quality filter on a hand-labeled sample of 100.
TP, FP, FN = 18, 7, 2
eq(TP + FP, 25)              # flagged 25
eq(TP + FN, 20)              # 20 truly low-quality
close(TP / (TP + FP), 0.72, 1e-12)      # precision
close(TP / (TP + FN), 0.90, 1e-12)      # recall

# LLM-as-filter cost at assumed $3/M input, $15/M output.
cost_in = 800 / 1e6 * 3
cost_out = 20 / 1e6 * 15
close(cost_in, 0.0024, 1e-12)
close(cost_out, 0.0003, 1e-12)
close(cost_in + cost_out, 0.0027, 1e-12)
close(100_000 * (cost_in + cost_out), 270.0, 1e-9)

# ---------------------------------------------------------------- §9 labeling
# Percent agreement and a re-label noise estimate.
close(82 / 100, 0.82, 1e-12)
close(0.3 * 0.3 + 0.7 * 0.7, 0.58, 1e-12)   # chance agreement at 30% base rate
close((0.82 - 0.58) / (1 - 0.58), 0.5714286, 1e-6)   # Cohen's kappa
close((0.82 - 0.58) / (1 - 0.58), 0.57, 5e-3)
close(16 / 200, 0.08, 1e-12)
close(1 - 16 / 200, 0.92, 1e-12)              # score ceiling vs noisy labels

# Labelling cost: 45 s per item.
eq(10_000 * 45, 450_000)
close(10_000 * 45 / 3600, 125, 1e-12)
eq(int(0.2 * 10_000), 2_000)
close(2_000 * 45 / 3600, 25, 1e-12)

# ---------------------------------------------------------------- §10 synthetic yield
N, pv, pd = 10_000, 0.6, 0.8
eq(N * pv, 6000)
close(N * pv * pd, 4800, 1e-9)
close(5000 / (pv * pd), 10416.666666666666, 1e-6)
import math
eq(math.ceil(5000 / (pv * pd)), 10417)
close(pv * pd, 0.48, 1e-12)
assert pv * pd < 0.5                          # "less than half"

# ---------------------------------------------------------------- §12 contamination
def ngrams(n_words, k):
    return n_words - k + 1


eq(ngrams(10, 8), 3)
eq(ngrams(9, 8), 2)
eq(1, 1)                     # one shared 8-gram
close(1 / 2, 0.5, 1e-12)     # overlap fraction of the eval item

# ---------------------------------------------------------------- §16 drift (PSI / KL)
def psi(p, q):
    return sum((pi - qi) * log(pi / qi) for pi, qi in zip(p, q))


def kl(q, p):
    return sum(qi * log(qi / pi) for qi, pi in zip(q, p) if qi > 0)


base = [0.5, 0.3, 0.2]
prod = [0.4, 0.4, 0.2]
close((0.5 - 0.4) * log(0.5 / 0.4), 0.0223144, 1e-6)
close((0.3 - 0.4) * log(0.3 / 0.4), 0.0287682, 1e-6)
close(psi(base, prod), 0.051083, 1e-5)
close(psi(base, prod), 0.051, 5e-4)
close(0.4 * log(0.4 / 0.5), -0.0892574, 1e-6)
close(0.4 * log(0.4 / 0.3), 0.1150728, 1e-6)
close(kl(prod, base), 0.025816, 1e-5)
close(kl(prod, base), 0.026, 5e-4)          # KL(q||p)
close(kl(base, prod), 0.025267, 1e-5)       # KL(p||q)
close(kl(base, prod), 0.025, 5e-4)
close(kl(base, prod) + kl(prod, base), psi(base, prod), 1e-12)   # PSI = sum

# ---------------------------------------------------------------- §17 mini pipeline
# Summary counts printed by the runnable pipeline (recomputed independently).
eq(10 - 1 - 1, 8)            # dedup: 1 exact + 1 near removed
eq(8, 8)                     # pii mask keeps count, masks 1
eq(8 - 2, 6)                 # quality filter drops 2 (junk + too short)
eq(6, 6)                     # final records
_d04 = "Invoice １２３ total ＄９９ due now."
eq(len(_d04), 30)                          # characters
eq(len(_d04.encode("utf-8")), 42)          # bytes: 6 full-width chars x 3 bytes
import re as _re


def _pshingles(t, k=2):
    w = _re.sub(r"[^a-z0-9 ]", "", t.lower()).split()
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}


_d02 = "Refunds are processed within 14 days of the request."
_d03n = "Refunds are processed within 14 DAYS of the request."   # d03 after normalize
assert _d02 != _d03n                       # not an exact (case-sensitive) dup
close(len(_pshingles(_d02) & _pshingles(_d03n))
      / len(_pshingles(_d02) | _pshingles(_d03n)), 1.0, 1e-12)

# ---------------------------------------------------------------- exercises
# Ex1 Jaccard
e1a = {"error 500", "500 on", "on checkout"}
e1b = {"error 500", "500 at", "at checkout"}
eq(len(e1a & e1b), 1)
eq(len(e1a | e1b), 5)
close(1 / 5, 0.2, 1e-12)

# Ex2 LSH at s=0.6, r=3, b=10
close(0.6 ** 3, 0.216, 1e-12)
close(1 - 0.216, 0.784, 1e-12)
close(0.784 ** 10, 0.0877320, 1e-6)
close(lsh(0.6, 3, 10), 0.912268, 1e-6)
close(lsh(0.6, 3, 10), 0.912, 5e-4)

# Ex3 dedup savings
eq(int(1_000_000 * 0.18), 180_000)
eq(1_000_000 - 180_000, 820_000)
close(820_000 * 0.88, 721_600, 1e-6)
eq(1_000_000 - 721_600, 278_400)

# Ex4 precision/recall
TP4, FP4, FN4 = 45, 15, 5
eq(TP4 + FP4, 60)
eq(TP4 + FN4, 50)
close(TP4 / (TP4 + FP4), 0.75, 1e-12)
close(TP4 / (TP4 + FN4), 0.90, 1e-12)

# Ex5 yield target
close(12_000 / (0.5 * 0.75), 32_000, 1e-9)

# Ex6 contamination
eq(ngrams(20, 13), 8)
close(3 / 8, 0.375, 1e-12)

# Ex7 PSI
b7 = [0.6, 0.25, 0.15]
p7 = [0.5, 0.30, 0.20]
close((0.6 - 0.5) * log(0.6 / 0.5), 0.0182322, 1e-6)
close((0.25 - 0.30) * log(0.25 / 0.30), 0.0091161, 1e-6)
close((0.15 - 0.20) * log(0.15 / 0.20), 0.0143841, 1e-6)
close(psi(b7, p7), 0.041732, 1e-5)
close(psi(b7, p7), 0.042, 5e-4)

# Ex8 percent agreement
close(120 / 150, 0.80, 1e-12)

print(f"ch60_data_prep: all {checks} checks passed")
