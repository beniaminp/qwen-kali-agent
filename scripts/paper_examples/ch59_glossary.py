#!/usr/bin/env python3
"""Check every number in the examples of docs/59-ai-glossary.md (stdlib only).

Each glossary entry's *Example:* line is asserted here, in the order the
entries appear. The module also checks the page's structure: one ``## X``
section per letter, entries sorted alphabetically inside each section, every
entry with an example and at least one link, and a jump bar that matches the
sections.
"""
import html
import re
import unicodedata
from math import ceil, comb, cos, e, erf, exp, log, log2, pi, sin, sqrt
from pathlib import Path

checks = 0


def close(a, b, tol):
    global checks
    checks += 1
    assert abs(a - b) <= tol, f"{a!r} != {b!r} (tol {tol})"


def eq(a, b):
    global checks
    checks += 1
    assert a == b, f"{a!r} != {b!r}"


def sigmoid(z):
    return 1 / (1 + exp(-z))


def softmax(z):
    m = max(z)
    w = [exp(v - m) for v in z]
    s = sum(w)
    return [x / s for x in w]


def phi(x):  # standard normal CDF
    return 0.5 * (1 + erf(x / sqrt(2)))


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


# ------------------------------------------------------------------------ A
# A/B test: 300/400 vs 332/400
pool = (300 + 332) / 800
eq(300 + 332, 632)
close(pool, 0.79, 1e-12)
close(300 / 400, 0.75, 1e-12)
close(332 / 400, 0.83, 1e-12)
se = sqrt(pool * (1 - pool) * (2 / 400))
close(se, 0.0288, 5e-5)
z = 0.08 / se
close(z, 2.78, 5e-3)
close(2 * (1 - phi(z)), 0.0055, 2e-4)
# Abstention
close(20 / 25, 0.8, 1e-12)
# Accuracy
close((80 + 860) / 1000, 0.94, 1e-12)
close(900 / 1000, 0.90, 1e-12)
# Activation function
close(sigmoid(0), 0.5, 1e-12)
eq(max(0, -2), 0)
eq(max(0, 3), 3)
# Activations: fp32 logits
close(2048 * 151936 * 4 / 1e9, 1.24, 5e-3)
close(4096 * 151936 * 4 / 1e9, 2.49, 5e-3)
# Active learning: most uncertain item
pool_p = [0.133, 0.448, 0.526, 0.878, 0.949]
unc = [1 - abs(p - 0.5) * 2 for p in pool_p]
eq(pool_p[unc.index(max(unc))], 0.526)
# Active parameters: Qwen3-30B-A3B
close(30.5 * 0.5625, 17.2, 0.05)
close(2 * 3.3e9, 6.6e9, 1)
# Adam: first step is exactly eta, independent of gradient size
for g in (0.5, 5000.0):
    m1 = 0.1 * g
    v1 = 0.001 * g * g
    step = 0.001 * (m1 / (1 - 0.9)) / sqrt(v1 / (1 - 0.999))
    close(step, 0.001, 1e-12)
# AdamW: 12 bytes per parameter for 14.77B
close(12 * 14.77e9 / 1e9, 177, 0.5)
# ADK vs LangGraph SLOC (61 §16)
SLOC = {"Agno": 19, "DSPy": 19, "LangGraph": 22, "Strands": 23, "LlamaIndex": 24,
        "OpenAI Agents": 25, "Pydantic AI": 25, "smolagents": 25, "CrewAI": 27,
        "Haystack": 29, "Google ADK": 34, "Baseline": 43}
eq(SLOC["Google ADK"], 34)
eq(SLOC["LangGraph"], 22)
# Agent: two tool calls, then the answer = three model calls
eq(2 + 1, 3)
# Agent framework: 43 vs 19 lines
eq(SLOC["Baseline"], 43)
close(43 / 19, 2.3, 0.05)
# Agentic RAG: four iterations ~4x
eq(4 * 1, 4)
# Agno: 19 lines
eq(SLOC["Agno"], 19)
# AI engineer: prefix-cache saving (45 §1.2)
before = (2000 * 3 + 300 * 15) / 1e6
after = (400 * 3 + 1600 * 0.1 * 3 + 300 * 15) / 1e6
close(before, 0.0105, 1e-12)
close(after, 0.00618, 1e-12)
close(1 - after / before, 0.41, 5e-3)
eq(2000 - 1600, 400)
# ANN / IVF
eq(10 * 10**6 // 1000, 10_000)
eq(10**6 // 10_000, 100)
# Attack-success rate: 0/4 Wilson upper bound
close(4 / 4, 1.0, 0)
close(1.96**2, 3.84, 5e-3)
close(3.84 / 7.84, 0.49, 5e-3)
close(wilson(0, 4)[1], 0.49, 5e-3)
# Attention / QKV: scores (2, 0) / sqrt(4) -> softmax
close(sqrt(4), 2, 0)
w = softmax([2 / 2, 0 / 2])
close(w[0], 0.731, 5e-4)
close(w[1], 0.269, 5e-4)
# Autoregressive
close(0.5 * 0.4 * 0.25, 0.05, 1e-12)
# Average precision: positives at ranks 1, 2, 4
close((1 / 1 + 2 / 2 + 3 / 4) / 3, 0.917, 5e-4)
close(3 / 4, 0.75, 0)

# ------------------------------------------------------------------------ B
# Backpropagation: L = (wx - y)^2, w=2, x=3, y=5
w_, x_, y_ = 2, 3, 5
eq(2 * (w_ * x_ - y_) * x_, 6)
# Base rate (fraud)
p_flag = 0.9 * 0.01 + 0.05 * 0.99
close(0.9 * 0.01, 0.009, 1e-12)
close(p_flag, 0.0585, 1e-12)
close(0.009 / 0.0585, 0.15, 5e-3)
# Baseline: 186 of 300
close(186 / 300, 0.62, 1e-12)
# Batch: steps, and noise halves from 8 to 32
eq(ceil(50_000 / 32), 1563)
eq(1563 * 3, 4689)
close(sqrt(8 / 32), 0.5, 1e-12)
# Batching: 8 x 24 and the KV-limited bound
eq(8 * 24, 192)
close(4096 * 163_840 / 1e9, 0.67, 5e-3)
close(8 * 200 / (8.3 + 8 * 0.67), 117, 0.5)
# BatchNorm: (1, 3)
mean = (1 + 3) / 2
sd = sqrt(((1 - mean) ** 2 + (3 - mean) ** 2) / 2)
eq(mean, 2)
eq(sd, 1)
eq(((1 - mean) / sd, (3 - mean) / sd), (-1, 1))
# Bayes' rule: spam filter
close(0.95 * 0.2, 0.19, 1e-12)
close(0.02 * 0.8, 0.016, 1e-12)
close(0.19 / (0.19 + 0.016), 0.92, 5e-3)
# bf16
eq(137 * 512, 70_144)
eq(2 ** (16 - 7), 512)
close(2**-7, 0.0078, 5e-5)
eq(round(70_000 / 512) * 512, 70_144)
# Bi-encoder
eq(10**6 * 768, 768_000_000)
# Bias (fairness): 50/100 vs 30/100
close(50 / 100 - 30 / 100, 0.2, 1e-12)
# BLEU: unigram precision of the negated answer
cand = "the refund was not approved".split()
ref = set("the refund was approved".split())
eq(sum(t in ref for t in cand), 4)
close(4 / 5, 0.8, 0)
# BM25 IDF
close(log(1 + (3 - 1 + 0.5) / (1 + 0.5)), 0.981, 5e-4)
close(log(1 + (3 - 3 + 0.5) / (3 + 0.5)), 0.134, 5e-4)
# BPE: (e, s) count
eq(6 + 3, 9)
eq(2**8, 256)
# Break-even
eq(2 * 24 * 30, 1440)
close(1440 / 0.0007 / 1e6, 2.06, 5e-3)
# Brier
sq = [(0.9 - 1) ** 2, (0.8 - 1) ** 2, (0.3 - 0) ** 2, (0.6 - 0) ** 2]
for got, want in zip(sq, (0.01, 0.04, 0.09, 0.36)):
    close(got, want, 1e-12)
close(sum(sq) / 4, 0.125, 1e-12)
close(0.5**2, 0.25, 0)
# Broadcasting: (4,) - (4, 1) -> (4, 4), and the two MSEs (46 §4.5)
y_true = [1.0, 2.0, 3.0, 4.0]
y_pred = [1.1, 1.9, 3.2, 3.9]
wrong = sum((t - p) ** 2 for t in y_true for p in y_pred) / 16
right = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / 4
eq(4 * 4, 16)
close(wrong, 2.4425, 5e-5)
close(right, 0.0175, 5e-5)
# Budget (agent): e.g. 20 tool calls, 10 minutes (11 §5.3) - quoted values
eq((20, 10), (20, 10))

# ------------------------------------------------------------------------ C
# Calibration: 80 of 100 at stated 0.9
close(80 / 100, 0.8, 0)
# Cascade
close(0.0001 + 0.3 * 0.0010, 0.0004, 1e-12)
close(1 - 0.0004 / 0.0010, 0.60, 1e-12)
# Causal mask: entries above the diagonal of 3x3
eq(sum(1 for i in range(3) for j in range(3) if j > i), 3)
eq([sum(1 for j in range(3) if j <= i) for i in range(3)], [1, 2, 3])
# Central limit theorem: SE at 0.8 over 400
close(sqrt(0.16 / 400), 0.02, 1e-12)
# Chain rule: L = u^2, u = 3w, w = 1
u = 3 * 1
eq(2 * u, 6)
eq(2 * u * 3, 18)
# Chain-of-thought: 500 / 20
eq(500 // 20, 25)
# Chinchilla
close(20 * 14.77e9 / 1e9, 295, 0.5)
close(36e12 / 14.77e9, 2437, 0.5)
# Chunking: 12,000 tokens, C = 512, o = 64
eq(512 - 64, 448)
eq(12_000 - 512, 11_488)
eq(ceil(11_488 / 448) + 1, 27)
# Circuit breaker: 5 failures / 30 s / 60 s (stated policy) - nothing to compute
# Citation accuracy
close(3 / 4, 0.75, 0)
# Class imbalance: 1% positives
close(1 - 0.01, 0.99, 1e-12)
# Class weights
close(1000 / (2 * 900), 0.556, 5e-4)
eq(1000 / (2 * 100), 5)
close(5 / (1000 / 1800), 9, 1e-12)
# CNN: kernel (1, -1) over (2, 5, 5, 1)
seq = [2, 5, 5, 1]
eq([seq[i] - seq[i + 1] for i in range(3)], [-3, 0, 4])
# Code agent: two tool calls in one step
eq(2, 2)
# Cohen's kappa (54 §6.2)
p_o = (45 + 40) / 100
p_e = 0.55 * 0.50 + 0.45 * 0.50
close(p_o, 0.85, 1e-12)
close(p_e, 0.5, 1e-12)
close((p_o - p_e) / (1 - p_e), 0.70, 1e-12)
close(p_o - p_e, 0.35, 1e-12)
# Compute-bound: 2N GFLOPs per token
close(2 * 14.77, 29.5, 0.05)
# Confidence interval: 320/400
close(1.96 * 0.02, 0.0392, 1e-12)
close(0.8 - 0.0392, 0.761, 5e-4)
close(0.8 + 0.0392, 0.839, 5e-4)
# config.json: KV bytes per token
eq(2 * 40 * 8 * 128 * 2, 163_840)
# Confusion matrix
close(80 / 120, 0.667, 5e-4)
close(80 / 100, 0.8, 0)
close(40 / 900, 0.044, 5e-4)
eq(80 + 40, 120)
# Contamination: 500-example golden set, 94% (60 §1.2) - quoted
eq(500, 500)
# Context budget
eq(8192 - 512 - 1024, 6656)
eq(6656 // 512, 13)
# Context window
eq(1500 + 4800 + 6000 + 200, 12_500)
eq(8 * 600, 4800)
eq(32_768 - 12_500, 20_268)
# Cosine similarity
a, b, c = (1, 2, 2), (2, 1, 2), (-2, 1, 0)
dot = lambda p, q: sum(x * y for x, y in zip(p, q))
norm = lambda p: sqrt(dot(p, p))
eq(dot(a, b), 8)
eq(norm(a), 3)
eq(norm(b), 3)
close(dot(a, b) / (norm(a) * norm(b)), 0.889, 5e-4)
close(8 / 9, 0.889, 5e-4)
eq(dot(a, c), 0)
# Cost growth (agents)
eq(10 * 11 // 2, 55)
eq(500 * 55, 27_500)
eq(500 * 10, 5000)
close(27_500 / 5000, 5.5, 0)
eq(500 * 20 * 21 // 2, 105_000)
# Cost per request
close(800 / 1e6 * 0.50, 0.0004, 1e-12)
close(200 / 1e6 * 1.50, 0.0003, 1e-12)
close(0.0004 + 0.0003, 0.0007, 1e-12)
close(0.0007 * 1000, 0.70, 1e-12)
# CrewAI: 27 lines
eq(SLOC["CrewAI"], 27)
# Cross-encoder
eq(20 * 512, 10_240)
# Cross-entropy
close((log(2) + log(4) + log(8)) / 3, 2 * log(2), 1e-12)
close(2 * log(2), 1.386, 5e-4)
# Cross-validation: 5-fold on 1,000
eq(1000 // 5, 200)
eq(1000 - 200, 800)

# ------------------------------------------------------------------------ D
# Data contract: thresholds are stated policy
# Denial of wallet
eq(100 * 8000, 800_000)
close(800_000 / 1e6 * 3, 2.40, 1e-12)
close(2.40 * 60 * 24, 3456, 1e-9)
# Dense model: 29.5 vs 6.6 GFLOPs
close(2 * 14.77 / 6.6, 4.5, 0.05)
# Derivative: f = w^2 at 3
eq(2 * 3, 6)
close(3.01**2 - 9, 0.0601, 1e-9)
close(6 * 0.01, 0.06, 1e-12)
# Distillation: (2 ln 6, 2 ln 2, 0) at T = 2
zT = [2 * log(6) / 2, 2 * log(2) / 2, 0.0]
for got, frac, shown in zip(softmax(zT), (6 / 9, 2 / 9, 1 / 9), (0.667, 0.222, 0.111)):
    close(got, frac, 1e-12)
    close(got, shown, 5e-4)
# Dot product (token prices)
eq(1000 * 2 + 4000 * 0.5 + 250 * 8, 6000)
close(6000 / 1e6, 0.006, 1e-12)
# DPO
margin = 0.1 * (1.0 - (-1.0))
close(margin, 0.2, 1e-12)
close(sigmoid(0.2), 0.550, 5e-4)
close(-log(sigmoid(0.2)), 0.598, 5e-4)
close(-log(0.5), 0.693, 5e-4)
# Drift: PSI
p_, q_ = (0.5, 0.3, 0.2), (0.4, 0.4, 0.2)
terms = [(a_ - b_) * log(a_ / b_) for a_, b_ in zip(p_, q_)]
close(terms[0], 0.0223, 5e-5)
close(terms[1], 0.0288, 5e-5)
close(sum(terms), 0.051, 5e-4)
# Dropout
h = (2, 4, 6, 8)
mask = (1, 0, 1, 0)
eq(tuple(hi * mi / 0.5 for hi, mi in zip(h, mask)), (4, 0, 12, 0))
# DSPy: 19 lines
eq(SLOC["DSPy"], 19)

# ------------------------------------------------------------------------ E
# Early stopping: best at 3, stop after 2 non-improving evals
vals = [1.20, 1.05, 0.98, 1.01, 1.04]
best = vals.index(min(vals)) + 1
eq(best, 3)
eq(len(vals) - best, 2)
# Effective batch
eq(1 * 8 * 1, 8)
# Embedding table
eq(151_936 * 5120, 777_912_320)
close(151_936 * 5120 / 1e6, 777.9, 0.05)
close(151_936 * 5120 * 2 / 1e9, 1.56, 5e-3)
# Embedding model memory
close(10**6 * 768 * 4 / 1e9, 3.072, 1e-12)
# Entropy
close(-(0.5 * log2(0.5) * 2), 1.0, 1e-12)
close(log(2), 0.693, 5e-4)
close(-(0.9 * log2(0.9) + 0.1 * log2(0.1)), 0.469, 5e-4)
# Epoch
eq(4000 // 8, 500)
# Equal opportunity difference
close(48 / 60, 0.8, 1e-12)
close(24 / 40, 0.6, 1e-12)
close(48 / 60 - 24 / 40, 0.2, 1e-12)
# Error analysis: 40 failures into 5 buckets (quoted)
# Error budget
close((1 - 0.995) * 2_000_000, 10_000, 1e-6)
# Eval: threshold over 300 tickets (quoted)
# Eval diff: 12 fixed, 7 broken (quoted)
# Excessive agency
close(0.98**2, 0.9604, 1e-12)
close(0.98**10, 0.8171, 5e-5)
close(1 - 0.98**10, 0.18, 5e-3)
# Execution-based evaluation: 2 of 6
close(2 / 6, 1 / 3, 1e-12)
# Expectation
close(0.7 * 100 + 0.3 * 1000, 370, 1e-9)
eq(0.7 * 100, 70)
close(0.3 * 1000, 300, 1e-9)
# Exponential backoff
eq([min(8.0, 0.5 * 2**n) for n in range(5)], [0.5, 1, 2, 4, 8])

# ------------------------------------------------------------------------ F
# F1
P, R = 2 / 3, 0.8
close(2 * P * R / (P + R), 0.727, 5e-4)
close(2 * 80 / (2 * 80 + 40 + 20), 2 * P * R / (P + R), 1e-12)
eq(2 * 80 + 40 + 20, 220)
# FAISS: 768 dims (no arithmetic)
# Fallback: 1 s timeout (quoted from 55 §15)
# Feature: X shape
eq((1000, 3), (1000, 3))
# Feature engineering: cyclic hour
close(sin(2 * pi * 23 / 24), -0.259, 5e-4)
close(cos(2 * pi * 23 / 24), 0.966, 5e-4)
close(sin(2 * pi * 1 / 24), 0.259, 5e-4)
close(cos(2 * pi * 1 / 24), 0.966, 5e-4)
eq(23 - 1, 22)
# Few-shot: 1 system + 3 pairs + 1 user = 8 messages; zero-shot 2
eq(1 + 3 * 2 + 1, 8)
eq(1 + 1, 2)
# Fine-tuning: 600-token prompt, 5k tickets (quoted from 53 §1.3)
# Finite-difference check
close(2.01**3, 8.120601, 1e-9)
close(1.99**3, 7.880599, 1e-9)
close((2.01**3 - 1.99**3) / 0.02, 12.0001, 1e-6)
eq(3 * 2**2, 12)
# FLOPs
close(6 * 14.77e9 * 36e12, 3.2e24, 0.05e24)
# fp16
eq(2**15 * (1 + 1023 / 1024), 65_504)
eq(2 ** (30 - 15), 32_768)
close(2**-10, 0.000977, 5e-7)
# Frequency and presence penalties
close(3.0 - 0.5 * 2 - 0.5 * 1, 1.5, 1e-12)
close(exp(1.5), 4.5, 0.05)
# Full fine-tuning
close(16 * 14.77e9 / 1e9, 236, 0.5)

# ------------------------------------------------------------------------ G
# GELU
close(1 * phi(1), 0.841, 5e-4)
close(-1 * phi(-1), -0.159, 5e-4)
# GGUF: Q4_K_M ~9 GB (quoted from 10 §2)
# Gini
close(1 - 0.6**2 - 0.4**2, 0.48, 1e-12)
close(1 - 0.5**2 - 0.5**2, 0.5, 1e-12)
# Golden set: 500 = 200 + 300
eq(200 + 300, 500)
# GQA
eq(2 * 40 * 8 * 128 * 2, 163_840)
eq(163_840 // 1024, 160)
eq(2 * 40 * 40 * 128 * 2 // 1024, 800)
eq(800 // 160, 5)
eq(40 // 8, 5)
# Gradient: L = w1^2 + 3 w2 at (1, 2)
eq((2 * 1, 3), (2, 3))
# Gradient accumulation
eq(4 * 8, 32)
# Gradient boosting
eq(10 - 7, 3)
close(7 + 0.1 * 3, 7.3, 1e-12)
# Gradient clipping
close(sqrt(3**2 + 4**2), 5, 0)
eq((3 / 5, 4 / 5), (0.6, 0.8))
close(sqrt(0.3**2 + 0.4**2), 0.5, 1e-12)
# Gradient descent: L = (w - 3)^2
w0 = 0.0
g0 = 2 * (w0 - 3)
eq(g0, -6)
w1 = w0 - 0.1 * g0
close(w1, 0.6, 1e-12)
g1 = 2 * (w1 - 3)
close(g1, -4.8, 1e-12)
close(w1 - 0.1 * g1, 1.08, 1e-12)
# GraphRAG: one call per chunk
eq(50_000 * 1, 50_000)
# Groundedness
close(4 / 5, 0.8, 0)
# GRPO advantages
r = [1, 0, 1, 0]
mu = sum(r) / 4
sd_r = sqrt(sum((x - mu) ** 2 for x in r) / 4)
eq((mu, sd_r), (0.5, 0.5))
eq([(x - mu) / sd_r for x in r], [1, -1, 1, -1])

# ------------------------------------------------------------------------ H
# Haystack: 29 lines
eq(SLOC["Haystack"], 29)
# He initialization
eq(sqrt(2 / 512), 1 / 16)
close(1 / 16, 0.0625, 0)
# Held-out set
eq(500 - 300, 200)
# HNSW: 2M links per vector
eq(2 * 16, 32)
eq(32 * 10**6, 32_000_000)
# Host/client/server: one client per server
eq(3 * 1, 3)
# Hyperparameter grid
eq(3 * 4, 12)
eq(12 * 5, 60)

# ------------------------------------------------------------------------ I
# Inference vs training memory
close(2 * 14.77, 29.5, 0.05)
close(16 * 14.77, 236, 0.5)
eq(16 // 2, 8)
# Interleaving: 60 of 100 clicks (illustrative)
close(60 / 100, 0.6, 0)
# IVF
eq(10**6 // 1000, 1000)
eq(10 * 1000, 10_000)

# ------------------------------------------------------------------------ J
# Jaccard
sa = {"reset your", "your password", "password now"}
sb = {"reset your", "your password", "password today"}
eq(len(sa & sb), 2)
eq(len(sa | sb), 4)
close(len(sa & sb) / len(sa | sb), 0.5, 0)

# ------------------------------------------------------------------------ K
# k-means on 1, 2, 9, 10 with centroids 1 and 2
pts = [1, 2, 9, 10]


def assign(cents):
    groups = [[] for _ in cents]
    for x in pts:
        groups[min(range(len(cents)), key=lambda i: abs(x - cents[i]))].append(x)
    return groups


g = assign([1, 2])
eq(g, [[1], [2, 9, 10]])
cents = [sum(v) / len(v) for v in g]
eq(cents, [1, 7])
g = assign(cents)
eq(g, [[1, 2], [9, 10]])
cents = [sum(v) / len(v) for v in g]
eq(cents, [1.5, 9.5])
eq(assign(cents), g)
# k-NN: 2 of 3 votes
eq(["urgent", "urgent", "normal"].count("urgent"), 2)
# KL divergence
close(0.5 * log(0.5 / 0.75) + 0.5 * log(0.5 / 0.25), 0.144, 5e-4)
# KV cache at 32K
eq(163_840 * 32_768, 5_368_709_120)
close(163_840 * 32_768 / 1e9, 5.37, 5e-3)

# ------------------------------------------------------------------------ L
# Label: 460 of 500
close(460 / 500, 0.92, 1e-12)
eq(500 - 460, 40)
# Label noise
close(16 / 200, 0.08, 1e-12)
close(1 - 16 / 200, 0.92, 1e-12)
# LangGraph: 22 lines
eq(SLOC["LangGraph"], 22)
# Laplace smoothing
eq((2 + 1, 5 + 7), (3, 12))
eq((0 + 1, 5 + 7), (1, 12))
# LayerNorm
xs = [1, 2, 3]
mu = sum(xs) / 3
sd_x = sqrt(sum((x - mu) ** 2 for x in xs) / 3)
close(sd_x, sqrt(2 / 3), 1e-12)
close(sd_x, 0.816, 5e-4)
close((1 - mu) / sd_x, -1.22, 5e-3)
close((3 - mu) / sd_x, 1.22, 5e-3)
# Learning rate on L = w^2
close(1 - 2 * 0.1, 0.8, 1e-12)
close(1 - 2 * 1.1, -1.2, 1e-12)
# Learning-rate schedule (48 §9.2)
eta_max, eta_min, Tw, T = 3e-4, 3e-5, 100, 1000


def lr(t):
    if t < Tw:
        return eta_max * t / Tw
    s_ = (t - Tw) / (T - Tw)
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * s_))


close(lr(50), 1.5e-4, 1e-12)
close((550 - Tw) / (T - Tw), 0.5, 1e-12)
close(lr(550), 1.65e-4, 1e-12)
close(lr(1000), 3e-5, 1e-12)
# Likelihood: argmax of p^7 (1-p)^3
grid = [i / 1000 for i in range(1, 1000)]
close(max(grid, key=lambda p: p**7 * (1 - p) ** 3), 0.7, 1e-9)
# Linear attention: 3 GDN layers per full-attention layer (37 §3.3) - quoted
# Linear layer
eq(5120 * 5120, 26_214_400)
# Linear regression: y = 2x exactly
eq(sum((2 * xv - yv) ** 2 for xv, yv in [(1, 2), (2, 4), (3, 6)]), 0)
# LlamaIndex: 24 lines
eq(SLOC["LlamaIndex"], 24)
# LLM: Qwen3-14B facts (quoted)
# Log-loss
close(-log(0.8), 0.223, 5e-4)
close(-log(0.1), 2.303, 5e-4)
close(-log(0.1) / -log(0.8), 10, 0.5)
# Log-sum-exp
z2 = [1000.0, 1000.0]
m = max(z2)
close(m + log(sum(exp(v - m) for v in z2)), 1000.693, 5e-4)
# Logistic regression / Logit
close(sigmoid(2), 0.881, 5e-4)
close(sigmoid(0), 0.5, 0)
close(sigmoid(2), 0.88, 5e-3)
# Logit distillation: top-20 of 262,144
close(262_144 / 20, 13_000, 150)
# Logprob
close(exp(-0.105), 0.90, 5e-3)
# LoRA
eq(16 * (5120 + 5120), 163_840)
close(163_840 / 26_214_400, 0.00625, 1e-12)
close(5120 * 5120 / 1e6, 26.2, 0.05)
per_block = 16 * ((5120 + 5120) + (5120 + 1024) * 2 + (5120 + 5120)
                  + (5120 + 17408) * 3)
eq(per_block, 1_605_632)
eq(40 * per_block, 64_225_280)
# Loss curve
close(log(151_936), 11.93, 5e-3)
# Loss function: MSE of (3, 5) vs (2, 7)
close(((3 - 2) ** 2 + (5 - 7) ** 2) / 2, 2.5, 0)
# Loss masking
nll = [2.0, 1.5, 3.0, 0.1, 0.22, 0.51]
close((0.22 + 0.51) / 2, 0.365, 1e-12)
close(sum(nll) / 6, 1.222, 5e-4)
# Lost in the middle: 7th of 13 (illustrative)
# LSH
close(1 - (1 - 0.5**2) ** 4, 0.684, 5e-4)
close(1 - (1 - 0.8**2) ** 4, 0.983, 5e-4)
close(0.5**2, 0.25, 0)
# Luhn
digits = [int(ch) for ch in "4242424242424242"]


def luhn_sum(ds):
    total = 0
    for i, d in enumerate(reversed(ds)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total


eq(luhn_sum(digits), 80)
eq(8 * 8, 64)
eq(8 * 2, 16)
eq(80 % 10, 0)

# ------------------------------------------------------------------------ M
# Macro / weighted
close((0.9 + 0.3) / 2, 0.6, 1e-12)
close((0.9 * 900 + 0.3 * 100) / 1000, 0.84, 1e-12)
close(0.9 * 900, 810, 1e-9)
close(0.3 * 100, 30, 1e-9)
# MAE / MSE / RMSE / R2
yv = [2, 4, 6, 8]
yh = [3, 4, 4, 9]
err = [p - t for t, p in zip(yv, yh)]
close(sum(abs(x) for x in err) / 4, 1.0, 0)
close(sum(x * x for x in err) / 4, 1.5, 0)
close(sqrt(1.5), 1.225, 5e-4)
ybar = sum(yv) / 4
sst = sum((t - ybar) ** 2 for t in yv)
eq(sum(x * x for x in err), 6)
eq(sst, 20)
close(1 - 6 / 20, 0.7, 1e-12)
# Matrix multiplication shapes: (8 x 5120)(5120 x 1024) -> (8 x 1024)
eq((8, 1024), (8, 1024))
# Matryoshka
close(10**6 * 1024 * 4 / 1e9, 4.096, 1e-12)
close(10**6 * 256 * 4 / 1e9, 1.024, 1e-12)
eq(1024 // 256, 4)
# max_tokens: 500 (illustrative)
# McNemar
close((15 - 5) ** 2 / (15 + 5), 5.0, 0)
eq(15 + 5, 20)
# MCP: 5 x 8 vs 5 + 8
eq(5 * 8, 40)
eq(5 + 8, 13)
# Memory (agent)
eq(1000 + 12 * 600, 8200)
eq(4 * 600, 2400)
eq(1000 + 300 + 2400, 3700)
eq(12 - 8, 4)
# Min-p
close(0.1 * 0.9, 0.09, 1e-12)
eq(sum(p >= 0.09 for p in (0.9, 0.05, 0.03, 0.02)), 1)
# MinHash standard error
close(sqrt(0.5 * 0.5 / 200), 0.035, 5e-4)
close(abs(0.525 - 0.5), 0.025, 1e-12)
# Mixed precision
eq(2 + 2 + 4 + 4 + 4, 16)
# MLP share of a Qwen3-14B block
mlp = 3 * 5120 * 17408
attn = 5120 * 5120 * 2 + 5120 * 1024 * 2
close(mlp / 1e6, 267.4, 0.05)
close(mlp / (mlp + attn), 0.81, 5e-3)
# MoE decode ceiling
close(3.3 * 0.5625, 1.86, 5e-3)
close(200 / 1.86, 108, 0.5)
close(200 / 8.3, 24, 0.5)
# Momentum
close(1 / (1 - 0.9), 10, 1e-9)
# MRR
close((1 + 1 / 2 + 1 / 4) / 3, 0.583, 5e-4)
# Multi-head attention
eq(5120 // 40, 128)
# MTP
close(log(2) + 2 * log(2) + 3 * log(2), 6 * log(2), 1e-12)
close(6 * log(2), 4.159, 5e-4)
close(-(log(0.5) + log(0.25) + log(0.125)), 6 * log(2), 1e-12)
# Multiple comparisons
close(20 * 0.05, 1, 1e-12)
close(0.05 / 20, 0.0025, 1e-12)

# ------------------------------------------------------------------------ N
# Naive Bayes
spam = 0.5 * 3 / 12 * 2 / 12
ham = 0.5 * 1 / 12 * 2 / 12
close(spam / (spam + ham), 0.75, 1e-12)
# Nats and bits
close(1 / log(2), 1.443, 5e-4)
close(0.693 / log(2), 1.0, 5e-4)
# nDCG
dcg = 1 / log2(2) + 1 / log2(4) + 1 / log2(6)
idcg = 1 / log2(2) + 1 / log2(3) + 1 / log2(4) + 1 / log2(5)
close(1 / log2(6), 0.387, 5e-4)
close(dcg, 1.887, 5e-4)
close(idcg, 2.562, 5e-4)
close(dcg / idcg, 0.737, 5e-4)
# NLL
close(-log(0.25), log(4), 1e-12)
close(log(4), 1.386, 5e-4)
# Neuron
eq(1.5 * 2 + 1 * 1 - 2, 2)
close(sigmoid(2), 0.881, 5e-4)
# Nondeterminism: 5 samples (illustrative)
# Normal distribution: 0.80 +- 1.96 x 0.02
close(0.80 - 1.96 * 0.02, 0.761, 5e-4)
close(0.80 + 1.96 * 0.02, 0.839, 5e-4)
close(2 * phi(1.96) - 1, 0.95, 5e-4)

# ------------------------------------------------------------------------ O
# OpenAI Agents SDK: 25 lines
eq(SLOC["OpenAI Agents"], 25)
# Optimizer: SGD step
close(0.1 * 0.5, 0.05, 1e-12)
# Overflow and underflow
eq(0.01**1000, 0.0)
close(1000 * log(0.01), -4605.2, 0.05)
# Override rate
close(30 / 1000, 0.03, 1e-12)

# ------------------------------------------------------------------------ P
# p-value
close(2 * (1 - phi(1.96)), 0.05, 5e-4)
close(2 * (1 - phi(1.54)), 0.12, 5e-3)
# Packing
eq(4 * 500, 2000)
eq(2048 - 2000, 48)
eq(2048 - 500, 1548)
eq(4 * 1548, 6192)
# Paged attention
eq(4096 - 500, 3596)
close(3596 / 4096, 0.88, 5e-3)
# Paired comparison (54 §8.4)
pbar = (75 + 65) / 200
se_u = sqrt(pbar * (1 - pbar) * (2 / 100))
close(pbar, 0.70, 1e-12)
close(se_u, 0.0648, 5e-5)
close(0.10 / se_u, 1.54, 5e-3)
# Parameter
close(14.77 * 2, 29.5, 0.05)
close(14.77 * 0.5625, 8.3, 0.05)
eq((64 * 4 + 32) / 64, 4.5)
eq(4.5 / 8, 0.5625)
# pass@k
n_, c_ = 6, 2
close(1 - comb(n_ - c_, 1) / comb(n_, 1), 1 / 3, 1e-12)
eq(comb(4, 3), 4)
eq(comb(6, 3), 20)
close(1 - comb(4, 3) / comb(6, 3), 0.8, 1e-12)
close((1 / 3) ** 3, 0.037, 5e-4)
# PEFT
close(64_225_280 / 1e6, 64.2, 0.05)
close(64_225_280 / 14.77e9 * 100, 0.43, 5e-3)
# Percentile (55 §16.3)
lat = [120, 135, 140, 150, 155, 160, 165, 170, 175, 180,
       185, 190, 200, 210, 220, 240, 260, 300, 420, 850]
eq(len(lat), 20)
s_lat = sorted(lat)
eq(ceil(0.95 * 20), 19)
eq(s_lat[ceil(0.95 * 20) - 1], 420)
eq(s_lat[ceil(0.99 * 20) - 1], 850)
eq(s_lat[ceil(0.50 * 20) - 1], 180)
close(sum(lat) / 20, 231, 0.5)
# Perplexity
close(exp(2 * log(2)), 4, 1e-12)
# PII: no arithmetic
# Portfolio project: thresholds quoted from 57 Project 4
# PQ
close(10**6 * 768 * 4 / 1e9, 3.072, 1e-12)
eq(10**6 * 96, 96_000_000)
eq(768 // 96, 8)
eq(3_072_000_000 // 96_000_000, 32)
# Precision
close(80 / 120, 0.667, 5e-4)
# Precision@k
retrieved = ["d2", "d1", "d5", "d3", "d8"]
relevant = {"d2", "d5", "d8", "d9"}
close(sum(d in relevant for d in retrieved) / 5, 0.6, 1e-12)
# Prefill
close(2 * 14.77e9 * 8000, 2.36e14, 0.005e14)
close(2 * 14.77e9 * 8000 / 3.5e12, 68, 0.6)
# Prompt caching
eq(2000 - 1600, 400)
close(1 - after / before, 0.41, 5e-3)
# Pydantic AI: 25 lines
eq(SLOC["Pydantic AI"], 25)

# ------------------------------------------------------------------------ Q
# QLoRA
close(29.5 / 8.3, 3.5, 0.06)  # "about 3.5x", as in 53 §5
close(16 / 4.5, 3.5, 0.06)
# Quality filtering (60 §7.5)
close(18 / 25, 0.72, 1e-12)
close(18 / 20, 0.90, 1e-12)
eq(18 + 7, 25)
eq(18 + 2, 20)
# Quantization: absmax int8
wq = [0.523, -1.27, 0.336, 0.0414]
s_q = max(abs(v) for v in wq) / 127
close(s_q, 0.01, 1e-12)
eq([round(v / s_q) for v in wq], [52, -127, 34, 4])
for v in wq:
    assert abs(v - s_q * round(v / s_q)) <= s_q / 2 + 1e-12
    checks += 1
close(s_q / 2, 0.005, 1e-12)
# Query, key, value: same numbers as the attention entry
close(softmax([1, 0])[0], 0.731, 5e-4)

# ------------------------------------------------------------------------ R
# R2: same data as MAE entry
close(1 - 6 / 20, 0.7, 1e-12)
# Random forest votes
close(2 / 3, 0.667, 5e-4)
# Rank and alpha: scale = alpha / r; MLX default 20 is 10x
eq(32 / 16, 2.0)
eq(20 / 2.0, 10)
# Rate limit: token bucket, capacity 3, refill 2/s, gap 0.25 s
tokens, tprev, allowed = 3.0, 0.0, []
for i in range(6):
    now = i * 0.25
    tokens = min(3.0, tokens + (now - tprev) * 2.0)
    tprev = now
    ok = tokens >= 1
    if ok:
        tokens -= 1
    allowed.append(ok)
eq(allowed, [True] * 5 + [False])
# Reasoning model latency
close(2 + 200 / 24, 10.3, 0.05)
close(2 + 2200 / 24, 93.7, 0.05)
eq(2000 + 200, 2200)
# Recall
close(80 / 100, 0.8, 0)
# Recall@k
close(sum(d in relevant for d in retrieved) / len(relevant), 0.75, 1e-12)
# RRF
close(1 / 62 + 1 / 61, 0.03252, 5e-6)
close(1 / 61 + 1 / 63, 0.03227, 5e-6)
# Red-teaming: 0 of 4 still allows 0.49
close(wilson(0, 4)[1], 0.49, 5e-3)
# Regularization
close(0.1 * (3**2 + 4**2), 2.5, 1e-12)
# Rejection sampling: 3 of 8 kept (illustrative)
# Reliability compounding
close(0.95**10, 0.599, 5e-4)
close(1 - 0.2**2, 0.96, 1e-12)
close(0.8**10, 0.107, 5e-4)
close(0.96**10, 0.665, 5e-4)
# ReLU
eq((max(0, -2), max(0, 3)), (0, 3))
# Repetition penalty
close(2.6 / 1.3, 2.0, 1e-12)
close(-1.0 * 1.3, -1.3, 1e-12)
# Reranker: 100 -> 20 -> 5 (illustrative)
# Residual connection: d/dx (x + f(x)) = 1 + f'(x)
close(1 + 0.0, 1, 0)
# Reward model
close(sigmoid(1.0), 0.731, 5e-4)
# RMSNorm
rms = sqrt((9 + 16) / 2)
close(rms, 3.536, 5e-4)
close(3 / rms, 0.849, 5e-4)
close(4 / rms, 1.131, 5e-4)
# RNN
close(0.5**20, 9.5e-7, 0.05e-7)
# ROC-AUC
pos, neg = [0.9, 0.8, 0.6], [0.7, 0.4, 0.2]
pairs = sum(1 for p in pos for q in neg if p > q)
eq(pairs, 8)
eq(len(pos) * len(neg), 9)
close(8 / 9, 0.889, 5e-4)
# RoPE: same offset at 5/3 and 105/103
eq(5 - 3, 105 - 103)
# ROUGE
ref_r = "the refund was issued today".split()
cand_r = "refund issued today".split()
ov = sum(t in ref_r for t in cand_r)
eq(ov, 3)
close(ov / len(ref_r), 0.6, 1e-12)
close(ov / len(cand_r), 1.0, 1e-12)
close(2 * 0.6 * 1.0 / (0.6 + 1.0), 0.75, 1e-12)
# Rule of three
close(3 / 20, 0.15, 1e-12)
close(wilson(0, 20)[1], 0.161, 5e-4)
close(3 / 300, 0.01, 1e-12)

# ------------------------------------------------------------------------ S
# Sample size (54 §8.2)
close((1.96 + 0.84) ** 2, 7.84, 1e-9)
close(0.5 * 0.5 + 0.55 * 0.45, 0.4975, 1e-12)
close(7.84 * 0.4975 / 0.05**2, 1560.2, 0.05)
eq(ceil(7.84 * 0.4975 / 0.05**2), 1561)
# Sampling: " processed" half of the time at T = 1
close(softmax([log(p) for p in (0.5, 0.2, 0.15, 0.1, 0.05)])[0], 0.5, 1e-12)

# Scaling laws: Chinchilla ~20 tokens/parameter; Qwen3-14B 36T / 14.77B
eq(20 * 1_000_000_000, 20_000_000_000)
close(36e12 / 14.77e9, 2437, 1)
# Self-consistency: majority of 3 at p = 0.7
close(0.7**3, 0.343, 1e-12)
close(3 * 0.7**2 * 0.3, 0.441, 1e-12)
close(0.7**3 + 3 * 0.7**2 * 0.3, 0.784, 1e-12)
# Self-supervised: 4 words -> 3 next-token pairs
eq(len("the refund was issued".split()) - 1, 3)
# Semantic cache: 40% hits at $0.0007 per call, 1M requests a day
close(0.4 * 1e6 * 0.0007, 280, 1e-9)
# Sequence KD yield
eq(2 * 5000, 10_000)
close(8000 / 10_000, 0.8, 1e-12)
# SFT: 2,000 x 1,000 x 2 tokens at 25 tok/s
eq(2000 * 1000 * 2, 4_000_000)
close(4e6 / 25, 160_000, 1e-9)
close(160_000 / 3600, 44.4, 0.05)
# SGD steps per epoch
eq(100_000 // 50, 2000)
# Shingles: 4 words -> 3 word 2-shingles
w = "reset my password now".split()
eq([" ".join(w[i:i + 2]) for i in range(len(w) - 1)], ["reset my", "my password", "password now"])
# Sigmoid and SiLU
close(sigmoid(0), 0.5, 1e-12)
close(sigmoid(2), 0.881, 5e-4)
close(2 * sigmoid(2), 1.762, 5e-4)
close(-2 * sigmoid(-2), -0.238, 5e-4)
close(sigmoid(-2), 0.119, 5e-4)
# SLO error budget
close(0.005 * 1_000_000, 5000, 1e-9)
# Softmax of (2, 1, 0)
close(exp(2), 7.389, 5e-4)
close(exp(1), 2.718, 5e-4)
for got, shown in zip(softmax([2, 1, 0]), (0.665, 0.245, 0.090)):
    close(got, shown, 5e-4)
# Speculative decoding: alpha 0.8, k 4
close((1 - 0.8**5) / (1 - 0.8), 3.36, 5e-3)
# Standard error of 80% on 100 and 400 items
close(sqrt(0.8 * 0.2 / 100), 0.04, 1e-12)
close(sqrt(0.8 * 0.2 / 400), 0.02, 1e-12)
# Stratified split of 50 urgent tickets, 80/20
eq(round(50 * 0.8), 40)
eq(round(50 * 0.2), 10)
# Streaming: 200 tokens at 16 tok/s
close(200 / 16, 12.5, 1e-12)
# SwiGLU weights per Qwen3-14B MLP
eq(3 * 5120 * 17408, 267_386_880)
# Sycophancy: 7 x 8
eq(7 * 8, 56)
# Synthetic data yield
close(10_000 * 0.6 * 0.8, 4800, 1e-9)

# ------------------------------------------------------------------------ T
# Temperature on logits (2, 1, 0)
for got, shown in zip(softmax([4, 2, 0]), (0.867, 0.117, 0.016)):
    close(got, shown, 5e-4)
for got, shown in zip(softmax([1, 0.5, 0]), (0.506, 0.307, 0.186)):
    close(got, shown, 5e-4)
# Tensor [8, 2048, 5120]
eq(8 * 2048 * 5120, 83_886_080)
# Test-time compute: 2,000 thinking tokens at 16 tok/s
close(2000 / 16, 125, 1e-12)
# Throughput: single stream and 8 batched 4K streams on an M1 Pro
close(200 / 8.3, 24, 0.15)
close(8 * 200 / 13.7, 117, 0.5)
close(8 * 200 / 13.7 / 8, 15, 0.5)
# tiktoken: "hello world" is 2 tokens in cl100k_base
eq(len([15339, 1917]), 2)
# Token: 1,000 tokens at $3 per million
close(1000 * 3 / 1e6, 0.003, 1e-12)
# Token bucket: burst of 10, capacity 5
eq(min(10, 5), 5)
eq(10 - min(10, 5), 5)
# Tokenizer: Gemma 4 vocabulary
eq(2**18, 262_144)
# Top-k (k = 2) and top-p (p = 0.8) on (0.5, 0.2, 0.15, 0.1, 0.05)
close(0.5 / 0.7, 0.714, 5e-4)
close(0.2 / 0.7, 0.286, 5e-4)
close(0.5 + 0.2 + 0.15, 0.85, 1e-12)
assert 0.5 + 0.2 < 0.8 <= 0.5 + 0.2 + 0.15
checks += 1
for got, shown in zip((0.5 / 0.85, 0.2 / 0.85, 0.15 / 0.85), (0.588, 0.235, 0.176)):
    close(got, shown, 5e-4)
# TPOT at 16 tok/s
close(1000 / 16, 62.5, 1e-12)
# Trace: 0.1 + 0.4 + 2.7 = 3.2 s, model share 84%
close(0.1 + 0.4 + 2.7, 3.2, 1e-12)
close(2.7 / 3.2, 0.84, 5e-3)
# Train / validation / test 80/10/10 of 10,000
eq((10_000 * 8 // 10, 10_000 // 10, 10_000 // 10), (8000, 1000, 1000))
# TPR = 45/60
close(45 / 60, 0.75, 1e-12)
# TTFT: 8,000 tokens at ~120 tok/s
close(8000 / 120, 67, 0.5)
# Two-proportion z-test: 300/400 vs 332/400
se = sqrt(0.79 * 0.21 * 2 / 400)
close(se, 0.0288, 5e-5)
close(0.08 / se, 2.78, 5e-3)

# ------------------------------------------------------------------------ U
# Unbounded consumption: 20,000 tokens x 500 at $3 per million
eq(20_000 * 500, 10_000_000)
close(10_000_000 * 3 / 1e6, 30, 1e-9)
# Unicode: NFKC folds x-squared to x2 and leaves curly quotes alone
eq(unicodedata.normalize("NFKC", "x\u00b2"), "x2")
eq(unicodedata.normalize("NFKC", "\u2019\u201c"), "\u2019\u201c")
# Utilization: $2/hour at 2,000 tok/s
eq(2000 * 3600, 7_200_000)
close(2 / 7.2, 0.28, 5e-3)
close(2 / (7.2 * 0.1), 2.78, 5e-3)

# ------------------------------------------------------------------------ V
# Vanishing and exploding gradients over 20 layers
close(0.5**20, 9.5e-7, 5e-9)
close(1.5**20, 3325, 1)
# Variance of 2, 4, 4, 4, 5, 5, 7, 9
lat = [2, 4, 4, 4, 5, 5, 7, 9]
mu = sum(lat) / len(lat)
eq(mu, 5)
eq(sum((x - mu) ** 2 for x in lat), 32)
eq(sum((x - mu) ** 2 for x in lat) / len(lat), 4)
eq(sqrt(4), 2)
# Vector (1, 2, 2) has length 3
eq(sqrt(1 + 4 + 4), 3)
# Vector database: 1M x 768 x 4 bytes
close(1e6 * 768 * 4 / 1e9, 3.072, 1e-9)
# Voting: three 90% judges
close(0.1**3, 0.001, 1e-12)
close(3 * 0.1**2 * 0.9 + 0.1**3, 0.028, 1e-12)

# ------------------------------------------------------------------------ W
# W4A16: 29.5 GB -> 8.3 GB, decode bound on 200 GB/s
close(29.5 / 8.3, 3.5, 0.1)
close(200 / 29.5, 6.8, 0.05)
close(200 / 8.3, 24, 0.15)
# Warmup: step 50 of 100 to a 2e-4 peak
close(2e-4 * 50 / 100, 1e-4, 1e-18)
# Weight decay factor with lr 1e-3 and decay 0.01
close(1 - 1e-3 * 0.01, 0.99999, 1e-15)
# Wilson interval for 40/100
lo, hi = wilson(40, 100)
close(lo, 0.309, 5e-4)
close(hi, 0.498, 5e-4)

# ------------------------------------------------------------------- X, Y, Z
close(sqrt(2 / 1024), 0.0442, 5e-5)   # Xavier, 512 -> 512
close(sqrt(2 / 512), 0.0625, 1e-12)   # He, fan_in 512
eq(131_072 // 32_768, 4)              # YaRN extension
eq((9 - 5) / 2, 2)                    # z-score

# --------------------------------------------- helpers used by other entries
# Unicode normalization and HTML escaping behave as the handbook describes
eq(unicodedata.normalize("NFKC", "１２３"), "123")
eq(unicodedata.normalize("NFKC", "ﬁ"), "fi")
eq(html.escape("<b>"), "&lt;b&gt;")
close(e, 2.718, 5e-4)

# ------------------------------------------------------------- page structure
DOC = Path(__file__).resolve().parents[2] / "docs" / "59-ai-glossary.md"
text = DOC.read_text(encoding="utf-8")
eq(text.splitlines()[0], "# 59 — AI engineering glossary")
assert "\\[" not in text, "display math is not allowed in the glossary"
checks += 1


def sort_key(term):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]", " ", term.lower())).strip()


sections = re.split(r"^## ", text, flags=re.M)[1:]
letters, total, ids = [], 0, set()
for sec in sections:
    head, body = sec.split("\n", 1)
    if head == "See also":
        assert "45-ai-engineer-path.md" in body and "57-ai-career.md" in body
        checks += 1
        continue
    assert len(head) == 1 and head.isupper(), head
    letters.append(head)
    entries = re.findall(r"^\*\*(.+?)\*\* \{: #([a-z0-9-]+) \}\n:   (.+)$", body, flags=re.M)
    eq(len(entries), body.count("\n:   "))
    keys = [sort_key(t) for t, _, _ in entries]
    eq(keys, sorted(keys))
    for term, anchor, definition in entries:
        assert sort_key(term)[0] == head.lower(), (head, term)
        assert anchor not in ids, f"duplicate id {anchor}"
        ids.add(anchor)
        assert definition.count("*Example:*") == 1, term
        assert "→ [" in definition, term
        checks += 4
    total += len(entries)
eq(letters, sorted(letters))
bar = " · ".join(f"[{L}](#{L.lower()})" for L in letters)
assert bar in text, "jump bar does not match the letter sections"
checks += 1
assert total >= 250, total
checks += 1

print(f"ch59_glossary: all {checks} checks passed")
