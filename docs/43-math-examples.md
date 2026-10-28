# 43 — Worked examples: every formula on a small model

*Last verified: 27 October 2026. Every number below was computed directly from the formula with a small model; the arithmetic is reproducible with a calculator or a short Python script.*

This chapter is the numerical companion to [32 — Mathematics](32-mathematics.md) and [38 — Derivations and proofs](38-derivations.md). Each section follows the same shape: it **states the formula**, says in one line **what it means**, grounds it in a **concrete situation** (a next token, a weight row, a batch of eval tasks), runs the arithmetic on a **tiny example you can check by hand**, and ends with **why it matters** on a real model.

The numbers are deliberately small. That is not unrealistic — it is the only way to verify a formula with a calculator instead of a GPU. A toy model that gets the softmax right gets it right for the same reason a 14B model does; the shapes and the signs are what you are checking, and those do not change with scale. Each box also carries a *reality bridge*: the same formula applied to Qwen3-14B or Gemma 4 12B, so you can see the toy number grow into the production one.

Run the arithmetic yourself:

```bash
python scripts/check_math_examples.py
```

It asserts 69 values from this chapter, including the Wilson interval, the hybrid and shared KV caches, and Gemma 4 12B, E4B and E2B parameter counts reproduced to the parameter from their published checkpoints. The shorter “On paper” cards after every formula in chapters 32–42 are checked by `python scripts/check_paper_examples.py`.

---

## 0. The toy model and seven things people get wrong

Most examples use one consistent toy model:

| Symbol | Value | Meaning |
|---|---:|---|
| \(d\) | 4 | hidden size |
| \(L\) | 2 | layers |
| \(H\) | 2 | query heads |
| \(d_h\) | 2 | head dimension |
| \(G\) | 1 | KV heads |
| \(d_{\text{ff}}\) | 8 | FFN width |
| \(V\) | 5 | vocabulary |
| \(T\) | 3 | sequence length |

Some examples use \(V=4\) or \(T=2\) when a smaller matrix is clearer; each box states its own numbers.

These are the seven places this chapter's arithmetic contradicts a plausible-looking shortcut. Each one is a mistake that looks right until you run the numbers; the section in parentheses has the full worked check:

1. **Gemma 4 12B parameter count.** It is tempting to count all 48 layers with the local shape `G=8, d_h=256`. But 8 of them are global layers with `num_global_key_value_heads = 1`, `global_head_dim = 512` and K=V — so they have **no `v_proj`**. Counting them correctly gives **11.91B** of text decoder (11.96B with the multimodal embedders); the shortcut gives 11.77B and silently drops ~0.14B (§14).
2. **Gemma 4 12B KV cache.** Applying the full-attention formula to all 48 layers gives 12.9 GB. That is an upper bound, not the deployed cache: the real hybrid cache is about **604 MB** at 32K and **2.48 GB** at 256K (§7).
3. **SwiGLU backward.** The gate's gradient is not just the scalar derivative \(g'(a)\). The correct form multiplies the upstream gradient \(W_d^\top\bar y\), the up-projection \(b\), and \(g'(a)\) elementwise (§9).
4. **Q4_0 scale.** The clean symmetric grid uses \(\max/(2^{k-1}-1)\), but real GGML Q4_0 uses a code range \(-8\ldots7\) and a scale of \(\max/8\). Close, but not identical (§16).
5. **Attention FLOPs.** The attention term scales with the query width \(H d_h\), not the hidden size \(d\). They are equal for Qwen3-14B but not for Gemma's global layers, where \(H d_h = 8192\) and \(d = 3840\) (§15).
6. **H100 peak.** NVIDIA lists the H100 SXM at 1,979 TFLOP/s BF16 *with sparsity*; the dense BF16 peak is half of that, about **989 TFLOP/s**. The 494.5 TFLOP/s figure that also circulates is the dense **TF32** peak. At 40% of the dense BF16 peak, the 36T-token Qwen3-14B run is ~2.24M GPU-hours, ~9.3 days and ~$5.6M at $2.50/GPU-hour. Planning with 494.5 doubles all three; planning with 1,979 as if it were dense halves them (§15).
7. **Smaller model, smaller cache.** Gemma 4 E4B computes with a third of the 12B's parameters (3.97B vs 11.91B), yet at 128K its KV cache is **2.17 GB against the 12B's 1.41 GB**: it keeps two KV heads with separate K and V in its global layers, while the 12B keeps one K=V head. The global-layer shape, not the parameter count, sets the cache (§7).

If you only remember one thing: a formula is a **shape and a count**. Check the shapes first, then the numbers.

---

## 1. Softmax and cross-entropy

For logits \(z\in\mathbb{R}^{V}\),

\[
p_j=\frac{e^{z_j}}{\sum_{k=1}^{V}e^{z_k}},
\qquad
\mathcal{L}_{\text{CE}}=-\log p_y.
\]

Softmax turns arbitrary scores into a probability distribution. Cross-entropy asks how much probability the model gave to the correct token.

### Example

The model has just read `return ` and must pick the next token. Suppose its vocabulary here is four candidates and it emits the logits \(z=(2,1,0,-1)\). The training text says the next token is `self`, so the target is \(y=0\).

| token | \(z_j\) | \(e^{z_j}\) | \(p_j\) |
|---|---:|---:|---:|
| `self` (target) | 2.0 | 7.3891 | **0.6439** |
| `None` | 1.0 | 2.7183 | 0.2369 |
| `0` | 0.0 | 1.0000 | 0.0871 |
| `[]` | -1.0 | 0.3679 | 0.0321 |

\[
Z=7.3891+2.7183+1.0000+0.3679=11.4752.
\]

\[
\mathcal{L}=-\log(0.6439)=0.4402,
\qquad
\text{perplexity}=e^{0.4402}=1.553.
\]

Interpretation: the model puts 64.4% on the correct token `self`, so the loss is low — a perplexity of 1.55 means it is effectively choosing between about 1.5 tokens, not 4. If it had no idea and spread probability evenly, \(p_y=0.25\), \(\mathcal{L}=\log 4=1.386\), and perplexity would be exactly 4, the full vocabulary.

A useful check: the four probabilities sum to 1. With these rounded figures \(0.6439+0.2369+0.0871+0.0321=1.0000\); rounding can leave the visible digits at 0.9999 or 1.0001, but the exact softmax always sums to exactly 1 — that is the whole point of the denominator \(Z\).

**Why it matters.** This is one position. A real forward pass over Qwen3-14B does exactly this at every position, but over \(V=151{,}936\) tokens, and the loss you watch in training is the mean of these per-token cross-entropies. A loss of 0.44 nats per token is a perplexity of 1.55; a freshly initialised model starts near \(\log V\approx11.9\).

---

## 2. Cross-entropy gradient

For one position,

\[
\nabla_z\mathcal{L}=p-e_y,
\]

where \(e_y\) is one-hot at the target.

### Example

Using the same \(p=(0.6439,0.2369,0.0871,0.0321)\) and target \(y=0\):

\[
\nabla_z\mathcal{L}
=(0.6439-1,\;0.2369,\;0.0871,\;0.0321)
=(-0.3561,\;0.2369,\;0.0871,\;0.0321).
\]

Checks:

- the entries sum to 0: \(-0.3561+0.2369+0.0871+0.0321=0\);
- the target logit is pushed **up** (negative gradient means “increase this logit” under gradient descent);
- the other logits are pushed down in proportion to their probability;
- the gradient is bounded in \([-1,1]\), no matter how large the logits are.

If the model were perfect, \(p=e_y\), the gradient would be exactly zero.

---

## 3. Softmax Jacobian

For \(p=\operatorname{softmax}(z)\),

\[
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j),
\qquad
J=\operatorname{diag}(p)-pp^\top.
\]

The Jacobian tells you how a small change in each logit changes every probability.

### Example

Let \(p=(0.5,0.3,0.2)\). Then

\[
J=
\begin{pmatrix}
0.25 & -0.15 & -0.10\\
-0.15 & 0.21 & -0.06\\
-0.10 & -0.06 & 0.16
\end{pmatrix}.
\]

Every row sums to 0. If we perturb \(z_0\) by \(+0.1\) and leave the others alone:

\[
J\begin{pmatrix}0.1\\0\\0\end{pmatrix}
=
\begin{pmatrix}0.025\\-0.015\\-0.010\end{pmatrix}.
\]

The first probability rises by 0.025; the other two fall, and the changes sum to zero. This is the “probability budget” constraint. \(J\) has rank \(V-1=2\): the all-ones direction is in its null space, because adding the same constant to every logit does not change softmax.

---

## 4. RMSNorm

\[
\operatorname{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}.
\]

It rescales the vector so its root-mean-square is 1, then applies a learned per-dimension scale \(\gamma\). It does **not** subtract the mean.

### Example

Let \(d=4\), \(x=(1,2,3,4)\), \(\epsilon=0\), \(\gamma=(1,1,1,1)\).

\[
\frac1d\sum_i x_i^2=\frac{1+4+9+16}{4}=7.5,
\qquad
r=\sqrt{7.5}=2.7386.
\]

\[
y=\frac{x}{r}=(0.3651,\;0.7303,\;1.0954,\;1.4606).
\]

Check: the RMS of \(y\) is

\[
\sqrt{\frac{0.3651^2+0.7303^2+1.0954^2+1.4606^2}{4}}=1.000.
\]

If \(x\) is multiplied by 10, \(y\) is unchanged (for \(\epsilon=0\)): RMSNorm is scale-invariant. With \(\gamma=(2,1,1,1)\), the first output becomes \(2\cdot0.3651=0.7303\).

### One gradient entry

For \(y_0=\gamma_0x_0/r\),

\[
\frac{\partial y_0}{\partial x_0}
=\frac{1}{r}\left(1-\frac{x_0^2}{d r^2}\right)
=0.3530,
\]

\[
\frac{\partial y_0}{\partial x_1}
=-\frac{x_0x_1}{d r^3}
=-0.02434.
\]

The diagonal is positive; the off-diagonal is negative. Increasing another coordinate slightly lowers \(y_0\), because it raises the denominator.

---

## 5. Scaled dot-product attention

For one head,

\[
\operatorname{Attn}(q,K,V)
=\operatorname{softmax}\left(\frac{qK^\top}{\sqrt{d_h}}\right)V.
\]

### Example

The current token wants to gather information from three earlier tokens. Its query is \(q=(1,0)\) — think of it as asking “which earlier token is aligned with direction 0?” The three earlier tokens expose keys

\[
k_1=(1,0),\quad k_2=(0,1),\quad k_3=(1,1).
\]

\(k_1\) points exactly along the query, \(k_2\) is orthogonal to it, and \(k_3\) is in between. The raw scores are

\[
qK^\top=(1,0,1),
\qquad
\frac{qK^\top}{\sqrt2}=(0.7071,\;0,\;0.7071).
\]

Exponentiate: \((2.0281,1,2.0281)\); sum \(5.0562\);

\[
p=(0.4011,\;0.1978,\;0.4011).
\]

With values \(v_1=(1,0)\), \(v_2=(0,1)\), \(v_3=(1,1)\):

\[
O=0.4011(1,0)+0.1978(0,1)+0.4011(1,1)
=(0.8022,\;0.5989).
\]

The output is a weighted average of the values. The weights sum to 1. Notice the aligned key \(k_1\) and the in-between key \(k_3\) each get weight 0.40, while the orthogonal key \(k_2\) gets only 0.20: attention copies more from the tokens whose keys point the way the query asks. That is the whole mechanism — a content-addressed, soft lookup.

### Causal mask

At position \(t=1\) (second token), the model may attend only to positions 0,1:

\[
\text{scores}=(0.7071,0),
\qquad
p=(0.6698,0.3302),
\]

\[
O=0.6698(1,0)+0.3302(0,1)=(0.6698,0.3302).
\]

The future key \(k_3\) is masked to \(-\infty\), so its weight is 0. In a full \(T\times T\) mask,

\[
M=
\begin{pmatrix}
0 & -\infty & -\infty\\
0 & 0 & -\infty\\
0 & 0 & 0
\end{pmatrix}.
\]

### Why divide by the square root of the head dimension

If each \(q_i,k_i\) has mean 0 and variance 1 and is independent, then

\[
\operatorname{Var}(q\cdot k)=\sum_{i=1}^{d_h}\operatorname{Var}(q_ik_i)=d_h.
\]

For \(d_h=2\), the dot product has variance 2 and standard deviation 1.414. Dividing by \(\sqrt2\) gives variance 1 and standard deviation 1, keeping softmax in a responsive range instead of saturating.

### Attention backward, numerically

Take \(T=2\), \(d_h=2\), \(V=I\), and

\[
P=
\begin{pmatrix}
0.5 & 0.5\\
0.2 & 0.8
\end{pmatrix},
\qquad
dO=
\begin{pmatrix}
1 & 0\\
0 & 1
\end{pmatrix}.
\]

Then \(dP=dO\,V^\top=dO\), and \(dV=P^\top dO=P^\top\).

For row 0: \(p=(0.5,0.5)\), \(dP=(1,0)\), \(s=dP\cdot p=0.5\),

\[
dS=(0.5(1-0.5),\;0.5(0-0.5))=(0.25,-0.25).
\]

For row 1: \(p=(0.2,0.8)\), \(dP=(0,1)\), \(s=0.8\),

\[
dS=(0.2(0-0.8),\;0.8(1-0.8))=(-0.16,0.16).
\]

Each row of \(dS\) sums to zero, just like the softmax Jacobian. That is the same probability-budget constraint.

**Why it matters.** Scale this up: Qwen3-14B runs 40 of these heads per layer across 40 layers, each with \(d_h=128\) instead of 2, over thousands of tokens instead of three. The arithmetic is identical; only the dimensions grow. The \(T\times T\) score matrix is also what makes long context expensive — at \(T=32{,}768\) it has a billion entries per head, which is why FlashAttention never materialises it and why §15's attention FLOP term scales as \(T^2\).

---

## 6. Multi-head and grouped-query attention

With \(H\) query heads and \(G\) KV heads:

\[
Q\in\mathbb{R}^{H\times d_h},
\qquad
K,V\in\mathbb{R}^{G\times d_h},
\]

and each KV head is shared by \(H/G\) query heads.

### Parameter example

Toy model: \(d=4\), \(H=2\), \(d_h=2\).

- Q projection: \(dHd_h=4\cdot2\cdot2=16\).
- K projection: \(dGd_h=4\cdot1\cdot2=8\).
- V projection: 8.
- O projection: \(Hd_hd=2\cdot2\cdot4=16\).

With \(G=1\), attention parameters are \(16+8+8+16=48\). With \(G=2\) (MHA), K and V are 16 each, total \(64\). GQA saves 16 parameters, 25%.

The KV cache ratio is \(G/H=1/2\). For Qwen3-14B, \(G/H=8/40=1/5\), so the cache is one fifth of MHA.

### What GQA does not do

It does **not** average the query heads. Each query head still computes its own attention:

\[
O_h=\operatorname{softmax}\left(\frac{Q_hK^\top}{\sqrt{d_h}}\right)V.
\]

Only \(K\) and \(V\) are shared. The saving is in the cache and the K/V projections, not in the query computation.

!!! example "On paper"

    Two query heads share one KV head holding two tokens, with values \(v_1=(1,0)\) and \(v_2=(0,1)\). Head 1's scaled scores against the shared keys are \((1,0)\); head 2's are \((0,1)\). With \(e\approx2.718\):

    \[
    \frac{e}{e+1}=\frac{2.718}{3.718}=0.731
    \]

    so head 1 attends \((0.731,\,0.269)\) and outputs \((0.731,\,0.269)\), while head 2 attends \((0.269,\,0.731)\) and outputs \((0.269,\,0.731)\). Averaging the two queries first would give scores \((0.5,0.5)\), weights \((0.5,0.5)\) and one blurred output \((0.5,0.5)\) for both heads. Sharing K and V keeps each head's own focus.

---

## 7. KV cache

For full attention,

\[
\text{KV bytes}=2\cdot L\cdot G\cdot d_h\cdot T\cdot b.
\]

The factor 2 is K and V; \(b\) is bytes per value (2 for bf16, 1 for int8).

### Toy example

\(L=2\), \(G=1\), \(d_h=2\), \(T=3\), \(b=2\):

\[
2\cdot2\cdot1\cdot2\cdot3\cdot2=48\text{ bytes}.
\]

That is tiny, but the formula scales linearly in every factor. Doubling context doubles the cache; doubling layers doubles it; using int8 halves it.

### Qwen3-14B

\(L=40\), \(G=8\), \(d_h=128\), \(T=32{,}768\), \(b=2\):

\[
2\cdot40\cdot8\cdot128\cdot32{,}768\cdot2=5.37\text{ GB}.
\]

### Gemma 4 12B: the corrected hybrid number

Gemma is not a full-attention stack:

- 40 local layers: \(G=8\), \(d_h=256\), window \(W=1024\);
- 8 global layers: \(G=1\), \(d_h=512\), full context, \(K=V\).

Local cache per layer is capped at \(W\), not \(T\):

\[
2\cdot40\cdot8\cdot256\cdot1024\cdot2=335.5\text{ MB}.
\]

Global cache stores only K because \(V=K\):

\[
8\cdot1\cdot512\cdot32{,}768\cdot2=268.4\text{ MB}.
\]

Total at 32K:

\[
335.5+268.4=603.9\text{ MB}.
\]

At 256K, the local term is unchanged and the global term becomes

\[
8\cdot1\cdot512\cdot262{,}144\cdot2=2.147\text{ GB},
\]

so the total is **2.48 GB**. The naive full-attention calculation \(2\cdot48\cdot8\cdot256\cdot32{,}768\cdot2=12.88\text{ GB}\) is an upper bound, not the deployed cache. This is the single most important correction in this chapter.

### Gemma 4 E4B and E2B: shared KV

The edge models add one more rule: the last `num_kv_shared_layers` layers reuse the K/V of an earlier layer and **cache nothing** ([37 §4.3](37-architecture.md)). E4B has 42 layers and shares the last 18, so only layers 0–23 cache — 20 local and 4 global. There is no K=V, so a global layer stores both K and V, with \(G=2\), \(d_h=512\); the local window is \(W=512\).

Local term, capped at the window:

\[
2\cdot20\cdot2\cdot256\cdot512\cdot2=21.0\text{ MB}.
\]

Global term, per token:

\[
4\cdot2\cdot2\cdot512\cdot2=16{,}384\text{ bytes}=16\text{ KB}.
\]

At 32K that is \(16{,}384\cdot32{,}768=536.9\) MB, so the cache is **558 MB**. At 128K (131,072 tokens) the global term is 2.15 GB and the cache **2.17 GB**.

Now compare with the 12B at the same 128K: \(335.5\text{ MB}+8\cdot1\cdot512\cdot131{,}072\cdot2=1.41\) GB. The 12B pays 8 KB per token for its global layers (one K=V head each); E4B pays 16 KB (two heads, K and V). **The 4B-class model needs the bigger long-context cache.**

E2B shares 20 of 35 layers, leaving 12 local and 3 global with \(G=1\):

\[
2\cdot12\cdot1\cdot256\cdot512\cdot2=6.3\text{ MB},
\qquad
3\cdot2\cdot1\cdot512\cdot2=6\text{ KB per token},
\]

so 208 MB at 32K and 812 MB at 128K. A runtime that ignored the sharing and cached every layer would need 0.98 GB (E4B) and 0.48 GB (E2B) at 32K; sharing removes 43% and 57%.

**Why it matters.** On the handbook's 32 GB Mac this is the difference between fitting a long context and not. A 4-bit Qwen3-14B is about 8.3 GB of weights; its 5.37 GB bf16 KV cache at 32K is the *second* largest consumer of memory, and int8 KV (§16) halves it to 2.68 GB. Gemma's hybrid design keeps even a 256K cache (2.48 GB) smaller than Qwen's 32K cache — the sliding window and single global KV head, not the parameter count, are what let a 12B model serve a quarter-million tokens.

---

## 8. RoPE

For dimension pair \(i\),

\[
\theta_{t,i}=t\,\omega_i,
\qquad
\omega_i=\text{base}^{-2i/d_h},
\]

and the pair is rotated by \(\theta_{t,i}\).

### Example

Let \(d_h=4\), base \(=10{,}000\). Then

\[
\omega_0=1,\qquad
\omega_1=10{,}000^{-2/4}=0.01.
\]

At position \(t=1\), pair 0 rotates by 1 radian:

\[
R_1(1,0)=(\cos1,\sin1)=(0.5403,0.8415).
\]

A key at position \(s=0\) is \((1,0)\). Its dot product with the query is

\[
0.5403\cdot1+0.8415\cdot0=0.5403=\cos(1).
\]

At \(t=2\), the dot product with the same key is \(\cos(2)=-0.4161\). It depends on \(t-s\).

For pair 1, \(\omega_1=0.01\), so at \(t=1\) the rotation is only 0.01 radian:

\[
(\cos0.01,\sin0.01)=(0.99995,0.01000).
\]

Slow dimensions encode long wavelengths; fast dimensions encode local order.

### Relative property

\[
R_a^\top R_b=R_{b-a}.
\]

A query at position 5 and key at position 3 see angle \((5-3)\omega=2\omega\), exactly the same as a query at 105 and key at 103. Shifting both positions by 100 changes nothing. That is why RoPE is relative.

### p-RoPE

With \(d_h=4\), \(p=0.5\), only the first \(0.5\cdot4=2\) dimensions rotate. Starting from \(q=(1,0,1,0)\) and angle \(0.5\):

\[
\tilde q=(0.8776,\;0.4794,\;1,\;0).
\]

The last two dimensions are untouched. Gemma 4 uses \(p=0.25\) on global layers: with \(d_h=512\), \(0.25\cdot512=128\) dimensions rotate and 384 do not.

---

## 9. SwiGLU and GeGLU

\[
\operatorname{SwiGLU}(x)=W_d\big(\operatorname{SiLU}(W_gx)\odot W_ux\big),
\qquad
\operatorname{SiLU}(a)=a\sigma(a).
\]

### Example

Let \(d=2\), \(d_{\text{ff}}=3\), \(x=(1,-1)\), and

\[
W_g=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix},
\quad
W_u=\begin{pmatrix}1&0\\0&1\\1&-1\end{pmatrix},
\quad
W_d=\begin{pmatrix}1&1&1\end{pmatrix}.
\]

Then

\[
a=W_gx=(1,-1,0),
\qquad
b=W_ux=(1,-1,2).
\]

SiLU:

\[
\operatorname{SiLU}(1)=0.7311,\quad
\operatorname{SiLU}(-1)=-0.2689,\quad
\operatorname{SiLU}(0)=0.
\]

Gate times up-projection:

\[
(0.7311,-0.2689,0)\odot(1,-1,2)=(0.7311,0.2689,0).
\]

Down-projection:

\[
y=0.7311+0.2689+0=1.0000.
\]

With GeGLU, \(\operatorname{GELU}(1)=0.8413\), \(\operatorname{GELU}(-1)=-0.1587\), \(\operatorname{GELU}(0)=0\), and the same weighted sum is also 1.0000. Both are valid; the difference is the nonlinearity and its gradient.

Parameter count for this layer:

\[
3\cdot d\cdot d_{\text{ff}}=3\cdot2\cdot3=18.
\]

### Backward check

For SiLU, with upstream \(\bar y=1\),

\[
g'(1)=0.9277,\qquad g'(-1)=0.0723,\qquad g'(0)=0.5.
\]

Then

\[
\frac{\partial\mathcal{L}}{\partial a}
=(W_d^\top\bar y)\odot b\odot g'(a)
=(1,1,1)\odot(1,-1,2)\odot(0.9277,0.0723,0.5)
=(0.9277,-0.0723,1.0),
\]

\[
\frac{\partial\mathcal{L}}{\partial b}
=(1,1,1)\odot(0.7311,-0.2689,0)
=(0.7311,-0.2689,0).
\]

The negative entry in \(\partial\mathcal{L}/\partial a\) is real: increasing \(a_1\) (which is negative) reduces the positive contribution from \(b_1=-1\). This is the kind of sign check a shape-only derivation misses.

---

## 10. Mixture of experts

\[
g(x)=\operatorname{softmax}(W_rx),
\qquad
y=\sum_{i\in\operatorname{TopK}(g(x))}
\frac{g_i(x)}{\sum_{j\in\operatorname{TopK}}g_j(x)}E_i(x).
\]

### Routing example

Let \(E=4\), \(k=2\), router logits \(r=(2,1,0,-1)\). From §1,

\[
g=(0.6439,\;0.2369,\;0.0871,\;0.0321).
\]

Top-2 experts are 0 and 1. Renormalised:

\[
\frac{g_0}{g_0+g_1}=0.7311,\qquad
\frac{g_1}{g_0+g_1}=0.2689.
\]

If \(E_0(x)=(1,0)\) and \(E_1(x)=(0,1)\):

\[
y=0.7311(1,0)+0.2689(0,1)=(0.7311,0.2689).
\]

Expert 2 and 3 receive no gradient for this token. That is why load balancing matters.

### Load-balancing loss

\[
\mathcal{L}_{\text{bal}}=\alpha E\sum_{i=1}^{E}f_iP_i.
\]

Balanced: \(f_i=P_i=1/4\) for all four experts. Then

\[
\sum_i f_iP_i=4\cdot\frac{1}{16}=0.25,
\qquad
\mathcal{L}_{\text{bal}}=\alpha\cdot4\cdot0.25=\alpha.
\]

Collapsed: all tokens go to expert 0, so \(f=(1,0,0,0)\), \(P=(1,0,0,0)\),

\[
\sum_i f_iP_i=1,
\qquad
\mathcal{L}_{\text{bal}}=4\alpha.
\]

The loss is four times larger when routing collapses. With \(\alpha=0.01\), the penalty goes from 0.01 to 0.04 — small in absolute terms, but its gradient is what pushes tokens to other experts.

### Active parameters

Toy numbers: attention 10M, router 0.1M, eight experts of 5M each, embeddings 2M, \(k=1\).

\[
N_{\text{active}}\approx10+0.1+\frac{1}{8}(40)+2
=17.1\text{M}.
\]

Total stored: \(10+0.1+40+2=52.1\text{M}\). The model stores 52.1M but computes with 17.1M per token, about 33%.

---

## 11. Gated DeltaNet

The delta rule:

\[
S_t=\alpha_tS_{t-1}
+\beta_t\big(v_t-\alpha_tS_{t-1}k_t\big)k_t^\top,
\qquad
o_t=S_t\,q_t,
\]

with the state \(S_t\in\mathbb{R}^{d_v\times d_k}\) mapping keys to values.

Use the scalar case \(d_k=d_v=1\) so \(S\) is a number. Let \(\alpha=1\), \(\beta=0.5\), \(S_0=0\).

**Step 1:** \(k_1=1\), \(v_1=2\).

\[
S_1=0+0.5(2-0)\cdot1=1.0,
\qquad
o_1=1.0.
\]

**Step 2:** \(k_2=1\), \(v_2=4\). The old state predicts 1, the new value is 4, error \(-3\).

\[
S_2=1.0+0.5(4-1)\cdot1=2.5,
\qquad
o_2=2.5.
\]

With forgetting \(\alpha=0.5\):

\[
S'=0.5\cdot1=0.5,
\]

\[
S_2=0.5+0.5(4-0.5)\cdot1=2.25,
\qquad
o_2=2.25.
\]

The gate pulls the state toward zero before the new association is written.

### It is a gradient step

\[
\mathcal{L}(S)=\frac12(Sk-v)^2.
\]

At \(S=1\), \(k=1\), \(v=4\):

\[
\frac{\partial\mathcal{L}}{\partial S}=(S-v)k=(1-4)\cdot1=-3.
\]

A step of size \(\beta=0.5\):

\[
S\leftarrow1-0.5(-3)=2.5,
\]

which is exactly the delta update. The delta rule is **one online gradient step** on the associative-memory loss.

In the real layer, \(k\) is L2-normalised. Without that, a large \(\|k\|^2\) makes the effective step \(\beta\|k\|^2\) and can destabilise the recurrence.

---

## 12. Sliding-window attention

A local layer with window \(W\) lets position \(t\) attend to

\[
[\max(0,t-W+1),\;t].
\]

### Example

\(W=4\), \(T=10\), position \(t=9\) (0-indexed) attends to positions 6,7,8,9 — four tokens, not ten. The cache is

\[
2Gd_hWb.
\]

Toy \(G=1,d_h=2,W=4,b=2\):

\[
2\cdot1\cdot2\cdot4\cdot2=32\text{ bytes}.
\]

The cache does not grow when \(T\) goes from 10 to 10,000. That is the entire point.

For Gemma 4 12B, the 40 local layers hold \(335.5\) MB at any context length beyond 1,024. The 8 global layers still grow with \(T\); that is why the total at 256K is 2.48 GB, not 335 MB.

---

## 13. Multi-token prediction and speculative decoding

An MTP head predicts \(n\) future tokens:

\[
\mathcal{L}_{\text{MTP}}
=-\sum_{j=1}^{n}\log p_\theta(x_{t+j}\mid x_{\le t}).
\]

Toy: \(n=2\), \(p(x_{t+1})=0.8\), \(p(x_{t+2})=0.5\).

\[
\mathcal{L}_{\text{MTP}}=-\log0.8-\log0.5=0.223+0.693=0.916.
\]

A perfect predictor would have loss 0; a uniform predictor over \(V=5\) would have \(2\log5=3.219\).

### Speculative speedup

A draft proposes \(\gamma\) tokens; the target accepts the longest correct prefix. If each token is accepted independently with probability \(\alpha\),

\[
\mathbb{E}[\text{accepted}]=\sum_{i=1}^{\gamma}\alpha^i
=\frac{\alpha(1-\alpha^\gamma)}{1-\alpha}.
\]

With \(\alpha=0.8\), \(\gamma=3\):

\[
\mathbb{E}=0.8+0.64+0.512=1.952.
\]

If the draft costs \(c=0.1\) target steps per token:

\[
\text{speedup}\approx\frac{1+1.952}{1+3(0.1)}
=\frac{2.952}{1.3}=2.27.
\]

A bad draft, \(\alpha=0.3\), \(c=0.5\), \(\gamma=3\):

\[
\mathbb{E}=0.3+0.09+0.027=0.417,
\qquad
\text{speedup}=\frac{1.417}{2.5}=0.567.
\]

It is **slower** than not drafting. Speculative decoding is a bet on acceptance rate and draft cost.

---

## 14. Parameter counting

For one layer,

\[
P_{\text{attn}}=dHd_h+2dGd_h+Hd_hd,
\qquad
P_{\text{MLP}}=3dd_{\text{ff}}.
\]

### Toy model

\(d=4\), \(H=2\), \(d_h=2\), \(G=1\), \(d_{\text{ff}}=8\), \(L=2\), \(V=5\), untied.

\[
P_{\text{attn}}=4\cdot2\cdot2+2\cdot4\cdot1\cdot2+2\cdot2\cdot4
=16+16+16=48.
\]

\[
P_{\text{MLP}}=3\cdot4\cdot8=96.
\]

Per layer: \(144\). Two layers: \(288\). Embedding plus untied head: \(2Vd=2\cdot5\cdot4=40\). Total:

\[
288+40=328.
\]

If the embedding is tied, the total is \(308\).

### Real models

| Model | Attention / layer | MLP / layer | Layers | Embedding | Total |
|---|---:|---:|---:|---:|---:|
| Qwen3-14B | 62.91M | 267.39M | 40 | 777.91M + 777.91M head | 14.77B |
| Gemma 4 12B local | 47.19M | 176.95M | 40 | 1,006.63M (tied) | — |
| Gemma 4 12B global | 64.88M | 176.95M | 8 | — | — |
| **Gemma 4 12B total** | — | — | 48 | — | **11.91B** |

Gemma’s global layer uses \(G=1\), \(d_h=512\) and K=V, so its attention is \(Q+K+O\) with **no value projection**:

\[
\underbrace{3840\cdot16\cdot512}_{Q}+\underbrace{3840\cdot1\cdot512}_{K=V}+\underbrace{16\cdot512\cdot3840}_{O}=64.88\text{M}.
\]

It is still larger than a local layer (47.19M), because its heads are twice as wide. If you count all 48 layers as local, you get 11.77B and miss about 0.14B; if you add a \(V\) projection to the global layers, you invent 15.7M weights. The published checkpoint holds 11,959,730,224 parameters: this 11.907B text decoder plus 52.4M of multimodal embedders (norms are only 0.7M). The model card rounds it to 11.95B.

### Edge models: total versus effective

The model card lists Gemma 4 E2B as “5.1B with embeddings, 2.3B effective”. Both numbers are real; they count different things. Besides the token embedding \(Vd\), E2B stores a per-layer embedding table of \(V\cdot L\cdot256\) values:

\[
\underbrace{262{,}144\cdot1{,}536}_{\text{token embedding}}=402.7\text{M},
\qquad
\underbrace{262{,}144\cdot35\cdot256}_{\text{per-layer table}}=2{,}348.8\text{M}.
\]

Those 2.75B are lookups: a token reads one row of each, so they cost storage but no matmuls. The checkpoint holds 5,123,178,979 parameters; subtract the tables and you get

\[
5.123\text{B}-2.751\text{B}=2.37\text{B},
\]

the “effective” size (the card rounds it to 2.3B). Like the file total, it still includes the 0.48B vision and audio encoders. For E4B the same subtraction gives \(7.996-3.490=4.51\)B (card: 4.5B). The text decoder that actually computes per token is smaller still: 1.88B for E2B and 3.97B for E4B.

Two edge-specific shapes make the count non-uniform. The last 20 layers of E2B share K/V, so they have no K/V projections (the file still ships 18.9M unused ones), and they have a **double-wide MLP**, \(3\cdot1536\cdot12{,}288=56.6\)M instead of 28.3M. A count that treats all 35 layers alike — what a generic estimator does — gets **1.69B** instead of 4.63B for the text decoder: it misses the per-layer table and the double-wide MLPs and wrongly adds K/V projections to the shared layers.

Check the shapes before trusting a parameter count: \(Hd_h\) is 4,096 for Gemma local, 8,192 for Gemma global, and 5,120 for Qwen3-14B. Only Qwen satisfies \(Hd_h=d\).

---

## 15. FLOPs and wall time

For dense pretraining,

\[
C_{\text{train}}\approx6ND.
\]

Forward is \(2ND\), backward is \(4ND\).

### Toy example

\(N=328\), \(D=1{,}000\):

\[
C=6\cdot328\cdot1000=1{,}968{,}000\ \text{FLOPs}.
\]

At 1 GFLOP/s effective, that is 0.00197 s. The arithmetic is trivial; the point is the unit check.

### Qwen3-14B

One million tokens:

\[
6\cdot14.77\times10^9\cdot10^6
=8.86\times10^{16}\ \text{FLOPs}.
\]

At an effective 100 TFLOP/s:

\[
\frac{8.86\times10^{16}}{10^{14}}=886\ \text{s}\approx15\ \text{min}.
\]

At 400 TFLOP/s, about 3.7 minutes. Real throughput is lower because of memory, attention and communication.

### Full pretraining

\(D=36\times10^{12}\):

\[
6\cdot14.77\times10^9\cdot36\times10^{12}
=3.19\times10^{24}\ \text{FLOPs}.
\]

At 10,000 GPUs × 40% MFU × 989 TFLOP/s **dense** BF16 (NVIDIA's 1,979 TFLOP/s figure is with sparsity):

\[
\text{GPU-hours}
=\frac{3.19\times10^{24}}{989\times10^{12}\cdot0.4\cdot3600}
\approx2.24\times10^6.
\]

That is 224 hours per GPU, or about **9.3 days**. At \$2.50/GPU-hour: **\$5.6M**. Two look-alike numbers move this by 2×: 494.5 TFLOP/s is the dense **TF32** peak (using it doubles the estimate), and 1,979 is the sparse BF16 peak (using it as if it were dense halves it). That is the order of magnitude; real runs add failures, data and evaluation.

### Attention term

The exact training term is

\[
12\,L\,T^2\,H\,d_h.
\]

Toy: \(L=2\), \(T=3\), \(H=2\), \(d_h=2\):

\[
12\cdot2\cdot9\cdot2\cdot2=864\ \text{FLOPs}.
\]

If you used \(d=4\) instead of \(Hd_h=4\), you would get the same number here. For Gemma global layers, \(Hd_h=8{,}192\) while \(d=3{,}840\), so the difference is large.

### Wall time

\[
\text{time}=\frac{\text{tokens}}{\text{tokens/sec}}.
\]

5M tokens at 25 tokens/s:

\[
\frac{5{,}000{,}000}{25}=200{,}000\ \text{s}\approx55.6\ \text{hours}.
\]

At 50 tokens/s: 27.8 hours. Always measure tokens/s on the actual machine; do not infer it from parameter count alone.

---

## 16. Quantization

Symmetric affine quantization:

\[
q=\operatorname{clamp}\left(\operatorname{round}\left(\frac{w}{s}\right),q_{\min},q_{\max}\right),
\qquad
\hat w=sq,
\qquad
s=\frac{\max|w|}{2^{k-1}-1}.
\]

### Example: 4-bit

Take five weights from one row of a projection matrix, \(w=(-1,-0.4,0,0.4,1)\), and store them in 4 bits (\(k=4\)), so the integer grid runs \(-7\ldots7\): \(2^{3}-1=7\). The scale is the largest magnitude divided by 7:

\[
s=\frac{\max|w|}{7}=\frac{1}{7}=0.142857.
\]

\[
q=(-7,-3,0,3,7),
\qquad
\hat w=(-1,-0.4286,0,0.4286,1).
\]

Errors: \(0,0.0286,0,0.0286,0\). The maximum error is \(0.0286\), below \(s/2=0.0714\).

The representable values are spaced by \(s\); the quantization error is at most half a step.

### Blockwise scales

Real weight rows are not uniform: one stretch can be large and another tiny. Split the row into two blocks, \(w_1=(-1,-0.4)\) with large values and \(w_2=(0.02,0.04)\) with small ones.

Per-tensor scale \(s=1/7=0.142857\):

- block 1: \(q=(-7,-3)\), errors \(0,0.0286\);
- block 2: \(q=(0,0)\), errors \(0.02,0.04\).

Per-block scales:

- \(s_1=1/7=0.142857\), same as above;
- \(s_2=0.04/7=0.005714\), \(q=(4,7)\), errors \(0.00286,0\).

The second block’s maximum error falls from 0.04 to 0.00286 — a 14× improvement on the small-value block, because its scale no longer has to reach all the way to 1. That is why blockwise quantization is better: each block gets a scale matched to its own range. Real GGUF Q4_0 does exactly this with 32-weight blocks, one fp16 scale each; a model's weights span orders of magnitude, so a single per-tensor scale would waste almost all of the 4-bit grid on the few largest weights.

**Q4_0 caveat.** The formula above uses a symmetric grid \([-7,7]\). GGML Q4_0 uses codes \(-8\ldots7\) and \(s=\max/8\), so the positive maximum maps to 7 and the negative extreme to -8. The error bound \(s/2\) is approximate for that format. The practical lesson is the same: smaller blocks and a suitable scale reduce error.

### QAT

Rounding has zero derivative almost everywhere, so QAT uses a straight-through estimator:

\[
\frac{\partial\hat w}{\partial w}\approx1.
\]

The forward pass sees the quantized weight; the backward pass pretends quantization is the identity. The model learns weights that remain good after rounding.

!!! example "On paper"

    A latent weight \(w=0.30\) with scale \(s=0.25\) is stored as \(q=\operatorname{round}(0.30/0.25)=\operatorname{round}(1.2)=1\), so the forward pass uses \(\hat w=0.25\). Suppose the loss gradient there is \(\partial\mathcal{L}/\partial\hat w=0.4\). The true derivative of rounding is 0, so plain backprop would never move \(w\). The straight-through estimator passes the 0.4 through unchanged; with learning rate 0.5:

    \[
    w\leftarrow0.30-0.5\cdot0.4=0.10
    \]

    Now \(\operatorname{round}(0.10/0.25)=\operatorname{round}(0.4)=0\): the next forward pass uses \(\hat w=0\). The latent weight moved smoothly, and the stored weight jumped to the neighbouring grid point once it crossed the rounding boundary at 0.125.

### KV quantization

The KV cache formula is linear in \(b\). Changing bf16 (\(b=2\)) to int8 (\(b=1\)) halves the cache:

- Qwen3-14B 32K: 5.37 GB → 2.68 GB;
- Gemma 4 12B 32K: 604 MB → 302 MB;
- Gemma 4 12B 256K: 2.48 GB → 1.24 GB.

Quantization is not free: it adds dequantization work and can hurt long-context retrieval. Measure both quality and latency.

---

## 17. AdamW and the learning-rate schedule

AdamW keeps

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\]

then corrects the bias:

\[
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},
\]

and updates

\[
\theta_t=\theta_{t-1}-\eta_t
\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
+\lambda\theta_{t-1}\right).
\]

### One-parameter example

Let \(\theta=1\), \(g=0.5\) at every step, \(\beta_1=0.9\), \(\beta_2=0.95\), \(\eta=0.1\), \(\lambda=0\), \(\epsilon\approx0\).

**Step 1:**

\[
m_1=0.05,\quad v_1=0.0125,
\]

\[
\hat m_1=\frac{0.05}{1-0.9}=0.5,
\qquad
\hat v_1=\frac{0.0125}{1-0.95}=0.25.
\]

\[
\theta_1=1-0.1\cdot\frac{0.5}{\sqrt{0.25}}=1-0.1=0.9.
\]

**Step 2:**

\[
m_2=0.9(0.05)+0.1(0.5)=0.095,
\]

\[
v_2=0.95(0.0125)+0.05(0.25)=0.024375,
\]

\[
\hat m_2=\frac{0.095}{1-0.81}=0.5,
\qquad
\hat v_2=\frac{0.024375}{1-0.9025}=0.25,
\]

\[
\theta_2=0.9-0.1=0.8.
\]

With a constant gradient, AdamW takes a constant step of size \(\eta\). Without bias correction at \(t=1\), the step would be \(0.1\cdot0.05/\sqrt{0.0125}=0.0447\), less than half as large. Bias correction matters most in the first few steps.

### Weight decay

If \(\lambda=0.1\), the decay term at \(\theta=1\) is \(0.1\cdot1=0.1\); multiplied by \(\eta=0.1\), it subtracts 0.01 before the gradient step. Weight decay is decoupled from the adaptive denominator; that is the “W” in AdamW.

### Cosine schedule

\[
\eta_t=\eta_{\min}+\frac12(\eta_{\max}-\eta_{\min})
\left(1+\cos\frac{\pi(t-T_w)}{T-T_w}\right).
\]

Let \(\eta_{\max}=10^{-4}\), \(\eta_{\min}=10^{-6}\), \(T_w=10\), \(T=100\), \(t=55\).

\[
\frac{t-T_w}{T-T_w}=\frac{45}{90}=0.5,
\qquad
\cos(\pi\cdot0.5)=0.
\]

\[
\eta_{55}=10^{-6}+\frac12(10^{-4}-10^{-6})(1+0)
=5.05\times10^{-5}.
\]

Halfway through the cosine, the LR is halfway between the minimum and maximum. At \(t=T_w\), it is \(\eta_{\max}\); at \(t=T\), it is \(\eta_{\min}\).

---

## 18. LoRA scaling

A LoRA layer replaces a frozen weight with

\[
W'=W+\frac{\alpha_{\text{PEFT}}}{r}BA.
\]

MLX-LM calls the multiplier `scale`:

\[
W'=W+\text{scale}\cdot BA.
\]

### Example

Let \(r=4\), \(\alpha_{\text{PEFT}}=8\), so

\[
\text{scale}=\frac{8}{4}=2.
\]

For a \(2\times2\) toy projection, let

\[
A=\begin{pmatrix}1&0\\0&1\end{pmatrix},
\qquad
B=\begin{pmatrix}0.5&0\\0&0.5\end{pmatrix}.
\]

Then

\[
BA=\begin{pmatrix}0.5&0\\0&0.5\end{pmatrix},
\qquad
\text{scale}\cdot BA=\begin{pmatrix}1&0\\0&1\end{pmatrix}.
\]

The adapter adds the identity to \(W\). If you mistakenly set `scale: 20` with rank 4, you add \(5I\), five times too large — the adapter overwhelms the pretrained weight and the model forgets what it knew. This is the single most common LoRA config error: `scale` is \(\alpha_{\text{PEFT}}/r\), not \(\alpha_{\text{PEFT}}\). For rank 16 and PEFT alpha 32, the correct scale is \(32/16=2.0\), not 32 and not 20. Run `scripts/check_lora.py` to confirm the value before a long run.

### Parameter count

For a \(4096\times4096\) projection, full fine-tuning has \(16.78\)M weights. LoRA rank 16 has

\[
r(d+k)=16(4096+4096)=131{,}072
\]

trainable parameters, about 0.78% of the full matrix. Rank 4 has 32,768, about 0.20%.

**Why it matters.** Across all attention and MLP projections, a rank-16 adapter on Qwen3-14B trains on the order of tens of millions of parameters while the 14.77B base stays frozen (and 4-bit). That ratio is the entire reason fine-tuning fits in 32 GB: you store the big model once, quantized, and only the small \(BA\) factors carry gradients and optimizer state.

---

## 19. Distillation: CE, KD and top-k

For a teacher \(T\) and student \(S\):

\[
p_T^{T}=\operatorname{softmax}\left(\frac{z_T}{T_{\text{temp}}}\right),
\qquad
p_S^{T}=\operatorname{softmax}\left(\frac{z_S}{T_{\text{temp}}}\right),
\]

\[
\mathcal{L}_{\text{KD}}
=T_{\text{temp}}^2\operatorname{KL}(p_T^T\|p_S^T),
\]

\[
\mathcal{L}=(1-\alpha)\mathcal{L}_{\text{CE}}+\alpha\mathcal{L}_{\text{KD}}.
\]

### Example

A large teacher (say Qwen3-14B) and a small student (say an 8B) both score the same next-token position over three candidates. The teacher's logits are \(z_T=(2,1,0)\) and the student's are \(z_S=(1,2,0)\); we soften both with temperature \(T_{\text{temp}}=2\), weight the two losses equally (\(\alpha=0.5\)), and the ground-truth token is \(y=0\).

At temperature 2:

\[
p_T^T=(0.5065,\;0.3072,\;0.1863),
\qquad
p_S^T=(0.3072,\;0.5065,\;0.1863).
\]

The teacher prefers token 0; the student prefers token 1. The KL is

\[
\operatorname{KL}
=0.5065\log\frac{0.5065}{0.3072}
+0.3072\log\frac{0.3072}{0.5065}
+0.1863\log1
=0.09964.
\]

\[
\mathcal{L}_{\text{KD}}=2^2\cdot0.09964=0.3986.
\]

For the hard-label CE, use the student at temperature 1:

\[
p_S=\operatorname{softmax}(1,2,0)=(0.2447,\;0.6652,\;0.0900),
\]

\[
\mathcal{L}_{\text{CE}}=-\log0.2447=1.4076.
\]

Total:

\[
\mathcal{L}=0.5(1.4076)+0.5(0.3986)=0.9031.
\]

The KD term is small because the student and teacher are not very far apart after temperature softening. If the student were confidently wrong, the KL term would dominate.

### Top-k

Keep the teacher’s top 2 tokens (0 and 1). Renormalise the teacher:

\[
q=(0.6225,\;0.3775).
\]

Student probabilities at those tokens (temperature 2):

\[
p_S=(0.3072,\;0.5065).
\]

\[
\mathcal{L}_{\text{top-k}}
=-(0.6225\log0.3072+0.3775\log0.5065)
=0.9915.
\]

Multiplying by \(T^2\) gives 3.966. This is **not** equal to the full KD term (0.3986; the KL itself is 0.0996): top-k drops the teacher’s tail and does not renormalise the student. It is a different estimator, not a free approximation. Use top-k for memory, then validate on the full distribution for a sample.

### Sequence KD

Sequence KD is the special case where you do not use teacher logits. You sample \(\tilde y\sim p_T\) and train

\[
\mathcal{L}_{\text{seq}}
=-\sum_t\log p_S(\tilde y_t\mid\tilde y_{<t},x).
\]

That is ordinary SFT on teacher traces. It transfers behaviour, not the full distribution. On a 32 GB Mac, this is the path MLX-LM supports.

!!! example "On paper"

    The teacher wrote the three-token trace `return`, `self`, `.` and the student gives those tokens probabilities 0.5, 0.25 and 0.8. With \(\ln0.5=-0.693\), \(\ln0.25=-1.386\) and \(\ln0.8=-0.223\):

    \[
    \mathcal{L}_{\text{seq}}=0.693+1.386+0.223=2.303
    \]

    That is \(-\ln(0.5\cdot0.25\cdot0.8)=-\ln0.1\): the loss is minus the log-probability of the whole trace, 0.768 per token. No teacher logits are needed — only the text the teacher wrote.

---

## 20. Evaluation math

### Pass@k

For \(n\) samples with \(c\) correct, the unbiased estimator is

\[
\text{pass@}k
=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}.
\]

Example: \(n=10\), \(c=4\), \(k=5\).

\[
\text{pass@}5
=1-\frac{\binom{6}{5}}{\binom{10}{5}}
=1-\frac{6}{252}
=0.9762.
\]

There is a 97.6% chance that at least one of five samples is correct, given 4 of 10 are correct. This is a capability estimate, not the deployment success rate. For deployment, use pass@1.

### Wilson interval

Your model solved 40 of 100 eval tasks, so \(\hat p=0.4\), \(n=100\). For a 95% interval, \(z=1.96\), so \(z^2=3.8416\). The Wilson score interval is

\[
\frac{\hat p+\dfrac{z^2}{2n}\;\pm\;z\sqrt{\dfrac{\hat p(1-\hat p)}{n}+\dfrac{z^2}{4n^2}}}{1+\dfrac{z^2}{n}}.
\]

The denominator is \(1+z^2/n=1.038416\). The centre is shifted slightly above \(\hat p\):

\[
\text{centre}=\frac{0.4+0.019208}{1.038416}=\frac{0.419208}{1.038416}=0.4037.
\]

The half-width uses both terms under the root, \(\hat p(1-\hat p)/n=0.0024\) and \(z^2/4n^2=0.000096\):

\[
\text{half-width}=\frac{1.96\sqrt{0.0024+0.000096}}{1.038416}
=\frac{1.96\cdot0.049960}{1.038416}=0.0943.
\]

\[
\text{interval}=0.4037\pm0.0943=(0.3094,\;0.4980).
\]

So “40 of 100” is consistent with a true pass rate anywhere from 31% to 50%. A two-point difference between two models on 100 tasks is well inside the noise. Report the interval, not just the point estimate — and if you need to distinguish a two-point gap, you need far more than 100 tasks.

### Agreement

\[
\text{agreement}=\frac1N\sum_i\mathbf{1}[\arg\max p_S=\arg\max p_T].
\]

Example: the student matches the teacher’s top token on 7 of 10 prompts: \(7/10=0.70\). This is a diagnostic, not a quality score; the student can agree and still fail the task.

### Expected calibration error

Partition predictions into confidence bins \(b\), with \(n_b\) predictions, accuracy \(\operatorname{acc}_b\) and mean confidence \(\operatorname{conf}_b\):

\[
\operatorname{ECE}=\sum_b\frac{n_b}{N}
\left|\operatorname{acc}_b-\operatorname{conf}_b\right|.
\]

Example: 10 predictions in two bins of 5.

- bin 1: mean confidence 0.9, 4/5 correct → \(|0.8-0.9|=0.1\);
- bin 2: mean confidence 0.6, 2/5 correct → \(|0.4-0.6|=0.2\).

\[
\operatorname{ECE}=0.5(0.1)+0.5(0.2)=0.15.
\]

A perfectly calibrated model has ECE 0. A model that is right 60% of the time but always says 95% has a large ECE even if its accuracy is fine.

---

## 21. The one-page check

If you can reproduce these numbers, you can debug most transformer math:

| Quantity | Formula | Toy answer |
|---|---|---:|
| Softmax of \((2,1,0,-1)\) | \(e^{z}/\sum e^{z}\) | \((0.6439,0.2369,0.0871,0.0321)\) |
| CE, target 0 | \(-\log p_0\) | 0.4402 |
| CE gradient | \(p-e_y\) | \((-0.3561,0.2369,0.0871,0.0321)\) |
| RMSNorm of \((1,2,3,4)\) | \(x/\mathrm{RMS}(x)\) | \((0.3651,0.7303,1.0954,1.4606)\) |
| Attention output, toy | \(\operatorname{softmax}(qK^\top/\sqrt2)V\) | \((0.8022,0.5989)\) |
| GQA params, toy | \(dHd_h+2dGd_h+Hd_hd\) | 48 (GQA) vs 64 (MHA) |
| KV cache, toy | \(2LGd_hTb\) | 48 bytes |
| Gemma 12B KV, 32K | hybrid sum | 604 MB |
| Gemma E4B KV, 128K | shared-KV sum | 2.17 GB (12B: 1.41 GB) |
| RoPE, \(t=1,s=0\) | \(\cos((t-s)\omega)\) | 0.5403 |
| SwiGLU, toy | \(W_d(\mathrm{SiLU}(W_gx)\odot W_ux)\) | 1.0000 |
| MoE top-2, toy | renormalised \(g\) | (0.7311,0.2689) |
| GDN, two steps | delta rule | \(S_2=2.5\) |
| Speculative, \(\alpha=.8,\gamma=3,c=.1\) | \((1+E)/(1+\gamma c)\) | 2.27× |
| Toy parameters | \(L(P_{\text{attn}}+P_{\text{MLP}})+2Vd\) | 328 |
| 6ND, toy | \(6ND\) | 1.968M FLOPs |
| 4-bit quant, \(s=1/7\) | \(\hat w=sq\) | max error 0.0286 |
| AdamW, \(g=.5,\eta=.1\) | bias-corrected step | \(\theta_1=0.9\) |
| LoRA, \(r=4,\alpha=8\) | \(W+\text{scale}BA\) | scale 2 |
| KD, toy | \(T^2\mathrm{KL}\) | 0.3986 |
| Pass@5, \(n=10,c=4\) | combinatorial | 0.9762 |

If one of these disagrees with your implementation, the bug is probably in shapes, masking or the temperature — not in the architecture.

---

## 22. Sources

- Attention Is All You Need — <https://arxiv.org/abs/1706.03762>
- RoPE — <https://arxiv.org/abs/2104.09864>
- GQA — <https://arxiv.org/abs/2305.13245>
- RMSNorm — <https://arxiv.org/abs/1910.07467>
- SwiGLU — <https://arxiv.org/abs/2002.05202>
- Gated DeltaNet — <https://arxiv.org/abs/2412.06464>
- DeltaNet parallel form — <https://arxiv.org/abs/2406.06484>
- AdamW — <https://arxiv.org/abs/1711.05101>
- Distilling the Knowledge in a Neural Network — <https://arxiv.org/abs/1503.02531>
- Pass@k estimator — <https://arxiv.org/abs/2107.03374>
- Calibration / ECE — <https://arxiv.org/abs/1706.04599>
- [32 — Mathematics](32-mathematics.md)
- [37 — Architecture anatomy](37-architecture.md)
- [38 — Derivations and proofs](38-derivations.md)
- [41 — Distillation training](41-distill-training.md)
- [42 — Distillation evaluation](42-distill-eval.md)
