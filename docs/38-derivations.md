# 38 — Derivations and proofs

*Last verified: 27 October 2026. These are the derivations behind [32 — Mathematics](32-mathematics.md); they are written for someone who wants to implement or debug a model, not just read about one.*

Notation follows chapter 32. \(B\) is batch, \(T\) sequence, \(d\) hidden size, \(H\) query heads, \(G\) KV heads, \(d_h\) head dimension, \(V\) vocabulary.

For numeric checks of every result in this chapter, see [43 — Worked examples](43-math-examples.md).

---

## 1. Cross-entropy and its gradient

For one position with logits \(z\in\mathbb{R}^{V}\) and target \(y\in\{1,\dots,V\}\):

\[
p_j=\frac{e^{z_j}}{\sum_{k=1}^{V}e^{z_k}},
\qquad
\mathcal{L}=-\log p_y.
\]

### 1.1 Gradient

Write \(Z=\sum_k e^{z_k}\). Then

\[
\frac{\partial \mathcal{L}}{\partial z_j}
=-\frac{1}{p_y}\frac{\partial p_y}{\partial z_j}.
\]

For \(j=y\):

\[
\frac{\partial p_y}{\partial z_y}
=p_y(1-p_y),
\]

so

\[
\frac{\partial \mathcal{L}}{\partial z_y}=-(1-p_y)=p_y-1.
\]

For \(j\ne y\):

\[
\frac{\partial p_y}{\partial z_j}=-p_y p_j,
\]

so

\[
\frac{\partial \mathcal{L}}{\partial z_j}=p_j.
\]

Combining:

\[
\boxed{\nabla_z\mathcal{L}=p-e_y}
\]

where \(e_y\) is the one-hot target. This is why the logits gradient is bounded: it is a probability vector minus a one-hot vector, with entries in \([-1,1]\).

### 1.2 Softmax Jacobian

For the full vector:

\[
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j),
\]

or in matrix form:

\[
J=\operatorname{diag}(p)-pp^\top.
\]

The Jacobian is symmetric positive semidefinite with rank \(V-1\) (the all-ones direction is in the null space because probabilities sum to 1). This is the mathematical reason softmax is stable to optimize but cannot represent a full-rank linear map.

### 1.3 Numerical stability

Computing \(e^{z_j}\) directly overflows for large logits. Subtract the maximum:

\[
p_j=\frac{e^{z_j-m}}{\sum_k e^{z_k-m}},
\qquad m=\max_k z_k.
\]

This is algebraically identical and keeps the largest exponent at 1. The log-sum-exp form is

\[
\log\sum_k e^{z_k}=m+\log\sum_k e^{z_k-m}.
\]

A **z-loss** penalizes large logits:

\[
\mathcal{L}_z=\lambda\left(\log\sum_k e^{z_k}\right)^2,
\]

which keeps the softmax denominator near 1 and reduces loss spikes in large-vocabulary models.

---

## 2. Attention forward and backward

For one head, with \(Q\in\mathbb{R}^{T\times d_h}\), \(K,V\in\mathbb{R}^{T\times d_h}\):

\[
S=\frac{QK^\top}{\sqrt{d_h}}+M,
\qquad
P=\operatorname{softmax}(S),
\qquad
O=PV.
\]

\(M\) is the causal mask: \(M_{ij}=0\) for \(j\le i\), \(-\infty\) otherwise.

### 2.1 Forward complexity

- \(QK^\top\): \(T^2d_h\) multiply-adds.
- softmax: \(O(T^2)\).
- \(PV\): \(T^2d_h\).
- per head: \(O(T^2d_h)\); for \(H\) heads and \(L\) layers: \(O(LHT^2d_h)=O(LT^2d)\).

The KV cache makes autoregressive decoding \(O(T)\) per token instead of \(O(T^2)\), because only the new query attends to cached keys/values.

### 2.2 Backward

Given \(dO\):

\[
dP=dO\,V^\top,
\qquad
dV=P^\top dO.
\]

For **each row** \(i\) with softmax row \(p_i\), let \(s_i=\sum_j dP_{ij}p_{ij}\). Then

\[
dS_{ij}=p_{ij}\left(dP_{ij}-s_i\right),
\]

or in matrix form \(dS=p\odot(dP-\mathbf{1}s^\top)\), where \(s_i=dP_i\cdot p_i\). Subtract the row’s probability-weighted \(dP\), then multiply elementwise by \(p\). Then

\[
dQ=\frac{dS\,K}{\sqrt{d_h}},
\qquad
dK=\frac{dS^\top Q}{\sqrt{d_h}}.
\]

The mask sets \(dS_{ij}=0\) for \(j>i\). In practice, FlashAttention fuses these steps and never materialises the \(T\times T\) matrix in global memory; it recomputes tiles in the backward pass. That is why long-context training is feasible at all.

### 2.3 Why scale by the square root of the head dimension

If \(q_i,k_i\) are independent with mean 0 and variance 1, then

\[
\operatorname{Var}(q\cdot k)=\sum_{i=1}^{d_h}\operatorname{Var}(q_i k_i)=d_h.
\]

Dividing by \(\sqrt{d_h}\) gives unit variance, keeping the softmax in a regime where gradients are informative. Without it, logits grow with \(d_h\), softmax saturates, and gradients vanish.

---

## 3. RMSNorm gradient

For \(x\in\mathbb{R}^{d}\):

\[
r=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon},
\qquad
y_i=\gamma_i\frac{x_i}{r}.
\]

For \(i\ne j\):

\[
\frac{\partial y_i}{\partial x_j}
=\gamma_i x_i\cdot\left(-\frac{1}{2}\right)r^{-3}\cdot\frac{2x_j}{d}
=-\frac{\gamma_i x_i x_j}{d r^3}.
\]

For \(i=j\):

\[
\frac{\partial y_i}{\partial x_i}
=\frac{\gamma_i}{r}-\frac{\gamma_i x_i^2}{d r^3}.
\]

Combining:

\[
\boxed{
\frac{\partial y_i}{\partial x_j}
=\frac{\gamma_i}{r}\left(\delta_{ij}-\frac{x_i x_j}{d r^2}\right)
}
\]

The Jacobian is \(\operatorname{diag}(\gamma)/r\) times a rank-one update of the identity. It is cheap to compute and does not center the input, which is the main difference from LayerNorm.

---

## 4. RoPE is relative: proof

For a 2D pair \((x_{2i},x_{2i+1})\), define

\[
R_{\theta}=
\begin{pmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{pmatrix}.
\]

RoPE applies \(R_{t\omega_i}\) to the pair at position \(t\). The key identity is

\[
R_a^\top R_b=R_{b-a}.
\]

Proof: rotation matrices commute and compose by adding angles:

\[
R_a^\top R_b=R_{-a}R_b=R_{b-a}.
\]

For a query at \(t\) and key at \(s\):

\[
q_t=R_{t\omega}W_qx_t,
\qquad
k_s=R_{s\omega}W_kx_s.
\]

Then

\[
q_t^\top k_s
=x_t^\top W_q^\top R_{t\omega}^\top R_{s\omega}W_kx_s
=x_t^\top W_q^\top R_{(s-t)\omega}W_kx_s.
\]

The dot product depends only on \(s-t\), not on absolute positions. That is the relative property.

### 4.1 Base frequency

\[
\omega_i=\text{base}^{-2i/d_h}.
\]

For \(i=0\), \(\omega=1\) (fast rotation); for \(i=d_h/2-1\), \(\omega=\text{base}^{-1+2/d_h}\) (slow). A larger base makes the slow dimensions rotate even more slowly, which helps long-context extrapolation. Qwen3 raises the base from \(10^4\) to \(10^6\) in the long-context stage and applies YaRN/DCA scaling.

### 4.2 p-RoPE

Gemma 4 rotates only the first \(p d_h\) dimensions:

\[
\tilde q=\big[R_{\theta}(q_{1:pd_h}),q_{pd_h+1:d_h}\big].
\]

The unrotated dimensions are position-invariant. This reduces the number of rotary operations and, more importantly, avoids imposing position on dimensions that global attention uses for content. The report uses \(p=0.25\).

---

## 5. GQA equivalence and cache

GQA can be viewed as MHA with tied KV projections. For a group of \(H/G\) query heads sharing one KV head:

\[
K_h=K,\quad V_h=V \quad\text{for all }h\text{ in the group}.
\]

The attention output for each query head is still computed independently:

\[
O_h=\operatorname{softmax}\left(\frac{Q_hK^\top}{\sqrt{d_h}}\right)V.
\]

So GQA is **not** averaging queries; it shares the keys and values. The cache saving is exact:

\[
\text{cache}_{\text{GQA}}=\frac{G}{H}\text{cache}_{\text{MHA}}.
\]

The quality cost is empirical, not a theorem. 5:1 is common because it retains most of MHA’s quality at 20% of the cache.

---

## 6. SwiGLU and GeGLU derivatives

For

\[
y=W_d\big(\operatorname{SiLU}(W_gx)\odot W_ux\big),
\]

let \(a=W_gx\), \(b=W_ux\), \(g=\operatorname{SiLU}(a)=a\sigma(a)\), and let \(\bar y=\partial\mathcal{L}/\partial y\) be the upstream gradient. The correct backward pass is

\[
\frac{\partial\mathcal{L}}{\partial a}
=\left(W_d^\top\bar y\right)\odot b\odot g'(a),
\qquad
g'(a)=\sigma(a)+a\sigma(a)\big(1-\sigma(a)\big),
\]

\[
\frac{\partial\mathcal{L}}{\partial b}
=\left(W_d^\top\bar y\right)\odot g.
\]

Do not confuse the scalar derivative \(g'(a)\) with the vector gradient: the upstream gradient \(W_d^\top\bar y\) must be multiplied elementwise by \(b\) and \(g'(a)\). The gate’s derivative has the extra \(a\sigma(a)(1-\sigma(a))\) term, which is why SiLU is smooth and avoids the dead-unit behaviour of ReLU. GeGLU replaces \(\operatorname{SiLU}\) with \(\operatorname{GELU}\):

\[
\operatorname{GELU}(a)=a\Phi(a),
\qquad
\Phi(a)=\frac12\left(1+\operatorname{erf}\left(\frac{a}{\sqrt2}\right)\right).
\]

Both have three matrices, so the parameter count is the same; the difference is the nonlinearity and its gradient.

---

## 7. MoE routing and load balancing

For \(E\) experts, router logits \(r=W_rx\), and top-k set \(\mathcal{T}\):

\[
g_i=\frac{\exp(r_i)}{\sum_{j=1}^{E}\exp(r_j)},
\qquad
y=\sum_{i\in\mathcal{T}}\frac{g_i}{\sum_{j\in\mathcal{T}}g_j}E_i(x).
\]

The router gradient flows only through the selected experts and the renormalisation. If one expert is selected for most tokens, the others receive no gradient and collapse further.

### 7.1 Auxiliary load-balancing loss

With \(f_i\) the fraction of tokens routed to expert \(i\) and \(P_i\) the mean router probability:

\[
\mathcal{L}_{\text{bal}}=\alpha E\sum_{i=1}^{E}f_iP_i.
\]

If routing is uniform, \(f_i=P_i=1/E\), so \(\mathcal{L}_{\text{bal}}=\alpha\). If routing collapses, \(f_iP_i\) is large for the dominant expert, increasing the loss. The gradient pushes probability mass toward underrepresented experts.

Qwen3 uses a **global-batch** version so specialisation can vary across a batch while staying balanced globally. The exact implementation (capacity factor, token dropping, expert bias) is not fully public.

### 7.2 Active compute

For \(k\) active experts out of \(E\):

\[
N_{\text{active}}\approx N_{\text{attn}}+N_{\text{router}}
+\frac{k}{E}N_{\text{experts}}+N_{\text{embed}}.
\]

For Qwen3-30B-A3B: \(E=128\), \(k=8\), so only 6.25% of expert parameters run per token. That is why it stores 30.5B but generates at ~3.3B speed.

---

## 8. Gated DeltaNet as online gradient descent

The delta rule updates an associative memory \(S\in\mathbb{R}^{d_k\times d_v}\) by correcting its prediction error for the current key/value:

\[
\mathcal{L}(S)=\frac12\|S^\top k_t-v_t\|_2^2.
\]

The gradient is

\[
\nabla_S\mathcal{L}=(S^\top k_t-v_t)k_t^\top.
\]

A gradient step with rate \(\beta_t\) gives

\[
S_t=S_{t-1}-\beta_t(S_{t-1}^\top k_t-v_t)k_t^\top.
\]

Gated DeltaNet first decays the state by \(\alpha_t\):

\[
S'=\alpha_tS_{t-1},
\]

then corrects:

\[
S_t=S'-\beta_t(S'^\top k_t-v_t)k_t^\top.
\]

Substituting \(S'\):

\[
\boxed{
S_t=\alpha_tS_{t-1}
+\beta_t\big(v_t-\alpha_tS_{t-1}^\top k_t\big)k_t^\top
}
\]

and

\[
o_t=S_t^\top q_t.
\]

This is exactly the recurrence in chapter 32. \(\alpha_t\) is a forget gate; \(\beta_t\) is the learning rate for the current association. In the actual GDN layer \(k_t\) is L2-normalised and \(\beta_t\in(0,1)\), which keeps the rank-one update bounded; without normalisation, a large \(\|k_t\|^2\) can make \(I-\beta_tk_tk_t^\top\) unstable.

### 8.1 Why it is not just “linear attention”

Linear attention is the special case \(\alpha=1,\beta=1\):

\[
S_t=S_{t-1}+v_tk_t^\top,
\]

which never erases. The delta rule subtracts the component of the state already aligned with \(k_t\), so a new value overwrites the old association instead of accumulating. Gating adds exponential forgetting. The combination improves in-context retrieval, which is exactly what pure linear attention lacks.

### 8.2 Chunkwise parallel form

The per-token transition is

\[
S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top.
\]

The matrix \((I-\beta_tk_tk_t^\top)\) is a rank-one update of the identity. It is a Householder reflection only when \(k_t\) is unit norm and \(\beta_t=2\); in GDN it is a damped rank-one update. A product of such updates over a chunk can be written as

\[
\prod_{t=1}^{C}(I-\beta_tk_tk_t^\top)=I-K^\top W K,
\]

where \(K\) stacks the chunk’s keys and \(W\) is a lower-triangular matrix computed from the \(\beta\)’s (the WY representation). This turns the sequential recurrence into matrix multiplications over a chunk, which is why GDN can train in parallel despite being recurrent at inference.

The exact WY construction is implementation detail; the key point is that training is chunk-parallel while inference is \(O(1)\) per token.

---

## 9. Sliding-window cache

For a local layer with window \(W\), position \(t\) attends to \([\max(0,t-W+1),t]\). The cache only needs the last \(W\) keys/values:

\[
\text{local cache}=2G d_h W b
\]

per layer, independent of \(T\). For Gemma 4 with \(W=1024\), \(G=8\), \(d_h=256\), bf16:

\[
2\cdot8\cdot256\cdot1024\cdot2=8.4\text{ MB per layer}.
\]

Forty local layers ≈ 335 MB, versus gigabytes for full attention at 32K. The global layers still grow with \(T\), but there are only eight of them.

---

## 10. Speculative decoding speedup

A draft model proposes \(\gamma\) tokens. The target verifies them in one forward pass and accepts the longest correct prefix. If each draft token is accepted independently with probability \(\alpha\), the expected number accepted is

\[
\mathbb{E}[\text{accepted}]=\sum_{i=1}^{\gamma}\alpha^i
=\frac{\alpha(1-\alpha^\gamma)}{1-\alpha}.
\]

If the draft costs \(c\) times a target step, the approximate speedup is

\[
\text{speedup}\approx
\frac{1+\mathbb{E}[\text{accepted}]}{1+\gamma c}.
\]

For \(\alpha=0.8\), \(\gamma=4\), \(c=0.1\):

\[
\mathbb{E}=0.8+0.64+0.512+0.4096=2.36,
\qquad
\text{speedup}\approx\frac{3.36}{1.4}=2.4.
\]

The formula explains why speculative decoding helps most when the draft is cheap and the acceptance rate is high. A bad draft can be slower than no draft.

---

## 11. Quantization error

For symmetric uniform quantization with \(k\) bits and scale

\[
s=\frac{\max_i|w_i|}{2^{k-1}-1},
\]

the quantized value is

\[
q=\operatorname{clamp}\left(\operatorname{round}\left(\frac{w}{s}\right),-2^{k-1}+1,2^{k-1}-1\right),
\qquad
\hat w=sq.
\]

The rounding error satisfies

\[
|w-\hat w|\le\frac{s}{2}.
\]

For a block of \(B_c\) weights, using a per-block scale \(s_b\) reduces the maximum error because each block’s range is smaller:

\[
s_b=\frac{\max_{i\in b}|w_i|}{2^{k-1}-1}.
\]

This is why Q4_0 (blockwise) is better than a single per-tensor scale.

### 11.1 QAT straight-through estimator

Quantization is not differentiable at the rounding step. QAT uses

\[
\frac{\partial\hat w}{\partial w}\approx 1
\]

in the backward pass (straight-through estimator), so the model learns weights that are robust to the quantization error. Gemma 4 uses QAT for its int2/int4 mobile and Q4_0 checkpoints.

---

## 12. Scaling-law fit

A common form is

\[
L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}.
\]

- \(E\) is the irreducible loss.
- \(A/N^\alpha\) is the finite-model term.
- \(B/D^\beta\) is the finite-data term.

To fit:

1. train several small models with different \(N\) and \(D\);
2. record final loss;
3. fit \(E,A,B,\alpha,\beta\) by least squares in log space;
4. extrapolate cautiously.

The fit is only valid in the explored range. It does not tell you that 36T tokens will beat 10T for a specific model; it tells you the trend under the same data distribution and recipe.

Chinchilla’s compute-optimal allocation is approximately

\[
N^\* \propto C^{0.5},\qquad D^\*\propto C^{0.5},
\]

with \(D\approx20N\). Production models often overtrain because inference cost dominates.

---

## 13. Exact FLOPs

The \(6ND\) estimate counts matrix multiplications but not attention scores. A more complete forward+backward estimate per layer is

\[
C_{\text{layer}}\approx
\underbrace{24d^2T}_{\text{projections}}
+\underbrace{12\,H\,d_h\,T^2}_{\text{attention scores/values}}
+\underbrace{18dd_{\text{ff}}T}_{\text{MLP}}
\]

for a dense block (the exact constants depend on whether biases/norms are counted). The attention term uses the total query width \(H d_h\), not \(d\); they are equal for Qwen3-14B but not for Gemma’s global layers. The attention term is why long-context training is more expensive than \(6ND\) suggests.

For a 14B model at 32K, attention is a significant fraction of compute; at 2K, it is small. Always measure tokens/sec rather than trusting a formula.

---

## 14. AdamW bias correction

Adam keeps

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
\]

At \(t=0\), \(m_0=v_0=0\), so the estimates are biased toward zero. The corrections are

\[
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}.
\]

The update is

\[
\theta_t=\theta_{t-1}-\eta_t
\left(
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
+\lambda\theta_{t-1}
\right).
\]

The \(\sqrt{\hat v_t}\) normalizes each parameter by its gradient scale, which is why AdamW is robust to heterogeneous gradients. Weight decay is decoupled from the adaptive step, unlike classic Adam.

---

## 15. Worked numerical example

Suppose \(V=4\), logits \(z=(2.0,1.0,0.0,-1.0)\), and the target is the first token, index \(0\).

\[
e^z=(7.389,2.718,1.000,0.368),\quad Z=11.475.
\]

\[
p=(0.644,0.237,0.087,0.032).
\]

\[
\mathcal{L}=-\log(0.644)=0.440.
\]

\[
\nabla_z\mathcal{L}=p-e_0=(-0.356,0.237,0.087,0.032).
\]

The gradient sums to zero (because probabilities sum to 1 and the target is one-hot). If the prediction were perfect (\(p=e_0\)), the gradient would be zero.

This is the entire training signal for next-token prediction: push the target logit up, push all others down proportionally to their probability.

---

## 16. Sources

- Attention Is All You Need — <https://arxiv.org/abs/1706.03762>
- RoPE — <https://arxiv.org/abs/2104.09864>
- GQA — <https://arxiv.org/abs/2305.13245>
- RMSNorm — <https://arxiv.org/abs/1910.07467>
- SwiGLU — <https://arxiv.org/abs/2002.05202>
- Gated DeltaNet — <https://arxiv.org/abs/2412.06464>
- DeltaNet parallel form — <https://arxiv.org/abs/2406.06484>
- FlashAttention — <https://arxiv.org/abs/2205.14135>
- AdamW — <https://arxiv.org/abs/1711.05101>
- Chinchilla — <https://arxiv.org/abs/2203.15556>
- [32 — Mathematics](32-mathematics.md)
- [37 — Architecture anatomy](37-architecture.md)
