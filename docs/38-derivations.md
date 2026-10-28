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

!!! example "On paper"

    After "The capital of France is", a three-token toy vocabulary {Paris, Lyon, Nice} gets logits \(z=(1,0,-1)\), and the target is Paris, \(y=1\). With \(e\approx2.718\) and \(e^{-1}\approx0.368\):

    \[
    \sum_k e^{z_k}=2.718+1+0.368=4.086
    \]

    \[
    p=(0.665,\;0.245,\;0.090)
    \]

    \[
    \mathcal{L}=-\log 0.665=0.408
    \]

    Equal logits would give \(p_y=1/3\) and \(\mathcal{L}=\log 3=1.099\), so this model is already well ahead of a uniform guess.

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

!!! example "On paper"

    Same logits \(z=(1,0,-1)\), target Paris (\(y=1\)), \(p=(0.665,0.245,0.090)\):

    \[
    \nabla_z\mathcal{L}=p-e_1=(-0.335,\;0.245,\;0.090)
    \]

    The entries sum to 0. Finite-difference check on the Lyon logit: since \(\mathcal{L}=\log\sum_k e^{z_k}-z_y\), moving \(z_2\) to \(\pm0.1\) changes only the sum, to \(4.1913\) and \(3.9910\) (\(e^{0.1}\approx1.10517\), \(e^{-0.1}\approx0.90484\)):

    \[
    \begin{aligned}
    \frac{\log(4.1913/3.9910)}{0.2}&=\frac{\log1.0502}{0.2}\\
    &=0.2449
    \end{aligned}
    \]

    That matches \(p_2=1/4.0862=0.2447\); the gap in the last digit comes from the finite step. Longer version: [43 §2](43-math-examples.md).

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

!!! example "On paper"

    Take \(p=(0.5,0.25,0.25)\), the softmax of logits \((\log2,0,0)\). Then \(\operatorname{diag}(p)=\tfrac1{16}\operatorname{diag}(8,4,4)\), and \(pp^\top\) has entries \(p_ip_j\):

    \[
    J=\frac{1}{16}\begin{pmatrix}4&-2&-2\\-2&3&-1\\-2&-1&3\end{pmatrix}
    \]

    Every row sums to 0, so \(J\mathbf{1}=0\): adding the same number to all logits changes nothing. Row 1 is minus the sum of rows 2 and 3, so the rank is \(2=V-1\). Finite-difference check of \(J_{11}=p_1(1-p_1)=0.25\): nudging \(z_1\) by \(\pm0.1\) gives \(p_1=0.52498\) and \(0.47502\), a slope of \(0.2498\).

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

!!! example "On paper"

    Logits \(z=(1000,999,998)\). Computing \(e^{1000}\) directly overflows every float format: fp32 and bf16 top out near \(3.4\times10^{38}\approx e^{88.7}\), fp64 near \(e^{709.8}\). With \(m=1000\) the shifted logits are \((0,-1,-2)\), and \(e^{-1}\approx0.3679\), \(e^{-2}\approx0.1353\):

    \[
    \sum_k e^{z_k-m}=1+0.3679+0.1353=1.5032
    \]

    \[
    p=(0.665,\;0.245,\;0.090)
    \]

    \[
    \begin{aligned}
    \log\sum_k e^{z_k}&=1000+\log1.5032\\
    &=1000.408
    \end{aligned}
    \]

    These are exactly the probabilities of \(z=(1,0,-1)\) above, because every logit moved by 999. The loss for the first token is \(1000.408-1000=0.408\), also unchanged.

A **z-loss** penalizes large logits:

\[
\mathcal{L}_z=\lambda\left(\log\sum_k e^{z_k}\right)^2,
\]

which keeps the softmax denominator near 1 and reduces loss spikes in large-vocabulary models.

!!! example "On paper"

    Use \(\lambda=10^{-4}\), the PaLM value. The logits \((1,0,-1)\) and \((1000,999,998)\) give the same probabilities and the same cross-entropy, 0.408, but their log-sum-exps are 1.408 and 1000.408:

    \[
    10^{-4}\cdot1.408^2=0.000198
    \]

    \[
    10^{-4}\cdot1000.408^2=100.08
    \]

    Cross-entropy cannot tell the two apart; the z-loss pushes the second back toward \(\log\sum_k e^{z_k}\approx0\). This matters in practice: bf16 numbers near 1000 are spaced 4 apart, so 1000, 999 and 998 would all round to 1000.

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

!!! example "On paper"

    Two tokens, \(d_h=4\), so \(\sqrt{d_h}=2\). Token 2 has query \(q_2=(1,1,1,1)\); the keys are \(k_1=(2,2,0,0)\) and \(k_2=(0,0,1,-1)\):

    \[
    S_{2\cdot}=\tfrac12\,(q_2\cdot k_1,\;q_2\cdot k_2)=(2,\;0)
    \]

    With \(e^2\approx7.389\):

    \[
    P_{2\cdot}=\Big(\tfrac{7.389}{8.389},\;\tfrac{1}{8.389}\Big)=(0.881,\;0.119)
    \]

    With values \(v_1=(1,0,0,0)\) and \(v_2=(0,1,0,0)\), \(O_2=0.881\,v_1+0.119\,v_2=(0.881,0.119,0,0)\). Token 1 has \(S_{12}=-\infty\) from the mask, so \(P_{1\cdot}=(1,0)\) and \(O_1=v_1\): the first token can only read itself.

### 2.1 Forward complexity

- \(QK^\top\): \(T^2d_h\) multiply-adds.
- softmax: \(O(T^2)\).
- \(PV\): \(T^2d_h\).
- per head: \(O(T^2d_h)\); for \(H\) heads and \(L\) layers: \(O(LHT^2d_h)=O(LT^2d)\).

The KV cache makes autoregressive decoding \(O(T)\) per token instead of \(O(T^2)\), because only the new query attends to cached keys/values.

!!! example "On paper"

    Qwen3-14B has \(d_h=128=2^7\) and \(H=40\) query heads. For a prompt of \(T=4{,}096=2^{12}\) tokens, one head's \(QK^\top\) costs

    \[
    T^2d_h=2^{24}\cdot2^{7}=2^{31}\approx2.15\times10^{9}
    \]

    multiply-adds, about \(8.6\times10^{10}\) for the 40 heads of one layer. With a cache, the next token needs only one query row against the \(T\) cached keys:

    \[
    T\,d_h=2^{12}\cdot2^{7}=2^{19}=524{,}288
    \]

    per head, \(2^{12}=4{,}096\) times less than recomputing the whole score matrix.

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

or in matrix form \(dS=P\odot(dP-s\mathbf{1}^\top)\), where \(s\) is the column vector of \(s_i=dP_i\cdot p_i\), so \((s\mathbf{1}^\top)_{ij}=s_i\). Subtract each row’s probability-weighted \(dP\), then multiply elementwise by \(P\). Then

\[
dQ=\frac{dS\,K}{\sqrt{d_h}},
\qquad
dK=\frac{dS^\top Q}{\sqrt{d_h}}.
\]

The mask sets \(dS_{ij}=0\) for \(j>i\). In practice, FlashAttention fuses these steps and never materialises the \(T\times T\) matrix in global memory; it recomputes tiles in the backward pass. That is why long-context training is feasible at all.

!!! example "On paper"

    Back-propagate row 2 of the forward example above, \(p_2=(0.881,0.119)\), with upstream gradient \(dO_2=(1,-1,0,0)\). Then \(dP_2=(dO_2\cdot v_1,\;dO_2\cdot v_2)=(1,-1)\) and

    \[
    s_2=1\cdot0.881-1\cdot0.119=0.762
    \]

    \[
    \begin{aligned}
    dS_2&=\big(0.881\cdot0.238,\;0.119\cdot(-1.762)\big)\\
    &=(0.210,\;-0.210)
    \end{aligned}
    \]

    Check with the softmax Jacobian of §1.2: for two entries \(J=p_1p_2\begin{pmatrix}1&-1\\-1&1\end{pmatrix}\), so \(J\,dP_2=2p_1p_2\,(1,-1)\) and \(2\cdot0.881\cdot0.119=0.210\). Then, with \(\sqrt{d_h}=2\):

    \[
    \begin{aligned}
    dq_2&=\tfrac12\,(0.210\,k_1-0.210\,k_2)\\
    &=(0.210,\;0.210,\;-0.105,\;0.105)
    \end{aligned}
    \]

    Longer version: [43 §5](43-math-examples.md).

### 2.3 Why scale by the square root of the head dimension

If \(q_i,k_i\) are independent with mean 0 and variance 1, then

\[
\operatorname{Var}(q\cdot k)=\sum_{i=1}^{d_h}\operatorname{Var}(q_i k_i)=d_h.
\]

Dividing by \(\sqrt{d_h}\) gives unit variance, keeping the softmax in a regime where gradients are informative. Without it, logits grow with \(d_h\), softmax saturates, and gradients vanish.

!!! example "On paper"

    Let every \(q_i\) and \(k_i\) be \(\pm1\) with equal odds, so each product \(q_ik_i\) is \(\pm1\): mean 0, variance 1. For \(d_h=2\) the four equally likely sums are \(2,0,0,-2\):

    \[
    \operatorname{Var}(q\cdot k)=\tfrac14\,(4+0+0+4)=2=d_h
    \]

    For Qwen3's \(d_h=128\) the raw scores have standard deviation \(\sqrt{128}\approx11.3\). Two keys one standard deviation apart would get softmax weights 0.99999 and 0.00001; after dividing by 11.3 the gap is 1 and the weights are 0.731 and 0.269.

---

## 3. RMSNorm gradient

For \(x\in\mathbb{R}^{d}\):

\[
r=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon},
\qquad
y_i=\gamma_i\frac{x_i}{r}.
\]

!!! example "On paper"

    Take \(d=4\), \(x=(1,-1,3,5)\), \(\gamma=(1,1,1,1)\) and Qwen3's \(\epsilon=10^{-6}\), which is negligible here:

    \[
    r=\sqrt{\tfrac14\,(1+1+9+25)}=\sqrt9=3
    \]

    \[
    \begin{aligned}
    y&=\big(\tfrac13,-\tfrac13,1,\tfrac53\big)\\
    &=(0.333,-0.333,1,1.667)
    \end{aligned}
    \]

    The mean of \(y_i^2\) is \(\tfrac14\cdot\tfrac{36}{9}=1\): RMSNorm rescales the vector to unit root-mean-square without subtracting its mean (here 2).

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

!!! example "On paper"

    Take \(d=2\), \(x=(1,7)\), \(\gamma=(1,1)\), \(\epsilon=0\). Then \(r=\sqrt{(1+49)/2}=5\) and \(dr^2=50\):

    \[
    \begin{aligned}
    J&=\tfrac15\begin{pmatrix}1-\frac{1}{50}&-\frac{7}{50}\\-\frac{7}{50}&1-\frac{49}{50}\end{pmatrix}\\
    &=\begin{pmatrix}0.196&-0.028\\-0.028&0.004\end{pmatrix}
    \end{aligned}
    \]

    Check 1: \(Jx=(0.196-7\cdot0.028,\;-0.028+7\cdot0.004)=(0,0)\), as it must be, because scaling \(x\) leaves \(y\) unchanged. Check 2: moving \(x_1\) to 1.1 and 0.9 gives \(r=\sqrt{25.105}\approx5.0105\) and \(\sqrt{24.905}\approx4.9905\), so \(y_1=0.21954\) and \(0.18034\), a slope of \((0.21954-0.18034)/0.2=0.1960=J_{11}\). Longer version: [43 §4](43-math-examples.md).

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

!!! example "On paper"

    Use a toy frequency \(\omega=\pi/2\), a quarter turn per position, so every cosine and sine is 0 or \(\pm1\), and fix the content vectors \(W_qx=(1,0)\) and \(W_kx=(2,1)\). Recall \(R_\theta(a,b)=(a\cos\theta-b\sin\theta,\;a\sin\theta+b\cos\theta)\). The query sits at position \(t\), the key at \(s\):

    | \(t,\,s\) | \(q_t\) | \(k_s\) | \(q_t^\top k_s\) |
    |---:|---:|---:|---:|
    | 1, 3 | \((0,1)\) | \((1,-2)\) | \(-2\) |
    | 2, 4 | \((-1,0)\) | \((2,1)\) | \(-2\) |

    Both pairs are two positions apart, and the right-hand side gives the same number directly: \((1,0)\cdot R_{\pi}(2,1)=(1,0)\cdot(-2,-1)=-2\). At distance 1 the score changes: \((1,0)\cdot R_{\pi/2}(2,1)=(1,0)\cdot(-1,2)=-1\).

### 4.1 Base frequency

\[
\omega_i=\text{base}^{-2i/d_h}.
\]

For \(i=0\), \(\omega=1\) (fast rotation); for \(i=d_h/2-1\), \(\omega=\text{base}^{-1+2/d_h}\) (slow). A larger base makes the slow dimensions rotate even more slowly, which helps long-context extrapolation. Qwen3 raises the base from \(10^4\) to \(10^6\) in the long-context stage and applies YaRN/DCA scaling.

!!! example "On paper"

    Qwen3 uses \(d_h=128\). For the middle pair, \(i=32\), the exponent is \(2i/d_h=64/128=1/2\), so \(\omega_{32}=\text{base}^{-1/2}\):

    | base | \(\omega_{32}\) | positions per full turn, \(2\pi/\omega_{32}\) |
    |---|---:|---:|
    | \(10^4\) | 0.01 | 628 |
    | \(10^6\) | 0.001 | 6,283 |

    Pair 0 has \(\omega_0=1\) at any base and turns once every \(2\pi\approx6.28\) positions. Going from \(10^4\) to \(10^6\) slows the middle pair 10× and the slowest pair (\(i=63\)) about 93×.

### 4.2 p-RoPE

Gemma 4 rotates only the first \(p d_h\) dimensions:

\[
\tilde q=\big[R_{\theta}(q_{1:pd_h}),q_{pd_h+1:d_h}\big].
\]

The unrotated dimensions are position-invariant. This reduces the number of rotary operations and, more importantly, avoids imposing position on dimensions that global attention uses for content. The report uses \(p=0.25\).

!!! example "On paper"

    Toy head with \(d_h=8\) and \(p=0.25\): only the first \(pd_h=2\) entries form a rotated pair. With \(q=(1,0,2,-1,0,3,1,1)\) and a quarter-turn angle \(\theta=\pi/2\):

    \[
    \tilde q=(0,1,\;2,-1,0,3,1,1)
    \]

    The last six entries contribute the same amount to a query–key dot product at every distance. A Gemma 4 global layer has \(d_h=512\), so \(0.25\cdot512=128\) dimensions (64 pairs) carry position and 384 carry content only.

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

!!! example "On paper"

    One KV head stores \(k_1=(1,1,0,0)\), \(k_2=(0,0,1,1)\), \(v_1=(1,0,0,0)\), \(v_2=(0,1,0,0)\) and serves two query heads, \(q_a=(2,2,0,0)\) and \(q_b=(0,0,2,2)\), with \(\sqrt{d_h}=2\). Their scores are \((2,0)\) and \((0,2)\), so with \(e^2\approx7.389\):

    \[
    \begin{aligned}
    O_a&=0.881\,v_1+0.119\,v_2\\
    O_b&=0.119\,v_1+0.881\,v_2
    \end{aligned}
    \]

    Same cache, two different outputs. Averaging the queries first, \(\bar q=(1,1,1,1)\), would give scores \((1,1)\) and the single output \(0.5\,v_1+0.5\,v_2\), which is not what GQA computes.

So GQA is **not** averaging queries; it shares the keys and values. The cache saving is exact:

\[
\text{cache}_{\text{GQA}}=\frac{G}{H}\text{cache}_{\text{MHA}}.
\]

The quality cost is empirical, not a theorem. 5:1 is common because it retains most of MHA’s quality at 20% of the cache.

!!! example "On paper"

    Qwen3-14B has \(H=40\), \(G=8\), \(d_h=128\), 40 layers and a bf16 cache (2 bytes). One token's K and V in one layer:

    \[
    \text{MHA: }2\cdot40\cdot128\cdot2=20{,}480\text{ bytes}
    \]

    \[
    \text{GQA: }2\cdot8\cdot128\cdot2=4{,}096\text{ bytes}
    \]

    That is \(8/40=20\%\). Over 40 layers GQA needs 160 KiB per token, so a 16,384-token context takes 2.5 GiB; MHA would take 12.5 GiB of a 32 GB Mac's memory. Longer version: [43 §7](43-math-examples.md).

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

!!! example "On paper"

    One token, two hidden units: suppose \(a=W_gx=(0,2)\), \(b=W_ux=(3,1)\) and \(W_d=(1,1)\), so \(y\) is a scalar. With \(\sigma(2)\approx0.881\), \(g=\operatorname{SiLU}(a)=(0,\;1.762)\) and

    \[
    y=1\cdot(0\cdot3)+1\cdot(1.762\cdot1)=1.762
    \]

    Backward with \(\bar y=1\), so \(W_d^\top\bar y=(1,1)\); \(g'(0)=0.5\) and \(g'(2)=0.881+2\cdot0.881\cdot0.119=1.091\):

    \[
    \frac{\partial\mathcal{L}}{\partial a}=(3\cdot0.5,\;1\cdot1.091)=(1.5,\;1.091)
    \]

    \[
    \frac{\partial\mathcal{L}}{\partial b}=(0,\;1.762)
    \]

    The closed first gate outputs 0 yet still gets gradient 1.5, so it can learn to open. Finite-difference check on \(a_2\): with \(\sigma(2.1)\approx0.89090\) and \(\sigma(1.9)\approx0.86989\), \(\operatorname{SiLU}(2.1)=1.8709\) and \(\operatorname{SiLU}(1.9)=1.6528\), so \((1.8709-1.6528)/0.2=1.0905\), against \(g'(2)=1.0908\). Longer version: [43 §9](43-math-examples.md).

Do not confuse the scalar derivative \(g'(a)\) with the vector gradient: the upstream gradient \(W_d^\top\bar y\) must be multiplied elementwise by \(b\) and \(g'(a)\). The gate’s derivative has the extra \(a\sigma(a)(1-\sigma(a))\) term, which is why SiLU is smooth and avoids the dead-unit behaviour of ReLU. GeGLU replaces \(\operatorname{SiLU}\) with \(\operatorname{GELU}\):

\[
\operatorname{GELU}(a)=a\Phi(a),
\qquad
\Phi(a)=\frac12\left(1+\operatorname{erf}\left(\frac{a}{\sqrt2}\right)\right).
\]

Both have three matrices, so the parameter count is the same; the difference is the nonlinearity and its gradient.

!!! example "On paper"

    About 68.27% of a standard normal lies within \(\pm1\), so \(\operatorname{erf}(1/\sqrt2)=0.6827\) and

    \[
    \operatorname{GELU}(1)=\tfrac12\,(1+0.6827)=0.841
    \]

    \[
    \operatorname{GELU}(-1)=-\tfrac12\,(1-0.6827)=-0.159
    \]

    Compare \(\operatorname{SiLU}(1)=\sigma(1)=0.731\) and \(\operatorname{SiLU}(-1)=-0.269\): the same shape with slightly different values, so GeGLU and SwiGLU differ only in the gate's values and slope.

---

## 7. MoE routing and load balancing

For \(E\) experts, router logits \(r=W_rx\), and top-k set \(\mathcal{T}\):

\[
g_i=\frac{\exp(r_i)}{\sum_{j=1}^{E}\exp(r_j)},
\qquad
y=\sum_{i\in\mathcal{T}}\frac{g_i}{\sum_{j\in\mathcal{T}}g_j}E_i(x).
\]

The router gradient flows only through the selected experts and the renormalisation. If one expert is selected for most tokens, the others receive no gradient and collapse further.

!!! example "On paper"

    \(E=4\) experts, top-\(k=2\), router logits \(r=(1,0.5,0,-1)\). With \(e^{0.5}\approx1.6487\), the exponentials are \((2.7183,1.6487,1,0.3679)\), summing to 5.7349:

    \[
    g=(0.474,\;0.287,\;0.174,\;0.064)
    \]

    Experts 1 and 2 win. In the renormalised weights the full sum cancels, leaving only their own exponentials:

    \[
    \frac{(2.7183,\;1.6487)}{2.7183+1.6487}=(0.622,\;0.378)
    \]

    So \(y=0.622\,E_1(x)+0.378\,E_2(x)\). Experts 3 and 4 held 24% of the router probability but get no gradient from this token. Longer version: [43 §10](43-math-examples.md).

### 7.1 Auxiliary load-balancing loss

With \(f_i\) the fraction of tokens routed to expert \(i\) and \(P_i\) the mean router probability:

\[
\mathcal{L}_{\text{bal}}=\alpha E\sum_{i=1}^{E}f_iP_i.
\]

If routing is uniform, \(f_i=P_i=1/E\), so \(\mathcal{L}_{\text{bal}}=\alpha\). If routing collapses, \(f_iP_i\) is large for the dominant expert, increasing the loss. The gradient pushes probability mass toward underrepresented experts.

!!! example "On paper"

    Take \(E=4\), top-1 routing, and Qwen3-30B-A3B's coefficient \(\alpha=0.001\). In a batch of 8 tokens, 7 go to expert 1 and 1 to expert 2, so \(f=(0.875,0.125,0,0)\), and the mean router probabilities are \(P=(0.6,0.2,0.1,0.1)\):

    \[
    \sum_i f_iP_i=0.875\cdot0.6+0.125\cdot0.2=0.55
    \]

    \[
    \mathcal{L}_{\text{bal}}=0.001\cdot4\cdot0.55=0.0022
    \]

    That is 2.2 times the balanced value \(\alpha=0.001\). Treating \(f\) as fixed counts, \(\partial\mathcal{L}_{\text{bal}}/\partial P_i=\alpha Ef_i\) is 0.0035 for expert 1 and 0 for experts 3 and 4, so the gradient lowers expert 1's probability fastest.

Qwen3 uses a **global-batch** version so specialisation can vary across a batch while staying balanced globally. The exact implementation (capacity factor, token dropping, expert bias) is not fully public.

### 7.2 Active compute

For \(k\) active experts out of \(E\):

\[
N_{\text{active}}\approx N_{\text{attn}}+N_{\text{router}}
+\frac{k}{E}N_{\text{experts}}+N_{\text{embed}}.
\]

For Qwen3-30B-A3B: \(E=128\), \(k=8\), so only 6.25% of expert parameters run per token. That is why it stores 30.5B but generates at ~3.3B speed.

!!! example "On paper"

    Check both numbers against the Qwen3-30B-A3B config: 48 layers, \(d=2048\), 128 experts of width 768, 32 query and 4 KV heads of size 128, and an untied 151,936-token vocabulary. One expert in one layer has \(3\cdot2048\cdot768=4.72\)M weights, so all experts hold \(4.72\text{M}\cdot128\cdot48=29.0\)B, of which \(\tfrac{8}{128}\cdot29.0=1.81\)B run per token. Attention is 0.91B, embeddings plus LM head 0.62B, routers 0.01B:

    \[
    \begin{aligned}
    N_{\text{active}}&\approx0.91+0.01+1.81+0.62\\
    &=3.35\text{B}
    \end{aligned}
    \]

    Counting every expert instead gives \(0.91+0.01+29.0+0.62=30.5\)B stored.

---

## 8. Gated DeltaNet as online gradient descent

The delta rule updates an associative memory \(S\in\mathbb{R}^{d_v\times d_k}\), which maps a key to a value (the convention of the Gated DeltaNet paper), by correcting its prediction error for the current key/value:

\[
\mathcal{L}(S)=\frac12\|S k_t-v_t\|_2^2.
\]

The gradient is

\[
\nabla_S\mathcal{L}=(S k_t-v_t)k_t^\top.
\]

A gradient step with rate \(\beta_t\) gives

\[
S_t=S_{t-1}-\beta_t(S_{t-1} k_t-v_t)k_t^\top.
\]

Gated DeltaNet first decays the state by \(\alpha_t\):

\[
S'=\alpha_tS_{t-1},
\]

then corrects:

\[
S_t=S'-\beta_t(S' k_t-v_t)k_t^\top.
\]

Substituting \(S'\):

\[
\boxed{
S_t=\alpha_tS_{t-1}
+\beta_t\big(v_t-\alpha_tS_{t-1} k_t\big)k_t^\top
}
\]

and

\[
o_t=S_t\, q_t.
\]

This is exactly the recurrence in chapter 32. \(\alpha_t\) is a forget gate; \(\beta_t\) is the learning rate for the current association. In the actual GDN layer \(k_t\) is L2-normalised and \(\beta_t\in(0,1)\), which keeps the rank-one update bounded; without normalisation, a large \(\|k_t\|^2\) can make \(I-\beta_tk_tk_t^\top\) unstable.

!!! example "On paper"

    Take \(d_k=2\) and a scalar value (\(d_v=1\)), so \(S\) is just two numbers. The memory holds \(S_{t-1}=(2,0)\): key \((1,0)\) returns value 2. A unit key \(k_t=(0.6,0.8)\) arrives with \(v_t=1\), \(\alpha_t=0.5\), \(\beta_t=0.5\):

    \[
    \alpha_tS_{t-1}k_t=0.5\cdot(2\cdot0.6+0\cdot0.8)=0.6
    \]

    \[
    \begin{aligned}
    S_t&=(1,0)+0.5\,(1-0.6)\,(0.6,0.8)\\
    &=(1.12,\;0.16)
    \end{aligned}
    \]

    Read back with \(q_t=k_t\): \(o_t=S_tk_t=0.672+0.128=0.8\). The decayed memory predicted 0.6, the target was 1, and \(\beta_t=0.5\) moved the answer exactly halfway. The two-step route gives the same state: \(S'=(1,0)\), then \(S'-0.5\,(0.6-1)(0.6,0.8)=(1.12,0.16)\). Longer version: [43 §11](43-math-examples.md).

### 8.1 Why it is not just “linear attention”

Linear attention keeps only the write, with no decay and no correction:

\[
S_t=S_{t-1}+v_tk_t^\top,
\]

which never erases. (Setting \(\alpha_t=\beta_t=1\) in the boxed update does not give this; it gives the ungated delta rule, which still erases.) The delta rule subtracts the component of the state already aligned with \(k_t\), so a new value overwrites the old association instead of accumulating. Gating adds exponential forgetting. The combination improves in-context retrieval, which is exactly what pure linear attention lacks.

!!! example "On paper"

    Same memory \(S_{t-1}=(2,0)\), key \(k_t=(0.6,0.8)\), value \(v_t=1\). The old memory answers \(S_{t-1}k_t=1.2\) for this key. Linear attention only adds:

    \[
    S_t=(2,0)+1\cdot(0.6,0.8)=(2.6,\;0.8)
    \]

    It reads back \(2.6\cdot0.6+0.8\cdot0.8=2.2\): the old 1.2 and the new 1 pile up. The delta rule with \(\alpha_t=\beta_t=1\) writes only the error \(1-1.2=-0.2\):

    \[
    S_t=(2,0)-0.2(0.6,0.8)=(1.88,-0.16)
    \]

    It reads back \(1.128-0.128=1.0\): the new value has replaced the old one.

### 8.2 Chunkwise parallel form

The per-token transition is

\[
S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top.
\]

!!! example "On paper"

    The numbers from the §8 example again: \(S_{t-1}=(2,0)\) as a row, \(k_t=(0.6,0.8)\), \(v_t=1\), \(\alpha_t=\beta_t=0.5\). The matrix \(k_tk_t^\top\) has entries 0.36, 0.48, 0.48, 0.64:

    \[
    I-0.5\,k_tk_t^\top=\begin{pmatrix}0.82&-0.24\\-0.24&0.68\end{pmatrix}
    \]

    \[
    \begin{aligned}
    &0.5\,(2,0)\begin{pmatrix}0.82&-0.24\\-0.24&0.68\end{pmatrix}\\
    &=(0.82,\;-0.24)
    \end{aligned}
    \]

    Adding \(\beta_tv_tk_t^\top=(0.3,0.4)\) gives \((1.12,0.16)\), the same state as the boxed form.

The matrix \((I-\beta_tk_tk_t^\top)\) is a rank-one update of the identity. It is a Householder reflection only when \(k_t\) is unit norm and \(\beta_t=2\); in GDN it is a damped rank-one update. A product of such updates over a chunk can be written as

\[
\prod_{t=1}^{C}(I-\beta_tk_tk_t^\top)=I-K^\top W K,
\]

where \(K\) stacks the chunk’s keys and \(W\) is a triangular matrix computed from the \(\beta\)’s and the dot products between the chunk’s keys (the WY representation). This turns the sequential recurrence into matrix multiplications over a chunk, which is why GDN can train in parallel despite being recurrent at inference.

!!! example "On paper"

    A chunk of two keys, \(k_1=(1,0)\) and \(k_2=(0.6,0.8)\), both with \(\beta=0.5\). Multiplying the two factors in the order the recurrence applies them:

    \[
    \begin{aligned}
    &\begin{pmatrix}0.5&0\\0&1\end{pmatrix}\begin{pmatrix}0.82&-0.24\\-0.24&0.68\end{pmatrix}\\
    &=\begin{pmatrix}0.41&-0.12\\-0.24&0.68\end{pmatrix}
    \end{aligned}
    \]

    Expanding instead gives \(I-\beta_1k_1k_1^\top-\beta_2k_2k_2^\top+\beta_1\beta_2(k_1\cdot k_2)\,k_1k_2^\top\): the identity minus a combination of the \(k_ik_j^\top\), whose coefficients 0.5, 0.5 and \(-0.25\cdot0.6=-0.15\) are the entries of \(W\). Numerically:

    \[
    \begin{aligned}
    &I-0.5\,k_1k_1^\top-0.5\,k_2k_2^\top+0.15\,k_1k_2^\top\\
    &=\begin{pmatrix}0.32&-0.24\\-0.24&0.68\end{pmatrix}+\begin{pmatrix}0.09&0.12\\0&0\end{pmatrix}
    \end{aligned}
    \]

    This is the same matrix. With orthogonal keys the cross coefficient would vanish and \(W\) would be diagonal.

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

!!! example "On paper"

    Four weights from one row, \(w=(0.46,-1.4,0.13,0.95)\), stored in 4 bits: \(2^{k-1}-1=7\), so \(s=1.4/7=0.2\).

    | \(w\) | \(w/s\) | \(q\) | \(\hat w\) | \(w-\hat w\) |
    |---:|---:|---:|---:|---:|
    | 0.46 | 2.3 | 2 | 0.4 | 0.06 |
    | −1.4 | −7 | −7 | −1.4 | 0 |
    | 0.13 | 0.65 | 1 | 0.2 | −0.07 |
    | 0.95 | 4.75 | 5 | 1.0 | −0.05 |

    Every error is at most \(s/2=0.1\). The row is stored as four 4-bit integers plus one scale. Longer version: [43 §16](43-math-examples.md).

For a block of \(B_c\) weights, using a per-block scale \(s_b\) reduces the maximum error because each block’s range is smaller:

\[
s_b=\frac{\max_{i\in b}|w_i|}{2^{k-1}-1}.
\]

This is why Q4_0 (blockwise) is better than a single per-tensor scale.

!!! example "On paper"

    Give the same row a second block of small weights, \(w=(0.05,-0.07,0.02,0.03)\). With the shared scale \(s=0.2\), every \(|w/s|\) is below 0.5, so \(q=(0,0,0,0)\) and the block is erased, with errors up to 0.07. Its own scale is

    \[
    s_b=\frac{0.07}{7}=0.01
    \]

    so \(w/s_b=(5,-7,2,3)\): exact here, and never off by more than \(s_b/2=0.005\), twenty times tighter than 0.1. The price is one extra scale per block; Q4_0 uses blocks of 32 weights.

### 11.1 QAT straight-through estimator

Quantization is not differentiable at the rounding step. QAT uses

\[
\frac{\partial\hat w}{\partial w}\approx 1
\]

in the backward pass (straight-through estimator), so the model learns weights that are robust to the quantization error. Gemma 4 uses QAT for its int2/int4 mobile and Q4_0 checkpoints.

!!! example "On paper"

    Take \(w=0.46\) from the row above (\(s=0.2\), \(\hat w=0.4\)), an upstream gradient \(\partial\mathcal{L}/\partial\hat w=2\) and a learning rate of 0.1. The straight-through estimator passes the gradient through unchanged:

    \[
    w\leftarrow0.46-0.1\cdot2=0.26
    \]

    \[
    \hat w=0.2\cdot\operatorname{round}(1.3)=0.2
    \]

    The full-precision weight moved smoothly, and \(\hat w\) dropped one step once \(w\) crossed the rounding boundary at 0.3. The true derivative of round is 0 almost everywhere, so without the STE \(w\) would never move.

---

## 12. Scaling-law fit

A common form is

\[
L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}.
\]

- \(E\) is the irreducible loss.
- \(A/N^\alpha\) is the finite-model term.
- \(B/D^\beta\) is the finite-data term.

!!! example "On paper"

    Use toy constants rounded from the Chinchilla fit (\(E\approx1.69\), \(A\approx406\), \(B\approx411\)): \(E=1.7\), \(A=B=400\), and \(\alpha=\beta=1/3\) so the roots are easy. For a 1B model on 64B tokens, \(N^{1/3}=1{,}000\) and \(D^{1/3}=4{,}000\):

    \[
    L=1.7+\frac{400}{1{,}000}+\frac{400}{4{,}000}=2.2
    \]

    An 8× larger model (\(N^{1/3}=2{,}000\)) gives \(1.7+0.2+0.1=2.0\); 8× more data instead (\(D^{1/3}=8{,}000\)) gives \(1.7+0.4+0.05=2.15\). Neither can get below \(E=1.7\).

To fit:

1. train several small models with different \(N\) and \(D\);
2. record final loss;
3. fit \(E,A,B,\alpha,\beta\) by least squares in log space;
4. extrapolate cautiously.

The fit is only valid in the explored range. It does not tell you that 36T tokens will beat 10T for a specific model; it tells you the trend under the same data distribution and recipe.

Chinchilla’s compute-optimal allocation is approximately

\[
N^{*} \propto C^{0.5},\qquad D^{*}\propto C^{0.5},
\]

with \(D\approx20N\). Production models often overtrain because inference cost dominates.

!!! example "On paper"

    Combine \(D\approx20N\) with the \(6ND\) estimate from §13: \(C\approx6N\cdot20N=120N^2\). For a budget of \(C=1.2\times10^{22}\) FLOPs:

    \[
    N=\sqrt{C/120}=\sqrt{10^{20}}=10^{10}
    \]

    That is 10B parameters trained on \(20N=200\)B tokens. Four times the compute doubles both (\(\sqrt4=2\)): 20B parameters on 400B tokens. Qwen3-14B (14.8B parameters) saw 36T tokens, about 2,400 per parameter or roughly 120 times the Chinchilla ratio. It is overtrained on purpose, because a smaller model is cheaper to serve.

---

## 13. Exact FLOPs

The \(6ND\) estimate counts matrix multiplications but not attention scores. A more complete forward+backward estimate per layer is

\[
C_{\text{layer}}\approx
\underbrace{6\,(2dHd_h+2dGd_h)\,T}_{\text{projections}}
+\underbrace{12\,H\,d_h\,T^2}_{\text{attention scores/values}}
+\underbrace{18dd_{\text{ff}}T}_{\text{MLP}}
\]

for a dense block (the exact constants depend on whether biases/norms are counted). With full-width K and V (\(Hd_h=Gd_h=d\)) the projection term is the familiar \(24d^2T\); with Qwen3-14B’s grouped-query attention it is \(14.4d^2T\). The attention term uses the total query width \(H d_h\), not \(d\); they are equal for Qwen3-14B but not for Gemma’s global layers. The attention term is why long-context training is more expensive than \(6ND\) suggests.

For a 14B model at 32K, attention is a significant fraction of compute; at 2K, it is small. Always measure tokens/sec rather than trusting a formula.

!!! example "On paper"

    For Qwen3-14B, \(Hd_h=40\cdot128=5{,}120=d\), \(Gd_h=8\cdot128=1{,}024\) and \(d_{\text{ff}}=17{,}408\). Divide the attention term by the other two; \(T\) and one \(d\) cancel:

    \[
    \frac{12\,dT^2}{(12d^2+12dGd_h+18dd_{\text{ff}})T}
    \]

    \[
    =\frac{12T}{61{,}440+12{,}288+313{,}344}=\frac{T}{32{,}256}
    \]

    | \(T\) | attention / rest | attention share |
    |---:|---:|---:|
    | 2,048 | 0.063 | 6% |
    | 32,768 | 1.02 | 50% |

    At 2K the scores add a small correction to the projection and MLP cost; at 32K they are half the layer.

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

!!! example "On paper"

    First step for one LoRA weight: \(g_1=-0.2\), \(\beta_1=0.9\), \(\beta_2=0.999\), \(m_0=v_0=0\).

    \[
    m_1=0.1\cdot(-0.2)=-0.02
    \]

    \[
    v_1=0.001\cdot0.04=0.00004
    \]

    Both are far below the gradient \(-0.2\) and its square \(0.04\). Dividing by \(1-0.9=0.1\) and \(1-0.999=0.001\) gives \(\hat m_1=-0.2\) and \(\hat v_1=0.04\): exactly the gradient and its square.

The update is

\[
\theta_t=\theta_{t-1}-\eta_t
\left(
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
+\lambda\theta_{t-1}
\right).
\]

The \(\sqrt{\hat v_t}\) normalizes each parameter by its gradient scale, which is why AdamW is robust to heterogeneous gradients. Weight decay is decoupled from the adaptive step, unlike classic Adam.

!!! example "On paper"

    Continue with \(\theta_0=0.5\), \(\eta=0.01\), \(\lambda=0.1\) and \(\epsilon\) negligible:

    \[
    \frac{\hat m_1}{\sqrt{\hat v_1}}=\frac{-0.2}{0.2}=-1
    \]

    \[
    \theta_1=0.5-0.01\,(-1+0.1\cdot0.5)=0.5095
    \]

    The first step has size \(\eta\) whatever the gradient's scale. Without the corrections, \(m_1/\sqrt{v_1}=-0.02/0.00632=-3.16\): a step more than three times too large. Longer version: [43 §17](43-math-examples.md).

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
