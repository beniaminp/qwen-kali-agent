# 32 — The mathematics inside Qwen and Gemma 4

*Last verified: 27 October 2026. Equations are standard transformer/linear-attention math; model-specific numbers come from the Qwen3, Qwen2.5-Coder, Qwen3.5 and Gemma 4 technical reports cited at the end.*

This chapter is the common mathematical foundation for the next four. It explains what a decoder-only language model computes, why Qwen and Gemma choose different pieces, and how to count parameters, FLOPs, memory and KV cache.

It is deliberately equation-heavy. There are three companion pages:

- **[37 — Architecture anatomy](37-architecture.md)** maps every equation to actual tensor shapes and `config.json` keys.
- **[38 — Derivations and proofs](38-derivations.md)** proves the softmax gradient, the RoPE relative property, the RMSNorm Jacobian, the GDN recurrence and the quantization error bound.
- **[43 — Worked examples](43-math-examples.md)** runs every formula on a toy model with real numbers, so you can check the arithmetic yourself.
- **[35 — From zero](35-from-zero.md)** and **[36 — Reproducing 12B–14B](36-reproducing-12b-14b.md)** turn the math into a training plan.

If you only want the practical fine-tuning recipe, skip to [08 — Fine-tuning](08-finetuning.md).

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| \(T\) | sequence length |
| \(V\) | vocabulary size |
| \(d\) | hidden size (model dimension) |
| \(L\) | number of transformer layers |
| \(H\) | number of query heads |
| \(G\) | number of key/value heads (GQA) |
| \(d_h\) | head dimension |
| \(d_{\text{ff}}\) | FFN intermediate size |
| \(N\) | parameter count |
| \(D\) | training tokens |
| \(\theta\) | all trainable parameters |
| \(x_t\) | token at position \(t\) |
| \(z_t\) | logits at position \(t\) |
| \(\eta\) | learning rate |
| \(\lambda\) | weight decay |
| \(b\) | bytes per value (2 for bf16, 1 for int8) |

A bold matrix is written \(W\); a vector is \(x\); \(\top\) is transpose; \(\odot\) is elementwise product; \(\sigma\) is the logistic sigmoid unless stated otherwise.

---

## 1. What “creating a model” means

A model like Qwen3-14B or Gemma 4 12B is the result of five coupled choices:

1. **Tokenization** — how text, code and multimodal inputs become integer sequences.
2. **Architecture** — what function \(f_\theta\) maps a token sequence to next-token probabilities.
3. **Objective** — what loss is minimised during pretraining.
4. **Data and optimisation** — what tokens are seen, in what order, with what schedule and precision.
5. **Post-training** — SFT, preference optimisation, RL and distillation that turn a base model into a usable assistant.

“From scratch” means learning \(\theta\) from random initialisation on a large corpus. Fine-tuning (including LoRA) starts from an existing \(\theta\) and is a different, much cheaper problem.

---

## 2. Tokenization

A tokenizer is a function

\[
\tau: \text{string} \rightarrow (x_1,\dots,x_T), \qquad x_t \in \{0,\dots,V-1\}.
\]

Qwen3 uses **byte-level BPE** with \(V=151{,}669\). Qwen2.5-Coder uses byte-level BPE with \(V=151{,}646\). Gemma 4 uses a **SentencePiece** tokenizer with \(V=262{,}144\), split digits, preserved whitespace and byte-level fallback.

The public Qwen3 report quotes 151,669 and Qwen2.5-Coder 151,646; the Hugging Face checkpoints pad the embedding to 151,936 and 152,064 respectively for tensor-parallel divisibility. The padding rows are unused but still cost memory.

### 2.1 BPE objective

Byte-pair encoding starts from bytes and repeatedly merges the most frequent adjacent pair:

\[
(a,b)^{*} = \arg\max_{(a,b)} \operatorname{count}(a,b),
\qquad
\text{merge }(a,b)\rightarrow c.
\]

!!! example "On paper"

    Take the 7-byte string `ababcab`. Its adjacent pairs are ab, ba, ab, bc, ca, ab, so \(\operatorname{count}(\text{a},\text{b})=3\) beats every other pair (1 each), and the first merge is \(\text{ab}\rightarrow\text{Z}\):

    \[
    \text{a b a b c a b}\ \longrightarrow\ \text{Z Z c Z}
    \]

    One merge turns 7 tokens into 4 and grows the vocabulary from 256 to \(256+1=257\). The next round counts the new pairs (ZZ, Zc, cZ) and repeats.

After \(M\) merges, the vocabulary is \(256+M\) (plus special tokens). The compression ratio is

\[
\rho = \frac{\text{bytes}}{\text{tokens}}.
\]

!!! example "On paper"

    The same 7 bytes were 7 tokens before the merge and are 4 tokens after it:

    \[
    \rho = \frac{7\ \text{bytes}}{4\ \text{tokens}} = 1.75
    \]

    Before the merge \(\rho=7/7=1\). At \(\rho=4\), a 2,000-byte Python file would cost 500 tokens of compute instead of 2,000.

A higher \(\rho\) means cheaper training and inference per character, but a larger embedding matrix \(V d\) and a larger output softmax.

### 2.2 Why vocabulary size matters

The input embedding and output head cost

\[
P_{\text{emb}} = V d \quad\text{(tied)},\qquad
P_{\text{emb}} = 2 V d \quad\text{(untied)}.
\]

For Qwen3-14B, \(V=151{,}936\), \(d=5{,}120\), untied: \(2Vd\approx1.56\text{B}\) parameters — about 10.5% of the model. For Gemma 4 12B, \(V=262{,}144\), \(d=3{,}840\), tied: \(Vd\approx1.01\text{B}\).

The softmax denominator at every position costs \(V\) operations:

\[
p(x_{t+1}=j\mid x_{\le t}) =
\frac{\exp(z_j)}{\sum_{k=1}^{V}\exp(z_k)}.
\]

!!! example "On paper"

    After `return`, suppose a toy vocabulary of \(V=4\) tokens (`self`, `None`, `x`, `0`) gets logits \(z=(2,0,0,0)\). With \(e^2\approx7.389\) and \(e^0=1\):

    \[
    p(\mathtt{self}) = \frac{7.389}{7.389+1+1+1} \approx 0.711
    \]

    Each of the other three tokens gets \(1/10.389\approx0.096\). The denominator needed all \(V\) exponentials: 4 here, 151,936 for Qwen3-14B at every position.

> **Worked example:** [43 §1–3](43-math-examples.md) computes the softmax, the CE and the gradient for \(z=(2,1,0,-1)\).

That is why a 262k vocabulary is expensive at inference even when the model is only 12B.

### 2.3 Special tokens

Chat and tool use need control tokens: BOS, EOS, turn separators, thinking delimiters, tool-call delimiters. Gemma 4 uses `<|turn>`, `<turn|>`, `<|tool>`, `<tool|>`, `<|tool_call>`, `<tool_call|>`, `<|channel>`, `<channel|>`. Qwen3 uses ChatML-style `<|im_start|>`, `<|im_end|>`, and `<tool_call>`. These tokens are part of the model’s contract; changing them after pretraining is a fine-tuning problem, not a formatting detail.

---

## 3. The autoregressive objective

A decoder-only model factorises the joint probability of a sequence:

\[
p_\theta(x_1,\dots,x_T) = \prod_{t=1}^{T} p_\theta(x_t\mid x_{<t}).
\]

!!! example "On paper"

    Score the three tokens `return`, `self`, `.` and suppose the model gives \(p(\mathtt{return})=0.5\), \(p(\mathtt{self}\mid\mathtt{return})=0.8\) and \(p(\mathtt{.}\mid\mathtt{return},\mathtt{self})=0.25\):

    \[
    p_\theta(x_1,x_2,x_3) = 0.5\cdot0.8\cdot0.25 = 0.1
    \]

    Every factor is at most 1, so each extra token can only keep or shrink the sequence probability; that is why training works with logs.

The pretraining loss is the mean negative log-likelihood:

\[
\mathcal{L}_{\text{NLL}}(\theta)
= -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
\]

!!! example "On paper"

    The same three tokens, with natural logs (\(\ln10\approx2.303\)):

    \[
    \begin{aligned}
    \mathcal{L}_{\text{NLL}} &= -\tfrac13(\ln0.5+\ln0.8+\ln0.25)\\
    &= -\tfrac13\ln0.1 = \tfrac{2.303}{3} \approx 0.768
    \end{aligned}
    \]

    The per-token terms are 0.693, 0.223 and 1.386, so the hard third token dominates. Perplexity is \(e^{0.768}\approx2.15\): on average the model is as unsure as a uniform pick among about two tokens.

If the model outputs logits \(z_t\in\mathbb{R}^{V}\), then

\[
p_\theta(x_t\mid x_{<t}) = \operatorname{softmax}(z_t)_{x_t},
\]

and the per-token loss is the cross-entropy

\[
\ell_t = -z_{t,x_t} + \log\sum_{j=1}^{V}\exp(z_{t,j}).
\]

!!! example "On paper"

    Back to the softmax example in §2.2: \(z=(2,0,0,0)\), the target is `self` (logit 2), and \(\ln10.389\approx2.341\):

    \[
    \begin{aligned}
    \ell_t &= -2 + \ln(e^2+1+1+1)\\
    &= -2 + 2.341 = 0.341
    \end{aligned}
    \]

    That is \(-\ln0.711\), the same probability as before. Had the target been `None` (logit 0), the loss would be \(0+2.341=2.341\).

Perplexity is \(\exp(\mathcal{L}_{\text{NLL}})\). It is comparable only across models with the same tokenizer; a 262k SentencePiece model and a 151k BPE model can have very different per-token losses for the same text quality.

### 3.1 Masked / completion loss

For SFT, the prompt tokens are often masked:

\[
\mathcal{L}_{\text{SFT}} =
-\frac{1}{|\mathcal{C}|}\sum_{t\in\mathcal{C}}
\log p_\theta(x_t\mid x_{<t}),
\]

where \(\mathcal{C}\) is the completion span. This is exactly what `mask_prompt: true` does in MLX-LM: only the final assistant turn contributes. It does not make a multi-turn example train every turn; split trajectories if you need that.

!!! example "On paper"

    A 5-token example: 3 prompt tokens, then a 2-token answer whose tokens get probabilities 0.5 and 0.25. Only the answer is in \(\mathcal{C}\), so \(|\mathcal{C}|=2\):

    \[
    \begin{aligned}
    \mathcal{L}_{\text{SFT}} &= -\tfrac12(\ln0.5+\ln0.25)\\
    &= \tfrac32\ln2 \approx 1.04
    \end{aligned}
    \]

    The prompt tokens never enter, however badly the model predicts them. Dividing the same sum by all 5 tokens would shrink the answer's loss to \(2.079/5\approx0.42\).

---

## 4. The transformer block

A modern decoder block is a residual stack:

\[
\begin{aligned}
u &= x + \operatorname{Attn}(\operatorname{Norm}_1(x)),\\
y &= u + \operatorname{MLP}(\operatorname{Norm}_2(u)).
\end{aligned}
\]

!!! example "On paper"

    Take \(d=2\) and \(x=(2,-1)\). Suppose the attention branch returns \((0.5,\,0.5)\) and the MLP branch returns \((-1,\,1)\):

    \[
    \begin{aligned}
    u &= (2,-1)+(0.5,\,0.5) = (2.5,\,-0.5)\\
    y &= (2.5,-0.5)+(-1,\,1) = (1.5,\,0.5)
    \end{aligned}
    \]

    Each branch only adds a correction to the stream. If both returned zeros, \(y=x\), which is why a deep pre-norm stack starts close to the identity.

Qwen3 and Qwen2.5 use **pre-norm** with RMSNorm. Gemma 4 uses **pre-norm and post-norm** with RMSNorm and QK-norm. Pre-norm keeps the residual path close to identity, which stabilises deep training; post-norm can improve final representation quality at the cost of stability.

![A pre-norm decoder block](assets/diagrams/decoder-block.svg)

The residual stream is the shared highway: every attention and MLP block reads from it and writes back to it. That is why all blocks have the same shape \(B\times T\times d\), and why LoRA can be inserted into projections without changing the interface. [37 — Architecture anatomy](37-architecture.md) traces the exact tensors for Qwen3-14B and Gemma 4 12B.

---

## 5. RMSNorm

LayerNorm is

\[
\operatorname{LN}(x) = \gamma \odot
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},
\quad
\mu=\frac{1}{d}\sum_i x_i,
\quad
\sigma^2=\frac{1}{d}\sum_i (x_i-\mu)^2.
\]

!!! example "On paper"

    A 4-wide residual vector \(x=(1,3,3,9)\), with \(\gamma=(1,1,1,1)\) and \(\epsilon=0\):

    \[
    \begin{aligned}
    \mu &= \tfrac14(1+3+3+9) = 4\\
    \sigma^2 &= \tfrac14(9+1+1+25) = 9\\
    \operatorname{LN}(x) &= \tfrac13(-3,-1,-1,5)\\
    &= (-1,\,-0.333,\,-0.333,\,1.667)
    \end{aligned}
    \]

    The output has mean 0 and variance 1. Shifting the input to \(x+10\) or scaling it to \(2x\) gives exactly the same result.

RMSNorm drops the mean subtraction:

\[
\operatorname{RMSNorm}(x) = \gamma \odot
\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}.
\]

!!! example "On paper"

    The same \(x=(1,3,3,9)\), \(\gamma=1\), \(\epsilon=0\):

    \[
    \begin{aligned}
    \tfrac14(1+9+9+81) &= 25\\
    \operatorname{RMSNorm}(x) &= \tfrac15(1,3,3,9)\\
    &= (0.2,\,0.6,\,0.6,\,1.8)
    \end{aligned}
    \]

    Unlike LayerNorm, nothing is subtracted: every entry keeps its sign and its ratio to the others, and only the overall scale changes (the result has root-mean-square 1).

It has one vector \(\gamma\in\mathbb{R}^d\) and no bias. [43 §4](43-math-examples.md) runs it on \(x=(1,2,3,4)\). Why it works: the residual stream’s scale is what matters, not its mean; removing the mean subtraction saves a reduction and a broadcast, and empirically trains stably. Qwen3, Qwen2.5 and Gemma 4 all use RMSNorm.

**QK-norm.** Qwen3 introduces QK-Norm to stabilise attention: normalise \(q\) and \(k\) per head before the dot product,

\[
\tilde q = \operatorname{RMSNorm}(q), \qquad
\tilde k = \operatorname{RMSNorm}(k).
\]

!!! example "On paper"

    One head, \(d_h=4\), \(\gamma=1\), \(\epsilon=0\). A large query \(q=(60,0,0,80)\) and key \(k=(30,0,0,40)\) give the logit \(q\cdot k/\sqrt4=5000/2=2500\), a one-hot softmax. Their RMS values are \(\sqrt{(3600+6400)/4}=50\) and 25:

    \[
    \begin{aligned}
    \tilde q &= q/50 = (1.2,\,0,\,0,\,1.6)\\
    \tilde k &= k/25 = (1.2,\,0,\,0,\,1.6)\\
    \tilde q\cdot\tilde k/\sqrt4 &= 4/2 = 2
    \end{aligned}
    \]

    After QK-norm each vector has length \(\sqrt{d_h}=2\), so with \(\gamma=1\) this logit can never exceed \(\sqrt{d_h}=2\), however large \(q\) and \(k\) grow.

Gemma 4 also uses QKNorm. This bounds the attention logits and reduces loss spikes at large learning rates.

---

## 6. Scaled dot-product attention

For one head with query \(q\in\mathbb{R}^{d_k}\), keys \(K\in\mathbb{R}^{T\times d_k}\), values \(V\in\mathbb{R}^{T\times d_v}\):

\[
\operatorname{Attn}(q,K,V) =
\operatorname{softmax}\left(\frac{qK^\top}{\sqrt{d_k}}\right)V.
\]

!!! example "On paper"

    One query, \(d_k=4\) (so \(\sqrt{d_k}=2\)), two earlier tokens: \(q=(1,1,0,0)\), keys \(k_1=(1,1,0,0)\) and \(k_2=(0,0,1,1)\), values \(v_1=(10,0)\) and \(v_2=(0,10)\). With \(e\approx2.718\):

    \[
    \begin{aligned}
    qK^\top/\sqrt{d_k} &= (2,\,0)/2 = (1,\,0)\\
    \operatorname{softmax}(1,0) &= \big(\tfrac{e}{e+1},\,\tfrac{1}{e+1}\big)\\
    &= (0.731,\,0.269)\\
    \text{output} &= (7.31,\,2.69)
    \end{aligned}
    \]

    The query mostly copies the value of the key it matches, but still blends in 27% of the other one.

For a whole sequence, with \(Q=XW_Q\), \(K=XW_K\), \(V=XW_V\):

\[
A = \operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right),
\qquad
O = AV,
\]

where the causal mask is

\[
M_{ij} =
\begin{cases}
0, & j\le i,\\
-\infty, & j>i.
\end{cases}
\]

!!! example "On paper"

    Two tokens with scaled scores \(QK^\top/\sqrt{d_k}=\begin{pmatrix}1&5\\2&2\end{pmatrix}\) and values \(V=\begin{pmatrix}4&0\\0&4\end{pmatrix}\). The mask puts \(-\infty\) above the diagonal, and the softmax runs row by row:

    \[
    A = \operatorname{softmax}\begin{pmatrix}1&-\infty\\2&2\end{pmatrix} = \begin{pmatrix}1&0\\0.5&0.5\end{pmatrix}
    \]

    \[
    O = AV = \begin{pmatrix}4&0\\2&2\end{pmatrix}
    \]

    Token 1 ignores token 2 even though that score (5) is the largest, because \(e^{-\infty}=0\). Token 2 sees both equally and averages their values.

The \(\sqrt{d_k}\) scaling keeps the logits at unit variance if \(q,k\) have unit variance: \(\operatorname{Var}(q\cdot k)=d_k\). [43 §5](43-math-examples.md) runs one query against three keys, then the causal mask and a \(2\times2\) backward pass.

### 6.1 Multi-head attention

With \(H\) heads and head dimension \(d_h=d/H\):

\[
\operatorname{MultiHead}(X) =
\operatorname{Concat}(\text{head}_1,\dots,\text{head}_H)W_O,
\]

\[
\text{head}_h = \operatorname{Attn}(XW_Q^h, XW_K^h, XW_V^h).
\]

!!! example "On paper"

    Qwen3-14B splits \(d=5120\) into \(H=40\) heads, so \(d_h=5120/40=128\). On paper take \(d=2\), \(H=2\), \(d_h=1\): for one token the two heads output 3 and \(-1\), and \(W_O=\begin{pmatrix}1&1\\0&2\end{pmatrix}\):

    \[
    \begin{aligned}
    (3,\,-1)\,W_O &= (3\cdot1-1\cdot0,\;3\cdot1-1\cdot2)\\
    &= (3,\,1)
    \end{aligned}
    \]

    Concat just places the head outputs side by side; \(W_O\) mixes them, so each output coordinate can draw on every head.

Each head can specialise: induction, syntax, copying, long-range retrieval.

### 6.2 Grouped-query attention (GQA)

GQA uses \(H\) query heads but only \(G\) key/value heads, with \(G<H\):

\[
Q\in\mathbb{R}^{H\times d_h},\quad
K,V\in\mathbb{R}^{G\times d_h},\quad
\text{each KV head is shared by }H/G\text{ query heads}.
\]

Qwen3-14B: \(H=40\), \(G=8\), ratio 5. Qwen2.5-Coder-14B: \(H=40\), \(G=8\), ratio 5. Gemma 4 12B: \(H=16\), \(G=8\), ratio 2.

GQA reduces the KV cache and memory bandwidth with a small quality cost. Multi-query attention is the extreme \(G=1\); multi-head attention is \(G=H\). [43 §6–7](43-math-examples.md) compares the parameter counts and computes the Gemma hybrid cache.

![Multi-head vs grouped-query attention](assets/diagrams/gqa.svg)

### 6.3 KV cache

At generation step \(t\), a transformer caches past keys and values instead of recomputing them. For \(L\) layers, \(G\) KV heads, head dimension \(d_h\), sequence length \(T\), and \(b\) bytes per value:

\[
\text{KV bytes} \approx 2 \cdot L \cdot G \cdot d_h \cdot T \cdot b.
\]

The factor 2 is for K and V. For Qwen3-14B at \(T=32{,}768\), bf16 (\(b=2\)):

\[
2\cdot40\cdot8\cdot128\cdot32{,}768\cdot2
\approx 5.37\ \text{GB}.
\]

For Gemma 4 12B, do **not** apply that formula to all 48 layers. The architecture is hybrid:

- 40 local layers: \(G=8\), \(d_h=256\), cache bounded by \(W=1024\);
- 8 global layers: \(G=1\), \(d_h=512\), full context, and \(K=V\).

At \(T=32{,}768\):

\[
\underbrace{2\cdot40\cdot8\cdot256\cdot1024\cdot2}_{335.5\ \text{MB local}}
+
\underbrace{8\cdot1\cdot512\cdot32{,}768\cdot2}_{268.4\ \text{MB global, K only}}
\approx 604\ \text{MB}.
\]

At 256K the local term is unchanged and the global term is 2.15 GB, so the cache is about **2.48 GB**. The naive “48 layers, \(G=8\), \(d_h=256\), full context” calculation gives 12.9 GB; it is a useful upper bound, not the real cache. The architecture gets its long context from the sliding window and the single global KV head, not from a huge cache.

**Rule:** long-context serving is a KV-cache problem, not just a weight problem. GQA, K=V, sliding windows, p-RoPE and quantized KV are all cache optimisations.

---

## 7. Rotary position embeddings (RoPE)

RoPE rotates pairs of dimensions by an angle proportional to position. For dimension pair \(i\) and position \(t\):

\[
\theta_{t,i} = t\,\omega_i,
\qquad
\omega_i = \text{base}^{-2i/d_h},
\]

\[
\begin{pmatrix} q'_{2i}\\ q'_{2i+1}\end{pmatrix}
=
\begin{pmatrix}
\cos\theta_{t,i} & -\sin\theta_{t,i}\\
\sin\theta_{t,i} & \cos\theta_{t,i}
\end{pmatrix}
\begin{pmatrix} q_{2i}\\ q_{2i+1}\end{pmatrix}.
\]

!!! example "On paper"

    Take \(d_h=8\), base \(10{,}000\), position \(t=5\) and pair \(i=1\), i.e. \((q_2,q_3)=(1,1)\). Use \(\cos0.5\approx0.8776\) and \(\sin0.5\approx0.4794\):

    \[
    \begin{aligned}
    \omega_1 &= 10{,}000^{-2/8} = 0.1\\
    \theta_{5,1} &= 5\cdot0.1 = 0.5\\
    q'_2 &= 0.8776-0.4794 = 0.3982\\
    q'_3 &= 0.4794+0.8776 = 1.3570
    \end{aligned}
    \]

    The pair turns by half a radian but keeps its length \(\sqrt2\approx1.414\). At the same position the other pairs turn by 5, 0.05 and 0.005 radians: the first pairs spin fast, the last ones slowly.

The key property is relative:

\[
R_t^\top R_s = R_{s-t}.
\]

!!! example "On paper"

    Pretend one pair turns a quarter turn per position (\(\omega=\pi/2\)), so every cos and sin is 0 or \(\pm1\); the property holds for any \(\omega\). One quarter turn maps

    \[
    R_1\begin{pmatrix}x\\y\end{pmatrix} = \begin{pmatrix}-y\\x\end{pmatrix}
    \]

    With \(q=(1,0)\) and \(k=(2,1)\):

    | \(t\) | \(s\) | \(R_tq\) | \(R_sk\) | \(q_t^\top k_s\) |
    |---:|---:|---|---|---:|
    | 1 | 0 | \((0,1)\) | \((2,1)\) | 1 |
    | 3 | 2 | \((0,-1)\) | \((-2,-1)\) | 1 |
    | 2 | 0 | \((-1,0)\) | \((2,1)\) | \(-2\) |
    | 4 | 2 | \((1,0)\) | \((-2,-1)\) | \(-2\) |

    Same gap, same score, wherever the pair sits: a gap of 1 always gives 1 and a gap of 2 always gives \(-2\).

So the dot product \(q_t^\top k_s\) depends on \(t-s\), not absolute position. RoPE has no learned position table, extrapolates better than absolute embeddings, and is the default in Qwen3, Qwen2.5, Qwen3.5 and Gemma 4 local layers. [43 §8](43-math-examples.md) rotates two actual vectors.

### 7.1 Base frequency and long context

The base controls the wavelength:

\[
\omega_i = \text{base}^{-2i/d_h}.
\]

!!! example "On paper"

    Qwen3 has \(d_h=128\). Take the middle pair \(i=32\), where \(2i/d_h=1/2\):

    \[
    \begin{aligned}
    \text{base }10^4:\ \omega_{32} &= (10^4)^{-1/2} = 0.01\\
    \text{base }10^6:\ \omega_{32} &= (10^6)^{-1/2} = 0.001
    \end{aligned}
    \]

    One full turn takes \(2\pi/\omega\) positions: about 628 tokens with base \(10^4\) and 6,283 with base \(10^6\). After ABF the same pair needs ten times as many tokens to wrap around.

A larger base (e.g. \(10^6\)) gives longer wavelengths and better long-context extrapolation. Qwen3 raises the base from \(10^4\) to \(10^6\) using **ABF** (adjusted base frequency) in the long-context stage, then applies **YaRN** and **Dual Chunk Attention** to reach 4× the trained length (32K → 128K).

### 7.2 p-RoPE

Gemma 4 uses **p-RoPE** on global attention layers: only a fraction \(p\) of the head dimensions are rotated, and the rest pass through. With \(p=0.25\),

\[
\tilde q = \big[R_{\theta}(q_{1:pd_h}),\; q_{pd_h+1:d_h}\big].
\]

!!! example "On paper"

    Gemma 4 global layers have \(d_h=512\) and \(p=0.25\):

    \[
    p\,d_h = 0.25\cdot512 = 128
    \]

    Dimensions 1–128 (64 pairs) are rotated and the other 384 pass through untouched. On a toy \(d_h=8\) query \(q=(1,0,3,1,4,1,5,9)\) with a quarter-turn rotation, only the first \(0.25\cdot8=2\) entries move: \(\tilde q=(0,1,3,1,4,1,5,9)\).

The motivation is that global layers already see the whole sequence; rotating every dimension is expensive and can hurt long-context stability. The report sets \(p=0.25\) on global layers, with RoPE base \(10^6\) global and \(10^4\) local.

---

## 8. The MLP: SwiGLU and GeGLU

A standard two-layer MLP is

\[
\operatorname{MLP}(x) = W_2\,\sigma(W_1x).
\]

!!! example "On paper"

    Take \(\sigma\) as the logistic sigmoid (the chapter's default), \(x=(1,-1)\), \(W_1=\begin{pmatrix}1&-1\\1&1\end{pmatrix}\), \(W_2=(1\ \ 1)\), and \(\sigma(2)=1/(1+e^{-2})\approx1/1.135\approx0.881\):

    \[
    \begin{aligned}
    W_1x &= (2,\,0)\\
    \sigma(W_1x) &= (0.881,\,0.5)\\
    W_2\,\sigma(W_1x) &= 0.881+0.5 = 1.381
    \end{aligned}
    \]

    The nonlinearity is what makes this more than one matrix: without \(\sigma\) the two layers would collapse into the single row \(W_2W_1=(2,\,0)\).

A gated linear unit uses two projections and a elementwise product:

\[
\operatorname{GLU}(x) = \big(\sigma(W_g x)\big)\odot(W_u x),
\qquad
\operatorname{MLP}(x)=W_d\,\operatorname{GLU}(x).
\]

!!! example "On paper"

    Keep the gate pre-activation \(W_gx=(2,0)\) (the \(W_1x\) above), let the up projection give \(W_ux=(1,3)\), and take \(W_d=(1\ \ 1)\):

    \[
    \begin{aligned}
    \operatorname{GLU}(x) &= (0.881\cdot1,\;0.5\cdot3)\\
    &= (0.881,\,1.5)\\
    W_d\operatorname{GLU}(x) &= 2.381
    \end{aligned}
    \]

    The gate decides how much of each up-projected channel passes: 88% of the first and 50% of the second.

**SwiGLU** uses SiLU (swish):

\[
\operatorname{SwiGLU}(x) = W_d\big(\operatorname{SiLU}(W_g x)\odot W_u x\big),
\quad
\operatorname{SiLU}(z)=z\,\sigma(z).
\]

!!! example "On paper"

    Same \(W_gx=(2,0)\), \(W_ux=(1,3)\) and \(W_d=(1\ \ 1)\), now with \(\operatorname{SiLU}(2)=2\cdot0.881=1.762\) and \(\operatorname{SiLU}(0)=0\):

    \[
    \begin{aligned}
    \operatorname{SiLU}(W_gx)\odot W_ux &= (1.762\cdot1,\;0\cdot3)\\
    \operatorname{SwiGLU}(x) &= 1.762+0 = 1.762
    \end{aligned}
    \]

    A gate pre-activation of 0 now closes its channel completely (the sigmoid gate let half through), and a negative one lets a little through with the sign flipped: \(\operatorname{SiLU}(-1)=-1\cdot0.269=-0.269\).

**GeGLU** uses GELU instead:

\[
\operatorname{GeGLU}(x) = W_d\big(\operatorname{GELU}(W_g x)\odot W_u x\big).
\]

!!! example "On paper"

    Same numbers, with \(\operatorname{GELU}(z)=z\,\Phi(z)\), where \(\Phi\) is the standard normal CDF and \(\Phi(2)\approx0.977\):

    \[
    \begin{aligned}
    \operatorname{GELU}(2) &= 2\cdot0.977 \approx 1.95\\
    \operatorname{GeGLU}(x) &= 1.95\cdot1+0\cdot3 = 1.95
    \end{aligned}
    \]

    GELU and SiLU have the same shape (0 at 0, close to \(z\) for large \(z\)); at \(z=2\) GELU passes 1.95 against SiLU's 1.76. The tanh approximation of GELU that Gemma's config uses also gives 1.95 here.

Qwen3/Qwen2.5 use SwiGLU. Gemma 4 uses GeGLU. A SwiGLU layer has three matrices, so its parameter count is

\[
P_{\text{MLP}} = 3\,d\,d_{\text{ff}}.
\]

For Qwen3-14B: \(3\cdot5120\cdot17{,}408\approx267\text{M}\) per layer. For Gemma 4 12B: \(3\cdot3840\cdot15{,}360\approx177\text{M}\). [43 §9](43-math-examples.md) computes a full SwiGLU/GeGLU forward and backward on a \(2\times3\) toy layer.

The expansion ratio is \(d_{\text{ff}}/d\): Qwen3-14B is 3.4, Qwen2.5-Coder-14B is 2.7, Gemma 4 12B is 4.0.

---

## 9. Mixture of experts (MoE)

A dense MLP activates all parameters for every token. An MoE replaces it with \(E\) experts and a router:

\[
g(x) = \operatorname{softmax}(W_r x),
\qquad
\mathcal{T}(x)=\operatorname{TopK}(g(x),k),
\]

\[
y = \sum_{i\in\mathcal{T}(x)} \frac{g_i(x)}{\sum_{j\in\mathcal{T}(x)}g_j(x)}\,E_i(x).
\]

!!! example "On paper"

    Four experts, \(k=2\), router logits \(W_rx=(1,1,0,-1)\), with \(e\approx2.718\) and \(e^{-1}\approx0.368\):

    \[
    \begin{aligned}
    g(x) &= \tfrac{1}{6.804}(2.718,\,2.718,\,1,\,0.368)\\
    &= (0.399,\,0.399,\,0.147,\,0.054)
    \end{aligned}
    \]

    The top 2 are experts 1 and 2. Their logits tie, so renormalising gives each \(g_1/(g_1+g_2)=0.5\). If \(E_1(x)=(2,0)\) and \(E_2(x)=(0,4)\):

    \[
    y = 0.5\,(2,0)+0.5\,(0,4) = (1,\,2)
    \]

    Experts 3 and 4 never run for this token: they cost no FLOPs and get no gradient from it.

Only \(k\) experts run per token. Total parameters scale with \(E\), but active parameters scale with \(k\).

Qwen3-30B-A3B: 128 experts, 8 active, no shared experts, 30.5B total / 3.3B active. Qwen3-235B-A22B: 94 layers, 64/4 heads, 128/8 experts, 235B/22B. Qwen3.5 uses 512 experts with 17B active out of 397B (4.28%).

### 9.1 Load balancing

Without a balancing loss, the router collapses onto a few experts. The Switch-style auxiliary loss is

\[
\mathcal{L}_{\text{bal}} =
\alpha\,E\sum_{i=1}^{E} f_i\,P_i,
\qquad
f_i=\frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[i\in\mathcal{T}(x_t)],
\quad
P_i=\frac{1}{T}\sum_{t=1}^{T}g_i(x_t).
\]

!!! example "On paper"

    Two experts, \(k=1\), a batch of \(T=4\) tokens: 3 go to expert 1 and 1 to expert 2, so \(f=(0.75,\,0.25)\); the router's mean probabilities are \(P=(0.7,\,0.3)\):

    \[
    \begin{aligned}
    \textstyle\sum_i f_iP_i &= 0.75\cdot0.7+0.25\cdot0.3 = 0.6\\
    \mathcal{L}_{\text{bal}} &= \alpha\cdot2\cdot0.6 = 1.2\,\alpha
    \end{aligned}
    \]

    A perfectly balanced router (\(f=P=(0.5,\,0.5)\)) scores \(\alpha\cdot2\cdot0.5=\alpha\); with \(\alpha=0.01\) this lopsided batch pays 0.012 instead of 0.010.

Qwen3 uses a **global-batch** load-balancing loss to encourage specialisation across the batch rather than forcing every sequence to be uniform. MoE training also needs capacity control, token dropping policy (or no dropping), expert parallelism, and careful all-to-all communication.

### 9.2 Active parameter count

For a dense model, inference FLOPs per token are \(2N\). For an MoE, use the **active** parameter count:

\[
N_{\text{active}} \approx N_{\text{attention}} + N_{\text{router}} + \frac{k}{E}N_{\text{experts}} + N_{\text{embed}}.
\]

!!! example "On paper"

    Qwen3-30B-A3B, rounded from its config (48 layers, \(d=2048\), 128 experts of width 768, \(k=8\), untied 151,936-token vocabulary): attention \(\approx0.9\)B, routers \(\approx0.01\)B, experts \(\approx29\)B, embedding + head \(\approx0.6\)B.

    \[
    \begin{aligned}
    N_{\text{active}} &\approx 0.9+0.01+\tfrac{8}{128}\cdot29+0.6\\
    &\approx 3.3\text{B}
    \end{aligned}
    \]

    The stored total is \(0.9+0.01+29+0.6\approx30.5\)B. Each token uses one-sixteenth of the experts, about 11% of all weights.

That is why Qwen3-30B-A3B generates at roughly 3B speed while storing 30B weights. [43 §10](43-math-examples.md) routes one token to the top 2 of four experts and computes the load-balancing loss.

---

## 10. Linear attention and Gated DeltaNet

Standard attention costs \(O(T^2)\) time and \(O(T)\) cache per layer. Linear attention replaces the softmax with a recurrent state \(S_t\in\mathbb{R}^{d_v\times d_k}\) (the convention of the Gated DeltaNet paper: the state maps a key to a value):

\[
S_t = S_{t-1} + v_t k_t^\top,
\qquad
o_t = S_t\, q_t.
\]

!!! example "On paper"

    Take \(d_k=d_v=2\) and \(S_0=0\). Store two facts with orthogonal keys: \(k_1=(1,0)\), \(v_1=(3,5)\), then \(k_2=(0,1)\), \(v_2=(2,4)\):

    \[
    S_2 = v_1k_1^\top+v_2k_2^\top = \begin{pmatrix}3&2\\5&4\end{pmatrix}
    \]

    \[
    S_2\, k_1 = (3,\,5) = v_1
    \]

    Querying with \(q=k_1\) returns \(v_1\) exactly. Now store \(k_3=(1,0)\), \(v_3=(1,1)\): the first column becomes \((4,6)\), and the same query returns \(v_1+v_3\). The old and new values are summed, not replaced.

The state is fixed-size, so inference is \(O(1)\) per token and training can be chunk-parallel. But additive memory overwrites nothing, so retrieval degrades.

**DeltaNet** adds the delta rule: before writing, correct the state’s prediction error:

\[
S_t = S_{t-1} + \beta_t\big(v_t - S_{t-1} k_t\big)k_t^\top,
\qquad
o_t = S_t\, q_t.
\]

This is one online gradient step on the associative memory loss. [43 §11](43-math-examples.md) runs two scalar updates and checks the gradient.

\[
\mathcal{L}(S)=\|S k_t - v_t\|_2^2.
\]

!!! example "On paper"

    Scalar case (\(d_k=d_v=1\)): the memory holds \(S_{t-1}=3\) for the key \(k_t=1\), the new value for that key is \(v_t=1\), and \(\beta_t=0.5\):

    \[
    S_t = 3+0.5\,(1-3\cdot1)\cdot1 = 2
    \]

    The loss \(\|S k_t-v_t\|^2\) falls from \((3-1)^2=4\) to \((2-1)^2=1\). With \(\beta_t=1\) the state becomes exactly 1, a clean overwrite; plain linear attention would add instead, giving \(3+1\cdot1=4\).

**Gated DeltaNet** adds a data-dependent decay \(\alpha_t\in(0,1)\) and a write rate \(\beta_t\in(0,1)\):

\[
S_t = \alpha_t S_{t-1}\big(I-\beta_t k_t k_t^\top\big) + \beta_t v_t k_t^\top,
\]

\[
o_t = S_t\, q_t.
\]

Equivalently:

\[
S_t = \alpha_t S_{t-1}
+ \beta_t\big(v_t-\alpha_t S_{t-1} k_t\big)k_t^\top.
\]

!!! example "On paper"

    The same scalar memory with forgetting: \(S_{t-1}=3\), \(k_t=1\), \(v_t=1\), \(\alpha_t=0.5\), \(\beta_t=0.5\). The first form gives

    \[
    S_t = 0.5\cdot3\cdot(1-0.5)+0.5\cdot1 = 1.25
    \]

    and the equivalent form gives the same number:

    \[
    S_t = 1.5+0.5\,(1-1.5) = 1.25
    \]

    With \(q_t=1\), \(o_t=1.25\). DeltaNet gave 2: the decay first halved the stale value 3 to 1.5, and only then was the correction written.

\(\alpha_t\) forgets stale associations; \(\beta_t\) controls how strongly the new key/value overwrites. In the real layer, \(k_t\) is L2-normalised and \(\beta_t\in(0,1)\); without normalisation the update can become unstable. Qwen3.5 uses GDN for most layers and standard gated attention every fourth layer (3:1). The chunkwise parallel form uses the WY representation of the product of rank-one Householder-like updates, so training is not strictly sequential.

![Gated DeltaNet state update](assets/diagrams/gdn.svg)

[38 — Derivations and proofs](38-derivations.md) shows that the delta rule is one online gradient step on \(\|S k_t-v_t\|^2\), and derives the WY chunkwise form.

**Why hybrid?** GDN gives cheap long-context memory; full attention every few layers recovers precise retrieval. Qwen3.5: 3 GDN : 1 gated attention. Gemma 4: 5 local sliding-window : 1 global attention (4:1 for E2B). Both are compromises between \(O(T^2)\) quality and \(O(T)\) cost.

---

## 11. Sliding-window and hybrid attention

A local layer attends only to the last \(W\) tokens:

\[
A_{ij} = 0 \quad\text{if } j<i-W+1.
\]

!!! example "On paper"

    Gemma 4 12B, \(W=1024\), query at position \(i=2000\):

    \[
    i-W+1 = 2000-1024+1 = 977
    \]

    So \(A_{2000,j}=0\) for \(j<977\), and the causal mask removes \(j>2000\). The token sees \(j=977,\dots,2000\), which is \(2000-977+1=1024\) positions, however long the document is.

Gemma 4 12B uses \(W=1024\), with a 5:1 local:global ratio (E2B/E4B use \(W=512\), and E2B a 4:1 ratio). Local layers cost \(O(TW)\) instead of \(O(T^2)\) and their KV cache is bounded by \(W\). Global layers keep full attention every sixth layer for long-range retrieval.

![Qwen3 full attention and Gemma 4 hybrid stack](assets/diagrams/hybrid-stacks.svg)

The cache for a hybrid stack is

\[
\text{KV} \approx \sum_{\ell\in\text{local}} 2G d_h W b
+ \sum_{\ell\in\text{global}} 2G d_h T b,
\]

and with K=V in global layers the second term loses the V factor. This is the math behind Gemma 4’s 256K context on a 12B model. The edge models add a third rule: KV-shared layers cache nothing, so only the first \(L-S\) layers appear in the sum ([37 §4.3](37-architecture.md)).

!!! example "On paper"

    Gemma 4 12B at \(T=8{,}192\), bf16 (\(b=2\)): 40 local layers with \(G=8\), \(d_h=256\), \(W=1024\), and 8 global layers with \(G=1\), \(d_h=512\) and K=V (no factor 2). Every factor except the 40 is a power of two:

    \[
    \begin{aligned}
    \text{local} &= 40\cdot2\cdot8\cdot256\cdot1024\cdot2\\
    &= 40\cdot2^{23} \approx 335.5\ \text{MB}\\
    \text{global} &= 8\cdot1\cdot512\cdot8192\cdot2\\
    &= 2^{26} \approx 67.1\ \text{MB}
    \end{aligned}
    \]

    The total is about 402.7 MB, and the global layers are exactly a sixth of it. At 32K they grow to 268.4 MB (the 604 MB in §6.3), while the local term never moves.

---

## 12. Multi-token prediction (MTP)

An MTP head predicts \(n\) future tokens from the same hidden state:

\[
\mathcal{L}_{\text{MTP}} =
-\sum_{j=1}^{n}\log p_\theta(x_{t+j}\mid x_{\le t}).
\]

!!! example "On paper"

    Three heads (\(n=3\)) at the position after `return` give the true next three tokens probabilities 0.5, 0.25 and 0.125: the further ahead, the harder. With \(\ln2\approx0.6931\):

    \[
    \begin{aligned}
    \mathcal{L}_{\text{MTP}} &= -(\ln0.5+\ln0.25+\ln0.125)\\
    &= \ln2+2\ln2+3\ln2\\
    &= 6\ln2 \approx 4.159
    \end{aligned}
    \]

    The third head alone contributes half of the loss (\(3\ln2\) of \(6\ln2\)).

Gemma 4 trains a small **autoregressive drafter** that takes the main model’s last-layer activations and token embeddings, and cross-attends to the main model’s KV cache. The drafter is used for speculative decoding: draft \(n\) tokens, verify them with the main model in one forward pass, accept the longest correct prefix. The expected speedup depends on the acceptance rate:

\[
\text{speedup} \approx \frac{1+\alpha+\alpha^2+\dots}{1+\text{draft cost}},
\]

where \(\alpha\) is the per-token acceptance probability. Gemma 4’s 12B drafter has ~400M parameters; the 31B/26B drafter uses a 4-layer transformer with model dimension 1024.

!!! example "On paper"

    The drafter proposes \(n=3\) tokens, each accepted with probability \(\alpha=0.5\), and each drafted token costs \(1/12\) of a main-model pass, so the draft cost is \(3/12=0.25\):

    \[
    \begin{aligned}
    \text{speedup} &\approx \frac{1+0.5+0.25+0.125}{1+0.25}\\
    &= \frac{1.875}{1.25} = 1.5
    \end{aligned}
    \]

    The series stops at \(\alpha^3\) because only three tokens were drafted. A better drafter with \(\alpha=0.9\) at the same cost gives \((1+0.9+0.81+0.729)/1.25\approx2.75\).

---

## 13. Parameter counting

For a dense decoder with \(L\) layers, hidden size \(d\), FFN size \(d_{\text{ff}}\), \(H\) query heads, \(G\) KV heads, head dimension \(d_h\), vocabulary \(V\):

\[
P_{\text{attn per layer}} =
\underbrace{d\,H\,d_h}_{Q}
+\underbrace{2\,d\,G\,d_h}_{K,V}
+\underbrace{H\,d_h\,d}_{O}.
\]

!!! example "On paper"

    Qwen3-14B: \(d=5120=5\cdot2^{10}\), \(Hd_h=40\cdot128=5120\) and \(Gd_h=8\cdot128=1024=2^{10}\):

    \[
    \begin{aligned}
    Q,\,O:\ 5120\cdot5120 &= 25\cdot2^{20}\ \text{each}\\
    K,\,V:\ 2\cdot5120\cdot1024 &= 10\cdot2^{20}\\
    P_{\text{attn}} = (25+10+25)\cdot2^{20} &\approx 62.9\text{M}
    \end{aligned}
    \]

    With \(2^{20}=1{,}048{,}576\) that is the 62.9M quoted below. K and V are only a sixth of it, because each has a fifth of the width of Q (\(G=8\) against \(H=40\)).

Do **not** assume \(H d_h=d\). It holds for Qwen3-14B, but not for Gemma 4 12B: local layers have \(H d_h=16\cdot256=4096\) while \(d=3840\), and global layers have \(H d_h=16\cdot512=8192\). The same asymmetry appears in the MLP input and output, so use the actual dimensions.

\[
P_{\text{MLP per layer}} = 3 d d_{\text{ff}},
\qquad
P_{\text{total}} \approx L(P_{\text{attn}}+P_{\text{MLP}})
+ (1+\mathbf{1}[\text{untied}])Vd.
\]

!!! example "On paper"

    A toy decoder you can count completely: \(d=8\), \(H=4\), \(G=2\), \(d_h=2\), \(d_{\text{ff}}=16\), \(L=2\), \(V=10\), tied. Its attention is \(8\cdot4\cdot2+2\cdot8\cdot2\cdot2+4\cdot2\cdot8=64+64+64=192\) per layer.

    \[
    \begin{aligned}
    P_{\text{MLP}} &= 3\cdot8\cdot16 = 384\\
    P_{\text{total}} &\approx 2\,(192+384)+1\cdot10\cdot8\\
    &= 1152+80 = 1232
    \end{aligned}
    \]

    Untied, the indicator adds a second \(Vd=80\), for 1,312. The \(\approx\) covers the small norm vectors the formula leaves out.

[43 §14](43-math-examples.md) counts a 328-parameter toy model by hand, then reconciles the Qwen and Gemma numbers.

Qwen3-14B: \(d=5120\), \(L=40\), \(d_{\text{ff}}=17408\), \(H=40\), \(G=8\), \(d_h=128\), \(V=151{,}936\), untied:

\[
P_{\text{attn}}=62.9\text{M},\quad
P_{\text{MLP}}=267.4\text{M},\quad
P_{\text{total}}\approx14.77\text{B}.
\]

Gemma 4 12B has two attention shapes:

| Layer type | \(H\) | \(G\) | \(d_h\) | \(P_{\text{attn}}\) | \(P_{\text{MLP}}\) | layers |
|---|---:|---:|---:|---:|---:|---:|
| local (sliding window) | 16 | 8 | 256 | 47.19M | 176.95M | 40 |
| global (full attention) | 16 | **1** | **512** | 64.88M | 176.95M | 8 |

The global layers use `num_global_key_value_heads = 1` and `global_head_dim = 512` in the checkpoint config, and they are **K=V**: the key projection doubles as the value, so the \(2dGd_h\) term becomes \(dGd_h\) — there is no `v_proj`. With tied embeddings:

\[
40(47.19+176.95) + 8(64.88+176.95) + 3840\cdot262{,}144
\approx 11.91\text{B}.
\]

The published checkpoint holds 11.96B: this 11.91B text decoder plus 52M of multimodal embedders; the model card rounds it to 11.95B. A naive count that uses 48 local layers gives 11.77B and undercounts the global attention by about 0.14B.

**Edge models.** Gemma 4 E2B and E4B add three terms the formula above does not have: a per-layer embedding table of \(V\cdot L\cdot256\) lookups, KV-shared final layers with no K/V projections, and (E2B) double-wide MLPs in those layers. Their “E” sizes — 2.3B and 4.5B effective out of 5.1B and 8.0B — are the totals minus the embedding tables. [37 §4](37-architecture.md) decomposes both checkpoints.

Run `scripts/inspect_architecture.py` for any config.

Qwen2.5-Coder-14B: \(d=5120\), \(L=48\), \(d_{\text{ff}}=13824\), \(H=40\), \(G=8\), \(V=152{,}064\), untied: about 14.7B.

Run `scripts/estimate_pretraining.py` for any config.

---

## 14. FLOPs, memory and time

### 14.1 Training compute

For a dense model with \(N\) parameters and \(D\) tokens, the standard estimate is below. [43 §15](43-math-examples.md) checks the units and the attention term on the toy model.

\[
C_{\text{train}} \approx 6ND.
\]

!!! example "On paper"

    A 1B-parameter model trained on 20B tokens (the Chinchilla-style 20 tokens per parameter):

    \[
    C_{\text{train}} \approx 6\cdot10^{9}\cdot20\cdot10^{9} = 1.2\times10^{20}
    \]

    Of that, the forward passes are \(2ND=4\times10^{19}\) FLOPs and the backward passes \(4ND=8\times10^{19}\).

Why 6: forward is \(2ND\) (one multiply-add per weight per token), backward is \(4ND\) (two matmuls: input gradient and weight gradient). Attention score/value FLOPs add roughly

\[
12\,L\,T^2\,H\,d_h
\]

for training (forward + backward), where \(H d_h\) is the total query width. This equals \(12LT^2d\) only when \(H d_h=d\), which is true for Qwen3-14B but not for Gemma’s global layers. It is negligible at 2K and significant at 32K. The exact constant depends on the implementation; measure tokens/sec rather than trusting a formula. See [38 — Derivations and proofs](38-derivations.md) §13 and [43 — Worked examples](43-math-examples.md) §15.

!!! example "On paper"

    Qwen3-14B (\(L=40\), \(Hd_h=5120=5\cdot2^{10}\)) on one sequence of \(T=2048=2^{11}\) tokens:

    \[
    \begin{aligned}
    12\cdot40\cdot2048^2\cdot5120 &= 2400\cdot2^{32}\\
    &\approx 1.03\times10^{13}
    \end{aligned}
    \]

    The \(6ND\) term for the same 2,048 tokens is \(6\cdot14.77\times10^9\cdot2048\approx1.81\times10^{14}\), so attention adds about 6%. At \(T=32{,}768\) attention grows \(16^2=256\) times but \(6ND\) only 16 times: \(2.64\times10^{15}\) against \(2.90\times10^{15}\), so it adds about 91%.

### 14.2 Inference compute

Per generated token:

\[
C_{\text{infer}} \approx 2N
\]

for dense models, plus attention and KV-cache reads. For MoE, replace \(N\) with \(N_{\text{active}}\).

!!! example "On paper"

    Qwen3-14B, \(N\approx14.77\)B:

    \[
    C_{\text{infer}} \approx 2\cdot14.77\times10^9 \approx 2.95\times10^{10}
    \]

    About 30 GFLOP per generated token, so a 500-token answer costs about \(1.5\times10^{13}\) FLOPs. Qwen3-30B-A3B, with \(N_{\text{active}}\approx3.3\)B, needs \(2\cdot3.3\times10^9=6.6\times10^9\) per token.

### 14.3 KV cache

As above:

\[
\text{KV bytes} \approx 2LGd_hTb.
\]

At 32K, Qwen3-14B bf16 is ~5.37 GB. Gemma 4 12B is hybrid, not full attention: 40 local layers with \(W=1024\) plus 8 global layers with \(G=1\), \(d_h=512\), and \(K=V\), for about **604 MB** at 32K and **2.48 GB** at 256K. The 12.9 GB full-attention calculation is only an upper bound. The edge models cache only their non-shared layers: Gemma 4 E2B needs about **208 MB** at 32K and **812 MB** at 128K, Gemma 4 E4B **558 MB** and **2.17 GB** — at 128K that is more than the 12B’s 1.41 GB, because E4B keeps two KV heads with separate K and V. Quantizing the cache to int8 halves any of these numbers. See [37 — Architecture anatomy](37-architecture.md) §2.1 and §4.3, and [43 — Worked examples](43-math-examples.md) §7.

### 14.4 Wall time and cost

\[
\text{GPU-hours} = \frac{6ND}{\text{peak FLOP/s}\times\text{MFU}\times3600},
\]

\[
\text{cost} = \text{GPU-hours}\times\text{price}.
\]

MFU (model FLOPs utilisation) is typically 30–50% for large training, lower at long context. Example: Qwen3-14B on 36T tokens at 40% MFU on 10,000 H100-class GPUs (989 TFLOP/s dense BF16; NVIDIA's 1,979 figure is with sparsity) is ~2.24M GPU-hours, ~9.3 days, ~$5.6M at $2.50/GPU-hour — before failures and data. Do not plan with 494.5 TFLOP/s: that is the dense TF32 peak, and it doubles the estimate.

---

## 15. Quantization mathematics

Affine quantization maps a real value \(x\) to an integer:

\[
q = \operatorname{clamp}\left(\operatorname{round}\left(\frac{x}{s}\right)+z,\;q_{\min},\;q_{\max}\right),
\]

\[
\hat x = s(q-z).
\]

!!! example "On paper"

    4-bit codes \(q_{\min}=0\), \(q_{\max}=15\), scale \(s=0.1\), zero point \(z=8\). Quantize \(x=0.33\):

    \[
    \begin{aligned}
    q &= \operatorname{clamp}(\operatorname{round}(3.3)+8,\,0,\,15) = 11\\
    \hat x &= 0.1\,(11-8) = 0.3
    \end{aligned}
    \]

    The error is 0.03, within half a step (\(s/2=0.05\)). A value outside the grid is clipped instead: \(x=1.0\) gives \(10+8=18\), clamped to 15, so \(\hat x=0.7\) and the error is 0.3.

For symmetric weights, \(z=0\). A **blockwise** format (Q4_0, GGUF) uses one scale per block of \(B\) weights:

\[
s_b = \frac{\max_{i\in b}|w_i|}{2^{k-1}-1}.
\]

!!! example "On paper"

    One block of \(B=4\) weights, \(w=(0.7,\,-0.33,\,0.12,\,-0.06)\), at \(k=4\) bits, so the grid is \(-7\dots7\) (\(2^{3}-1=7\)):

    \[
    s_b = \frac{0.7}{7} = 0.1
    \]

    Then \(q=\operatorname{round}(w/s_b)=(7,-3,1,-1)\) and \(\hat w=(0.7,-0.3,0.1,-0.1)\), with errors 0, 0.03, 0.02 and 0.04, all at most \(s_b/2=0.05\). A block of smaller weights gets a smaller \(s_b\) and proportionally smaller errors.

This is the clean symmetric grid \(-(2^{k-1}-1)\ldots(2^{k-1}-1)\). Real Q4_0 uses the code range \(-8\ldots7\) and a scale of \(\max/8\), so its positive maximum is represented by 7 rather than 8. The error bound below is therefore a good approximation, not the exact GGML quantizer behaviour; see [43 — Worked examples](43-math-examples.md) §16.

Quantization-aware training (QAT) inserts the quantize/dequantize function in the forward pass and uses a straight-through estimator for the gradient:

\[
\frac{\partial \hat x}{\partial x}\approx 1.
\]

!!! example "On paper"

    During QAT a weight \(x=0.33\) is used as \(\hat x=0.3\) (scale 0.1). Rounding is flat, so its true derivative is 0 and \(x\) would never move. With the STE, an upstream gradient \(\partial\mathcal{L}/\partial\hat x=2\) and \(\eta=0.01\):

    \[
    \begin{aligned}
    \partial\mathcal{L}/\partial x &\approx 2\cdot1 = 2\\
    x &\leftarrow 0.33-0.01\cdot2 = 0.31
    \end{aligned}
    \]

    The forward pass still sees 0.3, but after a few more such steps \(x\) drops below 0.25 and the quantized weight jumps to 0.2.

Gemma 4 uses QAT for mobile int2/int4 weights with int8 activations and for Q4_0; the report’s 12B memory goes from 24.0 GB bf16 to 7.65 GB Q4_0, and int8 KV at 32K adds 0.28 GiB (302 MB, half of the 604 MB bf16 cache). That is how a 12B model fits a laptop. [43 §16](43-math-examples.md) quantizes five weights and shows the blockwise error reduction.

---

## 16. Optimisation

The update is AdamW:

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,
\qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2,
\]

\[
\theta_t = \theta_{t-1} -
\eta_t\left(
\frac{m_t}{1-\beta_1^t}\Big/\left(\sqrt{\frac{v_t}{1-\beta_2^t}}+\epsilon\right)
+ \lambda\theta_{t-1}
\right).
\]

!!! example "On paper"

    One parameter at the first step (\(t=1\)): \(\theta_0=2\), \(g_1=-0.2\), \(\eta=0.01\), \(\lambda=0.1\), \(\beta_1=0.9\), \(\beta_2=0.95\), \(\epsilon\approx0\), \(m_0=v_0=0\):

    \[
    \begin{aligned}
    m_1 &= 0.1\cdot(-0.2) = -0.02\\
    v_1 &= 0.05\cdot0.04 = 0.002\\
    \frac{m_1/0.1}{\sqrt{v_1/0.05}} &= \frac{-0.2}{0.2} = -1\\
    \theta_1 &= 2-0.01\,(-1+0.1\cdot2) = 2.008
    \end{aligned}
    \]

    The bias-corrected first step has size \(\eta=0.01\) against the gradient, whatever the size of \(g\); weight decay takes \(0.01\cdot0.1\cdot2=0.002\) of it back.

Typical large-model values: \(\beta_1=0.9\), \(\beta_2=0.95\), \(\epsilon=10^{-8}\), weight decay 0.1, gradient clip 1.0, bf16 parameters/activations with fp32 master weights and optimizer state.

The learning rate schedule is usually warmup + cosine:

\[
\eta_t =
\begin{cases}
\eta_{\max}\frac{t}{T_w}, & t<T_w,\\
\eta_{\min}+\frac{1}{2}(\eta_{\max}-\eta_{\min})
\left(1+\cos\frac{\pi(t-T_w)}{T-T_w}\right), & t\ge T_w.
\end{cases}
\]

!!! example "On paper"

    \(\eta_{\max}=10^{-4}\), \(\eta_{\min}=10^{-5}\), \(T_w=100\), \(T=1000\). During warmup, at \(t=50\):

    \[
    \eta_{50} = 10^{-4}\cdot\tfrac{50}{100} = 5\times10^{-5}
    \]

    At \(t=400\), \((t-T_w)/(T-T_w)=300/900=1/3\) and \(\cos(\pi/3)=0.5\):

    \[
    \begin{aligned}
    \eta_{400} &= 10^{-5}+\tfrac12(9\times10^{-5})(1+0.5)\\
    &= 10^{-5}+6.75\times10^{-5} = 7.75\times10^{-5}
    \end{aligned}
    \]

    A third of the way through the cosine, the LR is still 75% of the way from \(\eta_{\min}\) to \(\eta_{\max}\).

Qwen3 develops scaling laws for learning rate and batch size across its three pretraining stages; you should not copy a 14B LR to a 100M model. [43 §17](43-math-examples.md) runs two AdamW steps and one cosine-schedule point.

### 16.1 Initialisation

GPT-2-style:

\[
W\sim\mathcal{N}(0,0.02^2),
\]

with residual output projections scaled by \(1/\sqrt{2L}\) in deeper models. Qwen/Gemma use standard normal init plus QK-norm and pre-norm for stability; exact init is not fully public.

!!! example "On paper"

    0.02 is a standard deviation, so about 95% of the initial weights lie within \(\pm2\cdot0.02=\pm0.04\). For the residual output projections of a 40-layer model (Qwen3-14B depth), with \(\sqrt{80}\approx8.944\):

    \[
    \frac{0.02}{\sqrt{2\cdot40}} = \frac{0.02}{8.944} \approx 0.0022
    \]

    There are \(2L=80\) residual writes (one attention and one MLP per layer); at variance \(0.02^2/80\) each, together they add roughly the variance of one unscaled write.

### 16.2 Stability tools

- QK-norm (Qwen3, Gemma 4).
- Pre-norm residual.
- Gradient clipping.
- bf16 instead of fp16 for large models (no loss scaling).
- z-loss on logits (optional).
- Careful data ordering and LR warmup.
- MoE load balancing and capacity control.
- Checkpoint rollback on loss spikes.

---

## 17. A worked comparison

| Quantity | Qwen3-14B | Qwen2.5-Coder-14B | Gemma 4 12B |
|---|---:|---:|---:|
| Parameters | 14.8B | 14.7B | 11.95B |
| Layers | 40 | 48 | 48 |
| \(d\) | 5,120 | 5,120 | 3,840 |
| FFN | 17,408 | 13,824 | 15,360 |
| Heads / KV | 40 / 8 | 40 / 8 | 16 / 8 |
| Head dim | 128 | 128 | 256 (global 512) |
| Vocab | 151,669 | 151,646 | 262,144 |
| Norm | RMSNorm + QK-norm | RMSNorm | RMSNorm pre+post + QK-norm |
| MLP | SwiGLU | SwiGLU | GeGLU |
| Attention | full, GQA | full, GQA | 5:1 local/global, K=V global, p-RoPE |
| Context | 32K train, 128K YaRN | 128K | 256K |
| Tokenizer | BBPE | BBPE | SentencePiece, split digits |

The table is the fastest way to see why “same parameter count” does not mean “same model”: depth, width, vocabulary, attention pattern and tokenizer all change compute, memory and behaviour.

---

## 18. What this means for fine-tuning

- LoRA adapts linear projections; it does not change attention pattern, tokenizer or KV cache.
- Training sequence length is a memory budget, not the model’s native context.
- Quantization changes the base weights; QLoRA adds adapters on top.
- A 12B model with a 262k vocabulary has a large logit tensor; generation is not free.
- GQA/K=V/sliding-window choices determine how much context you can actually serve.
- The loss you see in MLX is token-level cross-entropy; it is not comparable across tokenizers.

---

## 19. Sources

- Attention Is All You Need — <https://arxiv.org/abs/1706.03762>
- RoPE — <https://arxiv.org/abs/2104.09864>
- GQA — <https://arxiv.org/abs/2305.13245>
- RMSNorm — <https://arxiv.org/abs/1910.07467>
- SwiGLU — <https://arxiv.org/abs/2002.05202>
- QK-Norm / scalable transformers — <https://arxiv.org/abs/2106.04560>
- Chinchilla scaling laws — <https://arxiv.org/abs/2203.15556>
- Kaplan scaling laws — <https://arxiv.org/abs/2001.08361>
- Gated DeltaNet — <https://arxiv.org/abs/2412.06464>
- FlashAttention — <https://arxiv.org/abs/2205.14135>
- ZeRO — <https://arxiv.org/abs/1910.02054>
- Megatron-LM — <https://arxiv.org/abs/1909.08053>
- Qwen3 technical report — <https://arxiv.org/abs/2505.09388>
- Qwen2.5-Coder technical report — <https://arxiv.org/abs/2409.12186>
- Gemma 4 technical report — <https://arxiv.org/abs/2607.02770>
- Qwen3.5 — <https://qwen.ai/blog?id=qwen3.5>
- Qwen3.5 Transformers docs — <https://huggingface.co/docs/transformers/model_doc/qwen3_5>
