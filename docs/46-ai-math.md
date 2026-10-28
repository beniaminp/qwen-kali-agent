# 46 — The math you actually need

*Last verified: 28 October 2026.*

This chapter rebuilds the math an AI engineer uses every week, starting from zero: vectors and matrices (what an embedding and a layer are), derivatives and gradient descent (how a model learns), probability and information theory (what a model outputs and what its loss means), and the statistics you need to decide whether prompt B really beats prompt A. Every formula comes with a small "On paper" card you can redo with a pencil, and every code block is Python (mostly numpy) that was run to produce the output shown under it. Read [45 — From software engineer to AI engineer](45-ai-engineer-path.md) first if you want the big picture; after this chapter, [47 — Machine learning fundamentals](47-ai-classical-ml.md) puts the math to work. The transformer-specific math (attention, RoPE, the KV cache, parameter counts) is not repeated here: it lives in [32 — Mathematics](32-mathematics.md), [38 — Derivations and proofs](38-derivations.md) and [43 — Worked examples](43-math-examples.md), and §14 says where.

---

## 1. Notation and setup

### 1.1 How to read the formulas

You will meet the same dozen symbols in every paper, model card and library doc. Learn them once.

| Symbol | Read as | Meaning | Example |
|---|---|---|---|
| \(x\) | "x" | a scalar (one number) | a temperature, 0.7 |
| \(\mathbf x\) | "bold x" | a vector (a list of numbers) | an embedding, \((1, 2, 2)\) |
| \(x_i\) | "x sub i" | the \(i\)-th entry of \(\mathbf x\) | \(x_2 = 2\) |
| \(W\) | "W" | a matrix (a grid of numbers) | a layer's weights |
| \(W^\top\) | "W transpose" | rows and columns swapped | |
| \(\sum_{i=1}^{n} x_i\) | "sum over i" | add \(x_1\) to \(x_n\) | a for loop with `total +=` |
| \(\prod_{i=1}^{n} p_i\) | "product over i" | multiply \(p_1\) to \(p_n\) | a for loop with `total *=` |
| \(\lVert\mathbf x\rVert\) | "norm of x" | the length of a vector | |
| \(\frac{\partial L}{\partial w}\) | "partial L by w" | how much \(L\) moves when \(w\) moves | |
| \(\nabla L\) | "grad L" | all partial derivatives in one vector | |
| \(P(A\mid B)\) | "P of A given B" | probability of \(A\) once you know \(B\) | |
| \(\mathbb E[X]\) | "expected value of X" | the long-run average | |
| \(\hat p\) | "p hat" | an estimate of \(p\) from data | 0.8 from 320 of 400 |
| \(\ln\), \(\log\) | "natural log" | log base \(e\); ML papers write \(\log\) for \(\ln\) | \(\ln 2 \approx 0.693\) |

A formula is a function written compactly. When one looks dense, translate it into the loop you would write, then run the loop on three numbers.

### 1.2 numpy in one minute

numpy stores numbers in arrays with a **shape** (how many entries along each axis) and a **dtype** (how each number is stored). Almost every bug in numerical code is a wrong shape or a wrong dtype, so print both often.

```python
import numpy as np

s = np.float32(0.7)                              # scalar: a temperature
v = np.array([1.0, 2.0, 2.0])                    # vector: a tiny embedding
M = np.array([[1, 2], [3, 4], [0, -1]])          # matrix: 3 rows, 2 columns
T = np.zeros((8, 512, 5120), dtype=np.float16)   # tensor: a batch of activations

for name, a in [("s", s), ("v", v), ("M", M), ("T", T)]:
    print(f"{name}: ndim={np.ndim(a)} shape={np.shape(a)} dtype={a.dtype}")
print("T uses", T.nbytes, "bytes")
```

```text
s: ndim=0 shape=() dtype=float32
v: ndim=1 shape=(3,) dtype=float64
M: ndim=2 shape=(3, 2) dtype=int64
T: ndim=3 shape=(8, 512, 5120) dtype=float16
T uses 41943040 bytes
```

---

## 2. Scalars, vectors, matrices and tensors

### 2.1 The four kinds of objects

- A **scalar** is one number: a learning rate, a loss value, a temperature.
- A **vector** is an ordered list of numbers. An embedding is a vector: the text "refund my order" becomes, for example, 1,024 numbers, and texts with similar meaning get similar vectors.
- A **matrix** is a rectangle of numbers with rows and columns. A layer's weights are a matrix: Qwen3-14B's query projection is 5,120 × 5,120.
- A **tensor** is the general case: an array with any number of axes. A scalar is a 0-axis tensor, a vector 1-axis, a matrix 2-axis. In PyTorch and numpy, "tensor" and "array" mean the same thing.

Why you care: every operation in a model is defined by the shapes it accepts and returns. If you can say the shape of every tensor in a pipeline, you can debug it.

### 2.2 The shapes of a real batch of token embeddings

Follow a batch through the first step of Qwen3-14B. You send 8 prompts, each padded to 512 tokens.

| Tensor | Shape | Meaning |
|---|---|---|
| token ids | `(8, 512)` | batch × sequence, integers in \([0, 151936)\) |
| embedding table | `(151936, 5120)` | one 5,120-number row per vocabulary entry |
| embeddings | `(8, 512, 5120)` | batch × sequence × hidden size |
| logits at the end | `(8, 512, 151936)` | one score per vocabulary entry, per position |

The embedding step is a lookup: row `token_id` of the table is copied into the output. Shapes also give you memory: multiply the axes and the bytes per number.

\[
\text{bytes} = B \cdot T \cdot d \cdot \text{bytes per number}
\]

!!! example "On paper"

    The embeddings above in bf16 (2 bytes per number):

    \[
    \begin{aligned}
    8\cdot512\cdot5120 &= 20{,}971{,}520 \text{ numbers}\\
    20{,}971{,}520\cdot2 &= 41{,}943{,}040 \text{ bytes} = 40\text{ MiB}
    \end{aligned}
    \]

    The logits are another matter: \(8\cdot512\cdot151{,}936\cdot4\) bytes in fp32 is about 2.49 GB (2.32 GiB). On a 32 GB Mac, the last tensor of a batch, not the first, is the one that runs you out of memory.

### 2.3 The same lookup in code

A toy table with a vocabulary of 10 and a hidden size of 8 behaves exactly like the real one:

```python
import numpy as np

rng = np.random.default_rng(0)
vocab, d_model = 10, 8                    # Qwen3-14B: 151,936 and 5,120
E = rng.normal(size=(vocab, d_model)).astype(np.float32)   # embedding table

token_ids = np.array([[3, 7, 1, 0],       # prompt 1, padded to 4 tokens
                      [5, 5, 2, 9]])      # prompt 2
x = E[token_ids]                          # one row of E per token

print("token_ids", token_ids.shape, token_ids.dtype)
print("E        ", E.shape, E.dtype)
print("x        ", x.shape, x.dtype)
print("x[1, 0] is row 5 of E:", np.array_equal(x[1, 0], E[5]))
print("bytes of x:", x.nbytes)
```

```text
token_ids (2, 4) int64
E         (10, 8) float32
x         (2, 4, 8) float32
x[1, 0] is row 5 of E: True
bytes of x: 256
```

`E[token_ids]` returned a tensor whose shape is the ids' shape with the hidden size appended. That rule ("index shape + row shape") is how every embedding layer works.

### 2.4 Pitfall: shape bugs are silent

A wrong shape rarely crashes. It produces a tensor of the wrong size that flows on, and you see a bad metric hours later. Two habits fix most of it: write the expected shape as a comment next to every tensor (`# (B, T, d)`), and add `assert x.shape == (B, T, d)` at module boundaries. It is the same discipline as type annotations on a public API.

---

## 3. Vectors

### 3.1 Addition and scaling

You add two vectors entry by entry, and you scale a vector by multiplying every entry by the same number.

\[
\mathbf a+\mathbf b=(a_1+b_1,\dots,a_n+b_n),
\quad
c\,\mathbf a=(c\,a_1,\dots,c\,a_n)
\]

!!! example "On paper"

    Two 3-number embeddings, \(\mathbf a=(1,2,2)\) for "refund my order" and \(\mathbf b=(2,1,2)\) for "money back please":

    \[
    \begin{aligned}
    \mathbf a+\mathbf b &= (3,3,4)\\
    \tfrac12(\mathbf a+\mathbf b) &= (1.5,1.5,2)
    \end{aligned}
    \]

    The average of two embeddings lands between them. That is how a "mean pooled" sentence embedding is built from token embeddings, and why the average of two refund requests is still a refund request.

Why you care: residual connections in a transformer are vector addition (`x = x + attention(x)`), and a learning-rate step is scaling (`w = w - lr * grad`).

### 3.2 The dot product as a weighted sum

The dot product multiplies two vectors entry by entry and adds the results. Read it as "quantities times prices".

\[
\mathbf a\cdot\mathbf b=\sum_{i=1}^{n}a_i b_i
\]

!!! example "On paper"

    One request to a hosted model uses 1,000 fresh input tokens, 4,000 cached input tokens and 250 output tokens. Assume prices of $2, $0.50 and $8 per million tokens:

    \[
    \begin{aligned}
    \mathbf n &= (1000,\ 4000,\ 250)\\
    \mathbf p &= (2,\ 0.5,\ 8)\ \$\text{ per million}\\
    \mathbf n\cdot\mathbf p &= 2000+2000+2000 = 6000\\
    \text{cost} &= 6000/10^6 = \$0.006
    \end{aligned}
    \]

    A million such requests cost $6,000, and each kind of token contributes a third. A neuron does the same arithmetic: inputs times weights, summed.

Why you care: a linear layer is many dot products, attention scores are dot products, and a similarity search is a dot product per document. If you understand this one operation you understand most of the compute in a model.

### 3.3 The dot product as similarity

Geometrically, the dot product is the product of the two lengths and the cosine of the angle between the vectors.

\[
\mathbf a\cdot\mathbf b=\lVert\mathbf a\rVert\,\lVert\mathbf b\rVert\cos\theta
\]

!!! example "On paper"

    Take \(\mathbf a=(1,0)\) and \(\mathbf b=(1,1)\), 45° apart, with \(\cos45°\approx0.707\):

    \[
    \begin{aligned}
    \mathbf a\cdot\mathbf b &= 1\cdot1+0\cdot1 = 1\\
    \lVert\mathbf a\rVert\,\lVert\mathbf b\rVert\cos\theta &= 1\cdot\sqrt2\cdot0.707 \approx 1
    \end{aligned}
    \]

    Both sides agree. The sign tells you the direction: positive when the vectors point the same way, zero at 90°, negative when they point apart.

### 3.4 Norms: L1 and L2

A norm is the "size" of a vector. The **L2 norm** is the straight-line length (Pythagoras). The **L1 norm** is the sum of absolute values (the distance a taxi drives on a grid).

\[
\lVert\mathbf x\rVert_2=\sqrt{\sum_i x_i^2},
\quad
\lVert\mathbf x\rVert_1=\sum_i \lvert x_i\rvert
\]

!!! example "On paper"

    For \(\mathbf x=(3,-4)\):

    \[
    \begin{aligned}
    \lVert\mathbf x\rVert_2 &= \sqrt{9+16} = 5\\
    \lVert\mathbf x\rVert_1 &= 3+4 = 7
    \end{aligned}
    \]

    L1 is never smaller than L2. Regularization uses both: an L2 penalty shrinks all weights a little, an L1 penalty pushes small weights to exactly zero ([47](47-ai-classical-ml.md) shows it in code). Gradient clipping caps the L2 norm of the gradient.

### 3.5 Unit vectors and normalisation

Dividing a vector by its L2 norm gives a **unit vector**: same direction, length 1. \((3,4)/5=(0.6,0.8)\), and \(0.36+0.64=1\). Embedding APIs and libraries often return vectors already normalised; check before you normalise twice (harmless) or forget to (a real bug when you then use the dot product as a similarity).

### 3.6 Cosine similarity

Cosine similarity is the dot product of the two unit vectors: it keeps the direction and throws away the length. It ranges from −1 (opposite) through 0 (unrelated) to 1 (same direction).

\[
\cos(\mathbf a,\mathbf b)=\frac{\mathbf a\cdot\mathbf b}{\lVert\mathbf a\rVert\,\lVert\mathbf b\rVert}
\]

!!! example "On paper"

    \(\mathbf a=(1,2,2)\) "refund my order", \(\mathbf b=(2,1,2)\) "money back please", \(\mathbf c=(-2,1,0)\) "cancel my subscription":

    \[
    \begin{aligned}
    \mathbf a\cdot\mathbf b &= 2+2+4 = 8\\
    \lVert\mathbf a\rVert &= \sqrt{1+4+4} = 3\\
    \lVert\mathbf b\rVert &= 3\\
    \cos(\mathbf a,\mathbf b) &= 8/9 \approx 0.889\\
    \mathbf a\cdot\mathbf c &= -2+2+0 = 0
    \end{aligned}
    \]

    0.889 is close to 1: a search for one refund request finds the other. The cancellation request is at 90° to it (cosine 0): unrelated, as far as these three numbers can tell.

Why you care: it is the score behind semantic search, deduplication, clustering of user questions and "find similar tickets". Its pitfall: cosine says "similar topic", not "same answer". "Cancel my order" and "do not cancel my order" are close.

### 3.7 Cosine, dot product and distance agree on unit vectors

For unit vectors, squared Euclidean distance and cosine carry the same information, so ranking by either gives the same order.

\[
\lVert\hat{\mathbf a}-\hat{\mathbf b}\rVert^2=2-2\cos(\mathbf a,\mathbf b)
\]

!!! example "On paper"

    With \(\hat{\mathbf a}=(1,2,2)/3\) and \(\hat{\mathbf b}=(2,1,2)/3\):

    \[
    \begin{aligned}
    \hat{\mathbf a}-\hat{\mathbf b} &= (-1,1,0)/3\\
    \lVert\hat{\mathbf a}-\hat{\mathbf b}\rVert^2 &= 2/9 \approx 0.222\\
    2-2\cdot\tfrac89 &= 2/9
    \end{aligned}
    \]

    Same number both ways. This is why a vector database can offer "cosine", "dot" and "L2" metrics and give identical rankings when the vectors are normalised; it matters only when they are not.

### 3.8 Semantic search in a few lines

Stack the document vectors as rows of a matrix \(D\). One matrix–vector product then gives every cosine at once.

```python
import numpy as np

docs = ["refund my order", "money back please", "cancel my subscription", "reset my password"]
D = np.array([[1, 2, 2], [2, 1, 2], [-2, 1, 0], [0, -1, 3]], dtype=float)  # toy embeddings
q = np.array([1.5, 1.5, 2.0])          # embedding of the query "I want a refund"

D_unit = D / np.linalg.norm(D, axis=1, keepdims=True)   # each row to length 1
q_unit = q / np.linalg.norm(q)
scores = D_unit @ q_unit               # (4, 3) @ (3,) -> (4,): all cosines at once

for i in np.argsort(-scores):
    print(f"{scores[i]:+.3f}  {docs[i]}")
```

```text
+0.972  refund my order
+0.972  money back please
+0.488  reset my password
-0.230  cancel my subscription
```

The two refund documents tie at the top because the query is their exact average. A real embedding model returns 384 to 4,096 numbers instead of 3, and a real index holds millions of rows; [51 — RAG](51-ai-rag.md) covers both.

---

## 4. Matrices

### 4.1 Matrix–vector multiply

A matrix \(W\) with \(m\) rows and \(n\) columns times a vector \(\mathbf x\) of length \(n\) gives a vector of length \(m\). Each output entry is the dot product of one row of \(W\) with \(\mathbf x\).

\[
(W\mathbf x)_i=\sum_{j=1}^{n}W_{ij}\,x_j
\]

!!! example "On paper"

    \(W\) has rows \((1,2)\), \((3,4)\), \((0,-1)\) (shape 3 × 2) and \(\mathbf x=(5,6)\):

    \[
    \begin{aligned}
    \text{row 1} &: 1\cdot5+2\cdot6 = 17\\
    \text{row 2} &: 3\cdot5+4\cdot6 = 39\\
    \text{row 3} &: 0\cdot5-1\cdot6 = -6\\
    W\mathbf x &= (17,39,-6)
    \end{aligned}
    \]

    Two numbers went in, three came out. A layer is a function that changes the size of a vector: 5,120 in, 17,408 out in Qwen3-14B's MLP.

### 4.2 Matrix–matrix multiply and the shape rule

Multiplying \(A\) (shape \(m\times n\)) by \(B\) (shape \(n\times p\)) gives \(C\) (shape \(m\times p\)). Entry \((i,j)\) is row \(i\) of \(A\) dotted with column \(j\) of \(B\). The inner sizes must match and disappear: \((m\times n)(n\times p)\to(m\times p)\).

\[
C_{ij}=\sum_{k=1}^{n}A_{ik}B_{kj}
\]

!!! example "On paper"

    \(A\) has rows \((1,0,2)\) and \((0,1,1)\) (2 × 3); \(B\) has rows \((1,2)\), \((3,4)\), \((5,6)\) (3 × 2):

    \[
    \begin{aligned}
    C_{11} &= 1\cdot1+0\cdot3+2\cdot5 = 11\\
    C_{12} &= 1\cdot2+0\cdot4+2\cdot6 = 14\\
    C_{21} &= 0\cdot1+1\cdot3+1\cdot5 = 8\\
    C_{22} &= 0\cdot2+1\cdot4+1\cdot6 = 10
    \end{aligned}
    \]

    The result is 2 × 2 and took \(2\cdot3\cdot2=12\) multiplications. In general \(m n p\) multiply-adds: a 2,048-token prompt through one 5,120 × 5,120 projection is \(2048\cdot5120\cdot5120\approx5.4\times10^{10}\) multiply-adds, about 107 GFLOP.

Order matters: \(AB\) is 2 × 2 here but \(BA\) is 3 × 3, and in general \(AB\ne BA\). Matrix multiplication is the operation GPUs and the M1 Pro's GPU are built for; "FLOPs" in [32 §14](32-mathematics.md#14-flops-memory-and-time) counts these multiply-adds, two FLOPs each (one multiply, one add).

### 4.3 Transpose

The transpose \(A^\top\) swaps rows and columns: a 2 × 3 matrix becomes 3 × 2. Two facts are used constantly. A dot product is a row times a column, \(\mathbf a\cdot\mathbf b=\mathbf a^\top\mathbf b\). And the transpose of a product reverses the order.

\[
(AB)^\top=B^\top A^\top
\]

!!! example "On paper"

    With \(A\) (2 × 3) and \(B\) (3 × 2) from §4.2, \(AB\) has rows \((11,14)\) and \((8,10)\):

    \[
    \begin{aligned}
    (AB)^\top &= \text{rows } (11,8),\ (14,10)\\
    B^\top A^\top &: (2\times3)(3\times2) \to 2\times2\\
    (B^\top A^\top)_{11} &= 1\cdot1+3\cdot0+5\cdot2 = 11
    \end{aligned}
    \]

    The shapes only line up in the reversed order, which is the quickest way to remember the rule.

Why you care: PyTorch's `nn.Linear` stores its weight as `(out_features, in_features)` and computes `x @ W.T + b`. When you load weights by hand (LoRA merges, conversions between formats) a missing `.T` is the classic bug, and with square matrices it does not even raise an error.

### 4.4 Shape errors in code

```python
import numpy as np

A = np.array([[1, 0, 2], [0, 1, 1]])          # (2, 3)
B = np.array([[1, 2], [3, 4], [5, 6]])        # (3, 2)
print("A @ B =\n", A @ B)
print("B @ A shape:", (B @ A).shape)
try:
    A @ A                                     # (2, 3) @ (2, 3): inner sizes 3 and 2
except ValueError as e:
    print("A @ A fails:", e)
```

```text
A @ B =
 [[11 14]
 [ 8 10]]
B @ A shape: (3, 3)
A @ A fails: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 2 is different from 3)
```

The error names the mismatched sizes. Read it as "inner dimensions 3 and 2 do not match", then find which tensor has the axis you did not expect.

### 4.5 Broadcasting and the bug it causes

**Broadcasting** lets numpy and PyTorch combine arrays of different shapes by stretching axes of size 1. Used on purpose, it is convenient: `X - X.mean(axis=0)` subtracts each column's mean from every row. Used by accident, it silently turns a vector of errors into a matrix.

```python
import numpy as np

y_true = np.array([1.0, 2.0, 3.0, 4.0])            # shape (4,)
y_pred = np.array([[1.1], [1.9], [3.2], [3.9]])    # shape (4, 1): a model's column output

diff = y_true - y_pred                              # broadcasts to (4, 4), no error
print("diff shape:", diff.shape)
print("wrong MSE:", round(np.mean(diff ** 2), 4))
print("right MSE:", round(np.mean((y_true - y_pred.ravel()) ** 2), 4))
```

```text
diff shape: (4, 4)
wrong MSE: 2.4425
right MSE: 0.0175
```

The wrong number is not even close, and nothing warned you. Every pair of (4,) and (4, 1) entries was compared. The rule numpy applies: align shapes from the right; two sizes are compatible if they are equal or one of them is 1. `(4,)` against `(4, 1)` becomes `(1, 4)` against `(4, 1)`, so both stretch to `(4, 4)`. Fix it by making shapes explicit (`ravel()`, `squeeze(-1)`, `keepdims=True`) and by asserting the shape of a loss's inputs.

### 4.6 Identity and inverse

The **identity** matrix \(I\) has ones on the diagonal and zeros elsewhere; \(I\mathbf x=\mathbf x\), the matrix version of multiplying by 1. The **inverse** \(A^{-1}\) undoes \(A\): \(A^{-1}A=I\). Only square matrices with a non-zero determinant have one. For a 2 × 2 matrix there is a short formula.

\[
\begin{pmatrix}a&b\\c&d\end{pmatrix}^{-1}
=\frac{1}{ad-bc}\begin{pmatrix}d&-b\\-c&a\end{pmatrix}
\]

!!! example "On paper"

    \(A\) has rows \((2,1)\) and \((1,1)\); solve \(A\mathbf x=(3,2)\):

    \[
    \begin{aligned}
    ad-bc &= 2\cdot1-1\cdot1 = 1\\
    A^{-1} &= \text{rows } (1,-1),\ (-1,2)\\
    \mathbf x = A^{-1}(3,2) &= (3-2,\ -3+4) = (1,1)
    \end{aligned}
    \]

    Check: \(2\cdot1+1\cdot1=3\) and \(1\cdot1+1\cdot1=2\). If \(ad-bc\) were 0, the two rows would point the same way and there would be no unique answer.

That is the level you need. You will meet inverses in the closed-form solution of linear regression ([47](47-ai-classical-ml.md)), and you should know that code never computes them explicitly when it can avoid it: `np.linalg.solve` is faster and more accurate.

```python
import numpy as np

A = np.array([[2.0, 1.0], [1.0, 1.0]])
b = np.array([3.0, 2.0])
A_inv = np.linalg.inv(A)
print("A_inv =\n", A_inv)
print("A @ A_inv =\n", A @ A_inv)
print("x via solve:", np.linalg.solve(A, b))
```

```text
A_inv =
 [[ 1. -1.]
 [-1.  2.]]
A @ A_inv =
 [[1. 0.]
 [0. 1.]]
x via solve: [1. 1.]
```

### 4.7 The linear layer

A **linear layer** (also "dense" or "fully connected") multiplies its input by a weight matrix and adds a bias vector. It is the workhorse of every neural network: Qwen3-14B is mostly linear layers.

\[
\mathbf y=W\mathbf x+\mathbf b
\]

!!! example "On paper"

    \(W\) has rows \((1,0,-1)\) and \((2,1,0)\) (2 × 3), \(\mathbf b=(0.5,-1)\), \(\mathbf x=(1,2,3)\):

    \[
    \begin{aligned}
    W\mathbf x &= (1-3,\ 2+2) = (-2,4)\\
    \mathbf y &= (-2+0.5,\ 4-1) = (-1.5,3)\\
    \text{parameters} &= 2\cdot3+2 = 8
    \end{aligned}
    \]

    A layer from \(n\) inputs to \(m\) outputs has \(mn+m\) parameters. Qwen3-14B's query projection, 5,120 → 5,120 without a bias, has 26,214,400.

In code you process a whole batch at once: stack the inputs as rows of \(X\) (shape `(batch, in)`) and compute `X @ W.T + b`. Broadcasting adds the bias to every row, the intended use of §4.5.

```python
import numpy as np

W = np.array([[1.0, 0.0, -1.0],
              [2.0, 1.0,  0.0]])     # (out=2, in=3), the nn.Linear layout
b = np.array([0.5, -1.0])            # (2,)
X = np.array([[1.0, 2.0, 3.0],       # a batch of 4 inputs, shape (4, 3)
              [0.0, 0.0, 0.0],
              [1.0, 1.0, 1.0],
              [-1.0, 0.0, 2.0]])

Y = X @ W.T + b                      # (4, 3) @ (3, 2) + (2,) -> (4, 2)
print(Y)
print("parameters:", W.size + b.size)

d = 5120                             # Qwen3-14B hidden size
print(f"q_proj 5120->5120: {d * d:,} weights; k_proj 5120->1024: {d * 1024:,}")
```

```text
[[-1.5  3. ]
 [ 0.5 -1. ]
 [ 0.5  2. ]
 [-2.5 -3. ]]
parameters: 8
q_proj 5120->5120: 26,214,400 weights; k_proj 5120->1024: 5,242,880
```

The second row shows what the bias is for: with an all-zero input the output is exactly \(\mathbf b\). The key projection is smaller (1,024 outputs = 8 KV heads × 128) because Qwen3 uses grouped-query attention; [32 §6.2](32-mathematics.md#62-grouped-query-attention-gqa) explains why.

---

## 5. Derivatives

### 5.1 Slope: how much does the output move?

A derivative answers one question: if I nudge the input a little, how much does the output move, per unit of nudge? Take \(f(x)=x^2\) at \(x=3\), where \(f(3)=9\), and nudge \(x\) by smaller and smaller amounts \(h\):

| \(h\) | \(f(3+h)\) | change in \(f\) | change ÷ \(h\) |
|---:|---:|---:|---:|
| 1 | 16 | 7 | 7 |
| 0.1 | 9.61 | 0.61 | 6.1 |
| 0.01 | 9.0601 | 0.0601 | 6.01 |
| 0.001 | 9.006001 | 0.006001 | 6.001 |

The ratio settles at 6. At \(x=3\), the output moves 6 times as much as the input. That number is the derivative \(f'(3)\).

Why you care: training a model means asking this question for every weight: "if I move this weight a little, how much does the loss move?" Then you move each weight in the direction that lowers the loss.

### 5.2 The definition

\[
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
\]

!!! example "On paper"

    For \(f(x)=x^2\) the limit can be done exactly:

    \[
    \begin{aligned}
    \frac{(x+h)^2-x^2}{h} &= \frac{2xh+h^2}{h} = 2x+h\\
    h\to0 &: f'(x) = 2x\\
    f'(3) &= 6
    \end{aligned}
    \]

    The \(+h\) explains the table: with \(h=0.01\) the ratio was \(6+0.01\). That leftover is the error of a finite-difference estimate, which §6.6 uses to check gradients.

### 5.3 Derivatives you will see

You do not need to derive these; recognise them and use the table.

| Function \(f(x)\) | Derivative \(f'(x)\) | Value to check | Where it appears |
|---|---|---|---|
| \(c\) (constant) | \(0\) | | bias terms vanish from some gradients |
| \(x^n\) | \(n x^{n-1}\) | \(x^3\) at 2: 12 | squared error |
| \(e^x\) | \(e^x\) | at 0: 1 | softmax |
| \(\ln x\) | \(1/x\) | at 2: 0.5 | log-likelihood, cross-entropy |
| \(\sigma(x)=\frac{1}{1+e^{-x}}\) | \(\sigma(x)(1-\sigma(x))\) | at 0: 0.25 | logistic regression, gates |
| \(\tanh x\) | \(1-\tanh^2 x\) | at 0: 1 | older activations, GeGLU's tanh approximation |
| \(\max(0,x)\) (ReLU) | 1 if \(x>0\), else 0 | at 3: 1 | activations |
| \(a f(x)+b g(x)\) | \(a f'(x)+b g'(x)\) | | losses summed over examples |
| \(f(x)\,g(x)\) | \(f'g+fg'\) | | gated layers (SwiGLU) |

The sigmoid row is worth a card, because its derivative is written in terms of its own output.

\[
\sigma'(x)=\sigma(x)\bigl(1-\sigma(x)\bigr)
\]

!!! example "On paper"

    At \(x=0\), \(\sigma(0)=0.5\); at \(x=\ln3\approx1.099\), \(\sigma=3/(3+1)=0.75\):

    \[
    \begin{aligned}
    \sigma'(0) &= 0.5\cdot0.5 = 0.25\\
    \sigma'(\ln3) &= 0.75\cdot0.25 = 0.1875
    \end{aligned}
    \]

    The slope is largest at 0 and shrinks as the output nears 0 or 1. A sigmoid that is "saturated" (output 0.999) barely passes any gradient back: the vanishing-gradient problem that [48](48-ai-deep-learning.md) returns to.

### 5.4 The chain rule

Models are functions of functions: a loss of a prediction of a weighted sum. The **chain rule** says the derivative of a chain is the product of the derivatives of its links. If \(L\) depends on \(u\) and \(u\) depends on \(w\):

\[
\frac{dL}{dw}=\frac{dL}{du}\cdot\frac{du}{dw}
\]

!!! example "On paper"

    A one-weight model predicts latency \(wx\). With \(w=2\), input \(x=3\) and true value \(y=4\), the loss is \(L=(wx-y)^2\). Let \(u=wx-y\):

    \[
    \begin{aligned}
    u &= 2\cdot3-4 = 2\\
    L &= u^2 = 4\\
    dL/du &= 2u = 4\\
    du/dw &= x = 3\\
    dL/dw &= 4\cdot3 = 12
    \end{aligned}
    \]

    Check by nudging: at \(w=2.001\), \(u=2.003\) and \(L=4.012009\), a change of 0.012009 for a nudge of 0.001, so a slope of about 12.009. Backpropagation is this multiplication, done link by link from the loss back to every weight.

Why you care: when a gradient is zero or huge, the chain rule tells you which link did it. A saturated sigmoid contributes a factor near 0; a large input contributes a large factor. "Exploding" and "vanishing" gradients are long chains of factors above or below 1.

### 5.5 Partial derivatives and the gradient

When a function has several inputs, a **partial derivative** moves one input and holds the others fixed. The **gradient** collects all of them in a vector, one entry per input. A model with 14 billion weights has a 14-billion-entry gradient.

\[
\nabla f=\left(\frac{\partial f}{\partial w_1},\ \frac{\partial f}{\partial w_2}\right)
\]

!!! example "On paper"

    \(f(w_1,w_2)=w_1^2+3w_1w_2\) at \((1,2)\), where \(f=1+6=7\):

    \[
    \begin{aligned}
    \partial f/\partial w_1 &= 2w_1+3w_2 = 2+6 = 8\\
    \partial f/\partial w_2 &= 3w_1 = 3\\
    \nabla f(1,2) &= (8,3)
    \end{aligned}
    \]

    To get \(\partial f/\partial w_1\), treat \(w_2\) as a constant and differentiate as usual. \(w_1\) has the larger effect here: moving it by 0.01 changes \(f\) by about 0.08, moving \(w_2\) by 0.01 changes it by about 0.03.

### 5.6 The gradient points uphill

The gradient points in the direction of steepest increase. A small step of size \(\eta\) against it lowers the function by about \(\eta\lVert\nabla f\rVert^2\).

\[
f(\mathbf w-\eta\nabla f)\approx f(\mathbf w)-\eta\lVert\nabla f\rVert^2
\]

!!! example "On paper"

    Same \(f\), from \((1,2)\) with \(\eta=0.01\) and \(\nabla f=(8,3)\), so \(\lVert\nabla f\rVert^2=73\):

    \[
    \begin{aligned}
    \mathbf w' &= (1-0.08,\ 2-0.03) = (0.92,1.97)\\
    f(\mathbf w') &= 0.8464+5.4372 = 6.2836\\
    \text{predicted} &: 7-0.01\cdot73 = 6.27
    \end{aligned}
    \]

    The loss fell by 0.7164 against a predicted 0.73. The prediction is a straight-line approximation; it is good for small steps and gets worse for large ones, which is the whole story of choosing a learning rate.

---

## 6. Gradient descent

### 6.1 The update rule

Gradient descent repeats one step: compute the gradient of the loss, move every weight a little against it. The size of the step is the **learning rate** \(\eta\) (eta).

\[
w_{t+1}=w_t-\eta\,\frac{\partial L}{\partial w}(w_t)
\]

!!! example "On paper"

    \(L(w)=(w-3)^2\), so \(L'(w)=2(w-3)\). Start at \(w_0=0\) with \(\eta=0.1\):

    \[
    \begin{aligned}
    L'(0) &= -6\\
    w_1 &= 0-0.1\cdot(-6) = 0.6\\
    L'(0.6) &= -4.8\\
    w_2 &= 0.6+0.48 = 1.08
    \end{aligned}
    \]

    The gradient is negative (the loss falls to the right), so the update moves right, towards the minimum at 3. Each step is smaller because the slope flattens near the bottom.

### 6.2 One dimension by hand

Keep going with the same loss and learning rate. The distance to the minimum shrinks by the same factor every step.

| step \(t\) | \(w_t\) | distance \(3-w_t\) | loss \((w_t-3)^2\) |
|---:|---:|---:|---:|
| 0 | 0 | 3 | 9 |
| 1 | 0.6 | 2.4 | 5.76 |
| 2 | 1.08 | 1.92 | 3.6864 |
| 3 | 1.464 | 1.536 | 2.3593 |
| 4 | 1.7712 | 1.2288 | 1.5099 |
| 5 | 2.01696 | 0.98304 | 0.9664 |

Each distance is 0.8 times the last, and each loss 0.64 times the last. That is a "loss curve": fast at first, then slower, never quite zero.

### 6.3 The learning rate: too small, too big

For this loss the update can be rewritten so the factor is visible:

\[
w_{t+1}-3=(1-2\eta)\,(w_t-3)
\]

!!! example "On paper"

    Plug in the learning rates from the table below:

    \[
    \begin{aligned}
    \eta=0.01 &: 1-0.02 = 0.98\\
    \eta=0.1 &: 1-0.2 = 0.8\\
    \eta=0.5 &: 1-1 = 0\\
    \eta=1.0 &: 1-2 = -1\\
    \eta=1.1 &: 1-2.2 = -1.2
    \end{aligned}
    \]

    A factor strictly between −1 and 1 converges, a factor of −1 bounces forever, beyond it the distance grows every step. So for this loss, any \(\eta\) between 0 and 1 converges and anything above 1 diverges.

| \(\eta\) | \(w_0\) | \(w_1\) | \(w_2\) | \(w_3\) | what you see in training |
|---:|---:|---:|---:|---:|---|
| 0.01 | 0 | 0.06 | 0.1188 | 0.1764 | loss barely moves; "is it even training?" |
| 0.1 | 0 | 0.6 | 1.08 | 1.464 | steady decrease |
| 0.5 | 0 | 3 | 3 | 3 | perfect in one step (only for this exact bowl) |
| 1.0 | 0 | 6 | 0 | 6 | loss stuck at 9, oscillating |
| 1.1 | 0 | 6.6 | −1.32 | 8.184 | loss grows, then `nan` |

Real losses are not perfect bowls, so you never find the "one step" learning rate. You find a range that works with a learning-rate sweep, and you watch for the two failure signatures: a flat loss curve (too small) and a loss that spikes or becomes `nan` (too big). [48](48-ai-deep-learning.md) covers schedules and warm-up.

### 6.4 Two dimensions: the narrow valley

With two weights, the steepest direction sets the largest learning rate you can use. Take \(L(w_1,w_2)=w_1^2+4w_2^2\), a bowl four times steeper along \(w_2\).

\[
\nabla L=(2w_1,\ 8w_2)
\]

!!! example "On paper"

    Start at \((4,2)\), where \(L=16+16=32\). Each step multiplies \(w_1\) by \(1-2\eta\) and \(w_2\) by \(1-8\eta\):

    \[
    \begin{aligned}
    \eta=0.1 &: \text{factors } 0.8 \text{ and } 0.2\\
    \text{step 1} &: (3.2,\ 0.4),\ L = 10.88\\
    \text{step 2} &: (2.56,\ 0.08),\ L = 6.5792\\
    \eta=0.3 &: \text{factors } 0.4 \text{ and } -1.4\\
    \text{step 1} &: (1.6,\ -2.8),\ L = 33.92
    \end{aligned}
    \]

    At \(\eta=0.3\) the gentle direction improves faster but the steep direction overshoots and grows, so the total loss rises. The steep axis caps \(\eta\) below \(2/8=0.25\). Just under that cap the gentle axis still shrinks by about half per step, but in a valley 100 times steeper along one axis the cap is \(2/200=0.01\) and the gentle axis crawls, by a factor of about 0.98 per step. Adaptive optimizers such as Adam exist largely to fix this.

### 6.5 Gradient descent in code

```python
def gd(lr, steps=5, w=0.0):
    path = [w]
    for _ in range(steps):
        grad = 2 * (w - 3)        # dL/dw for L(w) = (w - 3)^2
        w = w - lr * grad
        path.append(w)
    return path

for lr in (0.01, 0.1, 0.5, 1.0, 1.1):
    print(f"lr={lr:<4}", " ".join(f"{w:7.3f}" for w in gd(lr)))
```

```text
lr=0.01   0.000   0.060   0.119   0.176   0.233   0.288
lr=0.1    0.000   0.600   1.080   1.464   1.771   2.017
lr=0.5    0.000   3.000   3.000   3.000   3.000   3.000
lr=1.0    0.000   6.000   0.000   6.000   0.000   6.000
lr=1.1    0.000   6.600  -1.320   8.184  -3.221  10.465
```

And the two-dimensional valley:

```python
import numpy as np

def loss(w):
    return w[0] ** 2 + 4 * w[1] ** 2

def grad(w):
    return np.array([2 * w[0], 8 * w[1]])

for lr in (0.1, 0.3):
    w = np.array([4.0, 2.0])
    print(f"lr={lr}")
    for step in range(4):
        print(f"  step {step}: w=({w[0]:7.3f}, {w[1]:7.3f})  loss={loss(w):8.3f}")
        w = w - lr * grad(w)
```

```text
lr=0.1
  step 0: w=(  4.000,   2.000)  loss=  32.000
  step 1: w=(  3.200,   0.400)  loss=  10.880
  step 2: w=(  2.560,   0.080)  loss=   6.579
  step 3: w=(  2.048,   0.016)  loss=   4.195
lr=0.3
  step 0: w=(  4.000,   2.000)  loss=  32.000
  step 1: w=(  1.600,  -2.800)  loss=  33.920
  step 2: w=(  0.640,   3.920)  loss=  61.875
  step 3: w=(  0.256,  -5.488)  loss= 120.538
```

### 6.6 Checking a gradient numerically

When you write a gradient by hand (a custom loss, a custom layer), check it against **finite differences**: nudge each weight both ways and measure the change in the loss. The central difference has an error proportional to \(h^2\), much smaller than the one-sided version's \(h\).

\[
f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}
\]

!!! example "On paper"

    \(f(x)=x^3\) at \(x=2\), where the true slope is \(3\cdot2^2=12\), with \(h=0.01\):

    \[
    \begin{aligned}
    f(2.01) &= 8.120601\\
    f(1.99) &= 7.880599\\
    \text{central} &: 0.240002/0.02 = 12.0001\\
    \text{one-sided} &: 0.120601/0.01 = 12.0601
    \end{aligned}
    \]

    The central estimate is off by 0.0001, the one-sided one by 0.06. Use the central form, with \(h\) around \(10^{-5}\) in float64.

Here is the check applied to the mean squared error of a linear model, together with a deliberately buggy gradient that forgot the 2 from the square:

```python
import numpy as np

rng = np.random.default_rng(0)
X, y, w = rng.normal(size=(5, 3)), rng.normal(size=5), rng.normal(size=3)

def loss(w):
    return np.mean((X @ w - y) ** 2)

def grad_analytic(w):             # chain rule: 2/n * X^T (Xw - y)
    return 2 * X.T @ (X @ w - y) / len(y)

def grad_buggy(w):                # forgot the 2 from the square
    return X.T @ (X @ w - y) / len(y)

def grad_numeric(f, w, h=1e-5):   # central differences, one coordinate at a time
    g = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w)
        e[i] = h
        g[i] = (f(w + e) - f(w - e)) / (2 * h)
    return g

num = grad_numeric(loss, w)
for name, g in [("analytic", grad_analytic(w)), ("buggy", grad_buggy(w))]:
    rel = np.linalg.norm(g - num) / np.linalg.norm(g + num)
    print(f"{name:8s} relative error = {rel:.1e}")
```

```text
analytic relative error = 3.3e-12
buggy    relative error = 3.3e-01
```

A relative error of \(10^{-12}\) means the gradient is right (in float64, anything below about \(10^{-7}\) is fine); anything above about \(10^{-4}\) means a bug. The buggy gradient scores exactly 1/3, because \(\lVert g/2-g\rVert/\lVert g/2+g\rVert=0.5/1.5\). PyTorch ships the same check as `torch.autograd.gradcheck`. It is a unit test for calculus.

### 6.7 Stochastic and mini-batch gradients

The true loss is an average over the whole training set, but computing its gradient over millions of examples for every step is too slow. **Stochastic gradient descent** estimates the gradient from a small random **mini-batch** (8 to a few thousand examples) instead. The estimate is noisy but unbiased: on average it points the right way. Bigger batches give less noisy gradients at a higher cost per step. In LoRA fine-tuning on a 32 GB Mac, `batch_size` and gradient accumulation are this trade-off ([08](08-finetuning.md), [48](48-ai-deep-learning.md)).

---

## 7. Exponentials and logarithms

### 7.1 The rules you need

\(e\approx2.71828\). The exponential \(e^x\) turns sums into products; the natural logarithm \(\ln x\) is its inverse and turns products into sums.

| Rule | Example |
|---|---|
| \(e^{a+b}=e^a e^b\) | \(e^{\ln2+\ln3}=2\cdot3=6\) |
| \(\ln(ab)=\ln a+\ln b\) | \(\ln 6=0.693+1.099=1.792\) |
| \(\ln(a/b)=\ln a-\ln b\) | \(\ln 0.5=-\ln 2=-0.693\) |
| \(\ln(a^k)=k\ln a\) | \(\ln 8=3\ln2=2.079\) |
| \(\ln 1=0\), \(e^0=1\) | a probability of 1 has log 0 |
| \(\ln p<0\) for \(0<p<1\) | log-probabilities are negative |
| \(\log_2 x=\ln x/\ln 2\) | \(\log_2 8=2.079/0.693=3\) |

The rule that matters most for ML is the product rule of logs:

\[
\ln(p_1p_2\cdots p_n)=\ln p_1+\ln p_2+\dots+\ln p_n
\]

!!! example "On paper"

    A model gives two tokens the probabilities 0.5 and 0.25:

    \[
    \begin{aligned}
    \ln(0.5\cdot0.25) &= \ln0.125 = -2.079\\
    \ln0.5+\ln0.25 &= -0.693-1.386 = -2.079
    \end{aligned}
    \]

    Same number. Multiplying many probabilities becomes adding many log-probabilities, which is what every language-model loss does.

### 7.2 Why we work in log space

The probability of a 200-token answer is the product of 200 per-token probabilities, so it shrinks exponentially with length. At a generous 0.6 per token the product is about \(4\times10^{-45}\), already below the smallest normal float32 (about \(1.2\times10^{-38}\)); at 0.01 per token it is \(10^{-400}\), below anything float64 can hold.

```python
import numpy as np

p = np.full(200, 0.01)                  # 200 tokens, each with probability 0.01
print("product   :", np.prod(p))         # 1e-400 is below the smallest float64
print("sum of log:", np.sum(np.log(p)))
print("per token :", np.mean(np.log(p)))
```

```text
product   : 0.0
sum of log: -921.0340371976183
per token : -4.605170185988092
```

The product **underflows** to exactly 0.0: every answer would look equally impossible. The sum of logs is an ordinary number. This is why APIs return `logprobs` instead of probabilities, why the training loss is a mean of negative log-probabilities, and why you compare two answers by their summed (or averaged) log-probabilities, never by multiplied probabilities.

### 7.3 The log-sum-exp trick

Softmax and cross-entropy need \(\ln\sum_j e^{z_j}\). With large logits, \(e^{z_j}\) **overflows** (\(e^{1000}\) is larger than any float). Subtracting the largest logit first gives the same answer with no overflow, because \(e^{z-m}=e^z/e^m\).

\[
\ln\sum_j e^{z_j}=m+\ln\sum_j e^{z_j-m},
\quad m=\max_j z_j
\]

!!! example "On paper"

    \(z=(1000,1001,1002)\), so \(m=1002\), with \(e^{-1}\approx0.3679\) and \(e^{-2}\approx0.1353\):

    \[
    \begin{aligned}
    \textstyle\sum_j e^{z_j-m} &= 0.1353+0.3679+1 = 1.5032\\
    \ln1.5032 &= 0.4076\\
    \text{LSE} &= 1002.4076
    \end{aligned}
    \]

    The softmax probabilities are \(e^{z_j-\text{LSE}}\): 0.0900, 0.2447 and 0.6652. Only the gaps between logits matter, never their absolute size. [38 §1.3](38-derivations.md#13-numerical-stability) proves it.

```python
import numpy as np

z = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(z) / np.exp(z).sum()    # inf / inf
print("naive  :", naive)

m = z.max()
lse = m + np.log(np.exp(z - m).sum())
print("lse    :", round(lse, 4))
print("softmax:", np.round(np.exp(z - lse), 4))
```

```text
naive  : [nan nan nan]
lse    : 1002.4076
softmax: [0.09   0.2447 0.6652]
```

Every serious library does this internally (`torch.logsumexp`, `torch.nn.functional.cross_entropy`). You need it when you write your own scoring code, for example when you sum `logprobs` from an API or compute a softmax over reranker scores.

### 7.4 Nats and bits

A log in base \(e\) measures information in **nats**; base 2 measures it in **bits**. One nat is \(1/\ln2\approx1.443\) bits. PyTorch losses are in nats. Information-theory examples are often in bits because the numbers come out whole. Convert by dividing by \(\ln 2\) (nats to bits) or multiplying by it (bits to nats), and always say which unit a loss is in.

---

## 8. Probability

### 8.1 Events and probabilities

A **probability** is a number between 0 and 1 attached to an **event**, something that can happen or not: "the next ticket is urgent", "the model's JSON parses", "the request takes more than 2 seconds". You estimate it from counts. Of 1,000 support tickets last week, 200 were urgent, so \(P(\text{urgent})\approx0.2\). Two basic rules follow from counting: the complement \(P(\text{not }A)=1-P(A)\), and for two events, \(P(A\text{ or }B)=P(A)+P(B)-P(A\text{ and }B)\), because the overlap would otherwise be counted twice.

Why you care: an LLM is a machine that outputs probabilities (one per possible next token), and a product built on it is a system whose behaviour you can only describe with rates: "valid JSON 99.2% of the time", "hallucinated citation in 3% of answers".

### 8.2 Joint and conditional probability

The **joint** probability \(P(A,B)\) is the chance both happen. The **conditional** probability \(P(A\mid B)\) is the chance of \(A\) among the cases where \(B\) happened: you shrink the world to \(B\) and count again.

\[
P(A\mid B)=\frac{P(A,B)}{P(B)}
\]

!!! example "On paper"

    Of the 1,000 tickets, 300 came by phone, and 120 were both urgent and by phone:

    \[
    \begin{aligned}
    P(\text{urgent}) &= 200/1000 = 0.2\\
    P(\text{phone}) &= 300/1000 = 0.3\\
    P(\text{urgent},\text{phone}) &= 120/1000 = 0.12\\
    P(\text{urgent}\mid\text{phone}) &= 0.12/0.3 = 0.4
    \end{aligned}
    \]

    Knowing the channel doubles the urgency rate, from 0.2 to 0.4. "Channel" is therefore a useful feature for an urgency classifier. And \(P(\text{urgent or phone})=0.2+0.3-0.12=0.38\).

### 8.3 Independence

Two events are **independent** if knowing one tells you nothing about the other: \(P(A\mid B)=P(A)\), or equivalently the joint probability is the product.

\[
P(A,B)=P(A)\,P(B)\quad\text{(only if independent)}
\]

!!! example "On paper"

    Channel and urgency from §8.2, then two model calls that fail independently with probability 0.1 each:

    \[
    \begin{aligned}
    P(\text{urgent})P(\text{phone}) &= 0.2\cdot0.3 = 0.06\\
    P(\text{urgent},\text{phone}) &= 0.12 \ne 0.06\\
    P(\text{both calls fail}) &= 0.1\cdot0.1 = 0.01
    \end{aligned}
    \]

    Urgency and channel are dependent. A retry cuts the failure rate from 10% to 1% only if failures are independent; if they come from an overloaded server, the retry fails too.

Pitfall: assuming independence is the most common modelling mistake in reliability math. Retries and judge votes are usually positively correlated (the input or the outage that defeats one attempt tends to defeat the next), so multiplying their failure probabilities overstates how much redundancy helps. Agent steps are correlated too, so the per-step compounding math of [52](52-ai-agents.md) is an approximation; measure end-to-end success on real tasks.

### 8.4 Bayes' rule: the fraud alert

Bayes' rule turns a conditional around: from "how often does the detector fire on fraud?" to "how often is a fired alert really fraud?".

\[
P(F\mid +)=\frac{P(+\mid F)\,P(F)}{P(+)}
\]

!!! example "On paper"

    1% of transactions are fraud (\(F\)). The detector flags 90% of fraud and 5% of legitimate transactions. How likely is a flagged one to be fraud?

    \[
    \begin{aligned}
    P(+\mid F)P(F) &= 0.9\cdot0.01 = 0.009\\
    P(+\mid\text{legit})P(\text{legit}) &= 0.05\cdot0.99 = 0.0495\\
    P(+) &= 0.009+0.0495 = 0.0585\\
    P(F\mid+) &= 0.009/0.0585 \approx 0.154
    \end{aligned}
    \]

    Only about 15% of alerts are fraud, although the detector "catches 90%". The rare class is swamped by the 5% false alarms on the huge legitimate class.

The same numbers as counts are easier to explain to a product manager. Out of 10,000 transactions:

| | flagged | not flagged | total |
|---|---:|---:|---:|
| fraud | 90 | 10 | 100 |
| legitimate | 495 | 9,405 | 9,900 |
| total | 585 | 9,415 | 10,000 |

90 of the 585 flags are fraud: \(90/585\approx0.154\). This is the **base-rate fallacy**, and it is why precision collapses for rare classes ([47](47-ai-classical-ml.md) on imbalance). A simulation confirms it:

```python
import numpy as np

rng = np.random.default_rng(42)
n = 1_000_000
fraud = rng.random(n) < 0.01                                   # 1% base rate
flag = np.where(fraud, rng.random(n) < 0.90, rng.random(n) < 0.05)

print("flagged:", flag.sum())
print("P(fraud | flag) ~", round(fraud[flag].mean(), 4))
```

```text
flagged: 58688
P(fraud | flag) ~ 0.1533
```

### 8.5 Random variables and expectation

A **random variable** is a number whose value depends on chance: the number of output tokens of the next response, the latency of the next request, 1 if the next answer is correct and 0 if not. Its **expectation** \(\mathbb E[X]\) is the probability-weighted average of its values: the long-run mean.

\[
\mathbb E[X]=\sum_x x\,P(X=x)
\]

!!! example "On paper"

    A support bot writes short answers (100 tokens) half the time, medium ones (200) 30% of the time and long ones (600) 20% of the time:

    \[
    \begin{aligned}
    \mathbb E[X] &= 100\cdot0.5+200\cdot0.3+600\cdot0.2\\
    &= 50+60+120 = 230 \text{ tokens}
    \end{aligned}
    \]

    Budget 230 output tokens per answer, not the "typical" 100. The rare long answers carry more than half the cost (120 of 230), which is why `max_tokens` and answer-length instructions matter for the bill.

Expectation is linear: \(\mathbb E[aX+bY]=a\,\mathbb E[X]+b\,\mathbb E[Y]\), whether or not \(X\) and \(Y\) are independent. 1,000 answers are expected to use 230,000 tokens.

### 8.6 Variance and standard deviation

The **variance** measures spread: the expected squared distance from the mean. The **standard deviation** (sd, \(\sigma\)) is its square root, back in the original units.

\[
\operatorname{Var}(X)=\mathbb E[X^2]-\bigl(\mathbb E[X]\bigr)^2
\]

!!! example "On paper"

    The same token distribution:

    \[
    \begin{aligned}
    \mathbb E[X^2] &= 10^4\cdot0.5+4\cdot10^4\cdot0.3+36\cdot10^4\cdot0.2\\
    &= 5000+12000+72000 = 89000\\
    \operatorname{Var}(X) &= 89000-230^2 = 36100\\
    \sigma &= \sqrt{36100} = 190
    \end{aligned}
    \]

    A standard deviation of 190 tokens on a mean of 230 says single answers are very unpredictable. Plan capacity from the distribution's tail, not from the mean.

### 8.7 Sums and averages: why the square root of n

For \(n\) independent draws, variances add, so the standard deviation of the sum grows like \(\sqrt n\) and the standard deviation of the average shrinks like \(1/\sqrt n\).

\[
\operatorname{sd}(\bar X)=\frac{\sigma}{\sqrt n}
\]

!!! example "On paper"

    100 independent answers from the distribution above (\(\sigma=190\)):

    \[
    \begin{aligned}
    \text{sd of the total} &= 190\cdot\sqrt{100} = 1900\\
    \text{sd of the average} &= 190/\sqrt{100} = 19
    \end{aligned}
    \]

    The total is 23,000 ± 1,900 tokens: relatively tight (8%), although single answers vary by 83%. Averages are predictable even when items are not. The same rule gives the standard error in §13.3.

---

## 9. Distributions you will meet

A **distribution** lists the possible values of a random variable and their probabilities. Five cover most of what an AI engineer sees.

### 9.1 Bernoulli: one yes-or-no

One trial that succeeds with probability \(p\): "is this answer correct?", "did the JSON parse?". Values 1 or 0.

\[
\mathbb E[X]=p,
\quad
\operatorname{Var}(X)=p(1-p)
\]

!!! example "On paper"

    One eval item that the model gets right 80% of the time:

    \[
    \begin{aligned}
    \mathbb E[X] &= 0.8\\
    \operatorname{Var}(X) &= 0.8\cdot0.2 = 0.16\\
    \sigma &= 0.4
    \end{aligned}
    \]

    The variance is largest at \(p=0.5\) (0.25) and zero at 0 or 1. That is why eval scores near 50% are the noisiest to measure.

### 9.2 Binomial: how many successes in n tries

The number of successes in \(n\) independent Bernoulli trials: "how many of 10 eval tasks pass?".

\[
P(K=k)=\binom nk p^k(1-p)^{n-k}
\]

!!! example "On paper"

    A model with a true pass rate of 0.8 on a 10-task eval; \(\binom{10}{8}=45\) and \(0.8^8\approx0.1678\):

    \[
    \begin{aligned}
    P(K=8) &= 45\cdot0.8^8\cdot0.2^2\\
    &= 45\cdot0.1678\cdot0.04 \approx 0.302
    \end{aligned}
    \]

    A model whose true rate is 80% scores exactly 8/10 only 30% of the time, and 6/10 or worse about 12% of the time. Ten tasks cannot tell a 70% model from an 80% one. Mean \(np=8\), sd \(\sqrt{np(1-p)}\approx1.26\) tasks.

### 9.3 Categorical: one of k outcomes

One draw from \(k\) outcomes with probabilities that sum to 1. The next-token distribution of a language model is categorical over the whole vocabulary (151,936 outcomes for Qwen3). So is a ticket router's output over "billing", "technical" and "other". Sampling with a temperature (§10.2) is drawing from a categorical distribution.

### 9.4 Uniform: every value equally likely

Every value between \(a\) and \(b\) is equally likely. You meet it as random jitter in retry backoff, as random initialisation, and as the "no idea" baseline.

\[
\mathbb E[X]=\frac{a+b}{2},
\quad
\operatorname{Var}(X)=\frac{(b-a)^2}{12}
\]

!!! example "On paper"

    Retry jitter drawn uniformly between 0 and 2 seconds:

    \[
    \begin{aligned}
    \mathbb E[X] &= (0+2)/2 = 1\text{ s}\\
    \operatorname{Var}(X) &= 4/12 \approx 0.333\\
    \sigma &\approx 0.577\text{ s}
    \end{aligned}
    \]

    Jitter spreads retries of many clients over two seconds instead of letting them hit the server at the same instant.

### 9.5 Normal: the bell curve

The **normal** (Gaussian) distribution has a mean \(\mu\) and a standard deviation \(\sigma\). It matters because averages of many independent things are approximately normal (the **central limit theorem**), which is what makes confidence intervals work. About 68% of values fall within 1 sd of the mean, 95% within 2 (precisely 1.96), 99.7% within 3. A **z-score** says how many standard deviations a value is from the mean.

\[
z=\frac{x-\mu}{\sigma}
\]

!!! example "On paper"

    Average daily cost is \(\mu=\$800\) with \(\sigma=\$100\), and today cost $1,000:

    \[
    \begin{aligned}
    z &= (1000-800)/100 = 2\\
    P(Z>2) &\approx 0.023
    \end{aligned}
    \]

    A day this expensive happens about 2.3% of the time by chance, roughly once every six weeks. A z-score above 3 (0.13%) is a better alert threshold if you do not want to be paged every six weeks by noise.

### 9.6 Heavy tails: latency and token counts are not normal

Latencies, response lengths and costs per user are **right-skewed**: most values are small, a few are huge, and nothing is below zero. A log-normal distribution fits them better than a normal one. For such data the mean sits well above the median, and "mean ± 2 sd" is meaningless (it can go negative). Report percentiles instead (§13.1).

### 9.7 Sampling and checking in code

Draw 100,000 samples from each distribution and compare the sample mean and variance with the formulas above:

```python
import numpy as np

rng = np.random.default_rng(7)
n = 100_000
samples = {
    "bernoulli p=0.8": (rng.random(n) < 0.8).astype(float),   # theory 0.8, 0.16
    "binomial n=10 p=0.8": rng.binomial(10, 0.8, n),          # theory 8, 1.6
    "uniform(0, 2)": rng.uniform(0, 2, n),                    # theory 1, 0.333
    "normal(800, 100)": rng.normal(800, 100, n),              # theory 800, 10000
}
for name, x in samples.items():
    print(f"{name:20s} mean={x.mean():9.3f}  var={x.var():10.3f}")

tokens = rng.choice(3, size=n, p=[0.6, 0.3, 0.1])             # categorical
print("categorical frequencies:", np.bincount(tokens) / n)

lat = rng.lognormal(mean=np.log(300), sigma=0.8, size=n)      # heavy-tailed latency (ms)
print(f"lognormal latency: median={np.median(lat):.0f} mean={lat.mean():.0f} "
      f"p99={np.percentile(lat, 99):.0f}")
```

```text
bernoulli p=0.8      mean=    0.799  var=     0.160
binomial n=10 p=0.8  mean=    7.999  var=     1.586
uniform(0, 2)        mean=    1.001  var=     0.334
normal(800, 100)     mean=  800.115  var= 10000.864
categorical frequencies: [0.60014 0.2995  0.10036]
lognormal latency: median=302 mean=416 p99=1969
```

The samples land within about 1% of the formulas. The log-normal latency has a median near 300 ms, a mean of 416 ms and a p99 of almost 2 seconds, six and a half times the median: a mean-based dashboard would hide what your slowest users see.

---

## 10. Softmax

### 10.1 From scores to probabilities

A classifier or a language model ends with one raw score per class, called a **logit**. **Softmax** turns logits into probabilities: exponentiate (so everything is positive), then divide by the total (so it sums to 1).

\[
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

!!! example "On paper"

    A ticket router outputs logits \(z=(\ln6,\ln3,0)\approx(1.792,1.099,0)\) for billing, technical and other:

    \[
    \begin{aligned}
    e^{z} &= (6,3,1)\\
    \textstyle\sum_j e^{z_j} &= 10\\
    p &= (0.6,0.3,0.1)
    \end{aligned}
    \]

    Billing wins with 60%. Adding the same number to every logit leaves \(p\) unchanged, because it multiplies every \(e^{z_j}\) and the total by the same factor.

### 10.2 Temperature

Dividing the logits by a **temperature** \(T\) before the softmax sharpens (\(T<1\)) or flattens (\(T>1\)) the distribution. This is the `temperature` parameter of every chat API.

\[
p_i(T)=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
\]

!!! example "On paper"

    The same logits at \(T=0.5\): dividing by 0.5 doubles them, which squares the weights:

    \[
    \begin{aligned}
    e^{2z} &= (36,9,1)\\
    \textstyle\sum &= 46\\
    p(0.5) &= (0.783,0.196,0.022)
    \end{aligned}
    \]

    The top class rises from 0.6 to 0.78 and "other" almost vanishes. At \(T=2\) the weights become square roots, \((2.449,1.732,1)\), and \(p\approx(0.473,0.334,0.193)\). As \(T\to0\) sampling becomes greedy (always the top token).

### 10.3 Sigmoid is softmax with two classes

For two classes with logits \(z\) and 0, softmax reduces to the **sigmoid** function. That is why a binary classifier outputs one number.

\[
\frac{e^{z}}{e^{z}+e^{0}}=\frac{1}{1+e^{-z}}=\sigma(z)
\]

!!! example "On paper"

    A spam classifier outputs \(z=\ln3\approx1.099\):

    \[
    \begin{aligned}
    \sigma(\ln3) &= 3/(3+1) = 0.75\\
    \sigma(-\ln3) &= 1/(1+3) = 0.25
    \end{aligned}
    \]

    The logit is the log-odds: \(\ln(0.75/0.25)=\ln3\). A logit of 0 means 50/50, each +0.693 doubles the odds.

### 10.4 A stable softmax in code

```python
import numpy as np

def softmax(z, T=1.0):
    z = np.asarray(z, dtype=float) / T
    z = z - z.max()                       # shift: same result, no overflow
    e = np.exp(z)
    return e / e.sum()

z = np.log([6.0, 3.0, 1.0])               # billing, technical, other
for T in (0.5, 1.0, 2.0):
    print(f"T={T}:", np.round(softmax(z, T), 4))
print("z+10 :", np.round(softmax(z + 10), 4))
print("big  :", np.round(softmax([1000, 1001, 1002]), 4))
```

```text
T=0.5: [0.7826 0.1957 0.0217]
T=1.0: [0.6 0.3 0.1]
T=2.0: [0.4727 0.3343 0.193 ]
z+10 : [0.6 0.3 0.1]
big  : [0.09   0.2447 0.6652]
```

The full derivation of the softmax Jacobian and the cross-entropy gradient \(p-e_y\) is in [38 §1](38-derivations.md#1-cross-entropy-and-its-gradient), with numbers in [43 §1–3](43-math-examples.md#1-softmax-and-cross-entropy).

---

## 11. Likelihood

### 11.1 The likelihood of data under a model

The **likelihood** asks: if the model's parameter had this value, how probable is the data we actually saw? It is the same formula as a probability, read the other way round: data fixed, parameter varying.

\[
L(p)=p^{k}(1-p)^{n-k}
\]

!!! example "On paper"

    A prompt passed 7 of 10 runs, in a particular order. Compare two hypotheses for its true pass rate, with \(0.7^7\approx0.0824\) and \(0.3^3=0.027\):

    \[
    \begin{aligned}
    L(0.5) &= 0.5^{10} \approx 0.000977\\
    L(0.7) &= 0.0824\cdot0.027 \approx 0.00222
    \end{aligned}
    \]

    The data is about 2.3 times as likely under \(p=0.7\). Both numbers are tiny because any particular sequence of 10 outcomes is unlikely; only the comparison means something.

### 11.2 Log-likelihood and its maximum

Taking logs turns the product into a sum and keeps the same maximum. The **maximum likelihood estimate** (MLE) is the parameter value that makes the data most probable; you find it where the derivative is zero.

\[
\ell(p)=k\ln p+(n-k)\ln(1-p)
\]

!!! example "On paper"

    With \(k=7\), \(n=10\):

    \[
    \begin{aligned}
    \ell'(p) &= 7/p-3/(1-p) = 0\\
    7(1-p) &= 3p \Rightarrow p = 0.7\\
    \ell(0.7) &= 7(-0.3567)+3(-1.2040) = -6.109\\
    \ell(0.5) &= 10(-0.6931) = -6.931
    \end{aligned}
    \]

    The MLE of a pass rate is the observed fraction, \(k/n\), which you would have guessed. The method matters because it works the same way for a model with 14 billion parameters.

### 11.3 Negative log-likelihood is the loss

Training maximises the log-likelihood of the training data. Optimizers minimise, so libraries minimise its negative, the **negative log-likelihood** (NLL), usually averaged per example or per token. For a language model the NLL of the correct next token is exactly the cross-entropy loss of §12.3 ([32 §3](32-mathematics.md#3-the-autoregressive-objective) writes it for a sequence).

```python
import numpy as np

k, n = 7, 10                                  # 7 passes in 10 runs
p = np.round(np.arange(0.1, 1.0, 0.1), 1)     # candidate pass rates
loglik = k * np.log(p) + (n - k) * np.log(1 - p)
for pi, ll in zip(p, loglik):
    print(f"p={pi:.1f}  log-likelihood={ll:7.3f}  NLL={-ll:6.3f}")
print("maximum at p =", p[np.argmax(loglik)])
```

```text
p=0.1  log-likelihood=-16.434  NLL=16.434
p=0.2  log-likelihood=-11.935  NLL=11.935
p=0.3  log-likelihood= -9.498  NLL= 9.498
p=0.4  log-likelihood= -7.947  NLL= 7.947
p=0.5  log-likelihood= -6.931  NLL= 6.931
p=0.6  log-likelihood= -6.325  NLL= 6.325
p=0.7  log-likelihood= -6.109  NLL= 6.109
p=0.8  log-likelihood= -6.390  NLL= 6.390
p=0.9  log-likelihood= -7.645  NLL= 7.645
maximum at p = 0.7
```

---

## 12. Entropy, cross-entropy and KL divergence

These three numbers measure surprise. They explain what a language model's loss is, why it never reaches zero, and what distillation optimises.

### 12.1 Surprise

The **surprise** (information content) of an outcome with probability \(p\) is \(-\log p\): certain events carry no surprise, rare ones carry a lot.

\[
I(x)=-\log_2 p(x)
\]

!!! example "On paper"

    Three outcomes with probabilities 1/2, 1/8 and 1:

    \[
    \begin{aligned}
    -\log_2(1/2) &= 1 \text{ bit}\\
    -\log_2(1/8) &= 3 \text{ bits}\\
    -\log_2 1 &= 0 \text{ bits}
    \end{aligned}
    \]

    One bit is the surprise of a fair coin flip. A token the model gave 1/8 is as surprising as three coin flips coming up heads.

### 12.2 Entropy

The **entropy** of a distribution is its expected surprise: how uncertain the outcome is on average.

\[
H(p)=-\sum_i p_i\log_2 p_i
\]

!!! example "On paper"

    A next-token distribution \(p=(0.5,0.25,0.25)\):

    \[
    \begin{aligned}
    H(p) &= 0.5\cdot1+0.25\cdot2+0.25\cdot2\\
    &= 1.5 \text{ bits}
    \end{aligned}
    \]

    A uniform distribution over 4 tokens has 2 bits, a certain one 0 bits. High-entropy next-token distributions are where sampling settings (temperature, top-p) change the output most.

### 12.3 Cross-entropy

The **cross-entropy** \(H(p,q)\) is the average surprise when outcomes follow \(p\) but you predict with \(q\). It is never smaller than \(H(p)\); predicting with the wrong distribution costs extra surprise.

\[
H(p,q)=-\sum_i p_i\log_2 q_i
\]

!!! example "On paper"

    The truth is \(p=(0.5,0.25,0.25)\) and the model predicts \(q=(0.25,0.5,0.25)\):

    \[
    \begin{aligned}
    H(p,q) &= 0.5\cdot2+0.25\cdot1+0.25\cdot2\\
    &= 1.75 \text{ bits}
    \end{aligned}
    \]

    1.75 bits against the 1.5 bits of \(H(p)\): the model pays 0.25 bits per outcome for its mistake. When the truth is one-hot (the training text says the next token is `self`), \(H(p,q)=-\log q_{\text{self}}\): the familiar cross-entropy loss, which libraries compute with \(\ln\), in nats.

### 12.4 KL divergence

The **Kullback–Leibler divergence** is the extra surprise: cross-entropy minus entropy. It is zero only when \(q=p\), and it is not symmetric, so it is not a distance.

\[
\operatorname{KL}(p\,\Vert\,q)=\sum_i p_i\log_2\frac{p_i}{q_i}=H(p,q)-H(p)
\]

!!! example "On paper"

    The same \(p\) and \(q\):

    \[
    \begin{aligned}
    \operatorname{KL} &= 0.5\log_2 2+0.25\log_2\tfrac12+0.25\log_2 1\\
    &= 0.5-0.25+0 = 0.25 \text{ bits}\\
    H(p,q)-H(p) &= 1.75-1.5 = 0.25
    \end{aligned}
    \]

    Both routes agree. For \(a=(0.9,0.1)\) and \(b=(0.5,0.5)\), \(\operatorname{KL}(a\Vert b)\approx0.531\) bits but \(\operatorname{KL}(b\Vert a)\approx0.737\): the order matters.

Why you care: distillation trains a student to minimise the KL from the teacher's distribution ([39](39-gemma-distillation.md)); preference tuning keeps a KL penalty to stop the model drifting from its starting point ([15](15-agentic-methods.md), [53](53-ai-fine-tuning.md)).

### 12.5 What this means for a language model's loss and perplexity

A language model's training loss is the average cross-entropy, in nats, of the true next token under the model's prediction. **Perplexity** is \(e^{\text{loss}}\) (or \(2^{\text{bits}}\)): the number of equally likely tokens the model is, on average, choosing between.

\[
\text{PPL}=\exp\Bigl(-\tfrac1T\textstyle\sum_{t}\ln q(x_t)\Bigr)
\]

!!! example "On paper"

    Four target tokens get probabilities 1/2, 1/4, 1/2 and 1/8, so surprises of 1, 2, 1 and 3 bits:

    \[
    \begin{aligned}
    \text{mean} &= (1+2+1+3)/4 = 1.75 \text{ bits}\\
    \text{loss} &= 1.75\ln2 \approx 1.213 \text{ nats}\\
    \text{PPL} &= 2^{1.75} \approx 3.364
    \end{aligned}
    \]

    The model is as unsure as a fair pick among 3.4 tokens. A loss of 2.0 nats would be a perplexity of \(e^2\approx7.39\).

Three consequences you will use:

- **The loss on new text cannot reach zero.** Cross-entropy = entropy of the text + KL. Real text has irreducible uncertainty (many good next words), so only the KL part can be trained away. A training loss near zero on a small fine-tuning set means the model memorised it, not that it got perfect.
- **Perplexity only compares models with the same tokenizer.** A tokenizer that splits text into more, shorter tokens changes the per-token loss without changing quality.
- **A falling training loss with a rising validation loss is overfitting**, the subject of [47](47-ai-classical-ml.md). For a fine-tune, watch the validation loss ([08](08-finetuning.md)).

### 12.6 The same numbers in code

```python
import numpy as np

def entropy(p):
    return -np.sum(p * np.log2(p))

def cross_entropy(p, q):
    return -np.sum(p * np.log2(q))

def kl(p, q):
    return np.sum(p * np.log2(p / q))

p, q = np.array([0.5, 0.25, 0.25]), np.array([0.25, 0.5, 0.25])
print("H(p)     =", entropy(p), " H(p,q) =", cross_entropy(p, q), " KL =", kl(p, q))

a, b = np.array([0.9, 0.1]), np.array([0.5, 0.5])
print("KL(a||b) =", round(kl(a, b), 3), " KL(b||a) =", round(kl(b, a), 3))

p_correct = np.array([0.5, 0.25, 0.5, 0.125])   # model's probability of each true token
nll = -np.log(p_correct)                        # nats, as PyTorch reports
print("loss =", round(nll.mean(), 4), "nats/token  perplexity =", round(np.exp(nll.mean()), 4))
```

```text
H(p)     = 1.5  H(p,q) = 1.75  KL = 0.25
KL(a||b) = 0.531  KL(b||a) = 0.737
loss = 1.213 nats/token  perplexity = 3.3636
```

---

## 13. Statistics for engineers

You will spend more time judging numbers than computing gradients: is p95 latency worse after the deploy? Is prompt B better, or was it luck? These are the tools.

### 13.1 Mean, median and percentiles

The **mean** is the sum divided by the count. The **median** is the middle value when sorted. The **p-th percentile** is the value below which p% of observations fall; p50 is the median, and p95 and p99 describe the slow tail that users complain about.

\[
\bar x=\frac1n\sum_{i=1}^{n}x_i
\]

!!! example "On paper"

    Ten request latencies in ms, sorted: 110, 120, 120, 130, 140, 150, 160, 180, 250, 2400. One request hit a cold cache:

    \[
    \begin{aligned}
    \textstyle\sum x_i &= 3760\\
    \bar x &= 376\text{ ms}\\
    \text{median} &= (140+150)/2 = 145\text{ ms}\\
    \text{p90 (nearest rank 9)} &= 250\text{ ms}
    \end{aligned}
    \]

    One outlier pulled the mean to 376 ms, more than twice the median. Nine of ten users waited 250 ms or less. Report p50 and p95/p99, and keep the mean for cost (which does add up).

Percentiles have several definitions, and tools disagree on small samples:

```python
import numpy as np

ms = np.array([110, 120, 120, 130, 140, 150, 160, 180, 250, 2400])
print("mean  :", ms.mean(), " median:", np.median(ms))
for q in (50, 90, 95, 99):
    lin = np.percentile(ms, q)                          # numpy default: interpolate
    nr = np.percentile(ms, q, method="inverted_cdf")    # nearest rank
    print(f"p{q}: linear={lin:7.1f}  nearest-rank={nr:6.0f}")
```

```text
mean  : 376.0  median: 145.0
p50: linear=  145.0  nearest-rank=   140
p90: linear=  465.0  nearest-rank=   250
p95: linear= 1432.5  nearest-rank=  2400
p99: linear= 2206.5  nearest-rank=  2400
```

With 10 samples, "p99" is a guess under any definition. You need at least a few hundred requests before p99 means anything. Two production rules: never average percentiles across hosts (p95 of host A and p95 of host B do not average to the fleet's p95; merge the raw data or histograms), and say which window and which definition a dashboard uses.

### 13.2 Sample standard deviation

From a sample, estimate the spread with \(n-1\) in the denominator (**Bessel's correction**). Dividing by \(n\) underestimates the spread, because the sample mean is fitted to the same data.

\[
s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar x)^2}
\]

!!! example "On paper"

    Eight eval runs take 2, 4, 4, 4, 5, 5, 7 and 9 minutes; the mean is 40/8 = 5:

    \[
    \begin{aligned}
    \textstyle\sum(x_i-\bar x)^2 &= 9+1+1+1+0+0+4+16 = 32\\
    s^2 &= 32/7 \approx 4.571\\
    s &\approx 2.138 \text{ min}
    \end{aligned}
    \]

    Dividing by 8 would give exactly 2. numpy's `np.std` divides by \(n\) unless you pass `ddof=1`; Python's `statistics.stdev` divides by \(n-1\).

### 13.3 Standard error

The **standard error** (SE) is the standard deviation of an estimate: how much your measured number would move if you repeated the measurement on fresh data. For a mean it is \(s/\sqrt n\) (§8.7). For a proportion, such as an accuracy or a pass rate, use the Bernoulli variance.

\[
\operatorname{SE}(\hat p)=\sqrt{\frac{\hat p(1-\hat p)}{n}}
\]

!!! example "On paper"

    A prompt answers 320 of 400 eval questions correctly, \(\hat p=0.8\):

    \[
    \begin{aligned}
    \hat p(1-\hat p) &= 0.8\cdot0.2 = 0.16\\
    \operatorname{SE} &= \sqrt{0.16/400} = \sqrt{0.0004} = 0.02
    \end{aligned}
    \]

    The accuracy is 80% give or take about 2 points. To halve the SE you need four times the data: 1,600 questions for ±1 point.

### 13.4 Confidence interval: the normal approximation

A **95% confidence interval** is a range built by a procedure that captures the true value in 95% of repeated experiments. The quick version is estimate ± 1.96 standard errors.

\[
\hat p\pm1.96\,\operatorname{SE}(\hat p)
\]

!!! example "On paper"

    The same 320 of 400:

    \[
    \begin{aligned}
    1.96\cdot0.02 &= 0.0392\\
    \text{interval} &= (0.7608,\ 0.8392)
    \end{aligned}
    \]

    Report "80% (95% CI 76–84%)". A competitor at 82% on the same size of test set is inside this range: you cannot claim either is better from these numbers alone.

The normal approximation breaks down for small \(n\) and for rates near 0 or 1. At 0 failures out of 20, it gives an interval of exactly (0, 0), a claim of certainty from 20 tests.

### 13.5 The Wilson interval

The **Wilson score interval** fixes both problems and is what you should use for pass rates, success rates and error rates. It pulls the centre towards 0.5 and never leaves \([0,1]\).

\[
\begin{aligned}
c&=\frac{\hat p+z^2/(2n)}{1+z^2/n}\\
h&=\frac{z}{1+z^2/n}\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}\\
\text{interval}&=(c-h,\ c+h)
\end{aligned}
\]

!!! example "On paper"

    A red-team suite of 20 prompt-injection attempts: 0 succeed, so \(\hat p=0\); \(z=1.96\), \(z^2=3.8416\):

    \[
    \begin{aligned}
    1+z^2/n &= 1.19208\\
    c &= 0.09604/1.19208 \approx 0.0806\\
    h &= 1.96\cdot0.049/1.19208 \approx 0.0806\\
    \text{interval} &= (0,\ 0.161)
    \end{aligned}
    \]

    Zero failures in 20 attempts is compatible with a true failure rate of up to 16%. "We tested it and nothing got through" needs a few hundred attempts before it means "rarely". The **rule of three** gives a quick upper bound for zero events: \(3/n\), here 0.15.

[43 §20](43-math-examples.md#wilson-interval) works the Wilson interval for 40 of 100, and [54](54-ai-evaluation.md) uses it throughout. In code:

```python
from math import sqrt

def normal_ci(k, n, z=1.96):
    p = k / n
    h = z * sqrt(p * (1 - p) / n)
    return max(0.0, p - h), min(1.0, p + h)

def wilson_ci(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)

for k, n in [(320, 400), (18, 20), (0, 20), (20, 20)]:
    a, b = normal_ci(k, n)
    c, d = wilson_ci(k, n)
    print(f"{k:3d}/{n:<3d} normal=({a:.3f}, {b:.3f})  wilson=({c:.3f}, {d:.3f})")
```

```text
320/400 normal=(0.761, 0.839)  wilson=(0.758, 0.836)
 18/20  normal=(0.769, 1.000)  wilson=(0.699, 0.972)
  0/20  normal=(0.000, 0.000)  wilson=(0.000, 0.161)
 20/20  normal=(1.000, 1.000)  wilson=(0.839, 1.000)
```

On 320/400 the two agree once rounded to two decimals. On 18/20 the normal interval runs past 100% and has to be clipped, and on 0/20 and 20/20 it collapses to a single point; Wilson stays honest.

### 13.6 An A/B test of two prompts by hand

You ship two prompts for a support bot to random halves of traffic. Prompt A resolves 300 of 400 conversations (75%), prompt B 332 of 400 (83%). Is B better, or was it luck? The **two-proportion z-test** asks: if the prompts were equally good, how surprising would an 8-point gap be?

1. State the **null hypothesis**: both prompts have the same true rate.
2. Under the null, estimate that common rate by pooling the data.
3. Compute the standard error of the difference under the null.
4. Divide the observed difference by it to get \(z\), then look up the **p-value**: the probability of a gap at least this large if the null were true.

\[
z=\frac{\hat p_B-\hat p_A}{\sqrt{\bar p(1-\bar p)\left(\frac1{n_A}+\frac1{n_B}\right)}}
\]

!!! example "On paper"

    The pooled rate is \(\bar p=(300+332)/800\):

    \[
    \begin{aligned}
    \bar p &= 632/800 = 0.79\\
    \bar p(1-\bar p) &= 0.79\cdot0.21 = 0.1659\\
    \operatorname{SE} &= \sqrt{0.1659\cdot(2/400)} \approx 0.0288\\
    z &= 0.08/0.0288 \approx 2.78\\
    p\text{-value} &\approx 0.0055
    \end{aligned}
    \]

    If the prompts were equally good, a gap this large (in either direction) would happen about 0.5% of the time. That is below the usual 0.05 threshold, so the result is called **statistically significant**: chance alone is a poor explanation of the gap.

How big is the improvement? Use a confidence interval for the difference, with the unpooled standard error (the null is no longer assumed):

\[
\operatorname{SE}_{\text{diff}}=\sqrt{\frac{\hat p_A(1-\hat p_A)}{n_A}+\frac{\hat p_B(1-\hat p_B)}{n_B}}
\]

!!! example "On paper"

    \(0.75\cdot0.25=0.1875\) and \(0.83\cdot0.17=0.1411\):

    \[
    \begin{aligned}
    \operatorname{SE}_{\text{diff}} &= \sqrt{(0.1875+0.1411)/400}\\
    &\approx 0.0287\\
    1.96\cdot0.0287 &\approx 0.056\\
    \text{CI} &= (0.024,\ 0.136)
    \end{aligned}
    \]

    B is better by somewhere between 2.4 and 13.6 points. The interval excludes 0, matching the test. It is wide: you know the direction, not the size.

```python
from math import sqrt
from statistics import NormalDist

def two_proportion_z(k_a, n_a, k_b, n_b):
    p_a, p_b = k_a / n_a, k_b / n_b
    pooled = (k_a + k_b) / (n_a + n_b)
    se = sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se
    p_value = 2 * (1 - NormalDist().cdf(abs(z)))        # two-sided
    return p_b - p_a, z, p_value

for counts in [(300, 400, 332, 400), (75, 100, 83, 100)]:
    diff, z, p = two_proportion_z(*counts)
    print(f"{counts}: diff={diff:.3f} z={z:.3f} p={p:.4f}")
```

```text
(300, 400, 332, 400): diff=0.080 z=2.778 p=0.0055
(75, 100, 83, 100): diff=0.080 z=1.389 p=0.1649
```

Same rates, a quarter of the traffic: \(z\) halves and the result is no longer significant. The test needs one condition you must enforce in the product: each conversation is assigned to A or B at random and counted once. If one user's ten conversations all land in B, they are not ten independent observations.

### 13.7 What a p-value is and is not

A p-value is the probability of data at least this extreme **if the null hypothesis were true**. It is not the probability that B is better, and it is not the probability that the result is a fluke. A tiny p-value with a tiny effect (0.2 points on a million users) is real and irrelevant; a large p-value means "not enough evidence", not "no difference". Decide in advance what difference matters to the product, and size the test to detect it (§13.8).

### 13.8 Sample size: how much traffic do you need?

Before running an A/B test, compute how many samples each arm needs to detect the smallest difference you care about. Two settings: \(\alpha\) (false-alarm rate, usually 0.05, \(z_{\alpha/2}=1.96\)) and **power** (the chance to detect a real difference of that size, usually 0.8, \(z_\beta=0.8416\)).

\[
n\approx\frac{(z_{\alpha/2}+z_\beta)^2\,\bigl[p_1(1-p_1)+p_2(1-p_2)\bigr]}{(p_1-p_2)^2}
\]

!!! example "On paper"

    The current prompt resolves 75%; you want to detect an improvement to 80%:

    \[
    \begin{aligned}
    (1.96+0.8416)^2 &\approx 7.849\\
    0.1875+0.16 &= 0.3475\\
    (0.80-0.75)^2 &= 0.0025\\
    n &\approx 7.849\cdot0.3475/0.0025 \approx 1091
    \end{aligned}
    \]

    About 1,100 conversations per prompt, 2,200 in total. Detecting a 2.5-point gain needs about four times as many: halving the difference quadruples the sample.

```python
from math import ceil
from statistics import NormalDist

def n_per_arm(p1, p2, alpha=0.05, power=0.80):
    z_a = NormalDist().inv_cdf(1 - alpha / 2)
    z_b = NormalDist().inv_cdf(power)
    var = p1 * (1 - p1) + p2 * (1 - p2)
    return ceil((z_a + z_b) ** 2 * var / (p1 - p2) ** 2)

for p1, p2 in [(0.75, 0.80), (0.75, 0.775), (0.75, 0.85), (0.02, 0.025)]:
    print(f"{p1} -> {p2}: {n_per_arm(p1, p2):,} per arm")
```

```text
0.75 -> 0.8: 1,091 per arm
0.75 -> 0.775: 4,545 per arm
0.75 -> 0.85: 248 per arm
0.02 -> 0.025: 13,807 per arm
```

The last row is the sobering one: detecting a change in a rare event (a 2% complaint rate moving to 2.5%) needs almost 14,000 samples per arm. For offline evals the same arithmetic tells you how big a golden set must be before a 2-point difference is real ([54](54-ai-evaluation.md)).

### 13.9 The multiple-comparisons trap

Each test at \(\alpha=0.05\) has a 5% chance of a **false positive**: a "significant" difference between two variants that are in fact equally good. Test 20 such variants, each against its own independent control, and the chance that at least one comes out "significant" is much higher.

\[
P(\text{at least one false positive})=1-(1-\alpha)^m
\]

!!! example "On paper"

    \(m=20\) variants, each tested at \(\alpha=0.05\):

    \[
    \begin{aligned}
    1-0.95^{20} &\approx 1-0.358 = 0.642\\
    \text{expected false positives} &= 20\cdot0.05 = 1\\
    \text{Bonferroni} &: \alpha/m = 0.05/20 = 0.0025
    \end{aligned}
    \]

    A 64% chance that at least one variant looks significantly different by luck alone. Testing each variant at 0.0025 brings the family-wide rate back under 5%: \(1-0.9975^{20}\approx0.049\).

A simulation of "A/A tests", where both arms are identical by construction:

```python
import numpy as np
from math import erfc, sqrt

rng = np.random.default_rng(1)
n, rate, variants, runs = 400, 0.75, 20, 2000      # every variant is truly identical

def p_value(k_a, k_b):
    pooled = (k_a + k_b) / (2 * n)
    z = abs(k_b - k_a) / n / sqrt(pooled * (1 - pooled) * 2 / n)
    return erfc(z / sqrt(2))                        # two-sided normal p-value

single = family = family_bonf = 0
for _ in range(runs):
    ps = [p_value(rng.binomial(n, rate), rng.binomial(n, rate)) for _ in range(variants)]
    single += ps[0] < 0.05
    family += min(ps) < 0.05
    family_bonf += min(ps) < 0.05 / variants
print(f"one test, false positive rate : {single / runs:.3f}")
print(f"20 tests, any false positive  : {family / runs:.3f}")
print(f"20 tests, Bonferroni, any     : {family_bonf / runs:.3f}")
```

```text
one test, false positive rate : 0.053
20 tests, any false positive  : 0.626
20 tests, Bonferroni, any     : 0.047
```

The simulation lands close to the formula. The same trap appears when you slice results ("B wins for French users on Tuesdays"), when you peek at a running test every hour and stop when it looks good, and when you try 50 prompt tweaks on the same eval set and keep the best. Fixes: decide the metric and sample size before you start, correct for the number of comparisons, and confirm a winner on fresh data.

### 13.10 Paired comparisons

When both prompts or models answer **the same** eval questions, the two scores are paired: compare them question by question, not as two independent samples. Many questions are easy for both and some hard for both, so pairing removes that shared noise and needs fewer questions. McNemar's test and the paired bootstrap are the tools; [54](54-ai-evaluation.md) works through McNemar's test and shows how much an unpaired comparison of the same data wastes, and [42](42-distill-eval.md) bootstraps a paired difference for a distilled model.

---

## 14. Where the transformer math lives

This chapter stops where the model-specific math begins. The rest is already in the handbook, with the same "On paper" cards:

| Topic | Where |
|---|---|
| The autoregressive loss, masked SFT loss, perplexity | [32 §3](32-mathematics.md#3-the-autoregressive-objective) |
| Attention, multi-head, GQA, KV cache | [32 §6](32-mathematics.md#6-scaled-dot-product-attention), [43 §5–7](43-math-examples.md#5-scaled-dot-product-attention) |
| RMSNorm, RoPE, SwiGLU | [32 §5](32-mathematics.md#5-rmsnorm), [32 §7](32-mathematics.md#7-rotary-position-embeddings-rope), [32 §8](32-mathematics.md#8-the-mlp-swiglu-and-geglu) |
| Softmax Jacobian, cross-entropy gradient, numerical stability | [38 §1](38-derivations.md#1-cross-entropy-and-its-gradient) |
| Attention backward, RMSNorm gradient | [38 §2](38-derivations.md#2-attention-forward-and-backward), [38 §3](38-derivations.md#3-rmsnorm-gradient) |
| Parameter counts, FLOPs, memory | [32 §13–14](32-mathematics.md#13-parameter-counting), [43 §14–15](43-math-examples.md#14-parameter-counting) |
| Quantization error | [32 §15](32-mathematics.md#15-quantization-mathematics), [38 §11](38-derivations.md#11-quantization-error) |
| AdamW and learning-rate schedules | [38 §14](38-derivations.md#14-adamw-bias-correction), [43 §17](43-math-examples.md#17-adamw-and-the-learning-rate-schedule) |
| Distillation losses (KL with temperature) | [43 §19](43-math-examples.md#19-distillation-ce-kd-and-top-k), [39](39-gemma-distillation.md) |

With §3–§12 of this chapter you have every prerequisite those pages assume: dot products, matrix shapes, the chain rule, softmax, logs and KL.

---

## 15. Key takeaways

- Say the **shape** of every tensor. Matrix multiply is \((m\times n)(n\times p)\to(m\times p)\); broadcasting bugs are silent; assert shapes at boundaries.
- The **dot product** is a weighted sum and a similarity. **Cosine similarity** is the dot product of unit vectors; on normalised vectors, cosine, dot and L2 rank alike.
- A **derivative** is "how much the output moves per unit of input". The **chain rule** multiplies link derivatives; the **gradient** collects them; **gradient descent** steps against it, and the learning rate decides between crawling and diverging.
- **Check hand-written gradients** with central finite differences.
- Work in **log space**: sums of log-probabilities, the log-sum-exp trick, never products of many probabilities.
- **Bayes' rule** with base rates: a 90%-recall detector on a 1% class can be wrong on most of its alerts.
- Use **expectation and variance** for budgets; use **percentiles**, not means, for latency.
- **Softmax** turns logits into probabilities; temperature rescales logits. A language model's loss is cross-entropy in nats; perplexity is its exponential; the loss floor is the entropy of the text.
- Every metric from a sample has a **standard error**. Use **Wilson intervals** for rates, a **two-proportion z-test** for A/B tests, compute the **sample size** before the test, and correct for **multiple comparisons**.

---

## 16. Exercises

!!! question "Exercise 16.1"

    Two embeddings are \(\mathbf u=(3,4)\) and \(\mathbf v=(4,3)\). Compute their dot product, both L2 norms and their cosine similarity. Then compute the squared distance between the unit vectors.

??? success "Answer"

    \(\mathbf u\cdot\mathbf v=12+12=24\). Both norms are \(\sqrt{9+16}=5\). Cosine \(=24/25=0.96\). Squared distance of the unit vectors \(=2-2\cdot0.96=0.08\); directly, \((0.6-0.8)^2+(0.8-0.6)^2=0.04+0.04=0.08\).

!!! question "Exercise 16.2"

    A batch of 32 sentence embeddings of size 768 goes through a linear layer to size 3,072, with a bias. What are the shapes of \(X\), \(W\) (in the `nn.Linear` layout), \(\mathbf b\) and the output? How many parameters does the layer have?

??? success "Answer"

    \(X\) is (32, 768), \(W\) is (3072, 768), \(\mathbf b\) is (3072,), and `X @ W.T + b` is (32, 3072). Parameters: \(768\cdot3072+3072=2{,}359{,}296+3{,}072=2{,}362{,}368\).

!!! question "Exercise 16.3"

    Minimise \(L(w)=(w-5)^2\) by gradient descent from \(w_0=1\) with \(\eta=0.25\). Give \(w_1\), \(w_2\), \(w_3\). Which learning rates make this diverge?

??? success "Answer"

    \(L'(w)=2(w-5)\). \(w_1=1-0.25\cdot(-8)=3\); \(w_2=3-0.25\cdot(-4)=4\); \(w_3=4-0.25\cdot(-2)=4.5\). The distance to 5 halves each step (factor \(1-2\eta=0.5\)). It diverges when \(\lvert1-2\eta\rvert>1\), that is \(\eta>1\); at exactly \(\eta=1\) it oscillates between 1 and 9.

!!! question "Exercise 16.4"

    A logistic model computes \(z=wx\) and the loss \(L=-\ln\sigma(z)\) for a positive example. Using \(dL/dz=\sigma(z)-1\), find \(dL/dw\) at \(w=0\), \(x=2\). Which way does gradient descent move \(w\)?

??? success "Answer"

    At \(w=0\), \(z=0\) and \(\sigma(0)=0.5\), so \(dL/dz=-0.5\). The chain rule gives \(dL/dw=dL/dz\cdot dz/dw=-0.5\cdot2=-1\). The gradient is negative, so \(w\) increases (\(w_1=0+\eta\cdot1\)), which raises \(z\) and the predicted probability of the positive class.

!!! question "Exercise 16.5"

    20% of incoming email is spam. The word "free" appears in 50% of spam and 5% of legitimate email. An email contains "free". What is the probability that it is spam?

??? success "Answer"

    \(P(\text{free}\mid\text{spam})P(\text{spam})=0.5\cdot0.2=0.10\). \(P(\text{free}\mid\text{ham})P(\text{ham})=0.05\cdot0.8=0.04\). \(P(\text{spam}\mid\text{free})=0.10/(0.10+0.04)=0.10/0.14\approx0.714\). In counts: of 1,000 emails, 100 spam emails and 40 legitimate ones contain "free".

!!! question "Exercise 16.6"

    An agent finishes a task in 1 model call 90% of the time and needs 3 calls 10% of the time. Compute the expected number of calls, the variance and the standard deviation. How many calls should you budget for 1,000 tasks?

??? success "Answer"

    \(\mathbb E[X]=0.9\cdot1+0.1\cdot3=1.2\). \(\mathbb E[X^2]=0.9\cdot1+0.1\cdot9=1.8\). \(\operatorname{Var}=1.8-1.2^2=0.36\), sd \(=0.6\). For 1,000 tasks expect 1,200 calls, with an sd of \(0.6\sqrt{1000}\approx19\) calls on the total.

!!! question "Exercise 16.7"

    A binary classifier outputs the logit \(z=\ln3\) for the correct class against 0 for the other. What probability does it give the correct class, what is the cross-entropy loss in nats, and what is the perplexity? What is the entropy, in bits, of a uniform distribution over 8 tokens, and its perplexity?

??? success "Answer"

    \(p=3/(3+1)=0.75\). Loss \(=-\ln0.75\approx0.288\) nats. Perplexity \(=1/0.75\approx1.333\). Uniform over 8: \(H=\log_2 8=3\) bits, perplexity \(2^3=8\): the model is choosing among 8 equally likely tokens.

!!! question "Exercise 16.8"

    Prompt A solves 160 of 200 tickets, prompt B 176 of 200. Run the two-proportion z-test. Is the difference significant at 0.05? Would it survive a Bonferroni correction if B was the best of 5 variants you tried?

??? success "Answer"

    \(\hat p_A=0.80\), \(\hat p_B=0.88\), pooled \(\bar p=336/400=0.84\). \(\operatorname{SE}=\sqrt{0.84\cdot0.16\cdot(2/200)}=\sqrt{0.001344}\approx0.0367\). \(z=0.08/0.0367\approx2.18\), two-sided p \(\approx0.029\). Significant at 0.05, but not at the Bonferroni level \(0.05/5=0.01\). Confirm B on fresh traffic before you call it.

!!! question "Exercise 16.9"

    Your guardrail blocked all 300 of 300 jailbreak attempts in a test suite. Give an approximate 95% upper bound on the true miss rate. How many attempts, all of them blocked, would you need before you could claim a miss rate below 0.1%?

??? success "Answer"

    Rule of three: \(3/300=0.01\), so up to about 1% of attempts could still get through. For an upper bound of 0.1% with zero misses you need about \(3/0.001=3{,}000\) attempts with no misses, ten times as many.

!!! question "Exercise 16.10"

    Ten latencies are 100, 100, 110, 120, 120, 130, 140, 150, 160 and 1,000 ms. Compute the mean, the median and the nearest-rank p90. Which would you put on the dashboard?

??? success "Answer"

    Sum \(=2{,}130\), mean \(=213\) ms. Median \(=(120+130)/2=125\) ms. Nearest-rank p90 is the 9th sorted value, 160 ms. Put p50 and p90/p95 on the dashboard; the mean is pulled up by one 1,000 ms outlier and describes no real request.

---

Next: [47 — Machine learning fundamentals](47-ai-classical-ml.md)
