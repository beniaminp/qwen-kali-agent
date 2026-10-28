# 48 — Deep learning and neural networks

*Last verified: 28 October 2026.*

This chapter builds a neural network from one neuron up to a PyTorch training loop, then explains every knob you will meet when you train or fine-tune one: optimizers, learning-rate schedules, batch size, initialization, normalization, regularization, gradient problems, number formats and memory. Every idea comes with numbers you can redo on paper or code that runs on a laptop CPU, with its real output. You need the derivatives and the chain rule from [46 — The math you need](46-ai-math.md) and the train/validation/test discipline from [47 — Classical machine learning](47-ai-classical-ml.md). The next chapter, [49 — How LLMs work](49-ai-llm-internals.md), applies all of it to transformers. The heavy transformer math lives in [32 — Mathematics](32-mathematics.md) and [37 — Architecture anatomy](37-architecture.md); this chapter links there instead of repeating it.

All code is Python 3.13 with `numpy`, `scikit-learn` and CPU `torch`. Seeds are fixed, so you should get the same numbers; on a different CPU or library version the last digit can differ.

---

## 1. From a linear model to a neuron

### 1.1 The neuron: a weighted sum plus a nonlinearity

A **neuron** multiplies each input by a weight, adds the products and a bias, and passes the total through a fixed function called the **activation**. That is all. A neural network is many neurons wired in layers, and **training** means finding the weights and biases that make the outputs useful.

\[
y=\phi(\mathbf w\cdot\mathbf x+b)=\phi\Big(\sum_i w_i x_i+b\Big)
\]

!!! example "On paper"

    An urgency score for a support ticket. Inputs: \(x_1=2\) exclamation marks per sentence, \(x_2=1\) because the customer pays. Weights \(\mathbf w=(1.5,\,1)\), bias \(b=-2\), activation the sigmoid \(\sigma(z)=1/(1+e^{-z})\), with \(e^{-2}\approx0.135\):

    \[
    \begin{aligned}
    z &= 1.5\cdot2+1\cdot1-2 = 2\\
    y &= \sigma(2) = 1/1.135 \approx 0.881
    \end{aligned}
    \]

    The neuron says "88% urgent". The bias of \(-2\) means a ticket with no signal at all starts at \(\sigma(-2)\approx0.119\), not at 50%.

The same neuron on three tickets at once is a matrix–vector product, which is how every framework computes it:

```python
import numpy as np

# Three support tickets, two features each:
# x1 = exclamation marks per sentence, x2 = 1 if the customer is on the paid plan
X = np.array([[2.0, 1.0],
              [0.0, 1.0],
              [1.0, 0.0]])
w = np.array([1.5, 1.0])   # weights: how much each feature pushes towards "urgent"
b = -2.0                   # bias: the default leans towards "not urgent"

z = X @ w + b              # weighted sum, one number per ticket
p = 1 / (1 + np.exp(-z))   # sigmoid squashes it into (0, 1)
for zi, pi in zip(z, p):
    print(f"z = {zi:+.2f}   p(urgent) = {pi:.3f}")
```

```text
z = +2.00   p(urgent) = 0.881
z = -1.00   p(urgent) = 0.269
z = -0.50   p(urgent) = 0.378
```

**Why an engineer cares.** A single neuron with a sigmoid *is* logistic regression from [47](47-ai-classical-ml.md). Everything deep learning adds is depth (layers feeding layers) and the machinery to train millions or billions of these weights at once. When a fine-tuned model misbehaves, you are debugging the same weighted sums, just 14.8 billion of them.

### 1.2 Why the nonlinearity matters: XOR

Stack two linear layers without an activation and you get one linear layer back, because matrix multiplication is associative:

\[
W_2(W_1\mathbf x)=(W_2W_1)\,\mathbf x
\]

!!! example "On paper"

    One input, two "layers" with weights 3 and 2 and no activation:

    \[
    2\cdot(3x) = 6x
    \]

    A single layer with weight 6 does exactly the same job. A hundred linear layers still draw one straight line (or one flat plane) through the data. Depth only buys power when something non-linear sits between the layers.

The classic proof is **XOR** ("exactly one of two flags is set"). Picture the four points (0,0), (0,1), (1,0), (1,1) on a square: the two positive points sit on one diagonal and the two negative points on the other. No straight line separates the diagonals. One hidden layer with a ReLU (\(\max(0,z)\), see §2.2) fixes it:

\[
\begin{aligned}
h_1&=\operatorname{ReLU}(x_1+x_2)\\
h_2&=\operatorname{ReLU}(x_1+x_2-1)\\
\hat y&=h_1-2h_2
\end{aligned}
\]

!!! example "On paper"

    Input \((1,1)\): \(h_1=\operatorname{ReLU}(2)=2\), \(h_2=\operatorname{ReLU}(1)=1\). Input \((0,1)\): \(h_1=1\), \(h_2=\operatorname{ReLU}(0)=0\).

    \[
    \begin{aligned}
    \hat y(1,1) &= 2-2\cdot1 = 0\\
    \hat y(0,1) &= 1-2\cdot0 = 1
    \end{aligned}
    \]

    \((0,0)\) gives \(0-0=0\) and \((1,0)\) gives 1. The second hidden unit only switches on when both flags are set and then cancels the first: the ReLU's kink is what bends the decision boundary.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0])                      # XOR: "exactly one is true"

linear = LogisticRegression().fit(X, y)
print("linear model accuracy:", linear.score(X, y))
print("linear model p(1):    ", linear.predict_proba(X)[:, 1].round(3))

def relu(v):
    return np.maximum(v, 0)

W1 = np.array([[1, 1], [1, 1]], dtype=float)    # both hidden units see x1 + x2
b1 = np.array([0, -1], dtype=float)             # ...but the second one only fires at 2
w2 = np.array([1, -2], dtype=float)
h = relu(X @ W1.T + b1)
out = h @ w2
print("hidden h:\n", h)
print("two-layer ReLU output:", out)
```

```text
linear model accuracy: 0.5
linear model p(1):     [0.5 0.5 0.5 0.5]
hidden h:
 [[0. 0.]
 [1. 0.]
 [1. 0.]
 [2. 1.]]
two-layer ReLU output: [0. 1. 1. 0.]
```

No straight line gets more than 3 of the 4 points right, and on this perfectly symmetric data logistic regression settles on 0.5 for every point, which is accuracy 0.5. In §6.3 a network learns XOR weights by itself instead of you choosing them.

**Analogy.** Linear layers are like composing functions that only scale and add: any pipeline of them can be collapsed into one step. A nonlinearity is the `if` statement that stops the collapse.

---

## 2. Activation functions

The activation decides what a neuron outputs and, just as important, what gradient flows back through it during training (§5). The derivative matters as much as the value.

### 2.1 Sigmoid and tanh

\[
\begin{aligned}
\sigma(x)&=\frac{1}{1+e^{-x}}\\
\sigma'(x)&=\sigma(x)\,\big(1-\sigma(x)\big)
\end{aligned}
\]

!!! example "On paper"

    At \(x=0\): \(e^{0}=1\), so \(\sigma(0)=1/2\). At \(x=2\), \(\sigma(2)\approx0.881\) as in §1.1.

    \[
    \begin{aligned}
    \sigma'(0) &= 0.5\cdot0.5 = 0.25\\
    \sigma'(2) &= 0.881\cdot0.119 \approx 0.105
    \end{aligned}
    \]

    0.25 is the *largest* slope the sigmoid ever has. Every sigmoid layer multiplies the backward signal by at most a quarter, which is the root of the vanishing-gradient problem in §14.

\[
\begin{aligned}
\tanh(x)&=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}\\
\tanh'(x)&=1-\tanh^2(x)
\end{aligned}
\]

!!! example "On paper"

    At \(x=1\), with \(e\approx2.718\) and \(e^{-1}\approx0.368\):

    \[
    \begin{aligned}
    \tanh(1) &= \frac{2.718-0.368}{2.718+0.368} = \frac{2.350}{3.086} \approx 0.7615\\
    \tanh'(1) &= 1-0.7615^2 \approx 0.420
    \end{aligned}
    \]

    tanh is a sigmoid stretched to \((-1,1)\) and centred on 0; its slope at 0 is 1, four times the sigmoid's. That is why it trained better in older networks and why RNNs (§19.2) use it.

Use the sigmoid today mainly at the **output** of a yes/no classifier, where you want a probability, and inside gates (LSTM gates, SiLU below).

### 2.2 ReLU

\[
\begin{aligned}
\operatorname{ReLU}(x)&=\max(0,x)\\
\operatorname{ReLU}'(x)&=\begin{cases}1,&x>0\\0,&x<0\end{cases}
\end{aligned}
\]

!!! example "On paper"

    Two pre-activations, \(z=(-2,\ 3)\):

    \[
    \begin{aligned}
    \operatorname{ReLU}(z) &= (0,\ 3)\\
    \operatorname{ReLU}'(z) &= (0,\ 1)
    \end{aligned}
    \]

    The positive unit passes its gradient through unchanged (slope 1, no shrinking); the negative one passes nothing. If a unit's input is negative for *every* example, it never gets a gradient again: a **dead ReLU**. At exactly 0 the slope is undefined; PyTorch uses 0.

ReLU is cheap, does not saturate for positive inputs, and made deep networks trainable around 2010–2012. Its weakness is dead units, which a too-high learning rate can cause in bulk.

### 2.3 GELU and SiLU (Swish)

Modern transformers use smooth versions of ReLU. **GELU** multiplies \(x\) by \(\Phi(x)\), the probability that a standard normal variable is below \(x\). **SiLU**, also called **Swish**, multiplies \(x\) by its own sigmoid:

\[
\begin{aligned}
\operatorname{GELU}(x)&=x\,\Phi(x)\\
\operatorname{SiLU}(x)&=x\,\sigma(x)
\end{aligned}
\]

!!! example "On paper"

    With \(\Phi(1)\approx0.841\), \(\Phi(-1)\approx0.159\), \(\sigma(1)\approx0.731\) and \(\sigma(-1)\approx0.269\):

    \[
    \begin{aligned}
    \operatorname{GELU}(1) &= 1\cdot0.841 = 0.841\\
    \operatorname{GELU}(-1) &= -1\cdot0.159 = -0.159\\
    \operatorname{SiLU}(1) &= 1\cdot0.731 = 0.731\\
    \operatorname{SiLU}'(1) &= 0.731\,(1+1\cdot0.269) \approx 0.928
    \end{aligned}
    \]

    Both behave like ReLU for large \(|x|\), but they let a little negative signal (and gradient) through near zero, so units do not die.

Here are all five at a few points, with their derivatives computed by autograd (§6):

```python
import torch
import torch.nn.functional as F

acts = {
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "relu": F.relu,
    "gelu": F.gelu,      # exact form: x * Phi(x)
    "silu": F.silu,      # x * sigmoid(x), also called Swish
}
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
print("x        " + "".join(f"{v:>8.1f}" for v in x.tolist()))
for name, f in acts.items():
    y = f(x)
    (grad,) = torch.autograd.grad(y.sum(), x)   # d f / d x at each point
    print(f"{name:<8} " + "".join(f"{v:>8.3f}" for v in y.tolist()))
    print(f"  d/dx   " + "".join(f"{v:>8.3f}" for v in grad.tolist()))
```

```text
x            -2.0    -1.0     0.0     1.0     2.0
sigmoid     0.119   0.269   0.500   0.731   0.881
  d/dx      0.105   0.197   0.250   0.197   0.105
tanh       -0.964  -0.762   0.000   0.762   0.964
  d/dx      0.071   0.420   1.000   0.420   0.071
relu        0.000   0.000   0.000   1.000   2.000
  d/dx      0.000   0.000   0.000   1.000   1.000
gelu       -0.046  -0.159   0.000   0.841   1.954
  d/dx     -0.085  -0.083   0.500   1.083   1.085
silu       -0.238  -0.269   0.000   0.731   1.762
  d/dx     -0.091   0.072   0.500   0.928   1.091
```

Read the derivative rows: sigmoid never exceeds 0.25, tanh peaks at 1, ReLU is exactly 0 or 1, and the GELU and SiLU slopes are slightly negative at \(-2\): both curves dip a little below zero and then turn back up.

### 2.4 Which activation goes where

| Place | Typical choice | Why |
|---|---|---|
| Hidden layers of an MLP or CNN | ReLU, GELU | Cheap, gradients do not shrink for positive inputs |
| Transformer MLP | SwiGLU (Qwen3, `hidden_act: "silu"`), GeGLU (Gemma) | Gated variants of SiLU/GELU; see [32 §8](32-mathematics.md#8-the-mlp-swiglu-and-geglu) |
| Output, yes/no | sigmoid | A probability for one class |
| Output, one of \(C\) classes | softmax | Probabilities that sum to 1 (see [46](46-ai-math.md)) |
| Output, a number (price, latency) | none (identity) | Any real value |
| Gates (LSTM, GLU) | sigmoid | A 0–1 "how much to let through" |

**Pitfall.** Putting a sigmoid or softmax at the output *and* using a loss that expects raw scores (logits) applies it twice. §4.4 shows the bug and its silent effect.

---

## 3. The MLP forward pass by hand

A **multilayer perceptron (MLP)** is layers of neurons where every output of one layer feeds every neuron of the next. The **forward pass** computes the prediction from the input, layer by layer. Take the smallest interesting one: 2 inputs, 2 hidden ReLU units, 1 sigmoid output.

\[
\begin{aligned}
\mathbf z_1&=W_1\mathbf x+\mathbf b_1, & \mathbf h&=\operatorname{ReLU}(\mathbf z_1)\\
z_2&=\mathbf w_2\cdot\mathbf h+b_2, & p&=\sigma(z_2)
\end{aligned}
\]

!!! example "On paper"

    Input \(\mathbf x=(1,2)\). Hidden weights: row 1 of \(W_1\) is \((0.5,\,0.25)\), row 2 is \((-0.5,\,1)\); \(\mathbf b_1=(0,\,0.5)\). Output weights \(\mathbf w_2=(1,\,-0.5)\), \(b_2=0\).

    \[
    \begin{aligned}
    z_{1,1} &= 0.5\cdot1+0.25\cdot2+0 = 1\\
    z_{1,2} &= -0.5\cdot1+1\cdot2+0.5 = 2\\
    \mathbf h &= \operatorname{ReLU}(1,\,2) = (1,\,2)\\
    z_2 &= 1\cdot1-0.5\cdot2+0 = 0\\
    p &= \sigma(0) = 0.5
    \end{aligned}
    \]

    The network has \(4+2+2+1=9\) parameters. With true label \(y=1\) it is maximally unsure, so training has something to fix; §5 computes how each of the 9 numbers should move.

```python
import numpy as np

x = np.array([1.0, 2.0])                  # one example, two features
y = 1.0                                   # its label
W1 = np.array([[0.5, 0.25],
               [-0.5, 1.0]])              # row j = weights of hidden unit j
b1 = np.array([0.0, 0.5])
w2 = np.array([1.0, -0.5])
b2 = 0.0

z1 = W1 @ x + b1                          # hidden pre-activations
h = np.maximum(z1, 0)                     # ReLU
z2 = w2 @ h + b2                          # output logit
p = 1 / (1 + np.exp(-z2))                 # sigmoid
loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))   # binary cross-entropy
print("z1 =", z1, " h =", h)
print(f"z2 = {z2:.3f}  p = {p:.3f}  loss = {loss:.4f}")
```

```text
z1 = [1. 2.]  h = [1. 2.]
z2 = 0.000  p = 0.500  loss = 0.6931
```

Vocabulary you will see everywhere: \(z_2\), the raw score before the sigmoid, is the **logit**. The shapes are the contract: \(W_1\) is `[hidden, inputs]` = `[2, 2]`, and a batch of \(B\) examples turns \(\mathbf x\) into a `[B, 2]` matrix with no other change. Qwen3-14B's MLP has the same structure with \(d=5{,}120\) inputs and \(17{,}408\) hidden units ([37 §1.1](37-architecture.md#11-shape-trace-for-qwen3-14b)).

---

## 4. Loss functions

A **loss** turns "how wrong was this prediction" into one number that training pushes down. The loss you choose defines what "good" means, so it must match the output type.

### 4.1 Mean squared error (MSE)

\[
\text{MSE}=\frac1n\sum_{i=1}^{n}(\hat y_i-y_i)^2
\]

!!! example "On paper"

    An invoice extractor predicts totals of 3 and 5 (hundreds of euros); the true totals are 2 and 7. The errors are \(+1\) and \(-2\):

    \[
    \begin{aligned}
    \text{MSE} &= \frac{1^2+(-2)^2}{2} = \frac{5}{2} = 2.5\\
    \text{RMSE} &= \sqrt{2.5} \approx 1.58
    \end{aligned}
    \]

    RMSE is back in the original unit: a typical miss of about 158 euros. Squaring makes the 2-unit miss count four times as much as the 1-unit miss.

Use MSE for regression (numbers). Do not use it for classification: with a sigmoid output its gradient vanishes exactly when the model is confidently wrong.

### 4.2 Binary cross-entropy (BCE)

\[
\text{BCE}=-\big[y\ln p+(1-y)\ln(1-p)\big]
\]

!!! example "On paper"

    The model says \(p(\text{urgent})=0.8\).

    \[
    \begin{aligned}
    y=1:\quad -\ln0.8 &\approx 0.223\\
    y=0:\quad -\ln0.2 &\approx 1.609\\
    y=0,\ p=0.999:\quad -\ln0.001 &\approx 6.91
    \end{aligned}
    \]

    Right and confident costs little; wrong and confident costs a lot, and the cost has no ceiling. That pressure is what makes classifiers produce useful probabilities.

### 4.3 Categorical cross-entropy

For \(C\) classes the network outputs \(C\) logits, softmax turns them into probabilities, and the loss is minus the log-probability of the true class \(y\):

\[
\text{CE}=-\ln\operatorname{softmax}(\mathbf z)_y=-z_y+\ln\sum_{k}e^{z_k}
\]

!!! example "On paper"

    A ticket router with three queues (billing, bugs, sales) outputs logits \(\mathbf z=(\ln2,\,0,\,0)\approx(0.693,0,0)\). Then \(e^{\mathbf z}=(2,1,1)\) and the sum is 4:

    \[
    \begin{aligned}
    \operatorname{softmax}(\mathbf z) &= (0.5,\ 0.25,\ 0.25)\\
    \text{CE if billing} &= -\ln0.5 \approx 0.693\\
    \text{CE if bugs} &= -\ln0.25 \approx 1.386
    \end{aligned}
    \]

    This is exactly the loss an LLM minimises, with 151,936 "queues" (one per token) instead of 3. [38 §1](38-derivations.md#1-cross-entropy-and-its-gradient) derives its gradient.

### 4.4 The same losses in PyTorch, and the classic bug

```python
import math
import torch
import torch.nn as nn

# MSE: predicted vs true invoice totals (hundreds of euros)
pred, true = torch.tensor([3.0, 5.0]), torch.tensor([2.0, 7.0])
print("MSE      ", nn.MSELoss()(pred, true).item())

# BCE: the model says p(urgent) = 0.8, i.e. logit ln(0.8 / 0.2) = ln 4
logit = torch.tensor([math.log(4)])
bce = nn.BCEWithLogitsLoss()          # takes logits, applies the sigmoid inside
print("BCE y=1  ", round(bce(logit, torch.tensor([1.0])).item(), 4))
print("BCE y=0  ", round(bce(logit, torch.tensor([0.0])).item(), 4))

# Categorical CE over 3 queues (billing, bugs, sales): logits (ln 2, 0, 0)
logits = torch.tensor([[math.log(2), 0.0, 0.0]])
ce = nn.CrossEntropyLoss()            # takes logits and a class index
print("softmax  ", torch.softmax(logits, dim=1).tolist())
print("CE y=0   ", round(ce(logits, torch.tensor([0])).item(), 4))
print("CE y=1   ", round(ce(logits, torch.tensor([1])).item(), 4))

# The classic bug: feeding probabilities where logits are expected
probs = torch.softmax(logits, dim=1)
print("CE on probs (wrong)", round(ce(probs, torch.tensor([0])).item(), 4))
```

```text
MSE       2.5
BCE y=1   0.2231
BCE y=0   1.6094
softmax   [[0.5, 0.25, 0.25]]
CE y=0    0.6931
CE y=1    1.3863
CE on probs (wrong) 0.9391
```

`nn.CrossEntropyLoss` applies softmax itself. Feed it probabilities and it softmaxes them again: the loss comes out as 0.939 instead of 0.693, nothing crashes, and the model trains worse. The same holds for `BCEWithLogitsLoss` versus a manual sigmoid plus `BCELoss`; the logits version is also numerically safer because it never computes \(\ln 0\).

| Output | Last layer | Loss |
|---|---|---|
| A number | linear | MSE (or MAE if outliers dominate) |
| Yes/no, or several independent yes/no labels | linear, logits | `BCEWithLogitsLoss` |
| One of \(C\) classes | linear, \(C\) logits | `CrossEntropyLoss` |
| Next token of a language model | linear, \(V\) logits | cross-entropy over the vocabulary ([49](49-ai-llm-internals.md)) |

---

## 5. Backpropagation by hand

**Backpropagation** computes the gradient of the loss with respect to every parameter: how much the loss changes when you nudge that parameter. It applies the chain rule from [46](46-ai-math.md) backwards through the forward pass, reusing each intermediate result once. The cost is about twice the forward pass, whatever the number of parameters; that is why you can train billions of them.

### 5.1 The starting point: sigmoid plus BCE

The first backward step, from the loss to the logit \(z_2\), collapses to a famous one-liner:

\[
\frac{\partial L}{\partial z_2}=\frac{\partial L}{\partial p}\cdot\frac{\partial p}{\partial z_2}=p-y
\]

!!! example "On paper"

    From §3, \(p=0.5\) and \(y=1\). For BCE with \(y=1\), \(\partial L/\partial p=-1/p\), and the sigmoid's slope is \(p(1-p)\):

    \[
    \begin{aligned}
    \partial L/\partial p &= -1/0.5 = -2\\
    \partial p/\partial z_2 &= 0.5\cdot0.5 = 0.25\\
    \partial L/\partial z_2 &= -2\cdot0.25 = -0.5 = p-y
    \end{aligned}
    \]

    The negative sign says "increase \(z_2\) to lower the loss", which pushes \(p\) towards the label 1. Softmax plus cross-entropy has the same shape: probabilities minus the one-hot label.

### 5.2 Every partial derivative

Going backwards one operation at a time:

\[
\begin{aligned}
\frac{\partial L}{\partial \mathbf w_2}&=\frac{\partial L}{\partial z_2}\,\mathbf h\\
\frac{\partial L}{\partial \mathbf h}&=\frac{\partial L}{\partial z_2}\,\mathbf w_2\\
\frac{\partial L}{\partial \mathbf z_1}&=\frac{\partial L}{\partial \mathbf h}\odot\mathbf 1[\mathbf z_1>0]\\
\frac{\partial L}{\partial W_1}&=\frac{\partial L}{\partial \mathbf z_1}\,\mathbf x^{\top}
\end{aligned}
\]

!!! example "On paper"

    With \(\partial L/\partial z_2=-0.5\), \(\mathbf h=(1,2)\), \(\mathbf w_2=(1,-0.5)\), \(\mathbf z_1=(1,2)\) (both positive, so the ReLU passes everything) and \(\mathbf x=(1,2)\):

    \[
    \begin{aligned}
    \partial L/\partial\mathbf w_2 &= -0.5\cdot(1,2) = (-0.5,\,-1)\\
    \partial L/\partial b_2 &= -0.5\\
    \partial L/\partial\mathbf h &= -0.5\cdot(1,-0.5) = (-0.5,\,0.25)\\
    \partial L/\partial\mathbf z_1 &= (-0.5,\,0.25)\odot(1,1)\\
    &= (-0.5,\,0.25)\\
    \partial L/\partial W_1[1,:] &= -0.5\cdot(1,2) = (-0.5,\,-1)\\
    \partial L/\partial W_1[2,:] &= 0.25\cdot(1,2) = (0.25,\,0.5)\\
    \partial L/\partial\mathbf b_1 &= (-0.5,\,0.25)
    \end{aligned}
    \]

    Hidden unit 2 gets a *positive* gradient: because its output weight is negative (\(-0.5\)), making it fire less raises \(p\). Backprop worked that out with no special case.

The same nine numbers as a table, each with its chain:

| Parameter | Chain | Value |
|---|---|---:|
| \(w_{2,1}\) | \((p-y)\cdot h_1\) | \(-0.5\) |
| \(w_{2,2}\) | \((p-y)\cdot h_2\) | \(-1.0\) |
| \(b_2\) | \((p-y)\cdot 1\) | \(-0.5\) |
| \(W_{1}[1,1]\) | \((p-y)\,w_{2,1}\cdot\mathbf 1[z_{1,1}>0]\cdot x_1\) | \(-0.5\) |
| \(W_{1}[1,2]\) | \((p-y)\,w_{2,1}\cdot\mathbf 1[z_{1,1}>0]\cdot x_2\) | \(-1.0\) |
| \(W_{1}[2,1]\) | \((p-y)\,w_{2,2}\cdot\mathbf 1[z_{1,2}>0]\cdot x_1\) | \(0.25\) |
| \(W_{1}[2,2]\) | \((p-y)\,w_{2,2}\cdot\mathbf 1[z_{1,2}>0]\cdot x_2\) | \(0.5\) |
| \(b_{1,1}\) | \((p-y)\,w_{2,1}\cdot\mathbf 1[z_{1,1}>0]\) | \(-0.5\) |
| \(b_{1,2}\) | \((p-y)\,w_{2,2}\cdot\mathbf 1[z_{1,2}>0]\) | \(0.25\) |

**Analogy.** Backprop is blame assignment in a call graph. The loss is the "incident"; each node passes blame to the nodes that produced its inputs, in proportion to how sensitive it was to each. A profiler attributing time from a slow endpoint back to its callees does the same walk in reverse.

### 5.3 One gradient-descent step

\[
\theta\leftarrow\theta-\eta\,\frac{\partial L}{\partial\theta}
\]

!!! example "On paper"

    Learning rate \(\eta=0.1\). Each parameter moves against its gradient by a tenth of it:

    \[
    \begin{aligned}
    \mathbf w_2 &= (1,-0.5)-0.1\,(-0.5,-1) = (1.05,\,-0.4)\\
    b_2 &= 0-0.1\cdot(-0.5) = 0.05\\
    W_1 &= \begin{pmatrix}0.55&0.35\\-0.525&0.95\end{pmatrix}\\
    \mathbf b_1 &= (0.05,\,0.475)
    \end{aligned}
    \]

    Re-run the forward pass: \(\mathbf z_1=(1.30,\,1.85)\), \(z_2=1.05\cdot1.3-0.4\cdot1.85+0.05=0.675\), \(p=\sigma(0.675)\approx0.6626\), and the loss drops from 0.693 to \(-\ln0.6626\approx0.412\).

### 5.4 Check it with finite differences

A **finite-difference check** nudges one parameter up and down by a tiny \(\epsilon\) and measures the slope of the loss directly. It is slow (two forward passes per parameter) but needs no calculus, so it is the standard test for a hand-written backward pass.

\[
\frac{\partial L}{\partial\theta}\approx\frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon}
\]

!!! example "On paper"

    Check \(\partial L/\partial b_2\) with \(\epsilon=0.01\). Here \(z_2=b_2\) and \(L=-\ln\sigma(b_2)=\ln(1+e^{-b_2})\):

    \[
    \begin{aligned}
    L(+0.01) &= \ln(1+e^{-0.01}) \approx 0.68816\\
    L(-0.01) &= \ln(1+e^{0.01}) \approx 0.69816\\
    \text{slope} &\approx (0.68816-0.69816)/0.02 = -0.5
    \end{aligned}
    \]

    It matches the \(-0.5\) from §5.2. The central difference (both sides) is far more accurate than a one-sided one.

```python
import numpy as np

x, y = np.array([1.0, 2.0]), 1.0
params = {"W1": np.array([[0.5, 0.25], [-0.5, 1.0]]), "b1": np.array([0.0, 0.5]),
          "w2": np.array([1.0, -0.5]), "b2": np.array(0.0)}

def forward(P):
    z1 = P["W1"] @ x + P["b1"]
    h = np.maximum(z1, 0)
    p = 1 / (1 + np.exp(-(P["w2"] @ h + P["b2"])))
    return z1, h, p, -(y * np.log(p) + (1 - y) * np.log(1 - p))

# backward pass by hand, one chain-rule step per line
z1, h, p, loss = forward(params)
dz2 = p - y                         # dL/dz2 for sigmoid + BCE
grads = {"w2": dz2 * h, "b2": np.array(dz2)}
dh = dz2 * params["w2"]             # dL/dh
dz1 = dh * (z1 > 0)                 # ReLU passes the gradient where z1 > 0
grads["W1"] = np.outer(dz1, x)      # dL/dW1[j, i] = dz1[j] * x[i]
grads["b1"] = dz1
for k, g in grads.items():
    print(f"dL/d{k} =", np.round(g, 4).tolist())

# finite differences: nudge each weight by +-eps and watch the loss
eps, worst = 1e-6, 0.0
for k, v in params.items():
    for idx in np.ndindex(v.shape):
        old = v[idx]
        v[idx] = old + eps; up = forward(params)[3]
        v[idx] = old - eps; down = forward(params)[3]
        v[idx] = old
        worst = max(worst, abs((up - down) / (2 * eps) - grads[k][idx]))
print(f"largest |hand - numeric| = {worst:.1e}")
```

```text
dL/dw2 = [-0.5, -1.0]
dL/db2 = -0.5
dL/dW1 = [[-0.5, -1.0], [0.25, 0.5]]
dL/db1 = [-0.5, 0.25]
largest |hand - numeric| = 1.3e-10
```

### 5.5 Check it with PyTorch autograd

```python
import torch
import torch.nn.functional as F

x, y = torch.tensor([1.0, 2.0]), torch.tensor(1.0)
W1 = torch.tensor([[0.5, 0.25], [-0.5, 1.0]], requires_grad=True)
b1 = torch.tensor([0.0, 0.5], requires_grad=True)
w2 = torch.tensor([1.0, -0.5], requires_grad=True)
b2 = torch.tensor(0.0, requires_grad=True)

def loss_fn():
    h = F.relu(W1 @ x + b1)
    return F.binary_cross_entropy_with_logits(w2 @ h + b2, y)

loss = loss_fn()
loss.backward()                       # autograd fills .grad on every leaf
for name, t in [("W1", W1), ("b1", b1), ("w2", w2), ("b2", b2)]:
    print(f"dL/d{name} =", t.grad.tolist())

with torch.no_grad():                 # one plain gradient-descent step, lr = 0.1
    for t in (W1, b1, w2, b2):
        t -= 0.1 * t.grad
print(f"loss before {loss.item():.4f}  after {loss_fn().item():.4f}")
```

```text
dL/dW1 = [[-0.5, -1.0], [0.25, 0.5]]
dL/db1 = [-0.5, 0.25]
dL/dw2 = [-0.5, -1.0]
dL/db2 = -0.5
loss before 0.6931  after 0.4116
```

Three methods, one answer: the hand derivation, the numerical slope and autograd agree, and the step lowers the loss to 0.4116 exactly as on paper. In real work you never write the backward pass yourself, but when a custom loss or a masking trick produces garbage, a finite-difference check on a tiny input is the fastest way to find out whether the gradient is wrong.

---

## 6. Computational graphs and autograd

### 6.1 The graph

A **computational graph** records every operation of the forward pass as a node, with edges to its inputs. Each node knows one thing: its **local derivative** with respect to each input (for \(a\cdot b\), the derivative with respect to \(a\) is \(b\)). **Reverse-mode automatic differentiation (autograd)** walks the graph from the loss back to the inputs and multiplies local derivatives along the way.

\[
\frac{\partial L}{\partial a}=\frac{\partial L}{\partial d}\cdot\frac{\partial d}{\partial a}
\]

!!! example "On paper"

    \(L=d^2\) with \(d=ab+c\), at \(a=2\), \(b=-3\), \(c=10\). Forward: \(d=-6+10=4\), \(L=16\). Backward, with \(\partial L/\partial d=2d\):

    \[
    \begin{aligned}
    \partial L/\partial d &= 2\cdot4 = 8\\
    \partial L/\partial a &= 8\cdot b = -24\\
    \partial L/\partial b &= 8\cdot a = 16\\
    \partial L/\partial c &= 8\cdot1 = 8
    \end{aligned}
    \]

    Each node only needed its own local rule. When a value is used twice (here \(L=d\cdot d\) uses \(d\) twice), the gradients from both uses add up.

### 6.2 A scalar autograd engine in 40 lines

This is the idea behind PyTorch's autograd, in the style of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd): every `Value` remembers its parents and the local derivative towards each, and `backward()` visits the graph in reverse topological order.

```python
import math


class Value:
    """A scalar that remembers how it was computed, so it can backpropagate."""

    def __init__(self, data, parents=(), local_grads=()):
        self.data, self.grad = data, 0.0
        self.parents, self.local_grads = parents, local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def tanh(self):
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t * t,))

    __radd__, __rmul__ = __add__, __mul__

    def __sub__(self, other):
        return self + (-1 * other)

    def backward(self):
        order, seen = [], set()

        def visit(v):                        # topological sort: parents before children
            if id(v) not in seen:
                seen.add(id(v))
                for p in v.parents:
                    visit(p)
                order.append(v)

        visit(self)
        self.grad = 1.0                      # dL/dL = 1
        for v in reversed(order):            # chain rule, from the output back
            for p, g in zip(v.parents, v.local_grads):
                p.grad += g * v.grad         # += : a value used twice collects both


a, b, c = Value(2.0), Value(-3.0), Value(10.0)
d = a * b + c
L = d * d
L.backward()
print(f"L = {L.data}, dL/da = {a.grad}, dL/db = {b.grad}, dL/dc = {c.grad}")
```

```text
L = 16.0, dL/da = -24.0, dL/db = 16.0, dL/dc = 8.0
```

### 6.3 Train a network with it

The same engine (compacted) trains a 2-4-1 tanh network on XOR. The three lines inside the loop, zero the gradients, backpropagate, step, are the heart of every training loop you will ever write.

```python
import math
import random


class Value:                                   # the engine from 6.2, compacted
    def __init__(self, data, parents=(), local_grads=()):
        self.data, self.grad, self.parents, self.local_grads = data, 0.0, parents, local_grads
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        return Value(self.data + o.data, (self, o), (1.0, 1.0))
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        return Value(self.data * o.data, (self, o), (o.data, self.data))
    def tanh(self):
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t * t,))
    __radd__, __rmul__ = __add__, __mul__
    def __sub__(self, o): return self + (-1 * o)
    def backward(self):
        order, seen = [], set()
        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v.parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0
        for v in reversed(order):
            for p, g in zip(v.parents, v.local_grads):
                p.grad += g * v.grad


random.seed(0)
H = 4                                          # a 2-4-1 network, tanh everywhere
W1 = [[Value(random.uniform(-1, 1)) for _ in range(2)] for _ in range(H)]
b1 = [Value(0.0) for _ in range(H)]
w2 = [Value(random.uniform(-1, 1)) for _ in range(H)]
b2 = Value(0.0)
params = [p for row in W1 for p in row] + b1 + w2 + [b2]

def net(x):
    h = [(sum(wi * xi for wi, xi in zip(row, x)) + b).tanh() for row, b in zip(W1, b1)]
    return (sum(wi * hi for wi, hi in zip(w2, h)) + b2).tanh()

data = [([0, 0], -1), ([0, 1], 1), ([1, 0], 1), ([1, 1], -1)]   # XOR, labels -1/+1
for step in range(201):
    errors = [net(x) - y for x, y in data]
    loss = sum(e * e for e in errors) * 0.25   # mean squared error
    for p in params:
        p.grad = 0.0                           # 1. zero the gradients
    loss.backward()                            # 2. backpropagate
    for p in params:
        p.data -= 0.2 * p.grad                 # 3. one gradient-descent step
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.data:.4f}")
print("predictions:", [round(net(x).data, 2) for x, _ in data])
```

```text
step   0  loss 1.0419
step  50  loss 0.5220
step 100  loss 0.0730
step 150  loss 0.0257
step 200  loss 0.0142
predictions: [-0.91, 0.88, 0.88, -0.86]
```

The network found weights that solve XOR on its own: the predictions have the signs of the labels \((-1,1,1,-1)\).

### 6.4 What PyTorch adds, and the zero-grad trap

PyTorch does the same thing for **tensors** (whole arrays per node instead of one scalar), with fast kernels for CPU, NVIDIA GPUs (CUDA) and Apple GPUs (MPS), and with a graph that is recorded fresh on every forward pass ("define by run"). Three rules follow from the engine you just wrote:

- A tensor is tracked only if it has `requires_grad=True` (parameters of `nn.Module` do by default).
- Gradients **accumulate** with `+=`, exactly like the `p.grad += ...` line above. You must clear them before each backward pass.
- The graph holds every intermediate tensor until `backward()` runs. That is the activation memory of §16.

```python
import torch

w = torch.tensor(3.0, requires_grad=True)
for i in range(3):
    loss = w * w            # dloss/dw = 2w = 6 every time
    loss.backward()
    print(f"after backward #{i + 1}: w.grad = {w.grad.item()}")
w.grad = None               # what optimizer.zero_grad() does by default
(w * w).backward()
print("after zeroing and one more backward:", w.grad.item())
```

```text
after backward #1: w.grad = 6.0
after backward #2: w.grad = 12.0
after backward #3: w.grad = 18.0
after zeroing and one more backward: 6.0
```

Forgetting `zero_grad()` turns each update into the sum of every gradient so far, so the steps keep growing and training soon turns unstable. Accumulating on purpose over several micro-batches is **gradient accumulation** (§10), a legitimate trick, as long as you divide the loss by the number of micro-batches.

---

## 7. The PyTorch training loop

### 7.1 Dataset and DataLoader

A **`Dataset`** answers two questions: how many examples (`__len__`) and give me example *i* (`__getitem__`). A **`DataLoader`** wraps it into an iterator of shuffled **batches** (stacked examples). Think of the pair as a repository plus a paginated, shuffled cursor over it.

```python
import torch
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, Dataset


class TicketDataset(Dataset):
    """Wraps feature rows and labels; DataLoader calls __len__ and __getitem__."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


X, y = make_classification(n_samples=3000, n_features=20, n_informative=8,
                           n_classes=3, n_clusters_per_class=2, random_state=0)
ds = TicketDataset(X, y)
loader = DataLoader(ds, batch_size=64, shuffle=True,
                    generator=torch.Generator().manual_seed(0))
xb, yb = next(iter(loader))
print("examples:", len(ds), " batches per epoch:", len(loader))
print("one batch:", tuple(xb.shape), xb.dtype, tuple(yb.shape), yb.dtype)
print("labels in this batch:", torch.bincount(yb).tolist())
```

```text
examples: 3000  batches per epoch: 47
one batch: (64, 20) torch.float32 (64,) torch.int64
labels in this batch: [24, 17, 23]
```

The synthetic data stands in for a three-queue ticket router with 20 numeric features per ticket. Features must be `float32`; class labels for `CrossEntropyLoss` must be `int64` (`torch.long`). A wrong dtype is one of the most common first errors. For text, `__getitem__` would tokenize and return token ids; with `num_workers > 0` the loader prepares batches in background processes while the model trains.

### 7.2 The whole loop

Model, loss, optimizer, then four lines per batch. Evaluation switches the model to **eval mode** (dropout and batch-norm behave differently, §12–13) and wraps the forward pass in **`torch.no_grad()`** so no graph is recorded.

```python
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
X, y = make_classification(n_samples=3000, n_features=20, n_informative=8,
                           n_classes=3, n_clusters_per_class=2, random_state=0)
X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.3, random_state=0)
X_va, X_te, y_va, y_te = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)
baseline = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
print(f"linear baseline: val acc {baseline.score(X_va, y_va):.3f}")

def loader(X, y, **kw):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y))
    return DataLoader(ds, **kw)

train = loader(X_tr, y_tr, batch_size=64, shuffle=True)
val, test = loader(X_va, y_va, batch_size=256), loader(X_te, y_te, batch_size=256)

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Dropout(0.1),
                      nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

def accuracy(dl):
    model.eval()                                  # dropout off
    with torch.no_grad():                         # no graph, less memory
        hits = sum((model(xb).argmax(1) == yb).sum().item() for xb, yb in dl)
    return hits / len(dl.dataset)

for epoch in range(1, 31):
    model.train()                                 # dropout on
    total = 0.0
    for xb, yb in train:
        opt.zero_grad()                           # 1. clear old gradients
        loss = loss_fn(model(xb), yb)             # 2. forward pass and loss
        loss.backward()                           # 3. backward pass
        opt.step()                                # 4. update the weights
        total += loss.item() * len(yb)
    if epoch % 5 == 0:
        print(f"epoch {epoch:2d}  train loss {total / len(train.dataset):.3f}"
              f"  val acc {accuracy(val):.3f}")
print(f"test acc (measured once, at the end): {accuracy(test):.3f}")
```

```text
linear baseline: val acc 0.767
epoch  5  train loss 0.463  val acc 0.804
epoch 10  train loss 0.354  val acc 0.867
epoch 15  train loss 0.289  val acc 0.878
epoch 20  train loss 0.238  val acc 0.898
epoch 25  train loss 0.212  val acc 0.896
epoch 30  train loss 0.192  val acc 0.889
test acc (measured once, at the end): 0.887
```

The MLP reaches 88.7% test accuracy (about 89% on validation) against 76.7% validation accuracy for the linear baseline: the data has curved class boundaries that a linear model cannot follow. Note the discipline from [47](47-ai-classical-ml.md): the validation set is watched during training (and would choose the epoch count), the test set is touched once.

### 7.3 Loop pitfalls

| Symptom | Cause |
|---|---|
| Validation accuracy jumps around between identical runs | forgot `model.eval()`, so dropout is still on |
| Memory grows every evaluation batch | forgot `torch.no_grad()`, graphs are kept |
| Memory grows every training step | `total += loss` instead of `loss.item()`: you keep every graph alive |
| Loss never moves | the optimizer was built before some parameters existed, or `requires_grad=False` |
| Loss explodes after a few hundred steps | `zero_grad()` missing (§6.4) or learning rate too high (§9) |
| Great validation, poor production | data leakage across splits, or features computed differently at serving time |

The four-line loop is the same one that fine-tunes a 14B model; libraries such as the MLX LoRA trainer in [08](08-finetuning.md) wrap it, add gradient accumulation and checkpointing, and stream the data from JSONL.

---

## 8. Optimizers

An **optimizer** turns gradients into parameter updates. Plain gradient descent works, but the update rule changes how fast and how stably you reach a good solution.

### 8.1 SGD

**Stochastic gradient descent (SGD)** is §5.3 on a random mini-batch instead of the full dataset:

\[
\theta_t=\theta_{t-1}-\eta\,g_t
\]

!!! example "On paper"

    \(\theta_0=1\), gradient \(g=0.5\), learning rate \(\eta=0.1\):

    \[
    \theta_1 = 1-0.1\cdot0.5 = 0.95
    \]

    The step size is proportional to the gradient. A steep direction gets a big step, a flat direction a tiny one, which is exactly the wrong way round on long, narrow valleys (§8.5).

### 8.2 Momentum

**Momentum** keeps a running velocity, so consistent gradients build up speed and oscillating ones cancel. PyTorch's form:

\[
v_t=\mu\,v_{t-1}+g_t,\qquad \theta_t=\theta_{t-1}-\eta\,v_t
\]

!!! example "On paper"

    \(\mu=0.9\), \(\eta=0.1\), the same gradient \(g=0.5\) twice, \(v_0=0\):

    \[
    \begin{aligned}
    v_1 &= 0.5\\
    \theta_1 &= 1-0.1\cdot0.5 = 0.95\\
    v_2 &= 0.9\cdot0.5+0.5 = 0.95\\
    \theta_2 &= 0.95-0.1\cdot0.95 = 0.855
    \end{aligned}
    \]

    The second step is almost twice the first. With a constant gradient the velocity tends to \(g/(1-\mu)=10g\): momentum 0.9 means up to ten times the plain step.

### 8.3 Adam

**Adam** keeps two running averages per parameter: \(m\), the mean gradient (momentum), and \(v\), the mean *squared* gradient. It divides the first by the square root of the second, so each parameter gets its own step size. The \(1-\beta^t\) terms correct for both averages starting at 0.

\[
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)\,g_t\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)\,g_t^2\\
\theta_t&=\theta_{t-1}-\eta\,\frac{m_t/(1-\beta_1^{t})}{\sqrt{v_t/(1-\beta_2^{t})}+\epsilon}
\end{aligned}
\]

!!! example "On paper"

    PyTorch defaults \(\beta_1=0.9\), \(\beta_2=0.999\), \(\eta=0.001\); take \(\epsilon\approx0\), \(\theta_0=1\), \(g_1=0.5\):

    \[
    \begin{aligned}
    m_1 &= 0.1\cdot0.5 = 0.05\\
    v_1 &= 0.001\cdot0.25 = 0.00025\\
    m_1/(1-0.9) &= 0.5\\
    \sqrt{v_1/(1-0.999)} &= \sqrt{0.25} = 0.5\\
    \theta_1 &= 1-0.001\cdot0.5/0.5 = 0.999
    \end{aligned}
    \]

    The first Adam step is exactly \(\eta\) against the sign of the gradient, whether \(g\) is 0.5 or 5,000. That scale-invariance is why one learning rate works across layers whose gradients differ by orders of magnitude.

### 8.4 AdamW: decoupled weight decay

**Weight decay** shrinks every weight a little each step (§13.2). **AdamW** applies it directly to the weights instead of mixing it into the gradient, where Adam's division would distort it. It is the default optimizer for transformers; [32 §16](32-mathematics.md#16-optimisation) runs one AdamW step with the large-model settings \(\beta_2=0.95\), \(\lambda=0.1\).

\[
\theta_t=\theta_{t-1}-\eta\,\big(\text{Adam step}+\lambda\,\theta_{t-1}\big)
\]

!!! example "On paper"

    The §8.3 step plus \(\lambda=0.01\):

    \[
    \begin{aligned}
    \text{decay} &= 0.001\cdot0.01\cdot1 = 0.00001\\
    \theta_1 &= 1-0.001-0.00001 = 0.99899
    \end{aligned}
    \]

    Tiny per step, but it compounds: with no gradient at all, \(10{,}000\) steps would shrink the weight by a factor \((1-10^{-5})^{10^4}\approx e^{-0.1}\approx0.905\).

The four updates in PyTorch, starting from \(\theta=1\) with a constant gradient of 0.5:

```python
import torch

def run(make_opt, steps=1):
    w = torch.tensor([1.0], requires_grad=True)
    opt = make_opt([w])
    for _ in range(steps):
        opt.zero_grad()
        (0.5 * w).sum().backward()        # loss = 0.5 w, so the gradient is always 0.5
        opt.step()
    return w.item()

print(f"SGD            1 step : {run(lambda p: torch.optim.SGD(p, lr=0.1)):.6f}")
print(f"SGD+momentum   2 steps: {run(lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9), 2):.6f}")
print(f"Adam           1 step : {run(lambda p: torch.optim.Adam(p, lr=1e-3)):.6f}")
print(f"AdamW (wd .01) 1 step : {run(lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=0.01)):.6f}")
```

```text
SGD            1 step : 0.950000
SGD+momentum   2 steps: 0.855000
Adam           1 step : 0.999000
AdamW (wd .01) 1 step : 0.998990
```

### 8.5 Why adaptive steps help: a narrow valley

The loss \(\tfrac12(100w_0^2+w_1^2)\) is 100 times steeper along \(w_0\) than along \(w_1\), like a real network whose layers have very different gradient scales. SGD's learning rate is capped by the steep direction (above 0.02 it diverges, §9.1), so the flat direction crawls.

```python
import torch

def steps_to_converge(make_opt, max_steps=5000):
    w = torch.tensor([1.0, 1.0], requires_grad=True)
    opt = make_opt([w])
    for step in range(1, max_steps + 1):
        loss = 0.5 * (100 * w[0] ** 2 + w[1] ** 2)   # steep in w0, flat in w1
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < 1e-4:
            return step
    return None

print("SGD lr=0.019          ", steps_to_converge(lambda p: torch.optim.SGD(p, lr=0.019)))
print("SGD lr=0.021          ", steps_to_converge(lambda p: torch.optim.SGD(p, lr=0.021)))
print("SGD+momentum lr=0.019 ", steps_to_converge(lambda p: torch.optim.SGD(p, lr=0.019, momentum=0.9)))
print("Adam lr=0.1           ", steps_to_converge(lambda p: torch.optim.Adam(p, lr=0.1)))
```

```text
SGD lr=0.019           224
SGD lr=0.021           None
SGD+momentum lr=0.019  126
Adam lr=0.1            83
```

`None` means SGD at 0.021 never converged: the steep coordinate overshoots further every step. Momentum and Adam get there faster without that cliff.

| Optimizer | State per parameter | Use it for |
|---|---|---|
| SGD | none | convex problems, some vision training, when memory is tight |
| SGD + momentum | 1 value | CNNs, classic vision recipes |
| Adam / AdamW | 2 values (\(m\), \(v\)) | transformers, LoRA, almost every LLM recipe |

The "2 values per parameter" is not free: for full fine-tuning it is the biggest memory line in §16.

---

## 9. The learning rate

The **learning rate** \(\eta\) is the step size. It is the single most important hyperparameter: too high and training diverges, too low and it crawls or stalls.

### 9.1 Too high, too low

Gradient descent on \(f(w)=w^2\), whose gradient is \(2w\):

\[
w_{t+1}=w_t-\eta\cdot2w_t=(1-2\eta)\,w_t
\]

!!! example "On paper"

    Start at \(w_0=1\); the minimum is at 0.

    \[
    \begin{aligned}
    \eta=0.1:\ w &= 1\to0.8\to0.64\\
    \eta=1.1:\ w &= 1\to-1.2\to1.44
    \end{aligned}
    \]

    Each step multiplies \(w\) by \(1-2\eta\). It shrinks only while \(|1-2\eta|<1\), so it converges only for \(\eta<1\): at \(\eta=1\) it bounces between \(+1\) and \(-1\) forever, and above 1 it diverges. A steeper loss (curvature 100 instead of 2) moves the cliff down to \(\eta=0.02\), as in §8.5.

```python
w0 = 1.0
for lr in (0.01, 0.1, 0.45, 0.9, 1.1):
    w, trace = w0, []
    for _ in range(5):
        w = w - lr * 2 * w            # gradient of w^2 is 2w
        trace.append(w)
    print(f"lr={lr:<5}", " ".join(f"{v:+.4f}" for v in trace))
```

```text
lr=0.01  +0.9800 +0.9604 +0.9412 +0.9224 +0.9039
lr=0.1   +0.8000 +0.6400 +0.5120 +0.4096 +0.3277
lr=0.45  +0.1000 +0.0100 +0.0010 +0.0001 +0.0000
lr=0.9   -0.8000 +0.6400 -0.5120 +0.4096 -0.3277
lr=1.1   -1.2000 +1.4400 -1.7280 +2.0736 -2.4883
```

Five regimes in one table: crawling (0.01), steady (0.1), nearly optimal (0.45), oscillating but converging (0.9), and diverging (1.1). The same regimes show up on a real network, with SGD on the §7 classifier:

```python
import torch
import torch.nn as nn
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=3000, n_features=20, n_informative=8,
                           n_classes=3, n_clusters_per_class=2, random_state=0)
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y)

def train(lr, steps=300):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 3))
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(steps):
        idx = torch.randint(0, len(y), (64,), generator=g)
        loss = nn.functional.cross_entropy(model(X[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses

for lr in (0.001, 0.1, 0.5, 1.0, 2.0):
    L = train(lr)
    avg = lambda a, b: sum(L[a:b]) / (b - a)
    print(f"lr={lr:<6} mean loss, steps 1-20: {avg(0, 20):9.3g}   "
          f"81-100: {avg(80, 100):.3f}   281-300: {avg(280, 300):.3f}")
```

```text
lr=0.001  mean loss, steps 1-20:      1.32   81-100: 1.171   281-300: 1.019
lr=0.1    mean loss, steps 1-20:     0.896   81-100: 0.486   281-300: 0.389
lr=0.5    mean loss, steps 1-20:     0.708   81-100: 0.336   281-300: 0.347
lr=1.0    mean loss, steps 1-20:      1.04   81-100: 0.414   281-300: 0.504
lr=2.0    mean loss, steps 1-20:  8.54e+05   81-100: nan   281-300: nan
```

At 0.001 the loss is still above 1.0 after 300 steps, close to the \(\ln3\approx1.10\) of a model that knows nothing (§18.1). Around 0.1–0.5 it trains well. At 1.0 it ends worse than at 0.5, and at 2.0 it explodes: a mean of about 854,000 over the first 20 steps, and `nan` (not a number) by steps 81–100. Once a `nan` appears, every later step is `nan`: production trainers check for it and stop or roll back.

### 9.2 Warmup and cosine decay

Large models use a **schedule**: a short **warmup** that ramps the rate up from 0 (early gradients are large and Adam's statistics are not yet reliable), then a **cosine decay** that lowers it smoothly so the final steps settle into a minimum. The first line below applies during warmup (\(t<T_w\)), the second afterwards, where \(s\) is the fraction of the decay phase already done:

\[
\begin{aligned}
\eta_t&=\eta_{\max}\,t/T_w\\
\eta_t&=\eta_{\min}+\tfrac12(\eta_{\max}-\eta_{\min})(1+\cos\pi s)\\
s&=(t-T_w)/(T-T_w)
\end{aligned}
\]

!!! example "On paper"

    \(\eta_{\max}=3\times10^{-4}\), \(\eta_{\min}=3\times10^{-5}\) (10% of the peak), \(T_w=100\) warmup steps, \(T=1000\) steps in total:

    \[
    \begin{aligned}
    \eta_{50} &= 3\times10^{-4}\cdot0.5 = 1.5\times10^{-4}\\
    s_{550} &= 450/900 = 0.5\\
    \eta_{550} &= 3\times10^{-5}+1.35\times10^{-4}\\
    &= 1.65\times10^{-4}\\
    \eta_{1000} &= 3\times10^{-5}
    \end{aligned}
    \]

    At \(t=550\), \(\cos(\pi/2)=0\): halfway through the decay the rate is halfway between peak and floor. At \(t=1000\), \(\cos\pi=-1\) leaves only the floor. Warmup is usually 1–5% of the steps.

```python
import math
import torch

opt = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=3e-4)
warmup, total, floor = 100, 1000, 0.1          # floor: end at 10% of the peak

def lr_factor(step):
    if step < warmup:
        return step / warmup                     # linear warmup from 0
    progress = (step - warmup) / (total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
for step in range(total + 1):
    if step in (0, 50, 100, 325, 550, 775, 1000):
        print(f"step {step:4d}  lr {opt.param_groups[0]['lr']:.3e}")
    opt.step()
    sched.step()
```

```text
step    0  lr 0.000e+00
step   50  lr 1.500e-04
step  100  lr 3.000e-04
step  325  lr 2.605e-04
step  550  lr 1.650e-04
step  775  lr 6.954e-05
step 1000  lr 3.000e-05
```

**Pitfalls.** Calling `sched.step()` once per *epoch* when the schedule was written in *steps* stretches warmup over the whole run. Resuming from a checkpoint without restoring the scheduler state restarts warmup in the middle of training. Copying a learning rate from a different model size or batch size is a guess, not a setting; LoRA typically uses higher rates (around \(10^{-4}\)) than full fine-tuning because far fewer weights move ([08](08-finetuning.md), [53](53-ai-fine-tuning.md)).

---

## 10. Batch size, epochs and steps

### 10.1 The arithmetic

An **epoch** is one pass over the training set. A **step** is one optimizer update, computed from one **batch**. The **effective batch** is micro-batch size × gradient-accumulation steps × number of devices.

\[
\text{steps}=\Big\lceil\frac{N}{B}\Big\rceil\times\text{epochs}
\]

!!! example "On paper"

    50,000 labelled tickets, batch 32, 3 epochs:

    \[
    \begin{aligned}
    \lceil50{,}000/32\rceil &= \lceil1562.5\rceil = 1{,}563\\
    1{,}563\cdot3 &= 4{,}689\ \text{steps}
    \end{aligned}
    \]

    With micro-batches of 4 and 8 accumulation steps the effective batch is still 32 and the run still makes 4,689 updates, but it runs \(50{,}000/4=12{,}500\) forward/backward passes per epoch with an eighth of the activation memory. For an LLM count tokens: 8 sequences of 2,048 tokens is \(16{,}384\) tokens per step.

Warmup and schedules are written in steps, so this number is the first thing to compute before a run. On the 32 GB Mac, the micro-batch is limited by memory ([02 §3.4](02-hardware.md#34-batch-size-and-sequence-length)); gradient accumulation recovers the effective batch.

### 10.2 What batch size changes

A mini-batch gradient is an *estimate* of the full-data gradient. Its error shrinks with the square root of the batch size, like any average of noisy samples ([46](46-ai-math.md), standard error):

\[
\text{gradient noise}\propto\frac{1}{\sqrt B}
\]

!!! example "On paper"

    Going from \(B=8\) to \(B=32\) multiplies the batch by 4:

    \[
    \frac{1}{\sqrt{4}} = 0.5
    \]

    The noise halves. The code below measures 1.922 at batch 8 and 0.960 at 32, a ratio of 0.4995. Four times the compute per step buys only half the noise.

```python
import torch
import torch.nn as nn
from sklearn.datasets import make_classification

torch.manual_seed(0)
X, y = make_classification(n_samples=20000, n_features=20, n_informative=8,
                           n_classes=3, random_state=0)
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y)
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 3))

def grad_vector(idx):
    model.zero_grad()
    nn.functional.cross_entropy(model(X[idx]), y[idx]).backward()
    return torch.cat([p.grad.flatten() for p in model.parameters()])

full = grad_vector(torch.arange(len(y)))       # the "true" gradient over all data
g = torch.Generator().manual_seed(0)
for B in (8, 32, 128, 512):
    errs = [(grad_vector(torch.randint(0, len(y), (B,), generator=g)) - full).norm().item()
            for _ in range(200)]
    print(f"batch {B:4d}: mean gradient error {sum(errs) / len(errs):.3f}")
```

```text
batch    8: mean gradient error 1.922
batch   32: mean gradient error 0.960
batch  128: mean gradient error 0.478
batch  512: mean gradient error 0.232
```

Practical consequences:

- **Bigger batches** give smoother gradients and better hardware use, and usually tolerate a higher learning rate. Past a point (the "critical batch size") they only waste compute.
- **Smaller batches** are noisier; the noise can help generalisation but needs a lower learning rate.
- **Change the batch, revisit the learning rate.** A common starting rule is to scale the rate with the square root of the batch change for Adam; then measure.
- **Epochs for fine-tuning are few.** LLM fine-tunes usually run 1–3 epochs; more mostly memorises (§13.3).

---

## 11. Initialization

### 11.1 Why not all zeros

If every weight starts equal, every hidden unit computes the same thing, receives the same gradient, and stays identical forever: the layer behaves like one unit. Random initialization **breaks the symmetry**.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
X = torch.randn(256, 4)
y = (X[:, 0] * X[:, 1] > 0).long()            # a non-linear rule

model = nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 2))
for p in model.parameters():
    nn.init.constant_(p, 0.1)                 # every weight equal (zeros is the extreme case)
opt = torch.optim.SGD(model.parameters(), lr=0.5)
for _ in range(200):
    loss = nn.functional.cross_entropy(model(X), y)
    opt.zero_grad(); loss.backward(); opt.step()
print("hidden-unit weight rows after 200 steps:")
print(model[0].weight.detach().round(decimals=3))
```

```text
hidden-unit weight rows after 200 steps:
tensor([[ 0.3150, -0.2560,  0.1430,  0.0230],
        [ 0.3150, -0.2560,  0.1430,  0.0230],
        [ 0.3150, -0.2560,  0.1430,  0.0230]])
```

200 steps later the three hidden units are still clones. With exact zeros it is worse: the output weights are 0, so no gradient reaches the hidden layer at all.

### 11.2 Xavier and He scaling

The *scale* of the random weights matters too. Each layer multiplies the variance of its signal by about \(n_{\text{in}}\cdot\operatorname{Var}(w)\). If that factor is above 1 the activations explode with depth; below 1 they vanish. The standard fixes choose \(\operatorname{Var}(w)\) to make it about 1:

\[
\begin{aligned}
\text{Xavier (tanh, sigmoid):}\ \operatorname{Var}(w)&=\frac{2}{n_{\text{in}}+n_{\text{out}}}\\
\text{He (ReLU):}\ \operatorname{Var}(w)&=\frac{2}{n_{\text{in}}}
\end{aligned}
\]

!!! example "On paper"

    A \(512\times512\) layer (\(n_{\text{in}}=n_{\text{out}}=512\)):

    \[
    \begin{aligned}
    \text{Xavier std} &= \sqrt{2/1024} = \sqrt{1/512} \approx 0.0442\\
    \text{He std} &= \sqrt{2/512} = 1/16 = 0.0625
    \end{aligned}
    \]

    He uses 2 because ReLU zeroes half its inputs and so halves the variance. With std 1 instead, each ReLU layer multiplies the variance by \(512/2=256\) and the std by 16: after 5 layers, \(16^5\approx10^6\).

```python
import torch

torch.manual_seed(0)
x0 = torch.randn(1024, 512)                    # a batch of inputs with std 1
for name, std in [("std 1.0", 1.0), ("std 0.01", 0.01), ("He: sqrt(2/512)", (2 / 512) ** 0.5)]:
    x, report = x0, []
    for layer in range(1, 21):                 # 20 layers of Linear(512, 512) + ReLU
        W = torch.randn(512, 512) * std
        x = torch.relu(x @ W)
        if layer in (1, 5, 10, 20):
            report.append(f"L{layer}: {x.std().item():.2e}")
    print(f"{name:<16}", "  ".join(report))
```

```text
std 1.0          L1: 1.32e+01  L5: 8.56e+05  L10: 8.03e+11  L20: 8.73e+23
std 0.01         L1: 1.32e-01  L5: 8.34e-05  L10: 8.83e-09  L20: 8.68e-17
He: sqrt(2/512)  L1: 8.27e-01  L5: 7.43e-01  L10: 7.42e-01  L20: 8.38e-01
```

Std 1.0 reaches \(10^{23}\) by layer 20 (in fp16, whose largest value is 65,504, the same stack already overflows to `inf` at layer 4, §15); std 0.01 shrinks to \(10^{-17}\), effectively zero; He keeps the signal near 1 at every depth. PyTorch's `nn.Linear` already uses a scaled uniform init, so you rarely set this by hand, but custom layers and very deep stacks need it. Transformers use a small normal init (std 0.02, Qwen3's `"initializer_range": 0.02`) plus normalization and residual scaling; see [32 §16.1](32-mathematics.md#161-initialisation).

---

## 12. Normalization

A **normalization layer** rescales activations to a standard range at every forward pass, so the next layer always sees inputs of a predictable size. It makes training faster and much less sensitive to the learning rate and initialization. The three you will meet differ in *which numbers they average over*.

### 12.1 BatchNorm

**BatchNorm** normalizes each feature using the mean and variance over the examples in the batch, then applies a learned scale \(\gamma\) and shift \(\beta\):

\[
\hat x_{ij}=\gamma_j\,\frac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}}+\beta_j,
\qquad \mu_j=\frac1B\sum_{i=1}^{B}x_{ij}
\]

!!! example "On paper"

    Feature 1 across a batch of three examples has values \((-1,\,3,\,1)\); take \(\gamma=1\), \(\beta=0\), \(\epsilon\approx0\):

    \[
    \begin{aligned}
    \mu &= (-1+3+1)/3 = 1\\
    \sigma^2 &= (4+4+0)/3 = 8/3 \approx 2.667\\
    \hat x &= (-2,\,2,\,0)/1.633 = (-1.225,\,1.225,\,0)
    \end{aligned}
    \]

    Every example's value depends on the *other examples in the batch*. At inference, with one example, BatchNorm switches to running averages collected during training: a train/serve difference that `model.eval()` controls.

### 12.2 LayerNorm and RMSNorm

**LayerNorm** normalizes each example on its own, over its features. **RMSNorm** skips the mean subtraction and divides by the root mean square; it is cheaper and is what Qwen3 and Gemma use (`rms_norm_eps` in the config):

\[
\begin{aligned}
\text{LayerNorm}(\mathbf x)&=\gamma\odot\frac{\mathbf x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta\\
\text{RMSNorm}(\mathbf x)&=\gamma\odot\frac{\mathbf x}{\sqrt{\tfrac1d\sum_i x_i^2+\epsilon}}
\end{aligned}
\]

!!! example "On paper"

    One example \(\mathbf x=(-1,-1,7,7)\), \(\gamma=1\), \(\beta=0\), \(\epsilon\approx0\). Its mean is 3, its deviations are \((-4,-4,4,4)\):

    \[
    \begin{aligned}
    \sigma^2 &= 16,\ \sigma = 4\\
    \text{LayerNorm} &= (-4,-4,4,4)/4\\
    &= (-1,-1,1,1)\\
    \tfrac14\textstyle\sum x_i^2 &= (1+1+49+49)/4 = 25\\
    \text{RMSNorm} &= (-1,-1,7,7)/5\\
    &= (-0.2,-0.2,1.4,1.4)
    \end{aligned}
    \]

    LayerNorm removes the offset and the scale; RMSNorm removes only the scale. Neither looks at other examples, so a batch of 1 at inference behaves exactly like training.

```python
import numpy as np
import torch
import torch.nn as nn

X = np.array([[-1.0, -1.0, 7.0, 7.0],      # example 1 (4 features)
              [3.0, 5.0, 3.0, 5.0],        # example 2
              [1.0, 2.0, 5.0, 0.0]])       # example 3
eps = 1e-6

bn = (X - X.mean(axis=0)) / np.sqrt(X.var(axis=0) + eps)          # per feature, across the batch
ln = (X - X.mean(axis=1, keepdims=True)) / np.sqrt(X.var(axis=1, keepdims=True) + eps)
rms = X / np.sqrt((X ** 2).mean(axis=1, keepdims=True) + eps)     # per example, no centring
for name, out in [("BatchNorm", bn), ("LayerNorm", ln), ("RMSNorm", rms)]:
    print(f"{name:<10}", np.round(out, 3).tolist())

T = torch.tensor(X, dtype=torch.float32)
print("torch agrees:",
      np.allclose(nn.BatchNorm1d(4, eps=eps)(T).detach().numpy(), bn, atol=1e-5),
      np.allclose(nn.LayerNorm(4, eps=eps)(T).detach().numpy(), ln, atol=1e-5),
      np.allclose(nn.RMSNorm(4, eps=eps)(T).detach().numpy(), rms, atol=1e-5))
```

```text
BatchNorm  [[-1.225, -1.225, 1.225, 1.019], [1.225, 1.225, -1.225, 0.34], [0.0, 0.0, 0.0, -1.359]]
LayerNorm  [[-1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, -1.0, 1.0], [-0.535, 0.0, 1.604, -1.069]]
RMSNorm    [[-0.2, -0.2, 1.4, 1.4], [0.728, 1.213, 0.728, 1.213], [0.365, 0.73, 1.826, 0.0]]
torch agrees: True True True
```

### 12.3 Which one, and why transformers chose per-example norms

| | BatchNorm | LayerNorm | RMSNorm |
|---|---|---|---|
| Averages over | the batch, per feature | the features, per example | the features, per example (no mean) |
| Train vs inference | different (running stats) | identical | identical |
| Works with batch size 1, variable lengths | poorly | yes | yes |
| Typical home | CNNs for vision | BERT, GPT-2, many transformers | Llama, Qwen, Gemma |

Text batches mix sequences of different lengths and padding, and generation runs one token at a time. Per-example norms avoid both problems. Transformers place the norm *before* each sub-layer ("pre-norm"), which keeps the residual path clean; [37 §1](37-architecture.md#1-one-block-one-tensor) shows the block and [38 §3](38-derivations.md#3-rmsnorm-gradient) derives the RMSNorm gradient.

---

## 13. Regularization in deep networks

**Regularization** is anything that stops a model from memorising its training data at the expense of new data. Deep networks have enough parameters to memorise almost anything, so you need it.

### 13.1 Dropout

**Dropout** zeroes each activation with probability \(p\) during training and scales the survivors by \(1/(1-p)\), so the expected value is unchanged. The network cannot rely on any single unit, which acts like training many thinned networks at once. At inference dropout is off.

\[
\tilde h_i=\frac{m_i\,h_i}{1-p},\qquad m_i\sim\text{Bernoulli}(1-p)
\]

!!! example "On paper"

    \(p=0.5\), \(\mathbf h=(2,4,6,8)\), and this step's random mask is \(\mathbf m=(1,0,1,0)\):

    \[
    \tilde{\mathbf h} = (2,0,6,0)/0.5 = (4,\,0,\,12,\,0)
    \]

    Each unit's *expected* output is \((1-p)\cdot h_i/(1-p)=h_i\), so the next layer sees the same average scale in training and inference. This particular draw has mean 4 instead of 5; over many draws it averages out.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
drop = nn.Dropout(p=0.5)
h = torch.tensor([2.0, 4.0, 6.0, 8.0])

drop.train()
print("train mode:", drop(h).tolist(), drop(h).tolist())
big = torch.ones(100_000)
print("train-mode mean of 100k ones:", round(drop(big).mean().item(), 3))
drop.eval()
print("eval mode: ", drop(h).tolist())
```

```text
train mode: [0.0, 0.0, 12.0, 0.0] [0.0, 0.0, 12.0, 16.0]
train-mode mean of 100k ones: 0.999
eval mode:  [2.0, 4.0, 6.0, 8.0]
```

Two calls in train mode give two different masks; eval mode is the identity. Modern LLM pretraining often uses no dropout at all (the data is too large to memorise in one pass; Qwen3's config has `"attention_dropout": 0.0`), but LoRA fine-tunes commonly use a small LoRA dropout.

### 13.2 Weight decay

**Weight decay** pulls every weight towards zero by a fixed fraction each step, which favours simpler functions built from small weights. With plain SGD it is equivalent to adding \(\tfrac{\lambda}{2}\lVert\theta\rVert^2\) to the loss (L2 regularization from [47](47-ai-classical-ml.md)); with Adam it is not, which is why AdamW exists (§8.4).

\[
\theta\leftarrow\theta-\eta\lambda\,\theta=(1-\eta\lambda)\,\theta
\]

!!! example "On paper"

    \(\eta=0.1\), \(\lambda=0.5\), a weight of 2 and (to isolate the effect) zero gradient:

    \[
    \begin{aligned}
    1-\eta\lambda &= 0.95\\
    \theta_1 &= 0.95\cdot2 = 1.9\\
    \theta_{10} &= 2\cdot0.95^{10} \approx 1.197
    \end{aligned}
    \]

    A weight that the data does not keep supporting decays away. Typical LLM values are \(\lambda=0.1\) at learning rates near \(10^{-4}\), so the per-step shrink is about \(10^{-5}\).

Weight decay is usually *not* applied to biases and normalization gains; most trainers exclude them by name.

### 13.3 Overfitting and early stopping

A big network trained on a small, noisy dataset first learns the real pattern, then starts memorising the noise. Training loss keeps falling; validation loss turns around. **Early stopping** keeps the checkpoint with the best validation loss and stops after `patience` epochs without improvement.

```python
import torch
import torch.nn as nn
from sklearn.datasets import make_classification

torch.manual_seed(0)
X, y = make_classification(n_samples=1200, n_features=20, n_informative=5,
                           flip_y=0.2, random_state=1)     # 20% of labels are noise
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y)
Xtr, ytr, Xva, yva = X[:200], y[:200], X[200:], y[200:]      # only 200 training rows

model = nn.Sequential(nn.Linear(20, 256), nn.ReLU(),
                      nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2))
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
history = []
for epoch in range(1, 301):                                  # full-batch training
    loss = nn.functional.cross_entropy(model(Xtr), ytr)
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logits = model(Xva)
        val = nn.functional.cross_entropy(logits, yva).item()
        acc = (logits.argmax(1) == yva).float().mean().item()
    history.append((loss.item(), val, acc))
    if epoch in (10, 25, 50, 100, 200, 300):
        print(f"epoch {epoch:3d}  train {loss.item():.3f}  val {val:.3f}  val acc {acc:.3f}")

best_epoch = min(range(300), key=lambda i: history[i][1]) + 1
patience, best, wait = 20, float("inf"), 0
for epoch, (_, val, _) in enumerate(history, start=1):      # replay early stopping
    best, wait = (val, 0) if val < best else (best, wait + 1)
    if wait == patience:
        break
print(f"lowest val loss at epoch {best_epoch}; patience {patience} stops at epoch {epoch}")
```

```text
epoch  10  train 0.591  val 0.620  val acc 0.684
epoch  25  train 0.466  val 0.564  val acc 0.721
epoch  50  train 0.324  val 0.591  val acc 0.715
epoch 100  train 0.108  val 0.770  val acc 0.686
epoch 200  train 0.009  val 1.280  val acc 0.677
epoch 300  train 0.003  val 1.536  val acc 0.679
lowest val loss at epoch 29; patience 20 stops at epoch 49
```

By epoch 300 the training loss is near 0 (the network has memorised all 200 rows, noise included) while the validation loss has almost tripled (0.564 at epoch 25, near its minimum, against 1.536 at the end). Early stopping with a patience of 20 stops at epoch 49, keeps the epoch-29 weights and saves about 250 epochs of compute. LLM fine-tunes show the same pattern: training loss on a small instruction set heading towards 0 is a warning, not a success ([53](53-ai-fine-tuning.md), [09](09-evaluation.md)).

**Analogy.** Early stopping is a canary rollout watching an error budget: keep going while the health metric improves, roll back to the last good version when it degrades for long enough.

---

## 14. Vanishing and exploding gradients

### 14.1 Products of many factors

The gradient that reaches an early layer is a product of one factor per layer above it:

\[
\frac{\partial L}{\partial h_0}=\frac{\partial L}{\partial h_n}\prod_{k=1}^{n}\frac{\partial h_k}{\partial h_{k-1}}
\]

!!! example "On paper"

    If every layer's factor is 0.25 (the sigmoid's best case, §2.1), ten layers give

    \[
    0.25^{10} \approx 9.5\times10^{-7}
    \]

    and the first layer learns a million times slower than the last. If every factor is 1.5 instead, twenty layers give \(1.5^{20}\approx3{,}325\): one bad batch then produces a huge update. Depth turns "slightly below 1" and "slightly above 1" into "nothing" and "infinity".

```python
import torch
import torch.nn as nn

def first_layer_grad(depth, act, residual=False, he=False):
    torch.manual_seed(0)
    layers = [nn.Linear(64, 64) for _ in range(depth)]
    if he:
        for layer in layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    h = torch.randn(32, 64)
    for layer in layers:
        h = h + act(layer(h)) if residual else act(layer(h))
    h.pow(2).mean().backward()
    return layers[0].weight.grad.norm().item()

for depth in (2, 10, 30):
    print(f"depth {depth:2d}:  sigmoid {first_layer_grad(depth, torch.sigmoid):.1e}"
          f"   relu+He {first_layer_grad(depth, torch.relu, he=True):.1e}"
          f"   sigmoid+residual {first_layer_grad(depth, torch.sigmoid, residual=True):.1e}")
```

```text
depth  2:  sigmoid 7.3e-03   relu+He 6.0e-01   sigmoid+residual 1.4e-01
depth 10:  sigmoid 6.3e-10   relu+He 9.6e-01   sigmoid+residual 4.5e-01
depth 30:  sigmoid 0.0e+00   relu+He 2.4e+00   sigmoid+residual 1.4e+00
```

At depth 30 the plain sigmoid stack's first-layer gradient has underflowed to exactly zero: that layer will never train. The two fixes that made deep networks work are both visible: ReLU with He init, and **residual connections**.

### 14.2 Residual connections

A residual block computes \(h_{k}=h_{k-1}+f(h_{k-1})\) instead of \(h_k=f(h_{k-1})\). Its local factor is \(1+f'\) instead of \(f'\), so the product along the identity path cannot collapse to zero. Every transformer block has two of them ([37 §1](37-architecture.md#1-one-block-one-tensor)); they are why Qwen3-14B can stack 40 blocks. Normalization (§12), careful init (§11) and gradient clipping (below) complete the toolkit ([32 §16.2](32-mathematics.md#162-stability-tools)).

### 14.3 Gradient clipping by norm

**Gradient clipping** caps the size of each update. If the norm of the whole gradient vector (all parameters together) exceeds a threshold \(c\), the vector is scaled down to length \(c\); its direction is kept.

\[
\mathbf g\leftarrow\mathbf g\cdot\min\Big(1,\ \frac{c}{\lVert\mathbf g\rVert}\Big)
\]

!!! example "On paper"

    Two parameters with gradients \((3,4)\), threshold \(c=1\):

    \[
    \begin{aligned}
    \lVert\mathbf g\rVert &= \sqrt{9+16} = 5\\
    \mathbf g &\leftarrow (3,4)\cdot\tfrac15 = (0.6,\,0.8)
    \end{aligned}
    \]

    A gradient of \((0.3,0.4)\) has norm 0.5 and is left alone. Clipping only fires on outliers, so it rarely changes normal training but stops a single bad batch from wrecking the weights.

```python
import torch

a = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
(3 * a + 4 * b).sum().backward()                 # gradients 3 and 4: global norm 5
norm = torch.nn.utils.clip_grad_norm_([a, b], max_norm=1.0)
print("norm before clipping:", norm.item())
print(f"gradients after: {a.grad.item():.4f} {b.grad.item():.4f}")

a.grad, b.grad = torch.tensor([0.3]), torch.tensor([0.4])   # already small: norm 0.5
torch.nn.utils.clip_grad_norm_([a, b], max_norm=1.0)
print(f"small gradients are untouched: {a.grad.item():.4f} {b.grad.item():.4f}")
```

```text
norm before clipping: 5.0
gradients after: 0.6000 0.8000
small gradients are untouched: 0.3000 0.4000
```

Call it between `loss.backward()` and `opt.step()`. The returned pre-clip norm is one of the most useful numbers to log: a run whose gradient norm suddenly jumps by 10× is about to spike. LLM recipes use \(c=1.0\). **Analogy:** a rate limiter on the update size.

---

## 15. Floating-point formats

### 15.1 Bits, range and precision

A floating-point number stores a sign, an **exponent** (which sets the range) and a **mantissa** (which sets the precision):

\[
x=(-1)^{s}\cdot2^{\,e-\text{bias}}\cdot\Big(1+\frac{m}{2^{M}}\Big)
\]

!!! example "On paper"

    fp16 has 5 exponent bits (bias 15) and \(M=10\) mantissa bits. Its largest finite value uses \(e=30\) and \(m=1023\):

    \[
    \begin{aligned}
    2^{30-15}\cdot\big(1+\tfrac{1023}{1024}\big) &= 32{,}768\cdot\tfrac{2047}{1024}\\
    &= 32\cdot2{,}047\\
    &= 65{,}504
    \end{aligned}
    \]

    The step between 1 and the next fp16 number is \(2^{-10}\approx0.000977\). bf16 keeps fp32's 8 exponent bits (so its range reaches about \(3.4\times10^{38}\)) but has only \(M=7\): a step of \(2^{-7}=0.0078\) near 1.

```python
import torch

print(f"{'dtype':<14}{'bits':>5}{'max':>13}{'step at 1.0':>14}{'smallest normal':>17}")
for dt in (torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
    fi = torch.finfo(dt)
    print(f"{str(dt).replace('torch.', ''):<14}{fi.bits:>5}{fi.max:>13.6g}{fi.eps:>14.4g}{fi.tiny:>17.4g}")
ii = torch.iinfo(torch.int8)
print(f"{'int8':<14}{ii.bits:>5}{ii.max:>13}{'1':>14}{'(integers)':>17}")

x = torch.tensor([70000.0, 1.001, 257.0, 1e-8])
for dt in (torch.float16, torch.bfloat16):
    print(f"{str(dt).replace('torch.', ''):<9}", [f"{float(v):.6g}" for v in x.to(dt).float()])
```

```text
dtype          bits          max   step at 1.0  smallest normal
float32          32  3.40282e+38     1.192e-07        1.175e-38
float16          16        65504     0.0009766        6.104e-05
bfloat16         16  3.38953e+38      0.007812        1.175e-38
float8_e4m3fn     8          448         0.125          0.01562
float8_e5m2       8        57344          0.25        6.104e-05
int8              8          127             1       (integers)
float16   ['inf', '1.00098', '257', '0']
bfloat16  ['70144', '1', '256', '1.00117e-08']
```

| Format | Sign/exponent/mantissa bits | Max | Step at 1.0 | Main use |
|---|---|---:|---:|---|
| fp32 | 1 / 8 / 23 | \(3.4\times10^{38}\) | \(1.2\times10^{-7}\) | master weights, optimizer state, reductions |
| fp16 | 1 / 5 / 10 | 65,504 | \(9.8\times10^{-4}\) | older mixed precision, needs loss scaling |
| bf16 | 1 / 8 / 7 | \(3.4\times10^{38}\) | \(7.8\times10^{-3}\) | default for training and serving LLMs |
| fp8 E4M3 | 1 / 4 / 3 | 448 | 0.125 | weights and activations on recent GPUs |
| fp8 E5M2 | 1 / 5 / 2 | 57,344 | 0.25 | gradients on recent GPUs |
| int8 | integer | 127 | 1 | quantized inference with a scale ([49](49-ai-llm-internals.md)) |

The last two lines of the output are the whole story of bf16 versus fp16:

- **70,000 overflows fp16 to `inf` but is fine in bf16**, rounded to 70,144. Between \(2^{16}=65{,}536\) and \(2^{17}\) bf16's step is \(2^{16-7}=512\), and \(137\cdot512=70{,}144\) is the nearest multiple.
- bf16 is *less precise*: \(1.001\) becomes \(1.0\) and 257 becomes 256. fp16 keeps both.
- \(10^{-8}\) underflows to 0 in fp16 and survives in bf16.

Activations of a few thousand are normal inside LLMs, and occasional outlier features reach far higher, so fp16 overflow is a real failure, not a curiosity. That is why bf16 became the default ([32 §16.2](32-mathematics.md#162-stability-tools)).

### 15.2 Mixed precision

**Mixed precision** runs the heavy matrix multiplications in a 16-bit format (much faster on NVIDIA tensor cores, and half the memory traffic everywhere; Apple GPUs have no double-rate half precision, so on the M1 Pro fp16/bf16 math runs at the same ~5.2 TFLOPS peak as fp32 and the gain is memory and bandwidth only, [02 §1](02-hardware.md#1-the-machine)) while keeping a 32-bit **master copy** of the weights and the optimizer state, and doing sums and the loss in fp32. Small updates such as \(10^{-4}\cdot\) a gradient would be lost if added directly to a bf16 weight of 1.0, whose step is 0.0078; the fp32 master copy keeps them.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Linear(512, 512)                       # master weights stay fp32
x = torch.randn(8, 512)
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    y = model(x)                                  # the matmul runs in bf16
    loss = y.float().pow(2).mean()                # reductions back in fp32
loss.backward()
print("weight dtype:", model.weight.dtype, " output dtype:", y.dtype,
      " grad dtype:", model.weight.grad.dtype)

# Why fp16 training needs loss scaling: tiny gradients underflow to zero
g = torch.tensor([1e-8, 3e-6, 2e-4])
print("fp16 as is:      ", [f"{v:.3g}" for v in g.half().float().tolist()])
scaled = (g * 65536).half()                       # scale up before the cast...
print("scaled, unscaled:", [f"{v:.3g}" for v in (scaled.float() / 65536).tolist()])
```

```text
weight dtype: torch.float32  output dtype: torch.bfloat16  grad dtype: torch.float32
fp16 as is:       ['0', '2.98e-06', '0.0002']
scaled, unscaled: ['1e-08', '3e-06', '0.0002']
```

fp16 cannot hold anything smaller than about \(6\times10^{-8}\) (its smallest positive value) and loses precision below \(6\times10^{-5}\), so small gradients vanish or get rounded coarsely; **loss scaling** multiplies the loss (and so every gradient) by a large factor before backward and divides it out in fp32 (`torch.amp.GradScaler` automates it). bf16's fp32-sized exponent makes this unnecessary. On a Mac, MLX trains LoRA adapters with the base weights quantized to 4 bits and the adapter in higher precision; see [08](08-finetuning.md).

---

## 16. Memory for training

### 16.1 The per-parameter bill

Full training with AdamW in mixed precision stores, for every parameter: the bf16 weight (2 bytes), its bf16 gradient (2), an fp32 master weight (4), and Adam's \(m\) and \(v\) in fp32 (4 + 4). Activations come on top.

\[
M_{\text{train}}\approx\underbrace{(2+2+4+4+4)}_{16\ \text{bytes}}\,N+M_{\text{act}}
\]

!!! example "On paper"

    A 1B-parameter model and Qwen3-14B (\(N\approx14.77\)B):

    \[
    \begin{aligned}
    1\text{B}:\ 16\cdot10^{9}\ \text{B} &= 16\ \text{GB}\\
    14\text{B}:\ 16\cdot14.77\times10^{9} &\approx 236.3\ \text{GB}\\
    \text{14B, inference, bf16: } 2N &\approx 29.5\ \text{GB}
    \end{aligned}
    \]

    Training costs 8 times the memory of serving, before activations. The 14B states alone need at least \(\lceil236.3/80\rceil=3\) GPUs of 80 GB, sharded with FSDP or ZeRO.

```python
def training_gb(n_params):
    """Mixed-precision AdamW, bytes per parameter, before activations."""
    parts = {"bf16 weights": 2, "bf16 gradients": 2, "fp32 master weights": 4,
             "fp32 Adam m": 4, "fp32 Adam v": 4}
    return {k: v * n_params / 1e9 for k, v in parts.items()}

for name, n in [("1B", 1.0e9), ("Qwen3-14B", 14.77e9)]:
    parts = training_gb(n)
    print(f"{name}: " + ", ".join(f"{k} {v:.1f}" for k, v in parts.items()))
    print(f"   total {sum(parts.values()):.1f} GB before activations; "
          f"inference in bf16 needs only {2 * n / 1e9:.1f} GB")
```

```text
1B: bf16 weights 2.0, bf16 gradients 2.0, fp32 master weights 4.0, fp32 Adam m 4.0, fp32 Adam v 4.0
   total 16.0 GB before activations; inference in bf16 needs only 2.0 GB
Qwen3-14B: bf16 weights 29.5, bf16 gradients 29.5, fp32 master weights 59.1, fp32 Adam m 59.1, fp32 Adam v 59.1
   total 236.3 GB before activations; inference in bf16 needs only 29.5 GB
```

The handbook's numbers agree: [37 §1.4](37-architecture.md#14-activation-memory) counts 177 GB of fp32 master weights plus moments (12 bytes per parameter) against 29.5 GB of bf16 weights, and [35 §5.4](35-from-zero.md#54-memory-math) uses the same 16 bytes per parameter. Some recipes drop the separate master copy or keep gradients in fp32, so 12–20 bytes per parameter is the realistic range.

### 16.2 Activations

**Activations** are the intermediate tensors the forward pass keeps for the backward pass (§6.4). They scale with batch size × sequence length × width × depth, not with the parameter count, so they decide whether a LoRA run fits once the weights are frozen. With batch size \(B\) and \(b\) bytes per value:

\[
M_{\text{act}}\propto B\times(\text{values per example})\times b
\]

!!! example "On paper"

    The MLP below (1024 → 4096 → GELU → 1024) keeps, per example, its input (1,024 values), the first linear output (4,096, needed by GELU's backward), the GELU output (4,096, needed by the second linear) and the final output (1,024):

    \[
    \begin{aligned}
    1024+4096+4096+1024 &= 10{,}240\ \text{values}\\
    10{,}240\cdot4\ \text{bytes} &= 40\ \text{KiB}\\
    512\cdot40\ \text{KiB} &= 20\ \text{MiB}
    \end{aligned}
    \]

    Double the batch, double the activations; the 32 MiB of weights do not change.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 1024))
weights = {p.data_ptr() for p in model.parameters()}
print(f"weights: {sum(p.numel() for p in model.parameters()) * 4 / 2**20:.1f} MiB in fp32")

for batch in (8, 64, 512):
    saved = []
    def pack(t):                       # called for every tensor autograd keeps for backward
        if t.data_ptr() not in weights:          # count activations, not weights
            saved.append(t.numel() * t.element_size())
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = model(torch.randn(batch, 1024)).pow(2).mean()
    print(f"batch {batch:3d}: activations kept for backward {sum(saved) / 2**20:6.2f} MiB")
```

```text
weights: 32.0 MiB in fp32
batch   8: activations kept for backward   0.31 MiB
batch  64: activations kept for backward   2.50 MiB
batch 512: activations kept for backward  20.00 MiB
```

For a transformer, "per example" means per *token* of every sequence, times 40 layers, plus attention scores that grow with the square of the sequence length unless a fused kernel avoids storing them. [37 §1.4](37-architecture.md#14-activation-memory) works the Qwen3-14B numbers; [02 §3](02-hardware.md#3-the-memory-budget) turns them into a peak-memory budget for QLoRA on the Mac.

### 16.3 The levers

| Lever | Saves | Costs |
|---|---|---|
| Smaller micro-batch + gradient accumulation | activations | wall time |
| Shorter `max_seq_length` | activations, logits | can truncate examples |
| Gradient (activation) checkpointing | most activations; recompute them in backward | ~20–30% more compute ([02](02-hardware.md#32-worked-example-qwen3-14b-at-2k)) |
| LoRA: freeze the base, train small adapters | gradients and optimizer state for 99%+ of weights | some quality ceiling ([53](53-ai-fine-tuning.md)) |
| QLoRA: base weights in 4-bit | ~75% of the weight memory | small quality loss, dequantization cost |
| 8-bit optimizers, sharding (FSDP/ZeRO) | optimizer state | complexity, multiple devices |

---

## 17. GPUs and Apple unified memory

Training and inference are dominated by two resources: **compute** (multiply-adds per second) and **memory** (capacity, and bandwidth: bytes per second between memory and the compute units).

| | Discrete NVIDIA GPU | Apple Silicon (this handbook's M1 Pro) |
|---|---|---|
| Memory | its own VRAM/HBM, separate from system RAM | one pool shared by CPU and GPU ("unified") |
| Capacity | fixed per card (consumer cards up to a few tens of GB; large data-center cards 80 GB or more) | the machine's RAM; 32 GB here, of which MLX gets ~21.3 GiB by default ([02 §1](02-hardware.md#1-the-machine)) |
| Moving data to the GPU | copy over PCIe | no copy |
| Bandwidth | from several hundred GB/s to several TB/s | ~200 GB/s on the M1 Pro |
| Software | CUDA: PyTorch, FlashAttention, fp8 kernels, every training library | MLX, PyTorch's `mps` backend, llama.cpp/Metal |
| Good for | pretraining, full fine-tuning, high-throughput serving | LoRA/QLoRA of models up to ~14B, local serving, development |

Two consequences recur in the next chapters:

- **Capacity decides what fits.** Qwen3-14B in bf16 is 29.5 GB: too big for the Mac's GPU budget and for a 24 GB card; at 4-bit (~8.3 GB) it fits both.
- **Bandwidth decides generation speed.** Generating one token reads every weight once, so tokens per second is capped by bandwidth divided by model bytes; [49](49-ai-llm-internals.md) works the number for the Mac.

Device selection in PyTorch is one line; the same code then runs on any of the three:

```python
import torch

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
x = torch.randn(1024, 1024, device=device)
print("device:", device, "| result lives on:", (x @ x).device)
```

```text
device: cpu | result lives on: cpu
```

This page was built on a CPU-only machine, hence `cpu`. On the M1 Pro the same line picks `mps`; on a CUDA machine, `cuda`. Moving tensors between devices (`.to(device)`) is explicit, and mixing devices in one operation is an error, the GPU equivalent of mixing up two database connections.

---

## 18. Reading loss curves

### 18.1 The first number to check

Before training, a classifier with random small weights outputs nearly equal probabilities for all \(C\) classes, so its loss should be close to \(-\ln(1/C)\):

\[
L_0\approx\ln C
\]

!!! example "On paper"

    Three ticket queues and Qwen3's vocabulary of 151,936 tokens:

    \[
    \begin{aligned}
    \ln3 &\approx 1.099\\
    \ln151{,}936 &\approx 11.93
    \end{aligned}
    \]

    The §9.1 runs started at about 1.3: the right order, since a random init is never perfectly uniform. A language model trained from scratch should start near 11.9; a first loss of 40 means a bug (labels misaligned, logits blown up by bad init, loss summed instead of averaged) before any tuning question.

### 18.2 Symptom → cause → fix

| What you see | Likely cause | What to do |
|---|---|---|
| Step-0 loss far from \(\ln C\) | wrong labels or label shift, init scale, loss reduction | check one batch by hand; print logits and labels |
| Loss is `nan` or `inf` | learning rate too high, fp16 overflow, \(\ln0\), corrupt data | lower LR, bf16, gradient clipping, `BCEWithLogitsLoss`, find the batch |
| Flat from the first step | LR far too low, parameters not in the optimizer, frozen weights, dead ReLUs | print gradient norms per layer; check `requires_grad` |
| Falls, then a sudden spike | LR too high after warmup, one bad batch, fp16 | longer warmup, clip at 1.0, skip or fix the batch, roll back |
| Training ↓, validation ↑ | overfitting (§13.3) | early stopping, more or cleaner data, dropout/weight decay, fewer epochs |
| Training and validation both high and close | underfitting | bigger model, longer training, higher LR, better features |
| Validation better than training | dropout active only in training, easier validation set, leakage | compare in eval mode on both; check the split |
| Very noisy | batch too small, LR too high | bigger effective batch, lower LR; plot a moving average |
| Saw-tooth at each epoch boundary | data not shuffled, or ordered by difficulty | shuffle each epoch |
| Training loss near 0 within an epoch | duplicates between train and validation, label leaked into the input | deduplicate; inspect the easiest examples |
| LLM fine-tune: loss on prompt tokens included | loss mask missing | mask everything but the answer ([06](06-data-format.md)) |

Always log training loss, validation loss, learning rate and gradient norm on the same step axis. A loss curve without the learning-rate curve next to it is half a stack trace.

---

## 19. A short tour: CNNs, RNNs and why transformers won text

### 19.1 Convolutional networks (CNNs)

A **convolution** slides a small set of weights (the **kernel**) along the input and takes a dot product at each position. The same weights are reused everywhere, so a pattern learned in one place is detected in every place.

\[
y_i=\sum_{j=0}^{k-1}w_j\,x_{i+j}
\]

!!! example "On paper"

    Daily ticket counts \(x=(1,2,4,7,7)\) and the kernel \(w=(-1,1)\), "next minus current":

    \[
    \begin{aligned}
    y &= (2-1,\ 4-2,\ 7-4,\ 7-7)\\
    &= (1,\ 2,\ 3,\ 0)
    \end{aligned}
    \]

    The two-weight kernel is a change detector that works on a series of any length. A \(3\times3\) image kernel from 64 channels to 64 channels has \(3\cdot3\cdot64\cdot64+64=36{,}928\) parameters, whatever the image size.

```python
import torch
import torch.nn as nn

signal = torch.tensor([[[1.0, 2.0, 4.0, 7.0, 7.0]]])     # shape [batch, channels, length]
conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, bias=False)
with torch.no_grad():
    conv.weight[:] = torch.tensor([[[-1.0, 1.0]]])        # "next minus current"
print("conv output:", conv(signal).flatten().tolist())

layer = nn.Conv2d(64, 64, kernel_size=3)                  # 64 channels in, 64 out, 3x3 window
print(f"Conv2d(64, 64, 3) parameters: {sum(p.numel() for p in layer.parameters()):,}")
print("32x32 input ->", tuple(layer(torch.randn(1, 64, 32, 32)).shape),
      " 224x224 input ->", tuple(layer(torch.randn(1, 64, 224, 224)).shape))
print(f"a dense layer for the 32x32 case: {64 * 32 * 32 * 64 * 30 * 30:,} weights")
```

```text
conv output: [1.0, 2.0, 3.0, 0.0]
Conv2d(64, 64, 3) parameters: 36,928
32x32 input -> (1, 64, 30, 30)  224x224 input -> (1, 64, 222, 222)
a dense layer for the 32x32 case: 3,774,873,600 weights
```

Weight sharing turns 3.8 billion dense weights into 36,928. CNNs build in the assumption that nearby pixels matter together, and they dominated vision for a decade; they are still common on small devices and as front ends for audio and images.

### 19.2 Recurrent networks (RNNs and LSTMs)

A **recurrent network** reads a sequence one step at a time and carries a **hidden state** forward, updating it with each input:

\[
h_t=\tanh\big(W h_{t-1}+U x_t+b\big)
\]

!!! example "On paper"

    A one-number RNN with \(W=0.5\), \(U=1\), \(b=0\), \(h_0=0\), reading the inputs \(1,0,0\):

    \[
    \begin{aligned}
    h_1 &= \tanh(1) \approx 0.762\\
    h_2 &= \tanh(0.5\cdot0.762) = \tanh(0.381) \approx 0.363\\
    h_3 &= \tanh(0.182) \approx 0.180
    \end{aligned}
    \]

    The first input's trace halves at every step. After 20 steps it is gone, and so is its gradient (§14). **LSTMs** add gates (sigmoids that decide what to keep, forget and output) and a separate cell state with a mostly additive update, which lets information survive hundreds of steps.

```python
import math
import torch
import torch.nn as nn

# The hand example: scalar RNN, h_t = tanh(w*h_{t-1} + u*x_t), w = 0.5, u = 1
h = 0.0
for t, x in enumerate([1.0, 0.0, 0.0, 0.0], start=1):
    h = math.tanh(0.5 * h + 1.0 * x)
    print(f"t={t}  h={h:.3f}")

# The same idea at size: an LSTM reads 32 steps one after another
torch.manual_seed(0)
lstm = nn.LSTM(input_size=16, hidden_size=64, batch_first=True)
out, (h_n, c_n) = lstm(torch.randn(4, 32, 16))             # [batch, time, features]
print("LSTM outputs:", tuple(out.shape), " final hidden:", tuple(h_n.shape))
print(f"LSTM parameters: {sum(p.numel() for p in lstm.parameters()):,}")
```

```text
t=1  h=0.762
t=2  h=0.363
t=3  h=0.180
t=4  h=0.090
LSTM outputs: (4, 32, 64)  final hidden: (1, 4, 64)
LSTM parameters: 20,992
```

### 19.3 Why transformers replaced them for text

| | CNN | RNN / LSTM | Transformer |
|---|---|---|---|
| Built-in assumption | nearby positions matter together | order matters, recent matters most | none; every token can look at every earlier token |
| Training parallel over positions | yes | no: step \(t\) waits for step \(t-1\) | yes |
| Steps between token 1 and token \(T\) | many layers | \(T\) recurrent steps | 1 attention layer |
| Cost per layer as \(T\) grows | linear | linear | quadratic (attention scores) |
| Memory while generating | fixed | fixed state | KV cache grows with \(T\) |

The transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) won text for two engineering reasons. Training processes all positions of a sequence at once, which uses GPU hardware fully; an RNN's sequential dependency leaves most of the chip idle. And any two tokens are one attention step apart, so long-range dependencies (a variable defined 2,000 tokens earlier) do not have to survive a chain of multiplications. The price is the quadratic attention cost and a cache that grows with the context, both covered in [49](49-ai-llm-internals.md) and [32](32-mathematics.md). Recurrence is not dead: linear-attention layers such as Gated DeltaNet ([32 §10](32-mathematics.md#10-linear-attention-and-gated-deltanet)) bring back a fixed-size state inside hybrid transformers.

---

## 20. Embeddings: learned lookup tables

An **embedding** layer maps an integer id (a token, a product, a user) to a vector of \(d\) learned numbers. It is a table with one row per id, and looking up row *i* is the same as multiplying a one-hot vector by the table:

\[
\mathbf e=E[\text{id}]=\mathbf 1_{\text{id}}^{\top}E
\]

!!! example "On paper"

    A vocabulary of 3 ids with \(d=2\), table rows \((0.1,0.2)\), \((0.3,0.4)\), \((0.5,0.6)\). Id 2 (counting from 0):

    \[
    (0,\,0,\,1)\,E = (0.5,\,0.6)
    \]

    The one-hot product picks one row, so frameworks skip the multiplication and index directly. Qwen3-14B's table is \(151{,}936\times5{,}120=777{,}912{,}320\) numbers, 1.56 GB in bf16.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
emb = nn.Embedding(num_embeddings=5, embedding_dim=3)     # vocabulary of 5, 3 numbers each
ids = torch.tensor([2, 0, 2])                              # a "sentence" of token ids
vectors = emb(ids)
print("lookup:", tuple(vectors.shape))
one_hot = nn.functional.one_hot(ids, num_classes=5).float()
print("same as one_hot @ table:", torch.allclose(vectors, one_hot @ emb.weight))

vectors.sum().backward()
print("gradient rows (only used ids get one):")
print(emb.weight.grad)
V, d = 151_936, 5_120                                       # Qwen3-14B
print(f"Qwen3-14B table: {V * d:,} parameters = {V * d * 2 / 1e9:.2f} GB in bf16")
```

```text
lookup: (3, 3)
same as one_hot @ table: True
gradient rows (only used ids get one):
tensor([[1., 1., 1.],
        [0., 0., 0.],
        [2., 2., 2.],
        [0., 0., 0.],
        [0., 0., 0.]])
Qwen3-14B table: 777,912,320 parameters = 1.56 GB in bf16
```

Only the rows that were used get a gradient (row 2 twice, because id 2 appears twice). Training moves ids that appear in similar contexts towards similar vectors; nobody designs the coordinates. That is the whole basis of semantic search with embedding models ([51](51-ai-rag.md)) and the first layer of every LLM ([49](49-ai-llm-internals.md)). Rare ids get few updates, so their vectors stay close to random: one reason models handle rare words and new product codes poorly.

---

## 21. Key takeaways

- A neuron is a weighted sum plus a nonlinearity; without the nonlinearity, depth collapses to one linear map.
- Backpropagation is the chain rule run backwards once; check a hand-written gradient with finite differences, then trust autograd.
- The training loop is always the same four lines: `zero_grad`, forward + loss, `backward`, `step`; evaluation needs `eval()` and `no_grad()`.
- AdamW with warmup and cosine decay is the default for transformers; its first step has size \(\eta\) regardless of the gradient's scale.
- The learning rate has a cliff: above it, training diverges to `nan`. Batch size, schedule and learning rate must be chosen together.
- Initialization scale, normalization and residual connections are what make deep stacks trainable; gradient clipping guards against the rest.
- bf16 has fp32's range and little precision; fp16 has more precision and overflows at 65,504.
- Full training costs about 16 bytes per parameter plus activations (236 GB for a 14B model); serving costs 2 bytes per parameter in bf16. LoRA and quantization close the gap on a laptop.
- Read loss curves against \(\ln C\), the learning-rate curve and the gradient norm; most curve shapes map to a short list of causes.
- Embeddings are trained lookup tables; transformers replaced RNNs for text because they train in parallel and connect any two tokens in one step.

---

## 22. Exercises

!!! question "Exercise 1"

    A neuron has weights \(\mathbf w=(1,\,-2)\), bias \(b=0.5\) and a sigmoid. Compute its output for \(\mathbf x=(2,\,0.75)\).

??? success "Answer"

    \(z=1\cdot2-2\cdot0.75+0.5=2-1.5+0.5=1\), and \(\sigma(1)=1/(1+e^{-1})=1/1.368\approx0.731\).

!!! question "Exercise 2"

    Take the §3 network but change the label to \(y=0\). What are the loss, \(\partial L/\partial z_2\) and \(\partial L/\partial\mathbf w_2\)?

??? success "Answer"

    The forward pass is unchanged, so \(p=0.5\) and \(L=-\ln(1-0.5)=\ln2\approx0.693\). Then \(\partial L/\partial z_2=p-y=0.5\) and \(\partial L/\partial\mathbf w_2=0.5\cdot\mathbf h=0.5\cdot(1,2)=(0.5,\,1)\): every gradient flips sign compared with §5.2, so the step now lowers \(p\).

!!! question "Exercise 3"

    In the §3 network, set \(\mathbf b_1=(0,\,-3)\). Which parameters get a zero gradient, and why?

??? success "Answer"

    \(z_{1,2}=-0.5+2-3=-1.5<0\), so \(h_2=0\) and the ReLU's slope there is 0. Row 2 of \(W_1\) and \(b_{1,2}\) get zero gradient (nothing flows back through a closed ReLU), and so does \(w_{2,2}\), because its gradient is \((p-y)\cdot h_2=0\). Unit 2 is dead for this input; if it is negative for all inputs it is dead for good.

!!! question "Exercise 4"

    Two parameters have gradients 1,000 and 0.001 on the first step. With Adam (\(\eta=10^{-3}\), \(\epsilon\approx0\)), how far does each move? With SGD at \(\eta=10^{-3}\)?

??? success "Answer"

    Adam's first step is \(\eta\cdot\hat m/\sqrt{\hat v}=\eta\cdot g/|g|\), so both move by exactly 0.001 against their gradient. SGD moves them by \(10^{-3}\cdot1{,}000=1\) and \(10^{-3}\cdot0.001=10^{-6}\): a millionfold difference. This is why Adam tolerates layers with very different gradient scales.

!!! question "Exercise 5"

    120,000 training examples, batch size 64, 2 epochs, warmup over the first 3% of steps. How many steps in total, and how many warmup steps?

??? success "Answer"

    Steps per epoch \(=120{,}000/64=1{,}875\) exactly, so \(1{,}875\cdot2=3{,}750\) steps. Warmup \(=0.03\cdot3{,}750=112.5\), so 113 steps (round up). If the schedule were stepped per epoch by mistake, warmup would never finish.

!!! question "Exercise 6"

    How much memory does full fine-tuning of a 3B model with mixed-precision AdamW need before activations? How much does serving it in bf16 need?

??? success "Answer"

    \(16\cdot3\times10^{9}=48\) GB for training, \(2\cdot3\times10^{9}=6\) GB for bf16 serving. 48 GB does not fit a 32 GB Mac, which is why the handbook trains LoRA adapters instead.

!!! question "Exercise 7"

    An activation has the value 80,000. What does it become in fp16 and in bf16?

??? success "Answer"

    fp16 overflows (max 65,504) and gives `inf`. In bf16, 80,000 lies between \(2^{16}\) and \(2^{17}\), where the step is \(2^{16-7}=512\); \(80{,}000/512=156.25\), so it rounds to \(156\cdot512=79{,}872\), an error of 128 (0.16%).

!!! question "Exercise 8"

    A gradient vector is \((6,\,8,\,0)\) and the clipping threshold is 2. What is the clipped gradient?

??? success "Answer"

    \(\lVert\mathbf g\rVert=\sqrt{36+64+0}=10>2\), so scale by \(2/10=0.2\): \((1.2,\,1.6,\,0)\). The direction is unchanged and the new norm is 2.

!!! question "Exercise 9"

    A 10-class image classifier reports a loss of 23.0 on its very first batch. Is that plausible? What would you check?

??? success "Answer"

    No. A freshly initialized 10-class classifier should start near \(\ln10\approx2.303\). Ten times that suggests huge logits (bad initialization or unnormalized inputs, e.g. pixels in 0–255), a loss summed over the batch instead of averaged, or probabilities passed where logits are expected. Print one batch's logits and labels before tuning anything.

!!! question "Exercise 10"

    Your fine-tune shows training loss falling from 1.2 to 0.05 over 5 epochs, while validation loss goes 1.1, 0.9, 0.95, 1.2, 1.6. What happened, which checkpoint do you keep, and name two fixes.

??? success "Answer"

    Overfitting: after epoch 2 the model memorises the training set. Keep the epoch-2 checkpoint (lowest validation loss, 0.9). Fixes: train fewer epochs (early stopping with a small patience), add or clean data, lower the learning rate, add regularization (dropout, weight decay) or, for LoRA, a smaller rank.

---

Next: [49 — How LLMs work, for engineers](49-ai-llm-internals.md).
