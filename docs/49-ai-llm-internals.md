# 49 — How LLMs work, for engineers

*Last verified: 28 October 2026.*

This chapter follows one request through a large language model (LLM): how text becomes tokens, how attention mixes them, how a stack of decoder blocks produces a probability for every possible next token, and how a sampler turns those probabilities into text. Then it covers what decides cost and speed (tokens, the KV cache, memory, quantization), how models are trained (pretraining, fine-tuning, preference tuning, reasoning), and the failure modes that follow from the training objective. Every concept has numbers you can redo by hand or code with its real output. It assumes [48 — Deep learning](48-ai-deep-learning.md) (neurons, losses, backprop, embeddings) and the softmax and cross-entropy of [46 — The math you need](46-ai-math.md). The full transformer mathematics, with proofs, is in [32 — Mathematics](32-mathematics.md), [37 — Architecture anatomy](37-architecture.md) and [38 — Derivations](38-derivations.md); this chapter gives you the working model and links there for depth. The running example is Qwen3-14B served locally by `mlx_lm.server` ([10](10-export-serving.md)).

---

## 1. The end-to-end picture

### 1.1 One request, end to end

An LLM is a function from a sequence of token ids to a probability distribution over the *next* token. Generation calls it in a loop. With Qwen3-14B's real sizes (\(d=5{,}120\), 40 blocks, vocabulary 151,936):

```text
"Refund my order"                              text
   |  tokenizer (section 2)
   v
[id_1, id_2, id_3, id_4]                       T token ids               [T]
   |  embedding lookup (section 3): one row of a 151,936 x 5,120 table
   v
X                                              one vector per token      [T, 5120]
   |  40 x decoder block (section 7):
   |  RMSNorm, attention (4-6), add, RMSNorm, MLP, add
   v
H                                              same shape, "in context"  [T, 5120]
   |  final RMSNorm, then the LM head (a 5,120 x 151,936 matrix)
   v
logits                                         one score per token       [T, 151936]
   |  take the LAST row, softmax, sample (section 9)
   v
next token id --> append it, run again (cheap thanks to the KV cache, section 10)
```

Everything an AI engineer tunes maps to a line of this diagram: the prompt is the token ids, `max_tokens` is the number of loop iterations, `temperature` and `top_p` act on the last arrow, context length is \(T\), and memory is the table sizes plus the cache.

### 1.2 The probability of a whole text

The model only ever predicts one token at a time, but the chain rule of probability turns those predictions into the probability of a whole sequence:

\[
p(x_1,\dots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_{<t})
\]

!!! example "On paper"

    Suppose that after the prompt the model gives "Refund" probability 0.5, then " approved" (given "Refund") 0.4, then "." (given "Refund approved") 0.9:

    \[
    p = 0.5\cdot0.4\cdot0.9 = 0.18
    \]

    The whole reply has probability 0.18. Training (§8) maximises this product over trillions of tokens, or equivalently minimises the average of \(-\ln\) of each factor. Generation (§9) builds the text one factor at a time, and never revisits an earlier choice.

### 1.3 The loop in code

The loop is the same whatever the model. Here the "model" is a bigram table (it predicts the next word from the last word only) built from three support replies, so the mechanics are visible:

```python
import numpy as np

corpus = "<s> your order shipped . <s> your refund arrived . <s> your order arrived late .".split()
vocab = sorted(set(corpus))
idx = {w: i for i, w in enumerate(vocab)}
counts = np.full((len(vocab), len(vocab)), 0.1)          # add-0.1 smoothing
for a, b in zip(corpus, corpus[1:]):
    counts[idx[a], idx[b]] += 1

def model(token_ids):
    """Stand-in for a transformer: returns logits for the NEXT token after the last one."""
    return np.log(counts[token_ids[-1]])                  # a bigram only looks at the last token

ids = [idx["<s>"]]                                         # the prompt
for _ in range(5):                                         # the generation loop
    logits = model(ids)                                    # 1. run the model
    probs = np.exp(logits) / np.exp(logits).sum()          # 2. softmax
    next_id = int(probs.argmax())                          # 3. pick (greedy)
    ids.append(next_id)                                    # 4. append and repeat
    print(f"{vocab[next_id]:<8} p={probs[next_id]:.2f}")
    if vocab[next_id] == ".":                              # 5. stop token
        break
print(" ".join(vocab[i] for i in ids))
```

```text
your     p=0.82
order    p=0.55
arrived  p=0.39
.        p=0.39
<s> your order arrived .
```

"your order arrived ." never appears in the corpus: the model stitched it from pieces it had seen. That recombination is what makes LLMs useful, and (§14.1) what makes them hallucinate. Where "order" could be followed by "shipped" or "arrived" with equal probability, greedy picked the one with the lower id; real ties are rare, but near-ties are common and matter in §9.8.

---

## 2. Tokenization

A **tokenizer** maps text to a sequence of integers from a fixed **vocabulary**, and back. The model never sees characters, only these ids, so the tokenizer decides how many steps a text costs and what the model can "see" inside a word.

### 2.1 Characters, words or subwords

| Unit | Vocabulary | Tokens for "unbelievably" | Problem |
|---|---|---|---|
| Characters (or bytes) | ~100 (256 bytes) | 12 | sequences become very long; attention cost grows with \(T^2\) |
| Words | hundreds of thousands, still incomplete | 1 | unseen words ("Kubernetesify", typos, product codes) have no id |
| Subwords (BPE, SentencePiece) | 32k–262k | 2–4 | the usual compromise: frequent words are one token, rare ones split into pieces |

Every current LLM uses subwords. Qwen3 uses **byte-level BPE** with 151,669 entries (padded to 151,936 rows in the checkpoint); Gemma 4 uses SentencePiece with 262,144 ([32 §2](32-mathematics.md#2-tokenization)). Byte-level means the base alphabet is the 256 possible bytes, so *any* string, in any language or encoding, can be represented: there is no "unknown token".

### 2.2 BPE trained by hand

**Byte-pair encoding (BPE)** builds the vocabulary bottom-up: start from single characters (or bytes), count every adjacent pair in the training corpus, merge the most frequent pair into a new token, repeat. The list of merges, in order, *is* the tokenizer.

\[
(a,b)^{*}=\arg\max_{(a,b)}\ \sum_{w}n_w\cdot\operatorname{count}_w(a,b)
\]

!!! example "On paper"

    A corpus of four words with frequencies \(n_w\): low ×5, lower ×2, newest ×6, widest ×3; `_` marks the end of a word. Count the pairs, weighted by word frequency:

    \[
    \begin{aligned}
    \text{count}(\text{e},\text{s}) &= 6+3 = 9\\
    \text{count}(\text{w},\text{e}) &= 2+6 = 8\\
    \text{count}(\text{l},\text{o}) &= 5+2 = 7
    \end{aligned}
    \]

    Three pairs tie at 9 (e s, s t, t _); ties go to the pair seen first, so merge 1 is \(\text{e}+\text{s}\rightarrow\text{es}\). Before any merge the corpus is \(5\cdot4+2\cdot6+6\cdot7+3\cdot7=95\) tokens.

Every merge of the first eight rounds, recounted after each one:

| Merge | Pair | Count | Why it wins |
|---:|---|---:|---|
| 1 | e + s → es | 9 | newest (6) + widest (3); first of three ties at 9 |
| 2 | es + t → est | 9 | the same two words |
| 3 | est + _ → est_ | 9 | "est" always ends a word here |
| 4 | l + o → lo | 7 | low (5) + lower (2) |
| 5 | lo + w → low | 7 | the same two words |
| 6 | n + e → ne | 6 | newest; ties with (e, w) and (w, est_), seen first |
| 7 | ne + w → new | 6 | newest |
| 8 | new + est_ → newest_ | 6 | newest is now a single token |

```python
from collections import Counter

# word -> how often it appears in the corpus; "_" marks the end of a word
corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
words = {tuple(w) + ("_",): n for w, n in corpus.items()}

def pair_counts(words):
    counts = Counter()
    for symbols, n in words.items():
        for pair in zip(symbols, symbols[1:]):
            counts[pair] += n
    return counts

def merge(words, pair):
    out = {}
    for symbols, n in words.items():
        s, i = [], 0
        while i < len(symbols):
            if symbols[i:i + 2] == pair:
                s.append(symbols[i] + symbols[i + 1]); i += 2
            else:
                s.append(symbols[i]); i += 1
        out[tuple(s)] = n
    return out

merges = []
for step in range(1, 9):
    counts = pair_counts(words)
    best = max(counts, key=counts.get)       # most frequent pair; ties go to the first seen
    merges.append(best)
    words = merge(words, best)
    print(f"merge {step}: {best[0]} + {best[1]} -> {best[0] + best[1]}  (count {counts[best]})")
print({" ".join(s): n for s, n in words.items()})

def encode(word):
    symbols = {tuple(word) + ("_",): 1}
    for pair in merges:                       # replay the merges in training order
        symbols = merge(symbols, pair)
    return list(next(iter(symbols)))

for w in ("lowest", "newer", "wider"):
    print(w, "->", encode(w))
```

```text
merge 1: e + s -> es  (count 9)
merge 2: es + t -> est  (count 9)
merge 3: est + _ -> est_  (count 9)
merge 4: l + o -> lo  (count 7)
merge 5: lo + w -> low  (count 7)
merge 6: n + e -> ne  (count 6)
merge 7: ne + w -> new  (count 6)
merge 8: new + est_ -> newest_  (count 6)
{'low _': 5, 'low e r _': 2, 'newest_': 6, 'w i d est_': 3}
lowest -> ['low', 'est_']
newer -> ['new', 'e', 'r', '_']
wider -> ['w', 'i', 'd', 'e', 'r', '_']
```

After 8 merges the vocabulary has the 11 starting symbols plus 8 new ones, and the corpus shrinks from 95 tokens to \(5\cdot2+2\cdot4+6\cdot1+3\cdot4=36\). "lowest" never appeared in training but encodes in two tokens, because it reuses "low" and "est_". "wider" shares no merge and falls back to single characters: rare words cost more tokens. A production tokenizer runs the same procedure for 150,000 merges over terabytes of text; byte-level BPE also pre-splits text on spaces and punctuation so merges never cross word boundaries in odd ways.

### 2.3 Counting tokens with tiktoken

`tiktoken` is a fast BPE library with public encodings. Qwen's tokenizer is different (it ships with the model; `transformers` or `mlx_lm` load it), so the exact counts below will not match Qwen's, but the pattern does:

```python
import tiktoken

texts = {
    "English": "Please refund order 4521; it arrived damaged.",
    "Romanian": "Vă rog să-mi returnați banii pentru comanda 4521; a ajuns deteriorată.",
    "Japanese": "注文4521の返金をお願いします。商品が破損していました。",
    "Python": "def total(items):\n    return sum(i.price * i.qty for i in items)\n",
    "Numbers": "3.14159265 1234567890 2026-10-28",
}
encs = [tiktoken.get_encoding(n) for n in ("cl100k_base", "o200k_base")]
print(f"{'text':<9}{'chars':>6}{'bytes':>6}{'cl100k':>8}{'o200k':>7}")
for name, t in texts.items():
    counts = [len(e.encode(t)) for e in encs]
    print(f"{name:<9}{len(t):>6}{len(t.encode('utf-8')):>6}{counts[0]:>8}{counts[1]:>7}")

enc = encs[1]
for t in ("Please refund order 4521", "1234567890", " strawberry"):
    print(repr(t), "->", [enc.decode([i]) for i in enc.encode(t)])
```

```text
text      chars bytes  cl100k  o200k
English      45    45      11     11
Romanian     70    74      24     23
Japanese     29    79      27     17
Python       65    65      17     17
Numbers      32    32      17     17
'Please refund order 4521' -> ['Please', ' refund', ' order', ' ', '452', '1']
'1234567890' -> ['123', '456', '789', '0']
' strawberry' -> [' strawberry']
```

What to read from it:

- **English prose** runs at about 4 characters per token (45 characters, 11 tokens).
- **The same request in Romanian costs about twice as many tokens** (23 against 11), and Japanese depends heavily on the vocabulary (27 versus 17 between two encodings). Non-English users pay more per word and fit less into the same context.
- **Numbers split into chunks** of up to three digits, and " 4521" becomes a space, "452" and "1". The model never sees the number 4521 as a unit, which is one reason arithmetic on long numbers is unreliable without a tool. (Gemma 4's tokenizer splits every digit separately, [32 §2](32-mathematics.md#2-tokenization); the effect on arithmetic is different, the lesson is the same.)
- **" strawberry" is one token.** The model gets one id, not ten letters, so "how many r's are in strawberry" asks it about spelling it never directly observed.
- Leading spaces are part of tokens: " refund" and "refund" are different ids.

### 2.4 Why token counts drive cost and latency

API prices are per token, local generation time is per token, and the context window is measured in tokens. With prices given as dollars per million tokens:

\[
\text{cost}=\frac{T_{\text{in}}}{10^{6}}\,p_{\text{in}}+\frac{T_{\text{out}}}{10^{6}}\,p_{\text{out}}
\]

!!! example "On paper"

    A support bot handles 100,000 requests a day, each with 2,000 input tokens (system prompt, retrieved documents, history) and 300 output tokens. **Assume** $3 per million input tokens and $15 per million output tokens:

    \[
    \begin{aligned}
    \text{input} &= 200\cdot\$3 = \$600\ \text{per day}\\
    \text{output} &= 30\cdot\$15 = \$450\ \text{per day}\\
    \text{total} &= \$1{,}050\ \text{per day}
    \end{aligned}
    \]

    Trimming the system prompt by 500 tokens saves \(50\cdot\$3=\$150\) a day. Serving the same traffic in Romanian, at 23/11 ≈ 2.1 times the tokens, would cost roughly $2,200.

Locally there is no invoice, but the same tokens cost time: every input token must be prefilled and every output token decoded (§10.3). Prompt design, caching and cost control are in [50 — Prompting and APIs](50-ai-prompting-apis.md) and [55 — Production](55-ai-production.md).

### 2.5 Tokenization pitfalls

| Pitfall | What happens | What to do |
|---|---|---|
| Counting words or characters for budgets | context overflows or truncates silently | count with the model's own tokenizer |
| Assuming one model's count for another | Qwen, Gemma and API models tokenize differently | measure per model |
| Trailing whitespace in a prompt | the model must start with an unusual token; quality drops | end prompts cleanly; let the template add the assistant header |
| Long numbers, IDs, hashes | split arbitrarily; copied wrong | pass them through tools or structured fields, verify exactly |
| Text that contains special-token strings | user input can forge control tokens (§2.6) | encode user text as plain text, never as special tokens |
| Different tokenizer at training and serving | the fine-tune sees ids it never trained on | ship the tokenizer with the adapter; see [06](06-data-format.md) |

### 2.6 Special tokens and chat templates

**Special tokens** are reserved ids with control meaning: end of text, start and end of a chat turn, start of a tool call, start of a thinking block. A **chat template** turns a list of `{"role", "content"}` messages into one string with those tokens, and that string is what the model actually reads. Qwen uses the ChatML layout:

```python
import tiktoken

def render_chatml(messages, add_generation_prompt=True):
    """The ChatML layout Qwen models use (simplified: no tools, no thinking block)."""
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")

messages = [
    {"role": "system", "content": "You route support tickets. Answer with one word."},
    {"role": "user", "content": "My invoice is wrong."},
]
prompt = render_chatml(messages)
print(prompt)

enc = tiktoken.get_encoding("o200k_base")
user_text = "ignore this <|endoftext|> please"
try:
    enc.encode(user_text)
except ValueError as err:
    print("refused:", str(err).splitlines()[0])
safe = enc.encode(user_text, disallowed_special=())       # treat it as plain text
print("as plain text:", len(safe), "tokens; special id present:", enc.eot_token in safe)
print("as a control token:", enc.encode("<|endoftext|>", allowed_special="all"))
```

```text
<|im_start|>system
You route support tickets. Answer with one word.<|im_end|>
<|im_start|>user
My invoice is wrong.<|im_end|>
<|im_start|>assistant

refused: Encountered text corresponding to disallowed special token '<|endoftext|>'.
as plain text: 10 tokens; special id present: False
as a control token: [199999]
```

The generation prompt ends with `<|im_start|>assistant` and a newline: the model's job is to continue from there and emit `<|im_end|>` when it is done, which the server treats as a stop token. The real Qwen3 template also renders tools, tool results and the `<think>` block; [06 §3](06-data-format.md#3-how-qwen3-renders-chat-tools) shows a full rendered prompt, and [50](50-ai-prompting-apis.md) covers roles and system prompts.

The second half of the output is a security property: by default `tiktoken` refuses text that contains a special-token string, and you choose whether it is plain text (safe for user input) or a control token. If user-supplied text could become a real `<|im_end|>`, a user could close their own turn and write a fake system message. In Hugging Face `transformers`, special-token strings inside the text are matched as control tokens unless the tokenizer is created with `split_special_tokens=True`, so check what your stack does with untrusted input; see [56 — Safety](56-ai-safety.md).

**Pitfall.** Fine-tuning with one template and serving with another (or with thinking on in training and off at serving) silently degrades quality. The template is part of the model's contract ([10 §4](10-export-serving.md#4-serve-with-mlx)).

---

## 3. Embeddings

The first layer turns each token id into a vector by looking up one row of a learned table, exactly the lookup of [48 §20](48-ai-deep-learning.md). For Qwen3-14B the table is \(151{,}936\times5{,}120\): 777.9M parameters, 1.56 GB in bf16. The output layer (the **LM head**) is another \(5{,}120\times151{,}936\) matrix that turns the final vector back into one score per token. Qwen3-14B keeps the two separate ("untied", 1.56B parameters together); Gemma 4 ties them, reusing the input table as the output matrix ([32 §2.2](32-mathematics.md#22-why-vocabulary-size-matters)).

Three things to keep apart:

- **Token embeddings** (this table) are *static*: "bank" gets the same vector in "river bank" and "bank transfer". They are only the starting point.
- **Hidden states** after the decoder blocks are *contextual*: by layer 40 the vector at "bank" has absorbed its neighbours through attention, so the two banks differ.
- **Embedding models** used for search ([51 — RAG](51-ai-rag.md)) are separate models trained to produce one vector per *text*, such that similar meanings have high cosine similarity. You do not get good search vectors by averaging an LLM's input table.

Rows for tokens that were rare in training are barely trained (only used rows get gradients, [48 §20](48-ai-deep-learning.md)), and padded rows (151,669 to 151,936) are never used at all.

---

## 4. Self-attention by hand

### 4.1 Queries, keys and values

**Self-attention** lets every token gather information from the other tokens before it. Each token's vector is projected three ways:

- a **query** \(\mathbf q\): what this token is looking for;
- a **key** \(\mathbf k\): what this token offers to others looking;
- a **value** \(\mathbf v\): what this token hands over if it is chosen.

Each query is compared with every key (a dot product), the scores become weights with a softmax, and the output is the weighted average of the values. **Analogy:** a fuzzy database lookup. A normal index returns the one row whose key equals the query; attention returns a blend of *all* rows, weighted by how well each key matches.

### 4.2 The formula

Stack the vectors of all \(T\) tokens as the rows of \(Q\), \(K\) and \(V\); \(d_k\) is the key width and \(M\) is the causal mask (§4.4), 0 where a token may look and \(-\infty\) where it may not:

\[
\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\Big(\frac{QK^{\top}}{\sqrt{d_k}}+M\Big)\,V
\]

!!! example "On paper"

    The third token "sat" in the example below has scaled scores \((1,1,2)\) against "the", "cat" and itself, and the values are \((1,0)\), \((0,1)\), \((1,1)\). With \(e\approx2.718\) and \(e^2\approx7.389\):

    \[
    \begin{aligned}
    \text{sum} &= 2.718+2.718+7.389 = 12.825\\
    \text{weights} &= (0.212,\ 0.212,\ 0.576)\\
    \text{output} &= 0.212\,(1,0)+0.212\,(0,1)\\
    &\quad+0.576\,(1,1)\\
    &= (0.788,\ 0.788)
    \end{aligned}
    \]

    "sat" keeps 58% of its own value and takes 21% from each earlier word. The output is a new vector for "sat" that now carries information about "the" and "cat".

### 4.3 Three tokens, every step

Take "the cat sat" with \(d_{\text{model}}=4\) and tiny 0/1 vectors so every product is mental arithmetic. In a real model the projections are learned \(5{,}120\times128\) matrices; here \(W_K\) is the identity (keys equal the inputs), \(W_Q\) swaps the two halves of each pair of coordinates (so a token looks for something other than itself), and \(W_V\) keeps the first two coordinates.

**Step 1: project.**

| Token | \(\mathbf x\) | \(\mathbf q=\mathbf xW_Q\) | \(\mathbf k=\mathbf xW_K\) | \(\mathbf v=\mathbf xW_V\) |
|---|---|---|---|---|
| the | (1, 0, 1, 0) | (0, 1, 0, 1) | (1, 0, 1, 0) | (1, 0) |
| cat | (0, 1, 0, 1) | (1, 0, 1, 0) | (0, 1, 0, 1) | (0, 1) |
| sat | (1, 1, 1, 1) | (1, 1, 1, 1) | (1, 1, 1, 1) | (1, 1) |

**Step 2: scores \(QK^\top\), then divide by \(\sqrt{d_k}=\sqrt4=2\).** Row = the query's token, column = the key's token.

| | the | cat | sat |
|---|---:|---:|---:|
| the | 0 → 0 | 2 → 1 | 2 → 1 |
| cat | 2 → 1 | 0 → 0 | 2 → 1 |
| sat | 2 → 1 | 2 → 1 | 4 → 2 |

**Step 3: causal mask.** A token may not look at later tokens, so everything above the diagonal becomes \(-\infty\): "the" sees only itself, "cat" sees the and cat, "sat" sees all three.

**Step 4: softmax per row**, using \(e^{-\infty}=0\):

| | the | cat | sat |
|---|---:|---:|---:|
| the | 1 | 0 | 0 |
| cat | \(e/(e+1)=0.731\) | \(1/(e+1)=0.269\) | 0 |
| sat | \(1/(2+e)=0.212\) | 0.212 | \(e/(2+e)=0.576\) |

**Step 5: weighted sum of values.** "the" → (1, 0); "cat" → \(0.731\,(1,0)+0.269\,(0,1)=(0.731,\,0.269)\); "sat" → (0.788, 0.788) as in the card above.

Two observations. Without the mask, "the" would have put 42% of its weight on each later word (scores 0, 1, 1), so the mask really changes the answer. And "cat" attends mostly to "the", not to itself, because \(W_Q\) made it look for a different pattern than it offers: that asymmetry is what learned projections buy.

```python
import numpy as np

X = np.array([[1, 0, 1, 0],          # "the"
              [0, 1, 0, 1],          # "cat"
              [1, 1, 1, 1]], float)  # "sat"
W_Q = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], float)  # swap pairs
W_K = np.eye(4)
W_V = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], float)

Q, K, V = X @ W_Q, X @ W_K, X @ W_V
scores = Q @ K.T / np.sqrt(K.shape[1])               # divide by sqrt(d_k) = 2
mask = np.triu(np.ones((3, 3), bool), k=1)           # True above the diagonal = future
scores[mask] = -np.inf
weights = np.exp(scores - scores.max(axis=1, keepdims=True))
weights /= weights.sum(axis=1, keepdims=True)        # softmax, row by row
out = weights @ V
print("scaled, masked scores:\n", scores)
print("attention weights:\n", weights.round(3))
print("output:\n", out.round(3))
```

```text
scaled, masked scores:
 [[  0. -inf -inf]
 [  1.   0. -inf]
 [  1.   1.   2.]]
attention weights:
 [[1.    0.    0.   ]
 [0.731 0.269 0.   ]
 [0.212 0.212 0.576]]
output:
 [[1.    0.   ]
 [0.731 0.269]
 [0.788 0.788]]
```

The same function with random weights matches PyTorch's built-in kernel, which is what real models call (fused, and on GPUs a FlashAttention-style implementation that never stores the full score matrix):

```python
import numpy as np
import torch
import torch.nn.functional as F

rng = np.random.default_rng(0)
T, d, d_k = 5, 16, 8                                  # 5 tokens, model width 16, head width 8
X = rng.normal(size=(T, d))
W_Q, W_K, W_V = (rng.normal(size=(d, d_k)) / np.sqrt(d) for _ in range(3))

def attention(X):
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    s = Q @ K.T / np.sqrt(d_k)
    s[np.triu(np.ones((T, T), bool), k=1)] = -np.inf
    w = np.exp(s - s.max(1, keepdims=True))
    return (w / w.sum(1, keepdims=True)) @ V

ours = attention(X)
t = lambda a: torch.tensor(a, dtype=torch.float64)
ref = F.scaled_dot_product_attention(t(X @ W_Q), t(X @ W_K), t(X @ W_V), is_causal=True)
print("matches torch:", np.allclose(ours, ref.numpy()))
print("output shape:", ours.shape)
```

```text
matches torch: True
output shape: (5, 8)
```

### 4.4 The causal mask

The mask is what makes a **decoder**. During training the model sees a whole sequence at once and predicts every next token in parallel; without the mask, position 3 could simply read token 4 and the task would be trivial. With it, every position is a fair "predict the next token from the past only" exercise, so one sequence of \(T\) tokens gives \(T-1\) training examples in a single forward pass. At generation time the mask costs nothing extra: the future does not exist yet.

### 4.5 Why divide by the square root of \(d_k\)

If the components of \(\mathbf q\) and \(\mathbf k\) are independent with variance 1, their dot product sums \(d_k\) such terms, so its variance is \(d_k\):

\[
\operatorname{Var}(\mathbf q\cdot\mathbf k)=\sum_{i=1}^{d_k}\operatorname{Var}(q_ik_i)=d_k
\]

!!! example "On paper"

    Qwen3-14B's heads have \(d_k=128\), so raw scores have a standard deviation of \(\sqrt{128}\approx11.3\). Two keys scoring 2 and 1 standard deviations, 22.6 and 11.3:

    \[
    \begin{aligned}
    \text{unscaled: } \frac{1}{1+e^{-11.3}} &\approx 0.99999\\
    \text{scaled to } (2,\,1):\ \frac{e}{e+1} &\approx 0.731
    \end{aligned}
    \]

    Unscaled, the softmax is almost a hard argmax: one key takes everything and the gradients through the others vanish. Dividing by \(\sqrt{d_k}\) brings the scores back to unit scale.

```python
import numpy as np

rng = np.random.default_rng(0)
d_k, T = 128, 64                                 # Qwen3-14B head width, 64 keys
q = rng.normal(size=(1000, d_k))                 # 1000 random queries
K = rng.normal(size=(T, d_k))
raw = q @ K.T
print(f"std of raw dot products: {raw.std():.1f}  (sqrt(128) = {np.sqrt(d_k):.1f})")

def softmax(s):
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

for name, s in [("unscaled", raw), ("scaled by 1/sqrt(d_k)", raw / np.sqrt(d_k))]:
    top = softmax(s).max(axis=1).mean()
    print(f"{name:<22} average weight on the top key: {top:.3f}")
```

```text
std of raw dot products: 11.3  (sqrt(128) = 11.3)
unscaled               average weight on the top key: 0.858
scaled by 1/sqrt(d_k)  average weight on the top key: 0.108
```

[38 §2.3](38-derivations.md#23-why-scale-by-the-square-root-of-the-head-dimension) has the proof; Qwen3 and Gemma 4 additionally normalise \(\mathbf q\) and \(\mathbf k\) (QK-norm) to keep these scores in range.

### 4.6 The cost grows with the square of the context

Every query is scored against every earlier key, so each head of each layer computes about \(T^2\) scores:

\[
\text{scores per layer}=H\cdot T^{2}
\]

!!! example "On paper"

    Qwen3-14B has \(H=40\) query heads. At \(T=2{,}048=2^{11}\) and at \(T=32{,}768=2^{15}\), in bf16 (2 bytes each) if they were all stored:

    \[
    \begin{aligned}
    40\cdot2^{22}\cdot2\ \text{B} &\approx 336\ \text{MB per layer}\\
    40\cdot2^{30}\cdot2\ \text{B} &\approx 85.9\ \text{GB per layer}
    \end{aligned}
    \]

    16 times the context costs 256 times the scores. That is why long-context training and serving use fused kernels that compute the softmax block by block without ever storing the full matrix ([37 §1.4](37-architecture.md#14-activation-memory)).

---

## 5. Multi-head and grouped-query attention

### 5.1 Several heads

One attention pattern per layer would be limiting: a token may need its subject, the previous line of code and the matching bracket at the same time. **Multi-head attention** splits the width into \(H\) independent heads, each with its own smaller \(W_Q,W_K,W_V\), runs attention in each, concatenates the results and mixes them with an output matrix \(W_O\). Qwen3-14B splits \(d=5{,}120\) into 40 heads of 128 ([32 §6.1](32-mathematics.md#61-multi-head-attention) has the formula and a card). Interpretability research on trained models finds heads with recognisable jobs, such as attending to the previous token or copying a pattern seen earlier in the context, though most heads are not that clean.

### 5.2 Grouped-query attention (GQA)

During generation, the keys and values of past tokens are cached (§10.2), and that cache is the main memory cost of long contexts. **Grouped-query attention** keeps all the query heads but shares each key/value head among several of them. Qwen3-14B has 40 query heads and only 8 KV heads, so each KV head serves 5 query heads ([32 §6.2](32-mathematics.md#62-grouped-query-attention-gqa)).

\[
\text{KV bytes per token}=2\cdot L\cdot G\cdot d_h\cdot b
\]

!!! example "On paper"

    Qwen3-14B: \(L=40\) layers, \(G=8\) KV heads, \(d_h=128\), bf16 (\(b=2\)); the 2 in front is for K and V:

    \[
    \begin{aligned}
    2\cdot40\cdot8\cdot128\cdot2 &= 163{,}840\ \text{B}\\
    &= 160\ \text{KiB}\\
    \text{with } G=40:\ 819{,}200\ \text{B} &= 800\ \text{KiB}
    \end{aligned}
    \]

    GQA cuts the cache to a fifth for a small quality cost. Every token of context costs 160 KiB of memory for as long as the conversation lives.

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, T, d, H, G, d_h = 1, 6, 32, 8, 2, 4          # 8 query heads share 2 KV heads
x = torch.randn(B, T, d)
W_q = torch.randn(d, H * d_h) / d ** 0.5
W_k = torch.randn(d, G * d_h) / d ** 0.5         # K and V are 4x narrower than Q
W_v = torch.randn(d, G * d_h) / d ** 0.5

q = (x @ W_q).view(B, T, H, d_h).transpose(1, 2)    # [B, H, T, d_h]
k = (x @ W_k).view(B, T, G, d_h).transpose(1, 2)    # [B, G, T, d_h]  <- this is what gets cached
v = (x @ W_v).view(B, T, G, d_h).transpose(1, 2)
print("q", tuple(q.shape), " k", tuple(k.shape), " v", tuple(v.shape))

k_rep = k.repeat_interleave(H // G, dim=1)           # each KV head serves H/G = 4 query heads
v_rep = v.repeat_interleave(H // G, dim=1)
out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
print("output", tuple(out.shape), " same as built-in GQA:", torch.allclose(out, ref, atol=1e-6))

L, d_h14, b = 40, 128, 2                             # Qwen3-14B, bf16
for name, kv_heads in [("MHA, 40 KV heads", 40), ("GQA, 8 KV heads ", 8)]:
    print(f"{name}: {2 * L * kv_heads * d_h14 * b:,} bytes of cache per token")
```

```text
q (1, 8, 6, 4)  k (1, 2, 6, 4)  v (1, 2, 6, 4)
output (1, 8, 6, 4)  same as built-in GQA: True
MHA, 40 KV heads: 819,200 bytes of cache per token
GQA, 8 KV heads : 163,840 bytes of cache per token
```

The output keeps all 8 query heads; only K and V are narrow. Reading `num_key_value_heads` in a config tells you the cache size before you download anything (§17.2).

---

## 6. Positional information

### 6.1 Why attention needs positions

Attention compares vectors by content only. Shuffle the input tokens and the outputs are shuffled the same way, with identical values: "dog bites man" and "man bites dog" would look the same to every head.

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(3, 4))                          # "dog bites man", one row per token
W_Q, W_K, W_V = (rng.normal(size=(4, 4)) for _ in range(3))

def attend(X):                                       # no mask, no positions
    s = (X @ W_Q) @ (X @ W_K).T / 2
    w = np.exp(s - s.max(1, keepdims=True))
    return (w / w.sum(1, keepdims=True)) @ (X @ W_V)

perm = [2, 1, 0]                                     # "man bites dog"
print("same outputs, just reordered:", np.allclose(attend(X)[perm], attend(X[perm])))

def rope(v, pos, omega=np.pi / 6):                   # rotate a 2-d pair by pos * 30 degrees
    c, s = np.cos(pos * omega), np.sin(pos * omega)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

for q, k in [((1.0, 0.0), (1.0, 0.0)), ((1.0, 2.0), (3.0, -1.0))]:
    q, k = np.array(q), np.array(k)
    dots = [f"({m},{n}) {rope(q, m) @ rope(k, n):+.3f}" for m, n in [(3, 1), (5, 3), (105, 103), (5, 1)]]
    print(f"q={q.tolist()} k={k.tolist()}:", "  ".join(dots))
```

```text
same outputs, just reordered: True
q=[1.0, 0.0] k=[1.0, 0.0]: (3,1) +0.500  (5,3) +0.500  (105,103) +0.500  (5,1) -0.500
q=[1.0, 2.0] k=[3.0, -1.0]: (3,1) -5.562  (5,3) -5.562  (105,103) -5.562  (5,1) -6.562
```

The causal mask breaks the symmetry a little (a token can tell how many tokens precede it), but models need explicit position information to know *how far apart* two tokens are.

### 6.2 RoPE in one paragraph

**Rotary position embedding (RoPE)**, used by Qwen and Gemma, rotates each pair of coordinates of \(\mathbf q\) and \(\mathbf k\) by an angle proportional to the token's position, with a different speed \(\omega\) for each pair. Because rotating both vectors only changes the angle *between* them by the difference of their positions, the attention score depends on the content and on the relative offset \(m-n\), not on the absolute positions. Nothing is added to the embeddings and there are no extra parameters. The slow-rotating pairs carry long-range position, the fast ones short-range; the base frequency (`rope_theta`, \(10^6\) for Qwen3) and tricks such as YaRN set how far the scheme stretches. [32 §7](32-mathematics.md#7-rotary-position-embeddings-rope) gives the full formula and [38 §4](38-derivations.md#4-rope-is-relative-proof) proves the relative property.

\[
\big(R_{m\omega}\,\mathbf q\big)\cdot\big(R_{n\omega}\,\mathbf k\big)=\mathbf q\cdot R_{(n-m)\omega}\,\mathbf k
\]

!!! example "On paper"

    \(\mathbf q=\mathbf k=(1,0)\), \(\omega=30^\circ\). Query at position 3 and key at 1: \(R_{90^\circ}\mathbf q=(0,1)\) and \(R_{30^\circ}\mathbf k=(0.866,\,0.5)\). Query at 5 and key at 3: \(R_{150^\circ}\mathbf q=(-0.866,\,0.5)\) and \(R_{90^\circ}\mathbf k=(0,1)\):

    \[
    \begin{aligned}
    (0,1)\cdot(0.866,\,0.5) &= 0.5\\
    (-0.866,\,0.5)\cdot(0,1) &= 0.5 = \cos60^\circ
    \end{aligned}
    \]

    Both pairs are 2 positions apart and score the same, as do positions 105 and 103 in the code above. At offset 4 the score is \(\cos120^\circ=-0.5\).

---

## 7. The decoder block

A **decoder block** is attention followed by an MLP, each wrapped in a normalization and a residual connection ([48 §12](48-ai-deep-learning.md), [48 §14](48-ai-deep-learning.md)):

1. `h = RMSNorm(x)`; `x = x + Attention(h)` — tokens exchange information.
2. `h = RMSNorm(x)`; `x = x + MLP(h)` — each token is processed on its own, by the widest matrices in the model.

The residual stream \(x\) keeps the shape `[batch, T, d]` all the way through, so blocks stack like middleware with the same interface. [37 §1](37-architecture.md#1-one-block-one-tensor) runs one token through a block with numbers. The block below is the Qwen3 layout at toy size (Qwen3 also normalises \(\mathbf q\) and \(\mathbf k\) per head and applies RoPE to them; both are left out to keep it short):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """Pre-norm block in the Qwen3 style: RMSNorm, GQA attention, SwiGLU MLP, two residuals."""

    def __init__(self, d=8, H=4, G=2, d_h=2, d_ff=16):
        super().__init__()
        self.H, self.G, self.d_h = H, G, d_h
        self.norm1, self.norm2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.q = nn.Linear(d, H * d_h, bias=False)
        self.k = nn.Linear(d, G * d_h, bias=False)
        self.v = nn.Linear(d, G * d_h, bias=False)
        self.o = nn.Linear(H * d_h, d, bias=False)
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):                                   # x: [B, T, d]
        B, T, _ = x.shape
        h = self.norm1(x)
        q = self.q(h).view(B, T, self.H, self.d_h).transpose(1, 2)
        k = self.k(h).view(B, T, self.G, self.d_h).transpose(1, 2)
        v = self.v(h).view(B, T, self.G, self.d_h).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, -1))  # residual 1
        h = self.norm2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))   # residual 2 (SwiGLU)


torch.manual_seed(0)
block = DecoderBlock()
x = torch.randn(2, 5, 8)                                     # 2 sequences, 5 tokens, d = 8
print("in", tuple(x.shape), "-> out", tuple(block(x).shape))
for name, p in block.named_parameters():
    print(f"  {name:<14}{str(tuple(p.shape)):<10}{p.numel():>5}")
print("total", sum(p.numel() for p in block.parameters()))
```

```text
in (2, 5, 8) -> out (2, 5, 8)
  norm1.weight  (8,)          8
  norm2.weight  (8,)          8
  q.weight      (8, 8)       64
  k.weight      (4, 8)       32
  v.weight      (4, 8)       32
  o.weight      (8, 8)       64
  gate.weight   (16, 8)     128
  up.weight     (16, 8)     128
  down.weight   (8, 16)     128
total 592
```

The count matches the toy decoder of [32 §13](32-mathematics.md#13-parameter-counting): 192 attention weights, 384 MLP weights and 16 norm gains. The MLP holds two thirds of the block; in Qwen3-14B it is 81% (267.4M of 330.3M per layer, [37 §1.3](37-architecture.md#13-parameter-budget)). A full model is an embedding table, 40 of these blocks, a final RMSNorm and the LM head. Attention moves information *between* tokens; interpretability research suggests that the MLPs, which hold most of the parameters, store much of the factual knowledge.

---

## 8. The language-model objective

### 8.1 Next-token prediction

Pretraining has one task: given the tokens so far, predict the next one. The labels are the input shifted by one position, so any text is training data and no human labelling is needed. A 2,048-token sequence gives 2,047 predictions per forward pass, thanks to the causal mask (§4.4).

### 8.2 The loss is the average negative log-likelihood

The loss is the cross-entropy of [48 §4.3](48-ai-deep-learning.md), averaged over positions: minus the log of the probability the model gave to the token that actually came next.

\[
\mathcal L=-\frac1T\sum_{t=1}^{T}\ln p_\theta(x_t\mid x_{<t})
\]

!!! example "On paper"

    A three-token continuation where the model gave the true tokens probabilities 0.5, 0.25 and 0.125. With \(\ln2\approx0.693\):

    \[
    \begin{aligned}
    \mathcal L &= \tfrac13(\ln2+\ln4+\ln8)\\
    &= \tfrac13(1+2+3)\ln2 = 2\ln2 \approx 1.386
    \end{aligned}
    \]

    Each halving of the probability adds 0.693 to that token's loss. A model guessing uniformly over Qwen3's 151,936 tokens would score \(\ln151{,}936\approx11.93\) (the starting loss of [48 §18.1](48-ai-deep-learning.md)).

### 8.3 Perplexity

**Perplexity** is the exponential of the loss: the number of equally likely choices the model is, on average, as uncertain as.

\[
\text{PPL}=e^{\mathcal L}
\]

!!! example "On paper"

    For the loss of §8.2:

    \[
    e^{2\ln2} = 2^2 = 4
    \]

    The model is as unsure as a fair four-way choice at each step. A uniform guess over the vocabulary has perplexity \(V=151{,}936\); a perplexity of 1 means certainty. Perplexities are only comparable between models with the same tokenizer, because a model with longer tokens makes fewer, harder predictions.

### 8.4 In code

The bigram model from §1.3, scored on three sentences, and the same number from the function every LLM trainer calls:

```python
import numpy as np
import torch
import torch.nn.functional as F

corpus = "<s> your order shipped . <s> your refund arrived . <s> your order arrived late .".split()
vocab = sorted(set(corpus))
idx = {w: i for i, w in enumerate(vocab)}
counts = np.full((len(vocab), len(vocab)), 0.1)
for a, b in zip(corpus, corpus[1:]):
    counts[idx[a], idx[b]] += 1
logprob_table = np.log(counts / counts.sum(axis=1, keepdims=True))

for text in ("<s> your order shipped .", "<s> your refund shipped .", "<s> order your . shipped"):
    ids = [idx[w] for w in text.split()]
    inputs, targets = ids[:-1], ids[1:]                   # shift by one: predict the next token
    nll = [-logprob_table[a, b] for a, b in zip(inputs, targets)]
    loss = float(np.mean(nll))
    print(f"{text:<28} per-token NLL {np.round(nll, 2).tolist()}  loss {loss:.3f}  ppl {np.exp(loss):.2f}")

# the same loss through the function every LLM trainer calls
ids = torch.tensor([idx[w] for w in "<s> your order shipped .".split()])
logits = torch.tensor(np.log(counts))[ids[:-1]]           # one row of logits per input position
print("F.cross_entropy:", round(F.cross_entropy(logits, ids[1:]).item(), 3))
```

```text
<s> your order shipped .     per-token NLL [0.2, 0.59, 0.93, 0.49]  loss 0.556  ppl 1.74
<s> your refund shipped .    per-token NLL [0.2, 1.24, 2.89, 0.49]  loss 1.207  ppl 3.34
<s> order your . shipped     per-token NLL [3.64, 3.33, 3.64, 3.33]  loss 3.485  ppl 32.62
F.cross_entropy: 0.556
```

A sentence the model saw scores perplexity 1.74; a plausible new one ("refund shipped") 3.34; word salad 32.6, worse than the uniform guess over 8 words. That is how an LLM's loss separates fluent from broken text. Note what loss does *not* measure: "your refund shipped" is fluent and false. Low perplexity is fluency, not truth (§14.1).

### 8.5 Loss masking for chat data

When fine-tuning on conversations you usually compute the loss only on the assistant's tokens, not on the system prompt and user turns: the model should learn to answer, not to write user messages. The trainer sets the label of every masked position to an ignore value (`-100` in PyTorch's `cross_entropy`). [06 §8](06-data-format.md#8-mask-prompt-truncation-and-trajectory-splitting) covers `--mask-prompt` in MLX and [32 §3.1](32-mathematics.md#31-masked-completion-loss) the formula.

---

## 9. Generation: from logits to text

### 9.1 Greedy decoding

**Greedy decoding** always picks the highest-probability token (`temperature=0`). It is deterministic in principle, good for extraction and classification, and prone to loops in long free-form text: once a phrase becomes likely it tends to stay likely. Qwen's model card explicitly warns against greedy decoding in thinking mode because it can cause endless repetition.

### 9.2 Temperature

**Temperature** \(T\) divides the logits before the softmax. \(T<1\) sharpens the distribution towards the top token; \(T>1\) flattens it.

\[
p_i(T)=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
\]

!!! example "On paper"

    After "Your refund was", five candidates with probabilities \((0.5,\,0.2,\,0.15,\,0.1,\,0.05)\) at \(T=1\). Since \(e^{z_i/T}\propto p_i^{1/T}\), \(T=0.5\) squares the probabilities and renormalises:

    \[
    \begin{aligned}
    \textstyle\sum p_i^2 &= 0.25+0.04+0.0225\\
    &\quad+0.01+0.0025 = 0.325\\
    p(T{=}0.5) &= (0.769,\,0.123,\,0.069,\,0.031,\,0.008)
    \end{aligned}
    \]

    \(T=2\) takes square roots instead (sum 2.081), giving \((0.340,\,0.215,\,0.186,\,0.152,\,0.107)\). The order never changes; only the gap between the candidates does.

```python
import numpy as np

tokens = [" processed", " approved", " denied", " sent", " delayed"]
logits = np.log([0.5, 0.2, 0.15, 0.1, 0.05])        # the model's next-token logits

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

for T in (0.5, 1.0, 2.0):
    p = softmax(logits / T)
    print(f"T={T}: " + "  ".join(f"{t.strip()} {v:.3f}" for t, v in zip(tokens, p)))
```

```text
T=0.5: processed 0.769  approved 0.123  denied 0.069  sent 0.031  delayed 0.008
T=1.0: processed 0.500  approved 0.200  denied 0.150  sent 0.100  delayed 0.050
T=2.0: processed 0.340  approved 0.215  denied 0.186  sent 0.152  delayed 0.107
```

### 9.3 Top-k, top-p and min-p

These filters cut the unlikely tail before sampling, then renormalise what is left. **Top-k** keeps the \(k\) most likely tokens. **Top-p** (nucleus sampling) keeps the smallest set of top tokens whose probabilities add up to at least \(p\):

\[
\mathcal V_p=\text{smallest } k \text{ such that } \sum_{i\le k}p_{(i)}\ge p
\]

!!! example "On paper"

    Sorted probabilities \((0.5,\,0.2,\,0.15,\,0.1,\,0.05)\), \(p=0.8\). The running totals are 0.5, 0.7, 0.85: the third token pushes the sum past 0.8, so three survive.

    \[
    \begin{aligned}
    \text{kept mass} &= 0.5+0.2+0.15 = 0.85\\
    \text{renormalised} &= (0.588,\ 0.235,\ 0.176)
    \end{aligned}
    \]

    "sent" and "delayed" can no longer be sampled. Top-\(k\) with \(k=2\) would keep \((0.5,0.2)\) and renormalise to \((0.714,\,0.286)\).

**Min-p** keeps every token whose probability is at least a fraction of the top token's:

\[
\text{keep } i\iff p_i\ge p_{\min}\cdot\max_j p_j
\]

!!! example "On paper"

    \(p_{\min}=0.25\) on the same distribution: the threshold is \(0.25\cdot0.5\), so 0.5, 0.2 and 0.15 survive. On a confident distribution \((0.9,\,0.05,\,0.03,\,0.02)\) with \(p_{\min}=0.1\):

    \[
    \text{threshold} = 0.1\cdot0.9 = 0.09
    \]

    Only the top token survives. Min-p adapts: it keeps many options when the model is unsure and few when it is sure, which top-\(k\) cannot do.

```python
import numpy as np

tokens = np.array([" processed", " approved", " denied", " sent", " delayed"])
probs = np.array([0.5, 0.2, 0.15, 0.1, 0.05])        # already sorted, highest first

def show(name, keep):
    p = np.where(keep, probs, 0.0)
    p = p / p.sum()                                   # renormalise what survives
    print(f"{name:<12}" + "  ".join(f"{t.strip()} {v:.3f}" for t, v in zip(tokens[keep], p[keep])))

show("top-k k=2", np.arange(len(probs)) < 2)
cum = np.cumsum(probs)                                # 0.5, 0.7, 0.85, 0.95, 1.0
show("top-p 0.8", cum - probs < 0.8)                  # keep until the total reaches 0.8
show("min-p 0.25", probs >= 0.25 * probs.max())       # keep if >= 25% of the best
```

```text
top-k k=2   processed 0.714  approved 0.286
top-p 0.8   processed 0.588  approved 0.235  denied 0.176
min-p 0.25  processed 0.588  approved 0.235  denied 0.176
```

### 9.4 Repetition, frequency and presence penalties

Penalties lower the logits of tokens that already appeared, to discourage loops. The OpenAI-style pair subtracts a fixed amount per occurrence (**frequency**) and once for having appeared at all (**presence**), where \(c_j\) is how often token \(j\) has appeared (OpenAI counts the tokens generated so far; `mlx_lm` counts the last 20 tokens of prompt plus output, §9.9):

\[
z_j\leftarrow z_j-\alpha_{\text{freq}}\,c_j-\alpha_{\text{pres}}\,\mathbf 1[c_j>0]
\]

!!! example "On paper"

    The token " sorry" has logit 3.0 and has already appeared twice; \(\alpha_{\text{freq}}=0.5\), \(\alpha_{\text{pres}}=0.5\):

    \[
    z = 3.0-0.5\cdot2-0.5\cdot1 = 1.5
    \]

    Its weight \(e^{z}\) falls by a factor \(e^{1.5}\approx4.5\). A third occurrence would cost another 0.5. Large penalties also punish tokens that *must* repeat, such as `"` in JSON or a variable name in code.

The **repetition penalty** (from the CTRL paper, used by Hugging Face and `mlx_lm`) divides positive logits and multiplies negative ones by \(r>1\) for every token already seen (in `mlx_lm`, seen within the last 20 tokens, §9.9):

\[
z_j\leftarrow\begin{cases}z_j/r, & z_j>0\\ z_j\cdot r, & z_j\le0\end{cases}
\]

!!! example "On paper"

    \(r=1.3\) on a seen token with logit 2.6 and another with logit \(-1.0\):

    \[
    \begin{aligned}
    2.6/1.3 &= 2.0\\
    -1.0\cdot1.3 &= -1.3
    \end{aligned}
    \]

    Both move down. Values of 1.05–1.2 are typical; above that, outputs start avoiding necessary words.

### 9.5 A complete sampler

Real servers apply the penalties first and the random draw last. Where the temperature sits relative to the filters varies: Hugging Face `transformers` divides by the temperature and then applies top-k, top-p and min-p, while `mlx_lm` applies the filters to the unscaled distribution and uses the temperature only in the final draw. The sampler below scales first. Writing one is a common interview task ([57](57-ai-career.md)):

```python
import numpy as np


def sample(logits, history, rng, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0,
           repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0):
    z = np.array(logits, dtype=np.float64)
    counts = np.bincount(history, minlength=len(z)) if history else np.zeros(len(z), int)
    seen = counts > 0
    if repetition_penalty != 1.0:                      # CTRL-style: shrink seen logits
        z[seen] = np.where(z[seen] > 0, z[seen] / repetition_penalty, z[seen] * repetition_penalty)
    z -= frequency_penalty * counts + presence_penalty * seen   # OpenAI-style penalties
    if temperature == 0:
        return int(z.argmax())                         # greedy
    p = np.exp((z - z.max()) / temperature)
    p /= p.sum()
    order = np.argsort(-p, kind="stable")              # most likely first
    keep = np.zeros(len(p), bool)
    keep[order[: top_k or len(p)]] = True              # top-k (0 = off)
    cum = np.cumsum(p[order])
    keep[order[(cum - p[order]) >= top_p]] = False     # top-p: drop the tail past p
    keep &= p >= min_p * p.max()                       # min-p
    p = np.where(keep, p, 0.0)
    return int(rng.choice(len(p), p=p / p.sum()))


vocab = [" processed", " approved", " denied", " sent", " delayed"]
logits = np.log([0.5, 0.2, 0.15, 0.1, 0.05])
name = lambda i: vocab[i].strip()
print("greedy:             ", name(sample(logits, [], None, temperature=0)))
print("seed 42, 3 requests:", [name(sample(logits, [], np.random.default_rng(42))) for _ in range(3)])
rng = np.random.default_rng(0)
print("top-p 0.8, 10 draws:", [name(sample(logits, [], rng, top_p=0.8)) for _ in range(10)])
said = [0, 0, 0]                                       # " processed" already appeared 3 times
print("greedy + frequency penalty 0.4:", name(sample(logits, said, None, temperature=0, frequency_penalty=0.4)))
```

```text
greedy:              processed
seed 42, 3 requests: ['denied', 'denied', 'denied']
top-p 0.8, 10 draws: ['approved', 'processed', 'processed', 'processed', 'approved', 'denied', 'approved', 'approved', 'processed', 'denied']
greedy + frequency penalty 0.4: approved
```

A fresh generator with the same seed gives the same draw every time (here "denied", a 15% token: sampling really does pick unlikely tokens). Ten draws with top-p 0.8 only ever produce the three nucleus tokens. And a frequency penalty of 0.4 after three uses of " processed" lowers its logit by 1.2, below " approved": greedy now picks a different word.

### 9.6 Stop sequences and max tokens

Generation ends when the model emits its end-of-turn token, when the output contains one of your **stop sequences**, or when it reaches **`max_tokens`**. The response's `finish_reason` tells you which: `"stop"` for the first two, `"length"` for the cap. Always check it; a `"length"` answer is truncated, and truncated JSON does not parse.

The client below talks to the handbook's local server ([10 §4](10-export-serving.md#4-serve-with-mlx)). `extra_body` passes fields the OpenAI client does not know about; `chat_template_kwargs` with `enable_thinking: false` asks Qwen3's template to skip the thinking block for this request.

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")

def extract(max_tokens):
    resp = client.chat.completions.create(
        model="mlx-community/Qwen3-14B-4bit",
        messages=[{"role": "user", "content": "Return JSON with keys number and total: "
                                              "'Invoice INV-7, total 120.50 EUR'"}],
        temperature=0,
        max_tokens=max_tokens,
        stop=["\n\n"],                       # cut at the first blank line
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0], resp.usage

for limit in (8, 200):
    choice, usage = extract(limit)
    text = choice.message.content or ""      # content can be None (e.g. only a tool call)
    try:
        result = f"parsed {json.loads(text)}"
    except json.JSONDecodeError:
        result = f"NOT JSON: {text!r}"
    print(f"max_tokens={limit:<4} finish_reason={choice.finish_reason:<7} "
          f"completion_tokens={usage.completion_tokens:<3} {result}")
```

```text
max_tokens=8    finish_reason=length  completion_tokens=8   NOT JSON: '{"number": "INV-7",'
max_tokens=200  finish_reason=stop    completion_tokens=17  parsed {'number': 'INV-7', 'total': 120.5}
```

This output came from a stub server that returns canned, OpenAI-shaped responses, because no model runs on the machine that built this page. Against the real `mlx_lm.server` the text and token counts will differ (the model may wrap the JSON in a code fence, for instance), but the behaviour of `finish_reason` is the same: `mlx_lm.server` reports `"length"` when it hits `max_tokens` and `"stop"` on the end token or a stop string, which it removes from the returned text. Structured output and validate-and-retry loops are in [50](50-ai-prompting-apis.md).

**Pitfalls.** A thinking model spends output tokens on its `<think>` block before the answer, so a `max_tokens` sized for the answer alone truncates it; Qwen's model card recommends an output length of 32,768 tokens for most queries. A stop sequence that can occur inside valid output (a blank line inside a code block) cuts answers short.

### 9.7 Logprobs as a confidence signal

A **logprob** is the natural log of the probability the model assigned to a token. Servers can return it for each generated token, plus the top alternatives at each position.

\[
p=e^{\ell}
\]

!!! example "On paper"

    A ticket classifier's first output token came back with logprob \(\ell=-0.094\); the top alternatives were "bug" at \(-2.813\) and "sales" at \(-3.912\):

    \[
    \begin{aligned}
    e^{-0.094} &\approx 0.910\\
    e^{-2.813} &\approx 0.060\\
    e^{-3.912} &\approx 0.020
    \end{aligned}
    \]

    The model put 91% on "billing". A rule such as "auto-route above 0.9, send to a human below" turns this into a product decision; the margin (0.91 against 0.06) is often a better signal than the top value alone.

```python
import math
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
resp = client.chat.completions.create(
    model="mlx-community/Qwen3-14B-4bit",
    messages=[
        {"role": "system", "content": "Classify the ticket as billing, bug or sales. One word."},
        {"role": "user", "content": "I was charged twice for October."},
    ],
    temperature=0,
    max_tokens=5,
    logprobs=True,
    top_logprobs=3,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # no <think> block first
)
choice = resp.choices[0]
print("answer:", choice.message.content or "", "| finish_reason:", choice.finish_reason)
first = choice.logprobs.content[0].top_logprobs          # candidates for the FIRST token
for alt in sorted(first, key=lambda a: -a.logprob):     # the server does not sort them
    print(f"  {alt.token:<8} logprob {alt.logprob:7.3f}   p = {math.exp(alt.logprob):.2f}")
```

```text
answer: billing | finish_reason: stop
  billing  logprob  -0.094   p = 0.91
  bug      logprob  -2.813   p = 0.06
  sales    logprob  -3.912   p = 0.02
```

Again this is stub output; the shape (`choices[0].logprobs.content`, one entry per generated token, each with `token`, `logprob` and `top_logprobs`) matches what the current `mlx_lm.server` code returns. With a real model:

- The **token strings** are the tokenizer's raw pieces (byte-level BPE writes a leading space as `Ġ`), and a label may split into several tokens. Compare the first token of each label, or score each full label.
- `mlx_lm.server` picks the alternatives with a partial sort, so `top_logprobs` is not in order (the code above sorts it), and each entry's own `token` and `logprob` are copied from the first of those alternatives, which is not necessarily the token that was generated. Read the alternatives.
- `mlx_lm` computes logprobs **after** the penalties and `logit_bias` and **before** temperature and the top-k/top-p/min-p filters: they include any penalties you set, but not your temperature or truncation. Check this for any other server.
- **Thinking first** means the first token is `<think>`, not the label: disable thinking or read the logprobs after the think block.
- Logprobs are **not calibrated** out of the box: "0.91" may be right 80% of the time. Calibrate on a labelled set before using thresholds ([47](47-ai-classical-ml.md) covers calibration, [54](54-ai-evaluation.md) evaluation).
- Streaming responses from `mlx_lm.server` carry no logprobs.

### 9.8 Seeds, and why outputs still differ

A **seed** fixes the random draws, so the same logits plus the same seed give the same token (§9.5). `mlx_lm.server` accepts a `seed` field. But identical requests can still produce different text, even at temperature 0, because the *logits* are not bit-identical from run to run:

- Floating-point addition is not associative, so summing the same numbers in a different order gives a slightly different result.
- Servers **batch** concurrent requests. The batch composition changes which kernels run and in what order the sums happen, so your request's logits depend on other users' traffic. (As of October 2026, `mlx_lm.server` serves requests that carry a `seed` one at a time rather than batching them.)
- Different hardware, library versions or quantization change the arithmetic.

A difference in the seventh digit only matters at a near-tie, but over hundreds of tokens one flipped choice sends the rest of the text down another path.

```python
import numpy as np

a, b, c = np.float32(1e8), np.float32(-1e8), np.float32(1.0)
print("(a + b) + c =", (a + b) + c, "   a + (b + c) =", a + (b + c))

rng = np.random.default_rng(0)
x = rng.normal(size=4096).astype(np.float32)
s1 = np.float32(0)
for v in x:                                    # one long left-to-right sum
    s1 += v
s2 = x.reshape(64, 64).sum(axis=1).sum()       # the same numbers, summed in blocks
print(f"sequential sum {s1:.7f}  blocked sum {s2:.7f}  equal: {s1 == s2}")

logits = np.array([5.0, 4.9999995], dtype=np.float32)       # a near tie, one float32 step apart
nudged = logits + np.array([0.0, 1e-6], dtype=np.float32)     # a different kernel, a different sum order
print("argmax:", logits.argmax(), "-> after a 1e-6 nudge:", nudged.argmax())
```

```text
(a + b) + c = 1.0    a + (b + c) = 0.0
sequential sum -66.0645828  blocked sum -66.0646133  equal: False
argmax: 0 -> after a 1e-6 nudge: 1
```

**What to do.** Treat outputs as samples, not as a pure function of the prompt. Test with several samples per input and assert on properties (valid JSON, correct label, contains the order number), not on exact strings ([54](54-ai-evaluation.md)). Log the model version, sampling settings and seed with every response so you can at least reproduce the conditions.

### 9.9 Sampling settings cheat-sheet

| Task | temperature | top_p / top_k / min_p | Other |
|---|---|---|---|
| Classification, extraction, routing | 0 | — | small `max_tokens`, stop on newline, logprobs |
| Code generation, tool calls | 0–0.3 | top_p 0.95 | validate output, retry on parse failure |
| Chat, summaries | 0.6–0.8 | top_p 0.8–0.95 | presence penalty if it repeats itself |
| Brainstorming, synthetic data | 0.8–1.2 | min_p 0.05–0.1 | several samples, deduplicate ([07](07-synthetic-data.md)) |
| Qwen3, thinking mode (model card) | 0.6 | top_p 0.95, top_k 20, min_p 0 | no greedy decoding |
| Qwen3, non-thinking mode (model card) | 0.7 | top_p 0.8, top_k 20, min_p 0 | in either mode, presence penalty 0–2 if it repeats endlessly |

`mlx_lm.server` accepts `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `seed`, `stop`, `max_tokens`, `logprobs` and `top_logprobs` in the request body; its documented defaults include temperature 0, top_p 1.0 and `max_tokens` 512, and it applies each penalty only over the last 20 tokens unless you set `repetition_context_size`, `presence_context_size` or `frequency_context_size`. The server's own flags in [10 §4](10-export-serving.md#4-serve-with-mlx) set the defaults per deployment.

---

## 10. Context windows and the KV cache

### 10.1 The context window

The **context window** is the maximum number of tokens the model can attend to at once: prompt plus everything it generates.

\[
T_{\text{prompt}}+T_{\text{output}}\le T_{\max}
\]

!!! example "On paper"

    Qwen3-14B natively supports 32,768 tokens. A RAG assistant sends a 1,500-token system prompt, 8 retrieved chunks of 600 tokens, 6,000 tokens of history and a 200-token question:

    \[
    \begin{aligned}
    T_{\text{prompt}} &= 1{,}500+4{,}800+6{,}000+200 = 12{,}500\\
    T_{\text{output}} &\le 32{,}768-12{,}500 = 20{,}268
    \end{aligned}
    \]

    Plenty for an answer, but in thinking mode the reasoning tokens come out of the same 20,268. Servers differ when a request does not fit: some reject it, some truncate the prompt, some stop at the limit; know which yours does.

A long window is not the same as good use of it: models retrieve facts from the middle of very long prompts less reliably than from the start or end. Put instructions and the most relevant material where they are easy to find, and measure on your own data ([51](51-ai-rag.md)). Qwen3's `max_position_embeddings` is 40,960; YaRN scaling extends it to about 131,072 tokens ([37 §5.1](37-architecture.md#51-traps)).

### 10.2 The KV cache

When generating token \(t+1\), attention needs the keys and values of all earlier tokens. They do not change, so the server stores them instead of recomputing them: the **KV cache**. Each new token then costs one projection and one attention row instead of a full pass over the prefix.

\[
\text{KV bytes}=2\cdot L\cdot G\cdot d_h\cdot T\cdot b
\]

!!! example "On paper"

    Qwen3-14B at its full 32,768-token context in bf16, using the 163,840 bytes per token from §5.2:

    \[
    \begin{aligned}
    163{,}840\cdot32{,}768 &= 5{,}368{,}709{,}120\ \text{B}\\
    &\approx 5.37\ \text{GB}\ (=5\ \text{GiB})
    \end{aligned}
    \]

    That is per conversation, on top of 8.3 GB of 4-bit weights: the same 5.37 GB figure as [32 §6.3](32-mathematics.md#63-kv-cache) and [02 §3.5](02-hardware.md#35-kv-cache-is-inference-only). Four users at 8,192 tokens each need the same 5.37 GB. An int8 cache roughly halves it and a 4-bit cache (`--kv-bits 4` in `mlx_lm`) roughly quarters it; the per-group scales add a little on top.

```python
import numpy as np

rng = np.random.default_rng(0)
d, T = 16, 6
W_Q, W_K, W_V = (rng.normal(size=(d, d)) / 4 for _ in range(3))
X = rng.normal(size=(T, d))                       # 6 token vectors entering one attention layer

def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def full_causal(X):                               # recompute everything for the whole prefix
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    s = Q @ K.T / np.sqrt(d)
    s[np.triu(np.ones((len(X), len(X)), bool), 1)] = -np.inf
    return softmax(s) @ V

K_cache, V_cache, outs, projections = [], [], [], 0
for t in range(T):                                # decode: one new token per step
    x = X[t]
    K_cache.append(x @ W_K); V_cache.append(x @ W_V)          # project the new token once
    projections += 2
    q = x @ W_Q
    outs.append(softmax(q @ np.array(K_cache).T / np.sqrt(d)) @ np.array(V_cache))
print("cached == recomputed:", np.allclose(np.array(outs), full_causal(X)))
print("K/V projections with cache:", projections,
      " without (recompute the prefix each step):", sum(2 * (t + 1) for t in range(T)))
```

```text
cached == recomputed: True
K/V projections with cache: 12  without (recompute the prefix each step): 42
```

The cached and recomputed outputs agree, and the work drops from quadratic to linear in the number of tokens. The cache is also why **prompt caching** works: if many requests share the same system prompt and tool definitions, the server can keep that prefix's cache and skip its prefill (`--prompt-cache-size` in `mlx_lm.server`, [55](55-ai-production.md)). [37 §2.1](37-architecture.md#21-cache-formula) shows how Gemma 4's hybrid layout shrinks the cache further.

### 10.3 Prefill and decode: why long prompts are slow

Serving has two phases with different bottlenecks:

- **Prefill** processes the whole prompt in one parallel pass and fills the cache. It does about \(2N\) FLOPs per prompt token, so it is **compute-bound**, and it sets the **time to first token (TTFT)**.
- **Decode** produces one token per step. Each step reads every weight (and the cache) from memory to do comparatively little arithmetic, so it is **memory-bandwidth-bound**.

\[
C_{\text{prefill}}\approx2N\,T_{\text{prompt}}
\]

!!! example "On paper"

    Qwen3-14B (\(N\approx14.77\)B) with an 8,000-token prompt. **Assume** the Mac sustains about 3.5 TFLOP/s in prefill. That figure comes from [llama.cpp's Apple Silicon benchmark](https://github.com/ggml-org/llama.cpp/discussions/4167): the 16-core M1 Pro prefills a 7B (6.74B-parameter) Q4_0 model at 266 tokens/s (pp512), which is \(2\cdot6.74\times10^{9}\cdot266\approx3.6\times10^{12}\) FLOP/s, or about 120 tokens/s once scaled to Qwen3-14B:

    \[
    \begin{aligned}
    C &= 2\cdot14.77\times10^{9}\cdot8{,}000 \approx 2.36\times10^{14}\\
    \text{TTFT} &\approx 2.36\times10^{14}/(3.5\times10^{12}) \approx 68\ \text{s}
    \end{aligned}
    \]

    More than a minute before the first word appears (8,000 tokens at about 120 tokens/s), entirely due to prompt length. Halving the prompt halves the wait; a cached prefix removes its share completely.

\[
\text{tokens/s}\lesssim\frac{\text{memory bandwidth}}{\text{bytes read per token}}
\]

!!! example "On paper"

    The M1 Pro has about 200 GB/s ([02 §1](02-hardware.md#1-the-machine)) and Qwen3-14B at 4-bit is about 8.3 GB. At a full 32K context the 5.37 GB cache is read too:

    \[
    \begin{aligned}
    200/8.3 &\approx 24\ \text{tokens/s (short context)}\\
    200/(8.3+5.37) &\approx 14.6\ \text{tokens/s (32K context)}
    \end{aligned}
    \]

    These are ceilings; [10 §8](10-export-serving.md#8-performance-expectations) expects about 15–17 tokens/s in practice at short context. A 300-token answer takes at least \(300/24=12.5\) s. Smaller or quantized weights decode faster because there are fewer bytes to read.

The engineering consequences: long prompts hurt TTFT, long outputs hurt total latency, and long *contexts* slow decode as well. Production serving (batching, speculative decoding, prefill/decode separation) is in [55 — Production](55-ai-production.md).

---

## 11. Model size, memory and quantization

### 11.1 Parameters times bytes

The weights dominate memory at short contexts, and their size is simple arithmetic:

\[
\text{weight bytes}=N\times\frac{\text{bits per weight}}{8}
\]

!!! example "On paper"

    Qwen3-14B, \(N\approx14.77\)B:

    \[
    \begin{aligned}
    \text{bf16 (16 bits)}: 14.77\cdot2 &\approx 29.5\ \text{GB}\\
    \text{8-bit}: 14.77\cdot1 &\approx 14.8\ \text{GB}\\
    \text{4-bit}: 14.77\cdot0.5 &\approx 7.4\ \text{GB}\\
    \text{MLX 4-bit, 4.5 bits}: 14.77\cdot0.5625 &\approx 8.3\ \text{GB}
    \end{aligned}
    \]

    MLX stores a 16-bit scale and a 16-bit offset for every group of 64 weights: \((64\cdot4+32)/64=4.5\) bits per weight. That is where the handbook's 8.3 GB comes from ([02 §3](02-hardware.md#3-the-memory-budget)).

| Model | bf16 | 8-bit | MLX 4-bit | Fits the Mac's ~21.3 GiB GPU budget? |
|---|---:|---:|---:|---|
| Qwen3-8B (8.2B) | 16.4 GB | 8.2 GB | ~4.6 GB | yes, at every precision but bf16 is tight |
| Qwen3-14B (14.8B) | 29.5 GB | 14.8 GB | ~8.3 GB | 4-bit and 8-bit; bf16 no |
| Qwen3-32B (32.8B) | 65.6 GB | 32.8 GB | ~18.5 GB | 4-bit only, with a short context ([01](01-landscape.md)) |
| Qwen3-30B-A3B MoE (30.5B) | 61 GB | 30.5 GB | ~17.2 GB | 4-bit only; fast because only 3.3B are active (§15) |

Add the KV cache (§10.2) and 2–4 GB of runtime to the weights before deciding. The 8B and 32B figures use the parameter counts from [01](01-landscape.md) and [26](26-one-model-for-all.md).

### 11.2 Quantization by hand

**Quantization** stores weights as small integers plus a scale. The simplest scheme, **absmax int8**, divides by the largest absolute value so that it maps to 127:

\[
\begin{aligned}
s&=\frac{\max_i|w_i|}{127}\\
q_i&=\operatorname{round}(w_i/s)\\
\hat w_i&=s\,q_i
\end{aligned}
\]

!!! example "On paper"

    Four weights \(\mathbf w=(0.523,\,-1.27,\,0.336,\,0.0414)\):

    \[
    \begin{aligned}
    s &= 1.27/127 = 0.01\\
    \mathbf w/s &= (52.3,\,-127,\,33.6,\,4.14)\\
    \mathbf q &= (52,\,-127,\,34,\,4)\\
    \hat{\mathbf w} &= (0.52,\,-1.27,\,0.34,\,0.04)\\
    |\mathbf w-\hat{\mathbf w}| &= (0.003,\,0,\,0.004,\,0.0014)
    \end{aligned}
    \]

    Every error is at most \(s/2=0.005\): rounding to a grid of step \(s\) can be off by at most half a step. Storage drops from 4 bytes per weight (fp32) to 1 byte, plus one shared scale.

```python
import numpy as np

def absmax_quant(w, bits):
    qmax = 2 ** (bits - 1) - 1                    # 127 for int8, 7 for int4
    scale = np.abs(w).max() / qmax
    q = np.round(w / scale).astype(np.int8)
    return q, scale, q * scale                    # codes, scale, dequantised values

w = np.array([0.523, -1.27, 0.336, 0.0414])
q, s, w_hat = absmax_quant(w, 8)
print("codes", q.tolist(), " scale", round(s, 4), " back", w_hat.round(4).tolist())
print("errors", np.abs(w - w_hat).round(4).tolist())

w_out = w.copy(); w_out[1] = -12.7                # one outlier stretches the scale 10x
q, s, w_hat = absmax_quant(w_out, 8)
print("with outlier: codes", q.tolist(), " back", w_hat.round(4).tolist())

rng = np.random.default_rng(0)                    # a 4096-weight row with one outlier
row = rng.normal(0, 0.02, 4096); row[100] = 0.5
for name, bits, group in [("int8, one scale", 8, 4096), ("int4, one scale", 4, 4096),
                          ("int4, groups of 64", 4, 64)]:
    w_hat = np.concatenate([absmax_quant(g, bits)[2] for g in row.reshape(-1, group)])
    print(f"{name:<20} mean |error| {np.abs(row - w_hat).mean():.5f}")
```

```text
codes [52, -127, 34, 4]  scale 0.01  back [0.52, -1.27, 0.34, 0.04]
errors [0.003, 0.0, 0.004, 0.0014]
with outlier: codes [5, -127, 3, 0]  back [0.5, -12.7, 0.3, 0.0]
int8, one scale      mean |error| 0.00099
int4, one scale      mean |error| 0.01492
int4, groups of 64   mean |error| 0.00202
```

The outlier line shows the weakness: one large weight (−12.7) stretches the scale to 0.1, and the small weight 0.0414 rounds to 0, a 100% error. Real weight rows have exactly such outliers. With 4 bits there are only 15 levels, so a single scale per row wipes out most small weights (mean error 0.0149 on weights of typical size 0.02); one scale per **group of 64** confines the damage to the outlier's group and cuts the error sevenfold. That is why MLX, GGUF and every practical 4-bit format are group-wise. [32 §15](32-mathematics.md#15-quantization-mathematics) covers zero points and the error bound, and [53](53-ai-fine-tuning.md) measures what quantization does to a fine-tune.

---

## 12. How a model is trained: the pipeline

### 12.1 Pretraining

**Pretraining** is next-token prediction (§8) over a huge, filtered corpus: web pages, books, code, papers, often with synthetic data mixed in. Qwen3 was pretrained on 36 trillion tokens in 119 languages and dialects, in three stages (general, reasoning-heavy, long-context) ([33 §4](33-qwen-creation.md)). The result is a **base model**: it continues text, but it does not follow instructions or stop politely. The compute is roughly six FLOPs per parameter per token ([32 §14.1](32-mathematics.md#141-training-compute)):

\[
C\approx6\,N\,D
\]

!!! example "On paper"

    Qwen3-14B: \(N\approx14.77\times10^{9}\) parameters, \(D=36\times10^{12}\) tokens:

    \[
    \begin{aligned}
    C &\approx 6\cdot14.77\times10^{9}\cdot36\times10^{12}\\
    &\approx 3.19\times10^{24}\ \text{FLOPs}
    \end{aligned}
    \]

    [35 §1.1](35-from-zero.md#11-worked-examples) turns this into about 9 days on 10,000 H100-class GPUs, or tens of thousands of years on one M1 Pro. You will not pretrain a 14B model; you will choose, prompt and adapt one.

Data quality matters more than any architectural detail at this stage: deduplication, filtering, language balance and decontamination of benchmarks decide what the model knows ([60 — Data preparation](60-ai-data-preparation.md), [36 §3](36-reproducing-12b-14b.md#3-data-the-hard-part)).

### 12.2 Supervised fine-tuning (SFT)

**Supervised fine-tuning** continues training on curated conversations: a prompt and the response you want, rendered with the chat template, with the loss on the response only (§8.5). Tens of thousands to a few million examples teach the base model the assistant format, instruction following, tool-call syntax and refusals. It changes *behaviour* far more easily than it adds *knowledge*. Your own fine-tunes (LoRA on the Mac, [08](08-finetuning.md), [53](53-ai-fine-tuning.md)) are small SFT runs; the data format is in [06](06-data-format.md) and data preparation in [60](60-ai-data-preparation.md).

### 12.3 Preference tuning: RLHF and DPO

SFT shows the model good answers; **preference tuning** teaches it which of two answers people (or a judge) prefer. The data is pairs: a prompt, a *chosen* response and a *rejected* one.

- **RLHF** (reinforcement learning from human feedback) first trains a **reward model** to score responses so that chosen beats rejected, then optimises the LLM with reinforcement learning (classically PPO) to get high reward, with a penalty for drifting too far from the SFT model.
- **DPO** (direct preference optimization) skips the reward model and the RL loop: a closed-form loss raises the probability of the chosen response relative to the rejected one, again relative to a frozen reference model. It is simpler and the common choice for small teams.
- **RL with verifiable rewards** (GRPO and relatives) replaces human preference with a checker, such as unit tests or a maths answer key, and is how reasoning skills are trained.

[15 — Agentic methods](15-agentic-methods.md) covers DPO, ORPO and GRPO for this handbook's models; [53](53-ai-fine-tuning.md) computes the DPO loss with numbers.

### 12.4 Reasoning models and test-time compute

A **reasoning model** is trained (with long chain-of-thought SFT and then RL on verifiable tasks) to write out intermediate reasoning, its **thinking tokens**, before the answer. Qwen3 does this inside a `<think>...</think>` block and can switch it on or off per request (`enable_thinking`, or `/think` and `/no_think` in a message). Spending more tokens at inference time to get better answers is called **test-time compute**. It helps on maths, code and multi-step planning, and is wasted on lookups and simple classification.

Qwen3 thinks **by default**. Unless a request disables it (`extra_body={"chat_template_kwargs": {"enable_thinking": False}}`, as in §9.6–9.7), the reply starts with a think block: the current `mlx_lm.server` moves that text into a separate `reasoning` field of the message, other servers call it `reasoning_content` or leave `<think>` inside `content`, and `content` can even be empty if `max_tokens` runs out while the model is still thinking. Logprobs, `max_tokens` and stop sequences all apply to the thinking tokens too, so read `content or ""` defensively. [06 §6](06-data-format.md#6-thinking-and-reasoning_content) shows how thinking is rendered in training data, which must match how you serve. The cost is paid in latency and tokens:

\[
\text{latency}\approx\text{TTFT}+\frac{T_{\text{think}}+T_{\text{answer}}}{\text{tokens/s}}
\]

!!! example "On paper"

    **Assume** 2 s to first token and the 24 tokens/s ceiling of §10.3. An answer of 200 tokens, with and without 2,000 thinking tokens:

    \[
    \begin{aligned}
    \text{thinking} &= 2+2{,}200/24 \approx 93.7\ \text{s}\\
    \text{no thinking} &= 2+200/24 \approx 10.3\ \text{s}
    \end{aligned}
    \]

    Nine times slower, and the thinking tokens also count against `max_tokens`, the context window and, with an API, the bill. Turn thinking on where an eval shows it pays ([54](54-ai-evaluation.md)).

### 12.5 Distillation

**Distillation** trains a small **student** model to imitate a large **teacher**, either on the teacher's generated answers (sequence-level) or on its full next-token probabilities (logit-level). The smaller Qwen3 models were trained largely this way from the flagship, at a fraction of the compute of running the full pipeline ([33 §4.4](33-qwen-creation.md#44-post-training-four-stages)). It is also how you can transfer a big model's skill on *your* task into a model that is cheap to serve; [39 — Distillation](39-gemma-distillation.md), [40](40-distill-data.md) and [41](41-distill-training.md) are the handbook's runbook.

| Stage | Data | Objective | Produces |
|---|---|---|---|
| Pretraining | trillions of tokens of raw text and code | next-token loss | base model: knowledge, fluency |
| SFT | thousands to millions of chat examples | next-token loss on responses | instruction-following assistant |
| Preference tuning | chosen/rejected pairs | reward model + RL, or DPO | preferred style, safety, helpfulness |
| Reasoning RL | problems with checkable answers | reward from a verifier (GRPO) | reasoning model with thinking tokens |
| Distillation | teacher outputs or logits | match the teacher | smaller model with similar skills |

---

## 13. Scaling laws

**Scaling laws** are empirical fits showing that pretraining loss falls smoothly, as a power law, when you increase parameters \(N\), training tokens \(D\) or compute \(C\). They let labs predict a large run's loss from small runs. The **Chinchilla** study ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)) found that for a fixed compute budget, parameters and tokens should grow together, at roughly 20 tokens per parameter:

\[
D_{\text{opt}}\approx20\,N
\]

!!! example "On paper"

    For Qwen3-14B's size:

    \[
    \begin{aligned}
    D_{\text{opt}} &\approx 20\cdot14.77\times10^{9}\\
    &\approx 295\times10^{9}\ \text{tokens}\\
    36\times10^{12}/14.77\times10^{9} &\approx 2{,}437
    \end{aligned}
    \]

    Qwen3-14B saw about 2,437 tokens per parameter, roughly 120 times the compute-optimal count. That is deliberate: Chinchilla minimises *training* cost for a given loss, but a model served billions of times is cheaper overall if it is smaller and trained longer ("overtrained").

What scaling laws mean for an engineer: bigger and longer-trained models are reliably better at next-token prediction, with diminishing returns; a smaller model trained on far more tokens can match an older larger one; and the curves describe average loss, not whether a specific skill your product needs will appear. [35 §1.2](35-from-zero.md#12-chinchilla-is-a-lower-bound-not-a-target), [36 §2.3](36-reproducing-12b-14b.md#23-chinchilla-vs-production) and [38 §12](38-derivations.md#12-scaling-law-fit) work the formulas.

---

## 14. Failure modes that come from the objective

### 14.1 Hallucination

A **hallucination** is fluent, confident output that is false or unsupported: an invented API parameter, a non-existent citation, a wrong refund policy. It follows directly from the objective. The model is trained to produce the most plausible continuation, and plausibility is not truth. When it lacks the fact, the plausible continuation still has the *shape* of an answer (a citation looks like author, year, title), and nothing in next-token training rewards saying "I don't know" unless post-training added it. §1.3's bigram produced "your order arrived ." without any order having arrived; scale that up and you get a confident tracking number.

What helps, in order of reliability:

- **Give it the facts**: retrieval ([51](51-ai-rag.md)) or tools ([52](52-ai-agents.md)), and instruct it to answer only from them and cite.
- **Make it checkable**: structured output with fields you validate; IDs, prices and dates checked against your systems.
- **Detect low confidence**: logprobs on key tokens (§9.7), disagreement between several samples.
- **Measure it**: a hallucination eval on real questions with known answers ([54](54-ai-evaluation.md)), and mitigation patterns in [56](56-ai-safety.md).

### 14.2 Knowledge cutoff

The weights contain what was in the training data up to the **knowledge cutoff** date, and nothing after. A model asked about a later library version or price change will either say it does not know or, more often, answer from the older state of the world as if it were current. It also does not know today's date unless you tell it. Put the date and any time-sensitive facts in the prompt, and prefer retrieval for anything that changes.

### 14.3 Sycophancy

**Sycophancy** is the tendency to agree with the user: to change a correct answer when the user pushes back, or to praise a flawed plan. Preference tuning rewards answers people rate highly, and people tend to rate agreement highly. Test it directly: ask a question, then reply "I think that's wrong, isn't it X?" and measure how often a correct answer flips. Mitigate with a system prompt that asks for corrections, with evals that include pushback turns, and by not leaking the expected answer in the question ("Is this code correct?" invites "yes").

### 14.4 Prompt sensitivity

Small changes that should not matter can change the output: the order of few-shot examples, the order of multiple-choice options, a missing newline, a synonym in the instruction, a different chat template. This follows from §9: when two continuations are close in probability, any change to the input moves the logits enough to swap them. Treat prompts as code: version them, evaluate each change on a fixed set with several paraphrases, and report variance, not one lucky run ([50](50-ai-prompting-apis.md), [54](54-ai-evaluation.md)).

| Failure | Root cause | First defence |
|---|---|---|
| Hallucination | plausible ≠ true; no abstention in the objective | retrieval and tools, validation, evals |
| Knowledge cutoff | weights frozen at a date | date and facts in context, retrieval |
| Sycophancy | preference data rewards agreement | pushback evals, neutral prompts |
| Prompt sensitivity | near-ties in the next-token distribution | prompt versioning, paraphrase evals, low temperature |

---

## 15. Dense vs mixture-of-experts

A **dense** model uses every parameter for every token. A **mixture-of-experts (MoE)** model replaces each MLP with many smaller "expert" MLPs and a small router that sends each token to the top few. Only the chosen experts run, so the **active** parameter count per token is a fraction of the **total**. Qwen3-30B-A3B has 128 experts per layer, uses 8 per token, and has 30.5B total and 3.3B active parameters ([32 §9](32-mathematics.md#9-mixture-of-experts-moe) has the routing math and the active count).

\[
\begin{aligned}
\text{memory}&\propto N_{\text{total}}\\
\text{FLOPs per token}&\approx2\,N_{\text{active}}
\end{aligned}
\]

!!! example "On paper"

    Qwen3-30B-A3B against dense Qwen3-14B, both at MLX 4-bit (4.5 bits):

    \[
    \begin{aligned}
    \text{memory} &= 30.5\cdot0.5625 \approx 17.2\ \text{GB}\\
    \text{FLOPs/token} &= 2\cdot3.3\times10^{9} = 6.6\times10^{9}\\
    \text{14B dense} &= 2\cdot14.77\times10^{9}\\
    &\approx 2.95\times10^{10}
    \end{aligned}
    \]

    About twice the memory of the 14B for about a quarter of the compute per token. Decode reads only the active weights, about \(3.3\cdot0.5625\approx1.86\) GB per token, so the bandwidth ceiling is near \(200/1.86\approx108\) tokens/s against 24 for the dense 14B.

The trade-off for a product: an MoE gives you the knowledge of a large model at the speed of a small one, if you have the memory to hold all experts. Fine-tuning MoE models is harder (router stability, experts that see little data), which is one reason this handbook fine-tunes dense models. When a model name includes "A3B" or "A22B", the second number is the active size.

---

## 16. Open-weight vs API models

An **open-weight** model (Qwen3, Gemma 4) is a set of files you download and run yourself under its license. An **API model** is a service you call per token; you never see the weights.

| Concern | Open-weight, self-hosted | API model |
|---|---|---|
| Quality ceiling | strong mid-size models; the very largest need serious hardware | usually the strongest models available |
| Data privacy | prompts never leave your machine or network | sent to the provider under their terms |
| Cost structure | hardware and operations, near zero per extra token | per token, no fixed cost |
| Latency | depends on your hardware; no network hop | network plus provider queue; often fast decode |
| Control | pin the exact version, change sampling, read logprobs, fine-tune freely | the provider may update or retire the model |
| Customisation | full: LoRA, distillation, quantization | limited to what the provider offers |
| Operations | you run, scale, patch and monitor it | the provider does |
| Licensing | read it: Qwen3 is Apache-2.0 ([12 §4](12-safety-legal.md#4-model-and-dataset-licenses)) | terms of service and usage policies |

Many products use both: an API model to prototype and for the hardest requests, a local or self-hosted model for high-volume, private or latency-sensitive paths. The break-even arithmetic and routing patterns are in [55 — Production](55-ai-production.md).

---

## 17. Reading a model card and a config.json

### 17.1 The model card

A **model card** is the README of a checkpoint. Read it for:

| Look for | Why | Qwen3-14B example |
|---|---|---|
| License | can you ship it? | Apache-2.0 |
| Parameter count (total, non-embedding, active for MoE) | memory and speed | 14.8B total, 13.2B non-embedding |
| Context length, native vs extended | prompt budget | 32,768 native; 131,072 validated with YaRN |
| Base or instruct/chat, thinking modes | which file to deploy | hybrid thinking, `enable_thinking`, `/think` and `/no_think` |
| Chat template and special tokens | must match at fine-tuning and serving | ChatML, in `tokenizer_config.json` ([06](06-data-format.md)) |
| Recommended sampling settings | the authors tuned them | §9.9 |
| Benchmarks | relative ordering only; vendor-run | verify on your own eval ([54](54-ai-evaluation.md)) |
| Training data and cutoff | what it can know | stated in the technical report, not always precisely |
| Known limitations and intended use | risk review | [56](56-ai-safety.md) |

### 17.2 config.json

`config.json` defines the shapes: every number in this chapter can be computed from it before you download a single weight. The keys are explained in [37 §5](37-architecture.md#5-reading-configjson); this script reads the shape-defining keys of Qwen3-14B's real config and derives size, cache and memory:

```python
import json

config = json.loads("""{
  "architectures": ["Qwen3ForCausalLM"], "hidden_act": "silu",
  "hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 40,
  "num_attention_heads": 40, "num_key_value_heads": 8, "head_dim": 128,
  "vocab_size": 151936, "tie_word_embeddings": false,
  "max_position_embeddings": 40960, "rope_theta": 1000000, "torch_dtype": "bfloat16"
}""")          # the shape-defining keys of Qwen/Qwen3-14B's config.json

d, L, ff = config["hidden_size"], config["num_hidden_layers"], config["intermediate_size"]
H, G = config["num_attention_heads"], config["num_key_value_heads"]
d_h = config.get("head_dim") or d // H           # head_dim may be missing: then d / H
V = config["vocab_size"]

attn = d * H * d_h + 2 * d * G * d_h + H * d_h * d   # q, k, v, o projections
mlp = 3 * d * ff                                     # gate, up, down (SwiGLU)
norms = 2 * d + 2 * d_h                              # two RMSNorms + q/k norms per layer
emb = V * d * (1 if config["tie_word_embeddings"] else 2)
total = L * (attn + mlp + norms) + emb + d           # + the final norm
print(f"per layer: attention {attn / 1e6:.2f}M, MLP {mlp / 1e6:.2f}M")
print(f"total {total / 1e9:.2f}B, of which embeddings + head {emb / 1e9:.2f}B")
print(f"GQA ratio {H // G}:1, KV cache {2 * L * G * d_h * 2:,} bytes/token in bf16")
for name, bits in [("bf16", 16), ("8-bit", 8), ("4-bit", 4), ("MLX 4-bit incl. scales", 4.5)]:
    print(f"weights at {name:<23} {total * bits / 8 / 1e9:5.1f} GB")
```

```text
per layer: attention 62.91M, MLP 267.39M
total 14.77B, of which embeddings + head 1.56B
GQA ratio 5:1, KV cache 163,840 bytes/token in bf16
weights at bf16                     29.5 GB
weights at 8-bit                    14.8 GB
weights at 4-bit                     7.4 GB
weights at MLX 4-bit incl. scales    8.3 GB
```

The result reproduces the model card's 14.8B and the 62.91M/267.39M per-layer split of [37 §1.3](37-architecture.md#13-parameter-budget). Things the config tells you at a glance: `hidden_act: "silu"` means SwiGLU MLPs; `num_key_value_heads` smaller than `num_attention_heads` means GQA; `tie_word_embeddings: false` means a second 0.78B-parameter matrix for the head; `max_position_embeddings` is a positional limit, not a promise of quality at that length; MoE configs add `num_experts` and `num_experts_per_tok`. The handbook's `scripts/inspect_architecture.py` does this for any Hugging Face config ([37 §5.2](37-architecture.md#52-inspect-a-checkpoint-before-downloading)).

---

## 18. Key takeaways

- An LLM maps token ids to a probability for every possible next token; generation is a loop of run, sample, append, with a KV cache so each step is cheap.
- Tokens are the unit of cost, latency and context. Count them with the model's own tokenizer; non-English text, numbers and code cost more than you expect.
- BPE builds the vocabulary by repeatedly merging the most frequent pair; byte-level BPE can encode any string.
- Attention is a soft lookup: scores \(QK^\top/\sqrt{d_k}\), a causal mask, a softmax, a weighted sum of values. Its cost grows with \(T^2\).
- GQA shares key/value heads to shrink the cache; RoPE encodes relative position by rotating queries and keys.
- Training minimises average negative log-likelihood; perplexity is its exponential. Low loss means fluent, not true.
- Temperature reshapes, top-k/top-p/min-p truncate, penalties discourage repeats. Check `finish_reason`, use logprobs with care, and expect run-to-run variation even with a seed.
- Memory = weights (params × bits/8) + KV cache (\(2LGd_hTb\)) + runtime. Qwen3-14B is 8.3 GB at MLX 4-bit plus 5.37 GB of cache at 32K.
- Prefill is compute-bound and sets time to first token; decode is bandwidth-bound and sets tokens per second.
- Models are built by pretraining, SFT, preference tuning and reasoning RL, often with distillation; hallucination, cutoff, sycophancy and prompt sensitivity follow from those objectives and must be engineered around.

---

## 19. Exercises

!!! question "Exercise 1"

    Train BPE by hand, with no end-of-word marker, on the corpus hug ×10, pug ×5, pun ×12, bun ×4, hugs ×5. What are the first three merges?

??? success "Answer"

    Pair counts: (u, g) = 10 + 5 + 5 = 20, (u, n) = 12 + 4 = 16, (p, u) = 5 + 12 = 17, (h, u) = 15. Merge 1 is u + g → ug (20). Recount: (h, ug) = 15, (u, n) = 16, (p, u) = 12, (p, ug) = 5. Merge 2 is u + n → un (16). Recount: (h, ug) = 15, (p, un) = 12. Merge 3 is h + ug → hug (15).

!!! question "Exercise 2"

    The second of two tokens has scaled attention scores \((0,\ \ln3)\) against the two tokens, and the values are \(\mathbf v_1=(4,0)\) and \(\mathbf v_2=(0,8)\). What does it output?

??? success "Answer"

    \(e^{0}=1\) and \(e^{\ln3}=3\), so the weights are \((1/4,\ 3/4)\). The output is \(0.25\,(4,0)+0.75\,(0,8)=(1,\ 6)\).

!!! question "Exercise 3"

    Next-token probabilities are \((0.4,\,0.3,\,0.2,\,0.1)\). Which tokens survive top-p 0.65, and with what probabilities? Which survive min-p 0.4? What is the distribution at temperature 0.5?

??? success "Answer"

    Top-p: running totals 0.4, 0.7; the second token crosses 0.65, so two survive, renormalised to \(0.4/0.7=0.571\) and \(0.3/0.7=0.429\). Min-p 0.4: the threshold is \(0.4\cdot0.4=0.16\), so 0.4, 0.3 and 0.2 survive, renormalised over 0.9 to \((0.444,\,0.333,\,0.222)\). Temperature 0.5 squares and renormalises: \((0.16,0.09,0.04,0.01)/0.30=(0.533,\,0.3,\,0.133,\,0.033)\).

!!! question "Exercise 4"

    A model assigns probability 0.25 to every correct next token of a test set. What are its loss and perplexity? A second model, with the same tokenizer, has loss 1.0. Which is better?

??? success "Answer"

    Loss \(=-\ln0.25=\ln4\approx1.386\), perplexity \(=4\). The second model has perplexity \(e^{1.0}\approx2.718\), lower, so it predicts the test set better. With different tokenizers the comparison would be invalid.

!!! question "Exercise 5"

    How much KV cache does Qwen3-14B need for one 8,192-token conversation in bf16? For four such conversations at once? With an int8 cache?

??? success "Answer"

    \(163{,}840\cdot8{,}192=1{,}342{,}177{,}280\) bytes \(\approx1.34\) GB for one; \(4\cdot1.342\approx5.37\) GB for four, the same as one 32K conversation. An int8 cache halves both: about 0.67 GB for one, 2.68 GB for four.

!!! question "Exercise 6"

    Qwen3-32B has 32.8B parameters, 64 layers, 8 KV heads and head dimension 128. Does it fit under the Mac's ~21.3 GiB (22.9 GB) GPU budget at MLX 4-bit with a full 32,768-token bf16 cache?

??? success "Answer"

    Weights: \(32.8\cdot4.5/8\approx18.45\) GB. Cache per token: \(2\cdot64\cdot8\cdot128\cdot2=262{,}144\) bytes, so \(262{,}144\cdot32{,}768\approx8.59\) GB at 32K. Total about 27 GB before the runtime: no. At 8K context the cache is about 2.15 GB and the total about 20.6 GB plus runtime, right at the edge; a 4-bit KV cache would help.

!!! question "Exercise 7"

    Quantize \(\mathbf w=(2.54,\,-0.8,\,0.013,\,1.0)\) with absmax int8. What are the codes, the dequantised values and the relative error on the smallest weight?

??? success "Answer"

    \(s=2.54/127=0.02\). Codes: \(\operatorname{round}(127,\,-40,\,0.65,\,50)=(127,\,-40,\,1,\,50)\). Dequantised: \((2.54,\,-0.8,\,0.02,\,1.0)\). The small weight became 0.02 instead of 0.013: an absolute error of 0.007 (under \(s/2=0.01\)) but a relative error of about 54%. Small weights next to a large one are where per-tensor quantization hurts; groups fix it.

!!! question "Exercise 8"

    Using the Chinchilla rule, how many tokens should a 7B model see, and how much training compute is that?

??? success "Answer"

    \(D\approx20\cdot7\times10^{9}=1.4\times10^{11}\) tokens (140B). \(C\approx6ND=6\cdot7\times10^{9}\cdot1.4\times10^{11}=5.88\times10^{21}\) FLOPs. Production models usually train far longer than this because they are cheaper to serve when small.

!!! question "Exercise 9"

    An MoE model has 2B parameters of attention and embeddings shared by every token, plus 64 experts totalling 20B parameters, of which each token uses 4. What are its total and active parameter counts, and its weight memory at 4.5 bits?

??? success "Answer"

    Active \(=2+20\cdot4/64=2+1.25=3.25\)B. Total \(=2+20=22\)B. Memory \(=22\times10^{9}\cdot4.5/8\approx12.4\) GB: you pay memory for 22B and compute for 3.25B.

!!! question "Exercise 10"

    A ticket classifier's first token comes back as "billing" with logprob \(-0.22\); the runner-up "bug" has \(-1.83\). Your policy auto-routes at probability 0.9 or above. What happens, and what should you check before trusting the threshold?

??? success "Answer"

    \(e^{-0.22}\approx0.80\) and \(e^{-1.83}\approx0.16\). 0.80 is below 0.9, so the ticket goes to a human. Before trusting 0.9, check calibration on labelled tickets (do 0.9-confidence answers really come out right 90% of the time?), check that "billing" is a single token for this tokenizer, and make sure thinking is off so the first token is the label.

---

Next: [50 — Prompting and APIs](50-ai-prompting-apis.md).
