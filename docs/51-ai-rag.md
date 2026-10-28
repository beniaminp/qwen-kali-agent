# 51 — Embeddings, vector search and RAG

*Last verified: 28 October 2026.*

Retrieval-augmented generation (RAG) is how you give a model knowledge it was never trained on: you search a corpus for the few passages that answer the question, paste them into the prompt, and ask the model to answer *from those passages only*. This chapter builds the whole pipeline from the arithmetic up — embeddings, chunking, exact and approximate search, lexical BM25, hybrid fusion, reranking, the prompt, access control, retrieval metrics, and a complete mini-RAG you can run on a laptop with no model at all. Read [46 — The math you need](46-ai-math.md) for vectors and cosine first if they are rusty, [49 — How LLMs work](49-ai-llm-internals.md) for tokens and context windows, and [50 — Prompting and APIs](50-ai-prompting-apis.md) for the chat request this chapter fills. Evaluation of the end-to-end system lives in [54 — Evaluation](54-ai-evaluation.md); when retrieval becomes a tool a model calls in a loop, see [52 — Agents](52-ai-agents.md).

---

## 1. Why RAG, and RAG versus fine-tuning

### 1.1 What RAG is

RAG has two phases. **Indexing** (offline): split your documents into chunks, turn each chunk into a vector with an embedding model, and store the vectors. **Retrieval and generation** (per request): embed the user's question, find the nearest chunks, format them into the prompt, and let the model answer. The model's weights never change; you change what you put in front of it.

The software analogy: RAG is a cache lookup followed by a template render. The embedding index is the cache key space, nearest-neighbour search is the lookup, and the prompt is the template. Fine-tuning, by contrast, recompiles the binary.

### 1.2 Four reasons to retrieve

You reach for RAG when one of these is true:

- **Knowledge the model lacks.** Your internal wiki, this quarter's contracts, a product manual. The model was never trained on it and cannot guess it.
- **Freshness.** Prices, inventory, today's incident status. A model has a knowledge cutoff; an index you re-run nightly does not.
- **Citations.** You must show *where* an answer came from. Retrieval hands you the source passage, so the answer can point at it. A fine-tuned model that "knows" a fact cannot cite it.
- **Access control.** Different users may see different documents. You can filter the index per user before retrieval; you cannot filter what a model memorised in its weights.

### 1.3 RAG versus fine-tuning: a decision

They answer different questions. **Fine-tuning changes behaviour** — tone, format, how the model uses tools, a skill it performs the same way every time. **RAG changes knowledge** — the facts available at answer time. They compose: fine-tune a model to follow your citation format and answer only from context, then feed it retrieved passages.

| Need | Reach for |
|---|---|
| "Answer from our 40,000-page handbook, with citations" | RAG |
| "Always reply as a terse JSON tool call" | fine-tune |
| "Use last night's prices" | RAG (re-index) |
| "Match our house writing style" | fine-tune |
| "Per-customer documents, strict isolation" | RAG with metadata filters |
| "A niche skill the base model fails at, static domain" | fine-tune (or both) |

A worked decision. A support team wants a bot that answers from a help centre that changes weekly and must cite the article it used. Knowledge the base model lacks, freshness, citations — three of the four reasons. **Start with RAG.** If, after retrieval is good, the model still rambles or ignores the "say you don't know" rule, fine-tune it on a few hundred examples of the format you want (see [53 — Fine-tuning](53-ai-fine-tuning.md)). Do not fine-tune *first*: you would bake in this week's articles and have to retrain next week.

---

## 2. Embeddings in practice

### 2.1 What an embedding model is

An embedding model is a neural network that maps a piece of text to a fixed-length list of numbers — a vector — so that texts with similar meaning land near each other. "Refund my order" and "I want my money back" share no words but point in nearly the same direction. That is the whole trick: meaning becomes geometry, and "find related text" becomes "find nearby vectors".

It is a different model from the chat model, usually much smaller, and it is trained with a contrastive objective: pull paraphrases together, push unrelated pairs apart. You do not need to train one; you call it like any other endpoint and get vectors back.

### 2.2 Dimensions and what they cost

The vector length \(d\) (its *dimension*) is fixed by the model: common sizes are 384, 768, 1024, 1536, 3072. More dimensions can carry more nuance but cost proportionally more memory and compute. Memory for a raw `float32` index is exactly

\[
\text{bytes}=N\cdot d\cdot 4
\]

!!! example "On paper"

    One million chunks at \(d=768\), four bytes per number:

    \[
    10^{6}\cdot 768\cdot 4 = 3.072\times10^{9}
    \]

    That is 3.072 GB just for the vectors, before any index overhead. Halve \(d\) and you halve the bill; §5.4 cuts it by 32x with quantization.

Some models support **Matryoshka** embeddings: you can truncate the vector to a shorter prefix and keep most of the quality, trading accuracy for memory without re-embedding. Treat that as a tuning knob, not a default.

### 2.3 Normalization

A vector has a direction and a length. For semantic search you almost always care about direction only, so you **normalize** each vector to unit length (divide by its norm). After normalization, the dot product of two vectors *is* their cosine similarity, which removes a division from every comparison. Many embedding endpoints return pre-normalized vectors; check, because mixing normalized and raw vectors silently corrupts your scores.

### 2.4 Dot product, cosine and L2 — and when they agree

Three ways to score "how close are these two vectors":

\[
\cos(\mathbf a,\mathbf b)=\frac{\mathbf a\cdot\mathbf b}{\lVert\mathbf a\rVert\,\lVert\mathbf b\rVert}
\]

!!! example "On paper"

    Two 3-number embeddings, a = (1, 2, 2) for "refund my order" and b = (2, 1, 2) for "money back please":

    \[
    \begin{aligned}
    \mathbf a\cdot\mathbf b &= 2+2+4 = 8\\
    \lVert\mathbf a\rVert &= \sqrt{1+4+4} = 3\\
    \lVert\mathbf b\rVert &= 3\\
    \cos &= 8/9 \approx 0.889
    \end{aligned}
    \]

    0.889 is close to 1: the two requests point the same way, so a search for one finds the other.

**L2** (Euclidean) distance measures straight-line gap: \(\lVert\mathbf a-\mathbf b\rVert\). For **unit-length** vectors it is tied to cosine exactly:

\[
\lVert\mathbf a-\mathbf b\rVert^{2}=2-2\cos(\mathbf a,\mathbf b)
\]

!!! example "On paper"

    Normalize a and b above (divide each by 3), so both are unit length. Their dot product is now the cosine, 8/9:

    \[
    \lVert\hat{\mathbf a}-\hat{\mathbf b}\rVert^{2}=2-2\cdot\tfrac89=\tfrac29
    \]

    So the L2 distance is \(\sqrt{2/9}\approx0.471\). Smaller L2 means larger cosine: on unit vectors the two orderings are identical, so it does not matter which you index by.

The takeaway: **on normalized vectors, dot product, cosine and L2 all rank neighbours the same way.** They diverge only when lengths vary — then dot product rewards long vectors and cosine ignores length. Pick one convention (usually cosine, i.e. normalized dot product) and apply it everywhere.

```python
import numpy as np

a = np.array([1.0, 2.0, 2.0])      # "refund my order"
b = np.array([2.0, 1.0, 2.0])      # "money back please"
c = np.array([3.0, 0.0, 0.0])      # "track my shipment"

def report(x, y):
    dot = x @ y
    cos = dot / (np.linalg.norm(x) * np.linalg.norm(y))
    l2 = np.linalg.norm(x - y)
    return dot, cos, l2

for name, v in [("b", b), ("c", c)]:
    dot, cos, l2 = report(a, v)
    print(f"a vs {name}: dot={dot:.3f}  cos={cos:.3f}  L2={l2:.3f}")

# when the vectors are unit length, the three agree on the ranking
A = a / np.linalg.norm(a)
for name, v in [("b", b), ("c", c)]:
    u = v / np.linalg.norm(v)
    print(f"unit a vs {name}: dot={A@u:.3f}  L2^2={np.sum((A-u)**2):.3f}  "
          f"(2-2dot={2-2*A@u:.3f})")
```

```text
a vs b: dot=8.000  cos=0.889  L2=1.414
a vs c: dot=3.000  cos=0.333  L2=3.464
unit a vs b: dot=0.889  L2^2=0.222  (2-2dot=0.222)
unit a vs c: dot=0.333  L2^2=1.333  (2-2dot=1.333)
```

The two "refund/money back" requests score cosine 0.889; the unrelated "track my shipment" scores 0.333. On the unit vectors, `L2^2` equals `2 - 2·dot` to the digit, which is the identity above.

### 2.5 Embedding the query versus the document

A subtle trap: some embedding models are **asymmetric**. They are trained so that a short *query* ("reset password") lands near a longer *passage* that answers it, not near other short queries. These models ship **instruction prefixes** — you prepend something like `"query: "` to questions and `"passage: "` to documents before embedding, or, for Qwen3-Embedding, a one-line task instruction on the query only (`Instruct: <task>\nQuery:<question>`, documents as-is; §15.1). Use the wrong prefix (or none) and retrieval quality drops, because queries no longer land where the model was trained to put them; the Qwen3-Embedding model card reports a 1–5% drop on most retrieval tasks without the query instruction.

Rules: read the model card; embed documents and queries with the prefixes it specifies; embed the index and the live query with the **same model and the same normalization**. If you ever swap embedding models, you must re-embed the entire index — old and new vectors are not comparable (§15.1 shows why).

---

## 3. Chunking

You rarely embed whole documents. A 40-page manual is one vector that means "manual" and retrieves nothing precisely. You split it into **chunks** — passages small enough to be about one thing, large enough to stand alone — and embed each. Chunking is the single highest-leverage decision in a RAG system; most "the model is dumb" bugs are "the chunk did not contain the answer".

### 3.1 Fixed-size with overlap

The simplest strategy: cut every \(C\) tokens, with an **overlap** of \(o\) tokens so a sentence straddling a boundary survives in at least one chunk. The chunks start every \(C-o\) tokens (the *stride*), so a document of \(T\) tokens needs

\[
n=\Big\lceil\frac{T-C}{C-o}\Big\rceil+1
\]

!!! example "On paper"

    A 12,000-token manual, 512-token chunks, 64-token overlap. The stride is \(512-64=448\):

    \[
    n=\lceil(12000-512)/448\rceil+1
    \]

    \[
    n=\lceil 25.64\rceil+1 = 26+1 = 27
    \]

    27 chunks. Drop the overlap to 0 and the stride becomes 512, giving \(\lceil 11488/512\rceil+1=24\) chunks — fewer, but any answer split across a boundary is now lost.

Measure chunk size in **tokens**, not characters, because the embedding model and the context budget are counted in tokens (see [49](49-ai-llm-internals.md)). Count with the tokenizer of the model that reads the chunk — the embedding model's for its input limit, the chat model's for the context budget — or with `tiktoken` as an estimate (it implements OpenAI's tokenizers, so its counts differ from Qwen's); never with `len(text)`.

### 3.2 Recursive by structure

Fixed-size cuts mid-sentence. **Recursive** chunking splits on a priority list of separators — paragraphs, then sentences, then words — and only falls back to a hard cut when a piece is still too big. It respects the document's shape: headings stay with their paragraph, list items stay whole. This is the sensible default for prose and Markdown.

### 3.3 Semantic chunking

**Semantic** chunking embeds each sentence and starts a new chunk when the similarity to the running average drops past a threshold — a topic shift becomes a boundary. It produces chunks that are each about one idea, at the cost of an embedding call per sentence at index time. Use it when documents are long and topically dense and retrieval quality justifies the extra indexing cost.

### 3.4 Parent–child (small-to-big)

A tension: small chunks retrieve precisely but give the model too little context; big chunks give context but retrieve fuzzily. **Small-to-big** resolves it. You embed and search over small child chunks (a sentence or two) for precision, but when a child is retrieved you hand the model its **parent** (the surrounding paragraph or section) for context. The index points children at parent IDs; retrieval dereferences them. This is one of the most effective upgrades to a basic RAG system.

### 3.5 Metadata

Every chunk carries structured fields alongside its text: source document ID, title, section heading, URL, author, timestamp, and any access-control tags (§11). Metadata powers filtering, citation (you can name the source), freshness (you can prefer recent chunks), and debugging (you can see *which* document a bad answer came from). Store it next to the vector; a vector with no provenance is a liability.

---

## 4. The indexing pipeline

Indexing is an ETL job: **load → clean → chunk → embed → store**.

1. **Load.** Pull documents from their sources (wiki, object store, database, crawl). Record a stable source ID and a content hash for each.
2. **Clean.** Strip boilerplate (nav bars, footers), normalize whitespace, drop empty or near-empty pages, convert tables/PDF to text deliberately (bad PDF extraction is a top cause of garbage chunks). Document parsing, cleaning, de-duplication and PII handling are a discipline of their own — see [60 — Data preparation](60-ai-data-preparation.md); keep only the RAG-specific concern here, that a chunk is only as good as the text you extracted.
3. **Chunk.** Apply the strategy from §3; attach metadata to each chunk.
4. **Embed.** Batch chunks through the embedding model. Batch for throughput; retry on failure; the vectors are deterministic for a fixed model, so you can cache by `(embedding model ID, chunk hash)` — the model ID belongs in the key, because a vector cached from another model is not comparable (§2.5).
5. **Store.** Write vectors plus metadata to the vector store, keyed by a deterministic chunk ID.

**Idempotent updates** are the part people skip and regret. Derive each chunk's ID from `(source_id, chunk_index, content_hash)`. When a document changes, re-chunk it and **upsert**: insert new chunk IDs, overwrite unchanged ones in place, and **delete** the chunk IDs that no longer exist for that source. If you only insert, edits leave stale chunks behind that keep getting retrieved — the "stale index" failure of §14. A correct reindex of one document touches only that document's chunks; a nightly full rebuild is a fallback, not a design.

!!! tip

    Keep a `source_id → set(chunk_ids)` map. Reindexing a document is then: compute the new chunk set, `upsert` the intersection and additions, `delete` the removals. This makes deletions — a GDPR "delete my data" request, a retracted article — a first-class operation, not an afterthought.

---

## 5. Search: exact versus approximate

### 5.1 Exact search cost

Exact (brute-force) search compares the query vector against every stored vector. Each comparison is a dot product of \(d\) multiply-adds, so one query costs

\[
\text{work}=N\cdot d\ \text{multiply-adds}
\]

!!! example "On paper"

    One million chunks, \(d=768\):

    \[
    10^{6}\cdot 768 = 7.68\times10^{8}
    \]

    768 million multiply-adds per query (1.536 billion floating-point operations, counting the multiply and the add), and all 3.072 GB of vectors (§2.2) read from memory once. Reading is the bottleneck: at an assumed 10–100 GB/s of memory bandwidth that is 0.03–0.3 s per query. At 300,000 vectors (0.92 GB) it is under 0.1 s even at the slow end, so **exact search is the right default up to a few hundred thousand vectors**. At ten million, or at high query rates, the linear term hurts and you switch to approximate search.

Exact search is `numpy`'s `M @ q` — it is simple, always correct, and returns true nearest neighbours. Do not reach for an approximate index before you need one; the approximation has its own failure modes.

### 5.2 IVF: inverted file with cells

**IVF** clusters the vectors into \(\text{nlist}\) cells (by k-means) once, at build time. At query time you compare the query only to the vectors in the few cells nearest to it — you *probe* `nprobe` cells instead of all of them.

\[
\text{vectors scanned}\approx\text{nprobe}\cdot\frac{N}{\text{nlist}}
\]

!!! example "On paper"

    One million vectors, 1,000 cells (so ~1,000 vectors each), probing 10 cells:

    \[
    10\cdot\frac{10^{6}}{1000}=10{,}000
    \]

    You scan 10,000 vectors instead of 1,000,000 — 100x less work. The risk: if the true neighbour sits in a cell you did not probe, you miss it. Raising `nprobe` recovers recall and costs time; it is the IVF accuracy knob.

### 5.3 HNSW: a navigable graph

**HNSW** (Hierarchical Navigable Small World) builds a multi-layer graph where each vector links to its near neighbours. Search starts at a top-layer entry point and greedily walks toward the query, dropping layers as it narrows in — like a skip list for geometry. It is the default index in most vector stores because it gives high recall at low latency, at the cost of build time and memory for the links.

Two parameters dominate:

- **M** — links per node. Higher M means a denser graph: better recall, more memory (FAISS, hnswlib and pgvector allow \(2M\) links per vector on the bottom layer, where every vector lives, so roughly \(2N\cdot M\) links), slower build. Typical 16–64.
- **efSearch** — the size of the candidate list kept during a search. Higher efSearch explores more of the graph: better recall, higher latency. It is tunable per query, so you can dial recall up for hard queries without rebuilding.

The recall/latency trade-off is the whole game: raise efSearch (or M) until recall@k against exact search is acceptable, then stop — every extra point of recall costs latency. Measure it (§5.5); do not guess.

### 5.4 Product quantization: compressing the vectors

**Product quantization** (PQ) shrinks memory by replacing each vector with a short code. Split the \(d\)-dimensional vector into \(m\) sub-vectors; learn 256 representative centroids for each subspace; store, per sub-vector, the one-byte ID of its nearest centroid. A vector becomes \(m\) bytes.

\[
\text{PQ bytes}=N\cdot m,\qquad \text{raw bytes}=N\cdot d\cdot 4
\]

!!! example "On paper"

    One million vectors, \(d=768\), split into \(m=96\) sub-vectors of 8 dimensions, one byte each:

    \[
    \text{raw}=10^{6}\cdot 768\cdot 4 = 3.072\ \text{GB}
    \]

    \[
    \text{PQ}=10^{6}\cdot 96 = 96\ \text{MB}
    \]

    A 32x reduction: 3.072 GB of `float32` becomes 96 MB of codes. The cost is approximation error — PQ distances are estimates — so PQ is usually combined with IVF (IVF-PQ) and a re-ranking pass over exact vectors for the final top-k.

### 5.5 Recall@k: measuring the approximation

An approximate index trades recall for speed. **Recall@k** is the fraction of the true top-k neighbours (from exact search) that the approximate index actually returned:

\[
\text{recall@}k=\frac{|\text{approx top-}k\cap\text{exact top-}k|}{k}
\]

!!! example "On paper"

    For one query, exact search says the true top-5 chunks are {A, B, C, D, E}. The HNSW index returns {A, B, C, D, F} — it found four of the five:

    \[
    \text{recall@}5=\tfrac45=0.8
    \]

    Average this over a few hundred held-out queries. If recall@10 is 0.98 you are losing 2% of true neighbours — usually fine. If it is 0.70, raise efSearch or nprobe. Always compute it against *exact* search on the same queries, not against ground-truth relevance (that is a different metric, §12).

### 5.6 Vector stores

Neutral one-liners; verify current features against each project's docs before you commit.

| Store | What it is |
|---|---|
| **pgvector** | A PostgreSQL extension adding a `vector` type with HNSW and IVFFlat indexes; keeps vectors beside your relational data. |
| **FAISS** | A library (not a server) of exact and approximate indexes (IVF, HNSW, PQ) from Meta; you embed it in your own process. |
| **Qdrant** | A standalone vector database with payload filtering and HNSW, written in Rust. |
| **Weaviate** | A vector database with a schema, hybrid search and module integrations. |
| **Milvus** | A distributed vector database aimed at very large, sharded indexes. |
| **Chroma** | A lightweight embedded/standalone store popular for prototypes and small apps. |
| **Elasticsearch / OpenSearch** | Search engines with dense-vector (HNSW) fields alongside their mature BM25 lexical search. |

For a first system on a single machine, pgvector (if you already run Postgres) or FAISS (if you want a library) covers most needs. Reach for a dedicated distributed store when the index outgrows one node. The full comparison — type, index support, hybrid, filtering, license and scale — and a how-to-choose guide are in §18.

---

## 6. Lexical search and BM25

### 6.1 Why lexical still matters

Embeddings miss exact strings: part numbers (`SKU-4471`), error codes (`ERR_503`), rare proper nouns, code identifiers. A dense model may place `ERR_503` near "server problem" and never return the page that literally contains `ERR_503`. **Lexical search** matches words directly, and **BM25** is its standard scoring function. Keep lexical search; it is the other half of hybrid retrieval (§7).

### 6.2 BM25 by hand

BM25 scores a document for a query by summing, over the query terms, three ideas: rare terms count more (**IDF**), more occurrences help but with diminishing returns (**term-frequency saturation**, knob \(k_1\)), and long documents are penalised so they do not win by sheer length (**length normalization**, knob \(b\)).

The inverse document frequency of a term appearing in \(n\) of \(N\) documents:

\[
\text{IDF}=\ln\!\Big(1+\frac{N-n+0.5}{n+0.5}\Big)
\]

!!! example "On paper"

    A 3-document help centre. Query: "refund policy". "refund" appears in 1 document, "policy" in all 3 (N = 3):

    \[
    \text{IDF}_{\text{refund}}=\ln(1+2.5/1.5)=\ln 2.667
    \]

    \[
    \text{IDF}_{\text{refund}}\approx 0.981
    \]

    \[
    \text{IDF}_{\text{policy}}=\ln(1+0.5/3.5)=\ln 1.143
    \]

    \[
    \text{IDF}_{\text{policy}}\approx 0.134
    \]

    "refund" is rare, so it carries 7x the weight of "policy", which is everywhere and nearly useless for ranking.

The full score, with term frequency \(f_t\), document length \(|d|\) and corpus average length `avgdl`:

\[
s=\sum_t \text{IDF}_t\cdot\frac{f_t\,(k_1+1)}{f_t+k_1\,(1-b+b\,|d|/\text{avgdl})}
\]

!!! example "On paper"

    Three documents, lengths 6, 5, 4 words, so avgdl = 5. Typical values \(k_1=1.5\), \(b=0.75\) (Lucene and Elasticsearch default to \(k_1=1.2\)).

    - Doc 1 (len 6): "refund policy covers late refund requests" — refund twice, policy once.
    - Doc 2 (len 5): "shipping policy and delivery schedule" — policy once.
    - Doc 3 (len 4): "general account policy page" — policy once.

    The length factor \(k_1(1-b+b\,|d|/\text{avgdl})\) for Doc 1:

    \[
    1.5\,(0.25+0.75\cdot 6/5)=1.5\cdot 1.15
    \]

    \[
    K_1=1.725
    \]

    Doc 1 "refund" term (\(f=2\)):

    \[
    0.981\cdot\frac{2\cdot 2.5}{2+1.725}=0.981\cdot 1.342
    \]

    \[
    \approx 1.317
    \]

    Doubling to \(f=2\) only raised the fraction from 0.917 (at \(f=1\)) to 1.342 — saturation, not a doubling. Add the small "policy" term (0.123) and Doc 1 scores ≈ 1.44. Doc 2 and Doc 3 have only "policy", so they score ≈ 0.13 and ≈ 0.15 — about ten times lower. Doc 3 edges Doc 2 only because it is shorter. The rare word "refund" decides the ranking, not raw counts.

The exact arithmetic is asserted in the checker module; the point to carry away is *which knob does what*: \(k_1\) controls how fast repeats stop helping, \(b\) controls how hard long documents are penalised, and IDF makes rare query words dominate — here "refund" outweighs "policy" by a factor of 7.3.

---

## 7. Hybrid search and reciprocal rank fusion

Dense (embedding) and lexical (BM25) search fail on different queries, so run **both** and merge their rankings. The scores are not comparable (cosine lives in [-1, 1]; BM25 is unbounded), so you fuse by **rank**, not score. **Reciprocal rank fusion** (RRF) gives each document a score from its position in each list:

\[
\text{RRF}(d)=\sum_{\text{lists}}\frac{1}{k+\text{rank}(d)}
\]

!!! example "On paper"

    Standard constant \(k=60\). Three documents, their ranks in the BM25 list and the vector list:

    - Doc A: BM25 rank 1, vector rank 3
    - Doc B: BM25 rank 2, vector rank 1
    - Doc C: BM25 rank 3, vector rank 2

    \[
    \text{RRF}(A)=\tfrac{1}{61}+\tfrac{1}{63}\approx 0.03227
    \]

    \[
    \text{RRF}(B)=\tfrac{1}{62}+\tfrac{1}{61}\approx 0.03252
    \]

    \[
    \text{RRF}(C)=\tfrac{1}{63}+\tfrac{1}{62}\approx 0.03200
    \]

    B wins, then A, then C. B was never top of the lexical list but was first in the vector list and second in the other — RRF rewards documents both methods like. The constant \(k=60\) damps the influence of any single high rank, so no one list can dominate.

RRF needs no tuning and no score calibration, which is why it is the default fusion method. If you have labelled data you can instead learn weights, but start with RRF.

---

## 8. Reranking with cross-encoders

Retrieval is fast and approximate; it reads the query and each document *separately* (a **bi-encoder**: two vectors, one dot product). A **cross-encoder** reads the query and one document **together** in a single forward pass, so it can model their interaction directly — far more accurate relevance, far more expensive. You cannot run it over a million documents, but you can run it over the top few that retrieval already found.

The pattern: retrieve 100 cheaply with the bi-encoder, **rerank** the top 20 with the cross-encoder, keep the best few.

\[
\text{cross-encoder passes}=k_{\text{rerank}}
\]

!!! example "On paper"

    Rerank the top 20, each pass over a ~512-token (query + document) input:

    \[
    20\cdot 512 = 10{,}240\ \text{tokens per query}
    \]

    That is a fixed, bounded cost per query. Scoring all one million documents with the cross-encoder would be one million forward passes — impossible in real time, which is exactly why retrieval does the cheap first pass and the cross-encoder only cleans up the top of the list.

Reranking typically buys a large jump in precision@k for a small latency cost, so it is one of the highest-value additions to a RAG system once basic retrieval works. A reranker is itself a model you can self-host (e.g. a BGE reranker via `llama-server --reranking`, exposing `/v1/rerank`) — a third dedicated model on its own port, next to the chat model and the embedding model of §15.1; neither of those can do its job.

---

## 9. The RAG prompt

### 9.1 Formatting, citations and grounding

Retrieval found the chunks; now the prompt must make the model *use them and only them*. A sturdy template:

- A **system instruction**: "Answer only from the context below. Cite each fact with its bracket number. If the context does not contain the answer, say you don't know." This is the single most important line in a RAG system — it is the difference between a grounded answer and a confident hallucination.
- The **context**, each chunk numbered so the model can cite it: `[1] ...`, `[2] ...`.
- The **question**.

Numbered chunks give you **citation markers**: the model writes "You can reset it from Settings [1]", and because chunk [1] carries metadata (§3.5), you resolve [1] to a real URL. Citations are not decoration — they are how a user (and your eval in §13) checks the answer against the source.

### 9.2 Context budget math

You cannot paste everything. The context window is a fixed token budget shared by the system prompt, the chunks, the conversation and the room the model needs to *write* its answer.

\[
\text{chunks that fit}=\Big\lfloor\frac{W-S-A}{C}\Big\rfloor
\]

!!! example "On paper"

    Window \(W=8192\) tokens, system + question \(S=512\), answer reserve \(A=1024\), chunk size \(C=512\):

    \[
    (8192-512-1024)/512 = 6656/512
    \]

    \[
    =13\ \text{chunks}
    \]

    So retrieve more than 13 if you like, but only the top 13 fit — which is why reranking (§8) matters: the 13 you keep must be the 13 best, not the first 13 retrieval happened to return. Reserve \(A\) generously; an answer truncated mid-sentence because you packed in one more chunk is a worse failure than one fewer source.

---

## 10. Query transformation

The user's words are often not the best search query. Transform them before (or alongside) retrieval.

- **Query rewriting.** Use a cheap model to turn a messy question into a clean search query — expand abbreviations, fix spelling, add synonyms. "cant login app keeps crashing" → "login failure mobile app crash".
- **Multi-query.** Generate several paraphrases of the question, retrieve for each, and union (or RRF-fuse) the results. This covers vocabulary the user did not use. Costs several retrievals per question.
- **HyDE** (Hypothetical Document Embeddings). Ask the model to *write a fake answer* to the question, embed that, and retrieve with it. The hypothetical answer is written in the vocabulary of real answers, so it often matches documents better than the terse question does. It can also hallucinate a wrong direction, so use it as one query among several.
- **Conversational condensation.** In a chat, "what about the refund?" only means something given the prior turns. **Condense** the history plus the new message into one standalone question ("How do I get a refund for a duplicate charge?") before retrieving. Skip this and follow-up questions retrieve nonsense, because the pronoun has no referent in the index.

---

## 11. Metadata filters and access control

### 11.1 Filter before retrieval

Metadata filters restrict the candidate set: `source = "handbook" AND language = "en" AND updated_after = 2026-01-01`. The vital rule for anything multi-tenant: **apply access-control filters *before* nearest-neighbour search, not after.** The store must only ever consider vectors the current user is allowed to see.

### 11.2 A leak story

A company builds a RAG bot over all internal documents and adds per-user permissions by filtering the *results*: retrieve the top 10 globally, then drop the ones the user may not see. One day an engineer asks about "Q3 compensation" and gets back three of their own HR documents plus — because the forbidden documents were the closest matches and got dropped — a top-10 that silently included a snippet of a colleague's salary review that slipped through a bug in the post-filter. Even when the post-filter works, **the ranking itself leaks**: the model's latency, the "N results found" count, or a partially-streamed token can reveal that a restricted document exists and is relevant.

The fix is structural: push `allowed_viewers CONTAINS current_user` into the search query so restricted vectors are never retrieved, never scored, never in memory. A post-filter is a filter you forgot to apply; a pre-filter is the only one that cannot leak. Store the ACL tags on the chunk at index time (§3.5), and treat a retrieval that ignores them as a security incident, not a bug. See [12 — Safety and legal](12-safety-legal.md) and [56 — Safety and security](56-ai-safety.md).

---

## 12. Retrieval evaluation by hand

Before you measure the model's *answers*, measure the *retrieval* — if the right chunk was never retrieved, no prompt can save the answer. Label a set of queries with their relevant chunk IDs, run retrieval, and compute:

- **recall@k**: of all relevant chunks, how many are in the top k. "Did we find them?"
- **precision@k**: of the top k, how many are relevant. "Is the top clean?"
- **MRR** (mean reciprocal rank): \(1/\text{rank}\) of the first relevant chunk, averaged over queries. "How high is the first good hit?"
- **nDCG@k**: a ranking-quality score that rewards relevant chunks for being near the top, normalised to [0, 1].

!!! example "On paper"

    One query. Four chunks are relevant: {d2, d5, d8, d9}. Retrieval returns, in order: [d2, d1, d5, d3, d8]. Relevant hits land at ranks 1, 3 and 5.

    \[
    \text{recall@}5=3/4=0.75
    \]

    \[
    \text{precision@}5=3/5=0.60
    \]

    \[
    \text{MRR}=1/1=1.0
    \]

    recall@5 is 0.75 because one relevant chunk (d9) never appeared; precision@5 is 0.60 because two of the five returned were junk; MRR is 1.0 because the very first result was relevant.

nDCG needs discounted cumulative gain. With binary relevance (1 if relevant, else 0) and a \(\log_2(i+1)\) discount at rank \(i\):

\[
\text{DCG@}5=\sum_{i=1}^{5}\frac{\text{rel}_i}{\log_2(i+1)}
\]

!!! example "On paper"

    Relevant at ranks 1, 3, 5 (gains 1), junk at 2, 4 (gains 0):

    \[
    \text{DCG}=\tfrac{1}{\log_2 2}+\tfrac{1}{\log_2 4}+\tfrac{1}{\log_2 6}
    \]

    \[
    \text{DCG}=1+0.5+0.387=1.887
    \]

    The ideal order puts all relevant first. With four relevant chunks:

    \[
    \text{IDCG}=1+\tfrac{1}{\log_2 3}+\tfrac12+\tfrac{1}{\log_2 5}
    \]

    \[
    \text{IDCG}=1+0.631+0.5+0.431=2.562
    \]

The normalised score divides the two:

\[
\text{nDCG@}5=\frac{\text{DCG}}{\text{IDCG}}
\]

!!! example "On paper"

    \[
    \text{nDCG@}5=1.887/2.562\approx 0.737
    \]

    0.737 out of a perfect 1.0: the relevant chunks are present but not ideally ordered (one sits at rank 5), and one relevant chunk is missing entirely, so the ceiling itself is set by four relevant items.

Track recall@k as your north star for retrieval — if it is low, fix chunking, embeddings or hybrid search before touching the prompt.

---

## 13. End-to-end RAG evaluation

Good retrieval is necessary, not sufficient — the model can still ignore the context or cite the wrong chunk. Three answer-level metrics, each gradeable by a human or an LLM judge (see [54 — Evaluation](54-ai-evaluation.md)):

- **Groundedness / faithfulness.** Is every claim in the answer supported by the retrieved context? An answer that adds facts not in the context is *ungrounded* — a hallucination, even if the fact is true. Grade per claim: decompose the answer into claims, check each against the context.
- **Answer relevance.** Does the answer actually address the question, or does it answer a nearby question? A grounded but off-topic answer still fails the user.
- **Citation accuracy.** Does each citation marker point at a chunk that actually supports the claim?

\[
\text{citation accuracy}=\frac{\text{correct citations}}{\text{total citations}}
\]

!!! example "On paper"

    An answer makes four cited claims; checking each citation against its chunk, three are supported and one points at an unrelated chunk:

    \[
    \text{citation accuracy}=3/4=0.75
    \]

    0.75 means a quarter of the answer's "sources" do not back up what they are attached to — a trust-destroying bug even when the prose reads well. Measure it explicitly; readers check citations.

A useful decomposition when an answer is wrong: was the chunk **retrieved** (check §12 metrics)? If yes, was it **in the prompt** (context budget, §9.2)? If yes, did the model **use it** (groundedness)? Each "no" points at a different fix.

---

## 14. Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Answer is confidently wrong | Wrong chunk retrieved; vocabulary mismatch between query and document | Add hybrid search (§7) and reranking (§8); query rewriting (§10) |
| Answer is half-right, trails off | The answer was split across a chunk boundary | Increase overlap; recursive or small-to-big chunking (§3) |
| Answer cites last week's price | Stale index — document changed, chunks did not | Idempotent upsert + delete (§4); monitor index freshness |
| Same fact repeated three times | Duplicate chunks (same content indexed twice) | Deduplicate by content hash at index time |
| Right chunk retrieved but ignored | "Lost in the middle" — models attend less to the middle of a long context | Fewer, better chunks (rerank); put the best chunk first or last |
| Exact code/ID never found | Dense-only search cannot match rare literal strings | Add BM25 lexical search (§6), fuse with RRF |
| "I don't know" to an answerable question | recall@k too low; chunk too small to stand alone | Fix chunking; raise k; check embedding prefixes (§2.5) |

Most of these are *retrieval* bugs wearing a *generation* costume. Diagnose with the §12/§13 decomposition before you touch the prompt.

---

## 15. A complete offline mini-RAG in numpy

This runs with no model and no server — it builds a TF-IDF index over eight tiny help-centre documents, retrieves by cosine similarity, and assembles the prompt. It is the whole pipeline of §§2–9 in 40 lines, so you can see every moving part.

```python
import re
import numpy as np

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

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

vocab = sorted({t for d in DOCS for t in tokenize(d)})
col = {t: i for i, t in enumerate(vocab)}
N = len(DOCS)

df = np.zeros(len(vocab))
for d in DOCS:
    for t in set(tokenize(d)):
        df[col[t]] += 1
idf = np.log((N + 1) / (df + 1)) + 1.0          # smoothed idf

def embed(text):
    v = np.zeros(len(vocab))
    for t in tokenize(text):
        if t in col:
            v[col[t]] += 1.0
    v *= idf
    norm = np.linalg.norm(v)
    return v / norm if norm else v

M = np.vstack([embed(d) for d in DOCS])          # 8 unit rows

def search(query, k=3):
    scores = M @ embed(query)                    # cosine, both normalised
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order]

query = "how do I reset my account password"
hits = search(query, k=3)
for rank, (i, s) in enumerate(hits, 1):
    print(f"{rank}. doc{i}  cos={s:.3f}  {DOCS[i]}")
```

```text
1. doc0  cos=0.649  Reset your password from the account settings page.
2. doc2  cos=0.276  Secure your account with a strong password and a backup email.
3. doc1  cos=0.134  To change your password, open Settings and click Security.
```

The top three are the three password documents. The order shows what TF-IDF measures: doc2 ("secure your account") beats doc1 ("change your password") only because it shares one more word, "account", with the query, although doc1 is the better answer. TF-IDF matches *words*, so a query that shared no words with a relevant document would miss it — the vocabulary-mismatch failure of §14 that a real embedding model fixes. Now assemble the grounded prompt from the hits:

```python
# continues from the mini-RAG index: DOCS and the retrieved hits
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
query = "how do I reset my account password"
hits = [(0, 0.649), (2, 0.276), (1, 0.134)]      # from the previous step

SYSTEM = ("Answer only from the context. Cite each fact as [n]. "
          "If the context does not contain the answer, say you don't know.")

blocks = [f"[{n}] {DOCS[i]}" for n, (i, _) in enumerate(hits, 1)]
context = "\n".join(blocks)
prompt = f"{SYSTEM}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
print(prompt)
```

```text
Answer only from the context. Cite each fact as [n]. If the context does not contain the answer, say you don't know.

Context:
[1] Reset your password from the account settings page.
[2] Secure your account with a strong password and a backup email.
[3] To change your password, open Settings and click Security.

Question: how do I reset my account password
Answer:
```

That prompt string is what you send to the chat model. The model reads three numbered, retrieved facts and the instruction to answer only from them and cite — a real Qwen3-14B would reply something like "Open the account settings page and reset your password [1]."

### 15.1 The same retrieval against an embedding model

Swap the hand-rolled TF-IDF for a real embedding model. It is a **separate model on a separate server**. The chat model on port 8080 cannot do this job: `mlx_lm.server` ([10 — Export and serving](10-export-serving.md)) serves chat and text completions and a model list, but no `/v1/embeddings` route, and a chat model is trained to predict the next token, not to place paraphrases near each other (§2.1). Serve a dedicated embedding model instead. Here that is Qwen3-Embedding-0.6B (1024 dimensions, 32K-token input), as a GGUF from the official `Qwen/Qwen3-Embedding-0.6B-GGUF` repository, through llama.cpp's `llama-server`, on its own port:

```bash
llama-server -hf Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 \
  --embedding --pooling last -ub 8192 \
  --host 127.0.0.1 --port 8081 --alias qwen3-embedding-0.6b-q8_0
```

- `-hf repo:Q8_0` downloads `Qwen3-Embedding-0.6B-Q8_0.gguf` (about 0.64 GB, small next to the chat model on a 32 GB Mac) from that repository; or download the file yourself and pass `-m`.
- `--embedding` restricts the server to embeddings; llama.cpp documents it for dedicated embedding models only. It serves the OpenAI-compatible `/v1/embeddings` and returns unit-length (Euclidean-normalized) vectors.
- `--pooling last` takes the hidden state of the last token as the text's vector, which is how Qwen3-Embedding pools (its model card's reference code does the same); `-ub 8192` raises the micro-batch from its default of 512 tokens. Both flags come from the model card.
- `--port 8081` keeps the chat server on 8080 untouched (pass a port explicitly; see [10](10-export-serving.md) for the default). `--alias` sets the model name the API reports, which the code below sends.

The retrieval code is the same as before; only `embed` becomes a network call:

```python
import numpy as np
from openai import OpenAI

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
# the embedding server on :8081, not the chat server on :8080
emb = OpenAI(base_url="http://127.0.0.1:8081/v1", api_key="local")
EMBED_MODEL = "qwen3-embedding-0.6b-q8_0"       # the --alias above
TASK = "Given a question, retrieve help-centre passages that answer it"

def embed(texts):
    resp = emb.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data])

# index time: documents as-is; store the model name with the vectors
index = {"model": EMBED_MODEL, "vectors": embed(DOCS)}

# query time: same model (checked), query side gets the instruction
assert index["model"] == EMBED_MODEL, "re-embed the index first"
q = embed([f"Instruct: {TASK}\nQuery:how do I reset my account password"])[0]
M = index["vectors"]
scores = M @ q / (np.linalg.norm(M, axis=1) * np.linalg.norm(q))
print("dims:", M.shape[1])
for rank, i in enumerate(np.argsort(-scores)[:3], 1):
    print(f"{rank}. doc{i}  cos={scores[i]:.3f}  {DOCS[i]}")
```

Run against a local **stub** of that server, which returns 1024-dimensional hashed word counts (no model needed to test the wiring):

```text
dims: 1024
1. doc0  cos=0.401  Reset your password from the account settings page.
2. doc2  cos=0.210  Secure your account with a strong password and a backup email.
3. doc1  cos=0.126  To change your password, open Settings and click Security.
```

The stub ignores the instruction line and its vectors only count shared words, so these scores are plain word overlap: doc0 shares 3 of its 8 distinct words with the 7-word query, \(3/\sqrt{8\cdot 7}\approx0.401\). The real model returns different numbers. It compares meaning, so doc1 ("change your password"), which shares only one word with the query, can score close to doc0, and unrelated passages rarely score exactly 0. The stub proves the API round trip and the cosine math; the ranking quality comes from the model. The division by the norms is a no-op here, because `llama-server` already returns unit vectors, but it keeps the code correct against a server that does not (§2.3).

**Why the index and the query must use the same embedding model.** Each embedding model learns its own coordinate system. Coordinate 17 of a Qwen3-Embedding vector has no counterpart in another model's vector, and two models with the same dimension still put "reset password" at unrelated points. A cosine between a vector from model A and a vector from model B is still a number between -1 and 1, so nothing crashes, but it measures nothing: retrieval silently returns junk. A dimension check catches some mismatches (Qwen3-Embedding-4B returns 2560 numbers, not 1024), but not two models of the same size. So store the model name, and the exact file (Q8_0 and F16 vectors differ slightly), with the index, and refuse to query an index built by a different model. In production, indexing and querying are separate processes days apart, so the stored name is what catches the mix-up — the `assert` in the code is that check in miniature. Because `--alias` is a name you choose, put the model and the quantization in it, as `qwen3-embedding-0.6b-q8_0` does (or store the file's hash), so a swapped file cannot hide behind an old alias. Changing the embedding model means re-embedding every chunk (§2.5). The same rule covers the prefixes: embed documents with none (for Qwen3-Embedding), and give every live query the same `Instruct:` line you used when you measured recall.

---

## 16. Production concerns

- **Freshness and reindexing.** Decide a staleness budget (minutes? a day?) and build the pipeline to meet it: event-driven upserts on document change, with a nightly full rebuild as a safety net. Track the oldest chunk's age as a metric; a creeping maximum age means the incremental path is broken.
- **Cost per query.** A RAG query costs an embedding call for the question, the retrieval, and the generation. Generation dominates. With a context of 13 chunks × 512 = 6,656 input tokens and ~500 output tokens, at assumed prices of $3 per million input and $15 per million output tokens: \(6656/10^{6}\cdot 3 + 500/10^{6}\cdot 15 = \$0.0275\) per query. Multiply by your query volume before you choose how many chunks to pack — every extra chunk is recurring spend. (Prices are an assumption for the arithmetic; see [55 — Production](55-ai-production.md) for the real cost model.)
- **Caching.** Cache embeddings by `(embedding model ID, chunk hash)` (they are deterministic per model, and useless to any other model) so reindexing an unchanged document is free. Cache whole answers for identical questions; cache retrieval results for near-identical questions with a short TTL. Exact-match and semantic caching are covered in [55](55-ai-production.md).
- **Monitoring retrieval quality.** Log, per query: the retrieved chunk IDs and scores, whether the model said "I don't know", and any user thumbs-up/down. Sample these weekly into the labelled set from §12 and recompute recall@k — retrieval quality drifts as the corpus grows and the query mix changes, and the only way to catch it is to keep measuring.

---

## 17. RAG architectures

The components above assemble into a handful of named patterns. They form a ladder: start at the bottom and climb only when a measured failure forces you to. Each step adds capability and cost. (Costs below assume $3 per million input tokens for the arithmetic; see [55 — Production](55-ai-production.md) for the real model.)

### 17.1 Naive RAG

One retrieval, one generation — the pipeline of §15.

```text
query ──► embed ──► top-k search ──► stuff context ──► LLM ──► answer
```

- **When to use:** a single, clean corpus; direct fact-lookup questions; a prototype.
- **Strengths:** simple, cheap, one LLM call, easy to debug.
- **Weaknesses:** no defence against bad queries, wrong chunks, multi-hop questions, or vocabulary mismatch.
- **Cost/latency:** one retrieval (~milliseconds) + one generation over ~4K context tokens ≈ \(4000/10^{6}\cdot 3=\$0.012\) per query.

### 17.2 Advanced RAG

Naive RAG with a pre-retrieval stage (query rewriting, HyDE — §10), hybrid retrieval (dense + BM25 fused by RRF — §7), and a post-retrieval stage (reranking — §8; small-to-big — §3.4).

```text
query ─► rewrite ─► [dense + BM25] ─► RRF fuse ─► rerank ─► small→big ─► LLM
```

- **When to use:** any serious single-corpus production system. This is the sensible default target.
- **Strengths:** much higher recall and precision; robust to messy queries and rare literals.
- **Weaknesses:** more moving parts; added latency from the rewrite call and the reranker.
- **Cost/latency:** on top of naive RAG, add a rewrite call (one more LLM round trip, so hundreds of milliseconds or more) and a reranker pass over ~10K tokens (§8). The reranker is a small model, usually self-hosted, so those tokens cost far less than chat-model tokens; billed at the assumed chat rate of $3 per million they would be \(10240/10^{6}\cdot 3\approx\$0.031\), more than naive RAG's whole $0.012 — which is why rerankers are small.

### 17.3 Modular RAG

A pipeline of swappable modules with a **router** that sends each query down the right path: a classifier picks a retriever (this corpus vs that one, vector vs SQL vs web), multiple retrievers run, and their results fuse.

```text
query ─► router ─┬─► corpus-A retriever ─┐
                 ├─► corpus-B retriever ─┼─► fuse ─► LLM ─► answer
                 └─► SQL / web tool ──────┘
```

- **When to use:** several data sources or query types that need different handling.
- **Strengths:** flexible; each module is independently testable and replaceable.
- **Weaknesses:** orchestration complexity and more to operate and monitor.
- **Cost/latency:** the router adds a cheap classification call; cost then depends on how many retrievers fan out.

### 17.4 GraphRAG (knowledge-graph RAG)

Build a knowledge graph from the corpus — extract entities and relations, cluster them into communities, summarise each — then retrieve subgraphs or community summaries instead of (or alongside) chunks.

```text
index:  docs ─► extract entities+relations ─► graph ─► community summaries
query:  question ─► find entities ─► gather subgraph/summaries ─► LLM
```

- **When to use:** multi-hop questions ("what connects X and Y?") and global questions ("what are the main themes across all documents?") that no single chunk answers.
- **Strengths:** captures relationships and whole-corpus structure that flat chunks lose.
- **Weaknesses:** expensive to build — roughly one extraction LLM call per chunk, so a 50,000-chunk corpus is ~50,000 calls — and the graph goes stale as documents change.
- **Cost/latency:** high one-time construction cost; query cost similar to advanced RAG once built.

### 17.5 Agentic RAG

Retrieval becomes a **tool** a model calls in a loop: retrieve, judge whether the result is sufficient, re-query or try a different source, and stop when it can answer (an iterative retrieve–reflect loop).

```text
┌─────────────────────────────────────────────┐
│ LLM ─► retrieve(tool) ─► reflect: enough?     │
│   ▲                          │ no             │
│   └──────── refine query ◄────┘               │
└─────────────── yes ─► answer ─────────────────┘
```

- **When to use:** complex or multi-step questions where one retrieval is not enough; questions that need to decompose into sub-questions.
- **Strengths:** adapts the search to what it finds; handles multi-hop.
- **Weaknesses:** cost and latency multiply per iteration; can loop or over-retrieve; harder to evaluate (judge the trajectory, not just the answer).
- **Cost/latency:** \(k\) iterations means roughly \(k\) times the LLM calls of naive RAG — four iterations is ~4x the per-query cost. See [52 — Agents](52-ai-agents.md) for the loop and [61 — Agent frameworks](61-ai-agent-frameworks.md) for the tooling.

### 17.6 Multimodal RAG

Index non-text content — images, tables, charts — either by embedding them with a multimodal model or by generating a text description (caption) and embedding that. Retrieval then spans modalities.

```text
index:  [text | image | table] ─► (embed or caption) ─► one index
query:  question ─► retrieve across modalities ─► LLM (sees text + images)
```

- **When to use:** documents whose answers live in figures, screenshots, scanned tables, or diagrams.
- **Strengths:** answers questions text-only RAG cannot.
- **Weaknesses:** multimodal embedding and table extraction are error-prone; parsing quality dominates (see [60 — Data preparation](60-ai-data-preparation.md)).
- **Cost/latency:** image tokens are expensive in the generation step; budget them like a large chunk.

### 17.7 Long context instead of RAG

Skip retrieval: put the whole corpus into a long context window and let the model read everything.

```text
query + [entire corpus, e.g. 200K tokens] ─► LLM ─► answer
```

- **When to use:** a corpus small enough to fit the window, low query volume, or questions that need whole-document reasoning no chunking preserves.
- **Strengths:** no retrieval failures (nothing is left out); the model sees full context.
- **Weaknesses:** cost and latency scale with the context; "lost in the middle" (§14) means the model under-uses the middle; weak citations; does not scale to a large or growing corpus.

\[
\frac{\text{long-context cost}}{\text{RAG cost}}=\frac{200000}{4000}
\]

!!! example "On paper"

    Stuffing 200K tokens vs retrieving 4K, at $3 per million input tokens:

    \[
    \text{long context}=200000/10^{6}\cdot 3=\$0.60
    \]

    \[
    \text{RAG}=4000/10^{6}\cdot 3=\$0.012
    \]

    \[
    \text{ratio}=0.60/0.012=50
    \]

    Long context costs 50x as much per query, and prefill latency is at least ~50x longer: the work of the linear layers is proportional to the token count, so it alone grows 50x, and attention's work grows with the square of the length on top of that. For one-off analysis of one document, pay it; for a high-traffic bot over a large corpus, retrieve. A common middle path does both: retrieve generously (say 100 chunks of 512 tokens, about 51K tokens) and let a long window read them all.

Climb the ladder on evidence: measure where naive RAG fails (§§12–13), and add exactly the stage that fixes that failure. Do not start at agentic GraphRAG because it sounds thorough — you will pay its cost and inherit its failure modes for a problem naive RAG may already solve.

## 18. Choosing a vector database

The §5.6 one-liners, expanded into the axes that actually decide the choice. Licenses and features move quickly; the table reflects October 2026, and you should confirm each project's current LICENSE file and docs before committing.

| Store | Type | Index types | Hybrid (lexical+vector) | Metadata filter | License | Scale sweet spot |
|---|---|---|---|---|---|---|
| **pgvector** | Postgres extension | HNSW, IVFFlat | via Postgres full-text + fusion | SQL `WHERE` | PostgreSQL License | up to a few million, already on Postgres |
| **FAISS** | library (in-process) | Flat, IVF, HNSW, PQ | no (build yourself) | no (build yourself) | MIT | any size you can host; you own the serving |
| **Qdrant** | server (Rust) | HNSW | yes | rich payload filters | Apache-2.0 | millions–billions, filtered search |
| **Weaviate** | server (Go) | HNSW | yes (BM25 + vector) | yes | BSD-3-Clause | millions, schema-driven apps |
| **Milvus** | distributed server | IVF, HNSW, DiskANN, more | yes | yes | Apache-2.0 | very large, sharded, billions |
| **Chroma** | embedded / server (Rust core since 1.0; Python and JS clients) | HNSW | limited | yes | Apache-2.0 | prototypes, small apps |
| **Pinecone** | managed (SaaS) | proprietary (serverless) | yes (dense + sparse) | yes | proprietary | any; you pay to skip ops |
| **Elasticsearch / OpenSearch** | server | HNSW (+ BM25) | yes (native) | yes | ES: SSPL/ELv2/AGPLv3; OpenSearch: Apache-2.0 | large; teams already running it for search |
| **LanceDB** | embedded (on Lance format) | IVF, HNSW | full-text + vector | SQL filter | Apache-2.0 | local / medium, disk-backed |
| **Redis** | server (in-memory) | FLAT, HNSW | yes (filtered KNN, `FT.HYBRID`) | yes | RSALv2 / SSPLv1 / AGPLv3 | low-latency, moderate size |

### 18.1 Selection criteria

- **Scale.** Thousands to a few million: pgvector, Chroma, LanceDB, or FAISS in-process. Many millions to billions, sharded: Milvus, Qdrant, or a managed service.
- **Metadata filtering.** If you filter heavily (per-tenant, per-date, per-category), favour a store with *filtered* ANN (Qdrant, Milvus, Weaviate, Elasticsearch) that applies the filter inside the search rather than after it — the post-filter leak and recall loss of §11 is a store-level concern too.
- **Hybrid.** If exact strings matter (§6), pick a store with native lexical+vector (Elasticsearch/OpenSearch, Weaviate, Qdrant, Redis) or pair FAISS/pgvector with a separate BM25 index and fuse with RRF yourself.
- **Multi-tenancy.** Strict per-user isolation (§11) is easier where the store supports partitions/namespaces and pre-filtering as first-class features.
- **Ops burden.** A library (FAISS) or an extension on a database you already run (pgvector) adds no new service; a standalone server (Qdrant, Milvus, Weaviate) is one more thing to deploy, back up and monitor; a managed service (Pinecone) trades money for zero ops.
- **Cost.** Self-hosted cost is mostly RAM (HNSW keeps the graph in memory); managed cost is per-vector and per-query. Quantization (§5.4) and disk-backed indexes (DiskANN, LanceDB) trade accuracy or latency for a smaller bill.

### 18.2 A sizing example

!!! example "On paper"

    Five million chunks at \(d=1024\), `float32`, HNSW in RAM:

    \[
    5\times10^{6}\cdot 1024\cdot 4 = 20.48\ \text{GB}
    \]

    20 GB of vectors (plus graph links, so budget ~30 GB RAM) fits one 64 GB server — pgvector or Qdrant on a single node. Product-quantize instead:

    \[
    5\times10^{6}\cdot 128 = 640\ \text{MB}
    \]

    PQ codes drop it to 640 MB (32x), so the whole index fits in memory on a laptop, at some recall cost. The sizing — not brand preference — picks the store: raw at a few million fits one node; past that, shard (Milvus) or quantize.

Default advice: already on Postgres and under a few million vectors — **pgvector**. Want a library you control — **FAISS**. Need filtered, hybrid search at scale without managing shards — **Qdrant** or **Weaviate**. Want zero ops — a managed service. Reviewing orchestration frameworks (LangChain, LlamaIndex, Haystack) that wrap these stores is out of scope here; see [61 — Agent frameworks](61-ai-agent-frameworks.md).

## 19. Key takeaways

- RAG supplies *knowledge* (facts at answer time, with citations and access control); fine-tuning supplies *behaviour*. They compose; choose by which one your problem needs.
- Normalize embeddings; then dot product, cosine and L2 rank neighbours identically. Embed queries and documents with the same dedicated embedding model (not the chat model; it runs as its own server), record its name with the index, and honour asymmetric models' instruction prefixes.
- Chunking is the highest-leverage decision. Measure chunk size in tokens, overlap to survive boundaries, and reach for recursive or small-to-big before blaming the model.
- Exact search is a `numpy` matmul and the right default up to a few hundred thousand vectors. Scale with IVF (probe fewer cells), HNSW (navigable graph, tune efSearch), and PQ (32x memory cut). Always measure recall@k against exact search.
- Keep lexical BM25 for exact strings and fuse it with dense search by reciprocal rank fusion; rerank the top few with a cross-encoder for a large precision gain at bounded cost.
- The system prompt "answer only from context, cite, say you don't know" is load-bearing. Budget context in tokens and reserve room for the answer.
- Filter access control *before* retrieval — a post-filter leaks. Evaluate retrieval (recall@k, nDCG) and answers (groundedness, citation accuracy) separately; most "dumb model" bugs are retrieval bugs.
- Climb the architecture ladder on evidence: naive → advanced (rewrite, hybrid, rerank, small-to-big) → modular → GraphRAG / agentic / multimodal, or long context when the corpus fits and traffic is low. Each step adds cost; add only the stage a measured failure demands.
- Choose the vector store by scale, filtering, hybrid, multi-tenancy, ops burden and cost — not brand. pgvector or FAISS to a few million; Qdrant/Weaviate/Milvus for filtered hybrid at scale; a managed service to skip ops.

---

## 20. Exercises

!!! question "Exercise 1"

    An embedding model returns 1024-dimensional `float32` vectors. You have 2 million chunks. How much memory does the raw vector index need, and what does product quantization into 128 one-byte codes per vector reduce it to?

??? success "Answer"

    Raw: \(2\times10^{6}\cdot 1024\cdot 4 = 8.192\times10^{9}\) bytes = 8.192 GB. PQ: \(2\times10^{6}\cdot 128 = 2.56\times10^{8}\) bytes = 256 MB. The ratio is \(1024\cdot4/128 = 32\)x, the same compression factor as §5.4 because it is set by (dimensions × 4 bytes) ÷ (codes), independent of \(N\).

!!! question "Exercise 2"

    A 30,000-token document is chunked at 1,000 tokens with 200-token overlap. How many chunks?

??? success "Answer"

    Stride \(= 1000 - 200 = 800\). \(n=\lceil(30000-1000)/800\rceil+1 = \lceil 36.25\rceil + 1 = 37 + 1 = 38\) chunks.

!!! question "Exercise 3"

    An IVF index has 4,000 cells over 4 million vectors. You probe 20 cells. How many vectors are scanned, and what is the speedup over exact search?

??? success "Answer"

    Vectors per cell \(\approx 4\times10^{6}/4000 = 1000\). Scanned \(= 20\cdot 1000 = 20{,}000\). Speedup \(= 4\times10^{6}/20{,}000 = 200\)x. Raising probes to 40 doubles the work and recovers recall; it is the accuracy/latency knob.

!!! question "Exercise 4"

    For a query, exact search's true top-10 is a known set. Your HNSW index returns 9 of those 10 in its top-10. What is recall@10? If you raise efSearch and now return all 10, what did it cost?

??? success "Answer"

    recall@10 \(= 9/10 = 0.90\). Raising efSearch to reach 1.0 costs latency: the search keeps a larger candidate list and visits more of the graph. You stop raising it once recall is acceptable, because each extra point costs time.

!!! question "Exercise 5"

    Two retrievers rank a document at position 1 (lexical) and position 5 (vector). Using RRF with \(k=60\), what is its fused score? A second document is ranked 4 and 2. Which ranks higher?

??? success "Answer"

    Doc 1: \(1/61 + 1/65 = 0.016393 + 0.015385 = 0.031778\). Doc 2: \(1/64 + 1/62 = 0.015625 + 0.016129 = 0.031754\). Doc 1 ranks higher, just barely — the rank-1 lexical hit edges out the more balanced second document. RRF rewards a strong placement in *either* list.

!!! question "Exercise 6"

    A retrieval returns, for one query, [d1, d2, d3, d4, d5]. The relevant chunks are {d2, d3}. Compute recall@5, precision@5 and MRR.

??? success "Answer"

    Both relevant chunks are in the top 5, so recall@5 \(= 2/2 = 1.0\). Of 5 returned, 2 are relevant, so precision@5 \(= 2/5 = 0.4\). The first relevant chunk is d2 at rank 2, so MRR \(= 1/2 = 0.5\).

!!! question "Exercise 7"

    Why must per-user access control be applied *before* nearest-neighbour search rather than by filtering the results afterwards? Give one way a post-filter leaks even when it drops every forbidden document.

??? success "Answer"

    A post-filter scores and ranks documents the user may not see, then removes them — but the *existence and relevance* of a restricted document can leak through timing, a "N results" count, or a partially-streamed token, and any bug in the filter returns the document itself. A pre-filter makes restricted vectors un-retrievable, so they are never scored or in memory. The ranking itself is the side channel.

!!! question "Exercise 8"

    A RAG answer contains five claims. Checking each against the retrieved context: four are supported, one is not in the context at all (though it happens to be true). What is the groundedness, and why is the true-but-unsupported claim still a failure?

??? success "Answer"

    Groundedness \(= 4/5 = 0.80\). The unsupported claim is a failure because RAG's contract is "answer from the context": a fact the model added from its own weights cannot be cited, cannot be verified by the user, and may be false for the *next* question where the model's memory is wrong. Ungrounded-but-true today is unverifiable-and-maybe-false tomorrow.

---

Next: [52 — Agents: model plus tools plus a loop](52-ai-agents.md).
