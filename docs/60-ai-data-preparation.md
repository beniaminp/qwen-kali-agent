# 60 — Data preparation for AI systems

*Last verified: 28 October 2026.*

Most of the time you spend building an AI feature is not spent on the model. It is spent getting data into a shape the model can learn from, retrieve from, or be judged against. This chapter is the engineering discipline of that work, across the whole lifecycle: where data comes from, how to parse the messy formats it arrives in, how to clean, deduplicate, filter, protect, label, synthesize, format, split, validate, version and document it, and how to monitor the data a live system produces. Read [50 — Prompting and APIs](50-ai-prompting-apis.md) first for how requests and tokens work. The chapters that own specific slices — [47 — Classical ML](47-ai-classical-ml.md) for splits, leakage and feature engineering, [51 — RAG](51-ai-rag.md) for chunking and indexing, [53 — Fine-tuning](53-ai-fine-tuning.md) for loss masking and packing, [54 — Evaluation](54-ai-evaluation.md) for golden sets and agreement, [56 — Safety](56-ai-safety.md) for PII redaction and privacy law — are linked where they take over. This repository's hands-on data chapters ([04 — Data strategy](04-data-strategy.md), [05 — Datasets](05-datasets.md), [06 — Data format](06-data-format.md), [07 — Synthetic data](07-synthetic-data.md)) are the worked instances of the principles here.

---

## 1. Why data work dominates AI engineering

### 1.1 Garbage in, garbage out

A model has no opinion about your data. It fits what you show it, retrieves what you indexed, and is scored on whatever you put in the eval set. If the input is wrong, stale, duplicated, mislabelled or contaminated, the model faithfully reproduces that defect — often amplifying it. "Garbage in, garbage out" is not a cliché here; it is the single most reliable predictor of whether an AI feature works. The uncomfortable corollary: when a model underperforms, the data is the prior suspect, not the architecture.

The software-engineering analogy is the test fixture. You can write a perfect function, but if the fixture it is tested against is wrong, every green check lies. Data is the fixture, the training set and the regression suite all at once.

### 1.2 Three stories

**The RAG bot that cited an outdated policy.** A support assistant retrieved from a folder of PDFs. One PDF was the 2023 refund policy, superseded but never deleted; the 2026 policy lived in a wiki the crawler missed. The bot confidently quoted "30 days" when the real window was 14. Nothing was wrong with the model or the retrieval math — the index contained a stale document and lacked an `updated_at` to prefer the current one. The fix was in data preparation: source the authoritative copy, record freshness metadata, and delete superseded chunks on reindex ([51 §4](51-ai-rag.md)).

**The fine-tune that learned signatures.** A team fine-tuned a model to draft replies from a corpus of real agent emails. They forgot to strip quoted history and signatures. The model learned that every answer ends with "Best regards,\nJordan\nSupport Team | ext. 4471" and began inventing names and extension numbers. The behaviour was learned perfectly from boilerplate that should never have been in the training data. The fix: parse the email thread down to the new message before training (§4.5).

**The eval set that leaked into training.** A team built a 500-example golden set, then later scraped "more training data" from the same internal tickets — including the 500 tickets the golden set was drawn from. Eval accuracy jumped to 94%. Production accuracy did not move. The model had memorised the answers. The gain was contamination, not skill, and it hid a real regression. The fix: decontaminate training data against the eval set by n-gram overlap and near-duplicate clustering *before* every run (§12, [54 §14](54-ai-evaluation.md)).

Each story is a data-preparation failure wearing a model-failure mask. That is why this work is the job.

---

## 2. The data lifecycle

Data moves through stages, each with a defined input and output. Drawing the pipeline and naming each stage's output is the fastest way to find where quality was lost.

```text
 source ─► ingest ─► parse ─► clean/normalize ─► deduplicate ─► filter quality
                                                                       │
   ┌───────────────────────────────────────────────────────────────── ┘
   ▼
 protect (PII/secrets/licence) ─► label/annotate ─► augment/synthesize
                                                                       │
   ┌───────────────────────────────────────────────────────────────── ┘
   ▼
 format ─► split ─► validate ─► version ─► document ─► monitor ──┐
   ▲                                                             │
   └──────────────── feedback from production ───────────────────┘
```

| Stage | Input | Output |
|---|---|---|
| Source | a need | a list of candidate sources with owners and licences |
| Ingest | sources | raw bytes/files landed in one place with provenance |
| Parse | raw files | plain text (plus structure: headings, tables) and metadata |
| Clean / normalize | text | canonical Unicode, no boilerplate, consistent labels |
| Deduplicate | clean records | unique records (exact + near) |
| Filter quality | unique records | records above a quality bar |
| Protect | records | records with PII/secrets removed and licence recorded |
| Label / annotate | records | records with trusted labels + an agreement estimate |
| Augment / synthesize | records | records plus verified synthetic ones |
| Format | records | the exact shape the consumer wants (chat JSONL, chunks, table) |
| Split | records | train / valid / test (or index / holdout) with no leakage |
| Validate | splits | splits that pass a schema and statistical contract |
| Version | validated splits | an immutable, hashed, named snapshot |
| Document | a snapshot | a data card describing it |
| Monitor | production traffic | drift alerts and new labelled examples |

The order is not sacred — you may dedup before or after cleaning — but every stage should produce something you can inspect. A pipeline where you cannot answer "what did this stage remove or change?" is a pipeline you cannot debug.

---

## 3. Sourcing

### 3.1 Where data comes from

- **Internal.** Support tickets, product docs, wiki pages, chat logs, application logs, source code and commits. This is usually your highest-value data because it matches your distribution exactly — and your biggest liability, because it is full of PII, secrets and access-controlled content. This handbook mines commits and docs for its tracks (see [19 — Java datasets](19-java-datasets.md), [23 — React datasets](23-react-datasets.md)).
- **Public datasets.** Curated corpora from Hugging Face and similar. Fast to start, but you inherit their licences, biases and contamination. The verified catalog and the licence traps are in [05 — Datasets](05-datasets.md).
- **Web data.** Crawled or scraped pages. Powerful and dangerous: `robots.txt` is a crawling convention, not a copyright licence, and a page being reachable does not make its text yours to train on. Check terms of use, copyright, and text-and-data-mining opt-outs; see [12 — Safety, law and licenses](12-safety-legal.md#4-model-and-dataset-licenses).
- **Synthetic.** Generated by a stronger model, then verified. Fills coverage gaps and teaches your exact formats, but amplifies the teacher's errors if unverified. The execution-first pipeline is [07 — Synthetic data](07-synthetic-data.md).

### 3.2 A sourcing checklist

Before a source enters the pipeline, answer five questions and record the answers:

| Question | Why it matters |
|---|---|
| **Ownership** — do we have the right to use it? | An unowned source can make the whole model unshippable. |
| **Consent** — did the people in it agree to this use? | Support tickets collected "to help you" may not be consented for training. |
| **Licence** — what does the licence permit? | CC-BY-NC data makes a commercial model non-commercial ([05 §4](05-datasets.md)). |
| **Freshness** — how old is it, how often does it change? | Stale data is the RAG-bot story of §1.2. |
| **Representativeness** — does it match what we ship against? | A model trained on easy tickets fails on the hard ones you never sampled. |

A source that fails ownership or licence is not "fix later" — it is excluded now. The other three are quality levers you tune.

---

## 4. Parsing documents

Parsing turns raw files into clean text while preserving the structure and metadata the downstream consumer needs. It is where most "garbage chunks" are born.

### 4.1 HTML to text

Stripping tags with a regex is a classic mistake — it leaves navigation menus, cookie banners and script contents in the text. Use a real parser and drop the structural noise. The standard library's `html.parser` is enough:

```python
from html.parser import HTMLParser

SKIP = {"script", "style", "nav", "header", "footer", "noscript"}


class Extractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip_depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag in SKIP and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if self.skip_depth == 0 and data.strip():
            self.parts.append(data.strip())


def html_to_text(html: str) -> str:
    p = Extractor()
    p.feed(html)
    return "\n".join(p.parts)


html = """<html><head><style>.b{color:red}</style>
<script>track()</script></head>
<body><nav>Home | Products | Login</nav>
<h1>Refund policy</h1>
<p>Refunds are processed within 14 days.</p>
<footer>(c) 2026 Example Corp</footer></body></html>"""

print(html_to_text(html))
```

```text
Refund policy
Refunds are processed within 14 days.
```

The nav bar, the style block, the tracking script and the footer are gone; the heading and paragraph remain. In production you would add a mature extractor (readability-style main-content detection) for messy pages, but the principle is this: skip structural elements, keep content, and never feed raw HTML to an embedding or training pipeline.

### 4.2 PDFs: the hard case

PDFs are a layout format, not a text format, and they fail in characteristic ways:

- **Text layer vs scanned.** A born-digital PDF has an extractable text layer; a scanned one is images and yields nothing until you run OCR. Detect it: if a page extracts fewer than a handful of characters but has a large image, it is scanned — route it to OCR, do not silently emit an empty chunk.
- **Columns.** Two-column layouts extract in the wrong reading order — the extractor reads across both columns line by line, interleaving them into nonsense. Detect it by checking for implausibly short "lines" that alternate topics; fix it with a layout-aware extractor that respects column boundaries.
- **Headers and footers.** Running headers, page numbers and footers repeat on every page and pollute every chunk. Detect them as lines that recur at the same position across many pages, and strip them.
- **Tables.** A table extracted as a flat string loses its rows and columns — "Q1 100 Q2 200" is meaningless. Detect tables and render them as Markdown (§4.6) so the structure survives.

The detection rule of thumb: **sample 20 pages, read the extracted text yourself, and compare to the PDF.** Bad PDF extraction is the top cause of garbage in document pipelines, and it is invisible unless you look. OCR adds its own errors (a scanned "l" becomes "1", "rn" becomes "m"); treat OCR output as noisy and run it through the cleaning stage hard.

### 4.3 Markdown

Markdown is already close to clean text, which is why it is the preferred intermediate format. Keep the heading hierarchy (it is free structure for chunking and citation), keep code fences intact (do not let a cleaner collapse their whitespace), and convert tables to a normalized form. Strip front-matter YAML into metadata rather than leaving it in the body.

### 4.4 Office files

`.docx`, `.xlsx` and `.pptx` are ZIP archives of XML. Extract the text through a library rather than by hand, but mind the traps: tracked changes and comments can leak deleted or private text; a spreadsheet's meaning lives in its cell grid, so flatten it to Markdown tables with headers, not a wall of numbers; a slide deck's reading order is not its visual order, so sort shapes top-to-bottom, left-to-right.

### 4.5 Email threads

Email is the boilerplate story of §1.2. A single message in a thread repeats the entire history below it, quoted with `>`, plus a signature. Train or index only the **new** message:

```python
import re

# Strip quoted replies and signatures from an email thread so the model
# trains on the new message, not the whole history repeated under it.
QUOTE = re.compile(r"^\s*>")
HEADER = re.compile(r"^On .* wrote:$")        # "On Tue, ... Jane wrote:"
SIG = re.compile(r"^--\s*$")                   # the standard signature cut


def latest_message(email: str) -> str:
    out = []
    for line in email.splitlines():
        if HEADER.match(line) or SIG.match(line):
            break
        if QUOTE.match(line):
            continue
        out.append(line)
    return "\n".join(out).strip()


thread = """Thanks, that fixed it!

--
Sent from my phone

On Tue, 14 Oct 2026, support wrote:
> Please restart the service and try again.
>> I cannot log in since the update.
"""
print(latest_message(thread))
```

```text
Thanks, that fixed it!
```

The quoted history and the signature are dropped; only the new line survives. Real threads vary (Outlook uses a `From:` header block, some clients indent instead of quoting), so maintain a small library of reply-boundary patterns and test it on your own mail.

### 4.6 Code and keeping structure

Source code parses cleanly but has its own rules: keep whole functions or classes together (a chunk that cuts a function in half teaches nothing), keep the file path as metadata, and decide deliberately whether comments and docstrings are signal or noise for your task.

Across every format, two things travel with the text:

- **Structure** — headings, lists and tables, best preserved as Markdown so the model and the chunker can see the document's shape.
- **Metadata** — `source`, `url`, `title`, `updated_at`, and an **access group** for per-user retrieval ([51 §11](51-ai-rag.md)). A record with no provenance cannot be cited, filtered, refreshed or deleted on request. Attach metadata at parse time, when you still know where the text came from.

---

## 5. Cleaning and normalization

Cleaning makes text canonical so that two things that mean the same compare equal. The biggest wins are invisible until you look at the bytes.

### 5.1 Unicode: normalize before you compare

The same visible character can have several byte encodings. Full-width digits (`１２３`), ligatures (`ﬁ`), accented letters composed two ways (`é` as one code point, or as `e` plus a combining accent), and "smart" quotes all break naive comparison, dedup and tokenization. **Normalize to a canonical form first.** NFC applies canonical composition, so the two spellings of `é` become the same code point. NFKC additionally folds *compatibility* characters: full-width to ASCII, ligatures to letters, a no-break space to a plain space. NFKC is the usual choice for match keys (dedup, search). For text a model will train on, remember that it is lossy (`x²` becomes `x2`, `½` becomes `1⁄2`, `™` becomes `TM`), so some pipelines store NFC text and use NFKC only for keys. Neither form touches smart quotes or dashes (`’`, `“`, `–` survive both); map those explicitly if they must match ASCII.

```python
import unicodedata

# 1. Unicode: full-width digits and an "fi" ligature look like ASCII but are not.
raw = "Invoice １２３ due; ﬁnal notice"   # "１２３" and "ﬁnal"
nfkc = unicodedata.normalize("NFKC", raw)
print("raw  :", raw, "| len", len(raw))
print("NFKC :", nfkc, "| len", len(nfkc))

# 2. Control characters: drop Unicode category C* (controls, format, private
#    use, unassigned) except \t and \n.
def strip_controls(s: str) -> str:
    return "".join(c for c in s if c in "\t\n" or unicodedata.category(c)[0] != "C")

dirty = "line1\x00\x07\tline2\x1b[31m"
print("controls stripped:", repr(strip_controls(dirty)))

# 3. Mojibake: UTF-8 text wrongly decoded as Latin-1 shows the classic "Ã©".
mojibake = "cafÃ©"                    # bytes of "café" read as Latin-1
try:
    repaired = mojibake.encode("latin-1").decode("utf-8")
except (UnicodeEncodeError, UnicodeDecodeError):
    repaired = mojibake
print("mojibake:", mojibake, "->", repaired)
```

```text
raw  : Invoice １２３ due; ﬁnal notice | len 28
NFKC : Invoice 123 due; final notice | len 29
controls stripped: 'line1\tline2[31m'
mojibake: cafÃ© -> café
```

Three cleaning steps at once. NFKC turned the full-width digits into `123` and the `ﬁ` ligature into two letters (the length *grew* from 28 to 29 because one ligature became two characters). The control-character pass dropped the null, bell and escape characters but kept the tab. Note that the rest of the ANSI colour code, `[31m`, survives: strip ANSI escape sequences with a regex before this pass. Category C also holds format characters such as the zero-width joiner, which emoji sequences and some scripts need; restrict the filter to `Cc` if your text contains them. The mojibake repair recovered `café` by reversing the bad decode — the classic `Ã©` pattern is the fingerprint of UTF-8 bytes read as Latin-1.

### 5.2 Whitespace, boilerplate and the rest

- **Whitespace.** Collapse runs of spaces and blank lines, strip trailing spaces, normalize line endings. Keep paragraph breaks; they carry structure.
- **Boilerplate.** Headers, footers, cookie banners, "Share this article", legal disclaimers. Detect by frequency: a line that appears on thousands of pages is boilerplate, not content. Remove it or it dominates your dedup and your model.
- **Language identification.** For a multi-source corpus, tag each record's language (a small classifier such as fastText's language ID is the usual tool) so you can filter to the languages you support and read per-language quality separately. A mis-tagged language is a silent quality sink.
- **Length filters.** Drop records that are too short to carry meaning (a "See page 2." fragment) or absurdly long (a concatenated log dump). Measure length in **tokens** when the consumer is a model, because that is the unit the model and the context budget count in ([49 — LLM internals](49-ai-llm-internals.md)).

### 5.3 Normalizing labels

If the data is labelled, the labels need cleaning too. "urgent", "Urgent", "URGENT", "p1" and "high" may all mean one class. Map them to a fixed taxonomy, trim whitespace, and reject labels outside the taxonomy rather than silently keeping a typo as a new class. A label set that grows every time someone fat-fingers a value is not a taxonomy.

---

## 6. Deduplication

Duplicates waste training compute, bias the model toward whatever is repeated, and — most dangerously — leak between train and eval so a memorised example looks like a correct prediction. There are two kinds.

### 6.1 Exact duplicates

Hash a *normalized* version of the text and keep the first occurrence. Normalizing first (§5) means "Reset your password" and "reset your   password" collapse to one key:

```python
import hashlib
import re
import unicodedata


def norm_key(text: str) -> str:
    """A normalized key: NFKC, lowercase, collapse whitespace."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"\s+", " ", t).strip()
    return hashlib.sha256(t.encode()).hexdigest()


docs = [
    "Reset your password from the Settings page.",
    "reset your   password from the settings page.",   # same, different spacing/case
    "Reset your password from the Settings page.",      # byte-for-byte dup
    "Upgrade your plan under Billing.",
]

seen, kept = set(), []
for d in docs:
    k = norm_key(d)
    if k in seen:
        continue
    seen.add(k)
    kept.append(d)

print(f"in : {len(docs)} docs")
print(f"out: {len(kept)} unique")
for d in kept:
    print(" -", d)
```

```text
in : 4 docs
out: 2 unique
 - Reset your password from the Settings page.
 - Upgrade your plan under Billing.
```

Three of the four docs collapsed to one because they are the same after normalization. Exact dedup is cheap and should run on every corpus.

### 6.2 Near-duplicates: shingles and Jaccard

Exact hashing misses "Refunds take 14 days" vs "Refunds take 14 business days" — different bytes, same content. The standard measure of overlap is **Jaccard similarity** on sets of **shingles** (overlapping k-word windows):

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
\]

!!! example "On paper"

    Split two short strings into 2-word shingles. "reset your password now" gives {reset your, your password, password now}; "reset your password today" gives {reset your, your password, password today}. Two shingles are shared and four are distinct in all:

    \[
    J=\frac{|A\cap B|}{|A\cup B|}=\frac{2}{4}=0.5
    \]

    So J = 0.5. A dedup threshold of 0.8 keeps both as distinct; a pair with three shared shingles out of four distinct (J = 3/4 = 0.75) is still kept, and only near-identical text (J ≥ 0.8) is merged.

### 6.3 MinHash: estimating Jaccard cheaply

Computing Jaccard between every pair of N documents is \(O(N^2)\) set operations — impossible at scale. **MinHash** turns each document's shingle set into a short fixed-length signature of integers, built so that the probability two documents agree in any given signature slot equals their Jaccard similarity. Comparing signatures is then cheap, and the fraction of matching slots is an unbiased estimate of Jaccard.

Why one slot agrees with probability exactly J: a slot uses one random hash function (think of it as a random shuffle of every possible shingle) and stores the smallest hash in the set. Hash every shingle in \(A\cup B\). Exactly one of them gets the smallest value, and each is equally likely to be that one. If it is in both sets, it is the minimum of A *and* of B, so the two slots are equal. If it is in only one set, that set's minimum is smaller than the other's, so the slots differ. The slots therefore agree with probability \(|A\cap B|/|A\cup B|=J\) (ignoring the negligible chance of two shingles hashing to the same value).

```python
import hashlib


def shingles(text: str, k: int = 2) -> set[str]:
    """Word k-shingles: overlapping windows of k words."""
    w = text.lower().split()
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b)


def signature(s: set[str], num_perm: int = 200) -> list[int]:
    """One min-hash per permutation, simulated by salting a SHA-1 hash."""
    sig = []
    for p in range(num_perm):
        mn = min(int.from_bytes(hashlib.sha1(f"{p}:{sh}".encode()).digest()[:8],
                                "big") for sh in s)
        sig.append(mn)
    return sig


def estimate(sig_a, sig_b) -> float:
    agree = sum(x == y for x, y in zip(sig_a, sig_b))
    return agree / len(sig_a)


s1 = shingles("reset your password now")
s2 = shingles("reset your password today")
print("shingles 1:", sorted(s1))
print("shingles 2:", sorted(s2))
print(f"true Jaccard : {jaccard(s1, s2):.3f}")
print(f"MinHash est. : {estimate(signature(s1), signature(s2)):.3f}")
```

```text
shingles 1: ['password now', 'reset your', 'your password']
shingles 2: ['password today', 'reset your', 'your password']
true Jaccard : 0.500
MinHash est. : 0.525
```

With 200 permutations the estimate (0.525) lands near the true Jaccard (0.5). Each slot is a yes/no trial with success probability J, so the estimate's standard error is \(\sqrt{J(1-J)/k}=\sqrt{0.25/200}\approx0.035\) for k = 200 slots: 0.525 is within one standard error, and four times as many permutations would halve the error. Production libraries such as `datasketch` implement this efficiently and pair it with LSH (below); the repository's dedup helpers use it ([07 §6.3](07-synthetic-data.md), `scripts/split_dataset.py --minhash`). The point of the toy version is to show there is no magic: it is minimums of salted hashes.

### 6.4 LSH: finding the duplicates without comparing every pair

MinHash makes a pair comparison cheap, but you still need to avoid \(O(N^2)\) pairs. **Locality-sensitive hashing (LSH)** splits each signature of \(r\cdot b\) slots into \(b\) bands of \(r\) rows, and buckets documents by each band. Two documents become candidates if they collide in *any* band. The probability they collide as a function of their Jaccard \(s\) is an S-curve:

\[
P=1-\bigl(1-s^{r}\bigr)^{b}
\]

!!! example "On paper"

    A pair with Jaccard s = 0.5, in an index with r = 2 rows per band and b = 4 bands (8 hashes total). The chance their band signatures match somewhere:

    \[
    s^{r}=0.5^{2}=0.25
    \]

    \[
    1-(1-0.25)^{4}=1-0.75^{4}
    \]

    \[
    =1-0.3164=0.684
    \]

    So a half-overlapping pair is caught about 68% of the time. A near-duplicate at s = 0.8 gives 1 − 0.36^4 ≈ 0.983 — caught almost always. Choosing r and b sets *where* the catch-rate jumps: roughly at \(s\approx(1/b)^{1/r}\), here \((1/4)^{1/2}=0.5\), which is why this pair sits in the middle of the curve. More rows per band push the jump to higher similarity (fewer false candidates); more bands pull it lower (fewer missed duplicates); a longer signature (more of both) makes the step steeper.

### 6.5 Why dedup matters per consumer

- **Training.** Duplicated examples are seen more often, so the model over-weights them — it memorises the repeated text and generalises worse. De-dup is one of the cheapest quality wins in a training corpus.
- **RAG.** A query that matches a 10×-duplicated chunk returns ten near-identical passages, crowding out diversity and wasting context budget. Dedup the corpus before indexing.
- **Evals.** The worst case. A duplicate that straddles train and eval turns the eval into a memorisation test (§1.2). Dedup *across* splits, not just within each, and treat near-duplicates as duplicates for leakage purposes.

---

## 7. Quality filtering

Not every unique, clean record deserves to be kept. Quality filtering removes the low-value tail so the model learns from the best data. Use cheap filters first, expensive ones last.

### 7.1 Heuristic rules

Fast, explainable, and catch the obvious junk: length bounds, symbol-to-text ratio (spam and markup are symbol-heavy), repeated-line ratio (templated or truncated content), and fraction of non-language characters.

```python
def symbol_ratio(text: str) -> float:
    """Fraction of characters that are not letters, digits or spaces."""
    if not text:
        return 1.0
    sym = sum(not (c.isalnum() or c.isspace()) for c in text)
    return sym / len(text)


def repeated_line_ratio(text: str) -> float:
    """Fraction of lines that are exact duplicates of an earlier line."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    seen, dup = set(), 0
    for ln in lines:
        if ln in seen:
            dup += 1
        seen.add(ln)
    return dup / len(lines)


good = "Refunds are processed within 14 days of the request."
junky = "$$$ BUY NOW >>> !!! @@@ #### %%% ^^^ &&& *** $$$"
spammy = "Click here\nClick here\nClick here\nBuy now\nClick here"

for name, t in [("good", good), ("junky", junky), ("spammy", spammy)]:
    print(f"{name:7} symbol_ratio={symbol_ratio(t):.2f} "
          f"repeat_ratio={repeated_line_ratio(t):.2f} len={len(t)}")

# A rule keeps a doc only if it is long enough and not mostly symbols/repeats.
def keep(t: str) -> bool:
    return len(t) >= 20 and symbol_ratio(t) < 0.30 and repeated_line_ratio(t) < 0.50

print("keep good  ?", keep(good))
print("keep junky ?", keep(junky))
print("keep spammy?", keep(spammy))
```

```text
good    symbol_ratio=0.02 repeat_ratio=0.00 len=52
junky   symbol_ratio=0.65 repeat_ratio=0.00 len=48
spammy  symbol_ratio=0.00 repeat_ratio=0.60 len=51
keep good  ? True
keep junky ? False
keep spammy? False
```

The good doc passes; the symbol spam fails on symbol ratio (0.65); the repeated spam fails on repeat ratio (0.60). Heuristics are a sieve — set thresholds from your own data's distribution, not from these toy numbers, and inspect what they drop.

### 7.2 Model-based classifiers

Heuristics cannot tell a well-written off-topic page from a well-written on-topic one. A cheap classifier can. Embed each record and train a small logistic-regression or gradient-boosted classifier on a few hundred hand-labelled "keep/drop" examples ([47 — Classical ML](47-ai-classical-ml.md)). Once a record is embedded the classifier itself costs microseconds, so the embedding pass sets the speed, and that is still far cheaper than an LLM call per record. It captures "is this the kind of text we want" far better than symbol counting. This is the workhorse quality filter for large corpora.

### 7.3 LLM-as-filter

For the hard middle — records a classifier is unsure about — an LLM can apply a written rubric ("keep if it is a factual support answer; drop if it is chit-chat, an auto-reply, or contains only a link"). It is the most accurate and the most expensive, so you cost it before you run it:

\[
c=\frac{t_\text{in}}{10^{6}}\,p_\text{in}+\frac{t_\text{out}}{10^{6}}\,p_\text{out}
\]

!!! example "On paper"

    One rubric call per doc: about 800 input tokens (doc + rubric) and 20 output tokens (a score). Assume $3 per million input tokens and $15 per million output tokens:

    \[
    c=\frac{800}{10^{6}}\,3+\frac{20}{10^{6}}\,15
    \]

    \[
    c=0.0024+0.0003=\$0.0027
    \]

    About a quarter-cent per doc; across 100,000 docs that is $270. (Prices are an assumption for the arithmetic, not a quote — see [55 — Production](55-ai-production.md).) A classifier costs a fraction of that, so the economical pattern is: heuristics drop the obvious junk for free, a classifier handles the bulk, and the LLM adjudicates only the uncertain remainder.

### 7.4 Perplexity filtering

A language model assigns a probability to text; low-probability (high-perplexity) text under a clean reference model is often garbled, machine-translated, or off-distribution. Scoring a corpus by perplexity and dropping the extreme tail is a classic pretraining filter. Use it with care: perplexity also flags rare-but-valuable technical text, so set the cut loosely and inspect both tails.

### 7.5 Measuring a filter

A filter is a classifier, so measure it like one. Hand-label a sample, treat "low-quality" as the positive class, and compute precision and recall (see [47](47-ai-classical-ml.md), [54](54-ai-evaluation.md)):

!!! example "On paper"

    A heuristic flags 25 of 100 sampled docs as low-quality. By hand you find 20 are truly low-quality; of the 25 flagged, 18 are real (TP), 7 are good docs wrongly dropped (FP), and 2 bad docs slipped through (FN):

    \[
    \text{precision}=\frac{18}{18+7}=\frac{18}{25}=0.72
    \]

    \[
    \text{recall}=\frac{18}{18+2}=\frac{18}{20}=0.90
    \]

    Precision 0.72 means 28% of what you drop was good data; recall 0.90 means you caught 90% of the junk. You cannot maximise both — tighten the rule to raise precision (keep more good data) or loosen it to raise recall (catch more junk). Decide which error is cheaper for your corpus.

---

## 8. Protecting data

Data preparation is where you remove what must not reach a model, a log, a third-party API or a shipped dataset.

### 8.1 PII and secrets scrubbing

Run redaction **in the pipeline, before** data is written anywhere it will persist — before logging, before a prompt is sent to a provider, before a training file is saved. A regex layer catches well-formed emails, phone numbers and card numbers (with a Luhn check to cut false positives); a trained NER classifier catches names and addresses that regexes miss. The detectors, the Luhn arithmetic and a runnable redactor live in [56 — Safety §8](56-ai-safety.md); do not reimplement them here, call them. What belongs to *this* stage is the policy:

- **Where it runs.** At ingest for anything that will be stored, and again at the boundary to any external service. Redacting "later" means the raw data already leaked into a log.
- **What to do with a hit.** Decide per field: mask (`[EMAIL]`), hash (to keep joinability without the value), drop the record, or quarantine it for human review. Masking is the default for training text; dropping is safest for secrets.
- **Secrets specifically.** API keys, tokens and private keys in training data teach the model to emit credential-shaped strings and may leak a live secret. Scan with a secret scanner and drop the record, do not mask-and-keep.

### 8.2 Licence and consent per record

Carry the licence and the consent basis on every record as metadata, not in a spreadsheet off to the side. When you build a release mix you then filter by licence programmatically ("exclude CC-BY-NC for the commercial model") and can prove provenance for an audit. A record whose licence you cannot name defaults to "all rights reserved" — exclude it from anything you publish ([05 §4](05-datasets.md), [12 §4](12-safety-legal.md#4-model-and-dataset-licenses)).

### 8.3 Retention and deletion requests

Under privacy law a person can ask you to delete their data, and that request has to reach *everywhere* their data went: the raw store, the processed splits, the vector index, and any fine-tune that memorised it ([56 §16](56-ai-safety.md)). Design for it:

- Key records by a stable user/document ID so you can find all of a person's data.
- Keep the `source_id → chunk_ids` map that makes a vector-index deletion a first-class operation, not a full rebuild ([51 §4](51-ai-rag.md)).
- Record which dataset version and which model were built from data that is now deleted, so you know what must be rebuilt or retrained. You usually cannot surgically remove one person from trained weights; the mitigation is retention limits and the ability to retrain a clean snapshot.

---

## 9. Labeling and annotation

When data needs labels — a class, a span, a preference, a quality score — the label quality caps the model quality. Treat labelling as a measured process, not a crowd task you throw over the wall.

### 9.1 Guidelines

Write a labelling guide as precise as a spec: the exact question, the label set, a worked **example** and a **counter-example** for each label, and tie-break rules for the ambiguous cases. Ambiguity in the guide shows up as disagreement between annotators ([54 §6.1](54-ai-evaluation.md)). A label taxonomy should be mutually exclusive where possible, cover the real distribution (including an "other" or "unclear" escape so people do not force-fit), and be versioned — changing the taxonomy invalidates old labels.

### 9.2 Pilots and agreement

Run a **pilot**: have two or more annotators label the same 100 items, then measure how often they agree before labelling thousands. Raw percent agreement is the first number:

\[
p_o=\frac{\text{agreements}}{\text{total}}
\]

!!! example "On paper"

    Two annotators label 100 tickets and agree on 82:

    \[
    p_o=\frac{82}{100}=0.82
    \]

    0.82 looks strong, but some agreement is luck. If both call a ticket "urgent" 30% of the time at random, they still agree by chance about 0.3·0.3 + 0.7·0.7 = 0.58 of the time. **Cohen's kappa** subtracts that chance agreement and rescales, here \((0.82-0.58)/(1-0.58)\approx0.57\), well below the raw 0.82. Kappa is the number to report — its formula and a worked example are in [54 — Evaluation §6.2](54-ai-evaluation.md). Low agreement means the guide is too vague; fix the guide, not the annotators.

### 9.3 Adjudication and label noise

When annotators disagree, **adjudicate**: a third senior labeller or the guide's author decides, and the decision feeds back into the guide as a new example. To estimate the noise left in your labels, **re-label a random sample** and compare to the originals:

- Re-label 200 items; if 16 come back different, the label-error rate is near 16/200 = 8%. (Disagreement mixes the original labeller's mistakes with the re-labeller's, so the estimate is only as good as the re-labeller; use your most careful annotator.) If those errors are random, even a perfect model would score only about 92% against these labels, so roughly 92% is the ceiling for an honest score. A model that scores 95% against labels that are 8% wrong is partly fitting the noise.

### 9.4 LLM-assisted labelling with human review

An LLM can pre-label at scale; a human then reviews, correcting the model's labels rather than labelling from scratch. This is usually much faster than labelling from scratch and keeps a human accountable for the final label. The failure mode is automation bias — reviewers rubber-stamp the model — so measure reviewer agreement with a held-out hand-labelled set, and never let the model both label and grade its own labels.

### 9.5 Active learning: label the examples that teach most

Labelling is expensive, so spend it where it helps most. **Uncertainty sampling** picks the unlabelled examples the current model is least sure about — near the decision boundary — because labelling a point the model already classifies confidently teaches it nothing.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# 6 labeled points in 1-D: class 0 below 0, class 1 above.
X_lab = np.array([[-3.], [-2.5], [-2.], [2.], [2.5], [3.]])
y_lab = np.array([0, 0, 0, 1, 1, 1])

# An unlabeled pool. Which one should a human label next?
X_pool = np.array([[-1.8], [-0.2], [0.1], [1.9], [2.8]])

clf = LogisticRegression().fit(X_lab, y_lab)
proba = clf.predict_proba(X_pool)[:, 1]        # P(class 1)
uncertainty = 1.0 - np.abs(proba - 0.5) * 2    # 1.0 at p=0.5, 0.0 at p in {0,1}

for x, p, u in zip(X_pool.ravel(), proba, uncertainty):
    print(f"x={x:+.1f}  P(1)={p:.3f}  uncertainty={u:.3f}")

pick = int(np.argmax(uncertainty))
print(f"label next: x={X_pool[pick, 0]:+.1f} (most uncertain)")
```

```text
x=-1.8  P(1)=0.133  uncertainty=0.266
x=-0.2  P(1)=0.448  uncertainty=0.896
x=+0.1  P(1)=0.526  uncertainty=0.948
x=+1.9  P(1)=0.878  uncertainty=0.243
x=+2.8  P(1)=0.949  uncertainty=0.103
label next: x=+0.1 (most uncertain)
```

The model picks `x=+0.1`, the point nearest the boundary where `P(1)` is closest to 0.5 — the label most likely to move the boundary. The confidently-classified points at the ends are a waste of a human's time. Over several rounds this often reaches a given accuracy with far fewer labels than random sampling; it is not guaranteed, because pure uncertainty sampling can fixate on noisy or outlying points, so mix in some random picks.

### 9.6 Costing a labelling project

Put numbers on it before you start. If one item takes 45 seconds to label, a 10,000-item set is 10,000 × 45 s = 450,000 s = 125 hours of annotator time; double-labelling a 20% sample (2,000 items) for agreement adds 25 hours; adjudication adds more. At a loaded rate this is a real budget line, and it is why active learning and LLM-assisted labelling exist. Decide the size from the precision your eval needs ([54 §8](54-ai-evaluation.md)), not from "as much as possible".

---

## 10. Synthetic data and augmentation

When you cannot source enough real data — rare cases, a new format, a class with few examples — you generate it. Done well it fills gaps cheaply; done badly it poisons the model.

### 10.1 When it helps and when it hurts

**Helps:** covering rare cases and the long tail; teaching an exact output format or your specific tool schemas; balancing a minority class; creating negative and refusal examples that rarely occur naturally ([07 §1](07-synthetic-data.md)).

**Hurts:**

- **Model collapse.** Train repeatedly on a model's own output and the distribution narrows — rare modes vanish, the model forgets the tails. Keep real data in the mix; do not build a model purely on its predecessors' output.
- **Bias amplification.** The teacher's skews become the student's, concentrated.
- **Teacher errors.** An unverified teacher invents confident, wrong content. This is the single biggest risk: a model that learns plausible-but-false output is worse than one with a gap.

### 10.2 Generate → verify → filter

The discipline that makes synthetic data safe is to never trust the generator. Generate candidates, **verify** each against ground truth (execute the code, check the fact against a source, validate the schema), then **filter and dedup** what passes. This is the execution-first pipeline of [07](07-synthetic-data.md) and the teacher-verification loop of [40 — Distillation data](40-distill-data.md). The yield math tells you how much to generate:

\[
K=N\cdot p_\text{verify}\cdot p_\text{dedup}
\]

!!! example "On paper"

    Generate N = 10,000 candidates. 60% pass verification; of those, 80% survive dedup:

    \[
    K=10000\cdot 0.6\cdot 0.8=4800
    \]

    So 4,800 usable rows from 10,000 generated. To land a target of 5,000 usable rows, work backwards:

    \[
    N=\frac{5000}{0.6\cdot 0.8}=\frac{5000}{0.48}\approx 10417
    \]

    Round up: generate at least 10,417. Yield compounds: two pass rates that each sound healthy (60% and 80%) together keep less than half (48%) of what you generate. The error can also run the other way: a verifier that lets errors through, or a dedup that is too loose, inflates K with junk. Measure both pass-rates on a pilot, and spot-check what passes, before you scale generation.

### 10.3 Augmentation techniques

Augmentation makes new examples from real ones, staying closer to the true distribution than free generation:

- **Paraphrase.** Reword an example to teach invariance to surface form. Verify the paraphrase preserves the label.
- **Back-translation.** Translate to another language and back; the round-trip produces a natural paraphrase. Watch for meaning drift on idioms and named entities.
- **Counterfactual augmentation for fairness.** Swap a protected attribute (a name, a gender, a nationality) and keep the label fixed, so the model learns the attribute should not change the decision ([56 §11](56-ai-safety.md)). This is a direct lever against learned bias.

Whatever the technique, record that a record is synthetic and how it was made, in its metadata and the data card (§15) — you will want to ablate with and without it.

---

## 11. Data for each consumer

The same cleaned corpus is shaped differently for each thing that consumes it. The preparation up to §10 is shared; the formatting below is per-consumer.

### 11.1 RAG corpora

Chunk-ready clean text with rich metadata and access-control tags. Chunk in tokens with overlap, attach `source`, `title`, `url`, `updated_at` and an access group to every chunk, and keep an idempotent `source_id → chunk_ids` map for updates and deletions. The full chunking and indexing treatment is [51 — RAG](51-ai-rag.md); this chapter's job is to deliver it clean, deduped and access-tagged.

### 11.2 Fine-tuning sets

Chat JSONL in the exact template the server uses, with a deliberate task mix:

- **Balance the task mix** so the model does not over-learn the most common task; under-represented behaviours need a minimum share ([04 §4](04-data-strategy.md)).
- **Include negatives and refusals** — examples where the right answer is to decline, ask a clarifying question, or say "I don't know" ([04 §5](04-data-strategy.md)). A set of only positive examples teaches a model that never refuses.
- **Check the token-length distribution** and set `max_seq_length` just above the 99th percentile, dropping or splitting longer rows, or loss masking silently trains on nothing ([06 §8](06-data-format.md), [53 §6](53-ai-fine-tuning.md)).

The canonical format, chat templates and loss masking are owned by [06 — Data format](06-data-format.md) and [53 — Fine-tuning](53-ai-fine-tuning.md).

### 11.3 Eval sets

Held out, stratified, versioned and contamination-checked. An eval set is drawn from the real distribution, labelled carefully (§9), tagged by slice so you can read per-slice pass rates, frozen with a version name, and checked for overlap with training data (§12). It is the asset the model is replaceable against — [54 — Evaluation §7](54-ai-evaluation.md) owns how to build one.

### 11.4 Classical ML tables

A feature table with a target column, built with the same hygiene plus the leakage rules specific to tabular ML: no feature computed with information from the future or from the target, splits that respect groups and time, and scaling fit on train only. [47 — Classical ML](47-ai-classical-ml.md) owns splits, leakage and feature engineering.

---

## 12. Splitting and leakage at scale

A split is wrong when information crosses from the data you train on into the data you measure on. At scale the subtle leaks dominate.

- **By time.** If the model will predict the future, split by time: train on the past, test on the later period. A random split lets the model "see" the future, inflating the score. This is the right split for anything with a temporal trend ([47](47-ai-classical-ml.md)).
- **By user/group.** If the same user, document or template appears in both train and test, the model can win by recognising the user, not solving the task. Split by group so every row from one group lands in one split.
- **By near-duplicate cluster.** Cluster near-duplicates (§6) and assign whole clusters to a split, so a paraphrase of a training example cannot appear in the test set. Splitting rows independently leaks exactly the examples dedup was meant to catch.

### 12.1 Contamination check by n-gram overlap

Before every training run, check that no eval item appears verbatim in training. The standard method flags any pair sharing a long n-gram (13-grams are common — long enough to be meaningful, short enough to catch near-verbatim copies). The arithmetic is the same at any k:

\[
g=L-k+1
\]

!!! example "On paper"

    Count 8-grams (k = 8) in two tiny texts. A train line of 10 words has g = 10 − 8 + 1 = 3 of them; an eval line of 9 words has g = 9 − 8 + 1 = 2. They share the leading 8-word window "the quick brown fox jumps over the lazy":

    \[
    \text{overlap}=\frac{\text{shared}}{\text{eval 8-grams}}=\frac{1}{2}=0.5
    \]

    Half the eval line's windows appear verbatim in training, so this eval item is contaminated — drop it from the eval set, or drop the training line. Production uses 13-grams and runs it across millions of rows (hashing the n-grams), but the arithmetic is identical. The basics of splits and leakage are in [47 — Classical ML](47-ai-classical-ml.md); the repository's procedure is [04 §7](04-data-strategy.md).

---

## 13. Validation

A dataset is a contract between the pipeline and its consumer. Validate it like you validate an API payload — at the boundary, every time, failing loudly.

### 13.1 Schema validation with pydantic

Check every record against a schema before it is written. `pydantic` turns the schema into code and gives precise errors:

```python
import json
from typing import Literal
from pydantic import BaseModel, ValidationError, field_validator


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str

    @field_validator("content")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("empty content")
        return v


class Example(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def shape(cls, v: list[Message]) -> list[Message]:
        if len(v) < 2:
            raise ValueError("need >= 2 messages")
        if v[-1].role != "assistant":
            raise ValueError("last message must be assistant")
        return v


rows = [
    '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}',
    '{"messages":[{"role":"user","content":"hi"},{"role":"user","content":"still me"}]}',
    '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"  "}]}',
]

ok, bad = 0, 0
for i, line in enumerate(rows):
    try:
        Example.model_validate_json(line)
        ok += 1
        print(f"row {i}: OK")
    except ValidationError as e:
        bad += 1
        print(f"row {i}: REJECT - {e.errors()[0]['msg']}")

print(f"valid={ok} invalid={bad}")
```

```text
row 0: OK
row 1: REJECT - Value error, last message must be assistant
row 2: REJECT - Value error, empty content
valid=1 invalid=2
```

Row 1 fails the "ends with assistant" rule, row 2 fails the "no empty content" rule, and both failures name the exact reason. The repository's `scripts/validate_dataset.py` ([06 §9](06-data-format.md)) applies the same idea to the full chat-with-tools format (valid roles, no empty assistant turn, calls only to tools that exist, argument JSON that parses) and adds destructive-command and secret patterns.

### 13.2 Data contracts and statistical checks

Schema validity is necessary, not sufficient — a well-formed dataset can still be wrong in aggregate. Add statistical checks as assertions:

- **Class balance.** Did one class suddenly drop to 2% because an upstream filter changed?
- **Length distribution.** Did the mean token length double because a parser started emitting whole pages?
- **Null rates.** Did a field go 30% null because a source changed its schema?
- **Drift between versions.** Compare this version's distributions to the last (§16). A sharp move is a bug until proven otherwise.

### 13.3 A data unit-test suite

Codify these as tests that run in CI, exactly like code tests. "Every row validates", "no train/eval overlap", "class balance within 5 points of last version", "token-length p99 below the serving limit", "zero PII hits in a 1,000-row sample". A failing data test blocks the dataset from being promoted, the same way a failing unit test blocks a merge. This is the single highest-leverage practice in the chapter: it turns silent data rot into a loud, early failure.

---

## 14. Versioning and lineage

You cannot reproduce a result, or roll back a bad one, if you cannot name the exact data it used.

### 14.1 Immutable snapshots, hashes and manifests

Treat a dataset version as immutable: once named, it never changes; a fix produces a new version. Capture it with a **manifest** — the list of files, their sizes and content hashes, plus a hash of the whole set — so any later copy can be verified byte-for-byte:

```python
import hashlib
import json
import tempfile
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(root: Path) -> dict:
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(root))] = {
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
    blob = json.dumps(files, sort_keys=True).encode()
    return {"files": files, "dataset_hash": hashlib.sha256(blob).hexdigest()}


# Build a tiny dataset directory and hash it into a manifest.
with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    (root / "train.jsonl").write_text('{"messages": []}\n')
    (root / "valid.jsonl").write_text('{"messages": []}\n')
    m = manifest(root)
    for name, meta in m["files"].items():
        print(f"{name}: {meta['bytes']} bytes  {meta['sha256'][:12]}...")
    print("dataset_hash:", m["dataset_hash"][:12], "...")
```

```text
train.jsonl: 17 bytes  19fc5b26092a...
valid.jsonl: 17 bytes  19fc5b26092a...
dataset_hash: eac34db49815 ...
```

The two files have the same content so the same hash; the `dataset_hash` summarises the whole set. Any change to any byte changes a file hash, which changes the dataset hash — tamper-evidence for free. The repository's `scripts/prepare_dataset.py` writes the same per-file `bytes` and `sha256` entries into its `manifest.json`, next to row counts, licences and sources ([04 §9](04-data-strategy.md)).

### 14.2 Tools

You rarely hand-roll versioning at scale. The common options, one neutral line each (verify the current feature set before you choose):

- **DVC** (Data Version Control) versions large files alongside git: git holds small `.dvc` metafiles with content hashes, and the bytes live in a local cache and a remote store (S3, Azure Blob, Google Cloud Storage, SSH, NFS and others).
- **lakeFS** gives an object store (S3, Azure Blob, GCS) git-like branches, commits and merges over the data itself.
- **Hugging Face dataset revisions**: a dataset on the Hub is a git repository, and `load_dataset(..., revision=...)` takes a branch, tag or commit hash. Pin a commit hash for an exact snapshot; a branch moves.
- **Delta Lake and Apache Iceberg** are table formats that keep table snapshots (versions) and support time-travel queries over large tabular data, within a retention window: Delta's `VACUUM` and log retention, and Iceberg's snapshot expiry, remove old versions, after which you cannot travel back to them.

### 14.3 Lineage

Record which data version produced which model or index version, and which sources produced the data version. A model card that says "trained on dataset `2026-10-28.v3`" and a manifest that says "built from sources X@rev, Y@rev" let you answer "what went into this model?" — required for debugging, for audits, and for a deletion request (§8.3). Lineage is the thread that ties §8, §14 and §15 together.

---

## 15. Documentation: data cards

A dataset without documentation is a liability the next engineer inherits. A **datasheet** ([Gebru et al., 2018](https://arxiv.org/abs/1803.09010)) or **data card** records what the data is, where it came from and where it fails. The repository generates a skeleton with `scripts/make_dataset_card.py`. A filled-in example for a support-ticket dataset:

```text
# Data card: support-tickets-triage 2026-10-28.v3

Summary: 18,400 resolved support tickets labelled with one of five
  intent classes, for fine-tuning a triage classifier.

Sources:        internal Zendesk export, 2024-01 to 2026-09.
Collection:     exported via API by the data team; tickets closed >30 days.
Licence/consent: internal use only; customer ToS permits product improvement;
                 NOT for external release.
Labels:         5-class taxonomy v2; 2 annotators + adjudication; kappa 0.78;
                re-label noise estimate 6%.
Preprocessing:  HTML->text; quoted replies and signatures stripped; NFKC;
                exact + MinHash dedup (threshold 0.8); PII masked
                (email/phone/card, Luhn-checked); low-quality filter.
Splits:         by ticket time (train <=2026-06, valid 07-08, test 09);
                decontaminated 13-gram vs test.
Class balance:  billing 31%, access 24%, bug 22%, how-to 15%, other 8%.
Known gaps:     English only; no voice-call transcripts; "other" is noisy;
                enterprise tickets under-represented (<3%).
Intended use:   triage routing suggestions with a human in the loop.
Out of scope:   automated ticket closing; non-English; legal/compliance intents.
Maintainer:     triage-ml team; review quarterly.
```

The sections that matter most are the ones people skip: **known gaps** ("English only", "enterprise under-represented") tell the reader exactly where not to trust a model built on it, and **out of scope** stops the data being reused for something it was never validated for. An honest limits section is worth more than a bigger dataset. Model and system cards, which sit above the data card, are in [56 §17](56-ai-safety.md).

---

## 16. Monitoring production data

Preparation does not stop at deployment. A live system produces data, and that data both feeds the next iteration and warns you when the world has moved.

### 16.1 Logging for future data, with consent

Log inputs and outputs so production becomes training and eval data — but under the §8 rules: redact PII before the log is written, record the consent basis, set a retention period, and restrict who can read traces ([55 §16.4](55-ai-production.md)). Logging without redaction is how a debugging convenience becomes a privacy incident.

### 16.2 Sampling for review

You cannot read every request. Sample — a small random slice plus an oversample of the interesting cases (low confidence, retries, thumbs-down) — and route the sample to human review. Reviewed samples become labelled data (§9) and eval cases (§11.3). This is the data flywheel of [55 §18.4](55-ai-production.md): real failures become labels become evals become fixes.

### 16.3 Detecting input drift

The input distribution shifts — new topics, new phrasing, a new customer segment — and a model tuned on the old distribution quietly degrades. Catch it by comparing this week's input histogram to a baseline. Two standard measures: **PSI** (Population Stability Index) and **KL divergence**.

\[
\text{PSI}=\sum_i (p_i-q_i)\,\ln\frac{p_i}{q_i}
\]

!!! example "On paper"

    A baseline input mix p = (0.5, 0.3, 0.2) over three length buckets; this week's traffic q = (0.4, 0.4, 0.2). Term by term:

    \[
    (0.5-0.4)\ln\tfrac{0.5}{0.4}=0.0223
    \]

    \[
    (0.3-0.4)\ln\tfrac{0.3}{0.4}=0.0288
    \]

    \[
    \text{PSI}=0.0223+0.0288+0=0.051
    \]

    The usual rules of thumb, inherited from credit-risk scoring rather than derived from a statistical test: PSI below 0.1 means "no real shift", 0.1–0.25 means "investigate", above 0.25 means "the input moved". By that convention 0.051 says traffic is stable this week; calibrate the cut-offs against your own normal week-to-week noise. KL divergence is directional: KL(q‖p), this week measured against the baseline, is 0.026 nats, and KL(p‖q) is 0.025. PSI is exactly their sum (0.025 + 0.026 ≈ 0.051), which is why it is symmetric. One pitfall: a bucket that is empty in either week makes the log infinite, so floor every bucket at a small value (say 0.0001) first. Wire one of these to an alert and you learn the world changed before your accuracy metric does.

### 16.4 Feedback loops

Close the loop deliberately: production surfaces failures, you sample and label them, they join the golden set and the training mix, you ship a fix and measure it on the eval, and the next failures are rarer and different. The danger is an *unmonitored* loop — training on the model's own outputs with no human check drifts the distribution (model collapse, §10.1). A healthy flywheel always has a human-labelled, real-data anchor.

---

## 17. A complete runnable mini pipeline

This ties the chapter together: a messy corpus of ten small documents, built in code, flows through parse → normalize → dedup → PII mask → quality filter → JSONL with metadata → manifest, printing a summary of what each stage removed or changed.

First, the corpus — ten documents with the problems this chapter covered: HTML with a nav bar, an exact duplicate, a near-duplicate, a full-width-digit invoice, an email with a quoted reply, a document with an email address, a junk string, a good doc, a second HTML page, and a too-short fragment:

```python
# A messy corpus of 10 small "documents", built in code. Each is (name, raw).
CORPUS = [
    ("d01.html",
     "<html><head><script>t()</script></head><body><nav>Home|Login</nav>"
     "<h1>Refund policy</h1><p>Refunds take up to 14 days to appear.</p>"
     "<footer>(c) 2026</footer></body></html>"),
    ("d02.txt", "Refunds are processed within 14 days of the request."),
    ("d03.txt", "Refunds   are processed   within 14 DAYS of the request."),   # d02 + spaces/case
    ("d04.txt", "Invoice １２３ total ＄９９ due now."),  # full-width
    ("d05.eml", "Thanks, resolved now.\n\nOn Mon, support wrote:\n> Try a restart.\n"),
    ("d06.txt", "Email the team at ops@example.com for access to the runbook."),
    ("d07.txt", "$$$ BUY NOW >>> !!! @@@ ### %%% ^^^ &&& *** $$$ CLICK <<<"),   # junk
    ("d08.txt", "Refunds are processed within 14 days of the request."),        # exact dup of d02
    ("d09.html",
     "<html><body><nav>Menu</nav><p>Enable two-factor auth in Security "
     "settings to protect the account.</p></body></html>"),
    ("d10.txt", "See page 2."),                                                 # too short
]

for name, raw in CORPUS:
    print(f"{name:9} {len(raw):3d} chars")
```

```text
d01.html  171 chars
d02.txt    52 chars
d03.txt    56 chars
d04.txt    30 chars
d05.eml    63 chars
d06.txt    60 chars
d07.txt    57 chars
d08.txt    52 chars
d09.html  115 chars
d10.txt    11 chars
```

`len()` counts characters (code points), not bytes: `d04`'s six full-width characters take 3 bytes each in UTF-8, so its 30 characters are 42 bytes on disk.

Now the pipeline. Each stage is one small function from earlier in the chapter, chained, with a counter of what it removed or changed:

```python
# continues from above: CORPUS is defined.
import hashlib
import json
import re
import unicodedata
from html.parser import HTMLParser

EMAIL = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
SKIP = {"script", "style", "nav", "header", "footer"}
stats = {}


class Strip(HTMLParser):
    def __init__(self):
        super().__init__(); self.d = 0; self.parts = []
    def handle_starttag(self, t, a):
        if t in SKIP: self.d += 1
    def handle_endtag(self, t):
        if t in SKIP and self.d: self.d -= 1
    def handle_data(self, s):
        if not self.d and s.strip(): self.parts.append(s.strip())


def parse(name, raw):
    if name.endswith(".html"):
        p = Strip(); p.feed(raw); return " ".join(p.parts)
    if name.endswith(".eml"):
        out = []
        for ln in raw.splitlines():
            if re.match(r"^On .* wrote:$", ln) or ln.startswith(">"): break
            out.append(ln)
        return " ".join(x for x in out if x.strip())
    return raw


def normalize(t):
    t = unicodedata.normalize("NFKC", t)
    return re.sub(r"\s+", " ", t).strip()


def shingles(t, k=2):
    w = re.sub(r"[^a-z0-9 ]", "", t.lower()).split()
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}


def near_dup(a, b, thr=0.7):
    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        return False
    return len(sa & sb) / len(sa | sb) >= thr


def symbol_ratio(t):
    return sum(not (c.isalnum() or c.isspace()) for c in t) / max(len(t), 1)


# ---- pipeline ----
rows = [(n, parse(n, r)) for n, r in CORPUS]
rows = [(n, normalize(t)) for n, t in rows]

seen, deduped = set(), []
exact = near = 0
for n, t in rows:
    h = hashlib.sha256(t.encode()).hexdigest()
    if h in seen:
        exact += 1; continue
    if any(near_dup(t, kt) for _, kt in deduped):
        near += 1; continue
    seen.add(h); deduped.append((n, t))
stats["removed_exact_dup"] = exact
stats["removed_near_dup"] = near

masked = 0
pii = []
for n, t in deduped:
    new = EMAIL.sub("[EMAIL]", t)
    masked += new != t
    pii.append((n, new))
stats["pii_masked"] = masked

kept = [(n, t) for n, t in pii if len(t) >= 20 and symbol_ratio(t) < 0.30]
stats["removed_low_quality"] = len(pii) - len(kept)

records = [{"id": n, "text": t, "source": "support-kb",
            "updated_at": "2026-10-28"} for n, t in kept]
jsonl = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
manifest = {"rows": len(records),
            "dataset_hash": hashlib.sha256(jsonl.encode()).hexdigest()}

print(f"{'stage':20} {'in':>4} {'out':>4} {'note':>18}")
print(f"{'parse':20} {len(CORPUS):>4} {len(rows):>4} {'html/email -> text':>18}")
print(f"{'normalize':20} {len(rows):>4} {len(rows):>4} {'NFKC + whitespace':>18}")
print(f"{'dedup':20} {len(rows):>4} {len(deduped):>4}"
      f" {'exact %d near %d' % (exact, near):>18}")
print(f"{'pii mask':20} {len(deduped):>4} {len(pii):>4} {'masked %d' % masked:>18}")
print(f"{'quality filter':20} {len(pii):>4} {len(kept):>4}"
      f" {'dropped %d' % stats['removed_low_quality']:>18}")
print(f"\nfinal: {len(records)} records")
print("dataset_hash:", manifest["dataset_hash"][:12], "...")
for r in records:
    print(" ", r["id"], "->", r["text"][:46])
```

```text
stage                  in  out               note
parse                  10   10 html/email -> text
normalize              10   10  NFKC + whitespace
dedup                  10    8     exact 1 near 1
pii mask                8    8           masked 1
quality filter          8    6          dropped 2

final: 6 records
dataset_hash: 4dc355bff2b1 ...
  d01.html -> Refund policy Refunds take up to 14 days to ap
  d02.txt -> Refunds are processed within 14 days of the re
  d04.txt -> Invoice 123 total $99 due now.
  d05.eml -> Thanks, resolved now.
  d06.txt -> Email the team at [EMAIL] for access to the ru
  d09.html -> Enable two-factor auth in Security settings to
```

Read the summary as the chapter in miniature. Parsing turned two HTML pages and an email into clean text (d01's nav/script/footer gone, d05's quoted reply gone). Normalization folded the full-width `１２３`/`＄９９` in d04 to `123`/`$99`. Dedup removed `d08` (a byte-for-byte copy of d02) and `d03`. After normalization `d03` differs from d02 only in the case of `DAYS`; this pipeline's hash does not lowercase (§6.1's key does, and would have called it exact), so the shingle check caught it instead, with Jaccard 1.0, above the 0.7 threshold. PII masking replaced the email in d06 with `[EMAIL]`. The quality filter dropped `d07` (symbol-heavy junk) and `d10` (too short). Ten messy documents became six clean, deduped, PII-safe records with metadata, plus a content hash that names this exact snapshot. Every stage reported what it did — which is the whole point: a pipeline you can audit stage by stage is a pipeline you can trust.

---

## 18. A data-readiness checklist

Before data feeds a model, an index, or an eval, every box should be ticked:

- [ ] **Sourced** with ownership, consent, licence, freshness and representativeness recorded (§3).
- [ ] **Parsed** with structure and metadata preserved; PDFs spot-checked by eye (§4).
- [ ] **Normalized** to canonical Unicode; boilerplate and control characters removed (§5).
- [ ] **Deduplicated** exact and near, *across* splits, not just within (§6).
- [ ] **Quality-filtered** with the filter itself measured (precision/recall) (§7).
- [ ] **Protected** — PII and secrets scrubbed in-pipeline; licence per record; deletion path exists (§8).
- [ ] **Labelled** with a versioned guide, measured agreement, and a noise estimate (§9).
- [ ] **Synthetic data** verified and filtered, with a measured yield, and flagged as synthetic (§10).
- [ ] **Formatted** for the specific consumer (RAG / fine-tune / eval / table) (§11).
- [ ] **Split** by time / group / near-duplicate cluster with no leakage (§12).
- [ ] **Validated** against a schema and a statistical contract, as data unit tests in CI (§13).
- [ ] **Versioned** as an immutable, hashed, named snapshot with lineage to sources and models (§14).
- [ ] **Documented** with a data card: sources, licence, preprocessing, known gaps, intended and out-of-scope use (§15).
- [ ] **Monitored** in production for drift, with redacted logging and a human-labelled feedback anchor (§16).

---

## 19. Key takeaways

- Data work is most of AI engineering, and most "model" failures are data failures wearing a mask — the stale PDF, the un-stripped signature, the leaked eval set.
- Draw the lifecycle and make every stage report what it removed or changed; a stage you cannot inspect is a stage you cannot debug.
- Parse with real parsers, not regexes; preserve structure and attach provenance metadata at parse time, when you still know where the text came from.
- Normalize Unicode (NFKC) before you compare, dedup or tokenize; exact-hash first, then near-dedup with shingles + MinHash/LSH, and dedup *across* splits.
- Filter quality cheapest-first: heuristics, then a classifier on embeddings, then an LLM rubric only for the uncertain middle — and measure the filter like the classifier it is.
- Scrub PII and secrets in the pipeline, carry licence and consent per record, and keep a real deletion path.
- Labels cap model quality: write a guide with examples and counter-examples, measure agreement and noise, and spend labels where they teach most (active learning).
- Verify synthetic data before you trust it; unverified teacher output is worse than a gap, and training on your own output collapses the distribution.
- Validate every record against a schema and statistical contract as data unit tests in CI; version datasets as immutable hashed snapshots with lineage; document them in a data card that is honest about gaps.
- Keep preparing after launch: monitor input drift (PSI/KL), sample for review under consent, and close the feedback loop with a human-labelled anchor.

---

## 20. Exercises

!!! question "Exercise 1"

    Split "error 500 on checkout" and "error 500 at checkout" into 2-word shingles and compute their Jaccard similarity.

??? success "Answer"

    A = {error 500, 500 on, on checkout}; B = {error 500, 500 at, at checkout}. They share only "error 500", so the intersection is 1 and the union is 5:

    \[
    J=\frac{1}{5}=0.2
    \]

    J = 0.2 — well below a 0.8 dedup threshold, so these are kept as distinct despite sharing a phrase.

!!! question "Exercise 2"

    An LSH index uses r = 3 rows per band and b = 10 bands. What is the chance a pair with Jaccard s = 0.6 is caught as a candidate?

??? success "Answer"

    Using \(P=1-(1-s^{r})^{b}\):

    \[
    s^{r}=0.6^{3}=0.216
    \]

    \[
    P=1-(1-0.216)^{10}=1-0.784^{10}
    \]

    \[
    P=1-0.0877\approx 0.912
    \]

    About 91%. With more bands or fewer rows per band the curve shifts to catch lower-similarity pairs — at the cost of more candidate pairs to verify.

!!! question "Exercise 3"

    A corpus of 1,000,000 documents has 18% exact duplicates. After removing them, near-duplicate detection removes a further 12% of what remains. How many documents survive, and how many were removed in total?

??? success "Answer"

    Exact dedup removes 180,000, leaving 820,000. Near-dedup removes 12% of 820,000 = 98,400, leaving:

    \[
    820000\cdot(1-0.12)=721600
    \]

    721,600 survive; 1,000,000 − 721,600 = 278,400 were removed. Near-duplicate removal is computed on what exact dedup already reduced, not on the original count.

!!! question "Exercise 4"

    On a hand-labelled sample of 200 docs, 50 are truly spam. A filter flags 60 docs, of which 45 are really spam. Compute the filter's precision and recall for the spam class.

??? success "Answer"

    TP = 45, FP = 60 − 45 = 15, FN = 50 − 45 = 5.

    \[
    \text{precision}=\frac{45}{60}=0.75
    \]

    \[
    \text{recall}=\frac{45}{50}=0.90
    \]

    It catches 90% of spam but a quarter of what it flags is clean data wrongly dropped — tighten the rule if that collateral loss matters more than the missed 10%.

!!! question "Exercise 5"

    You need 12,000 usable synthetic rows. 50% pass verification and 75% of those survive dedup. How many candidates must you generate?

??? success "Answer"

    The yield is \(0.5\cdot 0.75 = 0.375\), so:

    \[
    N=\frac{12000}{0.375}=32000
    \]

    Generate 32,000 candidates. If either pass-rate is lower than assumed, you fall short — measure both on a pilot before committing the generation budget.

!!! question "Exercise 6"

    A contamination check uses 13-grams. An eval item has 20 words. How many 13-grams does it contain, and if 3 of them appear verbatim in the training set, what fraction of its 13-grams overlap?

??? success "Answer"

    Number of 13-grams = \(20-13+1 = 8\). Overlap fraction:

    \[
    \frac{3}{8}=0.375
    \]

    Over a third of its windows appear verbatim in training — treat the item as contaminated and drop it from the eval (or drop the matching training rows).

!!! question "Exercise 7"

    Baseline input distribution over three buckets is p = (0.6, 0.25, 0.15); this week's is q = (0.5, 0.30, 0.20). Compute the PSI and say whether it signals drift.

??? success "Answer"

    \[
    (0.6-0.5)\ln\tfrac{0.6}{0.5}=0.0182
    \]

    \[
    (0.25-0.30)\ln\tfrac{0.25}{0.30}=0.0091
    \]

    \[
    (0.15-0.20)\ln\tfrac{0.15}{0.20}=0.0144
    \]

    \[
    \text{PSI}=0.0182+0.0091+0.0144=0.042
    \]

    PSI = 0.042, below the 0.1 rule-of-thumb cut-off, so no meaningful drift — the input is stable this week. You would still log it and watch the trend.

!!! question "Exercise 8"

    Two annotators label 150 items and agree on 120. What is the raw percent agreement, and why is it not enough to report on its own?

??? success "Answer"

    Percent agreement:

    \[
    p_o=\frac{120}{150}=0.80
    \]

    0.80, but some of that agreement is chance — if both annotators share a label's base rate, they agree sometimes by luck even without reading the item. Report **Cohen's kappa** ([54 §6.2](54-ai-evaluation.md)), which subtracts the chance agreement, so you know how much of the 0.80 is real reliability.

---

Next: [51 — Embeddings, vector search and RAG](51-ai-rag.md).
