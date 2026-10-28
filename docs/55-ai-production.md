# 55 — Production: serving, cost and LLMOps

*Last verified: 28 October 2026.*

This chapter is about everything between "the model works in a notebook" and "the feature is live, fast, affordable and observable". You will learn the shape of an LLM application in production, why decode speed is a memory-bandwidth problem (worked for Qwen3-14B on an M1 Pro and an 8B on a datacenter GPU), how batching, the KV cache, quantization and speculative decoding change the numbers, how to size hardware and model the cost against an API, the caching and routing tricks that cut that cost, and the operational layer — fallbacks, streaming, observability, percentiles, SLOs and the data flywheel. There is a runnable gateway in Python stdlib. Read [50 — Prompting and APIs](50-ai-prompting-apis.md) for the request itself and [54 — Evaluation](54-ai-evaluation.md) for the rollout gates; the inference math is grounded in [02 — Hardware](02-hardware.md), [10 — Export and serving](10-export-serving.md), [32 — Mathematics](32-mathematics.md) and [37 — Architecture](37-architecture.md).

---

## 1. The architecture of an LLM application

### 1.1 The request path

A production LLM feature is not "call the model". It is a pipeline, and most of the engineering is in the layers around the model.

```text
                 ┌────────────────────────────────────────────────┐
  client ──────► │ GATEWAY                                        │
  (stream back)  │ request-id · auth · rate limit · timeout       │
                 └───────┬────────────────────────────────┬───────┘
                         │                                │
                 ┌───────▼────────┐              ┌────────▼────────┐
                 │ CACHE          │  miss        │ LOGGING /       │
                 │ exact + sem.   ├───────┐      │ TRACING         │
                 └───────┬────────┘       │      │ (every step)    │
                    hit  │                │      └────────▲────────┘
                         │        ┌───────▼─────────┐     │
                         │        │ PROMPT TEMPLATE │     │
                         │        │ + RETRIEVAL(RAG)│     │
                         │        └───────┬─────────┘     │
                         │        ┌───────▼─────────┐     │
                         │        │ MODEL ROUTER    │     │
                         │        │ small/large/API │     │
                         │        └───────┬─────────┘     │
                         │        ┌───────▼─────────┐     │
                         │        │ MODEL SERVERS   │     │
                         │        │ vLLM/MLX/...    │     │
                         │        │ + batch queue   │     │
                         │        └───────┬─────────┘     │
                         │        ┌───────▼─────────┐     │
                         │        │ GUARDRAILS      │     │
                         │        │ in/out filters  │     │
                         │        └───────┬─────────┘     │
                         └────────────────┴───────────────┘
                                          ▼
                          EVAL PIPELINE (offline, on logs)
```

| Layer | Job |
|---|---|
| Gateway | one entry point: request IDs, auth, rate limiting, timeouts, streaming |
| Auth / rate limiter | who may call, how often (per key/tenant) |
| Prompt templates | versioned prompt assembly, not string concatenation in code |
| Retrieval | fetch context for RAG ([51](51-ai-rag.md)) |
| Model router | pick the cheapest model that can do this request ([§13](#13-routing-and-model-cascades)) |
| Model servers | the inference engine(s) ([§9](#9-serving-engines)) |
| Guardrails | input/output filters, PII, scope ([56](56-ai-safety.md)) |
| Cache | exact + semantic response cache ([§11](#11-caching-layers)) |
| Queue | batch requests to the model for throughput ([§4](#4-batching-and-continuous-batching)) |
| Logging / tracing | one span per step, token and cost accounting ([§16](#16-observability)) |
| Eval pipeline | run evals on production logs ([54](54-ai-evaluation.md), [§18.4](#184-the-data-flywheel)) |

!!! note "Software-engineering analogy"

    This is the request pipeline you already build for any API: an ingress/gateway, auth, a cache, a service tier, structured logs, a metrics pipeline. The model server is just one more backend — an unusually slow, expensive, non-deterministic one. Every instinct you have about timeouts, retries, caching and observability transfers; the numbers are different.

---

## 2. Inference basics: prefill vs decode

### 2.1 Two phases

A generation request has two phases with completely different performance characteristics:

- **Prefill** processes the whole prompt in one forward pass to build the KV cache. It is **compute-bound**: a big matmul over all prompt tokens at once, so it uses the GPU's FLOPS. Its cost scales with prompt length.
- **Decode** generates output tokens one at a time, each a forward pass that reads *all the weights* to produce *one* token. It is **memory-bandwidth-bound**: the matmuls are tiny (one token) but you must stream every weight from memory each step. Its cost scales with output length.

### 2.2 The latency vocabulary

| Term | Meaning | Driven by |
|---|---|---|
| **TTFT** (time to first token) | prompt in → first output token | prefill (prompt length) |
| **TPOT / ITL** (time per output token / inter-token latency) | gap between output tokens | decode (bandwidth) |
| **Total latency** | TTFT + TPOT × (output tokens − 1) | both |
| **Throughput** | tokens/s across all requests | batching |

Throughput and latency trade off: batching more requests raises total throughput but can raise each request's latency ([§4](#4-batching-and-continuous-batching)). In an agent loop the latency is dominated by prefill of the system prompt plus tools and by output length, not raw decode speed ([10 §8](10-export-serving.md)).

---

## 3. The decode speed bound

Because decode reads every weight once per token, the ceiling on single-stream decode speed is set by memory bandwidth, not FLOPS:

\[
\text{tokens/s} \approx \frac{\text{memory bandwidth}}{\text{bytes read per token}}.
\]

!!! example "On paper"

    Qwen3-14B 4-bit on an M1 Pro. Bandwidth ~200 GB/s ([02 §1](02-hardware.md)); the weights read per token are ~8.3 GB ([02 §3.2](02-hardware.md)):

    \[
    \frac{200}{8.3} \approx 24\ \text{tok/s}
    \]

    That 24 tok/s is the roofline. The handbook measures 15–17 tok/s ([10 §8](10-export-serving.md)), about 62–71% of it: real decode loses the rest to KV-cache reads, Python and imperfect kernels. You cannot beat the roofline by tuning code — only by reading fewer bytes (quantize) or faster memory (different hardware).

!!! example "On paper"

    An 8B model in fp16 on an H100 SXM. NVIDIA lists 3.35 TB/s = 3350 GB/s of HBM3 bandwidth; fp16 weights are ~16 GB:

    \[
    \frac{3350}{16} \approx 209\ \text{tok/s}
    \]

    Roofline ~209 tok/s; a real single stream reaches ~60–70% of it, ~125–145 tok/s. The datacenter GPU has ~17× the M1 Pro's bandwidth (3350/200), which is most of why it decodes far faster — not its FLOPS.

The bandwidth figures: M1 Pro ~200 GB/s from [02 §1](02-hardware.md); H100 SXM 3.35 TB/s from NVIDIA's H100 datasheet page.

---

## 4. Batching and continuous batching

### 4.1 Why batching raises throughput

Decode is bandwidth-bound: one step reads all the weights to make one token. If you process \(B\) requests *together*, you read the weights **once** and make \(B\) tokens. So aggregate throughput rises almost linearly with batch size until the matmuls grow large enough to become compute-bound.

\[
\text{throughput}_{\text{batch}} \approx B \times \text{throughput}_{\text{single}}.
\]

!!! example "On paper"

    One Qwen3-14B stream decodes at ~24 tok/s (roofline, [§3](#3-the-decode-speed-bound)), reading 8.3 GB per step. A batch of 8 reads the same 8.3 GB once and emits 8 tokens:

    \[
    8 \times 24 = 192\ \text{tok/s aggregate}
    \]

    Each request still sees ~24 tok/s, but the server does 8× the total work for ~1× the weight traffic. This is why a busy server batches: the weights are the expensive read, and batching amortises them. The KV cache is not shared, though: at 4K context each request adds 0.67 GB of reads per step ([§5](#5-kv-cache-memory-and-concurrency)), so 8 such requests read \(8.3+8\times0.67\approx13.7\) GB per step and the bound drops to \(8\times200/13.7\approx117\) tok/s. Compute caps it too: each token costs about \(2N\approx29.6\) GFLOP, so 192 tok/s would need ~5.7 TFLOPS, more than the M1 Pro GPU\'s ~5.2 TFLOPS peak ([02 §1](02-hardware.md)), while 117 tok/s needs ~3.5 TFLOPS, about two-thirds of it. On the Mac, batching gains flatten after a handful of requests; on a datacenter GPU with hundreds of TFLOPS the bandwidth bound holds much longer.

### 4.2 Continuous (in-flight) batching

Static batching waits to fill a batch, then runs it to completion — a short request is stuck behind a long one. **Continuous batching** adds and removes requests from the running batch *at every decode step*: when one request finishes, its slot is filled immediately by a waiting one. This keeps the batch full, which keeps throughput high, without making short requests wait for long ones. It is the default in vLLM, SGLang and TGI; MLX's server batches too (`--decode-concurrency`, default 32; [10 §4](10-export-serving.md)).

The trade-off: a fuller batch means each step's matmul is bigger, so per-token latency rises slightly, and a new request may wait a step or two to join. You tune batch size to balance throughput against tail latency.

---

## 5. KV cache memory and concurrency

Every concurrent request holds its own KV cache, and that is what limits how many fit. The per-token size is ([32 §6.3](32-mathematics.md), [37 §2.1](37-architecture.md)):

\[
\text{KV bytes/token} = 2 \cdot L \cdot G \cdot d_h \cdot b.
\]

!!! example "On paper"

    Qwen3-14B, bf16 cache: \(L=40\), \(G=8\) KV heads, \(d_h=128\), \(b=2\):

    \[
    2 \cdot 40 \cdot 8 \cdot 128 \cdot 2 = 163{,}840
    \]

    That is 160 KiB per token. A 4096-token request holds \(4096 \times 163{,}840\text{ B} \approx 0.67\text{ GB}\), matching [02 §3.5](02-hardware.md). The cache grows every step, so a long conversation costs more than a short one.

The number of concurrent requests that fit is the free memory divided by the per-request cache:

\[
N_{\text{concurrent}} \approx \frac{\text{mem} - \text{weights} - \text{runtime}}{\text{KV per request}}.
\]

!!! example "On paper"

    On a 32 GB M1 Pro, usable ~24 GB, minus 8.3 GB weights and ~2 GB runtime leaves ~13.7 GB for KV. At 0.67 GB per 4K request:

    \[
    \frac{13.7}{0.67} \approx 20
    \]

    About 20 concurrent 4K-context requests before the cache overflows — and fewer as each conversation grows. On an 80 GB datacenter GPU the same arithmetic allows about 70 with bf16 weights (\((80-29.5-2)/0.67\approx72\)) or about 100 with 4-bit weights, and hundreds at shorter contexts, which is why datacenter serving is throughput-oriented and laptop serving is not.

---

## 6. Paged attention

The naive KV cache reserves one contiguous block per request sized for the *maximum* context, so a request that uses 500 of 4096 tokens wastes 7/8 of its reservation, and memory fragments. **Paged attention** (from vLLM) borrows the operating system's virtual-memory trick: the KV cache is split into fixed-size *pages*, allocated on demand as the sequence grows, and an index maps logical positions to physical pages. Requests sharing a prefix can even share pages (copy-on-write). The effect is far less wasted and fragmented memory, so you fit more concurrent requests in the same RAM — the [§5](#5-kv-cache-memory-and-concurrency) count goes up without buying hardware.

!!! note "Software-engineering analogy"

    Paged attention is `malloc` with pages instead of one fixed-size array per request. The same reason the OS pages memory — avoid pre-reserving the worst case, avoid fragmentation, share read-only pages — applies to the KV cache.

---

## 7. Quantization in serving

Quantization shrinks the weights (and sometimes activations); in serving it buys both memory and decode speed.

- **Weights-only (e.g. W4A16).** Store weights in 4-bit, compute in fp16. Cuts weight memory ~3.5× (4.5 bits per weight once the group scales are counted) and, because decode is bandwidth-bound, speeds decode by about the same factor. This is what llama.cpp and MLX 4-bit do. Minimal accuracy loss with good quantizers.
- **Weights + activations (e.g. W8A8).** Also quantize activations to int8 so the matmuls use int8 tensor cores. This speeds the *compute-bound* prefill too, but quantizing activations is more accuracy-sensitive and needs calibration.

\[
\text{decode speedup} \approx \frac{\text{bytes}_{\text{fp16}}}{\text{bytes}_{\text{quant}}}.
\]

!!! example "On paper"

    Qwen3-14B on the M1 Pro. fp16 weights are ~29.5 GB; 4-bit ~8.3 GB:

    \[
    \frac{200}{29.5} \approx 6.8\ \text{tok/s (fp16)}
    \]

    \[
    \frac{200}{8.3} \approx 24\ \text{tok/s (4-bit)}
    \]

    The 4-bit model decodes \(24/6.8 \approx 3.5\times\) faster *and* frees ~21 GB. (The fp16 figure is only a bound: 29.5 GB does not fit in the ~24 GB the Mac's GPU can use.) On a bandwidth-bound workload, quantization is the single biggest speed lever — it directly cuts the bytes read per token.

---

## 8. Speculative decoding

Speculative decoding runs a small, fast **draft** model to propose \(k\) tokens, then verifies all \(k\) in one forward pass of the big model. Accepted tokens are free; the big model does one pass instead of \(k\). If each draft token is accepted with probability \(\alpha\), the expected number of tokens produced per big-model step is

\[
E = \frac{1 - \alpha^{\,k+1}}{1 - \alpha}.
\]

!!! example "On paper"

    Acceptance \(\alpha = 0.8\), draft length \(k = 4\). First \(0.8^5 = 0.32768\):

    \[
    E = \frac{1 - 0.32768}{1 - 0.8}
    \]

    \[
    E = \frac{0.67232}{0.2} \approx 3.36
    \]

    About 3.36 tokens per big-model pass instead of 1 — a ~3.4× speedup in big-model forward passes, minus the cheap draft cost. It is lossless: the output distribution is identical to the big model's. MLX supports it via `--draft-model` ([10 §4](10-export-serving.md)).

Speculative decoding helps when the draft agrees with the target often (high \(\alpha\)); on hard, surprising text \(\alpha\) drops and the draft work is wasted.

---

## 9. Serving engines

One neutral line each, verified October 2026. Pick by hardware and need, not hype.

| Engine | What it is |
|---|---|
| **vLLM** | open-source GPU serving library; introduced PagedAttention; continuous batching and prefix caching; OpenAI-compatible server. |
| **SGLang** | open-source serving framework optimized for agentic and large-scale workloads; RadixAttention for automatic prefix caching. |
| **TGI** (Text Generation Inference) | Hugging Face's server; **in maintenance mode** since December 2025, and its GitHub repository was archived (read-only) in March 2026; its README points to vLLM, SGLang, llama.cpp and MLX instead. |
| **llama.cpp** | C/C++ CPU/GPU engine; GGUF weights, excellent quantization support; `llama-server` is OpenAI-compatible. |
| **Ollama** | convenience wrapper (built on llama.cpp) for local model management; OpenAI-compatible `/v1` is a subset. |
| **MLX / `mlx_lm.server`** | Apple-silicon engine; OpenAI-compatible; batching, prompt cache and speculative decoding — the server's own docs say it is **not recommended for production** (basic security only). |
| **TensorRT-LLM** | NVIDIA's open-source, PyTorch-architected optimizer; `trtllm-serve` gives an OpenAI-compatible server; often the fastest on NVIDIA hardware, heaviest to set up. |

For this handbook's 32 GB Mac, the real choices are MLX (`mlx_lm.server`) and llama.cpp (`llama-server`); for a datacenter deployment, vLLM or SGLang, or TensorRT-LLM when you need NVIDIA's last few percent ([10](10-export-serving.md)).

---

## 10. Sizing hardware for a target load

Work backward from the load. The decode-bound resource is output tokens per second.

\[
\text{GPUs} = \left\lceil \frac{\text{req/s} \times \text{out tokens/req}}{\text{out tokens/s per GPU}} \right\rceil.
\]

!!! example "On paper"

    Target 50 req/s, each 200 output tokens, and assume one GPU sustains 2,000 output tok/s with continuous batching:

    \[
    50 \times 200 = 10{,}000\ \text{tok/s needed}
    \]

    \[
    \lceil 10{,}000 / 2{,}000 \rceil = 5\ \text{GPUs}
    \]

    Add one for headroom and failover: **6 GPUs**. The per-GPU throughput is the number to *measure* on your model and hardware, not assume; it depends on quantization, batch size and sequence length.

Always size on the tokens, not the request count: a feature that emits 2,000-token answers needs 10× the GPUs of one that emits 200, at the same request rate.

---

## 11. Caching layers

### 11.1 Exact-response cache

Hash the full request (model, messages, params); if seen before, return the stored response. Near-free, safe as long as the key covers everything that changes the answer (model and prompt version, parameters, and the user or tenant when answers are personalised), and surprisingly effective for repeated prompts (health checks, popular questions, retried requests). This is what the gateway in [§15](#15-a-minimal-gateway-in-python) implements.

### 11.2 Semantic cache

Embed the query; if its cosine similarity to a cached query is above a threshold, return that cached answer instead of calling the model ([51](51-ai-rag.md) for embeddings). It catches paraphrases the exact cache misses, but carries a **false-hit risk**: set the threshold too low and you return a cached answer for a query that only *looks* similar but needs a different answer. Tune the threshold up, and measure the false-hit rate on a labelled set before trusting it.

\[
\text{cost}_{\text{with cache}} = (1-h)\cdot\text{cost}_{\text{full}} + h\cdot\text{cost}_{\text{cache}}.
\]

!!! example "On paper"

    Full model cost \$0.0007/request ([§12](#12-cost-modeling)), cache cost ~\$0 (embed + lookup), hit rate \(h=0.4\):

    \[
    \text{cost} = 0.6 \times 0.0007 = 0.00042
    \]

    \[
    \text{saved} = 0.4 \times 0.0007 = 0.00028
    \]

    A 40% hit rate saves 40% of the model spend: at 1M requests/day that is \(0.00028 \times 10^{6} = \$280\)/day. The risk is the 40% you *did not* send to the model — each is correct only if the threshold was tight enough.

### 11.3 Prefix / prompt caching

Long system prompts and tool schemas are identical across requests. Prefix caching stores the KV cache for that shared prefix so prefill skips it, cutting TTFT. SGLang's RadixAttention does this automatically; MLX has `--prompt-cache-size`; it is why you keep the system/tool block stable and put the variable part last ([10 §8](10-export-serving.md)).

---

## 12. Cost modeling

### 12.1 Cost per request

State your prices as assumptions; they move, and this handbook does not quote live API prices. Assume **\$0.50 per 1M input tokens** and **\$1.50 per 1M output tokens**.

\[
\text{cost} = \frac{\text{in}}{10^{6}}\,p_{\text{in}} + \frac{\text{out}}{10^{6}}\,p_{\text{out}}.
\]

!!! example "On paper"

    A request with 800 input and 200 output tokens:

    \[
    \frac{800}{10^{6}} \times 0.50 = 0.00040
    \]

    \[
    \frac{200}{10^{6}} \times 1.50 = 0.00030
    \]

    \[
    \text{cost} = 0.00070
    \]

    \$0.0007 per request, \$0.70 per thousand. Output tokens cost 3× input here, so verbose answers are the expensive ones — cap `max_tokens`.

### 12.2 API vs self-host break-even

Self-hosting has a fixed hourly cost whether or not you use it. Assume a rented GPU at **\$2.00/hour**, so \$2 × 24 × 30 = **\$1,440/month** per GPU. The break-even monthly volume is where API spend equals that fixed cost:

\[
N^{*} = \frac{\text{fixed monthly}}{\text{cost per request (API)}}.
\]

!!! example "On paper"

    Fixed \$1,440/month, API \$0.0007/request:

    \[
    N^{*} = \frac{1440}{0.0007} \approx 2.06\times10^{6}
    \]

    Above ~2.06M requests/month, self-hosting is cheaper **if the GPU is busy**. One GPU at 10 req/s can serve \(10 \times 86400 \times 30 = 25.92\text{M}\)/month, so the capacity is there — the catch is utilization.

### 12.3 The utilization effect

Idle GPUs still bill. The effective self-host cost per request is the fixed cost divided by the requests you *actually* serve, which is capacity × utilization. Self-host beats the API only above a break-even utilization:

\[
u^{*} = \frac{\text{fixed monthly}}{\text{capacity} \times \text{cost per request (API)}}.
\]

!!! example "On paper"

    Fixed \$1,440, capacity 25.92M/month, API \$0.0007:

    \[
    u^{*} = \frac{1440}{25.92\times10^{6} \times 0.0007}
    \]

    \[
    u^{*} \approx \frac{1440}{18{,}144} \approx 0.079
    \]

    You need only ~8% utilization to beat the API — but a GPU sitting at 2% utilization costs *more* per request than the API. The lesson: self-hosting wins on steady, high volume; the API wins on spiky, low or unpredictable traffic because you pay only for what you use.

---

## 13. Routing and model cascades

Not every request needs the big model. A **router** sends each request to the cheapest model that can handle it; a **cascade** runs a cheap model first and escalates only the hard cases. Assume a small model at \$0.0001/request and a large at \$0.0010/request, with a fraction \(f\) escalated:

\[
E[\text{cost}] = c_{\text{small}} + f\,c_{\text{large}}.
\]

!!! example "On paper"

    \(c_{\text{small}}=0.0001\), \(c_{\text{large}}=0.0010\), escalate \(f=0.3\):

    \[
    E[\text{cost}] = 0.0001 + 0.3 \times 0.0010 = 0.0004
    \]

    \$0.0004 vs \$0.0010 for large-only: a **60% cut**. Every request pays the small cost; only 30% also pay the large cost.

\[
E[\text{acc}] = (1-f)\,a_{\text{small, kept}} + f\,a_{\text{large}}.
\]

!!! example "On paper"

    Of all queries, 70% are kept by the small model (accuracy 0.90 on those) and 30% escalate to the large (accuracy 0.95):

    \[
    E[\text{acc}] = 0.7 \times 0.90 + 0.3 \times 0.95
    \]

    \[
    E[\text{acc}] = 0.63 + 0.285 = 0.915
    \]

    91.5% accuracy at 40% of the cost, versus 95% for large-only if the large model also scores 0.95 on the queries the small one keeps. Whether trading 3.5 points for a 60% saving is worth it is a product call — and it only holds if the escalation rule actually sends the *hard* queries up.

---

## 14. Reliability: fallbacks, timeouts, circuit breakers, degradation

Models and their servers fail, hang and rate-limit. Treat them like any flaky dependency:

- **Timeouts.** Bound every call. A model that has not produced a token in N seconds is a failure, not a slow success.
- **Retries with backoff.** Retry transient 429/503 with exponential backoff and jitter — but only idempotent calls, and cap the attempts ([50](50-ai-prompting-apis.md)).
- **Fallbacks.** On failure, fall back: a smaller/faster model, a cached answer, or a canned "try again" — the [§15](#15-a-minimal-gateway-in-python) gateway does this.
- **Circuit breakers.** If a backend fails past a threshold, stop calling it for a cool-down so you fail fast instead of piling requests onto a sick server.
- **Graceful degradation.** Decide what "degraded but up" means: answer without RAG, drop to a shorter prompt, disable the expensive feature — anything better than a 500.

!!! note "Software-engineering analogy"

    Nothing here is LLM-specific. It is the standard resilience kit (timeouts, retries, bulkheads, circuit breakers) you apply to any remote dependency. The only twist is that a "success" can be a *bad answer*, so a timeout is not your only failure mode — see guardrails and evals ([56](56-ai-safety.md), [54](54-ai-evaluation.md)).

---

## 15. A minimal gateway in Python

This is a working gateway on Python's stdlib `http.server`: it adds a request ID, times each request, accounts tokens, serves an exact-match cache, and falls back to a second model server when the primary is down. The two model servers here are stubs standing in for `mlx_lm.server`; in production you point `MODEL_URL` at the real one ([10](10-export-serving.md)). It uses a fake clock so the printed timings are reproducible.

```python
import http.server, socketserver, json, threading, urllib.request, hashlib

# Fake clock so the demo's timings are reproducible; in production call
# time.perf_counter(). Safe here because requests are sent one at a time.
NOW = [0.0]
def clock():
    return NOW[0]

MODEL_URL = "http://127.0.0.1:18531/v1/chat/completions"     # primary model
FALLBACK_URL = "http://127.0.0.1:18532/v1/chat/completions"  # cheaper/smaller
CACHE, LOG, IDS = {}, [], [0]

def call_model(url, payload):
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1.0) as r:
        return json.loads(r.read())

class Gateway(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence default access log
        pass
    def do_POST(self):
        IDS[0] += 1
        rid = f"req-{IDS[0]}"            # a real gateway uses uuid.uuid4().hex
        start = clock()
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        key = hashlib.sha256(raw).hexdigest()
        hit = key in CACHE
        fallback = False
        if hit:
            NOW[0] += 0.001             # cache lookup is ~free
            out = CACHE[key]
        else:
            try:
                out = call_model(MODEL_URL, raw)
            except Exception:
                out = call_model(FALLBACK_URL, raw)   # graceful degradation
                fallback = True
            if not fallback:                # never cache a degraded answer
                CACHE[key] = out
        u = out["usage"]
        LOG.append({"id": rid, "cache": hit, "fallback": fallback,
                    "tok_in": u["prompt_tokens"], "tok_out": u["completion_tokens"],
                    "ms": round((clock() - start) * 1000)})
        body = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Request-Id", rid)
        self.end_headers()
        self.wfile.write(body)

# ---- stubs standing in for the model servers (not part of the gateway) ----
def model_stub(reply, tok_out, latency, fail=False):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            NOW[0] += latency
            if fail:
                self.send_response(500); self.end_headers(); return
            body = json.dumps({"choices": [{"message":
                    {"role": "assistant", "content": reply}}],
                    "usage": {"prompt_tokens": 42, "completion_tokens": tok_out}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
    return H

class Reusable(socketserver.TCPServer):
    allow_reuse_address = True

def serve(port, handler):
    httpd = Reusable(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd

serve(18531, model_stub("services: ssh, http", 12, 0.05))
serve(18532, model_stub("(degraded) retry later", 6, 0.05))
serve(18530, Gateway)

def send(text):
    payload = json.dumps({"model": "local",
                          "messages": [{"role": "user", "content": text}]}).encode()
    call_model("http://127.0.0.1:18530/v1/chat/completions", payload)

send("scan 10.10.10.5")     # 1: miss -> primary
send("scan 10.10.10.5")     # 2: exact cache hit
MODEL_URL = "http://127.0.0.1:18539/v1/chat/completions"  # primary now down
send("scan 10.10.10.6")     # 3: miss -> primary refused -> fallback

for r in LOG:
    print(f"{r['id']} cache={str(r['cache']):5} fallback={str(r['fallback']):5} "
          f"tok_in={r['tok_in']} tok_out={r['tok_out']} ms={r['ms']}")
```

```text
req-1 cache=False fallback=False tok_in=42 tok_out=12 ms=50
req-2 cache=True  fallback=False tok_in=42 tok_out=12 ms=1
req-3 cache=False fallback=True  tok_in=42 tok_out=6 ms=50
```

Request 1 misses the cache and hits the primary (50 ms, 12 output tokens). Request 2 is identical, so the exact cache returns it in 1 ms with no model call. The log still shows the cached response's token counts, so cost accounting must check the `cache` flag and *not* charge the model cost twice. Request 3 is new, the primary is down (its URL now points at a dead port), so the connection is refused and the gateway transparently falls back to the smaller server. The fallback answer is not cached, so the next identical request tries the primary again. A real server would stream tokens back ([§17](#17-streaming-and-async-jobs)), expose `/metrics`, and send each `LOG` record to a tracing backend ([§16](#16-observability)).

---

## 16. Observability

### 16.1 What to log per request

Log a structured record for every request — enough to reconstruct and bill it, never the raw secrets:

| Field | Why |
|---|---|
| request ID, timestamp, tenant/user | correlation, rate limiting |
| prompt version, model, params | reproducibility, A/B attribution |
| tokens in / out | cost, capacity |
| TTFT, total latency | SLOs, [§16.3](#163-percentiles) |
| cost (computed) | spend tracking |
| cache hit, fallback used, route taken | efficiency, debugging |
| tool calls made | agent behaviour ([52](52-ai-agents.md)) |
| guardrail verdicts | safety ([56](56-ai-safety.md)) |
| user feedback (thumbs, edit, retry) | the flywheel ([§18.4](#184-the-data-flywheel)) |

### 16.2 Tracing a RAG or agent request

One request fans out into steps; a trace is a tree of **spans**, each timed, so you can see where the latency and cost went:

```text
request  [2100 ms, $0.0012]
├─ retrieve        [120 ms]   embed query + vector search (k=8)
├─ rerank          [ 60 ms]   cross-encoder over 8 chunks
├─ prompt assemble [  5 ms]   template v7 + 4 chunks
├─ model call      [1850 ms]  TTFT 400 ms, 180 output tokens
│  └─ (tool) nmap  [  –   ]   agent step, logged separately
└─ guardrail out   [ 65 ms]   PII + scope check
```

Without spans, "the request was slow" is unactionable; with them you see it was the model call, or the reranker, or a slow retrieval — and fix the right thing.

### 16.3 Percentiles

Averages hide the tail. Report p50/p95/p99 latency. The **nearest-rank** method: sort the values, and the p-th percentile is the value at rank \(\lceil \tfrac{p}{100} N \rceil\) (1-based).

```python
import math

def percentile(values, p):
    s = sorted(values)
    rank = math.ceil(p / 100 * len(s))   # nearest-rank method
    return s[rank - 1]

latencies = [120, 135, 140, 150, 155, 160, 165, 170, 175, 180,
             185, 190, 200, 210, 220, 240, 260, 300, 420, 850]  # ms, n=20
for p in (50, 95, 99):
    print(f"p{p} = {percentile(latencies, p)} ms")
```

```text
p50 = 180 ms
p95 = 420 ms
p99 = 850 ms
```

!!! example "On paper"

    Nearest-rank on the 20 sorted samples above. For p95, \(N=20\):

    \[
    \text{rank} = \lceil 0.95 \times 20 \rceil = 19
    \]

    The 19th value is 420 ms. The mean of this list is ~231 ms, but 1 request in 20 takes 850 ms — the p99. Users feel the p99, not the mean, so alert on the tail.

Alert on **percentiles crossing an SLO**, error rate, cost per hour, cache hit rate and queue depth — not on single slow requests.

### 16.4 Privacy in logs

Prompts and outputs often contain PII, secrets or customer data. Before logging: **redact** (mask emails, keys, card numbers; [56](56-ai-safety.md)), set a **retention** window and delete past it, restrict who can read logs, and let tenants opt out of content logging while you keep the metadata (tokens, latency, cost). Logging the full prompt "for debugging" forever is how an LLM feature becomes a data-breach surface.

---

## 17. Streaming and async jobs

- **Streaming to the client.** Decode is slow, so stream tokens as they are produced (Server-Sent Events; [50](50-ai-prompting-apis.md)). The user sees the first token at TTFT instead of waiting for the whole answer — the same wall-clock, a far better experience. The gateway must pass the stream through, not buffer it.
- **Async / batch jobs for offline work.** Not everything is interactive. Nightly summarisation, bulk extraction, re-embedding a corpus, backfilling evals — run these as queued batch jobs at high batch size, off the latency-critical path. Throughput, not latency, is the goal, so pack the batches full and schedule them when the serving fleet is idle.

---

## 18. Versioning, rollouts, SLOs and the data flywheel

### 18.1 Version everything

An LLM feature's behaviour depends on three artifacts, and all three must be versioned and logged per request so you can attribute a change: the **prompt** (template v7, not an inline string), the **model** (base + adapter version), and the **retrieval index** (which corpus snapshot and embedding model). A silent prompt edit is a deploy; treat it like one.

### 18.2 Rollouts

Ship changes the way you ship any risky code ([54](54-ai-evaluation.md)):

- **Shadow.** Run the new version on real traffic without showing users its output; compare offline. Zero user risk.
- **Canary.** Route a small slice (1–5%) to the new version, watch the metrics and SLOs, widen if healthy.
- **A/B.** Split traffic and measure the outcome metric (task success, thumbs-up, conversion) with enough samples to be significant ([54](54-ai-evaluation.md)).

### 18.3 SLOs and error budgets

An SLO is a target on a user-visible metric over a window; the **error budget** is what is left over — the amount of failure you are allowed before you stop shipping features and fix reliability.

\[
\text{error budget} = (1 - \text{SLO}) \times \text{total}.
\]

!!! example "On paper"

    An SLO of 99.5% successful responses over 2M requests/month:

    \[
    (1 - 0.995) \times 2\times10^{6} = 10{,}000
    \]

    You may fail 10,000 requests a month. For an availability SLO of 99.9% over a 30-day month:

    \[
    (1 - 0.999) \times 30 \times 24 \times 60 = 43.2
    \]

    43.2 minutes of downtime budget. Spend the budget on rollouts; when it is gone, freeze features and fix stability.

### 18.4 The data flywheel

Production is your best source of training and eval data. The loop:

```text
production logs ──► find failures (feedback, low scores, retries)
      ▲                        │
      │                        ▼
   ship fix ◄── eval ◄── label the failures ◄── add to golden set
```

Real failures become labelled examples; labelled examples become eval cases ([54](54-ai-evaluation.md)) and fine-tuning data ([53](53-ai-fine-tuning.md)); the fix is measured on the eval and shipped; the next failures are rarer and different. This flywheel — not a bigger model — is what makes a feature improve month over month.

---

## 19. A production readiness checklist

- [ ] Gateway with request IDs, auth, rate limits and timeouts.
- [ ] Prompt, model and index versions logged per request.
- [ ] Structured per-request logs with tokens, latency and cost; PII redacted; retention set.
- [ ] Traces (spans) for multi-step RAG/agent requests.
- [ ] p50/p95/p99 latency, error rate, cost and cache-hit dashboards with SLO alerts.
- [ ] Timeouts, retries with backoff, a fallback model and a circuit breaker.
- [ ] Exact (and maybe semantic) response cache; prefix caching for the system/tool block.
- [ ] Streaming to the client; batch jobs off the interactive path.
- [ ] Capacity sized on output tokens/s with headroom; load tested.
- [ ] A cost model and an API-vs-self-host decision with your real prices.
- [ ] Rollout plan (shadow → canary → A/B) tied to an eval gate.
- [ ] SLOs, an error budget, and a flywheel from logs back into evals.

---

## 20. Key takeaways

- An LLM feature is a pipeline — gateway, cache, retrieval, router, model servers, guardrails, logging, evals — and most of the engineering is the layers around the model.
- Prefill is compute-bound; decode is memory-bandwidth-bound. Decode tops out at bandwidth / bytes-read-per-token: ~24 tok/s for Qwen3-14B 4-bit on an M1 Pro, ~209 tok/s for an 8B fp16 on an H100.
- Batching amortises the weight read across requests (throughput up, latency slightly up); the KV cache limits concurrency; paged attention, quantization and speculative decoding all push the numbers further.
- Cost is dominated by output tokens. Caching, routing/cascades and the API-vs-self-host-with-utilization math decide the bill; self-host wins only on steady high volume.
- Ship like any risky service: version prompts/models/indexes, roll out via shadow/canary/A/B against an eval gate, run on SLOs and an error budget, and close the data flywheel from production logs back into evals.

---

## 21. Exercises

!!! question "Exercise 21.1"

    A model has 24 GB of fp16 weights and serves on a GPU with 2 TB/s bandwidth. What is the single-stream decode roofline, and why is the real rate lower?

??? success "Answer"

    \(2000\ \text{GB/s} / 24\ \text{GB} \approx 83\) tok/s roofline. The real rate is lower (~60–70%) because each step also reads the growing KV cache, and Python/kernels/launch overhead eat bandwidth and add latency the roofline ignores. You beat it only by reading fewer bytes (quantize) or using faster memory.

!!! question "Exercise 21.2"

    Qwen3-14B bf16 KV cache is 160 KiB/token. A server has 40 GB free for KV after weights and runtime. How many concurrent 8K-token requests fit?

??? success "Answer"

    Per 8K request: \(8192 \times 163{,}840\ \text{B} \approx 1.34\ \text{GB}\) ([02 §3.5](02-hardware.md) lists 1.34 GB at 8K). \(40 / 1.34 \approx 29.9\), so 29 full 8K requests fit (about 30) — and fewer as each conversation grows past 8K.

!!! question "Exercise 21.3"

    Speculative decoding with acceptance \(\alpha = 0.9\) and \(k = 3\) draft tokens. Expected tokens per big-model step?

??? success "Answer"

    \(E = (1 - 0.9^{4})/(1 - 0.9) = (1 - 0.6561)/0.1 = 0.3439/0.1 = 3.439\). About 3.44 tokens per big-model pass versus 1, a ~3.4× speedup in big-model forwards (minus the draft cost). Higher \(\alpha\) is why a well-matched draft model pays off.

!!! question "Exercise 21.4"

    Assuming \$0.50/1M input and \$1.50/1M output tokens, what does a request with 2,000 input and 500 output tokens cost? Which part dominates?

??? success "Answer"

    Input: \(2000/10^6 \times 0.50 = 0.0010\). Output: \(500/10^6 \times 1.50 = 0.00075\). Total \$0.00175. Input dominates here because the prompt is large; but output is 3× the per-token price, so for a short prompt and long answer output dominates. Both levers matter: trim the prompt (prefix caching, RAG chunk limits) and cap `max_tokens`.

!!! question "Exercise 21.5"

    You rent one GPU at \$3/hour and your API cost is \$0.0007/request. The GPU can serve 20M requests/month at full load. What monthly volume and what utilization make self-hosting break even?

??? success "Answer"

    Fixed = \(3 \times 24 \times 30 = \$2{,}160\)/month. Break-even volume \(N^{*} = 2160/0.0007 \approx 3.09\text{M}\) requests. Break-even utilization \(u^{*} = 2160/(20\times10^{6} \times 0.0007) = 2160/14{,}000 \approx 0.154\), so ~15.4% utilization. Below that, the idle GPU costs more per request than the API.

!!! question "Exercise 21.6"

    A cascade keeps 80% of queries on a small model (accuracy 0.85) and escalates 20% to a large model (accuracy 0.96). Small costs \$0.0001, large \$0.0012. Expected accuracy and cost?

??? success "Answer"

    Accuracy \(= 0.8 \times 0.85 + 0.2 \times 0.96 = 0.68 + 0.192 = 0.872\). Cost \(= 0.0001 + 0.2 \times 0.0012 = 0.0001 + 0.00024 = 0.00034\). So 87.2% accuracy at \$0.00034/request versus 96% at \$0.0012 for large-only (if the large model scores 0.96 everywhere) — a 72% cost cut for ~9 points of accuracy. The escalation rule must send the genuinely hard queries up or the accuracy term collapses.

!!! question "Exercise 21.7"

    Compute p50, p90 and p99 by nearest-rank for these 10 latencies (ms): 90, 95, 100, 110, 120, 130, 150, 180, 250, 600.

??? success "Answer"

    Sorted already, \(N=10\). p50: \(\lceil 0.5 \times 10 \rceil = 5\) → 120 ms. p90: \(\lceil 0.9 \times 10 \rceil = 9\) → 250 ms. p99: \(\lceil 0.99 \times 10 \rceil = 10\) → 600 ms. The mean is 182.5 ms, below the p90 — the two slow requests drag the tail far past the typical experience.

!!! question "Exercise 21.8"

    Your feature has a 99.9% success SLO over 5M requests/month. How many failures are in budget? If a bad prompt deploy fails 0.3% of requests for a day (at ~167k requests/day), did it blow the budget?

??? success "Answer"

    Budget \(= (1 - 0.999) \times 5\times10^{6} = 5{,}000\) failures/month. The bad deploy failed \(0.003 \times 167{,}000 \approx 501\) requests in that day — about 10% of the monthly budget in one day. One such incident is survivable; about ten such days in a month (\(10\times501\approx5{,}010\)) exhaust the budget and should freeze feature work until reliability is fixed.

---

Next: [56 — Safety and security](56-ai-safety.md) covers the threat model, prompt injection, guardrails and the OWASP LLM Top 10 that the guardrail and logging layers here exist to enforce.
