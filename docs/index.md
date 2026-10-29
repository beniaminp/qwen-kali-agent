---
title: Qwen & Gemma Handbook
hide:
  - navigation
  - toc
---

<div class="gb-hero">
  <div class="gb-hero__main">
    <p class="gb-eyebrow"><span class="gb-dot"></span> Open handbook · Oct 2026</p>
    <h1 class="gb-hero__title">Build, fine-tune and ship <span class="gb-gradient">Qwen &amp; Gemma</span> agents.</h1>
    <p class="gb-hero__lead">A source-backed, practical handbook for local model training on a 32 GB Apple M1 Pro — agents for authorized Kali Linux work, Java/Quarkus services and React/Tailwind front-ends.</p>
    <div class="gb-actions">
      <a class="gb-btn gb-btn--primary" href="00-start/">Start here <span aria-hidden="true">→</span></a>
      <a class="gb-btn" href="17-finetune-2026-update/">2026 LoRA recipe</a>
      <a class="gb-btn gb-btn--ghost" href="https://github.com/beniaminp/qwen-kali-agent" target="_blank" rel="noopener">GitHub</a>
    </div>
  </div>
  <div class="gb-term" role="group" aria-label="Quick start commands">
    <div class="gb-term__bar" aria-hidden="true"><span></span><span></span><span></span><em>qwen-kali-agent — make</em></div>
    <div class="gb-term__body">
      <span class="gb-term__note"># 1 · environment and memory headroom</span>
      <span><b aria-hidden="true">$</b> make setup &amp;&amp; make check</span>
      <span class="gb-term__note"># 2 · validate the training splits</span>
      <span><b aria-hidden="true">$</b> make data-check</span>
      <span class="gb-term__note"># 3 · 8B smoke run, then the 14B adapter</span>
      <span><b aria-hidden="true">$</b> make train-8b</span>
      <span><b aria-hidden="true">$</b> make train-14b</span>
      <span class="gb-term__note"># 4 · score it on the golden set</span>
      <span><b aria-hidden="true">$</b> make eval<i class="gb-term__cursor" aria-hidden="true"></i></span>
    </div>
  </div>
  <div class="gb-stats">
    <div class="gb-stat"><strong>3</strong><span>specializations</span></div>
    <div class="gb-stat"><strong>2</strong><span>model families</span></div>
    <div class="gb-stat"><strong>64</strong><span>chapters</span></div>
    <div class="gb-stat"><strong>32 GB</strong><span>target hardware</span></div>
  </div>
</div>

## Choose your track

<div class="gb-track-grid">
  <a class="gb-track gb-track--kali" href="01-landscape/">
    <div class="gb-track__top"><span class="gb-pill">Agent</span><span class="gb-track__arrow">→</span></div>
    <h3>Kali &amp; network security</h3>
    <p>Qwen3-14B, tool calling, multi-turn trajectories, scope policy and an execution-verified Kali loop.</p>
    <ul>
      <li>Qwen3-14B / 8B smoke run</li>
      <li>BFCL, KaliBench, Cybench</li>
      <li>MCP + policy layer</li>
    </ul>
    <span class="gb-track__cta">Start the track</span>
  </a>
  <a class="gb-track gb-track--java" href="18-java-quarkus/">
    <div class="gb-track__top"><span class="gb-pill">Code</span><span class="gb-track__arrow">→</span></div>
    <h3>Java &amp; Quarkus</h3>
    <p>Java 25 LTS, Quarkus 3.40 LTS, hexagonal architecture and SOLID — verified with Maven and ArchUnit.</p>
    <ul>
      <li>Qwen2.5-Coder-14B</li>
      <li>repository + commit data</li>
      <li>compile and test gates</li>
    </ul>
    <span class="gb-track__cta">Start the track</span>
  </a>
  <a class="gb-track gb-track--react" href="22-react-tailwind/">
    <div class="gb-track__top"><span class="gb-pill">UI</span><span class="gb-track__arrow">→</span></div>
    <h3>React &amp; Tailwind</h3>
    <p>React 19.3, Tailwind 4.3, Vite 8 and accessible design systems — checked by TypeScript, Playwright and axe.</p>
    <ul>
      <li>Qwen2.5-Coder-14B</li>
      <li>components, pages, a11y</li>
      <li>browser-verified output</li>
    </ul>
    <span class="gb-track__cta">Start the track</span>
  </a>
</div>

<div class="gb-banner">
  <div>
    <span class="gb-pill gb-pill--gemma">Alternative</span>
    <h3>Gemma 4 12B — the second stack</h3>
    <p>Apache-2.0, 11.95B dense, 256K context, natively multimodal. Train separate adapters and compare with Qwen on the same golden set.</p>
  </div>
  <a class="gb-btn" href="27-gemma4-12b/">Explore Gemma 4</a>
</div>

<div class="gb-banner gb-banner--distill">
  <div>
    <span class="gb-pill gb-pill--distill">Compress</span>
    <h3>Gemma 12B distillation — teacher, student, or both</h3>
    <p>Distil a stronger teacher into 12B, compress 12B into E4B/E2B, or use 12B as a data teacher. Includes the KL math, a GPU logit-KD trainer and the 32 GB sequence-KD path.</p>
  </div>
  <a class="gb-btn" href="39-gemma-distillation/">Open the track</a>
</div>

## The core path

<div class="gb-path">
  <a class="gb-path__item" href="01-landscape/">
    <span class="gb-path__num">01</span>
    <strong>Pick the model</strong>
    <span>What fits, what does not, and why Qwen3-14B is the local default.</span>
  </a>
  <a class="gb-path__item" href="02-hardware/">
    <span class="gb-path__num">02</span>
    <strong>Check the memory</strong>
    <span>Unified-memory math, sequence length, batch and QLoRA peak.</span>
  </a>
  <a class="gb-path__item" href="04-data-strategy/">
    <span class="gb-path__num">03</span>
    <strong>Build the data</strong>
    <span>Capability mix, trajectories, negative examples and quality gates.</span>
  </a>
  <a class="gb-path__item" href="08-finetuning/">
    <span class="gb-path__num">04</span>
    <strong>Train the adapter</strong>
    <span>MLX runbook, flags, checkpoints and loss reading.</span>
  </a>
  <a class="gb-path__item" href="09-evaluation/">
    <span class="gb-path__num">05</span>
    <strong>Prove it works</strong>
    <span>Benchmarks, private golden set and release gates.</span>
  </a>
  <a class="gb-path__item" href="11-kali-agent/">
    <span class="gb-path__num">06</span>
    <strong>Ship with guardrails</strong>
    <span>Serving, MCP, scope enforcement, audit and kill switch.</span>
  </a>
</div>

## Go deeper

<div class="gb-mini-grid">
  <a class="gb-mini" href="17-finetune-2026-update/">
    <span class="gb-mini__icon">✦</span>
    <strong>2026 LoRA recipe</strong>
    <span>All-layer LoRA, the MLX scale trap and one-epoch training.</span>
  </a>
  <a class="gb-mini" href="26-one-model-for-all/">
    <span class="gb-mini__icon">◇</span>
    <strong>One model for all</strong>
    <span>Qwen3-14B plus three adapters, or one mixed adapter.</span>
  </a>
  <a class="gb-mini" href="39-gemma-distillation/">
    <span class="gb-mini__icon">⇣</span>
    <strong>Distillation</strong>
    <span>Teacher logits, sequence KD and 12B → E4B/E2B compression.</span>
  </a>
  <a class="gb-mini" href="35-from-zero/">
    <span class="gb-mini__icon">↑</span>
    <strong>From zero</strong>
    <span>Train a small model you can actually afford.</span>
  </a>
  <a class="gb-mini" href="32-mathematics/">
    <span class="gb-mini__icon">∑</span>
    <strong>Mathematics</strong>
    <span>Attention, RoPE, GQA, MoE, Gated DeltaNet, FLOPs and cache.</span>
  </a>
  <a class="gb-mini" href="37-architecture/">
    <span class="gb-mini__icon">▦</span>
    <strong>Architecture anatomy</strong>
    <span>From config.json to tensor shapes, with diagrams.</span>
  </a>
  <a class="gb-mini" href="38-derivations/">
    <span class="gb-mini__icon">ƒ</span>
    <strong>Derivations</strong>
    <span>Softmax, attention backward, RMSNorm, RoPE and GDN proofs.</span>
  </a>
  <a class="gb-mini" href="43-math-examples/">
    <span class="gb-mini__icon">✓</span>
    <strong>Worked examples</strong>
    <span>Every formula on a toy model, checked by a 69-value script.</span>
  </a>
</div>

## What is new

<div class="gb-new">
  <div class="gb-new__item">
    <span class="gb-new__date">29 Oct 2026</span>
    <strong><a href="62-ai-agent-integration/">Agents, integrated and applied</a></strong>
    <p>Eight ways to wire an agent into a product, the run contract and a runnable agent service, then <a href="63-ai-agent-applications/">twelve real applications</a> end to end: support, coding in CI, data analysis, incident response, documents, research, sales, e-commerce, helpdesk, pipelines and QA.</p>
  </div>
  <div class="gb-new__item">
    <span class="gb-new__date">28 Oct 2026</span>
    <strong><a href="45-ai-engineer-path/">New menu: AI Engineer</a></strong>
    <p>What it takes to move from software engineering to AI engineering: 19 chapters from vectors and gradients to data preparation, RAG, agents, MCP, agent frameworks, agent integration and real applications, evals, serving and interviews, with on-paper examples and tested code.</p>
  </div>
  <div class="gb-new__item">
    <span class="gb-new__date">28 Oct 2026</span>
    <strong><a href="32-mathematics/">Every formula, on paper</a></strong>
    <p>129 short examples you can redo with a pencil, one after every formula in chapters 32–42, each number checked by <code>make math-check</code>.</p>
  </div>
  <div class="gb-new__item">
    <span class="gb-new__date">28 Oct 2026</span>
    <strong><a href="43-math-examples/#15-flops-and-wall-time">H100 budget corrected</a></strong>
    <p>Dense BF16 is ~989 TFLOP/s (1,979 is the sparse figure): Qwen3-14B on 36T tokens is ~2.24M GPU-hours, ~9.3 days and ~$5.6M, not double that.</p>
  </div>
  <div class="gb-new__item">
    <span class="gb-new__date">27 Oct 2026</span>
    <strong><a href="37-architecture/#4-the-edge-models-gemma-4-e2b-and-e4b">Gemma 4 E2B and E4B, explained</a></strong>
    <p>Per-layer embeddings, KV sharing and the double-wide MLP, read from the checkpoints — and why the E4B cache at 128K outgrows the 12B’s.</p>
  </div>
  <div class="gb-new__item">
    <span class="gb-new__date">27 Oct 2026</span>
    <strong><a href="43-math-examples/">Worked examples, grounded</a></strong>
    <p>Every formula now starts from a real situation — a next token, a weight row, an eval batch — with a corrected Wilson interval and 64 checked values.</p>
  </div>
</div>

---

> **Authorized testing only.** A model is not authorization. Use a written scope, a lab you own, and a policy layer that enforces both.
>
> **A handbook, not a tested product.** Commands were checked against upstream source/docs on 25 Oct 2026, but the MLX runs were not executed on the target Mac. Run the preflight and a smoke test before a multi-day run.

<p class="gb-footer-links">
  <a href="00-start/">Quick start</a>
  <a href="14-references/">References</a>
  <a href="appendix-c-runbook/">Runbook</a>
  <a href="https://github.com/beniaminp/qwen-kali-agent">Repository</a>
</p>
