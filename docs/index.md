---
title: Qwen & Gemma Handbook
hide:
  - navigation
  - toc
---

<div class="gb-hero">
  <p class="gb-eyebrow"><span class="gb-dot"></span> Open handbook · Oct 2026</p>
  <h1 class="gb-hero__title">Build, fine-tune and ship <span class="gb-gradient">Qwen &amp; Gemma</span> agents.</h1>
  <p class="gb-hero__lead">A source-backed, practical handbook for local model training on a 32 GB Apple M1 Pro — agents for authorized Kali Linux work, Java/Quarkus services and React/Tailwind front-ends.</p>
  <div class="gb-actions">
    <a class="gb-btn gb-btn--primary" href="00-start/">Start here <span aria-hidden="true">→</span></a>
    <a class="gb-btn" href="17-finetune-2026-update/">2026 LoRA recipe</a>
    <a class="gb-btn gb-btn--ghost" href="https://github.com/beniaminp/qwen-kali-agent" target="_blank" rel="noopener">GitHub</a>
  </div>
  <div class="gb-stats">
    <div class="gb-stat"><strong>3</strong><span>specializations</span></div>
    <div class="gb-stat"><strong>2</strong><span>model families</span></div>
    <div class="gb-stat"><strong>45</strong><span>chapters</span></div>
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
  <a class="gb-mini" href="26-one-model-for-all/">
    <span class="gb-mini__icon">◇</span>
    <strong>One model for all</strong>
    <span>Qwen3-14B plus three adapters, or one mixed adapter.</span>
  </a>
  <a class="gb-mini" href="35-from-zero/">
    <span class="gb-mini__icon">↑</span>
    <strong>From zero</strong>
    <span>Train a small model you can actually afford.</span>
  </a>
  <a class="gb-mini" href="39-gemma-distillation/">
    <span class="gb-mini__icon">⇣</span>
    <strong>Distillation</strong>
    <span>Teacher logits, sequence KD and 12B → E4B/E2B compression.</span>
  </a>
</div>

## What is new

<div class="gb-new">
  <div class="gb-new__item">
    <span class="gb-new__date">25 Oct 2026</span>
    <strong><a href="37-architecture/">Architecture anatomy and proofs</a></strong>
    <p>Tensor-level diagrams, config-to-shape mapping and full derivations for attention, RoPE, RMSNorm, GDN and quantization.</p>
  </div>
  <div class="gb-new__item">
    <span class="gb-new__date">25 Oct 2026</span>
    <strong><a href="27-gemma4-12b/">Gemma 4 12B track</a></strong>
    <p>A parallel model family with its own configs, datasets, validation, synthetic pipeline and comparison protocol.</p>
  </div>
  <div class="gb-new__item">
    <span class="gb-new__date">25 Oct 2026</span>
    <strong><a href="17-finetune-2026-update/">Corrected LoRA recipe</a></strong>
    <p>All-layer adapters, the MLX <code>scale</code> trap, one-epoch training and new sweep tools.</p>
  </div>
  <div class="gb-new__item">
    <span class="gb-new__date">25 Oct 2026</span>
    <strong><a href="39-gemma-distillation/">Gemma 12B distillation</a></strong>
    <p>A three-direction track: teacher → 12B, 12B → E4B/E2B, and 12B as a data teacher, with sequence and logit KD.</p>
  </div>
  <div class="gb-new__item">
    <span class="gb-new__date">25 Oct 2026</span>
    <strong><a href="43-math-examples/">Math, checked with numbers</a></strong>
    <p>Every formula on a toy model, including the corrected Gemma parameter count, hybrid KV cache and dense-vs-sparse H100 peak.</p>
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
