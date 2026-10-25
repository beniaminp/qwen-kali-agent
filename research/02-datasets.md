# 02 — Datasets for a Qwen3 Kali/agentic LoRA

*Research notes, 25 Oct 2026. Target: Qwen3 8B–14B, 4-bit QLoRA on a 32 GB M1 Pro; authorized pentest/network-security assistant with reliable tool calling and ReAct loops.*

## 0. Method and budget

Every ID was resolved through the HF API and/or dataset card on 25 Oct 2026. Row counts come from the datasets-server `/size` endpoint; token counts are `chars/4` estimates from sampled rows, marked "est."; unreadable sizes are stated. Gated repos return HTTP 401 for both gated and nonexistent IDs, so gated IDs were cross-checked by enumerating the author's public datasets. Unverified claims are marked **UNVERIFIED**.

The binding constraint is tokens, not rows: OpenThoughts-Agent-SFT-100K is 94k rows but ≈2.0B tokens (≈21k/trajectory), and Toucan's SFT subset is ≈497M. Budget **50–120M tokens** on this hardware, and expect most cyber sources to be knowledge corpora rather than workflows.

## 1. General function-calling datasets

| Dataset | Rows | Size / tokens (est.) | License | Shape | Verdict |
|---|---:|---|---|---|---|
| `Salesforce/xlam-function-calling-60k` | 60,000 | 96 MB JSON; ≈23.7M tok | CC-BY-4.0 | `query`, `tools` (JSON string), `answers` (JSON string) | Strong single-turn breadth; **gated**; schemas non-standard |
| `Team-ACE/ToolACE` | 11,300 | 10.8 MB parquet; ≈19.5M tok | Apache-2.0 | `system`, `conversations` (`from`: user/assistant/tool) | Best open multi-turn with tool results; custom call syntax |
| `NousResearch/hermes-function-calling-v1` | 11,578 (5 configs) | 21.7 MB; ≈4.1M tok (func_calling) | Apache-2.0 | `conversations`, `tools`, `category` | Qwen-shaped; small; mixed quality |
| `glaiveai/glaive-function-calling-v2` | 112,960 | 97 MB; ≈75.9M tok | Apache-2.0 | `system`, `chat` strings | 2023-era; trivial; format noise |
| `nvidia/When2Call` | 15,000 SFT; 9,000 pref; 3,652 MCQ | 12.5 MB; ≈5.4M tok | CC-BY-4.0 | `tools`, `messages` | **Essential** for call-vs-ask-vs-abstain |
| `argilla/apigen-function-calling` | 109,402 | 60 MB; ≈44.4M tok | CC-BY-4.0 | `answers`, `query`, `tools`, `origin` | xLAM (60k) + Synth-APIGen (49.4k); needs prep script |
| `Salesforce/APIGen-MT-5k` | 5,000 | 10 MB; ≈26.0M tok | **CC-BY-NC-4.0** | `conversations`, `tools`, `system` | Best multi-turn realism; **non-commercial trap** |
| `Agent-Ark/Toucan-1.5M` | 1,646,546 total; 119,287 SFT | 21.8 GB; SFT ≈497M tok | Apache-2.0 | `messages`, `tools`, `question`, quality scores | Best 2025 set; MCP-centric; huge |
| `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | eval files (`BFCL_v3_*`) | ≈9 MB JSON | Apache-2.0 | questions + `possible_answer` | **Eval only — hold out** |

**xLAM.** The APIGen pipeline verified 60k samples over 3,673 executable APIs through format checking, real execution, and semantic verification ([arXiv 2406.18518](https://arxiv.org/abs/2406.18518)). The flaw is schema shape: `parameters` is a map of `param → {type: "str"|"int"|"List", description}`, with no top-level `"type":"object"` and no `required` array; 940 of 58,105 sampled tool definitions had a non-object top-level type. It is single-turn, has no tool responses, and ~53% of rows contain 2+ calls (median 3 available tools; 26% list 4–5 candidates). Convert to JSON Schema and invent no observations. The canonical repo is gated; mirrors exist but some relicense it MIT/Apache — **laundering; use the CC-BY-4.0 origin**.

**ToolACE.** 26,507-API pool, dual rule/model verification, and a formal thinking process ([arXiv 2409.00920](https://arxiv.org/abs/2409.00920)). The entire tool list sits in `system`, so rows are long. Calls are **not JSON**: `[Market Trends API(trend_type="MARKET_INDEXES")]`; tool results are `from:"tool"` JSON blobs. Tool names contain spaces (`"Get Manager Placeholder Image"`), and schemas use `"type":"dict"`. It has genuine multi-turn observation loops and chit-chat turns, which help "don't call a tool" behaviour — but every row needs parsing, renaming (`[A-Za-z0-9_-]`), and schema rewriting.

**Hermes.** Five configs (1,893 single-turn; 1,893 multi-turn; 5,209 Glaive; 1,342 + 1,241 JSON-mode). Format is already Qwen-adjacent: `<tools>[...]</tools>` in system, `<tool_call>{"name","arguments"}</tool_call>`, `<tool_response>`. The `glaive_*` config inherits Glaive's flaws; the `json_mode_*` configs teach structured output, not tools. Small but cheap and clean.

**Glaive.** `system` is a `"SYSTEM: ..."` string, `chat` uses `USER:`/`ASSISTANT:` with `<functioncall> {"name":..., "arguments": '{"k":"v"}'}` (arguments is a **string**) and `FUNCTION RESPONSE:` turns. Single-turn bias, trivial calls, inconsistent refusal/no-call behaviour, and occasional arithmetic errors inside arguments. Use only as format seasoning after filtering.

**When2Call.** Explicitly benchmarks "when to call, when to ask a follow-up, when to admit the tools cannot answer" ([arXiv 2504.18851](https://arxiv.org/abs/2504.18851)). This is the single most relevant public set for our scope/authorization behaviour, and it is CC-BY-4.0 with commercial use stated on the card. Caveat: synthetic and benchmark-shaped; add our own Kali-specific scope dialogues.

**APIGen-MT-5k.** Simulated human–agent interplay with clarification questions, `function_call`/`observation` turns, and an explicit `think` tool call ([arXiv 2504.03601](https://arxiv.org/abs/2504.03601)). 5.2k tokens/row — the best public multi-turn realism, but **CC-BY-NC-4.0**, so it can inform evaluation and internal experiments only.

**Toucan-1.5M.** 495 real MCP servers, 2,000+ tools, trajectories from Qwen3-32B, Kimi-K2 and GPT-OSS-120B, with LLM-judge fields and an `irrelevant` subset that is effectively negative/no-call data ([arXiv 2510.01179](https://arxiv.org/abs/2510.01179)). Apache-2.0. Start from the 119k-row SFT config; filter to the top judge bands (`question_quality_assessment`, `response_quality_assessment`) and trajectories under 6k tokens.

**BFCL.** Not training data. The repo ships `BFCL_v3_*` questions plus `possible_answer/`. Its Apache-2.0 licence does not make it safe to train on: it is the benchmark we will report, so any overlap invalidates results.

## 2. Agentic / terminal / software-agent trajectories

| Dataset | Rows | Size / tokens (est.) | License | What it teaches | Recovery? |
|---|---:|---|---|---|---|
| `SWE-Gym/SWE-Gym` | 2,438 tasks | 43.6 MB | MIT | Real Python issue-fixing with runnable tests | Yes (test failures) |
| `SWE-Gym/OpenHands-SFT-Trajectories` | 491 | 10.4 MB | MIT | OpenHands SFT traces | Some |
| `SWE-bench/SWE-smith` | 59,136 instances | 278 MB | MIT | Synthetic bugs across 128 repos | Yes (failing tests) |
| `SWE-bench/SWE-smith-trajectories` | 76,002 (tool/xml/ticks) | 3.17 GB | MIT | Full agent traces, 3 formats | **Yes** |
| `open-thoughts/OpenThoughts-Agent-SFT-100K` | 94,334 | 1.75 GB; ≈2.0B tok | Apache-2.0 | Terminal/code agent trajectories, Qwen3-8B base | Yes |
| `open-thoughts/AgentTrove` | 1,696,847 | 19.6 GB | Apache-2.0 | 219 sources, unified terminus-2 traces | Mixed |
| `nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1` | 31,111 | 1.27 GB; ≈272M tok | CC-BY-4.0 | Terminal pivots/recovery, bash tool | **Yes** |
| `nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1` | 96,968 | 1.56 GB; ≈395M tok | CC-BY-4.0 | 838 domains, conversational tool use | Some |
| `Agent-Ark/Toucan-1.5M` | see §1 | — | Apache-2.0 | MCP multi-tool, multi-turn | Some |

**Which teach error recovery and long horizons.** SWE-smith is the strongest open signal: 50k+ injected bugs, failing tests, and 76k full trajectories in three serializations (`tool`, `xml`, `ticks`), MIT ([arXiv 2504.21798](https://arxiv.org/abs/2504.21798)). SWE-Gym adds 2,438 real tasks and 491 curated OpenHands traces ([arXiv 2412.21139](https://arxiv.org/abs/2412.21139)). Nemotron Terminal Pivot is the most directly transferable to a shell agent: 31k trajectories built around recovery/pivot behaviour, `bash_command` as the tool, terminus-2 prompts.

**Format warning.** OpenThoughts, AgentTrove and Nemotron terminal use the **terminus-2 JSON convention**: `{"analysis": "...", "plan": "...", "commands": [{"keystrokes": ...}]}`, not OpenAI `tool_calls`. Convert `commands[]` into one `bash` call per command, move `analysis`/`plan` to `reasoning_content`, and keep environment output in `role:"tool"`. Do not train raw terminus-2 if you also want native `<tool_call>` reliability — the formats compete.

**ToolBench / ToolLLM.** 16,464 RapidAPI tools and DFS-generated solution paths ([arXiv 2307.16789](https://arxiv.org/abs/2307.16789)). The canonical HF repo does not resolve publicly (401; author listing empty), so use the GitHub release or a verified mirror — **UNVERIFIED** as an HF source in 2026. Largely superseded, and its API data is stale.

**TheAgentCompany** is a benchmark, not SFT data; only `TheAgentCompany/ubuntu_os_images` is public. `OpenHands/OpenHands-Trajectories` and `SWE-Gym/SWE-Gym-OpenHands-Trajectories` do not resolve publicly; the public SWE-Gym trajectory repo is `SWE-Gym/OpenHands-SFT-Trajectories`. **UNVERIFIED.**

## 3. Cybersecurity / pentest datasets

| Dataset | Rows | License | Type | Usability |
|---|---:|---|---|---|
| `RISys-Lab/KaliBench` | 8,504 (3,504 train / 5,000 test) | CC-BY-4.0 | NL→Kali command, structured args | **High for CLI fidelity; test split must be held out** |
| `Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset` | 53,201 | Apache-2.0 | Defensive Q&A (system/user/assistant) | High for domain + refusal policy |
| `theelderemo/pentesting-explanations` | 5,906 default (+3,228 HackTricks, +2,678 MITRE configs) | Apache-2.0 (claimed) | MCQ + explanation + think trace | Medium; convert to open QA |
| `7h3-R3v3n4n7/pentest-agent-dataset-chatml` | 322,511 | Apache-2.0 (claimed) | CVE FAQ, 2-turn | **Low** — 21.5% junk, 0.4% code |
| `iselabvn/Kali-tool-calling` | 790 | Apache-2.0 | Single-turn tool call + reasoning | High format, tiny |
| `0dAI/PentestingCommandLogic` | 10,188 | Apache-2.0 | Instruction/response pairs | **Junk in sample** |
| `suryanshp1/kali-linux-pentesting-data` | 343 | none | Trivial nmap Q&A | Low |
| `WNT3D/Ultimate-Offensive-Red-Team` | 25,620 (HF) | MIT | instruction/input/output | **Low** — outputs echo inputs |
| `yatin-superintelligence/White-Hat-Security-Agent-Prompts-600K` | 596,295 | CC-BY-4.0 | Prompt corpus, **no answers** | Not SFT as-is |
| `preemware/pentesting-eval` | 241 | Apache-2.0 | MCQ eval | Eval only |
| `resk-fr/pentesting-for-agents` | 54 | none | Synthetic OpenSSH/Redis scenarios | Tiny; inspect |
| `whysue/agent-real-pentest-trajectory` | 2.3 GB ShareGPT + 3.8 GB alpaca | MIT | files named `artex_*`; no card | **UNVERIFIED provenance** |
| `justinwangx/CTFtime` | 18,013 chunks | none | `text_chunk` only; likely scraped writeups | **Contamination/legal risk** |
| `ethz-spylab/ctf-satml24` | 137,151 | MIT | AI-control competition attack logs | Not pentest |
| `alibaba-AAIG/alibaba-ctf` | 87 tasks | Apache-2.0 | Docker CTF benchmark | Eval only |
| `WhiteRabbitNeo/WRN-Chapter-1`, `-2` | gated | "other" | Cyber Q&A | **Avoid** (terms unreadable) |
| `CyberNative/CyberSecurityEval` | 500 | none | MCQ eval | Eval |
| `s2e-lab/SecurityEval` | 121 | none | MCQ eval | Eval |

**KaliBench is the best CLI set and the weakest attack set.** 8,504 query–command pairs, 1,642 tools, 23 capability dimensions, documentation-grounded generation, multi-stage verification (LLM, sandboxed execution, human review); NeurIPS 2026 D&B ([arXiv 2610.02206](https://arxiv.org/abs/2610.02206)). I downloaded both splits: **0 exact query overlap**; 3,504 vs 5,000 unique queries; 962 vs 1,642 tools. But only **32 train rows (0.9%)** use a canonical offensive tool (nmap, sqlmap, hydra, msfconsole); the top tools are `gdb`, `mailer`, `airmon-ng`, `pwsh`, `pedis`, `ld.lld`, `rva2ofs`; 20.1% of commands have **no flags**; 5.5% are version-style; the average command is 5.13 tokens; 157 of 962 tools appear once; only 31 use `sudo`. It teaches flag/argument fidelity on obscure util-linux binaries, not recon or exploitation sequencing. Both splits are public, so the test set is not a clean holdout for models trained after its release.

**The 322k-row "pentest agent" set is a CVE FAQ.** All 322,511 rows scanned: 310,428 mention a CVE, only 1,343 (0.4%) contain a fenced code block, mean answer length is 59 words, only 714 prompts mention "exploit", and **69,480 rows (21.5%) are `** RESERVED **` or rejected-CVE placeholders**. There are 310,306 unique CVEs and zero multi-turn rows. Usable only as a filtered CVE subset (drop reserved, require ≥60 words and a reference), never as agentic data.

**Trendyol is clean but defensive.** 53,201 rows, zero exact duplicates, two malformed rows, mean 275 words, 0.3% code fences, 1,628 prompts containing offensive terms. Its uniform system prompt is a strong refusal/scope template but copying it verbatim will over-refuse; paraphrase a shorter policy.

**Junk is common.** `0dAI` answers a MediaWiki SVG question with `fdroid`/`curl root@x00` gibberish, `WNT3D` echoes its input, and `theelderemo` tests recognition, not procedure. `iselabvn/Kali-tool-calling` is the cleanest small set (`reasoning_content` + `tool_calls` for `exec`), but only 790 rows.

**Benchmarks to hold out.** Cybench: 40 professional CTF tasks from 4 competitions, Apache-2.0, Docker + subtasks ([arXiv 2408.08926](https://arxiv.org/abs/2408.08926)). NYU CTF Bench: 200 test + 55 dev CSAW challenges, **GPL-2.0** ([arXiv 2406.05590](https://arxiv.org/abs/2406.05590)). CyberMetric: 80/500/2,000/10,000 MCQ, RAG-generated from NIST/RFCs and human-validated for 200+ hours ([arXiv 2402.07688](https://arxiv.org/abs/2402.07688)); Apache-2.0 via the author's mirror `tihanyin/CyberMetric`. SecEval: 2,000+ MCQ over 9 domains ([XuanwuAI/SecEval](https://github.com/XuanwuAI/SecEval)); no licence in the original, and the RISys mirror is **CC-BY-NC-SA-4.0**.

## 4. Licensing and compliance

| Licence | Datasets | Publishable model? |
|---|---|---|
| Apache-2.0 | ToolACE, Hermes, Glaive, Toucan, OpenThoughts, AgentTrove, Trendyol, Nemotron sets, BFCL, Cybench | Yes, with NOTICE/attribution |
| MIT | SWE-Gym, SWE-smith, WNT3D (provenance caveat) | Yes |
| CC-BY-4.0 | xLAM, When2Call, argilla/apigen, KaliBench, CyberMetric (mirror), yatin | Yes, with attribution |
| CC-BY-NC-4.0 | **APIGen-MT-5k**, Salesforce APIGen-MT-30k | **No** for a commercial/published model |
| CC-BY-NC-SA-4.0 | SecEval mirror | **No**; share-alike compounds |
| GPL-2.0 | **NYU CTF Bench**, ExploitDB | Risky; exclude from a publishable mix |
| "other"/custom | WhiteRabbitNeo WRN, MITRE ATT&CK | Read terms; assume restricted |
| None | justinwangx/CTFtime, suryanshp1, resk-fr, XuanwuAI/SecEval, HackTricks | **No** — all rights reserved by default |

Three traps matter.

1. **Non-commercial contagion.** APIGen-MT-5k is the best multi-turn set and is CC-BY-NC. Keep it in an internal-only ablation, never in the release mix; the same applies to SecEval's NC-SA mirror. Whether NC-trained weights are a "derivative" is legally unsettled — treat them as unsafe to publish (**UNVERIFIED legal conclusion**).
2. **Copyleft data.** NYU CTF Bench is GPL-2.0. Training is probably not distribution; redistributing a derived dataset is, and the model-release question is unresolved. ExploitDB is GPL-2.0, which contaminates `7h3/...` despite its Apache-2.0 tag. `theelderemo` claims Apache-2.0 over HackTricks/MITRE content, and HackTricks has no SPDX licence file (**UNVERIFIED**). Prefer KaliBench/Trendyol for the cyber core.
3. **Licence laundering.** Mirrors relicense xLAM as MIT/Apache; neither can override Salesforce's CC-BY-4.0. Trace every mirror to the origin and record the origin ID.

**Leaked CTF/writeup risk.** Challenge files, flags and writeups are copyrighted by organisers; some flags are still live. `justinwangx/CTFtime` (18,013 writeup chunks, no licence) and `ethz-spylab/ctf-satml24` (AI-control transcripts, not pentest) should not enter a released model. Generate CTF-style reasoning in your own lab; use Apache-2.0 Docker benchmarks for evaluation only.

## 5. Recommended data mix

Token-weighted, ~90M tokens, ~89k rows — inside the 50k–150k "minimal viable" band. Percentages follow the brief's skeleton, adjusted because KaliBench is smaller and more repetitive than expected.

| Bucket | Share | Sources (rows sampled) | Est. tokens |
|---|---:|---|---:|
| General tool calling | 37% | xLAM 25k; Toucan SFT 3k; ToolACE 4k; Hermes 2k; Glaive 2k | 34M |
| Multi-turn / agentic | 18% | APIGen-MT 1.5k (internal only) or Toucan multi-turn; SWE-smith 1.5k; Nemotron Terminal 400; OpenThoughts 100 | 16M |
| Cyber QA + CLI | 20% | KaliBench train 3,504 ×2; Trendyol 12k; theelderemo 3k; 7h3 filtered 8k | 18M |
| Synthetic verified Kali trajectories | 15% | our harness, 6k trajectories × ~2.2k tok | 13.5M |
| Refusal / scope / no-call | 10% | When2Call 9k; curated scope dialogues 5k; negative/no-tool 3k | 9M |

**Minimal viable (~50k rows, ~45M tokens).** Halve every bucket; keep all 3,504 KaliBench train rows (×2), all 9k When2Call, and ≥3k synthetic trajectories; drop Toucan to 1k rows and OpenThoughts to 50 trajectories. One or two epochs.

**Stretch (~250k rows, ~250M tokens).** Toucan 20k, AgentTrove 20k, OpenThoughts 10k, filtered 7h3 (~50k), synthetic 30k, 5% general-chat replay. Roughly 4–6 days of 4-bit QLoRA at seq 4,096 (benchmark tokens/s first).

**Curriculum.** Avoid a strict multi-stage curriculum at this budget; use a **mild easy→hard ramp with a refusal-heavy anneal**. (1) **Warm-up 10%**: single-turn, short schemas — KaliBench, xLAM easy slice, Hermes single-turn — to learn `<tool_call>` syntax and JSON validity. (2) **Bulk 60%**: shuffled multi-turn tool+observation data (ToolACE, Toucan, APIGen-MT), terminal trajectories (SWE-smith, Nemotron) and cyber QA; shuffle by bucket, not by row, so batches mix formats. (3) **Cyber-heavy 20%**: KaliBench ×3, Trendyol, synthetic Kali trajectories, converted theelderemo. (4) **Anneal 10%**: refusal/scope/no-call at low LR plus 5% general-tool replay. Sprinkle ~30% of refusals through stages 1–3 so the model does not become uniformly over-refusing at the end.

**Conversion to Qwen3 + MLX.** Target JSONL: `{"messages":[...], "tools":[OpenAI-format...]}`. Verified against `Qwen/Qwen3-8B/tokenizer_config.json`: tools are JSON-dumped inside `<tools>...</tools>` in the system turn; calls render as `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`; `role:"tool"` messages are wrapped in `<tool_response>` inside a **user** turn (consecutive tool messages are grouped); `reasoning_content` is supported and only the final assistant turn is forced to open ` thinking`. Set `enable_thinking=false` for direct-call samples. `mlx-lm` accepts exactly this shape ([LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)).

Mapping: xLAM `tools` string → list, `parameters` map → `{type:"object", properties, required}`, `answers` → `tool_calls`. ToolACE `[Name(arg="v")]` → `tool_calls` plus tool rename and a split-out `tools` list. Hermes/Glaive: parse `<tool_call>`/`<functioncall>`, map `FUNCTION RESPONSE:` to `role:"tool"`. APIGen-MT: `human→user`, `gpt→assistant`, `function_call→tool_calls`, `observation→tool`, `think`→`reasoning_content`. Terminus-2: `commands[]` → one `bash` call each, `analysis`/`plan` → `reasoning_content`. Train on assistant tokens only (`--mask-prompt`), cap at 6,144–8,192 tokens, and truncate rather than pad.

## 6. Decontamination

Hold out **BFCL (all versions, including the HF v3 files), KaliBench test (5,000), Cybench, CyberMetric, SecEval, NYU CTF, preemware, Alibaba CTF**. Then run four passes:

1. **Normalise**: NFKC, lowercase, strip Markdown fences/HTML, collapse whitespace, keep punctuation and flags.
2. **Exact dedup**: SHA-256 over normalised (user turn + tool names); also dedup normalised `ground_truth_command` for KaliBench-derived rows.
3. **Near-dup**: 5-gram shingles + MinHash LSH (128 permutations, Jaccard ≥0.80), or 64-bit SimHash (Hamming ≤3). Shingle trajectories per turn, not per document, or a shared system prompt will make everything look similar.
4. **Benchmark containment**: 13-gram containment against every benchmark question/answer; flag >0.50, review 200 pairs, drop. For commands, compare canonicalised token sequences against KaliBench test commands (alias-aware).

Record per-source overlap in the manifest. KaliBench train↔test exact overlap is already 0, so any nonzero rate later indicates a pipeline bug. Never "dedup" against a benchmark and keep the row — drop it.

## 7. Quality filtering and cyber-specific red flags

**Structural filters (all sources).** Valid JSON against the declared schema; tool name exists in `tools`; every required argument present; no hallucinated tool; assistant turn non-empty; ≤8k tokens; language is English; no placeholder-only answers ("** RESERVED **", "N/A", "I cannot help with that" as the entire response).

**Cyber red flags — reject or rewrite:**

- **Destructive commands without lab framing**: `rm -rf /`, `mkfs*`, `dd of=/dev/sd*`, fork bombs, `iptables -F` on a remote host. Keep defensive lessons only with explicit "authorized lab" context and a warning.
- **Real third-party targets**: public IPs, real domains, company names in the target slot. Require RFC1918, `example.com`, `.lab`, `.local`, or `<TARGET>` — a safety *and* legal filter.
- **Live credentials/PII**: AWS `AKIA…`, private keys, JWTs, password hashes with usernames, emails. Redact; never train on them.
- **Illegal-targeting language** ("hack <real company>", "bypass auth on <live site>"). Reject and add a scope-refusal counterexample.
- **Exploit code without context**: no CVE, affected version, lab, or mitigation. Keep only with a CVE/MITRE ID and an authorized-lab frame.
- **Weaponisation uplift**: ransomware builders, EDR bypass, credential stealers, phishing kits. Exclude entirely.
- **Fabricated flags**: drop CTF `flag{...}` strings and rows quoting benchmark answers.
- **Over-refusal** is also a failure: "I can't help with that target, but here is how to scope it" must be a *correct* answer, and authorized lab requests must be fulfilled.

**Keep signals.** A non-zero exit followed by a corrected command (recovery), an explicit scope check before the first scan, and a report separating observations from inferences. Those are worth more than another 10k CVE descriptions.

## Key sources

- HF API + dataset cards: `Salesforce/xlam-function-calling-60k`, `Team-ACE/ToolACE`, `NousResearch/hermes-function-calling-v1`, `glaiveai/glaive-function-calling-v2`, `nvidia/When2Call`, `argilla/apigen-function-calling`, `Salesforce/APIGen-MT-5k`, `Agent-Ark/Toucan-1.5M`, `RISys-Lab/KaliBench`, `Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset`, `7h3-R3v3n4n7/pentest-agent-dataset-chatml`, `theelderemo/pentesting-explanations`, `iselabvn/Kali-tool-calling`, `SWE-bench/SWE-smith-trajectories`, `nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1`, `open-thoughts/OpenThoughts-Agent-SFT-100K`, `open-thoughts/AgentTrove`.
- Papers: [APIGen 2406.18518](https://arxiv.org/abs/2406.18518), [xLAM 2409.03215](https://arxiv.org/abs/2409.03215), [ToolACE 2409.00920](https://arxiv.org/abs/2409.00920), [APIGen-MT 2504.03601](https://arxiv.org/abs/2504.03601), [When2Call 2504.18851](https://arxiv.org/abs/2504.18851), [Toucan 2510.01179](https://arxiv.org/abs/2510.01179), [KaliBench 2610.02206](https://arxiv.org/abs/2610.02206), [SWE-Gym 2412.21139](https://arxiv.org/abs/2412.21139), [SWE-smith 2504.21798](https://arxiv.org/abs/2504.21798), [Cybench 2408.08926](https://arxiv.org/abs/2408.08926), [NYU CTF 2406.05590](https://arxiv.org/abs/2406.05590), [CyberMetric 2402.07688](https://arxiv.org/abs/2402.07688), [ToolLLM 2307.16789](https://arxiv.org/abs/2307.16789).
- Code/benchmarks: [Cybench](https://github.com/andyzorigin/cybench), [NYU CTF Bench](https://github.com/NYU-LLM-CTF/NYU_CTF_Bench), [SecEval](https://github.com/XuanwuAI/SecEval), [KaliBench](https://github.com/RISys-Lab/KaliBench), [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html), [mlx-lm LoRA](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md), [Qwen3-8B chat template](https://huggingface.co/Qwen/Qwen3-8B).
