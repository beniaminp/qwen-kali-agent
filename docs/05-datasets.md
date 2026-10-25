# 05 — Datasets: a verified catalog

*Last verified: 25 October 2026. Row counts come from the Hugging Face datasets-server where available; token counts are `chars/4` estimates and are marked “est.” Gated repos return 401 for both “gated” and “does not exist”, so gated IDs were cross-checked against the author's public listing.*

This chapter is the shopping list. It covers general tool calling, agentic trajectories, and cybersecurity/pentest data, with the license, schema, quality and conversion notes you need before downloading anything.

> **Read [`04-data-strategy.md`](04-data-strategy.md) first.** This catalog is not a recommendation to train on everything. Most of these sets are raw material; the mix decides what enters a run.

> Looking for **Java/Quarkus** or **React/Tailwind** data? They have their own catalogs in [19 — Java datasets](19-java-datasets.md) and [23 — React datasets](23-react-datasets.md).

---

## 1. General function-calling datasets

| Dataset | Rows | Size / tokens (est.) | License | Shape | Verdict |
|---|---:|---|---|---|---|
| [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 60,000 | 96 MB; ≈23.7M tok | CC-BY-4.0 | `query`, `tools` (JSON string), `answers` (JSON string) | Strong single-turn breadth; **gated**; schemas need repair. |
| [`Team-ACE/ToolACE`](https://huggingface.co/datasets/Team-ACE/ToolACE) | 11,300 | 10.8 MB; ≈19.5M tok | Apache-2.0 | `system`, `conversations` (`from`: user/assistant/tool) | Best open multi-turn with tool results; custom call syntax. |
| [`NousResearch/hermes-function-calling-v1`](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | 11,578 (5 configs) | 21.7 MB; ≈4.1M tok (func_calling) | Apache-2.0 | `conversations`, `tools`, `category` | Qwen-shaped, small, mixed quality. |
| [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | 112,960 | 97 MB; ≈75.9M tok | Apache-2.0 | `system`, `chat` strings | 2023-era; trivial calls, format noise. Use sparingly. |
| [`nvidia/When2Call`](https://huggingface.co/datasets/nvidia/When2Call) | 15,000 SFT; 9,000 pref; 3,652 MCQ | 12.5 MB; ≈5.4M tok | CC-BY-4.0 | `tools`, `messages` | **Essential** for call-vs-ask-vs-abstain. |
| [`argilla/apigen-function-calling`](https://huggingface.co/datasets/argilla/apigen-function-calling) | 109,402 | 60 MB; ≈44.4M tok | CC-BY-4.0 | `answers`, `query`, `tools`, `origin` | xLAM (60k) + Synth-APIGen (49.4k); needs prep. |
| [`Salesforce/APIGen-MT-5k`](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k) | 5,000 | 10 MB; ≈26.0M tok | **CC-BY-NC-4.0** | `conversations`, `tools`, `system` | Best multi-turn realism; **non-commercial trap**. |
| [`Agent-Ark/Toucan-1.5M`](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) | 1.65M total; 119k SFT | 21.8 GB; SFT ≈497M tok | Apache-2.0 | `messages`, `tools`, quality scores | Best 2025 set; MCP-centric; huge. |
| [`gorilla-llm/Berkeley-Function-Calling-Leaderboard`](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) | eval files | ≈9 MB | Apache-2.0 | questions + `possible_answer` | **Eval only — hold out.** |

### xLAM — breadth, with schema debt

The APIGen pipeline generated 60k samples over 3,673 executable APIs and verified them by format checking, real execution and semantic checks ([arXiv:2406.18518](https://arxiv.org/abs/2406.18518)). It is excellent for teaching the model to choose among many tools and to produce multi-argument calls.

Its schema is **not** OpenAI JSON Schema. `parameters` is a map of `param → {type: "str"|"int"|"List", description}`, with no top-level `"type":"object"` and no `required` array. A sample found 940 of 58,105 tool definitions with a non-object top-level type. Convert:

```python
# Old: {"location": {"type": "str", "description": "..."}}
# New:
{"type": "object",
 "properties": {"location": {"type": "string", "description": "..."}},
 "required": ["location"]}
```

Never invent a `required` value; infer it from the reference call's arguments only if the original schema marks it. If the type is ambiguous, drop the field from the schema rather than guess.

xLAM is single-turn and has no tool responses. It teaches **selection and arguments**, not observation handling.

### ToolACE — the best open multi-turn set

26,507 APIs, dual rule/model verification, and a formal thinking process ([arXiv:2409.00920](https://arxiv.org/abs/2409.00920)). The entire tool list sits in `system`, so rows are long. Calls are **not JSON**:

```text
[Market Trends API(trend_type="MARKET_INDEXES")]
```

Tool results are `from: "tool"` JSON blobs. Tool names contain spaces (`"Get Manager Placeholder Image"`), and schemas use `"type":"dict"`. Every row needs parsing, name normalization (`[A-Za-z0-9_-]`), and schema rewriting. It has genuine multi-turn observation loops and chit-chat turns, which help “don't call a tool” behavior.

### Hermes — Qwen-shaped and small

Five configs: single-turn (1,893), multi-turn (1,893), Glaive-derived (5,209), and JSON-mode sets. The format is already Qwen-adjacent: `<tools>[...]</tools>` in the system message, `<tool_call>{"name","arguments"}</tool_call>`, `<tool_response>`. The `glaive_*` config inherits Glaive's flaws; the `json_mode_*` configs teach structured output, not tools. Use the clean single/multi-turn configs as a small, cheap supplement.

### Glaive — volume, not quality

112,960 rows, but 2023-era: `system` is a `"SYSTEM: ..."` string, `chat` uses `USER:`/`ASSISTANT:` with `<functioncall> {"name":..., "arguments": '{"k":"v"}'}` (arguments is a **string**), and `FUNCTION RESPONSE:` turns. Single-turn bias, trivial calls, inconsistent refusal behavior, occasional arithmetic errors inside arguments. Filter aggressively; use only as format seasoning.

### When2Call — the refusal set you need

Explicitly benchmarks “when to call, when to ask a follow-up, when to admit the tools cannot answer” ([arXiv:2504.18851](https://arxiv.org/abs/2504.18851)). This is the single most relevant public set for scope/authorization behavior, and it is CC-BY-4.0 with commercial use stated on the card. Caveat: synthetic and benchmark-shaped; add your own Kali-specific scope dialogues.

### APIGen-MT-5k — great, but non-commercial

Simulated human–agent interplay with clarification questions, `function_call`/`observation` turns, and an explicit `think` tool call ([arXiv:2504.03601](https://arxiv.org/abs/2504.03601)). 5.2k tokens/row; the best public multi-turn realism. **CC-BY-NC-4.0** — internal experiments and evaluation only, never in a model you publish or sell.

### Toucan — the 2025 scale option

495 real MCP servers, 2,000+ tools, trajectories from Qwen3-32B, Kimi-K2 and GPT-OSS-120B, with LLM-judge quality fields and an `irrelevant` subset that is effectively negative/no-call data ([arXiv:2510.01179](https://arxiv.org/abs/2510.01179)). Apache-2.0. Start from the 119k-row SFT config; filter to the top judge bands and trajectories under ~6k tokens.

---

## 2. Agentic / terminal / software-agent trajectories

| Dataset | Rows | Size / tokens (est.) | License | What it teaches | Recovery? |
|---|---:|---|---|---|---|
| [`SWE-Gym/SWE-Gym`](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) | 2,438 tasks | 43.6 MB | MIT | Real Python issue fixing with runnable tests | Yes |
| [`SWE-Gym/OpenHands-SFT-Trajectories`](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories) | 491 | 10.4 MB | MIT | OpenHands SFT traces | Some |
| [`SWE-bench/SWE-smith`](https://huggingface.co/datasets/SWE-bench/SWE-smith) | 59,136 instances | 278 MB | MIT | Synthetic bugs across 128 repos | Yes |
| [`SWE-bench/SWE-smith-trajectories`](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) | 76,002 | 3.17 GB | MIT | Full agent traces in 3 formats | **Yes** |
| [`open-thoughts/OpenThoughts-Agent-SFT-100K`](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-100K) | 94,334 | 1.75 GB; ≈2.0B tok | Apache-2.0 | Terminal/code agent trajectories | Yes |
| [`open-thoughts/AgentTrove`](https://huggingface.co/datasets/open-thoughts/AgentTrove) | 1.7M | 19.6 GB | Apache-2.0 | 219 sources, unified terminus-2 traces | Mixed |
| [`nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1`](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1) | 31,111 | 1.27 GB; ≈272M tok | CC-BY-4.0 | Terminal pivots/recovery, `bash` tool | **Yes** |
| [`nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1`](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1) | 96,968 | 1.56 GB; ≈395M tok | CC-BY-4.0 | Conversational tool use, 838 domains | Some |

### Which teach recovery?

- **SWE-smith** is the strongest open signal: 50k+ injected bugs, failing tests, and 76k full trajectories in three serializations (`tool`, `xml`, `ticks`), MIT ([arXiv:2504.21798](https://arxiv.org/abs/2504.21798)). The trajectory subset is large; sample it.
- **SWE-Gym** adds 2,438 real tasks and 491 curated OpenHands traces ([arXiv:2412.21139](https://arxiv.org/abs/2412.21139)).
- **Nemotron Terminal Pivot** is the most directly transferable to a shell agent: 31k trajectories built around recovery/pivot behavior, `bash_command` as the tool, terminus-2 prompts.

### Format warning: terminus-2 vs tool_calls

OpenThoughts, AgentTrove and Nemotron terminal use the **terminus-2 JSON convention**:

```json
{"analysis": "...", "plan": "...", "commands": [{"keystrokes": "nmap -sV 10.10.10.5\n"}]}
```

This is not OpenAI `tool_calls`. Convert `commands[]` into one `bash` call per command, move `analysis`/`plan` to `reasoning_content`, and keep environment output in `role:"tool"`. Do not train raw terminus-2 if you also want native `<tool_call>` reliability — the formats compete.

### ToolBench / ToolLLM

16,464 RapidAPI tools and DFS-generated solution paths ([arXiv:2307.16789](https://arxiv.org/abs/2307.16789)). The canonical HF repo did not resolve publicly on 25 Oct 2026 (401; author listing empty), so use the GitHub release or a verified mirror — **UNVERIFIED** as an HF source. Largely superseded, and its API data is stale.

### TheAgentCompany

A benchmark, not SFT data; only `TheAgentCompany/ubuntu_os_images` is public. `OpenHands/OpenHands-Trajectories` and `SWE-Gym/SWE-Gym-OpenHands-Trajectories` did not resolve publicly on 25 Oct 2026 (**UNVERIFIED**). The public SWE-Gym trajectory repo is `SWE-Gym/OpenHands-SFT-Trajectories`.

---

## 3. Cybersecurity / pentest datasets

| Dataset | Rows | License | Type | Usability |
|---|---:|---|---|---|
| [`RISys-Lab/KaliBench`](https://huggingface.co/datasets/RISys-Lab/KaliBench) | 8,504 (3,504 train / 5,000 test) | CC-BY-4.0 | NL→Kali command, structured args | **High for CLI fidelity; test split must be held out.** |
| [`Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset`](https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset) | 53,201 | Apache-2.0 | Defensive Q&A | High for domain + refusal policy. |
| [`theelderemo/pentesting-explanations`](https://huggingface.co/datasets/theelderemo/pentesting-explanations) | 5,906 default (+3,228 HackTricks, +2,678 MITRE configs) | Apache-2.0 (claimed) | MCQ + explanation + think trace | Medium; convert to open QA. |
| [`7h3-R3v3n4n7/pentest-agent-dataset-chatml`](https://huggingface.co/datasets/7h3-R3v3n4n7/pentest-agent-dataset-chatml) | 322,511 | Apache-2.0 (claimed) | CVE FAQ, 2-turn | **Low** — 21.5% junk, 0.4% code. |
| [`iselabvn/Kali-tool-calling`](https://huggingface.co/datasets/iselabvn/Kali-tool-calling) | 790 | Apache-2.0 | Single-turn tool call + reasoning | High format, tiny. |
| [`0dAI/PentestingCommandLogic`](https://huggingface.co/datasets/0dAI/PentestingCommandLogic) | 10,188 | Apache-2.0 | Instruction/response | **Junk in sample.** |
| [`suryanshp1/kali-linux-pentesting-data`](https://huggingface.co/datasets/suryanshp1/kali-linux-pentesting-data) | 343 | none | Trivial nmap Q&A | Low. |
| [`WNT3D/Ultimate-Offensive-Red-Team`](https://huggingface.co/datasets/WNT3D/Ultimate-Offensive-Red-Team) | 25,620 | MIT | instruction/input/output | **Low** — outputs echo inputs. |
| [`yatin-superintelligence/White-Hat-Security-Agent-Prompts-600K`](https://huggingface.co/datasets/yatin-superintelligence/White-Hat-Security-Agent-Prompts-600K) | 596,295 | CC-BY-4.0 | Prompt corpus, **no answers** | Not SFT as-is. |
| [`preemware/pentesting-eval`](https://huggingface.co/datasets/preemware/pentesting-eval) | 241 | Apache-2.0 | MCQ eval | Eval only. |
| [`resk-fr/pentesting-for-agents`](https://huggingface.co/datasets/resk-fr/pentesting-for-agents) | 54 | none | Synthetic OpenSSH/Redis scenarios | Tiny; inspect. |
| [`whysue/agent-real-pentest-trajectory`](https://huggingface.co/datasets/whysue/agent-real-pentest-trajectory) | 2.3 GB ShareGPT + 3.8 GB alpaca | MIT | files named `artex_*`; no card | **UNVERIFIED provenance.** |
| [`justinwangx/CTFtime`](https://huggingface.co/datasets/justinwangx/CTFtime) | 18,013 chunks | none | `text_chunk` only; likely scraped writeups | **Contamination/legal risk.** |
| [`ethz-spylab/ctf-satml24`](https://huggingface.co/datasets/ethz-spylab/ctf-satml24) | 137,151 | MIT | AI-control competition attack logs | Not pentest. |
| [`alibaba-AAIG/alibaba-ctf`](https://huggingface.co/datasets/alibaba-AAIG/alibaba-ctf) | 87 tasks | Apache-2.0 | Docker CTF benchmark | Eval only. |
| `WhiteRabbitNeo/WRN-Chapter-1`, `-2` | gated | "other" | Cyber Q&A | **Avoid** (terms unreadable). |
| [`CyberNative/CyberSecurityEval`](https://huggingface.co/datasets/CyberNative/CyberSecurityEval) | 500 | none | MCQ eval | Eval only. |
| [`s2e-lab/SecurityEval`](https://huggingface.co/datasets/s2e-lab/SecurityEval) | 121 | none | MCQ eval | Eval only. |

### KaliBench — the best CLI set, the weakest attack set

8,504 query–command pairs, 1,642 tools, 23 capability dimensions, documentation-grounded generation, multi-stage verification (LLM, sandboxed execution, human review); NeurIPS 2026 D&B ([arXiv:2610.02206](https://arxiv.org/abs/2610.02206)).

Measured properties:

- 3,504 train / 5,000 test, **zero exact query overlap**;
- 962 vs 1,642 tools;
- only **32 train rows (0.9%)** use a canonical offensive tool (nmap, sqlmap, hydra, msfconsole);
- top tools are `gdb`, `mailer`, `airmon-ng`, `pwsh`, `pedis`, `ld.lld`, `rva2ofs`;
- **20.1%** of commands have no flags;
- 5.5% are version-style commands;
- average command length 5.13 tokens;
- 157 of 962 tools appear once;
- only 31 use `sudo`.

It teaches flag/argument fidelity on obscure util-linux binaries, **not** recon or exploitation sequencing. Both splits are public, so the test set is not a clean holdout for models trained after release.

### The 322k-row “pentest agent” set is a CVE FAQ

All 322,511 rows scanned: 310,428 mention a CVE, only 1,343 (0.4%) contain a fenced code block, mean answer length 59 words, only 714 prompts mention “exploit”, and **69,480 rows (21.5%) are `** RESERVED **` or rejected-CVE placeholders**. 310,306 unique CVEs, zero multi-turn rows. Usable only as a filtered CVE subset (drop reserved, require ≥60 words and a reference), never as agentic data.

### Trendyol is clean but defensive

53,201 rows, zero exact duplicates, two malformed rows, mean 275 words, 0.3% code fences, 1,628 prompts containing offensive terms. Its uniform system prompt is a strong refusal/scope template; copying it verbatim will over-refuse. Paraphrase a shorter policy.

### Junk is common

`0dAI` answers a MediaWiki SVG question with `fdroid`/`curl root@x00` gibberish; `WNT3D` echoes its input; `theelderemo` tests recognition, not procedure. `iselabvn/Kali-tool-calling` is the cleanest small set (`reasoning_content` + `tool_calls` for `exec`), but only 790 rows.

---

## 4. Licensing and compliance

| License | Datasets | Publishable model? |
|---|---|---|
| Apache-2.0 | ToolACE, Hermes, Glaive, Toucan, OpenThoughts, AgentTrove, Trendyol, Nemotron sets, BFCL, Cybench | Yes, with NOTICE/attribution |
| MIT | SWE-Gym, SWE-smith, WNT3D (provenance caveat) | Yes |
| CC-BY-4.0 | xLAM, When2Call, argilla/apigen, KaliBench, CyberMetric (mirror), yatin | Yes, with attribution |
| CC-BY-NC-4.0 | **APIGen-MT-5k**, Salesforce APIGen-MT-30k | **No** for a commercial/published model |
| CC-BY-NC-SA-4.0 | SecEval mirror | **No**; share-alike compounds |
| GPL-2.0 | **NYU CTF Bench**, ExploitDB | Risky; exclude from a publishable mix |
| "other"/custom | WhiteRabbitNeo WRN, MITRE ATT&CK | Read terms; assume restricted |
| None | justinwangx/CTFtime, suryanshp1, resk-fr, XuanwuAI/SecEval, HackTricks | **No** — all rights reserved by default |

Three traps:

1. **Non-commercial contagion.** APIGen-MT-5k is the best multi-turn set and is CC-BY-NC. Keep it in an internal-only ablation. Whether NC-trained weights are a “derivative” is legally unsettled; treat them as unsafe to publish (**UNVERIFIED legal conclusion**).
2. **Copyleft data.** NYU CTF Bench is GPL-2.0. Training is probably not distribution; redistributing a derived dataset is, and the model-release question is unresolved. ExploitDB is GPL-2.0, which contaminates the 322k set despite its Apache-2.0 tag. `theelderemo` claims Apache-2.0 over HackTricks/MITRE content, and HackTricks has no SPDX license file (**UNVERIFIED**). Prefer KaliBench/Trendyol for the cyber core.
3. **License laundering.** Mirrors relicense xLAM as MIT/Apache; neither can override Salesforce's CC-BY-4.0. Trace every mirror to the origin and record the origin ID.

**Leaked CTF/writeup risk.** Challenge files, flags and writeups are copyrighted by organisers; some flags are still live. `justinwangx/CTFtime` (18,013 writeup chunks, no license) and `ethz-spylab/ctf-satml24` (AI-control transcripts, not pentest) should not enter a released model. Generate CTF-style reasoning in your own lab; use Apache-2.0 Docker benchmarks for evaluation only.

---

## 5. Benchmarks to hold out

| Benchmark | Size | License | What it measures | Use |
|---|---:|---|---|---|
| **BFCL** | ~2k questions across v3/v4 categories | Apache-2.0 | function selection, argument validity, parallel/multiple calls, irrelevance | primary tool-call holdout |
| **KaliBench test** | 5,000 | CC-BY-4.0 | NL→Kali command accuracy | CLI holdout (public, so report contamination) |
| **Cybench** | 40 CTF tasks, 4 competitions | Apache-2.0 | end-to-end offensive security in Docker | agentic holdout |
| **NYU CTF Bench** | 200 test + 55 dev | GPL-2.0 | CTF challenge solving | evaluation only, not release mix |
| **AutoPenBench** | ~30 scenarios | see repo | autonomous pentest workflow | agentic holdout |
| **CyberMetric** | 80/500/2,000/10,000 MCQ | Apache-2.0 (mirror) | cybersecurity knowledge | knowledge check, not agent behavior |
| **SecEval** | 2,000+ MCQ | mirror CC-BY-NC-SA | security knowledge | evaluation only |
| **Terminal-Bench** | terminal tasks | see repo | shell agent competence | optional |

**Holdout discipline:** put the official files in `data/benchmarks/` and never symlink them into `data/processed/`. Run the 13-gram decontamination from [`04-data-strategy.md`](04-data-strategy.md) §7 and report the overlap rate.

---

## 6. Recommended downloads

Start with a minimal shopping list. All commands assume `hf` (Hugging Face CLI) is installed and you are logged in.

```bash
mkdir -p data/raw && cd data/raw

# General tool calling
hf download Team-ACE/ToolACE --repo-type dataset --local-dir ToolACE
hf download NousResearch/hermes-function-calling-v1 --repo-type dataset --local-dir hermes-fc
hf download nvidia/When2Call --repo-type dataset --local-dir When2Call
hf download Agent-Ark/Toucan-1.5M --repo-type dataset --local-dir Toucan --include "sft/*"

# Agentic / terminal (sample, do not download all 3 GB unless needed)
hf download SWE-bench/SWE-smith-trajectories --repo-type dataset --local-dir swe-smith-trajectories --include "tool/*"
hf download nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1 --repo-type dataset --local-dir nemotron-terminal

# Cyber
hf download RISys-Lab/KaliBench --repo-type dataset --local-dir KaliBench
hf download Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset --repo-type dataset --local-dir trendyol-cyber
hf download iselabvn/Kali-tool-calling --repo-type dataset --local-dir kali-tool-calling

# Gated: accept the terms on the dataset page first
hf download Salesforce/xlam-function-calling-60k --repo-type dataset --local-dir xlam
```

If a download fails with 401, check the dataset page for a gated agreement. Do not use a mirror that relicenses the data.

---

## 7. Conversion recipes

The target format is always:

```json
{"messages": [...], "tools": [OpenAI tool schemas]}
```

Detailed rules and examples are in [`06-data-format.md`](06-data-format.md). The per-dataset mapping:

### xLAM

```text
query                         → messages[0] user
tools (JSON string)           → tools (parse; repair parameters)
answers (JSON string)         → assistant tool_calls (arguments JSON-stringified)
```

Repair schema: `parameters = {"type":"object","properties":...,"required":[...]}`. If `answers` contains multiple calls, emit multiple `tool_calls` in one assistant message.

### ToolACE

```text
system                        → messages[0] system (keep; it contains the tool list)
conversations[from=user]      → user
conversations[from=assistant] → assistant; parse `[Name(arg="v")]` into tool_calls
conversations[from=tool]      → tool (content)
```

Normalize tool names to `[A-Za-z0-9_-]`; rewrite the `tools` array from the system block; `"dict"` → `"object"`.

### Hermes

Already close. Keep `<tools>` content as the `tools` array, parse `<tool_call>` JSON, map tool responses to `role:"tool"`, and strip the Hermes wrapper tokens (the Qwen template will add its own).

### Glaive

```text
system "SYSTEM: ..."           → system (strip prefix)
USER: ...                      → user
ASSISTANT: <functioncall> ...  → assistant tool_calls (arguments is a JSON string)
FUNCTION RESPONSE: ...         → tool
```

Filter rows with arithmetic errors or trivial calls.

### APIGen-MT

```text
human         → user
gpt           → assistant
function_call → assistant tool_calls
observation   → tool
think         → reasoning_content (or a `think` tool call, depending on the source)
```

Internal-only because of CC-BY-NC.

### When2Call

Keep the `tools` + `messages` structure. Preserve the “ask a follow-up” and “cannot answer with these tools” turns; they are the point. Do not convert every turn into a tool call.

### Terminus-2 (OpenThoughts, AgentTrove, Nemotron)

```text
analysis/plan → reasoning_content
commands[]    → one `bash` tool call per command (arguments: {"command": "..."})
environment   → tool response
```

Do not mix terminus-2 and native `<tool_call>` in the same training run without labeling the format in the system prompt.

### KaliBench

```text
query                         → user
command + args                → assistant tool_call to `kali_exec`
```

Use the structured arguments if present; otherwise pass the command string. Hold out the test split. Note that only 0.9% of train rows are offensive tools; supplement with synthetic recon/enumeration trajectories.

### Trendyol

Already `system/user/assistant`. Keep as cyber QA. Shorten the uniform system prompt; otherwise the model learns to over-refuse.

---

## 8. Quality filtering and red flags

**General filters:**

- exact duplicate removal (hash of normalized messages),
- near-duplicate removal (MinHash, Jaccard ≥ 0.8),
- token length within the chosen max sequence (or split),
- valid JSON for every tool call,
- tool names exist in the `tools` array,
- required arguments present,
- no empty assistant messages,
- no tool response without a preceding call.

**Cyber-specific red flags:**

- commands against public IPs/domains that are not lab targets,
- real credentials, API keys, tokens, hashes of live systems,
- destructive commands (`rm -rf /`, `mkfs`, fork bombs, `dd` to devices) presented without a lab context,
- exploit code with no defensive/authorized context,
- live flags from CTF challenges,
- personally identifiable data,
- instructions to evade logging or law enforcement,
- “scan the internet” / “find vulnerable hosts” style tasks.

Quarantine flagged rows for human review. Do not silently delete them; the quarantine set is useful for training refusal behavior.

---

## 9. Sources

- xLAM / APIGen — <https://arxiv.org/abs/2406.18518>
- ToolACE — <https://arxiv.org/abs/2409.00920>
- When2Call — <https://arxiv.org/abs/2504.18851>
- APIGen-MT — <https://arxiv.org/abs/2504.03601>
- Toucan — <https://arxiv.org/abs/2510.01179>
- SWE-smith — <https://arxiv.org/abs/2504.21798>
- SWE-Gym — <https://arxiv.org/abs/2412.21139>
- KaliBench — <https://arxiv.org/abs/2610.02206>
- Cybench — <https://arxiv.org/abs/2408.08926>
- NYU CTF Bench — <https://arxiv.org/abs/2406.05590>
- CyberMetric — <https://arxiv.org/abs/2402.07688>
- BFCL — <https://gorilla.cs.berkeley.edu/leaderboard.html>
- Hugging Face dataset API — <https://huggingface.co/docs/datasets-server>
