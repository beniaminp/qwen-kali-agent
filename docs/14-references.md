# 14 — References

*Annotated source list, verified 25 October 2026. Primary sources first.*

---

## Models and tokenizers

| Source | Why it matters |
|---|---|
| [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | The recommended local base; model card, benchmarks, license. |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Fast fallback; same template family. |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | Inference-only at 4-bit on 32 GB. |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Strong agentic base; best if renting a GPU. |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Small modern model; conditional local training. |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | Fast MoE inference; bulk synthetic generation. |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Best dense quality that fits 4-bit inference; oracle/judge. |
| [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | CLI/coding specialist; inference-only on 32 GB. |
| [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) | 80B-A3B; too large for 32 GB. |
| [Qwen3 technical report](https://arxiv.org/abs/2505.09388) | Architecture, training, benchmarks. |

## Fine-tuning and MLX

| Source | Why it matters |
|---|---|
| [MLX-LM](https://github.com/ml-explore/mlx-lm) | The training/serving framework. |
| [MLX-LM LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) | Data formats, flags, fuse, memory tips. |
| [MLX-LM server guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md) | OpenAI endpoint, security caveats. |
| [MLX-LM issues](https://github.com/ml-explore/mlx-lm/issues) | GDN training OOMs, descriptor leaks, MoE routing. |
| [Issue #1206](https://github.com/ml-explore/mlx-lm/issues/1206) | Qwen3.5-9B first-backward OOM at 36 GB. |
| [Issue #1185](https://github.com/ml-explore/mlx-lm/issues/1185) | Metal descriptor limit on long LoRA runs. |
| [PR #1217](https://github.com/ml-explore/mlx-lm/pull/1217) | GDN Metal VJP kernel; closed unmerged. |
| [MLX](https://github.com/ml-explore/mlx) | Core array framework. |
| [MLX memory limits](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html) | Wired limit API. |
| [LoRA](https://arxiv.org/abs/2106.09685) | Original method. |
| [QLoRA](https://arxiv.org/abs/2305.14314) | Quantized base + LoRA. |
| [LIMA](https://arxiv.org/abs/2305.11206) | Data quality over quantity. |

## Serving and conversion

| Source | Why it matters |
|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF conversion, quantization, `llama-server`. |
| [llama.cpp function calling](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md) | Jinja templates, tool parsers. |
| [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) | Flags, OpenAI compatibility. |
| [Ollama import](https://docs.ollama.com/import) | GGUF import and Modelfile. |
| [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility) | `/v1` subset and caveats. |
| [Ollama macOS](https://docs.ollama.com/macos) | App, `OLLAMA_HOST`, context. |
| [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) | Download/upload. |

## Datasets

| Source | Why it matters |
|---|---|
| [xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 60k function-calling; schema repair needed. |
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | Multi-turn tool use with observations. |
| [Hermes FC](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | Qwen-shaped tool data. |
| [Glaive FC v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | Volume; filter heavily. |
| [When2Call](https://huggingface.co/datasets/nvidia/When2Call) | Call vs ask vs abstain. |
| [APIGen](https://huggingface.co/datasets/argilla/apigen-function-calling) | Verified synthetic function calling. |
| [APIGen-MT-5k](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k) | Best multi-turn realism; **CC-BY-NC**. |
| [Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) | Large MCP/tool-use set. |
| [SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith) | Recovery-rich terminal/code trajectories. |
| [OpenThoughts-Agent](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-100K) | Terminal agent trajectories. |
| [Nemotron Terminal Pivot](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1) | Recovery/pivot behavior. |
| [KaliBench](https://huggingface.co/datasets/RISys-Lab/KaliBench) | NL→Kali command; hold out test. |
| [Trendyol Cybersecurity](https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset) | Defensive cyber QA. |
| [BFCL](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) | Evaluation only. |
| [Cybench](https://huggingface.co/datasets/penfever/cybench-harbor) | CTF evaluation. |
| [CyberMetric](https://huggingface.co/datasets/tihanyin/CyberMetric) | Security knowledge MCQ. |

## Synthetic data

| Source | Why it matters |
|---|---|
| [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing) | Current models/prices. |
| [DeepSeek JSON mode](https://api-docs.deepseek.com/guides/json_mode) | Structured output. |
| [DeepSeek tool calls](https://api-docs.deepseek.com/guides/tool_calls) | Function calling, strict schemas. |
| [DeepSeek thinking](https://api-docs.deepseek.com/guides/thinking_mode) | Reasoning control. |
| [OpenCode](https://github.com/sst/opencode) | Headless agent harness. |
| [OpenCode docs](https://opencode.ai/docs) | Providers, agents, MCP, permissions. |
| [Self-Instruct](https://arxiv.org/abs/2212.10560) | Seed expansion. |
| [APIGen](https://arxiv.org/abs/2406.18518) | Verifiable synthetic function calling. |
| [ToolACE](https://arxiv.org/abs/2409.00920) | Multi-agent tool synthesis. |
| [MinHash](https://arxiv.org/abs/2001.07219) | Dedup. |

## Agent frameworks and Kali

| Source | Why it matters |
|---|---|
| [OpenCode MCP](https://opencode.ai/docs/mcp-servers) | MCP config for the agent. |
| [MCP-Kali-Server](https://github.com/Wh0am123/MCP-Kali-Server) | Small, typed Kali tool surface. |
| [PentestGPT](https://github.com/GreyDGL/PentestGPT) | Guided pentest agent. |
| [hackingBuddyGPT](https://github.com/ipa-lab/hackingBuddyGPT) | Minimal research baseline. |
| [HexStrike AI](https://github.com/0x4m4/hexstrike-ai) | Large MCP tool surface (too many for 14B). |
| [CAI](https://github.com/aliasrobotics/cai) | Archived; reference only. |
| [Kali docs](https://www.kali.org/docs/) | Tool documentation. |
| [MCP spec](https://modelcontextprotocol.io/) | Protocol. |
| [OWASP WSTG](https://owasp.org/www-project-web-security-testing-guide/) | Testing methodology. |

## Safety and law

| Source | Why it matters |
|---|---|
| [CFAA](https://www.justice.gov/criminal/cybercrime-intellectual-property-section/computer-fraud-and-abuse-act) | US computer-misuse law. |
| [UK CMA 1990](https://www.legislation.gov.uk/ukpga/1990/18/contents) | UK computer-misuse law. |
| [GDPR](https://gdpr-info.eu/) | Personal data. |
| [NIS2](https://digital-strategy.ec.europa.eu/en/policies/nis2-directive) | EU critical-infrastructure security. |
| [NIST CSF](https://www.nist.gov/cyberframework) | Defensive framework. |
| [Qwen license](https://huggingface.co/Qwen/Qwen3-14B/blob/main/LICENSE) | Apache-2.0. |

---

## How to keep this list current

1. Check the model card and the GitHub repo, not a blog post.
2. Record the date and the revision/commit.
3. Prefer primary sources for licenses and prices.
4. When a source disappears, mark it **UNVERIFIED** rather than guessing.
5. Update the handbook’s “last verified” line when you change a command or a number.
