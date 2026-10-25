# 19 — Java and Quarkus datasets: what exists, what is missing, and how to mix it

*Last verified: 25 October 2026. Dataset sizes and licenses were checked against the Hugging Face API and dataset cards; re-check before a large download.*

There is no single “Java architecture and Quarkus” dataset. What exists falls into five families:

1. **Raw code corpora** — The Stack v3, StarCoderData, CodeSearchNet.
2. **Code instruction data** — OpenCodeInstruct, Magicoder, CodeAlpaca.
3. **Repository-level issue resolution** — SWE-smith-java, Multi-SWE-bench, SWE-bench Multilingual.
4. **Test generation** — Methods2Test.
5. **Quarkus-specific material** — the framework, quickstarts and guides, which you must turn into tasks yourself.

The missing piece — SOLID/hexagonal architecture, Quarkus idioms, code review, migration — is exactly what synthetic generation and execution verification are for. See [21 — Java synthetic data](21-java-synthetic.md).

---

## 1. Inventory

| Dataset | What it is | Size | License | Use |
|---|---|---|---:|---|
| [HuggingFaceCode/stack-v3-train](https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train) | 2025 GitHub crawl, deduped, filtered, inline source | ~3.6T train tokens / 15.9 TB, 713 languages | **ODC-BY** | raw Java corpus, continued pretraining, seed code |
| [bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2) | 2023 crawl from Software Heritage | 67.5 TB raw, ~900B tokens train | permissive + opt-out (`other`) | older raw corpus; gated |
| [bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) | StarCoder pretraining corpus, 86 languages | 783 GB, ~250B tokens | permissive (`other`), gated | raw code, commits, issues |
| [code_search_net](https://huggingface.co/datasets/code_search_net) | Java methods with docstrings | 1,569,889 Java functions; 542,991 with docs | repo MIT; data `other` | docstring→code, code search, summarization |
| [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | 5M code instructions, tests, execution status | 5M rows / ~19 GB | **CC-BY-4.0** | instruction SFT; filter Java |
| [ise-uiuc/Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | OSS-Instruct synthetic code instructions | 75K | MIT (GPT-3.5 output; OpenAI policy) | diversity; filter Java |
| [ise-uiuc/Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K) | Evol-Instruct complexity | 110K | Apache-2.0 | harder tasks; filter Java |
| [sahil2801/CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) | early code instructions | 20K | CC-BY-4.0 | small supplement; filter |
| [jitx/Methods2Test_java_unit_test_code](https://huggingface.co/datasets/jitx/Methods2Test_java_unit_test_code) | JUnit tests paired with focal methods | 624,022 train + 156,922 test | **MIT** | test generation |
| [SWE-bench/SWE-smith-java](https://huggingface.co/datasets/SWE-bench/SWE-smith-java) | synthesized Java bug-fix tasks with tests | 7,470 rows / 1.4 GB | repo MIT; card untagged | bug fix, recovery, execution |
| [ByteDance-Seed/Multi-SWE-bench](https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench) | multilingual issue resolution | 1,632 total; Java subset | **CC0** (source code licenses apply) | evaluation only |
| [SWE-bench/SWE-bench_Multilingual](https://huggingface.co/datasets/SWE-bench/SWE-bench_Multilingual) | 300 multilingual issue tasks | 300 total; Java subset | **MIT** | evaluation only |
| [nuprl/MultiPL-E](https://huggingface.co/datasets/nuprl/MultiPL-E) | HumanEval/MBPP translated to Java and others | 18 languages | MIT | evaluation only |
| [zai-org/humaneval-x](https://huggingface.co/datasets/zai-org/humaneval-x) | 820 multilingual tasks, Java included | 820 across 5 languages | Apache-2.0 | evaluation only |
| [quarkusio/quarkus](https://github.com/quarkusio/quarkus) | framework source and tests | large | **Apache-2.0** | Quarkus idioms, tests, seeds |
| [quarkusio/quarkus-quickstarts](https://github.com/quarkusio/quarkus-quickstarts) | runnable examples | medium | **Apache-2.0** | feature tasks |
| [quarkus.io guides](https://quarkus.io/guides/) | reference documentation | large | site says **CC BY 3.0**; check each guide | task/answer seeds |
| [Nan-Do/instructional_code-search-net-java](https://huggingface.co/datasets/Nan-Do/instructional_code-search-net-java) | Java CodeSearchNet instruction pairs | small | Apache-2.0 | auxiliary |
| [nvidia/OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) | code reasoning traces | large | CC-BY-4.0 | optional; filter Java |

> **The Stack v3** is the current large corpus (released July 2026). Its training subset is ODC-BY, not “no license”; attribution is required. The full 113.7 TB corpus is gated. Filter Java by the per-file `language` field and keep the per-file `license_type`/`detected_licenses` metadata.

---

## 2. Raw code corpora

### 2.1 The Stack v3

- **What:** 15.9 TB training subset, ~3.6T tokens, 713 languages, 173M repositories; the full corpus is 113.7 TB across 770 languages and 224M repositories.
- **Format:** repository rows with inline `files` (content, path, language, license metadata).
- **Why use it:** the cleanest large Java corpus for continued pretraining or for sampling real Quarkus/Jakarta projects.
- **Caveats:** ODC-BY attribution; per-file licenses vary; dedup and PII filtering are done but not perfect; gated download; huge.
- **How to use it for SFT:** don't train on raw files directly at 3T scale. Sample 50k–200k Java files from repositories that use Quarkus/Jakarta, then generate tasks from them. Or use it for a separate continued-pretraining run on rented GPUs, not for a local LoRA.

### 2.2 The Stack v2 and StarCoderData

Older but still useful:

- **The Stack v2:** 67.5 TB raw, ~900B tokens; permissive-license filter and opt-out; gated.
- **StarCoderData:** 783 GB, 86 languages, ~250B tokens, including commits and issues; permissive filter and opt-out; gated.

For a Java LoRA, these are **seed corpora**, not instruction data. Their Java share is large enough that a small sample is plenty.

### 2.3 CodeSearchNet

- Java: **1,569,889 functions**, **542,991 with documentation**.
- Pairs a method with its Javadoc/comment.
- Good for: docstring→method, code summarization, code search, “explain this method”.
- Not good for: architecture, multi-file changes, Quarkus.
- License: repository MIT; the data itself is tagged `other` and comes from permissively licensed projects. Keep provenance.

### 2.4 CodeXGLUE

Microsoft's CodeXGLUE is a benchmark suite with Java tasks for code summarization, code translation, clone detection and defect detection. The Hugging Face mirror (`microsoft/CodeXGLUE`) returned 401 for anonymous access, so use the GitHub release and its task-specific data rather than treating it as a bulk corpus. It is useful for **evaluation** and for small, well-defined auxiliary tasks; it is not a Quarkus or architecture dataset.

---

## 3. Instruction and code-writing datasets

### 3.1 OpenCodeInstruct

- 5M instruction/answer pairs, ~19 GB, **CC-BY-4.0**, commercial use allowed.
- Includes `domain`, `generation_algorithm`, `unit_tests`, `tests_execution_status`, `average_test_score`.
- This is the best large public starting point for code SFT, but it is multilingual and Python-heavy. Filter to Java by language detection or by the prompt/answer containing Java syntax; then prefer rows with passing tests.
- Its execution metadata is useful as a quality signal, but do not assume every row ran successfully.

### 3.2 Magicoder

- **OSS-Instruct-75K:** seed code snippets → diverse instructions; MIT, but generated by GPT-3.5, so OpenAI's usage policy applies.
- **Evol-Instruct-110K:** evolves instructions to be harder; Apache-2.0.
- Good for: diversity and harder reasoning.
- Weak for: Quarkus, Java 25, repository context, tests.
- Filter Java; drop Python-only and LeetCode-style puzzles if you want production Java style.

### 3.3 CodeAlpaca and small Java sets

- CodeAlpaca-20k: CC-BY-4.0, early synthetic instructions; small and dated.
- `Nan-Do/instructional_code-search-net-java`: Apache-2.0, Java docstring/instruction pairs; small.
- `XEUIPR/Java-Code-Large-text-only`: MIT tag, but provenance is unclear; avoid for a release.
- `dmsovetov/codeparrot-java`, `hrishizone/Java-GitHub-Codes`: no clear license; avoid.
- `semeru/code-code-CodeCompletion-TokenLevel-Java`: MIT, token-level completion; niche, not instruction.

Rule: **a permissive tag is not a provenance audit.** Prefer datasets with a paper, a maintainer, and per-row license metadata.

---

## 4. Repository-level and execution data

### 4.1 SWE-smith-java

- 7,470 task instances, 1.4 GB, from Java repositories; each row has a `patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `image_name`, `repo`, `problem_statement`.
- This is the closest public analogue to “fix a real Java issue and prove it with tests”.
- The SWE-smith project is MIT; the Java dataset card is untagged, so verify before redistribution.
- Use it for: bug-fix trajectories, recovery, test-driven repair, agent tool use.
- Caveat: synthesized bugs can be artificial; mix with real PRs.

### 4.2 Multi-SWE-bench

- 1,632 instances across seven languages; Java repos include `alibaba/fastjson2`, `elastic/logstash`, `mockito/mockito`.
- CC0 for the dataset, but the underlying repositories have their own licenses.
- **Evaluation only.** It is a benchmark; do not train on it. The Java subset is small, so use it as a held-out signal, not a training set.

### 4.3 SWE-bench Multilingual

- 300 tasks, Java among them; MIT.
- **Evaluation only.** Public and likely in pretraining; report the caveat.

### 4.4 What repository-level data teaches

- Read a file tree, find the relevant class, change multiple files.
- Keep tests green.
- Recover from a failed compile or a failed assertion.
- Produce a patch, not a lecture.

These are the behaviours a “coding agent” needs, and they are underrepresented in single-file instruction datasets.

---

## 5. Test-generation data

Methods2Test is the strongest public Java test corpus:

- 780k focal-method/JUnit-test pairs from 91k open-source projects.
- Variants add class name, constructor signatures, method signatures and fields.
- MIT license.
- Good for: “write a JUnit 5 test for this method”, edge cases, assertions.
- Caveats: old JUnit style, no Quarkus, no Mockito guarantees, tests may be weak or flaky.

Use it to teach **test shape**, then generate Quarkus-specific tests synthetically and execute them.

---

## 6. Quarkus-specific data: build it

There is no public Quarkus instruction dataset of meaningful size. The raw material is good:

- `quarkusio/quarkus` — framework code, tests, extensions.
- `quarkusio/quarkus-quickstarts` — runnable feature examples.
- `quarkus.io/guides` — reference guides with code snippets.
- `code.quarkus.io` — generated project skeletons.
- Quarkus blog posts — release notes and migration guides.
- Your own Quarkus repositories — the highest-value data, if you have the rights.

### 6.1 Turn a quickstart into tasks

For each quickstart module:

1. Read `pom.xml`, the main resource, the service/repository, and the tests.
2. Create a task that removes one behaviour (an endpoint, a validation rule, a repository method, a test).
3. Keep the original code/test as the gold patch.
4. Run `./mvnw -q -DskipITs test` at the base commit and after the patch.
5. Keep only tasks where the base fails and the gold passes.

This produces **execution-verified** feature tasks with a real Quarkus shape.

### 6.2 Turn a guide into Q&A and review tasks

Guides are good for:

- “Which extension provides X?”
- “Why does Quarkus do Y at build time?”
- “What is the correct annotation for Z?”
- “Review this code against the guide's recommended pattern.”

Do not train on the guide text verbatim; generate a task, then verify the answer against the guide and a build.

### 6.3 Turn commits into issue tasks

Use [`scripts/mine_java_commits.py`](https://github.com/beniaminp/qwen-kali-agent/blob/main/scripts/mine_java_commits.py):

```bash
python scripts/mine_java_commits.py \
  --repo ~/src/quarkus-quickstarts \
  --out data/java/interim/quickstart-commits.jsonl \
  --limit 300 --only-with-tests --license Apache-2.0
```

Each seed has a base commit, a commit message, changed files, the gold patch and a test command. Then ask a teacher to write a problem statement from the commit message and a **new** solution patch; apply, test, keep passes. This is how you get scale without training on the gold patch itself.

---

## 7. Architecture and SOLID data: the gap

There is no public dataset that reliably labels “this refactor improves SRP/DIP” or “this package violates hexagonal boundaries”. The honest approach:

1. **Mine real refactors** from permissive repositories (`git log` for commits touching interfaces, package moves, dependency inversion).
2. **Generate refactor tasks** from a working baseline: “move persistence behind a port; keep tests green; add ArchUnit rules”.
3. **Verify with the compiler, tests and ArchUnit**, not an LLM's opinion.
4. **Review with a rubric** for SRP/OCP/LSP/ISP/DIP, coupling, cohesion and readability.
5. **Keep a human-labeled slice** to calibrate the rubric.

ArchUnit is the closest thing to an executable definition of architecture. A refactor that passes the old tests and the new rules is a much stronger training example than a prose “SOLID” answer.

### 7.1 Example architecture task

```json
{
  "task": "Refactor the order feature so the domain does not depend on Quarkus, Panache or Jakarta persistence. Add ArchUnit rules for domain independence and package cycles. Keep the existing HTTP tests green.",
  "constraints": [
    "Java 25 LTS, Quarkus 3.40 LTS",
    "jakarta.* only",
    "no behavior change",
    "return a unified diff",
    "run ./mvnw -q verify"
  ],
  "acceptance": [
    "ArchitectureTest passes",
    "OrderResourceTest passes",
    "no class in ..domain.. imports ..adapters.., io.quarkus.. or jakarta.."
  ]
}
```

This is a task, not a training row. The training row is the verified diff plus the test output.

---

## 8. The mix

For a first Java/Quarkus adapter, aim for **2–4M tokens** (roughly 2,000–5,000 examples after truncation):

| Bucket | Share | Source | Why |
|---|---:|---|---|
| Quarkus feature/API implementation | 25% | quickstarts, guides, synthetic | teaches the framework shape |
| Repo bug fix / recovery | 20% | SWE-smith-java, commits, synthetic | teaches multi-file reasoning |
| SOLID/hexagonal refactor | 15% | synthetic + ArchUnit | teaches architecture |
| Test generation | 15% | Methods2Test, synthetic | teaches verification |
| Code/architecture review | 10% | PR comments, synthetic | teaches critique |
| Migration/upgrade | 5% | 3.33→3.40, 21→25 | teaches version hygiene |
| Refusal/uncertainty/security | 5% | synthetic negatives | teaches boundaries |
| General Java/code replay | 5% | OpenCodeInstruct/Magicoder | limits forgetting |

A mature adapter can grow to 8–12M tokens. Do not scale the same prompt 10×; scale the **task families** and verify.

---

## 9. Java-specific quality filters

Run these before tokenization:

- **Compiles** with `javac --release 25` or `./mvnw -q -DskipITs test`.
- **Tests pass**, and the test actually exercises the change.
- **No legacy imports:** `javax.persistence`, `javax.ws.rs`, `javax.inject`, `javax.enterprise`, `javax.transaction`, `javax.validation`.
- **No Spring idioms** (`org.springframework`, `@Autowired`) in Quarkus rows.
- **No legacy RESTEasy coordinates** (`quarkus-resteasy-reactive`) when `quarkus-rest` is expected.
- **Version tag:** `Java 25`, `Quarkus 3.40.x`; reject rows that mix 3.x and 4.x without a migration label.
- **No secrets** in code, config, logs or test data.
- **No `System.out`/`printStackTrace`** in production examples; use JBoss Logging or SLF4J.
- **No `TODO`/`FIXME`** in the final answer.
- **Patch applies** with `git apply --check` to the base commit.
- **License metadata** present; reject unknown/copyleft for a release mix.
- **Deduplicate** exact and near-duplicate patches; split by repo/commit.

`scripts/validate_java_dataset.py` automates the static checks. It is not a compiler.

---

## 10. Contamination and evaluation hygiene

- Hold out **MultiPL-E Java**, **HumanEval-X Java**, **Multi-SWE-bench Java** and **SWE-bench Multilingual Java**.
- Run 13-gram overlap between training rows and every benchmark.
- Split by **repository and base commit**, never by random row.
- Hold out entire Quarkus modules or releases for a time-based test.
- Keep a private golden set of 100–200 tasks written from your own repositories.
- Report the base model and adapter on the same prompts, at temperature 0, with the same tools and system prompt.

A model that memorizes `quarkus-quickstarts` will look good on a quickstart test and fail on your codebase. The private golden set is the only cure.

---

## 11. Canonical rows

A single-file Java task:

```json
{
  "messages": [
    {"role": "system", "content": "You are a senior Java/Quarkus engineer. Target Java 25 LTS and Quarkus 3.40 LTS. Use jakarta.* imports, constructor injection and tests."},
    {"role": "user", "content": "Write a Java 25 record `Money` with a non-negative amount and an ISO-4217 currency. Reject nulls and invalid currencies."},
    {"role": "assistant", "content": "```java\npackage com.example.money;\n\nimport java.math.BigDecimal;\nimport java.util.Objects;\n\npublic record Money(BigDecimal amount, String currency) {\n    public Money {\n        Objects.requireNonNull(amount, \"amount\");\n        Objects.requireNonNull(currency, \"currency\");\n        if (amount.signum() < 0) throw new IllegalArgumentException(\"amount must be >= 0\");\n        if (!currency.matches(\"[A-Z]{3}\")) throw new IllegalArgumentException(\"currency must be ISO-4217\");\n    }\n}\n```"}
  ],
  "meta": {
    "repo": "java-domain-example",
    "license": "Apache-2.0",
    "category": "java-domain",
    "test_command": "./mvnw -q -DskipITs test"
  }
}
```

A repository patch task with tools: see [`data/examples/java-train.example.jsonl`](https://github.com/beniaminp/qwen-kali-agent/blob/main/data/examples/java-train.example.jsonl). For Qwen2.5-Coder, tool arguments must be **objects**, not JSON strings:

```json
{"name": "read_file", "arguments": {"path": "src/main/java/com/example/OrderResource.java"}}
```

The Qwen2.5-Coder template applies `tojson` to `arguments`; a string becomes an escaped JSON string and the model learns the wrong shape. Use `prepare_dataset.py --arguments-format dict`.

---

## 12. Sources

- The Stack v3 — <https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train>
- The Stack v2 — <https://huggingface.co/datasets/bigcode/the-stack-v2>
- StarCoderData — <https://huggingface.co/datasets/bigcode/starcoderdata>
- CodeSearchNet — <https://github.com/github/CodeSearchNet>
- CodeXGLUE — <https://github.com/microsoft/CodeXGLUE>
- OpenCodeInstruct — <https://huggingface.co/datasets/nvidia/OpenCodeInstruct>
- Magicoder OSS-Instruct — <https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K>
- Magicoder Evol-Instruct — <https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K>
- Methods2Test — <https://huggingface.co/datasets/jitx/Methods2Test_java_unit_test_code>
- SWE-smith-java — <https://huggingface.co/datasets/SWE-bench/SWE-smith-java>
- Multi-SWE-bench — <https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench>
- SWE-bench Multilingual — <https://huggingface.co/datasets/SWE-bench/SWE-bench_Multilingual>
- MultiPL-E — <https://huggingface.co/datasets/nuprl/MultiPL-E>
- HumanEval-X — <https://huggingface.co/datasets/zai-org/humaneval-x>
- Quarkus quickstarts — <https://github.com/quarkusio/quarkus-quickstarts>
- Quarkus guides — <https://quarkus.io/guides/>
- Quarkus DDD/hexagonal — <https://quarkus.io/blog/quarkus-insights-254-ddd-hexagonal-architecture-part2>
- ArchUnit — <https://www.archunit.org/>
