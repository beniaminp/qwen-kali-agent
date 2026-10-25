# 21 — Generating Java/Quarkus training data with another LLM

*Last verified: 25 October 2026. The pipeline uses the DeepSeek API and OpenCode from [07 — Synthetic data](07-synthetic-data.md); the Java-specific prompts and verification rules are new.*

Public Java datasets teach syntax and single-file coding. They do not teach **your** Quarkus conventions, **your** package boundaries, **your** test command, or the difference between a plausible diff and one that compiles. Synthetic data fills that gap only when every candidate is **executed**.

This chapter describes a repo-grounded pipeline:

```text
permissive repo + docs + commits
        │
        ▼
teacher proposes a task (JSON)
        │
        ▼
teacher proposes a patch + tests (JSON)
        │
        ▼
validator: schema, scope, Java/Quarkus rules
        │
        ▼
sandbox: git apply --check → ./mvnw test → ArchUnit
        │
        ▼
judge + human spot check → keep / repair / discard
        │
        ▼
canonical JSONL (Qwen2.5-Coder arguments as objects)
```

The teacher can be DeepSeek V4.1 Flash/Pro, a local Qwen3-Coder-30B-A3B, or a larger API model. The verifier is not an LLM: it is Maven, the compiler, the tests and ArchUnit.

---

## 1. What synthetic data must add

| Gap in public data | Synthetic task |
|---|---|
| Quarkus 3.40 idioms | REST endpoint, Panache repository, validation, config, test |
| Java 25 style | records, sealed types, pattern matching, virtual threads |
| SOLID/hexagonal | refactor behind ports, add ArchUnit rules, keep tests green |
| Repository context | multi-file changes, read before edit, patch format |
| Recovery | inject a compile/test failure, ask for a fix, verify |
| Review | find real smells with file/line evidence |
| Migration | 3.33→3.40, 21→25, legacy `javax`→`jakarta` |
| Safety | refuse secrets, auth bypass, destructive changes, invented test results |

Do not generate 10,000 paraphrases of “write a Quarkus endpoint”. Generate a **matrix**: phase × layer × difficulty × failure mode.

---

## 2. Teacher choices

| Teacher | Where | Best for | Caveats |
|---|---|---|---|
| **DeepSeek V4.1 Flash** | API | bulk tasks, patches, tests, reviews | cheap; validate JSON; thinking tokens cost output |
| **DeepSeek V4-Pro** | API | hard architecture, migration, judge | ~3–4× price; reserve for hard slices |
| **Qwen3-Coder-30B-A3B-Instruct 4-bit** | local MLX/llama.cpp | private code, offline generation | ~18–20 GB weights; 8K–16K context practical; slower |
| **Qwen3-Coder-Next** | API/rented GPU | hardest repo tasks | 80B; not local |
| **Qwen3.8-27B 4-bit** | local | judge/reviewer | slow; good quality |
| **A frontier API** | API | gold examples, rubric calibration | cost/terms; do not send proprietary code without permission |

Use **Flash for volume**, **Pro for the hard 10–20%**, and a strong local model for private repositories.

---

## 3. Seed sources

Use only code you have the right to use:

- `quarkusio/quarkus-quickstarts` (Apache-2.0).
- `quarkusio/quarkus` (Apache-2.0).
- Quarkus guides (site says CC BY 3.0; check each guide).
- Your own Quarkus repositories (with permission).
- The Stack v3 Java files whose `license_type`/`detected_licenses` are permissive (ODC-BY attribution).
- Real PRs/commits from permissive repos.

Do **not** seed from proprietary code, GPL/AGPL code for a commercial release, leaked data, or repositories with no license.

---

## 4. The eight generation strategies

| # | Strategy | Output | Verification |
|---|---|---|---|
| 1 | **Repo-grounded feature** | task + patch + test | compile + tests |
| 2 | **Bug injection / recovery** | broken commit → fix | failing test then green |
| 3 | **SOLID/hexagonal refactor** | behavior-preserving diff + ArchUnit | tests + ArchUnit |
| 4 | **Test generation** | JUnit/Quarkus tests | tests compile and pass; mutation/coverage optional |
| 5 | **Code/architecture review** | findings with file/line | rubric + human spot check |
| 6 | **Migration** | 3.33→3.40 / 21→25 diff | build + tests |
| 7 | **Architecture Q&A** | explanation grounded in docs | citation + human check |
| 8 | **Negative examples** | refusal/uncertainty | rubric; no tool call |

### 4.1 Repo-grounded feature

Input: a file tree, `pom.xml`, 3–8 relevant files, existing tests, and a task.
Output: a diff that adds the feature and tests that pass.

This is the highest-value bucket. It teaches the exact package shape, imports, annotations and test style.

### 4.2 Bug injection / recovery

Take a working commit, inject a realistic bug (null handling, transaction boundary, missing validation, wrong HTTP status, N+1 query), keep the test failing, and ask for a fix. Or use SWE-smith-java seeds.

The training row should include the **failing output** and the **diagnosis**, not just the final patch. That is how the model learns recovery.

### 4.3 SOLID/hexagonal refactor

Start from a working “big ball of mud” slice. Ask for:

- a domain port;
- an adapter that implements it;
- no JPA entity leaking;
- ArchUnit rules that fail on the old structure;
- behavior preserved.

Verify with the old tests plus the new architecture tests. A refactor that passes both is a strong example; a prose “this is more SOLID” is not.

### 4.4 Test generation

Use Methods2Test for shape, then generate Quarkus tests:

- `@QuarkusTest` + RestAssured for HTTP;
- plain JUnit for domain;
- `@TestTransaction` or Dev Services for persistence;
- no `Thread.sleep`;
- assertions that fail on the old behavior.

Run the tests. Reject tests that pass on the broken code (they assert nothing).

### 4.5 Review

Ask for findings by severity with file/line, principle, issue and minimal fix. The judge checks:

- Is the finding real?
- Is the line plausible?
- Is the fix minimal?
- Does it avoid invented APIs?

Review data is weaker than executed patches; keep it a small slice and label it.

### 4.6 Migration

Two separate tasks:

- **Quarkus 3.33 → 3.40**: BOM, plugin, deprecated APIs, `quarkus-rest` coordinates, test fixes.
- **Java 21 → 25**: compiler release, language features, build/tooling, native image.

Do not mix Quarkus 2 `javax` migration with 3.x tasks unless the task explicitly says “legacy migration”. Quarkus 3 already uses `jakarta.*`.

Quarkus 4 migration is a separate future bucket; it is Beta and changes Jackson, Hibernate, CDI and the Java baseline.

### 4.7 Architecture Q&A

Generate questions from Quarkus guides and your architecture docs:

- “Why does Quarkus do X at build time?”
- “Which extension provides Y?”
- “Where should this transaction boundary live?”
- “Why is a JPA entity in the domain a problem?”

Require a citation (guide section or file) and a short answer. This is knowledge/style data, not a substitute for tests.

### 4.8 Negative examples

At least five families:

1. **Unsafe:** auth bypass, credential logging, `rm -rf`, disabling TLS verification.
2. **Under-specified:** “fix the bug” with no repo, no failing test, no expected behavior.
3. **Out of scope:** modify a dependency, change the license, add a paid service.
4. **Invented API:** uses a Quarkus annotation that does not exist.
5. **False claim:** “all tests pass” without running them.

Teach the model to ask, refuse, or say “I did not run it” — not to bluff.

---

## 5. Prompt templates

The reusable files live in `examples/java/`:

- `teacher-system.md` — global rules.
- `task-prompt.md` — task taxonomy.
- `prompts/solver.md` — plan + patch + tests.
- `prompts/test.md` — JUnit/Quarkus tests.
- `prompts/refactor.md` — SOLID/hexagonal.
- `prompts/review.md` — findings.
- `prompts/migration.md` — version upgrades.

### 5.1 Task generation with DeepSeek

```bash
export DEEPSEEK_API_KEY="..."
python scripts/generate_synthetic.py \
  --mode tasks --tasks 300 --batch 10 \
  --model deepseek-flash \
  --system-file examples/java/teacher-system.md \
  --prompt-file examples/java/task-prompt.md \
  --out data/java/interim/tasks.jsonl
```

The script uses JSON-object mode, retries, and tracks token cost. It does **not** execute anything and does not solve the tasks.

### 5.2 Solving a task

Use the OpenAI-compatible API with `prompts/solver.md` and the repo context. A minimal loop:

```python
import json
import os
from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
template = Path("examples/java/prompts/solver.md").read_text()

for task in tasks:
    prompt = (template
              .replace("{task}", task["task"])
              .replace("{repo_context}", task["repo_context"])
              .replace("{constraints}", json.dumps(task["constraints"]))
              .replace("{acceptance}", json.dumps(task["acceptance"]))
              .replace("{test_command}", task["test_command"]))
    r = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "system", "content": Path("examples/java/teacher-system.md").read_text()},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=8192,
    )
    candidate = json.loads(r.choices[0].message.content)
    # validate schema, apply patch, run tests, keep only verified
```

Use `.replace`, not `.format`: the prompt contains JSON braces.

### 5.3 A solver response

```json
{
  "plan": ["read OrderResource", "add POST", "map DomainException to 422", "add test"],
  "patch": "diff --git a/src/main/java/... b/src/main/java/...\n...",
  "tests": ["src/test/java/com/example/orders/OrderResourceTest.java"],
  "commands": ["./mvnw -q -DskipITs test"],
  "risks": ["validation status may be 400 vs 422"],
  "notes": "uses quarkus-rest and jakarta.validation"
}
```

Validate the patch with `git apply --check` before applying. Reject responses with prose outside JSON.

### 5.4 Repair loop

If the build fails:

1. Capture the first 100 lines of the Maven error.
2. Send the task, patch, and error back to the teacher: “Here is the real compiler/test output; produce a minimal fix.”
3. Max 2 attempts.
4. Keep the final **verified** trajectory, including the failed attempt if it teaches recovery.

Do not let the teacher “fix” by deleting the failing test.

---

## 6. Execution harness

### 6.1 Build the lab image

```bash
docker build -t java-lab -f examples/java/Dockerfile.java-lab .
```

The image has JDK 25, Maven 3.9 and git. Mount a Maven cache at `$HOME/.m2`; the examples set `HOME=/tmp` and mount the host cache at `/tmp/.m2`.

### 6.2 Run a candidate

```bash
rm -rf /tmp/java-work && cp -a ~/src/quarkus-orders /tmp/java-work
cd /tmp/java-work
git checkout <base_commit>

# apply candidate patch (from JSON)
git apply --check model.patch && git apply model.patch

docker run --rm \
  -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /tmp/java-work:/workspace \
  -v "$HOME/.m2":/tmp/.m2 \
  -w /workspace java-lab \
  ./mvnw -q -DskipITs test
```

For Quarkus `@QuarkusTest` with Dev Services (PostgreSQL, Kafka), the container needs Docker-in-Docker or an external service. Simpler:

- use H2 for tests (`quarkus-jdbc-h2`, test profile), or
- run a Postgres service on the lab network and set `quarkus.datasource.jdbc.url`, or
- disable Dev Services and use Testcontainers with a reachable Docker socket (less isolated).

Pick one and pin it. A test harness that needs Docker-in-Docker is fragile.

### 6.3 Offline builds

1. Run one online build to populate `/tmp/.m2` (the mounted cache).
2. Re-run with `-o` (offline).
3. Pin plugin and dependency versions; no snapshots.
4. Use a Maven mirror/proxy if the sandbox has no internet.

### 6.4 ArchUnit

```bash
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /tmp/java-work:/workspace -v "$HOME/.m2":/tmp/.m2 \
  -w /workspace java-lab \
  ./mvnw -q -Dtest=ArchitectureTest test
```

A refactor is only “architecture-verified” if the rule fails on the base commit and passes after the patch. Check both.

### 6.5 Safety

- No host mounts other than the temp repo and the Maven cache.
- No Docker socket, SSH keys, cloud credentials or `.env`.
- No `--network host`.
- Per-command timeout (300–900 s).
- Copy the repo; never edit the original.
- Scan for secrets before sending code to any API.
- Treat model output as hostile until it compiles and tests.

---

## 7. OpenCode as the agent in the loop

OpenCode can read files, edit them, run Maven, and emit JSON events. It is useful when you want a multi-step transcript rather than one-shot JSON.

### 7.1 Configure

`examples/java/opencode.json`:

```json
{
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {"baseURL": "http://127.0.0.1:8080/v1", "apiKey": "local"},
      "models": {"qwen25-coder-14b-java": {"name": "Qwen2.5-Coder-14B Java"}}
    }
  },
  "model": "local/qwen25-coder-14b-java",
  "permission": {"bash": "ask", "edit": "allow", "webfetch": "deny"}
}
```

For a DeepSeek teacher, use the provider block from [07 — Synthetic data](07-synthetic-data.md).

### 7.2 Add project rules

OpenCode reads `AGENTS.md` from the repository. Put your Java/Quarkus rules there (or use a skill): Java 25, Quarkus 3.40, `jakarta.*`, `quarkus-rest`, constructor injection, domain boundaries, `./mvnw test`, no new dependencies without asking.

### 7.3 Run headless

```bash
cd /tmp/java-work
opencode run -m local/qwen25-coder-14b-java --format json \
  "Add a validated POST /api/orders endpoint. Read the existing resource and tests first. Run ./mvnw -q -DskipITs test. Return a diff."
```

For unattended runs, do not enable `--auto` globally; define an agent whose bash permission allows only `./mvnw` and whose edit root is the temporary clone.

Parse the event stream into OpenAI-style messages:

- assistant text → `content`;
- tool call → `tool_calls`;
- tool result → `role: "tool"`;
- reasoning → drop for Qwen2.5-Coder (no thinking template).

Normalize tool arguments to objects for Qwen2.5-Coder. Reject transcripts that claim tests passed without a `run_maven` result.

### 7.4 What OpenCode is good and bad at

**Good:** multi-step repo navigation, edits, test-fix loops, generating a transcript.
**Bad:** bulk throughput, strict JSON schemas, deterministic cost. Use the API for volume; use OpenCode for a few hundred high-quality trajectories and for operating the sandbox.

---

## 8. Quality gates

### 8.1 Hard gates (reject)

- JSON invalid or missing fields.
- Patch does not apply with `git apply --check`.
- Compile fails.
- Tests fail (unless the task is explicitly “diagnose only”).
- Test deleted/weakened to pass.
- `javax.*` imports or Spring idioms in a Quarkus task.
- Wrong Java/Quarkus version.
- New dependency without justification.
- Secret/PII detected.
- License unknown or incompatible.
- Diff > 400 lines or > 10 files without a clear reason.
- Model claims tests passed but no test output exists.

### 8.2 Soft gates (score)

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| Correctness | does not compile | compiles/tests pass | tests cover edge cases |
| Architecture | leaks persistence/HTTP into domain | partial boundaries | clean ports/adapters + ArchUnit |
| Java/Quarkus idiom | Spring/javax/legacy | mixed | Java 25 + Quarkus 3.40 idiomatic |
| Test quality | no assertions | happy path | failure paths + meaningful assertions |
| Scope | unrelated rewrite | some noise | minimal diff |
| Evidence | invented output | partial | real commands/results |

Keep rows with average ≥ 4 and no hard-gate failure. Keep a 10% human-reviewed sample.

### 8.3 Dedup and splits

- MinHash near-dedup at 0.8 on normalized code + task.
- Exact dedup on patch hash.
- Split by **repository and base commit**, not row.
- Decontaminate against MultiPL-E, HumanEval-X, Multi-SWE-bench, SWE-bench Multilingual with 13-gram overlap.
- Keep a time-based holdout: train on commits before a date, test on later commits.

---

## 9. Cost

Assume a task is ~2k input tokens and ~6k output tokens, plus 30% retries and a judge call.

| Examples | Output | Input | Flash off-peak | V4-Pro off-peak |
|---:|---:|---:|---:|---:|
| 1,000 | 7.8M | 2.6M | ~$5 | ~$18 |
| 5,000 | 39M | 13M | ~$25 | ~$90 |
| 20,000 | 156M | 52M | ~$100 | ~$360 |

Add judge calls, repair loops and long architecture tasks; double the estimate for a comfortable budget. Use prompt caching for the stable system/tool block. A local Qwen3-Coder-30B teacher has no API bill but is slower and needs 18–20 GB of memory.

**Do not generate 20k before you have verified 200.** The first 200 expose the schema, sandbox and licensing mistakes that would otherwise be replicated.

---

## 10. Legal and licensing

- **Quarkus code:** Apache-2.0; keep NOTICE/attribution.
- **Quarkus guides:** the site footer says CC BY 3.0; check the specific guide before redistribution.
- **The Stack v3:** ODC-BY; attribution required; per-file licenses still apply.
- **OpenCodeInstruct:** CC-BY-4.0; attribution.
- **Magicoder OSS-Instruct:** MIT, but generated by GPT-3.5; OpenAI usage policy applies.
- **SWE-smith-java:** project MIT; dataset card untagged — verify before release.
- **Multi-SWE-bench:** CC0 for the dataset; source repositories have their own licenses.
- **Your repository:** do not send proprietary code to an API without permission; do not publish an adapter trained on it.
- **Generated code:** may reproduce training code; scan for long overlaps and keep provenance.
- **DeepSeek/OpenCode terms:** review the current terms; record the teacher model and version.

This is not legal advice. For a commercial release, have counsel review the dataset and model licenses.

---

## 11. Failure modes

| Failure | Cause | Fix |
|---|---|---|
| Confident but unbuildable code | no compile/test loop | execute every candidate; keep only passes |
| Invented Quarkus APIs | teacher hallucination | retrieve the guide; compile; reject |
| Patch applies but tests fail | wrong base commit/context | pin commit; include `pom.xml` and tests |
| Tests pass but assert nothing | weak tests | mutation/coverage; review assertions |
| Model learns `javax` | mixed Quarkus 2/3 data | tag versions; filter legacy imports |
| Model learns Quarkus 4 beta | mixed 3.x/4.x data | separate buckets; 4.x eval only |
| Over-refusal | too many negative examples | rebalance; add in-scope positives |
| Data leakage | random split | split by repo/commit; 13-gram decontamination |
| Cost blow-up | thinking + retries + judge | Flash for bulk, Pro for hard, cap tokens |
| License problem | scraped/no-license seed | provenance + license audit |

---

## 12. Sources

- DeepSeek pricing — <https://api-docs.deepseek.com/quick_start/pricing>
- DeepSeek JSON mode — <https://api-docs.deepseek.com/guides/json_mode>
- DeepSeek tool calls — <https://api-docs.deepseek.com/guides/tool_calls>
- OpenCode — <https://opencode.ai/docs>
- Quarkus guides — <https://quarkus.io/guides/>
- Quarkus quickstarts — <https://github.com/quarkusio/quarkus-quickstarts>
- Qwen3-Coder — <https://github.com/QwenLM/Qwen3-Coder>
- Magicoder — <https://arxiv.org/abs/2312.02120>
- Self-Instruct — <https://arxiv.org/abs/2212.10560>
- SWE-smith — <https://arxiv.org/abs/2504.21798>
- Multi-SWE-bench — <https://arxiv.org/abs/2504.02605>
- The Stack v3 — <https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train>
- [07 — Synthetic data](07-synthetic-data.md)
- [20 — Java fine-tuning](20-java-finetuning.md)
