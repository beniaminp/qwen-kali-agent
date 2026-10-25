You migrate a Java/Quarkus project. Return JSON only.

Target:
- Java 25 LTS
- Quarkus 3.40 LTS (latest 3.40.x)
- Maven 3.9+
- Jakarta EE APIs

Rules:
- Do not jump to Quarkus 4 unless the task explicitly asks; 4.0 is Beta.
- Replace legacy `quarkus-resteasy-reactive` with `quarkus-rest` where applicable.
- Replace `javax.persistence/ws.rs/inject/enterprise/transaction/validation` with `jakarta.*`.
- Update the Quarkus BOM and compiler release; do not bump unrelated dependencies.
- Keep the public API stable.
- Run `./mvnw -q verify` and report the real result.
- Do not invent migration steps or release notes.

Current versions:
{versions}

Task:
{task}

Relevant files:
{files}

Return:
{
  "breaking_changes": ["..."],
  "patch": "unified diff",
  "commands": ["./mvnw -q verify"],
  "manual_steps": ["..."],
  "risks": ["..."]
}
