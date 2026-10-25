# Java / Quarkus agent rules

These rules apply to coding-agent runs in this repository.

## Platform

- Java 25 LTS, `--release 25`.
- Quarkus 3.40 LTS (latest 3.40.x).
- Maven 3.9+; use `./mvnw`.
- Jakarta EE APIs; `jakarta.*` only.

## Non-negotiable

- Never emit `javax.persistence`, `javax.ws.rs`, `javax.inject`,
  `javax.enterprise`, `javax.transaction` or `javax.validation`.
- Prefer `quarkus-rest` over `quarkus-resteasy-reactive`.
- Constructor injection only; no field injection, no `@Autowired`.
- Domain must not import Quarkus, Panache, Jakarta persistence or HTTP types.
- JPA entities must not escape the persistence adapter.
- Add or update tests; run `./mvnw -q -DskipITs test`.
- Never claim tests passed without a real test run.
- Never add dependencies or change the Java version without saying so.
- Never print secrets, tokens or personal data.

## Output

- Single file: full file in a `java` block.
- Multi-file: unified diff in a `diff` block.
- Review: findings by severity with file/line and a minimal fix.
- Migration: breaking changes, diff, verification commands.

## Architecture

- Hexagonal boundaries: `domain → application → adapters`.
- SOLID where it removes a real coupling; no abstraction for its own sake.
- ArchUnit rules for domain independence and package cycles.
- `@QuarkusTest` + RestAssured at HTTP boundaries; plain JUnit for domain.
