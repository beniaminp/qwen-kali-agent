Propose {n} diverse Java/Quarkus training tasks for a repository-grounded coding
agent. Cover this mix, in roughly equal parts:

- quarkus-rest: endpoint, validation, DTO, exception mapping
- panache: repository/active-record choice, transaction boundary, query
- solid: SRP, OCP, LSP, ISP or DIP refactor with no behavior change
- hexagonal: move persistence or HTTP behind a port, add ArchUnit rules
- test: JUnit 5 or @QuarkusTest for an existing behavior
- review: find architecture/code smells without changing code
- migration: Quarkus 3.33 -> 3.40 or Java 21 -> 25
- negative: out-of-scope, unsafe, or under-specified request

Each task must include a concrete acceptance test and a test command.

Return:
{
  "tasks": [
    {
      "id": "short-stable-id",
      "category": "quarkus-rest|panache|solid|hexagonal|test|review|migration|negative",
      "difficulty": "easy|medium|hard",
      "task": "what the agent must do",
      "repo_context": ["files or modules the agent should read"],
      "constraints": ["Java 25", "Quarkus 3.40", "jakarta.* only", "..."],
      "acceptance": ["what must pass", "which tests"],
      "test_command": "./mvnw -q -DskipITs test",
      "why": "what capability this trains"
    }
  ]
}
