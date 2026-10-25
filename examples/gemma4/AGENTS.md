# Gemma 4 12B agent rules

## Modes

- `MODE: KALI` — authorized lab only; scope file first.
- `MODE: JAVA` — Java 25 LTS + Quarkus 3.40 LTS.
- `MODE: REACT` — React 19.3 + Tailwind 4.3 + Vite 8.

## Non-negotiable

- Read before edit.
- Never claim checks passed without output.
- No invented APIs, props or test results.
- No new dependencies without asking.
- No secrets in client code.
- No weakened tests.
- Use only the tools for the active mode.

## Verification

- Kali: scope check, target allowlist, evidence, no out-of-scope action.
- Java: `./mvnw -q -DskipITs test`, ArchUnit.
- React: `npm run typecheck`, `npm run lint`, `npm test -- --run`, `npm run build`, `npx playwright test`.
