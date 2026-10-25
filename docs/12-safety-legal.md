# 12 — Safety, law and licenses

*This chapter is engineering guidance, not legal advice. Laws and terms change; consult a lawyer for your jurisdiction and use case.*

A penetration-testing model is dual-use. The same capability that helps a defender validate a patch can help someone attack a system they do not own. The difference is **authorization**, **scope**, **intent**, and **control**. This chapter makes those four things explicit in the system, the data, and the release process.

---

## 1. Authorization is a precondition, not a prompt

Before the first scan:

1. **Written authorization** from the system owner or an authorized representative.
2. **Rules of engagement (RoE):** targets, excluded hosts, time window, allowed techniques, forbidden techniques, escalation contacts, data handling.
3. **A scope file** that the policy layer parses mechanically.
4. **A named human** accountable for the engagement.
5. **Insurance/contract** where required by the engagement.

If any of these is missing, the agent does not run. “The user said it's fine” is not authorization.

### Minimum RoE contents

| Item | Example |
|---|---|
| Client / owner | Lab owner or authorized representative |
| Targets | `10.10.10.0/24`, specific hostnames |
| Exclusions | `10.10.10.1` (hypervisor), production segments |
| Time window | 2026-10-25 08:00Z → 2026-10-26 08:00Z |
| Allowed techniques | read-only recon, authenticated web testing |
| Forbidden techniques | DoS, social engineering, data exfiltration, destructive commands |
| Data handling | no customer data; findings stored encrypted; delete after 30 days |
| Contacts | technical POC, emergency stop, escalation |
| Reporting | format, deadline, disclosure policy |

The agent's `scope.yaml` should be generated from the RoE, not typed from memory.

---

## 2. Jurisdictions (summary)

| Jurisdiction | Relevant law | Practical implication |
|---|---|---|
| **United States** | Computer Fraud and Abuse Act (CFAA), 18 U.S.C. § 1030; state laws | Unauthorized access or exceeding authorized access can be criminal. Written authorization and scope matter. |
| **United Kingdom** | Computer Misuse Act 1990 | Unauthorized access, unauthorized modification, and tools/articules for misuse. Authorization is central. |
| **European Union** | GDPR (personal data), NIS2 (critical infrastructure), national computer-misuse laws | Data minimization, lawful basis, breach notification, security obligations. |
| **Other** | National computer-misuse and data-protection statutes | Do not assume a US/UK analysis transfers. |

**Not legal advice.** A bug bounty program's scope is not a blanket authorization; read its terms. A lab VM you own is not a license to scan the host network. A “practice” target can still be a third party's system.

---

## 3. Data handling

Training data and logs can contain sensitive material even in a lab.

- **No real credentials.** Generate lab-only credentials; rotate anything that leaks.
- **No customer data.** Use synthetic or public data; if you must use real data, minimize, encrypt, and set a retention period.
- **No live flags.** CTF flags are often still valid; do not train on them.
- **Scrub secrets** before training: API keys, private keys, JWTs, password assignments.
- **Logs:** store audit logs outside the sandbox, restrict access, and delete according to the RoE.
- **Model artifacts:** an adapter can memorize data. Treat it as sensitive if the training data was sensitive.
- **Sharing:** do not upload a model trained on private data to a public hub.

A useful rule: if you would not paste it into a public GitHub issue, do not put it in a public dataset.

---

## 4. Model and dataset licenses

| License | Examples | Publish the adapter? |
|---|---|---|
| Apache-2.0 | Qwen models, ToolACE, Hermes, Toucan, OpenThoughts, Trendyol | Yes, with NOTICE/attribution |
| MIT | SWE-Gym, SWE-smith | Yes |
| CC-BY-4.0 | xLAM, When2Call, KaliBench, CyberMetric mirror | Yes, with attribution |
| CC-BY-NC-4.0 | **APIGen-MT-5k** | **No** for commercial/published use |
| CC-BY-NC-SA-4.0 | SecEval mirror | **No**; share-alike compounds |
| GPL-2.0 | NYU CTF Bench, ExploitDB | Evaluation only; avoid in a release mix |
| “Other”/custom | WhiteRabbitNeo, MITRE ATT&CK | Read the terms; assume restricted |
| None | many scraped writeup sets | **No** — all rights reserved by default |

**Three traps:**

1. **Non-commercial contagion.** CC-BY-NC data can make the resulting model unpublishable for commercial use. Whether weights are a “derivative” is legally unsettled; treat NC as internal-only.
2. **Copyleft.** GPL data is fine for internal evaluation but risky to redistribute. Do not include it in a released dataset or a commercial model without advice.
3. **License laundering.** A mirror that relicenses xLAM as MIT cannot override Salesforce's CC-BY-4.0. Trace every dataset to its origin.

The Qwen models themselves are Apache-2.0 as of 25 Oct 2026. That covers the base weights, not your data or your adapter's training data.

---

## 5. Dual-use and export

- **Dual-use:** the model can be used for defense or offense. Document intended use, limitations, and prohibitions in the model card.
- **Export controls:** some jurisdictions regulate intrusion software or “technology” for offensive cyber. A model that generates exploitation tooling could be in scope. Get advice before distributing internationally.
- **Sanctions:** do not provide the model or services to sanctioned parties.
- **Terms of service:** your model host, API provider, and dataset sources have terms. Violating them can terminate access and create liability.

---

## 6. Guardrails the system must enforce

| Control | Implementation |
|---|---|
| Scope | Parse `scope.yaml`; compare every target (IP/CIDR/domain) before execution. |
| Allowlist | Only approved binaries/tools; reject unknown commands. |
| Denylist | Destructive patterns (`rm -rf /`, `mkfs`, `dd of=/dev/`, fork bombs, `shutdown`). |
| Confirmation | Human approval for active enumeration, auth attempts, and anything not read-only. |
| Egress | Lab network `internal: true`; no route to the internet. |
| Timeout | Per-command wall clock; per-task budget. |
| Rate limit | Max tool calls per task; max requests per minute. |
| Audit | Append-only JSONL outside the sandbox; hash outputs. |
| Kill switch | Stop agent, kill child processes, tear down lab. |
| Model policy | Refusal/scope training; evaluate unsafe-action rate. |

**Defense in depth:** the model should refuse, the policy layer should deny, and the sandbox should make the action impossible. Any one of these can fail.

---

## 7. Incident response

If the agent does something unexpected:

1. **Stop** the agent and the lab (`docker compose down`, kill the bridge).
2. **Preserve** the audit log and the model's output; do not edit them.
3. **Identify** the target and whether it was in scope.
4. **Notify** the engagement owner and, if required, the affected party.
5. **Contain** any data exposure; rotate credentials.
6. **Root-cause** the failure: prompt, tool schema, policy gap, model behavior, or sandbox escape.
7. **Fix and re-test** the guardrail; add a regression example to the golden set.
8. **Document** the incident and the remediation.

A near miss is a finding. Treat it as one.

---

## 8. Release checklist

Before publishing an adapter, GGUF, or dataset:

- [ ] written authorization for the data and the evaluation targets;
- [ ] dataset manifest with source IDs, licenses, hashes, and row counts;
- [ ] license audit: no NC/SA/GPL/unknown data in a release mix;
- [ ] secret/PII scan clean;
- [ ] benchmark contamination reported;
- [ ] evaluation report with UAR = 0 on the forbidden-action suite;
- [ ] model card with intended use, limitations, and safety notes;
- [ ] no real credentials, customer data, or live flags in the repo;
- [ ] a named owner for maintenance and vulnerability reports;
- [ ] a way to withdraw the model if a serious issue is found.

If you cannot check every box, release privately or not at all.

---

## 9. Ethics

- **Do not automate harm.** Keep a human accountable for scope and impact.
- **Respect the target.** A lab VM is not a license to attack the hypervisor or the LAN.
- **Be honest about capability.** A 14B agent is a assistant, not an autonomous pentester. Do not market it as one.
- **Disclose responsibly.** If you find a real vulnerability during testing, follow the agreed disclosure process.
- **Minimize data.** Collect only what the task needs; delete when done.
- **Document uncertainty.** The model should say “I did not observe that” rather than invent a result.

---

## 10. Sources

- CFAA — <https://www.justice.gov/criminal/cybercrime-intellectual-property-section/computer-fraud-and-abuse-act>
- UK Computer Misuse Act 1990 — <https://www.legislation.gov.uk/ukpga/1990/18/contents>
- GDPR — <https://gdpr-info.eu/>
- NIS2 — <https://digital-strategy.ec.europa.eu/en/policies/nis2-directive>
- Qwen3-14B license — <https://huggingface.co/Qwen/Qwen3-14B/blob/main/LICENSE>
- APIGen-MT-5k license — <https://huggingface.co/datasets/Salesforce/APIGen-MT-5k>
- KaliBench license — <https://huggingface.co/datasets/RISys-Lab/KaliBench>
- NYU CTF Bench — <https://arxiv.org/abs/2406.05590>
- OWASP testing guide — <https://owasp.org/www-project-web-security-testing-guide/>
- NIST cybersecurity framework — <https://www.nist.gov/cyberframework>
