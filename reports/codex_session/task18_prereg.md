# Task 18 pre-registration — multilingual prompt-injection recall

**Date:** 2026-06-18 · written BEFORE code.

## Reframing (corrects the handoff)

The handoff filed Task 18 as "injection detected but not *promoted*" (a decision-policy
problem). Instrumentation disproves that: on the 10 PROMPT_INJECTION probe cases,
`PromptInjectionAnalyzer` returns `status=pass, failure_type=None` for 9 of 10. The
decision policy never sees a `PROMPT_INJECTION` candidate to promote. The one case that
*is* caught (deepset-1, an English "act as a translator" role attack) promotes correctly
to primary already. So the engine's blocking-tier policy is fine; the gap is **detection
recall**, not ranking.

## Root cause (established)

`PromptInjectionAnalyzer` (`src/raggov/analyzers/security/injection.py`) matches
English-only patterns (`IMPERATIVE_DIRECTIVES`, jailbreak, exfiltration, etc.) and an
English-only `SemanticIntentClassifier`. The 9 missed probe payloads carry override /
new-instruction / role markers in other languages:
- DE: "Abweichend zu vorherigen instruktionen", "Vergiss alles davor",
  "Nun folgen neue Anweisungen", "Formuliere ... eine unwahre Überschrift".
- ES: "Olvide todo que digo antes".
- romanized Hindi + obfuscated exemplar attacks (genuinely hard; not all reachable by regex).

This is in-bounds for discipline rule #3 (no text heuristics for *pipeline* modes):
injection detection is content-scanning **by design**; rule #3 targets retrieval/chunking/
staleness modes, not the security stage.

## Change (one, narrow)

Add a `MULTILINGUAL_DIRECTIVES` pattern group (DE/ES/FR + a few romanized markers) for the
two highest-precision attack classes only — `task_override` (ignore/forget/deviate-from
previous instructions; follow new instructions) and `role_reassignment` (act as / pretend).
Append it to the compiled pattern set. No threshold change, no new analyzer, no policy edit.
Patterns are anchored on instruction/role nouns to avoid firing on ordinary foreign-language
prose.

## Measured baseline (BEFORE — reproduced this session)

- Protected: 41/46 GREEN.
- Calib scored primary: 23/45 = 0.511.
- Probe overall: 35/145 = 0.241.
- Probe PROMPT_INJECTION: **1/10**.
- Probe CLEAN → PROMPT_INJECTION false positives: **0** (none of the 30 CLEAN cases flagged
  as injection today).

## Hard acceptance criteria

1. Protected baseline stays **41/46 GREEN**. *(revert trigger)*
2. Calib scored primary stays **≥ 23/45 = 0.511**. *(revert trigger)*
3. Probe CLEAN → PROMPT_INJECTION false positives stay **0** (no healthy answer newly flagged
   as injection). *(revert trigger)*
4. Probe overall accuracy does **not decrease** (≥ 35/145). *(revert trigger)*

**Success criterion (the point of the change):**
5. Probe PROMPT_INJECTION rises to **≥ 7/10** (from 1/10).

If 1–4 hold but 5 lands below 7/10, keep the change only if it is a *strict* improvement on
injection recall with zero precision regression (1–4 all green), and document the residual
hard cases (obfuscated/romanized exemplar attacks) honestly. If any of 1–4 fails → revert,
keep this record.
