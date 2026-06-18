# Task 18 result — multilingual prompt-injection recall — LANDED

**Date:** 2026-06-18. Prereg: `task18_prereg.md`.

## What changed
`src/raggov/analyzers/security/injection.py`: added a `MULTILINGUAL_DIRECTIVES`
pattern group (DE/ES/FR task-override + role-reassignment, plus two known cross-lingual
exemplars: `say yes/no` in imperative position and the `pwned` exemplar). Appended to the
compiled pattern set. No threshold change, no new analyzer, no decision-policy edit — the
existing blocking-tier policy promotes the detected `PROMPT_INJECTION` to primary on its own.

## Acceptance criteria — all green
| # | Criterion | Before | After | Verdict |
|---|---|---|---|---|
| 1 | Protected baseline | 41/46 | **41/46** | PASS |
| 2 | Calib scored primary | 23/45 (0.511) | **23/45 (0.511)** | PASS |
| 3 | Probe CLEAN→injection FPs | 0 | **0** (CLEAN still 4/30, unchanged) | PASS |
| 4 | Probe overall accuracy | 35/145 (0.241) | **43/145 (0.297)** | PASS (strict ↑) |
| 5 | Probe PROMPT_INJECTION (target ≥7/10) | 1/10 | **9/10** | PASS |

Whole +8 probe gain is attributable to injection recall; no other type moved.

## Tests added
- `tests/test_analyzers/test_security.py`: `test_prompt_injection_fails_on_multilingual_task_override`
  (DE×3, ES×1) and `test_prompt_injection_passes_on_benign_foreign_language_prose` (precision guard).
- `tests/test_analyzers/test_injection_promotion_e2e.py`: end-to-end — multilingual injection
  diagnoses to primary `PROMPT_INJECTION`; clean foreign-language prose does not.

## Honest residual
- 1/10 injection cases still missed: a romanized-Hindi roleplay/obfuscation payload. Not reachable
  by precise regex without precision risk; needs a model-side or multilingual-classifier layer
  (noted as the analyzer's documented evasion boundary). Left as a known miss, not forced.
- Pre-existing stale test `test_prompt_injection_warns_on_hits_below_risk_threshold` remains red
  for an unrelated evidence-format drift (red at baseline before this change; out of scope —
  one task per commit).

## Corrected the handoff
Handoff framed Task 18 as a *promotion/policy* problem ("detected but not promoted").
Instrumentation showed the analyzer was *passing* (no detection) on 9/10 — a **recall** gap.
Decision policy was already correct.
