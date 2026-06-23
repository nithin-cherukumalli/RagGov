# Phase C increment 1 (Codex) — Opus verification outcome

**Date:** 2026-06-22. Codex made two engine changes (privacy + contradiction) and reported a gain.
Per discipline, Opus re-ran the full guard suite Codex did NOT run. Outcome below.

## Finding 1 — privacy change REGRESSES protected → REVERTED
Codex tightened `PrivacyAnalyzer` (ignore iphone/email/"private university" without disclosure
intent + person/account target). This removed the 4 PRIVACY false alarms on the heldout BUT broke
the protected baseline: `security_privacy_sensitive_35` (a real privacy case the tool must catch)
is now **misdiagnosed** → protected 41/46, check **FAIL**.
- Action: reverted `privacy.py` and the new privacy test to HEAD. Protected back to 42/46 PASS.
- The intent was right (the 4 heldout PRIVACY alarms are genuine false positives) but the execution
  was too aggressive. Redo as a proper increment that preserves `security_privacy_sensitive_35`.

## Finding 2 — contradiction change is SAFE but does NOT move the metric → kept, honestly
Codex downgrades derivative citation/NCV contradiction signals to UNSUPPORTED_CLAIM when grounding
evidence is non-explicit. Verified: protected PASS, Calib 23/45 (native+default), regression tests
12/12 pass. BUT on the locked gold it only **relabels** the 5 false CONTRADICTED alarms as
UNSUPPORTED (UNSUPPORTED FP 5→10); it does **not** reduce CLEAN-FP.
- Kept: for a governance/faithfulness tool, not falsely screaming "CONTRADICTION" is a real (if
  modest) quality win, and it is regression-free with tests. But it is honestly NOT a numbers gain.

## Honest net effect of increment 1 (after verification)
| metric | pre-increment (locked gold) | after (privacy reverted + contradiction kept) |
|--------|------|------|
| exact | 27/75 = 0.36 | 27/75 = 0.36 |
| CLEAN-FP | 30/46 = 0.65 | 30/46 = 0.65 |
| detection | 18/29 = 0.62 | 18/29 = 0.62 |
| false high-severity CONTRADICTED alarms | 5 | 0 (downgraded to UNSUPPORTED) |

**The headline numbers did not move.** Codex's reported gain (detection 0.76) was an artifact of the
broken, protected-failing state. This is exactly why "Opus re-verifies every sidekick number" is a
hard rule — it caught a protected regression that would otherwise have shipped.

## Lesson logged
Sidekicks may draft engine changes, but every engine/policy change MUST run protected-46 + Calib-45
before it is reported as a win. Codex skipped both. Going forward, sidekicks do read-only diagnostics
to tee up the fix; Opus implements + verifies the trust-bearing change.

## Current state (guard-clean)
privacy.py + test_security.py = HEAD. decision_policy_support.py = increment-1 demotion + gate-v2
revert + Codex contradiction. engine.py = increment-1 warn-fallback. Protected PASS, Calib 23/45,
heldout 27/75 / CLEAN-FP 0.65 / detection 0.62. Only pre-existing prompt-injection test red.

## Next (proper Opus increments)
1. Privacy done right: remove the 4 heldout PRIVACY false alarms WITHOUT breaking
   `security_privacy_sensitive_35` (read what 35 needs; tighten around it).
2. SufficiencyAnalyzer scope-condition (×8) — the largest remaining CLEAN-FP bucket.
