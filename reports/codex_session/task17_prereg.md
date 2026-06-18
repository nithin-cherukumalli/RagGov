# Task 17 pre-registration — CLEAN over-firing via STALE_RETRIEVAL

**Date:** 2026-06-17 · written BEFORE any code change.

## Problem

The generalization probe (`generalization_probe_v1.md`) shows the engine
over-fires on CLEAN inputs. The dominant driver is **STALE_RETRIEVAL false
positives**: on 30 induced CLEAN wiki cases (HotpotQA/ALCE, empty chunk
metadata), `TemporalSourceValidityAnalyzerV1` and `RetrievalDiagnosisAnalyzerV0`
each fire STALE_RETRIEVAL 11×, and it becomes the primary verdict 5×.

Root cause (from code reading): `TemporalSourceValidityAnalyzerV1` builds a
`VersionValidityReport` using an **assumed query date + heuristic age check**,
so it can mark documents "stale" even when there is **no temporal/version
signal at all** in the evidence. `RetrievalDiagnosisAnalyzerV0` then reads that
same `run.version_validity_report` and re-emits STALE_RETRIEVAL. So both
over-firers derive from one report.

## Hypothesis

Staleness cannot be concluded without a temporal signal. If the version-validity
analyzer **skips when no temporal/version evidence is present** (in
`chunk.metadata`, `corpus_entries` timestamps, or `run.metadata`), the
false-positive STALE verdicts disappear, while true positives are preserved
because real STALE cases carry such signals (gc-012/gc-013 have `version`
metadata).

## Scope (one change)

Add a pipeline-agnostic precondition to `TemporalSourceValidityAnalyzerV1.analyze`:
if no temporal/version signal exists in structured evidence, return `skip`
(producing no stale report). This also stops `RetrievalDiagnosisAnalyzerV0`
over-firing because it consumes that report.

Predicate reads **metadata key presence and corpus timestamps only** — never the
query string, never passage text, never dataset-specific values.

## Measured baseline (BEFORE)

- Protected baseline (common benchmark): **41/46 GREEN**.
- Calib scored primary (live `govrag_calib_150`, train+dev+heldout, no placeholders):
  **23/45 = 0.511**.
- Real STALE cases: gc-012 ✅ STALE, gc-013 ✅ STALE, gc-011 already → CLEAN
  (pre-existing miss, not in scope).
- Probe CLEAN: **3/30 correct**, **5 STALE false positives**.

## Hard acceptance criteria

1. Protected baseline stays **≥ 41/46 GREEN**.
2. Calib scored primary **≥ 0.511**, and **gc-012 & gc-013 remain STALE_RETRIEVAL**.
3. Probe: STALE_RETRIEVAL false positives among the 30 CLEAN drop **5 → 0**, and
   probe CLEAN-correct **strictly increases** (> 3).
4. No new dangerous-miss: no real STALE/failure case flips to CLEAN (esp. gc-012/013).
5. Predicate is pipeline/domain-agnostic (metadata-only).

**Revert trigger:** if criterion 1, 2 (gc-012/013), or 4 fails → revert the
change, keep this pre-registration as the historical record.

## Out of scope (separate follow-ups)

CHUNKING_BOUNDARY_ERROR / ParserValidation over-firing, INCONSISTENT_CHUNKS,
citation over-eagerness (Task 16 / 3-v2), injection promotion (Task 18).
