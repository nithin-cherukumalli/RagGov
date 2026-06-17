# Calibration Dataset — Label Changelog

Every change to a golden (`expected_*`) field or to the case-ID set goes here.
Format per entry: date · case id(s) · field · old → new · rationale · adjudicator.

After any change, run `python scripts/check_dataset_lock.py --regenerate`.

---

## 2026-06-17 — `dataset_v1` frozen (baseline)

The file `govrag_calib_150.jsonl` is frozen as `dataset_v1` at SHA256
`fd55e009…`. This is the baseline; entries above this line going forward record
deviations from it.

### Pre-freeze discrepancy (UNRECONCILED — needs human adjudication)

A comparison between `reports/codex_session/tasks_3_4_5_result.md` (2026-06-15) and
the current file shows the labels referenced by those bare case numbers no longer
match. **It is not yet known whether labels were edited or whether cases were
renumbered** — both are governance failures this changelog now prevents. Recorded
here for a human to reconstruct (do NOT assume the v1 doc or the current file is
"correct" without checking the actual case content):

| case # (per v1 doc) | label in v1 result doc | current golden in dataset_v1 |
|---|---|---|
| 025 | UNSUPPORTED_CLAIM | CITATION_MISMATCH |
| 029 | CITATION_MISMATCH | UNSUPPORTED_CLAIM |
| 030 | CITATION_MISMATCH | CONTRADICTED_CLAIM |
| 004 | RETRIEVAL_DEPTH_LIMIT | CLEAN |
| 011 | CLEAN | STALE_RETRIEVAL |
| 012 | CLEAN | STALE_RETRIEVAL |
| 023 | UNSUPPORTED_CLAIM | CONTRADICTED_CLAIM |
| 020 | RERANKER_FAILURE (target) | UNSUPPORTED_CLAIM |

**Adjudicator:** _unassigned_ — requires a human to (a) confirm whether v1's "025"
is today's `gc-025`, (b) decide the correct label for each, (c) record the decision
as a dated entry below. Until then, prior (pre-2026-06-17) accuracy numbers should
be treated as **not comparable** to current ones.

---

<!-- New entries below this line, newest first. -->
