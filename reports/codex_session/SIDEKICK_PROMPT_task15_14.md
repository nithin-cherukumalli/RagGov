# Codex Sidekick Prompt — groundwork for Tasks 15 then 14

Paste this to Codex. It is READ-ONLY groundwork only. Opus implements afterward.

---

You are Codex, the RagGov sidekick supporting Claude Opus 4.8 on engine precision.
Your job is read-only groundwork that lets Opus implement fast. Follow the rules in
`reports/codex_session/codex_sidekick_session_plan.md` ("Non-Negotiable Rules" and
"Closeout Format"). Append your findings to a new section of that ledger; do not create
scattered new report files.

ABSOLUTE CONSTRAINTS
- Do NOT edit engine, analyzer, decision-policy, claims, label, fixture, threshold, gate,
  or production-flag files. No code changes at all — groundwork only.
- Do NOT revert or touch Opus/user changes or dirty files.
- Measure on the induced probe + protected baseline + Calib, reproduced (not quoted).
- Keep `calibration_status=not_production_calibrated`, `production_gating_eligible=false`.

ENV SETUP (sandbox is Python 3.10; the repo needs 3.11 shims)
```
mkdir -p /tmp/shim
printf 'import datetime as _dt\nif not hasattr(_dt,"UTC"): _dt.UTC=_dt.timezone.utc\n' > /tmp/shim/sitecustomize.py
printf 'from tomli import *\nfrom tomli import load, loads\n' > /tmp/shim/tomllib.py
pip install -q pytest pydantic typer rich numpy httpx tomli --break-system-packages
export PYTHONPATH=/tmp/shim:src:.
```
Reproduce baseline first: `python scripts/check_protected_baseline.py` (must be 41/46),
and score the induced probe `evals/govrag_calib/staging/raw/induced_candidates.jsonl` and
Calib `evals/govrag_calib/govrag_calib_150.jsonl` (split in train/dev/heldout) by building a
`RAGRun` per row (`RetrievedChunk(chunk_id,text,source_doc_id=doc_id,score)`,
`final_answer=answer`, `cited_doc_ids=citations`) and comparing `diagnose(run).primary_failure`.
Current anchors to confirm: probe ~80/145=0.552, Calib 23/45.

== PRIMARY TASK: Task 15 groundwork (lowest risk — pure stage attribution) ==
Target xfail test:
`tests/test_analyzers/test_answer_quality_confidence_metadata.py::test_quality_incomplete_38_has_generation_stage_candidate_if_supported`
Also see: `tests/test_pr5e_answer_quality.py::test_incomplete_answer_with_good_context_stage_generation`.

Known shape (from `task14_15_16_groundwork.md`): primary is correctly `UNSUPPORTED_CLAIM`;
only `root_cause_stage` is wrong (`GROUNDING`, expected `GENERATION`). Selected analyzer
`ClaimGroundingAnalyzer`; selection reason mentions `_suppress_citation_when_downstream_symptom`.
`AnswerQualityAnalyzer` has a GENERATION-stage candidate that loses trace selection.

Produce, in the ledger, all of:
1. The exact failing assertion (run with `--runxfail`) and the current vs expected
   `primary_failure`, `root_cause_stage`, selected analyzer.
2. The full decision trace for case 38: every candidate (analyzer, failure_type, stage, tier,
   weight, evidence_summary), the winner, and the suppression reason for the
   `AnswerQualityAnalyzer` GENERATION candidate.
3. The exact file+function+line where stage attribution / trace selection picks GROUNDING over
   the GENERATION candidate (likely in `decision_policy_support.py` or engine trace selection,
   and `analyzers/answer_quality/analyzer.py`).
4. Whether the `AnswerQualityAnalyzer` GENERATION candidate even exists in the pool for case 38,
   or must be produced (and under what condition: "context sufficient + answer omits required content").
5. A list of every other test that currently asserts `root_cause_stage`/analyzer for incomplete
   answers (grep `GENERATION`, `root_cause_stage`, `AnswerQualityAnalyzer`) so Opus knows the
   regression surface.
6. A filled prereg (hypothesis, exact change locus, hard criteria: primary stays
   `UNSUPPORTED_CLAIM`; protected 41/46; Calib >=23/45; probe overall not down; named test flips;
   no other stage-attribution test regresses; success = stage `GENERATION` + analyzer
   `AnswerQualityAnalyzer` for case 38).

== SECONDARY TASK: Task 14 groundwork (higher risk — profile staleness) ==
Target xfail: `tests/test_analyzers/test_version_validity_pipeline.py::test_stale_irrelevant_source_does_not_primary_fail`.
Root cause (confirmed): `profile.stale_doc_ids` is built from ABSOLUTE age in
`src/raggov/analyzers/retrieval/evidence_profile.py`, so it flags even the fresh, cited, active
`doc-ceo` (2024); `StaleRetrievalAnalyzer._from_profile` then fails STALE on both docs.

Produce, in the ledger:
1. The exact code in `evidence_profile.py` that populates `stale_doc_ids` (function, lines) and
   whether a "strictly newer dated retrieved alternative" and per-chunk query-relevance are
   available at that point.
2. For the protected case `version_stale_not_cited_32`: dump its chunks/docs, which are
   query-relevant vs noisy, citation status, and dated alternatives — so Opus can find a predicate
   that drops Task-14's `doc-lease` (irrelevant) and fresh `doc-ceo` WITHOUT dropping case 32.
3. Every test/fixture that depends on `stale_doc_ids` or STALE_RETRIEVAL primary (regression surface).
4. A candidate predicate (described, not coded) and a filled prereg with hard criteria including
   "protected baseline stays 41/46 incl. version_stale_not_cited_32" and "Calib gc rows with STALE
   gold stay STALE".

CLOSEOUT (use the ledger's Closeout Format): files inspected, changes made (None),
method status, tests run + results, protected files changed (no), labels/gates changed (no),
known limitations, next recommended step. Do not implement — hand back to Opus.
