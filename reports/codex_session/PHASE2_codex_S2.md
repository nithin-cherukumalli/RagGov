# Phase 2 — Codex (S2): human-audit worklist for the 25 contradiction labels + NLI provider readiness

READ-ONLY on engine/analyzer/policy/labels/gates/canonical dataset. Standalone `scripts/` + staging
+ ledger only. Reproduce on current committed code.

## Env (same shim as before; PYTHONPATH=/tmp/shim:src:.)

## Why
The real heldout v1 (`heldout_real_v1.jsonl`, 75 rows) gave the first honest number 0.24. The 25
CONTRADICTED rows carry RAGTruth-migrated labels (heuristic) and must be human-audited before they
anchor any trust claim. And Opus is building the entailment (NLI) tier — we need to know exactly how
a real provider plugs in.

## Tasks
1. **Human-audit worklist for the 25 CONTRADICTED rows.** For each: source_id, query, the answer,
   the cited passage text, and a one-line "is this a genuine contradiction (answer asserts X, source
   says not-X)?" prompt with a yes/no/unsure field for a human to fill. Output
   `reports/calibration/contradiction_audit_worklist.md`. Classify each as: (a) clear contradiction,
   (b) actually unsupported (not contradicted), (c) actually fine/mislabeled. Do NOT finalize — this
   is the sheet a human fills; flag your provisional read.
2. **Also spot-check 10 of the 50 CLEAN rows** the same way (are the reference answers truly
   faithful to their passages?) — since CLEAN scored only 24%, we must rule out bad CLEAN labels vs
   genuine over-firing. Output to the same worklist under a CLEAN section.
3. **NLI provider readiness note.** Inspect `src/raggov/analyzers/grounding/verifiers.py`
   (`ClaimEntailmentVerifierV1`, `conservative_ensemble`), the refchecker adapter, and
   `claim_grounding_verifier_policy`. Document the EXACT interface a real entailment provider must
   implement (method signature, inputs/outputs), how to enable it via config, what local/offline NLI
   options exist (e.g. a small HF NLI model), and how it degrades visibly when absent. This is the
   spec Opus codes against — accuracy matters; verify by reading the code, not assuming.

## Closeout
Files inspected; worklist + provider-readiness note paths; provisional audit counts (a/b/c);
protected/labels/gates changed (no); next step. HAND BACK — Opus owns label acceptance + the wiring.
