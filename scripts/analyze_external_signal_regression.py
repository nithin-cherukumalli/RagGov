"""Per-case diff harness: native vs external-enhanced diagnosis comparison.

Runs every gold-set case twice (native + external-enhanced) and outputs:
  - Per-case structured diffs (JSONL)
  - External signal trace per consumed signal (JSONL)
  - Human-readable Markdown summary

Usage:
    python scripts/analyze_external_signal_regression.py --gold-set v1
    python scripts/analyze_external_signal_regression.py --gold-set v2 --format markdown
    python scripts/analyze_external_signal_regression.py --case retrieval_noise --dump-traces

Output files (written to reports/):
    external_signal_regression_diff.jsonl
    external_signal_trace.jsonl
    external_signal_regression_analysis.md
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raggov.engine import diagnose
from raggov.models.chunk import RetrievedChunk
from raggov.models.corpus import CorpusEntry
from raggov.models.run import RAGRun

REPORTS_DIR = ROOT / "reports"
GOLD_V1 = ROOT / "tests" / "fixtures" / "govrag_pinpoint_eval" / "pinpoint_eval_gold_v1.json"
GOLD_V2 = ROOT / "tests" / "fixtures" / "govrag_pinpoint_eval" / "pinpoint_eval_gold_v2_30.json"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ExternalSignalTrace:
    """One external signal and its downstream consumption path."""
    case_id: str
    provider: str
    signal_type: str
    metric_name: str
    value: Any
    label: str | None
    affected_chunk_ids: list[str]
    affected_claim_ids: list[str]
    raw_payload: dict | None
    consumed_by_analyzer: list[str]
    consumed_by_ncv_node: list[str]
    consumed_by_a2p_rule: list[str]
    final_influence: str  # none | advisory | changed_node | changed_root_cause | changed_primary


@dataclass
class CaseDiff:
    """Per-case comparison of native vs external-enhanced diagnosis."""
    case_id: str
    description: str

    expected_primary_failure: str
    expected_first_failing_node: str | None
    expected_pinpoint_node: str | None
    expected_root_cause: str | None

    native_primary_failure: str
    external_primary_failure: str

    native_first_failing_node: str | None
    external_first_failing_node: str | None

    native_pinpoint_node: str | None
    external_pinpoint_node: str | None

    native_root_cause: str | None
    external_root_cause: str | None

    changed_primary_failure: bool
    changed_first_failing_node: bool
    changed_pinpoint_node: bool
    changed_root_cause: bool

    native_correct_primary: bool
    native_correct_node: bool
    native_correct_root_cause: bool
    external_correct_primary: bool
    external_correct_node: bool
    external_correct_root_cause: bool

    # External signal provenance
    external_signals_present: int
    external_signal_provider_names: list[str]
    external_signal_types: list[str]
    external_signal_labels: list[str]
    external_signal_metric_names: list[str]
    external_signal_values: list[Any]

    which_analyzer_consumed_external_signal: list[str]
    which_report_changed_after_external_signal: list[str]

    # A2P internals
    native_a2p_primary_cause: str | None
    external_a2p_primary_cause: str | None
    native_security_failure_present: bool
    external_security_failure_present: bool
    native_ncv_security_risk_status: str | None
    external_ncv_security_risk_status: str | None

    # Regression classification
    regression_type: str  # none | node_regression | root_cause_regression | both | primary_regression
    regression_evidence: list[str]

    # Traces (optional)
    native_trace: dict | None = None
    external_trace: dict | None = None


@dataclass
class RegressionSummary:
    total_cases: int
    native_primary_accuracy: float
    external_primary_accuracy: float
    native_node_accuracy: float
    external_node_accuracy: float
    native_root_cause_accuracy: float
    external_root_cause_accuracy: float

    native_correct_external_wrong_primary: int
    native_wrong_external_correct_primary: int
    both_wrong_primary: int
    both_correct_primary: int

    native_correct_external_wrong_node: int
    native_wrong_external_correct_node: int
    both_wrong_node: int
    both_correct_node: int

    native_correct_external_wrong_root_cause: int
    native_wrong_external_correct_root_cause: int
    both_wrong_root_cause: int
    both_correct_root_cause: int

    external_only_improved_any: int
    external_only_degraded_any: int

    # Confusion tables (expected → predicted)
    primary_confusion_native: dict[str, dict[str, int]]
    primary_confusion_external: dict[str, dict[str, int]]
    node_confusion_native: dict[str, dict[str, int]]
    node_confusion_external: dict[str, dict[str, int]]
    root_cause_confusion_native: dict[str, dict[str, int]]
    root_cause_confusion_external: dict[str, dict[str, int]]

    # External signal analysis
    total_external_signals: int
    signals_that_changed_node: int
    signals_that_changed_root_cause: int
    top_degraded_node_transitions: list[tuple[str, str, int]]  # (native, external, count)
    top_degraded_root_cause_transitions: list[tuple[str, str, int]]

    security_risk_over_trigger_count: int
    root_cause_collapse_count: int
    generic_root_causes_in_external: Counter

    per_case: list[CaseDiff] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------

def _native_config(case_config: dict) -> dict:
    return {
        "mode": "native",
        "enable_ncv": True,
        "enable_a2p": True,
        "use_llm": False,
        **{k: v for k, v in case_config.items() if k != "mode"},
    }


def _external_config(case_config: dict) -> dict:
    return {
        "mode": "external-enhanced",
        "enable_ncv": True,
        "enable_a2p": True,
        "use_llm": False,
        "enabled_external_providers": ["ragas", "deepeval"],
        **{k: v for k, v in case_config.items() if k not in ("mode", "enabled_external_providers")},
    }


def _load_run(run_fixture: str) -> RAGRun:
    fixture_path = (ROOT / run_fixture).resolve() if not Path(run_fixture).is_absolute() else Path(run_fixture)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if "run_id" in payload and "query" in payload and "final_answer" in payload:
        return RAGRun.model_validate(payload)
    return _build_run_from_payload(payload)


def _build_run_from_payload(payload: dict) -> RAGRun:
    chunks = [RetrievedChunk.model_validate(c) for c in payload.get("retrieved_chunks", [])]
    entries = [
        CorpusEntry.model_validate(e)
        for e in payload.get("corpus_entries", payload.get("corpus_metadata", {}).get("entries", []))
    ]
    metadata = dict(payload.get("metadata", {}))
    if "corpus_metadata" in payload:
        metadata["corpus_metadata"] = payload["corpus_metadata"]
    if payload.get("parser_validation_profile") is not None:
        metadata["parser_validation_profile"] = payload["parser_validation_profile"]
    if "citations" in payload:
        metadata["citations"] = payload["citations"]
    return RAGRun(
        run_id=payload.get("run_id", payload.get("case_id", "eval-case")),
        query=payload["query"],
        retrieved_chunks=chunks,
        final_answer=payload["final_answer"],
        cited_doc_ids=payload.get("cited_doc_ids", []),
        answer_confidence=payload.get("answer_confidence"),
        trace=payload.get("trace"),
        corpus_entries=entries,
        metadata=metadata,
    )


def _run_diagnosis(run: RAGRun, config: dict) -> Any:
    # Re-load run from fixture to ensure clean state (no shared mutable state between modes)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return diagnose(run, config=config)


def _first_failing_node(diagnosis: Any) -> str | None:
    if diagnosis.ncv_report:
        return diagnosis.ncv_report.get("first_failing_node")
    return diagnosis.first_failing_node


def _pinpoint_node(diagnosis: Any) -> str | None:
    if diagnosis.pinpoint_findings:
        finding = diagnosis.pinpoint_findings[0]
        return finding.location.ncv_node if finding.location else None
    return None


def _root_cause(diagnosis: Any) -> str | None:
    if diagnosis.causal_chains:
        return diagnosis.causal_chains[0].causal_hypothesis
    return None


def _a2p_primary_cause(diagnosis: Any) -> str | None:
    for result in diagnosis.analyzer_results:
        if result.analyzer_name == "A2PAttributionAnalyzer":
            for attr in result.claim_attributions or []:
                return attr.primary_cause
            for attr in result.claim_attributions_v2 or []:
                return attr.primary_cause
    return None


def _security_failure_present(diagnosis: Any) -> bool:
    from raggov.models.diagnosis import FailureType
    security_types = {
        FailureType.PROMPT_INJECTION,
        FailureType.SUSPICIOUS_CHUNK,
        FailureType.RETRIEVAL_ANOMALY,
        FailureType.PRIVACY_VIOLATION,
    }
    for result in diagnosis.analyzer_results:
        if result.failure_type in security_types and result.status in {"fail", "warn"}:
            return True
    return False


def _ncv_security_risk_status(diagnosis: Any) -> str | None:
    if not diagnosis.ncv_report:
        return None
    for node in diagnosis.ncv_report.get("node_results", []):
        if node.get("node") == "security_risk":
            return node.get("status")
    return None


def _extract_external_signals(diagnosis: Any) -> list[dict]:
    """Extract all external signals from metadata and retrieval diagnosis report."""
    signals = []
    meta = {}
    for result in diagnosis.analyzer_results:
        if result.analyzer_name == "RetrievalDiagnosisAnalyzerV0" and result.retrieval_diagnosis_report:
            report = result.retrieval_diagnosis_report
            for sig in getattr(report, "evidence_signals", []):
                source = getattr(sig, "source_report", "") or ""
                if "ExternalEvaluationResult" in str(source):
                    signals.append({
                        "metric_name": getattr(sig, "signal_name", ""),
                        "value": getattr(sig, "value", None),
                        "label": None,
                        "source": str(source),
                        "interpretation": getattr(sig, "interpretation", ""),
                        "consumed_by_analyzer": ["RetrievalDiagnosisAnalyzerV0"],
                    })
    return signals


def _ncv_node_results(diagnosis: Any) -> dict[str, str]:
    """Return {node_name: status} for all evaluated NCV nodes."""
    if not diagnosis.ncv_report:
        return {}
    return {
        node.get("node", ""): node.get("status", "")
        for node in diagnosis.ncv_report.get("node_results", [])
    }


def _build_regression_evidence(
    case: Any,
    native_diagnosis: Any,
    external_diagnosis: Any,
    native_node: str | None,
    external_node: str | None,
    native_root: str | None,
    external_root: str | None,
) -> list[str]:
    evidence: list[str] = []
    expected_node = case["expected"].get("first_failing_node")
    expected_root = case["expected"].get("root_cause")

    if native_node == expected_node and external_node != expected_node:
        evidence.append(f"node_regression: native={native_node!r} (correct) → external={external_node!r} (wrong)")

    if native_root == expected_root and external_root != expected_root:
        evidence.append(f"root_cause_regression: native={native_root!r} (correct) → external={external_root!r} (wrong)")

    # Check if security risk overrode correct native node
    if external_node == "security_risk" and native_node != "security_risk" and native_node == expected_node:
        evidence.append("security_risk_override: external mode selected security_risk over correct native node")

    # Detect A2P adversarial_context injection
    ext_a2p = _a2p_primary_cause(external_diagnosis)
    nat_a2p = _a2p_primary_cause(native_diagnosis)
    if ext_a2p == "adversarial_context" and nat_a2p != "adversarial_context":
        evidence.append(f"a2p_adversarial_injection: external A2P produced adversarial_context (native={nat_a2p!r})")

    # Detect generic root cause collapse
    generic = {"adversarial_or_unsafe_context", "retrieval_coverage_gap", "generation_grounding_failure"}
    if external_root in generic and native_root not in generic and native_root == expected_root:
        evidence.append(f"root_cause_collapse_to_generic: external→{external_root!r} displaced specific native→{native_root!r}")

    # Detect ncv priority rule 2 firing
    ext_ncv_security = _ncv_security_risk_status(external_diagnosis)
    nat_ncv_security = _ncv_security_risk_status(native_diagnosis)
    if ext_ncv_security in ("fail", "warn") and nat_ncv_security not in ("fail", "warn"):
        evidence.append(f"security_risk_node_difference: native={nat_ncv_security!r} external={ext_ncv_security!r}")

    return evidence


def _classify_regression(
    changed_node: bool,
    changed_root: bool,
    changed_primary: bool,
    evidence: list[str],
) -> str:
    if not any([changed_node, changed_root, changed_primary]):
        return "none"
    parts = []
    if changed_primary:
        parts.append("primary_regression")
    if changed_node:
        parts.append("node_regression")
    if changed_root:
        parts.append("root_cause_regression")
    return "+".join(parts) if parts else "none"


def _build_trace(diagnosis: Any, mode: str) -> dict:
    ncv = diagnosis.ncv_report or {}
    return {
        "mode": mode,
        "primary_failure": diagnosis.primary_failure.value,
        "first_failing_node": _first_failing_node(diagnosis),
        "root_cause": _root_cause(diagnosis),
        "a2p_primary_cause": _a2p_primary_cause(diagnosis),
        "security_failure_present": _security_failure_present(diagnosis),
        "ncv_security_risk_status": _ncv_security_risk_status(diagnosis),
        "ncv_priority_policy_decision": ncv.get("priority_policy_decision"),
        "ncv_original_first_failing_node": ncv.get("original_first_failing_node"),
        "ncv_node_statuses": _ncv_node_results(diagnosis),
        "analyzer_results": [
            {
                "name": r.analyzer_name,
                "status": r.status,
                "failure_type": r.failure_type.value if r.failure_type else None,
                "stage": r.stage.value if r.stage else None,
            }
            for r in diagnosis.analyzer_results
        ],
        "external_signals_present": len(_extract_external_signals(diagnosis)),
        "degraded_external_mode": diagnosis.degraded_external_mode,
        "missing_external_providers": list(diagnosis.missing_external_providers),
        "external_signals_used": list(diagnosis.external_signals_used),
    }


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------

def evaluate_case(case: dict, *, include_trace: bool = False) -> CaseDiff:
    run_fixture = case["run_fixture"]
    engine_config = case.get("engine_config", {})
    expected = case["expected"]

    # Load fresh run for each mode
    native_run = _load_run(run_fixture)
    external_run = _load_run(run_fixture)

    native_diag = _run_diagnosis(native_run, _native_config(engine_config))
    external_diag = _run_diagnosis(external_run, _external_config(engine_config))

    native_primary = native_diag.primary_failure.value
    external_primary = external_diag.primary_failure.value
    native_node = _first_failing_node(native_diag)
    external_node = _first_failing_node(external_diag)
    native_pinpoint = _pinpoint_node(native_diag)
    external_pinpoint = _pinpoint_node(external_diag)
    native_root = _root_cause(native_diag)
    external_root = _root_cause(external_diag)

    expected_primary = expected.get("primary_failure", "")
    expected_node = expected.get("first_failing_node")
    expected_pinpoint = expected.get("pinpoint_node") or expected.get("first_failing_node")
    expected_root = expected.get("root_cause")

    changed_primary = native_primary != external_primary
    changed_node = native_node != external_node
    changed_pinpoint = native_pinpoint != external_pinpoint
    changed_root = native_root != external_root

    ext_signals = _extract_external_signals(external_diag)
    ext_providers = list({s.get("source", "") for s in ext_signals})
    ext_types = list({s.get("interpretation", "")[:40] for s in ext_signals})
    ext_labels = [s.get("label") for s in ext_signals if s.get("label")]
    ext_metrics = [s.get("metric_name", "") for s in ext_signals]
    ext_values = [s.get("value") for s in ext_signals]
    ext_consumers = list({c for s in ext_signals for c in s.get("consumed_by_analyzer", [])})

    regression_evidence = _build_regression_evidence(
        case, native_diag, external_diag,
        native_node, external_node, native_root, external_root,
    )
    regression_type = _classify_regression(
        changed_node, changed_root, changed_primary, regression_evidence
    )

    which_report_changed: list[str] = []
    if changed_node:
        which_report_changed.append("first_failing_node")
    if changed_root:
        which_report_changed.append("root_cause")
    if changed_primary:
        which_report_changed.append("primary_failure")

    return CaseDiff(
        case_id=case["case_id"],
        description=case.get("description", ""),
        expected_primary_failure=expected_primary,
        expected_first_failing_node=expected_node,
        expected_pinpoint_node=expected_pinpoint,
        expected_root_cause=expected_root,
        native_primary_failure=native_primary,
        external_primary_failure=external_primary,
        native_first_failing_node=native_node,
        external_first_failing_node=external_node,
        native_pinpoint_node=native_pinpoint,
        external_pinpoint_node=external_pinpoint,
        native_root_cause=native_root,
        external_root_cause=external_root,
        changed_primary_failure=changed_primary,
        changed_first_failing_node=changed_node,
        changed_pinpoint_node=changed_pinpoint,
        changed_root_cause=changed_root,
        native_correct_primary=native_primary == expected_primary,
        native_correct_node=native_node == expected_node,
        native_correct_root_cause=native_root == expected_root,
        external_correct_primary=external_primary == expected_primary,
        external_correct_node=external_node == expected_node,
        external_correct_root_cause=external_root == expected_root,
        external_signals_present=len(ext_signals),
        external_signal_provider_names=ext_providers,
        external_signal_types=ext_types,
        external_signal_labels=ext_labels,
        external_signal_metric_names=ext_metrics,
        external_signal_values=ext_values,
        which_analyzer_consumed_external_signal=ext_consumers,
        which_report_changed_after_external_signal=which_report_changed,
        native_a2p_primary_cause=_a2p_primary_cause(native_diag),
        external_a2p_primary_cause=_a2p_primary_cause(external_diag),
        native_security_failure_present=_security_failure_present(native_diag),
        external_security_failure_present=_security_failure_present(external_diag),
        native_ncv_security_risk_status=_ncv_security_risk_status(native_diag),
        external_ncv_security_risk_status=_ncv_security_risk_status(external_diag),
        regression_type=regression_type,
        regression_evidence=regression_evidence,
        native_trace=_build_trace(native_diag, "native") if include_trace else None,
        external_trace=_build_trace(external_diag, "external-enhanced") if include_trace else None,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _confusion_table(
    per_case: list[CaseDiff],
    expected_fn: Any,
    predicted_fn: Any,
) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for case in per_case:
        expected = expected_fn(case)
        predicted = predicted_fn(case)
        if expected is not None:
            table[str(expected)][str(predicted)] += 1
    return {k: dict(v) for k, v in table.items()}


def compute_summary(per_case: list[CaseDiff], gold_status: str) -> RegressionSummary:
    n = len(per_case) or 1

    # Accuracy
    nat_primary_acc = sum(1 for c in per_case if c.native_correct_primary) / n
    ext_primary_acc = sum(1 for c in per_case if c.external_correct_primary) / n
    nat_node_acc = sum(1 for c in per_case if c.native_correct_node) / n
    ext_node_acc = sum(1 for c in per_case if c.external_correct_node) / n
    nat_root_acc = sum(1 for c in per_case if c.native_correct_root_cause) / n
    ext_root_acc = sum(1 for c in per_case if c.external_correct_root_cause) / n

    # Cross-table
    nc_ew_primary = sum(1 for c in per_case if c.native_correct_primary and not c.external_correct_primary)
    nw_ec_primary = sum(1 for c in per_case if not c.native_correct_primary and c.external_correct_primary)
    bw_primary = sum(1 for c in per_case if not c.native_correct_primary and not c.external_correct_primary)
    bc_primary = sum(1 for c in per_case if c.native_correct_primary and c.external_correct_primary)

    nc_ew_node = sum(1 for c in per_case if c.native_correct_node and not c.external_correct_node)
    nw_ec_node = sum(1 for c in per_case if not c.native_correct_node and c.external_correct_node)
    bw_node = sum(1 for c in per_case if not c.native_correct_node and not c.external_correct_node)
    bc_node = sum(1 for c in per_case if c.native_correct_node and c.external_correct_node)

    nc_ew_root = sum(1 for c in per_case if c.native_correct_root_cause and not c.external_correct_root_cause)
    nw_ec_root = sum(1 for c in per_case if not c.native_correct_root_cause and c.external_correct_root_cause)
    bw_root = sum(1 for c in per_case if not c.native_correct_root_cause and not c.external_correct_root_cause)
    bc_root = sum(1 for c in per_case if c.native_correct_root_cause and c.external_correct_root_cause)

    ext_improved = sum(
        1 for c in per_case
        if (not c.native_correct_node and c.external_correct_node)
        or (not c.native_correct_root_cause and c.external_correct_root_cause)
    )
    ext_degraded = sum(
        1 for c in per_case
        if (c.native_correct_node and not c.external_correct_node)
        or (c.native_correct_root_cause and not c.external_correct_root_cause)
    )

    # Confusion tables
    primary_nat = _confusion_table(per_case, lambda c: c.expected_primary_failure, lambda c: c.native_primary_failure)
    primary_ext = _confusion_table(per_case, lambda c: c.expected_primary_failure, lambda c: c.external_primary_failure)
    node_nat = _confusion_table(per_case, lambda c: c.expected_first_failing_node, lambda c: c.native_first_failing_node)
    node_ext = _confusion_table(per_case, lambda c: c.expected_first_failing_node, lambda c: c.external_first_failing_node)
    root_nat = _confusion_table(per_case, lambda c: c.expected_root_cause, lambda c: c.native_root_cause)
    root_ext = _confusion_table(per_case, lambda c: c.expected_root_cause, lambda c: c.external_root_cause)

    # Degraded node transitions
    node_transitions: Counter = Counter()
    root_transitions: Counter = Counter()
    for c in per_case:
        if c.native_correct_node and not c.external_correct_node:
            node_transitions[(c.native_first_failing_node, c.external_first_failing_node)] += 1
        if c.native_correct_root_cause and not c.external_correct_root_cause:
            root_transitions[(c.native_root_cause, c.external_root_cause)] += 1

    top_node_trans = [(n, e, cnt) for (n, e), cnt in node_transitions.most_common(10)]
    top_root_trans = [(n, e, cnt) for (n, e), cnt in root_transitions.most_common(10)]

    # Security over-trigger
    security_over_trigger = sum(
        1 for c in per_case
        if c.expected_first_failing_node not in ("security_risk", None)
        and c.external_first_failing_node == "security_risk"
    )

    # Root cause collapse
    generic_roots = {"adversarial_or_unsafe_context", "retrieval_coverage_gap", "generation_grounding_failure"}
    root_collapse = sum(
        1 for c in per_case
        if c.external_root_cause in generic_roots
        and c.native_root_cause not in generic_roots
        and c.native_correct_root_cause
    )

    generic_root_counter: Counter = Counter(
        c.external_root_cause for c in per_case if c.external_root_cause in generic_roots
    )

    total_signals = sum(c.external_signals_present for c in per_case)
    signals_changed_node = sum(
        1 for c in per_case if c.external_signals_present > 0 and c.changed_first_failing_node
    )
    signals_changed_root = sum(
        1 for c in per_case if c.external_signals_present > 0 and c.changed_root_cause
    )

    return RegressionSummary(
        total_cases=len(per_case),
        native_primary_accuracy=round(nat_primary_acc, 4),
        external_primary_accuracy=round(ext_primary_acc, 4),
        native_node_accuracy=round(nat_node_acc, 4),
        external_node_accuracy=round(ext_node_acc, 4),
        native_root_cause_accuracy=round(nat_root_acc, 4),
        external_root_cause_accuracy=round(ext_root_acc, 4),
        native_correct_external_wrong_primary=nc_ew_primary,
        native_wrong_external_correct_primary=nw_ec_primary,
        both_wrong_primary=bw_primary,
        both_correct_primary=bc_primary,
        native_correct_external_wrong_node=nc_ew_node,
        native_wrong_external_correct_node=nw_ec_node,
        both_wrong_node=bw_node,
        both_correct_node=bc_node,
        native_correct_external_wrong_root_cause=nc_ew_root,
        native_wrong_external_correct_root_cause=nw_ec_root,
        both_wrong_root_cause=bw_root,
        both_correct_root_cause=bc_root,
        external_only_improved_any=ext_improved,
        external_only_degraded_any=ext_degraded,
        primary_confusion_native=primary_nat,
        primary_confusion_external=primary_ext,
        node_confusion_native=node_nat,
        node_confusion_external=node_ext,
        root_cause_confusion_native=root_nat,
        root_cause_confusion_external=root_ext,
        total_external_signals=total_signals,
        signals_that_changed_node=signals_changed_node,
        signals_that_changed_root_cause=signals_changed_root,
        top_degraded_node_transitions=top_node_trans,
        top_degraded_root_cause_transitions=top_root_trans,
        security_risk_over_trigger_count=security_over_trigger,
        root_cause_collapse_count=root_collapse,
        generic_root_causes_in_external=generic_root_counter,
        per_case=per_case,
    )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_markdown(summary: RegressionSummary, gold_label: str) -> str:
    lines = [
        "# External Signal Regression Analysis",
        "",
        f"> Gold set: `{gold_label}` | Cases: `{summary.total_cases}`",
        "",
        "## Accuracy Comparison",
        "",
        "| metric | native | external-enhanced | delta |",
        "| --- | ---: | ---: | ---: |",
        f"| primary_failure_accuracy | {summary.native_primary_accuracy:.4f} | {summary.external_primary_accuracy:.4f} | {summary.external_primary_accuracy - summary.native_primary_accuracy:+.4f} |",
        f"| first_failing_node_accuracy | {summary.native_node_accuracy:.4f} | {summary.external_node_accuracy:.4f} | {summary.external_node_accuracy - summary.native_node_accuracy:+.4f} |",
        f"| root_cause_accuracy | {summary.native_root_cause_accuracy:.4f} | {summary.external_root_cause_accuracy:.4f} | {summary.external_root_cause_accuracy - summary.native_root_cause_accuracy:+.4f} |",
        "",
        "## External Signal Impact Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| external_signals_total | {summary.total_external_signals} |",
        f"| signals_that_changed_node | {summary.signals_that_changed_node} |",
        f"| signals_that_changed_root_cause | {summary.signals_that_changed_root_cause} |",
        f"| security_risk_over_trigger_count | {summary.security_risk_over_trigger_count} |",
        f"| root_cause_collapse_count | {summary.root_cause_collapse_count} |",
        "",
        "## Native-correct, External-wrong (Pure Regressions)",
        "",
        "| dimension | native✓ external✗ | native✗ external✓ | both✗ | both✓ |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| primary_failure | {summary.native_correct_external_wrong_primary} | {summary.native_wrong_external_correct_primary} | {summary.both_wrong_primary} | {summary.both_correct_primary} |",
        f"| first_failing_node | {summary.native_correct_external_wrong_node} | {summary.native_wrong_external_correct_node} | {summary.both_wrong_node} | {summary.both_correct_node} |",
        f"| root_cause | {summary.native_correct_external_wrong_root_cause} | {summary.native_wrong_external_correct_root_cause} | {summary.both_wrong_root_cause} | {summary.both_correct_root_cause} |",
        "",
        "## Degraded Node Transitions (native_correct → external_wrong)",
        "",
        "| native_node | external_node | count |",
        "| --- | --- | ---: |",
    ]
    for n, e, cnt in summary.top_degraded_node_transitions:
        lines.append(f"| {n} | {e} | {cnt} |")
    if not summary.top_degraded_node_transitions:
        lines.append("| (none) | | |")

    lines.extend([
        "",
        "## Degraded Root Cause Transitions (native_correct → external_wrong)",
        "",
        "| native_root_cause | external_root_cause | count |",
        "| --- | --- | ---: |",
    ])
    for n, e, cnt in summary.top_degraded_root_cause_transitions:
        lines.append(f"| {n} | {e} | {cnt} |")
    if not summary.top_degraded_root_cause_transitions:
        lines.append("| (none) | | |")

    lines.extend([
        "",
        "## Generic Root Causes in External Mode",
        "",
        "| root_cause | count |",
        "| --- | ---: |",
    ])
    for rc, cnt in summary.generic_root_causes_in_external.most_common():
        lines.append(f"| {rc} | {cnt} |")

    lines.extend([
        "",
        "## Per-Case Regression Table",
        "",
        "| case_id | expected_node | native_node | external_node | expected_root | native_root | external_root | regression_type |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for c in summary.per_case:
        nat_node_mark = "✓" if c.native_correct_node else "✗"
        ext_node_mark = "✓" if c.external_correct_node else "✗"
        nat_root_mark = "✓" if c.native_correct_root_cause else "✗"
        ext_root_mark = "✓" if c.external_correct_root_cause else "✗"
        lines.append(
            f"| {c.case_id} "
            f"| {c.expected_first_failing_node or '-'} "
            f"| {c.native_first_failing_node or '-'}{nat_node_mark} "
            f"| {c.external_first_failing_node or '-'}{ext_node_mark} "
            f"| {c.expected_root_cause or '-'} "
            f"| {c.native_root_cause or '-'}{nat_root_mark} "
            f"| {c.external_root_cause or '-'}{ext_root_mark} "
            f"| {c.regression_type} |"
        )

    lines.extend(["", "## Regression Evidence (cases with degradation)", ""])
    for c in summary.per_case:
        if c.regression_evidence:
            lines.append(f"### {c.case_id}")
            lines.append("")
            for ev in c.regression_evidence:
                lines.append(f"- {ev}")
            if c.native_ncv_security_risk_status or c.external_ncv_security_risk_status:
                lines.append(f"- ncv_security_risk: native={c.native_ncv_security_risk_status!r} external={c.external_ncv_security_risk_status!r}")
            if c.native_a2p_primary_cause or c.external_a2p_primary_cause:
                lines.append(f"- a2p_cause: native={c.native_a2p_primary_cause!r} external={c.external_a2p_primary_cause!r}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSONL output helpers
# ---------------------------------------------------------------------------

def _case_to_jsonl_line(diff: CaseDiff) -> str:
    d = asdict(diff)
    return json.dumps(d, default=str)


def _signal_trace_lines(per_case: list[CaseDiff]) -> list[str]:
    lines = []
    for case in per_case:
        if case.external_signals_present == 0:
            continue
        trace = ExternalSignalTrace(
            case_id=case.case_id,
            provider="; ".join(case.external_signal_provider_names),
            signal_type="; ".join(case.external_signal_types),
            metric_name="; ".join(case.external_signal_metric_names),
            value=case.external_signal_values,
            label="; ".join(str(l) for l in case.external_signal_labels) or None,
            affected_chunk_ids=[],
            affected_claim_ids=[],
            raw_payload=None,
            consumed_by_analyzer=case.which_analyzer_consumed_external_signal,
            consumed_by_ncv_node=[],
            consumed_by_a2p_rule=["adversarial_context"] if case.external_a2p_primary_cause == "adversarial_context" else [],
            final_influence=(
                "changed_root_cause" if case.changed_root_cause else
                "changed_node" if case.changed_first_failing_node else
                "advisory"
            ),
        )
        lines.append(json.dumps(asdict(trace), default=str))
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(
    gold_path: Path,
    *,
    case_filter: str | None = None,
    include_trace: bool = False,
) -> tuple[RegressionSummary, str]:
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold_label = gold_path.stem
    cases = gold.get("cases", [])

    if case_filter:
        cases = [c for c in cases if c["case_id"] == case_filter]

    per_case: list[CaseDiff] = []
    for case in cases:
        print(f"  evaluating {case['case_id']}...", file=sys.stderr)
        try:
            diff = evaluate_case(case, include_trace=include_trace)
            per_case.append(diff)
        except Exception as exc:
            print(f"  ERROR in {case['case_id']}: {exc}", file=sys.stderr)

    summary = compute_summary(per_case, gold.get("evaluation_status", gold_label))
    return summary, gold_label


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-set", choices=("v1", "v2"), default="v1")
    parser.add_argument("--gold-path", type=Path, default=None, help="Override gold set path")
    parser.add_argument("--case", default=None, help="Evaluate only this case_id")
    parser.add_argument("--format", choices=("json", "markdown", "both"), default="both")
    parser.add_argument("--dump-traces", action="store_true")
    parser.add_argument("--write-reports", action="store_true", help="Write output files to reports/")
    args = parser.parse_args(argv)

    gold_path = args.gold_path or (GOLD_V1 if args.gold_set == "v1" else GOLD_V2)
    if not gold_path.exists():
        print(f"Gold set not found: {gold_path}", file=sys.stderr)
        return 1

    print(f"Running analysis on {gold_path.name}...", file=sys.stderr)
    summary, gold_label = run_analysis(
        gold_path, case_filter=args.case, include_trace=args.dump_traces
    )

    if args.format in ("json", "both"):
        print(json.dumps(asdict(summary), indent=2, default=str))

    if args.format in ("markdown", "both"):
        md = render_markdown(summary, gold_label)
        print(md, file=sys.stderr)

    if args.write_reports:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        diff_path = REPORTS_DIR / "external_signal_regression_diff.jsonl"
        diff_path.write_text(
            "\n".join(_case_to_jsonl_line(c) for c in summary.per_case),
            encoding="utf-8",
        )
        trace_path = REPORTS_DIR / "external_signal_trace.jsonl"
        trace_path.write_text(
            "\n".join(_signal_trace_lines(summary.per_case)),
            encoding="utf-8",
        )
        md_path = REPORTS_DIR / "external_signal_regression_analysis.md"
        md_path.write_text(render_markdown(summary, gold_label), encoding="utf-8")
        print(f"Reports written to {REPORTS_DIR}/", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
