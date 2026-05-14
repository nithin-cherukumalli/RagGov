"""Offline evaluation harness for NCV pinpointing and A2P causal chains.

This harness measures structured diagnosis outputs against a tiny gold set.
It is deterministic, uncalibrated, and non-gating by design.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, ConfigDict, Field

from raggov.engine import diagnose
from raggov.models.chunk import RetrievedChunk
from raggov.models.corpus import CorpusEntry
from raggov.models.run import RAGRun


DEFAULT_GOLD_SET = ROOT / "tests" / "fixtures" / "govrag_pinpoint_eval" / "pinpoint_eval_gold_v1.json"
EVALUATION_STATUS = "pinpoint_eval_gold_v1_small_unvalidated"


class PinpointEvalExpected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_failure: str
    first_failing_node: str | None = None
    pinpoint_node: str | None = None
    root_cause: str | None = None
    fix_category: str | None = None
    secondary_nodes: list[str] = Field(default_factory=list)
    affected_claim_ids: list[str] = Field(default_factory=list)
    affected_chunk_ids: list[str] = Field(default_factory=list)
    affected_doc_ids: list[str] = Field(default_factory=list)
    human_review_required: bool = True
    recommended_for_gating: bool = False
    calibration_status: str = "uncalibrated"


class PinpointEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    description: str
    run_fixture: str
    engine_config: dict[str, Any] = Field(default_factory=dict)
    expected: PinpointEvalExpected


class PinpointEvalGoldSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_status: str
    cases: list[PinpointEvalCase] = Field(default_factory=list)


@dataclass(frozen=True)
class PinpointCaseResult:
    case_id: str
    primary_failure_expected: str
    primary_failure_observed: str
    primary_failure_pass: bool
    first_failing_node_expected: str | None
    first_failing_node_observed: str | None
    first_failing_node_pass: bool
    pinpoint_node_observed: str | None
    pinpoint_node_pass: bool
    root_cause_expected: str | None
    root_cause_observed: str | None
    root_cause_pass: bool
    fix_category_expected: str | None
    fix_category_observed: str
    fix_category_pass: bool
    affected_claim_ids_expected: list[str] = field(default_factory=list)
    affected_claim_ids_observed: list[str] = field(default_factory=list)
    affected_chunk_ids_expected: list[str] = field(default_factory=list)
    affected_chunk_ids_observed: list[str] = field(default_factory=list)
    affected_doc_ids_expected: list[str] = field(default_factory=list)
    affected_doc_ids_observed: list[str] = field(default_factory=list)
    is_clean: bool = False
    is_incomplete: bool = False
    false_clean: bool = False
    false_incomplete: bool = False
    recommended_for_gating_true: bool = False
    calibrated_confidence_present: bool = False
    calibration_violation: bool = False
    production_gating_violation: bool = False
    trust_decision: str | None = None
    mismatch_class: str | None = None
    suspected_fix_area: str | None = None
    trace: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PinpointEvalSummary:
    evaluation_status: str
    total_cases: int
    primary_failure_accuracy: float
    first_failing_node_accuracy: float
    pinpoint_node_accuracy: float
    root_cause_accuracy: float
    fix_category_accuracy: float | None
    false_clean_count: int
    false_incomplete_count: int
    recommended_for_gating_true_count: int
    calibrated_confidence_present_count: int
    non_uncalibrated_count: int
    uncalibrated_count: int
    production_gating_decision_count: int
    per_case: list[PinpointCaseResult]


def load_gold_set(path: Path) -> PinpointEvalGoldSet:
    return PinpointEvalGoldSet.model_validate_json(path.read_text(encoding="utf-8"))


def evaluate_pinpointing(
    gold_set_path: Path = DEFAULT_GOLD_SET,
    *,
    dump_traces: bool = False,
    trace_case: str | None = None,
) -> PinpointEvalSummary:
    gold = load_gold_set(gold_set_path)
    selected_cases = [
        case
        for case in gold.cases
        if trace_case is None or case.case_id == trace_case
    ]
    per_case = [
        _evaluate_case(
            case,
            include_trace=dump_traces or trace_case is not None,
        )
        for case in selected_cases
    ]
    total = len(per_case) or 1
    fix_cases = [case for case in per_case if case.fix_category_expected is not None]
    non_uncalibrated_count = sum(1 for case in per_case if case.calibration_violation)

    return PinpointEvalSummary(
        evaluation_status=gold.evaluation_status,
        total_cases=len(per_case),
        primary_failure_accuracy=sum(1 for case in per_case if case.primary_failure_pass) / total,
        first_failing_node_accuracy=sum(1 for case in per_case if case.first_failing_node_pass) / total,
        pinpoint_node_accuracy=sum(1 for case in per_case if case.pinpoint_node_pass) / total,
        root_cause_accuracy=sum(1 for case in per_case if case.root_cause_pass) / total,
        fix_category_accuracy=(
            sum(1 for case in fix_cases if case.fix_category_pass) / len(fix_cases)
            if fix_cases
            else None
        ),
        false_clean_count=sum(1 for case in per_case if case.false_clean),
        false_incomplete_count=sum(1 for case in per_case if case.false_incomplete),
        recommended_for_gating_true_count=sum(1 for case in per_case if case.recommended_for_gating_true),
        calibrated_confidence_present_count=sum(1 for case in per_case if case.calibrated_confidence_present),
        non_uncalibrated_count=non_uncalibrated_count,
        uncalibrated_count=len(per_case) - non_uncalibrated_count,
        production_gating_decision_count=sum(1 for case in per_case if case.production_gating_violation),
        per_case=per_case,
    )


def render_markdown(summary: PinpointEvalSummary) -> str:
    fix_accuracy = "n/a" if summary.fix_category_accuracy is None else f"{summary.fix_category_accuracy:.2f}"
    lines = [
        "# Pinpointing Evaluation Harness",
        "",
        f"- evaluation_status: `{summary.evaluation_status}`",
        f"- total_cases: `{summary.total_cases}`",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| primary_failure_accuracy | {summary.primary_failure_accuracy:.2f} |",
        f"| first_failing_node_accuracy | {summary.first_failing_node_accuracy:.2f} |",
        f"| pinpoint_node_accuracy | {summary.pinpoint_node_accuracy:.2f} |",
        f"| root_cause_accuracy | {summary.root_cause_accuracy:.2f} |",
        f"| fix_category_accuracy | {fix_accuracy} |",
        f"| false_clean_count | {summary.false_clean_count} |",
        f"| false_incomplete_count | {summary.false_incomplete_count} |",
        f"| recommended_for_gating_true_count | {summary.recommended_for_gating_true_count} |",
        f"| calibrated_confidence_present_count | {summary.calibrated_confidence_present_count} |",
        f"| non_uncalibrated_count | {summary.non_uncalibrated_count} |",
        f"| production_gating_decision_count | {summary.production_gating_decision_count} |",
        "",
        "## Per-case",
        "",
        "| case_id | primary | node | root cause | fix category |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in summary.per_case:
        lines.append(
            f"| {case.case_id} | {case.primary_failure_observed} | "
            f"{case.first_failing_node_observed or '-'} | "
            f"{case.root_cause_observed or '-'} | {case.fix_category_observed} |"
        )
    traced_cases = [case for case in summary.per_case if case.trace is not None]
    if traced_cases:
        lines.extend(["", "## Traces", ""])
        for case in traced_cases:
            lines.append(f"### {case.case_id}")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(case.trace, indent=2, sort_keys=True))
            lines.append("```")
            lines.append("")
    return "\n".join(lines)


def determine_exit_code(
    summary: PinpointEvalSummary,
    *,
    fail_under_node_accuracy: float | None = None,
    fail_on_gating_violations: bool = False,
) -> int:
    if (
        fail_under_node_accuracy is not None
        and summary.first_failing_node_accuracy < fail_under_node_accuracy
    ):
        return 1
    if fail_on_gating_violations and (
        summary.recommended_for_gating_true_count > 0
        or summary.calibrated_confidence_present_count > 0
        or summary.non_uncalibrated_count > 0
        or summary.production_gating_decision_count > 0
    ):
        return 1
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-set", type=Path, default=DEFAULT_GOLD_SET)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--fail-under-node-accuracy", type=float, default=None)
    parser.add_argument("--fail-on-gating-violations", action="store_true")
    parser.add_argument("--dump-traces", action="store_true")
    parser.add_argument("--trace-case", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        summary = evaluate_pinpointing(
            args.gold_set,
            dump_traces=args.dump_traces,
            trace_case=args.trace_case,
        )
    if args.format == "markdown":
        print(render_markdown(summary))
    else:
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return determine_exit_code(
        summary,
        fail_under_node_accuracy=args.fail_under_node_accuracy,
        fail_on_gating_violations=args.fail_on_gating_violations,
    )


def _evaluate_case(case: PinpointEvalCase, *, include_trace: bool = False) -> PinpointCaseResult:
    run = _load_run(case.run_fixture)
    diagnosis = diagnose(run, config=_engine_config(case.engine_config))

    first_finding = diagnosis.pinpoint_findings[0] if diagnosis.pinpoint_findings else None
    first_chain = diagnosis.causal_chains[0] if diagnosis.causal_chains else None
    actual_primary = diagnosis.primary_failure.value
    actual_node = diagnosis.first_failing_node
    actual_pinpoint_node = first_finding.location.ncv_node if first_finding is not None else None
    expected_pinpoint_node = case.expected.pinpoint_node if case.expected.pinpoint_node is not None else case.expected.first_failing_node
    actual_root_cause = first_chain.causal_hypothesis if first_chain is not None else None
    actual_fix_category = _fix_category(diagnosis.proposed_fix or diagnosis.recommended_fix)
    actual_claim_ids = list(first_finding.location.claim_ids) if first_finding is not None else []
    actual_chunk_ids = list(first_finding.location.chunk_ids) if first_finding is not None else []
    actual_doc_ids = list(first_finding.location.doc_ids) if first_finding is not None else []
    recommended_for_gating_true = _has_recommended_for_gating(diagnosis)
    calibrated_confidence_present = _has_calibrated_confidence(diagnosis)
    calibration_violation = _has_calibration_violation(diagnosis)
    production_gating_violation = _has_production_gating_violation(diagnosis)

    notes: list[str] = []
    if actual_primary != case.expected.primary_failure:
        notes.append(
            f"primary mismatch: expected {case.expected.primary_failure}, observed {actual_primary}"
        )
    if actual_node != case.expected.first_failing_node:
        notes.append(
            f"node mismatch: expected {case.expected.first_failing_node}, observed {actual_node}"
        )
    if actual_root_cause != case.expected.root_cause:
        notes.append(
            f"root cause mismatch: expected {case.expected.root_cause}, observed {actual_root_cause}"
        )

    mismatch_class, suspected_fix_area = _classify_mismatch(
        case=case,
        diagnosis=diagnosis,
        actual_node=actual_node,
        actual_root_cause=actual_root_cause,
    )

    return PinpointCaseResult(
        case_id=case.case_id,
        primary_failure_expected=case.expected.primary_failure,
        primary_failure_observed=actual_primary,
        primary_failure_pass=actual_primary == case.expected.primary_failure,
        first_failing_node_expected=case.expected.first_failing_node,
        first_failing_node_observed=actual_node,
        first_failing_node_pass=actual_node == case.expected.first_failing_node,
        pinpoint_node_observed=actual_pinpoint_node,
        pinpoint_node_pass=actual_pinpoint_node == expected_pinpoint_node,
        root_cause_expected=case.expected.root_cause,
        root_cause_observed=actual_root_cause,
        root_cause_pass=actual_root_cause == case.expected.root_cause,
        fix_category_expected=case.expected.fix_category,
        fix_category_observed=actual_fix_category,
        fix_category_pass=(
            actual_fix_category == case.expected.fix_category
            if case.expected.fix_category is not None
            else True
        ),
        affected_claim_ids_expected=list(case.expected.affected_claim_ids),
        affected_claim_ids_observed=actual_claim_ids,
        affected_chunk_ids_expected=list(case.expected.affected_chunk_ids),
        affected_chunk_ids_observed=actual_chunk_ids,
        affected_doc_ids_expected=list(case.expected.affected_doc_ids),
        affected_doc_ids_observed=actual_doc_ids,
        is_clean=actual_primary == "CLEAN",
        is_incomplete=actual_primary == "INCOMPLETE_DIAGNOSIS",
        false_clean=case.expected.primary_failure != "CLEAN" and actual_primary == "CLEAN",
        false_incomplete=(
            case.expected.primary_failure != "INCOMPLETE_DIAGNOSIS"
            and actual_primary == "INCOMPLETE_DIAGNOSIS"
        ),
        recommended_for_gating_true=recommended_for_gating_true,
        calibrated_confidence_present=calibrated_confidence_present,
        calibration_violation=calibration_violation,
        production_gating_violation=production_gating_violation,
        trust_decision=diagnosis.trust_decision.decision if diagnosis.trust_decision is not None else None,
        mismatch_class=mismatch_class,
        suspected_fix_area=suspected_fix_area,
        trace=(
            _build_trace(case, diagnosis, actual_node, actual_pinpoint_node, actual_root_cause, actual_fix_category, mismatch_class, suspected_fix_area)
            if include_trace
            else None
        ),
        notes=notes,
    )


def _engine_config(case_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "native",
        "enable_ncv": True,
        "enable_a2p": True,
        "use_llm": False,
        **case_config,
    }


def _load_run(run_fixture: str) -> RAGRun:
    fixture_path = (ROOT / run_fixture).resolve() if not Path(run_fixture).is_absolute() else Path(run_fixture)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if "run_id" in payload and "query" in payload and "final_answer" in payload:
        return RAGRun.model_validate(payload)
    return _build_run_from_case_payload(payload)


def _build_run_from_case_payload(payload: dict[str, Any]) -> RAGRun:
    chunks = [RetrievedChunk.model_validate(chunk) for chunk in payload.get("retrieved_chunks", [])]
    corpus_entries = [
        CorpusEntry.model_validate(entry)
        for entry in payload.get("corpus_entries", payload.get("corpus_metadata", {}).get("entries", []))
    ]
    metadata = dict(payload.get("metadata", {}))
    if "corpus_metadata" in payload:
        metadata["corpus_metadata"] = payload["corpus_metadata"]
    if payload.get("parser_validation_profile") is not None:
        metadata["parser_validation_profile"] = payload["parser_validation_profile"]
    if "citations" in payload:
        metadata["citations"] = payload["citations"]

    return RAGRun(
        run_id=payload.get("run_id", payload.get("case_id", "pinpoint-eval-case")),
        query=payload["query"],
        retrieved_chunks=chunks,
        final_answer=payload["final_answer"],
        cited_doc_ids=payload.get("cited_doc_ids", []),
        answer_confidence=payload.get("answer_confidence"),
        trace=payload.get("trace"),
        corpus_entries=corpus_entries,
        metadata=metadata,
    )


def _fix_category(text: str | None) -> str:
    if not text:
        return "other"
    normalized = text.lower()
    if any(token in normalized for token in ("retrieval", "recall", "query rewriting", "metadata filter")):
        return "retrieval_recall"
    if any(token in normalized for token in ("rerank", "precision", "filtering", "query routing")):
        return "retrieval_precision"
    if any(token in normalized for token in ("citation", "provenance", "claim-citation")):
        return "citation_alignment"
    if any(token in normalized for token in ("effective-date", "supersession", "stale", "fresh source")):
        return "freshness"
    if any(token in normalized for token in ("security", "prompt injection", "quarantine", "unsafe")):
        return "security"
    if any(token in normalized for token in ("parser", "chunking", "provenance profile")):
        return "parsing"
    if any(token in normalized for token in ("grounding", "abstention", "claim verification", "generation")):
        return "generation_grounding"
    if "completeness" in normalized:
        return "answer_completeness"
    if any(token in normalized for token in ("intent", "routing logic")):
        return "query_understanding"
    return "other"


def _has_recommended_for_gating(diagnosis: Any) -> bool:
    if diagnosis.trust_decision is not None and diagnosis.trust_decision.recommended_for_gating:
        return True
    for finding in diagnosis.pinpoint_findings:
        if finding.recommended_for_gating or finding.location.recommended_for_gating:
            return True
    return False


def _has_calibrated_confidence(diagnosis: Any) -> bool:
    for finding in diagnosis.pinpoint_findings:
        if finding.calibrated_confidence is not None:
            return True
    for chain in diagnosis.causal_chains:
        if chain.calibrated_confidence is not None:
            return True
    return False


def _has_calibration_violation(diagnosis: Any) -> bool:
    statuses = []
    if diagnosis.trust_decision is not None:
        statuses.append(diagnosis.trust_decision.calibration_status)
    for finding in diagnosis.pinpoint_findings:
        statuses.extend([finding.calibration_status, finding.location.calibration_status])
    for chain in diagnosis.causal_chains:
        statuses.append(chain.calibration_status)
    return any(status != "uncalibrated" for status in statuses if status is not None)


def _has_production_gating_violation(diagnosis: Any) -> bool:
    if diagnosis.trust_decision is None:
        return False
    return diagnosis.trust_decision.blocking_eligible or diagnosis.trust_decision.decision == "block"


def _classify_mismatch(
    *,
    case: PinpointEvalCase,
    diagnosis: Any,
    actual_node: str | None,
    actual_root_cause: str | None,
) -> tuple[str | None, str | None]:
    if actual_node == case.expected.first_failing_node and actual_root_cause == case.expected.root_cause:
        return None, None
    if actual_node is None and diagnosis.ncv_report is not None:
        return "ENGINE_PLUMBING_MISSING", "engine plumbing"
    if actual_node != case.expected.first_failing_node and actual_root_cause is None:
        return "CAUSAL_CHAIN_MISSING", "A2P causal chain"
    if diagnosis.first_failing_node is not None and diagnosis.pinpoint_findings:
        return "NCV_MAPPING_WRONG", "NCV node check"
    return "EVIDENCE_WRONG", "retrieval diagnosis"


def _build_trace(
    case: PinpointEvalCase,
    diagnosis: Any,
    actual_node: str | None,
    actual_pinpoint_node: str | None,
    actual_root_cause: str | None,
    actual_fix_category: str,
    mismatch_class: str | None,
    suspected_fix_area: str | None,
) -> dict[str, Any]:
    ncv_report = diagnosis.ncv_report or {}
    node_traces = []
    for node in ncv_report.get("node_results", []):
        node_traces.append(
            {
                "node": node.get("node"),
                "status": node.get("status"),
                "primary_reason": node.get("primary_reason"),
                "evidence_signals": [
                    {
                        "signal_name": sig.get("signal_name"),
                        "value": sig.get("value"),
                        "label": sig.get("label"),
                    }
                    for sig in node.get("evidence_signals", [])
                ],
            }
        )
    analyzer_traces = [
        {
            "analyzer_name": result.analyzer_name,
            "status": result.status,
            "failure_type": result.failure_type.value if result.failure_type is not None else None,
            "stage": result.stage.value if result.stage is not None else None,
            "top_evidence": list(result.evidence[:3]),
        }
        for result in diagnosis.analyzer_results
    ]
    first_finding = diagnosis.pinpoint_findings[0] if diagnosis.pinpoint_findings else None
    first_chain = diagnosis.causal_chains[0] if diagnosis.causal_chains else None
    return {
        "case_id": case.case_id,
        "expected": {
            "primary_failure": case.expected.primary_failure,
            "first_failing_node": case.expected.first_failing_node,
            "root_cause": case.expected.root_cause,
            "fix_category": case.expected.fix_category,
        },
        "actual": {
            "primary_failure": diagnosis.primary_failure.value,
            "first_failing_node": actual_node,
            "pinpoint_node": actual_pinpoint_node,
            "root_cause": actual_root_cause,
            "fix_category": actual_fix_category,
        },
        "analyzer_results": analyzer_traces,
        "ncv": {
            "original_first_failing_node": ncv_report.get("original_first_failing_node"),
            "priority_policy_decision": ncv_report.get("priority_policy_decision"),
            "final_first_failing_node": ncv_report.get("first_failing_node"),
            "node_results": node_traces,
        },
        "pinpoint": {
            "count": len(diagnosis.pinpoint_findings),
            "first_location": (
                {
                    "ncv_node": first_finding.location.ncv_node,
                    "pipeline_stage": first_finding.location.pipeline_stage,
                }
                if first_finding is not None
                else None
            ),
            "evidence_for": (
                [e.signal_name for e in first_finding.evidence_for]
                if first_finding is not None
                else []
            ),
            "missing_evidence": (
                list(first_finding.missing_evidence)
                if first_finding is not None
                else []
            ),
        },
        "a2p": {
            "status": next((r.status for r in diagnosis.analyzer_results if r.analyzer_name == "A2PAttributionAnalyzer"), None),
            "failure_type": next((r.failure_type.value for r in diagnosis.analyzer_results if r.analyzer_name == "A2PAttributionAnalyzer" and r.failure_type is not None), None),
            "stage": next((r.stage.value for r in diagnosis.analyzer_results if r.analyzer_name == "A2PAttributionAnalyzer" and r.stage is not None), None),
            "has_pinpoint_context": any(
                e.startswith("a2p_pinpoint_context:")
                for r in diagnosis.analyzer_results
                if r.analyzer_name == "A2PAttributionAnalyzer"
                for e in r.evidence
            ),
            "has_causal_chain": any(
                e.startswith("a2p_causal_chain:")
                for r in diagnosis.analyzer_results
                if r.analyzer_name == "A2PAttributionAnalyzer"
                for e in r.evidence
            ),
            "causal_chains_count": len(diagnosis.causal_chains),
            "first_causal_hypothesis": first_chain.causal_hypothesis if first_chain is not None else None,
        },
        "mismatch_class": mismatch_class,
        "suspected_fix_area": suspected_fix_area,
    }


if __name__ == "__main__":
    raise SystemExit(main())
