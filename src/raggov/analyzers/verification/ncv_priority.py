"""Deterministic NCV Priority Policy v1.

Selects the most semantically correct first-failing NCV node from an already-built
NCVReport by applying explicit priority rules.

This policy is:
- Deterministic and uncalibrated.
- NOT recommended for production gating.
- A practical heuristic layer that corrects known node-ordering artefacts.

Known corrections:
- citation_support beats context_assembly when citation failure evidence is explicit.
- retrieval_coverage beats retrieval_precision when retrieval_miss evidence is explicit.
- version_validity beats citation_support when stale/superseded cited docs are involved.
- security_risk overrides downstream symptom nodes when malicious content is detected.
- parser_validity overrides all when blocking structural damage is confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from raggov.models.ncv import NCVCalibrationStatus, NCVNode, NCVNodeResult, NCVNodeStatus, NCVReport


# ---------------------------------------------------------------------------
# Decision model
# ---------------------------------------------------------------------------

@dataclass
class NCVPriorityDecision:
    """Result of applying the NCV priority policy to an NCVReport."""

    selected_node: str | None
    original_first_failing_node: str | None
    changed: bool
    reason: str
    evidence_for_selection: list[str] = field(default_factory=list)
    alternative_nodes_considered: list[str] = field(default_factory=list)
    method_type: str = "deterministic_priority_policy"
    calibration_status: str = "uncalibrated"
    recommended_for_gating: bool = False
    limitations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_first_failing_node_v1(report: NCVReport | dict) -> NCVPriorityDecision:
    """Return the most semantically correct first-failing node for an NCVReport.

    Applies ten explicit priority rules in order.  Rules 1-3 (parser, security,
    version) are absolute overrides; rules 4-5 (citation vs context-assembly,
    coverage vs precision) are evidence-conditional; remaining nodes fall through
    to the original pipeline-order selection.

    Args:
        report: An NCVReport instance or a plain dict (e.g. from JSON).

    Returns:
        NCVPriorityDecision.  If ``changed`` is False, ``selected_node`` equals
        the report's existing ``first_failing_node``.
    """
    report = _coerce_report(report)
    if report is None:
        return NCVPriorityDecision(
            selected_node=None,
            original_first_failing_node=None,
            changed=False,
            reason="NCVReport could not be parsed.",
        )

    fail_map: dict[NCVNode, NCVNodeResult] = {
        r.node: r for r in report.node_results if r.status == NCVNodeStatus.FAIL
    }

    original = report.first_failing_node
    original_str = original.value if original is not None else None

    if not fail_map:
        return NCVPriorityDecision(
            selected_node=None,
            original_first_failing_node=original_str,
            changed=False,
            reason="No failing nodes in report.",
        )

    # Rule 1 — parser_validity: blocking structural/provenance damage
    if NCVNode.PARSER_VALIDITY in fail_map and _is_parser_blocking(fail_map[NCVNode.PARSER_VALIDITY]):
        return _decision(
            NCVNode.PARSER_VALIDITY,
            original_str,
            fail_map,
            "parser_validity has blocking structural/provenance damage (Rule 1)",
            ["parser_validity_blocking_damage"],
        )

    # Rule 2 — security_risk: malicious or adversarial context
    if NCVNode.SECURITY_RISK in fail_map and _is_explicit_security_failure(fail_map[NCVNode.SECURITY_RISK]):
        return _decision(
            NCVNode.SECURITY_RISK,
            original_str,
            fail_map,
            "security_risk has malicious or adversarial content (Rule 2)",
            ["security_risk_fail"],
        )

    # Rule 3 — version_validity: stale/superseded cited sources beat citation failures
    if NCVNode.VERSION_VALIDITY in fail_map and _is_version_stale(fail_map[NCVNode.VERSION_VALIDITY]):
        return _decision(
            NCVNode.VERSION_VALIDITY,
            original_str,
            fail_map,
            "version_validity has stale/superseded/withdrawn cited source evidence (Rule 3)",
            ["version_validity_stale_cited"],
        )

    # Rule 4 — citation_support beats downstream retrieval/context symptom nodes
    if (
        NCVNode.CITATION_SUPPORT in fail_map
        and _has_citation_failure(fail_map[NCVNode.CITATION_SUPPORT])
        and original in {NCVNode.CITATION_SUPPORT, NCVNode.CONTEXT_ASSEMBLY, NCVNode.RETRIEVAL_PRECISION}
    ):
        return _decision(
            NCVNode.CITATION_SUPPORT,
            original_str,
            fail_map,
            "citation_support has explicit citation failure evidence, overrides downstream retrieval/context symptoms (Rule 4)",
            ["citation_support_explicit_failure"],
        )

    # Rule 5 — retrieval_coverage beats retrieval_precision when retrieval_miss evidence is explicit
    if NCVNode.RETRIEVAL_COVERAGE in fail_map and _has_retrieval_miss(fail_map[NCVNode.RETRIEVAL_COVERAGE]):
        return _decision(
            NCVNode.RETRIEVAL_COVERAGE,
            original_str,
            fail_map,
            "retrieval_coverage has explicit retrieval_miss/insufficient_context evidence (Rule 5)",
            ["retrieval_coverage_retrieval_miss"],
        )

    # Rules 6-10 — fall through to original pipeline-order selection
    return NCVPriorityDecision(
        selected_node=original_str,
        original_first_failing_node=original_str,
        changed=False,
        reason=f"No priority rule overrode '{original_str}'; pipeline-order selection preserved.",
        alternative_nodes_considered=[n.value for n in fail_map if n.value != original_str],
        limitations=["Priority policy v1 has no rule for this failure combination."],
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STALE_KEYWORDS = frozenset({
    "invalid cited", "expired", "superseded", "withdrawn",
    "deprecated", "stale", "not yet effective",
})
_CITATION_FAIL_SIGNALS = frozenset({"citation_faithfulness_issues", "citation_probe_status"})
_CITATION_FAIL_REASONS = ("citation faithfulness", "citation support", "phantom", "unsupported citation", "missing citation")
_SECURITY_FAIL_SIGNALS = frozenset({"security_result_status"})
_SECURITY_FAIL_REASONS = (
    "prompt injection",
    "suspicious content",
    "poisoning",
    "privacy",
    "unsafe",
    "adversarial",
)


def _is_parser_blocking(node: NCVNodeResult) -> bool:
    if "blocking" in node.primary_reason.lower():
        return True
    return any(sig.signal_name == "parser_validation_finding" for sig in node.evidence_signals)


def _is_version_stale(node: NCVNodeResult) -> bool:
    reason = node.primary_reason.lower()
    if any(k in reason for k in _STALE_KEYWORDS):
        return True
    return any(sig.signal_name == "invalid_cited_doc_ids" for sig in node.evidence_signals)


def _has_citation_failure(node: NCVNodeResult) -> bool:
    reason = node.primary_reason.lower()
    if any(k in reason for k in _CITATION_FAIL_REASONS):
        return True
    for sig in node.evidence_signals:
        if sig.signal_name in _CITATION_FAIL_SIGNALS:
            return True
    return False


def _has_retrieval_miss(node: NCVNodeResult) -> bool:
    reason = node.primary_reason.lower()
    if "retrieval miss" in reason or "retrieval_miss" in reason:
        return True
    for sig in node.evidence_signals:
        if sig.signal_name == "retrieval_primary_failure":
            val = str(sig.value).lower() if sig.value is not None else ""
            if "retrieval_miss" in val:
                return True
        if sig.signal_name == "sufficiency_label":
            val = str(sig.value).lower() if sig.value is not None else ""
            if "insufficient" in val:
                return True
    return False


def _is_explicit_security_failure(node: NCVNodeResult) -> bool:
    reason = node.primary_reason.lower()
    if any(token in reason for token in _SECURITY_FAIL_REASONS):
        return True
    for sig in node.evidence_signals:
        if sig.signal_name not in _SECURITY_FAIL_SIGNALS:
            continue
        value = str(sig.value).lower() if sig.value is not None else ""
        source = str(sig.source_report).lower()
        if value == "fail":
            return True
        if any(token in source for token in ("promptinjection", "poison", "privacy")):
            return True
    return False


def _decision(
    selected: NCVNode,
    original_str: str | None,
    fail_map: dict[NCVNode, NCVNodeResult],
    reason: str,
    evidence_for_selection: list[str],
) -> NCVPriorityDecision:
    selected_str = selected.value
    changed = selected_str != original_str
    alternatives = [n.value for n in fail_map if n != selected]
    return NCVPriorityDecision(
        selected_node=selected_str,
        original_first_failing_node=original_str,
        changed=changed,
        reason=reason,
        evidence_for_selection=evidence_for_selection,
        alternative_nodes_considered=alternatives,
        limitations=["Heuristic priority policy v1; uncalibrated."],
    )


def _coerce_report(report: NCVReport | dict) -> NCVReport | None:
    if isinstance(report, NCVReport):
        return report
    try:
        return NCVReport.model_validate(report)
    except Exception:
        return None
