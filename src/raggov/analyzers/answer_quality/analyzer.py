"""Generation-stage answer-quality ownership signals."""

from __future__ import annotations

import re
from dataclasses import dataclass

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.findings import AnalyzerFinding, AnalyzerReport
from raggov.models.run import RAGRun
from raggov.models.signals import EvidenceSignalMetadata


@dataclass(frozen=True)
class _QualitySignal:
    signal_name: str
    method: str
    failure_type: FailureType
    stage: FailureStage
    status: str
    severity: str
    evidence_message: str
    evidence_ids: list[str]
    evidence_strength: str
    evidence_tier: str


class AnswerQualityAnalyzer(BaseAnalyzer):
    """Emit explicit generation-stage metadata for answer-quality failures."""

    weight = 0.95

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        signals: list[_QualitySignal] = []
        signals.extend(self._answer_completeness_signals(run))
        signals.extend(self._context_adherence_signals(run))
        signals.extend(self._overconfidence_signals(run))

        if not signals:
            return self._pass_with_report("No generation-stage answer-quality issue detected.")

        metadata = [self._metadata(signal) for signal in signals]
        findings = [
            self._finding(index=index, signal=signal, metadata=signal_metadata)
            for index, (signal, signal_metadata) in enumerate(zip(signals, metadata), start=1)
        ]
        report = AnalyzerReport(
            analyzer_name=self.name(),
            overall_status="fail" if any(s.status == "fail" for s in signals) else "warn",
            findings=findings,
            notes=["Answer-quality signals are uncalibrated and not production-gating eligible."],
        )

        primary_signal = next((signal for signal in signals if signal.status == "fail"), signals[0])
        return AnalyzerResult(
            analyzer_name=self.name(),
            status=primary_signal.status,  # type: ignore[arg-type]
            failure_type=primary_signal.failure_type,
            stage=primary_signal.stage,
            evidence=[signal.evidence_message for signal in signals],
            signal_metadata=metadata,
            analyzer_report=report,
            remediation=self._remediation(primary_signal),
        )

    def _answer_completeness_signals(self, run: RAGRun) -> list[_QualitySignal]:
        expected_items = self._expected_numbered_items(run.query)
        if not expected_items:
            return []

        context_text = " ".join(chunk.text for chunk in run.retrieved_chunks)
        context_items = self._numbered_items(context_text)
        answer_items = self._numbered_items(run.final_answer)
        if not expected_items.issubset(context_items):
            return []

        missing = sorted(expected_items - answer_items)
        if not missing:
            return []

        return [
            _QualitySignal(
                signal_name="answer_incomplete_despite_available_context",
                method="answer_completeness_check",
                failure_type=FailureType.UNSUPPORTED_CLAIM,
                stage=FailureStage.GENERATION,
                status="fail",
                severity="high",
                evidence_message=(
                    "Answer completeness check: retrieved context contains requested "
                    f"items {sorted(expected_items)}, but answer omits {missing}."
                ),
                evidence_ids=[chunk.chunk_id for chunk in run.retrieved_chunks],
                evidence_strength="strong",
                evidence_tier="structured",
            )
        ]

    def _context_adherence_signals(self, run: RAGRun) -> list[_QualitySignal]:
        grounding = self._prior_result("ClaimGroundingAnalyzer")
        if (
            grounding is None
            or grounding.status != "fail"
            or grounding.failure_type not in (FailureType.CONTRADICTED_CLAIM, FailureType.UNSUPPORTED_CLAIM)
        ):
            return []
            
        if not run.retrieved_chunks:
            return []
            
        sufficiency = self._prior_result("SufficiencyAnalyzer")
        if sufficiency and sufficiency.sufficiency_result and grounding.failure_type != FailureType.CONTRADICTED_CLAIM:
            if (
                not sufficiency.sufficiency_result.sufficient
                or sufficiency.sufficiency_result.sufficiency_label == "insufficient"
            ):
                return []

        evidence = (
            "Context adherence check: contradictory evidence exists; grounding found the answer contradicted "
            "retrieved context, so generation likely ignored available evidence."
            if grounding.failure_type == FailureType.CONTRADICTED_CLAIM
            else "Context adherence check: context is sufficient but answer contains unsupported claims, so generation likely hallucinated."
        )

        return [
            _QualitySignal(
                signal_name="answer_ignored_retrieved_context",
                method="context_adherence_check",
                failure_type=grounding.failure_type,
                stage=FailureStage.GENERATION,
                status="fail",
                severity="high",
                evidence_message=evidence,
                evidence_ids=self._claim_or_chunk_ids(grounding, run),
                evidence_strength="strong",
                evidence_tier="structured",
            )
        ]

    def _overconfidence_signals(self, run: RAGRun) -> list[_QualitySignal]:
        answer_lower = run.final_answer.lower()
        caller_confident = run.answer_confidence is not None and run.answer_confidence >= 0.8
        lexical_confident = bool(re.search(r"\b(definitely|certainly|clearly|without question)\b", answer_lower))
        weak_scores = [
            chunk.score
            for chunk in run.retrieved_chunks
            if chunk.score is not None and chunk.score < 0.5
        ]
        if not (caller_confident or lexical_confident) or not weak_scores:
            return []

        return [
            _QualitySignal(
                signal_name="overconfident_answer_with_weak_evidence",
                method="overconfidence_weak_evidence_check",
                failure_type=FailureType.LOW_CONFIDENCE,
                stage=FailureStage.CONFIDENCE,
                status="warn",
                severity="medium",
                evidence_message=(
                    "Overconfidence check: answer is presented confidently while "
                    f"{len(weak_scores)} retrieved chunk score(s) are below 0.50."
                ),
                evidence_ids=[chunk.chunk_id for chunk in run.retrieved_chunks if chunk.score is not None and chunk.score < 0.5],
                evidence_strength="medium",
                evidence_tier="proxy",
            )
        ]

    def _pass_with_report(self, evidence_message: str) -> AnalyzerResult:
        sig = EvidenceSignalMetadata(
            signal_name="answer_quality_no_issue_detected",
            source_analyzer=self.name(),
            method="answer_quality_screen",
            method_status="practical_approximation",
            calibration_status="uncalibrated",
            evidence_strength="advisory",
            evidence_tier="proxy",
            notes="No answer-quality failure was detected by the current uncalibrated checks.",
        )
        finding = AnalyzerFinding(
            finding_id="answer_quality:pass",
            analyzer_name=self.name(),
            failure_type=None,
            stage=FailureStage.GENERATION,
            status="pass",
            severity="none",
            evidence_message=evidence_message,
            signal_metadata=sig,
        )
        return AnalyzerResult(
            analyzer_name=self.name(),
            status="pass",
            evidence=[evidence_message],
            signal_metadata=[sig],
            analyzer_report=AnalyzerReport(
                analyzer_name=self.name(),
                overall_status="pass",
                findings=[finding],
                notes=["Answer-quality signals are uncalibrated."],
            ),
        )

    def _metadata(self, signal: _QualitySignal) -> EvidenceSignalMetadata:
        return EvidenceSignalMetadata(
            signal_name=signal.signal_name,
            source_analyzer=self.name(),
            method=signal.method,
            method_status="practical_approximation",
            calibration_status="uncalibrated",
            evidence_strength=signal.evidence_strength,  # type: ignore[arg-type]
            evidence_tier=signal.evidence_tier,  # type: ignore[arg-type]
            evidence_ids=signal.evidence_ids,
            notes="Calibration-ready answer-quality signal; not calibrated.",
        )

    def _finding(
        self,
        *,
        index: int,
        signal: _QualitySignal,
        metadata: EvidenceSignalMetadata,
    ) -> AnalyzerFinding:
        return AnalyzerFinding(
            finding_id=f"answer_quality:{signal.signal_name}:{index}",
            analyzer_name=self.name(),
            failure_type=signal.failure_type,
            stage=signal.stage,
            status=signal.status,  # type: ignore[arg-type]
            severity=signal.severity,  # type: ignore[arg-type]
            evidence_message=signal.evidence_message,
            signal_metadata=metadata,
            affected_chunk_ids=signal.evidence_ids,
        )

    def _prior_result(self, analyzer_name: str) -> AnalyzerResult | None:
        for result in self.config.get("prior_results", []):
            if result.analyzer_name == analyzer_name:
                return result
        return None

    def _claim_or_chunk_ids(self, result: AnalyzerResult, run: RAGRun) -> list[str]:
        if result.signal_metadata:
            ids: list[str] = []
            for signal in result.signal_metadata:
                ids.extend(signal.evidence_ids)
            if ids:
                return list(dict.fromkeys(ids))
        return [chunk.chunk_id for chunk in run.retrieved_chunks]

    def _expected_numbered_items(self, query: str) -> set[str]:
        query_lower = query.lower()
        match = re.search(r"\b(?:all\s+)?(\d+)\s+(?:steps?|requirements?|items?)\b", query_lower)
        if match is None:
            return set()
        count = int(match.group(1))
        if count <= 0 or count > 20:
            return set()
        return {str(index) for index in range(1, count + 1)}

    def _numbered_items(self, text: str) -> set[str]:
        return set(re.findall(r"\b(?:step|req(?:uirement)?)?\s*(\d{1,2})\b", text.lower()))

    def _remediation(self, signal: _QualitySignal) -> str:
        if signal.method == "answer_completeness_check":
            return "Regenerate the answer to cover all retrieved requirements requested by the query."
        if signal.method == "context_adherence_check":
            return "Strengthen generation grounding instructions so retrieved context controls the answer."
        return "Treat the answer confidence as an uncalibrated proxy and require stronger evidence before serving."
