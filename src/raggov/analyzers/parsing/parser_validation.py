"""Analyzer for detecting parser-stage structure degradation."""

from __future__ import annotations

from typing import Any

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.run import RAGRun
from raggov.parser_validation import (
    ChunkIR,
    MetadataNormalizer,
    NormalizedMetadata,
    PROFILE_LINT_REMEDIATION,
    ParserFailureType,
    ParserFinding,
    ParserSeverity,
    ParserValidationConfig,
    ParserValidationEngine,
    ParserValidationProfile,
    ProfileLintEngine,
    ProfileLintIssue,
    ProfileLintReport,
)
from raggov.parser_validation.adapters import (
    chunks_from_rag_run,
    _get_attr_or_key,
    parsed_doc_from_run_metadata,
)


class ParserValidationAnalyzer(BaseAnalyzer):
    """
    Detect parser-stage structural failures.

    v1 design:
    - Prefer parser/chunker metadata when available.
    - Use parser-agnostic structural validators.
    - Fall back to explicitly marked text-only heuristics when no rich parser IR is available.
    - Respect declared chunking strategy: validators only enforce guarantees the strategy promised.
    """

    weight = 0.85

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        profile: ParserValidationProfile | None = None,
    ) -> None:
        super().__init__(config)
        self.profile = profile

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")

        try:
            profile = self._resolve_profile(run)
        except (ValueError, TypeError, Exception) as exc:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="warn",
                stage=FailureStage.PARSING,
                failure_type=FailureType.METADATA_LOSS,
                evidence=[
                    "parser_validation_profile_malformed",
                    f"Parser profile validation failed: {str(exc)}",
                ],
                remediation="Provide a valid parser_validation_profile to enable parser-stage validation.",
            )

        if profile is None:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="warn",
                stage=FailureStage.PARSING,
                failure_type=FailureType.METADATA_LOSS,
                evidence=[
                    "parser_validation_profile_missing",
                    "parser validation skipped because no parser/chunking profile was provided",
                ],
                remediation="Provide parser_validation_profile metadata to enable parser-stage validation.",
            )

        config = self._build_config(profile)
        engine = ParserValidationEngine(config=config)

        parsed_doc = parsed_doc_from_run_metadata(run)
        chunks = self._chunks_from_run(run, profile)
        lint_report = self._lint_profile(chunks, config, profile)

        if lint_report.authority_blocked:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="fail",
                failure_type=FailureType.METADATA_LOSS,
                stage=FailureStage.PARSING,
                evidence=self._format_lint_evidence(lint_report.errors),
                remediation=PROFILE_LINT_REMEDIATION,
            )

        findings = engine.validate(parsed_doc=parsed_doc, chunks=chunks)

        actionable_findings = [
            finding
            for finding in findings
            if finding.severity in {ParserSeverity.FAIL, ParserSeverity.WARN}
        ]

        if not actionable_findings:
            if lint_report.warnings:
                return AnalyzerResult(
                    analyzer_name=self.name(),
                    status="warn",
                    failure_type=FailureType.METADATA_LOSS,
                    stage=FailureStage.PARSING,
                    evidence=self._format_lint_evidence(lint_report.warnings),
                    remediation=PROFILE_LINT_REMEDIATION,
                )
            return self._pass()

        primary = actionable_findings[0]

        return AnalyzerResult(
            analyzer_name=self.name(),
            status=self._status_from_severity(primary.severity),
            failure_type=self._map_failure_type(primary.failure_type),
            stage=FailureStage.PARSING,
            evidence=self._format_evidence(
                actionable_findings,
                config.chunking_profile.strategy_type.value,
            )
            + self._format_lint_evidence(lint_report.warnings),
            remediation=primary.remediation,
        )

    def _resolve_profile(self, run: RAGRun) -> ParserValidationProfile | None:
        """
        Resolve the parser validation profile from config or run metadata.
        Returns None if no profile is provided (calling code handles as 'warn').
        """
        if self.profile is not None:
            return self.profile

        config_profile = self.config.get("parser_validation_profile")
        if config_profile is not None:
            return self._coerce_profile(config_profile)

        run_metadata = getattr(run, "metadata", None)
        if isinstance(run_metadata, dict):
            run_profile = run_metadata.get("parser_validation_profile")
            if run_profile is not None:
                return self._coerce_profile(run_profile)

        return None

    def _coerce_profile(self, value: Any) -> ParserValidationProfile:
        if isinstance(value, ParserValidationProfile):
            return value
        if isinstance(value, dict):
            return ParserValidationProfile.model_validate(value)
        raise TypeError(
            "parser_validation_profile must be a ParserValidationProfile or dict"
        )

    def _build_config(self, profile: ParserValidationProfile) -> ParserValidationConfig:
        return ParserValidationConfig(
            min_table_structure_score=float(
                self.config.get("min_table_structure_score", 0.70)
            ),
            min_metadata_coverage=float(
                self.config.get("min_metadata_coverage", 0.80)
            ),
            min_provenance_coverage=float(
                self.config.get("min_provenance_coverage", 0.90)
            ),
            max_chunk_boundary_damage_ratio=float(
                self.config.get("max_chunk_boundary_damage_ratio", 0.10)
            ),
            enable_text_fallback_heuristics=bool(
                self.config.get("enable_text_fallback_heuristics", True)
            ),
            chunking_profile=profile.chunking_strategy,
        )

    def _lint_profile(
        self,
        chunks: list[ChunkIR],
        config: ParserValidationConfig,
        profile: ParserValidationProfile,
    ) -> ProfileLintReport:
        return ProfileLintEngine(
            min_metadata_coverage=config.min_metadata_coverage,
            min_provenance_coverage=config.min_provenance_coverage,
        ).lint(chunks, config.chunking_profile, profile)

    def _chunks_from_run(
        self,
        run: RAGRun,
        profile: ParserValidationProfile,
    ) -> list[ChunkIR]:
        if profile.infer_from_legacy:
            return chunks_from_rag_run(run)

        normalizer = MetadataNormalizer(profile.metadata_mapping)
        return [
            self._chunk_from_normalized_metadata(raw_chunk, normalizer)
            for raw_chunk in run.retrieved_chunks
        ]

    def _chunk_from_normalized_metadata(
        self,
        raw_chunk: Any,
        normalizer: MetadataNormalizer,
    ) -> ChunkIR:
        raw_payload = self._raw_chunk_payload(raw_chunk)
        normalized = normalizer.normalize(raw_payload)

        chunk_id = normalized.chunk_id or _get_attr_or_key(raw_chunk, "chunk_id", None)
        if chunk_id is None:
            chunk_id = _get_attr_or_key(raw_chunk, "id", "unknown_chunk")

        text = normalized.text
        if text is None:
            text = _get_attr_or_key(raw_chunk, "text", None)
        if text is None:
            text = _get_attr_or_key(raw_chunk, "content", None)
        if text is None:
            text = _get_attr_or_key(raw_chunk, "page_content", "")

        metadata = self._metadata_from_normalized(normalized)

        page_end = normalized.page_end
        if page_end is None:
            page_end = normalized.page_start

        return ChunkIR(
            chunk_id=str(chunk_id),
            text=str(text),
            source_element_ids=normalized.source_element_ids,
            source_table_ids=normalized.source_table_ids,
            page_start=normalized.page_start,
            page_end=page_end,
            section_path=normalized.section_path,
            metadata=metadata,
        )

    def _raw_chunk_payload(self, raw_chunk: Any) -> dict[str, Any]:
        metadata = _get_attr_or_key(raw_chunk, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}

        return {
            "chunk_id": _get_attr_or_key(raw_chunk, "chunk_id", None),
            "id": _get_attr_or_key(raw_chunk, "id", None),
            "text": _get_attr_or_key(raw_chunk, "text", None),
            "content": _get_attr_or_key(raw_chunk, "content", None),
            "page_content": _get_attr_or_key(raw_chunk, "page_content", None),
            "source_doc_id": _get_attr_or_key(raw_chunk, "source_doc_id", None),
            "score": _get_attr_or_key(raw_chunk, "score", None),
            "metadata": dict(metadata),
        }

    def _metadata_from_normalized(
        self,
        normalized: NormalizedMetadata,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        metadata.update(normalized.boundary_flags)
        metadata.update(normalized.domain_fields)

        if normalized.document_id is not None:
            metadata["document_id"] = normalized.document_id
        if normalized.parent_id is not None:
            metadata["parent_id"] = normalized.parent_id
        if normalized.chunking_strategy is not None:
            metadata["chunking_strategy"] = normalized.chunking_strategy

        metadata["domain_fields"] = normalized.domain_fields
        metadata["unmapped"] = normalized.unmapped

        return metadata

    def _status_from_severity(self, severity: ParserSeverity) -> str:
        if severity == ParserSeverity.FAIL:
            return "fail"
        if severity == ParserSeverity.WARN:
            return "warn"
        return "pass"

    def _map_failure_type(self, failure_type: ParserFailureType) -> FailureType:
        mapping = {
            ParserFailureType.TABLE_STRUCTURE_LOSS: FailureType.TABLE_STRUCTURE_LOSS,
            ParserFailureType.HIERARCHY_FLATTENING: FailureType.HIERARCHY_FLATTENING,
            ParserFailureType.METADATA_LOSS: FailureType.METADATA_LOSS,
            ParserFailureType.CHUNK_BOUNDARY_DAMAGE: FailureType.HIERARCHY_FLATTENING,
            ParserFailureType.PROVENANCE_MISSING: FailureType.METADATA_LOSS,
            ParserFailureType.OCR_DEGRADATION: FailureType.METADATA_LOSS,
        }

        return mapping.get(failure_type, FailureType.METADATA_LOSS)

    def _format_evidence(self, findings: list[ParserFinding], strategy: str) -> list[str]:
        formatted: list[str] = []

        for finding in findings:
            evidence_type = "heuristic" if finding.is_heuristic else "structural"

            prefix = (
                f"[{finding.validator_name}] "
                f"{finding.failure_type.value} "
                f"severity={finding.severity.value} "
                f"confidence={finding.confidence:.2f} "
                f"chunking_strategy={strategy} "
                f"evidence_type={evidence_type}"
            )

            if not finding.evidence:
                formatted.append(prefix)
                continue

            for evidence in finding.evidence:
                location_bits = []

                if evidence.chunk_id:
                    location_bits.append(f"chunk={evidence.chunk_id}")

                if evidence.table_id:
                    location_bits.append(f"table={evidence.table_id}")

                if evidence.element_id:
                    location_bits.append(f"element={evidence.element_id}")

                location = f" ({', '.join(location_bits)})" if location_bits else ""
                formatted.append(f"{prefix}{location}: {evidence.message}")

        return formatted

    def _format_lint_evidence(
        self,
        issues: tuple[ProfileLintIssue, ...],
    ) -> list[str]:
        formatted: list[str] = []

        for issue in issues:
            formatted.append(
                "[profile_lint] "
                f"{issue.code} "
                f"severity={issue.severity}: "
                f"{issue.message}"
            )

        return formatted
