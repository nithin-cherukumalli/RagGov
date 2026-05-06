"""Indexed access helpers for analyzer results."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType


class AnalyzerResultIndex:
    """Safe lookup helper over analyzer results.

    Downstream analyzers should prefer this index over ad hoc name matching or
    JSON scraping spread across the codebase.
    """

    def __init__(self, results: Iterable[AnalyzerResult]) -> None:
        self._results = list(results)
        self._by_name: dict[str, list[AnalyzerResult]] = defaultdict(list)
        self._by_failure: dict[FailureType, list[AnalyzerResult]] = defaultdict(list)

        for result in self._results:
            self._by_name[result.analyzer_name].append(result)
            if result.failure_type is not None:
                self._by_failure[result.failure_type].append(result)

    def all(self) -> list[AnalyzerResult]:
        return list(self._results)

    def by_name(self, analyzer_name: str) -> AnalyzerResult | None:
        matches = self._by_name.get(analyzer_name)
        return matches[-1] if matches else None

    def by_failure_type(self, failure_type: FailureType) -> AnalyzerResult | None:
        matches = self._by_failure.get(failure_type)
        return matches[-1] if matches else None

    def latest_with_field(self, field_name: str) -> AnalyzerResult | None:
        for result in reversed(self._results):
            if getattr(result, field_name, None) is not None:
                return result
        return None

    def parser_results(self) -> list[Any]:
        findings: list[Any] = []
        for result in self._results:
            if result.analyzer_name.lower().startswith("parser") or result.stage == FailureStage.PARSING:
                findings.append(result)
        return findings

    def grounding_result(self) -> AnalyzerResult | None:
        for result in reversed(self._results):
            if (
                result.analyzer_name == "ClaimGroundingAnalyzer"
                or result.claim_results
                or result.grounding_evidence_bundle is not None
            ):
                return result
        return None

    def structured_report(
        self,
        analyzer_name: str,
        field_name: str,
    ) -> dict[str, Any] | None:
        result = self.by_name(analyzer_name)
        if result is None:
            return None

        structured = getattr(result, field_name, None)
        if isinstance(structured, dict):
            return structured

        if result.evidence:
            try:
                payload = json.loads(result.evidence[0])
            except (TypeError, json.JSONDecodeError, IndexError):
                return None
            if isinstance(payload, dict):
                return payload

        return None
