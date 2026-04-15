"""Analyzer for assessing claim support against retrieved evidence."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal


logger = logging.getLogger(__name__)

from raggov.analyzers.base import BaseAnalyzer
from raggov.analyzers.grounding.claims import ClaimExtractor
from raggov.analyzers.retrieval.scope import STOPWORDS
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import (
    AnalyzerResult,
    ClaimResult,
    FailureStage,
    FailureType,
)
from raggov.models.run import RAGRun


NEGATION_SIGNALS = {"not", "never", "no", "no longer", "contrary to"}
REMEDIATION = (
    "{failed} of {total} claims are unsupported by retrieved context. "
    "Review retrieval quality or add source verification."
)


class ClaimGroundingAnalyzer(BaseAnalyzer):
    """Assess whether generated claims are grounded in retrieved chunks."""

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")

        claim_extractor_client = self.config.get("claim_extractor_client")
        extractor = ClaimExtractor(
            use_llm=claim_extractor_client is not None,
            llm_client=claim_extractor_client,
        )
        claims = extractor.extract(run.final_answer)
        if not claims:
            return self.skip("no claims extracted from final answer")

        claim_results = [self._evaluate_claim(claim, run.retrieved_chunks) for claim in claims]
        failed_results = [
            result
            for result in claim_results
            if result.label in {"unsupported", "contradicted"}
        ]
        contradicted_results = [
            result for result in claim_results if result.label == "contradicted"
        ]
        failed_fraction = len(failed_results) / len(claim_results)
        evidence = [
            result.model_dump_json(exclude_none=False) for result in claim_results
        ]
        remediation = REMEDIATION.format(
            failed=len(failed_results),
            total=len(claim_results),
        )

        if failed_fraction >= float(self.config.get("fail_threshold", 0.3)):
            return self._fail(
                FailureType.UNSUPPORTED_CLAIM,
                FailureStage.GROUNDING,
                evidence,
                remediation,
            )
        if contradicted_results:
            return self._warn(
                FailureType.CONTRADICTED_CLAIM,
                FailureStage.GROUNDING,
                evidence,
                remediation,
            )
        if failed_results:
            return self._warn(
                FailureType.UNSUPPORTED_CLAIM,
                FailureStage.GROUNDING,
                evidence,
                remediation,
            )

        return self._pass(evidence)

    def _evaluate_claim(
        self, claim: str, chunks: list[RetrievedChunk]
    ) -> ClaimResult:
        if bool(self.config.get("use_llm", False)) and self.config.get("llm_client") is not None:
            try:
                return self._evaluate_claim_with_llm(claim, chunks)
            except Exception as exc:
                logger.warning("LLM claim grounding failed, falling back to deterministic: %s", exc)
                return self._evaluate_claim_deterministic(claim, chunks)
        return self._evaluate_claim_deterministic(claim, chunks)

    def _evaluate_claim_deterministic(
        self, claim: str, chunks: list[RetrievedChunk]
    ) -> ClaimResult:
        claim_terms = self._terms(claim)
        if not claim_terms:
            return ClaimResult(
                claim_text=claim,
                label="unsupported",
                supporting_chunk_ids=[],
                confidence=0.0,
            )

        best_chunk: RetrievedChunk | None = None
        best_ratio = 0.0
        for chunk in chunks:
            chunk_terms = self._terms(chunk.text)
            overlap_ratio = len(claim_terms & chunk_terms) / len(claim_terms)
            if overlap_ratio > best_ratio:
                best_ratio = overlap_ratio
                best_chunk = chunk

        if best_chunk is not None and self._contains_negation_of_terms(
            best_chunk.text, claim_terms
        ):
            return ClaimResult(
                claim_text=claim,
                label="contradicted",
                supporting_chunk_ids=[best_chunk.chunk_id],
                confidence=best_ratio,
            )

        supporting_chunk_ids = [best_chunk.chunk_id] if best_chunk is not None else []
        if best_ratio >= 0.5:
            label: Literal["entailed", "unsupported", "contradicted"] = "entailed"
        else:
            label = "unsupported"

        return ClaimResult(
            claim_text=claim,
            label=label,
            supporting_chunk_ids=supporting_chunk_ids if best_ratio >= 0.2 else [],
            confidence=best_ratio,
        )

    def _evaluate_claim_with_llm(
        self, claim: str, chunks: list[RetrievedChunk]
    ) -> ClaimResult:
        payload = self._call_llm(claim, chunks)
        label = payload.get("label")
        if label not in {"entailed", "unsupported", "contradicted"}:
            raise ValueError("invalid LLM grounding label")
        evidence_chunk_id = payload.get("evidence_chunk_id")
        supporting_chunk_ids = [str(evidence_chunk_id)] if evidence_chunk_id else []
        return ClaimResult(
            claim_text=claim,
            label=label,
            supporting_chunk_ids=supporting_chunk_ids,
            confidence=float(payload.get("confidence", 0.0)),
        )

    def _call_llm(self, claim: str, chunks: list[RetrievedChunk]) -> dict[str, Any]:
        client = self.config["llm_client"]
        prompt = self._prompt(claim, chunks)
        if hasattr(client, "chat"):
            response = client.chat(prompt)
        elif hasattr(client, "complete"):
            response = client.complete(prompt)
        else:
            raise TypeError("llm_client must provide chat() or complete()")
        parsed = self._parse_response(response)
        if not isinstance(parsed, dict):
            raise ValueError("LLM grounding response must be a JSON object")
        return parsed

    def _prompt(self, claim: str, chunks: list[RetrievedChunk]) -> str:
        relevant_chunks = "\n\n".join(
            f"[{chunk.chunk_id}] {chunk.text}" for chunk in chunks
        )
        return (
            "Does the following retrieved context support, contradict, or neither "
            "support nor contradict this claim?\n"
            f"Context: {relevant_chunks}\n"
            f"Claim: {claim}\n"
            'Answer with JSON: {"label": "entailed"|"unsupported"|"contradicted", '
            '"confidence": 0.0-1.0, "evidence_chunk_id": "chunk_id or null"}'
        )

    def _parse_response(self, response: object) -> Any:
        if isinstance(response, dict):
            if "text" in response:
                response = response["text"]
            elif "content" in response:
                response = response["content"]
            else:
                return response
        if not isinstance(response, str):
            response = str(response)
        return json.loads(response)

    def _contains_negation_of_terms(self, text: str, terms: set[str]) -> bool:
        tokens = self._tokens(text)
        for index, token in enumerate(tokens):
            window = tokens[max(0, index - 5) : index + 6]
            window_text = " ".join(window)
            if token in terms and any(
                re.search(rf"\b{re.escape(signal)}\b", window_text)
                for signal in NEGATION_SIGNALS
            ):
                return True
        return False

    def _terms(self, text: str) -> set[str]:
        return {
            token
            for token in self._tokens(text)
            if token not in STOPWORDS
        }

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())
