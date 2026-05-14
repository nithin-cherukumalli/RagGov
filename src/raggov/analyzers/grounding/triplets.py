"""
RefChecker-inspired triplet extraction interface for GovRAG policy claims.

RefChecker-inspired triplet extraction interface.  Not full RefChecker until
triplet checking and benchmark evaluation are added.

RefChecker (Hu et al., 2023) represents claims as knowledge triplets
(subject, predicate, object) and verifies them against reference text.
GovRAG's implementation provides a compatible extraction interface as an
optional v2 path that runs on top of atomic claim sentences.  Triplet
*checking* against evidence chunks is a planned future step.

Hierarchy
---------
Claim (sentence) → 1-N ClaimTriplets

Extraction methods
------------------
GenericRuleTripletExtractorV0
    Domain-agnostic rule baseline for factual subject-predicate-object claims.

GovernmentPolicyTripletExtractorV0
    Optional non-core heuristic patterns tuned for Indian government/policy text:
    GO numbers, circulars, policy mandates, eligibility rules, deadlines,
    and numeric qualifiers.

LLMTripletExtractorV1  (optional, requires llm_client)
    Prompts an LLM to extract knowledge triplets in JSON.
    Falls back to GenericRuleTripletExtractorV0 on parse failure.

Usage
-----
    extractor = GenericRuleTripletExtractorV0()
    triplets = extractor.extract("The SDK supports retries in version 2.4.")
    # → [ClaimTriplet(subject="The SDK", predicate="supports",
    #                  object="subsidy for farmers", values=["60%"], ...)]

Configuration gate
------------------
Triplet extraction is disabled by default.  Enable by setting
``enable_triplet_extraction: true`` in the analyzer config.  No default
analyzer behavior changes unless this flag is set.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ClaimTriplet:
    """
    A (subject, predicate, object) knowledge triplet extracted from a claim.

    RefChecker-inspired.  Extraction method and calibration status are always
    recorded so downstream consumers know the provenance.

    Fields
    ------
    triplet_id          Unique identifier for this triplet.
    source_claim_id     claim_id from the parent ExtractedClaim / ClaimEvidenceRecord.
    subject             Entity or document that the predicate acts on/from.
    predicate           Relationship verb or policy action.
    object              Target of the predicate (entity, value, concept).
    qualifiers          Modifiers such as "effective from", "up to", "only for".
    values              Numeric, date, percentage, or identifier values.
    source_text_span    Exact substring of the claim that yielded this triplet.
    extraction_method   Identifier for the extractor that produced this triplet.
    confidence_or_raw_score
                        Extractor confidence (0-1) or raw overlap score.
                        None for rule-based extractors that do not score.
    calibration_status  "uncalibrated" — no calibration has been performed.
    """

    triplet_id: str
    source_claim_id: str
    subject: str
    predicate: str
    object: str
    qualifiers: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)
    source_text_span: str = ""
    extraction_method: str = "rule_based_policy_v0"
    confidence_or_raw_score: float | None = None
    calibration_status: str = "uncalibrated"


# ---------------------------------------------------------------------------
# Extractor interface
# ---------------------------------------------------------------------------

class TripletExtractor(ABC):
    """
    Abstract interface for claim triplet extractors.

    RefChecker-inspired triplet extraction interface.  Not full RefChecker
    until triplet checking and benchmark evaluation are added.
    """

    @abstractmethod
    def extract(
        self,
        claim_text: str,
        source_claim_id: str = "claim_000",
    ) -> list[ClaimTriplet]:
        """
        Extract knowledge triplets from a single claim sentence.

        Args:
            claim_text:      The atomic or compound claim string.
            source_claim_id: ID of the parent claim for provenance.

        Returns:
            A list of ClaimTriplet objects.  May be empty if no patterns match.
        """


# ---------------------------------------------------------------------------
# Rule-based V0
# ---------------------------------------------------------------------------

# Government Order / circular / notification subject pattern
_GO_SUBJECT_RE = re.compile(
    r"""
    (?:
        G\.O\.(?:Ms|Rt|P|D)\.?No\.?\s*\d+      # G.O.Ms.No. 42
      | G\.O\.[^\s,;.]{0,30}                    # G.O.anything
      | (?:circular|notification|order|act|rule|policy|scheme|directive)
        (?:\s+no\.?\s*\d+)?                     # circular no. 5
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Policy predicate verbs
_PREDICATE_RE = re.compile(
    r"\b("
    r"mandates?|requires?|stipulates?|directs?|orders?|notifies?"
    r"|prohibits?|bans?|restricts?|prevents?|disallows?"
    r"|permits?|allows?|authorizes?|grants?|exempts?"
    r"|provides?|extends?|confers?|entitles?|awards?"
    r"|specifies?|defines?|establishes?|sets?\s+out"
    r"|applies?\s+to|covers?"
    r"|(?:are|is)\s+entitled\s+to|entitles?\s+to"
    r"|(?:are|is)\s+eligible\s+(?:for|to)"
    r"|(?:are|is|must\s+be)\s+required\s+to"
    r"|(?:are|is)\s+prohibited\s+from"
    r"|must\s+be\s+(?:filed|submitted|paid|deposited|registered|completed)"
    r"|(?:is|are)\s+applicable(?:\s+for)?"
    r"|(?:is|are)\s+levied"
    r"|(?:is|are)\s+payable"
    r"|(?:is|are)\s+due"
    r"|may\s+(?:take|avail|claim|apply|be\s+taken)"
    r")\b",
    re.IGNORECASE,
)

# Eligibility actor pattern
_ELIGIBILITY_SUBJECT_RE = re.compile(
    r"\b("
    r"(?:all\s+)?(?:central|state)?\s*government\s+employees?"
    r"|(?:all\s+)?(?:loanee|non-loanee)?\s*farmers?"
    r"|(?:all\s+)?(?:women|men)?\s*employees?"
    r"|(?:all\s+)?citizens?"
    r"|(?:women|men|persons?|individuals?)\s*(?:employees?|workers?|beneficiaries?)?"
    r"|beneficiaries?"
    r"|taxpayers?"
    r"|registered\s+(?:persons?|dealers?|taxpayers?)"
    r")\b",
    re.IGNORECASE,
)

# Numeric / percentage / amount / date values
_VALUE_RE = re.compile(
    r"""
    (?:
        (?:Rs?\.?\s*)?[\d,]+(?:\.\d+)?\s*%         # 60% / Rs. 50%
      | (?:Rs?\.?\s*)?[\d,]+(?:\.\d+)?             # Rs. 200 / 5000
        \s*(?:lakhs?|crores?|thousands?)?           # optional unit
      | \d{1,2}[-/]\d{1,2}[-/]\d{2,4}              # 01-01-2023
      | \d{4}[-/]\d{2}[-/]\d{2}                    # 2023-01-01
      | \b(?:january|february|march|april|may|june|july|august|september|
             october|november|december)\s+\d{4}     # March 2023
      | \b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|
                       september|october|november|december)\s+\d{4}
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Qualifier phrases
_QUALIFIER_RE = re.compile(
    r"\b("
    r"effective\s+(?:from|date\s+of)"
    r"|with\s+effect\s+from"
    r"|up\s+to(?:\s+a\s+maximum\s+of)?"
    r"|not\s+exceeding"
    r"|subject\s+to"
    r"|only\s+for"
    r"|for\s+the\s+first\s+\w+"
    r"|per\s+(?:day|month|year|beneficiary|family|person)"
    r"|in\s+respect\s+of"
    r"|provided\s+that"
    r")\b",
    re.IGNORECASE,
)

# Object / noun chunk after predicate: anything up to comma / period / semicolon
_OBJECT_RE = re.compile(
    r"(?:that\s+)?(.{4,120}?)(?=[,;.]|$|\band\b|\bor\b)",
    re.IGNORECASE | re.DOTALL,
)


_GENERIC_SUBJECT_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9_.-]*(?:\s+[A-Z][A-Za-z0-9_.-]*){0,6}|"
    r"(?:the|a|an)\s+[A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+){0,5})\b"
)


class GenericRuleTripletExtractorV0(TripletExtractor):
    """Domain-agnostic rule triplet extractor for simple factual claims."""

    _METHOD = "generic_rule_triplet_v0"

    def extract(
        self,
        claim_text: str,
        source_claim_id: str = "claim_000",
    ) -> list[ClaimTriplet]:
        values = [m.group(0).strip() for m in _VALUE_RE.finditer(claim_text)]
        qualifiers = [m.group(0).strip() for m in _QUALIFIER_RE.finditer(claim_text)]
        pred_match = _PREDICATE_RE.search(claim_text)
        if pred_match is None:
            return []

        before_pred = claim_text[: pred_match.start()].strip().rstrip(",;:")
        subject = self._extract_subject(before_pred)
        predicate = pred_match.group(0).strip()
        obj = self._extract_object(claim_text[pred_match.end():].strip())
        if not subject or not obj:
            return []

        return [
            ClaimTriplet(
                triplet_id=f"triplet_{uuid.uuid4().hex[:8]}",
                source_claim_id=source_claim_id,
                subject=subject,
                predicate=predicate,
                object=obj,
                qualifiers=qualifiers,
                values=values,
                source_text_span=claim_text[:160].strip(),
                extraction_method=self._METHOD,
            )
        ]

    def _extract_subject(self, text: str) -> str:
        if not text:
            return "unspecified"
        match = _GENERIC_SUBJECT_RE.search(text)
        if match:
            return match.group(0).strip()
        tokens = text.split()
        return " ".join(tokens[-6:]).strip()

    def _extract_object(self, text: str) -> str:
        text = text.strip()
        m = _OBJECT_RE.match(text)
        if m:
            return m.group(1).strip()
        return text[:120].split(".")[0].split(";")[0].strip()


class GovernmentPolicyTripletExtractorV0(TripletExtractor):
    """
    Heuristic triplet extractor tuned for Indian government/policy text.

    Handles:
    - GO/circular/notification as subjects
    - policy predicate verbs (mandates, allows, prohibits, …)
    - eligibility actor subjects
    - date and numeric qualifiers

    Extraction method label: ``rule_based_policy_v0``

    RefChecker-inspired triplet extraction interface.  Not full RefChecker
    until triplet checking and benchmark evaluation are added.
    """

    _METHOD = "rule_based_policy_v0"

    def extract(
        self,
        claim_text: str,
        source_claim_id: str = "claim_000",
    ) -> list[ClaimTriplet]:
        """Extract triplets via heuristic pattern matching."""
        triplets: list[ClaimTriplet] = []
        values = self._extract_values(claim_text)
        qualifiers = self._extract_qualifiers(claim_text)

        # Strategy 1: GO / document subject → predicate → object
        for go_match in _GO_SUBJECT_RE.finditer(claim_text):
            subject = go_match.group(0).strip()
            after_subject = claim_text[go_match.end():]
            pred_match = _PREDICATE_RE.search(after_subject)
            if pred_match:
                predicate = pred_match.group(0).strip()
                after_pred = after_subject[pred_match.end():].strip()
                obj = self._extract_object(after_pred)
                triplets.append(
                    ClaimTriplet(
                        triplet_id=f"triplet_{uuid.uuid4().hex[:8]}",
                        source_claim_id=source_claim_id,
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        qualifiers=qualifiers,
                        values=values,
                        source_text_span=claim_text[
                            go_match.start(): min(
                                go_match.end() + pred_match.end() + len(obj) + 20,
                                len(claim_text),
                            )
                        ].strip(),
                        extraction_method=self._METHOD,
                    )
                )

        # Strategy 2: Eligibility subject → predicate → object
        for elig_match in _ELIGIBILITY_SUBJECT_RE.finditer(claim_text):
            subject = elig_match.group(0).strip()
            after_subject = claim_text[elig_match.end():]
            pred_match = _PREDICATE_RE.search(after_subject)
            if pred_match:
                predicate = pred_match.group(0).strip()
                after_pred = after_subject[pred_match.end():].strip()
                obj = self._extract_object(after_pred)
                triplets.append(
                    ClaimTriplet(
                        triplet_id=f"triplet_{uuid.uuid4().hex[:8]}",
                        source_claim_id=source_claim_id,
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        qualifiers=qualifiers,
                        values=values,
                        source_text_span=claim_text[
                            elig_match.start(): min(
                                elig_match.end() + pred_match.end() + len(obj) + 20,
                                len(claim_text),
                            )
                        ].strip(),
                        extraction_method=self._METHOD,
                    )
                )

        # Strategy 3: Bare predicate — no explicit named subject found
        if not triplets:
            pred_match = _PREDICATE_RE.search(claim_text)
            if pred_match:
                before_pred = claim_text[: pred_match.start()].strip().rstrip(",;:")
                subject = before_pred[-80:].strip() if before_pred else "unspecified"
                predicate = pred_match.group(0).strip()
                after_pred = claim_text[pred_match.end():].strip()
                obj = self._extract_object(after_pred)
                triplets.append(
                    ClaimTriplet(
                        triplet_id=f"triplet_{uuid.uuid4().hex[:8]}",
                        source_claim_id=source_claim_id,
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        qualifiers=qualifiers,
                        values=values,
                        source_text_span=claim_text[:160].strip(),
                        extraction_method=self._METHOD,
                    )
                )

        return triplets

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _extract_values(self, text: str) -> list[str]:
        return [m.group(0).strip() for m in _VALUE_RE.finditer(text)]

    def _extract_qualifiers(self, text: str) -> list[str]:
        return [m.group(0).strip() for m in _QUALIFIER_RE.finditer(text)]

    def _extract_object(self, text: str) -> str:
        """Return a cleaned object string from whatever follows the predicate."""
        text = text.strip()
        m = _OBJECT_RE.match(text)
        if m:
            return m.group(1).strip()
        # Fallback: first 120 chars up to natural boundary
        return text[:120].split(".")[0].split(";")[0].strip()


# Backward-compatible alias for existing government benchmark tests and callers.
RuleBasedPolicyTripletExtractorV0 = GovernmentPolicyTripletExtractorV0


# ---------------------------------------------------------------------------
# LLM V1 (optional)
# ---------------------------------------------------------------------------

class LLMTripletExtractorV1(TripletExtractor):
    """
    LLM-based triplet extractor.

    Prompts the LLM to return a JSON array of triplets.  Falls back to
    GenericRuleTripletExtractorV0 on any parse or call failure.

    Requirements:
    - ``llm_client`` must expose ``chat(prompt: str) -> str`` or
      ``complete(prompt: str) -> str``.

    Extraction method label: ``llm_triplet_v1``

    RefChecker-inspired triplet extraction interface.  Not full RefChecker
    until triplet checking and benchmark evaluation are added.
    """

    _METHOD = "llm_triplet_v1"

    def __init__(self, llm_client: Any) -> None:
        if llm_client is None:
            raise ValueError(
                "LLMTripletExtractorV1 requires a non-None llm_client."
            )
        self._client = llm_client
        self._fallback = GenericRuleTripletExtractorV0()

    def extract(
        self,
        claim_text: str,
        source_claim_id: str = "claim_000",
    ) -> list[ClaimTriplet]:
        """Extract triplets via LLM, falling back to rule-based on failure."""
        try:
            raw = self._call_llm(claim_text)
            return self._parse_response(raw, claim_text, source_claim_id)
        except Exception as exc:
            logger.warning(
                "LLMTripletExtractorV1 failed for claim '%s': %s. "
                "Falling back to GenericRuleTripletExtractorV0.",
                claim_text[:80],
                exc,
            )
            fallback = self._fallback.extract(claim_text, source_claim_id)
            for t in fallback:
                t.extraction_method = f"llm_fallback_to_{GenericRuleTripletExtractorV0._METHOD}"
            return fallback

    def _call_llm(self, claim_text: str) -> str:
        prompt = self._build_prompt(claim_text)
        if hasattr(self._client, "chat"):
            return self._client.chat(prompt)
        if hasattr(self._client, "complete"):
            return self._client.complete(prompt)
        raise TypeError("llm_client must expose chat() or complete()")

    def _build_prompt(self, claim_text: str) -> str:
        return (
            "Extract knowledge triplets from the following claim. "
            "Use only information present in the claim; do not add external knowledge.\n\n"
            "Rules:\n"
            "- Return a JSON array. Each element must have exactly these keys:\n"
            "    subject (string): the entity or document making or subject to the claim\n"
            "    predicate (string): the relationship or policy action\n"
            "    object (string): the target of the predicate\n"
            "    qualifiers (array of strings): modifiers like 'effective from', 'up to'\n"
            "    values (array of strings): all numeric, percentage, or date values\n"
            "- Preserve exact identifiers, versions, dates, numbers, and names.\n"
            "- Do not split or rephrase values.\n"
            "- Return [] if no clear triplet can be extracted.\n\n"
            f"Claim: {claim_text}\n\n"
            "Answer with JSON only:"
        )

    def _parse_response(
        self,
        raw: Any,
        claim_text: str,
        source_claim_id: str,
    ) -> list[ClaimTriplet]:
        # Normalise to string
        if isinstance(raw, dict):
            raw = raw.get("text") or raw.get("content") or json.dumps(raw)
        if not isinstance(raw, str):
            raw = str(raw)

        # Strip markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")

        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError(f"LLM triplet response must be a JSON array, got: {type(parsed)}")

        triplets: list[ClaimTriplet] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            subject = str(item.get("subject", "")).strip()
            predicate = str(item.get("predicate", "")).strip()
            obj = str(item.get("object", "")).strip()
            if not (subject and predicate and obj):
                continue
            triplets.append(
                ClaimTriplet(
                    triplet_id=f"triplet_{uuid.uuid4().hex[:8]}",
                    source_claim_id=source_claim_id,
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    qualifiers=list(item.get("qualifiers") or []),
                    values=list(item.get("values") or []),
                    source_text_span=claim_text[:160],
                    extraction_method=self._METHOD,
                    confidence_or_raw_score=float(item.get("confidence", 0.0)),
                )
            )
        return triplets


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_triplet_extractor(config: dict[str, Any]) -> TripletExtractor | None:
    """
    Return the appropriate TripletExtractor based on config, or None if
    triplet extraction is disabled.

    Config keys
    -----------
    enable_triplet_extraction : bool   (default False)
        Gate flag.  Nothing happens unless this is True.
    triplet_extractor_mode : str       (default "generic_rule_v0")
        "generic_rule_v0" → GenericRuleTripletExtractorV0
        "government_policy_v0" → GovernmentPolicyTripletExtractorV0
        "rule_based_v0" → deprecated alias for GovernmentPolicyTripletExtractorV0
        "llm_v1" → LLMTripletExtractorV1 (requires llm_client)
    llm_client : object                (default None)
        Required when triplet_extractor_mode = "llm_v1".
    """
    if not config.get("enable_triplet_extraction", False):
        return None

    mode = config.get("triplet_extractor_mode", "generic_rule_v0")
    if mode == "llm_v1":
        client = config.get("llm_client")
        if client is None:
            logger.warning(
                "triplet_extractor_mode='llm_v1' requires llm_client; "
                "falling back to generic_rule_v0."
            )
            return GenericRuleTripletExtractorV0()
        return LLMTripletExtractorV1(client)
    if mode in {"government_policy_v0", "rule_based_policy_v0", "rule_based_v0"}:
        return GovernmentPolicyTripletExtractorV0()

    return GenericRuleTripletExtractorV0()
