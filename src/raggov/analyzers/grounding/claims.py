"""Claim extraction substrate for GovRAG grounding."""

from __future__ import annotations

import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)

_ABBREV_CHAIN_RE = re.compile(r"^[A-Z](?:\.[A-Za-z0-9]\w*)+\.$")
_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_NUMBER_RE = re.compile(r"(?:[$€£₹])?\d[\d,]*(?:\.\d+)?%?")
_DATE_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b(?:\s+\d{1,2},?)?(?:\s+\d{4})?"
    r"|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_SUBSTANTIVE_RE = re.compile(
    r"\d"
    r"|\$|€|£|₹|%"
    r"|G\.O\b"
    r"|\b(?:shall|may|require(?:s|d)?|mandated?|prohibit(?:ed)?|exempt|approv(?:al|ed)|"
    r"authoriz(?:ed)?|allow(?:ed|s)?|eligib(?:le|ility)|"
    r"apply|applies|deprecat(?:e|es|ed)?|capital|benefits?|"
    r"found(?:ed|er)|headquarter(?:ed|s)?|visa|income|health|insurance|"
    r"profit|revenue|expenses|grant|citizens?|salary|"
    r"smoking|carpet|blue|team|race|won|winner|definitely|comply|compliance|"
    r"permit(?:ted)?|regulations?|rules?|act|polic(?:y|ies)|"
    r"deadline|threshold|applicable|effective|enforce(?:d)?|"
    r"mandatory|optional|waive(?:r)?|refund(?:s|able)?|renewal(?:s)?|"
    r"downgrade(?:s)?|credit(?:s)?|billing|subscriber(?:s)?|version|sdk|api|guideline|"
    r"dose|dosage|contraindicat(?:ion|ed)?|disclosure|manual|baseline|exception|"
    r"procedure|caus(?:e|al|es)|compar(?:e|ed|ison)|higher|lower|increase|decrease|"
    r"design|development|start(?:ed|s)?|done|complete(?:d)?)\b",
    re.IGNORECASE,
)
_ABSTENTION_RE = re.compile(
    r"\b(?:i\s+don'?t\s+have|i\s+do\s+not\s+have|not\s+provided|not\s+specified|"
    r"cannot\s+determine|unable\s+to\s+determine|no\s+information)\b",
    re.IGNORECASE,
)
_SHORT_ENTITY_RE = re.compile(
    r"^[A-Z][A-Za-z0-9_]{2,}(?:[._-][A-Za-z0-9]*)*(?:\s+[A-Z][A-Za-z0-9_.-]*){0,4}\.?$"
)
_SHORT_ENTITY_PLACEHOLDERS = {"Answer", "Response", "Result"}

ClaimAtomicity = Literal["atomic", "compound", "unclear"]
ClaimType = Literal[
    "numeric",
    "temporal",
    "policy_rule",
    "eligibility",
    "obligation",
    "prohibition",
    "causal",
    "comparison",
    "definition",
    "entity_attribute",
    "version_validity",
    "other",
]


class ExtractedClaim(BaseModel):
    """Structured factual claim extracted from a generated answer."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    claim_id: str
    claim_text: str
    source_sentence: str
    source_start_char: int
    source_end_char: int
    atomicity_status: ClaimAtomicity
    claim_type: ClaimType
    entities: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    extraction_method: str
    extraction_reason: str
    extraction_confidence: float | None = None
    extraction_warnings: list[str] = Field(default_factory=list)
    should_verify: bool
    skip_reason: str | None = None


class AtomicSubclaim(BaseModel):
    """
    An atomic subclaim produced by decomposing a compound claim.

    Each subclaim is independently verifiable.  The parent's final
    verification label is derived by aggregating all required subclaims:
      - parent supported  <=> all required subclaims supported
      - parent contradicted <=> any required subclaim contradicted
      - parent insufficient <=> any required subclaim insufficient (and none contradicted)
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    parent_claim_id: str
    subclaim_id: str
    text: str
    decomposition_method: str  # e.g. "heuristic_conjunction_split" | "llm_decomposer"
    atomicity_status: ClaimAtomicity = "atomic"
    required_support: bool = True  # False marks informational subclaims
    entities: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)


class BaseClaimExtractor(ABC):
    """Interface for answer-to-claims extraction."""

    def extract(self, answer: str) -> list[str]:
        claims = self.extract_structured(answer)
        return [claim.claim_text for claim in claims if claim.should_verify]

    @abstractmethod
    def extract_structured(self, answer: str) -> list[ExtractedClaim]:
        raise NotImplementedError

    # Pronoun-starting sentences: the LLM may legitimately resolve the
    # pronoun in claim_text while keeping the original sentence as source.
    _PRONOUN_START_RE = re.compile(
        r"^(?:It|They|This|These|That|Those|He|She|We|I|One)\b", re.IGNORECASE
    )

    def _validate_against_answer(self, answer: str, claims: list[ExtractedClaim]) -> list[ExtractedClaim]:
        validated: list[ExtractedClaim] = []
        for claim in claims:
            warning_messages: list[str] = []
            source_sentence = claim.source_sentence.strip()
            sentence_start = answer.find(source_sentence) if source_sentence else -1

            # Primary resolution: find source_sentence verbatim.
            if sentence_start == -1 and claim.claim_text:
                # If the claim_text differs from source_sentence (pronoun resolved),
                # try to locate source_sentence directly in the answer first.
                if source_sentence and source_sentence in answer:
                    sentence_start = answer.find(source_sentence)

            if sentence_start == -1 and claim.claim_text:
                raw_sentence = self._resolve_source_sentence(answer, claim.claim_text)
                if raw_sentence is not None:
                    source_sentence, sentence_start = raw_sentence

            if sentence_start == -1 or not source_sentence:
                # Last-chance: if source_sentence starts with a pronoun and
                # the sentence exists verbatim in the answer, accept it as
                # grounded — the claim_text is a pronoun-resolved rewrite.
                if source_sentence and self._PRONOUN_START_RE.match(source_sentence):
                    pos = answer.find(source_sentence)
                    if pos != -1:
                        sentence_start = pos

            if sentence_start == -1 or not source_sentence:
                warning_messages.append("source_sentence_not_found_in_answer")
            else:
                claim.source_sentence = source_sentence
                claim.source_start_char = sentence_start
                claim.source_end_char = sentence_start + len(source_sentence)

            if source_sentence and sentence_start != -1:
                # For pronoun-resolved claims the claim_text legitimately differs
                # from source_sentence; skip the invented-content check in that case
                # as long as the entities in claim_text are a superset of the
                # entities already present across the full answer.
                is_pronoun_resolved = (
                    self._PRONOUN_START_RE.match(source_sentence)
                    and claim.claim_text.strip() != source_sentence.strip()
                )
                if is_pronoun_resolved:
                    invented = self._invented_content(claim, answer)  # check against full answer
                else:
                    invented = self._invented_content(claim, source_sentence)
                if invented:
                    warning_messages.append("possible_extraction_hallucination")

            claim.extraction_warnings = [*claim.extraction_warnings, *warning_messages]
            if warning_messages:
                claim.should_verify = False
                claim.skip_reason = "claim_not_grounded_in_answer"
            validated.append(claim)
        return validated

    def _resolve_source_sentence(self, answer: str, claim_text: str) -> tuple[str, int] | None:
        raw_sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])(?:\s+|$)", answer)
            if sentence.strip()
        ]
        sentences = self._merge_abbreviation_splits(raw_sentences)
        for sentence in sentences:
            if claim_text.strip().rstrip(".") in sentence.rstrip("."):
                start = answer.find(sentence)
                if start != -1:
                    return sentence, start
        return None

    @staticmethod
    def _merge_abbreviation_splits(fragments: list[str]) -> list[str]:
        merged: list[str] = []
        i = 0
        while i < len(fragments):
            if i + 1 < len(fragments) and _ABBREV_CHAIN_RE.match(fragments[i]):
                merged.append(fragments[i] + " " + fragments[i + 1])
                i += 2
            else:
                merged.append(fragments[i])
                i += 1
        return merged

    def _invented_content(self, claim: ExtractedClaim, source_sentence: str) -> bool:
        source_lower = source_sentence.lower()
        for number in claim.numbers or _extract_numbers(claim.claim_text):
            if _normalize_number(number) not in _normalize_number(source_sentence):
                return True
        for date in claim.dates or _extract_dates(claim.claim_text):
            if date.lower() not in source_lower:
                return True
        source_entities = {_normalize_entity(entity) for entity in _extract_entities(source_sentence)}
        for entity in claim.entities or _extract_entities(claim.claim_text):
            normalized = _normalize_entity(entity)
            if normalized and normalized not in source_entities and normalized not in source_lower:
                return True
        return False


class HeuristicClaimExtractorV0(BaseClaimExtractor):
    """Visible heuristic fallback when no LLM extractor is available."""

    def extract_structured(self, answer: str) -> list[ExtractedClaim]:
        raw_sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])(?:\s+|$)", answer)
            if sentence.strip()
        ]
        sentences = self._merge_abbreviation_splits(raw_sentences)
        claims: list[ExtractedClaim] = []
        search_start = 0

        for index, sentence in enumerate(sentences, start=1):
            start_char = answer.find(sentence, search_start)
            if start_char == -1:
                start_char = search_start
            end_char = start_char + len(sentence)
            search_start = end_char

            substantive_matches = len(_SUBSTANTIVE_RE.findall(sentence))
            stripped_sentence = sentence.strip().rstrip(".")
            if (
                stripped_sentence not in _SHORT_ENTITY_PLACEHOLDERS
                and _SHORT_ENTITY_RE.match(sentence.strip())
            ):
                substantive_matches = max(substantive_matches, 1)
            conjunctions = len(re.findall(r"\b(?:and|or|but)\b", sentence, re.IGNORECASE))
            atomicity: ClaimAtomicity = (
                "compound" if conjunctions > 0 or substantive_matches > 1 else "atomic"
            )
            is_substantive = substantive_matches > 0
            is_long_enough = len(sentence.split()) > 4
            should_verify = True
            skip_reason = None
            if _ABSTENTION_RE.search(sentence):
                should_verify = False
                skip_reason = "answer_abstention"
            elif not is_substantive and not is_long_enough:
                should_verify = False
                skip_reason = "short_non_substantive"
            elif not is_substantive:
                should_verify = False
                skip_reason = "lacks_substantive_terms"

            claims.append(
                ExtractedClaim(
                    claim_id=f"claim_{index:03d}",
                    claim_text=sentence,
                    source_sentence=sentence,
                    source_start_char=start_char,
                    source_end_char=end_char,
                    atomicity_status=atomicity,
                    claim_type=_heuristic_claim_type(sentence),
                    entities=_extract_entities(sentence),
                    dates=_extract_dates(sentence),
                    numbers=_extract_numbers(sentence),
                    extraction_method="heuristic_atomic_claim_extractor_v0",
                    extraction_reason="heuristic_sentence_split",
                    extraction_confidence=None,
                    extraction_warnings=[],
                    should_verify=should_verify,
                    skip_reason=skip_reason,
                )
            )

        return self._validate_against_answer(answer, claims)


class LLMAtomicClaimExtractorV1(BaseClaimExtractor):
    """Primary structured extractor when an LLM client is configured."""

    def __init__(self, llm_client: object) -> None:
        self.llm_client = llm_client

    def extract_structured(self, answer: str) -> list[ExtractedClaim]:
        response = self._invoke(self._prompt(answer))
        try:
            parsed = self._parse_response(response)
        except Exception as exc:
            logger.warning("LLM claim extraction parse failed; requesting repair: %s", exc)
            repair_response = self._invoke(self._repair_prompt(answer, response))
            parsed = self._parse_response(repair_response)
            repaired = True
        else:
            repaired = False

        parsed_claims = self._coerce_claim_list(parsed)
        if parsed_claims is None:
            raise ValueError("Claim extractor response must be a JSON array or an object with a 'claims' array.")

        claims: list[ExtractedClaim] = []
        for item in parsed_claims:
            if not isinstance(item, dict):
                raise ValueError("Each extracted claim must be a JSON object.")
            claim_text = str(item.get("claim_text", "")).strip()
            if not claim_text:
                continue
            claim = ExtractedClaim(
                claim_id=str(item.get("claim_id") or f"claim_{uuid.uuid4().hex[:8]}"),
                claim_text=claim_text,
                source_sentence=str(item.get("source_sentence", "")).strip(),
                source_start_char=int(item.get("source_start_char", 0) or 0),
                source_end_char=int(item.get("source_end_char", 0) or 0),
                atomicity_status=_coerce_atomicity(item.get("atomicity_status")),
                claim_type=_coerce_claim_type(item.get("claim_type")),
                entities=_coerce_list(item.get("entities"), fallback=_extract_entities(claim_text)),
                dates=_coerce_list(item.get("dates"), fallback=_extract_dates(claim_text)),
                numbers=_coerce_list(item.get("numbers"), fallback=_extract_numbers(claim_text)),
                extraction_method="llm_atomic_claim_extractor_v1",
                extraction_reason=str(item.get("extraction_reason", "llm_atomic_decomposition")),
                extraction_confidence=None,
                extraction_warnings=_coerce_list(item.get("extraction_warnings"), fallback=[]),
                should_verify=bool(item.get("should_verify", True)),
                skip_reason=str(item.get("skip_reason")) if item.get("skip_reason") is not None else None,
            )
            if repaired:
                claim.extraction_warnings.append("json_repair_applied")
            claims.append(claim)
        return self._validate_against_answer(answer, claims)

    def _invoke(self, prompt: str) -> object:
        client = self.llm_client
        if hasattr(client, "chat"):
            return client.chat(prompt)  # type: ignore[union-attr]
        if hasattr(client, "complete"):
            return client.complete(prompt)  # type: ignore[union-attr]
        raise TypeError("llm_client must provide chat() or complete()")

    def _prompt(self, answer: str) -> str:
        return (
            "Split the answer into atomic, self-contained factual claims.\n"
            "Requirements:\n"
            "- Do not invent facts or use external knowledge.\n"
            "- Preserve the original source sentence for every claim.\n"
            "- Split compound statements into separate atomic claims.\n"
            "- Preserve dates, numbers, entities, qualifiers, conditions, and exceptions.\n"
            "- Mark rhetorical or unverifiable text with should_verify=false.\n"
            "- Return only strict JSON matching the schema below.\n\n"
            "Schema:\n"
            "{\n"
            '  "claims": [{\n'
            '    "claim_id": "string",\n'
            '    "claim_text": "string",\n'
            '    "source_sentence": "string",\n'
            '    "source_start_char": 0,\n'
            '    "source_end_char": 0,\n'
            '    "atomicity_status": "atomic|compound|unclear",\n'
            '    "claim_type": "numeric|temporal|policy_rule|eligibility|obligation|prohibition|causal|comparison|definition|entity_attribute|version_validity|other",\n'
            '    "entities": ["string"],\n'
            '    "dates": ["string"],\n'
            '    "numbers": ["string"],\n'
            '    "should_verify": true,\n'
            '    "skip_reason": null,\n'
            '    "extraction_reason": "string",\n'
            '    "extraction_warnings": []\n'
            "  }]\n"
            "}\n\n"
            "CRITICAL RULES:\n"
            "1. Every sentence in the answer MUST produce at least one separate claim entry.\n"
            "2. If a sentence begins with a pronoun (It, They, This, These, That, Those, He, She, We),\n"
            "   resolve the pronoun to its referent from the prior sentence before writing claim_text.\n"
            "   Example: 'The order applies to schools. It excludes private ones.' yields TWO claims:\n"
            "     claim 1 claim_text: 'The order applies to schools.'\n"
            "     claim 2 claim_text: 'The order excludes private schools.'  (pronoun resolved)\n"
            "3. NEVER merge two sentences into a single claim entry.\n"
            "4. claim_text must be a single, self-contained factual statement.\n\n"
            f"Answer:\n{answer}"
        )

    def _repair_prompt(self, answer: str, bad_response: object) -> str:
        return (
            "The previous response was malformed JSON. Repair it into valid strict JSON only.\n"
            "Do not add new claims or external facts.\n"
            f"Original answer:\n{answer}\n\n"
            f"Malformed response:\n{bad_response}"
        )

    def _parse_response(self, response: object) -> Any:
        if isinstance(response, dict):
            if "text" in response:
                response = response["text"]
            elif "content" in response:
                response = response["content"]
        if not isinstance(response, str):
            response = str(response)
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
        return json.loads(text)

    def _coerce_claim_list(self, parsed: object) -> list[object] | None:
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("claims", "extracted_claims", "items", "results"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return value
        return None


class ClaimExtractor(BaseClaimExtractor):
    """Facade preserving the old API while preferring LLM extraction."""

    def __init__(
        self,
        use_llm: bool = True,
        llm_client: object | None = None,
        extractor_mode: Literal["auto", "llm", "heuristic"] = "auto",
    ) -> None:
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.extractor_mode = extractor_mode

    def extract_structured(self, answer: str) -> list[ExtractedClaim]:
        prefer_llm = self.extractor_mode != "heuristic" and self.use_llm and self.llm_client is not None
        if prefer_llm:
            try:
                return LLMAtomicClaimExtractorV1(self.llm_client).extract_structured(answer)
            except Exception as exc:
                logger.warning("LLM claim extraction failed, falling back to heuristic: %s", exc)
                fallback_claims = HeuristicClaimExtractorV0().extract_structured(answer)
                for claim in fallback_claims:
                    claim.extraction_method = "llm_fallback"
                    claim.extraction_warnings.append(f"llm_extraction_failed:{type(exc).__name__}")
                return fallback_claims
        return HeuristicClaimExtractorV0().extract_structured(answer)


def _coerce_list(value: object, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return list(fallback)


def _coerce_atomicity(value: object) -> ClaimAtomicity:
    text = str(value or "unclear").strip().lower()
    if text in {"atomic", "compound", "unclear"}:
        return text  # type: ignore[return-value]
    return "unclear"


def _coerce_claim_type(value: object) -> ClaimType:
    text = str(value or "other").strip().lower().replace("-", "_")
    allowed = {
        "numeric",
        "temporal",
        "policy_rule",
        "eligibility",
        "obligation",
        "prohibition",
        "causal",
        "comparison",
        "definition",
        "entity_attribute",
        "version_validity",
        "other",
    }
    if text in allowed:
        return text  # type: ignore[return-value]
    return "other"


def _heuristic_claim_type(text: str) -> ClaimType:
    lowered = text.lower()
    if _NUMBER_RE.search(text):
        return "numeric"
    if _DATE_RE.search(text):
        return "temporal"
    if re.search(r"\bmust\b|\bshall\b", lowered):
        return "obligation"
    if re.search(r"\bmay not\b|\bmust not\b|\bprohibit", lowered):
        return "prohibition"
    if re.search(r"\beligib", lowered):
        return "eligibility"
    if re.search(r"\bcaus|because|due to|leads to\b", lowered):
        return "causal"
    if re.search(r"\bcompare|higher|lower|increase|decrease|grow|grew\b", lowered):
        return "comparison"
    if re.search(r"\bmeans\b|\bdefined\b|\brefers to\b", lowered):
        return "definition"
    if re.search(r"\bpolicy\b|\brule\b|\border\b|\bregulation\b|\bapplies\b", lowered):
        return "policy_rule"
    if re.search(r"\bversion\b|\bsupersed|\bdeprecated\b|\bwithdrawn\b", lowered):
        return "version_validity"
    if _extract_entities(text):
        return "entity_attribute"
    return "other"


def _extract_entities(text: str) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()
    for match in _ENTITY_RE.finditer(text):
        value = match.group(0).strip()
        normalized = _normalize_entity(value)
        if normalized and normalized not in seen and value not in _SHORT_ENTITY_PLACEHOLDERS:
            seen.add(normalized)
            entities.append(value)
    return entities


def _extract_dates(text: str) -> list[str]:
    seen: set[str] = set()
    dates: list[str] = []
    for match in _DATE_RE.finditer(text):
        value = match.group(0).strip()
        normalized = value.lower()
        if normalized not in seen:
            seen.add(normalized)
            dates.append(value)
    return dates


def _extract_numbers(text: str) -> list[str]:
    seen: set[str] = set()
    numbers: list[str] = []
    for match in _NUMBER_RE.finditer(text):
        value = match.group(0).strip()
        normalized = _normalize_number(value)
        if normalized not in seen:
            seen.add(normalized)
            numbers.append(value)
    return numbers


def _normalize_number(text: str) -> str:
    return re.sub(r"[\s,]", "", text.lower())


def _normalize_entity(text: str) -> str:
    return " ".join(text.lower().split())
