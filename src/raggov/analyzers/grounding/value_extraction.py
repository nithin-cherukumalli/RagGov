"""Deterministic value extraction helpers for claim grounding."""

from __future__ import annotations

import re
from dataclasses import dataclass


_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
}

_NUMBER_WORD_PATTERN = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|and)"

_TIME_UNITS = frozenset({"year", "month", "day", "hour"})

_ATTRIBUTE_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "for", "to", "of",
    "in", "on", "at", "by", "with", "and", "or", "not", "no",
    "from", "this", "that", "than", "then", "when", "where",
    "which", "also", "only", "just", "some", "more", "each",
    "such", "its", "per",
})


@dataclass(frozen=True)
class ValueMention:
    """Normalized value mention extracted from text."""

    raw_text: str
    normalized_value: str
    value_type: str
    unit: str
    start: int
    end: int
    context: str


def extract_value_mentions(text: str) -> list[ValueMention]:
    """Extract numeric and policy-value mentions from text."""
    lowered = text.lower()
    mentions: list[ValueMention] = []

    for match in re.finditer(r"\bg\.?\s*o\.?\s*(?:rt|ms)?\.?\s*no\.?\s*[:.]?\s*(\d+)\b", lowered):
        go_number = match.group(1)
        mentions.append(
            _build_mention(text, match.start(), match.end(), f"go:{go_number}", "go_number", "go")
        )

    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*%\b", lowered):
        value = match.group(1)
        mentions.append(_build_mention(text, match.start(), match.end(), value, "percentage", "percent"))
    for match in re.finditer(rf"\b({_NUMBER_WORD_PATTERN}(?:[\s-]+{_NUMBER_WORD_PATTERN})*)\s+percent\b", lowered):
        number = _words_to_number(match.group(1))
        if number is not None:
            mentions.append(
                _build_mention(text, match.start(), match.end(), _normalize_number(number), "percentage", "percent")
            )

    for match in re.finditer(r"(?:[$₹€£]\s*\d[\d,]*(?:\.\d+)?)\b", text):
        numeric = _parse_numeric_token(re.sub(r"[$₹€£\s]", "", match.group(0)))
        if numeric is not None:
            mentions.append(
                _build_mention(text, match.start(), match.end(), _normalize_number(numeric), "money", "currency")
            )
    for match in re.finditer(r"\b(?:rs\.?|rupees?|dollars?)\s*(\d[\d,]*(?:\.\d+)?)\b", lowered):
        numeric = _parse_numeric_token(match.group(1))
        if numeric is not None:
            mentions.append(
                _build_mention(text, match.start(), match.end(), _normalize_number(numeric), "money", "currency")
            )
    for match in re.finditer(
        rf"\b({_NUMBER_WORD_PATTERN}(?:[\s-]+{_NUMBER_WORD_PATTERN})*)\s+(dollars?|rupees?)\b",
        lowered,
    ):
        number = _words_to_number(match.group(1))
        if number is not None:
            mentions.append(
                _build_mention(text, match.start(), match.end(), _normalize_number(number), "money", "currency")
            )

    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s+(days?|months?|years?|hours?)\b", lowered):
        unit = _normalize_unit(match.group(2))
        mentions.append(
            _build_mention(text, match.start(), match.end(), match.group(1), "duration", unit)
        )
    for match in re.finditer(
        rf"\b({_NUMBER_WORD_PATTERN}(?:[\s-]+{_NUMBER_WORD_PATTERN})*)\s+(days?|months?|years?|hours?)\b",
        lowered,
    ):
        number = _words_to_number(match.group(1))
        if number is not None:
            unit = _normalize_unit(match.group(2))
            mentions.append(
                _build_mention(
                    text,
                    match.start(),
                    match.end(),
                    _normalize_number(number),
                    "duration",
                    unit,
                )
            )

    for match in re.finditer(r"\b(" + "|".join(_MONTHS.keys()) + r")\s+(\d{1,2})(?:,\s*(\d{4}))?\b", lowered):
        month = _MONTHS[match.group(1)]
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else 0
        mentions.append(
            _build_mention(
                text,
                match.start(),
                match.end(),
                f"{year:04d}-{month:02d}-{day:02d}",
                "date",
                "date",
            )
        )
    for match in re.finditer(r"\b(\d{1,2})\s+(" + "|".join(_MONTHS.keys()) + r")(?:\s+(\d{4}))?\b", lowered):
        day = int(match.group(1))
        month = _MONTHS[match.group(2)]
        year = int(match.group(3)) if match.group(3) else 0
        mentions.append(
            _build_mention(
                text,
                match.start(),
                match.end(),
                f"{year:04d}-{month:02d}-{day:02d}",
                "date",
                "date",
            )
        )

    for match in re.finditer(r"\b\d[\d,]*(?:\.\d+)?\b", lowered):
        numeric = _parse_numeric_token(match.group(0))
        if numeric is not None:
            mentions.append(
                _build_mention(
                    text,
                    match.start(),
                    match.end(),
                    _normalize_number(numeric),
                    "number",
                    "number",
                )
            )
    for match in re.finditer(rf"\b({_NUMBER_WORD_PATTERN}(?:[\s-]+{_NUMBER_WORD_PATTERN})+)\b", lowered):
        number = _words_to_number(match.group(1))
        if number is not None:
            mentions.append(
                _build_mention(
                    text,
                    match.start(),
                    match.end(),
                    _normalize_number(number),
                    "number",
                    "number",
                )
            )

    return _dedupe_mentions(mentions)


def find_value_alignment(
    claim_text: str,
    evidence_text: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], bool]:
    """Compare claim/evidence values and return matches, conflicts, and missing-critical flag."""
    claim_values = extract_value_mentions(claim_text)
    evidence_values = extract_value_mentions(evidence_text)
    matches: list[dict[str, str]] = []
    conflicts: list[dict[str, str]] = []
    missing_critical = False

    if not claim_values:
        return matches, conflicts, False

    for claim_value in claim_values:
        candidates = [
            evidence_value
            for evidence_value in evidence_values
            if _compatible_types(claim_value, evidence_value)
            and _context_overlap(claim_value, evidence_value)
        ]
        if not candidates:
            if claim_value.value_type in {"money", "duration", "percentage", "date", "go_number"}:
                missing_critical = True
            continue

        found_match = False
        for candidate in candidates:
            if _values_match(claim_value, candidate):
                found_match = True
                matches.append(
                    {
                        "claim_value": claim_value.raw_text,
                        "evidence_value": candidate.raw_text,
                        "value_type": claim_value.value_type,
                        "unit": claim_value.unit,
                    }
                )
                break
        if not found_match:
            chosen = candidates[0]
            conflicts.append(
                {
                    "claim_value": claim_value.raw_text,
                    "evidence_value": chosen.raw_text,
                    "value_type": claim_value.value_type,
                    "unit": claim_value.unit,
                    "reason": "conflicting value in similar context",
                }
            )

    return matches, conflicts, missing_critical


def _build_mention(
    text: str,
    start: int,
    end: int,
    normalized_value: str,
    value_type: str,
    unit: str,
) -> ValueMention:
    return ValueMention(
        raw_text=text[start:end],
        normalized_value=normalized_value,
        value_type=value_type,
        unit=unit,
        start=start,
        end=end,
        context=text[max(0, start - 25) : min(len(text), end + 25)].lower(),
    )


def _dedupe_mentions(mentions: list[ValueMention]) -> list[ValueMention]:
    unique: list[ValueMention] = []
    seen: set[tuple[str, str, str, int, int]] = set()
    for mention in mentions:
        key = (
            mention.normalized_value,
            mention.value_type,
            mention.unit,
            mention.start,
            mention.end,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(mention)
    return unique


def _normalize_unit(unit: str) -> str:
    lowered = unit.lower()
    if lowered.startswith("day"):
        return "day"
    if lowered.startswith("month"):
        return "month"
    if lowered.startswith("year"):
        return "year"
    if lowered.startswith("hour"):
        return "hour"
    return lowered


def _parse_numeric_token(token: str) -> float | None:
    cleaned = token.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalize_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _words_to_number(text: str) -> float | None:
    parts = re.split(r"[\s-]+", text.strip().lower())
    if not parts:
        return None
    total = 0
    current = 0
    seen_number = False
    for part in parts:
        if part == "and":
            continue
        if part not in _NUMBER_WORDS:
            return None
        seen_number = True
        value = _NUMBER_WORDS[part]
        if value == 100:
            current = max(1, current) * 100
        elif value == 1000:
            current = max(1, current) * 1000
            total += current
            current = 0
        else:
            current += value
    if not seen_number:
        return None
    return float(total + current)


def _compatible_types(claim_value: ValueMention, evidence_value: ValueMention) -> bool:
    if claim_value.value_type == evidence_value.value_type:
        if claim_value.value_type == "duration":
            return claim_value.unit == evidence_value.unit
        return True
    if {claim_value.value_type, evidence_value.value_type} == {"number", "money"}:
        return True
    if {claim_value.value_type, evidence_value.value_type} == {"number", "duration"}:
        return claim_value.unit == "number" or evidence_value.unit == "number"
    if {claim_value.value_type, evidence_value.value_type} == {"number", "percentage"}:
        return True
    return False


def _values_match(claim_value: ValueMention, evidence_value: ValueMention) -> bool:
    if not _compatible_types(claim_value, evidence_value):
        return False

    if claim_value.normalized_value != evidence_value.normalized_value:
        return False

    if claim_value.value_type == evidence_value.value_type == "duration":
        return claim_value.unit == evidence_value.unit

    if "number" not in {claim_value.value_type, evidence_value.value_type}:
        return claim_value.unit == evidence_value.unit

    return True


def _context_overlap(claim_context: ValueMention | str, evidence_context: ValueMention | str) -> bool:
    claim_context_text = claim_context.context if isinstance(claim_context, ValueMention) else claim_context
    evidence_context_text = (
        evidence_context.context if isinstance(evidence_context, ValueMention) else evidence_context
    )
    claim_tokens = {
        token
        for token in re.findall(r"[a-z]+", claim_context_text)
        if len(token) > 3 and token not in _NUMBER_WORDS
    }
    evidence_tokens = {
        token
        for token in re.findall(r"[a-z]+", evidence_context_text)
        if len(token) > 3 and token not in _NUMBER_WORDS
    }
    if not claim_tokens or not evidence_tokens:
        return True
    overlap = claim_tokens & evidence_tokens
    return len(overlap) >= 1
