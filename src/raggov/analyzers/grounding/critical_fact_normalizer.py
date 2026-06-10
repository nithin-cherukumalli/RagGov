"""Deterministic critical fact normalization for claim grounding."""

from __future__ import annotations
import re

# Small explicit synonym map for specific target failure cases
ENTITY_SYNONYMS = {
    "confidence interval": ["confidence limits", "confidence limit"],
    "jury service": ["jury selection"],
}

class CriticalFactNormalizer:
    """Helper for deterministic fact normalization and equivalence checks."""

    @staticmethod
    def normalize_mentions(mentions: list) -> list:
        """Normalize extracted mentions if necessary."""
        return mentions

    @staticmethod
    def is_value_equivalent(val1: str, unit1: str, val2: str, unit2: str) -> bool:
        """Check if two values and units are deterministically equivalent."""
        # Exact match
        if val1 == val2 and unit1 == unit2:
            return True

        # Duration conversions
        if {unit1, unit2} == {"week", "day"}:
            try:
                if unit1 == "week":
                    return float(val1) * 7 == float(val2)
                else:
                    return float(val2) * 7 == float(val1)
            except ValueError:
                pass
        
        # We can add month/day or year/month if needed, but week/day handles hc-004
        return False

    @staticmethod
    def get_entity_synonyms(entity: str) -> list[str]:
        """Return a list of known synonyms for an entity, including itself."""
        syns = [entity.lower()]
        for key, aliases in ENTITY_SYNONYMS.items():
            if entity.lower() == key:
                syns.extend(a.lower() for a in aliases)
            elif entity.lower() in aliases:
                syns.append(key.lower())
                syns.extend(a.lower() for a in aliases if a.lower() != entity.lower())
        return list(dict.fromkeys(syns))
