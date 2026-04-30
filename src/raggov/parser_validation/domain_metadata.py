"""Domain metadata governance kept separate from parser structural validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DomainMetadataRule:
    """One domain metadata governance rule."""

    field: str
    required: bool = False
    allowed_values: tuple[Any, ...] = ()


@dataclass(frozen=True)
class DomainMetadataIssue:
    """One domain metadata governance issue."""

    code: str
    message: str
    field: str
    severity: str = "error"
    document_id: str | None = None
    chunk_id: str | None = None
    observed: Any | None = None
    expected: Any | None = None


@dataclass(frozen=True)
class DomainMetadataReport:
    """Result from domain metadata governance checks."""

    issues: tuple[DomainMetadataIssue, ...] = field(default_factory=tuple)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)


class DomainMetadataGovernanceEngine:
    """Validate domain-specific metadata without coupling to parser validators."""

    def __init__(self, rules: tuple[DomainMetadataRule, ...] = ()) -> None:
        self.rules = rules

    def validate(
        self,
        domain_metadata: dict[str, Any],
        *,
        document_id: str | None = None,
        chunk_id: str | None = None,
    ) -> DomainMetadataReport:
        issues: list[DomainMetadataIssue] = []

        for rule in self.rules:
            value = domain_metadata.get(rule.field)

            if rule.required and value is None:
                issues.append(
                    DomainMetadataIssue(
                        code="domain_field_missing",
                        message=f"Required domain metadata field is missing: {rule.field}",
                        field=rule.field,
                        document_id=document_id,
                        chunk_id=chunk_id,
                        expected="present",
                        observed=None,
                    )
                )
                continue

            if value is not None and rule.allowed_values and value not in rule.allowed_values:
                issues.append(
                    DomainMetadataIssue(
                        code="domain_field_invalid",
                        message=f"Domain metadata field has an unexpected value: {rule.field}",
                        field=rule.field,
                        document_id=document_id,
                        chunk_id=chunk_id,
                        expected=rule.allowed_values,
                        observed=value,
                    )
                )

        return DomainMetadataReport(issues=tuple(issues))
