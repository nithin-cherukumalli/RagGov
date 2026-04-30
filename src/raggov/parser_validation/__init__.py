"""Parser-validation interfaces for parser-agnostic document IR checks."""

from raggov.parser_validation.engine import ParserValidationEngine
from raggov.parser_validation.ingestion import (
    DocumentParserQualitySummary,
    IngestionValidationReport,
    IngestionValidationRequest,
    validate_ingestion,
)
from raggov.parser_validation.metadata_normalizer import (
    MetadataNormalizer,
    NormalizedMetadata,
)
from raggov.parser_validation.models import (
    ChunkIR,
    ChunkingStrategyProfile,
    ChunkingStrategyType,
    ElementIR,
    LayoutBox,
    ParsedDocumentIR,
    ParserEvidence,
    ParserFailureType,
    ParserFinding,
    ParserSeverity,
    ParserValidationConfig,
    TableIR,
    default_chunking_profile,
)
from raggov.parser_validation.profile import (
    CanonicalMetadataMapping,
    MetadataFieldMapping,
    ParserProfile,
    ParserValidationProfile,
    ParserValidationProfileSet,
)
from raggov.parser_validation.profile_lint import (
    PROFILE_LINT_REMEDIATION,
    ProfileLintEngine,
    ProfileLintIssue,
    ProfileLintReport,
)

__all__ = [
    "ParserValidationEngine",
    "CanonicalMetadataMapping",
    "ChunkIR",
    "ChunkingStrategyProfile",
    "ChunkingStrategyType",
    "DocumentParserQualitySummary",
    "ElementIR",
    "IngestionValidationReport",
    "IngestionValidationRequest",
    "LayoutBox",
    "MetadataNormalizer",
    "MetadataFieldMapping",
    "NormalizedMetadata",
    "ParsedDocumentIR",
    "ParserEvidence",
    "ParserFailureType",
    "ParserFinding",
    "ParserProfile",
    "ParserSeverity",
    "ParserValidationProfile",
    "ParserValidationProfileSet",
    "ParserValidationConfig",
    "PROFILE_LINT_REMEDIATION",
    "ProfileLintEngine",
    "ProfileLintIssue",
    "ProfileLintReport",
    "TableIR",
    "default_chunking_profile",
    "validate_ingestion",
]
