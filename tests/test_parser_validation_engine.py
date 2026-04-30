from raggov.parser_validation.engine import ParserValidationEngine
from raggov.parser_validation.models import (
    ChunkIR,
    ParsedDocumentIR,
    ParserEvidence,
    ParserFailureType,
    ParserFinding,
    ParserSeverity,
    ParserValidationConfig,
)


class FakeValidator:
    name = "fake_validator"

    def validate(self, parsed_doc, chunks, config):
        return [
            ParserFinding(
                failure_type=ParserFailureType.METADATA_LOSS,
                severity=ParserSeverity.WARN,
                confidence=0.50,
                evidence=(ParserEvidence(message="warn"),),
                remediation="fix warn",
                validator_name=self.name,
            ),
            ParserFinding(
                failure_type=ParserFailureType.TABLE_STRUCTURE_LOSS,
                severity=ParserSeverity.FAIL,
                confidence=0.90,
                evidence=(ParserEvidence(message="fail"),),
                remediation="fix fail",
                validator_name=self.name,
            ),
        ]


def test_engine_runs_validators_and_sorts_findings():
    engine = ParserValidationEngine(validators=[FakeValidator()])
    findings = engine.validate(
        parsed_doc=ParsedDocumentIR(document_id="doc1"),
        chunks=[ChunkIR(chunk_id="c1", text="hello")],
    )

    assert len(findings) == 2
    assert findings[0].severity == ParserSeverity.FAIL
    assert findings[0].confidence == 0.90
    assert findings[1].severity == ParserSeverity.WARN


def test_engine_can_be_constructed_with_no_validators():
    engine = ParserValidationEngine(validators=[])
    findings = engine.validate(None, [ChunkIR(chunk_id="c1", text="hello")])

    assert findings == []
