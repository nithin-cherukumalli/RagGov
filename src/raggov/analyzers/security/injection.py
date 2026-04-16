"""Analyzer for prompt injection and instruction override risks.

Implements dual-channel structural detection based on:
- Greshake et al., "Not What You've Signed Up For" (AISec@CCS 2023)
- StruQ (Chen et al., UC Berkeley, USENIX Security 2025)

Attack taxonomy:
1. Imperative directives (Greshake: task injection, persona injection)
   - role_reassignment: AI-directed identity/role changes
   - task_override: Instructions to ignore/bypass system prompts
   - system_context: System/developer/admin mode manipulation

2. Exfiltration structure (Greshake: data theft)
   - credential_exfiltration: Verb-object-destination pattern for sensitive data
   - prompt_exfiltration: Instructions to reveal system configuration
   - indirect_exfiltration: Indirect data inclusion requests

3. Delimiter injection (StruQ: structural delimiter attacks)
   - special_token: Model control tokens in data channel (<|im_start|>, etc.)
   - llama_delimiter: LLaMA-style instruction markers ([INST], [/INST])
   - chat_delimiter: Chat template role markers (###System, ###Human, etc.)
   - xml_role_tag: XML-style role tags (<system>, <user>, etc.)

4. Jailbreak templates (Known attack frameworks)
   - known_jailbreak: DAN, AIM, UCAR, DevMode patterns
   - jailbreak_greeting: Jailbreak activation phrases
   - jailbreak_roleplay: Unrestricted mode requests

5. Prompt probing (Indirect prompt leakage)
   - prompt_probing: Questions about constraints, guidelines, instructions
"""

from __future__ import annotations

import base64
import logging
import re
import time
import urllib.parse
from typing import Any

from raggov.analyzers.base import BaseAnalyzer
from raggov.models.diagnosis import (
    AnalyzerResult,
    FailureStage,
    FailureType,
    SecurityRisk,
)
from raggov.models.run import RAGRun

logger = logging.getLogger(__name__)


# Detector 1: Imperative directive detection (Greshake taxonomy: task/persona injection)
# Checks for imperative verbs DIRECTED AT an AI system
IMPERATIVE_DIRECTIVES = [
    # Role/identity reassignment (Greshake: "persona injection")
    # Requires AI-directed context to avoid false positives like "you are now eligible"
    (r"(?i)\b(you are|you're)\s+(now\s+)?(an?|the)?\s*(AI|assistant|chatbot|bot|helper|agent|model|system|unrestricted|DAN|jailbreak)", "role_reassignment"),
    (r"(?i)\b(act as|pretend (to be|you are)|roleplay as)\b", "role_reassignment"),
    # Leetspeak and variations
    (r"(?i)\b(u r|u are|you r|ur)\s+(now|currently)?\s+.{0,30}\b(AI|assistant|chatbot|bot|agent|model)", "role_reassignment"),
    (r"(?i)\b(consider yourself|assume you are|pretend that you are)\s+.{0,30}\b(AI|assistant|chatbot|bot|agent|model)", "role_reassignment"),
    # Task override (Greshake: "task injection")
    (r"(?i)\b(ignore|disregard|forget|override|bypass|skip)\b.{0,40}\b(instruction|rule|guideline|prompt|context|above|previous|prior)\b", "task_override"),
    # System context manipulation
    (r"(?i)\b(system|developer|admin|operator)\s*(prompt|mode|instruction|override)\b", "system_context"),
    (r"(?i)\b(enable|activate|switch to|enter)\b.{0,20}\b(debug|privileged|root|admin|unrestricted|god)\s*(mode|access|context)\b", "system_context"),
]

# Detector 2: Exfiltration structure (Greshake taxonomy: data theft)
# Checks for: data-reference + transmission-verb + external-target
EXFILTRATION_PATTERNS = [
    # Verb-object-destination structure (direct)
    (r"(?i)\b(send|forward|transmit|POST|email|leak|export)\b.{0,50}\b(password|secret|key|token|api.?key|credential)\b", "credential_exfiltration"),
    (r"(?i)\b(print|output|display|repeat|echo|return)\b.{0,30}\b(system prompt|instruction|configuration)\b", "prompt_exfiltration"),
    # Indirect data exfiltration
    (r"(?i)\b(include|append|add|attach|incorporate)\b.{0,50}\b(password|secret|key|token|api.?key|credential)\b.{0,30}\b(response|answer|message|output)\b", "indirect_exfiltration"),
    (r"(?i)\byour (next )?(?:message|response|answer) should (contain|include).{0,50}\b(password|secret|key|token|api.?key)\b", "indirect_exfiltration"),
]

# Detector 3: Delimiter injection (StruQ: structural delimiter attacks)
# Checks for prompt-channel structural markers appearing in data channel
DELIMITER_PATTERNS = [
    (r"<\|.{1,20}\|>", "special_token"),  # <|im_start|>, <|endoftext|>
    (r"\[INST\]|\[/INST\]|\[SYS\]", "llama_delimiter"),
    (r"###\s*(System|Human|Assistant|Instruction)\b", "chat_delimiter"),
    (r"(?i)<(system|user|assistant|instruction)>", "xml_role_tag"),
]

# Detector 4: Jailbreak templates (Known attack frameworks)
JAILBREAK_PATTERNS = [
    (r"(?i)\b(DAN|AIM|UCAR|DevMode|Developer Mode|DUDE)\b.{0,30}\b(activated|enabled|mode|version)\b", "known_jailbreak"),
    (r"(?i)(Hello ChatGPT|Hi ChatGPT).{0,50}(DAN mode|developer mode|jailbreak)", "jailbreak_greeting"),
    (r"(?i)pretend.{0,30}(unfiltered|unrestricted|no rules|no guidelines)", "jailbreak_roleplay"),
]

# Detector 5: Prompt probing (Indirect prompt leakage)
PROMPT_PROBING_PATTERNS = [
    (r"(?i)what (constraints|restrictions|limitations|guidelines|rules).{0,30}(operating under|following|have)", "prompt_probing"),
    (r"(?i)(summarize|describe|explain).{0,30}(your instructions|your guidelines|your system prompt)", "prompt_probing"),
    (r"(?i)what were you (told|instructed|programmed).{0,30}(not to|to avoid)", "prompt_probing"),
]

# Public-facing remediation (generic, doesn't expose detection methodology)
REMEDIATION_PUBLIC = (
    "Retrieved content contains patterns consistent with prompt injection. "
    "Content has been blocked for security review."
)

# Internal remediation (detailed, for security team)
REMEDIATION_INTERNAL = (
    "Retrieved chunk(s) contain instruction-like content consistent with prompt "
    "injection (structural attack classes detected: Greshake task/persona injection, "
    "data exfiltration, or StruQ delimiter injection). Sanitize corpus or add a "
    "pre-retrieval content filter."
)


class PromptInjectionAnalyzer(BaseAnalyzer):
    """Scan retrieved chunks for prompt injection signals using structural detection.

    Configuration:
        risk_threshold (int): Number of matches to trigger HIGH risk (default: 1)
            Valid range: 1-100
            Recommended: 1 for imperative directives and exfiltration patterns

        warn_threshold (int): Number of matches to trigger LOW risk (default: 1)
            Valid range: 1-100
            Must be <= risk_threshold
            Recommended: 1 for delimiter patterns

        max_chunk_length (int): Maximum chunk length to process (default: 50000)
            Chunks exceeding this length are logged and skipped

        enable_multi_chunk_detection (bool): Enable cross-chunk analysis (default: True)
            When True, analyzes combined text of first 5 chunks for split attacks

        enable_encoding_detection (bool): Enable base64/URL encoding detection (default: True)
            When True, decodes and scans base64 and URL-encoded content

        max_execution_time (float): Maximum execution time in seconds (default: 5.0)
            Prevents DoS via expensive regex matching

        sanitize_evidence (bool): Sanitize evidence to prevent attack pattern disclosure (default: True)
            When True, evidence contains only generic messages for security

    Threshold justification:
        - Imperative directives: risk_threshold=1 because any AI-directed command in a
          retrieved chunk is suspicious (legitimate docs don't command the reader).
        - Exfiltration patterns: risk_threshold=1 because verb+object+destination
          structure is highly specific to data theft attempts.
        - Delimiter patterns: warn_threshold=1 because delimiters could appear in
          legitimate technical documentation. Context-aware detection reduces false positives.
    """

    weight = 1.0

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)

        # Validate and set thresholds
        self.risk_threshold = self._validate_threshold(
            "risk_threshold", default=1, min_val=1, max_val=100
        )
        self.warn_threshold = self._validate_threshold(
            "warn_threshold", default=1, min_val=1, max_val=100
        )

        # Ensure logical threshold relationship
        if self.warn_threshold > self.risk_threshold:
            raise ValueError(
                f"warn_threshold ({self.warn_threshold}) cannot exceed "
                f"risk_threshold ({self.risk_threshold})"
            )

        # Other configuration
        self.max_chunk_length = int(self.config.get("max_chunk_length", 50000))
        self.enable_multi_chunk = bool(self.config.get("enable_multi_chunk_detection", True))
        self.enable_encoding = bool(self.config.get("enable_encoding_detection", True))
        self.max_execution_time = float(self.config.get("max_execution_time", 5.0))
        self.sanitize_evidence = bool(self.config.get("sanitize_evidence", True))

        # Compile patterns for performance (do this once at initialization)
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), label)
            for pattern, label in (
                IMPERATIVE_DIRECTIVES
                + EXFILTRATION_PATTERNS
                + DELIMITER_PATTERNS
                + JAILBREAK_PATTERNS
                + PROMPT_PROBING_PATTERNS
            )
        ]

    def _validate_threshold(
        self, key: str, default: int, min_val: int, max_val: int
    ) -> int:
        """Validate and return a threshold value."""
        value = self.config.get(key, default)
        try:
            value = int(value)
        except (ValueError, TypeError):
            raise ValueError(f"{key} must be an integer, got {type(value).__name__}")

        if not min_val <= value <= max_val:
            raise ValueError(
                f"{key} must be between {min_val} and {max_val}, got {value}"
            )

        return value

    def _is_documentation_context(self, text: str, match_start: int) -> bool:
        """Check if match appears in a documentation/code context."""
        # Check if inside markdown code block (between opening and closing ```)
        preceding_text = text[:match_start]
        following_text = text[match_start:]

        # Count ``` occurrences before and after the match
        backtick_count_before = preceding_text.count("```")
        backtick_count_after = following_text.count("```")

        # If odd number of ``` before match and at least one after, we're inside a code block
        if backtick_count_before % 2 == 1 and backtick_count_after > 0:
            return True

        # Check for indented code (4 spaces at start of line)
        # Find the start of the line containing the match
        line_start = preceding_text.rfind('\n') + 1
        line_prefix = preceding_text[line_start:]
        if line_prefix.startswith("    "):
            return True

        # Check for documentation signal phrases (not just keywords)
        # These are phrases that indicate technical documentation about tokens/delimiters
        context_window = text[max(0, match_start - 50) : match_start + 50].lower()
        doc_phrases = [
            "uses the",
            "the model uses",
            "delimiter is",
            "token is",
            "example:",
            "syntax:",
            "format:",
            "mark sequence",
            "represents a",
        ]
        if any(phrase in context_window for phrase in doc_phrases):
            return True

        return False

    def _detect_encoded_injection(self, text: str) -> list[str]:
        """Detect base64 and URL-encoded injection attempts."""
        evidence = []

        # Base64 detection (look for long base64 strings)
        b64_pattern = r"[A-Za-z0-9+/]{20,}={0,2}"
        for match in re.finditer(b64_pattern, text):
            try:
                decoded = base64.b64decode(match.group()).decode("utf-8", errors="ignore")
                # Check decoded content for injection keywords
                injection_keywords = [
                    "ignore",
                    "system prompt",
                    "api key",
                    "password",
                    "secret",
                    "token",
                    "override",
                    "bypass",
                ]
                if any(kw in decoded.lower() for kw in injection_keywords):
                    evidence.append(f"base64_encoded_injection: {match.group()[:30]}...")
                    logger.warning(
                        "Base64-encoded injection detected",
                        extra={"encoded": match.group()[:50], "decoded": decoded[:100]},
                    )
            except Exception:
                pass

        # URL encoding detection
        if "%" in text and re.search(r"%[0-9A-Fa-f]{2}", text):
            try:
                decoded = urllib.parse.unquote(text)
                if decoded != text:  # Was actually encoded
                    # Re-run pattern matching on decoded text
                    for compiled_pattern, attack_class in self.compiled_patterns:
                        if compiled_pattern.search(decoded):
                            evidence.append(f"url_encoded_{attack_class}")
                            logger.warning(
                                "URL-encoded injection detected",
                                extra={"attack_class": attack_class, "decoded": decoded[:100]},
                            )
                            break
            except Exception:
                pass

        return evidence

    def _detect_multi_chunk_injection(self, chunks: list) -> list[str]:
        """Detect injection patterns split across multiple chunks."""
        if len(chunks) < 2:
            return []

        # Analyze combined text of first 5 chunks
        combined_text = " ".join(chunk.text for chunk in chunks[:5])
        evidence = []

        for compiled_pattern, attack_class in self.compiled_patterns:
            if compiled_pattern.search(combined_text):
                # Find which chunks contributed
                contributing = []
                keywords = [
                    "ignore",
                    "system",
                    "reveal",
                    "send",
                    "password",
                    "instruction",
                    "override",
                ]
                for chunk in chunks[:5]:
                    if any(kw in chunk.text.lower() for kw in keywords):
                        contributing.append(chunk.chunk_id)

                if len(contributing) > 1:
                    evidence.append(f"multi_chunk_{attack_class}: {', '.join(contributing)}")
                    logger.warning(
                        "Multi-chunk injection detected",
                        extra={
                            "attack_class": attack_class,
                            "contributing_chunks": contributing,
                        },
                    )

        return evidence

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        if not run.retrieved_chunks:
            return self.skip("no retrieved chunks available")

        start_time = time.time()
        detailed_evidence: list[str] = []
        total_hits = 0

        # Multi-chunk detection (if enabled)
        if self.enable_multi_chunk:
            multi_chunk_evidence = self._detect_multi_chunk_injection(run.retrieved_chunks)
            detailed_evidence.extend(multi_chunk_evidence)
            total_hits += len(multi_chunk_evidence)

        # Per-chunk analysis
        for chunk in run.retrieved_chunks:
            # Check execution timeout
            if time.time() - start_time > self.max_execution_time:
                logger.warning(
                    "Prompt injection analysis timeout",
                    extra={"elapsed_time": time.time() - start_time},
                )
                return self.skip(f"analysis timeout after {self.max_execution_time}s")

            # Skip excessively long chunks
            if len(chunk.text) > self.max_chunk_length:
                logger.warning(
                    "Chunk exceeds max length, skipping",
                    extra={"chunk_id": chunk.chunk_id, "length": len(chunk.text)},
                )
                continue

            chunk_matches: list[str] = []

            # Encoding detection (if enabled)
            if self.enable_encoding:
                encoding_matches = self._detect_encoded_injection(chunk.text)
                chunk_matches.extend(encoding_matches)

            # Pattern matching with context-aware detection
            for compiled_pattern, attack_class in self.compiled_patterns:
                match = compiled_pattern.search(chunk.text)
                if match:
                    # Context-aware filtering: check if match is in code/documentation context
                    # This applies to ALL patterns, not just delimiters, since any injection
                    # pattern in a code block is likely benign (code examples, documentation)
                    if self._is_documentation_context(chunk.text, match.start()):
                        logger.debug(
                            "Pattern in documentation/code context, skipping",
                            extra={
                                "chunk_id": chunk.chunk_id,
                                "attack_class": attack_class,
                            },
                        )
                        continue

                    chunk_matches.append(attack_class)

            if chunk_matches:
                total_hits += len(chunk_matches)
                detailed_evidence.append(
                    f"{chunk.chunk_id}: {len(chunk_matches)} hit(s): {', '.join(chunk_matches)}"
                )

        if total_hits == 0:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="pass",
                security_risk=SecurityRisk.NONE,
            )

        # Determine security level
        if total_hits >= self.risk_threshold:
            security_risk = SecurityRisk.HIGH
            status = "fail"
        elif total_hits >= self.warn_threshold:
            security_risk = SecurityRisk.LOW
            status = "warn"
        else:
            return AnalyzerResult(
                analyzer_name=self.name(),
                status="pass",
                security_risk=SecurityRisk.NONE,
            )

        # Sanitize evidence if configured (prevent attack pattern disclosure)
        if self.sanitize_evidence:
            sanitized_evidence = [
                f"Detected {total_hits} potential injection pattern(s) across {len(detailed_evidence)} chunk(s)"
            ]
        else:
            sanitized_evidence = detailed_evidence

        # Structured logging for security monitoring
        logger.warning(
            "Prompt injection detected",
            extra={
                "total_hits": total_hits,
                "matched_patterns": list(
                    set([match for evidence in detailed_evidence for match in evidence.split(": ")[1].split(", ") if ": " in evidence])
                ),
                "affected_chunks": len([e for e in detailed_evidence if ":" in e]),
                "security_risk": security_risk.value,
            },
        )

        return AnalyzerResult(
            analyzer_name=self.name(),
            status=status,
            failure_type=FailureType.PROMPT_INJECTION,
            stage=FailureStage.SECURITY,
            security_risk=security_risk,
            evidence=sanitized_evidence,
            remediation=REMEDIATION_PUBLIC if self.sanitize_evidence else REMEDIATION_INTERNAL,
        )
