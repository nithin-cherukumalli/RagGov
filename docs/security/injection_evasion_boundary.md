## Prompt Injection Evasion Boundary

`PromptInjectionAnalyzer` provides three detection layers:

1. Structural signature matching for imperative override, exfiltration, delimiter, jailbreak, and prompt-probing patterns.
2. Encoded-content inspection for base64 and URL-encoded payloads.
3. Semantic intent classification for paraphrased override, role-shift, and exfiltration language.

This improves detection coverage, but it is not a complete defense.

Known gaps:
- Highly context-specific paraphrasing with no stable verb-object structure.
- Multi-session or cross-turn attacks that are only visible over longer dialogue state.
- Adversarial prompts optimized against this detector's concepts or thresholds.

Operational guidance:
- Treat this analyzer as detection and alerting, not prevention.
- Keep evidence sanitized for untrusted audiences.
- Pair this analyzer with model-side hardening and grounded-response controls for production use.
