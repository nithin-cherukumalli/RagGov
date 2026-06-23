# Heldout v1 — proposed gold labels (Kimi labeler)

- model: `llama-3.3-70b-versatile`  rows: 75
- agree with provisional: 0/75  disagree: 0  errored: 75
- worklist (human spot-check needed): 75

## Confusion: provisional -> proposed_gold
- **CLEAN** -> NEEDS_HUMAN:50
- **CONTRADICTED_CLAIM** -> NEEDS_HUMAN:25

## Next
1. Human-adjudicate the worklist (every disagreement + low-confidence + error).
2. Promote adjudicated labels to a locked gold heldout; re-run the scorer against it.
3. Only then are CLEAN-FP and failure-recall numbers measured against truth.
