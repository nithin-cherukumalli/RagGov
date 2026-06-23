# Heldout v1 — two-labeler merge (adjudication-ready)

- rows compared: 75
- raw inter-annotator agreement: 0/75 = 0.00
- AGREED + both confident (provisional gold): 0/75
- REVIEW (human must adjudicate): 75

## Confusion: label_a x label_b
- **A=CLEAN** -> NEEDS_HUMAN:41
- **A=INSUFFICIENT_CONTEXT** -> NEEDS_HUMAN:5
- **A=UNSUPPORTED_CLAIM** -> NEEDS_HUMAN:29

## Next
1. Hand-adjudicate ONLY `heldout_real_v1_gold_review.jsonl` (the disagreements).
2. Merge your adjudications with `heldout_real_v1_gold_agreed.jsonl` -> locked gold heldout.
3. Re-run the scorer against the locked gold. Now the numbers mean something.
