# Substrate Order

## Order Of Trust

The project should be developed in this order:

1. claim extraction
2. evidence candidate selection
3. claim grounding
4. citation faithfulness
5. sufficiency
6. retrieval diagnosis
7. version validity
8. security detection
9. NCV
10. Layer6
11. A2P
12. semantic entropy / confidence
13. final decision policy

## Why This Order Matters

Higher layers depend on lower layers:

- citation faithfulness depends on claim evidence quality
- sufficiency is stronger when tied to claim/evidence structure
- retrieval diagnosis uses upstream reports
- NCV aggregates reports and falls back when reports are absent
- Layer6 maps existing failures to a taxonomy
- A2P explains failures using prior results
- decision policy chooses among candidate failures generated elsewhere

If lower layers are weak, upper layers become polished uncertainty.

## Practical Rule

Do not prioritize meta-layer sophistication before substrate hardening.
