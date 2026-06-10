# PR Protocol

Every future PR must be bounded.

Each PR must define:

1. Problem
2. Substrate affected
3. Files allowed to change
4. Files forbidden to change
5. Expected behavior
6. Tests required
7. Evaluation command
8. Rollback plan
9. Risks
10. Limitations
11. Wiki files to update
12. Context affected

## Additional GovRAG Rules

- Prefer one substrate per PR.
- Do not mix substrate hardening with decision policy changes.
- Do not combine documentation, analyzer logic, and benchmark chasing into one PR.
- New analyzers are forbidden unless explicitly approved.
- Do not frame external-enhanced improvements as progress on the primary product path unless native mode improved too.
- If the PR changes meaning, not just code, it must update the wiki.
- If the PR changes terminology, it must update the glossary or registry files.

## Special Rule For Policy Changes

Any change to `src/raggov/decision_policy.py` requires a separate plan PR first.

That plan PR must explain:

- why substrate evidence is insufficient
- why the change belongs in policy rather than analyzer logic
- why the change is not benchmark-specific patching
- what regressions are expected

## Obsolescence Prevention Rule

The wiki must not be allowed to become obsolete.

For every meaningful PR, reviewers should ask:

1. Did architecture meaning change?
2. Did fallback behavior change?
3. Did trust semantics change?
4. Did the operator mental model change?
5. Which `llm_wiki/` files were updated to reflect that?
