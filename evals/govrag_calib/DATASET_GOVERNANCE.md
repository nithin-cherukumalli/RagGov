# Calibration Dataset Governance

**Canonical file:** `evals/govrag_calib/govrag_calib_150.jsonl`
**Version:** `dataset_v1` (frozen 2026-06-17)
**Lock manifest:** `DATASET_MANIFEST.json` — enforced by `scripts/check_dataset_lock.py`

This file is the single source of truth for calibration/heldout evaluation. The
rules below exist because golden labels were silently relabeled/renumbered between
2026-06-15 and 2026-06-17, which invalidated prior accuracy numbers and broke the
v2 task queue (see `reports/codex_session/v2_feasibility_blocker.md`). Never again.

## Hard rules

1. **Case IDs are immutable.** A `gc-0NN` must refer to the same case forever.
   Never renumber. Never reuse a retired ID. New cases get new, higher IDs.
2. **No silent label changes.** Any change to an `expected_*` field requires:
   (a) a dated entry in `LABEL_CHANGELOG.md` with rationale and adjudicator, and
   (b) regenerating the manifest: `python scripts/check_dataset_lock.py --regenerate`.
3. **No tuning labels to make code pass.** If a label looks wrong, file a review
   note for human adjudication — do not edit it to match analyzer output. (Mirrors
   the project Prime Directive.)
4. **Placeholders are never scored.** Cases marked `TODO`/`placeholder` must stay in
   `split=unset` until filled with a real case. Filling one is a label change → rule 2.
5. **The lock check runs in CI / before any eval claim.** A failing
   `check_dataset_lock.py` means the dataset drifted without a record — stop and
   reconcile before trusting any number.

## Splits

- `train`, `dev`, `heldout` are **scored**.
- `unset` is **not scored** (holds placeholders and not-yet-assigned cases).
- The `heldout` split must be frozen and only rotated by a deliberate, logged
  decision — never edited in place to chase a number.

## Adding or changing a case (checklist)

1. Append the new case with the next free `gc-0NN` id (or edit in place only for a
   genuine correction).
2. Add a `LABEL_CHANGELOG.md` entry: date, case id(s), field, old→new, why, who.
3. Run `python scripts/check_dataset_lock.py --regenerate`.
4. Re-run the benchmark and record the new accuracy with its sample size.

## Why so strict

The whole value of this project is *trustworthy* diagnosis. A diagnosis tool whose
ground truth moves underneath it cannot be trusted, no matter how good the code is.
These rules make the data as disciplined as the code already is.
