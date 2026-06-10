# GovRAG-Calib Dataset Splits

This directory holds split-specific subsets of `govrag_calib_150.jsonl`.

Splits are extracted by `scripts/validate_govrag_calib.py` when `--write-splits` is passed.

| File | Contents |
|---|---|
| `train.jsonl` | Cases with `split=train` |
| `dev.jsonl` | Cases with `split=dev` |
| `heldout.jsonl` | Cases with `split=heldout` (do not inspect during development) |
| `unset.jsonl` | Placeholder cases with `split=unset` |

**Warning:** `heldout.jsonl` must not be used during active development or threshold tuning.
