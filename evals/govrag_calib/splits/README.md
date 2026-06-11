# GovRAG-Calib-150 Splits

Split values:

- `train`: development and exploratory model/threshold work.
- `calibration`: calibration fitting and confidence interval estimation.
- `heldout`: locked final evaluation split.
- `unset`: seed records that are not assigned yet.

Heldout records must use `calibration_status = heldout_locked`.

The seed file intentionally uses `unset` until the dataset is reviewed and split policy is finalized.
