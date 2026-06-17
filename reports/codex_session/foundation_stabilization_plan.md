# Foundation Stabilization Plan — read this before any more analyzer work

**Date:** 2026-06-17
**Why this exists.** The task queue (Tasks 3–13) is built on an evaluation
foundation that is too small and not version-locked. Golden labels were
relabeled/renumbered between sessions, several fixtures are `TODO` placeholders,
and some failure types in the taxonomy have zero supporting data. Until that is
fixed, every accuracy number is unreliable and every analyzer change is measured
against a moving target. This plan reorders the work so the foundation is solid
*before* more detection logic is added.

**The one-sentence problem:** the project can write more analyzer code faster than
it can produce trustworthy ground truth to evaluate it — so the bottleneck is
data and labels, not code.

---

## P0 — Lock the ground truth (do this first, nothing else matters until it's done)

1. **Freeze a canonical dataset.** Pick one file as the single source of truth
   (`govrag_calib_150.jsonl`), tag it with a version (e.g. `dataset_v1`), and stop
   renumbering. Case IDs become **permanent and immutable** — a `gc-0NN` must mean
   the same case forever.
2. **Add a label changelog + provenance.** Every golden-label change gets a dated
   line with who/why. The silent relabeling that happened between 2026-06-15 and now
   must never happen again — it invalidates prior results.
3. **Resolve the placeholders.** `gc-047`, `gc-051`, `gc-052` (and any other
   `TODO`/`WRONG`-marked fixtures) are either filled with real cases or removed from
   the scored set. Placeholders must not count toward accuracy.
4. **Re-pin the open tasks** (3-v2/4-v2/5-v2, and 14/15/16) to current immutable IDs
   once the freeze is done. See `v2_feasibility_blocker.md`.

**Done when:** the dataset has a version tag, a changelog, zero placeholders in the
scored set, and a written rule that case IDs are immutable.

---

## P1 — Right-size the taxonomy to the data

5. **Audit every `FailureType`.** Count golden cases per type. Today the spread is
   extreme (CONTRADICTED_CLAIM=11 … RERANKER_FAILURE=**0**, several at 1–3).
6. **Quarantine unsupported types.** Any failure type with **zero** goldens
   (`RERANKER_FAILURE`) is marked `experimental` or removed from the public enum —
   it is currently advertised but cannot be detected or validated.
7. **Set a per-type minimum** (e.g. ≥5 real cases) before a type counts toward
   headline accuracy or appears in README claims.

**Done when:** the advertised failure modes == the ones with real detection + data.

---

## P2 — Grow and split the dataset properly

8. **Scale up.** ~52 cases (with a 3-case heldout slice in the file) is too small to
   trust. Target hundreds, with a per-failure-type floor.
9. **Real, frozen heldout.** A held-out split large enough to be meaningful (tens of
   cases minimum), frozen, and rotated on a schedule — not edited in place.

**Done when:** accuracy numbers come with a sample size you'd defend to a skeptic.

---

## P3 — Fix the packaging/runtime bug (small, do it anytime)

10. Code uses Python 3.11 features (`datetime.UTC`, `tomllib`) but `pyproject.toml`
    declares `requires-python >=3.10`. Either bump to `>=3.11` or add 3.10 shims.
    Right now a clean 3.10 install fails to import.

---

## P4 — Only now: fix the 3 real routing bugs (Tasks 14/15/16)

These are genuine production bugs found in the red-test triage (see
`red_test_triage.md`). They are routing/stage-attribution gaps, measured against the
dataset — so they must wait until P0/P1 make the dataset trustworthy. Each needs its
own pre-registration.

---

## P5 — Then revisit the original queue, re-scoped

In order: Task 3-v2 (re-scoped to real CITATION_MISMATCH cases) → Task 4-v2 / 5-v2
(only if/when introspection signals and reranker data are added by deliberate schema
decision) → Task 11 (warn-promotion removal) → Task 12 (calibration pack) → Task 13
(README/release). Task 11's "Tasks 1–5 must land" precondition stays blocked until
the re-scoped v2 work actually lands honestly.

---

## What "trustworthy diagnosis" looks like (the bar)

- A frozen dataset of hundreds of cases, immutable IDs, label changelog.
- Every advertised failure type has ≥5 real cases and an analyzer the engine
  actually selects.
- Primary-failure accuracy reported with confidence intervals over multiple seeds.
- A calibration curve (predicted confidence vs. empirical accuracy).
- README claims that match measured numbers, with sample sizes.

Until those exist, the honest framing is: **RagGov gives a useful first-pass signal
on the failure modes it has data for, and is not yet an authoritative diagnosis.**

---

## Status note: nothing already committed is wasted

The CLI provenance block (Tasks 6/7), the framework examples (8/9), the red-test
cleanup (Task 10), and all the triage/feasibility documentation are independent of
this data work and remain valid. This plan changes the *order* of what comes next;
it does not undo what landed.
