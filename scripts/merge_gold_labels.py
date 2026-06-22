#!/usr/bin/env python3
"""Merge two independent LLM labelers into an adjudication-ready gold set.

A credible benchmark isn't one model's opinion. This takes two `*_gold_proposed.jsonl` files from
`label_heldout_gold.py` (run with two different models/providers) and partitions every row:

  - AGREED  : both labelers chose the same label, neither errored, neither low-confidence.
              -> provisional gold (still worth a fast human skim, but trustworthy).
  - REVIEW  : labelers disagree, OR either errored / was low-confidence.
              -> the focused human worklist (this is the only set you must hand-adjudicate).

Reports inter-annotator agreement (the headline trust metric) and the merged label distribution.
Reads only proposed-label staging files; never writes canonical/gold dataset files.

    PYTHONPATH=src:. python scripts/merge_gold_labels.py \
        --a evals/govrag_calib/staging/raw/heldout_real_v1_gold_proposed.jsonl \
        --b evals/govrag_calib/staging/raw/heldout_real_v1_gold_proposed_modelB.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "evals/govrag_calib/staging/raw"
REPORT = ROOT / "reports/calibration/heldout_real_v1_gold_merge_report.md"

_LOWQ = {"low", ""}
_BAD = {"NEEDS_HUMAN", ""}


def _load(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["case_id"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True, help="labeler A proposed file")
    ap.add_argument("--b", type=Path, required=True, help="labeler B proposed file")
    ap.add_argument("--agreed-out", type=Path, default=STAGE / "heldout_real_v1_gold_agreed.jsonl")
    ap.add_argument("--review-out", type=Path, default=STAGE / "heldout_real_v1_gold_review.jsonl")
    ap.add_argument("--report", type=Path, default=REPORT)
    args = ap.parse_args()

    a, b = _load(args.a), _load(args.b)
    ids = sorted(set(a) & set(b))
    agreed, review = [], []
    both_high_agree = 0
    raw_agree = 0
    confusion: dict[str, dict[str, int]] = {}

    for cid in ids:
        ra, rb = a[cid], b[cid]
        la, lb = ra["proposed_gold"], rb["proposed_gold"]
        ca, cb = ra.get("confidence", "low"), rb.get("confidence", "low")
        confusion.setdefault(la, {}).setdefault(lb, 0)
        confusion[la][lb] += 1
        if la == lb and la not in _BAD:
            raw_agree += 1
        clean = la == lb and la not in _BAD and ca not in _LOWQ and cb not in _LOWQ
        rec = {
            "case_id": cid,
            "query": ra.get("query"),
            "provisional_label": ra.get("provisional_label"),
            "label_a": la, "conf_a": ca, "rationale_a": ra.get("rationale"),
            "label_b": lb, "conf_b": cb, "rationale_b": rb.get("rationale"),
        }
        if clean:
            rec["gold"] = la
            agreed.append(rec)
            both_high_agree += 1
        else:
            rec["reason"] = (
                "disagree" if la != lb else "low_confidence" if (ca in _LOWQ or cb in _LOWQ) else "error"
            )
            review.append(rec)

    args.agreed_out.write_text("\n".join(json.dumps(r) for r in agreed) + "\n", encoding="utf-8")
    args.review_out.write_text("\n".join(json.dumps(r) for r in review) + "\n", encoding="utf-8")

    n = len(ids)
    lines = [
        "# Heldout v1 — two-labeler merge (adjudication-ready)",
        "",
        f"- rows compared: {n}",
        f"- raw inter-annotator agreement: {raw_agree}/{n} = {raw_agree/n:.2f}" if n else "- n/a",
        f"- AGREED + both confident (provisional gold): {both_high_agree}/{n}",
        f"- REVIEW (human must adjudicate): {len(review)}",
        "",
        "## Confusion: label_a x label_b",
    ]
    for la in sorted(confusion):
        cells = ", ".join(f"{lb}:{c}" for lb, c in sorted(confusion[la].items()))
        lines.append(f"- **A={la}** -> {cells}")
    lines += [
        "",
        "## Next",
        "1. Hand-adjudicate ONLY `heldout_real_v1_gold_review.jsonl` (the disagreements).",
        "2. Merge your adjudications with `heldout_real_v1_gold_agreed.jsonl` -> locked gold heldout.",
        "3. Re-run the scorer against the locked gold. Now the numbers mean something.",
    ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"agreed: {args.agreed_out}\nreview: {args.review_out}\nreport: {args.report}")
    print(f"raw agreement {raw_agree}/{n}; provisional gold {both_high_agree}; review {len(review)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
