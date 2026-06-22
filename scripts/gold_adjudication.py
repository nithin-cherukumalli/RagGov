#!/usr/bin/env python3
"""Human adjudication -> locked gold heldout -> honest re-score (v1 taxonomy).

Two-LLM-judge is nice when both judges work; when one provider is flaky, the human IS the second
judge. This turns the Kimi proposal into an editable sheet, then finalizes the human-confirmed
labels into a LOCKED gold heldout and re-scores the engine against it (mapping legacy engine output
to the 6-bucket v1 taxonomy so the numbers are honest).

Usage:
  # 1) make an editable sheet from the Kimi proposal (final_label prefilled = proposed):
  python scripts/gold_adjudication.py sheet \
      --from evals/govrag_calib/staging/raw/heldout_real_v1_gold_kimi.jsonl

  # 2) edit heldout_real_v1_gold_sheet.csv in a spreadsheet — change `final_label` where you
  #    disagree (valid: CLEAN PROMPT_INJECTION STALE_RETRIEVAL INSUFFICIENT_CONTEXT
  #    UNSUPPORTED_CLAIM CONTRADICTED_CLAIM). The `bucket` column flags where to focus.

  # 3) finalize -> locked gold + re-score:
  python scripts/gold_adjudication.py finalize \
      --sheet evals/govrag_calib/staging/raw/heldout_real_v1_gold_sheet.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
STAGE = ROOT / "evals/govrag_calib/staging/raw"
SOURCE = STAGE / "heldout_real_v1.jsonl"
SHEET = STAGE / "heldout_real_v1_gold_sheet.csv"
LOCKED = STAGE / "heldout_real_v1_gold.jsonl"

V1_LABELS = {
    "CLEAN", "PROMPT_INJECTION", "STALE_RETRIEVAL",
    "INSUFFICIENT_CONTEXT", "UNSUPPORTED_CLAIM", "CONTRADICTED_CLAIM",
}

# legacy engine FailureType -> v1 bucket (see taxonomy_v1.md)
LEGACY_TO_V1 = {
    "CLEAN": "CLEAN",
    "PROMPT_INJECTION": "PROMPT_INJECTION", "SUSPICIOUS_CHUNK": "PROMPT_INJECTION",
    "STALE_RETRIEVAL": "STALE_RETRIEVAL",
    "INSUFFICIENT_CONTEXT": "INSUFFICIENT_CONTEXT", "SCOPE_VIOLATION": "INSUFFICIENT_CONTEXT",
    "RETRIEVAL_ANOMALY": "INSUFFICIENT_CONTEXT", "RETRIEVAL_DEPTH_LIMIT": "INSUFFICIENT_CONTEXT",
    "EMBEDDING_DRIFT": "INSUFFICIENT_CONTEXT", "RERANKER_FAILURE": "INSUFFICIENT_CONTEXT",
    "UNSUPPORTED_CLAIM": "UNSUPPORTED_CLAIM", "CITATION_MISMATCH": "UNSUPPORTED_CLAIM",
    "POST_RATIONALIZED_CITATION": "UNSUPPORTED_CLAIM", "INCONSISTENT_CHUNKS": "UNSUPPORTED_CLAIM",
    "CONTRADICTED_CLAIM": "CONTRADICTED_CLAIM",
}


def _bucket(case_id: str, provisional: str) -> str:
    if "alce" in case_id:
        return "ALCE_list"  # list-factoid: UNSUPPORTED is often correct (faithfulness)
    if provisional == "CONTRADICTED_CLAIM":
        return "ex-CONTRADICTED"  # all re-derived; confirm a few
    if "hotpotqa" in case_id:
        return "hotpotqa"
    return "other"


def make_sheet(src: Path) -> None:
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    with SHEET.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "bucket", "provisional", "proposed", "confidence",
                    "final_label", "rationale", "query"])
        for r in rows:
            prov = r.get("provisional_label", "")
            proposed = r.get("proposed_gold", "")
            w.writerow([
                r.get("case_id", ""), _bucket(r.get("case_id", ""), prov), prov, proposed,
                r.get("confidence", ""), proposed, (r.get("rationale", "") or "")[:200],
                (r.get("query", "") or "")[:120],
            ])
    print(f"wrote {SHEET}  ({len(rows)} rows). Edit the `final_label` column, then run finalize.")


def finalize(sheet: Path) -> None:
    final = {}
    bad = []
    with sheet.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lbl = (row.get("final_label") or "").strip().upper()
            if lbl not in V1_LABELS:
                bad.append((row.get("case_id"), lbl))
            final[row["case_id"]] = lbl
    if bad:
        print("ERROR: invalid final_label values (fix these and re-run finalize):")
        for cid, lbl in bad[:20]:
            print(f"  {cid}: {lbl!r}")
        raise SystemExit(1)

    src = [json.loads(l) for l in SOURCE.read_text(encoding="utf-8").splitlines() if l.strip()]
    locked = []
    for r in src:
        cid = r.get("case_id")
        if cid not in final:
            continue
        r = dict(r)
        r["expected_primary_failure"] = final[cid]
        r["label_source"] = "gold_human_adjudicated_v1"
        r["label_confidence"] = "gold"
        locked.append(r)
    LOCKED.write_text("\n".join(json.dumps(r) for r in locked) + "\n", encoding="utf-8")

    # honest re-score against the locked gold (map engine legacy output -> v1)
    import scripts.raggov_score as rs  # noqa: E402
    eng = rs._engine("native")
    import collections
    dist = collections.Counter(final[c] for c in final)
    n = correct = 0
    clean_tot = clean_fp = 0
    fail_tot = fail_detected = 0
    per = collections.defaultdict(lambda: [0, 0])
    for r in locked:
        gold = r["expected_primary_failure"]
        pred_legacy = eng.diagnose(rs.build_run(r)).primary_failure.value
        pred = LEGACY_TO_V1.get(pred_legacy, "OTHER")
        n += 1
        per[gold][0] += 1
        if pred == gold:
            correct += 1
            per[gold][1] += 1
        if gold == "CLEAN":
            clean_tot += 1
            clean_fp += pred != "CLEAN"
        else:
            fail_tot += 1
            fail_detected += pred != "CLEAN"  # any failure flagged (detection, not exact type)
    print(f"\nLOCKED gold: {LOCKED}  ({n} rows)")
    print("gold distribution:", dict(dist))
    print(f"v1 exact-match accuracy: {correct}/{n} = {correct/n:.3f}")
    print(f"CLEAN false-positive rate: {clean_fp}/{clean_tot} = {clean_fp/max(1,clean_tot):.3f}")
    print(f"failure DETECTION rate (any flag): {fail_detected}/{fail_tot} = {fail_detected/max(1,fail_tot):.3f}")
    print("per-type exact accuracy:", {k: f"{v[1]}/{v[0]}" for k, v in sorted(per.items())})


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet"); s.add_argument("--from", dest="src", type=Path, required=True)
    f = sub.add_parser("finalize"); f.add_argument("--sheet", type=Path, default=SHEET)
    args = ap.parse_args()
    if args.cmd == "sheet":
        make_sheet(args.src)
    else:
        finalize(args.sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
