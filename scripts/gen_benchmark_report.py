#!/usr/bin/env python3
"""Generate BENCHMARK.md — the honest, reproducible evaluation of RagGov on the locked gold set.

Single source of truth for the numbers quoted in the README. Run after any engine change.
    PYTHONPATH=src:. python scripts/gen_benchmark_report.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
GOLD = ROOT / "evals/govrag_calib/staging/raw/heldout_real_v1_gold.jsonl"
OUT = ROOT / "BENCHMARK.md"

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


def main() -> int:
    import raggov_score as rs  # noqa: E402
    rows = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    eng = rs._engine("native")
    n = correct = ctot = cfp = ftot = fdet = 0
    per = collections.defaultdict(lambda: [0, 0])
    dist = collections.Counter(r["expected_primary_failure"] for r in rows)
    for r in rows:
        gold = r["expected_primary_failure"]
        pred = LEGACY_TO_V1.get(eng.diagnose(rs.build_run(r)).primary_failure.value, "OTHER")
        n += 1
        per[gold][0] += 1
        if pred == gold:
            correct += 1
            per[gold][1] += 1
        if gold == "CLEAN":
            ctot += 1
            cfp += pred != "CLEAN"
        else:
            ftot += 1
            fdet += pred != "CLEAN"

    md = [
        "# RagGov Benchmark — Real Heldout (locked gold)",
        "",
        "Reproducible: `PYTHONPATH=src:. python scripts/gen_benchmark_report.py`. Native (heuristic)",
        "mode, no LLM. Evaluated against `heldout_real_v1_gold.jsonl` — 75 real RAG runs from RAGTruth /",
        "HotpotQA / ALCE, relabeled into the 6-class v1 taxonomy via a two-stage protocol (strong-LLM",
        "proposal -> independent chunk-by-chunk adjudication -> human/maintainer verification).",
        "",
        "## Headline (native, no LLM)",
        f"- **CLEAN false-positive rate: {cfp}/{ctot} = {cfp/ctot:.2f}** (lower is better — faithful answers wrongly flagged)",
        f"- **Failure detection rate: {fdet}/{ftot} = {fdet/ftot:.2f}** (real failures that get flagged)",
        f"- **Exact-class accuracy: {correct}/{n} = {correct/n:.2f}** (correct v1 class incl. CLEAN)",
        "",
        "## Gold label distribution",
        "".join(f"\n- {k}: {v}" for k, v in dist.most_common()),
        "",
        "## Per-class exact accuracy",
        "| class | correct / total |",
        "|-------|-----------------|",
        *[f"| {k} | {v[1]}/{v[0]} |" for k, v in sorted(per.items())],
        "",
        "## Honesty notes (read these)",
        "- The engine is **uncalibrated heuristics**, not an ML classifier; confidence is advisory.",
        "- The benchmark is small (75 rows) and the labels are maintainer-adjudicated, not a community",
        "  gold standard. Treat numbers as directional, not authoritative.",
        "- The original migrated labels were mislabeled (25 rows tagged CONTRADICTED; 0 held up under",
        "  re-adjudication). Fixing the benchmark was step one — see `reports/codex_session/`.",
        "- CLEAN false-positive reduction (0.65 -> 0.48 on this set) came from removing analyzers/",
        "  signals that had zero true-positives on real data, each change guarded against a protected",
        "  regression suite. No tuning on the heldout.",
    ]
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"CLEAN-FP {cfp}/{ctot}={cfp/ctot:.2f} | detection {fdet}/{ftot}={fdet/ftot:.2f} | exact {correct}/{n}={correct/n:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
