#!/usr/bin/env python3
"""Pull a seed-intake batch into staging/raw/ (run on a machine with internet).

Hardened version of the original uploaded puller:
  - HotpotQA fixed for the real HF columnar schema (context/supporting_facts are
    dicts of parallel lists, not lists of tuples).
  - Each source is wrapped so one failing source doesn't lose the others.
  - Prints each dataset's actual feature schema, so any field-name guess for
    RAGTruth/ALCE is easy to correct against reality.
  - Deterministic sampling (seeded) and emits the full seed-intake schema
    (source_dataset/source_id/domain/...), matching staging/README.md.

Default output: evals/govrag_calib/staging/raw/starter_seed_intake.jsonl
Fresh heldout helper:
  python scripts/pull_seed_intake.py --fresh-preset
  # -> evals/govrag_calib/staging/raw/fresh_intake_v1.jsonl

NOTE: must be run where huggingface.co is reachable (it is blocked in the RagGov
build sandbox). Run `pip install datasets` first.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - only hit without HF datasets.
    load_dataset = None
    _DATASETS_IMPORT_ERROR = exc
else:
    _DATASETS_IMPORT_ERROR = None

STARTER_OUT_PATH = Path("evals/govrag_calib/staging/raw/starter_seed_intake.jsonl")
FRESH_OUT_PATH = Path("evals/govrag_calib/staging/raw/fresh_intake_v1.jsonl")


def write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def coerce_license(ds) -> str:
    meta = getattr(getattr(ds, "info", None), "metadata", None)
    if isinstance(meta, dict):
        for k in ("license", "licenses"):
            if meta.get(k):
                return str(meta[k])
    return "UNKNOWN (check dataset card before redistribution)"


def _show_schema(name: str, ds) -> None:
    try:
        print(f"[{name}] features: {list(ds.features.keys())}")
    except Exception:
        pass


def _require_datasets() -> None:
    if load_dataset is None:
        raise SystemExit(
            "Missing dependency: install Hugging Face datasets first, e.g. "
            "`pip install datasets`."
        ) from _DATASETS_IMPORT_ERROR


# ---------- RAGTruth (labelled_failure) ----------
def _ctx_to_passages(ctx, base_id: str) -> list[dict]:
    """Split a RAGTruth context string into passage chunks."""
    ctx_str = ctx if isinstance(ctx, str) else json.dumps(ctx)
    parts = [p.strip() for p in re.split(r"\n\n+|passage\s*\d+\s*:", ctx_str,
                                         flags=re.I) if p.strip()]
    parts = parts or [ctx_str]
    return [{"doc_id": f"{base_id}-p{i}", "text": t} for i, t in enumerate(parts[:6])]


def pull_ragtruth(
    n_conflict: int = 15,
    n_baseless: int = 15,
    skip_conflict: int = 0,
    skip_baseless: int = 0,
) -> list[dict]:
    _require_datasets()
    # Real schema: query, context, output, hallucination_labels_processed, task_type, ...
    ds = load_dataset("wandb/RAGTruth-processed", split="train")
    _show_schema("ragtruth", ds)
    lic = coerce_license(ds)
    conflict, baseless = [], []
    skipped_task = 0
    for ex in ds:
        # Only the QA task is a genuine RAG question->context->answer case.
        # (RAGTruth also has Summary and Data2txt tasks, which don't fit.)
        task = str(ex.get("task_type") or "").lower()
        if "qa" not in task and "question" not in task:
            skipped_task += 1
            continue
        q, ctx, out = ex.get("query"), ex.get("context"), ex.get("output")
        if not (q and ctx and out):
            continue
        raw = ex.get("hallucination_labels_processed") or ex.get("hallucination_labels")
        if not raw:
            continue  # no hallucination annotated -> not a failure case
        blob = json.dumps(raw).lower()
        # Conflict = contradicts the source; Baseless Info = unsupported addition.
        label = "contradicted" if "conflict" in blob else "unsupported"
        item = {
            "source_dataset": "ragtruth",
            "source_id": str(ex.get("id") or ""),
            "domain": "wiki",
            "query": q,
            "passages": _ctx_to_passages(ctx, f"ragtruth-{ex.get('id')}"),
            "reference_answer": out,  # the (flawed) model response being judged
            "kind": "labelled_failure",
            "source_label": label,
            "supporting_doc_ids": None,
            "license": lic,
            # Carry the raw label so the contradicted/unsupported split is auditable.
            "notes": f"RAGTruth QA; raw_label={json.dumps(raw)[:160]}",
        }
        (conflict if label == "contradicted" else baseless).append(item)
    random.shuffle(conflict)
    random.shuffle(baseless)
    print(f"  (ragtruth: skipped {skipped_task} non-QA rows; "
          f"{len(conflict)} conflict / {len(baseless)} baseless QA failures found)")
    return (
        conflict[skip_conflict:skip_conflict + n_conflict]
        + baseless[skip_baseless:skip_baseless + n_baseless]
    )


# ---------- HotpotQA (clean) ----------
def pull_hotpotqa(n: int = 20, skip: int = 0) -> list[dict]:
    _require_datasets()
    ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train")
    _show_schema("hotpotqa", ds)
    lic = coerce_license(ds)
    items = []
    for idx, ex in enumerate(ds):
        if idx < skip:
            continue
        ctx = ex["context"]
        passages = [{"doc_id": t, "text": " ".join(s)}
                    for t, s in zip(ctx["title"], ctx["sentences"])]
        supporting = sorted(set(ex["supporting_facts"]["title"]))
        items.append({
            "source_dataset": "hotpotqa",
            "source_id": str(ex.get("id") or ""),
            "domain": "wiki",
            "query": ex.get("question"),
            "passages": passages,
            "reference_answer": ex.get("answer"),
            "kind": "clean",
            "source_label": None,
            "supporting_doc_ids": supporting,
            "license": lic,
            "notes": "",
        })
        if len(items) >= n:
            break
    return items


# ---------- ALCE (labelled_failure; citation) ----------
def pull_alce(n: int = 10, skip: int = 0) -> list[dict]:
    _require_datasets()
    # ALCE-data is a WebDataset: each row is a FILE -> data lives in ex['json'].
    # No pre-labelled mismatches, so treat as CLEAN (question + gold docs + answer)
    # and let induce_cases.py derive citation/sufficiency variants.
    ds = load_dataset("princeton-nlp/ALCE-data", split="train")
    _show_schema("alce", ds)
    lic = coerce_license(ds)
    items: list[dict] = []
    seen_records = 0
    for ex in ds:
        raw = ex.get("json")
        try:
            obj = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
        except Exception:
            continue
        records = obj.get("data") if isinstance(obj, dict) else obj
        if not isinstance(records, list):
            continue
        fname = str(ex.get("__key__") or "")
        for rec in records:
            if seen_records < skip:
                seen_records += 1
                continue
            seen_records += 1
            q = rec.get("question") or rec.get("query")
            ans = rec.get("answer") or rec.get("gold_answer") or ""
            docs = rec.get("docs") or rec.get("passages") or rec.get("ctxs") or []
            passages = []
            for j, d in enumerate(docs[:8]):
                if isinstance(d, dict):
                    txt = d.get("text") or d.get("content") or d.get("snippet") or ""
                    if txt:
                        passages.append({"doc_id": d.get("id") or d.get("title")
                                         or f"alce-{j}", "text": txt})
            if not (q and passages):
                continue
            items.append({
                "source_dataset": "alce",
                "source_id": f"{fname}:{rec.get('sample_id') or rec.get('id') or len(items)}",
                "domain": "wiki",
                "query": q,
                "passages": passages,
                "reference_answer": ans if isinstance(ans, str) else json.dumps(ans),
                "kind": "clean",
                "source_label": None,
                "supporting_doc_ids": [passages[0]["doc_id"]],
                "license": lic,
                "notes": f"ALCE {fname}; treated as clean base for induction.",
            })
            if len(items) >= n:
                break
        if len(items) >= n:
            break
    return items


# ---------- deepset/prompt-injections (labelled_failure; injection) ----------
def pull_prompt_injections(n: int = 10) -> list[dict]:
    _require_datasets()
    ds = load_dataset("deepset/prompt-injections", split="train")
    _show_schema("prompt_injections", ds)
    lic = coerce_license(ds)
    inj = [ex for ex in ds if int(ex.get("label", 0)) == 1]
    random.shuffle(inj)
    items = []
    for i, ex in enumerate(inj[:n]):
        items.append({
            "source_dataset": "prompt_injections",
            "source_id": f"deepset-{i}",
            "domain": "software",
            "query": None,
            "passages": [{"doc_id": f"deepset-inj-{i}", "text": ex.get("text", "")}],
            "reference_answer": None,
            "kind": "labelled_failure",
            "source_label": "prompt_injection",
            "supporting_doc_ids": None,
            "license": lic,
            "notes": "",
        })
    return items


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pull GovRAG seed-intake rows from public source datasets."
    )
    parser.add_argument("--out", type=Path, default=STARTER_OUT_PATH)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--ragtruth-conflict", type=int, default=15)
    parser.add_argument("--ragtruth-baseless", type=int, default=15)
    parser.add_argument("--ragtruth-conflict-skip", type=int, default=0)
    parser.add_argument("--ragtruth-baseless-skip", type=int, default=0)
    parser.add_argument("--hotpotqa", type=int, default=20)
    parser.add_argument("--hotpotqa-skip", type=int, default=0)
    parser.add_argument("--alce", type=int, default=10)
    parser.add_argument("--alce-skip", type=int, default=0)
    parser.add_argument("--prompt-injections", type=int, default=10)
    parser.add_argument(
        "--fresh-preset",
        action="store_true",
        help=(
            "Pull a fresh non-starter slice for heldout intake: seed=99, "
            "out=staging/raw/fresh_intake_v1.jsonl, RAGTruth 25+25, "
            "HotpotQA 30 after skipping 20, ALCE 20 after skipping 10, "
            "and no prompt-injection rows."
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.fresh_preset:
        args.out = FRESH_OUT_PATH
        args.seed = 99
        args.ragtruth_conflict = 25
        args.ragtruth_baseless = 25
        args.ragtruth_conflict_skip = 15
        args.ragtruth_baseless_skip = 15
        args.hotpotqa = 30
        args.hotpotqa_skip = 20
        args.alce = 20
        args.alce_skip = 10
        args.prompt_injections = 0

    random.seed(args.seed)
    items: list[dict] = []
    pullers = [
        (
            "ragtruth",
            lambda: pull_ragtruth(
                n_conflict=args.ragtruth_conflict,
                n_baseless=args.ragtruth_baseless,
                skip_conflict=args.ragtruth_conflict_skip,
                skip_baseless=args.ragtruth_baseless_skip,
            ),
        ),
        ("hotpotqa", lambda: pull_hotpotqa(n=args.hotpotqa, skip=args.hotpotqa_skip)),
        ("alce", lambda: pull_alce(n=args.alce, skip=args.alce_skip)),
    ]
    if args.prompt_injections:
        pullers.append(
            ("prompt_injections", lambda: pull_prompt_injections(n=args.prompt_injections))
        )

    for name, fn in pullers:
        try:
            got = fn()
            items += got
            print(f"  {name}: {len(got)} items")
        except Exception as exc:  # one bad source must not lose the rest
            print(f"  {name}: SKIPPED ({type(exc).__name__}: {exc})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, items)
    print(f"Wrote {len(items)} items -> {args.out}")


if __name__ == "__main__":
    main()
