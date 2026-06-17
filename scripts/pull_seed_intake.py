#!/usr/bin/env python3
"""Pull a starter seed-intake batch into staging/raw/ (run on a machine with internet).

Hardened version of the original uploaded puller:
  - HotpotQA fixed for the real HF columnar schema (context/supporting_facts are
    dicts of parallel lists, not lists of tuples).
  - Each source is wrapped so one failing source doesn't lose the others.
  - Prints each dataset's actual feature schema, so any field-name guess for
    RAGTruth/ALCE is easy to correct against reality.
  - Deterministic sampling (seeded) and emits the full seed-intake schema
    (source_dataset/source_id/domain/...), matching staging/README.md.

Output: evals/govrag_calib/staging/raw/starter_seed_intake.jsonl

NOTE: must be run where huggingface.co is reachable (it is blocked in the RagGov
build sandbox). Run `pip install datasets` first.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

random.seed(13)
OUT_PATH = Path("evals/govrag_calib/staging/raw/starter_seed_intake.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


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


# ---------- RAGTruth (labelled_failure) ----------
def pull_ragtruth(n_conflict: int = 15, n_baseless: int = 15) -> list[dict]:
    ds = load_dataset("wandb/RAGTruth-processed", split="train")
    _show_schema("ragtruth", ds)
    lic = coerce_license(ds)
    conflict, baseless = [], []
    for ex in ds:
        label = str(ex.get("label") or ex.get("label_type") or "")
        docs = []
        for doc in (ex.get("documents") or ex.get("docs") or ex.get("context") or []):
            if isinstance(doc, dict):
                docs.append({"doc_id": doc.get("id") or doc.get("doc_id")
                             or doc.get("title") or "", "text": doc.get("text")
                             or doc.get("content") or ""})
            else:
                docs.append({"doc_id": "", "text": str(doc)})
        src_label = ("contradicted" if label == "Conflict"
                     else "unsupported" if label == "Baseless Info" else label)
        item = {
            "source_dataset": "ragtruth",
            "source_id": str(ex.get("id") or ex.get("source_id") or ""),
            "domain": "wiki",
            "query": ex.get("prompt") or ex.get("question") or ex.get("query"),
            "passages": docs,
            "reference_answer": ex.get("response") or ex.get("answer")
            or ex.get("reference") or ex.get("gold"),
            "kind": "labelled_failure",
            "source_label": src_label,
            "supporting_doc_ids": None,
            "license": lic,
            "notes": "",
        }
        if src_label == "contradicted":
            conflict.append(item)
        elif src_label == "unsupported":
            baseless.append(item)
    random.shuffle(conflict)
    random.shuffle(baseless)
    return conflict[:n_conflict] + baseless[:n_baseless]


# ---------- HotpotQA (clean) ----------
def pull_hotpotqa(n: int = 20) -> list[dict]:
    ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train",
                      trust_remote_code=True)
    _show_schema("hotpotqa", ds)
    lic = coerce_license(ds)
    items = []
    for ex in ds:
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
def pull_alce(n: int = 10) -> list[dict]:
    ds = load_dataset("princeton-nlp/ALCE-data", split="train")
    _show_schema("alce", ds)
    lic = coerce_license(ds)
    items = []
    for ex in ds:
        ctx = ex.get("context") or ex.get("passage") or ex.get("input") or ""
        items.append({
            "source_dataset": "alce",
            "source_id": str(ex.get("qid") or ex.get("doc_id") or ""),
            "domain": "wiki",
            "query": ex.get("question") or ex.get("input") or ex.get("prompt"),
            "passages": [{"doc_id": ex.get("doc_id") or ex.get("qid") or "",
                          "text": ctx if isinstance(ctx, str) else json.dumps(ctx)}],
            "reference_answer": ex.get("answer") or ex.get("target") or "",
            "kind": "labelled_failure",
            "source_label": "citation",
            "supporting_doc_ids": None,
            "license": lic,
            "notes": f"citation={ex.get('citation')}",
        })
        if len(items) >= n:
            break
    return items


# ---------- deepset/prompt-injections (labelled_failure; injection) ----------
def pull_prompt_injections(n: int = 10) -> list[dict]:
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


def main() -> None:
    items: list[dict] = []
    for name, fn in (("ragtruth", pull_ragtruth), ("hotpotqa", pull_hotpotqa),
                     ("alce", pull_alce), ("prompt_injections", pull_prompt_injections)):
        try:
            got = fn()
            items += got
            print(f"  {name}: {len(got)} items")
        except Exception as exc:  # one bad source must not lose the rest
            print(f"  {name}: SKIPPED ({type(exc).__name__}: {exc})")
    write_jsonl(OUT_PATH, items)
    print(f"Wrote {len(items)} items -> {OUT_PATH}")


if __name__ == "__main__":
    main()
