"""Validation of the lenient answer parser against labelled responses (reviewer round 2, comment 3).

    python -m mpir_study.lenient_validation sample   # writes results/revision/lenient_sample.json (+ hidden key)
    python -m mpir_study.lenient_validation score    # reads results/revision/lenient_labels.json

A labeller reads a response (head and tail, without the gold answer and without either parser's verdict) and names the
option the response commits to as its final answer. We then compare, for every response, the strict and the lenient
parser with the labelled truth. Two strata are sampled separately, binary tasks (hyperbaton, sports_understanding) and
multi-option tasks, each half strict-correct and half strict-incorrect, since lenient parsing only matters when
strict parsing fails.
"""
from __future__ import annotations

import json
import random
import sys

from .analyze import REV
from .answers import is_correct_lenient, is_correct_strict, normalize
from .data import load_split
from .revision2 import clopper_pearson_upper

BINARY_TASKS = ("hyperbaton", "sports_understanding")
SEED = 20261102
PER_STRATUM_CLASS = 100  # responses per (stratum, strict-correct / strict-incorrect)
HEAD, TAIL = 500, 900
SAMPLE = REV / "lenient_sample.json"
KEY = REV / "lenient_sample_key.json"
LABELS = REV / "lenient_labels.json"
OUT = REV / "lenient_validation.json"


def _pool() -> dict[tuple[str, bool], list[dict]]:
    """All test responses of the session's score cache, by (stratum, strict_correct)."""
    pool: dict[tuple[str, bool], list[dict]] = {}
    for path in sorted((REV / "score_cache").glob("*.jsonl")):
        task, _, seed = path.stem.split("__")
        gold = [ex["answer"] for ex in load_split(task, int(seed[1:]), "test")]
        stratum = "binary" if task in BINARY_TASKS else "multi"
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["split"] != "test" or row["rep"] != 0 or len(row["outputs"]) != len(gold):
                    continue
                for i, (out, g) in enumerate(zip(row["outputs"], gold)):
                    strict = is_correct_strict(out, g)
                    pool.setdefault((stratum, strict), []).append(
                        {"cell": path.stem, "hash": row["hash"], "index": i, "task": task, "gold": g, "output": out})
    return pool


def make_sample() -> None:
    rng = random.Random(SEED)
    pool = _pool()
    chosen = []
    for stratum in ("binary", "multi"):
        for strict in (True, False):
            items = pool.get((stratum, strict), [])
            chosen += [dict(x, stratum=stratum, strict_correct=strict)
                       for x in rng.sample(items, min(PER_STRATUM_CLASS, len(items)))]
    rng.shuffle(chosen)
    public, key = [], {}
    for n, item in enumerate(chosen):
        sid = f"r{n:03d}"
        text = item["output"]
        shown = text if len(text) <= HEAD + TAIL else text[:HEAD] + "\n[... omitted ...]\n" + text[-TAIL:]
        kind = "yes/no" if item["task"] == "sports_understanding" else "letters (A), (B), ..."
        public.append({"id": sid, "task": item["task"], "answer_format": kind, "response": shown})
        key[sid] = {k: item[k] for k in ("cell", "hash", "index", "task", "gold", "stratum", "strict_correct")}
    SAMPLE.write_text(json.dumps(public, indent=1), encoding="utf-8")
    KEY.write_text(json.dumps(key), encoding="utf-8")
    print("sample", len(public), {f"{s}|{c}": len(pool.get((s, c), [])) for s in ("binary", "multi") for c in (True, False)})


def _norm_label(label: str) -> str:
    return normalize(label) if label else "none"


def score() -> None:
    labels = {x["id"]: x["committed"] for x in json.loads(LABELS.read_text(encoding="utf-8"))}
    key = json.loads(KEY.read_text(encoding="utf-8"))
    cache = {}
    rows = []
    for sid, k in key.items():
        if sid not in labels:
            continue
        path = REV / "score_cache" / f"{k['cell']}.jsonl"
        if k["cell"] not in cache:
            cache[k["cell"]] = {}
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    r = json.loads(line)
                    if r["split"] == "test" and r["rep"] == 0:
                        cache[k["cell"]][r["hash"]] = r["outputs"]
        out = cache[k["cell"]][k["hash"]][k["index"]]
        truth = _norm_label(labels[sid]) == normalize(k["gold"])
        rows.append({"id": sid, "stratum": k["stratum"], "strict": is_correct_strict(out, k["gold"]),
                     "lenient": is_correct_lenient(out, k["gold"]), "truth": truth,
                     "committed": _norm_label(labels[sid])})
    result = {"n": len(rows)}
    for stratum in ("binary", "multi", "all"):
        sel = [r for r in rows if stratum == "all" or r["stratum"] == stratum]
        entry = {"n": len(sel)}
        for metric in ("strict", "lenient"):
            tp = sum(r[metric] and r["truth"] for r in sel)
            fp = sum(r[metric] and not r["truth"] for r in sel)
            fn = sum((not r[metric]) and r["truth"] for r in sel)
            tn = sum((not r[metric]) and not r["truth"] for r in sel)
            entry[metric] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                             "precision": tp / (tp + fp) if tp + fp else None,
                             "recall": tp / (tp + fn) if tp + fn else None,
                             "agreement": (tp + tn) / len(sel) if sel else None,
                             "fp_upper_95": clopper_pearson_upper(fp, tp + fp) if tp + fp else None}
        # among responses that strict parsing rejects (where the lenient parser matters)
        rej = [r for r in sel if not r["strict"]]
        entry["strict_rejected"] = {"n": len(rej), "truly_correct": sum(r["truth"] for r in rej),
                                    "lenient_accepts": sum(r["lenient"] for r in rej),
                                    "lenient_accepts_and_correct": sum(r["lenient"] and r["truth"] for r in rej)}
        result[stratum] = entry
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "n"}, indent=1)[:1500])


if __name__ == "__main__":
    {"sample": make_sample, "score": score}[sys.argv[1]]()
