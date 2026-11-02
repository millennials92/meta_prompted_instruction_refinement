"""Rescore the prompts behind the revision's headline claims in ONE serving session, with repeated scorings.

Answers reviewer concerns about mixed server settings (restarts, KV/prefix-cache changes) and the 6.5 % flip rate:
the original per-run scorings were produced across several server configurations, so every prompt that a claim in the
paper rests on is scored again here under a single configuration (2 repeats), and the outputs are kept for the
lenient-parser validation and the failure-mode audit.

    python -m mpir_study.rescore --set headline
    python -m mpir_study.rescore --set protected
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from .analyze import REV, cell_of, load_runs, runs_where
from .data import Cell, load_p0, tier1_cells
from .scoring import Scorer, prompt_hash

OUT = REV / "rescore.json"


def final_prompt(run: dict) -> str:
    cell = cell_of(run)
    if run["final_is_p0"]:
        return load_p0(cell)
    for x in run["rounds"]:
        if x.get("cand_hash") == run["final_hash"] and x.get("candidate"):
            return x["candidate"]
    raise KeyError(run["spec"])


def headline_items(runs: list[dict]) -> list[tuple[Cell, str, str]]:
    """(cell, label, prompt) for P0 and the final prompts of the exploratory strong-refiner variants."""
    items = []
    for c in tier1_cells(seeds=(42,)):
        items.append((c, "p0", load_p0(c)))
    for refiner, variant, mseed in (("qwen235", "v2", 1), ("qwen235", "v2", 2), ("qwen235", "v3", 1),
                                    ("gpt6luna", "v2", 1)):
        for r in runs_where(runs, variant=variant, refiner=refiner, mseed=mseed):
            if r["spec"]["cell"]["seed"] == 42:
                items.append((cell_of(r), f"{variant}|{refiner}|m{mseed}", final_prompt(r)))
    return items


def protected_items(runs: list[dict]) -> list[tuple[Cell, str, str]]:
    items = []
    for variant in ("v1p", "v2p"):
        for r in runs_where(runs, variant=variant, refiner="1.7b"):
            for x in r["rounds"]:
                if x["outcome"] == "validated" and "test" in x:
                    items.append((cell_of(r), f"{variant}|cand|m{r['spec']['mseed']}|t{x['round']}", x["candidate"]))
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", required=True, choices=("headline", "protected"))
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    out_path = OUT.with_name("rescore.json" if args.set == "headline" else "rescore_protected.json")
    runs = load_runs()
    items = headline_items(runs) if args.set == "headline" else protected_items(runs)
    store = json.loads(out_path.read_text()) if out_path.exists() else {}
    scorers: dict[str, Scorer] = {}

    def work(item):
        cell, label, prompt = item
        scorer = scorers.setdefault(cell.key, Scorer(cell))
        reps = args.reps if args.set == "headline" else 1
        vectors = [list(scorer.score(prompt, "test", rep).correct) for rep in range(reps)]
        return f"{cell.key}|{label}", {"hash": prompt_hash(prompt), "test": vectors}

    with ThreadPoolExecutor(args.workers) as pool:
        for key, value in pool.map(work, items):
            store[key] = value
            print("rescored", key, flush=True)
            out_path.write_text(json.dumps(store), encoding="utf-8")


if __name__ == "__main__":
    main()
