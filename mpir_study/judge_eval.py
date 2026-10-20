"""Judge-validity data (protocol G4): rubric totals for P0 and every oracle-scored candidate, from the
1.7B judge and from a stronger reference judge, both at temperature 0.

    python -m mpir_study.judge_eval --judge 1.7b      # target server up
    python -m mpir_study.judge_eval --judge 8b        # 8B server up
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from .analyze import candidate_table, load_runs
from .client import REFINERS, chat
from .data import Cell, load_p0
from .scoring import prompt_hash
from .templates import CORRECTED_RUBRIC, REPO, parse_rubric_total

OUT = REPO / "results" / "revision" / "judge_scores.json"


def prompts_to_judge() -> dict[str, str]:
    """hash -> prompt text for P0 and every oracle candidate."""
    runs = load_runs()
    texts: dict[str, str] = {}
    for r in runs:
        c = r["spec"]["cell"]
        cell = Cell(c["task"], c["optimizer"], c["seed"])
        texts[r["p0_hash"]] = load_p0(cell)
        for x in r["rounds"]:
            if x["outcome"] == "validated" and "test" in x:
                texts[x["cand_hash"]] = x["candidate"]
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", required=True, choices=list(REFINERS))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    endpoint = REFINERS[args.judge]
    store = json.loads(OUT.read_text()) if OUT.exists() else {}
    done = store.setdefault(args.judge, {})
    todo = {h: t for h, t in prompts_to_judge().items() if h not in done}
    print(f"{args.judge}: {len(todo)} prompts to judge", flush=True)

    def judge(item):
        h, text = item
        out = chat(endpoint, [{"role": "user", "content": CORRECTED_RUBRIC.format(prompt=text)}],
                   temperature=0.0, max_tokens=1500)
        total, out_of = parse_rubric_total(out)
        return h, {"total": total, "out_of": out_of}

    with ThreadPoolExecutor(args.workers) as pool:
        for h, result in pool.map(judge, todo.items()):
            done[h] = result
    # Re-read before writing so concurrent judge processes do not overwrite each other's scores.
    latest = json.loads(OUT.read_text()) if OUT.exists() else {}
    latest.setdefault(args.judge, {}).update(done)
    OUT.write_text(json.dumps(latest, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
