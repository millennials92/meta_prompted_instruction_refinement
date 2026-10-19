"""Orchestrate refinement runs for the protocol's experiment groups (resumable)."""
from __future__ import annotations

import argparse
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from .data import tier1_cells, tier2_cells
from .mpir import RunSpec, run_refinement
from .scoring import Scorer


def specs_for(group: str, mseeds: tuple[int, ...]) -> list[tuple[RunSpec, bool]]:
    """(spec, oracle) pairs for a protocol group."""
    pilot_cells = [c for c in tier1_cells() if c.seed == 42]
    if group == "g1":
        return [(RunSpec(c, "v2", "1.7b", m, val_reps=1), False) for c in tier1_cells() + tier2_cells() for m in mseeds]
    if group == "g2":
        out: list[tuple[RunSpec, bool]] = []
        for variant in ("v0", "v1", "v2", "v2l", "v3"):
            seeds = (1,) if variant == "v0" else mseeds
            out += [(RunSpec(c, variant, "1.7b", m), m == 1) for c in pilot_cells for m in seeds]
        return out
    if group == "g3-4b":
        return [(RunSpec(c, v, "4b", m), m == 1) for c in pilot_cells for v in ("v1", "v2") for m in mseeds]
    if group == "g3-8b":
        return [(RunSpec(c, "v1", "8b", m), m == 1) for c in pilot_cells for m in mseeds]
    raise SystemExit(f"unknown group {group}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True)
    parser.add_argument("--mseeds", default="1,2,3")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    mseeds = tuple(int(s) for s in args.mseeds.split(","))
    todo = [(s, o) for s, o in specs_for(args.group, mseeds) if not s.path.exists()]
    print(f"{args.group}: {len(todo)} runs to do", flush=True)
    scorers: dict[str, Scorer] = {}

    def work(item):
        spec, oracle = item
        scorer = scorers.setdefault(spec.cell.key, Scorer(spec.cell))
        started = time.time()
        try:
            result = run_refinement(spec, scorer, oracle=oracle)
            outcomes = [r["outcome"][:3] for r in result["rounds"]]
            print(f"DONE {spec.key} {time.time() - started:.0f}s final_is_p0={result['final_is_p0']} "
                  f"test {result['p0_test']:.3f}->{result['final_test']:.3f} {outcomes}", flush=True)
        except Exception:
            print(f"FAILED {spec.key}\n{traceback.format_exc()}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))


if __name__ == "__main__":
    main()
