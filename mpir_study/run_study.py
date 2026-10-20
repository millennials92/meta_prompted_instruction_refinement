"""Orchestrate refinement runs for the protocol's experiment groups (resumable)."""
from __future__ import annotations

import argparse
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from .data import DEMOS, TIER1_TASKS, tier1_cells, tier2_cells
from .mpir import RunSpec, generate_independent, run_refinement
from .scoring import Scorer


def specs_for(group: str, mseeds: tuple[int, ...]) -> list[tuple[RunSpec, bool]]:
    """(spec, oracle) pairs for a protocol group."""
    pilot_cells = [c for c in tier1_cells() if c.seed == 42]
    if group == "g1":
        # Tier-1 seed-42 cells are covered by G2 (same V2 runs, with oracle scoring and three validation repeats).
        cells = [c for c in tier1_cells() + tier2_cells() if not (c.seed == 42 and c.task in TIER1_TASKS)]
        return [(RunSpec(c, "v2", "1.7b", m, val_reps=1), False) for c in cells for m in mseeds]
    if group == "g2":
        out: list[tuple[RunSpec, bool]] = []
        for variant in ("v0", "v1", "v2", "v2l", "v3"):
            seeds = (1,) if variant in ("v0", "v2l") else mseeds
            oracle = variant in ("v1", "v2", "v3")
            out += [(RunSpec(c, variant, "1.7b", m), oracle and m == 1) for c in pilot_cells for m in seeds]
        return out
    if group == "g3-api":
        # Stronger refiners through OpenRouter; the target stays the local 1.7B model.
        return [(RunSpec(c, "v2", r, m), m == 1) for r in ("gpt6luna", "qwen235") for c in pilot_cells
                for m in mseeds]
    if group == "g3-follow":
        # Exploratory follow-up (not in the original protocol): after the Qwen3-235B refiner showed gains,
        # (a) the rubric-free gated rewrite with the same refiner, and (b) a second MPIR seed.
        return ([(RunSpec(c, "v3", "qwen235", 1), True) for c in pilot_cells]
                + [(RunSpec(c, "v2", "qwen235", 2, val_reps=1), False) for c in pilot_cells])
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
    parser.add_argument("--generate-only", action="store_true",
                        help="only run the refiner (independent variants); score later with the refiner server down")
    args = parser.parse_args()
    mseeds = tuple(int(s) for s in args.mseeds.split(","))
    def has_p0(spec: RunSpec) -> bool:
        return (DEMOS / "results" / f"{spec.cell.optimizer}_{spec.cell.task}_seed{spec.cell.seed}.pkl").exists()

    # MPIR seed 1 for every variant first, then seeds 2 and 3, so the run can be stopped at any point.
    todo = sorted(((s, o) for s, o in specs_for(args.group, mseeds) if has_p0(s) and not s.path.exists()),
                  key=lambda item: item[0].mseed)
    print(f"{args.group}: {len(todo)} runs to do", flush=True)
    scorers: dict[str, Scorer] = {}

    def work(item):
        spec, oracle = item
        if args.generate_only:
            try:
                generate_independent(spec)
                print(f"GENERATED {spec.key}", flush=True)
            except Exception:
                print(f"FAILED {spec.key}\n{traceback.format_exc()}", flush=True)
            return
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
