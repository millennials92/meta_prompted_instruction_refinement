"""Analysis of the intermediate open-weight refiner (Qwen3-8B-AWQ, local) with V2 (rubric) and V3 (rubric-free).

    python -m mpir_study.mid_refiner     # writes results/revision/rev2_mid_refiner.json

Same nine Tier-1 seed-42 cells and MPIR seed 1 as the strong-refiner probes, so V2 - V3 is paired per cell.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from .analyze import REV, _lenient_vector, cell_of, load_runs, runs_where
from .data import tier1_cells
from .rescore import final_prompt
from .scoring import prompt_hash
from .stats import paired_bootstrap

OUT = REV / "rev2_mid_refiner.json"
MDE = 0.05


def _lenient_final(run: dict):
    cell = cell_of(run)
    return _lenient_vector(cell, run["final_hash"]), _lenient_vector(cell, run["p0_hash"])


def summarize(runs: list[dict], variant: str) -> dict:
    group = [r for r in runs_where(runs, variant=variant, refiner="8b", mseed=1) if r["spec"]["cell"]["seed"] == 42]
    rounds = [x for r in group for x in r["rounds"]]
    counts = Counter(x["outcome"] for x in rounds)
    validated = [x for x in rounds if x["outcome"] == "validated"]
    cells = {}
    for r in group:
        res = paired_bootstrap(r["paired"]["p0"], r["paired"]["final"])
        len_final, len_p0 = _lenient_final(r)
        lenient = paired_bootstrap(len_p0, len_final) if len_final is not None and len_p0 is not None else None
        cells[cell_of(r).key] = {"strict": res, "lenient": lenient, "accepted": not r["final_is_p0"]}
    diffs = [c["strict"]["diff"] for c in cells.values()]
    ldiffs = [c["lenient"]["diff"] for c in cells.values() if c["lenient"]]
    return {"runs": len(group), "rounds": len(rounds), **{k: counts.get(k, 0) for k in
                                                          ("fallback", "echo", "duplicate", "validated")},
            "val_better": sum(1 for x in validated if x["val"] > x["baseline_val"]),
            "accepted_runs": sum(1 for r in group if not r["final_is_p0"]),
            "strict_wins": sum(c["strict"]["diff"] > 0 and c["strict"]["lo"] > 0 for c in cells.values()),
            "strict_mean": float(np.mean(diffs)) if diffs else None,
            "cells_ge_mde": sum(d >= MDE for d in diffs),
            "lenient_wins": sum(1 for c in cells.values() if c["lenient"] and c["lenient"]["diff"] > 0
                                and c["lenient"]["lo"] > 0),
            "lenient_losses": sum(1 for c in cells.values() if c["lenient"] and c["lenient"]["diff"] < 0
                                  and c["lenient"]["hi"] < 0),
            "lenient_mean": float(np.mean(ldiffs)) if ldiffs else None, "cells": cells,
            "raw_keep_format": float(np.mean([("Wrap the final answer" in (x.get("raw_candidate") or ""))
                                              for x in rounds if x.get("raw_candidate")] or [0]))}


def paired_v2_v3(runs: list[dict]) -> dict:
    out, diffs = {}, []
    for c in tier1_cells(seeds=(42,)):
        v2 = [r for r in runs_where(runs, variant="v2", refiner="8b", mseed=1) if cell_of(r) == c]
        v3 = [r for r in runs_where(runs, variant="v3", refiner="8b", mseed=1) if cell_of(r) == c]
        if v2 and v3:
            res = paired_bootstrap(v3[0]["paired"]["final"], v2[0]["paired"]["final"])
            out[c.key] = res
            diffs.append(res["diff"])
    if not diffs:
        return {}
    rng = np.random.default_rng(20261019)
    boot = rng.choice(np.asarray(diffs), size=(10_000, len(diffs)), replace=True).mean(axis=1)
    return {"cells": out, "n": len(diffs), "mean": float(np.mean(diffs)), "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)), "v2_better": int(sum(d > 0 for d in diffs)),
            "v3_better": int(sum(d < 0 for d in diffs)), "tied": int(sum(d == 0 for d in diffs))}


def main() -> None:
    runs = load_runs()
    result = {"v2": summarize(runs, "v2"), "v3": summarize(runs, "v3"), "paired": paired_v2_v3(runs)}
    if result["v2"]["runs"] and result["v3"]["runs"]:
        v2, v3, pr = result["v2"], result["v3"], result["paired"]
        pp = lambda x: f"{100 * x:+.1f}"
        result["abstract_sentence"] = (
            f"An intermediate open-weight refiner (Qwen3-8B) gave a mean strict change of {pp(v2['strict_mean'])} "
            f"points with the rubric and {pp(v3['strict_mean'])} without it.")
        result["response_sentence"] = (
            f"We added a local intermediate open-weight refiner, Qwen3-8B-AWQ, with the rubric-guided (V2) and the "
            f"rubric-free (V3) variants on the nine Tier-1 seed-42 cells: V2 changed the prompt in {v2['accepted_runs']} "
            f"of {v2['runs']} runs (strict mean {pp(v2['strict_mean'])} points, {v2['lenient_wins']} lenient wins and "
            f"{v2['lenient_losses']} lenient losses) and V3 in {v3['accepted_runs']} of {v3['runs']} (strict mean "
            f"{pp(v3['strict_mean'])}); the paired V2-V3 mean is {pp(pr['mean'])} points (interval "
            f"[{pp(pr['lo'])}, {pp(pr['hi'])}]) (Section 5.6.2).")
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
