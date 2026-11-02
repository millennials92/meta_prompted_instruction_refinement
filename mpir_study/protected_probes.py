"""Analysis of the protected-treatment follow-up: V3F (rubric-free, same protection as V2F) with the 1.7B refiner on the
V2F grid, and V2F / V3F with the strong refiners (Qwen3-235B, GPT-6 Luna) on the nine Tier-1 seed-42 cells.

    python -m mpir_study.protected_probes     # writes results/revision/rev2_protected_probes.json
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from .analyze import REV, _lenient_vector, cell_of, load_runs, runs_where
from .data import tier1_cells
from .stats import paired_bootstrap

OUT = REV / "rev2_protected_probes.json"
MDE = 0.05


def _lenient_pair(run: dict):
    cell = cell_of(run)
    return _lenient_vector(cell, run["p0_hash"]), _lenient_vector(cell, run["final_hash"])


def summarize(runs: list[dict], variant: str, refiner: str, cells_filter=None) -> dict:
    group = [r for r in runs_where(runs, variant=variant, refiner=refiner, mseed=1)
             if cells_filter is None or cell_of(r) in cells_filter]
    rounds = [x for r in group for x in r["rounds"]]
    counts = Counter(x["outcome"] for x in rounds)
    validated = [x for x in rounds if x["outcome"] == "validated"]
    cells, vecs = {}, {}
    for r in group:
        strict = paired_bootstrap(r["paired"]["p0"], r["paired"]["final"])
        l0, l1 = _lenient_pair(r)
        lenient = paired_bootstrap(l0, l1) if l0 is not None and l1 is not None else None
        cells[cell_of(r).key] = {"strict": strict, "lenient": lenient, "accepted": not r["final_is_p0"],
                                 "start_test": r["p0_test"], "final_test": r["final_test"]}
        vecs[cell_of(r).key] = (np.asarray(r["paired"]["final"], float), None if l1 is None else np.asarray(l1, float))
    sd = [c["strict"]["diff"] for c in cells.values()]
    ld = [c["lenient"]["diff"] for c in cells.values() if c["lenient"]]
    return {"runs": len(group), "rounds": len(rounds),
            **{k: counts.get(k, 0) for k in ("fallback", "echo", "duplicate", "validated")},
            "val_better": sum(1 for x in validated if x["val"] > x["baseline_val"]),
            "accepted_runs": sum(1 for r in group if not r["final_is_p0"]),
            "strict_wins": sum(c["strict"]["diff"] > 0 and c["strict"]["lo"] > 0 for c in cells.values()),
            "strict_mean": float(np.mean(sd)) if sd else None,
            "cells_ge_mde": sum(d >= MDE for d in sd),
            "lenient_wins": sum(1 for c in cells.values() if c["lenient"] and c["lenient"]["diff"] > 0 and c["lenient"]["lo"] > 0),
            "lenient_losses": sum(1 for c in cells.values() if c["lenient"] and c["lenient"]["diff"] < 0 and c["lenient"]["hi"] < 0),
            "lenient_mean": float(np.mean(ld)) if ld else None, "cells": cells}


def paired_v2f_v3f(runs: list[dict], refiner: str, cells_filter=None) -> dict:
    """Per-cell V2F minus V3F on the final prompts (strict and lenient), over the cells both completed."""
    a = {cell_of(r).key: r for r in runs_where(runs, variant="v2f", refiner=refiner, mseed=1)
         if cells_filter is None or cell_of(r) in cells_filter}
    b = {cell_of(r).key: r for r in runs_where(runs, variant="v3f", refiner=refiner, mseed=1)
         if cells_filter is None or cell_of(r) in cells_filter}
    keys = sorted(set(a) & set(b))
    strict, lenient = [], []
    for k in keys:
        strict.append(paired_bootstrap(b[k]["paired"]["final"], a[k]["paired"]["final"]))
        l0 = _lenient_vector(cell_of(a[k]), a[k]["final_hash"])
        l1 = _lenient_vector(cell_of(b[k]), b[k]["final_hash"])
        lenient.append(paired_bootstrap(l1, l0) if l0 is not None and l1 is not None else None)
    out = {"n": len(keys)}
    if not keys:
        return out
    rng = np.random.default_rng(20261019)
    for name, rows in (("strict", strict), ("lenient", [x for x in lenient if x])):
        d = np.array([x["diff"] for x in rows])
        if not len(d):
            continue
        boot = rng.choice(d, size=(10_000, len(d)), replace=True).mean(axis=1)
        out[name] = {"mean": float(d.mean()), "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
                     "v2f_better": int((d > 0).sum()), "v3f_better": int((d < 0).sum()), "tied": int((d == 0).sum()),
                     "ci_excludes_zero": int(sum(1 for x in rows if x["lo"] > 0 or x["hi"] < 0)), "cells": len(d)}
    return out


def main() -> None:
    runs = load_runs()
    tier1 = set(tier1_cells(seeds=(42,)))
    result = {"v3f_1.7b": summarize(runs, "v3f", "1.7b"), "v2f_1.7b": summarize(runs, "v2f", "1.7b"),
              "paired_1.7b_all": paired_v2f_v3f(runs, "1.7b"),
              "paired_1.7b_tier1": paired_v2f_v3f(runs, "1.7b", tier1)}
    for refiner in ("qwen235", "gpt6luna"):
        for variant in ("v2f", "v3f"):
            result[f"{variant}_{refiner}"] = summarize(runs, variant, refiner)
        result[f"paired_{refiner}"] = paired_v2f_v3f(runs, refiner)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
