"""Second-revision analyses of the NEW runs: format-preserving grid (v2f), positive and bare-tag controls, and the
single-session rescoring.

    python -m mpir_study.revision2_new       # writes results/revision/rev2_new.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from .analyze import REV, _lenient_vector, cell_of, load_runs, runs_where
from .data import load_p0
from .scoring import prompt_hash
from .mpir import protect_candidate  # noqa: F401  (documented dependency of the v2f candidates)
from .revision2 import BARE_TAG, clopper_pearson_upper
from .stats import paired_bootstrap
from .templates import ANSWER_FORMAT, EXEMPLAR_HEADER

OUT = REV / "rev2_new.json"
MDE_POINTS = 0.05  # the paper's own minimum detectable effect (Section 5.7)


def grid_v2f(runs: list[dict]) -> dict:
    """Format- and example-preserving chained refinement (v2f), 1.7B self-refiner, all cells that finished."""
    group = runs_where(runs, variant="v2f", refiner="1.7b")
    cells, rounds = {}, []
    for r in group:
        cell = cell_of(r)
        vs_start = paired_bootstrap(r["paired"]["p0"], r["paired"]["final"])
        vs_p0 = paired_bootstrap(r["paired_true_p0"], r["paired"]["final"])
        # lenient vectors of the start prompt (P0 + exemplar), the true P0 and the final prompt, from the session cache
        from .mpir import start_prompt, RunSpec  # local import: avoids a cycle at module import
        from .rescore import final_prompt
        spec = RunSpec(cell, "v2f")
        len_start = _lenient_vector(cell, prompt_hash(start_prompt(spec)))
        len_p0 = _lenient_vector(cell, prompt_hash(load_p0(cell)))
        len_fin = _lenient_vector(cell, prompt_hash(final_prompt(r)))
        lenient = None
        if len_start is not None and len_fin is not None:
            res = paired_bootstrap(len_start, len_fin)
            lenient = {"vs_start": res, "win_vs_start": res["diff"] > 0 and res["lo"] > 0,
                       "loss_vs_start": res["diff"] < 0 and res["hi"] < 0}
            if len_p0 is not None:
                res0 = paired_bootstrap(len_p0, len_fin)
                lenient.update({"vs_p0": res0, "win_vs_p0": res0["diff"] > 0 and res0["lo"] > 0})
        cells[cell.key] = {"lenient": lenient, "task": cell.task, "optimizer": cell.optimizer, "seed": cell.seed,
                           "accepted": not r["final_is_p0"], "start_test": r["p0_test"],
                           "true_p0_test": r["true_p0_test"], "final_test": r["final_test"],
                           "vs_start": vs_start, "vs_true_p0": vs_p0,
                           "win_vs_start": vs_start["diff"] > 0 and vs_start["lo"] > 0,
                           "win_vs_p0": vs_p0["diff"] > 0 and vs_p0["lo"] > 0}
        rounds += r["rounds"]
    counts = Counter(x["outcome"] for x in rounds)
    validated = [x for x in rounds if x["outcome"] == "validated"]
    raw = [x["raw_candidate"] for x in rounds if x.get("raw_candidate") and x["outcome"] != "echo"]
    n = len(group)
    log = REV.parent.parent / "logs" / "g5-fix.log"
    seconds = []
    if log.exists():
        import re
        seconds = [int(m.group(1)) for m in re.finditer(r"^DONE \S+ (\d+)s", log.read_text(encoding="utf-8"), re.M)
                   if int(m.group(1)) > 0]
    accepted = sum(c["accepted"] for c in cells.values())
    diffs = [c["vs_start"]["diff"] for c in cells.values()]
    diffs_p0 = [c["vs_true_p0"]["diff"] for c in cells.values()]
    return {
        "wall_clock_median_s": float(np.median(seconds)) if seconds else None,
        "wall_clock_total_s_shared_server": float(np.sum(seconds)) if seconds else None,
        "cells": cells, "runs": n, "accepted_runs": accepted,
        "accept_upper_one_sided_95": clopper_pearson_upper(accepted, n) if n else None,
        "rounds": len(rounds), **{k: counts.get(k, 0) for k in ("fallback", "echo", "duplicate", "validated")},
        "share_fallback_echo": (counts["fallback"] + counts["echo"]) / len(rounds) if rounds else None,
        "val_better": sum(1 for x in validated if x["val"] > x["baseline_val"]),
        "val_equal": sum(1 for x in validated if x["val"] == x["baseline_val"]),
        "val_worse": sum(1 for x in validated if x["val"] < x["baseline_val"]),
        "lenient_cells": sum(1 for c in cells.values() if c["lenient"]),
        "lenient_wins_vs_start": sum(1 for c in cells.values() if c["lenient"] and c["lenient"]["win_vs_start"]),
        "lenient_losses_vs_start": sum(1 for c in cells.values() if c["lenient"] and c["lenient"]["loss_vs_start"]),
        "lenient_wins_vs_p0": sum(1 for c in cells.values() if c["lenient"] and c["lenient"].get("win_vs_p0")),
        "lenient_mean_diff_vs_start": float(np.mean([c["lenient"]["vs_start"]["diff"] for c in cells.values()
                                                     if c["lenient"]] or [0])),
        "wins_vs_start": sum(c["win_vs_start"] for c in cells.values()),
        "wins_vs_p0": sum(c["win_vs_p0"] for c in cells.values()),
        "mean_diff_vs_start": float(np.mean(diffs)) if diffs else None,
        "mean_diff_vs_p0": float(np.mean(diffs_p0)) if diffs_p0 else None,
        "raw_keep_format_line": float(np.mean([ANSWER_FORMAT in t for t in raw])) if raw else None,
        "raw_keep_exemplar_header": float(np.mean([EXEMPLAR_HEADER in t for t in raw])) if raw else None,
        "raw_keep_bare_tag": float(np.mean([bool(BARE_TAG.search(t)) for t in raw])) if raw else None,
        "accepted_by_optimizer": dict(Counter(c["optimizer"] for c in cells.values() if c["accepted"])),
        "cells_by_optimizer": dict(Counter(c["optimizer"] for c in cells.values())),
    }


def controls_summary() -> dict:
    path = REV / "controls.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    rows = {}
    for key, v in data.items():
        row = {"optimizer": key.split("__")[1], "p0_val": v["p0"]["val"], "ex_val": v["exemplar"]["val"],
               "gate_accepts_exemplar": v["exemplar"]["val"] > v["p0"]["val"]}
        if "test" in v["p0"]:
            row.update({"p0_test": v["p0"]["test"], "ex_test": v["exemplar"]["test"],
                        "ex_vs_p0": paired_bootstrap(v["p0"]["test_vector"], v["exemplar"]["test_vector"])})
        if "bare_tag" in v:
            row.update({"tag_val": v["bare_tag"]["val"], "gate_accepts_bare_tag": v["bare_tag"]["val"] > v["p0"]["val"],
                        "tag_test": v["bare_tag"]["test"],
                        "tag_vs_p0": paired_bootstrap(v["p0"]["test_vector"], v["bare_tag"]["test_vector"])})
        rows[key] = row
    by_opt: dict[str, dict] = {}
    for opt in sorted({r["optimizer"] for r in rows.values()}):
        sel = [r for r in rows.values() if r["optimizer"] == opt]
        entry = {"cells": len(sel), "gate_accepts_exemplar": sum(r["gate_accepts_exemplar"] for r in sel)}
        if opt != "promptwizard":
            entry.update({
                "gate_accepts_bare_tag": sum(r["gate_accepts_bare_tag"] for r in sel),
                "exemplar_test_gain_mean": float(np.mean([r["ex_test"] - r["p0_test"] for r in sel])),
                "bare_tag_test_gain_mean": float(np.mean([r["tag_test"] - r["p0_test"] for r in sel])),
                "exemplar_wins": sum(r["ex_vs_p0"]["diff"] > 0 and r["ex_vs_p0"]["lo"] > 0 for r in sel),
                "bare_tag_wins": sum(r["tag_vs_p0"]["diff"] > 0 and r["tag_vs_p0"]["lo"] > 0 for r in sel)})
            better = [r for r in sel if r["ex_test"] > r["p0_test"]]
            entry["exemplar_better_on_test"] = len(better)
            entry["exemplar_better_and_accepted"] = sum(r["gate_accepts_exemplar"] for r in better)
            entry["exemplar_not_better_but_accepted"] = sum(r["gate_accepts_exemplar"] and r["ex_test"] <= r["p0_test"]
                                                            for r in sel)
        by_opt[opt] = entry
    allr = list(rows.values())
    known = [r for r in allr if "ex_test" in r and r["ex_test"] > r["p0_test"]]
    return {"cells": rows, "by_optimizer": by_opt, "n": len(allr),
            "gate_accepts_exemplar": sum(r["gate_accepts_exemplar"] for r in allr),
            "known_good_cells": len(known), "known_good_accepted": sum(r["gate_accepts_exemplar"] for r in known)}


def rescore_summary(runs: list[dict]) -> dict:
    """Original versus single-session rescored test accuracy of the exploratory strong-refiner finals."""
    path = REV / "rescore.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out: dict = {"groups": {}}
    by_label: dict[str, dict] = defaultdict(dict)
    for key, v in data.items():
        cell, label = key.split("|", 1)
        by_label[label][cell] = v
    p0 = by_label.get("p0", {})
    for label, cells in by_label.items():
        if label == "p0":
            continue
        refiner, mseed = label.split("|")[1], int(label.split("|")[2][1:])
        variant = label.split("|")[0]
        rows = []
        for cell, v in cells.items():
            if cell not in p0:
                continue
            orig = [r for r in runs_where(runs, variant=variant, refiner=refiner, mseed=mseed)
                    if cell_of(r).key == cell][0]
            orig_diff = orig["final_test"] - orig["p0_test"]
            new_final = np.mean([np.mean(vec) for vec in v["test"]])
            new_p0 = np.mean([np.mean(vec) for vec in p0[cell]["test"]])
            fin = np.mean(np.asarray(v["test"], dtype=float), axis=0)
            base = np.mean(np.asarray(p0[cell]["test"], dtype=float), axis=0)
            res = paired_bootstrap(base, fin) if not (v["hash"] == p0[cell]["hash"]) else None
            rows.append({"cell": cell, "orig_diff": orig_diff, "rescored_diff": float(new_final - new_p0),
                         "same_prompt_as_p0": v["hash"] == p0[cell]["hash"],
                         "rescored_ci": res})
        diffs_o = [r["orig_diff"] for r in rows]
        diffs_n = [r["rescored_diff"] for r in rows]
        out["groups"][label] = {
            "cells": len(rows), "orig_mean": float(np.mean(diffs_o)), "rescored_mean": float(np.mean(diffs_n)),
            "orig_cells_gt_mde": sum(d >= MDE_POINTS for d in diffs_o),
            "rescored_cells_gt_mde": sum(d >= MDE_POINTS for d in diffs_n),
            "orig_cells_pos": sum(d > 0 for d in diffs_o), "rescored_cells_pos": sum(d > 0 for d in diffs_n),
            "max_abs_change": float(max(abs(a - b) for a, b in zip(diffs_o, diffs_n))), "rows": rows}
    # scoring-to-scoring flips within the session (rep 0 vs rep 1 of the same prompt)
    flips = []
    for v in data.values():
        if len(v["test"]) >= 2:
            a, b = np.asarray(v["test"][0]), np.asarray(v["test"][1])
            flips.append(float(np.mean(a != b)))
    out["within_session_flip_rate"] = {"mean": float(np.mean(flips)) if flips else None,
                                       "max": float(max(flips)) if flips else None, "n_prompts": len(flips)}
    return out


def main() -> None:
    runs = load_runs()
    result = {"grid_v2f": grid_v2f(runs), "controls": controls_summary(), "rescore": rescore_summary(runs)}
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
