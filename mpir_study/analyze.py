"""Aggregate every revision experiment into results/revision/summary_numbers.json (read by the manuscript).

Run:  python -m mpir_study.analyze
Every analysis tolerates missing inputs (a section is simply absent), so it can be re-run while the
experiments are still going.
"""
from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .answers import is_correct_lenient
from .data import Cell, OPTIMIZERS, TIER1_TASKS, TIER2_TASKS
from .stats import hierarchical_bootstrap, is_win, minimum_detectable_effect, paired_bootstrap, spearman
from .templates import REPO

REV = REPO / "results" / "revision"
CACHE = REV / "score_cache"
OUTCOMES = ("fallback", "echo", "duplicate", "validated")


# ---------------------------------------------------------------------------------------------- loading
def load_runs() -> list[dict]:
    runs = []
    for path in sorted(glob.glob(str(REV / "rounds" / "*.json"))):
        runs.append(json.loads(Path(path).read_text(encoding="utf-8")))
    return runs


def run_key(run: dict) -> tuple:
    s = run["spec"]
    c = s["cell"]
    return (c["task"], c["optimizer"], c["seed"], s["variant"], s["refiner"], s["mseed"])


_CACHE_MEMO: dict[str, dict] = {}


def cell_cache(cell_key: str, with_outputs: bool = False) -> dict:
    """(hash, split, rep) -> tuple of per-example correctness (and outputs if requested)."""
    memo_key = f"{cell_key}:{with_outputs}"
    if memo_key not in _CACHE_MEMO:
        out = {}
        path = CACHE / f"{cell_key}.jsonl"
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    out[(row["hash"], row["split"], row["rep"])] = (
                        (tuple(row["correct"]), tuple(row["outputs"])) if with_outputs else tuple(row["correct"]))
        _CACHE_MEMO[memo_key] = out
    return _CACHE_MEMO[memo_key]


def cell_of(run: dict) -> Cell:
    c = run["spec"]["cell"]
    return Cell(c["task"], c["optimizer"], c["seed"])


def runs_where(runs: Iterable[dict], **kw) -> list[dict]:
    keep = []
    for r in runs:
        s = r["spec"]
        if all((s[k] == v) for k, v in kw.items()):
            keep.append(r)
    return keep


# ---------------------------------------------------------------------------------------------- A1 composition
def composition(runs: list[dict]) -> dict:
    """Outcome counts per variant/refiner, with validated candidates split by validation score vs baseline."""
    table = {}
    for (variant, refiner) in sorted({(r["spec"]["variant"], r["spec"]["refiner"]) for r in runs}):
        group = runs_where(runs, variant=variant, refiner=refiner)
        rounds = [x for r in group for x in r["rounds"]]
        counts = Counter(x["outcome"] for x in rounds)
        validated = [x for x in rounds if x["outcome"] == "validated"]
        better = sum(1 for x in validated if x["val"] > x["baseline_val"])
        equal = sum(1 for x in validated if x["val"] == x["baseline_val"])
        worse = sum(1 for x in validated if x["val"] < x["baseline_val"])
        distinct = [len({x["cand_hash"] for x in r["rounds"] if x["cand_hash"]}) for r in group]
        table[f"{variant}|{refiner}"] = {
            "runs": len(group), "rounds": len(rounds), **{o: counts.get(o, 0) for o in OUTCOMES},
            "accepted": sum(1 for x in rounds if x["accepted"]),
            "val_better": better, "val_equal": equal, "val_worse": worse,
            "mean_distinct_candidates": float(np.mean(distinct)) if distinct else 0.0,
            "runs_with_acceptance": sum(1 for r in group if not r["final_is_p0"]),
        }
    return table


# ---------------------------------------------------------------------------------------------- A2 main grid
def cell_difference(runs_for_cell: list[dict], cell_key: str) -> dict:
    """Paired difference (final - P0) pooled over the MPIR seeds of one cell: per-example mean of the
    per-seed differences, bootstrapped over examples."""
    p0 = np.asarray(runs_for_cell[0]["paired"]["p0"], dtype=float)
    finals = np.mean([np.asarray(r["paired"]["final"], dtype=float) for r in runs_for_cell], axis=0)
    x = finals - p0
    n = len(x)
    rng = np.random.default_rng(20261019)
    if not np.any(x):
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "n": n, "accepted_runs": 0, "seeds": len(runs_for_cell)}
    idx = rng.integers(0, n, size=(10_000, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": n,
            "accepted_runs": sum(1 for r in runs_for_cell if not r["final_is_p0"]), "seeds": len(runs_for_cell)}


def main_grid(runs: list[dict], variant="v2", refiner="1.7b") -> dict:
    group = runs_where(runs, variant=variant, refiner=refiner)
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in group:
        by_cell[cell_of(r).key].append(r)
    cells = {}
    for key, rs in sorted(by_cell.items()):
        cells[key] = cell_difference(rs, key)
        cells[key]["task"], cells[key]["optimizer"], cells[key]["seed"] = (
            rs[0]["spec"]["cell"]["task"], rs[0]["spec"]["cell"]["optimizer"], rs[0]["spec"]["cell"]["seed"])
        cells[key]["p0_test"], cells[key]["final_test_mean"] = rs[0]["p0_test"], float(np.mean([r["final_test"] for r in rs]))
        cells[key]["win"] = bool(cells[key]["diff"] > 0 and cells[key]["lo"] > 0)
    wins = sum(c["win"] for c in cells.values())
    diffs = [c["diff"] for c in cells.values()]
    by_task: dict[str, list[float]] = defaultdict(list)
    for c in cells.values():
        by_task[c["task"]].append(c["diff"])
    n_cells = len(cells)
    mean_diff = float(np.mean(diffs)) if diffs else 0.0
    verdict = ("reproduces" if (wins >= n_cells / 3 and mean_diff >= 0.02) else
               "null" if (wins <= 2 * n_cells / 45 and abs(mean_diff) <= 0.01) else "mixed") if n_cells else "n/a"
    return {"cells": cells, "n_cells": n_cells, "wins": wins, "mean_diff": mean_diff,
            "hier": hierarchical_bootstrap(by_task) if by_task else None,
            "cells_any_acceptance": sum(1 for c in cells.values() if c["accepted_runs"] > 0),
            "runs": len(group), "runs_accepting": sum(1 for r in group if not r["final_is_p0"]),
            "verdict": verdict}


# ---------------------------------------------------------------------------------------------- A3 oracle / gate
def candidate_table(runs: list[dict]) -> list[dict]:
    """One row per validated candidate with oracle test accuracy (runs scored with oracle=True)."""
    rows = []
    for r in runs:
        cell = cell_of(r)
        cache = cell_cache(cell.key)
        p0_hash = r["p0_hash"]
        p0_test = cache.get((p0_hash, "test", 0))
        for x in r["rounds"]:
            if x["outcome"] != "validated" or "test" not in x:
                continue
            test_correct = cache.get((x["cand_hash"], "test", 0))
            rows.append({"run": "|".join(map(str, run_key(r))), "cell": cell.key, "task": cell.task,
                         "optimizer": cell.optimizer, "variant": r["spec"]["variant"],
                         "refiner": r["spec"]["refiner"], "mseed": r["spec"]["mseed"], "round": x["round"],
                         "val": x["val"], "val_reps": x["val_reps"], "baseline_val": x["baseline_val"],
                         "test": x["test"], "p0_test": r["p0_test"], "judge_total": x["judge_total"],
                         "cand_hash": x["cand_hash"], "p0_hash": p0_hash,
                         "test_correct": test_correct, "p0_test_correct": p0_test})
    return rows


def oracle_summary(rows: list[dict], variant: str, refiner: str = "1.7b") -> dict:
    """Per-cell best candidate (by test) minus P0, ungated mean candidate delta and acceptance-rule variants."""
    sel = [r for r in rows if r["variant"] == variant and r["refiner"] == refiner]
    per_cell = {}
    for cell in sorted({r["cell"] for r in sel}):
        c = [r for r in sel if r["cell"] == cell]
        deltas = [r["test"] - r["p0_test"] for r in c]
        best = max(c, key=lambda r: r["test"])
        per_cell[cell] = {"n_candidates": len(c), "best_delta": best["test"] - best["p0_test"],
                          "mean_delta": float(np.mean(deltas)), "frac_better": float(np.mean([d > 0 for d in deltas])),
                          "frac_worse": float(np.mean([d < 0 for d in deltas]))}
    return {"cells": per_cell,
            "cells_best_ge_3pts": sum(1 for v in per_cell.values() if v["best_delta"] >= 0.03),
            "mean_best_delta": float(np.mean([v["best_delta"] for v in per_cell.values()])) if per_cell else None,
            "mean_candidate_delta": float(np.mean([r["test"] - r["p0_test"] for r in sel])) if sel else None,
            "n_candidates": len(sel)}


def acceptance_rules(rows: list[dict], variant: str, refiner: str = "1.7b") -> dict:
    """Re-apply alternative gates to the oracle candidates of one (variant, refiner) offline, per cell.

    Rules: strict (as run), non-strict (>=; the best-scoring tie wins), repeated-mean (mean of the three
    validation scorings vs. the mean of P0's three), larger gate (25 validation + half of test; accuracy
    reported on the other half). Selection is over all candidates of the cell (MPIR seed 1)."""
    sel = [r for r in rows if r["variant"] == variant and r["refiner"] == refiner and r["mseed"] == 1]
    out = {}
    for cell in sorted({r["cell"] for r in sel}):
        c = [r for r in sel if r["cell"] == cell]
        cache = cell_cache(cell)
        p0_hash = c[0]["p0_hash"]
        p0_val_reps = [np.mean(cache[(p0_hash, "val", k)]) for k in range(3) if (p0_hash, "val", k) in cache]
        p0_test = np.asarray(c[0]["p0_test_correct"], dtype=float)
        n = len(p0_test)
        half_a, half_b = np.arange(n) % 2 == 0, np.arange(n) % 2 == 1
        p0_val = np.asarray(cache[(p0_hash, "val", 0)], dtype=float)

        def pick(score_fn, strict=True):
            scored = [(score_fn(r), r) for r in c]
            base = score_fn(None)
            best_score, best = max(scored, key=lambda t: t[0])
            ok = best_score > base if strict else best_score >= base
            return best if ok else None

        res = {}
        base_val = c[0]["baseline_val"]
        for name, fn, strict in [
            ("strict", lambda r: (r["val"] if r else base_val), True),
            ("nonstrict", lambda r: (r["val"] if r else base_val), False),
            ("repeated", lambda r: (float(np.mean(r["val_reps"])) if r else float(np.mean(p0_val_reps))), True),
        ]:
            chosen = pick(fn, strict)
            res[name] = {"accepted": chosen is not None,
                         "delta": (chosen["test"] - chosen["p0_test"]) if chosen else 0.0}

        def gate_score(r):
            cand = np.asarray(r["test_correct"], dtype=float) if r else p0_test
            val = np.asarray(cache[(r["cand_hash"], "val", 0)], dtype=float) if r else p0_val
            return (val.sum() + cand[half_a].sum()) / (len(val) + half_a.sum())

        chosen = pick(gate_score, True)
        if chosen:
            cand = np.asarray(chosen["test_correct"], dtype=float)
            res["large_gate"] = {"accepted": True, "delta": float(cand[half_b].mean() - p0_test[half_b].mean())}
        else:
            res["large_gate"] = {"accepted": False, "delta": 0.0}
        out[cell] = res
    summary = {}
    for name in ("strict", "nonstrict", "repeated", "large_gate"):
        vals = [v[name] for v in out.values() if name in v]
        summary[name] = {"cells": len(vals), "accepted_cells": sum(v["accepted"] for v in vals),
                         "mean_delta": float(np.mean([v["delta"] for v in vals])) if vals else None,
                         "cells_worse": sum(1 for v in vals if v["delta"] < 0),
                         "cells_better": sum(1 for v in vals if v["delta"] > 0)}
    return {"per_cell": out, "summary": summary}


def val_test_correlation(rows: list[dict]) -> dict:
    val = [r["val"] for r in rows]
    test = [r["test"] - r["p0_test"] for r in rows]
    val_rel = [r["val"] - r["baseline_val"] for r in rows]
    per_cell = {}
    for cell in sorted({r["cell"] for r in rows}):
        c = [r for r in rows if r["cell"] == cell]
        per_cell[cell] = spearman([r["val"] - r["baseline_val"] for r in c], [r["test"] - r["p0_test"] for r in c])
    valid = [v for v in per_cell.values() if v is not None]
    return {"pooled_spearman": spearman(val_rel, test), "n": len(rows),
            "median_within_cell": float(np.median(valid)) if valid else None, "cells": len(valid)}


def legacy_acceptance(runs: list[dict], variant="v1", refiner="1.7b") -> dict:
    """Quantify the original defect (A_best = -inf): the round-1 outcome is accepted unconditionally.

    Uses the oracle-scored independent-variant logs of MPIR seed 1. A fallback or echo round yields P0
    itself (the pre-fix code kept the current prompt), a validated round yields its candidate; later rounds
    replace it only if strictly better on validation than the best so far."""
    out = {}
    for r in runs_where(runs, variant=variant, refiner=refiner, mseed=1):
        rounds = r["rounds"]
        if any(x["outcome"] == "validated" and "test" not in x for x in rounds):
            continue
        base_val, base_test = r["baseline_val"], r["p0_test"]
        best_val, best_test, accepted_first = None, None, None
        for i, x in enumerate(rounds):
            if x["outcome"] == "validated":
                val, test = x["val"], x["test"]
            else:
                val, test = base_val, base_test
            if best_val is None or val > best_val:
                best_val, best_test = val, test
                if i == 0:
                    accepted_first = x["outcome"]
        out[cell_of(r).key] = {"legacy_test": best_test, "p0_test": base_test, "gated_test": r["final_test"],
                               "first_round_outcome": rounds[0]["outcome"]}
    deltas = [v["legacy_test"] - v["p0_test"] for v in out.values()]
    gated = [v["gated_test"] - v["p0_test"] for v in out.values()]
    return {"cells": out, "n": len(out), "mean_legacy_delta": float(np.mean(deltas)) if deltas else None,
            "mean_gated_delta": float(np.mean(gated)) if gated else None,
            "legacy_worse": sum(1 for d in deltas if d < 0), "legacy_better": sum(1 for d in deltas if d > 0),
            "legacy_changed_prompt": sum(1 for v in out.values() if v["legacy_test"] != v["p0_test"])}


# ---------------------------------------------------------------------------------------------- A5 capability
def capability(runs: list[dict]) -> dict:
    out = {}
    for refiner in ("1.7b", "4b", "8b", "qwen235", "gpt6luna"):
        for variant in ("v1", "v2"):
            group = [r for r in runs_where(runs, variant=variant, refiner=refiner) if r["spec"]["cell"]["seed"] == 42
                     and r["spec"]["cell"]["task"] in TIER1_TASKS]
            if not group:
                continue
            by_cell: dict[str, list[dict]] = defaultdict(list)
            for r in group:
                by_cell[cell_of(r).key].append(r)
            cells = {k: cell_difference(v, k) for k, v in by_cell.items()}
            wins = sum(1 for c in cells.values() if c["diff"] > 0 and c["lo"] > 0)
            out[f"{variant}|{refiner}"] = {
                "cells": len(cells), "wins": wins, "mean_diff": float(np.mean([c["diff"] for c in cells.values()])),
                "runs": len(group), "runs_accepting": sum(1 for r in group if not r["final_is_p0"]),
                "per_cell": {k: {"diff": v["diff"], "lo": v["lo"], "hi": v["hi"]} for k, v in cells.items()}}
    return out


# ---------------------------------------------------------------------------------------------- A7 noise
def noise_floor() -> dict:
    path = REV / "noise_floor.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    cells = {}
    for key, v in data.items():
        t = v["test"]
        cells[key] = {"sd": float(np.std(t["accs"], ddof=1)), "range": float(max(t["accs"]) - min(t["accs"])),
                      "discordance": t["discordance_mean"], "n": t["n"], "accs": t["accs"],
                      "val_range": float(max(v["val"]["accs"]) - min(v["val"]["accs"]))}
    disc = [c["discordance"] for c in cells.values()]
    mean_disc = float(np.mean(disc))
    n_typ = int(np.median([c["n"] for c in cells.values()]))
    return {"cells": cells, "mean_discordance": mean_disc, "max_discordance": float(max(disc)),
            "mean_sd": float(np.mean([c["sd"] for c in cells.values()])),
            "mean_range": float(np.mean([c["range"] for c in cells.values()])),
            "mde_noise_only": minimum_detectable_effect(n_typ, mean_disc),
            "mde_for_discordance": {str(d): minimum_detectable_effect(n_typ, d) for d in (0.05, 0.10, 0.20)},
            "n_typical": n_typ}


# ---------------------------------------------------------------------------------------------- A9 parsing / exemplar
def p0_parsing(cells: Iterable[Cell]) -> dict:
    """Strict vs lenient accuracy of every P0 on test, from the cached raw outputs."""
    from .data import load_p0
    from .scoring import prompt_hash
    from .data import load_split
    out = {}
    for cell in cells:
        try:
            p0 = load_p0(cell)
        except FileNotFoundError:
            continue
        cache = cell_cache(cell.key, with_outputs=True)
        hit = cache.get((prompt_hash(p0), "test", 0))
        if hit is None:
            continue
        correct, outputs = hit
        gold = [ex["answer"] for ex in load_split(cell.task, cell.seed, "test")]
        lenient = [is_correct_lenient(o, g) for o, g in zip(outputs, gold)]
        out[cell.key] = {"strict": float(np.mean(correct)), "lenient": float(np.mean(lenient)), "n": len(correct),
                         "optimizer": cell.optimizer, "task": cell.task, "seed": cell.seed}
    return out


def exemplar() -> dict:
    path = REV / "exemplar_intervention.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    cells = {}
    for key, v in data.items():
        r = paired_bootstrap(v["paired"]["p0"], v["paired"]["ext"])
        cells[key] = {"p0": v["p0"], "with_exemplar": v["with_exemplar"], "diff": r["diff"], "lo": r["lo"], "hi": r["hi"]}
    by_opt = defaultdict(list)
    for key, v in cells.items():
        by_opt[key.split("__")[1]].append(v["with_exemplar"] - v["p0"])
    return {"cells": cells, "mean_gain_by_optimizer": {k: float(np.mean(v)) for k, v in by_opt.items()},
            "n": len(cells)}


# ---------------------------------------------------------------------------------------------- A6 judge
def judge_repeatability(runs: list[dict]) -> dict:
    """The independent variants re-judge the same prompt P0 every round: spread of those totals."""
    sds, parsed, total = [], 0, 0
    for r in runs_where(runs, variant="v1", refiner="1.7b"):
        totals = [x["judge_total"] for x in r["rounds"]]
        total += len(totals)
        parsed += sum(t is not None for t in totals)
        ok = [t for t in totals if t is not None]
        if len(ok) >= 3:
            sds.append(float(np.std(ok, ddof=1)))
    return {"mean_sd_across_rounds": float(np.mean(sds)) if sds else None, "parse_rate": parsed / total if total else None,
            "n_runs": len(sds)}


def judge_validity(rows: list[dict]) -> dict:
    """Spearman correlation of rubric totals with accuracy (within cells), and agreement between judges."""
    path = REV / "judge_scores.json"
    if not path.exists():
        return {}
    scores = json.loads(path.read_text())
    out: dict = {"judges": {}}
    for judge, table in scores.items():
        usable = {h: v["total"] for h, v in table.items() if v["total"] is not None}
        parse_rate = len(usable) / len(table) if table else None
        per_cell_test, per_cell_val, pooled = [], [], {"x": [], "test": [], "val": []}
        for cell in sorted({r["cell"] for r in rows}):
            c = {r["cand_hash"]: r for r in rows if r["cell"] == cell}
            hashes = [h for h in c if h in usable]
            if len(hashes) < 4:
                continue
            x = [usable[h] for h in hashes]
            per_cell_test.append(spearman(x, [c[h]["test"] for h in hashes]))
            per_cell_val.append(spearman(x, [c[h]["val"] for h in hashes]))
            pooled["x"] += [usable[h] - np.mean(x) for h in hashes]
            pooled["test"] += [c[h]["test"] - np.mean([c[k]["test"] for k in hashes]) for h in hashes]
        valid_t = [v for v in per_cell_test if v is not None]
        valid_v = [v for v in per_cell_val if v is not None]
        out["judges"][judge] = {
            "n_prompts": len(table), "parse_rate": parse_rate,
            "mean_total": float(np.mean(list(usable.values()))) if usable else None,
            "sd_total": float(np.std(list(usable.values()), ddof=1)) if len(usable) > 1 else None,
            "median_spearman_test_within_cell": float(np.median(valid_t)) if valid_t else None,
            "median_spearman_val_within_cell": float(np.median(valid_v)) if valid_v else None,
            "pooled_centred_spearman_test": spearman(pooled["x"], pooled["test"]),
            "cells_used": len(valid_t)}
    judges = list(scores)
    if len(judges) >= 2:
        a, b = scores[judges[0]], scores[judges[1]]
        shared = [h for h in a if h in b and a[h]["total"] is not None and b[h]["total"] is not None]
        out["agreement"] = {"judges": judges[:2], "n": len(shared),
                            "spearman": spearman([a[h]["total"] for h in shared], [b[h]["total"] for h in shared])}
    return out


# ---------------------------------------------------------------------------------------------- tables for the paper
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(centre - half), float(centre + half)


def p0_landscape(cells: Iterable[Cell]) -> dict:
    """Strict and lenient accuracy of every P0 with Wilson intervals and the chance level of its task."""
    from .data import load_p0, load_split
    from .scoring import prompt_hash
    out = {}
    for cell in cells:
        try:
            p0 = load_p0(cell)
        except FileNotFoundError:
            continue
        hit = cell_cache(cell.key, with_outputs=True).get((prompt_hash(p0), "test", 0))
        if hit is None:
            continue
        correct, outputs = hit
        gold = [ex["answer"] for ex in load_split(cell.task, cell.seed, "test")]
        lenient = [is_correct_lenient(o, g) for o, g in zip(outputs, gold)]
        k, n = int(sum(correct)), len(correct)
        lo, hi = wilson(k, n)
        out[cell.key] = {"task": cell.task, "optimizer": cell.optimizer, "seed": cell.seed, "n": n,
                         "strict": k / n, "lo": lo, "hi": hi, "lenient": float(np.mean(lenient)),
                         "chance": 1 / len(set(gold)), "with_delims": float(np.mean(["<ANS_START>" in o and "<ANS_END>" in o for o in outputs]))}
    return out


def reference_conditions() -> dict:
    """Pilot reference conditions (zero-shot, expert few-shot, ungated free rewrite), seed 42, with intervals."""
    path_dir = REPO / "demos" / "results" / "predictions"
    out = {}
    for task in TIER1_TASKS:
        for cond in ("zero_shot_cot", "expert_few_shot_cot", "free_rewrite"):
            path = path_dir / f"{task}_{cond}_42.jsonl"
            if not path.exists():
                continue
            rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            k, n = sum(r["is_correct"] for r in rows), len(rows)
            lo, hi = wilson(k, n)
            out[f"{task}|{cond}"] = {"acc": k / n, "lo": lo, "hi": hi, "n": n,
                                     "lenient": float(np.mean([is_correct_lenient(r["llm_output"], r["actual"]) for r in rows]))}
    return out


def grid_by_task_optimizer(grid: dict) -> dict:
    out: dict = {}
    for c in grid["cells"].values():
        key = f"{c['task']}|{c['optimizer']}"
        out.setdefault(key, []).append(c)
    return {k: {"cells": len(v), "mean_diff": float(np.mean([c["diff"] for c in v])),
                "accepted_runs": int(sum(c["accepted_runs"] for c in v)), "seeds": int(sum(c["seeds"] for c in v))}
            for k, v in sorted(out.items())}


# ---------------------------------------------------------------------------------------------- driver
def main() -> None:
    runs = load_runs()
    summary: dict = {"n_runs_total": len(runs)}
    summary["composition"] = composition(runs)
    summary["main_grid"] = main_grid(runs)
    rows = candidate_table(runs)
    summary["oracle"] = {f"{v}|{ref}": oracle_summary(rows, v, ref) for v in ("v1", "v2", "v2l", "v3")
                         for ref in ("1.7b", "4b", "8b", "qwen235", "gpt6luna") if any(r["variant"] == v and r["refiner"] == ref for r in rows)}
    summary["acceptance_rules"] = {f"{v}|{ref}": acceptance_rules(rows, v, ref) for v in ("v1", "v2", "v3")
                                   for ref in ("1.7b", "4b", "8b", "qwen235", "gpt6luna") if any(
                                       r["variant"] == v and r["refiner"] == ref and r["mseed"] == 1 for r in rows)}
    summary["val_test_corr"] = {"all": val_test_correlation(rows)} if rows else {}
    summary["legacy"] = legacy_acceptance(runs)
    summary["capability"] = capability(runs)
    summary["noise"] = noise_floor()
    from .data import tier1_cells, tier2_cells
    summary["p0_parsing"] = p0_parsing(tier1_cells() + tier2_cells())
    summary["p0_landscape"] = p0_landscape(tier1_cells() + tier2_cells())
    summary["reference_conditions"] = reference_conditions()
    summary["grid_by_task_optimizer"] = grid_by_task_optimizer(summary["main_grid"])
    summary["exemplar"] = exemplar()
    summary["judge_repeat"] = judge_repeatability(runs)
    summary["judge_validity"] = judge_validity(rows)
    REV.mkdir(parents=True, exist_ok=True)
    out = REV / "summary_numbers.json"
    out.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print("wrote", out, "runs", len(runs))


if __name__ == "__main__":
    main()
