"""Second-revision analyses that need no new generations (reviewer round 2, Access-2026-54235).

    python -m mpir_study.revision2          # writes results/revision/rev2_offline.json

Every number the revised manuscript cites for: the exact acceptance bound, the candidate-format audit, the
V2-versus-V3 paired comparison, judge correlations without format-zero candidates, the echo/fallback audit
samples, per-variant cost, and the classification of the 45 base prompts.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import beta

from .analyze import REV, candidate_table, cell_cache, cell_of, load_runs, runs_where
from .data import TIER1_TASKS, Cell, load_p0, tier1_cells, tier2_cells
from .stats import paired_bootstrap, spearman
from .templates import ANSWER_FORMAT, ECHO_MARKERS

OUT = REV / "rev2_offline.json"
BARE_TAG = re.compile(r"<ANS_START>\s*(?:\([A-R]\)|[A-R]|yes|no|true|false|valid|invalid)\s*<ANS_END>", re.I)
CHARS_PER_TOKEN = 4.36  # pooled ratio over 1,500 logged generations, Qwen3 tokenizer (tokens.py, chars_per_token.json)


# ------------------------------------------------------------------------------------------ exact bounds
def clopper_pearson_upper(k: int, n: int, level: float = 0.95, one_sided: bool = True) -> float:
    """Exact upper confidence bound for a binomial proportion."""
    alpha = (1 - level) if one_sided else (1 - level) / 2
    return float(beta.ppf(1 - alpha, k + 1, n - k)) if k < n else 1.0


def acceptance_bounds(runs: list[dict]) -> dict:
    group = runs_where(runs, variant="v2", refiner="1.7b")
    accepting = [r for r in group if not r["final_is_p0"]]
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in group:
        by_cell[cell_of(r).key].append(r)
    cell_any = sum(1 for rs in by_cell.values() if any(not r["final_is_p0"] for r in rs))
    by_opt, by_task = Counter(), Counter()
    for r in accepting:
        by_opt[r["spec"]["cell"]["optimizer"]] += 1
        by_task[r["spec"]["cell"]["task"]] += 1
    n_runs, k = len(group), len(accepting)
    return {"runs": n_runs, "accepting_runs": k, "cells": len(by_cell), "cells_with_acceptance": cell_any,
            "run_rate": k / n_runs,
            "run_upper_one_sided_95": clopper_pearson_upper(k, n_runs),
            "run_upper_two_sided_95": clopper_pearson_upper(k, n_runs, one_sided=False),
            "cell_upper_one_sided_95": clopper_pearson_upper(cell_any, len(by_cell)),
            "cell_upper_two_sided_95": clopper_pearson_upper(cell_any, len(by_cell), one_sided=False),
            "accepting_by_optimizer": dict(by_opt), "accepting_by_task": dict(by_task),
            "rule_of_three_check": 3 / n_runs}


# ------------------------------------------------------------------------------------------ round typology
def round_typology(runs: list[dict]) -> dict:
    out = {}
    for variant in ("v1", "v2", "v1p", "v2p"):
        rounds = [x for r in runs_where(runs, variant=variant, refiner="1.7b") for x in r["rounds"]]
        counts = Counter(x["outcome"] for x in rounds)
        n = len(rounds)
        out[variant] = {"rounds": n, **{k: counts.get(k, 0) for k in ("fallback", "echo", "duplicate", "validated")},
                        "share_fallback_echo": (counts["fallback"] + counts["echo"]) / n if n else None,
                        "share_fallback_echo_duplicate":
                            (counts["fallback"] + counts["echo"] + counts["duplicate"]) / n if n else None}
    chained = [x for r in runs_where(runs, variant="v2", refiner="1.7b") for x in r["rounds"]]
    both = [x for r in runs_where(runs, variant="v1", refiner="1.7b") for x in r["rounds"]] + chained
    c = Counter(x["outcome"] for x in both)
    out["v1_v2_pooled"] = {"rounds": len(both), "fallback": c["fallback"], "echo": c["echo"],
                           "duplicate": c["duplicate"], "validated": c["validated"]}
    return out


# ------------------------------------------------------------------------------------------ candidate audit
def p0_features(cell: Cell) -> dict:
    p0 = load_p0(cell)
    return {"chars": len(p0), "has_format_line": ANSWER_FORMAT in p0, "has_bare_tag_example": bool(BARE_TAG.search(p0)),
            "worked_examples": p0.count("[Question]") + p0.count("Q:"),
            "kind": ("induction_template" if "I gave a friend an instruction" in p0 else
                     "seed_plus_format" if len(p0) <= 120 else
                     "promptwizard_long" if len(p0) > 2000 else "short_instruction")}


def _retains_examples(raw: str, p0_examples: int) -> bool | None:
    if p0_examples == 0:
        return None
    return raw.count("[Question]") + raw.count("Q:") >= max(1, p0_examples // 2)


def candidate_format_audit(runs: list[dict]) -> dict:
    """What the 1.7B refiner's raw candidates keep of the base prompt (answers reviewer question 1)."""
    out = {}
    for variant in ("v1", "v2", "v1p", "v2p", "v2f"):
        rows = []
        for r in runs_where(runs, variant=variant, refiner="1.7b"):
            feats = p0_features(cell_of(r))
            for x in r["rounds"]:
                raw = x.get("raw_candidate")
                if not raw or x["outcome"] == "echo":
                    continue
                rows.append({"format": ANSWER_FORMAT in raw, "bare": bool(BARE_TAG.search(raw)),
                             "examples": _retains_examples(raw, feats["worked_examples"]),
                             "had_examples": feats["worked_examples"] > 0, "p0_bare": feats["has_bare_tag_example"]})
        with_ex = [x for x in rows if x["had_examples"]]
        no_fmt = [x for x in rows if not x["format"]]
        out[variant] = {
            "candidates": len(rows),
            "keep_format_line": float(np.mean([x["format"] for x in rows])) if rows else None,
            "keep_bare_tag_example": float(np.mean([x["bare"] for x in rows])) if rows else None,
            "from_prompts_with_examples": len(with_ex),
            "keep_examples_given_p0_has_examples": float(np.mean([bool(x["examples"]) for x in with_ex])) if with_ex else None,
            "lost_format": len(no_fmt),
            "lost_format_and_examples_given_p0_has_examples":
                sum(1 for x in no_fmt if x["had_examples"] and not x["examples"]),
            "lost_format_from_prompts_with_examples": sum(1 for x in no_fmt if x["had_examples"])}
    return out


def base_prompt_inventory() -> dict:
    cells = [c for c in tier1_cells() + tier2_cells()
             if (load_p0_exists(c))]
    rows = {c.key: p0_features(c) for c in cells}
    kinds = Counter(v["kind"] for v in rows.values())
    return {"cells": rows, "kinds": dict(kinds),
            "with_bare_tag_example": sum(1 for v in rows.values() if v["has_bare_tag_example"]),
            "n": len(rows)}


def load_p0_exists(cell: Cell) -> bool:
    try:
        load_p0(cell)
        return True
    except FileNotFoundError:
        return False


def protected_failure_modes(runs: list[dict]) -> dict:
    """Why V1P/V2P candidates still score far below P0: what the target does with the answer tags."""
    out = {}
    for variant in ("v1p", "v2p"):
        stats = {"candidates": 0, "no_tag": 0, "sentence_in_tag": 0, "bare_in_tag": 0, "outputs": 0}
        diffs, retain = [], []
        for r in runs_where(runs, variant=variant, refiner="1.7b"):
            cell = cell_of(r)
            cache = cell_cache(cell.key, with_outputs=True)
            for x in r["rounds"]:
                if x["outcome"] != "validated" or "test" not in x:
                    continue
                scored = cache.get((x["cand_hash"], "test", 0))
                if not scored:
                    continue
                stats["candidates"] += 1
                diffs.append(x["test"] - r["p0_test"])
                retain.append(bool(BARE_TAG.search(x["candidate"])))
                for text in scored[1]:
                    stats["outputs"] += 1
                    start, end = text.find("<ANS_START>"), text.find("<ANS_END>")
                    if start < 0 or end < start:
                        stats["no_tag"] += 1
                        continue
                    span = text[start + 11:end].strip()
                    stats["bare_in_tag" if len(span) <= 12 else "sentence_in_tag"] += 1
        n = max(stats["outputs"], 1)
        out[variant] = {**stats, "mean_diff_vs_p0": float(np.mean(diffs)) if diffs else None,
                        "candidates_with_bare_tag_example": int(sum(retain)),
                        "share_no_tag": stats["no_tag"] / n, "share_sentence_in_tag": stats["sentence_in_tag"] / n,
                        "share_bare_in_tag": stats["bare_in_tag"] / n}
    return out


# ------------------------------------------------------------------------------------------ V2 vs V3
def rubric_vs_free(runs: list[dict], refiner: str = "qwen235") -> dict:
    """Per-cell paired V2 - V3 difference on test (final prompts, same refiner, same cells)."""
    cells = {}
    for c in tier1_cells(seeds=(42,)):
        v3 = [r for r in runs_where(runs, variant="v3", refiner=refiner, mseed=1) if cell_of(r) == c]
        v2 = {m: [r for r in runs_where(runs, variant="v2", refiner=refiner, mseed=m) if cell_of(r) == c]
              for m in (1, 2)}
        if not v3 or not v2[1]:
            continue
        entry = {"v3_final_is_p0": v3[0]["final_is_p0"], "v2_m1_final_is_p0": v2[1][0]["final_is_p0"],
                 "p0_test": v3[0]["p0_test"]}
        entry["v2m1_minus_v3"] = paired_bootstrap(v3[0]["paired"]["final"], v2[1][0]["paired"]["final"])
        if v2[2]:
            entry["v2m2_minus_v3"] = paired_bootstrap(v3[0]["paired"]["final"], v2[2][0]["paired"]["final"])
        entry["v2m1_minus_p0"] = paired_bootstrap(v3[0]["paired"]["p0"], v2[1][0]["paired"]["final"])
        entry["v3_minus_p0"] = paired_bootstrap(v3[0]["paired"]["p0"], v3[0]["paired"]["final"])
        cells[c.key] = entry
    diffs = np.array([e["v2m1_minus_v3"]["diff"] for e in cells.values()]) if cells else np.array([])
    out = {"cells": cells, "n_cells": len(cells)}
    if len(diffs):
        rng = np.random.default_rng(20261019)
        boot = rng.choice(diffs, size=(10_000, len(diffs)), replace=True).mean(axis=1)
        out.update({"mean_v2_minus_v3": float(diffs.mean()), "lo": float(np.percentile(boot, 2.5)),
                    "hi": float(np.percentile(boot, 97.5)),
                    "cells_v2_better": int((diffs > 0).sum()), "cells_v3_better": int((diffs < 0).sum()),
                    "cells_tied": int((diffs == 0).sum()),
                    "cells_ci_excludes_zero": int(sum(1 for e in cells.values()
                                                      if e["v2m1_minus_v3"]["lo"] > 0 or e["v2m1_minus_v3"]["hi"] < 0))})
        if all("v2m2_minus_v3" in e for e in cells.values()):
            d2 = np.array([e["v2m2_minus_v3"]["diff"] for e in cells.values()])
            out["mean_v2seed2_minus_v3"] = float(d2.mean())
    return out


# ------------------------------------------------------------------------------------------ judge validity
def judge_without_format_zeros(runs: list[dict]) -> dict:
    scores = json.loads((REV / "judge_scores.json").read_text())
    rows = candidate_table(runs)
    out = {}
    for judge, table in scores.items():
        usable = {h: v["total"] for h, v in table.items() if v["total"] is not None}
        for label, keep in (("all", lambda r: True), ("test_above_zero", lambda r: r["test"] > 0)):
            per_cell = []
            for cell in sorted({r["cell"] for r in rows}):
                sel = {r["cand_hash"]: r for r in rows if r["cell"] == cell and keep(r) and r["cand_hash"] in usable}
                if len(sel) < 4:
                    continue
                rho = spearman([usable[h] for h in sel], [sel[h]["test"] for h in sel])
                if rho is not None:
                    per_cell.append(rho)
            kept = [r for r in rows if keep(r) and r["cand_hash"] in usable]
            out[f"{judge}|{label}"] = {"candidates": len(kept), "cells_used": len(per_cell),
                                        "median_spearman_test": float(np.median(per_cell)) if per_cell else None,
                                        "zero_share": float(np.mean([r["test"] == 0 for r in rows
                                                                     if r["cand_hash"] in usable])) if rows else None}
    return out


# ------------------------------------------------------------------------------------------ audit samples
def audit_samples(runs: list[dict], n_each: int = 40, seed: int = 20261102) -> dict:
    """Random echo/fallback rounds (chained and independent 1.7B variants, unprotected+protected) for manual audit."""
    pool = {"echo": [], "fallback": []}
    for r in runs:
        if r["spec"]["refiner"] != "1.7b" or r["spec"]["variant"] not in ("v1", "v2", "v1p", "v2p"):
            continue
        for x in r["rounds"]:
            if x["outcome"] in pool:
                pool[x["outcome"]].append((r["spec"]["variant"], cell_of(r).key, r["spec"]["mseed"], x))
    rng = random.Random(seed)
    samples, stats = {}, {}
    for kind, items in pool.items():
        picked = rng.sample(items, min(n_each, len(items)))
        samples[kind] = [{"id": f"{kind}-{i:02d}", "variant": v, "cell": c, "mseed": m, "round": x["round"],
                          "refine_text": x["refine_text"], "raw_candidate": x["raw_candidate"]}
                         for i, (v, c, m, x) in enumerate(picked)]
        stats[kind] = len(items)
    # automatic fallback sub-causes over the full pool
    causes = Counter()
    lengths = []
    for variant, cell_key, m, x in pool["fallback"]:
        text = x["refine_text"] or ""
        lengths.append(len(text))
        if "<START>" not in text:
            causes["no_start_delimiter"] += 1
        elif "<END>" not in text:
            causes["start_without_end"] += 1
        else:
            causes["empty_span"] += 1
    marker_hits = Counter()
    for variant, cell_key, m, x in pool["echo"]:
        for marker in ECHO_MARKERS:
            if marker in (x["raw_candidate"] or ""):
                marker_hits[marker] += 1
    near_limit = {"refine_ge_90pct_of_3000": 0, "judge_ge_90pct_of_1500": 0, "rounds": 0}
    for r in runs:
        if r["spec"]["refiner"] != "1.7b":
            continue
        for x in r["rounds"]:
            near_limit["rounds"] += 1
            near_limit["refine_ge_90pct_of_3000"] += len(x.get("refine_text") or "") >= 0.9 * 3000 * CHARS_PER_TOKEN
            near_limit["judge_ge_90pct_of_1500"] += len(x.get("judge_text") or "") >= 0.9 * 1500 * CHARS_PER_TOKEN
    (REV / "rev2_audit_samples.json").write_text(json.dumps(samples, indent=1), encoding="utf-8")
    return {"pool_sizes": stats, "fallback_causes": dict(causes),
            "fallback_length_chars_median": float(np.median(lengths)) if lengths else None,
            "fallback_length_chars_p90": float(np.percentile(lengths, 90)) if lengths else None,
            "echo_marker_hits": dict(marker_hits), "near_token_limit": near_limit}


def audit_label_summary() -> dict:
    """Counts from the blind manual audit (results/revision/rev2_audit_labels.json)."""
    path = REV / "rev2_audit_labels.json"
    if not path.exists():
        return {}
    labels = json.loads(path.read_text(encoding="utf-8"))
    echo = Counter(x["verdict"] for x in labels["echo"])
    fall = Counter(x["cause"] for x in labels["fallback"])
    n_e, n_f = len(labels["echo"]), len(labels["fallback"])
    fp = echo.get("false_positive", 0)
    rec = sum(1 for x in labels["fallback"] if x.get("recoverable"))
    return {"echo_n": n_e, "echo_false_positive": fp, "echo_true": echo.get("true_echo", 0),
            "echo_fp_rate": fp / n_e, "echo_fp_upper_95_one_sided": clopper_pearson_upper(fp, n_e),
            "fallback_n": n_f, "fallback_causes": dict(fall), "fallback_recoverable": rec,
            "fallback_recoverable_rate": rec / n_f,
            "fallback_recoverable_upper_95_one_sided": clopper_pearson_upper(rec, n_f)}


# ------------------------------------------------------------------------------------------ cost
def cost_per_variant(runs: list[dict]) -> dict:
    out = {}
    keys = sorted({(r["spec"]["variant"], r["spec"]["refiner"]) for r in runs})
    for variant, refiner in keys:
        group = runs_where(runs, variant=variant, refiner=refiner)
        calls_ref, chars_ref, calls_val, rounds = [], [], [], []
        for r in group:
            n_calls = 0
            chars = 0
            for x in r["rounds"]:
                n_calls += 1 if variant == "v3" else 2
                chars += len(x.get("judge_text") or "") + len(x.get("refine_text") or "")
            validated = [x for x in r["rounds"] if x["outcome"] == "validated"]
            reps = r["spec"]["val_reps"]
            calls_ref.append(n_calls)
            chars_ref.append(chars)
            calls_val.append(len(validated) * reps * 25 + 25)  # validation split = 25 items; +baseline scoring
            rounds.append(len(r["rounds"]))
        out[f"{variant}|{refiner}"] = {
            "runs": len(group), "refiner_calls_per_run": float(np.mean(calls_ref)),
            "refiner_generated_tokens_per_run": float(np.mean(chars_ref) / CHARS_PER_TOKEN),
            "target_validation_calls_per_run": float(np.mean(calls_val)),
            "target_test_calls_per_run": float(2 * np.mean([r["n_test"] for r in group]))}
    return out


def main() -> None:
    runs = load_runs()
    result = {
        "acceptance_bounds": acceptance_bounds(runs),
        "round_typology": round_typology(runs),
        "candidate_format_audit": candidate_format_audit(runs),
        "base_prompt_inventory": base_prompt_inventory(),
        "protected_failure_modes": protected_failure_modes(runs),
        "rubric_vs_free": {r: rubric_vs_free(runs, r) for r in ("qwen235",)},
        "judge_without_format_zeros": judge_without_format_zeros(runs),
        "audit_samples": audit_samples(runs),
        "audit_labels": audit_label_summary(),
        "cost": cost_per_variant(runs),
    }
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
