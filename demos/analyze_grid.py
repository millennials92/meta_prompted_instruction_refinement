"""Statistical analysis for REBUILD.md §3.2 / §4.2 -- Phase 5 deliverables.

Reads results/predictions/<task>_<condition>_<seed>.jsonl (written by
GluePromptOpt.evaluate(), REBUILD.md §5's per-example provenance
requirement) and computes, for each requested (baseline, refined) pair such
as ("protegi", "protegi_mpir"):

  1. PRIMARY: example-level McNemar's test, pooled across every (task, seed)
     cell where both conditions were run on the identical held-out test
     partition (guaranteed by demos/data_prep.py's seed-scoped split -- the
     same seed always produces the same test set for every technique run
     against that task/seed, which is what makes this pairing valid), plus a
     task-clustered bootstrap CI on the pooled accuracy difference (resamples
     tasks, not individual examples, to respect the examples-within-task
     nesting).
  2. SECONDARY: a GEE logistic regression (condition as predictor, task as
     the clustering/exchangeable-correlation unit) -- the practical,
     well-supported approximation to REBUILD.md §4.2's "GLMM, task as random
     effect" that this environment can fit robustly. A GEE is a marginal
     (population-averaged) model rather than a true mixed-effects model;
     documented here rather than silently presented as equivalent.
  3. CONSERVATIVE: task-level Wilcoxon signed-rank test + sign test on
     per-task mean accuracy (averaged across seeds) -- matches the
     manuscript's original methodology, recomputed from real per-example
     predictions instead of the old aggregate xlsx files.
  4. A per-(task, condition) accuracy table with across-seed variance.

Usage:
    python analyze_grid.py --predictions-dir results/predictions \
        --pairs protegi:protegi_mpir,ape:ape_mpir,promptwizard:promptwizard_mpir
"""
import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar


def load_predictions(predictions_dir: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(predictions_dir, "*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"No prediction files found under {predictions_dir}")
    df = pd.DataFrame(rows)
    df["is_correct"] = df["is_correct"].astype(bool)
    return df


def _pooled_2x2_table(df: pd.DataFrame, condition_a: str, condition_b: str) -> Tuple[np.ndarray, int]:
    """Sum discordant/concordant counts across every (task, seed) cell where
    both conditions share the same example_index set -- valid because
    McNemar's statistic is a sum over paired observations, and summing the
    counts from several independent paired samples before testing is the
    standard way to pool matched-pairs data across strata."""
    table = np.zeros((2, 2), dtype=int)
    n_cells = 0
    for (task, seed), group in df.groupby(["task", "seed"]):
        a = group[group.condition == condition_a].set_index("example_index")["is_correct"]
        b = group[group.condition == condition_b].set_index("example_index")["is_correct"]
        common = a.index.intersection(b.index)
        if len(common) == 0:
            continue
        a, b = a.loc[common], b.loc[common]
        table[0, 0] += int(((a) & (b)).sum())
        table[0, 1] += int(((a) & (~b)).sum())
        table[1, 0] += int(((~a) & (b)).sum())
        table[1, 1] += int(((~a) & (~b)).sum())
        n_cells += 1
    return table, n_cells


def _pooled_accuracy_diff(df: pd.DataFrame, condition_a: str, condition_b: str,
                          tasks: List[str] = None) -> float:
    subset = df if tasks is None else df[df.task.isin(tasks)]
    a = subset[subset.condition == condition_a]
    b = subset[subset.condition == condition_b]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return b["is_correct"].mean() - a["is_correct"].mean()


def task_clustered_bootstrap_ci(df: pd.DataFrame, condition_a: str, condition_b: str,
                                n_bootstrap: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    """Bootstrap CI on the pooled accuracy difference (B - A), resampling at
    the TASK level (with replacement) rather than the example level, since
    examples within a task are not independent draws (REBUILD.md §3.2)."""
    rng = np.random.default_rng(seed)
    tasks = sorted(df.task.unique())
    point_estimate = _pooled_accuracy_diff(df, condition_a, condition_b)
    boot_diffs = []
    for _ in range(n_bootstrap):
        resampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        diff = _pooled_accuracy_diff(df, condition_a, condition_b, tasks=list(resampled_tasks))
        if not np.isnan(diff):
            boot_diffs.append(diff)
    if not boot_diffs:
        return point_estimate, float("nan"), float("nan")
    lower, upper = np.percentile(boot_diffs, [2.5, 97.5])
    return point_estimate, lower, upper


def primary_mcnemar(df: pd.DataFrame, condition_a: str, condition_b: str) -> Dict:
    table, n_cells = _pooled_2x2_table(df, condition_a, condition_b)
    n_discordant = table[0, 1] + table[1, 0]
    if n_discordant == 0:
        result = {"statistic": 0.0, "pvalue": 1.0}
    else:
        use_exact = n_discordant < 25
        test_result = mcnemar(table, exact=use_exact, correction=not use_exact)
        result = {"statistic": float(test_result.statistic), "pvalue": float(test_result.pvalue)}
    diff, ci_lower, ci_upper = task_clustered_bootstrap_ci(df, condition_a, condition_b)
    return {
        "n_task_seed_cells": n_cells,
        "table_a_correct_b_correct": int(table[0, 0]),
        "table_a_correct_b_wrong": int(table[0, 1]),
        "table_a_wrong_b_correct": int(table[1, 0]),
        "table_a_wrong_b_wrong": int(table[1, 1]),
        "n_discordant": int(n_discordant),
        "mcnemar_statistic": result["statistic"],
        "mcnemar_pvalue": result["pvalue"],
        "pooled_accuracy_diff": diff,
        "bootstrap_ci_95_lower": ci_lower,
        "bootstrap_ci_95_upper": ci_upper,
    }


def secondary_gee(df: pd.DataFrame, condition_a: str, condition_b: str) -> Dict:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    subset = df[df.condition.isin([condition_a, condition_b])].copy()
    subset["condition_indicator"] = (subset.condition == condition_b).astype(int)
    subset["is_correct_int"] = subset["is_correct"].astype(int)
    try:
        model = smf.gee("is_correct_int ~ condition_indicator", groups="task", data=subset,
                        family=sm.families.Binomial())
        fit = model.fit()
        coef = fit.params.get("condition_indicator", float("nan"))
        pvalue = fit.pvalues.get("condition_indicator", float("nan"))
        return {"coefficient": float(coef), "pvalue": float(pvalue), "converged": True}
    except Exception as e:
        return {"coefficient": float("nan"), "pvalue": float("nan"), "converged": False, "error": str(e)}


def conservative_task_level(df: pd.DataFrame, condition_a: str, condition_b: str) -> Dict:
    per_task = (df[df.condition.isin([condition_a, condition_b])]
                .groupby(["task", "condition"])["is_correct"].mean().unstack())
    per_task = per_task.dropna(subset=[condition_a, condition_b]) if condition_a in per_task and condition_b in per_task else per_task
    if condition_a not in per_task.columns or condition_b not in per_task.columns or len(per_task) < 2:
        return {"n_tasks": len(per_task), "wilcoxon_statistic": float("nan"), "wilcoxon_pvalue": float("nan"),
               "sign_test_pvalue": float("nan")}

    a_vals = per_task[condition_a].values
    b_vals = per_task[condition_b].values
    diffs = b_vals - a_vals
    non_zero = diffs[diffs != 0]

    if len(non_zero) == 0:
        wilcoxon_stat, wilcoxon_p = float("nan"), 1.0
    else:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(a_vals, b_vals)

    n_positive = int((diffs > 0).sum())
    n_negative = int((diffs < 0).sum())
    n_nonzero = n_positive + n_negative
    sign_p = stats.binomtest(n_positive, n_nonzero, 0.5).pvalue if n_nonzero > 0 else float("nan")

    return {
        "n_tasks": len(per_task),
        "wilcoxon_statistic": float(wilcoxon_stat),
        "wilcoxon_pvalue": float(wilcoxon_p),
        "n_tasks_favoring_b": n_positive,
        "n_tasks_favoring_a": n_negative,
        "sign_test_pvalue": float(sign_p) if n_nonzero > 0 else float("nan"),
    }


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    per_seed = df.groupby(["task", "condition", "seed"])["is_correct"].mean().reset_index()
    summary = per_seed.groupby(["task", "condition"])["is_correct"].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["task", "condition", "accuracy_mean", "accuracy_std_across_seeds", "n_seeds"]
    return summary.sort_values(["task", "condition"])


def analyze_pair(df: pd.DataFrame, condition_a: str, condition_b: str) -> Dict:
    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "primary_mcnemar": primary_mcnemar(df, condition_a, condition_b),
        "secondary_gee": secondary_gee(df, condition_a, condition_b),
        "conservative_task_level": conservative_task_level(df, condition_a, condition_b),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", default="results/predictions")
    parser.add_argument("--pairs", required=True,
                        help="Comma-separated condition_a:condition_b pairs, e.g. "
                             "protegi:protegi_mpir,ape:ape_mpir")
    parser.add_argument("--out", default="results/analysis_summary.json")
    args = parser.parse_args()

    df = load_predictions(args.predictions_dir)
    print(f"Loaded {len(df)} prediction rows across "
         f"{df.task.nunique()} tasks, {df.condition.nunique()} conditions, {df.seed.nunique()} seeds")

    pairs = []
    for pair_str in args.pairs.split(","):
        a, b = pair_str.split(":")
        pairs.append((a.strip(), b.strip()))

    results = {"accuracy_table": accuracy_table(df).to_dict(orient="records"), "pairs": []}
    for condition_a, condition_b in pairs:
        print(f"\n=== {condition_a} vs {condition_b} ===")
        pair_result = analyze_pair(df, condition_a, condition_b)
        results["pairs"].append(pair_result)

        primary = pair_result["primary_mcnemar"]
        print(f"Primary (McNemar, pooled over {primary['n_task_seed_cells']} task/seed cells, "
             f"n_discordant={primary['n_discordant']}): "
             f"stat={primary['mcnemar_statistic']:.3f}, p={primary['mcnemar_pvalue']:.4f}")
        print(f"  Pooled accuracy diff: {primary['pooled_accuracy_diff']:+.4f} "
             f"[95% CI: {primary['bootstrap_ci_95_lower']:+.4f}, {primary['bootstrap_ci_95_upper']:+.4f}]")

        secondary = pair_result["secondary_gee"]
        if secondary["converged"]:
            print(f"Secondary (GEE, task-clustered): coef={secondary['coefficient']:.4f}, "
                 f"p={secondary['pvalue']:.4f}")
        else:
            print(f"Secondary (GEE): did not converge -- {secondary.get('error')}")

        conservative = pair_result["conservative_task_level"]
        print(f"Conservative (task-level Wilcoxon, n={conservative['n_tasks']} tasks): "
             f"stat={conservative['wilcoxon_statistic']:.3f}, p={conservative['wilcoxon_pvalue']:.4f}; "
             f"sign test p={conservative['sign_test_pvalue']:.4f} "
             f"({conservative.get('n_tasks_favoring_b', '?')} tasks favor B, "
             f"{conservative.get('n_tasks_favoring_a', '?')} favor A)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
