# -*- coding: utf-8 -*-
"""Results. Every number comes from results/revision/summary_numbers.json (mpir_study/analyze.py)."""
import json
import os

from blocks_api import H1, H2, H3, P, BULLETS, TABLE, CODE  # noqa: F401
from study_numbers import N

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS_T1 = ("hyperbaton", "ruin_names", "penguins_in_a_table")
TASKS_T2 = ("temporal_sequences", "movie_recommendation", "disambiguation_qa", "geometric_shapes",
            "salient_translation_error_detection", "sports_understanding")
OPTS = ("promptwizard", "ape", "protegi")
OPT_NAME = {"promptwizard": "PromptWizard", "ape": "APE", "protegi": "ProTeGi"}


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def pp(x, d=1):
    return f"{100 * x:+.{d}f}"


def share(part, whole):
    return f"{100 * part / whole:.0f}"


def cell_label(key):
    task, opt = key.split("__")[:2]
    return f"{task} / {OPT_NAME[opt]}"


C = N["composition"]
MG = N["main_grid"]
CAP = N["capability"]
CAPL = N["capability_lenient"]
NOISE = N["noise"]
LEG = N["legacy"]
EX, EXL = N["exemplar"], N["exemplar_lenient"]
JV = N["judge_validity"]["judges"]
LO = N["lenient_oracle"]
OR = N["oracle"]
AR = N["acceptance_rules"]
LAND = N["p0_landscape"]
PVR = N["pilot_vs_rescore"]
V0, V1, V2, V3 = C["v0|1.7b"], C["v1|1.7b"], C["v2|1.7b"], C["v3|1.7b"]
V2L, QW, LU = C["v2l|1.7b"], C["v2|qwen235"], C["v2|gpt6luna"]
V1P, V2P = C.get("v1p|1.7b"), C.get("v2p|1.7b")
VC = N["val_test_corr"]["all"]


def wins_and_losses(group):
    wins = [(cell_label(k), v["diff"]) for k, v in group["per_cell"].items() if v["diff"] > 0 and v["lo"] > 0]
    losses = [(cell_label(k), v["diff"]) for k, v in group["per_cell"].items() if v["diff"] < 0 and v["hi"] < 0]
    return sorted(wins, key=lambda t: -t[1]), sorted(losses, key=lambda t: t[1])


def listing(pairs):
    return ", ".join(f"{n} {pp(d)}" for n, d in pairs) if pairs else "none"


# ===========================================================================
H1("5. Results")

H2("5.1. Base Prompts and the Scoring Harness")
P("Table 4 gives the strict and lenient test accuracy of the nine pilot base prompts (seed 42) with Wilson "
  "intervals ‹125› and the chance level of each task, and Table 5 the reference conditions. Several baselines "
  "are at or below chance under strict extraction, the largest being ProTeGi: "
  f"{pct(LAND['hyperbaton__protegi__s42']['strict'])}% on hyperbaton, "
  f"{pct(LAND['ruin_names__protegi__s42']['strict'])}% on ruin_names and "
  f"{pct(LAND['penguins_in_a_table__protegi__s42']['strict'])}% on penguins_in_a_table. Under lenient "
  "extraction the same responses score "
  f"{pct(LAND['hyperbaton__protegi__s42']['lenient'])}%, {pct(LAND['ruin_names__protegi__s42']['lenient'])}% and "
  f"{pct(LAND['penguins_in_a_table__protegi__s42']['lenient'])}%. The raw outputs show the cause: "
  "the model reasons to an option but wraps a whole explanatory sentence in the answer tags, which "
  "the strict rule rejects. This happens for exemplar-free prompts (APE, ProTeGi, zero-shot) and not for "
  "prompts that contain a worked example of the bare tag (PromptWizard, expert few-shot). Lenient extraction "
  "accepts everything strict extraction accepts, plus a response whose tagged span (or, without tags, last 300 "
  "characters) names exactly the gold option; it is approximate, and a prompt that scores well on it "
  "is not guaranteed to be reasoning correctly.")

rows = []
for t in TASKS_T1:
    for o in OPTS:
        v = LAND[f"{t}__{o}__s42"]
        rows.append([t, OPT_NAME[o], f"{pct(v['strict'])} [{pct(v['lo'])}, {pct(v['hi'])}]",
                     pct(v["lenient"]), pct(v["chance"]), pct(v["with_delims"], 0)])
TABLE("Table 4. Test accuracy (%) of the nine pilot base prompts (seed 42, Qwen3-1.7B, greedy, one scoring "
      "pass). Strict is the primary metric with a 95% Wilson interval; lenient is the secondary metric. "
      "Chance is 1 over the number of distinct gold labels. Delims is the share of responses containing both tags.",
      header=["Task", "Optimizer", "Strict [95% CI]", "Lenient", "Chance", "Delims %"],
      rows=rows, full=True, colw=[0.27, 0.15, 0.24, 0.12, 0.10, 0.12])

REF = N["reference_conditions"]
rows = []
for t in TASKS_T1:
    row = [t]
    for cond in ("zero_shot_cot", "expert_few_shot_cot", "free_rewrite"):
        v = REF[f"{t}|{cond}"]
        row.append(f"{pct(v['acc'])} ({pct(v['lenient'])})")
    rows.append(row)
TABLE("Table 5. Reference conditions, strict (lenient) test accuracy %, seed 42. Free rewrite is the ungated "
      "rubric-free rewrite of the PromptWizard prompt from the pilot.",
      header=["Task", "Zero-shot CoT", "Expert few-shot CoT", "Free rewrite (ungated)"],
      rows=rows, full=True, colw=[0.34, 0.22, 0.22, 0.22])

P("Two checks bear on how much of this is a harness effect. First, appending one worked format exemplar (the "
  "first BBH-author exemplar, with its bare tag) to an exemplar-free base prompt raised strict accuracy "
  f"by {pp(EX['mean_gain_by_optimizer']['ape'])} points on average for APE and "
  f"{pp(EX['mean_gain_by_optimizer']['protegi'])} points for ProTeGi over {EX['cells_by_optimizer']['ape']} cells "
  f"each (the paired-bootstrap interval excludes zero in {EX['wins_by_optimizer']['ape']} and "
  f"{EX['wins_by_optimizer']['protegi']} cells; the range is {pp(EX['min_gain'])} to {pp(EX['max_gain'])} "
  "points, the negative end being APE's degenerate hyperbaton prompt). Under lenient extraction the same "
  f"intervention gains {pp(EXL['mean_by_optimizer']['ape'])} and {pp(EXL['mean_by_optimizer']['protegi'])} points, "
  "so most of the strict gain is the tag, and the exemplar's effect on reasoning is modest. Second, the same "
  "prompt does not score the same twice: the strict accuracy of the identical nine prompts, scored in the "
  f"pilot and again for this study, differs by {pct(PVR['mean_abs_diff'])} points on average and up to "
  f"{pct(PVR['max_abs_diff'])} points (Section 5.7). Accuracy differences of this size are not reliably "
  "attributable to a prompt on this serving stack.")

# ===========================================================================
H2("5.2. What the Rounds Were")
P("The composition of the refinement rounds is in Table 6 (all rounds of all runs; per-cell, per-round "
  "records for the primary variant are in Appendix E). The legacy loop of the pilot "
  f"(V0, nine cells) produced on average {V0['mean_distinct_candidates']:.1f} distinct candidate prompts per "
  "run although decoding was greedy: the seven rounds were not identical copies, but they differed only "
  "through serving non-determinism, not by design, so 'seven rounds without acceptance' overstated the "
  "evidence. (The pilot's own prompts were byte-identical in all nine cells; in our re-run of the same loop "
  f"{V0['runs_with_acceptance']} of nine runs changed its prompt, which shows how fragile "
  "even a greedy comparison is.) In the repaired variants the rounds differ by sampling and, for V2, by history.")

rows = []
for key, label in [("v0|1.7b", "V0 legacy"), ("v1|1.7b", "V1 independent"), ("v2|1.7b", "V2 chained"),
                   ("v2l|1.7b", "V2L original rubric"), ("v3|1.7b", "V3 free rewrite (gated)"),
                   ("v1p|1.7b", "V1P protected"), ("v2p|1.7b", "V2P protected"),
                   ("v2|qwen235", "V2, refiner Qwen3-235B"), ("v3|qwen235", "V3, refiner Qwen3-235B"),
                   ("v2|gpt6luna", "V2, refiner GPT-6 Luna")]:
    if key not in C:
        continue
    c = C[key]
    rows.append([label, str(c["runs"]), str(c["rounds"]), str(c["fallback"]), str(c["echo"]), str(c["duplicate"]),
                 str(c["validated"]), f"{c['val_better']}/{c['val_equal']}/{c['val_worse']}", str(c["accepted"]),
                 f"{c['runs_with_acceptance']}/{c['runs']}"])
TABLE("Table 6. Outcome of every refinement round (all runs of a variant pooled; refiner Qwen3-1.7B unless "
      "stated). Fallback: no delimiters. Echo: template phrase. Duplicate: same text as an earlier prompt. "
      "Validated rounds are split by validation score against the baseline (better/equal/worse). The last "
      "column counts runs whose final prompt differs from P0.",
      header=["Variant", "Runs", "Rounds", "Fallback", "Echo", "Dup.", "Validated", "Val. better/equal/worse",
              "Accepted", "Runs changed"],
      rows=rows, full=True)

_prot = ""
if V1P and V2P:
    _prot = (f"With protection the 1.7B refiner still reaches validation in only "
             f"{share(V1P['validated'], V1P['rounds'])}% (V1P, {V1P['rounds']} rounds) and "
             f"{share(V2P['validated'], V2P['rounds'])}% (V2P, {V2P['rounds']} rounds) of rounds, and "
             f"{V1P['val_better'] + V2P['val_better']} of its {V1P['validated'] + V2P['validated']} validated "
             "candidates beat the baseline on validation (Table 6). The protected variants were run on the nine "
             "Tier-1 seed-42 cells and finished on "
             f"{N['capability']['v1p|1.7b']['cells']} (V1P) and {N['capability']['v2p|1.7b']['cells']} (V2P) of them; "
             "the rest were not completed for compute reasons. ")
P(f"For the 1.7B refiner, {share(V2['fallback'] + V2['echo'], V2['rounds'])}% of the chained rounds "
  f"({V2['fallback'] + V2['echo']} of {V2['rounds']}) never reached validation because the response had no "
  f"delimiters or echoed a template; of the {V2['validated']} validated candidates {V2['val_worse']} scored "
  f"below the baseline, {V2['val_equal']} tied and {V2['val_better']} scored higher. Most validated candidates "
  "are far worse, not tied. An audit found a large part of the reason: the refinement request does not "
  "ask the refiner to keep the answer-format line, and "
  f"{pct(LO['v1|1.7b']['share_without_tag'], 0)}% of the validated V1 and {pct(LO['v2|1.7b']['share_without_tag'], 0)}% "
  "of the validated V2 candidates lack it, so they score exactly zero under strict extraction; the "
  "rubric-free rewrite, whose prompt asks to preserve the answer format, loses it in none. The "
  "unprotected numbers therefore measure a failure of the 1.7B refiner to keep the format as much as a "
  "failure of the rubric. To separate the two we added the format-protected variants V1P and V2P, which "
  "append the answer-format line to every candidate. " + _prot +
  f"Refinement with the original, inconsistent rubric text (V2L) was no better: {V2L['accepted']} accepted "
  f"candidates in {V2L['rounds']} rounds and {share(V2L['fallback'] + V2L['echo'], V2L['rounds'])}% of rounds "
  "lost to fallbacks or echoes, so the inconsistency in the rubric text is not what limits the 1.7B refiner.")

# ===========================================================================
H2("5.3. The Repaired Layer on 45 Cells (RQ1, D1)")
GO = N["grid_by_optimizer"]
P(f"With the 1.7B model refining its own prompts through the chained variant, the final prompt equals P0 in "
  f"{MG['runs'] - MG['runs_accepting']} of {MG['runs']} runs ({MG['n_cells']} cells, one to three MPIR seeds each). "
  f"Only {MG['runs_accepting']} run accepted a candidate. No cell is a win: the mean paired difference "
  f"over cells is {pp(MG['mean_diff'], 2)} points (hierarchical bootstrap interval "
  f"[{pp(MG['hier']['lo'], 2)}, {pp(MG['hier']['hi'], 2)}], which is almost degenerate because 44 of 45 cells "
  f"are exactly zero), and {MG['wins']} of {MG['n_cells']} cells satisfy the win criterion. By the "
  f"pre-specified rule (D1) the outcome is '{MG['verdict']}'; a rule-of-three bound puts the "
  f"acceptance rate of this procedure below about {100 * 3 / MG['runs']:.0f}% of runs. The result does not "
  "depend on the base optimizer: "
  + "; ".join(f"{OPT_NAME[o]} {GO[o]['accepted_runs']} of {GO[o]['runs']} runs changed" for o in OPTS) + ". "
  "D1 was specified and run for the unprotected V2, and Section 5.2 shows that this procedure is limited by "
  "the 1.7B refiner's inability to produce usable, format-preserving candidates; the null is therefore a "
  "statement about this procedure with this refiner, and not about rubric-guided refinement in general. "
  "Table 7 gives the per-task view. The tests of the pilot (McNemar, GEE, Wilcoxon on three tasks) are not "
  "reported, because with identical prompts they cannot be informative.")

GT = N["grid_by_task_optimizer"]
rows = []
for t in TASKS_T1 + TASKS_T2:
    row = [t + (" *" if t in TASKS_T1 else "")]
    for o in OPTS:
        g = GT.get(f"{t}|{o}")
        row.append("n/a" if g is None else f"{g['accepted_runs']}/{g['seeds']}, {pp(g['mean_diff'])}")
    rows.append(row)
TABLE("Table 7. Chained variant, 1.7B refiner (PW = PromptWizard): in each entry, the number of runs whose final "
      "prompt differs from P0 out of all MPIR runs of that task and optimizer, then the mean paired test-accuracy "
      "difference over "
      "those cells in points. * marks the Tier-1 tasks (split seeds 42 to 44, three cells per optimizer); "
      "Tier-2 tasks have split seed 42 only.",
      header=["Task", "PW", "APE", "ProTeGi"], rows=rows, full=True, colw=[0.37, 0.21, 0.21, 0.21])

# ===========================================================================
H2("5.4. Is the Gate the Bottleneck? (RQ2, D3)")
O1, O2, O3 = OR["v1|1.7b"], OR["v2|1.7b"], OR["v3|1.7b"]
OQ = OR["v2|qwen235"]
P("To separate the candidates from the gate, every validated candidate of MPIR seed 1 was also scored on the "
  f"full test split. Ungated, the mean test-accuracy change of a 1.7B-refiner candidate relative to its P0 is "
  f"{pp(O1['mean_candidate_delta'])} points for the independent variant ({O1['n_candidates']} candidates) and "
  f"{pp(O2['mean_candidate_delta'])} points for the chained variant ({O2['n_candidates']}), and no cell has a "
  f"candidate that beats P0 by 3 points or more ({O1['cells_best_ge_3pts']} and {O2['cells_best_ge_3pts']} of 9 "
  "cells); these candidates mostly lack the answer-format line (Section 5.2), so the figure largely "
  "reflects format loss. Candidates that keep the format are still not better on average: "
  + (f"{pp((OR['v1p|1.7b']['mean_candidate_delta'] * OR['v1p|1.7b']['n_candidates'] + OR['v2p|1.7b']['mean_candidate_delta'] * OR['v2p|1.7b']['n_candidates']) / (OR['v1p|1.7b']['n_candidates'] + OR['v2p|1.7b']['n_candidates']))} "
     f"points for the format-protected 1.7B candidates of V1P and V2P ({OR['v1p|1.7b']['n_candidates'] + OR['v2p|1.7b']['n_candidates']} candidates, none beating P0 by 3 points or more), "
     if 'v1p|1.7b' in OR and 'v2p|1.7b' in OR else "")
  + f"{pp(O3['mean_candidate_delta'])} points for the 1.7B rewrite ({O3['n_candidates']} candidates) and "
  f"{pp(OQ['mean_candidate_delta'])} points for the Qwen3-235B candidates ({OQ['n_candidates']}; best candidate "
  f"per cell {pp(OQ['mean_best_delta'])} points on average, {OQ['cells_best_ge_3pts']} cells at +3 or more). D3 "
  "requires the best candidate to beat P0 by 3 points in at least 3 of 9 cells while the gated final prompt "
  "does not; it is not met for the 1.7B refiner. Validation accuracy tracks test accuracy across candidates "
  f"(Spearman {VC['pooled_spearman']:.2f} over {VC['n']} oracle-scored candidates of all refiners), but this is "
  "driven by candidates that score zero; among 1.7B-refiner candidates with positive validation accuracy it is "
  f"{VC['pos_val_1.7b']['spearman']:.2f} (n = {VC['pos_val_1.7b']['n']}). The gate can therefore rank "
  "broken candidates but is a weaker guide among plausible ones, and a 25-question gate is noisy (Section 5.7). "
  "Table 8 re-applies other acceptance rules offline to the same candidates; cells with no validated candidate "
  "are omitted.")

rows = []
for key, label in [("v1|1.7b", "V1 independent"), ("v2|1.7b", "V2 chained"), ("v3|1.7b", "V3 free rewrite"),
                   ("v2|qwen235", "V2, Qwen3-235B")]:
    if key not in AR:
        continue
    S = AR[key]["summary"]
    for rule, rname in [("strict", "strict (as run)"), ("nonstrict", "non-strict (ties accepted)"),
                        ("repeated", "mean of 3 validation scorings"), ("large_gate", "larger gate (val + half of test)")]:
        s = S[rule]
        rows.append([label, rname, f"{s['accepted_cells']}/{s['cells']}", pp(s["mean_delta"]),
                     f"{s['cells_better']}/{s['cells_worse']}"])
TABLE("Table 8. Acceptance-rule sensitivity on the oracle-scored candidates (MPIR seed 1, Tier-1 seed-42 cells). "
      "Cells accepting: cells in which the rule accepts some candidate, out of the cells with at least one "
      "validated candidate. Mean delta is the mean change in strict test accuracy of the selected prompt "
      "versus P0, in points, over those cells (for the larger gate, measured on the held-out half of the test "
      "split). Better/worse is the number of cells with a positive or negative delta.",
      header=["Variant", "Acceptance rule", "Cells accepting", "Mean delta", "Better/worse cells"],
      rows=rows, full=True, colw=[0.20, 0.36, 0.16, 0.12, 0.16])

# ===========================================================================
H2("5.5. Free Rewriting Versus the Rubric")
fr = {t: REF[f"{t}|free_rewrite"]["acc"] - LAND[f"{t}__promptwizard__s42"]["strict"] for t in TASKS_T1}
w3, l3 = wins_and_losses(CAP["v3|1.7b"])
P("In the pilot, the ungated rubric-free rewrite of the PromptWizard prompt changed strict accuracy by "
  f"{pp(fr['hyperbaton'])}, {pp(fr['ruin_names'])} and {pp(fr['penguins_in_a_table'])} points on hyperbaton, "
  "ruin_names and penguins_in_a_table (Tables 4 and 5); the first is 4 test items, within the serving noise of "
  "Section 5.7, and the others are large losses, so the ungated rewrite was harmful on two of three tasks. The "
  f"gated rewrite (V3) accepted a rewrite in {V3['runs_with_acceptance']} of {V3['runs']} runs, against "
  f"{V1['runs_with_acceptance']} of {V1['runs']} for the independent rubric variant and "
  f"{CAP['v2|1.7b']['runs_accepting']} of {CAP['v2|1.7b']['runs']} for the chained one (Tier-1 runs). Part of this "
  "gap is the format asymmetry of Section 5.2, not the rewriting style. V3 has "
  f"{CAP['v3|1.7b']['wins']} winning cells ({listing(w3)}) and {len(l3)} cells in which the gate accepted a "
  f"harmful rewrite ({listing(l3)}), for a mean difference of {pp(CAP['v3|1.7b']['mean_diff'])} points. So the "
  "gate does not reliably protect against harm either, since 25 questions are noisy. We cannot say that "
  "unstructured rewriting is worse than rubric-guided refinement: the rubric-free variant did at least as well "
  f"({pp(CAP['v3|1.7b']['mean_diff'])} against {pp(CAP['v1|1.7b']['mean_diff'])} and "
  f"{pp(CAP['v2|1.7b']['mean_diff'])} points), and the rubric adds nothing detectable with this refiner.")

# ===========================================================================
H2("5.6. A More Capable Refiner (RQ3, D2)")
rows = []
for key, label in [("v1|1.7b", "Qwen3-1.7B (self), V1"), ("v2|1.7b", "Qwen3-1.7B (self), V2"),
                   ("v3|1.7b", "Qwen3-1.7B (self), V3"), ("v2|qwen235", "Qwen3-235B-A22B, V2"),
                   ("v3|qwen235", "Qwen3-235B-A22B, V3"), ("v2|gpt6luna", "GPT-6 Luna, V2")]:
    if key not in CAP:
        continue
    cap, capl = CAP[key], CAPL.get(key)
    rows.append([label, f"{share(cap['validated'], cap['rounds'])}%", str(cap["val_better"]),
                 f"{cap['runs_accepting']}/{cap['runs']}", f"{cap['wins']}/{cap['cells']}", pp(cap["mean_diff"]),
                 f"{capl['wins']}/{capl['losses']}" if capl else "n/a", pp(capl["mean_diff"]) if capl else "n/a"])
TABLE("Table 9. Refiner capability on the nine Tier-1 seed-42 cells; the target stays Qwen3-1.7B. Validated is "
      "the share of rounds reaching validation. Runs changed, strict wins and the strict mean refer to the final "
      "prompt versus P0 under strict extraction; the last two columns give wins/losses and the mean "
      "difference under lenient extraction. 1.7B rows pool up to three MPIR seeds, the others use MPIR seed 1.",
      header=["Refiner, variant", "Validated", "Val. better", "Runs changed", "Strict wins", "Strict mean",
              "Lenient wins/losses", "Lenient mean"],
      rows=rows, full=True)

wq, _ = wins_and_losses(CAP["v2|qwen235"])
wl, _ = wins_and_losses(CAP["v2|gpt6luna"])
FS2 = N["follow_seed2"]
P(f"With Qwen3-235B-A22B as refiner, {QW['validated']} of {QW['rounds']} rounds (both MPIR seeds pooled in "
  "Table 6) reach validation, almost no round is an echo, no candidate lacks the answer tag, "
  f"{CAP['v2|qwen235']['runs_accepting']} of 9 first-seed runs accept a candidate, and the final prompt is a "
  f"significant strict win in {CAP['v2|qwen235']['wins']} of 9 cells ({listing(wq)}); the mean difference is "
  f"{pp(CAP['v2|qwen235']['mean_diff'])} points. A second MPIR seed gives the same pattern: "
  f"{FS2['wins']} winning cells, mean {pp(FS2['mean_diff'])} points. GPT-6 Luna, run with minimal reasoning, "
  f"produces valid candidates in {share(CAP['v2|gpt6luna']['validated'], CAP['v2|gpt6luna']['rounds'])}% of "
  f"rounds but only {CAP['v2|gpt6luna']['val_better']} of them beat the baseline, with "
  f"{CAP['v2|gpt6luna']['wins']} winning cell ({listing(wl)}) and a mean difference of "
  f"{pp(CAP['v2|gpt6luna']['mean_diff'])} points. Under the pre-specified rule D2, with GPT-6 Luna as the "
  "strongest refiner, capability dependence is not supported: the mean exceeds that of the 1.7B refiner by "
  f"{pp(CAP['v2|gpt6luna']['mean_diff'] - CAP['v2|1.7b']['mean_diff'])} points but only "
  f"{CAP['v2|gpt6luna']['wins']} cell wins, against the required 3. With Qwen3-235B the strict rule is met, but "
  "only post hoc, since the role was assigned to GPT-6 Luna.")

P("Strict accuracy is, however, the wrong judge of these gains. Under lenient extraction (Table 9) "
  f"Qwen3-235B wins in {CAPL['v2|qwen235']['wins']} cell and loses in {CAPL['v2|qwen235']['losses']}, with a mean "
  f"difference of {pp(CAPL['v2|qwen235']['mean_diff'])} points. The three ProTeGi cells in which the strict "
  "gain is large are cells where the base prompt reasons about as well as before but wraps whole sentences "
  "in the answer tags; the accepted candidate (Figure B.3) adds an explicit instruction to wrap the final "
  "answer on its own line and a worked output example, which repairs the tag, while the model's "
  "reasoning under the long structured prompt is worse (lenient accuracy falls, for example from "
  f"{pct(N['winner_cells']['qwen235|hyperbaton__protegi__s42']['p0_lenient'])}% to "
  f"{pct(N['winner_cells']['qwen235|hyperbaton__protegi__s42']['final_lenient'])}% on hyperbaton with ProTeGi). "
  "A gate that uses strict extraction accepts such candidates, which is a concrete way in which answer "
  "extraction and the gate interact. The single gain that holds under both metrics is "
  f"hyperbaton with PromptWizard, whose base prompt already had the tag (strict = lenient = "
  f"{pct(LAND['hyperbaton__promptwizard__s42']['strict'])}%) and whose accepted candidates, from two different "
  "refiners, add a role, a step-back and numbered reasoning steps and reach "
  f"{pct(N['winner_cells']['qwen235|hyperbaton__promptwizard__s42']['final_strict'], 0)}% and "
  f"{pct(N['winner_cells']['gpt6luna|hyperbaton__promptwizard__s42']['final_strict'], 0)}%. One cell out of nine, "
  "on a base prompt that opens with an irrelevant sentence (Appendix A), is a gain that a rubric-based "
  "rewrite can plausibly claim; there is no multiplicity correction and no replication on another task. "
  "The two strong refiners differ, one belongs to the target's family and one was run with minimal reasoning, "
  "so capability, family and reasoning budget are confounded, and the experiment is an upper-bound probe, not a "
  "controlled capability scale.")

H3("5.6.1. Follow-Up Runs (Exploratory)")
V3Q = CAP["v3|qwen235"]
wv3q, lv3q = wins_and_losses(V3Q)
P("Because the strong-refiner signal appeared after the pre-specified analysis, two follow-up runs were added "
  "and are exploratory (Appendix F). First, the rubric-free gated rewrite (V3) with the same Qwen3-235B refiner "
  f"gives a mean strict difference of {pp(V3Q['mean_diff'])} points with {V3Q['wins']} winning cells of 9 "
  f"({listing(wv3q)}), against {pp(CAP['v2|qwen235']['mean_diff'])} points and {CAP['v2|qwen235']['wins']} wins "
  "for the rubric-guided variant, and a lenient mean of "
  f"{pp(CAPL['v3|qwen235']['mean_diff'])} points. A refiner that does not use the rubric therefore reaches "
  "gains of the same kind, so the rubric is not what produces them. Second, a second MPIR seed of the "
  f"rubric-guided variant gives {pp(FS2['mean_diff'])} points and {FS2['wins']} winning cells, so the strict "
  "pattern is stable across seeds.")

# ===========================================================================
H2("5.7. Serving Noise and Minimum Detectable Effect")
P("Each of the nine pilot base prompts was scored five times on the test and validation splits (identical "
  "prompt, greedy decoding). Test accuracy varies with a standard deviation of "
  f"{pct(NOISE['mean_sd'])} points and a range of {pct(NOISE['mean_range'])} points per prompt on average; between two "
  f"scorings of the same prompt {pct(NOISE['mean_discordance'])}% of examples change outcome on average "
  f"(up to {pct(NOISE['max_discordance'])}%), and the pilot and present scorings of the same prompt differ by "
  f"up to {pct(PVR['max_abs_diff'])} points. This is larger than the 2% assumed in the first submission and "
  "is the reason a 25-question validation split is noisy. For 200 test examples the minimum detectable "
  f"paired effect at 80% power and α = 0.05 is {pct(NOISE['mde_noise_only'])} points if only serving noise "
  f"separates two prompts (paired discordance {pct(NOISE['mean_discordance'])}%), and "
  f"{pct(NOISE['mde_for_discordance']['0.1'])} to {pct(NOISE['mde_for_discordance']['0.2'])} points if two different "
  "prompts disagree on 10-20% of examples, as candidates here do. The strict gains of the strong refiner "
  f"({', '.join(pp(d, 0) for _, d in wq)} points) are above this floor; effects of a few points are not detectable "
  "at these test sizes.")

# ===========================================================================
H2("5.8. Does the Rubric Judge Measure Anything? (RQ4)")
J1, J2 = JV["1.7b"], JV["gpt6luna"]
rows = []
for name, j in (("Qwen3-1.7B", J1), ("GPT-6 Luna (reference)", J2)):
    rows.append([name, pct(j["parse_rate"], 0) + "%", f"{j['mean_total']:.1f}", f"{j['sd_total']:.1f}",
                 f"{j['median_spearman_val_within_cell']:.2f}", f"{j['median_spearman_test_within_cell']:.2f}"])
TABLE("Table 10. Rubric totals (out of 35) assigned at temperature 0 to the base prompts and oracle-scored "
      f"candidates ({J1['n_prompts']} distinct prompts). Spearman correlations of the total with validation and "
      f"with test accuracy are computed within each of the nine cells over the {VC['n']} oracle-scored "
      "candidates (median shown); the two judges' totals correlate at "
      f"{N['judge_validity']['agreement']['spearman']:.2f}.",
      header=["Judge", "Totals parsed", "Mean total", "SD", "Spearman, val.", "Spearman, test"],
      rows=rows, full=True, colw=[0.31, 0.14, 0.13, 0.10, 0.16, 0.16])
P(f"The 1.7B judge gives a parseable total for {pct(J1['parse_rate'], 0)}% of prompts and scores them "
  f"{J1['mean_total'] - J2['mean_total']:.1f} points higher than the reference judge on average; its "
  f"totals for the same unchanged P0 vary with a standard deviation of {N['judge_repeat']['mean_sd_across_rounds']:.1f} points "
  "across sampled rounds. Both judges' totals correlate only weakly with accuracy "
  f"(median within-cell Spearman {J1['median_spearman_test_within_cell']:.2f} and "
  f"{J2['median_spearman_test_within_cell']:.2f} with test accuracy), and the two judges agree with each other "
  f"at {N['judge_validity']['agreement']['spearman']:.2f}. The rubric score is therefore at best a weak signal "
  "of prompt quality. No human ratings were collected; the reference judge is a stronger model, not a person, "
  "so this analysis bounds the judge's usefulness but does not validate the rubric against human judgement.")

# ===========================================================================
H2("5.9. Effect of the Original Defect")
LG6 = LEG["literal_g6"]
P("The original loop accepted the first round's outcome whatever it scored. With the 1.7B refiner and nine "
  "cells (MPIR seed 1, independent variant, oracle-scored candidates), applying that rule gives a mean "
  f"test-accuracy change of {pp(LEG['mean_legacy_delta'])} points relative to P0 when a fallback or echo round "
  f"returns P0, as the pilot's code did: the prompt changed in {LEG['legacy_changed_prompt']} of {LEG['n']} cells "
  f"and was worse than P0 in {LEG['legacy_worse']}, better in {LEG['legacy_better']}. Under the literal "
  "reading of protocol G6 (the first validated candidate is accepted unconditionally) the mean change is "
  f"{pp(LG6['mean_delta'])} points. Two cells account for most of either loss, the repaired loop changes accuracy "
  f"by {pp(LEG['mean_gated_delta'])} points on the same candidates, and the figure applies to this refiner only; "
  "its effect with the original GPT-4o refiner is unknown, because that configuration was not rerun.")

H2("5.10. Decision Rules")
TABLE("Table 11. Outcome of the decision rules (Section 4.1).",
      header=["Rule", "Criterion", "Observed", "Outcome"],
      rows=[
          ["D1", "At least 15 of 45 cells win and mean at least +2.0 pts; null if at most 2 wins and |mean| at most 1.0",
           f"{MG['wins']} wins, mean {pp(MG['mean_diff'], 2)}", MG["verdict"].capitalize()],
          ["D2, GPT-6 Luna", "Mean +2.0 over 1.7B and at least 3 of 9 wins",
           f"{pp(CAP['v2|gpt6luna']['mean_diff'])}; {CAP['v2|gpt6luna']['wins']} win", "Not met"],
          ["D2, Qwen3-235B (post hoc)", "Same",
           f"{pp(CAP['v2|qwen235']['mean_diff'])}; {CAP['v2|qwen235']['wins']} wins (lenient: "
           f"{pp(CAPL['v2|qwen235']['mean_diff'])}; {CAPL['v2|qwen235']['wins']} win)", "Met under strict only"],
          ["D3", "Best oracle candidate at least +3 pts in at least 3 of 9 cells although the gated final is not",
           f"{O1['cells_best_ge_3pts']}, {O2['cells_best_ge_3pts']}, {O3['cells_best_ge_3pts']} cells (V1, V2, V3)",
           "Not met"],
      ], full=True, colw=[0.15, 0.37, 0.30, 0.18])
