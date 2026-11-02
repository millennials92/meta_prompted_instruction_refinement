# -*- coding: utf-8 -*-
"""Results. Every number comes from results/revision/summary_numbers.json (mpir_study/analyze.py)."""
import json
import os

from blocks_api import H1, H2, H3, P, BULLETS, TABLE, CODE  # noqa: F401

from study_numbers import N
from rev2_numbers import OFF, NEW, AUDIT, LEN, MID

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
AB = OFF["acceptance_bounds"]
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
                 f"{c['runs_with_acceptance']}/{c['runs']}", f"{c['mean_distinct_candidates']:.1f}"])
TABLE("Table 6. Outcome of every refinement round (all runs of a variant pooled; refiner Qwen3-1.7B unless "
      "stated). Fallback: no delimiters. Echo: template phrase. Duplicate: same text as an earlier prompt. "
      "Validated rounds are split by validation score against the baseline (better/equal/worse). The last "
      "column pair gives the runs whose final prompt differs from P0 and the mean number of distinct candidate "
      "prompts per run.",
      header=["Variant", "Runs", "Rounds", "Fallback", "Echo", "Dup.", "Validated", "Val. better/equal/worse",
              "Accepted", "Runs changed", "Distinct per run"],
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
  f"{pct(LO['v1|1.7b']['share_without_tag'], 0)}% of the oracle-scored V1 and {pct(LO['v2|1.7b']['share_without_tag'], 0)}% "
  "of the oracle-scored V2 candidates (MPIR seed 1) lack it, so they score exactly zero under strict extraction; "
  "the candidates of the rubric-free rewrite, which has the line appended in code, and of the protected "
  "variants always carry it. The "
  "unprotected numbers therefore measure a failure of the 1.7B refiner to keep the format as much as a "
  "failure of the rubric. To separate the two we added the format-protected variants V1P and V2P, which "
  "append the answer-format line to every candidate. " + _prot +
  f"Refinement with the original, inconsistent rubric text (V2L) was no better: {V2L['accepted']} accepted "
  f"candidates in {V2L['rounds']} rounds and {share(V2L['fallback'] + V2L['echo'], V2L['rounds'])}% of rounds "
  "lost to fallbacks or echoes, so the inconsistency in the rubric text is not what limits the 1.7B refiner.")

FA, AL, AS_, PFM, RT = (OFF["candidate_format_audit"], OFF["audit_labels"], OFF["audit_samples"],
                        OFF["protected_failure_modes"], OFF["round_typology"])
BP = OFF["base_prompt_inventory"]
P(f"What the candidates keep. The second review asked what the refiner drops besides the format line. Among the "
  f"{FA['v1']['candidates']} V1 and {FA['v2']['candidates']} V2 candidates that were not echoes, none keeps the "
  f"answer-format line, and only {pct(FA['v1']['keep_bare_tag_example'], 0)}% and "
  f"{pct(FA['v2']['keep_bare_tag_example'], 0)}% contain a bare-tag worked example. This is largely because the "
  f"base prompts contain none: only {BP['with_bare_tag_example']} of the {BP['n']} base prompts (the PromptWizard "
  "prompts, whose in-context examples carry the bare tag) have such an example, and the exemplar-free APE and "
  "ProTeGi prompts have nothing for the refiner to keep. Where the base prompt had worked examples, "
  f"{FA['v2']['lost_format_and_examples_given_p0_has_examples']} of {FA['v2']['from_prompts_with_examples']} V2 "
  f"candidates (V1: {FA['v1']['lost_format_and_examples_given_p0_has_examples']} of "
  f"{FA['v1']['from_prompts_with_examples']}) lost the format line and also dropped at least half of the "
  "examples (an example is counted by its question marker). Appending the format line (V1P, V2P) therefore "
  "restores an instruction but not a demonstration of it.")
if PFM["v1p"]["outputs"] and PFM["v2p"]["outputs"]:
    P("Why the protected candidates are still far below P0. We re-scored the oracle-scored protected candidates in "
      f"one session ({PFM['v1p']['candidates']} V1P and {PFM['v2p']['candidates']} V2P candidates) and classified "
      "each response by what it does with the answer tags. In "
      f"{pct(PFM['v1p']['share_sentence_in_tag'], 0)}% (V1P) and {pct(PFM['v2p']['share_sentence_in_tag'], 0)}% (V2P) of "
      "responses the model wraps a whole sentence (more than 12 characters) inside the tags, which strict "
      f"extraction rejects, and in {pct(PFM['v1p']['share_no_tag'], 0)}% and {pct(PFM['v2p']['share_no_tag'], 0)}% it "
      f"writes no tags at all; only {pct(PFM['v1p']['share_bare_in_tag'], 0)}% and "
      f"{pct(PFM['v2p']['share_bare_in_tag'], 0)}% contain a bare option inside the tags. Only "
      f"{PFM['v1p']['candidates_with_bare_tag_example']} of {PFM['v1p']['candidates']} V1P and "
      f"{PFM['v2p']['candidates_with_bare_tag_example']} of {PFM['v2p']['candidates']} V2P candidates contain a "
      "bare-tag example. So the format-protected variants do not protect what matters: appending the instruction "
      "line does not teach the 1.7B target the tag convention, which is what the one-exemplar intervention of "
      "Section 5.1 does. This is why V2F (Section 5.3.1) re-attaches an example and not only the line.")
P(f"Audit of the round classification. The echo guard (five marker phrases) and the missing-delimiter rule decide "
  f"{RT['v1_v2_pooled']['fallback'] + RT['v1_v2_pooled']['echo']} of the {RT['v1_v2_pooled']['rounds']} V1 and V2 "
  f"rounds. Counting duplicates, {pct(RT['v2']['share_fallback_echo_duplicate'], 0)}% of the chained rounds never "
  f"reached validation; the 60% quoted in the first version of this paper counted fallbacks and echoes only "
  f"({RT['v2']['fallback'] + RT['v2']['echo']} of {RT['v2']['rounds']}, {pct(RT['v2']['share_fallback_echo'], 0)}%). "
  f"We drew a random sample of {AL['echo_n']} echo and {AL['fallback_n']} fallback rounds (1.7B refiner, V1, V2, V1P, "
  "V2P) and labelled each from the logged text. The labelling was done by a language-model assistant, not by a "
  "human rater, working from the head and tail of each text, so it is a screening audit; the sample file and labels "
  "are in the released logs for re-checking. "
  f"{AL['echo_false_positive']} of {AL['echo_n']} sampled echoes were false positives, usable APE-style "
  f"induction prompts that contain a marker phrase (rate {pct(AL['echo_fp_rate'], 0)}%, one-sided 95% upper bound "
  f"{pct(AL['echo_fp_upper_95_one_sided'], 0)}%); the rest were the rubric text, a critique or a degenerate induction "
  f"text. Of the sampled fallbacks, {AL['fallback_causes'].get('no_refined_prompt', 0)} never presented a refined prompt "
  f"(critique, suggestions or an answer to a question), {AL['fallback_causes'].get('refined_prompt_without_end', 0)} "
  f"presented one without a closing tag, none was cut off by a length limit, and {AL['fallback_recoverable']} of "
  f"{AL['fallback_n']} ({pct(AL['fallback_recoverable_rate'], 0)}%) were recoverable by a tolerant parser. Over all "
  f"fallbacks, {AS_['fallback_causes'].get('start_without_end', 0)} of {AS_['pool_sizes']['fallback']} contain the "
  f"opening delimiter but no valid pair (the closing tag is usually malformed), and the median fallback response is "
  f"{AS_['fallback_length_chars_median'] / 4.36:.0f} tokens long against the 3000-token limit "
  f"(90th percentile {AS_['fallback_length_chars_p90'] / 4.36:.0f}); only {AS_['near_token_limit']['refine_ge_90pct_of_3000']} "
  f"of {AS_['near_token_limit']['rounds']} refinement responses come within 10% of that limit, so truncation of "
  "the refinement output does not explain the fallbacks. The evaluation (judge) call is closer to its 1500-token "
  f"limit: {AS_['near_token_limit']['judge_ge_90pct_of_1500']} of {AS_['near_token_limit']['rounds']} judge "
  "responses are within 10% of it, which can cut a rubric critique but not the rewrite that follows. We conclude that "
  "the guards are mostly right (about 90% precision for echoes), that the dominant failure is a refiner that "
  "critiques instead of rewriting, and that a tolerant parser would recover only a small minority of the lost rounds.")

# ===========================================================================
H2("5.3. The Repaired Layer on 45 Cells (RQ1, D1)")
GO = N["grid_by_optimizer"]
P(f"With the 1.7B model refining its own prompts through the chained variant, the final prompt equals P0 in "
  f"{MG['runs'] - MG['runs_accepting']} of {MG['runs']} runs ({MG['n_cells']} cells, one to three MPIR seeds each). "
  f"Only {MG['runs_accepting']} run accepted a candidate. No cell is a win: the mean paired difference "
  f"over cells is {pp(MG['mean_diff'], 2)} points (hierarchical bootstrap interval "
  f"[{pp(MG['hier']['lo'], 2)}, {pp(MG['hier']['hi'], 2)}], which is almost degenerate because 44 of 45 cells "
  f"are exactly zero), and {MG['wins']} of {MG['n_cells']} cells satisfy the win criterion. By the "
  f"decision rule fixed in advance (D1) the outcome is '{MG['verdict']}'. The informative quantity is the acceptance "
  f"rate, not the degenerate interval: the exact (Clopper-Pearson) upper confidence bound for {AB['accepting_runs']} "
  f"acceptance in {AB['runs']} runs is {100 * AB['run_upper_one_sided_95']:.1f}% one-sided at 95% "
  f"({100 * AB['run_upper_two_sided_95']:.1f}% two-sided); the first submission's '7%' was the two-sided value "
  "rounded up and was not labelled as such. Runs of one cell share a base prompt and are not independent, so "
  f"we also give the bound over cells: {AB['cells_with_acceptance']} of {AB['cells']} cells ever accepted a "
  f"candidate, with an upper bound of {100 * AB['cell_upper_one_sided_95']:.1f}% one-sided "
  f"({100 * AB['cell_upper_two_sided_95']:.1f}% two-sided). The accepting run is one PromptWizard run on "
  "hyperbaton; the acceptance rate by cell is 0 for every other task and optimizer. The result does not "
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
  f"candidate that beats P0 by 3 points or more ({O1['cells_best_ge_3pts']} of 9 and {O2['cells_best_ge_3pts']} of 8 "
  "cells with a validated candidate); these candidates mostly lack the answer-format line (Section 5.2), so the figure largely "
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
  f"{CAP['v3|1.7b']['wins']} winning cells ({listing(w3)}; a win of 3 points or less is below the 5-point minimum "
  "detectable effect of Section 5.7 and is within serving noise) and {len(l3)} cell(s) in which the gate accepted a "
  f"significantly harmful rewrite ({listing(l3)}), for a mean strict difference of {pp(CAP['v3|1.7b']['mean_diff'])} "
  f"points; under lenient extraction it has {CAPL['v3|1.7b']['wins']} wins and {CAPL['v3|1.7b']['losses']} losses "
  f"(mean {pp(CAPL['v3|1.7b']['mean_diff'])} points, against {pp(CAPL['v1|1.7b']['mean_diff'])} for V1 and "
  f"{pp(CAPL['v2|1.7b']['mean_diff'])} for V2). So the gate does not reliably protect against harm, since 25 "
  "questions are noisy, and neither variant helps with this refiner: the rubric-free rewrite has more strict "
  "wins, partly because of the format asymmetry, and is not better under lenient scoring, so the rubric adds "
  "nothing detectable and we cannot say that unstructured rewriting is worse.")

# ===========================================================================
H2("5.6. A More Capable Refiner (RQ3, D2)")
rows = []
for key, label in [("v1|1.7b", "Qwen3-1.7B (self), V1"), ("v2|1.7b", "Qwen3-1.7B (self), V2"),
                   ("v3|1.7b", "Qwen3-1.7B (self), V3"), ("v1p|1.7b", "Qwen3-1.7B (self), V1P"),
                   ("v2p|1.7b", "Qwen3-1.7B (self), V2P"), ("v2|qwen235", "Qwen3-235B-A22B, V2"),
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
  f"{pp(CAP['v2|gpt6luna']['mean_diff'])} points. Under the rule D2 fixed in advance, with GPT-6 Luna as the "
  "strongest refiner, capability dependence is not supported: the mean exceeds that of the 1.7B refiner by "
  f"{pp(CAP['v2|gpt6luna']['mean_diff'] - CAP['v2|1.7b']['mean_diff'])} points but only "
  f"{CAP['v2|gpt6luna']['wins']} cell wins, against the required 3. With Qwen3-235B the strict rule is met, but "
  "only post hoc, since the role was assigned to GPT-6 Luna.")

FS2L = N["follow_seed2_lenient"]
P("Strict accuracy is, however, a poor judge of these gains. Under lenient extraction (Table 9), in the first "
  f"MPIR seed Qwen3-235B wins in {CAPL['v2|qwen235']['wins']} cell and loses significantly in "
  f"{CAPL['v2|qwen235']['losses']}, with a mean difference of {pp(CAPL['v2|qwen235']['mean_diff'])} points; in the "
  f"second seed the lenient mean is {pp(FS2L['mean_diff'])} points with {FS2L['wins']} win and "
  f"{FS2L['losses']} significant losses, so the other strict gains vanish there instead of reversing. The three "
  "ProTeGi cells in which the strict gain is large are cells where the base prompt reasons about as well as "
  "before but wraps whole sentences in the answer tags; the accepted candidate (Figure B.3) adds an explicit "
  "instruction to wrap the final answer on its own line and a worked output example, which repairs the tag, "
  "while lenient accuracy does not improve, and in the first seed falls (we did not test why; the long "
  "structured prompt may make the 1.7B target reason worse), for example from "
  f"{pct(N['winner_cells']['qwen235|hyperbaton__protegi__s42']['p0_lenient'])}% to "
  f"{pct(N['winner_cells']['qwen235|hyperbaton__protegi__s42']['final_lenient'])}% on hyperbaton with ProTeGi. "
  "A gate that uses strict extraction accepts such candidates, which is a concrete way in which answer "
  "extraction and the gate interact. The single gain that holds under both metrics is "
  f"hyperbaton with PromptWizard, whose base prompt already had the tag (strict = lenient = "
  f"{pct(LAND['hyperbaton__promptwizard__s42']['strict'])}%) and whose accepted candidates, from two different "
  "refiners, add a role, a step-back and numbered reasoning steps and reach "
  f"{pct(N['winner_cells']['qwen235|hyperbaton__promptwizard__s42']['final_strict'], 0)}% and "
  f"{pct(N['winner_cells']['gpt6luna|hyperbaton__promptwizard__s42']['final_strict'], 0)}%. One cell out of nine, "
  "on a near-chance base prompt (59% against 50% chance) whose instruction is garbled, is a gain that a "
  "rubric-based rewrite can plausibly claim; there is no multiplicity correction and no replication on another task. "
  "The two strong refiners differ, one belongs to the target's family and one was run with minimal reasoning, "
  "so capability, family and reasoning budget are confounded, and the experiment is an upper-bound probe, not a "
  "controlled capability scale.")

H3("5.6.1. Follow-Up Runs (Exploratory)")
V3Q = CAP["v3|qwen235"]
wv3q, lv3q = wins_and_losses(V3Q)
P("Because the strong-refiner signal appeared after the analysis fixed in advance, two follow-up runs were added "
  "and are exploratory (Appendix F). First, the rubric-free gated rewrite (V3) with the same Qwen3-235B refiner "
  f"gives a mean strict difference of {pp(V3Q['mean_diff'])} points with {V3Q['wins']} winning cells of 9 "
  f"({listing(wv3q)}), against {pp(CAP['v2|qwen235']['mean_diff'])} points and {CAP['v2|qwen235']['wins']} wins "
  "for the rubric-guided variant, and a lenient mean of "
  f"{pp(CAPL['v3|qwen235']['mean_diff'])} points. A refiner that does not use the rubric therefore reaches "
  "gains of the same kind, so the rubric is not what produces them. Second, a second MPIR seed of the "
  f"rubric-guided variant gives {pp(FS2['mean_diff'])} points and {FS2['wins']} winning cells, so the strict "
  f"pattern is stable across seeds; under lenient extraction its mean is {pp(FS2L['mean_diff'])} points with "
  f"{FS2L['wins']} win (hyperbaton with PromptWizard) and {FS2L['losses']} significant losses.")

RV = OFF["rubric_vs_free"]["qwen235"]
rows = []
for key, e in RV["cells"].items():
    a, b, c = e["v3_minus_p0"], e["v2m1_minus_p0"], e["v2m1_minus_v3"]
    rows.append([cell_label(key), pct(e["p0_test"]), pp(a["diff"]), pp(b["diff"]),
                 f"{pp(c['diff'])} [{pp(c['lo'])}, {pp(c['hi'])}]"])
TABLE("Table 9b. Paired comparison of the rubric-guided (V2) and rubric-free (V3) variants with the same "
      "Qwen3-235B refiner on the nine Tier-1 seed-42 cells, MPIR seed 1 for both. Differences are in test-accuracy "
      "points of the final prompt, paired over the 200 test examples (96 for penguins_in_a_table); the last column "
      "is V2 minus V3 with a 95% paired-bootstrap interval. One run per condition and cell; the minimum "
      f"detectable effect is {pct(NOISE['mde_noise_only'])} points (Section 5.7).",
      header=["Cell", "P0 strict %", "V3 - P0", "V2 - P0", "V2 - V3 [95% CI]"],
      rows=rows, full=True, colw=[0.30, 0.12, 0.14, 0.14, 0.30])
P(f"Reviewer 1 asked for a cell-level paired comparison. Table 9b gives it. Averaged over the {RV['n_cells']} cells, "
  f"V2 exceeds V3 by {pp(RV['mean_v2_minus_v3'])} points (bootstrap over cells [{pp(RV['lo'])}, {pp(RV['hi'])}]); "
  f"V2 is ahead in {RV['cells_v2_better']} cells, V3 in {RV['cells_v3_better']} and {RV['cells_tied']} are tied, and "
  f"the paired interval excludes zero in {RV['cells_ci_excludes_zero']} cells. With the second V2 seed the mean "
  f"difference is {pp(RV['mean_v2seed2_minus_v3'])} points, so the sign is stable but the size is not. The data "
  "therefore cannot separate 'the rubric adds nothing' from 'the rubric adds a modest benefit' with this refiner: "
  "the interval for the mean difference includes zero and benefits of several points, each condition has one or two "
  "runs per cell, and the refiner's capability, family and reasoning budget are confounded with the choice of "
  "model. We therefore withdraw the sentence that a rubric-free rewrite 'reaches similar strict gains' as a "
  "statement of equivalence: it reaches gains of the same kind, and the experiment is exploratory. The one V3 win "
  "with the 1.7B refiner (Section 5.5), ruin_names with ProTeGi at +2.8 points, is below the minimum detectable "
  "effect of 5.0 points and within serving noise; we label it as such there.")

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

JZ = OFF["judge_without_format_zeros"]
_jz = lambda k: JZ.get(k)
_rows = []
for judge, name in (("1.7b", "Qwen3-1.7B"), ("gpt6luna", "GPT-6 Luna"), ("8b", "Qwen3-8B-AWQ (independent of all reported refiners)")):
    a_, b_ = _jz(f"{judge}|all"), _jz(f"{judge}|test_above_zero")
    if a_ and b_:
        _rows.append([name, str(a_["candidates"]), pct(a_["zero_share"], 0) + "%", f"{a_['median_spearman_test']:.2f}",
                      str(b_["candidates"]), f"{b_['median_spearman_test']:.2f}"])
if _rows:
    TABLE("Table 10b. Within-cell median Spearman correlation of the rubric total with test accuracy over all "
          "oracle-scored candidates and after excluding candidates whose strict test accuracy is exactly zero "
          "(format loss). The 8B judge is a local Qwen3-8B-AWQ model that is not one of the reported refiners.",
          header=["Judge", "Candidates", "Zero accuracy", "Spearman (all)", "Candidates > 0", "Spearman (> 0)"],
          rows=_rows, full=True, colw=[0.34, 0.12, 0.12, 0.14, 0.14, 0.14])
    P("The second review noted that a large share of the oracle-scored candidates score exactly zero because they "
      f"lost the answer format ({pct(JZ['1.7b|all']['zero_share'], 0)}%), which can inflate or deflate a "
      "correlation with the rubric total, and that GPT-6 Luna is both the reference judge and one of the refiners. "
      "Table 10b removes the zero-accuracy candidates: the median correlation falls to "
      f"{JZ['1.7b|test_above_zero']['median_spearman_test']:.2f} (1.7B judge) and "
      f"{JZ['gpt6luna|test_above_zero']['median_spearman_test']:.2f} (GPT-6 Luna) over "
      f"{JZ['1.7b|test_above_zero']['candidates']} and {JZ['gpt6luna|test_above_zero']['candidates']} candidates, so the "
      "weak association of Table 10 was partly the rubric rewarding candidates that had kept the format, and among "
      "usable candidates the rubric score says very little about accuracy. "
      + ("A judge that is not one of the refiners (the local 8B model) gives "
         f"{JZ['8b|all']['median_spearman_test']:.2f} over all candidates and "
         f"{JZ['8b|test_above_zero']['median_spearman_test']:.2f} without the zero-accuracy ones. "
         if "8b|all" in JZ else "")
      + "A criterion-level ablation, which would show which of the seven criteria carry any signal, was not run; it is "
      "listed among the limitations.")

# ===========================================================================
H2("5.9. Effect of the Original Defect")
LG6 = LEG["literal_g6"]
P("The original loop accepted the first round's outcome whatever it scored. With the 1.7B refiner and nine "
  "cells (MPIR seed 1, independent variant, oracle-scored candidates), applying that rule gives a mean "
  f"test-accuracy change of {pp(LEG['mean_legacy_delta'])} points relative to P0 when a fallback or echo round "
  f"returns P0, as the pilot's code did: the prompt changed in {LEG['legacy_changed_prompt']} of {LEG['n']} cells "
  f"and was worse than P0 in {LEG['legacy_worse']}, better in {LEG['legacy_better']}. Under the literal "
  "reading of protocol G6 (the first validated candidate is accepted unconditionally) the mean change is "
  f"{pp(LG6['mean_delta'])} points. Two cells account for most of the smaller loss, the repaired loop changes accuracy "
  f"by {pp(LEG['mean_gated_delta'])} points on the same candidates, and the figures apply to this refiner only; "
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
