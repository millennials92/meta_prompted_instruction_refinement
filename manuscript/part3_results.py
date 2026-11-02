# -*- coding: utf-8 -*-
"""Results. Every number comes from results/revision/summary_numbers.json (mpir_study/analyze.py)."""
import json
import os

from blocks_api import H1, H2, H3, P, BULLETS, TABLE, CODE  # noqa: F401

from study_numbers import N
from rev2_numbers import OFF, NEW, AUDIT, LEN, MID, PP

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

P("Reader's guide. The variants of Table 2 fall into three groups. The main grid is V2 (chained, no format protection), the primary test of decision rule D1 on 45 cells; on the nine Tier-1 cells it is accompanied by V0 (the pilot's loop), V1 (independent rounds), V2L (original rubric text) and V3 (rubric-free gated rewrite). The format repairs are V1P and V2P, which append the answer-format line, and V2F, which also keeps a worked example and extends the gate's baseline (44 cells), with V3F as its rubric-free control (45 cells). The stronger-refiner probes run V2 and V3 with Qwen3-235B, Qwen3-8B-AWQ and, for V2, GPT-6 Luna, and the protected V2F and V3F with Qwen3-235B and GPT-6 Luna. Unless stated otherwise the refiner is Qwen3-1.7B, 'strict' and 'lenient' are the two answer-extraction metrics of Section 4.5, and 'the nine Tier-1 cells' are the seed-42 cells of hyperbaton, ruin_names and penguins_in_a_table (Table R.4 lists every variant).")

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

P("Two checks bear on how much of this is a harness effect. First, appending one worked format exemplar (the first BBH-author exemplar, with its bare tag) to an exemplar-free base prompt raised strict accuracy by +19.5 points on average for APE and +42.3 points for ProTeGi over 15 cells each (the paired-bootstrap interval excludes zero in 9 and 15 cells; the range is -45.0 to +72.9 points, the negative end being two APE prompts, on disambiguation_qa and on ruin_names with seed 43, whose accuracy fell to zero). Under lenient extraction the same intervention gains +2.3 and +10.2 points, so most of the strict gain is the tag, and the exemplar's effect on reasoning is modest (Section 5.3.1 separates the tag from the worked reasoning). Second, the same prompt does not score the same twice: the strict accuracy of the identical nine prompts, scored in the pilot and again for this study, differs by 2.6 points on average and up to 7.0 points (Section 5.7), so accuracy differences of this size are not reliably attributable to a prompt on this serving stack.")

H3("5.1.1. Validating the Lenient Parser")
LV = LEN
from scipy.stats import beta as _beta
_cpu = lambda k, n: float(_beta.ppf(0.95, k + 1, n - k))
_a, _b, _c = LV["binary"], LV["multi"], LV["all"]
P("Neither metric is ground truth, and the lenient parser has to be checked against labels. We drew 400 test responses at random from the scoring cache of the 2026-11-02 session, about 100 for each of "
  "four groups: binary tasks (hyperbaton, sports_understanding) and multi-option tasks, each split into responses that "
  "strict extraction accepts and responses that it rejects, since the lenient parser matters only for the latter. "
  "Each response was shown without the gold answer and without either parser's verdict, as its first 500 and last 900 "
  "characters, to a language-model assistant that named the option the response commits to (or 'none'). The "
  "labelling was therefore done by an AI assistant, not by human raters, and for the longest responses it saw "
  "mainly the end of the text; we release the sample, the key and the labels for re-checking. The labelled option was "
  "compared with the gold answer to give a truth value for every response.")
_rows = []
for _name, _d in (("Binary tasks", _a), ("Multi-option tasks", _b), ("All", _c)):
    for _m in ("strict", "lenient"):
        _e = _d[_m]
        _rows.append([_name, _m, str(_d["n"]), pct(_e["precision"]), pct(_e["recall"]), pct(_e["agreement"]),
                      str(_e["fp"]), str(_e["fn"])])
TABLE("Table R.1. Agreement of the strict and the lenient parser with labelled truth on 400 sampled test responses "
      "(half of each task group strict-accepted, half strict-rejected, so recall and precision describe this sample, "
      "not the population). Precision is the share of accepted responses that are truly correct and recall the share "
      "of truly correct responses that are accepted; fp and fn are counts.",
      header=["Tasks", "Parser", "n", "Precision %", "Recall %", "Agreement %", "fp", "fn"],
      rows=_rows, full=True, colw=[0.22, 0.12, 0.08, 0.14, 0.12, 0.14, 0.09, 0.09])
P(f"Among the {_c['strict_rejected']['n']} sampled responses that strict extraction rejects, "
  f"{_c['strict_rejected']['truly_correct']} ({share(_c['strict_rejected']['truly_correct'], _c['strict_rejected']['n'])}%) "
  f"are truly correct, so about a quarter of the responses that strict scoring rejects are in fact correct, and the lenient "
  f"parser accepts {_c['strict_rejected']['lenient_accepts_and_correct']} of those {_c['strict_rejected']['truly_correct']} "
  f"and wrongly accepts {_c['strict_rejected']['lenient_accepts'] - _c['strict_rejected']['lenient_accepts_and_correct']}. "
  f"Lenient precision is {pct(_c['lenient']['precision'])}% and recall {pct(_c['lenient']['recall'])}% overall "
  f"(binary tasks {pct(_a['lenient']['precision'])}% and {pct(_a['lenient']['recall'])}%, multi-option tasks "
  f"{pct(_b['lenient']['precision'])}% and {pct(_b['lenient']['recall'])}%), against a strict recall of "
  f"{pct(_c['strict']['recall'])}%. For two-option tasks, where a parser that credits a response "
  "naming exactly one option could be fooled by a stray mention, precision is still "
  f"{pct(_a['lenient']['precision'])}% (two false positives among "
  f"{_a['lenient']['fp'] + _a['lenient']['tn']} truly incorrect binary-task responses; one-sided 95% upper bound on the "
  f"false-positive rate {pct(_cpu(_a['lenient']['fp'], _a['lenient']['fp'] + _a['lenient']['tn']), 0)}%). On this AI-labelled sample the lenient parser tracks "
  "what the model answers far better than strict extraction does, and strict extraction mainly measures formatting; a "
  "human re-labelling of a subsample, which we did not do, would be needed to confirm the labels. From here on the "
  "decision rules are restated on the lenient metric as well, and both are reported; the gate itself still uses the "
  "strict metric, which is how MPIR is defined.")

# ===========================================================================
H2("5.2. What the Rounds Were")
P("Table 6 lists the outcome of every round (per-cell records for V1 and V2 are in Appendix E). The pilot's legacy loop (V0, nine cells) produced on average 4.3 distinct candidate prompts per run although decoding was greedy, so the rounds were not identical copies; they differed only through serving non-determinism, not by design. The pilot's own prompts were byte-identical in all nine cells, yet in our re-run of the same loop 1 of nine runs changed its prompt, which shows how fragile even a greedy comparison is. In the repaired variants the rounds differ by sampling and, for V2, by history.")

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
P("For the 1.7B refiner, 337 of the 560 chained rounds (60%) ended in a fallback or echo and never reached validation (66% once the 30 duplicates are counted). Of the 193 validated candidates, 126 scored below the baseline, 66 tied and 1 scored higher, so most validated candidates are far worse, not tied. The main reason is that the refinement request does not ask the refiner to keep the answer-format line: 93% of the oracle-scored V1 and 95% of the oracle-scored V2 candidates (MPIR seed 1) lack it and score exactly zero under strict extraction, whereas candidates of the rubric-free rewrite (which has the line appended in code) and of the protected variants always carry it. The unprotected rows of Table 6 therefore measure the refiner's failure to keep the format at least as much as any failure of the rubric. To separate the two we added V1P and V2P, which append the format line to every candidate. They still reached validation in only 30% (V1P, 70 rounds) and 27% (V2P, 56 rounds) of rounds, and 5 of their 36 validated candidates beat the baseline on validation. They were run on the nine Tier-1 seed-42 cells and finished on 5 (V1P) and 8 (V2P) of them; the rest were not completed for compute reasons. The original, inconsistent rubric text (V2L) was no better: 0 accepted candidates in 63 rounds and 63% of rounds lost to fallbacks or echoes, so the inconsistency in the rubric text is not what limits the 1.7B refiner.")

FA, AL, AS_, PFM, RT = (OFF["candidate_format_audit"], OFF["audit_labels"], OFF["audit_samples"],
                        OFF["protected_failure_modes"], OFF["round_typology"])
BP = OFF["base_prompt_inventory"]
P('What the candidates keep. Among the 120 V1 and 223 V2 rounds that returned delimiters and were not echoes, none keeps the exact answer-format sentence (the 93-95% above counts oracle-scored candidates that lack the answer tag altogether), and only 8% and 7% contain a bare-tag worked example. Only 15 of the 45 base prompts (the PromptWizard prompts) contain such an example, so the exemplar-free APE and ProTeGi prompts gave the refiner nothing to keep; where the base prompt did have worked examples, 48 of 67 V2 candidates (V1: 19 of 32) lost the format line and also dropped at least half of the examples. Appending the format line (V1P, V2P) restores an instruction but not a demonstration of it. We re-scored 5 V1P and 15 V2P candidates in one session: in 40% (V1P) and 31% (V2P) of responses the model wraps a whole sentence (more than 12 characters) inside the tags, which strict extraction rejects, in 20% and 59% it writes no tags at all, and only 40% and 9% contain a bare option inside the tags. Only 3 of 5 V1P and 1 of 15 V2P candidates contain a bare-tag example. The instruction line alone therefore does not teach the 1.7B target the tag convention, which the one-exemplar intervention of Section 5.1 does; this is why V2F (Section 5.3.1) re-attaches an example and not only the line.')
P('Audit of the round classification. The echo guard (five marker phrases) and the missing-delimiter rule decide 406 of the 749 V1 and V2 rounds. A language-model assistant, not a human rater, labelled a random sample of 40 echo and 40 fallback rounds (1.7B refiner, V1, V2, V1P, V2P) from the head and tail of each logged text, so this is a screening audit; the sample and labels are released for re-checking. 4 of 40 sampled echoes were false positives (rate 10%, one-sided 95% upper bound 21%). Of the sampled fallbacks, 25 never presented a refined prompt, 10 presented one without a closing tag, none was cut off by a length limit, and 6 of 40 (15%) were recoverable by a tolerant parser. Truncation does not explain the fallbacks: the median fallback response is 631 tokens long against the 3000-token limit (90th percentile 986), and only 2 of 1498 refinement responses come within 10% of it (the judge call is closer to its 1500-token limit, 77 of 1498, which can cut a critique but not the rewrite that follows). The guards are therefore mostly right (about 90% precision for echoes), the dominant failure is a refiner that critiques instead of rewriting, and a tolerant parser would recover only a small minority of the lost rounds.')

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
  f"({100 * AB['run_upper_two_sided_95']:.1f}% two-sided). Runs of one cell share a base prompt and are not independent, so "
  f"we also give the bound over cells: {AB['cells_with_acceptance']} of {AB['cells']} cells ever accepted a "
  f"candidate, with an upper bound of {100 * AB['cell_upper_one_sided_95']:.1f}% one-sided "
  f"({100 * AB['cell_upper_two_sided_95']:.1f}% two-sided). The accepting run is one PromptWizard run on "
  "hyperbaton; the acceptance rate by cell is 0 for every other task and optimizer. The result does not "
  "depend on the base optimizer: "
  + "; ".join(f"{OPT_NAME[o]} {GO[o]['accepted_runs']} of {GO[o]['runs']} runs changed" for o in OPTS) + ". "
  "D1 was specified and run for the unprotected V2, and Section 5.2 shows that this procedure is limited by "
  "the 1.7B refiner's inability to produce usable, format-preserving candidates; the null is therefore a "
  "statement about this procedure with this refiner, and not about rubric-guided refinement in general. "
  "Table 7 lists the only task-optimizer combinations that changed. The tests of the pilot (McNemar, GEE, Wilcoxon on three tasks) are not "
  "reported, because with identical prompts they cannot be informative.")

GT = N["grid_by_task_optimizer"]
_nonzero = [(t, o, g) for t in TASKS_T1 + TASKS_T2 for o in OPTS
            for g in [GT.get(f"{t}|{o}")] if g and (g["accepted_runs"] > 0 or g["mean_diff"] != 0)]
_rows = [[t, OPT_NAME[o], f"{g['accepted_runs']} of {g['seeds']}", pp(g["mean_diff"])] for t, o, g in _nonzero]
TABLE("Table 7. Per task and optimizer, chained variant, 1.7B refiner: the only task-optimizer combinations in which "
      "any run's final prompt differs from P0 (runs changed out of all MPIR runs, and the mean paired test-accuracy "
      f"difference in points). The other {len(TASKS_T1 + TASKS_T2) * len(OPTS) - len(_nonzero)} of the "
      f"{len(TASKS_T1 + TASKS_T2) * len(OPTS)} combinations are identical to P0 in every run, with a difference of exactly zero.",
      header=["Task", "Optimizer", "Runs changed", "Mean difference"], rows=_rows, full=True, colw=[0.40, 0.22, 0.20, 0.18])

H3("5.3.1. A Format- and Example-Preserving Re-Run (V2F) and a Positive Control")
G, CT = NEW["grid_v2f"], NEW["controls"]
CTO = CT["by_optimizer"]
P("Two things are needed to interpret the V2 null: a re-run of the main grid with a treatment that preserves the "
  "answer format, and evidence that the gate and the harness can accept a known-good candidate. We report the positive "
  "control first, because it says what a null result could mean.")
P("Positive control. For each of the 45 base prompts we offered the gate a candidate known to be good under strict scoring, P0 plus the first worked example of the BBH authors' prompt (the intervention of Section 5.1), and applied the usual rule, strict improvement of the 25-question validation score over P0. The gate accepted it in 29 of 45 cells overall: 15 of 15 ProTeGi cells, 9 of 15 APE cells and 5 of 15 PromptWizard cells; the PromptWizard prompts already contain bare-tag examples, so those acceptances do not test a known-good candidate. Of the 26 APE and ProTeGi cells in which the exemplar raises strict test accuracy, the gate accepted 24, and it never accepted the exemplar where the exemplar did not help on test (0 cases). Those 26 cells are defined by the test outcome, so 24 of 26 is a check on sensitivity to a large effect, not an acceptance rate for unknown candidates. The gate and harness can therefore detect a large strict-metric effect when one is offered, and the null of V2 is not an artefact of a gate that accepts nothing.")
P('Bare-tag control. The exemplar adds a worked chain of thought as well as the tag. Replacing it by one sentence that shows only the final-answer tag raises strict test accuracy by +19.4 points for APE and +34.2 points for ProTeGi, against +20.0 and +42.8 points for the full exemplar (the paired interval excludes zero in 10 of 15 APE and 15 of 15 ProTeGi cells for the bare tag). Most of the effect is therefore the tag convention, and the worked reasoning adds about 1 point for APE and 9 for ProTeGi. The effect is heterogeneous: it ranges from losses of up to 45 points on two APE prompts to gains of more than 70 points on ProTeGi prompts. These values were rescored in a later session; Section 5.1 reports +19.5 and +42.3 from the earlier session.')

_ver = ("reproduces" if (G["wins_vs_start"] >= G["runs"] / 3 and G["mean_diff_vs_start"] >= 0.02) else
        "null" if (G["wins_vs_start"] <= 2 * G["runs"] / 45 and abs(G["mean_diff_vs_start"]) <= 0.01) else "mixed")
_len_ver = ("reproduces" if (G["lenient_wins_vs_start"] >= G["runs"] / 3 and G["lenient_mean_diff_vs_start"] >= 0.02) else
            "null" if (G["lenient_wins_vs_start"] <= 2 * G["runs"] / 45 and abs(G["lenient_mean_diff_vs_start"]) <= 0.01)
            else "mixed")
P("V2F. The format-preserving variant (Section 3.2, Table 2) was run with the 1.7B model as refiner and target on "
  f"{G['runs']} of the 45 cells, one MPIR seed each (the 45th cell, geometric_shapes with PromptWizard, "
  "could not be run: its refinement request exceeds the 8192-token context window by one token). Its gate baseline is P0 extended with the bare-tag worked "
  "example, so a candidate cannot win merely by restoring the format, and its re-attachment step guarantees that every "
  f"candidate carries the format line and the example. The final prompt differs from the baseline in {G['accepted_runs']} of "
  f"{G['runs']} runs (V2: {AB['accepting_runs']} of {AB['runs']}), an acceptance rate of {pct(G['accepted_runs'] / G['runs'], 0)}%; {share(G['fallback'] + G['echo'], G['rounds'])}% of the "
  f"{G['rounds']} rounds still ended in a fallback or echo, {G['validated']} rounds were validated and "
  f"{G['val_better']} of those beat the baseline on validation ({G['val_worse']} scored below it). Even when told to keep them, the refiner's raw reply keeps "
  f"the format line in {pct(G['raw_keep_format_line'], 0)}% of its candidates and the example header in "
  f"{pct(G['raw_keep_exemplar_header'], 0)}%, so the re-attachment step does real work. Measured against its own "
  f"baseline, V2F's final prompts win in {G['wins_vs_start']} cells under strict scoring (mean "
  f"{pp(G['mean_diff_vs_start'])} points) and in {G['lenient_wins_vs_start']} cell{'' if G['lenient_wins_vs_start'] == 1 else 's'} under lenient scoring, with "
  f"{G['lenient_losses_vs_start']} significant lenient losses (mean {pp(G['lenient_mean_diff_vs_start'])} points). "
  f"Measured against the unextended P0, which is what a practitioner would compare with, V2F wins in "
  f"{G['wins_vs_p0']} cells strictly (mean {pp(G['mean_diff_vs_p0'])} points), but that difference is the exemplar, "
  f"which the positive control already shows, and the lenient wins against P0 number {G['lenient_wins_vs_p0']}. "
  f"Applying decision rule D1 to V2F against its own baseline gives '{_ver}' on strict scoring and '{_len_ver}' on "
  "lenient scoring (the rule scaled to the number of cells run).")
P("What this shows. Giving the 1.7B refiner a treatment that can succeed raises the acceptance rate far above that of V2 (15 of 44 runs against 1 of 80), so the V2 null was in part a property of the unprotected treatment. It does not, however, produce reliable gains: against its own baseline the accepted candidates win in 4 cells under strict scoring and in 1 under lenient scoring, with 3 significant lenient losses; its rubric-free control, which does improve the prompts, is reported in Section 5.3.2. One reading is that a 1.7B model refining its own prompt behaves like a noisy search on a 25-question gate; we did not test this against a random-acceptance baseline. Three limits apply. First, V2F is a hybrid treatment: it changes the gate's baseline and re-attaches the format and the example in code, and its rubric-free counterpart under the same protection (V3F) is reported in Section 5.3.2. Second, V2F was run with the 1.7B refiner on 44 cells and one seed (strong refiners are in Section 5.6.3), and its decision rule was scaled post hoc to that number of cells. Third, the positive control shows that the gate accepts a hand-built candidate, not that a refiner can produce one. Finally, ProTeGi returned the seed instruction plus the format line in 15 of its 15 cells (Appendix G), which is why its exemplar effect is the largest; for ProTeGi the experiments test the refinement of a seed prompt, not of an optimized one.")

H3("5.3.2. A Rubric-Free Control Under the Same Protection (V3F)")
_P1, _P2, _PR = PP["v3f_1.7b"], PP["v2f_1.7b"], PP["paired_1.7b_all"]
_by = {}
for _k, _c in _P1["cells"].items():
    _o = _k.split("__")[1]
    _by.setdefault(_o, []).append(_c["lenient"]["diff"] if _c["lenient"] else 0.0)
_vol = lambda o: sum(_by[o]) / len(_by[o])
P("V3F isolates the rubric. It has the same start prompt (P0 plus the bare-tag example), the same gate, the same "
  "re-attachment of the format line and the example, the same sampling and seven rounds as V2F, but the refiner receives "
  "the rubric-free rewrite request of V3 and no evaluation step, and the rounds are independent. It was run "
  f"with the 1.7B refiner on the same cells as V2F ({_P1['runs']} cells, one MPIR seed). V3F never ended in a fallback or an "
  f"echo, against {_P2['fallback'] + _P2['echo']} of {_P2['rounds']} rounds for V2F; {_P1['validated']} of {_P1['rounds']} rounds "
  f"were validated ({_P2['validated']} of {_P2['rounds']} for V2F) and {_P1['val_better']} of them beat the baseline on "
  f"validation ({_P2['val_better']} for V2F). Candidates were accepted in {_P1['accepted_runs']} of {_P1['runs']} runs "
  f"(V2F: {_P2['accepted_runs']} of {_P2['runs']}). Against its own baseline the final prompts win in {_P1['strict_wins']} "
  f"cells under strict scoring (mean {pp(_P1['strict_mean'])} points) and in {_P1['lenient_wins']} under lenient scoring "
  f"(mean {pp(_P1['lenient_mean'])} points), with {_P1['lenient_losses']} significant lenient losses; V2F's lenient mean is "
  f"{pp(_P2['lenient_mean'])} points. The lenient gain is spread over the optimizers (mean {pp(_vol('ape'))} points for APE, "
  f"{pp(_vol('protegi'))} for ProTeGi and {pp(_vol('promptwizard'))} for PromptWizard).")
P(f"Over the {_PR['n']} cells that both variants completed, the paired difference between the rubric-guided and the "
  f"rubric-free variant (V2F minus V3F) is {pp(_PR['strict']['mean'])} points under strict scoring (interval over cells "
  f"[{pp(_PR['strict']['lo'])}, {pp(_PR['strict']['hi'])}]) and {pp(_PR['lenient']['mean'])} points under lenient scoring "
  f"([{pp(_PR['lenient']['lo'])}, {pp(_PR['lenient']['hi'])}]); under lenient scoring V3F is ahead in "
  f"{_PR['lenient']['v3f_better']} cells, V2F in {_PR['lenient']['v2f_better']} and {_PR['lenient']['tied']} are tied. "
  "Scaled to the number of cells run, decision rule D1 gives 'mixed' for V3F (more than 2 and fewer than a third of the cells "
  "win, and the lenient mean is above +2.0 points). So with the 1.7B refiner and the answer format protected, a "
  "rubric-free rewrite gives a modest gain and the rubric-guided variant gives none, which isolates the "
  "rubric: for this refiner it hurts and does not merely fail to help. The likely mechanism is the length of the rubric "
  "meta-prompt, which the 1.7B model answers with critiques instead of rewrites (the fallbacks and echoes above); we did "
  "not test it, for example with a shortened rubric. The caveats are those of every probe here: one MPIR seed, a "
  "25-question gate, no correction for multiple comparisons, and a gain measured against a baseline that already "
  "contains the bare-tag example.")

# ===========================================================================
H2("5.4. Is the Gate the Bottleneck? (RQ2, D3)")
O1, O2, O3 = OR["v1|1.7b"], OR["v2|1.7b"], OR["v3|1.7b"]
OQ = OR["v2|qwen235"]
P("To separate the candidates from the gate, every validated candidate of MPIR seed 1 was also scored on the "
  f"full test split. Ungated, the mean test-accuracy change of a 1.7B-refiner candidate relative to its P0 is "
  f"{pp(O1['mean_candidate_delta'])} points for the independent variant ({O1['n_candidates']} candidates) and "
  f"{pp(O2['mean_candidate_delta'])} points for the chained variant ({O2['n_candidates']}), and no cell has a "
  f"candidate that beats P0 by 3 points or more ({O1['cells_best_ge_3pts']} of 9 and {O2['cells_best_ge_3pts']} of 8 "
  "cells with a validated candidate; for the 1.7B rewrite V3 one cell has such a candidate); these candidates mostly lack the answer-format line (Section 5.2), so the figure largely "
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
      "versus P0, in points, over all cells with at least one validated candidate (a cell in which the rule accepts "
      "nothing counts as zero; for the larger gate, measured on the held-out half of the test split). Better/worse is the number of cells with a positive or negative delta.",
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
  "detectable effect of Section 5.7 and is within serving noise) and "
  f"{len(l3)} cell{'' if len(l3) == 1 else 's'} in which the gate accepted a "
  f"significantly harmful rewrite ({listing(l3)}), for a mean strict difference of {pp(CAP['v3|1.7b']['mean_diff'])} "
  f"points; under lenient extraction it has {CAPL['v3|1.7b']['wins']} wins and {CAPL['v3|1.7b']['losses']} losses "
  f"(mean {pp(CAPL['v3|1.7b']['mean_diff'])} points, against {pp(CAPL['v1|1.7b']['mean_diff'])} for V1 and "
  f"{pp(CAPL['v2|1.7b']['mean_diff'])} for V2). The gate therefore does not reliably prevent harm, since 25 "
  "questions are noisy. The rubric-free rewrite has more strict wins, partly because of the format asymmetry, and is "
  "no better under lenient scoring; we can neither credit the rubric nor say that unstructured rewriting is worse.")

# ===========================================================================
H2("5.6. A More Capable Refiner (RQ3, D2)")
rows = []
for key, label in [("v1|1.7b", "Qwen3-1.7B (self), V1†"), ("v2|1.7b", "Qwen3-1.7B (self), V2†"),
                   ("v3|1.7b", "Qwen3-1.7B (self), V3†"), ("v1p|1.7b", "Qwen3-1.7B (self), V1P†"),
                   ("v2p|1.7b", "Qwen3-1.7B (self), V2P†"), ("v2|qwen235", "Qwen3-235B-A22B, V2"),
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
      "difference under lenient extraction. Rows marked † pool up to three MPIR seeds, the others use MPIR seed 1 only; read the † rows as pooled and not as seed-1 results.",
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
P('Strict accuracy is, however, a poor judge of these gains. Lenient extraction was applied to every run and both MPIR seeds are reported; nothing was selected, and the seed-2 runs have only their final prompt scored on test. Under lenient extraction (Table 9), in the first MPIR seed Qwen3-235B wins in 1 cell and loses significantly in 3, with a mean difference of -3.2 points; in the second seed the lenient mean is +0.9 points with 1 win and 0 significant losses, so the other strict gains vanish there instead of reversing. The large strict gains in the ProTeGi cells come from base prompts that reason about as well as before but wrap whole sentences in the answer tags. The accepted candidate (Figure B.3) adds an explicit instruction to wrap the final answer on its own line and a worked output example, which repairs the tag, while lenient accuracy does not improve and in the first seed falls, for example from 58.0% to 49.0% on hyperbaton with ProTeGi (we did not test why). A gate that uses strict extraction accepts such candidates, which is a concrete way in which answer extraction and the gate interact.')
P("The single gain that holds under both metrics is hyperbaton with PromptWizard, whose base prompt already had the tag (strict = lenient = 59.0%) and whose accepted candidates, from two different refiners, add a role, a step-back and numbered reasoning steps and reach 89% and 86%. It is one cell out of nine, on a near-chance base prompt (59% against 50% chance) whose instruction is garbled, with no multiplicity correction and no replication on another task; a rubric-based rewrite can plausibly claim it, and no more. The two strong refiners also differ from each other, one belonging to the target's family and one run with minimal reasoning, so capability, family and reasoning budget are confounded and the experiment is an upper-bound probe, not a controlled capability scale.")

H3("5.6.1. Follow-Up Runs (Exploratory)")
V3Q = CAP["v3|qwen235"]
wv3q, lv3q = wins_and_losses(V3Q)
P("Because the strong-refiner signal appeared after the analysis fixed in advance, two follow-up runs were added "
  "and are exploratory (Appendix F). First, the rubric-free gated rewrite (V3) with the same Qwen3-235B refiner "
  f"gives a mean strict difference of {pp(V3Q['mean_diff'])} points with {V3Q['wins']} winning cells of 9 "
  f"({listing(wv3q)}), against {pp(CAP['v2|qwen235']['mean_diff'])} points and {CAP['v2|qwen235']['wins']} wins "
  "for the rubric-guided variant, and a lenient mean of "
  f"{pp(CAPL['v3|qwen235']['mean_diff'])} points. A refiner that does not use the rubric therefore reaches "
  "gains of the same kind, so the rubric is not shown to be what produces them (Table R.2 gives the paired comparison). Second, a second MPIR seed of the "
  f"rubric-guided variant gives {pp(FS2['mean_diff'])} points and {FS2['wins']} winning cells, so the strict "
  f"pattern is stable across seeds; under lenient extraction its mean is {pp(FS2L['mean_diff'])} points with "
  f"{FS2L['wins']} win (hyperbaton with PromptWizard) and {FS2L['losses']} significant losses.")

RV = OFF["rubric_vs_free"]["qwen235"]
rows = []
for key, e in RV["cells"].items():
    a, b, c = e["v3_minus_p0"], e["v2m1_minus_p0"], e["v2m1_minus_v3"]
    rows.append([cell_label(key), pct(e["p0_test"]), pp(a["diff"]), pp(b["diff"]),
                 f"{pp(c['diff'])} [{pp(c['lo'])}, {pp(c['hi'])}]"])
TABLE("Table R.2. Paired comparison of the rubric-guided (V2) and rubric-free (V3) variants with the same "
      "Qwen3-235B refiner on the nine Tier-1 seed-42 cells, MPIR seed 1 for both. Differences are in test-accuracy "
      "points of the final prompt, paired over the 200 test examples (96 for penguins_in_a_table); the last column "
      "is V2 minus V3 with a 95% paired-bootstrap interval. One run per condition and cell; the minimum "
      f"detectable effect is {pct(NOISE['mde_noise_only'])} points (Section 5.7).",
      header=["Cell", "P0 strict %", "V3 - P0", "V2 - P0", "V2 - V3 [95% CI]"],
      rows=rows, full=True, colw=[0.30, 0.12, 0.14, 0.14, 0.30])
P(f"Table R.2 gives the cell-level paired comparison. Averaged over the {RV['n_cells']} cells, "
  f"V2 exceeds V3 by {pp(RV['mean_v2_minus_v3'])} points (bootstrap over cells [{pp(RV['lo'])}, {pp(RV['hi'])}]); "
  f"V2 is ahead in {RV['cells_v2_better']} cells, V3 in {RV['cells_v3_better']} and {RV['cells_tied']} are tied, and "
  f"the paired interval excludes zero in {RV['cells_ci_excludes_zero']} cells. With the second V2 seed the mean "
  f"difference is {pp(RV['mean_v2seed2_minus_v3'])} points, so the sign is stable but the size is not. The data "
  "therefore cannot separate 'the rubric adds nothing' from 'the rubric adds a modest benefit' with this refiner: "
  "the interval for the mean difference includes zero and benefits of several points, each condition has one or two "
  "runs per cell, and the refiner's capability, family and reasoning budget are confounded with the choice of "
  "model. We therefore do not claim equivalence: the rubric-free rewrite reaches gains of the same kind, and the "
  "experiment is exploratory. The one V3 win "
  "with the 1.7B refiner (Section 5.5), ruin_names with ProTeGi at +2.8 points, is below the minimum detectable "
  "effect of 5.0 points and within serving noise; we label it as such there.")

H3("5.6.2. An Intermediate Open-Weight Refiner (Exploratory)")
P("The protocol planned local 4B and 8B refiners for D2, but they were not run in the main study. We therefore served Qwen3-8B-AWQ locally next to the 1.7B target and ran the rubric-guided chained variant (V2) and the rubric-free gated rewrite (V3) on the nine Tier-1 seed-42 cells, MPIR seed 1, with oracle scoring (Appendix F, amendment 10; exploratory). With V2, 63% of 63 rounds reached validation (37% were fallbacks or echoes), 1 validated candidate beat the baseline, and 1 of 9 runs changed the prompt: strict mean +1.8 points with 1 winning cell, lenient mean +0.2 points with 0 wins and 0 significant losses. With V3, 1 of 9 runs changed the prompt: strict mean +1.9 points with 1 win, lenient mean +1.9 points with 1 win and 0 losses. The paired V2 minus V3 difference over the nine cells is -0.1 points (interval over cells [-5.7, +5.3]; V2 ahead in 1 cell, V3 in 1, 7 tied). D2 (a mean at least 2.0 points above the 1.7B refiner's and at least 3 winning cells) is not met under strict scoring (+1.8 points, 1 winning cell). Because the 8B refiner changed the prompt in only one run per variant, it adds little to the 235B result, which it neither confirms nor contradicts: gains occur mostly in cells whose base prompt fails on the answer format or scores near chance, a rubric-free rewrite obtains gains of the same kind in some of them, and the paired rubric effect is not distinguishable from zero in nine cells with one run each.")

H3("5.6.3. Strong Refiners Under the Protected Treatment (Exploratory)")
_t1 = {k for k in PP["v2f_qwen235"]["cells"]}


def _tier_mean(variant_key, metric="lenient"):
    cs = [c for k, c in PP[variant_key]["cells"].items() if k in _t1 and c[metric]]
    return sum(c[metric]["diff"] for c in cs) / len(cs) if cs else 0.0


_rows = []
for _key, _label in [("v2f_1.7b", "Qwen3-1.7B, V2F"), ("v3f_1.7b", "Qwen3-1.7B, V3F"), ("v2f_qwen235", "Qwen3-235B-A22B, V2F"),
                     ("v3f_qwen235", "Qwen3-235B-A22B, V3F"), ("v2f_gpt6luna", "GPT-6 Luna, V2F"), ("v3f_gpt6luna", "GPT-6 Luna, V3F")]:
    _d = PP[_key]
    _rows.append([_label, str(_d["runs"]), f"{share(_d['validated'], _d['rounds'])}%", f"{_d['accepted_runs']}/{_d['runs']}",
                  f"{_d['strict_wins']}", pp(_d["strict_mean"]), f"{_d['lenient_wins']}/{_d['lenient_losses']}", pp(_d["lenient_mean"])])
TABLE("Table R.6. The protected treatment (start prompt P0 plus a bare-tag example, format line and example re-attached) with "
      "three refiners, rubric-guided (V2F) and rubric-free (V3F). Differences are final prompt minus the start prompt, in test-accuracy "
      "points; the 1.7B rows cover 44 or 45 cells, the strong-refiner rows the nine Tier-1 seed-42 cells, MPIR seed 1. Wins and "
      "losses are cells whose paired interval excludes zero.",
      header=["Refiner, variant", "Runs", "Validated", "Accepted", "Strict wins", "Strict mean", "Lenient wins/losses", "Lenient mean"],
      rows=_rows, full=True)
_Q, _L = PP["paired_qwen235"], PP["paired_gpt6luna"]
P("The second probe repeats the protected comparison with the strong refiners, Qwen3-235B-A22B and GPT-6 Luna, through the "
  "router on 2026-11-02 (same model slugs as in Section 4.3; the provider and fingerprint were again not logged), on the "
  "nine Tier-1 seed-42 cells with oracle scoring. With the format and example protected, these refiners produce gains that "
  f"hold under lenient scoring: Qwen3-235B {pp(PP['v2f_qwen235']['lenient_mean'])} points with {PP['v2f_qwen235']['lenient_wins']} "
  f"winning cells and {PP['v2f_qwen235']['lenient_losses']} losses, GPT-6 Luna {pp(PP['v2f_gpt6luna']['lenient_mean'])} points with "
  f"{PP['v2f_gpt6luna']['lenient_wins']} wins and {PP['v2f_gpt6luna']['lenient_losses']} loss, against {pp(_tier_mean('v2f_1.7b'))} points "
  "for the 1.7B refiner on the same cells. This is the capability dependence that the unprotected probes could not show "
  "(rule D2 applied to lenient scoring and post hoc: for Qwen3-235B the mean exceeds that of the 1.7B refiner by more than 2.0 "
  "points and at least 3 cells win). The rubric, however, is not what produces it: the rubric-free V3F gives "
  f"{pp(PP['v3f_qwen235']['lenient_mean'])} points with Qwen3-235B and {pp(PP['v3f_gpt6luna']['lenient_mean'])} with GPT-6 Luna, and "
  f"the paired V2F minus V3F difference is {pp(_Q['lenient']['mean'])} points [{pp(_Q['lenient']['lo'])}, {pp(_Q['lenient']['hi'])}] "
  f"for Qwen3-235B and {pp(_L['lenient']['mean'])} [{pp(_L['lenient']['lo'])}, {pp(_L['lenient']['hi'])}] for GPT-6 Luna, each over nine "
  "cells with one run per condition. The data therefore support three statements and no more: a capable refiner that keeps the answer "
  "format can improve these prompts under a validated lenient metric; a rubric-free rewrite does about as well as the rubric-guided "
  "variant, so the rubric is not shown to help (and it hurts the 1.7B refiner, Section 5.3.2); and nine cells with single runs "
  "cannot detect a modest rubric benefit with a strong refiner.")

# ===========================================================================
H2("5.7. Serving Noise and Minimum Detectable Effect")
P("Each of the nine pilot base prompts was scored five times on the test and validation splits (identical "
  "prompt, greedy decoding). Test accuracy varies with a standard deviation of "
  f"{pct(NOISE['mean_sd'])} points and a range of {pct(NOISE['mean_range'])} points per prompt on average; between two "
  f"scorings of the same prompt {pct(NOISE['mean_discordance'])}% of examples change outcome on average "
  f"(up to {pct(NOISE['max_discordance'])}%), and the pilot and present scorings of the same prompt differ by "
  f"up to {pct(PVR['max_abs_diff'])} points. This is larger than the 2% assumed earlier in the project and "
  "is the reason a 25-question validation split is noisy. For 200 test examples the minimum detectable "
  f"paired effect at 80% power and α = 0.05 is {pct(NOISE['mde_noise_only'])} points if only serving noise "
  f"separates two prompts (paired discordance {pct(NOISE['mean_discordance'])}%), and "
  f"{pct(NOISE['mde_for_discordance']['0.1'])} to {pct(NOISE['mde_for_discordance']['0.2'])} points if two different "
  "prompts disagree on 10-20% of examples, as candidates here do. The strict gains of the strong refiner "
  f"({', '.join(pp(d, 0) for _, d in wq)} points) are above this floor; effects of a few points are not detectable "
  "at these test sizes.")

RS = NEW["rescore"]
_g = RS["groups"]
P("Mixed serving settings. The inference server was restarted several times during the study, with changed memory "
  "and prefix-caching settings, so the original scorings of some runs combine configurations, and the per-run "
  "configuration was not logged. An exclusion analysis is therefore not possible. We instead rescored every prompt "
  "behind an exploratory claim, the base prompts and the final prompts of the Qwen3-235B (two MPIR seeds, V2; one, V3) "
  f"and GPT-6 Luna runs ({RS['within_session_flip_rate']['n_prompts']} prompts, nine cells each), twice each in a single "
  "serving session on 2026-11-02, and compared the cell differences with the original ones. The mean strict gain of "
  f"Qwen3-235B V2 moves from {pp(_g['v2|qwen235|m1']['orig_mean'])} to {pp(_g['v2|qwen235|m1']['rescored_mean'])} points "
  f"(seed 1) and from {pp(_g['v2|qwen235|m2']['orig_mean'])} to {pp(_g['v2|qwen235|m2']['rescored_mean'])} (seed 2), "
  f"V3 from {pp(_g['v3|qwen235|m1']['orig_mean'])} to {pp(_g['v3|qwen235|m1']['rescored_mean'])} and GPT-6 Luna V2 from "
  f"{pp(_g['v2|gpt6luna|m1']['orig_mean'])} to {pp(_g['v2|gpt6luna|m1']['rescored_mean'])}. The number of cells "
  f"with a gain of at least the 5-point minimum detectable effect is unchanged "
  f"({_g['v2|qwen235|m1']['orig_cells_gt_mde']} and {_g['v2|qwen235|m1']['rescored_cells_gt_mde']} for V2 seed 1; "
  f"{_g['v3|qwen235|m1']['orig_cells_gt_mde']} and {_g['v3|qwen235|m1']['rescored_cells_gt_mde']} for V3), and no cell "
  f"difference moves by more than {pct(max(v['max_abs_change'] for v in _g.values()))} points. The headline "
  "conclusions of the exploratory probes therefore do not depend on which server configuration scored them. Between "
  f"two scorings in the same session, {pct(RS['within_session_flip_rate']['mean'])}% of examples flip on average "
  f"(largest {pct(RS['within_session_flip_rate']['max'])}%), the same order as the 6.5% measured earlier, so the "
  "gate's noise is a property of the stack and not of the restarts.")

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
for judge, name in (("1.7b", "Qwen3-1.7B"), ("gpt6luna", "GPT-6 Luna"), ("8b", "Qwen3-8B-AWQ (a refiner only in Section 5.6.2)")):
    a_, b_ = _jz(f"{judge}|all"), _jz(f"{judge}|test_above_zero")
    if a_ and b_:
        _rows.append([name, str(a_["candidates"]), pct(a_["zero_share"], 0) + "%", f"{a_['median_spearman_test']:.2f}",
                      str(b_["candidates"]), f"{b_['median_spearman_test']:.2f}"])
if _rows:
    TABLE("Table R.3. Within-cell median Spearman correlation of the rubric total with test accuracy over all "
          "oracle-scored candidates and after excluding candidates whose strict test accuracy is exactly zero "
          "(format loss). The 8B judge is the local Qwen3-8B-AWQ model, which is also the intermediate refiner of Section 5.6.2; it is not the refiner of the 1.7B, Qwen3-235B or GPT-6 Luna candidates, but its scores of its own candidates are not independent.",
          header=["Judge", "Candidates", "Zero accuracy", "Spearman (all)", "Candidates > 0", "Spearman (> 0)"],
          rows=_rows, full=True, colw=[0.34, 0.12, 0.12, 0.14, 0.14, 0.14])
    P("A large share of the oracle-scored candidates score exactly zero because they "
      f"lost the answer format ({pct(JZ['1.7b|all']['zero_share'], 0)}%), which can inflate or deflate a "
      "correlation with the rubric total, and GPT-6 Luna is both the reference judge and one of the refiners. "
      "Table R.3 removes the zero-accuracy candidates: the median correlation falls to "
      f"{JZ['1.7b|test_above_zero']['median_spearman_test']:.2f} (1.7B judge) and "
      f"{JZ['gpt6luna|test_above_zero']['median_spearman_test']:.2f} (GPT-6 Luna) over "
      f"{JZ['1.7b|test_above_zero']['candidates']} and {JZ['gpt6luna|test_above_zero']['candidates']} candidates, so the "
      "weak association of Table 10 was partly the rubric rewarding candidates that had kept the format, and among "
      "usable candidates the correlation is small and not stable across judges. "
      + ("A third judge, the local 8B model (scored on a random subset of 150 prompts, "
         f"{JZ['8b|all']['candidates']} parsed), gives "
         f"{JZ['8b|all']['median_spearman_test']:.2f} over all candidates and "
         f"{JZ['8b|test_above_zero']['median_spearman_test']:.2f} without the zero-accuracy ones, but that value rests on "
         f"{JZ['8b|test_above_zero']['candidates']} candidates in {JZ['8b|test_above_zero']['cells_used']} cells and is not "
         "reliable. "
         if "8b|all" in JZ else "")
      + "A criterion-level ablation, which would show which of the seven criteria carry any signal, was not run; it is "
      "listed among the limitations.")

# ===========================================================================
H2("5.9. Effect of the Original Defect")
LG6 = LEG["literal_g6"]
P("The original loop accepted the first round's outcome whatever it scored. With the 1.7B refiner and nine "
  "cells (MPIR seed 1, independent variant, oracle-scored candidates), applying that rule gives a mean "
  f"test-accuracy change of {pp(LEG['mean_legacy_delta'])} points relative to P0, where a fallback round "
  f"returns P0 (as the pilot's code did) and an echo round is accepted as a candidate: test accuracy differed from P0's in {LEG['legacy_changed_prompt']} of {LEG['n']} cells "
  f"(worse in {LEG['legacy_worse']}, better in {LEG['legacy_better']}). The candidates are those of the independent variant V1, MPIR seed 1, on the nine Tier-1 seed-42 cells, scored on the "
  "full test split; the first-round candidate was validated in "
  f"{sum(1 for c in LEG['cells'].values() if c['first_round_outcome'] == 'validated')} cells, was a fallback in "
  f"{sum(1 for c in LEG['cells'].values() if c['first_round_outcome'] == 'fallback')} (where the simulated legacy rule returns P0) and an echo in "
  f"{sum(1 for c in LEG['cells'].values() if c['first_round_outcome'] == 'echo')} (where it accepts the echoed prompt). "
  "This is an offline simulation: the actual legacy loop produced identical prompts in all nine pilot cells, so the "
  "figures describe what the accept-first rule would have done to these candidates, not what the pilot did. Under the literal "
  "reading of the revision protocol (group G6: the first validated candidate is accepted unconditionally) the mean change is "
  f"{pp(LG6['mean_delta'])} points. The repaired loop changes accuracy by {pp(LEG['mean_gated_delta'])} points on the "
  "same candidates. These figures apply to this refiner only; their counterpart for the original GPT-4o refiner is "
  "unknown, because that configuration was not rerun.")

H2("5.10. Decision Rules")
_G2 = NEW["grid_v2f"]
P("Table 11 gives the outcome of each decision rule. D1 (RQ1) is null for the unprotected chained variant with the 1.7B "
  "refiner (0 wins in 45 cells, one accepting run of 80). Restated for the format- and example-preserving variant V2F against "
  "its own baseline it gives 4 strict wins (mixed) and 1 lenient win with 3 significant lenient losses in 44 cells (lenient: "
  f"null), while its rubric-free control V3F gives {PP['v3f_1.7b']['lenient_wins']} lenient wins and a lenient mean of "
  f"{pp(PP['v3f_1.7b']['lenient_mean'])} points (mixed). D2 (RQ3) is not met with GPT-6 Luna or the 8B refiner under the "
  "unprotected treatment and is met for Qwen3-235B under strict scoring only and post hoc; under the protected treatment and "
  "lenient scoring it is met for Qwen3-235B (post hoc), but the rubric-free control does as well. D3 (RQ2) is not met for the "
  "1.7B refiner: the candidates, not the gate, limit the result. The rubric score is weakly related to accuracy (RQ4).")

_VR = []
def _vrow(label, refiner, cells, tests, outcome):
    _VR.append([label, refiner, cells, tests, outcome])
_vrow("V0 legacy", "Qwen3-1.7B", f"9 ({V0['runs']} runs)", "the pilot's loop re-run",
      f"{V0['runs_with_acceptance']} of {V0['runs']} runs changed the prompt")
_vrow("V1 independent", "Qwen3-1.7B", "9", "rubric, independent rounds, unprotected",
      f"{V1['runs_with_acceptance']} of {V1['runs']} runs changed; mostly format loss")
_vrow("V2 chained (primary)", "Qwen3-1.7B", f"{MG['n_cells']} ({MG['runs']} runs)", "D1, unprotected",
      f"{MG['runs_accepting']} of {MG['runs']} runs changed; {MG['wins']} wins; null")
_vrow("V2L original rubric text", "Qwen3-1.7B", "9", "rubric text inconsistency", f"{V2L['accepted']} accepted; no better")
_vrow("V3 rubric-free gated", "Qwen3-1.7B", "9", "rubric versus free rewrite",
      f"{V3['runs_with_acceptance']} of {V3['runs']} runs changed; a few within noise")
_vrow("V1P, V2P protected", "Qwen3-1.7B", "5 and 8 of 9", "format line appended",
      "still far below P0: no bare-tag example kept")
_vrow("V2F preserving", "Qwen3-1.7B", f"{_G2['runs']} of 45", "D1 with a treatment that can succeed",
      f"{_G2['accepted_runs']} of {_G2['runs']} changed; strict wins {_G2['wins_vs_start']}, lenient wins "
      f"{_G2['lenient_wins_vs_start']}, lenient losses {_G2['lenient_losses_vs_start']}")
_vrow("V3F rubric-free, preserving", "Qwen3-1.7B", f"{PP['v3f_1.7b']['runs']}", "rubric isolated under the same protection",
      f"{PP['v3f_1.7b']['accepted_runs']} of {PP['v3f_1.7b']['runs']} changed; lenient wins {PP['v3f_1.7b']['lenient_wins']}, "
      f"losses {PP['v3f_1.7b']['lenient_losses']}, mean {pp(PP['v3f_1.7b']['lenient_mean'])}")
_vrow("V2F, V3F preserving", "Qwen3-235B-A22B, GPT-6 Luna", "9 each", "strong refiners under the protected treatment (exploratory)",
      f"lenient mean V2F {pp(PP['v2f_qwen235']['lenient_mean'])} / {pp(PP['v2f_gpt6luna']['lenient_mean'])}, V3F "
      f"{pp(PP['v3f_qwen235']['lenient_mean'])} / {pp(PP['v3f_gpt6luna']['lenient_mean'])}")
_vrow("V2 chained, V3 free", "Qwen3-8B-AWQ", "9", "intermediate refiner (exploratory)",
      (f"V2 strict {pp(MID['v2']['strict_mean'])}, V3 {pp(MID['v3']['strict_mean'])}; see Section 5.6.2"
       if MID.get("v2") else "not available"))
_vrow("V2, V3 chained/free", "Qwen3-235B-A22B", "9", "strong refiner (exploratory)",
      f"V2 {CAP['v2|qwen235']['wins']} strict wins, V3 {CAP['v3|qwen235']['wins']}; mostly format repairs")
_vrow("V2 chained", "GPT-6 Luna", "9", "strong closed refiner (exploratory)",
      f"{CAP['v2|gpt6luna']['wins']} strict win; D2 not met")
TABLE("Table R.4. Summary of the variants: refiner, cells, what each tests and its outcome (strict scoring unless "
      "stated; cells are Tier-1 seed-42 cells unless a count is given).",
      header=["Variant", "Refiner", "Cells", "What it tests", "Outcome"], rows=_VR, full=True,
      colw=[0.20, 0.15, 0.12, 0.24, 0.29])
TABLE("Table 11. Outcome of the decision rules (Section 4.1).",
      header=["Rule", "Criterion", "Observed", "Outcome"],
      rows=[
          ["D1", "At least 15 of 45 cells win and mean at least +2.0 pts; null if at most 2 wins and |mean| at most 1.0",
           f"{MG['wins']} wins, mean {pp(MG['mean_diff'], 2)}", MG["verdict"].capitalize()],
          ["D1, V2F (own baseline)", "Same rule, scaled to the cells run",
           f"strict wins {_G2['wins_vs_start']}, mean {pp(_G2['mean_diff_vs_start'])}; lenient wins {_G2['lenient_wins_vs_start']}, "
           f"losses {_G2['lenient_losses_vs_start']}, mean {pp(_G2['lenient_mean_diff_vs_start'])}", "Strict mixed; lenient null"],
          ["D1, V3F (own baseline)", "Same rule, scaled to the cells run",
           f"strict wins {PP['v3f_1.7b']['strict_wins']}, mean {pp(PP['v3f_1.7b']['strict_mean'])}; lenient wins "
           f"{PP['v3f_1.7b']['lenient_wins']}, losses {PP['v3f_1.7b']['lenient_losses']}, mean {pp(PP['v3f_1.7b']['lenient_mean'])}",
           "Mixed (more than 2 wins, fewer than a third)"],
          ["D2, protected V2F, Qwen3-235B (post hoc, lenient)", "Mean +2.0 over the 1.7B refiner and at least 3 of 9 wins",
           f"{pp(PP['v2f_qwen235']['lenient_mean'])}; {PP['v2f_qwen235']['lenient_wins']} wins", "Met, but not specific to the rubric (V3F similar)"],
          ["D2, GPT-6 Luna", "Mean +2.0 over 1.7B and at least 3 of 9 wins",
           f"{pp(CAP['v2|gpt6luna']['mean_diff'])}; {CAP['v2|gpt6luna']['wins']} win", "Not met"],
          ["D2, Qwen3-8B-AWQ (exploratory)", "Same",
           (f"{pp(MID['v2']['strict_mean'] - CAP['v2|1.7b']['mean_diff'])}; {MID['v2']['strict_wins']} win"
            if MID.get("v2") else "n/a"), "Not met"],
          ["D2, Qwen3-235B (post hoc)", "Same",
           f"{pp(CAP['v2|qwen235']['mean_diff'])}; {CAP['v2|qwen235']['wins']} wins (lenient: "
           f"{pp(CAPL['v2|qwen235']['mean_diff'])}; {CAPL['v2|qwen235']['wins']} win)", "Met under strict only"],
          ["D3", "Best oracle candidate at least +3 pts in at least 3 of 9 cells although the gated final is not",
           f"{O1['cells_best_ge_3pts']}, {O2['cells_best_ge_3pts']}, {O3['cells_best_ge_3pts']} cells (V1, V2, V3)",
           "Not met"],
      ], full=True, colw=[0.15, 0.37, 0.30, 0.18])

# ===========================================================================
H2("5.11. Cost")
CO = OFF["cost"]
_rows = []
for key, label in [("v2|1.7b", "V2, 1.7B refiner"), ("v1|1.7b", "V1, 1.7B refiner"), ("v3|1.7b", "V3, 1.7B refiner"),
                   ("v2f|1.7b", "V2F, 1.7B refiner"), ("v2|qwen235", "V2, Qwen3-235B"), ("v3|qwen235", "V3, Qwen3-235B"),
                   ("v2|gpt6luna", "V2, GPT-6 Luna")]:
    if key in CO:
        c = CO[key]
        _rows.append([label, str(c["runs"]), f"{c['refiner_calls_per_run']:.0f}",
                      f"{c['refiner_generated_tokens_per_run'] / 1000:.1f}k", f"{c['target_validation_calls_per_run']:.0f}",
                      f"{c['target_test_calls_per_run']:.0f}"])
TABLE("Table R.5. Cost of one run (seven rounds), averaged over the runs of each variant. Refiner calls: evaluation plus "
      "refinement (one call per round for V3). Refiner tokens are generated tokens, estimated from the logged texts at "
      "4.36 characters per token (measured with the Qwen3 tokenizer on 1,500 generations). Target calls are the "
      "validation questions answered for the baseline and every validated candidate (25 per scoring) and the two test "
      "scorings of P0 and the final prompt; the oracle scoring of every candidate used for analysis is not included.",
      header=["Variant", "Runs", "Refiner calls", "Refiner tokens", "Target val. calls", "Target test calls"],
      rows=_rows, full=True, colw=[0.28, 0.08, 0.14, 0.16, 0.18, 0.16])
P("A run issues 14 refiner calls (7 for V3) and generates about 12 thousand refiner tokens, plus about 134 target calls for validation and 356 for the two test scorings that any evaluation needs anyway; in absolute terms this is small next to the base optimizers, which score many candidates on minibatches. Wall-clock time was not logged for the earlier runs; a V2F run took a median of 9 minutes on the shared local GPU while other jobs were running, which is not a stand-alone figure. Because 66% of the V2 rounds never reached validation, the effective cost per validated candidate is about three times the nominal cost of a round. The 'cheap layer' argument of the Introduction therefore holds only if the layer works: in our data none of the 80 V2 runs improved accuracy, so the cost bought nothing, and it argues for caution rather than for use.")
