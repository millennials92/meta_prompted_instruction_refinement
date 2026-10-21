# -*- coding: utf-8 -*-
"""Builds the IEEE Access response-to-reviewers document from the official template.

Every number comes from results/revision/summary_numbers.json via study_numbers.N, so the response and the
manuscript cannot drift apart. Run after mpir_study/analyze.py.
"""
import os
import sys

import docx
from docx.shared import Pt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from study_numbers import N  # noqa: E402

TEMPLATE = os.path.join(HERE, "templates", "IEEE-Access-Response-to-Reviewers-template.docx")
OUT = os.path.join(HERE, "response_to_reviewers.docx")

MG, CAP, NOISE, LEG, EX = N["main_grid"], N["capability"], N["noise"], N["legacy"], N["exemplar"]
CAPL = N["capability_lenient"]
C, OR, PVR = N["composition"], N["oracle"], N["pilot_vs_rescore"]
JV = N["judge_validity"]["judges"]


def pp(x, d=1):
    return f"{100 * x:+.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


V0, V2, V3 = C["v0|1.7b"], C["v2|1.7b"], C["v3|1.7b"]
QW, LU = CAP["v2|qwen235"], CAP["v2|gpt6luna"]
FOLLOW = []
if CAP.get("v3|qwen235"):
    FOLLOW.append(f"the rubric-free gated rewrite with the same refiner gives {pp(CAP['v3|qwen235']['mean_diff'])} points "
                  f"({CAP['v3|qwen235']['wins']} winning cells of 9)")
if N.get("follow_seed2"):
    FOLLOW.append(f"a second MPIR seed gives {pp(N['follow_seed2']['mean_diff'])} points "
                  f"({N['follow_seed2']['wins']} winning cells)")
FOLLOW_TEXT = (" Exploratory follow-up: " + "; ".join(FOLLOW) + " (Section 5.6.1).") if FOLLOW else ""

# (reviewer label, concern, response, action)
ENTRIES = [
    # ------------------------------------------------------------------ Reviewer 1, major
    ("Reviewer#1, Major comment 1",
     "Refinement rounds in Algorithm 1 do not appear to be independent (Evaluate and Refine always applied to P0, temperature 0). State what varies between rounds; if nothing does, redo with independently sampled candidates and/or conditioning on rejected candidates and report the number of distinct candidates per cell.",
     f"The reviewer is right about the design. In the pilot every round received identical input at temperature 0, so rounds differed only through serving non-determinism. We re-measured this: the legacy loop (variant V0), re-run on the nine pilot cells, produced on average {V0['mean_distinct_candidates']:.1f} distinct candidates per run, but only through serving noise, not by design. We then repaired the loop and report two repaired variants: V1 (independent rounds, refiner sampled at temperature 0.7) and V2 (chained: each round refines the current incumbent and is shown the earlier candidates with their validation scores). The number of distinct candidates per run is reported in Table 6 ({V2['mean_distinct_candidates']:.1f} for V2 with the 1.7B refiner; more for stronger refiners).",
     "Sections 3.2 and 5.2, Algorithm 1, Tables 2 and 6, and Appendix E rewrite and replace the pilot's description. The claim 'seven rounds without acceptance' was removed."),
    ("Reviewer#1, Major comment 2",
     "The composition of the nine 'byte-identical' outcomes is not reported (fallbacks, echo-rejected, validated candidates and their scores relative to the baseline; rubric totals).",
     f"We now log the outcome of every round (fallback, echo, duplicate, validated), the validation score against the baseline, and the judge's rubric total. Across all {V2['rounds']} chained rounds with the 1.7B refiner, {V2['fallback']} were fallbacks, {V2['echo']} echoes, {V2['duplicate']} duplicates and {V2['validated']} reached validation, of which {V2['val_worse']} scored below the baseline, {V2['val_equal']} tied and {V2['val_better']} scored higher. So 'scored at or below' hid mostly losses and mostly guard-intercepted outputs, as the reviewer suspected.",
     "Table 6 (all variants), Tables 15 and 16 in Appendix E (per cell and per round, including rubric totals), and Table 10 (judge totals). The full per-round records, including candidate texts, are in the repository under results/revision."),
    ("Reviewer#1, Major comment 3",
     "The statistical analysis is degenerate; Wilcoxon/sign tests on n = 3; GEE/bootstrap with three clusters; no power analysis; quantify the serving-noise floor and give a minimum detectable effect.",
     f"We removed McNemar, GEE and Wilcoxon from the inference and report per-cell paired differences with bootstrap intervals, win counts and a hierarchical bootstrap descriptive interval. The number of cells is increased from 9 to {MG['n_cells']} (three optimizers, nine tasks, up to three seeds). The serving-noise floor was measured by scoring each pilot prompt five times: standard deviation {pct(NOISE['mean_sd'])} points, range {pct(NOISE['mean_range'])} points, {pct(NOISE['mean_discordance'])}% of examples changing outcome between scorings (up to {pct(NOISE['max_discordance'])}%), which is larger than the 2% assumed. The minimum detectable effect for 200 test examples is {pct(NOISE['mde_noise_only'])} points if only noise separates two prompts and {pct(NOISE['mde_for_discordance']['0.1'])}-{pct(NOISE['mde_for_discordance']['0.2'])} points if prompts disagree on 10-20% of examples.",
     "Sections 4.6, 5.3 and 5.7 and Table 7 replace the old statistics; Section 5.10 (Table 11) gives the decision rules and outcomes."),
    ("Reviewer#1, Major comment 4",
     "Task selection (winner's curse), 'pre-registration' without registry/timestamp/rule text, single seed at temperature 0.",
     "(a) We now state that the pilot tasks were chosen on a biased statistic, rank them in the original (leaked) results (Appendix D), and add six tasks drawn uniformly at random with a stated generator and seed (random.Random(20261019)). (b) 'Pre-registered' is withdrawn. The pilot's go/no-go rule was written in a version-controlled planning document (commit 7612aa0, 2026-09-03) before the pilot ran; several pipeline fixes were made while it ran, and these are listed. For this revision, REVISION_PROTOCOL.md (commit 8fd1397, 2026-10-19) fixed the decision rules before any revised experiment; its dated amendments, including two exploratory additions made after results were seen, are listed. Neither is a registry entry and the paper says so. (c) Tier 1 now uses three seeds (42-44), and the main grid uses up to three MPIR seeds. We did not run all BBH tasks: nine of 27 task files were used.",
     "Sections 4.1 and 4.2, Appendix D and Appendix F; REVISION_PROTOCOL.md in the repository."),
    ("Reviewer#1, Major comment 5",
     "The capability-gap interpretation is untested; original results and the Gemini check are not shown; either rerun GPT-4o/GPT-3.5 (and Gemini) with a mid-sized open model added, or remove the claim.",
     f"We could not rerun the GPT-4o/GPT-3.5-turbo configuration (no API access for that setup), so the claim that the layer 'depends on the refiner's capability' is no longer stated as a finding. We instead varied the refiner with the target fixed at Qwen3-1.7B: Qwen3-235B-A22B and GPT-6 Luna (through a router API). With Qwen3-235B the layer produced significant gains in {QW['wins']} of 9 cells (mean {pp(QW['mean_diff'])} points); with GPT-6 Luna in {LU['wins']} cell (mean {pp(LU['mean_diff'])} points). Under lenient answer extraction only one of these gains remains in the first seed (hyperbaton with PromptWizard; mean {pp(CAPL['v2|qwen235']['mean_diff'])} points, {CAPL['v2|qwen235']['losses']} significant losses), and in the second seed the lenient mean is {pp(N['follow_seed2_lenient']['mean_diff'])} points. The strict gains occur where the base prompt fails on answer format, and the accepted candidates add a format instruction and example, so they do not isolate the rubric.{FOLLOW_TEXT} Local 4B/8B refiners were not run because the shared GPU was too slow for the extra servers. The original per-task GPT-3.5 results, with the validation leak flagged, and the Gemini check are in Appendix D.",
     "Sections 5.6 and 6.1 and Table 9; Appendix D (Table 14); the abstract and conclusion state the limited scope."),
    ("Reviewer#1, Major comment 6",
     "The evaluation harness confounds the comparison; 'format compliance' is under-evidenced; the six pipeline bugs are described only in a log; give the serving configuration.",
     f"We report strict and lenient accuracy for every base prompt (Table 4; ProTeGi on hyperbaton rises from {pct(N['p0_landscape']['hyperbaton__protegi__s42']['strict'])}% to {pct(N['p0_landscape']['hyperbaton__protegi__s42']['lenient'])}%), and ran the single-format-exemplar intervention on {EX['n']} APE and ProTeGi cells: mean gain {pp(EX['mean_gain_by_optimizer']['ape'])} points (APE) and {pp(EX['mean_gain_by_optimizer']['protegi'])} points (ProTeGi). All defects, with symptoms and fixes, are in a table, together with the serving configuration (vLLM version, quantization, context, token limits, hardware).",
     "Section 5.1, Tables 4 and 5, Appendix C (Tables 12 and 13)."),
    ("Reviewer#1, Major comment 7",
     "The validation gate is coarse; the free-rewrite control has no gate; add a gated free rewrite, sensitivity to validation size and acceptance rule, and the P0 score distribution across repeated runs.",
     f"We added a gated free rewrite (V3), scored every validated candidate on the test split, and re-applied four acceptance rules offline (strict, non-strict, mean of three validation scorings, and a larger gate formed from the validation split plus half of the test split). The gate is not the bottleneck: ungated candidates are on average {pp(OR['v1|1.7b']['mean_candidate_delta'])} and {pp(OR['v2|1.7b']['mean_candidate_delta'])} points relative to P0 (V1, V2), no cell has a candidate that beats P0 by 3 points, and validation tracks test accuracy across candidates (Spearman {N['val_test_corr']['all']['pooled_spearman']:.2f}), although this is driven by candidates that score zero and is {N['val_test_corr']['all']['pos_val_1.7b']['spearman']:.2f} among plausible 1.7B candidates (n = {N['val_test_corr']['all']['pos_val_1.7b']['n']}). The statement that unstructured rewriting is 'at least as unreliable' as MPIR was removed. The repeated-scoring distribution of P0 is in Section 5.7.",
     "Sections 5.4, 5.5 and 5.7, Table 8."),
    ("Reviewer#1, Major comment 8",
     "The rubric and the judge are not validated; 'closely approximating expert human review' is unsupported; Objective 1 is declared achieved.",
     f"We report the judge's totals per round and for all candidates, their relation to validation and test accuracy, and a comparison with a stronger reference judge. Within-cell Spearman correlation with test accuracy is only {JV['1.7b']['median_spearman_test_within_cell']:.2f} for the 1.7B judge and {JV['gpt6luna']['median_spearman_test_within_cell']:.2f} for the reference judge, and the judges agree at {N['judge_validity']['agreement']['spearman']:.2f}. The 'expert review' claim and the statement that Objective 1 was achieved are removed. No human ratings were collected; we say so.",
     "Sections 3.3 and 5.8, Table 10, Section 6.3.3."),
    ("Reviewer#1, Major comment 9",
     "The contribution is over-claimed and the framing is inconsistent; the missing baseline initialization is presented as a general defect; quantify its effect by running the uncorrected variant.",
     f"Title, abstract, introduction and Sections 2-3 are reframed as a negative-result study; novelty and superiority statements are removed and Table 1 states that it describes design differences only. The initialization is now described as an implementation defect of ours, not a general property. We quantified its effect on the oracle-scored candidates: the uncorrected loop changes accuracy by {pp(LEG['mean_legacy_delta'])} points on average relative to P0 (changed the prompt in {LEG['legacy_changed_prompt']} of {LEG['n']} cells, worse in {LEG['legacy_worse']}).",
     "Title, abstract, Sections 1, 2.5, 3.2 and 5.9, Table 1."),
    ("Reviewer#1, Major comment 10",
     "Reference-quality problems and missing literature ([1], [3], [17], [22], [25], [48]-[51], venues).",
     "All references were audited against primary sources. Mismatched citations were replaced: translation now cites a translation study, question answering a QA evaluation, the few-shot claims canonical sources, and role prompting both supporting and contradicting evidence; the step-back and positional-bias statements are re-worded to what the sources support. Industry sources now carry URLs, access dates where retrievable and archive copies, and are described as practitioner guidance. Missing venue or identifier information was completed. We added and discuss literature on LLM judges, the limits of self-refinement, format and answer-extraction sensitivity, and reporting of negative results.",
     "Sections 2.1-2.4, 3.3 and the reference list."),
    # ------------------------------------------------------------------ Reviewer 1, minor
    ("Reviewer#1, Minor comment 1", "The abstract is long and mixes content.",
     "The abstract was rewritten to state the main finding, the scope (one model family, nine tasks) and the key limitation.", "Abstract."),
    ("Reviewer#1, Minor comment 2", "Objective excludes P0 although the algorithm retains it; Fig. 1 caption.",
     "The candidate set is now {P0} plus the refinements, and Figure 1's caption describes the legacy and repaired loops.", "Sections 3.1 and 3.2, Figure 1."),
    ("Reviewer#1, Minor comment 3", "'frequently does return P_final = P0' versus 'in all nine cells'.",
     "Both statements were replaced by the counts of Section 5.3.", "Sections 3.2 and 5.3."),
    ("Reviewer#1, Minor comment 4", "'Evaluated across three upstream APO methods and two model families' refers to unreported results.",
     "The sentence was removed; no model-family claim is made.", "Section 2."),
    ("Reviewer#1, Minor comment 5", "Table 1 is illegible and asserts design differences.",
     "Table 1 was reformatted and its caption states that it reports design differences only.", "Table 1."),
    ("Reviewer#1, Minor comment 6", "Table 2: values used, MPIR parameters, near-identical descriptions.",
     "Table 3 now lists the values used, the actual parameter names from the configuration files, and the MPIR parameters (rounds, sampling, token limits, guards).", "Section 4.4, Table 3."),
    ("Reviewer#1, Minor comment 7", "Table 3: confidence intervals, identical '+MPIR' columns, seed/temperature, chance level.",
     "Table 4 gives strict accuracy with 95% Wilson intervals, lenient accuracy and chance level; the identical '+MPIR' columns were dropped, and the caption states seed and decoding.", "Table 4."),
    ("Reviewer#1, Minor comment 8", "Hyperparameters 'unmodified' versus omitted synthesis components; Algorithm 2.",
     "We state exactly what was run (synthetic example generation disabled, expert identity not used at evaluation) and removed the algorithm listing that did not match.", "Section 4.4."),
    ("Reviewer#1, Minor comment 9", "Partition procedure; serving configuration.",
     "The seeded partition procedure, the use of original example indices and the shared partition across optimizers are described; the serving configuration is in Table 13.", "Section 4.2, Appendix C."),
    ("Reviewer#1, Minor comment 10", "Code identifiers; frequency of failure modes.",
     "Code identifiers were removed from the text; the frequency of each outcome is in Table 6.", "Sections 3.4 and 5.2."),
    ("Reviewer#1, Minor comment 11", "Vllm non-determinism is unverified.",
     f"Verified and quantified by repeated scoring (Section 5.7); the pilot and present scorings of the same prompts differ by up to {pct(PVR['max_abs_diff'])} points.", "Section 5.7."),
    ("Reviewer#1, Minor comment 12", "Practitioner recommendations generalize from one model size.",
     "The recommendations are rewritten with explicit scope qualifiers.", "Section 6.2."),
    ("Reviewer#1, Minor comment 13", "Free-rewrite results called 'inconsistent' and 'harmful'; +3.0 pp is 6 items.",
     "The text now notes that the hyperbaton difference is 4 test items and within noise.", "Section 5.5."),
    ("Reviewer#1, Minor comment 14", "Appendix A: the optimized prompt opens with an irrelevant question.",
     "Appendix A now states that the base prompts show the quality of the 1.7B-driven upstream baselines, which limits what refining them can show; the degenerate APE prompt is disclosed.", "Appendix A, Section 6.3.1."),
    ("Reviewer#1, Minor comment 15", "Cross-references (Sec. 4.5 as decision rule).",
     "All cross-references were checked by script; the decision rules are in Section 4.1 and the amendments in Appendix F. Section numbering in the compiled PDF now matches the prose.", "Throughout."),
    ("Reviewer#1, Minor comment 16", "Repetition of the null-result explanation.",
     "The manuscript was condensed and the explanation appears in the results and discussion only.", "Sections 1, 5, 6."),
    ("Reviewer#1, Minor comment 17", "Front and back matter placeholders; biographies.",
     "Volume, year, history and DOI fields were replaced (the DOI is marked as assigned on acceptance) and the author biographies are complete.", "Title page, biographies."),
    ("Reviewer#1, Minor comment 18", "'neither prior review round' and 'earlier drafts' unclear for an initial submission.",
     "These phrases were removed; the history of the work is stated once, in the introduction.", "Section 1, Acknowledgment."),
    ("Reviewer#1, Minor comment 19", "Reference entries lack venue or identifier information.",
     "Completed (DOIs, venues and arXiv identifiers), as in Major comment 10.", "Reference list."),
    ("Reviewer#1, Minor comment 20", "[25] CogBench: which result supports step-back.",
     "The sentence now states what the source shows (a behavioural benchmark of 35 LLMs in which step-back prompting fosters model-based behaviour) and that this is weaker support than the original step-back evidence.", "Section 2.1."),
    # ------------------------------------------------------------------ Reviewer 1, questions
    ("Reviewer#1, Question 1", "What differs between rounds; how many distinct candidates per cell?", "See Major comment 1.", "Section 3.2, Table 6."),
    ("Reviewer#1, Question 2", "Per cell and round: fallbacks, echo-rejected, validated candidates and their scores.", "See Major comment 2.", "Appendix E, Table 6."),
    ("Reviewer#1, Question 3", "Rubric totals for P0 and candidates; correlation with accuracy.", "See Major comment 8.", "Section 5.8, Table 10."),
    ("Reviewer#1, Question 4", "Original GPT-4o/GPT-3.5 results; the Gemini check; why not analysed.",
     "The original per-task results, including the Gemini check, are reproduced with the leak flagged (Appendix D). The Gemini model was both target and refiner, had the same defects and a 92% baseline with little headroom; it was not analysed further, and is not used as evidence.", "Appendix D."),
    ("Reviewer#1, Question 5", "Where and when was the pilot pre-registered; what changed after it began?",
     "It was not pre-registered in a registry. The rule was in REBUILD.md (commit 7612aa0, 2026-09-03), before the pilot. Pipeline changes made while the pilot ran are in Appendix C; the revision protocol and its amendments are in Appendix F.", "Sections 4.1, Appendices C and F."),
    ("Reviewer#1, Question 6", "Conclusions under lenient extraction; fraction of responses containing a correct option.",
     f"Lenient accuracy is reported for every base prompt and for the final prompts of the strong refiners. The apparent "
     f"weakness of exemplar-free baselines is largely a format effect, and it changes the reading of the strong-refiner "
     f"gains: under lenient extraction only one of them remains in the first seed (Section 5.6). The exemplar "
     f"intervention gains {pp(EX['mean_gain_by_optimizer']['protegi'])} points for ProTeGi under strict and "
     f"{pp(N['exemplar_lenient']['mean_by_optimizer']['protegi'])} under lenient extraction.", "Section 5.1, Table 4."),
    ("Reviewer#1, Question 7", "Stronger refiner than target with gate, harness and splits fixed.",
     f"Done with Qwen3-235B-A22B and GPT-6 Luna (Section 5.6): strict wins in {QW['wins']} and {LU['wins']} of 9 cells; "
     f"under lenient extraction the first-seed Qwen3-235B result is {CAPL['v2|qwen235']['wins']} win and "
     f"{CAPL['v2|qwen235']['losses']} significant losses, and the format-protected variants of the 1.7B refiner remain "
     f"worse than P0.", "Section 5.6."),
    ("Reviewer#1, Question 8", "Was this work or the original evaluation described in an earlier submission, preprint or publication?",
     "The original evaluation was described in a manuscript submitted to another journal (AI Open) and desk-rejected without external review. It has not been published or posted as a preprint. This manuscript is a new study.", "Section 1."),
    # ------------------------------------------------------------------ Reviewer 2
    ("Reviewer#2, Concern 1", "Clarify what differs across the N = 7 rounds of Algorithm 1 given deterministic decoding.", "See Reviewer#1, Major comment 1.", "Sections 3.2 and 5.2."),
    ("Reviewer#2, Concern 2", "Add Prompt-Level Distillation and CROP to the related work.",
     "Both were added and discussed with other work on small-model prompt optimization, after checking that they exist and what they report.", "Section 2.2."),
    ("Reviewer#2, Concern 3", "Contradiction about exemplars for APE and ProTeGi; APE's saved prompt is an unextracted induction meta-prompt.",
     "The statement that APE and ProTeGi used the three BBH exemplars was wrong: both ran exemplar-free (few_shot_count = 0) and only the expert few-shot condition uses the exemplars. This is corrected, and the degenerate APE prompt is disclosed as an artifact of our implementation.", "Section 4.4, Appendix A, Section 6.3.1."),
    ("Reviewer#2, Concern 4", "The novelty is incremental; position against PROPEL, PE2, Rethinking Prompt Optimizers, Principled Instructions, GReaTer and the self-correction papers.",
     "We agree and no longer claim novelty beyond the combination of an explicit rubric with a post-hoc gated role. The work is now positioned against these papers, and the self-correction and small-verifier results are used to interpret the findings.", "Sections 2.2, 2.3 and 2.5."),
    ("Reviewer#2, Concern 5", "Report A(L, P0) and round-by-round trajectories.",
     "Reported in Tables 15 and 16 and Table 6, separating degraded candidates, ties and guard-intercepted outputs; full logs are in the repository.", "Appendix E, Table 6."),
    # ------------------------------------------------------------------ Reviewer 3
    ("Reviewer#3, Concern 1", "Run the experiment that answers the research question; the pilot removes the capability gap and shrinks the judge at the same time.",
     f"We varied the refiner with the target fixed (Section 5.6). The question is only partly answered: a much larger refiner gains under strict scoring in some cells, GPT-6 Luna in fewer, and most of the strict gains do not hold "
     f"under lenient scoring.{FOLLOW_TEXT} The original GPT-4o/GPT-3.5 setting was not rerun.", "Sections 5.6 and 6.1."),
    ("Reviewer#3, Concern 2", "Refine from the current best prompt or carry the critique history forward.",
     "Done (variant V2, chained); the independent variant V1 is also reported.", "Sections 3.2 and 5.3."),
    ("Reviewer#3, Concern 3", "Reframe the paper around what it shows.", "Title, abstract, introduction, Table 1 and conclusion are reframed.", "Throughout."),
    ("Reviewer#3, Statistical tests", "Once prompts are byte-identical, the tests compare a prompt with itself.", "Agreed; those tests were removed (see Reviewer#1, Major comment 3).", "Section 4.6."),
    # ------------------------------------------------------------------ Reviewer 4
    ("Reviewer#4, Concern 1", "Justify the statistical analysis: paired examples, task clustering, serving variability.",
     "Per-cell paired bootstrap over test examples, hierarchical bootstrap over tasks, optimizers and seeds as a descriptive summary, and a measured serving-noise floor with a minimum detectable effect.", "Sections 4.6, 5.3 and 5.7."),
    ("Reviewer#4, Concern 2", "What supports generalization beyond the nine configurations?",
     f"The evidence now covers {MG['n_cells']} cells (three optimizers, nine tasks, up to three seeds). It still covers one model family and nine tasks; the scope limits are stated.", "Sections 4.2 and 6.3.2."),
    ("Reviewer#4, Concern 3", "Why is one seed sufficient; sensitivity to seed and serving conditions?",
     "It was not; Tier 1 now uses three seeds with up to three MPIR seeds, and sensitivity to serving conditions is measured.", "Sections 4.2 and 5.7."),
]


def fmt(paragraph, bold_label, text):
    run = paragraph.add_run(bold_label)
    run.bold = True
    run.font.name, run.font.size = "Calibri", Pt(11)
    run = paragraph.add_run(text)
    run.font.name, run.font.size = "Calibri", Pt(11)


def main():
    doc = docx.Document(TEMPLATE)
    paras = doc.paragraphs
    paras[0].runs[0].text = "Original Manuscript ID: "
    for r in paras[0].runs[1:]:
        r.text = ""
    paras[0].runs[0].text = "Original Manuscript ID: Access-2026-44920"
    for r in paras[1].runs:
        r.text = ""
    paras[1].runs[0].text = ("Original Article Title: Meta-Prompted Instruction Refinement: Bridging Manual Prompting "
                             "Techniques and Automatic Prompt Optimization")
    paras[15].runs[0].text = "Quang-Vinh Dang et al."
    for r in paras[15].runs[1:]:
        r.text = ""
    note = paras[59]
    for p in paras[30:59]:
        p._element.getparent().remove(p._element)
    intro = [
        ("Summary of changes. ",
         "The revised manuscript is a new study. We repaired the refinement loop, added a time-stamped protocol, "
         f"and ran {MG['n_cells']} base prompts with three optimizers, nine tasks and up to three seeds, plus a "
         "mechanism study, a refiner-capability study, format-protected variants, a judge-validity analysis, a "
         "serving-noise measurement and a format-exemplar intervention. We reframed the paper as a negative-result report. Two things we could "
         "not do: rerun the GPT-4o/GPT-3.5-turbo configuration, and collect human ratings of the rubric. We "
         "say so in the manuscript. The new title is 'Does Rubric-Guided Refinement Improve Automatically "
         "Optimized Prompts? A Pre-Specified Negative-Result Study'. Section "
         "numbers below refer to the revised manuscript; code, per-round logs and generations are public."),
    ]
    for label, text in intro:
        p = note.insert_paragraph_before("")
        fmt(p, label, text)
        note.insert_paragraph_before("")
    for label, concern, response, action in ENTRIES:
        p = note.insert_paragraph_before("")
        fmt(p, label + ": ", concern)
        p = note.insert_paragraph_before("")
        fmt(p, "Author response: ", response)
        p = note.insert_paragraph_before("")
        fmt(p, "Author action: ", "We updated the manuscript. " + action + ("" if action.endswith(".") else "."))
        note.insert_paragraph_before("")
    doc.save(OUT)
    print("wrote", OUT, "entries", len(ENTRIES))


if __name__ == "__main__":
    main()
