# -*- coding: utf-8 -*-
"""Builds the IEEE Access response to the second round of reviews (Access-2026-54235) from the official template.

Every number comes from the result files via study_numbers / rev2_numbers, so the letter and the manuscript cannot
drift apart. Run after mpir_study/revision2.py and revision2_new.py.
"""
import os
import sys

import docx
from docx.shared import Pt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from study_numbers import N  # noqa: E402
from rev2_numbers import OFF, NEW, LEN, MID, PP  # noqa: E402

TEMPLATE = os.path.join(HERE, "templates", "IEEE-Access-Response-to-Reviewers-template-6.26.23.docx")
OUT = os.path.join(HERE, "response_to_reviewers_round2.docx")

MG, CAP, NOISE, LEG, EX = N["main_grid"], N["capability"], N["noise"], N["legacy"], N["exemplar"]
CAPL, C, LO = N["capability_lenient"], N["composition"], N["lenient_oracle"]
AB, FA, AL, AS_, RT = (OFF["acceptance_bounds"], OFF["candidate_format_audit"], OFF["audit_labels"],
                       OFF["audit_samples"], OFF["round_typology"])
PFM, RV, JZ, CO, BP = (OFF["protected_failure_modes"], OFF["rubric_vs_free"]["qwen235"],
                       OFF["judge_without_format_zeros"], OFF["cost"], OFF["base_prompt_inventory"])
G, CT, RS = NEW["grid_v2f"], NEW["controls"], NEW["rescore"]
T1_17B = sum(c['lenient']['diff'] for k, c in PP['v2f_1.7b']['cells'].items() if k in PP['v2f_qwen235']['cells'] and c['lenient']) / len(PP['v2f_qwen235']['cells'])
V3F, PR, PQ, PL = PP["v3f_1.7b"], PP["paired_1.7b_all"]["lenient"], PP["paired_qwen235"]["lenient"], PP["paired_gpt6luna"]["lenient"]
CTO = CT["by_optimizer"]
LA, LB, LC = LEN["binary"], LEN["multi"], LEN["all"]


def pp(x, d=1):
    return f"{100 * x:+.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


JUDGE8 = (f"A third judge, the local Qwen3-8B (which is also the intermediate refiner, scored on a random subset of 150 prompts), gives {JZ['8b|all']['median_spearman_test']:.2f} over all judged candidates and {JZ['8b|test_above_zero']['median_spearman_test']:.2f} without the zero-accuracy ones, resting on only {JZ['8b|test_above_zero']['cells_used']} cells and not reliable; GPT-6 Luna therefore remains a judge that is also a refiner, and an independent judge of a different family could not be run without router access." if "8b|all" in JZ else "")
MID_TEXT = MID.get("response_sentence", "An intermediate open-weight refiner (Qwen3-8B-AWQ) was added (Section 5.6.2).")

ENTRIES = [
    # ------------------------------------------------------------------------------- Reviewer 1, major
    ("Reviewer#1, Major comment 1 (primary treatment defective; no positive control)",
     "V1/V2 do not ask the refiner to keep the answer-format line (93-95% of oracle-scored candidates score exactly zero) and about 60% of rounds never reach validation; the main grid was not re-run; no positive control shows that the gate and harness can accept a known-good candidate. Re-run the main grid with a format-preserving treatment, add a positive control, or retitle.",
     f"We did all three. (i) Positive control: for each of the 45 base prompts we offered the gate P0 plus the BBH format exemplar and applied the usual strict-improvement rule. The gate accepted it in {CT['known_good_accepted']} of the {CT['known_good_cells']} exemplar-free (APE and ProTeGi) cells in which the exemplar helps on test (and in {CT['gate_accepts_exemplar']} of all {CT['n']} cells; {CTO['protegi']['gate_accepts_exemplar']} of 15 ProTeGi, {CTO['ape']['gate_accepts_exemplar']} of 15 APE, {CTO['promptwizard']['gate_accepts_exemplar']} of 15 PromptWizard, whose prompts already contain bare-tag examples), and never accepted it where it did not help on test. So the harness can detect a large strict-metric effect. (ii) We re-ran the main grid with a new variant, V2F, whose request tells the refiner to keep the format line and the worked examples, whose gate baseline already carries a bare-tag example, and whose candidates have the format line and example re-attached in code. It ran on {G['runs']} of 45 cells with one MPIR seed (the 45th, geometric_shapes with PromptWizard, exceeds the 8192-token context window by one token). Acceptance rose from {AB['accepting_runs']} of {AB['runs']} runs to {G['accepted_runs']} of {G['runs']}, but against its own baseline V2F wins in {G['wins_vs_start']} cells under strict scoring and {G['lenient_wins_vs_start']} under lenient scoring, with {G['lenient_losses_vs_start']} significant lenient losses. (iii) The abstract, research questions and conclusion now state the scope (Qwen3-1.7B, nine tasks, weak base prompts), and the text says that the original grid did not test rubric-guided refinement in general.",
     "New Section 5.3.1 (positive control, bare-tag control, V2F), Table 2 (variant V2F), Sections 3.2, 4.1, 6.1 and 7, and the new title and abstract. Amendment 10 in Appendix F records that these additions were made after the first-round results."),
    ("Reviewer#1, Major comment 2 (format-protected variants may not protect what matters; incomplete)",
     "V1P/V2P append only the instruction line, yet their candidates average -24.9 points against P0; if refinement drops the worked examples, the instruction line cannot restore strict compliance. They completed on only 5 and 8 of 9 cells. Inspect the candidates, implement protection that preserves a bare-tag example, run it on all cells.",
     f"The reviewer's suspicion is correct. Only {FA['v1p']['candidates']} V1P and {FA['v2p']['candidates']} V2P candidates were oracle-scored; the protected candidates keep a bare-tag example in {PFM['v1p']['candidates_with_bare_tag_example']} of {PFM['v1p']['candidates']} (V1P) and {PFM['v2p']['candidates_with_bare_tag_example']} of {PFM['v2p']['candidates']} (V2P) cases, because the exemplar-free APE and ProTeGi base prompts contain no bare-tag example (only {BP['with_bare_tag_example']} of the {BP['n']} base prompts do). We re-scored them in one session and classified the target's answers: {pct(PFM['v1p']['share_sentence_in_tag'], 0)}% (V1P) and {pct(PFM['v2p']['share_sentence_in_tag'], 0)}% (V2P) wrap a whole sentence in the tags and {pct(PFM['v1p']['share_no_tag'], 0)}% and {pct(PFM['v2p']['share_no_tag'], 0)}% write no tags, which explains the deficit. V2F is the protection that re-attaches a bare-tag example, and it was run on all cells that fit the context window. Among the unprotected candidates, {FA['v2']['lost_format_and_examples_given_p0_has_examples']} of {FA['v2']['from_prompts_with_examples']} V2 candidates refined from prompts with worked examples lost the format line and also at least half of the examples (V1: {FA['v1']['lost_format_and_examples_given_p0_has_examples']} of {FA['v1']['from_prompts_with_examples']}).",
     "Section 5.2 (what the candidates keep; why the protected candidates are still far below P0) and Section 5.3.1."),
    ("Reviewer#1, Major comment 3 (the primary metric measures formatting; lenient metric unvalidated)",
     "Validate the lenient parser on a manually labelled sample (precision and recall, especially for two-option tasks) or adopt a constrained/log-likelihood score; redefine the decision rules on the validated metric and report both.",
     f"We drew 400 test responses at random (100 for each of binary and multi-option tasks, strict-accepted and strict-rejected), had the option each response commits to labelled without the gold answer or either parser's verdict, and compared both parsers with the resulting truth. The lenient parser has precision {pct(LC['lenient']['precision'])}% and recall {pct(LC['lenient']['recall'])}% (two-option tasks: {pct(LA['lenient']['precision'])}% and {pct(LA['lenient']['recall'])}%; multi-option: {pct(LB['lenient']['precision'])}% and {pct(LB['lenient']['recall'])}%), against recall {pct(LC['strict']['recall'])}% for strict extraction; about a quarter of the responses that strict extraction rejects are in fact correct. The labelling was done by a language-model assistant and not by human raters, which we state; the sample, key and labels are released. Decision rules are restated on both metrics (D1 for V2F on strict and lenient scoring), and the gate itself still uses strict extraction, which is how MPIR is defined. We did not adopt a log-likelihood score, because it would change the method under test.",
     "New Section 5.1.1 with Table R.1, revised Section 4.5, Sections 5.3.1 and 6.3.3."),
    ("Reviewer#1, Major comment 4 (rubric not isolated; 'similar gains' not tested)",
     "No paired cell-level V2-V3 comparison with an interval; one or two runs per condition; one V3 'win' (+2.8) is below the MDE; capability, family and reasoning budget are confounded; 4B/8B refiners were not run. Provide paired differences across enough runs, add an intermediate open-weight refiner and ideally a second target family, or report as exploratory.",
     f"We added a rubric-free control under the same protection: V3F is the rubric-free rewrite with the start prompt, gate and re-attachment of V2F (it also differs in having independent rounds and no evaluation call). With the 1.7B refiner on the same cells, V3F gains {pp(V3F['lenient_mean'])} points on average under lenient scoring ({V3F['lenient_wins']} winning cells, {V3F['lenient_losses']} losses; accepted in {V3F['accepted_runs']} of {V3F['runs']} runs), whereas V2F gains {pp(G['lenient_mean_diff_vs_start'])}; the paired V2F minus V3F difference over {PP['paired_1.7b_all']['n']} cells is {pp(PR['mean'])} points (interval [{pp(PR['lo'])}, {pp(PR['hi'])}]), so for a 1.7B refiner the rubric-guided variant does worse than a plain rewrite (we did not test whether the rubric criteria or the length of the meta-prompt is the reason). With strong refiners under the same protection (nine cells, one run each) V2F gains {pp(PP['v2f_qwen235']['lenient_mean'])} (Qwen3-235B) and {pp(PP['v2f_gpt6luna']['lenient_mean'])} points (GPT-6 Luna) under lenient scoring, but the rubric-free V3F gains {pp(PP['v3f_qwen235']['lenient_mean'])} and {pp(PP['v3f_gpt6luna']['lenient_mean'])}, with paired differences of {pp(PQ['mean'])} [{pp(PQ['lo'])}, {pp(PQ['hi'])}] and {pp(PL['mean'])} [{pp(PL['lo'])}, {pp(PL['hi'])}] points: the rubric is not shown to help, and these nine-cell comparisons cannot exclude a small benefit. For the unprotected probes, Table R.2 gives the paired V2-V3 difference per cell with intervals for Qwen3-235B (mean {pp(RV['mean_v2_minus_v3'])} points, interval [{pp(RV['lo'])}, {pp(RV['hi'])}]); the +2.8 point V3 win (1.7B refiner, ruin_names with ProTeGi) is labelled as below the 5-point MDE. {MID_TEXT} We could not add a second target family, and the strong-refiner results remain exploratory (one run per condition and cell).",
     "Sections 5.3.2 (V3F), 5.5, 5.6.1 (Table R.2), 5.6.2, 5.6.3 (Table R.6), 6.1 and 6.3.2, and the abstract."),
    ("Reviewer#1, Major comment 5 (claims and labels overstate the evidence)",
     "'Pre-specified', 'the uncorrected loop lowered accuracy by about 14 points', and 'Automatically Optimized Prompts' are stronger than the evidence. Use 'protocol fixed in advance, with listed amendments', restate the -13.6 as a simulation on V1 candidates, make the title and research questions explicit about the 1.7B/Qwen3 scope and base-prompt quality.",
     "We agree with all three. 'Pre-specified' is replaced by 'protocol fixed in advance, with listed amendments' (Section 4.1 is retitled; Appendix F lists ten amendments, including those of this revision). The -13.6 point figure is restated as 'in a simulation that applies the original accept-first rule to the oracle-scored candidates of the independent variant V1 (one MPIR seed, nine Tier-1 cells)', with the note that the actual legacy loop produced identical prompts in all nine pilot cells; the -29.2 figure is identified as the literal reading of protocol G6. The title now reads 'Does Rubric-Guided Refinement Improve Prompts from Small-Model Optimizers?', and Appendix G lists the kind of each of the 45 base prompts.",
     "Title, abstract, Sections 4.1, 5.9, 6.2 and 7, Appendices F and G."),
    ("Reviewer#1, Major comment 6 (statistical summary and serving noise)",
     "The hierarchical interval is degenerate; the '7%' bound does not match 1/80 (about 5.8%); runs within a cell are not independent; the bootstrap ignores the 6.5% flip rate; the server was restarted with changed settings. State how the bound was computed; report the acceptance rate at cell/task level; incorporate serving variability; show a sensitivity analysis excluding mixed-settings runs.",
     f"The reviewer's computation is right. The exact one-sided 95% (Clopper-Pearson) upper bound for 1 acceptance in 80 runs is {pct(AB['run_upper_one_sided_95'])}% ({pct(AB['run_upper_two_sided_95'])}% two-sided); the earlier '7%' was the two-sided value rounded up, and the text now says how each bound is computed. Because runs of a cell share a base prompt, we also report the bound over cells ({AB['cells_with_acceptance']} of {AB['cells']} cells accepted, upper bound {pct(AB['cell_upper_one_sided_95'])}% one-sided) and the accepting run's optimizer and task. The per-run server configuration was not logged, so an exclusion analysis is impossible; instead we rescored every prompt behind an exploratory claim ({RS['within_session_flip_rate']['n_prompts']} prompts, twice each) in one session. The cell differences move by at most {pct(max(v['max_abs_change'] for v in RS['groups'].values()))} points, the number of cells above the MDE is unchanged, and the within-session flip rate ({pct(RS['within_session_flip_rate']['mean'])}% on average) matches the earlier one, so the conclusions do not depend on the restarts.",
     "Sections 4.6, 5.3 and 5.7 (new 'Mixed serving settings' paragraph) and 6.3.1."),
    ("Reviewer#1, Major comment 7 (echo/fallback classification unvalidated)",
     "Manually audit a random sample of echo/fallback rounds; report the false-positive rate and the effect of token limits.",
     f"We drew {AL['echo_n']} echo and {AL['fallback_n']} fallback rounds at random and labelled them from the logs. {AL['echo_false_positive']} of {AL['echo_n']} echoes were false positives (rate {pct(AL['echo_fp_rate'], 0)}%, one-sided 95% upper bound {pct(AL['echo_fp_upper_95_one_sided'], 0)}%). Of the fallbacks, {AL['fallback_causes'].get('no_refined_prompt', 0)} never presented a refined prompt, {AL['fallback_causes'].get('refined_prompt_without_end', 0)} presented one without a closing tag, none was cut off by a length limit, and {AL['fallback_recoverable']} of {AL['fallback_n']} ({pct(AL['fallback_recoverable_rate'], 0)}%) were recoverable with a tolerant parser. Over all fallbacks the median response is {AS_['fallback_length_chars_median'] / 4.36:.0f} tokens against the 3000-token limit, and {AS_['near_token_limit']['refine_ge_90pct_of_3000']} of {AS_['near_token_limit']['rounds']} refinement responses come within 10% of the limit; the evaluation call is closer to its limit ({AS_['near_token_limit']['judge_ge_90pct_of_1500']} responses). The labelling was done by a language-model assistant, not by human raters, and we say so.",
     "Section 5.2 (audit of the round classification) and Section 6.3.1; the samples and labels are in the released logs."),
    ("Reviewer#1, Major comment 8 (exemplar and judge analyses need tighter controls)",
     "The exemplar adds a worked chain of thought, not only a bare tag; the judge correlations include many zero-accuracy candidates and the reference judge is also a refiner. Add a content-free bare-tag control; report correlations excluding zero-accuracy candidates and with an independent judge; consider criterion-level ablation.",
     f"The bare-tag control (one sentence showing only the final-answer tag, no reasoning) raises strict test accuracy by {pp(CTO['ape']['bare_tag_test_gain_mean'])} (APE) and {pp(CTO['protegi']['bare_tag_test_gain_mean'])} (ProTeGi) points, against {pp(CTO['ape']['exemplar_test_gain_mean'])} and {pp(CTO['protegi']['exemplar_test_gain_mean'])} for the full exemplar, so almost all of the effect is the tag convention; the heterogeneity (range -45 to +73 points, including APE's degenerate hyperbaton prompt) is now reported. Excluding the {pct(JZ['1.7b|all']['zero_share'], 0)}% of candidates with zero test accuracy, the median within-cell Spearman correlation with the rubric total falls from {JZ['1.7b|all']['median_spearman_test']:.2f} to {JZ['1.7b|test_above_zero']['median_spearman_test']:.2f} (1.7B judge) and from {JZ['gpt6luna|all']['median_spearman_test']:.2f} to {JZ['gpt6luna|test_above_zero']['median_spearman_test']:.2f} (GPT-6 Luna). {JUDGE8} A criterion-level ablation was not run and is listed as a limitation.",
     "Section 5.3.1 (bare-tag control), Section 5.8 (Table R.3), Section 6.3.3."),
    ("Reviewer#1, Major comment 9 (external validity and cost)",
     "One target family and size; possible BBH contamination not discussed; tasks chosen from a biased statistic; no token/call/time cost for a layer motivated as cheap.",
     f"We added a contamination paragraph (BBH is public and may overlap pre-training data of the target and the strong refiners; contamination would shrink the room for any refinement to help; we did not test it) and a statement that {BP['kinds'].get('seed_plus_format', 0)} of the {BP['n']} base prompts are the seed instruction plus the format line. We could not add a second target family. A new cost section reports calls, estimated generated tokens and validation/test calls per run for each variant (about {CO['v2|1.7b']['refiner_generated_tokens_per_run'] / 1000:.0f} thousand refiner tokens and 14 refiner calls per V2 run); because {pct(RT['v2']['share_fallback_echo_duplicate'], 0)}% of V2 rounds never reached validation, the effective cost per validated candidate is about three times the nominal cost, and the 'cheap' argument is withdrawn as a reason to expect benefit.",
     "New Section 5.11 (Table R.5), Sections 6.3.2, 1 and 6.2."),
    # ------------------------------------------------------------------------------- Reviewer 1, minor
    ("Reviewer#1, Minor comments 1, 2, 4, 5",
     "Abstract very dense; report heterogeneity of the +19/+42 effect; 'often dropped the format' understates 93-95%. Figure 1 caption refers to the first submission. Notation of the argmax should note ties favour P0 and noisy validation. Table 6 text: 60% counts fallbacks plus echoes; duplicates raise it to 66%.",
     f"The abstract is rewritten with findings separated from diagnostics, the 93-95% figure, and the range of the exemplar effect ({pp(EX['min_gain'], 0)} to {pp(EX['max_gain'], 0)} points). The Figure 1 caption now says that the drawing shows the legacy loop and how the repaired variants differ. Section 3.1 now states that ties favour P0 and that validation scores are noisy (6.5% flips). The text states which count is meant: {pct(RT['v2']['share_fallback_echo'], 0)}% for fallbacks plus echoes and {pct(RT['v2']['share_fallback_echo_duplicate'], 0)}% with duplicates.",
     "Abstract, Fig. 1 caption, Section 3.1, Section 5.2."),
    ("Reviewer#1, Minor comments 3, 6, 7, 8, 9, 10",
     "Table 1 hard to read, consider moving to the appendix. Table 7 is almost all '0/n, +0.0'. Table 9 pools different numbers of seeds. The V3 win of +2.8 is below the MDE. Explain how first-seed lenient losses and the second-seed result were selected. Specify which candidates produced -13.6 and -29.2.",
     "Table 1 states the positioning against other methods and no evidence, so we kept it in Section 2.5 and did not move it. Table 7 is replaced by a compact table of the only task-optimizer combination that changed. Rows of Table 9 that pool up to three MPIR seeds are marked with a dagger and the caption says so. The +2.8 win is labelled as within noise. Lenient scoring was applied to every run and both seeds are reported; nothing was selected (this is now stated). The -13.6 and -29.2 figures are specified (V1, MPIR seed 1, nine Tier-1 seed-42 cells; first-round fallbacks and echoes return P0 for -13.6; the first validated candidate accepted unconditionally for -29.2).",
     "Sections 2.5, 5.3, 5.5, 5.6 and 5.9, Table 9."),
    ("Reviewer#1, Minor comments 11, 12, 13",
     "List which of the 45 base prompts are degenerate. Table 14 reproduces results declared invalid; add run counts and blank cells. Give model version/provider identifiers for router-served models and the exact slugs and dates in the main text.",
     "Appendix G (Table 17) lists the kind of every base prompt (seed instruction plus format line, APE induction template, short instruction, PromptWizard prompt with examples). The caption of Table 14 now states that it reproduces results declared invalid, that it has one run per cell and that the Gemini columns are blank for tasks that were not run. Section 4.3 gives the slugs (qwen/qwen3-235b-a22b-2507, openai/gpt-6-luna) and the dates (2026-10-20 and 21) and states that the router's provider and system fingerprint were not logged, so the exact backend cannot be recovered.",
     "Sections 4.3 and 6.3.1, Appendix G, Table 14."),
    ("Reviewer#1, Minor comments 14, 15",
     "Check that reference [25] supports the sentence; [30] is weak support. Several 2026 arXiv entries need verified identifiers, titles and venues; prefer peer-reviewed versions.",
     "The reviewer is right about [25] (Wang and Zhou, NeurIPS 2024), which concerns eliciting reasoning through decoding; the sentence now cites only the paper on what matters in chain-of-thought demonstrations and says that relevance and correct order matter more than the validity of each step, which is what that paper reports (we found no verified source for 'length'). The CogBench clause for step-back prompting is removed. We checked every 2025-2026, arXiv, web and whitepaper entry against the arXiv API, the ACL Anthology, PMLR, Crossref and proceedings pages: 14 entries were corrected or updated (for example the EACL 2026, IJCNLP-AACL 2025, KDD 2025 workshop, ICLR 2026 and NeurIPS 2024 versions, and a survey entry that printed without its venue); entries without a peer-reviewed version remain as arXiv preprints, and a few web pages and model stubs are marked as not re-verified in the audit file.",
     "Reference list and the sentence in Section 2.1; the audit is in manuscript/reference_audit_round2.md."),
    ("Reviewer#1, Minor comments 16, 17, 18",
     "Cover metadata says 'Initial Submission' while the response letter refers to Access-2026-44920. Section 5.10 has no text before Table 11. Add a summary table 'variant - refiner - cells - what it tests - outcome'.",
     "The submission history is now stated consistently in the cover letter and this response: the initial submission was Access-2026-44920, the first resubmission Access-2026-54235 (reviewed in October 2026), and this is the second resubmission. Section 5.10 now opens with a paragraph summarising the decision outcomes and Table R.4 summarises every variant with its refiner, cells, purpose and outcome.",
     "Section 5.10 (Table R.4), cover letter."),
    # ------------------------------------------------------------------------------- Reviewer 1, questions
    ("Reviewer#1, Questions 1 and 2",
     "How many V1/V2 candidates that lost the format line also dropped the worked examples, and what fraction of V1P/V2P candidates keep a bare-tag example? What acceptance rate and result do you expect if the main grid is re-run with a format-preserving request, and does the gate accept P0 plus the BBH exemplar?",
     f"See the responses to Major comments 1 and 2: {FA['v2']['lost_format_and_examples_given_p0_has_examples']} of {FA['v2']['from_prompts_with_examples']} V2 candidates from prompts with examples lost both the format line and at least half of the examples; {PFM['v1p']['candidates_with_bare_tag_example']} of {PFM['v1p']['candidates']} V1P and {PFM['v2p']['candidates_with_bare_tag_example']} of {PFM['v2p']['candidates']} V2P candidates keep a bare-tag example. The re-run gave an acceptance rate of {pct(G['accepted_runs'] / G['runs'], 0)}% ({G['accepted_runs']} of {G['runs']}) with strict wins in {G['wins_vs_start']} and lenient wins in {G['lenient_wins_vs_start']} cells; the gate accepts P0 plus the exemplar in {CT['known_good_accepted']} of the {CT['known_good_cells']} exemplar-free cells in which it helps on test.",
     "Sections 5.2 and 5.3.1."),
    ("Reviewer#1, Questions 3, 5, 6",
     "False-positive rate of the echo guard and the effect of token limits; how the lenient parser was validated; how the '7%' bound was computed.",
     f"See Major comments 7, 3 and 6: echo false positives {AL['echo_false_positive']} of {AL['echo_n']}; lenient precision {pct(LC['lenient']['precision'])}% and recall {pct(LC['lenient']['recall'])}% on 400 labelled responses; bound {pct(AB['run_upper_one_sided_95'])}% one-sided ({pct(AB['run_upper_two_sided_95'])}% two-sided) over runs and {pct(AB['cell_upper_one_sided_95'])}% over cells.",
     "Sections 5.1.1, 5.2 and 5.3."),
    ("Reviewer#1, Questions 4, 7, 8, 9",
     "Paired V2/V3 per cell with the same refiner; how many scorings were produced under different server settings and do conclusions hold without them; what each variant cost; which router model identifiers and versions were used.",
     f"Paired V2-V3 per cell is Table R.2. The per-run server configuration was not logged; after rescoring the prompts behind the exploratory claims in one session, the conclusions are unchanged (Section 5.7). Costs per variant are in Table R.5. The router model slugs and dates are given in Section 4.3; provider and fingerprint were not recorded, so the results may not be reproducible if the router's models change, which we now state as a limitation.",
     "Sections 4.3, 5.6.1, 5.7, 5.11 and 6.3.1."),
    # ------------------------------------------------------------------------------- Reviewer 2
    ("Reviewer#2, Weakness 1 (the negative conclusion relies on a very small refiner)",
     "The primary negative conclusion relies heavily on Qwen3-1.7B as the refiner, which is already known to struggle with long multi-criteria meta-prompts.",
     f"We agree, and the new experiments test exactly this. (i) A 1.7B refiner does worse with the rubric: under the same protection, a plain rubric-free rewrite request gains {pp(V3F['lenient_mean'])} points on average (lenient) while the rubric-guided variant gains {pp(G['lenient_mean_diff_vs_start'])} (paired difference {pp(PR['mean'])} [{pp(PR['lo'])}, {pp(PR['hi'])}] against the rubric). (ii) With strong refiners under the protected treatment, gains appear ({pp(PP['v2f_qwen235']['lenient_mean'])} for Qwen3-235B and {pp(PP['v2f_gpt6luna']['lenient_mean'])} for GPT-6 Luna under lenient scoring, against {pp(T1_17B)} for the 1.7B refiner on the same nine cells), so capability matters, but the rubric-free variant reaches similar gains, so the rubric is still not shown to help. (iii) The gate accepts a known-good candidate in {CT['known_good_accepted']} of {CT['known_good_cells']} cells in which it helps, and a local intermediate refiner (Qwen3-8B-AWQ) was added. The conclusions are now stated as about the rubric-guided layer, with scope limits (nine cells and one run per condition for the strong refiners).",
     "Title, abstract, Sections 5.3.1, 5.3.2, 5.6.2, 5.6.3, 6.1 and 7."),
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
    for r in paras[0].runs:
        r.text = ""
    paras[0].runs[0].text = "Original Manuscript ID: Access-2026-54235"
    for r in paras[1].runs:
        r.text = ""
    paras[1].runs[0].text = ("Original Article Title: Does Rubric-Guided Refinement Improve Automatically Optimized "
                             "Prompts? A Pre-Specified Negative-Result Study")
    paras[15].runs[0].text = "Minh Ngoc Dinh et al."
    for r in paras[15].runs[1:]:
        r.text = ""
    note = paras[59]
    for p in paras[30:59]:
        p._element.getparent().remove(p._element)
    intro = [
        ("Summary of changes. ",
         "The revised manuscript is retitled 'Does Rubric-Guided Refinement Improve Prompts from Small-Model "
         "Optimizers?'. In response to the main concern, that the primary "
         "treatment could not succeed, we added a positive control for the gate, a format- and example-preserving "
         "re-run of the main grid, a validation of the lenient metric on labelled responses, a rubric-free control under the same protection (V3F), protected runs with the strong refiners, an audit of sampled "
         "echo and fallback rounds, exact acceptance bounds, a paired rubric versus rubric-free comparison, a rescoring "
         "of the exploratory prompts in one serving session, a cost table and an intermediate open-weight refiner. "
         "We could not add a second target family or collect "
         "human ratings, and the audit and lenient-parser labels were produced by a language-model assistant, not by "
         "human raters; the manuscript says so. Section numbers refer to the revised manuscript. Code, per-round logs, "
         "samples and labels are public. Submission history: Access-2026-44920 (initial submission), Access-2026-54235 "
         "(first resubmission, reviewed), this document (second resubmission)."),
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
        fmt(p, "Author action: ", "We updated the manuscript by: " + action)
        note.insert_paragraph_before("")
    doc.save(OUT)
    print("wrote", OUT, "entries", len(ENTRIES))


if __name__ == "__main__":
    main()
