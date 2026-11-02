# Final proofread, IEEE Access resubmission (round 2)

Scope: read all of manuscript_plaintext.txt (954 lines), then re-checked every finding against latex_ieee/manuscript.tex, because the .tex was edited while this review ran (12:55) and is newer than the plaintext dump (12:49). Numbers were checked against summary_numbers.json, rev2_offline.json, rev2_new.json, rev2_mid_refiner.json and lenient_validation.json.

Verified as consistent (no action): abstract, 5.3.1, 5.10, Table 11, Table R.4 and Section 6/7 all agree on V2F strict 4 wins (mean +0.7) and lenient 1 win / 3 losses (mean -0.4), 15 of 44 accepted, positive control 24 of 26 (29 of 45 overall), 8B refiner 1 of 9 runs per variant (+1.8 / +1.9, paired -0.1 [-5.7, +5.3]), 235B +8.7 / 4 wins, V2 minus V3 +2.7 [-3.5, +9.7], lenient parser table R.1 (all rows reconcile with lenient_validation.json), Table 6 row sums, Table R.5 cost, Table 15/16 counts (22 and 41 validated), Appendix G counts (21/4/5/15), Appendix D ranks 1/3/6, rescoring figures in 5.7, Table R.3 counts, Spearman values, Clopper-Pearson bounds.

## A. Numbers and claims that are wrong or unsupported

| Location (current .tex) | Problem | Exact replacement text |
|---|---|---|
| Table 8 caption: "Mean delta is the mean change in strict test accuracy" | Wrong denominator. The JSON mean is over all cells with a validated candidate, cells that accept nothing count as 0. Proof: V1 strict accepts in 1 cell (hyperbaton / PromptWizard, -8.5) and reports -0.9 = -8.5/9; V2 Qwen3-235B reports +8.7 = 78.3/9 (it would be +15.7 over its 5 accepting cells). The caption says "over those cells", which a reader will read as accepting cells. | "Mean delta is the mean change in strict test accuracy of the selected prompt versus P0, in points, over all cells with at least one validated candidate (a cell in which the rule accepts nothing counts as zero; for the larger gate, measured on the held-out half of the test split)." |
| 5.9 "applying that rule gives a mean test-accuracy change of -13.6 points ... when a fallback or echo round returns P0, as the pilot's code did" | Contradicted by summary_numbers.json legacy block. The pilot code returned P0 only on a fallback. First-round outcomes are 2 fallbacks, 2 echoes, 5 validated; the hyperbaton / PromptWizard echo cell scores 50.5% under the accept-first rule against P0 59.0%, so the echo candidate is accepted, and that cell is one of the 4 cells that "differed". | "applying that rule gives a mean test-accuracy change of -13.6 points relative to P0, where a fallback round returns P0 (as the pilot's code did) and an echo round is accepted as a candidate: test accuracy differed from P0's in 4 of 9 cells (worse in 4, better in 0)." |
| 5.9 "was validated in 5 cells and was a fallback or echo in 4, where the simulated legacy rule returns P0" | Same error, second occurrence. | "was validated in 5 cells, was a fallback in 2 (where the simulated legacy rule returns P0) and an echo in 2 (where it accepts the echoed prompt; one of these, hyperbaton / PromptWizard, scored 50.5% against 59.0%)." |
| 5.2 "Among the 120 V1 and 223 V2 candidates that were not echoes" | 223 is not "V2 candidates that were not echoes" (that would be 400 = 560 - 160). 223 = 560 - 177 fallbacks - 160 echoes (rounds with delimiters, includes 30 duplicates); V1 120 = 189 - 23 - 46. | "Among the 120 V1 and 223 V2 rounds that returned delimiters and were not echoes" |
| Table 7 caption "All other 26 of the 27 combinations" | Garbled quantifier (not wrong in number). | "The other 26 of the 27 combinations" |

## B. Template artefacts, plural errors, grammar

| Location | Problem | Exact replacement text |
|---|---|---|
| 5.3.1 V2F: "and in 1 cells under lenient scoring, with 3 significant" | plural error ("1 cells") | "and in 1 cell under lenient scoring, with 3 significant" |
| 5.5: "and 1 cell(s) in which the gate accepted a significantly harmful" | unfilled "(s)" template | "and 1 cell in which the gate accepted a significantly harmful" (generator: part3_results.py line 339 `cell(s)`; make it a plural helper) |
| Table 11 row "D1, V2F": "lenient: 1 wins, 3 losses, mean -0.4" | plural error | "lenient: 1 win, 3 losses, mean -0.4" |
| Declaration of Generative AI (tex, Acknowledgment paragraph): "readability of the manuscript text to generate initial code snippets during development" | Two clauses fused, sentence is ungrammatical and says AI was used "to improve text to generate code". | "Generative AI tools were used to improve the grammar and readability of the manuscript text, and an AI-assisted coding tool generated initial code snippets during development. Language-model assistants also labelled sampled echo and fallback rounds and responses for the lenient-parser validation, and helped verify bibliographic entries; these labels are screening aids released for re-checking, not human annotations. All AI-assisted text and code were reviewed, verified, and adapted by the authors, who take full responsibility for the manuscript's content, analyses, and conclusions." |
| Introduction: "An earlier version of this study was submitted elsewhere and was not externally reviewed" versus 4.1 "first submission to this journal, 2026-09-05; decision and reviews, 2026-10-19" | The two statements read as contradictory (was it reviewed or not). Removing the journal chronology in 4.1 (row C1) resolves it; if the intro sentence refers only to the first (leaked) evaluation, say so. | "The first evaluation was written up as a manuscript that was not externally reviewed; its numbers are exploratory and are not used as evidence here." |

## C. Leftover revision-history narration (all present in current .tex)

| # | Location (8-word quote) | Problem | Exact replacement text |
|---|---|---|---|
| C1 | 4.1 "first submission to this journal, 2026-09-05; decision and reviews, 2026-10-19; ... second review and the second-revision experiments, 2026-11-02" | journal process narrated | "The chronology is: first evaluation and audit (before September 2026); pilot, 2026-09-04 and 05; revision protocol committed 2026-10-19 (commit 8fd1397), before any revised experiment; revised experiments, 2026-10-20 and 21; further controls, audits and probes, 2026-11-02 (Appendix F, amendment 10)." |
| C2 | 4.1 D2 "added exploratorily in the second revision, Section 5.6.2" | | "a local 8B refiner was added exploratorily, Section 5.6.2" |
| C3 | 4.3 "The runs added in the second revision (V2F, the controls" | | "The later runs (V2F, the controls and the rescoring of Section 5.7) were produced on 2026-11-02 in a single serving session" |
| C4 | 4.4 "the first submission's statement that APE and ProTeGi also used them was wrong" | | delete the clause; keep "Only the expert few-shot condition uses the three BBH-author exemplars (Suzgun et al., 2023)." |
| C5 | 4.5 "In the second revision we therefore validated the lenient parser" | | "We therefore validated the lenient parser on a labelled sample, labelled by an AI assistant (Section 5.1.1)," |
| C6 | 5.1.1 "from the scoring cache of the second-revision session" | | "from the scoring cache of the 2026-11-02 session" |
| C7 | 5.3 "the first submission's '7%' was the two-sided value rounded up and was not labelled as such" | | "(6.8% two-sided)." (delete the rest of the clause) |
| C8 | 5.3.1 "These values were rescored in the second-revision session; Section 5.1 reports" | | "These values were rescored in a later session; Section 5.1 reports +19.5 and +42.3 from the earlier session." |
| C9 | 5.6.2 "The first-round protocol planned local 4B and 8B refiners ... In the second revision we therefore served" | | "The protocol planned local 4B and 8B refiners for D2, but they were not run in the main study. We therefore served Qwen3-8B-AWQ locally next to the 1.7B target ..." |
| C10 | 5.7 "larger than the 2% assumed in the first submission and is the reason" | | "This is larger than the 2% assumed at the outset and is the reason a 25-question validation split is noisy." |
| C11 | 6.3.1 "The second revision re-ran the grid with the format- and example-preserving variant" | | "The grid was re-run with the format- and example-preserving variant V2F, with one MPIR seed per cell and on 44 of 45 cells (one request exceeded the context window)." |
| C12 | Appendix C intro "found in the audit that preceded the first revision, and items 12 and 13 during it" | | "items 9-11 were found in an audit after the pilot, and items 12 and 13 during the main grid, after it had run." |
| C13 | Table 13 row "Intermediate refiner (second revision)" | | "Intermediate refiner" |
| C14 | Appendix F intro "Amendment 10 lists the additions made in response to the second round of review; they were decided after the first-round results were known" | | "Amendment 10 lists the later additions; they were decided after the main-grid results were known and are post hoc in that sense, although none was chosen after seeing its own result." |
| C15 | Appendix F item 4 "were not run in the first revision (an 8B refiner was added in the second, amendment 10)" | | "were not run in the main study (an 8B refiner was added later, amendment 10)" |
| C16 | Appendix F item 10 "2026-11-02, second revision. Added in response to the reviews:" | | "2026-11-02, additions. Added after the main-grid results:" |

## D. Stale text in manuscript_plaintext.txt only (already fixed in the .tex; regenerate the dump)

- Abstract and Section 7: "won in 1 cell(s)" (fixed in .tex).
- 5.3.1: "adds about 1 points" and "Three limits apply" followed by four clauses (the .tex now says "about 1 point" and enumerates First/Second/Third plus "Finally").
- 5.6.2: "1 winning cells", "1 wins", "ahead in 1 cells" (the .tex has the singular forms).
- 5.2: "the 60% reported in the first submission" sentence (removed in .tex).
- References: entry "reynolds2021b. ?." is not in cas-refs.bib or manuscript.bbl, so it is a dump artefact only; Table 1 markers such as "‹19›" are likewise dump artefacts of \cite.
- Conclusion: "and the first evaluation's GPT-4o results remain unverified" as a list item (rewritten in .tex).

Generator sources that still emit the "(s)" and singular/plural problems: part4_discussion_back.py:124 and part_abstract.py:27 use `cell(s)`; part3_results.py:339 likewise. If the .tex is regenerated from these, B-rows 1-2 return.
