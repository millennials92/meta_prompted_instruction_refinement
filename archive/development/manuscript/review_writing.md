# Independent review and writing edit of the current manuscript

Source: manuscript/review_text_current.txt (extracted text of the compiled PDF). Section numbers below follow the PDF (Results subsections A..J = 5.1..5.10). Letters such as "fi"/"ff" missing in the extraction (e.g. "renement", "dierence") are probably extraction ligature artefacts; verify once in the PDF.

Abstract word count: 223 words, single paragraph. This is within the 250-word IEEE Access limit. A tighter 241-word replacement is proposed in item M10.

---------------------------------------------------------------------

## A. Coverage of the original criticisms (short)

| Criticism | Status | Residual weakness |
|---|---|---|
| Non-independent rounds | Addressed (V1/V2, Alg. 1, Sec. 5.2) | Fine. The V0 numbers conflict with the Introduction (M5). |
| Unreported round composition | Addressed (Table 6, Appendix E) | Table 9 vs Table 6 inconsistency (M5). |
| Degenerate statistics | Mostly addressed (descriptive, MDE, noise floor) | The "null" is driven by the refiner failing to produce usable candidates. The hierarchical CI [-0.16, +0.00] is itself degenerate. |
| Biased tasks, "pre-registration" | Addressed in wording | The timeline is inconsistent (M8). The 235B/Luna runs were not in the protocol (M3). "Dated amendments" carry no dates. |
| Untested capability-gap claim | Tested, but the interpretation is unbalanced and partly post hoc | See M2, M3, M4. |
| Harness/format confounds | Largely addressed | The oracle analysis (RQ2) is still strict-extraction only, and MPIR's refine prompt does not tell the refiner to preserve the answer tags (M4). |
| Coarse gate; ungated free rewrite | Addressed (Table 8, V3) | Spearman 0.92 is driven by zero-accuracy candidates. See S3. |
| Unvalidated rubric/judge | Addressed, honestly bounded | OK. Judge totals have no human anchor, which is stated. |
| Over-claiming, framing | Much improved | Several over-claims remain (M2, M6, S1). |
| Reference problems | NOT addressed | Wrong citation numbers, a missing BBH reference, "UNVERIFIED" tags in the list (M1). |
| Repetition | Partly | The "first submission" and "pilot" narrative is repeated 8+ times (S6). |
| Cross-references | NOT addressed | Several wrong (M7). |
| Placeholders | NOT addressed | Empty heading "5.6.1", "UNVERIFIED" tags, "Reviewer comments asked..." (M1, M9). |

---------------------------------------------------------------------

## B. MUST FIX

**M1. Reference list and citations are still broken.**
- (a) BBH is cited as [20], but [20] is Yuksekgonul et al. (TextGrad, Nature). The BBH paper (Suzgun et al., 2023) is not in the reference list. [20] is used wrongly in Sec. 1 ("23 BBH tasks [20]"), Sec. 4.B and 4.D ("three BBH-author exemplars [20]"), and Appendix B. Add Suzgun et al. and re-point those citations.
- (b) Table 1 row labels carry stale numbers: APE 19, ProTeGi 20, EvoPrompt 21, OPRO 22, PromptWizard 23, PE2 24, PROPEL 26, GEPA 59. They should be [14], [15], [16], [17], [18], [39], [41], [43].
- (c) The reference list contains internal editorial notes that must not be published:
  - [46] "UNVERIFIED; cite as arXiv preprint unless confirmed"
  - [49] "accepted at FEVER 2026 (per arXiv comment)"
  - [51] "proceedings pages UNVERIFIED"
  - [52] "(per arXiv comment)"
  - [53] and [54] "no peer-reviewed venue found"
  - [59] "NeurIPS 2024 listing UNVERIFIED"
  - [69] "Publication month UNVERIFIED"
  - [71] "no access date asserted"
  
  Resolve each one (verify, or cite plainly as an arXiv preprint with its ID) and delete the notes.
- (d) Entries with no venue or arXiv ID: [7], [19] (no venue), [42], [44], [45], [47], [67], [68], [72]. IEEE Access needs a complete entry for each.
- (e) Missing references for tools and methods the paper depends on: vLLM (Kwon et al., 2023), AWQ, Qwen3-235B-A22B, GPT-6 Luna, Gemini 3.5 Flash-Lite (model/version/provider/date), the paired bootstrap and the Wilson interval.
- (f) Citation-claim mismatches:
  - [7], [8] (SaulLM, AlpaCare) are domain LLMs. They do not show that "prompt design has become a bottleneck in law and medicine" (Sec. 1).
  - [33] is cited in Sec. 2.A as reporting "recency effects" and in Sec. 3.C as stating that instruction placement made "no difference" for GPT-4-class models. Reconcile these.
  - [19] (a survey) is cited for "may forfeit the structure that human-written prompts carry". Cite a primary source or soften.

**M2. Logical gap: "the gains do not show the rubric helps" is not coherent as stated.**
Criterion 5 of the rubric is "Output Format with Examples", and criterion 3 is guided chain of thought. The 235B-accepted candidate (Fig. B.3) adds exactly an output-format instruction and a worked example. Sec. 5.6 and the Discussion therefore say the gains are "the same lever as the single-exemplar intervention ... the data do not show that the other criteria add anything". That is a fair statement, but the rubric itself is not exonerated.
- Accurate framing: the gains are consistent with criterion 5 alone. No ablation separates it from the other six criteria. A one-line format-repair step would deliver the same benefit more cheaply (+42 points for ProTeGi with no refiner).
- Replace "so they do not isolate the rubric" with "so they are consistent with a single rubric criterion (output format) and do not show that the full rubric is needed".
- The same applies in the Abstract and Conclusion.

**M3. The capability-gap evidence is post hoc and its reporting departs from the protocol.**
- Protocol rule D2 (Sec. 4.1) is defined for the 8B refiner. Sec. 4.C and Table 13 describe 4B-AWQ and 8B-AWQ refiners. No 4B or 8B result appears anywhere in Sec. 5. Instead D2 is evaluated for GPT-6 Luna and Qwen3-235B.
- Threats to validity admit that the follow-up runs "were added after the results were seen", yet Table 11 states "met (Qwen3-235B)".
- Either report the 4B/8B results, or state plainly (Sec. 4.1 and Appendix F) that D2 was re-specified from 8B to the 235B and Luna refiners, when and why, and mark the D2 outcome as exploratory.
- Qwen3-235B is the same family as the target, so "capability" and "family" are confounded. The paper notes this in one clause only.
- GPT-6 Luna ("minimal reasoning") and Qwen3-235B (reasoning setting not stated) are not described in the Methods at all. Add serving, API or version, date, temperature, thinking setting and token limits for both.
- Balance: Luna reaches validation in 90% of rounds, yet only 3 of 57 candidates beat the baseline and 1 cell wins. The Discussion calls the evidence "mixed" (good), but the Abstract and Introduction lead with "a much larger refiner can help". Say what the data show:
  - one large refiner out of two produced wins;
  - the wins are confined to cells whose base prompt was broken;
  - Luna's mean +3.0 comes from a single cell (about +27 points in 1 of 9);
  - there are no confidence intervals on the 9-cell means;
  - no multiplicity correction was applied to the four "significant" cells.

**M4. Two undermining confounds are not acknowledged.**
- (a) MPIR's refine request (Fig. 2) never tells the refiner to preserve the answer-format line, whereas the free-rewrite control prompt (Fig. A.4) does ("Preserve the original task, answer format, and examples"). Tables E1/E2 show that most validated 1.7B candidates score exactly 0.00, which is what dropping the answer tags would produce. So "candidates are far worse" (Sec. 5.2, 5.4) may mostly measure format loss rather than rubric-guided quality.
  - Report the share of candidates lacking the delimiter instruction.
  - Re-score the oracle candidates with lenient extraction. If the sign still holds, the claim becomes much stronger.
  - State the asymmetry between the MPIR prompt and the free-rewrite prompt as a limitation.
- (b) The claim that the PromptWizard/hyperbaton +30 gain is a "harness reason" (Sec. 5.6) contradicts Table 4: that base prompt has 100% delimiters and strict = lenient = 59.0%. Either show that its accepted candidate also repairs the format, or drop it from the "format-repair" explanation. The same check is needed for ruin_names/protegi (+12.0) and penguins/protegi (+11.5): Fig. B.3 shows only one of four winning cells.
- A cheap, decisive test: lenient re-scoring of the 235B final prompts versus P0 (and the single-exemplar appended to the same P0 versus 235B refinement on the same cells).

**M5. Internal numeric inconsistencies.**
- (a) The Introduction says the pilot's refined prompt was "byte-identical to the base prompt in all nine cells", but Table 6 row V0 shows 1/9 runs changed and 1 accepted candidate. Explain (re-run under different serving?) or correct.
- (b) Table 9 row "Qwen3-1.7B (self), V2" gives Val. better = 1 but Runs changed = 0/27. Under a strict gate a validation-better candidate is accepted, so one of the two numbers is wrong, or the single better candidate lies outside the 27 Tier-1 runs. Say which.
- (c) Sec. 5.5: free-rewrite deltas (+3.0, -15.5, -14.6) do not match Table 4 vs Table 5. Table 4 PW strict is 59.0, 63.5, 71.9 and Table 5 free rewrite is 61.0, 44.5, 55.2, i.e. +2.0, -19.0, -16.7. If the text uses the pilot's earlier PW scores (58.0, 60.0, 69.8), say so. Otherwise correct either the text or the tables.
- (d) The sample sizes behind Spearman: 211 pooled candidates (Sec. 5.4) versus "223 prompts" in the Table 10 caption (the arithmetic does not obviously reconcile). Table 8 chained rows have 8 cells while the others have 9, with no explanation.
- (e) The 6% vs 6.5% vs "6%": the noise paragraph says 6.5%, the Abstract and Discussion say 6%. Use 6.5%.
- (f) Sec. 5.7 and Sec. 6.C say repeated scoring "moves accuracy by 3.3 points on average". 3.3 is the range per prompt (SD 1.4). Rephrase.
- (g) Table 7 cell values (e.g. "1/7 changed, -0.6") are ambiguous: is -0.6 the mean over 7 runs, or the one changed run? Spell it out in the caption.
- (h) Sec. 5.5 says the rubric-free gated rewrite "does no better". But V3 is marginally better than V1/V2 (+0.7 vs -0.5/0.0; 2 winning cells vs 0; 16/27 accepted). Replace with "does at least as well, and its two winning cells are not accompanied by any rubric effect (Sec. 5.5)", and say which cells win.

**M6. Over-claims in the Abstract, Discussion and Conclusion.**
- (a) Conclusion: "The uncorrected loop of the first evaluation would have reduced accuracy". Sec. 5.9 states the effect with the original GPT-4o refiner "is unknown". Rephrase to "With a 1.7B refiner, the uncorrected loop would have reduced accuracy by about 14 points (9 cells, one MPIR seed)". The Abstract and Discussion (ii) also need "with a 1.7B refiner" attached to the 14-point figure.
- (b) Abstract: "a single answer-format exemplar outperformed any refinement". It was tested only on exemplar-free APE/ProTeGi prompts (+19.5/+42.3 average), not against the 235B refiner on the same cells (+30/+29/+12/+11 on 4 cells). Replace with "raised accuracy of exemplar-free prompts by 20 to 42 points on average, far more than any refinement we observed".
- (c) "three optimizers" and "optimizers tested": Appendix A shows that the APE and ProTeGi base prompts for two of the three pilot tasks are the unchanged seed instruction ("Let's think step by step") and that the APE hyperbaton prompt is the induction template itself. In Table 5, zero-shot CoT often beats these "optimized" prompts under strict scoring. A large share of the base population is therefore not an optimized prompt at all. State this in the main text (Sec. 4.D / 6.C) and temper "refining optimized prompts of three optimizers"; also label PromptWizard as "reduced configuration" because synthetic-example generation was disabled.
- (d) Title says "Small Open-Weight Models" but the headline contrast involves a 235B refiner and a closed model. Suggest: "...Automatically Optimized Prompts? A Pre-Specified Negative-Result Study with a Small Self-Refining Model". The running header ("Rubric-Guided Prompt Refinement: A Pre-Specified Negative Result") differs from the title; unify.
- (e) "Pre-specified" is defensible only for the primary rules D1 and D3 and the 1.7B grid. State in the Abstract, in one clause, that the larger-refiner and exemplar analyses were exploratory or added later.

**M7. Cross-reference errors.**
- Sec. 3.B: "the defect Section 5.8 quantifies" should be Sec. 5.9. (5.8 is the judge analysis.)
- Sec. 3: "(Figure 1)" for the three-stage cycle, but Figure 1 is the evaluation meta-prompt and no cycle diagram exists. Sec. 3.D cites "Figure 2" (evaluation) and "Figure 3" (refinement), but the captions are Figure 1 and Figure 2. Either add the cycle diagram or fix the numbers.
- Sec. 5.6 and Sec. 6.C cite Section 5.6.1 ("Follow-Up Runs (Exploratory)"), which has a heading and no text (see M9).
- Appendices are headed only "APPENDIX." with no letters, but the text cites Appendices A.3, B, C, D, E, F. The first appendix's subsections are lettered A, B, C, whereas the text says "Appendix A.3" for what is subsection "C". Use IEEE style (APPENDIX A, B, ...) and cite consistently.
- Table numbering: "TABLE 12: Table C1.", "TABLE 13: Table C2.", "TABLE 14: Table D1.", "TABLE 15: Table E1.". The text says "Table E2" while the table is Table 16. Use one scheme.
- Sec. 4.A refers to "Appendix F" for dated amendments, but the amendments listed carry no dates.

**M8. Timeline and pre-specification wording need checking.**
- "Manuscript received August 2026" versus the pilot's go/no-go rule committed 2026-09-03 and described as preceding the pilot, versus a first submission that already contained the pilot. Make the chronology explicit in one place (a short table or timeline: first evaluation, audit, desk rejection, first submission, pilot, review, protocol commit 10-19, runs, revision 10-20).
- Protocol committed 2026-10-19; manuscript revised 2026-10-20. An editor will ask whether the 45-cell grid, the oracle scoring, two strong refiners and the follow-ups were run in one day. State the actual run dates (or say that runs spanned the dates in the commit log).
- Sec. 4.A: "Dated amendments are listed in Appendix F". The appendix has an undated list. Add the dates and commit hashes.

**M9. Placeholders and reviewer-response text left in the manuscript.**
- Empty subsection heading "1) Follow-Up Runs (Exploratory)" at Sec. 5.6.1: write it or remove it and its three references. If the second-seed Qwen3-235B result is wanted, it needs a table row and its own text.
- Appendix D: "Reviewer comments asked for its numbers". This is response-letter language. Replace with "reproduced for completeness".
- Repeated remarks addressed to the first submission ("the first manuscript's statement that APE and ProTeGi also used them was wrong"; "the claim of the first submission ... is withdrawn"; "tests of the first submission are not reported"; "assumed in the first submission"; Acknowledgment thanking reviewers). A standalone paper should state the correct facts and move the corrections to the response letter. Keep one history paragraph (see S6).
- "Table 14" ("Di", "Gem. PW") has non-standard abbreviations; define them or spell out. "Di" probably is a lost "Diff"/"Delta" glyph; check in the PDF.

**M10. Abstract (223 words, compliant, but with problems).**
Problems:
- "leaving its refinement rounds, statistics and harness open to criticism" is review-speak.
- "cells" and "runs" are undefined.
- "the validation gate tracked test accuracy" is vague.
- The 14-point figure has no scope.
- The 6% should be 6.5%.
- The exemplar claim is over-stated.

Proposed replacement (241 words):

> Meta-Prompted Instruction Refinement (MPIR) is a post-hoc layer that asks a language model to score an automatically optimized prompt against a seven-criterion rubric, rewrite it, and keep the rewrite only if it beats the original on held-out questions. An earlier evaluation with a closed refiner reported mixed gains, but audits found a validation leak and an unscored baseline, and a small pilot showed no effect while leaving its refinement rounds, statistics and answer-extraction harness open to criticism. We report a re-test whose decision rules were fixed in a time-stamped protocol before the runs. We repaired the refinement loop so that rounds are distinct, logged every round, and refined 45 base prompts (PromptWizard, APE and ProTeGi on nine Big-Bench Hard tasks, up to three seeds) with Qwen3-1.7B as both target and refiner. The layer changed the prompt in 1 of 80 runs and won in 0 of 45 cells. Most candidates were worse than the original, validation accuracy tracked test accuracy, and the uncorrected loop would have cost about 14 accuracy points. Identical prompts disagree on 6.5% of examples between scorings, and appending one answer-format exemplar to exemplar-free prompts raised accuracy by 20 to 42 points on average, more than any refinement. With a 235B-parameter refiner the layer won in 4 of 9 cells, but the gains coincide with a repaired answer format, so they do not isolate the rubric. Conclusions are limited to Qwen3 targets, nine tasks and the optimizers tested.

Still to fix after adopting it:
- Add "(exploratory)" before "With a 235B-parameter refiner" if M3 is resolved that way.
- Add "with a 1.7B refiner" to the 14-point clause.
- Define "cell" once in the Introduction (see S4).

---------------------------------------------------------------------

## C. SHOULD FIX

**S1. Capability-gap discussion balance (Sec. 5.6, 6.A, Intro, Conclusion).**
Present the two strong refiners symmetrically. Suggested replacement for the last paragraph of Sec. 5.6:

> The two strong refiners behave differently. Qwen3-235B-A22B, which belongs to the target's family, produced a valid candidate in 92% of rounds and a significant gain in 4 of 9 cells; GPT-6 Luna, run with minimal reasoning, produced a valid candidate in 90% of rounds but a gain in only 1 of 9 cells. Nine cells, one MPIR seed per refiner, no multiplicity correction and post hoc selection of both refiners mean that we cannot say whether capability, model family, or reasoning budget explains the difference. What the data do show is that the gains occur in cells whose base prompt scored poorly, and that the accepted candidates in the one case we inspected (Fig. B.3) add an answer-format instruction and a worked output example.

**S2. Winner's curse and regression to the mean.**
Gains of +30/+29 occur on the weakest P0s (P0 val 0.16, 10.5% test). Part of a gain is regression from a harness-broken baseline. Add one sentence and, ideally, report gains on the strict-vs-lenient base for those cells.

**S3. Logic of RQ2 ("gate is not the bottleneck").**
- A Spearman of 0.92 pooled over cells inherits between-cell and zero-accuracy variance. The relevant question (can the 25-question gate rank near-P0 candidates?) is answered only by the within-cell figure and by Table 8.
- Say that the oracle result holds for the 1.7B refiner's candidates, which are mostly broken, so the gate has nothing to find. It says nothing about a strong refiner's candidates. Tie this to M4a.
- D3 is "met/not met" on seed-1 candidates only (9 cells, 8 for the chained variant). State why 8.

**S4. Terminology.**
- "Cell", "run", "seed" (data-split seed 42-44), "MPIR seed" (refinement RNG), "variant" (V0..V3, V2L), "condition", "Tier", "P0" and "base prompt" are used interchangeably in places. "Cell" is defined only in Sec. 4.B but used from the Abstract. The Introduction's "nine (task, optimizer) cells" uses a different meaning from "45 cells (task, optimizer, seed)".
- Add a short notation box in Sec. 3 or 4: cell = (task, optimizer, split seed) with its own P0; run = one MPIR execution on a cell, with a given MPIR seed (1 to 3); variant = V0..V3/V2L; round = one critique-rewrite-validate step (N = 7).
- Use "split seed" and "MPIR seed" consistently.

**S5. Introduction motivation and story.**
- The Introduction opens well but spends its second half on provenance (the desk rejection, the code audit). The reader reaches the research questions only after the history.
- Suggested order: (1) the problem, (2) the idea and why it is plausible, (3) why a clean test is needed and what a negative result is worth to practitioners (do not add a refinement layer without a format check; do not trust a rewrite without scoring the original), (4) one short history paragraph, (5) RQs and contributions.
- State why a negative result matters: the answer changes what practitioners should spend compute on.
- The contributions list sets "Harness findings" against "Diagnostics", and both include the same noise and format analyses; merge.
- Roadmap sentence ("Section 2 reviews ...") can be cut for a journal this size.

**S6. Repetition.**
- The pilot narrative (identical prompts in 9 cells, the unscored baseline, the leak) appears in the Abstract, Sec. 1, Sec. 3.B, Sec. 4.A, Sec. 4.G, Sec. 5.2, Sec. 5.9 and Appendices C/D.
- The "format lever" result (+42.3 points) appears in Sec. 5.1, 5.6, 6.A, 6.B(i), the Conclusion and Fig. B.3. State it once in the Results with full numbers and refer back.
- The headline "1 of 80 / 0 of 45" is repeated six times; keep Abstract, Sec. 5.3 and Conclusion.
- Sec. 6.A recaps Sec. 5.2 to 5.5 almost sentence by sentence. Compress it to the findings, plus what is *not* shown.

**S7. Rubric provenance.**
Sec. 3.C says the rubric came from the literature "and iterative experimentation on BBH-style tasks". If any BBH task (including the three Tier-1 tasks) informed the rubric, that is a design-on-test leak. Say which tasks, or state that none of the nine were used.

**S8. Baselines and clean-room implementations.**
ProTeGi scores 1.0% to 10.5% (strict) and APE 15.6% on penguins, both below the zero-shot CoT seed (Table 5). Readers will ask whether the implementations are faithful. Add a one-paragraph sanity check, or say outright that these baselines are strict-extraction-limited and not representative of published performance.

**S9. Statistical presentation.**
- The hierarchical interval [-0.16, +0.00] with 44 of 45 cells exactly zero is nearly degenerate. Say so. Add "the one non-zero cell" (hyperbaton/PW, -0.6) and a lower bound on the conditional effect, e.g. a rule-of-three style bound on the acceptance rate (acceptance 1/80, upper 95% about 6%).
- Report the exact MDE logic ("5.0 points" is for 200 examples and paired discordance 6.5%) in the Methods, not only in the Results.
- The 235B "significant win" per cell has no multiplicity correction and no pooled interval across the 9 cells. Provide a hierarchical interval as for the 1.7B grid.

**S10. Tables and figures (visible in the text).**
- Table 2: the variant labels "V2 chained (primary)" and "V2L" collide with Table 6's "V2L original rubric". The cell "incumbentscored candidates" is missing a separator.
- Table 3: header "ComponentParameter"; the 'see Section 3.4' entry in the value column should read "Sec. 3.4".
- Table 7: header "PromptWizardAPE"; each entry is "n/m changed, delta" in one cell. Use two sub-columns, or split into a runs-changed table and a delta table. Tier-1 tasks are starred but the footnote hides the 7/7/5 run counts (3 seeds x 1-3 MPIR seeds); add a column "split seeds".
- Table 8: define "Better/worse cells"; the "larger gate" row uses half of the test set, which changes the sample for Delta. State the denominator.
- Table 10: the caption says "223 prompts"; reconcile with the 211 candidates in the text. Add the inter-judge Spearman (0.38) quoted in the text.
- Table 11: "Not met (Luna); met (Qwen3-235B)" is cramped; use two rows.
- Tables 12-16: the doubled labels (see M7). Table 14 has 12 columns and is hard to read; consider moving it to a supplementary file, since it is "exploratory only" and not evidence.
- Fig. 1/2 are code-style text boxes, not figures; consider a proper diagram of the three-stage cycle, which the text promises.
- Algorithm 1: line 9 has the line break garbled in the text; check the PDF.
- All figure and table captions should be self-contained; several refer to "(Section X)" for the main definition.

**S11. Language and hedging (examples).**
- "Accuracy differences smaller than this cannot be attributed to a prompt." Too strong for a result that depends on one serving stack: "...are not reliably attributable to a prompt on this serving stack."
- "an ungated rewrite by this model is harmful": based on three tasks, one scoring each. Use "was harmful on two of three tasks".
- "That conclusion survives the points raised against the pilot." Informal; use "The conclusion holds after addressing the weaknesses of the pilot."
- "The honest summary is ..." (Sec. 5.6). Remove "honest"; it implies the rest is not.
- "Most refinement attempts by this refiner never reach validation, which is a failure of instruction following, not a verdict on the rubric." Fine, but then the RQ1 answer is about a 1.7B refiner's instruction following. Say so in the RQ1 summary.
- Sec. 2.A: "MPIR does not use it." (about self-consistency) is irrelevant; delete. The step-back sentence "which is weaker support ... than the original step-back evidence" is awkward: "although this is weaker evidence for knowledge-intensive reasoning than the original step-back results".
- Sec. 1 first sentence is aphoristic and uncited ("the specification, the interface and the test fixture ... cannot be checked by a compiler"). Either cite or simplify.
- Sec. 4.C: "Everything is served with vLLM ... under WSL2 on one 16 GB RTX 5000 Ada GPU" followed by a long parenthetical chain. Split into two sentences.
- Sec. 5.2 first paragraph: the 4.3 distinct candidates sentence has three clauses in one sentence; split.
- Uncited claims: "unlike source code it cannot be checked by a compiler" (Sec. 1); "Non-expert authors struggle..." (fine, [12]); "it scales but may forfeit the structure" ([19] is a survey, see M1f); the "standard in APO loops" claim (Sec. 3.B) needs a citation (e.g., OPRO [17], APE [14]).

**S12. Threats to validity.**
Add:
- The base-prompt population is partly degenerate (M6c).
- MPIR's refine prompt does not protect the answer format (M4a).
- All 45 cells use one target model; Tier 2 has seed 42 only and 1 MPIR seed.
- Rubric authorship overlap with the evaluation domain (S7).
- Strict extraction is the gate metric and the primary metric.
- The tasks in Tier 2 are not representative of BBH's free-form tasks.

---------------------------------------------------------------------

## D. OPTIONAL

**O1.** Keywords: not alphabetical and inconsistently capitalised ("Big-Bench Hard, Prompt engineering, automatic prompt optimization, large language models, ..."). Use IEEE Access style, alphabetical, e.g. "Automatic prompt optimization, Big-Bench Hard, large language models, meta-prompting, negative results, prompt engineering".

**O2.** Front matter vs back matter: Funding appears twice (front-page footnote and Sec. 8). Numbered Sections 8-12 (Funding, Ethics, Competing interest, Data availability, Author contributions) are non-standard for IEEE Access; make them unnumbered. Author biographies are present without photographs; IEEE Access asks for photographs, so add them at camera-ready.

**O3.** Mention of the desk-rejection by "another journal" is not needed to evaluate the work and may prejudice an editor. Consider: "An earlier version of this study was submitted elsewhere and not reviewed externally". At minimum use neutral wording.

**O4.** Related work could add a short sentence on recent negative or sceptical results about prompt optimizers and on instruction-following of small models as refiners, to position the negative result (the current Sec. 2.D is about reporting practice, not about prior sceptical findings). Add MIPROv2 or similar as an additional optimizer family.

**O5.** Sec. 3.A: the formalisation defines R(P0) as refinements of P0, which does not describe chained mode. Define the candidate set generically (candidates generated by the loop), and write the tie rule once.

**O6.** Sec. 4.B gives code-level detail (random.Random(seed).sample); keep it, but put the code detail in a footnote or appendix.

**O7.** Report GPU hours and wall-clock cost for the grid and for the 235B/Luna runs, as the budget is part of the reporting guidance the paper cites ([63]).

**O8.** In Appendix A, the prompts are said to be "exactly as saved" yet are "abridged". Say "excerpted", and point to the files in the repository.

---------------------------------------------------------------------

## E. Proposed sentence-level rewrites (in addition to the items above)

1. Sec. 1 (RQ summary):
   - Original: "the gate is not the cause (RQ2); a much larger refiner can help in some cells, but where it does the gains coincide with a repaired answer format (RQ3); and the rubric score is only weakly related to accuracy (RQ4)."
   - Proposed: "ungated candidates are worse than P0 and validation accuracy tracks test accuracy, so the gate is not the cause (RQ2); one of two much larger refiners won in several cells, but the gains coincide with a repaired answer format, which one exemplar also provides (RQ3); and the rubric score is only weakly related to accuracy (RQ4)."
2. Sec. 3.B:
   - Original: "which is the defect Section 5.8 quantifies."
   - Proposed: "which is the defect quantified in Section 5.9."
3. Sec. 5.5 last sentences:
   - Original: "We therefore cannot say that unstructured rewriting is worse than MPIR; with this refiner neither helps, and the gate is what protects against the harm of the ungated versions."
   - Proposed: "The rubric-free gated rewrite did at least as well as the rubric-guided variants (+0.7 against -0.5 and +0.0 points), so with this refiner the rubric adds nothing detectable; what protects against harm is the gate, not the rubric."
4. Sec. 5.6 last sentence:
   - Original: "The data therefore do not show that the rubric's other criteria add anything beyond a fix for output format."
   - Proposed: "The gains are therefore consistent with a single rubric criterion (output format with examples); the data do not show that the other six criteria contribute, because no ablation was run."
5. Sec. 6.A:
   - Original: "a rubric-free gated rewrite does no better (Section 5.5)."
   - Proposed: "a rubric-free gated rewrite does at least as well (Section 5.5)."
6. Sec. 6.B (ii):
   - Original: "the uncorrected loop lowered accuracy by about 14 points on average."
   - Proposed: "with a 1.7B refiner, the uncorrected loop lowered accuracy by about 14 points on average (nine cells, one MPIR seed)."
7. Conclusion:
   - Original: "The uncorrected loop of the first evaluation would have reduced accuracy, ..."
   - Proposed: "With a 1.7B refiner the uncorrected loop would have reduced accuracy by about 14 points, although its effect with the original GPT-4o refiner remains unknown; strict answer extraction accounts for much of the apparent weakness of exemplar-free baselines; and identical prompts do not score identically on this serving stack."
8. Sec. 5.1:
   - Original: "Accuracy differences smaller than this cannot be attributed to a prompt."
   - Proposed: "Accuracy differences of this size are not reliably attributable to a prompt on this serving stack."
9. Sec. 6.C.1:
   - Original: "repeated scoring of one prompt moves accuracy by 3.3 points on average"
   - Proposed: "repeated scoring of one prompt spans 3.3 points on average (SD 1.4)".
10. Sec. 4.A:
    - Original: "The first pilot's go/no-go rule was written in the project's planning document ... and the pilot was not pre-registered in the strict sense in which we used that word in the first submission."
    - Proposed: "The pilot's go/no-go rule was written in the project's planning document (REBUILD.md, commit 7612aa0, 2026-09-03) before the pilot ran. That is a time-stamped plan in a version-controlled repository, not an entry in a public registry, so we describe the study as pre-specified, not pre-registered."
11. Appendix D:
    - Original: "It is reproduced here only because the manuscript's framing referred to it, and Reviewer comments asked for its numbers."
    - Proposed: "It is reproduced here for completeness because the Introduction refers to it."

---------------------------------------------------------------------

## F. Overall assessment

The revision is a large improvement in honesty and design. It repairs the loop, logs every round, reports composition, adds an oracle analysis, a noise floor, MDE, a judge validation and an explicit statement of what the evidence does not show. The negative result for the 1.7B self-refiner is credible and sufficiently supported, with the caveat that it is largely a result about a small model's instruction following and format compliance (60% of rounds never validate; many candidates score 0.00 through lost answer tags), and that a large part of the base-prompt population is not a genuinely optimized prompt.

The remaining problems fall into four groups:

1. Mechanical but pervasive. Wrong citation numbers (Table 1, BBH), a missing BBH reference, "UNVERIFIED" notes in the reference list, wrong figure and section cross-references, appendix and table numbering, and one empty subsection (5.6.1).
2. A handful of numeric inconsistencies that a careful reader will find: Table 9 vs Table 6/7 for the V2 row, the V0 "byte-identical" claim, the free-rewrite deltas against Tables 4/5, and 211 vs 223.
3. The capability-gap evidence. It was re-specified after the fact (8B to 235B/Luna), is shown on 9 cells and one seed, is confounded by model family, and its format-repair interpretation is demonstrated for one of four winning cells and contradicted by Table 4 for the PromptWizard/hyperbaton win. The discussion also treats the "format" explanation as exonerating the rubric, although output format with examples is one of the rubric's seven criteria.
4. Residual over-claims (14-point figure for the "first evaluation", "outperformed any refinement") and review-response language in a standalone paper.

Recommendation: **Major revision**. Not because the core result is doubtful, but because the fixes in M1 to M9 change tables, claims and references, and two of them need new analyses (lenient re-scoring of the oracle candidates and of the 235B finals; an explicit D2 reporting for 4B/8B or a re-specification note). All are feasible with existing saved outputs. If M1, M5, M7, M9 are done and M2, M3, M4, M6 are handled by rewording plus the two cheap re-scorings, it would be acceptable after a minor check-up round.
