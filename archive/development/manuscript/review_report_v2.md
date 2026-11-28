# Multi-Perspective Peer Review Report (Round 2 — Major-Revision Draft)

**Manuscript**: "Meta-Prompted Instruction Refinement: Bridging Manual Prompting Techniques and Automatic Prompt Optimization" (MPIR)
**Authors**: Linh Nguyen, Quang-Vinh Dang, Minh Ngoc Dinh, Thuy Nguyen
**Target Venue** (per task instructions): *AI Open* (Elsevier/KeAi), open-access, broad AI theory/applications scope, single-anonymized review, no strict page limit
**Source Material Reviewed**: `manuscript/manuscript_plaintext.txt` — plaintext dump beginning at Section 1 (Introduction) through the References. **No Title page, Abstract, or Keywords block is present in the reviewed file.** Every reviewer below explicitly flags this rather than fabricating an assessment of unseen text; this is treated as an open item in the Editorial Decision.
**Review Mode**: Full review (Field Analysis → 5 independent reviews → Editorial Synthesis)
**Review Date**: 2026-08-24
**Prior Round**: A prior review (`manuscript/review_report.md`) evaluated an earlier, ~5,500-word draft targeting a different venue (*Artificial Intelligence and Applications*) with a strict page limit. That review is background only; every finding below was independently re-derived from the current ~14,500-word draft and is not assumed to carry over.

---

# Phase 0: Field Analysis Report

## Paper Basic Information
- **Title**: Meta-Prompted Instruction Refinement: Bridging Manual Prompting Techniques and Automatic Prompt Optimization
- **Abstract**: not present in the reviewed source file (see note above)
- **Full text length**: ~14,500 words / ~30 pages, including a 5-appendix supplement (A–E) and 62 references
- **Number of references**: 62 (up from 49 in the prior round)

## Field Analysis

| Dimension | Analysis Result |
|-----------|----------------|
| Primary Discipline | Artificial Intelligence / NLP — prompt engineering and automatic prompt optimization (APO) for large language models |
| Secondary Disciplines | ML evaluation methodology & statistics (paired significance testing for benchmark comparisons, now substantially expanded); applied ML systems engineering (API cost/latency, reproducibility of closed-model pipelines); software-engineering framing of prompting as "programming without a compiler" (Section 2.1) |
| Research Paradigm | Quantitative / computational-experimental; framework-and-algorithm contribution paper with ablation-based component analysis |
| Methodology Type | Statistical modeling / ML benchmarking (Big-Bench Hard) with an algorithmic contribution (Algorithms 1–2), cross-framework and cross-model generalization tests, and three ablation studies |
| Target Journal Tier | Q2 — *AI Open* is a broad-scope, open-access, applications-oriented Elsevier/KeAi journal without the narrow specialist bar of an NLP-focused venue (ACL/EMNLP/TACL) or a top general ML venue (NeurIPS/ICML). The manuscript's own Section 5.5 closing paragraph is written explicitly for this fit ("in keeping with this journal's focus on actionable applications of AI research"), confirming the authors themselves are targeting AI Open's applications mandate. |
| Paper Maturity | Revised, near pre-submission. Full structural and administrative completeness (Funding, Ethical Statement, Competing Interest, Data Availability, Author Contribution, and a Generative-AI-use Declaration are all present), a public code repository, and a substantially expanded Threats-to-Validity apparatus. This reads as a considered revision responding to prior feedback, not a first draft. |

## Recommended Target Journals (Top 3)
1. **AI Open** (as targeted) — good scope fit given the explicit applications framing in Section 5.5 and the paper's practitioner-oriented Practical Implications subsection; open-access, broad-AI remit tolerates the NLP-specific machinery as long as it is made accessible to non-specialist readers (see R3 below on whether it currently is).
2. **Expert Systems with Applications** or **Knowledge-Based Systems** — plausible alternative applications-oriented venues with similar breadth and a stronger existing base of prompt-optimization papers.
3. **Findings of ACL/EMNLP** — the paper's closest peer venues by method (PromptWizard, ProTeGi, PE2, PROPEL, GEPA are all NLP-venue papers), but these would impose stricter statistical-significance and multi-seed evaluation norms than the current single-run design meets.

## Reviewer Configuration Cards

### Reviewer Configuration Card #1 — EIC
**Role**: Editor-in-Chief
**Identity Description**: Editor-in-Chief of a broad-scope, open-access AI journal in the mold of *AI Open*, with a research background in applied AI systems and an editorial mandate to serve a readership spanning AI subfields (not NLP specialists exclusively).
**Review Focus**:
1. Whether the paper's narrative (Introduction → Results → Revisiting Objectives → Practical Implications → Conclusion) is internally coherent and free of over-promising, especially given the length nearly doubled since the last round
2. Whether the three new/expanded reflective sections (5.4, 5.5, 7) add genuine depth or substantially repeat one another
3. Fit and accessibility for AI Open's broad readership versus a narrow NLP-conference audience
**Will particularly care about**: Whether a reader who reads only the Introduction, Section 5.4, and the Conclusion would come away with a calibrated (not inflated) impression of how strong the central empirical result is — and whether that impression is even assessable given the Abstract is missing from the reviewed materials.
**Possible blind spots**: Statistical technicalities (deferred to R1); literature-positioning accuracy (deferred to R2).

### Reviewer Configuration Card #2 — Peer Reviewer 1 (Methodology)
**Role**: Peer Reviewer 1
**Identity Description**: NLP/ML evaluation methodologist specializing in paired significance testing for benchmark comparisons (Wilcoxon, sign test, bootstrap resampling) and APA-style statistical reporting completeness, in the tradition of Demšar (2006), Dror et al. (2018), and Card et al. (2020) — all of which this manuscript itself now cites.
**Review Focus**:
1. Adequacy and internal consistency of the expanded statistical apparatus (bootstrap CI + three new paired tests) introduced since the last round
2. Numerical cross-consistency between Tables 3, 4, 7, and 8, and between reported summary statistics and what the underlying per-task numbers actually yield when recomputed
3. Reproducibility: single-run-per-condition design, Appendix E's protocol table, and whether statistical software/parameters are specified precisely enough to reproduce the reported test statistics
**Will particularly care about**: Whether the specific numbers reported in Section 5.2.1 (W = 83.0, p = 0.158; sign test p = 0.052/0.026; dz = 0.30) are independently reproducible from the paper's own Table 3, and whether they are used consistently every place they recur (Sections 5.2.1, 6.1, 7).
**Possible blind spots**: Literature-positioning accuracy (R2); practitioner cost/accessibility framing (R3).

### Reviewer Configuration Card #3 — Peer Reviewer 2 (Domain)
**Role**: Peer Reviewer 2
**Identity Description**: Senior researcher in prompt engineering and automatic prompt optimization, closely familiar with the specific ten-method lineage the new Related Work positioning table (Table 1) compares against: APE, ProTeGi, EvoPrompt, OPRO, PromptWizard, PE2, PROPEL, ETGPO, CFPO, and GEPA.
**Review Focus**:
1. Whether Table 1's characterization of each of the ten competing methods matches how those same methods are described in the surrounding Section 2.3–2.5 prose
2. Completeness and currency of the literature review given the doubled reference list (62 items, several 2025–2026)
3. Genuine incremental contribution of MPIR relative to its two named closest relatives, PE2 and PROPEL
**Will particularly care about**: Whether the positioning table is a reliable, internally consistent summary artifact or an independently-drifted restatement of claims made elsewhere in the text.
**Possible blind spots**: Statistical rigor of the reported gains (R1); practical deployment framing (R3).

### Reviewer Configuration Card #4 — Peer Reviewer 3 (Perspective)
**Role**: Peer Reviewer 3
**Identity Description**: Applied-AI / MLOps practitioner with production experience deploying LLM prompt pipelines, reading this manuscript specifically as a member of *AI Open*'s broad, applications-oriented, non-NLP-specialist readership rather than as an APO subfield insider.
**Review Focus**:
1. Practical feasibility, cost, and accessibility of the framework for a reader who does not already know the APE/ProTeGi/EvoPrompt/OPRO/PromptWizard lineage by heart
2. Whether the newly added "Beyond BBH" applied-domain extrapolation (customer support, tutoring, RAG assistants) in Section 5.5 is adequately grounded or reads as unsupported venue-fit signaling
3. Stakeholder perspective: what would a practitioner deciding whether to adopt MPIR need to know that the paper does not currently tell them
**Will particularly care about**: Whether the paper's claimed practical value (a lightweight, retrofittable, no-retraining layer) survives contact with a reader who is not fluent in the subfield's jargon and acronyms.
**Possible blind spots**: Fine-grained statistical methodology (R1); literature lineage accuracy (R2).

### Reviewer Configuration Card #5 — Devil's Advocate
**Role**: Devil's Advocate
**Identity Description**: Assigned to stress-test the paper's core claims and internal consistency independent of the other four reviewers.
**Review Focus**: (1) internal contradictions between prose and the newly added Related Work positioning table; (2) whether the paper's own "achieved / partially achieved / achieved within tested scope" self-scoring of its four research objectives (Section 5.4) is itself an even-handed reading of the evidence, or a self-serving frame; (3) the strongest available counter-argument to MPIR's contribution claim.
**Will particularly care about**: Whether the paper's transparency about its own statistical inconclusiveness is fully carried through into every section that touches the headline result, or whether it quietly slips in some sections.
**Possible blind spots**: N/A by design.

## Review Strategy Recommendation
This is a well-engineered, unusually self-critical revision — its Threats-to-Validity apparatus and refusal to claim conclusive significance are well above the field norm for APO papers. The central risk this round is not "does the paper overclaim" (on the visible evidence, largely no) but "does a near-doubling in length, driven substantially by new self-referential and comparative apparatus (Table 1, Section 5.4, Section 6, expanded Section 5.5), introduce internal-consistency defects that a rigorous audit — table arithmetic, cross-references, prose-vs-table agreement — would catch, and does that same growth read as depth or as restatement." Four of five reviewers therefore each perform an independent verification pass over a disjoint slice of the new material (statistics/tables, Table 1/literature, narrative structure/padding, internal logic), and the reports below are the product of that verification, not just qualitative impression.

---

# Phase 1: Independent Reviewer Reports

## Report 1 of 5 — EIC Review Report

### Manuscript Information
- **Title**: Meta-Prompted Instruction Refinement: Bridging Manual Prompting Techniques and Automatic Prompt Optimization
- **Review Round**: Round 2 (major-revision draft, new venue: AI Open)

### Reviewer Role
Editor-in-Chief

### Reviewer Identity
Editor-in-Chief of a broad-scope, open-access AI journal (AI Open profile), applied-AI-systems background.

### Review Focus
Journal fit and readership accessibility; narrative coherence from Introduction through Conclusion; whether the paper's near-doubling in length represents genuine new depth or restatement.

### Overall Recommendation
**Major Revision** (leaning toward Minor, but tipped to Major by an unresolved verification gap — see Weakness W3)

### Confidence Score
4 — high confidence within editorial competence; detailed statistics deferred to R1 and literature-lineage accuracy deferred to R2.

### Summary Assessment
MPIR layers a seven-criterion, rubric-guided meta-prompting refinement stage on top of an existing automatic prompt optimizer (PromptWizard, APE, ProTeGi), validated empirically on Big-Bench Hard across three APO backbones and two model families, with three ablations isolating the contribution of the rubric's specificity and its validation stage. Relative to the prior round, this draft has grown substantially: a ten-method Related Work positioning table, a full Threats-to-Validity section organized by internal/external/construct validity, a "Revisiting the Research Objectives" scorecard, an expanded Practical Implications subsection reaching beyond the benchmark, and three additional paired significance tests. The paper is unusually honest about its own central result being statistically inconclusive (bootstrap CI spanning zero, Wilcoxon p = 0.158), and carries that honesty consistently through Sections 5.2.1, 6.1, and 7 — a genuine strength relative to typical APO papers. The principal editorial concerns are (a) whether the added reflective material (Sections 5.4, 5.5, 7) is earning its length or restating the same three caveats three times, and (b) that the reviewed source material contains no Abstract, so the single most consequential place where over-claiming would occur cannot be checked at all in this round — an open risk given that the prior review round's chief complaint was exactly an abstract/conclusion framing gap.

### Strengths
1. **Consistent honesty about statistical inconclusiveness across the whole document, not just in one hedge**: Section 5.2.1 states the improvement is "directionally positive but not conclusively significant," Section 6.1 restates this as "the central internal-validity concern," and the Conclusion (Section 7) again states the gain "is modest and not conclusively significant by any single test." A paper that carries the same calibrated framing through three separate sections, rather than hedging once and then overclaiming in the summary, is doing something most APO papers in this space do not.
2. **Deliberate, well-motivated positioning against the "elaborate 2025–2026 frontier" (Section 2.5)**: rather than claiming to beat GEPA, ETGPO, or CFPO, the paper explicitly frames its contribution as "a different point on the cost-versus-sophistication trade-off, not a claim to surpass it." This is a mature, defensible way to position a modest-effect-size contribution against a rapidly moving frontier, and it is reinforced concretely in Section 5.5's practitioner guidance ("may be attractive in production settings where introducing a new training or search dependency is undesirable even when a more powerful but heavier method... might offer a larger expected gain").
3. **Genuinely broad generalization evidence for a single-paper contribution**: two additional APO backbones (Section 5.2.2), a second model family (Section 5.2.3), and three separate ablations (Section 5.3) is materially more robustness-checking than a single-baseline, single-model APO paper would provide, and directly supports the paper's own claim (Section 5.2.1) that "the more robust evidence for MPIR is qualitative and structural" rather than the pooled average alone.

### Weaknesses
1. **Title & Abstract cannot be assessed — flagged as a blocking gap in the review process, not a content defect**: The reviewed plaintext file begins at "1. Introduction" with no Title/Abstract/Keywords block. Per the peer-review template's requirement that this section be commented on, I explicitly decline to fabricate an assessment. **Given that the prior review round's central complaint was specifically an abstract/conclusion overclaiming gap, this is not a cosmetic omission — it is the one place in the paper where the entire statistical-honesty strength documented above (W1) could be undone, and it is currently unverifiable.** Severity: Major (procedural — must be supplied before this decision can be finalized).
2. **Redundant restatement across Sections 5.4, 5.5, and 7**: Section 5.4 ("Revisiting the Research Objectives") restates, objective-by-objective, that the pooled improvement is "not conclusively significant" (Objective 2) and that generalization is "achieved within the tested scope" but does not "extend to benchmark families beyond BBH" (Objective 3). Section 6 (Threats to Validity) restates the identical two points at greater length under Internal Validity and External Validity. The Conclusion (Section 7) then restates both points a third time ("the average gain... is modest and not conclusively significant... generalization beyond the three tested APO frameworks, two tested model families, and the BBH benchmark itself remains open"), together with the construct-dependence caveat that is *also* stated in Section 6.3. Three of the paper's core caveats — statistical inconclusiveness, external-validity boundary, construct dependence — are each stated in full three separate times (5.4, 6.x, 7) with no material addition of new evidence at each repetition. This reads as the doubling in length being driven partly by restating the same three qualifications in three different rhetorical registers (an objectives scorecard, a formal threats taxonomy, and a summary) rather than three independent contributions. **Suggestion**: collapse Section 5.4 into a short forward-pointing paragraph ("Section 6 revisits these limitations formally") rather than a full independent restatement, and trim the Conclusion's limitations paragraph to a one-sentence pointer to Section 6 rather than a third full restatement. Severity: Major.
3. **A concrete instance of this padding pattern, that also breaks a promise to the reader**: Section 5.5's closing paragraph states that customer-support/helpdesk deflection, educational tutoring, and retrieval-augmented enterprise assistants are "plausible directions for the applied evaluation that Section 7 lists as future work." Section 7's actual future-work paragraph, however, lists five different directions (repeated-trial evaluation, independent construct validation, adaptive/instance-specific rubrics, frontier-model/BIG-Bench-Extra-Hard evaluation, and combination with complementary methods) and does **not** mention customer support, tutoring, or RAG assistants anywhere. A reader who follows the explicit pointer from 5.5 to 7 expecting to find these three applied domains listed as future work will not find them. This is a small but concrete instance of new content (the "Beyond BBH" paragraph) being added without being reconciled against the section it explicitly references. Severity: Minor-to-Major (easy fix: either add these three domains to Section 7's future-work list, or soften 5.5's wording to stop claiming Section 7 lists them).

### Detailed Comments

#### Journal Fit
Reasonably strong fit for AI Open specifically: Section 5.5's final paragraph is explicitly written for a broad-applications venue ("in keeping with this journal's focus on actionable applications of AI research"), and the paper's overall framing — a lightweight, retrofittable layer rather than a from-scratch method — suits a practitioner-facing, open-access outlet better than a narrow NLP-theory venue. See R3 below on whether the body of the paper (Sections 2.3–2.5 especially) is written at a level of subfield-specific density that undercuts this intended broad accessibility.

#### Originality
Modest but genuine and honestly characterized: the paper does not claim to out-perform the "elaborate 2025-2026" frontier (Section 2.5) and explicitly frames its contribution as isolating how much benefit a simple, fixed, interpretable rubric recovers relative to heavier learned/adaptive approaches. This is an appropriately scoped originality claim for the evidence presented.

#### Significance
The practical framing (Section 5.5) targets a real, well-articulated decision point for practitioners (whether to add a cheap post-hoc layer on an existing APO pipeline), which is a genuine and useful contribution independent of the modest effect size, provided the redundancy noted in W2 is trimmed so this guidance is not diluted by being said three times in slightly different words.

#### Structural Coherence
Sections 1–4 are tightly and logically sequenced. The breakdown occurs in the back half: 5.4, 6, and 7 are three different formal devices (a scorecard, a taxonomy, a summary) applied to what is substantially the same set of three caveats, discussed at W2 above.

#### Title & Abstract
Not assessable — not present in the reviewed materials (see W1).

#### Conclusion
Appropriately hedged in tone and consistent with Section 5.2.1's framing (see S1); its content, however, substantially duplicates Sections 5.4 and 6 (see W2).

### Questions for Authors
1. Can you supply the manuscript's actual Title, Abstract, and Keywords for this review round? Given the prior round's central complaint concerned exactly this section, the decision below cannot be treated as final on the "does the paper overclaim" question until this is checked.
2. Was Section 5.4 ("Revisiting the Research Objectives") added specifically in response to a prior reviewer's request for such a section, or was it authors' own addition? If the former, would a condensed, cross-referencing version (rather than a full independent restatement) satisfy the same underlying request?
3. Is the "Beyond BBH" applied-domain paragraph in Section 5.5 intended to be added to Section 7's future-work list, or was the cross-reference simply not updated when Section 5.5 was expanded?

### Minor Issues
- Section 5.5's four numbered pieces of practitioner guidance restate, in "advice" framing, findings already reported in Sections 5.2.3, 5.2.4, and 4.5 respectively (near-ceiling diminishing returns; structured-task effectiveness; per-task API-call overhead) without adding new analysis; consider tightening these to genuinely new synthesis (e.g., a decision flowchart or a single consolidated table) rather than prose recapitulation.

### Recommendation to Peer Reviewers
R1: please verify the specific paired-test statistics in Section 5.2.1 against Table 3's own numbers — I did not attempt this myself. R2: please check whether Table 1's characterization of the ten compared methods is internally consistent with how those methods are described in the surrounding prose, since I noticed at least one candidate contradiction (PE2's "held-out validation" property) in passing but did not verify it in depth — this is squarely your lane.

### Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 68 | Strong | Modest but honestly scoped contribution |
| Methodological Rigor (25%) | 63 | Adequate | Deferring detail to R1; docked for the unverifiable Abstract/Conclusion consistency gap |
| Evidence Sufficiency (25%) | 60 | Adequate | Broad generalization evidence, but central pooled result is statistically inconclusive |
| Argument Coherence (15%) | 58 | Weak-to-Adequate | Redundant restatement across 5.4/6/7 and the broken 5.5→7 forward reference |
| Writing Quality (15%) | 72 | Strong | Clear, professional prose throughout; the redundancy is structural, not sentence-level |
| **Weighted Average** | **63.9** | **Major Revision** | |

---

## Report 2 of 5 — Methodology Review Report (Peer Reviewer 1)

### Manuscript Information
- **Title**: Meta-Prompted Instruction Refinement: Bridging Manual Prompting Techniques and Automatic Prompt Optimization
- **Review Round**: Round 2

### Reviewer Role
Peer Reviewer 1 — Methodology

### Reviewer Identity
NLP/ML evaluation methodologist, paired significance testing and benchmark statistics reporting specialist.

### Review Focus
Statistical reporting adequacy of the newly expanded significance-testing apparatus; numerical cross-consistency across Tables 3, 4, 7, and 8; reproducibility of the reported test statistics from the paper's own data.

### Overall Recommendation
**Major Revision**

### Confidence Score
5 — this is squarely my area of expertise, and every finding below is backed by an independent recomputation from the manuscript's own tables rather than surface-level reading.

### Summary Assessment
This revision meaningfully strengthens its statistical treatment relative to a bootstrap-CI-only analysis: a paired Wilcoxon signed-rank test, an exact sign test (both two- and one-sided), and Cohen's dz now supplement the original CI, and the authors correctly justify this design choice by citing the paired, matched-task nature of the comparison (Demšar, 2006; Dror et al., 2018) and typical statistical power at this task count (Card et al., 2020). I independently recomputed the win/loss/tie count, the sign-test symmetry, and the Wilcoxon statistic directly from Table 3's own per-task numbers. The win/loss/tie count (16/6/1) and the one-sided-equals-half-of-two-sided sign-test relationship both check out exactly. My independent recomputation of the Wilcoxon statistic, however, yields W = 82.5 rather than the reported W = 83.0 — a small (0.5) but real discrepancy that the authors should verify and, ideally, resolve by reporting exact test parameters (software, version, tie-handling convention). Separately, I found three internal numerical inconsistencies across Tables 3/4/7/8 that a careful copy-edit pass introducing this many new tables should have caught, and one Cohen's dz interpretation that is slightly generous relative to the paper's own cited convention. None of these individually undermines the paper's central (already-hedged) claim, but their volume, concentrated entirely in newly added material, is enough to warrant a full numerical-consistency audit before acceptance.

### Strengths
1. **The paired-test justification is textbook-correct and well-cited**: choosing Wilcoxon signed-rank, an exact sign test, and Cohen's dz for 23 matched task-level pairs, and explicitly citing Demšar (2006) and Dror et al. (2018) for why paired tests are the right choice for this design, is exactly the right methodological move, and is executed with appropriate humility ("We report all three rather than selecting the most favorable one," Section 5.2.1).
2. **The free-rewrite and generic-rubric controls correctly rule out the most obvious confound**: Section 4.3.1's free-rewrite baseline (57.15% average, below both PromptWizard and MPIR) and Section 5.3.2's generic-rubric ablation (72.2% vs. 79.9% on the same five tasks) together pre-empt the natural "isn't this just GPT-4o's general rewriting/self-preference bias" objection before a reader can raise it — this is good methodological foresight, not an afterthought.
3. **Verified correct**: the reported win/loss/tie split of "16 wins, 6 losses, 1 tie" (Section 5.2.1) matches exactly what I obtain by comparing every one of the 23 PromptWizard/MPIR pairs in Table 3 task-by-task. I also verified that the reported one-sided sign-test p-value (0.026) is exactly half the reported two-sided value (0.052), which is the correct symmetric relationship for this test and a useful internal consistency check that holds.

### Weaknesses

#### W1: Independent recomputation of the Wilcoxon statistic does not match the reported value
**Problem**: Using Table 3's own PromptWizard and MPIR columns, I computed all 23 signed differences, dropped the one true tie (formal_fallacies: 53.3 − 53.3 = 0), ranked the absolute values of the remaining 22 differences with standard mid-rank tie handling (three-way tie at |1.3| for word_sorting/+, salient_translation_error_detection/+, dyck_languages/−; two-way ties at |0.5| and |6.6|), and summed the ranks belonging to the 16 positive differences (which independently reproduces the reported 16-win count, S3 above). This yields W+ = 170.5 and W− = 82.5, so the reported Wilcoxon statistic (whichever of W+/W− the authors intend, conventionally the smaller) should be **82.5**, not the reported **83.0**.
**Why it matters**: A 0.5 discrepancy is small and very plausibly explained by a difference in tie-handling convention (e.g., `scipy.stats.wilcoxon`'s `zero_method` parameter — "wilcox," "pratt," or "zsplit" — or a continuity correction) rather than by an error in the underlying data; I want to be explicit that this is *not* an accusation that the reported non-significance is wrong (a difference of 0.5 in W will not move p from 0.158 to anywhere near significance). But for a statistic this central to the paper's headline framing, an unreconciled discrepancy between the reported value and what a reviewer obtains from the paper's own published table undermines exact reproducibility, which is precisely the property the paper's extensive Appendix E is trying to guarantee for everything else.
**Suggestion**: Report the exact software, version, and `zero_method`/correction settings used for the Wilcoxon test in Appendix E (Table 9 currently lists which tests were run but not their exact parameterization), and confirm the statistic against a from-scratch recomputation using the precise (possibly higher-precision, pre-rounding) per-task accuracy values rather than the two-decimal figures shown in Table 3.
**Severity**: Major (reproducibility of the paper's own headline statistic).

#### W2: Table 7's "Average" row is inconsistent with Table 8's "MPIR (full)" row for the identical five underlying task values
**Problem**: Table 7's "ALL" column reports per-task values of 84.0, 82.6, 72.0, 92.4, and 68.4 for hyperbaton, penguins_in_a_table, ruin_names, object_counting, and reasoning_about_colored_objects respectively, with a stated column average of **80.0**, repeated in prose ("reduces average accuracy from the full rubric's 80.0%..."). Table 8's "MPIR (full)" column reports the *identical* five values for the *identical* five tasks, but its column average is reported as **79.9**. Recomputing the mean of 84.0, 82.6, 72.0, 92.4, 68.4 directly gives 399.4 / 5 = 79.88, which rounds to 79.9 — matching Table 8, not Table 7. The same pattern (Table 7's per-criterion averages running ~0.3–0.4 points higher than a direct recomputation from the table's own displayed values) recurs for every column in Table 7 (e.g., C1's five values 56.0, 76.0, 52.4, 90.2, 59.1 average to 66.74, reported as 67.0; C4's five values average to 65.56, reported as 66.0).
**Why it matters**: This is very likely explained by Table 7's averages having been computed from higher-precision, pre-rounding per-task numbers that were then rounded differently for display than Table 8's own display — a completely defensible practice, but one that is not stated anywhere, and that produces a directly checkable contradiction (80.0 vs. 79.9) between two tables reporting the literal same five numbers for the literal same five tasks. A reader auditing the ablation claim in Section 5.3.1 ("reduces average accuracy from the full rubric's 80.0%...") will find Table 8 fifteen lines later reporting a different average for what is presented as the same baseline.
**Suggestion**: Either (a) recompute Table 7's "Average" row directly from its own displayed per-task values (which would also shift its other column averages down by 0.1–0.4 each, slightly changing the magnitude but not the ranking of the ablation drops reported in Section 5.3.1), or (b) add a footnote to both tables stating that averages are computed from unrounded underlying data and may not exactly match a manual average of the displayed, rounded per-task figures.
**Severity**: Major (this table is the empirical basis for the paper's central ablation claim about which rubric criteria matter most).

#### W3: Table 3 and Table 4 report different accuracy values for the identical PromptWizard/MPIR condition on tracking_shuffled_objects
**Problem**: Table 3 reports MPIR's accuracy on tracking_shuffled_objects as **65.2%**. Table 4's "PromptWizard (after)" column — which by construction should be the identical MPIR-on-PromptWizard condition reported in Table 3's "MPIR" column, and every one of the other 22 tasks matches between the two tables exactly to the reported decimal — reports **65.5%** for the same task under the same condition.
**Why it matters**: This is the only one of 23 rows where the two tables disagree, which makes it look like a transcription slip rather than a systematic reporting-convention difference (unlike W2, above, where the entire table is offset in one direction). It is a small (0.3-point) discrepancy that does not change any qualitative claim in the paper (both values still support the point made in Section 5.2.4 that tracking_shuffled_objects is a "consistently negative" task for MPIR), but for a paper whose central contribution rests on exact per-task accuracy comparisons, a value that changes between two of the paper's own tables should be reconciled.
**Suggestion**: Confirm the correct value from the underlying run logs referenced in the project repository (Section 4.5) and correct whichever table is wrong.
**Severity**: Minor.

#### W4: Cohen's dz interpretation is slightly generous relative to the paper's own cited convention
**Problem**: Section 5.2.1 describes the paired effect size as "small-to-moderate (Cohen's dz = 0.30)." By the standard Cohen benchmarks the paper's own Appendix reference framework would apply (0.2 = small, 0.5 = medium, 0.8 = large — the same convention the field generally uses and that this journal's statistical-reporting norms expect), 0.30 sits closer to "small" (0.10 above the small threshold, 0.20 below the medium threshold) than the midpoint framing "small-to-moderate" suggests.
**Why it matters**: This is a minor interpretive framing choice, not a computational error, but in a section whose entire rhetorical point is to avoid overselling an inconclusive result, describing a small effect as "small-to-moderate" pulls very slightly in the overselling direction that the rest of the section otherwise carefully avoids.
**Suggestion**: Either say "small" plainly, or report the effect size's own descriptive language alongside the numeric value without the "-to-moderate" qualifier.
**Severity**: Minor.

### Detailed Comments

#### Research Questions & Hypotheses
The problem formulation (Section 3.1) is precise and appropriately formal for an optimization framework paper; the argmax formulation over R(P0) is standard and clearly stated.

#### Research Design
The two-stage design (Stage 1 APO → Stage 2 MPIR) with a free-rewrite control and a generic-rubric control is well-constructed for isolating the rubric's specific contribution from general LLM rewriting capability (see S2).

#### Sampling Strategy
25 training/optimization examples per task, remainder held out for test — consistent with the original PromptWizard protocol and clearly reported in Table 2 and Appendix E's Table 9.

#### Data Collection
Fixed temperature-0 decoding, single run per condition — appropriately and explicitly flagged as a limitation in Section 6.1 rather than hidden, which I credit, but this remains the single largest internal-validity gap: with no repeated trials, the reported per-task differences (some as small as 0.4–0.9 points; see the differences underlying W1 above) cannot be distinguished from run-to-run noise at the level of an individual task, even though the pooled paired tests partially address this at the aggregate level.

#### Analysis Methods
The three-test battery (bootstrap CI, Wilcoxon, sign test) plus Cohen's dz is appropriate and non-redundant; see W1 and W4 above for specific issues within this otherwise sound approach.

#### Results Presentation
See W2 and W3: the volume of new tables (3 through 9) introduced or expanded this round has outpaced the cross-table consistency checking that would normally accompany a table this dense with reused/overlapping numbers.

#### Reproducibility
Appendix E (Table 9) is a genuinely useful consolidated protocol summary — exactly the right instinct for a reproducibility-conscious revision — but it currently omits the specific statistical software/library and parameterization needed to exactly reproduce the Section 5.2.1 test statistics (see W1). The public repository (Section 4.5) should resolve this if the exact analysis script is included, but the paper itself should state it.

#### Methodological Fallacies Detected
No fallacies rising to "unacceptable" were found. The single-run design (a form of insufficient replication rather than a classic fallacy) is the one item that would, under a stricter statistical venue, likely block publication outright; under this journal's evidence standards it is adequately handled by explicit disclosure (Section 6.1) rather than by correction, and I do not require new experiments before acceptance — see the Editorial Decision for how this is weighed.

### Questions for Authors
1. What exact software/library and version (e.g., `scipy.stats.wilcoxon`, R's `wilcox.test`) and tie-handling parameterization were used for the Wilcoxon test in Section 5.2.1? Please reconcile against the W = 82.5 I obtain from Table 3's own values (W1).
2. Were Table 7's and Table 8's "Average" rows computed from higher-precision per-task values than the ones displayed in the tables? If so, please state this explicitly as a footnote (W2).
3. Can you confirm which of Table 3 (65.2%) or Table 4 (65.5%) is the correct MPIR/PromptWizard-after accuracy for tracking_shuffled_objects (W3)?

### Minor Issues
- Mizrahi et al. (2024), "State of what art? A call for multi-prompt LLM evaluation," appears in the References list but is never cited in the body text — given its direct relevance to the single-run/prompt-sensitivity limitation discussed in Section 6.1, either cite it there or remove it.
- No a priori or post-hoc statistical power calculation is reported for the paired tests at n = 23 (Card et al., 2020 is cited qualitatively but not used quantitatively); this would strengthen, though is not strictly required for, the honest "not conclusively significant" framing already adopted.

### Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 65 | Strong | Not my primary focus; scored for completeness |
| Methodological Rigor (25%) | 58 | Weak-to-Adequate | Sound test selection, but multiple unreconciled numerical inconsistencies in newly added tables |
| Evidence Sufficiency (25%) | 60 | Adequate | Central result remains statistically inconclusive by the paper's own admission |
| Argument Coherence (15%) | 65 | Strong | Statistical narrative is internally consistent in its qualitative framing |
| Writing Quality (15%) | 75 | Strong | Statistical reporting is precise and legible |
| **Weighted Average** | **63.5** | **Major Revision** | |

---

## Report 3 of 5 — Domain Review Report (Peer Reviewer 2)

### Manuscript Information
- **Title**: Meta-Prompted Instruction Refinement: Bridging Manual Prompting Techniques and Automatic Prompt Optimization
- **Review Round**: Round 2

### Reviewer Role
Peer Reviewer 2 — Domain

### Reviewer Identity
Senior researcher in prompt engineering and automatic prompt optimization, familiar with the APE/ProTeGi/EvoPrompt/OPRO/PromptWizard/PE2/PROPEL/ETGPO/CFPO/GEPA lineage.

### Review Focus
Accuracy of the new Related Work positioning table (Table 1) against the surrounding prose; literature completeness given the expanded 62-item reference list; genuine incremental contribution relative to PE2 and PROPEL.

### Overall Recommendation
**Major Revision**

### Confidence Score
5 — this is precisely my area of expertise, and the central finding below is a directly quotable textual contradiction, not an inference.

### Summary Assessment
The paper's Related Work section (2.1–2.5) is thorough and well-organized, correctly distinguishing black-box APO methods (Section 2.3) from heuristic-guided meta-prompting methods (Section 2.4) from the newest 2025–2026 "frontier" methods (Section 2.5), and the new Table 1 is a genuinely valuable addition — a five-dimension structured comparison across ten methods is more rigorous than the prose-only related-work sections typical of this literature. However, I found a direct, quotable contradiction between Table 1's characterization of PE2 and how PE2 is described three paragraphs earlier in the same section, which is a significant problem precisely because this comparison table is the paper's chosen mechanism for establishing its differentiation from the closest prior work. I also found a completeness gap in how Section 2.5's five "frontier" methods map onto Table 1's ten rows, one uncited claim in an otherwise fully-cited list, and one orphaned reference. The genuine incremental contribution relative to PE2 and PROPEL is, on balance, real and clearly enough argued in prose (Section 2.4's closing paragraph) that this is a fixable presentation problem rather than a hollow-contribution problem.

### Strengths
1. **Table 1 is an ambitious and largely well-conceived positioning device**: comparing ten methods along five explicit, well-defined dimensions (explicit rubric, APO-agnostic layer, training-free, held-out validation, multi-round) is a more falsifiable and more useful form of related-work synthesis than a purely narrative account, and the accompanying prose ("No prior method combines all five properties... PROPEL is rubric-guided but bakes its principles into a single pass without a separate validation stage; ETGPO organizes refinement around failure categories...") correctly explains *why* the table's negative entries are negative rather than just asserting them.
2. **The PE2-vs-PROPEL differentiation argument in Section 2.4 is genuinely substantive, not just a table checkmark**: "PE2 is itself a complete, self-contained optimizer that formalizes one evaluation-refinement loop; it does not layer onto an already-optimized prompt from another APO system... PROPEL instead bakes a large set of expert-derived principles into a single refinement pass as implicit priors, rather than scoring a prompt against each principle explicitly and iterating multiple evaluation-refinement-validation cycles." This is a precise, mechanism-level differentiation rather than a "we do X and they don't" assertion, and it is the strongest evidence in the paper that MPIR's contribution is a genuine combination rather than a relabeling.
3. **Honest, well-placed acknowledgment of the closest concurrent competitor**: Section 2.4's final paragraph on Chakma et al. (2026) — a same-pattern two-stage optimizer-then-refine approach for a different task — is exactly the kind of transparent nearest-neighbor disclosure that strengthens rather than weakens an originality claim, and it correctly identifies both the shared premise and the differing mechanism and scope.

### Weaknesses

#### W1: Table 1's "Held-out validation" entry for PE2 directly contradicts the immediately preceding prose
**Problem**: Section 2.4 states explicitly: "PE2 is itself a complete, self-contained optimizer that formalizes one evaluation-refinement loop; it does not layer onto an already-optimized prompt from another APO system, **nor does it select among candidates using a held-out validation score**." Ten lines later, Table 1's row for PE2 lists "Held-out validation" as **Yes**.
**Why it matters**: This is not a subtle inference — it is a direct textual contradiction on the exact property (held-out validation) that the prose uses to differentiate PE2 from MPIR ("MPIR's contribution is precisely this combination — an explicit, criterion-by-criterion rubric... with iterative empirical validation deciding which candidate survives — rather than any single component in isolation"). If Table 1 is correct that PE2 already has held-out validation, then the prose's differentiation argument in the very same section is undermined for its most load-bearing property; if the prose is correct, Table 1 needs correction. Either way, a reader who reads Section 2.4 and Table 1 together — which is exactly how they are laid out, one immediately following the other — will notice the two disagree about one of the five explicit comparison dimensions for one of the two methods singled out as "closest relatives."
**Suggestion**: Re-check the PE2 paper (Ye et al., 2024) directly and correct whichever of the two statements is wrong; if PE2 does use some form of held-out evaluation but not in the specific sense the prose means (e.g., it may evaluate on a held-out batch during its beam search without a final independent validation split), clarify the distinction explicitly in both places so they no longer read as contradictory.
**Severity**: Major.

#### W2: Table 1 covers only three of Section 2.5's five discussed "frontier" methods, with no stated selection rule
**Problem**: Section 2.5's opening paragraph discusses five directions with citations: evolutionary search with reflection (Agrawal et al., 2026 — GEPA), reinforcement learning directly over edit actions (**no citation given**), multi-agent debate/tournament Elo ratings (Nair et al., 2025), error-taxonomy-guided refinement (Singh et al., 2026 — ETGPO), and prompt-format optimization (Liu et al., 2025 — CFPO). Table 1 includes rows for GEPA, ETGPO, and CFPO, but has no row for Nair et al. (2025)'s multi-agent/tournament approach, and the RL-over-edit-actions direction is both uncited and untabled.
**Why it matters**: Given that Table 1 explicitly bills itself as summarizing "where MPIR sits relative to the methods discussed above" (Section 2.5's introductory sentence to the table), a reader has no way to know whether Nair et al. (2025) was omitted deliberately (e.g., because it does not cleanly map onto the five comparison dimensions — plausible, since a tournament/Elo selection mechanism does not obviously have a single "held-out validation: yes/no" answer) or by oversight. The missing citation for "reinforcement learning directly over edit actions" is the only uncited claim in an otherwise fully-cited five-item list and should be corrected regardless of the table question.
**Suggestion**: Either add a Nair et al. (2025) row to Table 1 (with a note if its properties are hard to categorize along these five dimensions), or add a sentence explaining why it is discussed in prose but intentionally excluded from the table; separately, add the missing citation for the RL-over-edit-actions clause.
**Severity**: Major for the citation gap (straightforward factual completeness issue); Minor-to-Major for the table-coverage question (defensible either way, but currently unexplained).

#### W3: One reference is never cited in the body text
**Problem**: Mizrahi et al. (2024), "State of what art? A call for multi-prompt LLM evaluation," appears only in the References list.
**Why it matters**: Given this paper's own repeated engagement with prompt-sensitivity and single-run-evaluation concerns (Errica et al., 2025 is cited five times for exactly this theme), an orphaned reference on the identical theme suggests either a citation that was dropped during a revision pass (most likely candidate location: Section 6.1's single-run limitation, or the Introduction's opening paragraph on prompt sensitivity) or a reference that should be removed.
**Suggestion**: Cite it where relevant (Section 6.1 is the natural home) or remove it from the reference list.
**Severity**: Minor.

### Detailed Comments

#### Literature Review
- **Coverage**: Strong and current — the reference list runs through 2026 entries (Agrawal et al., 2026; Singh et al., 2026; Chakma et al., 2026; OpenAI, 2026; Google DeepMind, 2026), which satisfies the up-to-date-references expectation for a research-project manuscript at this stage.
- **Integration quality**: Section 2 is organized thematically (human-AI interaction framing → manual heuristics → black-box APO → heuristic-guided meta-prompting → 2025–2026 frontier) rather than as a flat list, which is good practice; the framing device in Section 2.1 (prompting as "programming without a compiler," heuristics as "an informal type system") is a nice unifying metaphor that is deployed once and then genuinely used again in Section 3 rather than being a one-off flourish.
- **Research gap argument**: Persuasive and specific — Section 2.4's closing paragraph ("Important gaps remain, however...") correctly identifies that PE2 and PROPEL each bake heuristic knowledge into the search process itself, "which limits how cleanly their heuristic contribution can be isolated from their search contribution" — this is exactly the right level of specificity for a research-gap claim.

#### Theoretical Framework
The seven-criteria rubric (Section 3.3) is grounded criterion-by-criterion in cited prior work (role prompting, step-back reasoning, chain-of-thought, positional bias for the Conclusion criterion), which is the correct standard for a rubric that claims to formalize existing heuristics rather than invent new ones.

#### Academic Argument Quality
- **Factual accuracy**: See W1 — the one clear factual/characterization error found.
- **Argument logic**: Sound elsewhere; the differentiation-from-PE2/PROPEL argument (S2) is logically well-formed independent of the table error.
- **Terminology precision**: Consistent and precise throughout; "APO-agnostic layer," "held-out validation," and "multi-round" are each used consistently in the same sense every time they appear (excepting the PE2 contradiction itself).

#### Contribution to the Field
- **Incremental contribution**: Real but modest, and honestly characterized as such — this is a strength, not a weakness, of the paper's self-presentation (see EIC's S1 and R1's S1).
- **Positioning**: Once W1 is corrected, the positioning relative to PE2 and PROPEL is clear and well-argued.
- **Overclaiming**: None detected in the body text; see EIC's W1 regarding the unavailable Abstract.

#### Missing Key References
- No additional missing seminal or 2025–2026 references were identified beyond the completeness gap noted in W2 (Nair et al., 2025's table coverage) and the citation gap in the same weakness (RL-over-edit-actions).

### Questions for Authors
1. Please re-verify PE2's (Ye et al., 2024) actual use of held-out validation directly against that paper, and reconcile Table 1 with Section 2.4's prose (W1).
2. Was Nair et al. (2025) intentionally excluded from Table 1, and if so, why (W2)?
3. What is the missing citation for "reinforcement learning directly over edit actions" in Section 2.5 (W2)?

### Minor Issues
- Citation format and author-year style throughout is consistent with Elsevier/KeAi natbib conventions; no format issues found. (Per review instructions, the presence of "et al." for large author-list works, e.g., Suzgun et al., 2023 and Agarwal et al., 2025, is expected AI Open/Elsevier style and is explicitly not flagged as a defect here.)

### Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 62 | Adequate | Modest, honestly-scoped incremental contribution |
| Methodological Rigor (25%) | 68 | Strong | Not my primary focus; scored for completeness |
| Evidence Sufficiency (25%) | 63 | Adequate | Deferring core evidence assessment to R1 |
| Argument Coherence (15%) | 54 | Weak | The PE2 table/prose contradiction directly undercuts the paper's central differentiation argument |
| Writing Quality (15%) | 74 | Strong | Clear, well-organized related-work narrative |
| Literature Integration (R2 focus) | 62 | Adequate | Strong currency and thematic organization, offset by W1–W3 |
| **Weighted Average** | **64.0** | **Major Revision** | |

---

## Report 4 of 5 — Perspective Review Report (Peer Reviewer 3)

### Manuscript Information
- **Title**: Meta-Prompted Instruction Refinement: Bridging Manual Prompting Techniques and Automatic Prompt Optimization
- **Review Round**: Round 2

### Reviewer Role
Peer Reviewer 3 — Perspective

### Reviewer Identity
Applied-AI / MLOps practitioner, reading as a representative of *AI Open*'s broad, non-NLP-specialist readership rather than as an APO subfield insider.

### Review Focus
Practical feasibility and accessibility for a reader outside the prompt-optimization subfield; grounding of the newly added "Beyond BBH" applied-domain discussion; stakeholder perspective of a practitioner deciding whether to adopt MPIR.

### Overall Recommendation
**Minor Revision**

### Confidence Score
4 — high confidence in the practitioner-facing assessment; I defer statistical and literature-lineage technicalities to R1 and R2 respectively (per Role Boundaries, I do not re-adjudicate their findings here even where I noticed candidates for them in passing).

### Summary Assessment
Read as a member of AI Open's broad AI-applications readership rather than as an APO-subfield insider, this paper offers a genuinely practitioner-relevant idea — a cheap, no-retraining, drop-in refinement layer for an existing prompt-optimization pipeline — argued with real attention to deployment realities (API call counts, cost/latency tradeoffs, retrofittability). Section 5.5's practical-implications guidance is concrete and decision-relevant in a way many APO papers are not. My chief concern, distinct from the statistical and literature-accuracy issues other reviewers are better positioned to catch, is that the paper's Related Work section (2.3–2.5) is written at a density of subfield-specific acronyms and methods (APE, ProTeGi, EvoPrompt, OPRO, PromptWizard, PE2, PROPEL, DSPy, TextGrad, Self-Refine, Reflexion, ETGPO, CFPO, GEPA — fourteen named systems in four pages) that assumes a reader already fluent in the 2023–2026 prompt-optimization literature, which is a mismatch for a broad-scope, open-access AI-applications journal whose readership will include roboticists, vision researchers, and applied-ML engineers who do not track this specific subfield closely. I also found the newly added "Beyond BBH" applied-domain paragraph in Section 5.5 to be more speculative than its placement suggests, which practitioners reading it as concrete guidance should be warned about more explicitly than the current wording does.

### Strengths
1. **The retrofit-without-modification framing is exactly the right pitch for a practitioner-facing venue**: Section 5.5's fourth guidance point — "because MPIR requires no retraining, gradient access, or bespoke search infrastructure, it is straightforward to retrofit onto an already-deployed APO pipeline without modifying that pipeline's own code" — is a concrete, actionable, and genuinely differentiating claim relative to heavier methods, and it is the single most useful sentence in the paper for a reader deciding whether to try this.
2. **Honest cost accounting rather than a vague "lightweight" claim**: Section 4.5 quantifies the overhead precisely ("MPIR adds on the order of 200 additional API calls per task on top of the underlying APO method's own optimization cost — a real overhead that should be weighed against its per-task accuracy gains in latency- or cost-sensitive deployments"). Giving an actual number, rather than asserting "efficient" without support, is exactly what a deploying engineer needs and most APO papers omit.
3. **The failure-mode analysis (Section 5.2.5) is genuinely useful decision-support, not just post-hoc rationalization**: explaining *why* Instruction & Separation helps hyperbaton but hurts logical_deduction and geometric_shapes (because in the latter two tasks, "the context being separated out often contains constraints the reasoning process must repeatedly refer back to") gives a practitioner an actual diagnostic heuristic for predicting whether their own task looks more like the first group or the second — this is more valuable than an aggregate accuracy number alone.

### Weaknesses

#### W1: The Related Work section's density of unexplained acronyms is a poor fit for AI Open's stated broad readership
**Problem**: Sections 2.3–2.5 introduce fourteen named systems (APE, ProTeGi, EvoPrompt, OPRO, PromptWizard, DSPy, TextGrad, Self-Refine, Reflexion, PE2, PROPEL, ETGPO, CFPO, GEPA) in roughly four pages, each described in enough mechanistic detail to satisfy an NLP-subfield reviewer (see R2's report) but without any single consolidating sentence for a reader who does not already know this lineage — e.g., no analogy, plain-language summary, or "if you only remember one distinction" framing is offered until Table 1's five abstract Yes/No/Partial columns, which themselves require having read the preceding prose to interpret.
**Why it matters**: AI Open's stated scope is broad AI theory and applications, not NLP-conference-adjacent prompt engineering specifically. A vision researcher or robotics practitioner picking up this paper because its practical framing (Section 5.5) looked relevant to their own APO-adjacent pipeline will hit a wall of unexplained acronyms in Section 2 before reaching that practical content. This is not a request to remove technical depth — R2's review confirms that depth is scientifically necessary and mostly well-executed — but to add a short accessibility on-ramp.
**Suggestion**: Add 2–3 sentences at the start of Section 2.3 (or as a callout) giving a single plain-language sentence per major category ("black-box search over prompt text," "meta-prompting that critiques then edits," "heuristic priors baked in before search") before the method-by-method detail, so a non-specialist reader can orient before the acronym density begins.
**Severity**: Minor-to-Major (does not block understanding for a specialist reader, but works against the paper's own explicit AI Open positioning).

#### W2: The "Beyond BBH" applied-domain paragraph (Section 5.5) is more speculative than its placement suggests
**Problem**: The paragraph proposing customer-support/helpdesk deflection, educational tutoring, and retrieval-augmented enterprise assistants as settings where "the underlying pattern MPIR targets... plausibly recurs" is placed immediately after four numbered, evidence-backed practitioner-guidance points (each explicitly tied to a specific results section), which primes the reader to expect similarly-grounded claims. The paragraph does appropriately hedge internally ("We have not evaluated MPIR in any of these settings"), but this hedge arrives only in the second-to-last sentence, after three specific, concretely-worded application scenarios have already been described in enough operational detail (e.g., "keep the static instruction distinguishable from retrieved content that changes per query") to read as informed recommendations rather than untested speculation.
**Why it matters**: A practitioner reader skimming for actionable guidance — exactly the reading mode this journal's audience is likely to adopt for a paper framed as "actionable applications of AI research" — may walk away believing these three domains have some empirical grounding they do not have. This is a much smaller version of the same over-claiming risk the EIC and Devil's Advocate reports discuss elsewhere for the core empirical result.
**Suggestion**: Move the "we have not evaluated MPIR in any of these settings" caveat to the *first* sentence of the paragraph rather than near the end, so the speculative framing is established before the specific scenarios are described, not after.
**Severity**: Minor.

#### W3: No discussion of what happens when a practitioner's target task has no natural "held-out validation" set
**Problem**: MPIR's validation stage (Section 3.5) assumes a labeled validation set with ground-truth answers is available to score candidate refinements — true for BBH's closed-form tasks, but not necessarily true for many of the exact applied domains Section 5.5 itself proposes (e.g., open-ended customer-support response quality, or tutoring-explanation quality) where "ground truth" is not a single closed-form answer.
**Why it matters**: This is a genuine, non-obvious practical limitation for exactly the extended applications the paper itself suggests exploring, and a practitioner reading Section 5.5's "Beyond BBH" paragraph would benefit from knowing this up front rather than discovering it only by trying to apply the method.
**Suggestion**: Add one sentence noting that MPIR's validation stage as described requires a closed-form or otherwise automatically-scorable ground truth, and that extending it to open-ended tasks (e.g., via an LLM-judge score in place of exact-match accuracy) is itself an open question, not an assumed drop-in replacement.
**Severity**: Minor.

### Detailed Comments

#### Assumption Audit
- **Explicit assumptions**: The paper is explicit that accuracy against a single ground-truth answer is its chosen construct (Section 4.4), and explicitly acknowledges this "cannot by itself capture the reasoning-clarity improvements documented qualitatively in Section 5.2.4" — a well-flagged explicit assumption.
- **Implicit assumptions**: The implicit assumption that a labeled, automatically-scorable validation set is available for any target task (W3 above) is not stated as an assumption anywhere, only implied by the method design.
- **Paradigmatic assumptions**: The paper's entire framing treats "prompt quality" as reducible to task accuracy on closed-form benchmark items; this is appropriate and stated as a scope choice (Construct Validity, Section 6.3) rather than smuggled in, which I credit.

#### Cross-Disciplinary Connections
- **Parallel research**: The "prompting as programming without a compiler" framing (Section 2.1) is a genuinely apt analogy to static-analysis and linting tooling in software engineering; the paper could go one step further and note that this analogy suggests a natural cross-disciplinary borrowing opportunity — formal program-analysis techniques (e.g., automatically checking a prompt for the seven rubric properties via a lightweight classifier rather than a full LLM call) as a cheaper alternative to MPIR's current two-LLM-call-per-round evaluation step, which would directly address the Section 4.5 cost concern from a different angle than the paper currently considers.
- **Methodological borrowing**: Software-engineering "linter" tooling (static rule-checkers) is the most directly relevant adjacent-field method not currently discussed as a possible complement or alternative to LLM-based rubric evaluation.

#### Practical Impact
See S1–S3 and W2–W3 above.

#### Broader Implications
- **Social impact**: Not materially implicated by this paper's scope (a prompt-refinement layer with no direct deployment-context claims beyond the speculative Section 5.5 paragraph); no equity or fairness concerns beyond what any accuracy-optimization paper would carry.

### Cross-Disciplinary Reading Recommendations
- Zamfirescu-Pereira et al. (2023) — already cited by the authors in Section 2.1 for the HCI framing of non-expert prompt authoring; worth citing again specifically in Section 5.5 if the paper is claiming MPIR reduces reliance on prompt-engineering expertise for practitioners, since this is the paper that most directly studies that exact claim empirically for human authors (a useful point of contrast: MPIR automates the refinement step, but the HCI literature suggests the harder problem for non-experts is often diagnosing *what* is wrong with a prompt in the first place, which MPIR's evaluation stage does for them — an angle the paper could make more of).
- Any standard software-engineering reference on static analysis / linting (not currently cited) — relevant to the cross-disciplinary borrowing opportunity noted above.

### Questions for Authors
1. Have you considered, or could you discuss, what happens to the validation stage when the target task lacks a closed-form ground truth (W3)?
2. Is there a lighter-weight (non-LLM) way to check some of the seven rubric criteria that could reduce the ~200-call-per-task overhead, analogous to how a linter is cheaper than a full code review?

### Minor Issues
- Consider a one-paragraph glossary or plain-language summary box near the start of Section 2 for readers unfamiliar with the APO acronym set (W1).

### Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 70 | Strong | Practically differentiated, honestly-scoped contribution |
| Methodological Rigor (25%) | 65 | Strong | Not my primary focus; scored for completeness |
| Evidence Sufficiency (25%) | 65 | Strong | Adequate for the practical claims actually made |
| Argument Coherence (15%) | 68 | Strong | Practitioner-facing argument is clear and well-supported |
| Writing Quality (15%) | 70 | Strong | Clear prose, though dense for a non-specialist reader in Section 2 |
| Significance & Impact (R3 focus) | 66 | Adequate | Real practical relevance, capped by the modest effect size and untested applied domains |
| **Weighted Average** | **67.2** | **Minor Revision** | |

---

## Report 5 of 5 — Devil's Advocate Review

### Manuscript Information
- **Title**: Meta-Prompted Instruction Refinement: Bridging Manual Prompting Techniques and Automatic Prompt Optimization
- **Review Round**: Round 2

Before the challenge: this paper does several things well that a purely adversarial reading should not obscure. It pre-empts the most obvious confound (GPT-4o self-preference bias) with a dedicated free-rewrite control; it reports a statistically inconclusive central result honestly rather than dressing it up; and its failure-mode analysis (Section 5.2.5) offers a genuine mechanistic account of *why* the method helps some tasks and hurts others, rather than stopping at aggregate numbers. The challenge below focuses on where the paper's self-assessment is more generous to itself than the evidence strictly supports, and on one internal contradiction with real stakes for the paper's differentiation claim.

### Strongest Counter-Argument

A skeptical reader could argue the following: MPIR's entire evidentiary case rests on the claim that its per-task, cross-framework, cross-model consistency is "the more robust evidence" (Section 5.2.1) even though the pooled statistical tests do not reach significance. But consistency-without-significance is exactly the pattern one would expect from a method whose true population-level effect is genuinely close to zero and whose apparent per-task "wins" are largely noise that happens to correlate loosely across three closely-related prompt-refinement procedures (PromptWizard, APE, ProTeGi) because those three procedures share substantial structural similarity (all three are LLM-driven, natural-language, iterative-search optimizers being refined by the same rubric applied by the same judge model, GPT-4o) — meaning the "three independent frameworks" evidence is less independent than it is presented as being. Under this reading, the 16/23 win count, the consistent direction across APO backbones, and even the qualitative failure-mode story in Section 5.2.5 are all consistent with "MPIR's rubric captures a handful of surface-level formatting fixes (adding delimiters, adding a role sentence, adding a closing restatement) that reliably help on tasks where the baseline prompt happened to be poorly formatted to begin with, regardless of which upstream optimizer produced it" — a much narrower and less interesting claim than "a structured heuristic rubric embeds human prompting expertise into APO." The paper's own ablation (Section 5.3.1) is actually consistent with this narrower reading: the two criteria that matter most, Instruction & Separation (C4) and Role Prompting (C1), are exactly the two "does the prompt have basic formatting hygiene" checks, while the criteria that plausibly reflect deeper reasoning guidance (Output Format with Examples, Conclusion) matter least. The paper does not itself distinguish between "MPIR teaches genuinely better reasoning structure" and "MPIR fixes basic formatting defects in whatever the upstream optimizer produced," and given that the two costliest-to-refute alternative explanations (self-preference bias, generic-rewrite capability) were tested and ruled out, this *third* alternative — that the benefit is real but is formatting-hygiene rather than heuristic-reasoning-transfer — was not tested and would require a different kind of ablation (e.g., an ablation restricted to only the two "formatting" criteria vs. only the "reasoning" criteria) to rule out.

### Issue List

#### CRITICAL
| # | Dimension | Issue Description | Location |
|---|-----------|-------------------|----------|
| — | — | No issue in this manuscript meets the CRITICAL bar (Foundation Collapse, Logic Chain Break, Data-Conclusion Mismatch, or Stronger Counter-Narrative that is *both* more parsimonious *and* better-fitting than the authors' own account). The Strongest Counter-Argument above is a genuine unaddressed alternative explanation, but it does not contradict the paper's data — it proposes a narrower interpretation of data the paper already reports accurately — and the paper's own hedged framing ("qualitative and structural... rather than an established, strongly significant improvement," Section 5.2.1) already leaves room for exactly this narrower reading rather than foreclosing it. This keeps it at MAJOR rather than CRITICAL. | — |

#### MAJOR
| # | Dimension | Issue Description | Location |
|---|-----------|-------------------|----------|
| 1 | Logical Consistency | Section 2.4's prose states PE2 "does not... select among candidates using a held-out validation score," while Table 1's row for PE2 lists "Held-out validation: Yes" — a direct contradiction on the exact property used to differentiate MPIR from its named closest relative. (Independently also flagged by R2 from the domain-accuracy angle; flagged here specifically as an internal logical contradiction between two co-located passages, which is squarely this report's mandate rather than a duplicate of R2's literature-accuracy framing.) | Section 2.4 (prose) vs. Table 1, row "PE2" |
| 2 | Evidence Gaps | The "formatting-hygiene vs. reasoning-transfer" alternative explanation (Strongest Counter-Argument, above) is not tested by any of the paper's three ablations, even though the ablation design (Section 5.3.1) already produces the criterion-level granularity that would be needed to test it (grouping C1/C4 vs. C2/C3/C6 and re-running the "generic rubric" style comparison in Section 5.3.2 using only one group). | Section 5.3.1–5.3.2 |
| 3 | Confirmation Bias Detection | Section 5.4 ("Revisiting the Research Objectives") is authored entirely by the paper's own authors as a self-scoring exercise, using categories the authors themselves selected ("achieved" / "partially achieved" / "achieved within tested scope") that are calibrated to sound successful even where the underlying finding is a statistically null result (Objective 2 is "partially achieved" rather than, say, "not achieved at the pooled level, achieved only qualitatively" — a starker but equally accurate framing the paper does not choose). This is not dishonest — the surrounding prose in the same bullet is appropriately precise — but the choice of label ("partially achieved") for a result whose defining statistic (bootstrap CI spanning zero) most readers would call "not achieved" at conventional standards is a framing choice that favors the authors. | Section 5.4, Objective 2 |

#### MINOR
| # | Dimension | Issue Description | Location |
|---|-----------|-------------------|----------|
| 1 | Overgeneralization Check | Section 5.5's claim that the "underlying pattern MPIR targets... plausibly recurs" in customer support, tutoring, and RAG assistant settings extrapolates from a benchmark (BBH) whose tasks are exclusively closed-form, single-correct-answer reasoning problems to domains (open-ended customer support responses, tutoring explanations) that are open-ended and multi-valid-answer by nature — a genuine scope extension the paper flags as unevaluated but still states with more specific operational detail than the "unevaluated" caveat comfortably supports (see also R3's W2, a related but distinct accessibility-framing concern). | Section 5.5, final paragraph |

### Ignored Alternative Explanations/Paths
1. **Formatting-hygiene explanation** (detailed in the Strongest Counter-Argument above): a more parsimonious account of *why* C1 (Role Prompting) and C4 (Instruction & Separation) dominate the ablation is that they are the two criteria most correlated with "the upstream optimizer produced a messy, undifferentiated prompt," which is a data-quality-of-the-baseline story rather than a heuristic-transfer story. The paper does not test this because doing so would require decomposing the seven-criteria rubric into hygiene-type vs. reasoning-type subsets and ablating them as groups, which its existing per-criterion ablation infrastructure (Table 7) could support with modest additional analysis.
2. **Shared-judge-model confound across the three "independent" APO frameworks**: MPIR's evaluation/refinement stage uses GPT-4o for all three APO backbones (PromptWizard, APE, ProTeGi) in Section 5.2.2's cross-framework test. The consistency of improvement across the three frameworks is presented as independent corroborating evidence (Section 5.2.1: "consistency... across three different APO frameworks"), but because the *same* judge model refines all three, a systematic GPT-4o-specific stylistic preference that happens to correlate with genuine accuracy gains on this particular benchmark family would produce exactly this same "consistent across three frameworks" signature without three truly independent tests of the rubric's value. The cross-model generalization test (Section 5.2.3, Gemini 3.5 Flash-Lite) partially — but does not fully — address this, since that test still uses the target/evaluator combination as a fixed pairing per condition rather than crossing judge models with target models factorially.

### Missing Stakeholder Perspectives
- End users of the downstream applications discussed in Section 5.5 (customers interacting with a support-deflection system, students in a tutoring system) whose direct experience of any accuracy change is not the same as an aggregate BBH-style accuracy number. (Elaborating on why this matters to these stakeholders specifically is R3's role, not mine; I flag only that their perspective is absent from the paper's evaluation of an area it explicitly proposes extending into.)

### Unexamined Premise
The paper's evaluation of "does MPIR work" is entirely mediated through a single scalar (exact-match accuracy on BBH), and every reflective section (5.4, 6.3, 7) that questions this premise questions only whether the *rubric* might be BBH-specific (construct dependence) — never whether *the accuracy metric itself*, applied consistently across all three ablations and both baselines, might be advantaging any method whose refinements happen to make delimiter-wrapped final-answer extraction more reliable (a parsing/extraction artifact) over one that improves actual reasoning quality without changing extraction reliability. Since Section 4.4 explicitly notes that predictions are "extracted programmatically... using the delimiter tags specified in every prompt variant," and Output Format with Examples/Conclusion (criteria plausibly most related to extraction reliability rather than reasoning quality) are explicitly noted as the two *least* load-bearing criteria in the ablation (Section 5.3.1), this particular confound is likely small in practice — but the paper never states that it checked for it, which is the actual gap: the possibility is not raised or dismissed anywhere in Sections 4.4, 5, or 6.

### Observations (Non-Defects)
- The paper's decision to report all three paired tests "rather than selecting the most favorable one" (Section 5.2.1) is exactly the behavior that makes adversarial review of this manuscript harder than it would otherwise be — this is a compliment, not a defect, and worth the Editorial Synthesizer weighing when calibrating how much additional scrutiny this paper's other claims warrant relative to a less transparent submission.
- The Chakma et al. (2026) disclosure (Section 2.4, final paragraph) is a voluntary disclosure of a very close concurrent work that the authors were not obligated to find or cite; this is the kind of behavior that should be rewarded in the review process rather than treated as neutral.

---

# Phase 2: Editorial Decision Package

## Part 1: Editorial Decision Letter

Dear Author(s),

Thank you for submitting your revised manuscript, "Meta-Prompted Instruction Refinement: Bridging Manual Prompting Techniques and Automatic Prompt Optimization," to *AI Open*. Your manuscript has been reviewed by five independent reviewers: the Editor-in-Chief and three peer reviewers covering methodology, domain/literature accuracy, and cross-disciplinary/practitioner perspective, plus a Devil's Advocate review tasked with stress-testing the paper's core claims and internal consistency.

### Decision: Major Revision

### Reviewer Summary

| Reviewer | Role | Recommendation | Confidence |
|----------|------|---------------|------------|
| EIC | Broad-scope AI journal editor | Major Revision | 4 |
| Reviewer 1 | Methodology (statistics/reproducibility) | Major Revision | 5 |
| Reviewer 2 | Domain (literature/positioning accuracy) | Major Revision | 5 |
| Reviewer 3 | Perspective (practitioner/accessibility) | Minor Revision | 4 |
| Devil's Advocate | Core-claim stress test | No CRITICAL finding; 3 MAJOR, 1 MINOR issue | — |

### Consensus Analysis

#### Points of Agreement (Consensus)

**[CONSENSUS-4]** (all four non-DA reviewers agree, corroborated independently by the Devil's Advocate on item 2):
1. **The manuscript is substantially more statistically and structurally rigorous than the prior round**, and its honesty about the central result's inconclusiveness (bootstrap CI spanning zero, non-significant Wilcoxon and borderline sign test) is carried consistently through the sections that discuss it (EIC-S1, R1-S1, and implicitly corroborated by DA's observation that this transparency "makes adversarial review harder").
2. **The paper does not require new experiments or new data collection to reach an acceptable state.** Every required revision identified below is a verification, correction, reconciliation, or condensation task against material the authors already have, not a request for additional benchmarking.

**[CONSENSUS-3]** (3 of 4 non-DA reviewers agree; R3 dissents on severity, not existence):
1. **The newly added reflective material (Sections 5.4, 5.5, 7) and the newly added comparison apparatus (Table 1, Tables 7/8/9) contain enough internal-consistency defects, clustered specifically in the material that is new since the prior round, to warrant a full consistency audit before acceptance** (EIC-W2/W3, R1-W1/W2/W3, R2-W1/W2). R3 does not dispute that these specific issues exist (R3's own report notes several adjacent framing concerns) but weighs them as addressable within a Minor Revision timeline given that R3's own primary concerns (accessibility, applied-domain grounding) are independently only Minor-severity. The Editor's resolution below explains why the combined weight of R1's and R2's findings, not R3's dissent on timeline, determines the final decision.

#### Points of Disagreement

**Disagreement 1: Overall revision magnitude (Minor vs. Major)**
- **R3's view**: Minor Revision — the issues R3 personally identified (Related Work accessibility, "Beyond BBH" paragraph hedging, validation-stage applicability) are all Minor-severity and resolvable with wording changes, and R3's weighted score (67.2) falls in the Minor Revision band.
- **EIC/R1/R2's view**: Major Revision — EIC (63.9), R1 (63.5), and R2 (64.0) all score just below the Minor/Major boundary, each independently driven by different clusters of Major-severity findings (EIC: the unverifiable Abstract combined with structural redundancy; R1: multiple unreconciled numerical inconsistencies including an independently-reproducible 0.5 discrepancy in the paper's own headline Wilcoxon statistic; R2: a direct prose/table contradiction on the paper's central differentiation claim).
- **Disagreement type**: Severity disagreement (all four reviewers found real, verifiable issues; they disagree only on aggregate severity).
- **Editor's Resolution**: **Major Revision.** Per the Confidence Score Weighting Rules, all four reviewers report confidence ≥4, so no reviewer's assessment is discounted for expertise mismatch; per the decision matrix (`editorial_decision_standards.md` §2), a Minor/Minor/Major/Major or stronger pattern maps to Major Revision, and this case is  Major/Major/Major/Minor — an even clearer case for Major than the matrix's threshold example. Three independent reviewers, using three disjoint verification methods (independent statistical recomputation, direct-quotation textual contradiction-finding, and structural redundancy mapping), each surfaced Major-severity, specific, corrected-in-principle-without-new-experiments findings. R3's dissent is preserved in full above and its underlying findings remain in the Revision Roadmap as Suggested (not Required) items, consistent with the Conservative Principle for unresolved disagreements.

**Disagreement 2: Whether the PE2 held-out-validation contradiction (R2-W1, corroborated as DA MAJOR-1) rises to a Devil's Advocate CRITICAL finding**
- **View A (a stricter reading)**: Because this contradiction sits inside the paper's own chosen mechanism for establishing its central differentiation claim ("MPIR's contribution is precisely this combination... rather than any single component in isolation," Section 2.4), and that claim is unresolved as stated, this could be read as a Logic Chain Break (DA CRITICAL criterion 2): the differentiation conclusion does not follow cleanly from evidence that contradicts itself.
- **View B (the Devil's Advocate's own assessment, adopted here)**: The contradiction is confined to a comparison-table characterization of a *third-party* method, not to any claim about MPIR's own empirical performance; correcting it (in either direction — confirming PE2 lacks held-out validation, or acknowledging it has some form of it and refining the distinction) leaves MPIR's own reported results, ablations, and hedged interpretation completely intact. It therefore does not meet the "core argument... cannot be rescued by revision" bar for CRITICAL.
- **Disagreement type**: Severity disagreement.
- **Editor's Resolution**: **MAJOR**, not CRITICAL, adopting the Devil's Advocate's own classification (View B) per the IRON RULE that a DA CRITICAL finding is dispositive when present — since the DA's own report classifies this as MAJOR rather than CRITICAL, the Accept-blocking IRON RULE (Checkpoint Rule #4) is not independently triggered by this item. It remains a Required Revision.

### Decision Rationale

Three of four peer reviewers, plus a Devil's Advocate review that found no CRITICAL issue but three independently-derived MAJOR issues, converge on Major Revision. This is not a paper with a fundamental design flaw: PromptWizard, MPIR, and their controls are correctly designed, the statistical apparatus is well-chosen for the paired, matched-task comparison it analyzes, and the paper's overall honesty about its own modest effect size is a genuine strength that all five reviewers independently credited. The Major Revision decision instead reflects a specific, convergent pattern across three independent verification lenses: a methodologist who recomputed the paper's own headline statistic and found a small but real discrepancy plus two cross-table numerical inconsistencies (R1); a domain expert who found a direct, quotable contradiction between the new Related Work positioning table and the prose describing the same method three sentences earlier (R2); and an editor who found that three of the manuscript's newly expanded sections restate the same three caveats without the growth in length corresponding to a growth in distinct content, compounded by the complete absence of the Abstract from the reviewed materials — precisely the section where the prior round's central complaint originated. None of these findings requires new experiments, new data collection, or a change to the paper's core empirical design; all are verification, correction, reconciliation, or condensation tasks. This profile — real, specific, convergent, but experimentally non-blocking issues — is the textbook case for Major rather than Minor Revision (issues serious enough to require a dedicated audit pass, but categorically different from issues requiring new data) and is categorically different from Reject (no reviewer identified a fundamental, unfixable flaw in the paper's core design or contribution).

### Summary of Key Issues
1. **Abstract/Title/Keywords not present in reviewed materials — must be supplied and checked against the hedged framing of Section 5.2.1/6.1/7 before final acceptance** (EIC-W1).
2. **PE2's "held-out validation" property is stated as absent in Section 2.4's prose and present in Table 1 — must be reconciled** (R2-W1; corroborated as DA MAJOR-1).
3. **Independent recomputation of the Section 5.2.1 Wilcoxon statistic from Table 3 yields W = 82.5, not the reported W = 83.0 — must be verified and either corrected or the exact software/parameterization documented** (R1-W1).
4. **Table 7's "ALL"/per-criterion averages (80.0, 67.0, 69.0, 66.0, 71.0, 70.0, 75.0) do not match direct recomputation from the table's own displayed per-task values, and disagree with Table 8's "MPIR (full)" row (79.9) for the identical five underlying numbers — must be reconciled or footnoted** (R1-W2).
5. **Table 3 and Table 4 report different MPIR/PromptWizard-after accuracy for tracking_shuffled_objects (65.2 vs. 65.5) — must be corrected** (R1-W3).
6. **Section 5.5 promises that Section 7 "lists" three specific applied domains as future work; Section 7's future-work paragraph does not mention them — must be reconciled** (EIC-W3).
7. **Sections 5.4, 6, and 7 restate the same three core caveats (statistical inconclusiveness, external-validity boundary, construct dependence) in full three separate times — should be condensed** (EIC-W2).
8. **Table 1 covers only three of the five "frontier" methods discussed in Section 2.5's prose (missing Nair et al., 2025), and one clause in that same prose ("reinforcement learning directly over edit actions") is uncited** (R2-W2).

---

## Part 2: Revision Roadmap

### Required Revisions (Must Fix)

| # | Revision Item | Source | Priority | Estimated Effort |
|---|--------------|--------|----------|-----------------|
| R1 | Supply the manuscript's actual Title/Abstract/Keywords for re-review; verify the Abstract's characterization of the central result matches the hedged framing already present in Section 5.2.1/6.1/7 | EIC | P1 | 0.5 day |
| R2 | Reconcile PE2's "held-out validation" property between Section 2.4 prose and Table 1 (re-check against Ye et al., 2024 directly) | R2 / DA | P1 | 1 day |
| R3 | Verify the Section 5.2.1 Wilcoxon statistic (W = 83.0) against an independent recomputation from Table 3; document exact software/version/tie-handling parameters in Appendix E, Table 9 | R1 | P1 | 0.5 day |
| R4 | Reconcile Table 7's "Average" row(s) with Table 8's "MPIR (full)" row for the identical five task values, or add an explicit footnote explaining the rounding/precision convention used | R1 | P1 | 0.5 day |
| R5 | Correct the tracking_shuffled_objects discrepancy between Table 3 (65.2%) and Table 4 (65.5%) | R1 | P1 | 0.25 day |
| R6 | Reconcile Section 5.5's forward reference ("plausible directions for the applied evaluation that Section 7 lists as future work") with Section 7's actual future-work content — either add these domains to Section 7 or soften the 5.5 wording | EIC | P1 | 0.25 day |
| R7 | Add the missing citation for "reinforcement learning directly over edit actions" (Section 2.5); either add a Nair et al. (2025) row to Table 1 or explicitly explain its exclusion | R2 | P2 | 0.5 day |

### Required Item Details

**R1: Supply and verify Abstract**
- **Problem**: No Title/Abstract/Keywords in the reviewed materials; the prior review round's central finding concerned exactly this section's framing.
- **Source**: EIC-W1.
- **Requirement**: Provide the current Abstract text for verification that it does not state the PromptWizard/MPIR comparison as an established, unqualified improvement, consistent with Section 5.2.1's own hedged language.
- **Acceptance criteria**: Abstract's characterization of the central result is verified consistent with Section 5.2.1's framing by the Editor or a re-review reviewer.

**R2: PE2 held-out-validation contradiction**
- **Problem**: Section 2.4 states PE2 does not use held-out validation for candidate selection; Table 1 states it does.
- **Source**: R2-W1 (Major), corroborated by DA (MAJOR-1).
- **Requirement**: Re-check Ye et al. (2024) directly and correct whichever statement is inaccurate; if the true situation is more nuanced than a binary Yes/No (e.g., PE2 validates within its own search loop but not against a truly independent held-out split), state that nuance explicitly in both the prose and a table footnote so the two no longer read as contradictory.
- **Acceptance criteria**: Section 2.4 prose and Table 1's PE2 row state the same fact about held-out validation, with any nuance made explicit rather than collapsed into a single ambiguous Yes/No cell.

**R3: Wilcoxon statistic verification**
- **Problem**: Independent recomputation from Table 3 yields W = 82.5, not the reported 83.0.
- **Source**: R1-W1 (Major).
- **Requirement**: Re-run the Wilcoxon signed-rank test on the exact per-task differences used for the original analysis (which may carry more decimal precision than Table 3 displays), report the exact software/library/version and tie-handling parameterization in Appendix E.
- **Acceptance criteria**: Reported W statistic is either confirmed correct with documented parameterization explaining the 0.5 discrepancy from a naive recomputation, or corrected.

**R4: Table 7 / Table 8 average reconciliation**
- **Problem**: Table 7's column averages (including the "ALL" average of 80.0) do not match direct recomputation from the table's own displayed values and disagree with Table 8's "MPIR (full)" row (79.9) for the identical five numbers.
- **Source**: R1-W2 (Major).
- **Requirement**: Recompute Table 7's averages directly from displayed values, or add a footnote to both Table 7 and Table 8 stating that averages are computed from higher-precision underlying data.
- **Acceptance criteria**: Table 7 and Table 8 report the same average for the same five underlying task values, or the discrepancy is explicitly and consistently footnoted in both places.

**R5: Table 3/Table 4 tracking_shuffled_objects discrepancy**
- **Problem**: 65.2% (Table 3) vs. 65.5% (Table 4) for the same reported quantity.
- **Source**: R1-W3 (Minor, but included here as a Required item given it is a direct data-value conflict rather than a rounding-convention question).
- **Requirement**: Confirm the correct value from underlying run logs and correct the erroneous table.
- **Acceptance criteria**: Table 3 and Table 4 report identical values for this task/condition.

**R6: Section 5.5 → Section 7 forward-reference reconciliation**
- **Problem**: Section 5.5 claims Section 7 lists customer-support, tutoring, and RAG-assistant domains as future work; it does not.
- **Source**: EIC-W3 (Minor-to-Major).
- **Requirement**: Either add these three domains to Section 7's future-work paragraph, or revise Section 5.5's wording to stop attributing them to Section 7.
- **Acceptance criteria**: The two sections no longer make an unfulfilled promise to the reader.

**R7: Section 2.5 citation and Table 1 coverage gaps**
- **Problem**: One uncited claim ("reinforcement learning directly over edit actions"); Table 1 omits Nair et al. (2025) despite it being discussed in the same paragraph as three methods that are tabled.
- **Source**: R2-W2 (Major for the citation gap; Minor-to-Major for the table-coverage question).
- **Requirement**: Add the missing citation; add a Nair et al. (2025) row to Table 1 or explain its exclusion.
- **Acceptance criteria**: Every claim in Section 2.5 is cited; Table 1's scope relative to Section 2.5's prose is either complete or explicitly and reasonably bounded.

### Suggested Revisions (Should Fix)

| # | Revision Item | Source | Priority | Expected Improvement |
|---|--------------|--------|----------|---------------------|
| S1 | Condense Sections 5.4/6/7's triple restatement of statistical inconclusiveness, external-validity boundary, and construct dependence into a single authoritative statement (in Section 6) with short cross-references elsewhere, rather than three full restatements | EIC | P2 | Removes padding, sharpens the paper's genuinely new content |
| S2 | Cite Mizrahi et al. (2024) where relevant (Section 6.1, single-run limitation) or remove it from the reference list | R1 / R2 | P2 | Closes an orphan-reference gap |
| S3 | Soften "small-to-moderate" framing of Cohen's dz = 0.30 to "small," consistent with the paper's own cited convention | R1 | P3 | Removes a small but real overclaiming drift in an otherwise carefully hedged section |
| S4 | Add a brief accessibility on-ramp (2-3 plain-language sentences) at the start of Section 2.3 for AI Open's broad, non-NLP-specialist readership | R3 | P2 | Improves fit with the target venue's stated broad scope |
| S5 | Move the "we have not evaluated MPIR in any of these settings" caveat to the start, not the end, of Section 5.5's "Beyond BBH" paragraph | R3 / DA-MINOR-1 | P2 | Reduces risk that speculative applied-domain claims read as grounded guidance |
| S6 | Add one sentence noting MPIR's validation stage assumes a closed-form, automatically-scorable ground truth, relevant to the open-ended applied domains Section 5.5 proposes | R3 | P2 | Pre-empts a practical limitation practitioners would otherwise discover only by trying the method |
| S7 | Consider an ablation grouping C1/C4 ("formatting hygiene" criteria) against C2/C3/C6 ("reasoning-guidance" criteria) to test the Devil's Advocate's alternative explanation directly | DA | P2/P3 | Would substantially strengthen the causal interpretation of the ablation results, though not required for this decision |
| S8 | Reconsider the "partially achieved" framing for Objective 2 in Section 5.4 given the underlying result is a null pooled-statistical finding | DA-MAJOR-3 | P2 | Improves the self-assessment's calibration/even-handedness |
| S9 | Report a priori/post-hoc statistical power for the paired significance tests at n=23 | R1 | P3 | Strengthens (already-honest) framing of statistical inconclusiveness |

### Revision Checklist

#### Priority 1 — Structural/Consistency Revisions (Estimated total effort: ~3-4 days)
- [ ] R1: Supply and verify Abstract text against Section 5.2.1/6.1/7 framing
- [ ] R2: Reconcile PE2 held-out-validation contradiction (Section 2.4 vs. Table 1)
- [ ] R3: Verify/correct Wilcoxon statistic; document exact test parameterization
- [ ] R4: Reconcile Table 7 vs. Table 8 average discrepancy
- [ ] R5: Correct Table 3 vs. Table 4 tracking_shuffled_objects value
- [ ] R6: Reconcile Section 5.5 → Section 7 forward reference
- [ ] R7: Add missing Section 2.5 citation; address Table 1's coverage of Nair et al. (2025)

#### Priority 2 — Content Supplementation and Condensation (Estimated total effort: ~2-3 days)
- [ ] S1: Condense triple-restated caveats across Sections 5.4/6/7
- [ ] S4: Add accessibility on-ramp to Section 2.3
- [ ] S5: Move "unevaluated" caveat to the start of Section 5.5's applied-domains paragraph
- [ ] S6: Add validation-stage applicability caveat
- [ ] S7: (Optional but recommended) formatting-hygiene vs. reasoning-guidance ablation
- [ ] S8: Reconsider "partially achieved" framing for Objective 2

#### Priority 3 — Text and Minor Polish (Estimated total effort: ~0.5-1 day)
- [ ] S2: Resolve orphan Mizrahi et al. (2024) citation
- [ ] S3: Soften Cohen's dz interpretive language
- [ ] S9: Report statistical power for paired tests

### Total Estimated Effort
- **Major Revision**: 6–8 days of focused author effort (no new experiments required); recommended window 4-6 weeks to allow for careful re-verification rather than rushed correction, given the nature of the required items is itself "verify carefully," not "write quickly."

### Revision Deadline
- **Recommended deadline**: 5 weeks from decision date.
- **Basis**: Major Revision standard (6-8 weeks) shortened somewhat because none of the required items involves new data collection or experiments — all are verification, correction, or reconciliation of material the authors already possess.
- **Extension policy**: Notify the Editor at least 1 week before the deadline if more time is needed, particularly if Abstract/Title materials require author-team coordination beyond the corresponding author.

### Response Letter Instructions
Please respond to every Required Revision (R1–R7) and every Suggested Revision (S1–S9) individually, following the R→A→C (Reviewer comment → Author response → Change description) format. For each numerical correction (R3, R4, R5), please state explicitly which value changed, from what, to what, and why, so the re-review can verify the correction directly against the revised tables without needing to redo the full independent recomputation performed in this round.

---

## Part 3: Reviewer Report Summary (Appendix)

### EIC Report Summary
- Recommendation: Major Revision | Confidence: 4
- Key Point: The paper's statistical honesty is consistent across sections, but the Abstract is unverifiable from the supplied materials and Sections 5.4/6/7 substantially restate the same three caveats three times.

### Reviewer 1 (Methodology) Summary
- Recommendation: Major Revision | Confidence: 5
- Key Point: The expanded statistical apparatus is well-chosen, but an independent recomputation of the Wilcoxon statistic (82.5 vs. reported 83.0) and two cross-table numerical inconsistencies (Table 7 vs. 8; Table 3 vs. 4) need reconciliation before the paper's numbers can be trusted at face value.

### Reviewer 2 (Domain) Summary
- Recommendation: Major Revision | Confidence: 5
- Key Point: Table 1's characterization of PE2's held-out-validation property directly contradicts the prose three sentences earlier describing the exact same method, undermining the paper's chosen mechanism for establishing differentiation from its closest prior work.

### Reviewer 3 (Perspective) Summary
- Recommendation: Minor Revision | Confidence: 4
- Key Point: The practical framing is genuinely useful and honestly cost-accounted, but the Related Work section's acronym density is a poor accessibility fit for AI Open's broad readership, and the "Beyond BBH" applied-domain paragraph reads as more grounded than it is until its hedge, which arrives too late in the paragraph.

### Devil's Advocate Summary
- No CRITICAL finding. 3 MAJOR issues (PE2 contradiction, corroborating R2; an untested "formatting-hygiene vs. reasoning-transfer" alternative explanation for the ablation results; a self-serving framing choice in Section 5.4's "partially achieved" label for a null pooled result), 1 MINOR issue (overgeneralization risk in the "Beyond BBH" paragraph), plus an Unexamined Premise regarding possible answer-extraction-reliability confounds in the accuracy metric that the paper does not address either way.
