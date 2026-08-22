# Multi-Perspective Peer Review Report

**Manuscript**: "Meta-Prompted Instruction Refinement: Bridging Manual Prompting Techniques and Automatic Prompt Optimization" (MPIR)
**Authors**: Linh Nguyen, Quang-Vinh Dang, Minh Ngoc Dinh, Thuy Nguyen
**Target Venue** (as stated in manuscript): *Artificial Intelligence and Applications* (AIA)
**Review Mode**: Full review (Field Analysis → 5 independent reviews → Editorial Synthesis)
**Review Date**: 2026-08-22

---

# Phase 0: Field Analysis Report

## Paper Basic Information
- **Title**: Meta-Prompted Instruction Refinement: Bridging Manual Prompting Techniques and Automatic Prompt Optimization
- **Abstract length**: ~180 words
- **Full text length**: ~5,500 words (approx. 12 journal pages, double-column, per stated constraint)
- **Number of references**: 49

## Field Analysis

| Dimension | Analysis Result |
|-----------|----------------|
| Primary Discipline | Artificial Intelligence / Natural Language Processing — specifically prompt engineering and automatic prompt optimization (APO) for large language models |
| Secondary Disciplines | ML evaluation methodology & statistics (significance testing for benchmark comparisons); Human-Computer Interaction (prompting as human-AI interaction, cognitive heuristics); Applied ML systems/engineering (cost, latency, reproducibility of API-based pipelines) |
| Research Paradigm | Quantitative / computational-experimental (benchmark evaluation + ablation studies); framework/algorithm contribution paper |
| Methodology Type | Statistical modeling / machine learning experimentation on a fixed benchmark (Big-Bench Hard), with algorithmic contribution (Algorithm 1) and ablation-based component analysis |
| Target Journal Tier | Q3 — *Artificial Intelligence and Applications* (Bon View Publishing) is a specialized, applications-oriented AI venue rather than a top-tier NLP/ML outlet (e.g., ACL, NeurIPS, TACL). Basis: journal scope description in the manuscript header, moderate reference-list prestige (mix of arXiv preprints and mid-tier venues alongside strong ACL/NeurIPS/ICLR entries), and the paper's own framing as an applied, practitioner-relevant contribution rather than a foundational theoretical one. |
| Paper Maturity | Pre-submission / finalized draft. The manuscript has complete structure, a DOI placeholder, full author/affiliation/ethics/funding boilerplate, and has clearly been condensed to a strict page limit (explicitly noted for Tables 2–3). This is a "camera-ready-adjacent" draft, not an early draft. |

## Recommended Target Journals (Top 3)
1. **Artificial Intelligence and Applications** (as submitted) — good scope fit; applied/practitioner framing matches the journal's applications focus.
2. **Findings of the ACL / EMNLP** — the work's closest peer venues (PromptWizard, ProTeGi, PE2, PROPEL, dual-phase optimization are all Findings papers); would face substantially higher rigor expectations on statistical significance and multi-seed evaluation.
3. **Natural Language Engineering** or **Expert Systems with Applications** — plausible alternative applications-oriented venues if AIA is not available.

## Reviewer Configuration Cards

### Reviewer Configuration Card #1 — EIC
**Role**: Editor-in-Chief
**Identity Description**: Editor-in-Chief of *Artificial Intelligence and Applications*, with a research background in applied machine learning systems and a mandate to serve a readership of applied AI researchers and practitioners rather than pure NLP theorists.
**Review Focus**:
1. Whether the paper's contribution and framing fit AIA's applied/practitioner scope
2. Whether the narrative (title → abstract → results → conclusion) is internally consistent and not over-promising
3. Overall originality and significance relative to the crowded APO/meta-prompting literature
**Will particularly care about**: Whether a reader skimming only the abstract and conclusion would come away with an accurate impression of how strong the empirical result actually is.
**Possible blind spots**: Statistical technicalities (deferred to R1); fine-grained citation accuracy (deferred to R2).

### Reviewer Configuration Card #2 — Peer Reviewer 1 (Methodology)
**Role**: Peer Reviewer 1
**Identity Description**: NLP evaluation methodologist specializing in statistical significance testing and reproducibility for LLM benchmarking (in the tradition of significance-testing-in-NLP and power-analysis literature), with particular interest in benchmark-level bootstrap methodology.
**Review Focus**:
1. Statistical treatment of the headline PromptWizard-vs-MPIR comparison (bootstrap CI, significance framing)
2. Experimental design rigor: single-run vs. repeated-seed evaluation, ablation sample sizes
3. Internal numerical/table consistency (cross-checking reported statistics against the tables that generate them)
**Will particularly care about**: Whether the paper's claims of "improvement" are calibrated to what the reported confidence interval actually supports.
**Possible blind spots**: Domain positioning relative to prior meta-prompting work (R2); practical deployment cost (R3).

### Reviewer Configuration Card #3 — Peer Reviewer 2 (Domain)
**Role**: Peer Reviewer 2
**Identity Description**: Senior researcher in prompt engineering and automatic prompt optimization, closely familiar with the APE/ProTeGi/EvoPrompt/OPRO/PromptWizard/DSPy/TextGrad/PE2/PROPEL lineage that the manuscript's Related Work section draws on.
**Review Focus**:
1. Completeness and accuracy of the Related Work section, including whether cited works are characterized the way their titles/known content would support
2. Positioning of MPIR's contribution relative to PE2, PROPEL, and dual-phase optimization (the three closest prior works)
3. Genuine incremental contribution vs. re-labeling of known ideas
**Will particularly care about**: Whether MPIR is actually differentiated from PROPEL (expert priors as guidance) and PE2 (structured step-by-step refinement), which pursue very similar goals.
**Possible blind spots**: Statistical rigor of the reported gains (R1); practical/engineering feasibility (R3).

### Reviewer Configuration Card #4 — Peer Reviewer 3 (Perspective)
**Role**: Peer Reviewer 3
**Identity Description**: Applied ML/MLOps engineer with experience deploying LLM pipelines in production, bringing a practitioner's-eye view of cost, latency, and reproducibility risk from reliance on closed, versioned commercial APIs (GPT-3.5-turbo, GPT-4o, Gemini).
**Review Focus**:
1. Practical feasibility and cost/latency implications of the two-model, multi-round (N=7) refinement pipeline
2. Reproducibility risk from dependence on closed, frequently-updated commercial model endpoints
3. Accessibility/clarity of the paper's narrative for readers outside the immediate APO subfield (AIA's broader applied readership)
**Will particularly care about**: Whether the claimed "scalability" and "reduced reliance on expert prompt engineering" actually translate into a cheaper or simpler real-world pipeline than just prompting an expert once.
**Possible blind spots**: Fine-grained NLP methodology (R1); detailed literature lineage (R2).

### Reviewer Configuration Card #5 — Devil's Advocate
**Role**: Devil's Advocate
**Identity Description**: Assigned to stress-test the paper's core empirical claim independent of the other four reviewers.
**Review Focus**: Whether the central claim — "MPIR outperforms its PromptWizard baseline" — is actually supported by the reported statistics, given a 95% bootstrap CI of [−0.46, 4.70] on the average improvement.
**Will particularly care about**: The gap between the abstract/conclusion's confident framing and the hedged, easy-to-miss admission buried in Section 5.2.1.
**Possible blind spots**: N/A by design (DA does not soften for balance).

## Review Strategy Recommendation
This is a well-organized, clearly written engineering/algorithms paper whose central risk is not "is the idea good" (it is a sensible, modular idea) but "does the evidence support the confidence with which the result is stated." Four of five reviewers should therefore weight the statistical framing question heavily, while dividing the remaining ground (originality/fit, literature accuracy, practical feasibility) into non-overlapping lanes per the IRON RULE.

---

# Phase 1: Independent Reviewer Reports

## Report 1 of 5 — EIC Review Report

### Manuscript Information
- **Title**: Meta-Prompted Instruction Refinement: Bridging Manual Prompting Techniques and Automatic Prompt Optimization
- **Review Round**: Round 1

### Reviewer Role
Editor-in-Chief, *Artificial Intelligence and Applications*

### Reviewer Identity
Editor-in-Chief of *Artificial Intelligence and Applications*, applied ML systems background, evaluating for fit with AIA's applications-oriented readership.

### Review Focus
Journal fit, originality/significance in the broader field, and — per this journal's emphasis on accessible, honest applied reporting — whether the paper's narrative arc (title → abstract → conclusion) accurately represents the strength of its own results.

### Recommendation
**Major Revision**

### Confidence Score
4 — high confidence; within my editorial competence, though I defer detailed statistics to Reviewer 1.

### Summary Assessment
This paper proposes MPIR, a rubric-guided meta-prompting layer that refines prompts produced by automatic prompt optimization (APO) systems, and evaluates it on Big-Bench Hard across three APO backbones and two model families. The idea is sensible and the engineering is thorough: a seven-criterion rubric, an evaluate-refine-validate loop, and a genuinely broad set of generalization checks (three APO methods, a cross-model transfer test, three ablations). The writing is clear and the structure is textbook-appropriate for an applied AI venue. However, the paper's headline claim — that MPIR "outperforms its PromptWizard baseline" — rests on an average improvement of +1.97 points whose own reported 95% bootstrap CI is [−0.46, 4.70], i.e., it cannot be distinguished from zero at conventional confidence. The abstract and conclusion state the result as an established fact ("MPIR outperforms... on 16 of 23 tasks") without conveying this uncertainty, and the hedge that does exist, buried in Section 5.2.1, undersells how consequential this is for the top-line claim. Combined with a table-numbering/arithmetic issue in Section 5.3 (detailed by R1), this is fixable but real work, warranting Major rather than Minor Revision.

### Strengths
1. **Clear, modular framing of a real gap**: The Introduction (p.1, "The trade-off between the effectiveness and interpretability of manual prompting and the efficiency and scalability of APO remains unresolved") crisply motivates the work, and the framework is explicitly designed as a drop-in layer "on top of an existing APO system" (Section 3), which is a genuinely useful design choice for practitioners who already have an APO pipeline.
2. **Breadth of generalization testing**: Beyond the primary PromptWizard result, the authors test two additional APO backbones (APE, ProTeGi; Table 2) and a different model family (Gemini; Table 3), which is more generalization evidence than most single-baseline APO papers provide.
3. **Honest self-reported limitations**: The Conclusion (Section 6) proactively flags that "its average gain over baseline APO systems is relatively modest" and that the rubric "was developed in close proximity to the BBH tasks themselves, which may introduce construct dependence" — this kind of self-critical limitations section is commendable and above the field norm.

### Weaknesses
1. **Abstract/Conclusion overstate a statistically inconclusive headline result** (Severity: Critical — see also Devil's Advocate report for full treatment). **Problem**: The abstract states MPIR "outperforms its PromptWizard baseline on 16 of 23 tasks" without qualification; the body (5.2.1) reports a 95% CI of [−0.46, 4.70] for the average effect. **Why it matters**: A reader relying on the abstract (the majority of readers) would reasonably conclude the improvement is established; it is not, at the level of statistical confidence the authors themselves computed. **Suggestion**: Add an explicit hedge to the abstract itself (e.g., "with a positive but not statistically decisive average improvement") and move the CI/significance caveat out of a single mid-paragraph sentence into a clearly flagged statement in both the Results summary and the Conclusion.
2. **"Over-promising and under-delivering" risk in the framing of scalability**: The abstract and Introduction position MPIR as "reducing reliance on expert prompt engineering," yet Section 5.2.5 shows MPIR still trails expert-crafted prompting by 5+ points on average and by 20–30+ points on four specific tasks (Table 4: navigate, web_of_lies). **Why it matters**: The gap to the "expert-crafted" ceiling is a first-order result, not a footnote, and should be reflected in how strongly the abstract frames the "reduced reliance on expert prompt engineering" claim. **Suggestion**: Soften the abstract's framing to "narrows, but does not close, the gap to expert-crafted prompting" or similar.
3. **Structural coherence gap between Section 5.3 ablations and their own conclusions**: As detailed in R1's report, an internal numerical inconsistency in the ablation table undermines the confidence with which Section 5.3.3's "substantial decline" claim can currently be read. **Suggestion**: See R1's Weakness #1 for the specific fix required; I flag it here only because it affects whether the paper's conclusion ("validation anchors MPIR's refinements... ensuring gains are both real") is currently fully earned by the presented evidence.

### Detailed Comments

#### Journal Fit
Reasonable fit. AIA's applications orientation matches the paper's framing as a practical, plug-in refinement layer rather than a purely theoretical contribution. The paper's length and structure (Introduction/Related Work/Method/Implementation/Results/Conclusion) match a typical AIA research article.

#### Originality
The core idea — treating heuristic-guided refinement as an explicit, separately-validated *stage* applied after APO has already run, rather than as an initialization seed or loose in-loop guidance — is a genuine, if incremental, contribution. It is closely adjacent to PROPEL and PE2 (see R2's report for the detailed positioning critique); the paper's own Related Work (Section 2.4) acknowledges this closeness but could sharpen the differentiation further.

#### Significance
Moderate. If the effect were established at the claimed magnitude, this would be a useful, generalizable technique. Given the non-significant primary result, the paper's significance currently rests more on its qualitative findings (which task types benefit, why validation matters, which rubric criteria matter) than on the headline accuracy number — which is a fine paper to have, but the current framing doesn't fully own that reality.

#### Structural Coherence
Title → Abstract → Introduction are consistent with each other. The Conclusion is honest about limitations. The weak link is specifically the abstract's silence on statistical uncertainty combined with the Results section's under-emphasized hedge — an internal coherence gap between how confidently the claim is stated in three of four places (title implicitly, abstract, conclusion) versus how it is actually supported in the one place with the numbers (5.2.1).

#### Title & Abstract
Title is accurate and appropriately scoped. Abstract is well-written and readable for a non-specialist, but — per Weakness #1 — omits the single most important caveat about its own headline number.

#### Conclusion
Good-faith limitations discussion; recommend cross-referencing the statistical caveat here as well (currently the Conclusion states the positive framing without revisiting the CI).

### Questions for Authors
1. Given the reported 95% CI of [−0.46, 4.70], how would the authors characterize the strength of the primary claim if asked to state it in one sentence for a non-specialist reader — and does the current abstract match that characterization?
2. Was the +1.97-point average improvement tested against any alternative null (e.g., a permutation test comparing MPIR vs. PromptWizard per-task, rather than only a bootstrap CI on the mean difference)? Would results differ using a paired test that accounts for per-task pairing rather than resampling tasks independently?
3. Table 2's caption says the five rows "best illustrate the pattern of gains and regressions" — what was the actual selection procedure (e.g., top-N gain, top-N loss, or a priori interest)? Stating this explicitly would pre-empt any concern about selective reporting.

### Minor Issues
- Consider giving the abstract's "gains of up to 20 percentage points" a matching downside statement (e.g., largest regression magnitude), for balance.
- The corresponding author email is listed as "[to be provided]" — ensure this is finalized before typesetting.

### Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 65 | Adequate | Genuine but incremental combination of known ideas |
| Methodological Rigor (25%) | 58 | Weak-Adequate | See R1 for detail; non-significance not adequately reflected in claims |
| Evidence Sufficiency (25%) | 55 | Weak | Single-run design; headline effect statistically inconclusive |
| Argument Coherence (15%) | 62 | Adequate | Abstract/Conclusion outrun what Results actually shows |
| Writing Quality (15%) | 80 | Strong | Clear, well-organized prose |
| **Weighted Average** | **62.6** | **Major Revision** | |

---

## Report 2 of 5 — Methodology Review Report (Peer Reviewer 1)

### Manuscript Information
- **Title**: Meta-Prompted Instruction Refinement: Bridging Manual Prompting Techniques and Automatic Prompt Optimization
- **Review Round**: Round 1

### Reviewer Role
Peer Reviewer 1 (Methodology)

### Reviewer Identity
NLP evaluation methodologist specializing in statistical significance testing and reproducibility for LLM benchmark comparisons.

### Review Focus
Statistical validity of the reported comparisons, experimental design (single-run vs. repeated-trial evaluation), and cross-checking the internal numerical consistency of the tables that support the paper's quantitative claims.

### Recommendation
**Major Revision**

### Confidence Score
5 — this is squarely within my area of expertise.

### Summary Assessment
The paper reports accuracy differences across a large number of task/method/model combinations, computed from what appears to be a single run per condition (temperature = 0, one random draw of 25 training examples per method). The one genuine inferential statistic reported — a 10,000-iteration bootstrap 95% CI of [−0.46, 4.70] for the average MPIR-vs-PromptWizard improvement — includes zero, meaning the headline result cannot be distinguished from no effect at conventional confidence. This is not itself disqualifying (negative/null results are publishable), but the paper's downstream statistical practice does not fully absorb this finding: no significance testing is applied anywhere else (the three ablation studies, the cross-model test, and the two-APO generalization test all report point-outcome averages with no interval or test), and I identified an internal arithmetic inconsistency in the Section 5.3 ablation table that the authors need to resolve before the ablation conclusions can be trusted as reported. The single-run design (no repeated seeds/example-subsets) is the deeper issue underlying all of this: without repeated trials, we cannot separate "genuine method effect" from "which 25 training examples happened to be drawn."

### Strengths
1. **Correct instinct to run a bootstrap analysis at all**: Most APO papers in this space (PromptWizard, APE, ProTeGi, EvoPrompt, OPRO as cited) report only point accuracy differences with no uncertainty quantification whatsoever; the authors' 10,000-iteration bootstrap in Section 5.2.1 is genuinely more rigorous than the norm in this specific subfield, and the number of tasks on which MPIR wins (16/23) is correctly and verifiably computed from Table 1 (I independently re-derived this: MPIR beats PromptWizard on 16 tasks, loses on 6, and ties on 1 [formal_fallacies, 53.3=53.3], summing to 23).
2. **Transparent hyperparameter reporting**: Table 6 (hyperparameters) and the explicit statement of temperature=0, the 25/rest train-test split, and N=7 refinement rounds (Section 4.5) support partial reproducibility — better documentation than many comparable papers provide.
3. **Deliberate baseline design to isolate the rubric's contribution**: The "Free Rewrite" baseline (Section 4.3.1), which asks GPT-4o to freely rewrite prompts without the rubric, is a well-chosen control that correctly isolates rubric-specificity from generic LLM rewriting ability — and the reported gap (57.15% free-rewrite vs. 64.37% MPIR) does support the authors' claim that the rubric, not just rewriting, drives the gain.

### Weaknesses
1. **Internal arithmetic inconsistency in the second "Table 6"** (Severity: Critical). **Problem**: The manuscript contains two tables both labeled "Table 6" — the hyperparameter table in Section 4.5, and the rubric/validation ablation table in Section 5.3.2–5.3.3. In the second Table 6, the "MPIR (full rubric)" column and the "MPIR (with validation)" column report *identical* per-task values across all five tasks (84.0/82.6/72.0/92.4/68.4 in both columns) — which is internally sensible, since "full rubric" and "with validation" describe the same underlying MPIR condition reused as the shared reference point for two separate ablations. However, the two columns' reported **averages differ** (79.9% for "full rubric" vs. 75.4% for "with validation"), which is mathematically impossible if the row values are identical. Section 5.3.3's claim that "removing validation drops average accuracy from 75.4% to 61.9%, a substantial decline" is therefore currently built on a column average that does not match its own displayed rows. **Why it matters**: This is not a stylistic nitpick — the validation ablation (Section 5.3.3) is used to support one of the paper's stronger causal claims ("validation anchors MPIR's refinements... ensuring gains are both real"), and right now that number cannot be verified from the table as printed. **Suggestion**: The authors must recompute and correct the "MPIR (with validation)" average (it should equal 79.9% if the row values are correct as printed, or the row values need correcting if 75.4% is the intended average) and — separately — renumber the tables so there is only one "Table 6" (the hyperparameter table should likely remain Table 6, with the rubric/validation comparison retitled "Table 7," matching the in-text reference to "(Table 7)" in Section 5.3.3, which currently points to a table that does not exist under that number).
2. **No repeated-trial variance anywhere in the study** (Severity: Major). **Problem**: Every reported accuracy — across Tables 1–7 — appears to derive from a single run per condition: one draw of 25 training/optimization examples, one optimization trajectory, one set of N=7 refinement rounds, evaluated once on the held-out test split. The only uncertainty estimate offered (the bootstrap CI in 5.2.1) resamples *across tasks*, which quantifies "how consistent is the effect across different BBH task types" but not "how much would results change if we reran the same pipeline with a different random seed or a different draw of the 25 training examples." These are different sources of variance, and the paper does not distinguish them. **Why it matters**: Given that PromptWizard's own optimization process involves stochastic mutation/refinement, and that MPIR's meta-prompting stage uses an LLM to critique and rewrite, some portion of the observed 1.97-point average gain (and, more importantly, some portion of the CI width) may reflect run-to-run noise rather than a stable causal effect of the MPIR layer. **Suggestion**: Repeat the primary PromptWizard-vs-MPIR comparison across at least 3–5 random seeds/training-example draws and report both between-task and between-seed variance; a paired-task bootstrap (resampling task-level *differences* directly, which the paper may already be doing — this should be stated explicitly) would also be more appropriate than resampling raw scores if not already the method used.
3. **Ablation sample sizes are very small relative to their claims** (Severity: Major). **Problem**: The three ablation studies (Section 5.3, Tables 5–7) are each conducted on only 5 of the 23 BBH tasks, with conclusions phrased in general terms ("every criterion plays a meaningful role," "the seven-criteria rubric's specificity... drives MPIR's effectiveness"). **Why it matters**: A 5-task sample is a reasonable computational-cost compromise, but claims phrased as general properties of the rubric (rather than "on these five representative tasks") risk overgeneralizing from a small, non-randomly-selected subset. **Suggestion**: Either (a) explicitly state and justify the criterion used to select these five tasks (currently unstated for Table 5, though Table 2/3's captions do state a selection criterion), or (b) soften ablation conclusions to explicitly scope them to "on this five-task subset."
4. **Minor internal rounding inconsistency**: Table 5's row values sum to a mean of 79.88% (my recomputation) for the "ALL" (full-rubric) column, but the table's own "Average" row and the accompanying prose (Section 5.3.1) both report "80.0%" — a 0.1-point discrepancy from strict rounding of the displayed row values, which is trivial but adds to the general impression that the numeric layer of Section 5.3 needs a careful pass. **Suggestion**: Recompute all displayed averages directly from the displayed row values as a final consistency check before resubmission.

### Detailed Comments

#### Research Questions & Hypotheses
Clear, sensible, and directly operationalized (Section 3.1's formal problem statement is a nice touch that most APO papers skip).

#### Research Design
Sound overall design (evaluate → refine → validate loop, held-out validation split). The core design limitation is the lack of repetition (Weakness #2).

#### Sampling Strategy
25 training examples drawn once per method; adequate for the optimization stage itself but insufficient, without repetition, to support claims about the *stability* of the resulting gains.

#### Data Collection
BBH is an appropriate, widely-used, difficulty-selected benchmark (23 tasks, ~250 examples each); good choice for this kind of study.

#### Analysis Methods
Accuracy is the sole metric — reasonable for BBH's multiple-choice/exact-match tasks. The bootstrap CI is the right kind of test to have run; see Weaknesses #1–2 for how its results should shape the paper's framing.

#### Results Presentation
Generally complete and non-selective — the paper does report several tasks where MPIR regresses (geometric_shapes, tracking_shuffled_objects, temporal_sequences, causal_judgment) rather than hiding them, which is commendable transparency. The Table 6/7 numbering and arithmetic issue (Weakness #1) is the main defect here.

#### Reproducibility
Code, prompts, and results are stated to be publicly available (GitHub link given), and hyperparameters are reported in full (Table 6/hyperparameters) — good practice. Reproducibility is nonetheless bounded by dependence on versioned commercial APIs (GPT-3.5-turbo, GPT-4o) whose behavior can drift or be deprecated over time (this specific angle is developed further by Reviewer 3).

#### Methodological Fallacies Detected
- **Borderline overclaiming from a non-significant result** (see Weakness #1 in the EIC report and the Devil's Advocate report for full treatment; from a purely statistical standpoint, I concur this is the central issue).
- No p-hacking or multiple-comparisons red flags detected — the paper does not appear to have selectively run and reported only favorable comparisons; the free-rewrite and cross-model checks in particular look like planned, not post-hoc, tests.

### Questions for Authors
1. Is the bootstrap CI in Section 5.2.1 computed by resampling raw task accuracies independently for each method, or by resampling the paired per-task differences? This materially affects the interpretation and should be stated explicitly in the text.
2. Can the authors clarify and correct the "MPIR (with validation)" average in the Section 5.3 rubric/validation table (Weakness #1)?
3. How many random seeds / training-example draws were used for the primary PromptWizard and MPIR runs reported in Table 1 — one, or an average over several?

### Minor Issues
- Two tables are both numbered "Table 6" (Section 4.5's hyperparameter table, and Section 5.3.2–5.3.3's rubric/validation table); the latter is referred to in text as both "(Table 6)" (5.3.2) and "(Table 7)" (5.3.3), neither of which currently exists as a uniquely-numbered table. Please renumber Tables 6 onward sequentially.
- Task name spelling is inconsistent: "causal_judgment" (Table 1) vs. "causal_judgement" (Table 3, Section 5.2.3) — pick one spelling throughout.

### Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 65 | Adequate | Not my primary focus; deferring to R2 |
| Methodological Rigor (25%) | 52 | Weak | Single-run design, no seed variance, internal table arithmetic error |
| Evidence Sufficiency (25%) | 55 | Weak | Headline result statistically inconclusive; small ablation samples |
| Argument Coherence (15%) | 58 | Weak-Adequate | Ablation conclusions outrun what corrected tables would show |
| Writing Quality (15%) | 78 | Strong | Clear technical exposition |
| **Weighted Average** | **60.2** | **Major Revision** | |

---

## Report 3 of 5 — Domain Review Report (Peer Reviewer 2)

### Manuscript Information
- **Title**: Meta-Prompted Instruction Refinement: Bridging Manual Prompting Techniques and Automatic Prompt Optimization
- **Review Round**: Round 1

### Reviewer Role
Peer Reviewer 2 (Domain)

### Reviewer Identity
Senior researcher in prompt engineering and automatic prompt optimization, with close familiarity with the APE/ProTeGi/EvoPrompt/OPRO/PromptWizard/DSPy/TextGrad/PE2/PROPEL literature this manuscript builds on.

### Review Focus
Completeness and accuracy of the literature review; whether cited works are characterized in ways consistent with their actual content; and the genuineness/scope of MPIR's incremental contribution relative to its three closest prior works.

### Recommendation
**Minor Revision**

### Confidence Score
4 — high confidence in this subfield.

### Summary Assessment
The paper surveys the prompt-engineering and APO literature capably, correctly grouping manual heuristics (Section 2.2), discrete/black-box APO methods (Section 2.3), and heuristic-guided meta-prompting hybrids (Section 2.4). Most citations are used in ways consistent with what their titles and known content support; I found a small number of loosely-matched citations (detailed below) but nothing that suggests systematic misrepresentation. The paper's central positioning claim — that prior heuristic-APO hybrids inject heuristics only "at initialization" or "loosely during optimization," whereas MPIR treats heuristic-guided refinement as an "independent stage... after an APO prompt has already been produced" (Section 2.4) — is a real and fair distinction from the two closest works (PROPEL, dual-phase optimization), though it undersells how close PE2's "step-by-step reasoning templates and structured analysis" already comes to the same idea. Overall, literature coverage is good; the main ask is to sharpen differentiation from PE2/PROPEL and tighten two loosely-matched citations.

### Strengths
1. **Well-organized, appropriately scoped literature taxonomy**: Sections 2.1–2.4 cleanly separate "prompting as interaction," "manual heuristics," "APO," and "meta-prompting/hybrids," which is the right taxonomy for this subfield and makes the paper easy to place for a reader new to it.
2. **Accurate characterization of the core cited APO methods**: PromptWizard [33] ("self-evolving, self-adapting framework... feedback-driven critique and synthesis"), ProTeGi [14] ("mimics gradient descent with textual feedback"), APE [31], EvoPrompt [15], and OPRO [32] are each described in ways that match their actual titles/known contributions — I did not find a misattribution among the primary APO baselines.
3. **Fair, specific articulation of the research gap** (Section 2.4, final paragraph): "manual prompting offers interpretability and effectiveness but does not scale, while automatic optimization offers efficiency and scalability but often neglects heuristic grounding" is a genuinely useful, quotable framing of the field's central tension, and the paper's contribution is correctly positioned as an attempt to bridge it.

### Weaknesses
1. **Differentiation from PE2 and PROPEL is under-argued** (Severity: Major). **Problem**: Section 2.4 states that PE2 "formalizes refinement through step-by-step reasoning templates" and PROPEL "integrates a large set of expert-derived prompting principles as priors to guide refinement" — both descriptions sound, on their face, extremely close to what MPIR itself does (a structured rubric applied via meta-prompted refinement). The paper's differentiation rests on a single sentence: heuristics are "typically injected at initialization... or loosely during optimization... rather than treated as an independent stage." **Why it matters**: For a domain expert, this is the single most important paragraph in the Related Work section, and it is currently under-defended — a reviewer unfamiliar with PE2/PROPEL's exact mechanics would not be able to verify the "loosely" characterization from the text alone. **Suggestion**: Add 1–2 concrete sentences contrasting *where in the pipeline* PE2/PROPEL apply their heuristics versus MPIR's clean separation into a distinct post-hoc evaluate-refine-validate stage with its own held-out validation set — this is MPIR's most defensible differentiator and deserves more than one sentence.
2. **Two citations are loosely matched to the claims they support** (Severity: Minor). **Problem**: (a) Citation [26] (Coda-Forno et al., "CogBench: A large language model walks into a psychology lab") is cited alongside [25] (Zheng et al., step-back prompting) to support the step-back reasoning claim (Section 3.3); CogBench's title suggests a broader psychological/behavioral evaluation framework for LLMs, and it is not obvious from the manuscript how it specifically supports step-back reasoning without further explanation. (b) Citation [21] (Wang & Zhou, "Chain-of-thought reasoning without prompting") is cited alongside [22] to support "coherent, relevant reasoning steps matter more than reasoning length" (Section 2.2, Section 5.2.4); [22]'s title directly supports this claim, but [21] is about eliciting CoT via decoding *without* explicit prompting — a related but distinct claim. **Why it matters**: Neither is likely to be a fabricated or invented reference, but both create small friction for a domain-expert reader trying to trace the specific evidentiary chain. **Suggestion**: Either add a half-sentence clarifying the specific connection, or replace with a more directly on-point citation.
3. **Contribution is real but incremental, and the paper could be more forthright about this** (Severity: Minor). **Problem**: Once PE2/PROPEL/dual-phase-optimization are properly weighed (see Weakness #1), MPIR's novelty is best described as "a specific, well-engineered instantiation and empirical validation of an idea (post-hoc heuristic-guided refinement of APO output) that the field was already converging on," rather than a conceptually new idea. **Suggestion**: The Introduction's four objectives (page 1) could explicitly acknowledge this framing — "we do not claim the idea of heuristic-guided refinement is new; we contribute the first systematic rubric, cross-framework validation, and ablation of its components" — which would, if anything, make the contribution claim more credible and specific, not weaker.

### Detailed Comments

#### Literature Review
- **Coverage**: Comprehensive within its immediate subfield (APO/meta-prompting); appropriately covers seminal CoT/few-shot/role-prompting work as background. No major foundational works appear to be missing.
- **Integration quality**: Genuinely synthetic rather than a list — Section 2.4's closing paragraph in particular does real critical synthesis work.
- **Research gap argument**: Persuasive and specific (see Strengths #3), modulo the PE2/PROPEL differentiation gap (Weakness #1).

#### Theoretical Framework
- **Appropriateness**: The paper does not lean on a named theoretical framework beyond the seven-criteria rubric itself, which is appropriate for an applied/engineering contribution of this kind.
- **Application depth**: The rubric is applied concretely and its components are separately ablated (Section 5.3.1) — genuine depth, not superficial naming.

#### Academic Argument Quality
- **Factual accuracy**: I did not identify factual errors in how the cited APO/prompting techniques are described (see Strengths #2).
- **Terminology precision**: Consistent and field-standard use of "APO," "meta-prompting," "in-context learning," etc.

#### Contribution to the Field
- **Incremental contribution**: Real but modest (see Weakness #3); the empirical breadth (3 APO backbones, cross-model, 3 ablations) is arguably the paper's stronger claim to contribution than the conceptual novelty of the idea itself.
- **Overclaiming**: The domain-positioning claims are appropriately scoped; the main overclaiming risk in this manuscript is statistical (per EIC/R1/DA reports), not a literature-positioning issue.

#### Missing Key References
- No major omissions identified. One suggestion for completeness: work specifically on prompt sensitivity/instability under paraphrase (the paper already cites Errica et al. [6] for this) could be supplemented with at least one additional empirical study quantifying paraphrase sensitivity, since this motivates the entire validation-stage design — but this is a "nice to have," not a gap that undermines the paper.

### Questions for Authors
1. Can the authors more precisely characterize *where* PE2 and PROPEL apply their heuristics in the optimization pipeline, to substantiate the "loosely" vs. MPIR's "independent stage" distinction?
2. Was PE2 or PROPEL considered as an additional baseline (alongside PromptWizard/APE/ProTeGi) for direct empirical comparison, rather than only a conceptual comparison in Related Work?

### Minor Issues
- Task name spelling inconsistency ("causal_judgment" vs. "causal_judgement") — noted independently by R1; flagging jointly for completeness of the citation/terminology review.
- Citations [11] and [17] (both Reynolds & McDonell, 2021) appear to be the same paper cited twice under two different reference numbers — please deduplicate.

### Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 68 | Adequate-Strong | Real but incremental relative to PE2/PROPEL |
| Methodological Rigor (25%) | 65 | Adequate | Deferring to R1's detailed statistical assessment |
| Evidence Sufficiency (25%) | 62 | Adequate | Good breadth of generalization tests |
| Argument Coherence (15%) | 68 | Adequate | Related Work argument is coherent modulo Weakness #1 |
| Writing Quality (15%) | 82 | Strong | Clear, well-organized |
| Literature Integration | 78 | Strong | Comprehensive, minor citation-matching looseness |
| **Weighted Average** | **67.9** | **Minor Revision** | |

---

## Report 4 of 5 — Perspective Review Report (Peer Reviewer 3)

### Manuscript Information
- **Title**: Meta-Prompted Instruction Refinement: Bridging Manual Prompting Techniques and Automatic Prompt Optimization
- **Review Round**: Round 1

### Reviewer Role
Peer Reviewer 3 (Perspective)

### Reviewer Identity
Applied ML/MLOps engineer with production LLM-pipeline deployment experience; reviewing from a practical feasibility and reproducibility-risk angle rather than an NLP-research angle. As a non-specialist in prompt-optimization theory, my confidence is calibrated accordingly.

### Review Focus
Practical/economic feasibility of the MPIR pipeline, reproducibility risk from dependence on closed commercial model APIs, and accessibility of the paper's narrative for AIA's broader applied readership.

### Recommendation
**Minor Revision**

### Confidence Score
3 — moderate confidence; core NLP methodology questions are outside my primary expertise and are deferred to R1/R2.

### Summary Assessment
MPIR is pitched as reducing "reliance on expert prompt engineering" and improving scalability, but from a deployment-cost perspective, the actual pipeline is not obviously cheaper or simpler than the alternatives it's compared against: it requires running PromptWizard (itself an iterative optimization process) *and then* seven additional rounds of GPT-4o-based evaluation-refinement plus GPT-3.5-turbo validation calls per round. As an outsider to the specific NLP methodology, I cannot assess the statistical claims (which R1 and the Devil's Advocate cover in depth), but from a practical standpoint the paper would benefit from explicitly discussing the compute/API cost of the full pipeline relative to its alternatives, and from acknowledging that both target models used (GPT-3.5-turbo, GPT-4o) are commercial, versioned endpoints whose behavior is not guaranteed to be stable over time — a real risk for anyone trying to reproduce or build on this work eighteen months from now. On accessibility: the paper is well-written for an NLP-literate reader but assumes familiarity with APO terminology that AIA's broader applied AI audience may not have.

### Strengths
1. **Genuinely modular design that respects practitioner constraints**: The framing of MPIR as a layer that sits "on top of an existing APO system" without requiring changes to the underlying optimizer (Section 3, opening paragraph) is exactly the right design goal for a technique meant to be adopted by teams that already have an APO pipeline in production — this is a real practical strength, not just a rhetorical one.
2. **Concrete before/after examples aid interpretability for non-specialists**: The worked example in Section 5.1 (Figure 6, the penguins_in_a_table case) and the hyperbaton/web_of_lies prompt-rewrite examples (Section 5.2.4) are genuinely useful for a reader like me who doesn't work in this exact subfield day-to-day — they make an otherwise abstract accuracy-table paper tangible.
3. **Public code/data release**: The GitHub link (Section 4.5, Data Availability Statement) is good practice and lowers the barrier for practitioners who want to actually try the technique rather than just read about it.

### Weaknesses
1. **No discussion of the pipeline's actual compute/API cost** (Severity: Major, from a practical-adoption standpoint). **Problem**: The full pipeline requires (a) running PromptWizard's own iterative optimization (mutate_refine_rounds=3, mutate_rounds=3, max_seq_iter=3 per Table 6), then (b) N=7 rounds of MPIR evaluation+refinement (GPT-4o calls) each followed by a validation pass (GPT-3.5-turbo calls on the held-out set). **Why it matters**: The abstract's framing — "making prompt optimization more effective, interpretable, and scalable, while reducing reliance on expert prompt engineering" — implicitly promises efficiency gains, but a practitioner reading this paper has no way to judge whether MPIR's total API cost/wall-clock time is smaller, comparable to, or larger than simply hiring/consulting a prompt-engineering expert once (the "expert-crafted" baseline this paper compares against, which still outperforms MPIR on average per Table 1). **Suggestion**: Report approximate total API calls / token cost / wall-clock time for the full MPIR pipeline per task, and compare this against a rough estimate of expert-engineering time cost, so readers can judge the actual scalability tradeoff being offered.
2. **Reproducibility risk from closed, versioned commercial endpoints is unacknowledged** (Severity: Minor-Major). **Problem**: GPT-3.5-turbo, GPT-4o, and (in the cross-model experiment) Gemini 3.5 Flash-Lite are all commercial APIs that can be silently updated or deprecated by their providers. **Why it matters**: A reader trying to reproduce Table 1's exact numbers in even one year may be unable to, through no fault of the authors' methodology — this is a known, field-wide limitation of LLM benchmarking research, but the paper's own Limitations paragraph (Section 6) does not mention it at all, despite mentioning several other, arguably less consequential limitations (rubric-benchmark construct dependence, shared validation examples). **Suggestion**: Add one sentence to Section 6 acknowledging that reported absolute numbers are tied to specific model API versions/snapshot dates and may not be exactly reproducible on future model updates.
3. **Some APO-specific terminology is introduced without a bridge for non-specialist AIA readers** (Severity: Minor). **Problem**: Terms like "few-shot count," "mutate_refine_rounds," and the Algorithm 2 pseudocode assume the reader is already fluent in the APO literature's internal vocabulary; a reader of AIA's broader applied-AI scope (e.g., someone applying LLMs in a non-NLP-research domain) may find Section 4.2–4.3 hard to follow without first reading the PromptWizard paper itself. **Suggestion**: A one-paragraph, plain-language summary of "what does PromptWizard actually produce, and why does MPIR need it as a starting point" placed before the Algorithm 2 pseudocode would make Section 4 much more accessible to AIA's stated readership.

### Detailed Comments

#### Introduction
Motivation is accessible and well-pitched even to a reader outside prompt-engineering research specifically (the "trade-off between effectiveness/interpretability... and efficiency/scalability" framing, page 1, is intuitive).

#### Methodology / Research Design
As a non-specialist in the specific optimization mechanics, I defer technical assessment to R1; from a systems-design view, the two-stage architecture (APO then MPIR) is clean and its modularity is a genuine practical asset (Strength #1).

#### Discussion
The Discussion (Section 5.2) usefully identifies *which kinds* of tasks benefit most (structured/rule-based reasoning) versus least (symbolic manipulation, spatial abstraction) — this task-type-conditional framing is more practically actionable for a deploying engineer than a single aggregate number would be, and I'd encourage the authors to foreground this more (e.g., in the abstract) since it may be the paper's most immediately useful takeaway for practitioners.

#### Conclusion
Limitations are honest but incomplete from a deployment-risk perspective (Weakness #2).

### Cross-Disciplinary Reading Recommendations
- Any recent MLOps/production-LLM-serving literature on API cost/latency benchmarking for multi-call agentic pipelines, to give the compute-cost discussion (Weakness #1) a quantitative anchor.
- Work on the reproducibility crisis in benchmarking closed commercial LLMs (a growing subliterature adjacent to, but broader than, the specific paper on prompt sensitivity [6] already cited) — would strengthen the Limitations discussion.

### Questions for Authors
1. What is the approximate total number of API calls (or estimated cost) required to run the full MPIR pipeline (PromptWizard optimization + 7 rounds of evaluation-refinement-validation) for a single task, and how does this compare to the effort of having a human expert write a prompt once?
2. Which specific model snapshots/versions of GPT-3.5-turbo and GPT-4o were used, and on what date were the experiments run? This affects reproducibility given API version drift.

### Minor Issues
- Consider adding a one-line glossary or forward-reference the first time PromptWizard-specific hyperparameter names (e.g., "mutate_refine_rounds") appear in running text, for readers who have not read the original PromptWizard paper.

### Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 66 | Adequate | Deferring depth to R2 |
| Methodological Rigor (25%) | 60 | Adequate | Deferring depth to R1 |
| Evidence Sufficiency (25%) | 60 | Adequate | Deferring depth to R1 |
| Argument Coherence (15%) | 64 | Adequate | Practical-scalability claim not fully substantiated |
| Writing Quality (15%) | 80 | Strong | Clear, though APO jargon is a barrier for broader AIA readers |
| Significance & Impact | 70 | Strong-Adequate | Real practical utility, bounded by cost/reproducibility unknowns |
| **Weighted Average** | **64.8** | **Minor-to-Major boundary; recommending Minor** | Core contribution sound; asks are additive discussion, not re-experimentation |

---

## Report 5 of 5 — Devil's Advocate Review

### Strongest Counter-Argument

Before challenging the paper, it is worth acknowledging what it does well: MPIR is a sensibly engineered, modular refinement layer, and the authors deserve credit for testing it across three APO backbones and two model families rather than resting on a single favorable comparison, and for an unusually candid Limitations section.

That said, here is the strongest case that the paper's central claim is not established. The title and abstract assert that MPIR "outperforms its PromptWizard baseline," and the Conclusion restates that "experiments on Big-Bench Hard show that rubric-guided meta-prompt refinement can improve APO-generated prompts." But the paper's own statistical analysis (Section 5.2.1) reports a 95% bootstrap CI of [−0.46, 4.70] for the average improvement over PromptWizard — an interval that includes zero and even a small *negative* effect. By the authors' own reported statistic, the null hypothesis that MPIR provides no average improvement over PromptWizard cannot be rejected at the 95% confidence level they themselves chose to report. A more parsimonious explanation of the data than "MPIR causally improves prompts" is: "MPIR and PromptWizard perform similarly on average, with task-to-task variation that happens to favor MPIR on this particular sample of 23 BBH tasks, run once each." The paper's own follow-on evidence — 16/23 task wins, but also a fairly even spread of both large gains (hyperbaton: +21.8) and real regressions (geometric_shapes: −8.5, tracking_shuffled_objects: −4.3, temporal_sequences: −6.2) — is entirely consistent with this more modest "similar performance, high task-to-task variance" story, and does not on its own adjudicate between it and the paper's stronger causal framing. The paper does contain the honest number; the problem is that the abstract, title, and conclusion are written as though the causal claim were established, and only a single hedged clause deep in Section 5.2.1 signals otherwise. A reader who reads only the abstract and conclusion — which is most readers — will come away believing something the authors' own statistics do not support.

### Issue List

#### CRITICAL

| # | Dimension | Issue Description | Location |
|---|-----------|-------------------|----------|
| 1 | Data-Conclusion Mismatch / Logic Chain Break | The abstract ("MPIR outperforms its PromptWizard baseline on 16 of 23 tasks") and Conclusion ("experiments... show that rubric-guided meta-prompt refinement can improve APO-generated prompts") both state the primary result as an established, unqualified fact. The paper's own bootstrap 95% CI for the average improvement is [−0.46, 4.70] (Section 5.2.1), which includes zero — i.e., the primary comparison is not statistically distinguishable from a null effect at the confidence level the authors themselves report. The main conclusion does not follow from the presented evidence at the level of confidence asserted. | Abstract (p.1); Section 5.2.1; Section 6 (Conclusion, first paragraph) |

#### MAJOR

| # | Dimension | Issue Description | Location |
|---|-----------|-------------------|----------|
| 2 | Alternative Paths Analysis | The paper does not consider or rule out the more parsimonious alternative explanation that MPIR and PromptWizard perform equivalently on average, with the observed task-level spread attributable to normal task-to-task and run-to-run variance rather than a genuine causal effect of the MPIR refinement stage (compounded by the single-run design R1 identifies). | Section 5.2.1; Section 6 |
| 3 | Confirmation Bias Detection (framing, not data selection) | The "16 of 23 tasks" statistic is reported prominently and repeatedly (abstract, 5.2.1) as supporting evidence for the improvement claim, while the CI that undercuts the same claim is reported once, in a subordinate clause ("although the interval includes zero, the overall trend is positive..."). This is not evidence fabrication or cherry-picking of data — the regressions are shown in Table 1, to the authors' credit — but it is a rhetorical asymmetry in how the same body of evidence is foregrounded versus hedged. | Section 5.2.1 |

#### MINOR

| # | Dimension | Issue Description | Location |
|---|-----------|-------------------|----------|
| 4 | Overgeneralization Check | "MPIR is particularly effective for structured, rule-based reasoning tasks" (Conclusion) is a reasonable qualitative read of Table 1/5.2.4's patterns, but is stated with more confidence than a 4-task illustrative pattern (hyperbaton, object_counting, boolean_expression, temporal_sequences — note temporal_sequences actually *regresses* under PromptWizard per Table 1, which weakens its use as a supporting example) can fully bear. | Section 5.2.4; Section 6 |

### Ignored Alternative Explanations/Paths
1. **Equivalence-with-noise, not improvement**: As argued above, the data are at least as consistent with "MPIR ≈ PromptWizard, with sampling/task variance" as with "MPIR > PromptWizard." The paper does not present this alternative or explain why the causal-improvement reading should be preferred over it.
2. **Free-rewrite baseline result could itself indicate GPT-4o-rewriting variance rather than rubric specificity**: The paper interprets the gap between Free Rewrite (57.15%) and MPIR (64.37%) as evidence the rubric drives the gain (a reasonable interpretation), but does not consider whether the free-rewrite baseline was itself only run once — if so, part of that ~7-point gap could likewise reflect single-run variance rather than a stable rubric effect. This does not overturn the finding, but it is an alternative explanation the paper does not address.

### Missing Stakeholder Perspectives
- Readers/downstream users who would make an adoption decision (e.g., "should my team invest engineering time integrating MPIR into our APO pipeline?") based on the abstract alone are not given the information they would need to correctly discount the confidence of that decision.
- Future researchers who might cite this paper's abstract as evidence that "meta-prompting refinement improves APO outputs" in a related-work paragraph, propagating the overclaim forward without themselves reading Section 5.2.1 closely.

### Unexamined Premise
The paper's framing throughout assumes that "improvement" is the right lens for evaluating MPIR's contribution at all, when the more defensible and equally interesting finding the data actually support is about *task-dependent behavior*: MPIR reliably helps on structured/rule-based tasks and is neutral-to-harmful on symbolic/spatial tasks (Section 5.2.4's own honest analysis). A paper framed around "when does heuristic-guided refinement help, and when does it not" — rather than "MPIR outperforms PromptWizard" — would be fully supported by the presented evidence and arguably more scientifically interesting, without requiring any additional experiments.

### Observations (Non-Defects)
- The condensation of Tables 2 and 3 to representative rows plus the average, with full data pointed to the project's GitHub repository, is a reasonable and clearly disclosed response to the stated 12-page limit; I independently checked whether the selected representative rows appear cherry-picked to favor MPIR and found they do not — Table 2's five rows include two tasks where all three APO methods regress after MPIR refinement (geometric_shapes, tracking_shuffled_objects), and Table 3's rows are explicitly the "tasks with the largest movement" in both directions (3 gains, 2 losses). This is fair reporting practice, not a defect.
- The authors' own Limitations paragraph already shows they are aware their gains are "relatively modest" — the CRITICAL finding above is about the abstract/conclusion's framing not yet matching that awareness, not about the authors having hidden anything.

---

# Phase 2: Editorial Synthesis & Decision

# Editorial Decision

## Manuscript Information
- **Title**: Meta-Prompted Instruction Refinement: Bridging Manual Prompting Techniques and Automatic Prompt Optimization
- **Target Venue**: Artificial Intelligence and Applications
- **Decision Date**: 2026-08-22
- **Review Round**: Round 1

---

## Decision

### Major Revision

---

## Reviewer Summary

| Reviewer | Role | Recommendation | Confidence |
|----------|------|---------------|------------|
| EIC | Editor, applied ML/AI systems background | Major Revision | 4 |
| Reviewer 1 | NLP evaluation methodologist | Major Revision | 5 |
| Reviewer 2 | Prompt-engineering/APO domain expert | Minor Revision | 4 |
| Reviewer 3 | Applied ML/MLOps engineer | Minor Revision | 3 |
| Devil's Advocate | Adversarial challenge | — (1 CRITICAL finding) | — |

---

## Consensus Analysis

### Points of Agreement (Consensus)

**[CONSENSUS-4]** (All four peer reviewers agree, independently and from different angles):
1. **The paper is well-written, clearly organized, and the underlying idea — a modular, post-hoc, rubric-guided refinement layer on top of an existing APO system — is sound and well-engineered.** (EIC Strength #1; R1 Strength #2–3; R2 Strength #1–3; R3 Strength #1–3)
2. **The breadth of generalization testing (3 APO backbones, cross-model transfer, 3 ablation studies) exceeds the norm for this subfield and is a genuine strength of the empirical design.** (EIC Strength #2; R1 Strength #1; R3 Strength #1)
3. **The average improvement over the primary PromptWizard baseline (+1.97 points) is real but modest, and the paper's confident framing of it in the abstract/conclusion does not yet fully reflect the uncertainty the authors themselves report in Section 5.2.1.** (EIC Weakness #1; R1 Weakness #1–2, corroborating from a pure-statistics angle; R2 does not raise this directly but does not contest it; independently and most sharply identified by the Devil's Advocate as CRITICAL)

**[CONSENSUS-3]** (3 of 4 peer reviewers; one dissent or non-overlapping focus noted):
1. **A concrete numerical/table-labeling defect exists in Section 5.3 that must be corrected.** EIC and R1 both flag this directly (R1 in full technical detail: two tables both numbered "Table 6," and an arithmetic inconsistency where two columns with identical row values report different averages). R3, reviewing from a non-NLP-methodology angle, did not independently catch this (outside R3's stated focus) but does not contest it once flagged. R2 focused on citation/positioning issues instead, consistent with the non-overlapping-lanes design of this review, and likewise does not contest the finding.

### Points of Disagreement

**Disagreement 1: Overall severity of the paper's issues (Major vs. Minor Revision)**
- **EIC and R1 view**: Major Revision — the combination of (a) the abstract/conclusion's overstatement of a statistically inconclusive headline result and (b) the Table 6 numbering/arithmetic defect are significant enough, together, to require a further verification pass before acceptance is appropriate.
- **R2 and R3 view**: Minor Revision — R2's concerns (literature positioning) and R3's concerns (practical-cost discussion, reproducibility caveat) are each addressable through additive text changes rather than re-analysis or re-experimentation, and do not on their own rise to Major Revision.
- **Disagreement type**: Severity disagreement (all four reviewers substantively agree on *what* needs fixing; they disagree on how much fixing "the statistical framing + table defect" cluster of issues requires).
- **Editor's Resolution**: **Major Revision**, following EIC/R1's assessment on this specific point.
- **Resolution Rationale**: Per `editorial_decision_standards.md` §2, expertise-based arbitration applies: statistical framing and table-integrity concerns fall most squarely within R1's declared area of primary competence (statistical rigor, Confidence 5) and are independently corroborated by the Devil's Advocate's CRITICAL finding. Per the IRON RULE governing this review process, a Devil's Advocate CRITICAL finding — here, that the paper's core claim is not supported at the confidence level the authors themselves report — categorically forecloses an Accept or Minor-Revision-only outcome, regardless of how the four peer reviewers' recommendations alone would combine. Additionally, per the standard decision matrix (`editorial_decision_standards.md`), a Minor/Minor/Major/Major pattern across four reviewers maps to Major Revision even before the DA finding is factored in — so both the peer-panel arithmetic and the DA override converge on the same outcome.

---

## Decision Rationale

Four independent reviewers and a Devil's Advocate examined this manuscript from non-overlapping angles: editorial fit and narrative coherence (EIC), statistical/methodological rigor (R1), literature positioning and domain accuracy (R2), practical/deployment feasibility (R3), and adversarial stress-testing of the core claim (DA). All four peer reviewers independently converged on the same underlying observation — that MPIR's average improvement over its primary PromptWizard baseline, while positive in direction (16/23 task wins), carries a 95% bootstrap confidence interval that includes zero, and that the paper's abstract and conclusion do not currently convey this uncertainty as clearly as the body text (partially) does. The Devil's Advocate elevated this to a CRITICAL finding: the title, abstract, and conclusion state the improvement as an established fact, which the authors' own reported statistic does not support at conventional confidence. Per this review process's governing rule, a DA CRITICAL finding forecloses an Accept-track decision regardless of how favorably the peer panel alone would have scored the paper.

Independently, R1 identified a genuine, correctable data-integrity issue: Section 5.3's ablation table is duplicated in numbering (two "Table 6"s) and contains an internal arithmetic inconsistency (two columns with identical per-task values reporting different averages) that currently undermines the verifiability of the validation-ablation claim ("removing validation drops average accuracy... a substantial decline"). This is a Major, not Critical, issue in isolation — but combined with the statistical-framing concern, it tips the overall panel recommendation to Major rather than Minor Revision, consistent with the standard decision matrix for a Minor/Minor/Major/Major reviewer split.

R2 and R3's Minor-Revision recommendations reflect that, in their respective lanes (literature positioning, practical feasibility), the paper's issues are real but addressable through additive revision rather than re-analysis. Combining all inputs, **Major Revision** is the appropriate decision: the paper's core contribution and empirical design are sound and worth publishing, but the abstract/conclusion framing must be brought into alignment with the paper's own reported statistics, the Table 5.3 numbering/arithmetic must be corrected and verified, and several literature-positioning and practical-discussion additions should be made before the manuscript is ready for this journal.

---

## Required Revisions (Must Fix)

| # | Revision Item | Source Reviewer | Severity | Section | Estimated Effort |
|---|--------------|----------------|----------|---------|-----------------|
| R1 | Rewrite the abstract, and the relevant sentences in the Conclusion, to explicitly convey that the average improvement over PromptWizard is directionally positive but not statistically distinguishable from zero at 95% confidence (CI [−0.46, 4.70]); do not state the improvement as an unqualified fact. | EIC (Weakness #1), R1 (Weakness #1–2), Devil's Advocate (CRITICAL #1) | Critical | Abstract; Section 5.2.1; Section 6 | 1 day |
| R2 | Correct the Section 5.3 ablation table: resolve the duplicate "Table 6" numbering (renumber sequentially so the in-text reference to "(Table 7)" in Section 5.3.3 points to an actual, correctly labeled table), and recompute/correct the "MPIR (with validation)" column average so it is consistent with its own displayed per-task row values. Re-verify all other reported averages in Tables 1–7 against their displayed rows as a final consistency pass. | R1 (Weakness #1) | Critical | Section 5.3 (Tables 6–7) | 1-2 days |
| R3 | Add repeated-trial evidence (at minimum, 3–5 seeds/training-example draws) for the primary PromptWizard-vs-MPIR comparison, and report between-seed variance alongside the existing between-task bootstrap, OR explicitly acknowledge the single-run design as a limitation and clarify exactly what source of variance the reported CI does and does not capture. | R1 (Weakness #2) | Major | Section 4.5; Section 5.2.1; Section 6 (Limitations) | 3-7 days (re-running experiments) or 0.5 day (limitation-only path) |

### Required Item Details

**R1: Recalibrate the paper's claim of improvement to match its own statistics**
- **Problem**: Abstract and Conclusion state MPIR "outperforms its PromptWizard baseline" as fact; Section 5.2.1's own 95% CI [−0.46, 4.70] includes zero.
- **Source**: Independently identified by EIC, R1, and the Devil's Advocate (CRITICAL).
- **Requirement**: Add explicit hedging language to the abstract (e.g., "a positive but not statistically decisive average improvement") and to the Conclusion's first paragraph; ensure the CI is referenced, not just the "16 of 23 tasks" statistic, wherever the headline comparison is summarized.
- **Acceptance criteria**: A reader who reads only the abstract and conclusion should come away with the same impression of the result's strength as a reader who reads Section 5.2.1 in full.

**R2: Fix the Table 6/7 numbering and arithmetic**
- **Problem**: Two tables labeled "Table 6"; the second's "MPIR (with validation)" column average (75.4%) is inconsistent with its own identical-to-"full rubric"-column row values (which average to 79.9%, per the "full rubric" column's own stated average).
- **Source**: R1 (Weakness #1), full technical detail in that report.
- **Requirement**: Renumber tables sequentially from Table 6 onward; recompute and correct the affected average(s); ensure Section 5.3.3's in-text reference to "(Table 7)" matches the corrected numbering.
- **Acceptance criteria**: Every table's own displayed "Average" row is independently verifiable by summing/averaging its own displayed data rows.

**R3: Address the single-run design**
- **Problem**: No repeated-seed/repeated-draw variance is reported anywhere; the only uncertainty estimate (task-level bootstrap) does not capture run-to-run variance.
- **Source**: R1 (Weakness #2).
- **Requirement**: Either add repeated-trial results, or explicitly and precisely scope what the existing bootstrap CI does and does not measure, in both the Results and Limitations sections.
- **Acceptance criteria**: A statistically literate reader can correctly state, after reading the paper, what source(s) of variance the reported CI accounts for.

---

## Suggested Revisions (Should Fix)

| # | Revision Item | Source Reviewer | Priority | Section | Expected Improvement |
|---|--------------|----------------|----------|---------|---------------------|
| S1 | Sharpen the differentiation from PE2 and PROPEL with 1-2 concrete contrasting sentences about *where in the pipeline* heuristics are applied. | R2 (Weakness #1) | P2 | Section 2.4 | Strengthens the originality/positioning argument |
| S2 | State explicitly the selection criterion used for Table 2's five representative rows and Table 5's five ablation tasks (Table 3 already states its criterion; extend the same practice). | EIC (Question #3), R1 (Weakness #3) | P2 | Section 5.1 caption; Section 5.3.1 | Pre-empts any concern about selective reporting; improves transparency |
| S3 | Add a compute/API cost discussion (approximate calls, tokens, or wall-clock time) for the full MPIR pipeline, and compare qualitatively to the cost of expert prompt engineering. | R3 (Weakness #1) | P2 | Section 4.5 or new subsection | Substantiates the "scalability" claim with concrete numbers |
| S4 | Add an explicit Limitations sentence acknowledging reproducibility risk from dependence on versioned, closed commercial model APIs. | R3 (Weakness #2) | P2 | Section 6 | Rounds out an already-strong Limitations section |
| S5 | Reconsider foregrounding the task-type-conditional finding ("helps on structured/rule-based reasoning, neutral-to-harmful on symbolic/spatial tasks") as a primary contribution rather than a secondary observation. | Devil's Advocate (Unexamined Premise) | P2 | Abstract; Section 6 | Offers a fully-evidence-supported alternative framing that strengthens rather than weakens the paper |
| S6 | Clarify two loosely-matched citations ([26] CogBench for step-back reasoning; [21] for the "steps matter more than length" claim) and deduplicate citations [11]/[17] (same paper, two reference numbers). | R2 (Weakness #2, Minor Issues) | P3 | Section 3.3; Section 2.2; References | Improves citation precision |
| S7 | Standardize task-name spelling ("causal_judgment" vs. "causal_judgement") throughout. | R1 and R2 (Minor Issues, both independently) | P3 | Tables 1 and 3; Section 5.2.3 | Copy-editing consistency |
| S8 | Add a one-paragraph plain-language bridge before Algorithm 2 explaining what PromptWizard produces and why MPIR needs it as input, for readers unfamiliar with the APO literature. | R3 (Weakness #3) | P3 | Section 4.2 | Improves accessibility for AIA's broader applied readership |

---

## Revision Roadmap

### Priority 1 — Structural Revisions (Estimated total effort: 2-9 days, depending on whether repeated-trial experiments are re-run)
- [ ] R1: Recalibrate abstract/conclusion language to match the reported (non-significant) confidence interval — remove unqualified "outperforms" framing
- [ ] R2: Fix the Table 6/7 duplicate numbering and the internal arithmetic inconsistency in the validation-ablation table; re-verify all table averages
- [ ] R3: Add repeated-seed variance evidence, or explicitly and precisely scope the existing CI's limitations in Results and Limitations

### Priority 2 — Content Supplementation (Estimated total effort: 3-4 days)
- [ ] S1: Sharpen PE2/PROPEL differentiation in Related Work
- [ ] S2: State the selection criterion for Table 2 and Table 5's representative-task subsets
- [ ] S3: Add compute/API cost discussion for the full pipeline
- [ ] S4: Add reproducibility-risk limitation regarding versioned commercial APIs
- [ ] S5: Consider foregrounding the task-type-conditional finding as a primary contribution

### Priority 3 — Text and Formatting (Estimated total effort: 0.5-1 day)
- [ ] S6: Clarify/replace two loosely-matched citations; deduplicate [11]/[17]
- [ ] S7: Standardize "causal_judgment" spelling throughout
- [ ] S8: Add plain-language bridge before Algorithm 2 for non-specialist readers
- [ ] Finalize corresponding-author email (currently a placeholder)

### Total Estimated Effort
- **Major Revision**: 6-8 weeks (standard journal allowance), comfortably covering the above even if repeated-seed experiments are re-run

---

## Revision Deadline
- **Recommended deadline**: 6-8 weeks from decision date, per standard Major Revision policy.
- **Basis**: Major Revision per `editorial_decision_standards.md` §1.
- **Extension policy**: If additional experimental runs (Priority 1, R3) require more time, authors should notify the editor at least 1 week before the deadline.

---

## Response Letter Instructions
Please respond to every item in the Required Revisions and Suggested Revisions tables individually, in the format of `templates/revision_response_template.md` (Reviewer comment → Author response → Change location), and provide a redlined or change-marked version of the revised manuscript alongside a clean copy.

---

## Closing

We thank the authors for a clearly written, carefully engineered contribution to an active and crowded subfield. The core idea is sound, the generalization testing is more thorough than the norm for this literature, and the Limitations section already shows real self-critical awareness. The central issue raised across four independent reviews and a dedicated adversarial pass is not that the work lacks value, but that the manuscript's framing of its headline result currently outruns what its own reported statistics support, compounded by a correctable data-integrity issue in the ablation tables. We are confident these are addressable within a single revision cycle and look forward to receiving a revised manuscript that brings its framing fully into alignment with its evidence.

Please note that the revised manuscript will undergo another round of review, with particular attention to whether the abstract/conclusion language and the corrected Table 6/7 have been satisfactorily addressed.

---

## Part 3: Reviewer Report Summary (Appendix)

### EIC Report Summary
- Recommendation: Major Revision | Confidence: 4
- Key Point: The abstract/conclusion state a statistically inconclusive result as established fact; fixable, but material.

### Reviewer 1 (Methodology) Summary
- Recommendation: Major Revision | Confidence: 5
- Key Point: 95% CI includes zero; single-run design with no seed variance; a genuine table-numbering/arithmetic defect in Section 5.3 needs correction.

### Reviewer 2 (Domain) Summary
- Recommendation: Minor Revision | Confidence: 4
- Key Point: Literature is well-covered and accurately characterized overall; differentiation from PE2/PROPEL needs sharpening; two citations are loosely matched.

### Reviewer 3 (Perspective) Summary
- Recommendation: Minor Revision | Confidence: 3
- Key Point: Practical cost/reproducibility risk of the multi-call commercial-API pipeline is undiscussed; core contribution otherwise sound and accessible.

### Devil's Advocate Summary
- CRITICAL finding: The core claim ("MPIR outperforms PromptWizard") is not supported at the confidence level the authors themselves report (95% CI includes zero); abstract/title/conclusion do not convey this. Per the IRON RULE, this finding forecloses an Accept-track decision independent of the peer panel's own scores.

*(Full reviewer reports appear in Phase 1 above; this appendix is a summary index only.)*
