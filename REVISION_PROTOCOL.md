# Revision protocol (written before any revised experiment was run)

Date written: 2026-10-19. This file is committed to version control before the
revision experiments start, so the commit timestamp fixes the decision rules below.
It is a time-stamped, version-controlled plan; it is **not** an entry in a public
registry (OSF, AsPredicted, etc.), and the paper says so.

Context: the IEEE Access reviews of the first submission (Access-2026-44920) found
that (i) the seven refinement rounds of the pilot are not independent attempts, (ii) the
composition of the nine byte-identical outcomes was not reported, (iii) the statistical
tests were degenerate, (iv) task selection, single seed and "pre-registration" were not
substantiated, (v) the capability-gap interpretation was untested, (vi) strict answer
parsing floors some baselines, (vii) the gate is coarse and the free-rewrite control is
ungated, (viii) rubric and judge were never validated, (ix) the framing over-claimed.
This protocol specifies the new experiments that answer those points. What was already
known before writing it: the pilot results in `demos/results/grid_summary.jsonl`
(9 cells, seed 42, no accepted refinement).

## 0. What the pilot's earlier "pre-registration" actually was

The go/no-go rule of the pilot ("if the effect does not reproduce on all three tasks, stop and
switch to the reframe-only path") was written in `REBUILD.md` section 4.0, commit `7612aa0`
(2026-09-03 10:35 +07), before the pilot ran (2026-09-04/05). It was a version-controlled
plan, not a registry entry. Several pipeline and algorithm fixes were made while the pilot
was running (bugs 1-7 in `REBUILD.md` section 8); the paper lists them as deviations.

## 1. Algorithm fix

Algorithm 1 as run in the pilot (`legacy`, V0): every round calls Evaluate and Refine on
the same P0 at temperature 0, so the N=7 rounds are one request repeated. The revised
experiments use two corrected variants, both with the validation gate unchanged
(strict improvement over the baseline score of P0 on the 25-item validation split):

* V1 `indep`: every round refines P0 independently; the meta-model samples at
  temperature 0.7 (plain temperature sampling, no top-p) with a per-round seed, so rounds are genuinely distinct draws.
* V2 `chained` (the corrected MPIR, primary): round t refines the current incumbent
  (best validated prompt so far, initially P0) and the meta-prompt carries the history of
  every earlier candidate with its validation score and the judge's critique; temperature 0.7.
* V3 `free_gated`: the rubric-free rewrite control, same N rounds, same sampling, same gate
  as V1 (independent rounds), so MPIR-vs-free-rewrite differs in the rubric only.
* V2L: V2 with the original rubric text (which says "five criteria", "6 criteria", "X/25"
  while listing seven criteria) instead of the corrected 7-criteria/35-point text, as a
  sensitivity check on the prompt inconsistency found during the audit.

Every round is logged with an outcome category: `fallback` (no delimiters), `echo`
(evaluation-template phrase), `duplicate` (text identical to P0/incumbent or to an earlier
candidate), `validated` (scored on validation), plus validation score, baseline score,
judge total, and the full texts. Validation scoring is repeated 3 times per prompt.

## 2. Cells and seeds

* Tier 1 (pilot tasks): hyperbaton, ruin_names, penguins_in_a_table. These were chosen
  because the leaked original evaluation showed the largest gains there; the paper says so
  and treats them as a biased sample. Optimizer/split seeds 42, 43, 44; three optimizers
  (PromptWizard, APE, ProTeGi): 27 cells.
* Tier 2 (random tasks): six tasks drawn uniformly at random, without replacement, from the
  24 other BBH task files, `random.Random(20261019).sample(sorted(rest), 6)`:
  temporal_sequences, movie_recommendation, disambiguation_qa, geometric_shapes,
  salient_translation_error_detection, sports_understanding. Seed 42, three optimizers:
  18 cells.
* Total 45 base prompts P0. Target model Qwen3-1.7B, greedy decoding (temperature 0),
  max 1024 new tokens, thinking disabled. Same three-way split (25 train / 25 validation /
  rest test) as the pilot.
* MPIR sampling seeds: 3 per cell and variant.

Experiment groups:

* G1 main: V2 (self refiner Qwen3-1.7B) on all 45 cells, 3 MPIR seeds.
* G2 mechanism: V0, V1, V2, V3, V2L on the 9 Tier-1 seed-42 cells. All distinct validated
  candidates are also scored on the full test split ("oracle" scoring), so any acceptance
  rule can be evaluated offline.
* G3 capability: refiners Qwen3-4B-AWQ and Qwen3-8B-AWQ (target stays Qwen3-1.7B) with V1 on
  the 9 Tier-1 seed-42 cells (plus V2 where both servers fit in memory), oracle scoring.
* G4 judge validation: rubric totals from the 1.7B judge and from the 8B reference judge for
  P0 and all candidates; Spearman correlation with validation and test accuracy; judge
  repeatability across rounds.
* G5 noise floor: P0 scored 5 times on validation and test per cell; discordance of identical
  prompts; minimum detectable effect.
* G6 uncorrected-variant impact: legacy acceptance rule (`A_best = -inf`, first non-fallback
  candidate always accepted, later rounds strictly better) applied to the V1 candidate logs.
* G7 parsing: strict vs lenient answer extraction recomputed offline from raw outputs for every
  condition (pilot and new); one-format-exemplar intervention on APE/ProTeGi prompts.
* G8 acceptance-rule sensitivity (offline, from G2/G3 oracle data): strict, non-strict (ties
  accepted), repeated-scoring mean, and a larger gate (validation plus half of the test
  split, evaluated on the other half).

## 3. Outcomes and decision rules

Primary outcome per cell: paired difference in test accuracy, final MPIR prompt minus P0
(paired bootstrap, 10,000 example resamples, 95% percentile CI). A cell counts as a
"win" if the difference is positive and the CI excludes zero.

* D1 (does MPIR help on this model?): "effect reproduces" iff at least 15 of 45 cells
  (one third) are wins **and** the mean cell difference is at least +2.0 percentage points.
  "Null" iff at most 2 of 45 cells are wins and the mean cell difference lies within +/-1.0
  point. Anything else is reported as "mixed". Tests that cannot be informative when prompts are
  identical (McNemar, GEE, Wilcoxon on n=3) are not reported as inference; cell-level
  differences, counts and hierarchical-bootstrap intervals are descriptive.
* D2 (capability dependence): supported iff the mean cell difference with the 8B refiner
  exceeds that with the 1.7B refiner by at least +2.0 points **and** at least 3 of the 9 cells
  are wins with the 8B refiner. Otherwise the paper states the hypothesis was tested and
  not supported at these scales.
* D3 (gate-limited?): the gate is "limiting" iff the best oracle candidate beats P0 by at
  least +3.0 points on test in at least 3 of the 9 cells although the gated final prompt did not.
* All other analyses (rubric-accuracy correlation, lenient parsing, uncorrected-variant impact,
  noise floor, MDE) are descriptive.

## 4. Things that will not be done and will be stated as limitations

The GPT-4o / GPT-3.5-turbo configuration is not re-run (no paid API access in this study);
the original per-task numbers are reported from the saved spreadsheet with the leak flagged.
No human rating of rubric quality is collected; the judge is compared with a stronger
open-weight reference judge only. One model family (Qwen3) is used.

## 5. Amendments (dated; made after the protocol commit, before any result was inspected)

* 2026-10-20: the serving stack is vLLM 0.28.0 under WSL2 on one RTX 5000 Ada (16 GB), shared with
  other desktop workloads. Measured aggregate throughput is about 1.2k generated tokens/s, so the
  compute plan was reduced: oracle test scoring of every validated candidate is done for MPIR
  sampling seed 1 only (seeds 2 and 3 are scored on the validation split and, for the final prompt,
  on test); the main grid G1 records one validation score per candidate instead of three repeats
  (the three-repeat scoring is kept in G2/G3, where the acceptance-rule sensitivity analysis needs it).
* 2026-10-20: V2L (original rubric text) uses MPIR seed 1 only.
* 2026-10-20: the extra-seed and random-task base prompts (P0) are produced by the same optimizer
  implementations and hyperparameters as the pilot (no changes to APE, ProTeGi or PromptWizard code).
* 2026-10-20: refiner sampling uses temperature 0.7 without top-p (the protocol text said top-p 0.95);
  vLLM's top-p path was measured to be about 3x slower on this stack.
* 2026-10-20: the Tier-1 seed-42 cells are run only in G2 (which includes the V2 runs with oracle
  scoring); G1 covers the remaining 36 cells. V2L (original rubric text) is not oracle-scored.
