# MPIR Rebuild — rejection diagnosis and plan

**Status:** AIOPEN-D-26-00563 desk-rejected by *AI Open* on 2026-09-03. Not sent to
external review. Editor's stated grounds: "insufficient novelty, lack of experimental
validation, or issues with the overall presentation."

**Interactive version of this document:**
https://claude.ai/code/artifact/477f4c7e-4401-41cc-bf46-c94b9818ee7e

This file is the durable record. It covers (1) why the manuscript was rejected,
(2) two code-level defects found while planning the rebuild that are more serious than
anything either prior review round caught, (3) the rebuild plan, (4) what must not slip.

---

## 1. Why it was rejected

Ranked by how much each alone would justify a desk reject. The first three are, in my
read, individually sufficient.

### 1.1 The abstract concedes the headline result is not significant — FATAL

> "the pooled average gain is directionally positive but not conclusively significant by
> paired statistical tests"

*Location:* Abstract, sentence 6. Corroborated in §5.2.1 (W = 83.0, p = 0.158; bootstrap
CI [−0.46, 4.70]; Cohen's dz = 0.31), §6.1, §7.

This is intellectually honest and it is also a self-administered rejection. An editor
triaging submissions does not forward a paper whose own abstract states the main claim
did not reach significance. The calibration that makes this a virtue in full reading is
invisible at triage speed; only the concession lands.

### 1.2 The entire experimental backbone runs on a deprecated model — FATAL

*Location:* §4.5, paragraph 1.

GPT-3.5-turbo is the target model for all 23 BBH tasks and for MPIR's validation stage,
with GPT-4o as the meta-model. In a 2026 submission this reads as "these results do not
speak to any LLM anyone currently deploys." Neither prior review round flagged this.

### 1.3 Our own tables show simpler methods winning — FATAL

*Location:* Table 3 vs. Table 4, §4.3.1.

| Condition | Avg. accuracy |
|---|---|
| MPIR + PromptWizard (**the flagship number**) | 64.38 |
| Expert-crafted few-shot CoT | 69.5 |
| APE, untouched | 70.16 |
| ProTeGi, untouched | 72.81 |
| **MPIR + APE** | **74.02** |
| **MPIR + ProTeGi** | **74.26** |

Whatever the intended framing, the numbers as presented say "skip MPIR and use the
benchmark authors' hand-written prompts." The cause is a choice, not a result:
PromptWizard is our *weakest* optimizer and we made it the primary baseline. See §3.1 —
this one reverses at zero cost.

### 1.4 Novelty is argued through a checkbox table — MAJOR

*Location:* Table 1, §2.4 closing paragraph, §2.5.

Ten methods scored on five binary columns, with MPIR as the only all-Yes row. Editors
read a table whose columns were evidently chosen so that one row wins as evidence of a
thin delta — particularly when §2.4 concedes in prose that PE2 and PROPEL differ from
MPIR in staging rather than mechanism.

### 1.5 The "held-out validation" claim is not accurate as written — MAJOR

*Location:* §4.5 paragraph 2 vs. Table 1 column 4. See §2.1 below — this is worse than
a wording problem; it is a real data leak in the code.

### 1.6 Single run, no variance, no replication — MAJOR

*Location:* §4.5, acknowledged §6.1.

One optimization run per condition at temperature 0. No seeds, no error bars, no variance
estimate anywhere. Every venue in this method's peer group (the ProTeGi / PE2 / PROPEL /
GEPA line) now expects multi-seed reporting.

### 1.7 Contributing factors

- **The cross-model experiment reports a null and calls it generalization.** Gemini
  baseline averages 92%; MPIR "maintains the same rounded average after refinement"
  (§5.2.3, Table 5). A baseline with no headroom cannot test a refinement layer.
- **Length reads as inflation.** ~14,500 words, 62 references, 5 appendices, and the same
  three caveats stated in full in §5.4, §6 and §7 — flagged by our own round-2 R1.

### 1.8 Note on the in-house review pipeline

`manuscript/review_report.md` and `review_report_v2.md` both returned **Major Revision**
on a manuscript that never reached a reviewer. The pipeline audited table arithmetic and
cross-references thoroughly and never asked the two questions an editor asks first: *is
the model current*, and *does the abstract claim a result*. Worth recalibrating before
relying on it again.

---

## 2. Code-level defects found while planning the rebuild

These are new findings, not in either review report. Both are in `demos/MPIR.ipynb`.
Both mean the reported numbers do not measure what the manuscript says they measure.

### 2.1 MPIR validates on PromptWizard's own training examples

The chain:

- `demos/MPIR.ipynb` cell 9: `train_samples = shuffled[:25]`, `test_samples = shuffled[25:]`
- cell 13: `GluePromptOpt(promptopt_config_path, setup_config_path, train_file_name, bbh_processor)`
- `promptwizard/glue/promptopt/instantiate.py:99`: `training_dataset = dataset[:seen_set_size]`
- `instantiate.py:108`: that dataset is passed to `Heuristic.__init__` as `self.dataset`
- `heuristic/core_logic.py:150`: `improve_prompt_with_score_check` scores every candidate
  by looping over `self.dataset`

So MPIR's validation set **is** the 25 examples PromptWizard used for optimization and
in-context example construction. `seen_set_size: 25` in
`demos/configs/heuristic/promptopt_config.yaml`.

There is no held-out validation stage. Table 1 nonetheless credits MPIR with held-out
validation and denies it to PROPEL, and §3.5 / §3 describe validation "on a held-out set."
Candidate selection across the 7 rounds (`validation_round: 7`) is therefore selecting on
data the upstream optimizer already fit to.

**This must be fixed before any rebuild run**, or the rebuild reproduces the same defect.

### 2.2 Accuracy is LLM-judged equivalence, not exact match

`demos/MPIR.ipynb` cell 6 sets `llm_as_judge_eval = True`, so `BBH.access_answer` routes
every correctness decision through `llm_eval()` (cell 4), which asks a model "compare them
and check they mean the same" and parses True/False. The exact-match branch
(`predicted_answer.lower() == gt_answer.lower()`) exists but is disabled. The judge runs
on `os.environ["OPENAI_MODEL_NAME"]` — the same model family being evaluated.

§4.4 says accuracy is "the proportion of model outputs matching ground-truth labels" and
argues explicitly for "exact-match accuracy" over "softer metrics." Delimiter extraction
is indeed programmatic, as §4.4 says — but the *comparison* is a free-form LLM judgement
at temperature 0, not string equality.

Two consequences: the metric described in the paper is not the metric that was run, and a
model judging its own output introduces exactly the self-preference bias the Devil's
Advocate reviewer raised and the free-rewrite baseline was designed to rule out.

**Before anything else:** confirm which setting produced the numbers in Tables 3–5. If
`llm_as_judge_eval = True` was live for the reported runs, §4.4 is wrong as written and
every table needs recomputing under exact match regardless of the rebuild.

---

## 3. Two fixes that cost no compute

### 3.1 Reframe around the strongest configuration

MPIR + ProTeGi (74.26) and MPIR + APE (74.02) both clear expert-crafted few-shot CoT
(69.5) — the same three BBH-author exemplars in both conditions, so the comparison is
legitimate. That is roughly **+4.5 points over the human ceiling the paper set for
itself**, and the current draft omits it while running a subsection titled "Remaining Gap
to Expert-Crafted Prompting" (§5.2.6) about a gap the best configuration closes.

Make the strongest optimizer the primary baseline. Report MPIR as a layer with per-optimizer
deltas: APE +3.86, ProTeGi +1.45, PromptWizard +1.99. Let PromptWizard sit where it
belongs — the low-baseline case, not the headline.

### 3.2 Move the statistics to example level

Collapsing ~5,900 test examples into 23 task means and running Wilcoxon at n = 23 is why
p = 0.158. The right analysis for paired binary outcomes is at the example level: McNemar
on matched correct/incorrect pairs, with a task-clustered bootstrap or a GLMM carrying
task as a random effect to respect the nesting. On that many matched trials a 2-point
difference has a real chance of clearing significance.

This cannot be run on existing data. `results/Big_bench_hard.xlsx` and `results/Albation.xlsx`
hold only aggregate per-task accuracies; no prediction-level output survives anywhere in
the repo. Hence §5.

Keep the task-level tests as a conservative secondary report; add the example-level test
as primary.

---

## 4. Rebuild plan

Constraint as set by the user: **no new API spend.** Closed models are out. Work moves to
GPU machines running open-weight models — see `HANDOFF-GPU.md` for the operational detail.

Framed correctly this is an upgrade, not a concession. Three standing objections dissolve:
the pipeline becomes reproducible by anyone with a GPU, closed-model drift stops
threatening replication, and using the same model as target, optimizer and rubric judge
removes the "MPIR just distills GPT-4o into a prompt" confound. The honest claim gets
stronger: a cheap refinement layer that works *without* a stronger teacher model.

What the rebuild is actually for: **current models** and **variance**. Everything else
follows.

### 4.0 GO / NO-GO GATE — run this before committing to the full grid

Run `hyperbaton`, `ruin_names` and `penguins_in_a_table` end to end — optimizer, MPIR
layer, full test set, one seed — on a single open-weight model. These are where the draft
reports its biggest wins (+21.8, +5.8, +7.6 over PromptWizard), so they are where the
effect is most likely to survive a weaker optimizer and judge. The measured throughput
also turns the schedule below from a guess into a plan.

**If the effect does not reproduce on all three, stop and switch to the reframe-only
path.** An 8B-class model critiquing its own prompts is genuinely weaker than GPT-4o doing
it, and the rubric gains may not survive the substitution. Better to learn that in a day
than across 23 tasks.

### 4.1 Condition grid (per model family)

| Condition | Role | Seeds | Runs |
|---|---|---|---|
| Zero-shot CoT | Manual floor | — | 1 |
| Expert few-shot CoT | Human ceiling | — | 1 |
| ProTeGi | Primary baseline | 3 | 3 |
| ProTeGi + MPIR | Primary result | 3 | 3 |
| APE | Cross-optimizer baseline | 3 | 3 |
| APE + MPIR | Cross-optimizer result | 3 | 3 |
| PromptWizard | Low-baseline case | 3 | 3 |
| PromptWizard + MPIR | Low-baseline result | 3 | 3 |
| Free rewrite | Rubric-vs-rewriting control | 3 | 3 |
| **Total** | | | **23** |

6,511 BBH examples across 23 task groups (`BIG-Bench-Hard/bbh/`, 250 each except
causal_judgement 187, penguins_in_a_table 146, snarks 178), less 25 optimizer-training
examples per task, less the new validation partition. Seeds vary the training sample and
the optimizer's own stochasticity; test decoding stays greedy, so the seed-independent
reference conditions run once.

Ablations — seven criterion removals plus generic-rubric and no-validation controls — run
as nine variants on a six-task subset at one seed, roughly 13,500 further generations.

### 4.2 Phases

| # | Phase | Effort | GPU |
|---|---|---|---|
| 1 | Retarget inference to a local OpenAI-compatible endpoint | ~1 day | no |
| 2 | Fix the validation split (three-way partition) | ~half day | no |
| 3 | Pilot + gate (§4.0) | ~1 day | yes |
| 4 | Full grid, checkpointed and resumable | days | yes |
| 5 | Analysis on prediction-level data | ~2 days | no |
| 6 | Rewrite around what the new numbers support | ~1 week | no |

Phases 1 and 2 are the only code changes and neither needs a GPU — do them first, on any
machine. `HANDOFF-GPU.md` specifies both.

Phase 5 deliverables:
- Primary: McNemar at example level, task-clustered bootstrap CIs
- Secondary: GLMM, task as random effect, seed as nuisance term
- Conservative: existing task-level Wilcoxon and sign test, retained
- Across-seed variance for every cell, not just means
- The ablation split the Devil's Advocate asked for: formatting-hygiene criteria (C1, C4)
  against reasoning-guidance criteria (C2, C3, C6)

---

## 5. The one requirement that cannot slip

**Save per-example predictions for every condition, every seed, every task.**

One row per example: task, condition, seed, example index, prompt hash, raw model output,
extracted answer, gold answer, correct flag, and the judge decision if a judge is used.

This is why the current data cannot be rescued by better statistics, and it is the cheapest
insurance in the plan. It buys the example-level significance test §3.2 depends on,
per-example paired analysis across conditions, the §5.2.5 failure-mode analysis on real
evidence rather than hand-picked cases, and a reviewer's ability to recompute any number
in the paper.

`GluePromptOpt.evaluate()` (`instantiate.py:124`) already accumulates the right fields per
example via `self.iolog.append_dict_to_chained_logs(result)` but writes them only under
`logs/`, which `.gitignore` excludes, and drops task/condition/seed/index. Phase 1 should
widen that record and write it to a tracked results directory.

Ship these files with the repository. "Every table in this paper is recomputable from
released predictions" is a stronger reproducibility claim than Appendix E currently makes,
and it is what a rigor-focused venue rewards.

---

## 6. Manuscript surgery (independent of results)

- [ ] **Rewrite the abstract to claim a result.** State what the evidence supports; move
      calibration to limitations. Honest is not the same as leading with the negative.
- [ ] **Cut ~14,500 words to ~9,000.** Collapse §5.4 into a forward pointer to §6; trim
      §7's limitations paragraph to one sentence.
- [ ] **Replace Table 1 with a mechanism comparison** — keyed on *how* methods differ, not
      five binary columns engineered so one row wins.
- [ ] **Thin Related Work.** Fourteen named systems across §2.3–2.5 is insider density.
      Keep the four MPIR is actually positioned against.
- [ ] **Retitle §5.2.6** — it describes a gap the best configuration closes.
- [ ] **Fix §4.4** to describe the metric actually used (see §2.2), or re-run under exact
      match and keep §4.4 as written.
- [ ] **Fix §3.5 / §4.5 / Table 1** on held-out validation (see §2.1).
- [ ] **Drop or reframe the Gemini experiment** — a 92% baseline has no headroom.
- [ ] **Reconcile the Wilcoxon statistic** — round-2 R1 recomputed W and got a 0.5
      discrepancy; a tie-handling convention, but fix it and state the convention.

---

## 7. Where to send it

**TMLR** is the strongest fit and the recommended target. It explicitly does not require
novelty or state-of-the-art results — only that claims are correct and that some segment
of the community would find them interesting. A rigorously established modest effect is
precisely what it publishes and precisely what an impact-screening journal will not.
Rolling submission, public reviews, no page limit.

Fallbacks in order: **Findings of ACL / EMNLP** once multi-seed results exist — the natural
peer venue by method, though it applies the strictest statistical norms of any option.
Then **Neurocomputing** or **Expert Systems with Applications**, both of which take applied
prompt-optimization work without hard novelty desk-screening.

Do not resubmit to another broad-scope AI journal with the current framing intact; the
failure mode that produced this letter would repeat.

**Do not submit anywhere before the §4.0 pilot has run.** If the effect does not survive
the move to open-weight models, the paper to write is a different one — an honest
measurement of how much a cheap post-hoc rubric layer buys, with the answer being "less
than the field assumes" — and that paper wants a different abstract and a different venue.
