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

These are new findings, not in either review report. The first two are in
`demos/MPIR.ipynb`; the third is a defect in the repository itself, found while
starting the rebuild. All three mean the reported numbers do not measure what
the manuscript says they measure, or cannot be traced to anything in the repo.

### 2.0 APE and ProTeGi have no implementation anywhere in the repository

Tables 3 and 4 report full results for "Iterative APE" and "ProTeGi" as baselines
and as APO backbones refined by MPIR (§4.3.2, §5.2.2), and §6.2's External
Validity claim leans on "three APO backbones" for generalization. But
`promptwizard/glue/promptopt/techniques/` contained exactly two optimizers
(`critique_n_refine` = PromptWizard, `heuristic` = MPIR) before this rebuild —
no APE or ProTeGi code exists in the working tree, in `.gitignore`d paths, or
anywhere in git history on any branch. The two tracked results files
(`results/Big_bench_hard.xlsx`, `results/Albation.xlsx`) hold only
PromptWizard/MPIR columns. Roughly a third of the manuscript's cross-framework
evidence has no traceable source in this repository. Neither prior review round
caught this, since both worked from the manuscript text and aggregate xlsx
files rather than asking whether the underlying code existed.

**Resolution (2026-09-03):** clean-room implementations of both, added under
`promptwizard/glue/promptopt/techniques/ape/` and `.../protegi/`, following the
published algorithms (Zhou et al. 2023; Pryzant et al. 2023) and matching this
repo's existing `PromptOptimizer`/`PromptPool` framework so they plug into
`GluePromptOpt` exactly like the other two techniques. See §8 for what shipped
and what remains a documented simplification.

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

---

## 8. Rebuild progress log

Kept as a running record so anyone picking this up mid-stream knows what shipped and
what is still assumed rather than verified.

### 2026-09-03 — Phase 1 & 2, and APE/ProTeGi implementations

**Phase 1 (retarget + prediction logging), done:**
- `llm_mgr.call_api()` gains a local-endpoint branch (`LOCAL_OPENAI_BASE_URL` /
  `LOCAL_MODEL_NAME`), checked before every closed-model path, with optional seed
  passthrough. Both notebooks' duplicated judge `call_api` now import this shared
  function instead of hardcoding OpenAI/Azure.
- Removed the hardcoded `META_MODEL_NAME="gpt-4o"` default in the heuristic (MPIR)
  technique.
- `GluePromptOpt.evaluate()` writes one tracked JSONL row per example under
  `results/predictions/<task>_<condition>_<seed>.jsonl` (§5's requirement) in addition
  to the existing gitignored iolog dump.

**Phase 2 (validation-split fix), done:**
- Fixed §2.1's leak: a new shared `demos/data_prep.py` produces an idempotent
  three-way partition (optimizer-train/mpir-validation/test) per (task, seed), used by
  every optimizer notebook instead of each one reshuffling its own split.
  `GluePromptOpt` gained an explicit `validation_dataset_jsonl` constructor arg so the
  heuristic technique scores candidates against the disjoint validation partition.
- `llm_as_judge_eval` now defaults to `False` (exact match primary), centralized in a
  new shared `demos/bbh_processor.py` (previously duplicated per notebook).

**§2.0 defect resolution — APE and ProTeGi implemented, done:**
- `promptwizard/glue/promptopt/techniques/ape/` and `.../protegi/`: clean-room
  implementations following Zhou et al. 2023 (forward-mode generation + iterative
  Monte Carlo resampling) and Pryzant et al. 2023 (textual-gradient edits + beam
  search), registered in `constants.py`/`utils.py` alongside the existing two
  techniques. Demo configs and notebooks (`demos/ape.ipynb`, `demos/protegi.ipynb`)
  mirror `promptwizard.ipynb`'s structure.
- **Documented deviation from Pryzant et al. 2023:** candidate selection scores every
  successor on a full minibatch each beam-search round rather than the paper's
  UCB-bandit sampling scheme (which exists to cut evaluation cost over larger candidate
  pools). Flagged in the class docstring in `protegi/core_logic.py`. Should not change
  which candidates survive at the pool sizes configured in
  `demos/configs/protegi/promptopt_config.yaml`, but has not been checked against the
  original ProTeGi codebase's reported numbers on any shared task.
- **Not yet done:** neither implementation has been run against a real LLM (local or
  closed). Correctness so far rests on code review and prompt-template inspection, not
  execution. Running the §4.0 pilot is what actually validates them.

### 2026-09-03 — Two-reviewer pass (code-reviewer + python-reviewer), fixes applied

Triggered a review of the full diff against pre-rebuild HEAD, on the theory that a
rebuild whose entire premise is "the previous numbers were wrong" should not ship its
own new measurement bugs unreviewed. Two independent agents found five real issues,
two of them CRITICAL/severe enough to have invalidated the §4.0 pilot outright. All
five are now fixed and verified with unit-level smoke tests (mocked LLM, no real API
calls) that specifically reproduce each bug and confirm the fix. Recorded here in
detail because this is exactly the class of mistake this document exists to catch.

1. **[CRITICAL, python-reviewer] APE/ProTeGi's search-time scoring never told the model
   the answer-delimiter format.** `eval_prompt` (used by `score_candidate` /
   `evaluate_on_batch` to score bare candidate instructions during search) had no
   `answer_format` substitution — only the *final* prompt (built once, after search
   ends, via `final_prompt.format(...)`) included it. Every candidate scored during
   search would therefore answer in free text, `access_answer`'s `<ANS_START>` /
   `<ANS_END>` extraction would find nothing, and every candidate would score ~0
   regardless of quality — meaning `top_n`/beam-search selection during search was
   driven by noise, not signal, even though the final reported accuracy would still
   look plausible. My own smoke test had missed this because its fake LLM was too
   lenient (always emitted the delimiter). Fixed by baking `answer_format` into the
   instruction text passed to `eval_prompt.format()` inside `score_candidate` /
   `evaluate_on_batch`, mirroring how `critique_n_refine`'s own `solve_template` already
   solves the identical problem. (First attempt at this fix added `{answer_format}`
   directly to the `eval_prompt` *template* — wrong, because that template is also used
   by the shared `GluePromptOpt.predict_and_access()` final-eval path, which never
   passes `answer_format` and already receives a fully-formatted instruction; that
   broke every technique's final evaluation with a `KeyError`. Reverted the template
   change, fixed the call sites instead.)
2. **[CRITICAL, both reviewers independently] `data_prep.py`'s three-way split was not
   actually seed-namespaced.** `train.jsonl`/`val.jsonl`/`test.jsonl` were shared across
   every seed for a task; only the audit-trail `partitions_seed{N}.json` was
   seed-specific. Preparing a second seed for an already-split task silently overwrote
   the first seed's data files, while the first seed's partition-index file still
   reported `"seed": 1` and passed the cache-hit check — so re-requesting seed 1 later
   would silently return seed 2's data under seed 1's name. Exactly the multi-seed grid
   in §4.1 would have hit this on task 2 of every 3-seed condition. Fixed: train/val/test
   filenames now include the seed; the partitions file also now records `num_examples`,
   `optimizer_train_size`, and `mpir_validation_size` so a config or source-data change
   invalidates a stale cache instead of silently reusing it.
3. **[HIGH, both reviewers] `seed` never seeded anything that generates variance.**
   `GluePromptOpt.evaluate(seed=...)` only stamped `seed` into filenames/log rows; nothing
   called `random.seed(seed)`, so APE/ProTeGi/critique_n_refine's internal
   `random.sample()` calls (demo selection, minibatch sampling) drew from whatever
   ambient global-RNG state existed at that point in the process -- not reproducible
   from, or actually varying with, the recorded seed. Fixed: `GluePromptOpt.__init__`
   now accepts an optional `seed` and calls `random.seed(seed)` before constructing the
   technique. All four demo notebooks now pass `seed=seed`. (LLM decode-level seeding
   for closed/local APIs remains partially wired -- see open item below.)
4. **[HIGH, code-reviewer] APE scored different candidates in the same selection round
   on different random minibatches.** Comparisons feeding `top_n` selection were not
   apples-to-apples. ProTeGi already did this correctly (one `eval_batch` per round,
   reused for every candidate). Fixed: APE now draws one shared scoring minibatch per
   round and reuses it for every candidate compared in that round, including re-scoring
   retained top candidates on the new round's minibatch so they stay comparable to
   newly resampled ones.
5. **[HIGH→addressed, code-reviewer; MEDIUM, python-reviewer] `LLMMgr.chat_completion`
   swallowed every exception and returned a fixed placeholder string**, which then
   fails delimiter extraction and gets recorded as an ordinary wrong answer in
   `results/predictions/*.jsonl` — indistinguishable from a genuine model mistake, and
   directly undermining §5's "every table recomputable from released predictions"
   claim. `tenacity` was already imported but unused. Fixed: `call_api` now retries
   (3 attempts, fixed+random backoff) and re-raises after exhausting retries; the
   swallow-to-sentinel in `chat_completion` is removed entirely, so a persistent
   failure now crashes the run loudly (to be resumed) instead of silently corrupting a
   result.

**Also fixed, lower severity:**
- `call_api`'s Azure-only imports (`azure.identity`, `AzureOpenAI`) now sit inside the
  branches that actually need them, after the local-endpoint check — a machine set up
  purely for local inference no longer needs `azure-identity` installed at all.
- `data_prep.py` now opens every file with explicit `encoding="utf-8"` (Windows defaults
  to the locale codepage otherwise) and asserts a task has more examples than
  `OPTIMIZER_TRAIN_SIZE + MPIR_VALIDATION_SIZE` before slicing, so an undersized task
  fails loudly at split time with a clear message instead of silently producing an
  empty test set.
- **Cross-task demo bug, caught by its own stale notebook output:** `promptwizard.ipynb`
  and `MPIR.ipynb` used different `dataset_to_run` values while `MPIR.ipynb` loaded a
  fixed `results/promptwizard.pkl` regardless of task — running the two "documented
  workflow" notebooks back-to-back would refine and evaluate the wrong task's prompt.
  `MPIR.ipynb`'s own saved output literally showed reasoning_about_colored_objects
  content despite its source reading `dataset_to_run = 'hyperbaton'`, confirming this
  had actually happened. Fixed: aligned both notebooks on the same demo task, and every
  pkl filename (`promptwizard`/`ape`/`protegi`) is now scoped by task+seed and
  constructed identically by both the producer and consumer notebook, so a future
  mismatch raises `FileNotFoundError` instead of silently loading the wrong prompt.

**Open items from the review, deliberately deferred (not correctness-blocking for a
single-seed pilot):**
- The debug-only `iolog`/`glue_logs` path shares one `experiment_name` across every
  grid cell, making the gitignored debug log undifferentiated once the grid runs at
  scale. Does not affect the tracked `results/predictions/*.jsonl` provenance files.
  Worth revisiting once the Phase 4 grid runner is built.
- `seed` is threaded to the local vLLM endpoint's decoding but not to the closed-model
  (OpenAI/Azure) paths -- low priority given the rebuild's local-only direction.
- `call_local_api` constructs a new `OpenAI` client per call rather than reusing one
  instance -- a performance nit, not a correctness issue, across what will be a large
  number of calls in the full grid.

### 2026-09-03 — Pilot scaffolding: manual baseline conditions + missing task configs

Model decision for the §4.0 pilot: **Qwen/Qwen3-1.7B**, serving as target, optimizer,
and judge simultaneously (per §4's design). This machine's GPU (RTX 5000 Ada, 16GB) is
the serving target.

Before this, only PromptWizard/APE/ProTeGi/MPIR had runnable notebooks — three of the
nine §4.1 grid conditions (Zero-shot CoT, Expert few-shot CoT, Free rewrite) had no
code at all, and `demos/configs/promptwizard/` was missing task-specific configs for
two of the three §4.0 gate tasks (`hyperbaton`, `penguins_in_a_table` — only
`ruin_names` existed). Added:

- `demos/configs/promptwizard/promptopt_config_hyperbaton.yaml` and
  `..._penguins_in_a_table.yaml`, mirroring the existing `ruin_names` config's
  hyperparameters with each task's own `task_description`.
- `demos/cot_prompts.py`: extracts the BBH authors' own one-line task description and
  full three-exemplar chain-of-thought prefix directly from
  `BIG-Bench-Hard/cot-prompts/<task>.txt` (stripping the canary/separator header), so
  Zero-shot CoT and Expert few-shot CoT share the same authoritative task framing
  instead of a hand-duplicated description.
- `demos/baselines.ipynb`: runs all three manual conditions for one task/seed.
  `GluePromptOpt` is constructed with the heuristic (MPIR) config purely as a
  lightweight vehicle for `evaluate()`'s logging and `data_processor`/`setup_config`
  plumbing -- no optimizer search runs; `BEST_PROMPT` is set directly per condition,
  each formatted through the framework's own `final_prompt` template (so
  `answer_format` is threaded consistently with every other condition).
  - **Free rewrite**'s exact methodology was never specified anywhere reproducible
    (this baseline is part of §2.0's "no source in the repo" defect) — operationalized
    here as: load the same PromptWizard-optimized prompt MPIR refines
    (`results/promptwizard_<task>_seed<seed>.pkl`), ask the pilot model once to improve
    it with no rubric/criteria, evaluate the result. Matches HANDOFF-GPU.md §6.4's
    description of the control's *purpose* (isolate MPIR's structured rubric from
    generic rewriting ability) but is a new operationalization, not a recovered one --
    flagged here so it isn't mistaken for a verified detail.
- Verified end-to-end against the **real** hyperbaton BBH data and real cot-prompts
  file (only the LLM call itself mocked): task description extraction, delimiter-format
  threading, answer extraction, and `results/predictions/*.jsonl` provenance all
  confirmed correct for all three conditions.

**Still open before the §4.0 pilot can execute:**
- `vllm` is not installed in the `py313` environment yet.
- No `.env` exists yet with `LOCAL_OPENAI_BASE_URL` / `LOCAL_MODEL_NAME` pointed at a
  served Qwen3-1.7B instance.
- Two pre-existing, now-superseded files remain tracked from before the seed-scoped
  split fix: `demos/data/hyperbaton/train.jsonl` and `test.jsonl` (the old, un-seeded
  naming scheme). Nothing reads them any more; left in place rather than deleted
  without being asked, but worth cleaning up.

**Still unstarted:** vLLM serving setup on the GPU machine, the §4.0
go/no-go pilot itself (executing all nine conditions × three tasks against Qwen3-1.7B
and checking the reported deltas reproduce), the full grid, and all manuscript surgery
in §6.

### 2026-09-05 — §4.0 pilot executed: NO-GO. Effect does not reproduce on Qwen3-1.7B.

vLLM served on WSL2 (native Windows vLLM has no CUDA wheels), Qwen3-1.7B as target,
optimizer and judge, thinking mode disabled. All 27 cells (9 conditions × 3 gate tasks:
`hyperbaton`, `ruin_names`, `penguins_in_a_table`, seed 42) completed with 0 failures.

**Six further real correctness bugs found live** (invisible to every mocked smoke test,
each confirmed against actual model output before being fixed) — recorded in the detail
this document exists to catch:

1. Qwen3's `<think>...</think>` trace and verbose style needed `enable_thinking: false`
   and an explicit `max_tokens` cap; without the latter, PromptWizard's long optimized
   prompt left too little of a small `--max-model-len` for the response to reach its
   answer tag, truncating mid-sentence and scoring 0/50.
2. `access_answer` did exact string match; Qwen3 sometimes answers a bare letter (`A`)
   against a `(A)`-formatted ground truth. Fixed with paren/whitespace-stripping
   normalization (`bbh_processor.py`'s `_normalize_for_comparison`).
3. `expert_few_shot_cot` scored exactly 0.0: the raw BBH exemplars end in unwrapped
   prose, so a model told separately to "wrap the final answer" wrapped its *entire*
   response. Fixed by appending a bare `<ANS_START>(X)<ANS_END>` tag after each
   exemplar's concluding sentence (`cot_prompts.py`), matching the format the
   manuscript's own Appendix A.3 already demonstrates.
4. MPIR's `Heuristic.improve_prompt` crashed (`IndexError`) when the meta-model's
   refinement response had no `<START>/<END>` wrapper at all — took down every
   remaining condition for that task via `run_grid.py`'s per-task exception handling.
   Fixed with an empty-match fallback that keeps the prompt unchanged for that round.
5. Same method, a *worse* variant with no crash to flag it: a well-formed
   `<START>/<END>` match whose wrapped content was the meta-model echoing its own
   `prompt_evaluation` rubric-template scaffolding verbatim, with no answer-format
   instruction — scored a whole condition exactly 0.0 silently. Fixed with a
   marker-string rejection check, but recognized as whack-a-mole against specific
   phrasings (see #6).
6. **The real fix, generalizing #4/#5:** `improve_prompt_with_score_check` initialized
   `best_score = float('-inf')`, so round 1's candidate always "won" regardless of
   actual quality — the two fixes above only caught contamination patterns matching a
   known string. Found live a *third* variant (the meta-model echoing APE's own
   induction-template opener, "I gave a friend an instruction...") that neither marker
   check recognized, scoring `hyperbaton/ape_mpir` exactly 0.0. Fixed properly: score
   the pre-refinement prompt as a baseline and only accept a round's candidate if it
   *strictly* beats that baseline — a general safety net independent of recognizing any
   particular failure text.
7. **Separately, a methodology bug (not a meta-model quality issue):** `run_grid.py`
   evaluated every `<optimizer>_mpir` condition through `gp_mpir` (built from
   `configs/heuristic/*`), silently swapping in Heuristic's own `eval_prompt` template
   instead of the base optimizer's. APE's and ProTeGi's templates differ from
   Heuristic's by trailing whitespace only (`"[Answer]"` vs `"[Answer] "`), but that's
   enough to flip a greedy decode's next token — confirmed via direct, repeated
   (5×, deterministic) API calls that the *same* rendered prompt reliably produces
   different outputs depending on which template wrapped it. `hyperbaton/ape_mpir`
   scored 0.07 through the wrong template on a prompt that is byte-identical to `ape`'s
   own 0.515-scoring prompt. Fixed: `<optimizer>_mpir` is now evaluated through the
   same `GluePromptOpt` instance as its base `<optimizer>` condition (only the
   refinement step itself still uses `gp_mpir`). PromptWizard's own template happens to
   be byte-identical to Heuristic's, so `promptwizard_mpir` results were never affected
   by this specific bug.

**Final pilot numbers** (`results/grid_summary.jsonl`, `results/pilot_analysis.json`,
one seed, exact match, all six live bugs above fixed before the numbers below were
accepted):

| Task | promptwizard → +mpir | ape → +mpir | protegi → +mpir |
|---|---|---|---|
| hyperbaton | 0.580 → 0.580 | 0.515 → 0.520 | 0.170 → 0.170 |
| ruin_names | 0.600 → 0.600 | 0.410 → 0.410 | 0.045 → 0.045 |
| penguins_in_a_table | 0.698 → 0.698 | 0.177 → 0.167 | 0.021 → 0.021 |

MPIR changed the base optimizer's accuracy in exactly 1 of 9 (task, optimizer) cells,
by +0.005 (one example out of 200) — every other cell is unchanged to three decimal
places, meaning `improve_prompt_with_score_check`'s baseline safety net (fix #6) almost
never found a refinement round that beat doing nothing. `demos/analyze_grid.py`
confirms this formally: pooled McNemar p=1.0 and GEE p∈{1.0, nan} for all three
base-vs-MPIR pairs, pooled accuracy differences of 0.0000 (ape: [-0.0104, +0.0050] 95%
CI), 0 or 1 of 3 tasks favoring MPIR by the conservative sign test.

**Decision, per §4.0's pre-committed rule ("if the effect does not reproduce on all
three, stop and switch to the reframe-only path"): NO-GO.** The effect does not survive
the move to a same-model-as-everything open-weight pilot on any of the three highest-
signal tasks from the original draft. Per §7: "the paper to write is a different one —
an honest measurement of how much a cheap post-hoc rubric layer buys, with the answer
being 'less than the field assumes'." Next: manuscript surgery per §6, targeting TMLR
per §7, **not** the full 23-task grid in §4.1 — running 20 more tasks cannot revive an
effect that failed to appear on the three tasks most favorable to it.
