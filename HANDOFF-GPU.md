# GPU machine handoff

Operational companion to `REBUILD.md`. That file says *why*; this one says *what to run*.

Nothing in the repo has been changed yet. Everything below is unstarted work.

---

## 0. Read first

- `REBUILD.md` §2 — two code defects that mean the published numbers do not measure what
  the manuscript claims. Both must be fixed before any rebuild run, or the rebuild
  reproduces them.
- `REBUILD.md` §4.0 — the go/no-go gate. Do not start the full grid before the 3-task
  pilot passes.
- `REBUILD.md` §5 — save per-example predictions. Non-negotiable; it is the reason the
  existing data is unusable for the analysis that matters.

---

## 1. Environment

Local machine used for the diagnosis was an Apple M5 / 32 GB (MLX-capable, no CUDA).
The rebuild moves to CUDA. Assumptions below are for a CUDA box; adjust as needed.

```bash
conda activate py313          # per global convention
pip install -r requirements.txt
pip install vllm              # not currently in requirements.txt
```

`requirements.txt` currently pins the closed-model client stack (openai, azure-identity,
google-genai, llama-index). None of that is needed for local inference but none of it
conflicts either — leave it until Phase 1 lands, then prune.

### Serving

vLLM is the right server on CUDA — it speaks the OpenAI chat-completions API, so Phase 1
becomes a base-URL change rather than a rewrite:

```bash
vllm serve <model-id> \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 8192 \
  --dtype auto
```

Model choice is a decision for whoever runs this, not something to inherit from this
document. Two constraints from the design:

- **The same model plays target, optimizer and rubric judge.** That is deliberate (see
  `REBUILD.md` §4) — it removes the "MPIR just distills GPT-4o" confound. It also means
  the model must be strong enough to critique a prompt usefully, which is the risk the
  §4.0 gate exists to test.
- **A second, unrelated family is needed** for the cross-model claim, replacing the Gemini
  experiment that reported a null (`REBUILD.md` §1.7).

Record for every run: model id, revision/build, quantization, `--max-model-len`, vLLM
version, GPU model and count.

---

## 2. Phase 1 — retarget inference (~1 day, no GPU needed)

**File:** `promptwizard/glue/common/llm/llm_mgr.py`, function `call_api()` at line 35.

Every model call in the repo funnels through this one function, and it already branches on
model name for Gemini (line 39). Add a local branch in the same shape:

```python
if os.environ.get("LOCAL_OPENAI_BASE_URL"):
    client = OpenAI(
        base_url=os.environ["LOCAL_OPENAI_BASE_URL"],   # http://127.0.0.1:8000/v1
        api_key="EMPTY",
    )
    model = model_name or os.environ["LOCAL_MODEL_NAME"]
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.0, seed=<seed>,
    )
    return response.choices[0].message.content
```

Nothing downstream needs to know. Note `Heuristic.__init__` already accepts
`meta_model_name` and `target_model_name` (`heuristic/core_logic.py:44`), so pointing the
meta-model and target model at the same local endpoint needs no signature change — but
`META_MODEL_NAME = "gpt-4o"` at `core_logic.py:25` is a hardcoded default that must go.

Also in Phase 1, and equally important:

**Widen the prediction record.** `GluePromptOpt.evaluate()` (`instantiate.py:124`) already
builds a per-example dict with question, predicted, actual and llm_output, and passes it to
`self.iolog.append_dict_to_chained_logs()` at line 152. Three problems: it writes only
under `logs/` (excluded by `.gitignore`), it omits task / condition / seed / example-index,
and the accuracy string is accumulated rather than the per-example flag being stored
cleanly. Fix all three — write one JSONL per (task, condition, seed) into a tracked
`results/predictions/` directory, one row per example with the fields listed in
`REBUILD.md` §5.

There is also a duplicated `call_api` implementation inside `demos/MPIR.ipynb` cell 4,
used by the LLM judge. It does not go through `LLMMgr` and will silently keep calling
OpenAI after Phase 1 unless it is also retargeted. Same for `demos/promptwizard.ipynb`.

---

## 3. Phase 2 — fix the validation split (~half day, no GPU needed)

This is defect `REBUILD.md` §2.1. Currently `demos/MPIR.ipynb` cell 9 splits
`shuffled[:25]` / `shuffled[25:]` and MPIR validates on the *same* 25 examples the
optimizer trained on.

Required: a **three-way** partition per task, from one seed, with indices recorded.

```
optimizer-train    25 examples   (PromptWizard / APE / ProTeGi optimization + ICL examples)
mpir-validation    25 examples   (MPIR candidate selection across the 7 rounds — disjoint)
test               remainder     (never touched during optimization or refinement)
```

Write the partition indices to disk per (task, seed) and load them rather than reshuffling
at run time — this is what makes the runs reproducible and the held-out claim true.

Plumbing note: `GluePromptOpt.__init__` derives the technique's dataset from the single
`dataset_jsonl` argument (`instantiate.py:99`), so `Heuristic` currently has no way to
receive a validation set distinct from the optimizer's training set. Either pass the
validation file separately, or construct the `Heuristic` with the validation partition
while the optimizer gets the train partition. Prefer the explicit route — a second
argument — over anything implicit.

Task inventory: 23 task groups in `BIG-Bench-Hard/bbh/`, 6,511 examples total. 250 per
task except `causal_judgement` 187, `snarks` 178, `penguins_in_a_table` 146. The three
`logical_deduction_*` and three `tracking_shuffled_objects_*` files are each reported as
one task group in the manuscript — check how they were pooled before re-running, since
Tables 3 and 4 report single rows for both.

While in this file, decide `llm_as_judge_eval` (defect `REBUILD.md` §2.2). The manuscript
§4.4 claims exact match. Recommendation: set it `False`, take exact match as primary, and
if a judge is kept for genuinely free-form tasks, log its verdict as a separate column so
both metrics are recomputable from the same predictions.

---

## 4. Phase 3 — the gate (~1 day, GPU)

Three tasks, one seed, one model, end to end: `hyperbaton`, `ruin_names`,
`penguins_in_a_table`. Full test partition, all conditions in the grid.

Reported deltas over PromptWizard in the current draft: +21.8, +5.8, +7.6. These are the
draft's largest wins, so they are where the effect is most likely to survive a weaker
optimizer and judge.

Record measured tokens/sec and wall-clock per condition — that converts the §4.1 grid into
a schedule.

**If the effect does not reproduce on all three, stop.** Switch to the reframe-only path
in `REBUILD.md` §3 and §7. Do not spend a week discovering it across 23 tasks.

---

## 5. Phase 4 — full grid (days, GPU)

23 evaluation runs per model family (`REBUILD.md` §4.1), then a reduced grid on the second
family. Ablations: nine variants on a six-task subset at one seed.

Make it resumable. Write predictions incrementally so an interrupted run continues instead
of restarting, and key the output files on (task, condition, seed) so a partial grid is
obvious from a directory listing.

Log with every result file: model id and build, quantization, seed, partition indices,
vLLM version, `validation_round`, and the full config used.

---

## 6. Open questions to resolve before Phase 4

1. **Which `llm_as_judge_eval` setting produced Tables 3–5?** If the judge was live, §4.4
   is wrong as written and the old tables need recomputing under exact match regardless of
   the rebuild. Ask Linh — this is not answerable from the repo.
2. **How were the three `logical_deduction_*` and three `tracking_shuffled_objects_*`
   files pooled** into the single rows reported in Tables 3 and 4? `BIG-Bench-Hard/bbh/`
   holds 27 files but the manuscript reports 23 task groups, and the pooling rule
   (concatenate then score, or score then average — which differ, since the sub-files have
   equal 250-example sizes here but the choice still needs stating) is not documented
   anywhere in the repo or the manuscript. Fix the rule, state it in §4.1, and apply it
   consistently in the rebuild.
3. **Which model families**, per §1 above.
4. **Whether the free-rewrite baseline is still meaningful** when target and rewriter are
   the same local model. It was designed to isolate MPIR's rubric from GPT-4o's general
   rewriting ability; with one model doing everything it becomes a cleaner control, but the
   manuscript's framing of it needs rewording.

---

## 7. What is already known and needs no re-derivation

- No per-example predictions survive anywhere. `results/Big_bench_hard.xlsx` (38×9) and
  `results/Albation.xlsx` (24×16) hold aggregate per-task accuracies only. This is why the
  example-level significance test in `REBUILD.md` §3.2 cannot be run on existing data.
- `AutoPromptTechnique_LinhNguyen.zip` in the repo root is a LaTeX template bundle
  (elsarticle .bst/.cls/.dtx, reference.bib, and the manuscript figures). It duplicates
  `manuscript/latex/` and is left untracked deliberately — do not commit it.
- `manuscript/latex/manuscript.tex` is the live build; `manuscript/build_latex.py` +
  `content_blocks.py` generate it, and `dump_plaintext.py` produces
  `manuscript_plaintext.txt`. Edit the generator, not the generated file.
- The `.env` route is gitignored. `demos/.env.example` documents the closed-model
  variables; add `LOCAL_OPENAI_BASE_URL` and `LOCAL_MODEL_NAME` there in Phase 1.
