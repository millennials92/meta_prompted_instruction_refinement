# Second revision (IEEE Access, Access-2026-54235): work-in-progress status

Snapshot of an unfinished revision. The manuscript in `manuscript/` does not yet build to a final PDF with the
second-revision numbers; do not upload anything from `submission_revision/` until this file says the revision is complete.

## Reviewer requests and where each is handled

| Comment | Handling | State |
|---|---|---|
| R1 major 1, 2 (defective treatment, no positive control, protection ineffective) | New variant `v2f` (`mpir_study/mpir.py`, `templates.py`): format- and example-preserving chained refinement, run on all 45 cells (`run_study --group g5-fix`). Positive control and bare-tag control in `mpir_study/controls.py`. Candidate audit in `revision2.py`. | grid running (21/45 cells at snapshot), controls 41/45 |
| R1 major 3 (lenient parser unvalidated) | `mpir_study/lenient_validation.py` (sample, blind labels, score). Needs the score cache to be complete. | code written, not run |
| R1 major 4 (rubric not isolated, V2 vs V3) | Paired per-cell V2 - V3 with intervals: `revision2.rubric_vs_free`, Table 9b in `part3_results.py`. Intermediate open-weight refiner: group `g6-8b` (Qwen3-8B-AWQ, V2 and V3, nine Tier-1 cells), not yet run. | paired analysis done, 8B not run |
| R1 major 5 (labels) | Title changed; "pre-specified" replaced by "protocol fixed in advance, with listed amendments"; Fig. 1 caption fixed; the -13.6 simulation wording still to be restated in 5.9 and the conclusion. | partly done |
| R1 major 6 (statistics, noise) | Exact Clopper-Pearson bounds (1/80: 5.8% one-sided, 6.8% two-sided; cells 1/45) in `revision2.acceptance_bounds` and Section 5.3. Single-session rescoring of the headline prompts: `mpir_study/rescore.py` (headline set done, protected set running); analysis in `revision2_new.rescore_summary`. | bounds done, rescoring analysis text pending |
| R1 major 7 (echo/fallback unvalidated) | Random sample of 40 echo and 40 fallback rounds labelled by an LLM assistant from excerpts (NOT a human): `results/revision/rev2_audit_samples.json`, `rev2_audit_labels.json`. Echo false positives 4/40; 25/40 fallbacks never present a refined prompt; none cut off by the length limit; 6/40 recoverable. Token ratio measured (4.36 chars/token, `mpir_study/tokens.py`). Text in Section 5.2. | done; a human spot-check is advisable |
| R1 major 8 (exemplar/judge controls) | Bare-tag control (`controls.py`); judge correlations without zero-accuracy candidates (`revision2.judge_without_format_zeros`, Table 10b). Independent 8B judge: `judge_eval --judge 8b`, not yet run. | partly done |
| R1 major 9 (external validity, cost) | Per-variant cost computed (`revision2.cost_per_variant`); contamination discussion and cost section still to be written. | pending |
| R1 minor 1-18, questions 1-9 | See response letter (to be rebuilt from `manuscript/build_response.py`). Done so far: 5 (66% vs 60%), 8, 11 (Appendix G, `part5b_appendix_g.py`), 12, 13 (slugs), 14, 15 (bibliography, `manuscript/reference_audit_round2.md` applied to `cas-refs.bib`), 4 (ties and noisy validation). Pending: 1, 3 (kept in place), 6, 7, 9, 10, 16, 17, 18. | partly done |
| R2 weakness 1 (small refiner) | Same as R1 major 1 and 4: V2F, intermediate 8B refiner, stronger-refiner probes. | pending experiments |

## Facts to keep in mind

- No OpenRouter key is available on this machine, so the Qwen3-235B and GPT-6 Luna runs cannot be repeated or extended.
- The earlier raw-output cache (`results/revision/score_cache/`, git-ignored) is absent; prompts needed for new analyses are re-scored in the 2026-11-02 serving session (vLLM 0.28.0, `--max-num-seqs 192`, `VLLM_WSL2_ENABLE_PIN_MEMORY=1`).
- Do not run `mpir_study.analyze` blindly: it would recompute older numbers from a cache that is missing. `results/revision/summary_numbers.json` holds the earlier numbers.
- Correct statement for the response letter: the "7%" bound in the earlier text was the two-sided value (6.8%) rounded up.
- The V3 "win" of +2.8 points belongs to the 1.7B refiner (ruin_names / ProTeGi), not to the 235B refiner.

## Remaining steps

1. Wait for `g5-fix`, `controls`, `rescore --set protected`; then `python -m mpir_study.revision2`, `revision2_new`.
2. Lenient validation: `lenient_validation sample`, label the sample blind (LLM labels, say so in the paper), `score`.
3. Start Qwen3-4B/8B-AWQ servers, run `run_study --group g6-8b`, `judge_eval --judge 8b`, write `rev2_mid_refiner.json`.
4. Write Sections 5.1.1, 5.3.1, 5.6.2, 5.7 additions, 5.10 summary table, cost section, discussion and conclusion edits; rewrite the abstract.
5. Rebuild the letter (`build_response.py`), cover letter, highlighted PDF (`highlight_diff.py`), upload package.
6. Final fact-check pass, then push.
