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

## Update (later the same day)

Done since the first snapshot: V2F grid (44/45 cells; the 45th exceeds the 8192-token context by one token), controls (45 cells),
single-session rescoring (headline and protected sets), lenient-parser validation (400 responses labelled blind by an LLM
assistant, merged into `lenient_labels.json`), intermediate Qwen3-8B-AWQ refiner (`g6-8b`, V2 and V3, nine cells;
`mpir_study/mid_refiner.py`), new Sections 5.1.1, 5.3.1, 5.6.2, 5.10, 5.11, Appendix G, rewritten abstract, discussion and
conclusion, response-letter builder (`manuscript/build_response_round2.py`), table-numbering fix (new tables are lettered
R.1-R.5 so the auto-numbered Tables 1-17 stay aligned).
Reviews (fact-check, simulated reviewer, consistency) are in `manuscript/review_round2_*.md`; their findings are being fixed.
Still pending: 8B judge pass (`judge_eval --judge 8b --sample 150`), fixes from the reviews, final rebuild, highlighted PDF,
upload package, cover letter. Do not upload anything before the review reports are resolved.

## Final state (2026-11-02)

The second-resubmission package is built in `submission_revision2/upload_ready/` (see `README_UPLOAD.txt` there):
clean LaTeX source and PDF, response letter (`manuscript/build_response_round2.py`), highlighted PDF against the PDF the
reviewers saw, cover letter, supplementary archive. Rebuild with `logs`-independent steps: `mpir_study.revision2`,
`revision2_new`, `mid_refiner`, then `manuscript/build_latex_ieee.py`, `build_response_round2.py`, `make_round2_package.py`.

Review reports are in `manuscript/review_round2_*.md`. The fact-check found a bug in the V2F lenient comparison (non-accepted runs used
the plain P0 instead of P0 plus exemplar); it is fixed (`mpir_study/rescore.py::final_prompt`) and the corrected result is in the
manuscript: lenient 1 win, 3 losses, mean -0.4 points (null), strict 4 wins (mixed). Most blocking and numerical findings of the three reviews
were fixed; the longer list of style suggestions (items 44-67 of the consistency report: roadmap paragraph, repeated numbers across
Sections 5, 6.1 and 7, splitting the conclusion) was not applied.

Known limits stated in the manuscript: AI-assistant labels (lenient validation, audit), no second target family, no V3F or capable-refiner
run of V2F, 8B refiner changed one run per variant, no router access for 235B/Luna reruns, 44 of 45 V2F cells (context limit), 8B judge on a
150-prompt subset, no criterion ablation. Open decision for the authors: the portal history (Access-2026-44920 initial; Access-2026-54235 reviewed;
this document as second resubmission) should be checked against what the portal shows.

## Final (after the last proofread)

Final proofread and flow reviews were applied (`manuscript/review_round2_final_*.md`): corrected Table 8 caption, Section 5.9 wording
(fallbacks return P0, echoes are accepted), the 5.2 denominator, AI-use declaration, removal of review-history phrasing, restructured
Sections 5.2, 5.3.1, 5.6, 5.10, 6.1 and 7, new reader's guide and roadmap. The upload set is in `submission_revision2/upload_ready/`.
`submission_revision2/human_spotcheck_sheet.csv` (not for upload) lets the authors re-label 90 items to check the AI-made labels.

## Final addition (rubric-free control and protected strong refiners)

After the last proofread, V3F (rubric-free rewrite under the V2F protection) was run on 45 cells with the 1.7B refiner, and V2F and V3F
were run with Qwen3-235B and GPT-6 Luna on the nine Tier-1 cells (`g7-v3f`, `g7-api`; analysis in `mpir_study/protected_probes.py`,
results in `results/revision/rev2_protected_probes.json`). Findings: V3F lenient mean +3.8 points (11 wins of 45), V2F -0.4; paired V2F minus V3F
-3.7 [-6.8, -1.1] over 44 cells; strong refiners under protection +7.3 (235B) and +3.4 (Luna) lenient, with V3F similar. An independent check
(`manuscript/review_round2_v3f_check.md`) confirmed the numbers and led to softer causal wording (the variants also differ in chaining and
in the evaluation call). The package in `submission_revision2/upload_ready/` was rebuilt with these results.
The OpenRouter key used for these runs was passed by environment variable only and is not stored in the repository.
