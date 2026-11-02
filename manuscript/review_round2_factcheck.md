# Round-2 factual review: manuscript vs result files

Scope: new or changed material of the second revision. Sources are in `results/revision/` (rev2_offline, rev2_new, rev2_mid_refiner, lenient_validation, controls, summary_numbers, rescore, judge_scores) and the code in `mpir_study/`. Numbers were recomputed independently where the underlying vectors or counts exist.

## Critical finding (read first)

The V2F lenient figures ("2 wins, 13 losses, mean -2.7", "lenient wins against P0 number 3") come from a bug in `mpir_study/revision2_new.py::grid_v2f`. For the 29 of 44 runs in which no candidate was accepted, the lenient "final" vector is built with `final_prompt(r)`, which returns the true P0 (`rescore.py`, line 26), not the loop's start prompt (P0 plus the bare-tag example). The lenient "vs_start" comparison for those cells is therefore P0+exemplar against P0, i.e. the exemplar effect, not zero. The strict side is unaffected (it uses `r["paired"]`). Evidence: in `rev2_new.json`, non-accepted cells show a nonzero lenient `vs_start` diff (for example disambiguation_qa/APE +0.085 with 77 discordant of 200) while the strict diff is exactly 0 and the lenient `vs_p0` diff is exactly 0.

Recomputed with the final prompt set to the start prompt for non-accepted runs (script in scratchpad, uses the same cached vectors):

- lenient vs own baseline: 1 win, 3 losses (hyperbaton/PW s42 -7.0, hyperbaton/ProTeGi s42 -8.0, ruin_names/PW s44 -11.0; win: temporal_sequences/PW +7.5), mean -0.43 points over 44 cells. All 3 losses and the 1 win are among the 15 accepted runs; the other 11 accepted runs are indistinguishable from zero.
- lenient vs unextended P0: 13 wins, 3 losses, mean +3.9 points.
- Consequence for D1 on V2F: strict = 4 wins, mean +0.7 (mixed, unchanged); lenient = 1 win (at most 2), mean -0.4 (inside +-1.0), so the lenient outcome is "null", not "mixed".
- The headline "13 significant lenient losses" and "mean -2.7" appear in the abstract, 5.3.1, 5.10, Table R.4, Table 11, 6.1 and 7 and must all be replaced. The narrative that "a format-preserving treatment accepts more but hurts under lenient scoring" weakens to "accepts more, 1 win and 3 losses of 15 accepted runs, the rest indistinguishable from noise".

## Verdict counts

| Verdict | Count |
|---|---|
| WRONG | 17 |
| UNSUPPORTED | 11 |
| CANNOT VERIFY | 5 |
| OK | 31 (grouped rows; each covers several individual numbers) |

## WRONG and UNSUPPORTED items

| # | Location | Claim | Verified value | Verdict | Suggested fix |
|---|---|---|---|---|---|
| 1 | Abstract | V2F "won in 2 cells and lost in 13 under a lenient metric" | Bugged. Corrected: 1 win, 3 losses (all among 15 accepted runs) | WRONG | "won in 1 cell and lost in 3 (mean -0.4 points)"; rerun `grid_v2f` with `start_prompt` for non-accepted runs |
| 2 | 5.3.1 V2F paragraph | "2 cells under lenient scoring, with 13 significant lenient losses (mean -2.7 points)" | 1 win, 3 losses, mean -0.43 | WRONG | Replace; also update the next sentence |
| 3 | 5.3.1 V2F paragraph | "the lenient wins against P0 number 3" | 13 wins, 3 losses, mean +3.9 (non-accepted runs are really P0+exemplar, whose lenient gain over P0 counts) | WRONG | Replace with 13 (mostly the exemplar effect) |
| 4 | 5.3.1 V2F paragraph | D1 for V2F gives "mixed" on lenient scoring | Rule: 1 win (<=2) and mean -0.4 (within +-1.0) gives "null" | WRONG | State strict mixed, lenient null |
| 5 | 5.10 text, Table 11 row "D1, V2F", Table R.4 row "V2F preserving" | "lenient: 2 wins, 13 losses, mean -2.7 ... Mixed"; "lenient losses 13" | As above | WRONG | Replace; Table 11 outcome "Mixed (strict) / Null (lenient)" |
| 6 | 6.1 paragraph 1 | "2 under lenient scoring, with 13 significant lenient losses, so the treatment can now succeed and still does not help" | 1 win and 3 losses; claim "still does not help" survives for lenient, but the "13 losses" evidence does not | WRONG | Reword to the corrected counts |
| 7 | Section 7 | "won in 2 cells and lost significantly in 13 under a lenient metric" | 1 and 3 | WRONG | Replace |
| 8 | 3.2 (V2F description) | "V2F is run on all 45 cells with one MPIR seed" | 44 cells (rev2_new: runs = 44); 5.3.1, Table R.4 and 6.3.1 say 44 | WRONG | "on 44 of the 45 cells" |
| 9 | Appendix F, amendment 10 | "variant V2F on all 45 cells" | 44 | WRONG | "44 of 45 cells (one request exceeded the context window)" |
| 10 | 5.5 | "and {len(l3)} cell(s) in which the gate accepted a significantly harmful rewrite" | Unrendered template placeholder. Data: 1 cell (hyperbaton/PW -9.3, CI [-14.7, -4.2]); the +2.8 ruin_names/ProTeGi win has CI [+1.0, +4.7] | WRONG | Replace by "1 cell" |
| 11 | 5.1.1 | "precision is still 98.4% (upper bound on the false-positive rate 5% at 95%)" | `fp_upper_95` is Clopper-Pearson on fp/(tp+fp)=2/124, i.e. an upper bound on 1-precision (error share among accepted). The false-positive rate fp/(fp+tn)=2/74 has upper bound 8.3% | WRONG | "upper 95% bound on the share of accepted responses that are wrong: 5.0%" (or give 8.3% for the FPR) |
| 12 | 5.1 and 5.3.1 | Range of exemplar gain "-45.0 ... the negative end being APE's degenerate hyperbaton prompt"; 5.3.1 "large losses on APE's degenerate hyperbaton prompt" | Hyperbaton/APE gains +28.5, +23.5, +0.5. The losses are disambiguation_qa/APE (-45.0, an induction-template prompt) and ruin_names/APE s43 (-41.0, a short instruction); hyperbaton APE is never negative | WRONG | Name disambiguation_qa/APE and ruin_names/APE |
| 13 | 5.3.1 vs 5.1 | Full exemplar gain "+20.0 and +42.8" (5.3.1) vs "+19.5 and +42.3" (5.1) for the same intervention; wins 10 vs 9 for APE | Both exist (controls.json: 0.1997/0.4282; exemplar_intervention: 0.1946/0.4226), from different scoring sessions | WRONG | Say they are two scorings, or use one set |
| 14 | Appendix C item 12 | "92-95% of validated 1.7B candidates lack it" | share_without_tag: V1 92.7%, V2 95.5% (abstract, 5.2 and 6.1 say 93-95%) | WRONG | "93-95%" |
| 15 | Table R.3 caption | "The 8B judge is a local Qwen3-8B-AWQ model that is not one of the reported refiners" | Table has no 8B rows; Qwen3-8B-AWQ IS a reported refiner (5.6.2, Table R.4). judge_scores.json holds 8b judge scores that are not reported | WRONG | Drop the sentence or add the 8B judge rows and correct "not a refiner" |
| 16 | Section 7 | 235B strict gains "do not hold under lenient scoring" | Hyperbaton/PW +30.0 holds under both (and Luna's +27.0, 8B V3's +17.0). Intro says "most of them do not hold", which is right | WRONG | "most of them do not hold" |
| 17 | 5.11 | "no accepted improvement in 79 of 80 V2 runs" | The single accepting run lowered test accuracy (cell mean -0.6, Table 7). There are 0 improvements in 80 runs | WRONG (minor) | "no improvement in any of 80 runs (one accepted candidate, which lowered accuracy)" |
| 18 | Abstract, 5.3.1 | "almost all of it the tag convention ... the worked reasoning adds at most a few points" | APE: bare tag +19.4 of +20.0 (97%). ProTeGi: +34.2 of +42.8 (80%); worked reasoning adds 8.6 points on average and up to about 37 points in single cells (penguins/ProTeGi s43 +75.0 vs +37.5 bare) | UNSUPPORTED (ProTeGi) | "about 97% (APE) and 80% (ProTeGi) of the strict gain is the tag; the worked reasoning adds about 0.6 and 8.6 points" |
| 19 | 5.6.1 | "A refiner that does not use the rubric therefore reaches gains of the same kind, so the rubric is not what produces them" | Table R.2: V2 beats V3 with CI excluding zero in 2 cells (hyperbaton/ProTeGi +24.0, penguins/ProTeGi +11.5), V3 beats V2 in 1 (ruin_names/ProTeGi -14.0); mean +2.7 [-3.5, +9.7]. V3 wins in only 2 of the 4 V2 winning cells | UNSUPPORTED | "so the rubric is not shown to be what produces them" (the next paragraph already says this) |
| 20 | Abstract, Table R.4 (235B row) | 235B wins are "mostly format repairs that a rubric-free rewrite also obtains" | Format-repair half supported (3 of 4 wins fall under lenient: -9, -27, -19 on the ProTeGi cells). "Also obtains" holds for 2 of 4 winning cells (HPW, RPro); in HPro V3 gets +5.0 against +29.0 and in PPro 0.0 against +11.5 | UNSUPPORTED (partly) | "a rubric-free rewrite reaches wins in 2 of the same 4 cells" |
| 21 | 5.6.2 last sentence | "the picture is the same at three scales: gains ... are mostly repairs of the answer format that a rubric-free rewrite also makes" | 8B: V2's only win is hyperbaton/ProTeGi (+16.0 strict, +1.5 lenient, a format repair); V3's only win is hyperbaton/PW (+17.0 strict AND lenient, so not a format repair). Different cells; paired V2-V3 is +16.0 and -17.0 in them | UNSUPPORTED | Say that at 8B the two variants each win one different cell, one of them under both metrics |
| 22 | 5.8 (after Table R.3) | "among usable candidates the rubric score says very little about accuracy"; "weak association ... partly the rubric rewarding candidates that had kept the format" | Median of 8 within-cell Spearman values (0.13) over 74 and 82 candidates, about 9 per cell, no interval, range restriction after removing zeros. 0.13 and 0.27 cannot be told apart at this n. The causal "rewarding" reading is not tested | UNSUPPORTED | "the median within-cell correlation falls from 0.27 to 0.13; at about 9 candidates per cell we cannot say whether any signal remains" |
| 23 | 5.3.1 and abstract | Positive control "accepted a known-good candidate in 29 of 45 cells" | The 5 PromptWizard acceptances are not known-good: test was measured for 2 of 15 PW cells; in hyperbaton/PW s42 the exemplar was accepted (val .72 to .76) and lowered test (.57 to .54). The controlled figure is 24 of 26 (APE/ProTeGi cells where the exemplar helps) | UNSUPPORTED | Report 24 of 26 as the headline; 29 of 45 includes 5 noise acceptances |
| 24 | 5.3.1 | "it never accepted the exemplar where the exemplar did not help on test (0 cases)" | True within APE and ProTeGi (0 of 4 and 0 of 0). For PW the test effect was not measured for most cells and one counter-case exists (above) | UNSUPPORTED (scope) | Add "among APE and ProTeGi cells" |
| 25 | 5.3.1 "What this shows" | V2F "behaves like a noisy search over rewrites that the gate cannot separate from noise" | With the uncorrected 13-vs-2 split, noise alone is rejected (two-sided sign test on 13 of 15 gives p = 0.007). With the corrected 1 win, 3 losses of 15 accepted, "noise" is plausible. Acceptance (34%) against V2 (1.25%) is numerically right, but acceptance is not improvement | UNSUPPORTED as written | Re-state after the fix; do not cite acceptance as evidence the treatment "can succeed" |
| 26 | 5.2, "What the candidates keep" vs paragraph before | "93% ... and 95% ... lack it [the format line]" and, next paragraph, "none keeps the answer-format line" | Two measures: share without "ANS_START" in the scored candidate (93%/95%) vs share containing the exact format sentence in the raw reply (0%). Both right, but the reader sees 93-95% and 100% for the same thing | UNSUPPORTED (ambiguous) | Say "none contains the exact line; 93% and 95% contain no answer-tag instruction at all" |
| 27 | 5.9 | "-13.6 ... prompt changed in 4 of 9 cells ... first-round outcome is a fallback or echo in 4 cells, where the legacy rule returns P0" | Simulation matches (-13.6, 4 changed, 4 worse). But one changed cell (hyperbaton/PW, delta -8.5) has an echo first round and changes in a later round through the gate, so it would also occur under the repaired loop (-0.94 mean). Defect-attributable loss is about -12.7 | UNSUPPORTED (minor) | Note that -0.9 of the -13.6 arises with the repaired loop |
| 28 | 6.1 paragraph 1 | "about 60% of rounds never produced a usable candidate" | 5.2 and 5.11 now use 66% (with duplicates); 60% counts fallbacks and echoes only (337 of 560) | UNSUPPORTED (inconsistent) | "60% (66% counting duplicates)" |

## CANNOT VERIFY

| # | Location | Claim | Note | Verdict |
|---|---|---|---|---|
| 29 | 5.2 audit | "Over all fallbacks, 230 of 236 contain the opening delimiter ... median 631 tokens (p90 986); only 2 of 1498 refinement responses within 10%" | 230/236, median 2750 chars (631 tokens), p90 4297.5 chars (986) and 2/1498 are in the file. But 236 covers only V1, V2, V1P, V2P (353 fallbacks exist across all 1.7B variants including V0, V2L, V2F). 1498 equals all 1.7B-refiner rounds (V0+V1+V2+V2L+V3+V1P+V2P+V2F), a different population. Rewrite "all fallbacks" as "the 236 fallbacks of V1, V2, V1P and V2P" | CANNOT VERIFY (scope mismatch) |
| 30 | 5.3.1 | geometric_shapes/PW "exceeds the 8192-token context window by one token" | Only the absence of the cell is verifiable (44 runs) | CANNOT VERIFY |
| 31 | 5.1.1, 5.2 | AI-assistant labels (400 and 80) | Counts recomputed from the label files and match; the labels themselves were not re-judged | CANNOT VERIFY |
| 32 | 5.7 "Mixed serving settings" | one session on 2026-11-02, twice each | Rescored values are present, the session is not logged | CANNOT VERIFY |
| 33 | 4.1, 4.3, Appendix F | commit hashes, dates, router behaviour, 1.2k tokens/s throughput, "no round is an echo" for hosted refiners beyond counts | Not in the result files | CANNOT VERIFY |

## OK (verified against the result files)

| Location | Claim | Verified value |
|---|---|---|
| Abstract, 5.3, 6.1 | 1 of 80 runs; exact upper bound 5.8% (6.8% two-sided); cells: 1 of 45, 10.1% (11.8%) | rev2_offline acceptance_bounds |
| 5.3 | mean -0.04, hierarchical interval [-0.16, +0.00]; 0 of 45 wins; PW 1 of 28, APE 0 of 31, ProTeGi 0 of 21 | summary_numbers main_grid, grid_by_optimizer |
| Table 7 | hyperbaton/PW 1 of 7 runs, -0.6 | -0.58, 7 runs |
| 5.3.1 positive control | 29 of 45 (ProTeGi 15, APE 9, PW 5); 24 of 26 APE/ProTeGi cells where the exemplar helps | controls summary |
| 5.3.1 bare-tag control | +19.4 and +34.2 vs +20.0 and +42.8; excludes zero in 10 of 15 and 15 of 15 | controls summary |
| 5.3.1 V2F | 15 of 44 (34%); 308 rounds, 48% fallback/echo; 92 validated, 20 better, 60 worse; raw keeps line 22.5%, header 63.8%; strict 4 wins mean +0.7; 27 strict wins vs P0, mean +22.6 | rev2_new grid_v2f |
| 5.2 / Table 6 | all 10 rows (rounds, F/E/D/validated, better/equal/worse, accepted, runs changed, distinct per run) | summary_numbers composition |
| 5.2 | 337 of 560 = 60%; 66% with duplicates; 406 of 749 | 177+160; 0.655; 200+206 |
| 5.2 | V1P 30%, V2P 27%; 5 of 36 validated beat baseline; V2L 63% F/E | 21/70, 15/56, 5, 40/63 |
| 5.2 "What the candidates keep" | 7.5% (shown 8%) and 7.2% bare-tag; 15 of 45 base prompts; 48 of 67 and 19 of 32 | candidate_format_audit |
| 5.2 "Why the protected candidates" | 40/31% sentence-in-tag; 20/59% no tag; 40/9% bare; 3 of 5 and 1 of 15 with example | protected_failure_modes |
| 5.2 audit labels | echo FP 4 of 40, 10%, upper 21%; fallbacks 25 / 10 / 5 other; 6 of 40 recoverable (15%); 77 of 1498 judge calls within 10% | audit_labels, audit_samples |
| 5.1.1 / Table R.1 | all 12 cells (tp/fp/fn, precision, recall, agreement); 201 rejected, 52 correct, lenient accepts 44 of them and 3 wrongly | lenient_validation |
| 5.4 | -27.0 (41), -15.2 (22), 0 of 9 and 0 of 8; -24.9 (20); -8.8 (33); -4.0 (58), best +4.0, 4 cells; Spearman 0.92 (n 275) and 0.37 (n 27) | oracle, val_test_corr |
| Table 8 | all 16 rows (cells accepting, mean delta over all cells, better/worse) | recomputed from acceptance_rules |
| 5.5 | gated rewrite 16/27, 2/27, 0/27; +17.0 and +2.8 wins, -9.3 loss, strict +0.7, lenient 0 wins 2 losses -1.8; V1 -0.6, V2 0.0; pilot rewrite +2.0, -19.0, -16.7 | capability, capability_lenient, Table 5 |
| 5.6 / Table 9 | all rows (validated share, val better, runs changed, wins, strict and lenient means) | capability, capability_lenient |
| 5.6 | 235B +30.0, +29.0, +12.0, +11.5; mean +8.7; seed 2 4 wins +12.3; lenient 1 win 3 losses -3.2; seed 2 lenient +0.9, 1 win, 0 losses; Luna 1 win +27.0 mean +3.0; 89% and 86%; 58.0 to 49.0 | summary_numbers |
| 5.6.1 / Table R.2 | all cell values, mean +2.7 [-3.5, +9.7], 4 / 3 / 2, 3 CIs exclude zero, seed 2 +6.3 | rubric_vs_free |
| 5.6.2 | V2: 63% reach validation (37% F/E), 1 better, 1 of 9 changed, strict +1.8, lenient +0.2; V3: strict +1.9, lenient +1.9, 1 win; paired -0.1 [-5.7, +5.3], 1 / 1 / 7 | rev2_mid_refiner |
| 5.7 | 1.4 SD, 3.3 range, 6.5% (max 16.6%), MDE 5.0 (6.3 to 8.9), pilot vs now 2.6 / 7.0 | noise, pilot_vs_rescore |
| 5.7 "Mixed serving settings" | +8.7 to +9.2, +12.3 to +11.6, +6.0 to +6.3, +3.0 to +3.3; counts 4/4 and 3/3; max shift 4.2; flip 7.1% (24.5%); 45 prompts = 9 base + 36 finals | rescore |
| 5.8 / Table 10 | 93%, 27.9, 9.1, 0.28/0.27; 20.7, 8.2, 0.27/0.27; agreement 0.38; +7.2; SD 4.7 | judge_validity, judge_repeat |
| Table R.3 | 196, 62%, 0.27, 74, 0.13; 211, 61%, 0.27, 82, 0.13 (8 cells used in the > 0 rows) | judge_without_format_zeros |
| 5.9 | -13.6, 4 changed, 4 worse, 0 better, -29.2, -0.9 | legacy |
| Table 11 | D1, D2 (Luna, 235B), D3 (0, 0, 1) | capability, oracle |
| Table R.5 / 5.11 | all 7 rows; 14 calls, about 12k tokens, 134 and 356; 551 s median (about 9 min); 4.36 chars/token; 66% gives about 3x | cost, chars_per_token |
| Appendix G | 21 / 4 / 5 / 15 kinds; 15 with bare tag | base_prompt_inventory |
| 6.3.2 | 21 of 45 are seed plus format line | inventory |
| 6.3.3 | lenient precision and recall above 96% | 98.4%, 96.8% |
| 5.1 | exemplar +19.5 and +42.3 (shown), 9 and 15 wins, range -45.0 to +72.9, lenient +2.3 and +10.2 | exemplar, exemplar_lenient |

## Top problems to fix

1. V2F lenient numbers are produced by a code bug (non-accepted runs compared as P0 vs P0+exemplar). Correct values: 1 win, 3 losses, mean -0.4; lenient D1 outcome is null; lenient wins vs P0 are 13, not 3. Affects abstract, 5.3.1, 5.10, Table 11, Table R.4, 6.1, 7.
2. Section 3.2 and Appendix F say V2F ran on "all 45 cells"; it ran on 44.
3. Section 5.5 contains the unrendered placeholder "{len(l3)} cell(s)" (value 1).
4. "Almost all ... tag convention / adds at most a few points" is not true for ProTeGi (80%, 8.6 points, up to about 37 points per cell).
5. "The rubric is not what produces them" (5.6.1) and "a rubric-free rewrite also obtains" (abstract, R.4, 5.6.2) overstate Table R.2 (V2 significantly beats V3 in 2 cells); at 8B the two variants win in different cells and V3's win holds under lenient scoring.
6. "Upper bound on the false-positive rate 5%" is actually the upper bound on 1-precision; the false-positive-rate bound is 8.3%.
7. The negative end of the exemplar-gain range is disambiguation_qa/APE (-45.0) and ruin_names/APE s43, not APE's hyperbaton prompt.
8. Section 7 says the 235B gains "do not hold under lenient scoring", but hyperbaton/PW holds under both metrics (and the Introduction correctly says "most").

Also: the positive-control headline 29 of 45 includes 5 PW noise acceptances (24 of 26 is the controlled figure); Table R.3 caption describes an 8B judge that is absent and "not a refiner" while Qwen3-8B-AWQ is a refiner; the same exemplar intervention is reported as +19.5/+42.3 (5.1) and +20.0/+42.8 (5.3.1); Appendix C says 92-95% (elsewhere 93-95%); 5.6.2 has grammar slips ("1 validated candidates", "1 winning cells", "V2 ahead in 1 cells", "1 wins").
