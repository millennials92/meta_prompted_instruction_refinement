# Factual check of the V3F / protected-probe material (round 2)

Recomputed from rev2_protected_probes.json, rev2_new.json, rev2_mid_refiner.json. All differences in points; "paired" = V2F minus V3F as stored in the JSON (negative = V3F ahead).

## Recomputed ground truth

| Variant | runs | accepted | strict wins/losses, mean | lenient wins/losses, mean |
|---|---|---|---|---|
| 1.7B V2F | 44 | 15 | 4/2, +0.71 | 1/3, -0.43 |
| 1.7B V3F | 45 | 22 | 11/3, +4.13 | 11/3, +3.81 |
| Qwen3-235B V2F | 9 | 8 | 4/0, +9.24 | 3/0, +7.31 |
| Qwen3-235B V3F | 9 | 9 | 4/0, +10.08 | 3/0, +7.63 |
| GPT-6 Luna V2F | 9 | 7 | 3/1, +5.86 | 3/1, +3.36 |
| GPT-6 Luna V3F | 9 | 9 | 3/0, +3.93 | 2/1, +1.48 |

- V3F lenient by optimizer: APE +4.88, ProTeGi +4.57, PromptWizard +2.00.
- Paired, 1.7B, 44 cells: strict -2.97 [-6.42, +0.03]; lenient -3.70 [-6.81, -1.07]; lenient V3F better 17, V2F better 9, tied 18.
- Paired, 1.7B, 9 Tier-1 seed-42 cells: lenient -3.19 [-8.94, +2.15] (includes zero).
- Paired Qwen3-235B lenient: -0.31 [-9.39, +7.74]. Paired GPT-6 Luna lenient: +1.88 [-4.62, +8.37]. Both include zero.
- 1.7B on the nine Tier-1 seed-42 cells: V2F lenient -1.61 (0 wins, 2 losses), V3F lenient +1.57 (2/1).
- Rounds: V2F 308 rounds, 88 fallback + 60 echo = 148, 92 validated, 20 better. V3F 315 rounds, 0 fallback/echo, 124 duplicates, 191 validated, 66 better.
- D1 (scaled): V3F lenient 11 wins vs 2n/45 = 2.0 and n/3 = 15 -> mixed (OK). V2F lenient 1 win <= 1.96 and |mean| 0.43 <= 1 -> null (OK). V2F strict 4 wins -> mixed (OK).
- D2 (lenient, protected, vs 1.7B on the same nine cells, using V2F -1.61): Qwen3-235B +8.9 and 3 wins -> met. GPT-6 Luna +5.0 and 3 wins -> also met.
- Positive control: 29/45 accepted overall, 24 of 26 APE/ProTeGi cells where the exemplar helps (OK). 8B refiner: V2 strict +1.78, lenient +0.17; V3 strict +1.89 (OK).

## Findings (WRONG / UNSUPPORTED first)

| # | Location | Claim | Verified value | Verdict | Exact replacement |
|---|---|---|---|---|---|
| 1 | Response letter, Weakness 1 (ii) (build_response_round2.py line ~119) | "+7.3 for Qwen3-235B and +3.4 for GPT-6 Luna ... against -0.4 for the 1.7B refiner" | -0.4 is the 44-cell 1.7B mean. The strong-refiner numbers are on the nine Tier-1 cells, where 1.7B V2F is -1.6 (the manuscript's 5.6.3 says -1.6). Different cell sets, and the letter contradicts the manuscript. | WRONG | "against -1.6 for the 1.7B refiner on the same nine cells" |
| 2 | Abstract, Sec 5.3.2, Sec 6.1, Sec 7, Response Major 4 | "the rubric hurts a 1.7B refiner" / "for this refiner it hurts and does not merely fail to help" / "so for a 1.7B refiner the rubric is not neutral" | Supported only as: V2F did worse than V3F (paired lenient -3.7 [-6.8, -1.1], 44 cells). V2F's own lenient mean is -0.4 (1 win, 3 losses), which is null under D1, so there is no absolute harm. On the nine Tier-1 cells the paired interval includes zero. V2F and V3F also differ in more than the rubric (chained vs independent rounds, an evaluation call, score history, 148/308 fallback/echo rounds vs 0, so 92 vs 191 validated candidates). One MPIR seed, no multiplicity correction. | UNSUPPORTED (overclaim) | Sec 5.3.2: replace "which isolates the rubric: for this refiner it hurts and does not merely fail to help" with "which separates the rubric-guided request (with its evaluation step, chained history and longer prompt) from a plain rewrite request: with this refiner the rubric-guided variant did worse than the plain rewrite, and it did not do better than its own baseline." Sec 7 / 6.1: "so the rubric hurts a 1.7B refiner" -> "so the rubric-guided variant did worse than a rubric-free rewrite with a 1.7B refiner". Response Major 4: "so for a 1.7B refiner the rubric hurts" -> "so, with a 1.7B refiner, the rubric-guided variant does worse than the rubric-free one (one MPIR seed; the two variants also differ in chaining and evaluation call)". |
| 3 | Sec 5.3.2 first line; Table R.4 row V3F ("rubric isolated"); Response Major 4 ("We isolated the rubric directly") | "V3F isolates the rubric" | The text of 5.3.1 itself calls V2F a hybrid; V3F differs also by mode (independent vs chained), evaluation step and history. | UNSUPPORTED | "V3F is a rubric-free counterpart of V2F." Table R.4: "rubric-free counterpart under the same protection". Response: "We added a rubric-free counterpart under the same protection." |
| 4 | Sec 6.1: "the long rubric meta-prompt makes the refiner worse"; Response Weakness 1 (i): "The long rubric meta-prompt is indeed what a 1.7B refiner handles badly" | Mechanism stated as fact | Sec 5.3.2 says "we did not test it, for example with a shortened rubric". Fallback/echo rates (48% vs 0%) are consistent with it but the length is confounded with chaining and the evaluation call. | UNSUPPORTED | Sec 6.1: "the rubric-guided request was answered with fallbacks or echoes in 48% of rounds against none for the plain rewrite request, which is consistent with, but does not show, an effect of the long meta-prompt". Response (i): "The rubric-guided request is where the 1.7B refiner fails more often (48% of rounds ended in a fallback or echo, against none for a plain rewrite request)". |
| 5 | Sec 5.10 and Table 11 | "under the protected treatment and lenient scoring it is met for Qwen3-235B (post hoc)" (no GPT row) | By the same D2 criterion GPT-6 Luna is also met (+3.4 vs -1.6 = +5.0 over the 1.7B refiner, 3 wins of 9). Table 11 still lists only the unprotected GPT-6 Luna row ("Not met"), which reads as if D2 fails for GPT-6 Luna overall. | WRONG (selective) | Sec 5.10: "under the protected treatment and lenient scoring it is met for Qwen3-235B and for GPT-6 Luna (both post hoc), but the rubric-free control does as well." Add Table 11 row: "D2, protected V2F, GPT-6 Luna (post hoc, lenient) | Mean +2.0 over the 1.7B refiner and at least 3 of 9 wins | +5.0 over 1.7B (+3.4 vs -1.6); 3 wins | Met, but not specific to the rubric (V3F +1.5)". |
| 6 | Table 11 row "D2, protected V2F, Qwen3-235B" | Observed "+7.3; 3 wins" against criterion "mean +2.0 over the 1.7B refiner" | The criterion is a difference over the 1.7B refiner: +7.3 - (-1.6) = +8.9. +7.3 is the refiner's own mean. | WRONG (inconsistent unit) | "+8.9 over 1.7B (+7.3 vs -1.6); 3 wins" |
| 7 | Abstract, last-but-two sentence; Sec 1 Step 3; Sec 6.1 para 2; Sec 7 | "a rubric-free rewrite did as well" / "does about as well" / "the rubric is not what helps" | Qwen3-235B: -0.3 [-9.4, +7.7], fine. GPT-6 Luna: V3F +1.5 vs V2F +3.4, V2F ahead by 1.9 (interval [-4.6, +8.4]); nine cells and one run cannot show equivalence (5.6.3 says so). Sec 1: "so the rubric is not what helps" is stronger than "not shown to help". | UNSUPPORTED (overclaim) | Abstract: "but a rubric-free rewrite was not detectably worse (paired differences include zero)". Sec 1 Step 3: "but a rubric-free rewrite does about as well in these nine cells, so the rubric is not shown to help". Sec 6.1: "does as well (+7.6 and +1.5 points)" -> "reaches +7.6 and +1.5 points". |
| 8 | Sec 6.1 para 2: "the picture is the reverse in size and the same in kind" | After "for a 1.7B refiner the rubric ... worse" | Not the same in kind: with the 1.7B refiner V3F beats V2F (interval excludes zero), with Qwen3-235B and GPT-6 Luna the difference includes zero. Also compares a 45-cell result with a 9-cell one. | UNSUPPORTED | "With strong refiners the gains are larger and the rubric-guided and rubric-free variants are not distinguishable (exploratory: nine cells, one run per condition)." |
| 9 | Sec 1 Step 2 (and Response Major 1 style summaries) | "the rubric-guided variant still did not win under a validated lenient metric" | It won in 1 of 44 cells and lost in 3. | WRONG (imprecise) | "the rubric-guided variant won in 1 of 44 cells and lost in 3 under a validated lenient metric" |
| 10 | Sec 5.3.2: "run with the 1.7B refiner on the same cells as V2F (45 cells, one MPIR seed)" | V2F has 44 cells, V3F 45 | Self-contradictory. | WRONG | "on the 44 cells of V2F and on the 45th cell (geometric_shapes with PromptWizard), 45 cells in all, one MPIR seed". The paired comparison correctly uses 44. |
| 11 | Sec 6.2 item (v) | "start from a short rewrite request and not a long rubric" | Rests on the confounded V2F/V3F contrast at 1.7B (one seed), and with strong refiners there is no difference. | UNSUPPORTED (soft) | "(v) If a refinement layer is used with a small model, a short rewrite request with the format line and one worked example did better than the long rubric request here (one MPIR seed, Section 5.3.2); validate on held-out data." |
| 12 | Sec 1 roadmap (line 34) | "the repaired grid, positive control and V2F (5.3); the gate and rubric-free rewriting (5.4, 5.5)" | 5.3 now also holds V3F (5.3.2) and 5.6.3 holds the protected strong refiners. | WRONG (stale) | "the repaired grid, positive control, V2F and its rubric-free counterpart V3F (5.3); the gate and rubric-free rewriting (5.4, 5.5); stronger refiners, including the protected V2F and V3F (5.6)" |
| 13 | Abstract / Sec 5.3.2 / Sec 6.1 mixing 45-cell and 9-cell numbers | +3.8 (45 cells, 1.7B) next to +7.3, +3.4 (9 cells, strong) | Different cell sets. On the same nine cells the 1.7B refiner gives V2F -1.6 and V3F +1.6. The manuscript only gives the -1.6 in 5.6.3. | OK but flag | Add to abstract or 5.6.3: "On the same nine cells the 1.7B refiner gives -1.6 (V2F) and +1.6 (V3F)." |
| 14 | Sec 5.3.1 / 5.3.2: "noisy search ... not tested against a random-acceptance baseline" | | Correctly disclosed. But V3F's +3.8 and V2F's -0.4 are both read causally without that baseline. | OK (keep; add to 6.3.1 if absent: V3F gains may partly reflect more validated candidates, 191 vs 92) | - |
| 15 | Sec 5.3.2 text: "V3F never ended in a fallback or echo, against 148 of 308 rounds for V2F; 191 of 315 validated (92 of 308); 66 beat the baseline (20); accepted 22 of 45 (15 of 44)" | | 0/0; 88+60 = 148; 191; 92; 66; 20; 22; 15 | OK | - |
| 16 | Sec 5.3.2: strict +4.1, lenient +3.8, 11 and 11 wins, 3 lenient losses, V2F lenient -0.4; by optimizer +4.9 APE, +4.6 ProTeGi, +2.0 PromptWizard | | +4.13; +3.81; 11/11; 3; -0.43; 4.88/4.57/2.00 | OK | - |
| 17 | Sec 5.3.2 paired: strict -3.0 [-6.4, +0.0]; lenient -3.7 [-6.8, -1.1]; V3F ahead 17, V2F 9, tied 18 | | -2.97 [-6.42, +0.03]; -3.70 [-6.81, -1.07]; 17/9/18. Sign "V2F minus V3F" stated explicitly. | OK | - |
| 18 | Sec 5.3.2 D1: "mixed for V3F (more than 2 and fewer than a third of the cells win, mean above +2.0)" | | 11 wins; 2 < 11 < 15; mean +3.8 | OK | - |
| 19 | Abstract / 6.1: "paired difference is +3.7 points [+1.1, +6.8] in its favour" | | Flip of -3.70 [-6.81, -1.07] is +3.70 [+1.07, +6.81] | OK (sign convention correct) | - |
| 20 | Abstract / 5.6.3 / Table R.6 / Table R.4 / Table 11 / Sec 7 strong-refiner numbers | +7.3 (3/0), +3.4 (3/1), V3F +7.6 (4 strict wins, +10.1 strict), +1.5 (2/1, strict +3.9); paired -0.3 [-9.4, +7.7], +1.9 [-4.6, +8.4]; validated 87/100/87/98%; accepted 8/9, 9/9, 7/9, 9/9 | All match (V2F Qwen 55/63 = 87%, V3F Qwen 63/63, V2F GPT 55/63, V3F GPT 62/63 = 98%) | OK | - |
| 21 | Sec 5.6.3: "-1.6 points for the 1.7B refiner on the same cells"; "D2 ... for Qwen3-235B the mean exceeds that of the 1.7B refiner by more than 2.0 and at least 3 cells win" | | -1.61; +8.9; 3 wins | OK | - |
| 22 | Sec 5.3.1 / 5.10 / Table 11 V2F rows: strict 4 wins mean +0.7 (mixed), lenient 1 win 3 losses mean -0.4 (null) | | Match | OK | - |
| 23 | Table R.6 1.7B rows, Table R.4 V2F/V3F rows, Appendix F amendment 11 | | Numbers match; amendment 11 correctly says the V2F vs V3F comparison is post hoc and V3F was added after V2F results. | OK | - |
| 24 | Sec 5.6.2 / Table R.4 8B row | V2 strict +1.8, V3 +1.9; V2 lenient +0.2 | +1.78, +1.89, +0.17 | OK | - |
| 25 | Positive control "24 of 26", 29 of 45 | | 24/26, 29/45 | OK | - |

## Leftover-contradiction scan

- No remaining sentence says that no rubric-free control exists under protection, that the strong refiners could not be re-run, or that the rubric "cannot be isolated". The Section 5.6.1 sentence "cannot separate 'the rubric adds nothing' from 'the rubric adds a modest benefit'" is about the unprotected Qwen3-235B probe and still holds, but it now sits next to the stronger Section 5.3.2 claim that the rubric "hurts"; consider adding "for the 1.7B refiner only" there.
- Section 6.3.1 mentions "V2F against V3F use one MPIR seed" (fine). It does not state that no random-acceptance baseline exists for V3F gains; add one clause (Section 5.3.1 has it only for V2F).
- Section 6.3.3 "the rubric's criteria were not ablated" is consistent with the new text, and it is exactly why finding 2 is an overclaim.

## Summary counts

OK: 13 rows (15-25 plus 13 and 14 as flagged OK). WRONG: 6 (rows 1, 5, 6, 9, 10, 12). UNSUPPORTED/overclaim: 6 (rows 2, 3, 4, 7, 8, 11).

Top five problems:
1. "The rubric hurts a 1.7B refiner" (abstract-adjacent text, 5.3.2, 6.1, 7, response Major 4) is a causal claim from a confounded one-seed contrast (V2F vs V3F differ in chaining, evaluation call, history and fallback rate); V2F is null against its own baseline, and the nine-cell paired interval includes zero.
2. Response letter Weakness 1 (ii) compares nine-cell strong-refiner gains with "-0.4 for the 1.7B refiner" (44-cell mean); the same-cell figure is -1.6, as the manuscript says.
3. D2 reporting: GPT-6 Luna also meets D2 under the protected lenient comparison (+5.0 over 1.7B, 3 wins) but only Qwen3-235B is listed; Table 11 shows "+7.3" where the criterion is a difference (+8.9).
4. "V3F isolates the rubric" and "the long rubric meta-prompt makes the refiner worse" present an untested mechanism as fact (5.3.2 itself says it was not tested).
5. "A rubric-free rewrite did as well" / "the rubric is not what helps" is stronger than nine cells with one run each allow (GPT-6 Luna V2F is +1.9 ahead); also the stale Section 1 "did not win" (it won 1, lost 3), the 44-vs-45 cell contradiction in 5.3.2, and the outdated roadmap sentence.
