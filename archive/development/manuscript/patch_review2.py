# -*- coding: utf-8 -*-
"""Patches after the final fact-check (kept for provenance)."""


def edit(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert old in s, (path, old[:80])
        s = s.replace(old, new, 1)
    open(path, "w", encoding="utf-8").write(s)


edit("part3_results.py", [
    # Table 6: distinct candidates per run
    ('''                 str(c["validated"]), f"{c['val_better']}/{c['val_equal']}/{c['val_worse']}", str(c["accepted"]),
                 f"{c['runs_with_acceptance']}/{c['runs']}"])''',
     '''                 str(c["validated"]), f"{c['val_better']}/{c['val_equal']}/{c['val_worse']}", str(c["accepted"]),
                 f"{c['runs_with_acceptance']}/{c['runs']}", f"{c['mean_distinct_candidates']:.1f}"])'''),
    ('''"column counts runs whose final prompt differs from P0.",
      header=["Variant", "Runs", "Rounds", "Fallback", "Echo", "Dup.", "Validated", "Val. better/equal/worse",
              "Accepted", "Runs changed"],''',
     '''"column pair gives the runs whose final prompt differs from P0 and the mean number of distinct candidate "
      "prompts per run.",
      header=["Variant", "Runs", "Rounds", "Fallback", "Echo", "Dup.", "Validated", "Val. better/equal/worse",
              "Accepted", "Runs changed", "Distinct per run"],'''),
    # 5.2 wording on tags
    ('''  f"{pct(LO['v1|1.7b']['share_without_tag'], 0)}% of the validated V1 and {pct(LO['v2|1.7b']['share_without_tag'], 0)}% "
  "of the validated V2 candidates lack it, so they score exactly zero under strict extraction; the "
  "rubric-free rewrite, whose prompt asks to preserve the answer format, loses it in none. The "''',
     '''  f"{pct(LO['v1|1.7b']['share_without_tag'], 0)}% of the oracle-scored V1 and {pct(LO['v2|1.7b']['share_without_tag'], 0)}% "
  "of the oracle-scored V2 candidates (MPIR seed 1) lack it, so they score exactly zero under strict extraction; "
  "the candidates of the rubric-free rewrite, which has the line appended in code, and of the protected "
  "variants always carry it. The "'''),
    # 5.3 rule of three -> exact bound
    ('''f"pre-specified rule (D1) the outcome is '{MG['verdict']}'; a rule-of-three bound puts the "
  f"acceptance rate of this procedure below about {100 * 3 / MG['runs']:.0f}% of runs. The result does not "''',
     '''f"pre-specified rule (D1) the outcome is '{MG['verdict']}'; the exact one-sided 95% upper bound "
  f"on the acceptance rate of this procedure is {100 * _beta.ppf(0.975, MG['runs_accepting'] + 1, MG['runs'] - MG['runs_accepting']):.0f}% of runs. The result does not "'''),
    ('from study_numbers import N\n', 'from scipy.stats import beta as _beta\n\nfrom study_numbers import N\n'),
    # 5.4 cell counts
    ('''  f"candidate that beats P0 by 3 points or more ({O1['cells_best_ge_3pts']} and {O2['cells_best_ge_3pts']} of 9 "
  "cells); these candidates''',
     '''  f"candidate that beats P0 by 3 points or more ({O1['cells_best_ge_3pts']} of 9 and {O2['cells_best_ge_3pts']} of 8 "
  "cells with a validated candidate); these candidates'''),
    # 5.5 conclusion on the rubric-free rewrite
    ('''  f"{CAP['v3|1.7b']['wins']} winning cells ({listing(w3)}) and {len(l3)} cells in which the gate accepted a "
  f"harmful rewrite ({listing(l3)}), for a mean difference of {pp(CAP['v3|1.7b']['mean_diff'])} points. So the "
  "gate does not reliably protect against harm either, since 25 questions are noisy. We cannot say that "
  "unstructured rewriting is worse than rubric-guided refinement: the rubric-free variant did at least as well "
  f"({pp(CAP['v3|1.7b']['mean_diff'])} against {pp(CAP['v1|1.7b']['mean_diff'])} and "
  f"{pp(CAP['v2|1.7b']['mean_diff'])} points), and the rubric adds nothing detectable with this refiner.")''',
     '''  f"{CAP['v3|1.7b']['wins']} winning cells ({listing(w3)}) and {len(l3)} cell(s) in which the gate accepted a "
  f"significantly harmful rewrite ({listing(l3)}), for a mean strict difference of {pp(CAP['v3|1.7b']['mean_diff'])} "
  f"points; under lenient extraction it has {CAPL['v3|1.7b']['wins']} wins and {CAPL['v3|1.7b']['losses']} losses "
  f"(mean {pp(CAPL['v3|1.7b']['mean_diff'])} points, against {pp(CAPL['v1|1.7b']['mean_diff'])} for V1 and "
  f"{pp(CAPL['v2|1.7b']['mean_diff'])} for V2). So the gate does not reliably protect against harm, since 25 "
  "questions are noisy, and neither variant helps with this refiner: the rubric-free rewrite has more strict "
  "wins, partly because of the format asymmetry, and is not better under lenient scoring, so the rubric adds "
  "nothing detectable and we cannot say that unstructured rewriting is worse.")'''),
    # Table 9 rows for protected variants
    ('''                   ("v3|1.7b", "Qwen3-1.7B (self), V3"), ("v2|qwen235", "Qwen3-235B-A22B, V2"),''',
     '''                   ("v3|1.7b", "Qwen3-1.7B (self), V3"), ("v1p|1.7b", "Qwen3-1.7B (self), V1P"),
                   ("v2p|1.7b", "Qwen3-1.7B (self), V2P"), ("v2|qwen235", "Qwen3-235B-A22B, V2"),'''),
    # 5.6 strict/lenient paragraph
    ('''P("Strict accuracy is, however, the wrong judge of these gains. Under lenient extraction (Table 9) "
  f"Qwen3-235B wins in {CAPL['v2|qwen235']['wins']} cell and loses in {CAPL['v2|qwen235']['losses']}, with a mean "
  f"difference of {pp(CAPL['v2|qwen235']['mean_diff'])} points. The three ProTeGi cells in which the strict "
  "gain is large are cells where the base prompt reasons about as well as before but wraps whole sentences "
  "in the answer tags; the accepted candidate (Figure B.3) adds an explicit instruction to wrap the final "
  "answer on its own line and a worked output example, which repairs the tag, while the model's "
  "reasoning under the long structured prompt is worse (lenient accuracy falls, for example from "''',
     '''FS2L = N["follow_seed2_lenient"]
P("Strict accuracy is, however, a poor judge of these gains. Under lenient extraction (Table 9), in the first "
  f"MPIR seed Qwen3-235B wins in {CAPL['v2|qwen235']['wins']} cell and loses significantly in "
  f"{CAPL['v2|qwen235']['losses']}, with a mean difference of {pp(CAPL['v2|qwen235']['mean_diff'])} points; in the "
  f"second seed the lenient mean is {pp(FS2L['mean_diff'])} points with {FS2L['wins']} win and "
  f"{FS2L['losses']} significant losses, so the other strict gains vanish there instead of reversing. The three "
  "ProTeGi cells in which the strict gain is large are cells where the base prompt reasons about as well as "
  "before but wraps whole sentences in the answer tags; the accepted candidate (Figure B.3) adds an explicit "
  "instruction to wrap the final answer on its own line and a worked output example, which repairs the tag, "
  "while lenient accuracy does not improve, and in the first seed falls (we did not test why; the long "
  "structured prompt may make the 1.7B target reason worse), for example from "'''),
    ('''"on a base prompt that opens with an irrelevant sentence (Appendix A), is a gain that a rubric-based "
  "rewrite can plausibly claim;''',
     '''"on a near-chance base prompt (59% against 50% chance) whose instruction is garbled, is a gain that a "
  "rubric-based rewrite can plausibly claim;'''),
    # 5.6.1 seed 2 lenient
    ('''f"rubric-guided variant gives {pp(FS2['mean_diff'])} points and {FS2['wins']} winning cells, so the strict "
  "pattern is stable across seeds.")''',
     '''f"rubric-guided variant gives {pp(FS2['mean_diff'])} points and {FS2['wins']} winning cells, so the strict "
  f"pattern is stable across seeds; under lenient extraction its mean is {pp(FS2L['mean_diff'])} points with "
  f"{FS2L['wins']} win (hyperbaton with PromptWizard) and {FS2L['losses']} significant losses.")'''),
    # 5.9 restrict the two-cells statement
    ('''f"{pp(LG6['mean_delta'])} points. Two cells account for most of either loss, the repaired loop changes accuracy "
  f"by {pp(LEG['mean_gated_delta'])} points on the same candidates, and the figure applies to this refiner only; "''',
     '''f"{pp(LG6['mean_delta'])} points. Two cells account for most of the smaller loss, the repaired loop changes accuracy "
  f"by {pp(LEG['mean_gated_delta'])} points on the same candidates, and the figures apply to this refiner only; "'''),
])

edit("part4_discussion_back.py", [
    ('''"(Section 5.5). The main reason is the refiner''', '''"(Section 5.5). The main reason is the refiner''') if False else ('''not caused by the inconsistent rubric text, and a rubric-free gated rewrite does at least as well "
  "(Section 5.5).''', '''not caused by the inconsistent rubric text, and a rubric-free gated rewrite is not worse under lenient "
  "scoring (Section 5.5).'''),
    ('''  "Most of the strict gains are, however, repairs of the answer tag on base prompts that reason about as well "
  "as before; under lenient extraction they reverse, because the long structured rewrite makes the 1.7B "
  "target reason worse, and a rubric-free gated rewrite by the same refiner reaches gains of the same kind. "''',
     '''  "Most of the strict gains are, however, repairs of the answer tag on base prompts that reason about as well "
  "as before; under lenient extraction they reverse in the first MPIR seed and vanish in the second, and a "
  "rubric-free gated rewrite by the same refiner reaches strict gains of the same kind. "'''),
])
print("patched part3/part4")
