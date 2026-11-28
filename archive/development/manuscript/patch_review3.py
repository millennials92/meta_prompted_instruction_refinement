# -*- coding: utf-8 -*-
"""Patches for the abstract, appendix caption and response letter after the final fact-check."""


def edit(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert old in s, (path, old[:80])
        s = s.replace(old, new, 1)
    open(path, "w", encoding="utf-8").write(s)


edit("part5_appendix.py", [(
    '''"The strict accuracy of this prompt is higher than that of P0 (39.5% against 10.5%) but its lenient "
           "accuracy is lower (16% against 58%, Section 5.6).", full=False)''',
    '''f"The strict accuracy of this prompt is higher than that of P0 (39.5% against 10.5%) but its lenient "
           f"accuracy is lower ({100 * N['winner_cells']['qwen235|hyperbaton__protegi__s42']['final_lenient']:.0f}% "
           f"against {100 * N['winner_cells']['qwen235|hyperbaton__protegi__s42']['p0_lenient']:.0f}%, Section 5.6).", full=False)''')])

edit("part_abstract.py", [(
    '''f"{_MG['wins']} of {_MG['n_cells']} cells, mainly because the small refiner rarely produced a usable candidate "
    "that kept the answer format. Identical prompts disagree on "''',
    '''f"{_MG['wins']} of {_MG['n_cells']} cells: the small refiner rarely produced a usable candidate, and "
    "unprotected candidates often dropped the answer format. Identical prompts disagree on "'''),
    ('''won in {_CAP['v2|qwen235']['wins']} of 9 cells under strict scoring, but most of these gains reverse under "
    "lenient scoring and a rubric-free rewrite reaches similar gains, so they do not isolate the rubric. "''',
     '''won in {_CAP['v2|qwen235']['wins']} of 9 cells under strict scoring, but under lenient scoring only one "
    "gain remains and a rubric-free rewrite reaches similar strict gains, so they do not isolate the rubric. "'''),
])

edit("part1_intro_related.py", [(
    '''"of them reverse under lenient scoring and a rubric-free rewrite reaches gains of the same kind (RQ3); and "''',
    '''"of them do not hold under lenient scoring and a rubric-free rewrite reaches gains of the same kind (RQ3); and "'''),
    ('''mainly because that refiner rarely produces a usable candidate that keeps "
  "the answer format (RQ1, RQ2)''', '''because that refiner rarely produces a usable candidate and often drops "
  "the answer format (RQ1, RQ2)'''),
])

edit("part4_discussion_back.py", [(
    '''"obtained strict-metric gains in several cells, but most reverse under lenient scoring and a rubric-free "
  "rewrite reaches gains of the same kind''',
    '''"obtained strict-metric gains in several cells, but they do not hold under lenient scoring and a rubric-free "
  "rewrite reaches strict gains of the same kind'''),
    ('''"mainly because that refiner rarely produces a usable, format-preserving candidate.''',
     '''"because that refiner rarely produces a usable candidate and often drops the answer format.''') if False else
    ('''"mainly because that refiner rarely produces a usable, format-preserving candidate. A much larger refiner "''',
     '''"because that refiner rarely produces a usable candidate and often drops the answer format. A much larger refiner "'''),
])

# response letter
edit("build_response.py", [
    ('MG, CAP, NOISE, LEG, EX = N["main_grid"], N["capability"], N["noise"], N["legacy"], N["exemplar"]',
     'MG, CAP, NOISE, LEG, EX = N["main_grid"], N["capability"], N["noise"], N["legacy"], N["exemplar"]\nCAPL = N["capability_lenient"]'),
    ('''We re-measured this: the legacy loop (variant V0), re-run on the nine pilot cells, produced on average {V0['mean_distinct_candidates']:.1f} distinct candidates per run, but only through serving noise, not by design.''',
     '''We re-measured this: the legacy loop (variant V0), re-run on the nine pilot cells, produced on average {V0['mean_distinct_candidates']:.1f} distinct candidates per run, but only through serving noise, not by design.'''),
    ('''"Original Article Title: ' if False else ''', '''"Original Article Title: ' if False else ''') if False else
    ('''"The new title is 'Does Rubric-Guided Refinement Improve Automatically "
         "Optimized Prompts? A Pre-Specified Negative-Result Study with Small Open-Weight Models'.''',
     '''"The new title is 'Does Rubric-Guided Refinement Improve Automatically "
         "Optimized Prompts? A Pre-Specified Negative-Result Study'.'''),
    ('''"a mechanism study, a refiner-capability study, a judge-validity analysis, a serving-noise measurement and "
         "a format-exemplar intervention.''',
     '''"a mechanism study, a refiner-capability study, format-protected variants, a judge-validity analysis, a "
         "serving-noise measurement and a format-exemplar intervention.'''),
    ('''including one exploratory follow-up added after the results were seen''', '''including two exploratory additions made after results were seen''') if False else
    ('''the protocol and its dated amendments, including one exploratory follow-up added after the results were seen, are listed.''',
     '''the protocol and its dated amendments, including two exploratory additions made after results were seen, are listed.'''),
    ('''(6 test items, within the noise''', '''(4 test items, within the noise''') if False else
    ("The text now notes that the hyperbaton difference is 6 items and within noise.",
     "The text now notes that the hyperbaton difference is 4 test items and within noise."),
    ('''and validation tracks test accuracy (Spearman {N['val_test_corr']['all']['pooled_spearman']:.2f}).''',
     '''and validation tracks test accuracy across candidates (Spearman {N['val_test_corr']['all']['pooled_spearman']:.2f}), although this is driven by candidates that score zero and is {N['val_test_corr']['all']['pos_val_1.7b']['spearman']:.2f} among plausible 1.7B candidates (n = {N['val_test_corr']['all']['pos_val_1.7b']['n']}).'''),
    ('''The gains occur where the base prompt fails on answer format, and the accepted candidates add a format instruction and example, so they do not isolate the rubric.{FOLLOW_TEXT}''',
     '''Under lenient answer extraction only one of these gains remains in the first seed (hyperbaton with PromptWizard; {pp(CAPL['v2|qwen235']['mean_diff'])} points on average, {CAPL['v2|qwen235']['losses']} significant losses), and in the second seed the lenient mean is {pp(N['follow_seed2_lenient']['mean_diff'])} points. The strict gains occur where the base prompt fails on answer format, and the accepted candidates add a format instruction and example, so they do not isolate the rubric.{FOLLOW_TEXT}'''),
    ('''"Lenient accuracy is reported for every base prompt; the conclusions about MPIR do not change, but the apparent weakness of exemplar-free baselines is largely a format effect."''',
     '''f"Lenient accuracy is reported for every base prompt and for the final prompts of the strong refiners. The apparent weakness of exemplar-free baselines is largely a format effect, and it changes the reading of the strong-refiner gains: under lenient extraction only one of them remains in the first seed (Section 5.6). The exemplar intervention gains {pp(EX['mean_gain_by_optimizer']['protegi'])} points for ProTeGi under strict and {pp(N['exemplar_lenient']['mean_by_optimizer']['protegi'])} under lenient extraction."'''),
    ('''f"Done with Qwen3-235B-A22B and GPT-6 Luna (Section 5.6): wins in {QW['wins']} and {LU['wins']} of 9 cells."''',
     '''f"Done with Qwen3-235B-A22B and GPT-6 Luna (Section 5.6): strict wins in {QW['wins']} and {LU['wins']} of 9 cells; under lenient extraction the first-seed Qwen3-235B result is {CAPL['v2|qwen235']['wins']} win and {CAPL['v2|qwen235']['losses']} significant losses, and the format-protected variants of the 1.7B refiner remain worse than P0."'''),
    ('''f"We varied the refiner with the target fixed (Section 5.6). The question is only partly answered: a much larger refiner helps in some cells, GPT-6 Luna in fewer, and the gains coincide with a repaired answer format.{FOLLOW_TEXT} The original GPT-4o/GPT-3.5 setting was not rerun."''',
     '''f"We varied the refiner with the target fixed (Section 5.6). The question is only partly answered: a much larger refiner gains under strict scoring in some cells, GPT-6 Luna in fewer, and most of the strict gains do not hold under lenient scoring.{FOLLOW_TEXT} The original GPT-4o/GPT-3.5 setting was not rerun."'''),
])
print("patched abstract, appendix and response")
