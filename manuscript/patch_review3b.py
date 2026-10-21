# -*- coding: utf-8 -*-
"""Response-letter patches after the final fact-check (kept for provenance)."""

p = "build_response.py"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new, 1)


rep('MG, CAP, NOISE, LEG, EX = N["main_grid"], N["capability"], N["noise"], N["legacy"], N["exemplar"]',
    'MG, CAP, NOISE, LEG, EX = N["main_grid"], N["capability"], N["noise"], N["legacy"], N["exemplar"]\n'
    'CAPL = N["capability_lenient"]')
rep("\"mechanism study, a refiner-capability study, a judge-validity analysis, a serving-noise measurement and \"\n"
    "         \"a format-exemplar intervention.",
    "\"mechanism study, a refiner-capability study, format-protected variants, a judge-validity analysis, a \"\n"
    "         \"serving-noise measurement and a format-exemplar intervention.")
rep("Optimized Prompts? A Pre-Specified Negative-Result Study with Small Open-Weight Models'. Section \"",
    "Optimized Prompts? A Pre-Specified Negative-Result Study'. Section \"")
rep("including one exploratory follow-up added after the results were seen, are listed.",
    "including two exploratory additions made after results were seen, are listed.")
rep("The text now notes that the hyperbaton difference is 6 items and within noise.",
    "The text now notes that the hyperbaton difference is 4 test items and within noise.")
rep("and validation tracks test accuracy (Spearman {N['val_test_corr']['all']['pooled_spearman']:.2f}).",
    "and validation tracks test accuracy across candidates (Spearman {N['val_test_corr']['all']['pooled_spearman']:.2f}), "
    "although this is driven by candidates that score zero and is {N['val_test_corr']['all']['pos_val_1.7b']['spearman']:.2f} "
    "among plausible 1.7B candidates (n = {N['val_test_corr']['all']['pos_val_1.7b']['n']}).")
rep("The gains occur where the base prompt fails on answer format, and the accepted candidates add a format instruction and example, so they do not isolate the rubric.{FOLLOW_TEXT}",
    "Under lenient answer extraction only one of these gains remains in the first seed (hyperbaton with PromptWizard; "
    "mean {pp(CAPL['v2|qwen235']['mean_diff'])} points, {CAPL['v2|qwen235']['losses']} significant losses), and in the second "
    "seed the lenient mean is {pp(N['follow_seed2_lenient']['mean_diff'])} points. The strict gains occur where the base prompt "
    "fails on answer format, and the accepted candidates add a format instruction and example, so they do not isolate "
    "the rubric.{FOLLOW_TEXT}")
rep("\"Lenient accuracy is reported for every base prompt; the conclusions about MPIR do not change, but the apparent weakness of exemplar-free baselines is largely a format effect.\"",
    "f\"Lenient accuracy is reported for every base prompt and for the final prompts of the strong refiners. The apparent \"\n"
    "     f\"weakness of exemplar-free baselines is largely a format effect, and it changes the reading of the strong-refiner \"\n"
    "     f\"gains: under lenient extraction only one of them remains in the first seed (Section 5.6). The exemplar \"\n"
    "     f\"intervention gains {pp(EX['mean_gain_by_optimizer']['protegi'])} points for ProTeGi under strict and \"\n"
    "     f\"{pp(N['exemplar_lenient']['mean_by_optimizer']['protegi'])} under lenient extraction.\"")
rep("f\"Done with Qwen3-235B-A22B and GPT-6 Luna (Section 5.6): wins in {QW['wins']} and {LU['wins']} of 9 cells.\"",
    "f\"Done with Qwen3-235B-A22B and GPT-6 Luna (Section 5.6): strict wins in {QW['wins']} and {LU['wins']} of 9 cells; \"\n"
    "     f\"under lenient extraction the first-seed Qwen3-235B result is {CAPL['v2|qwen235']['wins']} win and \"\n"
    "     f\"{CAPL['v2|qwen235']['losses']} significant losses, and the format-protected variants of the 1.7B refiner remain \"\n"
    "     f\"worse than P0.\"")
rep("a much larger refiner helps in some cells, GPT-6 Luna in fewer, and the gains coincide with a repaired answer format.{FOLLOW_TEXT}",
    "a much larger refiner gains under strict scoring in some cells, GPT-6 Luna in fewer, and most of the strict gains do not hold \"\n"
    "     f\"under lenient scoring.{FOLLOW_TEXT}")
open(p, "w", encoding="utf-8").write(s)
print("response patched")
