# -*- coding: utf-8 -*-
from study_numbers import N

_MG, _CAP, _CAPL, _NOISE, _LEG = N["main_grid"], N["capability"], N["capability_lenient"], N["noise"], N["legacy"]
_EX, _EXL = N["exemplar"], N["exemplar_lenient"]


def _pp(x, d=0):
    return f"{100 * x:+.{d}f}"


ABSTRACT = (
    "Meta-Prompted Instruction Refinement (MPIR) is a post-hoc layer that asks a language model to score an "
    "automatically optimized prompt against a seven-criterion rubric, rewrite it, and keep the rewrite only if it "
    "beats the original on held-out questions. An earlier evaluation with a closed refiner reported mixed gains, "
    "but audits found a validation leak and an unscored baseline, and a small pilot left its refinement rounds, "
    "statistics and answer-extraction harness open to criticism. We report a re-test whose decision rules were "
    "fixed in a time-stamped protocol. We repaired the refinement loop, logged every round, and refined 45 base "
    "prompts (PromptWizard, APE and ProTeGi on nine Big-Bench Hard tasks, up to three seeds) with Qwen3-1.7B as "
    f"target and refiner. The layer changed the prompt in {_MG['runs_accepting']} of {_MG['runs']} runs and won in "
    f"{_MG['wins']} of {_MG['n_cells']} cells, mainly because the small refiner rarely produced a usable candidate "
    "that kept the answer format. Identical prompts disagree on "
    f"{100 * _NOISE['mean_discordance']:.1f}% of examples between scorings, and one answer-format exemplar raised the "
    f"strict accuracy of exemplar-free prompts by {_pp(_EX['mean_gain_by_optimizer']['ape'])} and "
    f"{_pp(_EX['mean_gain_by_optimizer']['protegi'])} points. In an exploratory probe, a 235B-parameter refiner "
    f"won in {_CAP['v2|qwen235']['wins']} of 9 cells under strict scoring, but most of these gains reverse under "
    "lenient scoring and a rubric-free rewrite reaches similar gains, so they do not isolate the rubric. "
    "Conclusions are limited to Qwen3 targets, nine tasks and the optimizers tested."
)
