# -*- coding: utf-8 -*-
from study_numbers import N
from rev2_numbers import OFF, NEW, LEN, MID

_MG, _CAP, _NOISE = N["main_grid"], N["capability"], N["noise"]
_EX, _LO = N["exemplar"], N["lenient_oracle"]
_G, _CT, _AB, _RV = NEW["grid_v2f"], NEW["controls"], OFF["acceptance_bounds"], OFF["rubric_vs_free"]["qwen235"]


def _pp(x, d=0):
    return f"{100 * x:+.{d}f}"


_lo = 100 * min(_LO['v1|1.7b']['share_without_tag'], _LO['v2|1.7b']['share_without_tag'])
_hi = 100 * max(_LO['v1|1.7b']['share_without_tag'], _LO['v2|1.7b']['share_without_tag'])

ABSTRACT = (
    "Meta-Prompted Instruction Refinement (MPIR) asks a language model to score an optimized prompt against a "
    "seven-criterion rubric, rewrite it, and keep the rewrite only if it beats the original on validation questions. "
    "We re-test it after an earlier evaluation was found to have a validation leak, using a repaired, fully logged loop "
    "and decision rules fixed in advance with dated amendments. The scope is narrow: Qwen3-1.7B as target, 45 base "
    "prompts from three small-model optimizers on nine Big-Bench Hard tasks, many of them weak. "
    f"With the 1.7B model refining its own prompts, the original treatment changed the prompt in {_MG['runs_accepting']} "
    f"of {_MG['runs']} runs (exact 95% upper bound {100 * _AB['run_upper_one_sided_95']:.1f}%), because "
    f"{_lo:.0f}-{_hi:.0f}% of candidates lost the answer format. A format- and example-preserving re-run (V2F), after a positive control "
    f"in which the gate accepted a known-good candidate in {_CT['known_good_accepted']} of {_CT['known_good_cells']} cells, had candidates "
    f"accepted in {_G['accepted_runs']} of {_G['runs']} runs but won in {_G['lenient_wins_vs_start']} cell(s) and lost "
    f"in {_G['lenient_losses_vs_start']} under a lenient metric validated on 400 AI-labelled responses. In exploratory "
    f"probes, a 235B refiner won in {_CAP['v2|qwen235']['wins']} of 9 cells under strict scoring, mostly format repairs "
    f"for which a rubric-free rewrite obtains gains of the same kind (paired difference {_pp(_RV['mean_v2_minus_v3'], 1)} points, interval "
    f"[{_pp(_RV['lo'], 1)}, {_pp(_RV['hi'], 1)}]); an 8B refiner changed the prompt in "
    f"{MID['v2']['accepted_runs']} of {MID['v2']['runs']} runs. One answer-format exemplar "
    f"changed strict accuracy by {_pp(_EX['mean_gain_by_optimizer']['ape'])} (APE) and "
    f"{_pp(_EX['mean_gain_by_optimizer']['protegi'])} (ProTeGi) points on average, mostly the tag "
    f"convention, and identical prompts disagree on {100 * _NOISE['mean_discordance']:.1f}% of examples between "
    "scorings. Conclusions are limited to Qwen3 targets, nine tasks and the optimizers tested."
)
