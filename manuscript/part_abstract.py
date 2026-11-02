# -*- coding: utf-8 -*-
"""Abstract (final text, 250 words or fewer; every number comes from results/revision/*.json)."""
from rev2_numbers import NEW, PP

_p = lambda x, d=1: f"{100 * x:+.{d}f}"
_G, _V3, _PR = NEW["grid_v2f"], PP["v3f_1.7b"], PP["paired_1.7b_all"]["lenient"]
_CT = NEW["controls"]

ABSTRACT = (
    "Meta-Prompted Instruction Refinement (MPIR) asks a language model to score an optimized prompt against a "
    "rubric, rewrite it, and keep the rewrite only if it beats the original on validation questions. We re-test it after "
    "an earlier evaluation was found to have a validation leak, with a repaired, logged loop and decision rules fixed in "
    "advance. The scope is narrow: Qwen3-1.7B as target, 45 base prompts from three small-model optimizers, nine "
    "Big-Bench Hard tasks. With the 1.7B model refining its own prompts, the original treatment changed the prompt in 1 of "
    "80 runs, but 93-95% of its candidates had lost the answer format, so the test could not separate a failed idea from a "
    f"failed treatment. With the format protected (positive control: the gate accepted a known-good candidate in "
    f"{_CT['known_good_accepted']} of {_CT['known_good_cells']} cells), the rubric-guided variant won in "
    f"{_G['lenient_wins_vs_start']} cell and lost in {_G['lenient_losses_vs_start']} under a lenient metric validated on 400 "
    f"AI-labelled responses, whereas a rubric-free rewrite with the same protection gained {_p(_V3['lenient_mean'])} points "
    f"on average ({_V3['lenient_wins']} winning cells of {_V3['runs']}); the paired difference is "
    f"{_p(-_PR['mean'])} points [{_p(-_PR['hi'])}, {_p(-_PR['lo'])}] in its favour. Qwen3-235B and GPT-6 Luna improved the prompts under "
    f"protection ({_p(PP['v2f_qwen235']['lenient_mean'])} and {_p(PP['v2f_gpt6luna']['lenient_mean'])} points, nine cells, one run "
    "each), but a rubric-free rewrite reached similar gains. One answer-format exemplar changed strict accuracy by +19.5 (APE) and +42.3 "
    "(ProTeGi) points, mostly tag format, and identical prompts disagree on 6.5% of examples. The "
    "rubric is not shown to help; conclusions are limited to Qwen3 targets, nine tasks and these optimizers."
)
