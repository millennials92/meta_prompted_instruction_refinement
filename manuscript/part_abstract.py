# -*- coding: utf-8 -*-
"""Abstract (final text, 250 words or fewer; every number comes from results/revision/*.json)."""
from rev2_numbers import NEW, PP, OFF

_p = lambda x, d=1: f"{100 * x:+.{d}f}"
_G, _V3, _PR = NEW["grid_v2f"], PP["v3f_1.7b"], PP["paired_1.7b_all"]["lenient"]
_CT = NEW["controls"]
_K = OFF["base_prompt_inventory"]["kinds"]

ABSTRACT = (
    "Meta-Prompted Instruction Refinement (MPIR) asks a language model to score an optimized prompt against a "
    "rubric, rewrite it, and keep it only if it beats the original on validation questions. We re-test it with a "
    "repaired, logged loop and decision rules fixed in advance, after an earlier evaluation leaked validation data. "
    "Scope: Qwen3-1.7B target, nine Big-Bench Hard tasks. Of the 45 base prompts from "
    f"three small-model optimizers, {_K['seed_plus_format']} are the seed plus a format line, {_K['induction_template']} an induction template, "
    f"{_K['short_instruction']} short and {_K['promptwizard_long']} substantive. With a 1.7B refiner, the original treatment "
    "changed the prompt in 1 of 80 runs, but 93-95% of candidates had lost the answer format, so the test could not "
    "separate idea from treatment. With the format protected (a positive control, passed in "
    f"{_CT['known_good_accepted']} of {_CT['known_good_cells']} cells, shows sensitivity to large strict-metric effects only), the rubric-guided variant won in "
    f"{_G['lenient_wins_vs_start']} cell and lost in {_G['lenient_losses_vs_start']} under a lenient metric validated on 400 "
    f"AI-labelled responses; a rubric-free rewrite gained {_p(_V3['lenient_mean'])} points "
    f"on average ({_V3['lenient_wins']} winning cells of {_V3['runs']}). The paired difference is "
    f"{_p(-_PR['mean'])} points [{_p(-_PR['hi'])}, {_p(-_PR['lo'])}], but the variants also differ in chaining, evaluation call and "
    "candidate throughput (one post hoc seed). With Qwen3-235B and GPT-6 Luna, nine single-run cells gave higher mean "
    f"lenient accuracy ({_p(PP['v2f_qwen235']['lenient_mean'])} and {_p(PP['v2f_gpt6luna']['lenient_mean'])} points), with no reliable difference from a "
    "rubric-free rewrite. One format exemplar changed strict accuracy by +19.5 (APE) and +42.3 "
    "(ProTeGi) points; identical prompts disagree on 6.5% of examples. The rubric is not shown to help."
)
