# -*- coding: utf-8 -*-
"""Abstract (final text, 250 words or fewer; every number was checked against results/revision/*.json)."""
ABSTRACT = (
    "Meta-Prompted Instruction Refinement (MPIR) has a language model score an optimized prompt against a rubric, "
    "rewrite it, and keep the rewrite only if it beats the original on validation questions. "
    "We re-test it after an earlier evaluation was found to have a validation leak, with a repaired, logged loop and "
    "decision rules fixed in advance. The scope is narrow: Qwen3-1.7B as target, 45 base prompts from three small-model "
    "optimizers, nine Big-Bench Hard tasks. With the 1.7B model refining its own prompts, the "
    "original treatment changed the prompt in 1 of 80 runs, but 93-95% of its candidates "
    "had lost the answer format, so the test could not separate a failed idea from a failed treatment. A "
    "format-preserving re-run (V2F), after a positive control in which the gate accepted a known-good candidate in 24 "
    "of 26 cells, had candidates accepted in 15 of 44 runs, but the final prompts won in 1 cell and lost in 3 under a "
    "lenient metric validated on 400 AI-labelled responses. In exploratory single-run probes, a 235B refiner won in 4 of "
    "9 cells under strict scoring, mostly format repairs, matched in some cells by a rubric-free rewrite "
    "(paired difference +2.7 points, interval [-3.5, +9.7]); an 8B refiner changed the prompt in 1 of 9 runs. One "
    "answer-format exemplar changed strict accuracy by +19.5 (APE) and +42.3 (ProTeGi) points, mostly the tag "
    "convention, and identical prompts disagree on 6.5% of examples between scorings. Conclusions are limited to Qwen3 "
    "targets, nine tasks and these optimizers."
)
