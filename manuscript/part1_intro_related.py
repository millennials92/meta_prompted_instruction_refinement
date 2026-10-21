# -*- coding: utf-8 -*-
from blocks_api import H1, H2, H3, P, BULLETS, TABLE  # noqa: F401
from study_numbers import N  # noqa: F401

# ===========================================================================
H1("1. Introduction")

P("Large language models (LLMs) are used for translation ‹101›, summarization ‹35› and question answering "
  "‹102› (see ‹34› for a survey), and are being applied in domains such as law and medicine ‹37,38›. Their "
  "accuracy depends on the prompt, and small changes in phrasing or formatting can change it substantially "
  "‹33,110›. Writing and tuning prompts is therefore a practical bottleneck for reliable deployment.")

P("Two families of approaches address that bottleneck. Manual prompting applies heuristics distilled from "
  "practice, such as chain-of-thought reasoning ‹7›, role assignment ‹12› and few-shot demonstration "
  "‹4›; these are interpretable but need expertise and trial and error ‹32,3›. Automatic prompt "
  "optimization (APO) searches the space of prompts with the help of an LLM, using critique, mutation, "
  "evolution or optimizer-style proposal loops ‹19,20,21,22,23›; it scales, but the prompts it produces "
  "are often less structured and harder to interpret than human-written ones ‹47›. A natural idea, which we "
  "call Meta-Prompted Instruction Refinement (MPIR), is to add a cheap post-hoc stage after any APO run: ask "
  "an LLM to score the optimized prompt against a rubric of manual heuristics, ask it to rewrite the prompt "
  "accordingly, and keep the rewrite only if it scores better on held-out validation questions.")

P("Whether such a layer helps matters in practice. It costs a few model calls and can be added to any APO "
  "pipeline, but if it does not help those calls are wasted, and an ungated version can make prompts worse. "
  "A clean test is not trivial: the rewrite is produced by a model that may be no more capable than the "
  "target, the gate uses a small validation set, and benchmark accuracy depends on how answers are "
  "extracted. We therefore report a pre-specified test, and we treat a null result as informative.")

P("The idea was first evaluated with GPT-4o as the refining model and GPT-3.5-turbo as the target on the "
  "tasks of Big-Bench Hard (BBH) ‹27›. That evaluation reported per-task gains over PromptWizard ‹23› but a "
  "pooled gain that was not significant (Appendix D reproduces its numbers). A later audit found that its "
  "validation stage scored candidates on the optimizer's own training examples and that its loop never "
  "scored the unrefined prompt. An earlier version of this study was submitted elsewhere and was not "
  "externally reviewed; the first evaluation's numbers are exploratory and are not used as evidence "
  "here. A pilot with Qwen3-1.7B in every role then found no effect, but its seven refinement rounds were "
  "not independent attempts, and its statistics, task choice and answer-extraction harness were open to "
  "criticism. This paper replaces the pilot with a new study.")

P("We ask four research questions. RQ1: with the refinement loop repaired so that rounds are genuinely "
  "distinct attempts, does gated MPIR improve a base optimizer's prompt over many tasks, optimizers and "
  "seeds? RQ2: is a failure to improve a property of the candidates or of the coarse validation gate? "
  "RQ3: does a more capable refining model change the answer? RQ4: does the rubric judge measure "
  "anything related to accuracy? (Terms such as cell and run are defined in Section 4.2.) "
  "In brief: with a 1.7B refiner the layer changed the prompt in "
  f"{N['main_grid']['runs_accepting']} of {N['main_grid']['runs']} runs and won in {N['main_grid']['wins']} "
  f"of {N['main_grid']['n_cells']} cells, mainly because that refiner rarely produces a usable candidate that keeps "
  "the answer format (RQ1, RQ2); a much larger refiner obtained strict-metric gains in several cells, but most "
  "of them reverse under lenient scoring and a rubric-free rewrite reaches gains of the same kind (RQ3); and "
  "the rubric score is only weakly related to accuracy (RQ4).")

P("The contributions are:")
BULLETS([
    ("A repaired, instrumented refinement loop.",
     "Rounds differ by sampling and, in the chained variant, by the history of rejected candidates; every "
     "round is logged with its outcome (fallback, echo, duplicate, validated), its validation and test scores "
     "and the judge's rubric total."),
    ("A pre-specified, multi-seed, multi-task evaluation.",
     "45 base prompts (three optimizers, nine tasks, up to three seeds), decision rules fixed in a "
     "time-stamped protocol, and a random, documented choice of six tasks to complement the three pilot tasks."),
    ("Diagnostics that make the result interpretable.",
     "Oracle scoring of all candidates, acceptance-rule sensitivity, a serving-noise floor and minimum "
     "detectable effect, gated and format-protected controls, strict and lenient answer extraction, a "
     "comparison with the uncorrected algorithm and a judge-validity analysis; defects of the pipeline "
     "and the serving configuration are catalogued (Appendix C)."),
])

# ===========================================================================
H1("2. Related Work")

H2("2.1. Manual Prompting Heuristics")
P("Prompting steers an LLM through in-context learning without changing its weights ‹2,4›, and has been "
  "described as a form of natural-language programming ‹3›. Non-expert authors struggle because they get "
  "no structural feedback and over-fit to the few examples they try ‹32›. Experienced authors rely on "
  "heuristics. Few-shot demonstrations help ‹4,103›, though Min et al. show that format, label space and "
  "input distribution matter more than label correctness ‹6›. Chain-of-thought prompting decomposes "
  "multistep problems ‹7,8›, and coherent, relevant reasoning steps matter more than length ‹9,10›. Role "
  "prompting assigns a persona to guide reasoning ‹12›, but its benefit is inconsistent: poorly aligned "
  "personas can hurt ‹16›, and a large study found that adding personas to system prompts does not "
  "reliably improve performance ‹104›. Step-back prompting abstracts key principles before solving ‹13›; a "
  "behavioural benchmark of 35 LLMs found that it fosters model-based behaviour in sequential decision "
  "tasks ‹17›, although this is weaker evidence for knowledge-intensive reasoning than the original step-back "
  "results. Models attend unevenly to information in long contexts ‹15,14›, and vendor guidance reports "
  "recency effects for some models while reporting no difference for GPT-4-class models ‹43›; whether either matters for the short prompts studied here is an empirical "
  "question rather than a premise. Self-consistency, a decoding-time heuristic, samples several reasoning "
  "paths and votes ‹66›;")

H2("2.2. Automatic Prompt Optimization and Meta-Prompting")
P("APO methods generate, evaluate and select prompts iteratively ‹47›. APE induces instructions from "
  "input-output demonstrations ‹19›; ProTeGi treats LLM critique as a textual gradient and runs a beam "
  "search ‹20›; EvoPrompt applies evolutionary operators ‹21›; OPRO conditions an LLM on a trajectory of "
  "scored prompts ‹22›; PromptWizard combines critique-based refinement, example selection and "
  "self-generated reasoning ‹23›. DSPy, TextGrad, Self-Refine and Reflexion generalize the idea of "
  "optimizing or refining text with LLM feedback ‹51,52,53,54›. Meta-prompting supplies an LLM with a "
  "template that structures the critique (see ‹18› for a survey): PE2 uses a step-by-step diagnosis "
  "template ‹24›, dual-phase methods add a meta-instruction-guided initialization ‹25›, and PROPEL "
  "injects expert-derived principles as priors ‹26›. Closest in spirit to MPIR, a two-stage approach for "
  "few-shot relation extraction first searches and then locally edits the winner ‹64›. More recent work "
  "explores evolutionary reflection ‹59›, error taxonomies ‹60›, format optimization ‹61›, debate-based "
  "fitness ‹62› and RL over edits ‹67›, and several studies target small or open-weight models: "
  "gradient-guided optimization with small models ‹120›, minimal-edit optimization for on-device "
  "models ‹121›, and error-driven optimization of arithmetic reasoning ‹122›. Others distill reasoning "
  "behaviour into prompts ‹116›, regularize prompts for token efficiency ‹117›, question what drives "
  "optimizer gains ‹118›, or codify principled instructions ‹119›.")

H2("2.3. Limits of Self-Refinement and of LLM Judges")
P("MPIR asks a model to critique and rewrite its own prompt, which places it in the self-refinement "
  "family ‹53,54›. Evidence on intrinsic self-correction is sobering: LLMs cannot reliably self-correct "
  "reasoning without external feedback ‹108›, and small models need strong verifiers to benefit from "
  "correction ‹109›. MPIR's critic is also an LLM judge, and LLM judges show position, verbosity and "
  "self-preference biases ‹105,106,107›. Finally, benchmark accuracy depends on prompt format and on "
  "how answers are extracted ‹110,111,112›, which matters here because MPIR's gate and our metrics both "
  "rely on a delimiter-based extraction. These strands predict that a 1.7B model refining its own prompts "
  "may fail for reasons unrelated to the rubric, which is why our design varies the refiner, validates "
  "the judge and re-scores answers leniently.")

H2("2.4. Reporting Negative Results")
P("Reproducibility programmes ‹113› and guidance on reporting experimental results ‹114› call for "
  "budgets, variance and full protocols. Variance across seeds and data splits can exceed the effect of "
  "a method ‹115›, significance testing in NLP needs care ‹56›, and small test sets leave studies "
  "under-powered ‹57›. We follow this guidance by fixing decision rules in advance, reporting "
  "per-cell paired differences with intervals rather than tests that cannot be informative, and "
  "quantifying the noise floor and the minimum detectable effect.")

H2("2.5. Positioning")
P("Table 1 contrasts how the methods above propose and select candidates. It states design differences, "
  "not evidence of superiority: whether any of them outperforms the others is not tested here.")

TABLE("Table 1. Design comparison of MPIR with representative methods. The table describes design "
      "differences only; it reports no empirical comparison.",
      header=["Method", "Candidate generation", "Selection", "Role relative to an APO run"],
      rows=[
          ["APE ‹19›", "Induction from demonstrations", "Execution accuracy", "Optimizer"],
          ["ProTeGi ‹20›", "Textual-gradient edits", "Beam search on minibatches", "Optimizer"],
          ["EvoPrompt ‹21›", "Crossover and mutation", "Fitness per generation", "Optimizer"],
          ["OPRO ‹22›", "LLM proposal given score history", "Best of trajectory", "Optimizer"],
          ["PromptWizard ‹23›", "Critique and sequential re-optimization", "Scores on training batches",
           "Optimizer"],
          ["PE2 ‹24›", "Structured diagnosis template", "Held-out scoring", "Optimizer"],
          ["PROPEL ‹26›", "Expert principles as priors", "Implicit in the priors", "Optimizer"],
          ["GEPA ‹59›", "Evolution with reflection", "Reflection-guided selection", "Optimizer"],
          ["MPIR (tested here)", "Rubric-guided critique and rewrite",
           "Validation accuracy vs. the unrefined prompt", "Post-hoc layer"],
      ], full=True, colw=[0.18, 0.30, 0.30, 0.22])

P("What distinguishes MPIR in this table is its role, not a mechanism: it adds one more "
  "critique-rewrite-validate cycle on top of whatever an upstream method produced. That makes it cheap and "
  "optimizer-agnostic, but also means its mechanism is close to existing critique-and-revise loops ‹20,24,53›; "
  "we do not claim novelty beyond the combination of an explicit rubric with a post-hoc, gated "
  "role.")
