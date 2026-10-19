# -*- coding: utf-8 -*-
from blocks_api import H1, H2, H3, P, BULLETS, TABLE  # noqa: F401
from study_numbers import N  # noqa: F401

# ===========================================================================
H1("1. Introduction")

P("The prompt is the specification, the interface and the test fixture of an LLM-based component at "
  "once, and unlike source code it cannot be checked by a compiler. Large language models (LLMs) perform "
  "strongly on tasks such as translation ‹101›, summarization ‹35› and question answering ‹102› "
  "(see ‹34› for a survey), yet small changes in phrasing or formatting can change their accuracy "
  "substantially ‹33,110›. Prompt design has therefore become a bottleneck for reliable deployment in "
  "domains such as law and medicine ‹37,38›.")

P("Two families of approaches address that bottleneck. Manual prompting applies heuristics distilled from "
  "practice, such as chain-of-thought reasoning ‹7›, role assignment ‹12› and few-shot demonstration "
  "‹4›; these are interpretable but need expertise and trial and error ‹32,3›. Automatic prompt "
  "optimization (APO) searches the space of prompts with the help of an LLM, using critique, mutation, "
  "evolution or optimizer-style proposal loops ‹19,20,21,22,23›; it scales but may forfeit the "
  "structure that human-written prompts carry ‹47›. A natural idea, which we call Meta-Prompted "
  "Instruction Refinement (MPIR), is to add a cheap post-hoc stage after any APO run: ask an LLM to score "
  "the optimized prompt against a rubric of manual heuristics, ask it to rewrite the prompt accordingly, "
  "and keep the rewrite only if it scores better on held-out validation questions.")

P("This paper does not propose MPIR as a new method. It reports a pre-specified test of whether that "
  "idea works when the same small open-weight model plays every role, and what the evidence does and does "
  "not allow us to conclude. The study has a history that readers need in order to weigh it. An earlier "
  "evaluation used GPT-4o as the refining model and GPT-3.5-turbo as the target on 23 Big-Bench Hard (BBH) "
  "tasks ‹52›. It reported per-task gains over PromptWizard ‹23› on a majority of tasks but a pooled gain "
  "that was not statistically significant (Appendix D reproduces those numbers). A later code audit found "
  "two defects: the validation stage scored candidate prompts on the optimizer's own training examples, "
  "and the refinement loop never scored the unrefined prompt, so a refinement was always accepted "
  "whatever it scored. A manuscript describing that evaluation was desk-rejected by another journal "
  "without external review. The numbers in it should be treated as exploratory and are not used as "
  "evidence here.")

P("We therefore asked a narrower question that the first evaluation could not answer: are the apparent "
  "gains attributable to the rubric-guided refinement mechanism, or to the refining model being more "
  "capable than the model it refines for? A first pilot with Qwen3-1.7B in every role found the refined "
  "prompt byte-identical to the base prompt in all nine (task, optimizer) cells. Review of that pilot, "
  "however, exposed weaknesses that we take seriously: the seven refinement rounds were not independent "
  "attempts, the composition of the outcomes was not reported, the statistics were degenerate, the task "
  "choice was biased, the capability-gap explanation was untested, and the evaluation harness floored "
  "several baselines. This revision answers each of them with new experiments, specified in a "
  "version-controlled protocol before they were run (Section 4.1).")

P("We ask four research questions. RQ1: with the refinement loop repaired so that rounds are genuinely "
  "distinct attempts, does gated MPIR improve a base optimizer's prompt over many tasks, optimizers and "
  "seeds? RQ2: is the failure to improve a property of the candidates or of the coarse validation gate? "
  "RQ3: does a more capable refining model change the answer? RQ4: does the rubric judge measure "
  "anything related to accuracy? @@SUMMARY@@")

P("The contributions are:")
BULLETS([
    ("A repaired and instrumented refinement loop.",
     "Rounds differ by sampling and, in the chained variant, by the history of rejected candidates; every "
     "round is logged with an outcome category (fallback, echo-rejected, duplicate, validated), its "
     "validation and test scores, and the judge's rubric total."),
    ("A pre-specified, multi-seed, multi-task evaluation.",
     "45 base prompts (three optimizers, nine tasks, up to three seeds), with decision rules fixed in a "
     "time-stamped protocol and a random, documented choice of six tasks to complement the three pilot tasks."),
    ("Diagnostics that make a null result interpretable.",
     "Oracle scoring of all candidates on the test split, acceptance-rule sensitivity, a serving-noise "
     "floor and minimum detectable effect, a gated free-rewrite control, an uncorrected-algorithm "
     "comparison, and a judge-validity analysis against a stronger reference judge."),
    ("Harness findings.",
     "Strict answer extraction floors some baselines; lenient re-scoring of the saved raw outputs and a "
     "catalogue of the pipeline defects, with serving configuration, are reported (Appendix C)."),
])

P("Section 2 reviews related work. Section 3 describes the layer under test. Section 4 gives the design "
  "and statistics. Section 5 reports results, Section 6 discusses what they support and their limits, and "
  "Section 7 concludes. Code, per-example predictions and per-round logs are public.")

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
  "tasks ‹17›, which is weaker support for knowledge-intensive reasoning than the original step-back "
  "evidence. Models attend unevenly to information in long contexts ‹15,14›, and vendor guidance reports "
  "recency effects ‹43›; whether either matters for the short prompts studied here is an empirical "
  "question rather than a premise. Self-consistency, a decoding-time heuristic, samples several reasoning "
  "paths and votes ‹66›; MPIR does not use it.")

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
