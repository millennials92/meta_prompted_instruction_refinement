# -*- coding: utf-8 -*-
from blocks_api import H1, H2, H3, P, BULLETS, TABLE, CODE, ALGO, FIG  # noqa: F401
from study_numbers import N  # noqa: F401

# ===========================================================================
H1("3. The MPIR Layer Under Test")

P("MPIR is a three-stage cycle applied to a prompt P0 that an upstream APO method has already produced: "
  "(1) a language model scores P0 against a seven-criteria rubric of manual prompting heuristics and "
  "recommends improvements; (2) it rewrites P0 accordingly; (3) the rewrite is scored on validation "
  "questions and kept only if it beats the incumbent (Figure 1). We describe it here as the object "
  "under test, not as a contribution.")

FIG("figure_3.png",
    "Figure 1. The MPIR cycle as drawn for the first submission: a prompt P0 from an APO run enters "
    "evaluation, refinement and validation for N rounds, and the best validated prompt is returned. In the "
    "legacy loop every round starts from P0; the repaired variants of Section 3.2 sample the refiner and, in "
    "the chained variant, start each round from the incumbent prompt.", full=False)

H2("3.1. Problem Formulation")
P("Let (Qv, Av) = {(qi, ai)}, i = 1..M, be a validation set of M question-answer pairs and A(L, P) the "
  "accuracy of the target model L on it under prompt P. Let R(P0) be the set of candidate prompts that the "
  "refinement loop produces from P0. MPIR returns")
P("Pfinal = argmax over P′ in {P0} ∪ R(P0) of A(L, P′),")
P("with ties resolved in favour of P0 (a candidate must be strictly better). Including P0 in the "
  "candidate set is what makes it possible for the procedure to conclude that no refinement helps.")

H2("3.2. Algorithm and Variants")
P("Algorithm 1 is the repaired loop. It scores P0 first, so the incumbent score Abest starts at A(L, P0) "
  "instead of minus infinity. In the original implementation the incumbent score started at minus infinity, "
  "so the first round's candidate was accepted whatever it scored, which is the defect Section 5.9 "
  "quantifies. Choosing among {P0} ∪ candidates by held-out score is standard in APO loops, so we regard "
  "this as an implementation defect of ours, not as a general property of the method.")

ALGO(
    "Algorithm 1. MPIR with a validation gate (repaired)",
    require=("L: target LLM; L′: refining LLM; Meval, Mrefine: meta-prompts; R: rubric; P0: prompt from an "
              "APO run; (Qv, Av): validation set; N: rounds; mode in {independent, chained}; τ: sampling "
              "temperature"),
    ensure="Pfinal",
    steps=[
        "Pbest ← P0; Abest ← Validate(L, P0, (Qv, Av)); H ← empty list      ▷ score the baseline first",
        "for t = 1 to N do",
        "    Psrc ← P0 if mode = independent else Pbest",
        "    S ← Evaluate(Meval, L′, Psrc, R; τ)                          ▷ rubric critique",
        "    P′ ← Refine(Mrefine, L′, Psrc, S, H; τ)                      ▷ H is empty when independent",
        "    outcome ← Classify(P′)         ▷ fallback, echo, duplicate or validated",
        "    if outcome = validated then",
        "        A′ ← Validate(L, P′, (Qv, Av)); append (P′, A′) to H",
        "        if A′ > Abest then Pbest ← P′; Abest ← A′ end if      ▷ strict: a tie keeps the incumbent",
        "    end if",
        "end for",
    ],
    ret="Pbest",
)

P("The pilot ran what we call the legacy loop: Psrc = P0 in every round, H empty and τ = 0. Because "
  "Evaluate and Refine then receive identical input in every round, the seven rounds were one request "
  "repeated, up to serving non-determinism, and 'seven rounds without acceptance' meant roughly one "
  "distinct attempt per cell. The repaired loop differs between rounds in two ways: independent mode "
  "samples at τ = 0.7 so each round is a fresh draw, and chained mode refines the current incumbent and "
  "shows the model the earlier candidates with their validation scores, as OPRO does with its "
  "trajectory ‹22›. We study seven variants (Table 2). Two of them, V1P and V2P, are format-protected: the "
  "answer-format line is appended to every candidate, as it is for the rubric-free rewrite and for the "
  "prompts of the base optimizers. They were added after an audit found that the refinement request does not "
  "ask the refiner to keep the answer-format line, which the 1.7B refiner then dropped in most candidates "
  "(Section 5.2); they are exploratory.")

TABLE("Table 2. Refinement variants studied. All use the same gate (strict improvement over the "
      "incumbent on 25 validation questions) and N = 7 rounds. V3 candidates, and those of V1P and V2P, carry "
      "the answer-format line; V1 and V2 candidates are scored as the refiner wrote them.",
      header=["Variant", "Refines", "History", "Sampling", "Rubric text"],
      rows=[
          ["V0 legacy (as in the pilot)", "P0", "none", "greedy", "original"],
          ["V1 independent", "P0", "none", "τ = 0.7", "corrected"],
          ["V2 chained (primary)", "incumbent", "scored candidates", "τ = 0.7", "corrected"],
          ["V2L chained", "incumbent", "scored candidates", "τ = 0.7", "original"],
          ["V3 gated free rewrite", "P0", "none", "τ = 0.7", "none (no rubric)"],
          ["V1P independent, format-protected", "P0", "none", "τ = 0.7", "corrected"],
          ["V2P chained, format-protected", "incumbent", "scored candidates", "τ = 0.7", "corrected"],
      ], full=True, colw=[0.27, 0.17, 0.22, 0.12, 0.22])

H2("3.3. The Seven-Criteria Rubric")
P("The rubric was assembled from the prompting literature ‹44,45›, industry guidance ‹40,41,42,43› and "
  "iterative experimentation on BBH-style tasks. It scores a prompt from 1 to 5 on: (1) Role Prompting, "
  "(2) Step-Back Reasoning, (3) Guided Chain of Thought, (4) Instruction and Separation, (5) Output "
  "Format with Examples, (6) Worked Reasoning in Examples and (7) Conclusion, for a total out of 35. "
  "The evidence behind the criteria is uneven. Chain-of-thought and demonstrations are well supported "
  "‹7,10,6›. Role prompting is the least supported, since its benefit is inconsistent ‹16,104›. The "
  "Conclusion criterion extrapolates positional-bias results from long contexts ‹15,14› to short prompts, "
  "and the vendor page we cite for it states that instruction placement made no difference for "
  "GPT-4-class models ‹43›. Industry sources are practitioner guidance rather than empirical evidence; "
  "they are living documents, so the reference list gives archived copies and access dates. The rubric was "
  "refined during the first evaluation using BBH tasks, including the three Tier-1 tasks, so Tier 1 is not "
  "independent of the rubric's development; the six random Tier-2 tasks are.")
P("The rubric text used in the pilot was internally inconsistent: it announced 'five defined "
  "criteria', asked for 'all 6 criteria' and 'Total Score: X/25' while listing seven criteria. We found "
  "this during the audit. The primary variants use a corrected text (seven criteria, X/35); variant V2L "
  "keeps the original text as a sensitivity check.")

H2("3.4. Evaluation, Refinement and Output Guards")
P("The evaluation meta-prompt (Figure 2) asks the refining model to apply the rubric to the prompt and "
  "report scores, strengths, weaknesses and 7-10 suggestions. The refinement request (Figure 3) asks "
  "for a rewrite wrapped in <START> and <END>; in chained mode it also lists the earlier candidates "
  "with their validation accuracy and asks the model not to repeat them. Three output guards exist, and "
  "each outcome is logged per round. A response without delimiters is a fallback: it cannot be "
  "validated, and the pilot's code returned P0 in that case, which ties the baseline by construction. A "
  "candidate containing phrases verbatim from the evaluation template, or APE's induction template, is an "
  "echo and is rejected before validation. A candidate textually identical to the incumbent, to P0 or to "
  "an earlier candidate is a duplicate. Only the remaining candidates are validated. The refinement request of V1 and V2 does not ask the refiner "
  "to keep the answer-format line (the free-rewrite prompt does), and a candidate is scored exactly as "
  "written; this asymmetry matters for small refiners and is examined in Section 5.2.")

CODE([
    "### Task Overview",
    "You are a Senior Prompt Engineer with two main tasks:",
    "1. Evaluate prompts using a 7-criteria rubric.  2. Refine prompts by applying evaluation feedback.",
    "### Prompt Under Evaluation",
    "[Prompt]: {prompt}",
    "### Instructions",
    "1. Review Prompt  2. Apply Rubric (seven defined criteria)",
    "3. Score & Justify: score 1-5, one strength, one weakness, a rationale per criterion",
    "4. Calculate Total: cumulative score out of 35",
    "5. Recommend 7-10 actionable improvements  6. Report in the standardized format",
    "### Seven-Criteria Evaluation Rubric",
    "1. Role Prompting  2. Step Back  3. Guided Chain of Thought  4. Instruction & Separation",
    "5. Output Format with Examples  6. Worked Reasoning in Examples  7. Conclusion",
    "### Output Format",
    "1. Role Prompting - X/5 - Strength: ... - Improvement: ... - Rationale: ...",
    "(continue for all 7 criteria)   **Total Score: X/35**   Refinement Summary: ...",
], caption="Figure 2. Evaluation meta-prompt (corrected text; abridged from the repository file).",
    full=True)

CODE([
    "Refine the prompt by applying all suggestions from the evaluation.",
    "Make sure to wrap the refined prompt with <START> and <END>",
    "",
    "[chained mode adds: Earlier refinement attempts ... did NOT beat the current best prompt on the",
    " validation questions; do not repeat them: Attempt k (validation accuracy a vs current best b): ...]",
], caption="Figure 3. Refinement request used by the independent (top) and chained (bottom) variants.",
    full=False)

H2("3.5. The Validation Gate")
P("A candidate is accepted only if its accuracy on the 25 validation questions strictly exceeds the "
  "incumbent's. With 25 questions accuracy moves in steps of 4 percentage points, and one question of "
  "serving noise equals a step, so acceptance is rare by design. Whether the gate or the candidates "
  "explain the absence of accepted refinements is research question RQ2, which we answer by scoring every "
  "candidate on the full test split (the oracle analysis) and by re-applying other acceptance rules "
  "offline (Section 5.4).")

# ===========================================================================
H1("4. Experimental Design")

H2("4.1. Pre-Specification and Decision Rules")
P("The pilot's go/no-go rule was written in the project's planning document (REBUILD.md, commit 7612aa0, "
  "2026-09-03) before the pilot ran. That is a time-stamped plan in a version-controlled repository, not an "
  "entry in a public registry, so we describe the study as pre-specified, not pre-registered; several pipeline "
  "fixes were also made while the pilot ran (Appendix C). The chronology is: first evaluation and audit "
  "(before September 2026); pilot, 2026-09-04 and 05; first submission to this journal, 2026-09-05; decision "
  "and reviews, 2026-10-19; revision protocol committed 2026-10-19 (commit 8fd1397), before any revised "
  "experiment; revised experiments, 2026-10-20 and 21. The decision rules below are reproduced from that "
  "protocol. Its dated amendments, which include one change of the refiners used for D2 and later "
  "exploratory additions, are listed with their dates in Appendix F.")
BULLETS([
    ("D1 (RQ1).",
     "Primary outcome per cell: paired difference in test accuracy between the final MPIR prompt and P0 "
     "(10,000-resample paired bootstrap, 95% percentile interval); a cell is a win if the difference is "
     "positive and the interval excludes zero. The effect 'reproduces' if at least 15 of 45 cells are wins "
     "and the mean cell difference is at least +2.0 points; it is 'null' if at most 2 of 45 are wins and the "
     "mean lies within ±1.0 point; otherwise 'mixed'. The protocol did not say how the up to three MPIR seeds "
     "of a cell are combined: we pool them by averaging the per-example outcomes of the final prompts over "
     "the seeds before the bootstrap. The planned 135 runs (45 cells times three MPIR seeds) were cut to "
     "80 for compute reasons, so cells have one to three runs."),
    ("D2 (RQ3).",
     "Capability dependence is supported if the mean cell difference with the strongest refiner exceeds that "
     "with the 1.7B refiner by at least 2.0 points and at least 3 of 9 cells are wins with that refiner. The "
     "protocol named a local 8B refiner; because the shared GPU was too slow for the extra servers, the "
     "strongest-refiner role was reassigned on 2026-10-20 (before any result with these refiners was "
     "inspected) to GPT-6 Luna, with Qwen3-235B-A22B as an intermediate rung. Any statement about "
     "Qwen3-235B against D2 is therefore post hoc."),
    ("D3 (RQ2).",
     "The gate is limiting if the best oracle candidate beats P0 on test by at least 3.0 points in at "
     "least 3 of 9 cells although the gated final prompt did not."),
])
P("McNemar, GEE and Wilcoxon tests on identical prompts cannot be informative, and the Wilcoxon test on "
  "three tasks cannot go below p = 0.25, so none is reported as inference. Rubric-accuracy correlations, "
  "lenient parsing, the uncorrected-algorithm comparison, the noise floor and the minimum detectable "
  "effect are descriptive.")

H2("4.2. Data, Splits and Task Selection")
P("BBH has 23 tasks in 27 task files, about 6,511 examples ‹27›. Each task is split three ways with "
  "random.Random(seed).sample over the original example order: 25 optimizer-training examples, 25 "
  "validation examples for the gate, and the remainder as test (200 examples for every task used here except "
  "penguins_in_a_table, which has 96). The partition is stored with the "
  "original example indices, is shared by all optimizers and conditions for a given (task, seed), and "
  "no stage touches test data during optimization or refinement. Tier 1 is the three pilot tasks "
  "(hyperbaton, ruin_names, penguins_in_a_table) at seeds 42, 43 and 44. They were chosen because the "
  "leaked first evaluation showed the largest gains there, so they are a biased sample and may be "
  "subject to winner's-curse effects. Tier 2 is six tasks drawn uniformly at random, without replacement, "
  "from the other 24 task files with random.Random(20261019): temporal_sequences, movie_recommendation, "
  "disambiguation_qa, geometric_shapes, salient_translation_error_detection and sports_understanding, at "
  "seed 42. In total there are 45 cells, each with its own base prompt P0.")
BULLETS([
    ("Cell.", "A (task, optimizer, split seed) combination with its own split and its own base prompt P0."),
    ("Run.", "One execution of a refinement variant on a cell with one MPIR seed (the seed of the refiner's "
              "sampling); a cell has one to three runs."),
    ("Variant, round.", "A refinement procedure (V0 to V3, V1P, V2P, V2L; Table 2); one round is one "
                        "evaluate-refine-validate step, with N = 7 rounds per run."),
    ("Split seed, MPIR seed.", "The seed that draws the train, validation and test split and the optimizer's "
                               "own randomness (42, 43, 44), and the seed of the refiner's sampling (1 to 3)."),
])

H2("4.3. Models and Serving")
P("The target model is Qwen3-1.7B ‹68› with thinking disabled and greedy decoding, at most 1024 new "
  "tokens. The refining model is the same model in the main grid. In the capability experiment the refiner is "
  "Qwen3-235B-A22B (the 2507 instruct release, a mixture-of-experts model from the target's family) or GPT-6 "
  "Luna ‹127›, both reached through the OpenRouter API on 2026-10-20 and 21 with temperature 0.7, a 3000-token "
  "limit and, for GPT-6 Luna, minimal reasoning effort. Meta-prompt calls allow 1500 (evaluation) and 3000 "
  "(refinement) new tokens. The target and the 1.7B refiner are served with vLLM ‹123› 0.28.0 under WSL2 on one "
  "16 GB RTX 5000 Ada GPU, bf16 weights, a maximum context of 8192 tokens (Appendix C). Greedy decoding is not bit-reproducible under continuous batching, so we measure the "
  "resulting noise (Section 5.7).")

H2("4.4. Base Optimizers and Conditions")
P("APE and ProTeGi are clean-room implementations of the published algorithms ‹19,20›, run with few_shot_count "
  "= 0, hence exemplar-free; ProTeGi scores every successor on a full minibatch instead of using the "
  "original bandit scheme. PromptWizard ‹23› is run with synthetic-example generation disabled; its "
  "three in-context examples are drawn from the 25 training examples with model-generated reasoning, and "
  "its generated expert identity is not used at evaluation. Hyperparameters are in Table 3. Many of the APE and ProTeGi base prompts are not optimized in any strong sense: "
  "for two of the three pilot tasks they are the seed instruction \"Let's think step by step.\" plus the "
  "answer-format line, and the APE prompt for hyperbaton is its own induction template (Appendix A). "
  "PromptWizard is a reduced configuration, because synthetic-example generation was disabled. The base "
  "prompts are therefore a mix of weak and degenerate starting points, which limits what refining them can show. Only the "
  "expert few-shot condition uses the three BBH-author exemplars ‹27›; the first manuscript's statement "
  "that APE and ProTeGi also used them was wrong. Reference conditions are zero-shot CoT ‹8› and expert "
  "few-shot CoT. The free-rewrite control (Appendix A.3) rewrites P0 with no rubric; V3 adds the gate.")

TABLE("Table 3. Hyperparameters of the base optimizers and of MPIR as actually used.",
      header=["Component", "Parameter", "Description", "Value used"],
      rows=[
          ["PromptWizard", "mutate_refine_iterations", "Outer rounds of mutate-then-refine over the task description", "3"],
          ["PromptWizard", "mutation_rounds", "Style-mutation rounds inside one outer round", "3"],
          ["PromptWizard", "style_variation", "Prompt variants generated per call", "5"],
          ["PromptWizard", "refine_task_eg_iterations", "Critique rounds on instruction and examples", "3"],
          ["PromptWizard", "few_shot_count / seen_set_size", "In-context examples / training examples", "3 / 25"],
          ["APE", "num_candidates, iterations", "Induced candidates; resampling rounds", "10, 2"],
          ["APE", "num_scoring_examples, top_n", "Scoring minibatch; candidates kept", "5, 3"],
          ["ProTeGi", "beam_width, num_steps", "Beam width; expansion rounds", "3, 3"],
          ["ProTeGi", "minibatch_size, eval_batch_size", "Error minibatch; scoring batch", "5, 8"],
          ["MPIR", "N, validation size", "Rounds; gate questions", "7, 25"],
          ["MPIR", "refiner temperature, max tokens", "Sampling; evaluation / refinement", "0.7, 1500 / 3000"],
          ["MPIR", "history kept (chained)", "Last candidates shown; characters each", "4, 800"],
          ["MPIR", "guards", "Fallback, echo (5 phrases), duplicate", "see Section 3.4"],
      ], full=True, colw=[0.15, 0.27, 0.40, 0.18])

H2("4.5. Metrics")
P("Accuracy is the share of test examples whose delimited answer equals the gold answer after "
  "removing parentheses, whitespace and case (strict extraction, as in the pilot). Because strict "
  "extraction penalizes correct reasoning that is wrapped badly, we add a lenient accuracy computed "
  "offline from the saved raw outputs: the response counts as correct if the delimited span (or, if there "
  "is none, the last 300 characters) names exactly one option and it is the gold one. Both are reported; "
  "strict extraction remains primary because MPIR's gate uses it.")

H2("4.6. Statistical Analysis")
P("For every cell we report the paired difference in test accuracy between the final MPIR prompt and P0 "
  "with a 95% paired-bootstrap interval over test examples. Cells with identical prompts have a "
  "difference of zero by construction, apart from serving noise. Across cells we report counts of wins, "
  "the mean difference and a hierarchical bootstrap interval that resamples tasks and then the cells "
  "within each selected task. The noise floor is the distribution of accuracy differences between repeated "
  "scorings of the same prompt, and the minimum detectable effect at 80% power and α = 0.05 follows from the "
  "paired discordance rate ‹56,57›. Intervals use the percentile bootstrap ‹126› and, for single accuracies, "
  "the Wilson interval ‹125›.")

H2("4.7. Harness Defects and Deviations")
P("Running against a live model exposed defects that mocked tests could not (truncation of long "
  "responses, answer normalization, delimiter mismatches in expert exemplars, template mismatches between "
  "an optimizer's own and MPIR's evaluation harness, two echo patterns and the unscored baseline). All "
  "are described with their magnitude in Appendix C and were fixed before the numbers reported here "
  "were produced.")
