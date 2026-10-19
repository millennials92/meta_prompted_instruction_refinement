# -*- coding: utf-8 -*-
from blocks_api import H1, H2, H3, P, BULLETS, TABLE, CODE, ALGO  # noqa: F401
from study_numbers import N  # noqa: F401

# ===========================================================================
H1("3. The MPIR Layer Under Test")

P("MPIR is a three-stage cycle applied to a prompt P0 that an upstream APO method has already produced: "
  "(1) a language model scores P0 against a seven-criteria rubric of manual prompting heuristics and "
  "recommends improvements; (2) it rewrites P0 accordingly; (3) the rewrite is scored on validation "
  "questions and kept only if it beats the incumbent (Figure 1). We describe it here as the object "
  "under test, not as a contribution.")

H2("3.1. Problem Formulation")
P("Let (Qv, Av) = {(qi, ai)}, i = 1..M, be a validation set of M question-answer pairs and A(L, P) the "
  "accuracy of the target model L on it under prompt P. Let R(P0) be the set of refinements that the "
  "refining model L′ produces from P0. MPIR returns")
P("Pfinal = argmax over P′ in {P0} ∪ R(P0) of A(L, P′),")
P("with ties resolved in favour of P0 (a candidate must be strictly better). Including P0 in the "
  "candidate set is what makes it possible for the procedure to conclude that no refinement helps.")

H2("3.2. Algorithm and Variants")
P("Algorithm 1 is the repaired loop. It scores P0 first, so the incumbent score Abest starts at A(L, P0) "
  "instead of minus infinity. In the original implementation the incumbent score started at minus infinity, "
  "so the first round's candidate was accepted whatever it scored, which is the defect Section 5.8 "
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
  "trajectory ‹22›. We study five variants (Table 2).")

TABLE("Table 2. Refinement variants studied. All use the same gate (strict improvement over the "
      "incumbent on 25 validation questions) and N = 7 rounds.",
      header=["Variant", "Refines", "History", "Sampling", "Rubric text"],
      rows=[
          ["V0 legacy (as in the pilot)", "P0", "none", "greedy", "original"],
          ["V1 independent", "P0", "none", "τ = 0.7", "corrected"],
          ["V2 chained (primary)", "incumbent", "scored candidates", "τ = 0.7", "corrected"],
          ["V2L chained", "incumbent", "scored candidates", "τ = 0.7", "original"],
          ["V3 gated free rewrite", "P0", "none", "τ = 0.7", "none (no rubric)"],
      ], full=True, colw=[0.30, 0.14, 0.22, 0.14, 0.20])

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
  "they are living documents, so the reference list gives archived copies and access dates.")
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
  "an earlier candidate is a duplicate. Only the remaining candidates are validated.")

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
P("The first pilot's go/no-go rule was written in the project's planning document (REBUILD.md, commit "
  "7612aa0, 2026-09-03) before the pilot ran. It was a version-controlled plan, not an entry in a public "
  "registry, and the pilot was not pre-registered in the strict sense in which we used that word in the "
  "first submission. Several pipeline fixes were also made while the pilot was running (Appendix C). For "
  "this revision we wrote the protocol in REVISION_PROTOCOL.md and committed it (commit 8fd1397, "
  "2026-10-19) before running any revised experiment; its commit time-stamps the rules below, which are "
  "reproduced from it. Dated amendments are listed in Appendix F. This is also not a registry entry.")
BULLETS([
    ("D1 (RQ1).",
     "Primary outcome per cell: paired difference in test accuracy between the final MPIR prompt and P0 "
     "(10,000-resample paired bootstrap, 95% percentile interval); a cell is a win if the difference is "
     "positive and the interval excludes zero. The effect 'reproduces' if at least 15 of 45 cells are wins "
     "and the mean cell difference is at least +2.0 points; it is 'null' if at most 2 of 45 are wins and the "
     "mean lies within ±1.0 point; otherwise 'mixed'."),
    ("D2 (RQ3).",
     "Capability dependence is supported if the mean cell difference with the 8B refiner exceeds that "
     "with the 1.7B refiner by at least 2.0 points and at least 3 of 9 cells are wins with the 8B refiner."),
    ("D3 (RQ2).",
     "The gate is limiting if the best oracle candidate beats P0 on test by at least 3.0 points in at "
     "least 3 of 9 cells although the gated final prompt did not."),
])
P("McNemar, GEE and Wilcoxon tests on identical prompts cannot be informative, and the Wilcoxon test on "
  "three tasks cannot go below p = 0.25, so none is reported as inference. Rubric-accuracy correlations, "
  "lenient parsing, the uncorrected-algorithm comparison, the noise floor and the minimum detectable "
  "effect are descriptive.")

H2("4.2. Data, Splits and Task Selection")
P("BBH has 23 tasks in 27 task files, about 6,511 examples ‹52›. Each task is split three ways with "
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
  "seed 42. In total there are 45 cells, one per (task, optimizer, seed), each with its own P0.")

H2("4.3. Models and Serving")
P("The target model is Qwen3-1.7B ‹68› with thinking disabled and greedy decoding, at most 1024 new "
  "tokens. The refining model is the same model in the main grid and Qwen3-4B-AWQ or Qwen3-8B-AWQ in the "
  "capability experiment; meta-prompt calls allow 1500 (evaluation) and 3000 (refinement) new tokens. "
  "Refiner sampling uses temperature 0.7. Everything is served with vLLM 0.28.0 under WSL2 on one "
  "16 GB RTX 5000 Ada GPU, bf16 weights for the 1.7B model, a maximum context of 8192 tokens "
  "(Appendix C). Greedy decoding is not bit-reproducible under continuous batching, so we measure the "
  "resulting noise (Section 5.7).")

H2("4.4. Base Optimizers and Conditions")
P("APE and ProTeGi are clean-room implementations of the published algorithms ‹19,20›, run with few_shot_count "
  "= 0, hence exemplar-free; ProTeGi scores every successor on a full minibatch instead of using the "
  "original bandit scheme. PromptWizard ‹23› is run with synthetic-example generation disabled; its "
  "three in-context examples are drawn from the 25 training examples with model-generated reasoning, and "
  "its generated expert identity is not used at evaluation. Hyperparameters are in Table 3. Only the "
  "expert few-shot condition uses the three BBH-author exemplars ‹52›; the first manuscript's statement "
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
  "the mean difference and a hierarchical bootstrap interval that resamples tasks, then optimizers and "
  "seeds within tasks. The noise floor is the distribution of accuracy differences between repeated "
  "scorings of the same prompt, and the minimum detectable effect at 80% power and α = 0.05 follows from the "
  "paired discordance rate ‹56,57›.")

H2("4.7. Harness Defects and Deviations")
P("Running against a live model exposed defects that mocked tests could not (truncation of long "
  "responses, answer normalization, delimiter mismatches in expert exemplars, template mismatches between "
  "an optimizer's own and MPIR's evaluation harness, two echo patterns and the unscored baseline). All "
  "are described with their magnitude in Appendix C and were fixed before the numbers reported here "
  "were produced.")
