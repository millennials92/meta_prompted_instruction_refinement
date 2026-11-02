# -*- coding: utf-8 -*-
from blocks_api import H1, H2, H3, P, BULLETS, TABLE, CODE  # noqa: F401
from study_numbers import N
from rev2_numbers import OFF, NEW, LEN, MID

CAP, CAPL, MG, NOISE, LEG = N["capability"], N["capability_lenient"], N["main_grid"], N["noise"], N["legacy"]
C, EX, EXL = N["composition"], N["exemplar"], N["exemplar_lenient"]


def pp(x, d=1):
    return f"{100 * x:+.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


V1P, V2P = C.get("v1p|1.7b"), C.get("v2p|1.7b")
_prot_validated = (V1P["validated"] + V2P["validated"]) if V1P and V2P else 0
_prot_better = (V1P["val_better"] + V2P["val_better"]) if V1P and V2P else 0

# ===========================================================================
H1("6. Discussion")

H2("6.1. What the Evidence Supports")
P('With Qwen3-1.7B as target and refiner, the rubric-guided layer as originally specified (V2) did not improve the prompts of any of three optimizers on any of nine tasks: the final prompt equalled P0 in 79 of 80 runs (exact one-sided 95% upper bound on the acceptance rate 5.8%), and by the decision rule fixed in advance the outcome is null (Section 5.3). The treatment, not only the idea, was at fault: the refinement request did not protect the answer-format line, so 93-95% of the candidates lost it and scored zero under strict extraction, and about 60% of rounds never produced a usable candidate. We therefore re-ran the grid with a format- and example-preserving treatment (V2F, 44 cells) after a positive control showed that the gate and the harness accept a known-good candidate in 24 of 26 cells in which it helps. Acceptance rose to 15 of 44 runs, but against its own baseline the final prompts win in 4 cells under strict scoring and in 1 under lenient scoring, with 3 significant lenient losses. In this setup the treatment can succeed and still does not help. This is a result about a 1.7B model refining its own, mostly weak, prompts, and it says little about rubric-guided refinement by a capable model.')
P("The evidence on more capable refiners is exploratory (nine cells, one or two runs per condition). Qwen3-235B produced significant strict gains in four cells in two MPIR seeds and GPT-6 Luna in one. Most of these gains repair the answer tag on base prompts that reason about as well as before; under lenient extraction, which we validated on 400 labelled responses, they reverse in the first MPIR seed and vanish in the second. A rubric-free gated rewrite by the same refiner reaches strict gains of the same kind, and the paired difference between the rubric-guided and the rubric-free variant is +2.7 points with an interval [-3.5, +9.7] that includes zero (Section 5.6.1), so the data cannot separate 'the rubric adds nothing' from 'the rubric adds a modest benefit'. The intermediate open-weight refiner (Section 5.6.2) does not change this picture. The one gain that survives both metrics, hyperbaton with PromptWizard, is a single cell on a base prompt that opens with a garbled instruction and scores near chance. Criterion-level ablations were not run and the original GPT-4o and GPT-3.5-turbo configuration was not repeated, so the first evaluation's per-task gains remain unexplained and should not be cited as evidence for MPIR.")
P('A second lesson concerns measurement. Answer extraction decides what the gate rewards: in our runs a strict gate accepted rewrites that repaired the tag even where lenient accuracy did not improve, and exemplar-free baselines look far weaker under strict scoring than under lenient scoring. Reporting a lenient metric next to the strict one exposed both effects.')


H2("6.2. Practical Guidance, Scoped to What Was Tested")
P("The following points follow from these experiments, with Qwen3 models, nine BBH tasks and the "
  "optimizers listed; they are not general rules. (i) Before adding a refinement layer, inspect how "
  "answers are extracted and check a lenient metric: one worked example of the answer format moved strict "
  f"accuracy of exemplar-free prompts by {pp(EX['mean_gain_by_optimizer']['ape'], 1)} and "
  f"{pp(EX['mean_gain_by_optimizer']['protegi'], 1)} points (APE, ProTeGi), and "
  f"{pp(EXL['mean_by_optimizer']['ape'], 1)} and {pp(EXL['mean_by_optimizer']['protegi'], 1)} points under lenient "
  "extraction. (ii) Do not accept a model's rewrite of a prompt without scoring the original prompt as a "
  f"baseline; in a simulation that applies the original accept-first rule to the oracle-scored candidates of the "
  f"independent variant V1 (one MPIR seed, nine cells), accuracy fell by about {abs(100 * LEG['mean_legacy_delta']):.0f} "
  f"points on average, whereas the actual legacy loop produced identical prompts in all nine pilot cells. (iii) Gate on held-out data, but note that identical "
  f"prompts scored twice differ on {pct(NOISE['mean_discordance'], 1)}% of examples, so differences of a few "
  "points on a few hundred questions are noise. (iv) When the refining model is as small as the target, expect "
  "most rewrites to be unusable (echoes, missing delimiters, a dropped answer format) and monitor that rate; "
  "protect the answer-format line and show one worked example of the bare tag, since the line alone did not restore "
  "accuracy (Section 5.2).")

H2("6.3. Threats to Validity")
H3("6.3.1. Internal Validity")
P('The serving stack is not deterministic (Section 5.7): repeated scoring of one prompt spans 3.3 points on average (standard deviation 1.4), which the gate and the minimum detectable effect (5.0 points) take into account but cannot remove. The server was restarted several times with changed settings, so a few interrupted runs combine scorings taken under different configurations; rescoring the prompts behind every exploratory claim in one session left the conclusions unchanged (Section 5.7). The main grid has 80 runs instead of the planned 135, oracle scoring covers MPIR seed 1 only, the strong-refiner runs use one MPIR seed (a second for one refiner), and the format-protected variants cover only part of the nine Tier-1 seed-42 cells (Section 5.2). Two audit findings, the format asymmetry of V1 and V2 and a lenient parser that rejected bare letters, were found late; the parser was corrected and the asymmetry was addressed by new variants (V1P, V2P, V2F) while the original V1 and V2 results were kept as run. The grid was re-run with the format- and example-preserving variant V2F, with one MPIR seed per cell and on 44 of 45 cells (one request exceeded the context window). '
  "The follow-up runs and the protected variants were added after results were seen. The "
  "refiner's rewrite is gated on 25 questions, so even V2F's accepted candidates may be noise; the strong-refiner "
  "models were reached through a router that does not report the serving provider, so those runs may not be "
  "reproducible; echo and fallback rounds were audited on a sample of 80 by a language-model assistant and not by "
  "human raters, and the same holds for the lenient-parser labels. The APE prompt for hyperbaton "
  "is a degenerate artifact of our implementation (Appendix A), and many APE and ProTeGi base prompts are weak, "
  "which limits what refining them can show.")
H3("6.3.2. External Validity")
P("One target model family and size was used in every run. Tier 1 was chosen from a biased statistic and "
  "its tasks were also used while the rubric was developed; Tier 2 is a random draw of six tasks from 24, with "
  "one split seed. The strong-refiner experiment covers nine cells. BBH is a public benchmark, and its questions or close variants may "
  "occur in the pre-training data of Qwen3 and of the strong refiners; contamination would raise all base "
  "accuracies and shrink the room for any refinement layer to help, and we did not test for it. The base prompts "
  "are produced by re-implemented optimizers driven by a 1.7B model, and 21 of the 45 are the seed instruction "
  "plus the format line (Appendix G), so the study speaks to refining such prompts and not to refining the output of "
  "well-tuned optimizers. The original closed-model "
  "configuration, including GPT-4o as refiner, was not rerun; GPT-6 Luna is a different model and not a "
  "replication of it.")
H3("6.3.3. Construct and Conclusion Validity")
P("Strict answer extraction largely measures output formatting; the lenient metric is an approximation that we "
  "validated on 400 labelled responses (precision 98.4% and recall 96.8% against labels from an AI assistant). "
  "The rubric and its judge are weakly related to accuracy (Section 5.8), no human ratings were "
  "collected, and the rubric's criteria were not ablated. The statistics are descriptive by design: with "
  "identical prompts in most cells, tests of differences would be uninformative, the nine pilot cells share "
  "three tasks and three optimizers and are not independent, and the cell-level wins of the strong refiners "
  "carry no multiplicity correction.")

# ===========================================================================
H1("7. Conclusion")
P("We re-tested a rubric-guided refinement layer on prompts from small-model optimizers after repairing its refinement loop and fixing the decision rules in advance, with listed amendments. Three findings stand. First, with a 1.7B model refining its own prompts the original treatment changed the prompt in one of 80 runs and won in none of 45 cells, because that refiner often dropped the answer format and rarely produced a usable candidate; this test therefore could not separate a failed idea from a failed treatment. Second, a format- and example-preserving treatment, run after a positive control showed that the gate accepts a known-good candidate in 24 of 26 cells in which it helps, had candidates accepted in 15 of 44 runs, but the final prompts won in 1 cell and lost significantly in 3 under a lenient metric that we validated on labelled responses. Third, a much larger refiner obtained strict-metric gains in several cells, but most do not hold under lenient scoring, and a rubric-free rewrite reaches gains of the same kind in some cells, so the rubric is not shown to be what helps; the comparison is exploratory and cannot exclude a modest benefit. Two measurement lessons accompany these findings: strict answer extraction accounts for most of the apparent weakness of exemplar-free baselines, and identical prompts do not score identically on this serving stack. These conclusions are limited to Qwen3-1.7B as target and as the main refiner, nine BBH tasks, and three re-implemented optimizers whose base prompts are often weak. Whether a capable refiner with a protected format gains from a rubric, and whether the first evaluation's GPT-4o results hold, remain unknown. Code, per-round logs and generations are public.")

# ===========================================================================
H1("Acknowledgement")
P("The authors thank the reviewers for their constructive comments on earlier versions of this manuscript.")

H1("Funding")
P("This research received no specific grant from any funding agency in the public, commercial, or "
  "not-for-profit sectors.")

H1("Ethical Statement")
P("This study did not involve human participants, human data, or animal subjects, and did not require "
  "ethical approval.")

H1("Declaration of Competing Interest")
P("The authors declare that they have no conflicts of interest.")

H1("Data Availability Statement")
P("The datasets used in this study are drawn from the publicly available Big-Bench Hard benchmark. Code, "
  "prompts, the revision protocol, per-round logs, generations, and analysis outputs are openly available at "
  "https://github.com/millennials92/meta_prompted_instruction_refinement.")

H1("Author Contribution Statement")
P("Linh Nguyen: Conceptualization, Methodology, Software, Investigation, Formal analysis, Data curation, "
  "Writing - original draft, Visualization. Quang-Vinh Dang: Conceptualization, Methodology, Supervision, "
  "Writing - review & editing, Project administration. Minh Ngoc Dinh: Methodology, Validation, Writing - "
  "review & editing. Thuy Nguyen: Investigation, Validation, Writing - review & editing.")

H1("Declaration of Generative AI and AI-Assisted Technologies in the Writing Process")
P("Generative AI tools were used solely to improve the grammar and readability of the text. All ideas, "
  "analyses, and conclusions are solely those of the authors. An AI-assisted coding tool was used to "
  "generate initial code snippets during development; all code was reviewed, verified, and adapted by "
  "the authors. Language-model assistants were also used to label sampled echo and fallback "
  "rounds and to label responses for the lenient-parser validation (Sections 5.1.1 and 5.2), to verify "
  "bibliographic entries, and to help edit the analysis code; these labels are screening aids released for "
  "re-checking, not human annotations, and all analyses, claims and conclusions are the authors' responsibility.")
