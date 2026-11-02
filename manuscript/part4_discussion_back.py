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
_G, _CT, _AB = NEW["grid_v2f"], NEW["controls"], OFF["acceptance_bounds"]
P(f"With Qwen3-1.7B as target and refiner, the rubric-guided layer as originally specified (V2) did not improve the "
  f"prompts of any of three optimizers on any of nine tasks: the final prompt equalled P0 in "
  f"{MG['runs'] - MG['runs_accepting']} of {MG['runs']} runs (exact one-sided 95% upper bound on the acceptance rate "
  f"{pct(_AB['run_upper_one_sided_95'])}%), and by the decision rule fixed in advance the outcome is null "
  "(Section 5.3). The main reason was the treatment: the refinement request did not protect the answer-format line, so "
  "93-95% of the candidates lost it and scored zero under strict extraction, and about 60% of rounds never produced a "
  "usable candidate. We therefore re-ran the grid with a format- and example-preserving treatment (V2F, "
  f"{_G['runs']} cells) after a positive control showed that the gate and the harness accept a known-good candidate in "
  f"{_CT['known_good_accepted']} of {_CT['known_good_cells']} cells in which it helps. Acceptance rose to {_G['accepted_runs']} of {_G['runs']} runs, "
  f"but against its own baseline the final prompts win in {_G['wins_vs_start']} cells under strict scoring and in "
  f"{_G['lenient_wins_vs_start']} under lenient scoring, with {_G['lenient_losses_vs_start']} significant lenient losses, "
  "so the treatment can now succeed and still does not help. This is a result about a 1.7B model refining its own "
  "prompts on weak base prompts, and it says little about rubric-guided refinement by a capable model.")
P("The evidence on a more capable refiner is exploratory. A much larger open-weight refiner produced significant "
  "strict gains in four cells in two MPIR seeds, and a strong closed refiner in one. Most of the strict gains are "
  "repairs of the answer tag on base prompts that reason about as well as before; under lenient extraction, which "
  "we validated on 400 labelled responses, they reverse in the first MPIR seed and vanish in the second. A rubric-free "
  "gated rewrite by the same refiner reaches strict gains of the same kind, and the paired difference between the "
  "rubric-guided and the rubric-free variant is a few points with an interval that includes zero (Section 5.6.1), so "
  "the data cannot separate 'the rubric adds nothing' from 'the rubric adds a modest benefit'. "
  + ("The intermediate open-weight refiner (Section 5.6.2) does not change this picture. " if MID else "")
  + "The one gain that survives both metrics, hyperbaton with PromptWizard, is on a base prompt that opens with a "
  "garbled instruction and scores near chance, and it is a single cell. Criterion-level ablations were not run and the "
  "original GPT-4o and GPT-3.5-turbo configuration was not repeated, so the first evaluation's per-task gains remain "
  "unexplained and should not be cited as evidence for MPIR.")
P("A second lesson is about measurement. Answer extraction decides what the gate rewards: a gate that uses "
  "strict extraction accepts rewrites that fix the tag even when they harm reasoning, and exemplar-free "
  "baselines look far weaker than they are. Reporting a lenient metric next to the strict one exposed both "
  "effects.")

H2("6.2. Practical Guidance, Scoped to What Was Tested")
P("The following points follow from these experiments, with Qwen3 models, nine BBH tasks and the "
  "optimizers listed; they are not general rules. (i) Before adding a refinement layer, inspect how "
  "answers are extracted and check a lenient metric: one worked example of the answer format moved strict "
  f"accuracy of exemplar-free prompts by {pp(EX['mean_gain_by_optimizer']['ape'], 0)} and "
  f"{pp(EX['mean_gain_by_optimizer']['protegi'], 0)} points (APE, ProTeGi), and "
  f"{pp(EXL['mean_by_optimizer']['ape'], 0)} and {pp(EXL['mean_by_optimizer']['protegi'], 0)} points under lenient "
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
P(f"The serving stack is not deterministic: repeated scoring of one prompt spans {pct(NOISE['mean_range'])} points "
  f"on average (standard deviation {pct(NOISE['mean_sd'])}), which the gate and the minimum detectable effect "
  f"({pct(NOISE['mde_noise_only'])} points) take into account but cannot remove. The inference server was "
  "restarted several times and its memory settings were changed during the study; cached scores and "
  "generations were kept across restarts, so a few interrupted runs combine scorings taken under different "
  "settings, with differences of the order of the measured noise. The main grid has 80 runs instead of the "
  "planned 135, oracle scoring covers MPIR seed 1 only, the strong-refiner runs use one MPIR seed (a second "
  "for one refiner), and the format-protected variants cover only part of the nine Tier-1 seed-42 cells (Section 5.2). Two "
  "audit findings, the format asymmetry of V1 and V2 and a lenient parser that rejected bare letters, were "
  "found late: the lenient parser was corrected, and the format asymmetry was addressed by new variants (V1P, V2P, "
  "V2F) while the original V1 and V2 results were kept as run; the second revision re-ran the grid with a "
  "format-preserving treatment (V2F), with one MPIR seed per cell and on 44 of 45 cells (one request exceeded the "
  "context window). The follow-up runs and the protected variants were added after results were seen. The "
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
  "validated on 400 labelled responses (precision and recall above 96%), by an AI labeller. "
  "The rubric and its judge are weakly related to accuracy (Section 5.8), no human ratings were "
  "collected, and the rubric's criteria were not ablated. The statistics are descriptive by design: with "
  "identical prompts in most cells, tests of differences would be uninformative, the nine pilot cells share "
  "three tasks and three optimizers and are not independent, and the cell-level wins of the strong refiners "
  "carry no multiplicity correction.")

# ===========================================================================
H1("7. Conclusion")
P(f"We re-tested a rubric-guided refinement layer on prompts from small-model optimizers after repairing its "
  f"refinement loop and fixing the decision rules in advance, with listed amendments. With a 1.7B model refining its "
  f"own prompts the original treatment changed the prompt in one of {MG['runs']} runs and won in none of "
  f"{MG['n_cells']} cells, because that refiner often dropped the answer format and rarely produced a usable candidate. "
  f"A format- and example-preserving treatment, run after a positive control showed that the gate accepts a "
  f"known-good candidate in {NEW['controls']['known_good_accepted']} of {NEW['controls']['known_good_cells']} cells in which it helps, was accepted "
  f"in {NEW['grid_v2f']['accepted_runs']} of {NEW['grid_v2f']['runs']} runs but won in "
  f"{NEW['grid_v2f']['lenient_wins_vs_start']} cell(s) and lost significantly in "
  f"{NEW['grid_v2f']['lenient_losses_vs_start']} under a lenient metric that we validated on labelled responses. A much "
  "larger refiner obtained strict-metric gains in several cells, but most do not hold under lenient scoring, and a "
  "rubric-free rewrite reaches gains of the same kind in some cells, so the rubric is not shown to be what helps; the "
  "comparison is exploratory and cannot exclude a modest benefit. In a simulation that applies the original "
  f"accept-first rule to the candidates of the independent variant, accuracy would have fallen by about "
  f"{abs(100 * LEG['mean_legacy_delta']):.0f} points (the actual legacy loop produced identical prompts), strict answer "
  "extraction accounts for most of the apparent weakness of exemplar-free baselines, and identical prompts do not score "
  "identically on this serving stack. These conclusions are limited to Qwen3-1.7B as target and as the main refiner, "
  "nine BBH tasks, three re-implemented optimizers whose base prompts are often weak, and the first evaluation's GPT-4o "
  "results remain unverified. Code, per-round logs and generations are public.")

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
  "the authors. Language-model assistants were also used in the second revision to label sampled echo and fallback "
  "rounds and to label responses for the lenient-parser validation (Sections 5.1.1 and 5.2), to verify "
  "bibliographic entries, and to help edit the analysis code; these labels are screening aids released for "
  "re-checking, not human annotations, and all analyses, claims and conclusions are the authors' responsibility.")
