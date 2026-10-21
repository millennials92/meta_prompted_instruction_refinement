# -*- coding: utf-8 -*-
from blocks_api import H1, H2, H3, P, BULLETS, TABLE, CODE  # noqa: F401
from study_numbers import N

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
P(f"With Qwen3-1.7B as target and refiner, the rubric-guided layer, as specified, did not improve the prompts of "
  f"any of three optimizers on any of nine tasks: the final prompt equalled P0 in "
  f"{MG['runs'] - MG['runs_accepting']} of {MG['runs']} runs, and by the pre-specified rule the outcome is "
  f"null (Section 5.3). The refinement rounds were made distinct by sampling and history, the failure is "
  "not caused by the inconsistent rubric text, and a rubric-free gated rewrite is not worse under lenient "
  "scoring (Section 5.5). The main reason is the refiner: most rounds never produce a usable candidate, and the "
  "unprotected candidates lose the answer-format line, so under strict extraction they score zero. "
  + (f"With the format line appended the refiner still reaches validation in a minority of rounds and "
     f"{_prot_better} of {_prot_validated} validated candidates beat the baseline (Section 5.2). " if V1P and V2P else "")
  + "The null is therefore a result about a 1.7B model's ability to follow a long refinement instruction, "
  "and it says little about rubric-guided refinement by a capable model.")
P("The evidence on a more capable refiner is mixed and exploratory. A much larger open-weight refiner "
  "produced significant strict gains in four cells in two MPIR seeds, and a strong closed refiner in one. "
  "Most of the strict gains are, however, repairs of the answer tag on base prompts that reason about as well "
  "as before; under lenient extraction they reverse in the first MPIR seed and vanish in the second, and a "
  "rubric-free gated rewrite by the same refiner reaches strict gains of the same kind. "
  "The one gain that survives both metrics, hyperbaton with PromptWizard, is on a base prompt that opens "
  "with a garbled instruction and scores near chance, and it is a single cell. So the data do not show that the seven-criteria "
  "rubric helps; criterion-level ablations were not run, and the original GPT-4o and GPT-3.5-turbo "
  "configuration was not repeated, so the first evaluation's per-task gains remain unexplained and should "
  "not be cited as evidence for MPIR.")
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
  f"baseline; with a 1.7B refiner the uncorrected loop lowered accuracy by about {abs(100 * LEG['mean_legacy_delta']):.0f} "
  f"points on average in our nine cells (one MPIR seed). (iii) Gate on held-out data, but note that identical "
  f"prompts scored twice differ on {pct(NOISE['mean_discordance'], 1)}% of examples, so differences of a few "
  "points on a few hundred questions are noise. (iv) When the refining model is as small as the target, expect "
  "most rewrites to be unusable (echoes, missing delimiters, a dropped answer format) and monitor that rate; "
  "protect the answer-format line explicitly.")

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
  "found late and corrected after the main grid had run; the main grid was not re-run with protection. The "
  "follow-up runs and the protected variants were added after results were seen. The APE prompt for hyperbaton "
  "is a degenerate artifact of our implementation (Appendix A), and many APE and ProTeGi base prompts are weak, "
  "which limits what refining them can show.")
H3("6.3.2. External Validity")
P("One target model family and size was used in every run. Tier 1 was chosen from a biased statistic and "
  "its tasks were also used while the rubric was developed; Tier 2 is a random draw of six tasks from 24, with "
  "one split seed. The strong-refiner experiment covers nine cells. The original closed-model "
  "configuration, including GPT-4o as refiner, was not rerun; GPT-6 Luna is a different model and not a "
  "replication of it.")
H3("6.3.3. Construct and Conclusion Validity")
P("Strict answer extraction partly measures output formatting, and the lenient metric is an approximation. "
  "The rubric and its judge are weakly related to accuracy (Section 5.8), no human ratings were "
  "collected, and the rubric's criteria were not ablated. The statistics are descriptive by design: with "
  "identical prompts in most cells, tests of differences would be uninformative, the nine pilot cells share "
  "three tasks and three optimizers and are not independent, and the cell-level wins of the strong refiners "
  "carry no multiplicity correction.")

# ===========================================================================
H1("7. Conclusion")
P(f"We re-tested a rubric-guided refinement layer for automatically optimized prompts after repairing its "
  f"refinement loop and fixing the decision rules in advance. With a 1.7B model refining its own prompts the "
  f"layer produced no improvement in {MG['n_cells']} cells (one of {MG['runs']} runs accepted a candidate), "
  "because that refiner rarely produces a usable candidate and often drops the answer format. A much larger refiner "
  "obtained strict-metric gains in several cells, but they do not hold under lenient scoring and a rubric-free "
  "rewrite reaches strict gains of the same kind, so the rubric is not shown to be what helps; one gain, on a "
  "single task, holds under both metrics. With a 1.7B refiner the uncorrected loop would have reduced "
  f"accuracy by about {abs(100 * LEG['mean_legacy_delta']):.0f} points (its effect with the original GPT-4o "
  "refiner is unknown), strict answer extraction accounts for much of the apparent weakness of exemplar-free "
  "baselines, and identical prompts do not score identically on this serving stack. These conclusions are "
  "limited to Qwen3-based targets, nine tasks and the optimizers tested, and the first evaluation's GPT-4o "
  "results remain unverified. Code, per-round logs and generations are public.")

# ===========================================================================
H1("Acknowledgement")
P("The authors thank the reviewers for their constructive comments on the first submission.")

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
  "the authors.")
