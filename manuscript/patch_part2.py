# -*- coding: utf-8 -*-
"""One-off patch of part2_method_design.py after the audit reports (kept for provenance)."""
p = "part2_method_design.py"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


rep('Let R(P0) be the set of refinements that the "\n  "refining model L′ produces from P0. MPIR returns")',
    'Let R(P0) be the set of candidate prompts that the "\n  "refinement loop produces from P0. MPIR returns")')
rep('which is the defect Section 5.8 "\n  "quantifies.', 'which is the defect Section 5.9 "\n  "quantifies.')
rep('trajectory ‹22›. We study five variants (Table 2).")',
    'trajectory ‹22›. We study seven variants (Table 2). Two of them, V1P and V2P, are format-protected: the "\n'
    '  "answer-format line is appended to every candidate, as it is for the rubric-free rewrite and for the "\n'
    '  "prompts of the base optimizers. They were added after an audit found that the refinement request does not "\n'
    '  "ask the refiner to keep the answer-format line, which the 1.7B refiner then dropped in most candidates "\n'
    '  "(Section 5.2); they are exploratory.")')
rep('          ["V3 gated free rewrite", "P0", "none", "τ = 0.7", "none (no rubric)"],',
    '          ["V3 gated free rewrite", "P0", "none", "τ = 0.7", "none (no rubric)"],\n'
    '          ["V1P independent, format-protected", "P0", "none", "τ = 0.7", "corrected"],\n'
    '          ["V2P chained, format-protected", "incumbent", "scored candidates", "τ = 0.7", "corrected"],')
rep('"incumbent on 25 validation questions) and N = 7 rounds.",',
    '"incumbent on 25 validation questions) and N = 7 rounds. V3 candidates, and those of V1P and V2P, carry "\n'
    '      "the answer-format line; V1 and V2 candidates are scored as the refiner wrote them.",')
rep('they are living documents, so the reference list gives archived copies and access dates.")',
    'they are living documents, so the reference list gives archived copies and access dates. The rubric was "\n'
    '  "refined during the first evaluation using BBH tasks, including the three Tier-1 tasks, so Tier 1 is not "\n'
    '  "independent of the rubric\'s development; the six random Tier-2 tasks are.")')
rep('Only the remaining candidates are validated.")',
    'Only the remaining candidates are validated. The refinement request of V1 and V2 does not ask the refiner "\n'
    '  "to keep the answer-format line (the free-rewrite prompt does), and a candidate is scored exactly as "\n'
    '  "written; this asymmetry matters for small refiners and is examined in Section 5.2.")')

a = s.index("P(\"The first pilot's go/no-go rule")
b = s.index('BULLETS([\n    ("D1 (RQ1).",')
s = s[:a] + '''P("The pilot's go/no-go rule was written in the project's planning document (REBUILD.md, commit 7612aa0, "
  "2026-09-03) before the pilot ran. That is a time-stamped plan in a version-controlled repository, not an "
  "entry in a public registry, so we describe the study as pre-specified, not pre-registered; several pipeline "
  "fixes were also made while the pilot ran (Appendix C). The chronology is: first evaluation and audit "
  "(before September 2026); pilot, 2026-09-04 and 05; first submission to this journal, 2026-09-05; decision "
  "and reviews, 2026-10-19; revision protocol committed 2026-10-19 (commit 8fd1397), before any revised "
  "experiment; revised experiments, 2026-10-20 and 21. The decision rules below are reproduced from that "
  "protocol. Its dated amendments, which include one change of the refiners used for D2 and later "
  "exploratory additions, are listed with their dates in Appendix F.")
''' + s[b:]

rep('''    ("D2 (RQ3).",
     "Capability dependence is supported if the mean cell difference with the 8B refiner exceeds that "
     "with the 1.7B refiner by at least 2.0 points and at least 3 of 9 cells are wins with the 8B refiner."),''',
    '''    ("D2 (RQ3).",
     "Capability dependence is supported if the mean cell difference with the strongest refiner exceeds that "
     "with the 1.7B refiner by at least 2.0 points and at least 3 of 9 cells are wins with that refiner. The "
     "protocol named a local 8B refiner; because the shared GPU was too slow for the extra servers, the "
     "strongest-refiner role was reassigned on 2026-10-20 (before any result with these refiners was "
     "inspected) to GPT-6 Luna, with Qwen3-235B-A22B as an intermediate rung. Any statement about "
     "Qwen3-235B against D2 is therefore post hoc."),''')
rep('''"mean lies within ±1.0 point; otherwise 'mixed'."),''',
    '''"mean lies within ±1.0 point; otherwise 'mixed'. The protocol did not say how the up to three MPIR seeds "
     "of a cell are combined: we pool them by averaging the per-example outcomes of the final prompts over "
     "the seeds before the bootstrap. The planned 135 runs (45 cells times three MPIR seeds) were cut to "
     "80 for compute reasons, so cells have one to three runs."),''')
rep('seed 42. In total there are 45 cells, one per (task, optimizer, seed), each with its own P0.")',
    '''seed 42. In total there are 45 cells, each with its own base prompt P0.")
BULLETS([
    ("Cell.", "A (task, optimizer, split seed) combination with its own split and its own base prompt P0."),
    ("Run.", "One execution of a refinement variant on a cell with one MPIR seed (the seed of the refiner's "
              "sampling); a cell has one to three runs."),
    ("Variant, round.", "A refinement procedure (V0 to V3, V1P, V2P, V2L; Table 2); one round is one "
                        "evaluate-refine-validate step, with N = 7 rounds per run."),
    ("Split seed, MPIR seed.", "The seed that draws the train, validation and test split and the optimizer's "
                               "own randomness (42, 43, 44), and the seed of the refiner's sampling (1 to 3)."),
])''')
rep('''The refining model is the same model in the main grid and Qwen3-4B-AWQ or Qwen3-8B-AWQ in the "
  "capability experiment; meta-prompt calls allow 1500 (evaluation) and 3000 (refinement) new tokens. "
  "Refiner sampling uses temperature 0.7. Everything is served with vLLM 0.28.0 under WSL2 on one "
  "16 GB RTX 5000 Ada GPU, bf16 weights for the 1.7B model, a maximum context of 8192 tokens "
  "(Appendix C).''',
    '''The refining model is the same model in the main grid. In the capability experiment the refiner is "
  "Qwen3-235B-A22B (the 2507 instruct release, a mixture-of-experts model from the target's family) or GPT-6 "
  "Luna ‹127›, both reached through the OpenRouter API on 2026-10-20 and 21 with temperature 0.7, a 3000-token "
  "limit and, for GPT-6 Luna, minimal reasoning effort. Meta-prompt calls allow 1500 (evaluation) and 3000 "
  "(refinement) new tokens. The target and the 1.7B refiner are served with vLLM ‹123› 0.28.0 under WSL2 on one "
  "16 GB RTX 5000 Ada GPU, bf16 weights, a maximum context of 8192 tokens (Appendix C).''')
rep("Hyperparameters are in Table 3.",
    '''Hyperparameters are in Table 3. Many of the APE and ProTeGi base prompts are not optimized in any strong sense: "
  "for two of the three pilot tasks they are the seed instruction \\"Let's think step by step.\\" plus the "
  "answer-format line, and the APE prompt for hyperbaton is its own induction template (Appendix A). "
  "PromptWizard is a reduced configuration, because synthetic-example generation was disabled. The base "
  "prompts are therefore a mix of weak and degenerate starting points, which limits what refining them can show.''')
rep("the mean difference and a hierarchical bootstrap interval that resamples tasks, then optimizers and \"\n  \"seeds within tasks.",
    "the mean difference and a hierarchical bootstrap interval that resamples tasks and then the cells \"\n  \"within each selected task.")
rep("paired discordance rate ‹56,57›.", "paired discordance rate ‹56,57›. Intervals use the percentile bootstrap ‹126› and, for single accuracies, \"\n  \"the Wilson interval ‹125›.")
open(p, "w", encoding="utf-8").write(s)
print("patched")
