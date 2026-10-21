# -*- coding: utf-8 -*-
"""One-off text patches after the audit reports: RQ summary, title, keywords, appendix defect table, captions."""


def edit(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert old in s, (path, old[:70])
        s = s.replace(old, new, 1)
    open(path, "w", encoding="utf-8").write(s)


edit("part1_intro_related.py", [(
    "@@RQSUMMARY@@",
    'In brief: with a 1.7B refiner the layer changed the prompt in "\n'
    '  f"{N[\'main_grid\'][\'runs_accepting\']} of {N[\'main_grid\'][\'runs\']} runs and won in {N[\'main_grid\'][\'wins\']} "\n'
    '  f"of {N[\'main_grid\'][\'n_cells\']} cells, mainly because that refiner rarely produces a usable candidate that keeps "\n'
    '  "the answer format (RQ1, RQ2); a much larger refiner obtained strict-metric gains in several cells, but most "\n'
    '  "of them reverse under lenient scoring and a rubric-free rewrite reaches gains of the same kind (RQ3); and "\n'
    '  "the rubric score is only weakly related to accuracy (RQ4).")')])
# the replacement above sits inside a plain string literal; make the surrounding strings consistent
s = open("part1_intro_related.py", encoding="utf-8").read()
s = s.replace('(Terms such as cell and run are defined in Section 4.2.) In brief: with a 1.7B refiner',
              '(Terms such as cell and run are defined in Section 4.2.) "\n  "In brief: with a 1.7B refiner')
open("part1_intro_related.py", "w", encoding="utf-8").write(s)

edit("content_blocks.py", [
    ('TITLE = ("Does Rubric-Guided Refinement Improve Automatically Optimized Prompts? "\n         "A Pre-Specified Negative-Result Study with Small Open-Weight Models")',
     'TITLE = ("Does Rubric-Guided Refinement Improve Automatically Optimized Prompts? "\n         "A Pre-Specified Negative-Result Study")'),
    ('KEYWORDS = ("Prompt engineering, automatic prompt optimization, meta-prompting, negative results, "\n            "large language models, Big-Bench Hard")',
     'KEYWORDS = ("Automatic prompt optimization, Big-Bench Hard, Large language models, Meta-prompting, "\n            "Negative results, Prompt engineering")'),
])
edit("build_latex_ieee.py", [
    ('shorttitle="Rubric-Guided Prompt Refinement: A Pre-Specified Negative Result"',
     'shorttitle="Does Rubric-Guided Refinement Improve Optimized Prompts?"'),
    ("Manuscript received August 2026; revised October 20, 2026.", "Manuscript received September 5, 2026; revised October 21, 2026."),
])

edit("part5_appendix.py", [
    ('''          ["11", "APE search selected a degenerate prompt", "hyperbaton P0 is APE's own induction template",
           "Left in place and disclosed; same code as the pilot"],''',
     '''          ["11", "APE search selected a degenerate prompt", "hyperbaton P0 is APE's own induction template",
           "Left in place and disclosed; same code as the pilot"],
          ["12", "Candidates of V1 and V2 scored without the answer-format line", "The refinement request does not ask "
           "to keep it; 93-95% of validated 1.7B candidates lack it and score exactly zero under strict extraction",
           "Format-protected variants V1P and V2P (Section 3.2); asymmetry reported"],
          ["13", "Lenient parser rejected a bare letter inside the tags", "Strict accepted what lenient rejected",
           "Lenient now accepts everything strict accepts"],'''),
    ('Items 1-8 were fixed before any number reported in Section 5; items 9-11 were found "\n  "during this revision.',
     'Items 1-8 were fixed before any number reported in Section 5; items 9-13 were found "\n  "during this revision, items 12 and 13 late, after the main grid had run.'),
    ('''], caption="Figure B.3. An accepted candidate from the strong refiner (abridged). It adds a role, step-back "
           "and numbered steps, but also an explicit instruction to wrap the final answer on its own line and a "
           "worked output example, which is the format lever of Section 5.1; the example is generic, not about "
           "adjective order.", full=False)''',
     '''], caption="Figure B.3. An accepted candidate from the strong refiner (abridged). It adds a role, step-back "
           "and numbered steps, and also an explicit instruction to wrap the final answer on its own line and a "
           "worked output example, which repairs the answer tag; the example is generic, not about adjective order. "
           "The strict accuracy of this prompt is higher than that of P0 (39.5% against 10.5%) but its lenient "
           "accuracy is lower (16% against 58%, Section 5.6).", full=False)'''),
])
print("final text patches applied")
