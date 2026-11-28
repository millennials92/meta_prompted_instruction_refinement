# -*- coding: utf-8 -*-
"""One-off patch of the appendix and cross-references after the audit reports (kept for provenance)."""
import re

p = "part5_appendix.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, count)


# table numbering: appendix tables use the running auto-numbers 12..16
for old, new in [("Table C1.", "Table 12."), ("Table C2.", "Table 13."), ("Table D1.", "Table 14."),
                 ('f"Table {_label}. Round-level', 'f"Table {_label}. Round-level')]:
    s = s.replace(old, new)
s = s.replace('for _variant, _label in (("v2", "E1"), ("v1", "E2")):', 'for _variant, _label in (("v2", "15"), ("v1", "16")):')
s = s.replace("exactly as saved by the pipeline, not cleaned", "excerpted from the files saved by the pipeline, not cleaned")
s = s.replace("This appendix reproduces prompt text exactly as saved", "This appendix reproduces prompt text excerpted from the files saved")

rep('''P("The first evaluation used GPT-3.5-turbo as target and GPT-4o as refiner on all BBH tasks. It is "
  "reproduced here only because the manuscript's framing referred to it, and Reviewer comments asked for "
  "its numbers. Its validation stage''', '''P("The first evaluation used GPT-3.5-turbo as target and GPT-4o as refiner on all BBH tasks. It is "
  "reproduced here for completeness, because the Introduction refers to it. Its validation stage''')
rep('"Gem. PW", "Gem. MPIR"', '"Gemini PW", "Gemini MPIR"')
rep("PW = PromptWizard; Diff = PW+MPIR minus PW. Exploratory only.", "PW = PromptWizard; Diff = PW+MPIR minus PW (points). Exploratory only.")

# serving table: strong refiners instead of the 4B/8B servers that were never run
rep('''          ["Capability-experiment refiners", "Qwen/Qwen3-4B-AWQ (74d4bd2) and Qwen/Qwen3-8B-AWQ (4da05a8), 4-bit AWQ"],''',
    '''          ["Capability-experiment refiners", "qwen/qwen3-235b-a22b-2507 and openai/gpt-6-luna (reasoning effort minimal) "
           "through the OpenRouter API, 2026-10-20 and 21, temperature 0.7, 3000 new tokens; no local server"],''')
rep('''          ["Server flags", "--gpu-memory-utilization 0.80 --max-num-seqs 96 --seed 0; prefix caching on"],''',
    '''          ["Server flags", "--gpu-memory-utilization 0.80-0.86 --max-num-seqs 64-128 --seed 0; the server was restarted "
           "several times and its memory settings changed (fp8 KV cache was tried and not usable without nvcc; "
           "prefix caching was off for one period); cached scores and generations were kept across restarts"],''')

# Appendix F with dates
a = s.index('H1("Appendix F. Protocol and Deviations")')
s = s[:a] + '''H1("Appendix F. Protocol and Deviations")
P("The decision rules of Section 4.1 were fixed in REVISION_PROTOCOL.md before any revised experiment "
  "ran (commit 8fd1397, 2026-10-19). The amendments below are recorded in that file; none is an entry in a "
  "registry. Amendments 1 to 7 were made before the results of the affected experiment were inspected; "
  "amendments 8 and 9 were made after results were seen and are exploratory.")
BULLETS([
    ("1. 2026-10-20, compute.", "Serving stack vLLM 0.28.0 under WSL2 with measured throughput of about 1.2k "
     "tokens/s; oracle test scoring of every validated candidate for MPIR seed 1 only; one validation score per "
     "candidate in the main grid G1; V2L with one seed and no oracle scoring."),
    ("2. 2026-10-20, sampling.", "Refiner sampling uses temperature 0.7 without top-p (the protocol text said "
     "top-p 0.95), because top-p was about three times slower on this stack."),
    ("3. 2026-10-20, grouping.", "Tier-1 seed-42 cells are run only in the mechanism group G2, which contains the "
     "V2 runs with oracle scoring; G1 covers the other 36 cells."),
    ("4. 2026-10-20, refiners for D2.", "Local 4B and 8B refiners were not run; the strongest-refiner role of D2 "
     "went to GPT-6 Luna and Qwen3-235B-A22B through OpenRouter (Section 4.1)."),
    ("5. 2026-10-20 and 21, scale.", "The main grid has 80 runs instead of the planned 135; MPIR seeds beyond 1 "
     "were cut after seed 1 was complete."),
    ("6. 2026-10-20, pooling.", "Cell-level differences pool the MPIR seeds of a cell by averaging per-example "
     "outcomes (Section 4.1)."),
    ("7. 2026-10-20, optimizer implementations.", "No change to APE, ProTeGi or PromptWizard code or hyperparameters "
     "between the pilot and the extra seeds and tasks."),
    ("8. 2026-10-21, exploratory.", "After the Qwen3-235B refiner showed strict-metric gains, the rubric-free gated "
     "rewrite with that refiner and a second MPIR seed were added (Section 5.6.1)."),
    ("9. 2026-10-21, exploratory.", "After an audit found that V1 and V2 candidates are scored without the "
     "answer-format line, the format-protected variants V1P and V2P were added, and the lenient parser was "
     "corrected to accept everything the strict parser accepts (Section 5.2)."),
])
'''
open(p, "w", encoding="utf-8").write(s)

# cross-reference updates elsewhere
for fname in ("part3_results.py", "part4_discussion_back.py", "part2_method_design.py", "part1_intro_related.py"):
    t = open(fname, encoding="utf-8").read()
    t = t.replace("Tables E1 and E2", "Tables 15 and 16").replace("Table E1", "Table 15").replace("Table E2", "Table 16")
    t = t.replace("Table C1", "Table 12").replace("Table C2", "Table 13").replace("Table D1", "Table 14")
    open(fname, "w", encoding="utf-8").write(t)
print("patched appendix")
