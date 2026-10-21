# -*- coding: utf-8 -*-
import glob
import json
import os

from blocks_api import (APPENDIX_START, H1, H2, H3, P, BULLETS, TABLE, CODE)  # noqa: F401
from study_numbers import N  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.join(HERE, "..", "results", "revision")

APPENDIX_START()

# ===========================================================================
H1("Appendix A. Prompt Examples")
H2("A.1. Base Prompts Produced by a 1.7B-Driven Optimizer")
P("These are prompts exactly as saved by the pipeline (seed 42), not cleaned. They show the quality of the "
  "upstream, 1.7B-driven baselines that MPIR was asked to refine, which limits how informative refining "
  "them can be. PromptWizard's prompt for penguins_in_a_table opens with a detached question that is "
  "not a task instruction; the APE prompt for hyperbaton is APE's own instruction-induction "
  "meta-prompt, not a task instruction, because the search selected a candidate that echoed the induction "
  "template; the APE and ProTeGi prompts for the other two tasks are the seed instruction \"Let's think "
  "step by step.\" plus the answer-format line.")
CODE([
    "How could I determine the accuracy of my solution to this problem?",
    "",
    "[Question] Here is a table where the first line is a header and",
    "each subsequent line is a penguin: name, age, height (cm),",
    "weight (kg) Louis, 7, 50, 11 Bernard, 5, 80, 13 Vincent, 9, 60, 11 ...",
    "What is the name of the 60 cm tall penguin?",
    "Options: (A) Louis (B) Bernard (C) Vincent (D) Gwen (E) James",
    "[Answer] ... Therefore, the name of the 60 cm tall penguin is Vincent.",
    "**Final Answer:** (C) Vincent <ANS_START>(C)<ANS_END>",
    "[... two further worked examples omitted ...]",
    "Wrap the final answer between <ANS_START> and <ANS_END>",
], caption="Figure A.1. PromptWizard's saved prompt for penguins_in_a_table (abridged).", full=False)
CODE([
    "I gave a friend an instruction. Based on the instruction they",
    "produced the following input-output pairs:",
    "[Question] Which sentence has the correct adjective order: ...",
    "[Answer] (B)",
    "[... further input-output pairs omitted ...]",
    "What was the instruction I gave?",
    "Wrap the final answer between <ANS_START> and <ANS_END>",
], caption="Figure A.2. APE's saved prompt for hyperbaton (abridged): an unextracted induction "
           "meta-prompt, not a task instruction.", full=False)

H2("A.2. Reference Prompts")
P("The zero-shot condition uses the BBH authors' task description plus \"Let's think step by step\" and "
  "the answer-format line. The expert few-shot condition appends the authors' three worked "
  "chain-of-thought exemplars ‹27›, each followed by a bare <ANS_START>(X)<ANS_END> tag. This tag "
  "is the property that exemplar-free prompts lack (Section 5.1).")
CODE([
    "Q: Which of the following is a humorous edit of this artist or movie name: 'whitesnake'?",
    "Options: (A) whitesnape (B) whitesnapke (C) whitesnuake (D) mwhitesnake",
    "A: Let's think step by step. ... So the answer is (A).",
    "<ANS_START>(A)<ANS_END>",
], caption="Figure A.3. First expert exemplar for ruin_names, abridged. The same exemplar is the single "
           "format exemplar used in the intervention of Section 5.1.", full=False)

H2("A.3. Free-Rewrite Control Prompt")
CODE([
    "You are given a task-solving prompt generated for a Big-Bench Hard task.",
    "Rewrite the prompt to improve clarity, readability, and instruction organization while",
    "preserving the original task meaning. Preserve the original task, answer format, and examples.",
    "Do not use any prompt-evaluation rubric. Output only the rewritten task-solving prompt.",
    "Original prompt: <START> {prompt} <END>",
], caption="Figure A.4. Meta-prompt of the free-rewrite control (abridged).", full=False)

# ===========================================================================
H1("Appendix B. Echo Failure Transcripts")
P("Two ways in which a small refiner failed silently, found by inspecting saved outputs after an "
  "anomalous accuracy reading in the pilot. Both are well-formed <START>/<END> responses with no usable "
  "task instruction. They are now caught by the echo guard (Section 3.4) and, independently, by the "
  "validation gate.")
CODE([
    "<START>",
    "I gave a friend an instruction. Based on the instruction they produced the following",
    "input-output pairs: [Question] Which sentence has the correct adjective order: ...",
    "The task is to determine the correct adjective order in a sentence ...",
    "<END>",
], caption="Figure B.1. hyperbaton / APE: the refiner echoed APE's induction template; no answer-format "
           "line, so validation accuracy fell to zero (abridged).", full=False)
CODE([
    "<START>",
    "### Task Overview  You are a Senior Prompt Engineer ...  ### Prompt Under Evaluation ...",
    "### Seven-Criteria Evaluation Rubric ...",
    "<END>",
], caption="Figure B.2. ruin_names / PromptWizard: the refiner echoed the evaluation meta-prompt "
           "itself; the round scored 0.0 (abridged).", full=False)

CODE([
    "P0 (ProTeGi, hyperbaton):  Let's think step by step.  Wrap the final answer between <ANS_START> and <ANS_END>",
    "Accepted candidate (Qwen3-235B refiner, round 7; validation 0.52 vs 0.16, test 39.5% vs 10.5%):",
    "  You are a Senior Reasoning Assistant ... ### Instructions  1. Step Back ... 2. Reason Step-by-Step ...",
    "  5. Follow Format Exactly ...  7. Final Answer: On a new line, wrap only the final answer with",
    "  <ANS_START> and <ANS_END>.   ### Output Format Examples  [Question] A bat and a ball cost $1.10 ... [Answer] ...",
], caption="Figure B.3. An accepted candidate from the strong refiner (abridged). It adds a role, step-back "
           "and numbered steps, and also an explicit instruction to wrap the final answer on its own line and a "
           "worked output example, which repairs the answer tag; the example is generic, not about adjective order. "
           f"The strict accuracy of this prompt is higher than that of P0 (39.5% against 10.5%) but its lenient "
           f"accuracy is lower ({100 * N['winner_cells']['qwen235|hyperbaton__protegi__s42']['final_lenient']:.0f}% "
           f"against {100 * N['winner_cells']['qwen235|hyperbaton__protegi__s42']['p0_lenient']:.0f}%, Section 5.6).", full=False)

# ===========================================================================
H1("Appendix C. Pipeline Defects and Serving Configuration")
P("Defects found while running the pipeline against a live model. Magnitudes are the symptoms recorded "
  "at the time. Items 1-8 were fixed before any number reported in Section 5; items 9-13 were found "
  "during this revision, items 12 and 13 late, after the main grid had run.")
TABLE("Table 12. Defects found in the pipeline, their observed symptom and the fix.",
      header=["#", "Defect", "Observed symptom", "Fix"],
      rows=[
          ["1", "Response truncated by an under-provisioned context", "PromptWizard prompts left too little "
           "context: responses cut mid-sentence, 0/50 on a task", "Explicit max new tokens (1024) and context 8192"],
          ["2", "Answer normalization gap", "Bare letter 'A' scored wrong against gold '(A)'",
           "Strip parentheses and case on both sides"],
          ["3", "Delimiter convention absent from expert exemplars", "Expert few-shot scored exactly 0.0 "
           "on hyperbaton (whole response wrapped)", "Append a bare tag after each exemplar's conclusion"],
          ["4", "Missing delimiters crashed the loop", "IndexError ended every remaining condition of a task",
           "Fallback to the unrefined prompt (now logged as an outcome)"],
          ["5", "Echo of the rubric meta-prompt", "ruin_names / PromptWizard candidate scored 0.0",
           "Marker-phrase echo guard"],
          ["6", "Echo of APE's induction template; unscored baseline", "hyperbaton / APE candidate scored 0.0 "
           "and was accepted because the incumbent score started at minus infinity",
           "Score P0 first; accept only on strict improvement"],
          ["7", "Evaluation template mismatch", "Same prompt text scored 0.07 through one template and 0.515 "
           "through another (trailing whitespace changes greedy decoding)",
           "Score every condition through the base optimizer's own template"],
          ["8", "Search-time scoring omitted the answer format", "APE/ProTeGi candidates scored near 0 during "
           "search, so selection ran on noise", "Bake the answer-format line into scored instructions"],
          ["9", "Rubric text inconsistent", "'five criteria', 'all 6 criteria', 'X/25' for a seven-criterion, "
           "35-point rubric", "Corrected text for V1-V3; original kept as V2L"],
          ["10", "Refinement rounds not independent", "Same input, temperature 0, seven times",
           "Independent sampling and chained variants (Section 3.2)"],
          ["11", "APE search selected a degenerate prompt", "hyperbaton P0 is APE's own induction template",
           "Left in place and disclosed; same code as the pilot"],
          ["12", "Candidates of V1 and V2 scored without the answer-format line", "The refinement request does not ask "
           "to keep it; 92-95% of validated 1.7B candidates lack it and score exactly zero under strict extraction",
           "Format-protected variants V1P and V2P (Section 3.2); asymmetry reported"],
          ["13", "Lenient parser rejected a bare letter inside the tags", "Strict accepted what lenient rejected",
           "Lenient now accepts everything strict accepts"],
      ], full=True, colw=[0.04, 0.26, 0.40, 0.30])
P("Earlier pre-pilot fixes (a per-seed data-split collision, an unseeded random generator, mismatched "
  "scoring minibatches in APE, and swallowed API errors) are recorded in the repository's REBUILD.md.")

TABLE("Table 13. Serving configuration.",
      header=["Item", "Value"],
      rows=[
          ["Inference engine", "vLLM 0.28.0 (V1 engine, model runner V2 disabled, flashinfer sampler disabled), PyTorch 2.13.0+cu132"],
          ["Target and default refiner", "Qwen/Qwen3-1.7B, revision 70d244c, bf16, no quantization"],
          ["Capability-experiment refiners", "qwen/qwen3-235b-a22b-2507 and openai/gpt-6-luna (reasoning effort minimal) "
           "through the OpenRouter API, 2026-10-20 and 21, temperature 0.7, 3000 new tokens; no local server"],
          ["Context and limits", "max model length 8192; max new tokens 1024 (target), 1500 / 3000 (meta calls); thinking disabled"],
          ["Decoding", "target: temperature 0; refiner: temperature 0.7 (0 for V0); seeds recorded per call"],
          ["Server flags", "--gpu-memory-utilization 0.80-0.86 --max-num-seqs 64-128 --seed 0; the server was restarted "
           "several times and its memory settings changed (fp8 KV cache was tried and not usable without nvcc; "
           "prefix caching was off for one period); cached scores and generations were kept across restarts"],
          ["Hardware", "NVIDIA RTX 5000 Ada Generation Laptop GPU, 16 GB, driver 596.58; WSL2 (Ubuntu 24.04, kernel 5.15.167.4) on Windows 11"],
          ["Non-determinism", "Continuous batching; identical prompts are re-scored to measure it (Section 5.7)"],
      ], full=True, colw=[0.26, 0.74])

# ===========================================================================
H1("Appendix D. The Original Evaluation (Exploratory; Validation Leak)")
P("The first evaluation used GPT-3.5-turbo as target and GPT-4o as refiner on all BBH tasks. It is "
  "reproduced here for completeness, because the Introduction refers to it. Its validation stage scored candidates on the optimizer's own training examples and its "
  "loop never scored the unrefined prompt (Section 3.2), single runs at temperature 0. The numbers are "
  "exploratory, are not evidence for MPIR, and were not used to compute any result in Section 5. The "
  "Gemini 3.5 Flash-Lite check used the same model as target and refiner, with the same defects; its "
  "baseline already averaged 92%, leaving little headroom, and it was not analysed further.")

_orig_path = os.path.join(REV, "original_results.json")
if os.path.exists(_orig_path):
    _orig = json.load(open(_orig_path, encoding="utf-8"))
    _main, _apo, _gem = _orig["gpt35_main"]["rows"], _orig["gpt35_apo"]["rows"], _orig["gemini"]["rows"]

    def _f(x):
        try:
            return f"{float(x):.1f}"
        except ValueError:
            return x

    rows, deltas = [], []
    for m, a, g in zip(_main, _apo, _gem):
        if m[0].startswith("Average") or a[0].startswith("Average"):
            continue
        pw, mp = float(m[2]), float(m[4])
        deltas.append((m[0], mp - pw))
        rows.append([m[0], _f(m[1]), _f(m[5]), _f(m[2]), _f(m[4]), f"{mp - pw:+.1f}", _f(a[1]), _f(a[4]),
                     _f(a[2]), _f(a[5]), g[1], g[2]])
    avg_m = next(r for r in _main if r[0].startswith("Average"))
    avg_a = next(r for r in _apo if r[0].startswith("Average"))
    avg_g = next(r for r in _gem if r[0].startswith("Average"))
    rows.append(["Average", _f(avg_m[1]), _f(avg_m[5]), _f(avg_m[2]), _f(avg_m[4]),
                 f"{float(avg_m[4]) - float(avg_m[2]):+.1f}", _f(avg_a[1]), _f(avg_a[4]), _f(avg_a[2]),
                 _f(avg_a[5]), avg_g[1], avg_g[2]])
    TABLE("Table 14. Original GPT-3.5-turbo evaluation (accuracy %, single run, leaked validation split) and "
          "Gemini 3.5 Flash-Lite check. PW = PromptWizard; Diff = PW+MPIR minus PW (points). Exploratory only.",
          header=["Task", "Zero-shot", "Expert", "PW", "PW+MPIR", "Diff", "APE", "APE+MPIR", "ProTeGi",
                  "ProTeGi+MPIR", "Gemini PW", "Gemini MPIR"],
          rows=rows, full=True)
    ranked = sorted(deltas, key=lambda t: -t[1])
    rank_of = {name: i + 1 for i, (name, _) in enumerate(ranked)}
    N_ORIG = {"rank_hyperbaton": rank_of.get("hyperbaton"), "rank_penguins": rank_of.get("penguins_in_a_table"),
              "rank_ruin": rank_of.get("ruin_names"), "n_tasks": len(ranked)}
    P(f"The three pilot tasks ranked {N_ORIG['rank_hyperbaton']}, {N_ORIG['rank_penguins']} and "
      f"{N_ORIG['rank_ruin']} of {N_ORIG['n_tasks']} by the original PW+MPIR gain (hyperbaton +21.8, "
      "penguins_in_a_table +7.6, ruin_names +5.8 points). Selecting tasks on the maximum of a noisy, leaked "
      "statistic invites winner's-curse effects, which is why Tier 2 of the revised design draws tasks at random.")

# ===========================================================================
H1("Appendix E. Round-Level Record of the Primary Variant")
P("Per cell (task / optimizer, seed 42) and round, the outcome of the chained variant V2 with MPIR seed 1 "
  "(F fallback, E echo, D duplicate, V validated with its validation accuracy), then the baseline "
  "validation accuracy of P0 and the judge's rubric total out of 35 at round 1. Table 16 gives the same for "
  "the independent variant V1. Full logs for every variant, seed and round, including candidate texts and "
  "judge outputs, are in results/revision/rounds.")


def _round_rows(variant):
    rows = []
    for task in ("hyperbaton", "ruin_names", "penguins_in_a_table"):
        for opt in ("promptwizard", "ape", "protegi"):
            path = os.path.join(REV, "rounds", f"{task}__{opt}__s42__{variant}__r1.7b__m1.json")
            if not os.path.exists(path):
                continue
            run = json.load(open(path, encoding="utf-8"))
            cells = []
            for x in run["rounds"]:
                code = {"fallback": "F", "echo": "E", "duplicate": "D"}.get(x["outcome"])
                cells.append(code if code else f"V {x['val']:.2f}" + ("*" if x["accepted"] else ""))
            jt = run["rounds"][0]["judge_total"]
            rows.append([f"{task} / {opt}", f"{run['baseline_val']:.2f}"] + cells +
                        [f"{jt:.0f}" if jt is not None else "n/a"])
    return rows


for _variant, _label in (("v2", "15"), ("v1", "16")):
    _rows = _round_rows(_variant)
    if _rows:
        TABLE(f"Table {_label}. Round-level outcomes, variant {_variant.upper()}, MPIR seed 1 (an asterisk marks "
              "an accepted candidate).",
              header=["Cell", "P0 val"] + [f"R{i}" for i in range(1, 8)] + ["Judge /35"],
              rows=_rows, full=True)

# ===========================================================================
H1("Appendix F. Protocol and Deviations")
P("The decision rules of Section 4.1 were fixed in REVISION_PROTOCOL.md before any revised experiment "
  "ran (commit 8fd1397, 2026-10-19). The amendments below are recorded in that file; none is an entry in a "
  "registry. Amendments 1 to 7 were made before the results of the affected experiment were inspected; "
  "amendments 8 and 9 were made after results were seen and are exploratory; V1P and V2P were stopped before all nine cells finished.")
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
