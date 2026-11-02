# -*- coding: utf-8 -*-
"""Build the second-resubmission upload package into submission_revision2/upload_ready/.

Run after: mpir_study.revision2, revision2_new, mid_refiner; build_latex_ieee.py; build_response_round2.py.
Steps: self-contained LaTeX package (test-compiled), clean PDF, highlighted PDF against the PDF the reviewers saw,
cover letter, response letter, supplementary archive.
"""
import os
import re
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from study_numbers import N  # noqa: E402
from rev2_numbers import OFF, NEW, LEN, MID, PP  # noqa: E402

OUT = os.path.join(ROOT, "submission_revision2")
PKG = os.path.join(OUT, "_main_pkg")
UP = os.path.join(OUT, "upload_ready")
LATEX = os.path.join(HERE, "latex_ieee")
OLD_PDF = os.path.join(ROOT, "submission_revision", "upload_ready", "1b_Main_Manuscript.pdf")
KEEP_EXT = (".png", ".bib", ".cls", ".bst", ".sty", ".pfb", ".tfm", ".map", ".bbl", ".tex")


def safe_copy(src, dst, tries=8):
    """shutil.copy with retries: Windows occasionally refuses to open a freshly written file (virus scanner, indexer)."""
    import time
    for attempt in range(tries):
        try:
            if os.path.exists(dst):
                os.remove(dst)
            shutil.copy(src, dst)
            return dst
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(3)


def run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_pkg():
    shutil.rmtree(PKG, ignore_errors=True)
    os.makedirs(os.path.join(PKG, "figures"))
    for f in os.listdir(LATEX):
        if f.endswith(KEEP_EXT) and f not in ("cover_letter.tex",) and not f.startswith("cover_letter"):
            shutil.copy(os.path.join(LATEX, f), PKG)
    shutil.copy(os.path.join(HERE, "figures", "figure_3.png"), os.path.join(PKG, "figures"))
    tex = open(os.path.join(PKG, "manuscript.tex"), encoding="utf-8").read().replace("../figures/", "figures/")
    open(os.path.join(PKG, "manuscript.tex"), "w", encoding="utf-8").write(tex)
    for _ in range(2):
        run(["pdflatex", "-interaction=nonstopmode", "manuscript.tex"], PKG)
    run(["bibtex", "manuscript"], PKG)
    for _ in range(2):
        run(["pdflatex", "-interaction=nonstopmode", "manuscript.tex"], PKG)


def conflict_of_interest():
    tex = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{times}
\usepackage{parskip}
\setlength{\parindent}{0pt}
\begin{document}
\section*{Conflict of Interest Statement}
\textbf{Manuscript:} Does Rubric-Guided Self-Refinement Improve Prompts from Small-Model Optimizers? A Scoped
Negative-Result Study with Qwen3-1.7B (revised submission; Access-2026-54235, initial submission Access-2026-44920)

None of the authors (Linh Nguyen, Quang-Vinh Dang, Minh Ngoc Dinh and Thuy Nguyen) has a conflict of interest to
disclose. The same statement appears in the manuscript under ``Declaration of Competing Interest''.
\end{document}
"""
    path = os.path.join(OUT, "conflict_of_interest.tex")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(tex)
    run(["pdflatex", "-interaction=nonstopmode", "conflict_of_interest.tex"], OUT)
    return os.path.join(OUT, "conflict_of_interest.pdf")


def cover_letter():
    G, CT, AB = NEW["grid_v2f"], NEW["controls"], OFF["acceptance_bounds"]
    LC = LEN["all"]
    mid = MID.get("v2", {})
    text = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{times}
\usepackage{url}
\usepackage{parskip}
\setlength{\parindent}{0pt}
\begin{document}
\begin{flushright}
November 3, 2026
\end{flushright}

\textbf{Manuscript ID:} Access-2026-54235 (initial submission: Access-2026-44920)\\
\textbf{Revised title:} Does Rubric-Guided Self-Refinement Improve Prompts from Small-Model Optimizers? A Scoped
Negative-Result Study with Qwen3-1.7B\\
\textbf{Manuscript type:} Negative Result\\
\textbf{Corresponding author:} Quang-Vinh Dang (British University Vietnam), \url{vinh.dq4@buv.edu.vn}

\bigskip
Dear Editor,

Thank you for the opportunity to resubmit. The reviews of Access-2026-54235 identified one central problem: the primary
treatment dropped the answer format in almost all candidates, so the null result could not show that rubric-guided
refinement fails. The revised manuscript addresses it with new experiments and a narrower scope.

\medskip
What changed. (i) A positive control: the validation gate accepted a known-good candidate in %(gate)d of %(ncells)d cells in which it helps.
(ii) A re-run of the main grid with a format- and example-preserving treatment on %(runs)d cells: candidates were accepted in
%(acc)d runs, but under a lenient metric the final prompts won in %(lw)d cell%(lws)s and lost significantly in %(ll)d.
(iii) A validation of the lenient parser on 400 labelled responses (precision %(prec).1f\%%, recall %(rec).1f\%%); the labels
and the audit of 80 sampled echo and fallback rounds were produced by a language-model assistant, not by human raters, and
the manuscript says so. (iv) A rubric-free control under the same protection: a plain rewrite request gained %(v3m).1f points on average under lenient scoring (%(v3w)d winning cells of %(v3n)d), while the rubric-guided variant did not, and strong refiners (Qwen3-235B, GPT-6 Luna) improved the prompts under the protected treatment (%(q).1f and %(l).1f points, nine cells, one run each), but a rubric-free rewrite reached similar gains; so the rubric is not shown to help. (v) Exact acceptance bounds (%(ub).1f\%% one-sided for 1 acceptance in 80 runs; the earlier 7\%% was
the two-sided value), a paired comparison of the rubric-guided and rubric-free variants, a single-session rescoring of the
exploratory prompts, an intermediate open-weight refiner (Qwen3-8B, %(midacc)s of 9 runs changed the prompt), cost per
variant, and a list of all base prompts. (vi) The title and claims now state the scope (Qwen3-1.7B, nine tasks, weak base
prompts), and ``pre-specified'' is replaced by ``protocol fixed in advance, with listed amendments''.

\medskip
What we could not do. We have no second target family and no human ratings, and the strong-refiner comparisons have one run per condition. The paper lists these limits.

\medskip
This manuscript is not under review elsewhere, and all authors have approved this submission. The author list is unchanged,
so no byline-change form is enclosed. All code, per-round logs, samples and labels are at
\url{https://github.com/millennials92/meta_prompted_instruction_refinement}.

\bigskip
On behalf of all authors,\\
Quang-Vinh Dang
\end{document}
""" % {"gate": CT["known_good_accepted"], "ncells": CT["known_good_cells"], "runs": G["runs"], "acc": G["accepted_runs"],
       "lw": G["lenient_wins_vs_start"], "lws": "" if G["lenient_wins_vs_start"] == 1 else "s", "ll": G["lenient_losses_vs_start"], "prec": 100 * LC["lenient"]["precision"],
       "rec": 100 * LC["lenient"]["recall"], "ub": 100 * AB["run_upper_one_sided_95"],
       "midacc": mid.get("accepted_runs", "n/a"), "v3m": 100 * PP["v3f_1.7b"]["lenient_mean"], "v3w": PP["v3f_1.7b"]["lenient_wins"], "v3n": PP["v3f_1.7b"]["runs"], "q": 100 * PP["v2f_qwen235"]["lenient_mean"], "l": 100 * PP["v2f_gpt6luna"]["lenient_mean"]}
    path = os.path.join(OUT, "cover_letter.tex")
    open(path, "w", encoding="utf-8").write(text)
    run(["pdflatex", "-interaction=nonstopmode", "cover_letter.tex"], OUT)
    return os.path.join(OUT, "cover_letter.pdf")


def main():
    os.makedirs(UP, exist_ok=True)
    build_pkg()
    clean = os.path.join(PKG, "manuscript.pdf")
    safe_copy(clean, os.path.join(UP, "1b_Main_Manuscript.pdf"))
    with zipfile.ZipFile(os.path.join(UP, "1a_Main_Manuscript_LaTeX_source.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(PKG):
            for f in files:
                if f.endswith((".pdf", ".aux", ".log", ".out", ".blg", ".toc")):
                    continue
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, PKG))
    safe_copy(os.path.join(HERE, "response_to_reviewers_round2.docx"), os.path.join(UP, "2_Author_Response.docx"))
    safe_copy(conflict_of_interest(), os.path.join(UP, "3_Conflict_of_Interest.pdf"))
    subprocess.run([sys.executable, os.path.join(HERE, "highlight_diff.py"), OLD_PDF, clean,
                    os.path.join(UP, "4_Highlighted_Manuscript_of_Changes.pdf")], check=True)
    safe_copy(cover_letter(), os.path.join(UP, "5_Cover_Letter.pdf"))
    with zipfile.ZipFile(os.path.join(UP, "6_Supplementary_Material.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for p in ("REVISION_PROTOCOL.md", "REVISION2_STATUS.md", "REBUILD.md", "README.md"):
            fp = os.path.join(ROOT, p)
            if os.path.exists(fp):
                z.write(fp, f"supplementary/{p}")
        for d in ("mpir_study", "tests", "results/revision/rounds", "results/revision/gen_cache"):
            for root, _, files in os.walk(os.path.join(ROOT, d)):
                if "__pycache__" in root:
                    continue
                for f in files:
                    p = os.path.join(root, f)
                    z.write(p, "supplementary/" + os.path.relpath(p, ROOT).replace(os.sep, "/"))
        rev = os.path.join(ROOT, "results", "revision")
        for f in os.listdir(rev):
            p = os.path.join(rev, f)
            if os.path.isfile(p) and f.endswith(".json") and "backup" not in f:
                z.write(p, "supplementary/results/revision/" + f)
        for f in ("review_round2_factcheck.md", "review_round2_reviewer.md", "review_round2_consistency.md",
                  "reference_audit_round2.md"):
            fp = os.path.join(HERE, f)
            if os.path.exists(fp):
                z.write(fp, f"supplementary/manuscript/{f}")
    open(os.path.join(UP, "README_UPLOAD.txt"), "w", encoding="utf-8").write(
        "IEEE Access second resubmission, Access-2026-54235. Which file goes into which field:\n\n"
        "REQUIRED\n"
        "  Main Manuscript (LaTeX)  -> 1a_Main_Manuscript_LaTeX_source.zip\n"
        "  Main Manuscript (PDF)    -> 1b_Main_Manuscript.pdf\n"
        "  Author Response          -> 2_Author_Response.docx\n"
        "  Conflict of Interest     -> 3_Conflict_of_Interest.pdf\n"
        "OPTIONAL\n"
        "  Highlighted PDF          -> 4_Highlighted_Manuscript_of_Changes.pdf (new or changed words in yellow, against the PDF the reviewers saw)\n"
        "  Cover letter             -> 5_Cover_Letter.pdf\n"
        "  Supplementary material   -> 6_Supplementary_Material.zip\n"
        "The author list is unchanged, so no byline-change form is needed.\n")
    for f in sorted(os.listdir(UP)):
        print(f, round(os.path.getsize(os.path.join(UP, f)) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
