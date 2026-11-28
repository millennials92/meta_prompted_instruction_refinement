# -*- coding: utf-8 -*-
"""Builds the IEEE Access LaTeX submission from content_blocks.py, reusing the
tested block-rendering logic in build_latex.py (the Elsevier/AI Open build)
rather than duplicating it. The two builds differ only in: citation command
(\\cite numeric vs \\citep author-year), document class/front matter, and a
handful of IEEE-specific back-matter requirements (author biographies,
Acknowledgment-section AI disclosure, ORCID). Everything else -- section
numbering, figure/table/code/algorithm rendering, the appendix-figure
double-numbering fix -- is imported and reused unmodified.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LATEX_IEEE_DIR = os.path.join(HERE, "latex_ieee")
sys.path.insert(0, HERE)

import build_latex as BL
import content_blocks as C

# Reuse the already-loaded reference_N -> bibkey mapping (venue-independent);
# only LATEX_DIR needs to point at latex_ieee/ so figure paths in latex_body()
# are computed relative to *this* build's output location.
BL.LATEX_DIR = LATEX_IEEE_DIR


def resolve_citations_ieee(text):
    """IEEEtran.bst is a numeric style: \\cite{key1,key2} renders as "[3], [7]",
    not the author-year \\citep{} the Elsevier build uses."""
    def _sub(m):
        keys = [BL.REFKEY_TO_BIBKEY[f"reference_{n.strip()}"] for n in m.group(1).split(",")]
        return r"\cite{" + ",".join(keys) + "}"
    return BL.CITE_RE.sub(_sub, BL.esc(text))


# Monkey-patch: latex_body/latex_algo/latex_table/caption_command/plain_caption
# all look up the module-global name `resolve_citations` at call time, so
# redirecting it here redirects every call inside build_latex.py too, without
# forking (and risking drift in) ~250 lines of already-verified rendering code.
BL.resolve_citations = resolve_citations_ieee

# H1 blocks handled specially for IEEE's required back-matter shape (see
# ieee_backmatter() below) rather than passed through generically.
_SKIP_H1_TITLES = {
    "Acknowledgement",
    "Declaration of Generative AI and AI-Assisted Technologies in the Writing Process",
}


def filtered_body_blocks():
    """The main-text BLOCKS list with the two backmatter sections IEEE wants
    consolidated (see ieee_acknowledgment()) removed from the generic pass,
    plus their immediately-following P block (each is a single-paragraph
    H1+P pair in content_blocks.py)."""
    out = []
    skip_next_p = False
    for b in C.BLOCKS:
        if skip_next_p and b["type"] == "p":
            skip_next_p = False
            continue
        skip_next_p = False
        if b["type"] == "h1" and b["text"] in _SKIP_H1_TITLES:
            skip_next_p = True
            continue
        out.append(b)
    return out


def latex_body_ieee():
    original_blocks = C.BLOCKS
    C.BLOCKS = filtered_body_blocks()
    try:
        return BL.latex_body()
    finally:
        C.BLOCKS = original_blocks


def ieee_acknowledgment():
    """IEEE Access requires (a) funding/thanks in an unnumbered "Acknowledgment"
    section (American spelling, no "e") and (b) AI-generated-text use disclosed
    in that same section, with a citation to the AI system used -- per the
    user's explicit choice, worded generically here rather than naming a
    specific product, consistent with the project's standing no-AI-identity
    rule."""
    return (
        r"\section*{Acknowledgment}" + "\n"
        + BL.resolve_citations(
            "The authors thank the reviewers for their constructive comments on earlier versions of this manuscript. "
            "Generative AI tools were used to improve the grammar and readability of the manuscript text, and an "
            "AI-assisted coding tool (Claude Code, Anthropic) generated initial code snippets during development. Language-model assistants (Claude, Anthropic) "
            "also labelled sampled echo and fallback rounds and responses for the lenient-parser validation, and helped "
            "verify bibliographic entries; these labels are screening aids released for re-checking, not human "
            "annotations. All AI-assisted text and code were reviewed, verified, and adapted by the authors, who take "
            "full responsibility for the manuscript's content, analyses, and conclusions."
        )
    )


def author_bio_placeholders():
    """IEEE Access requires a short biography (photo optional) for every
    author, placed after the references. Role/institution supplied directly
    by the corresponding author; kept to what was actually provided rather
    than expanded with invented degree history, dates, or research-interest
    claims."""
    bios = {
        "Linh Nguyen": "is a Ph.D. student at RMIT University Vietnam, Ho Chi Minh City, Vietnam. "
                       "This work began while Linh Nguyen was a Master's student at the same university.",
        "Quang-Vinh Dang": "is a Senior Lecturer at British University Vietnam, Hung Yen, Vietnam.",
        "Minh Ngoc Dinh": "is Dean of the School of Information Technology and Data Science, "
                          "Millenia Education, Ho Chi Minh City, Vietnam.",
        "Thuy Nguyen": "is a Senior Lecturer at RMIT University Vietnam, Ho Chi Minh City, Vietnam.",
    }

    lines = []
    for a in C.AUTHORS:
        bio = bios.get(a["name"], "")
        lines.append(f"\\begin{{IEEEbiographynophoto}}{{{BL.esc(a['name'])}}}")
        lines.append(BL.esc(bio))
        lines.append(r"\end{IEEEbiographynophoto}")
        lines.append("")
    return "\n".join(lines)


def author_block_ieee():
    lines = []
    author_strs = []
    for i, a in enumerate(C.AUTHORS, start=1):
        marks = "".join(f"\\authorrefmark{{{j}}}" for j in a["affil_idx"])
        author_strs.append(f"\\uppercase{{{BL.esc(a['name'])}}}{marks}")
    # IEEE Access byline convention: "A, B, AND C" (last joined with "AND").
    if len(author_strs) > 1:
        author_line = ", ".join(author_strs[:-1]) + ", AND " + author_strs[-1]
    else:
        author_line = author_strs[0]
    lines.append(f"\\author{{{author_line}}}")
    lines.append("")
    for i, aff in enumerate(C.AFFILIATIONS, start=1):
        lines.append(f"\\address[{i}]{{{BL.esc(aff)}}}")
    lines.append("")
    corresp_name = next(a["name"] for a in C.AUTHORS if a.get("corresponding"))
    lines.append(
        f"\\corresp{{Corresponding author: {BL.esc(corresp_name)} "
        f"(e-mail: {C.CORRESPONDING_AUTHOR_EMAIL})}}"
    )
    return "\n".join(lines)


# The accepted abstract has one source: part_abstract.py via C.ABSTRACT.


LUNA_FOOTNOTE = (
    r"\footnote{GPT-6 Luna (OpenAI; model slug openai/gpt-6-luna) is documented only through its OpenRouter "
    r"model listing, \url{https://openrouter.ai/openai/gpt-6-luna}, accessed on Oct. 20 and Nov. 2, 2026. No technical "
    r"report or snapshot identifier was available to us and the router chooses the serving provider, so exact "
    r"re-runs may not be possible.}"
)


def build():
    os.makedirs(LATEX_IEEE_DIR, exist_ok=True)
    body = latex_body_ieee().replace("@@LUNAFN@@", LUNA_FOOTNOTE)
    authors = author_block_ieee()
    keywords = ", ".join(sorted(k.strip() for k in C.KEYWORDS.split(",")))
    acknowledgment = ieee_acknowledgment()
    bios = author_bio_placeholders()

    doc = TEMPLATE.format(
        title=BL.esc(C.TITLE),
        shorttitle="Does Rubric-Guided Refinement Improve Optimized Prompts?",
        authors=authors,
        abstract=BL.resolve_citations(C.ABSTRACT),
        keywords=keywords,
        body=body,
        acknowledgment=acknowledgment,
        bios=bios,
    )
    out_path = os.path.join(LATEX_IEEE_DIR, "manuscript.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    print("Wrote", out_path)


TEMPLATE = r"""\documentclass{{ieeeaccess}}
\usepackage{{cite}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{algorithm}}
\usepackage{{algpseudocode}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{textcomp}}
\usepackage{{caption}}
\usepackage{{array}}
\usepackage{{xurl}}

\begin{{document}}
\renewcommand{{\thesection}}{{\arabic{{section}}}}
\renewcommand{{\thesubsection}}{{\thesection.\arabic{{subsection}}}}
\renewcommand{{\thesubsubsection}}{{\thesubsection.\arabic{{subsubsection}}}}
\def\thesectiondis{{\thesection.}}
\def\thesubsectiondis{{\thesection.\arabic{{subsection}}}}
\def\thesubsubsectiondis{{\thesection.\arabic{{subsection}}.\arabic{{subsubsection}}}}

\vol{{14}}
\year{{2026}}
\history{{Manuscript received September 5, 2026; revised November 3, 2026; accepted November 22, 2026.}}
\doi{{10.1109/ACCESS.2026.3757210}}

\title{{{title}}}

{authors}

\tfootnote{{This research received no specific grant from any funding agency in the public,
commercial, or not-for-profit sectors.}}

\markboth{{Nguyen \headeretal: {shorttitle}}}
{{Nguyen \headeretal: {shorttitle}}}

\begin{{abstract}}
{abstract}
\end{{abstract}}

\begin{{keywords}}
{keywords}
\end{{keywords}}

\titlepgskip=-21pt

\maketitle

{body}

{acknowledgment}

\bibliographystyle{{IEEEtran}}
\bibliography{{cas-refs}}

{bios}

\vfill

\EOD

\end{{document}}
"""

if __name__ == "__main__":
    build()
