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
            "The authors thank the reviewers for their constructive comments on earlier drafts of "
            "this manuscript. Generative AI tools were used to assist with improving the grammar "
            "and readability of the manuscript text and to generate initial code snippets during "
            "development; all AI-assisted text and code were reviewed, verified, and adapted by the "
            "authors, who take full responsibility for the manuscript's content, analyses, and "
            "conclusions."
        )
    )


def author_bio_placeholders():
    """IEEE Access requires a short biography (photo optional) for every
    author, placed after the references. Real biographical content about
    named individuals should not be fabricated -- these are explicit
    placeholders (affiliation only, drawn from the manuscript's own author
    block) for the authors to complete before submission."""
    aff_by_author = []
    for a in C.AUTHORS:
        affs = [C.AFFILIATIONS[i - 1] for i in a["affil_idx"]]
        aff_by_author.append("; ".join(affs))

    lines = []
    for a, aff in zip(C.AUTHORS, aff_by_author):
        lines.append(f"\\begin{{IEEEbiographynophoto}}{{{BL.esc(a['name'])}}}")
        lines.append(
            f"[PLACEHOLDER -- author to complete before submission] {BL.esc(a['name'])} is affiliated "
            f"with {BL.esc(aff)}. [Add: degrees held, institution/year; prior or current role; "
            f"research interests; any relevant honors or memberships.]"
        )
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


# IEEE Access requires a 150-250 word, single-paragraph, citation-free
# abstract -- the master ABSTRACT in content_blocks.py (written for the
# Elsevier/AI Open submission, no strict word cap there) runs longer, so this
# build uses its own trimmed version rather than shortening the shared master
# copy and affecting the other build.
IEEE_ABSTRACT = (
    "Large language models remain critically dependent on prompt quality, and automatic prompt "
    "optimization (APO) methods often produce output lacking the structured guidance human-crafted "
    "prompts provide. This paper reports a pre-registered go/no-go pilot testing Meta-Prompted "
    "Instruction Refinement (MPIR), a lightweight rubric-guided layer that refines an APO-generated "
    "prompt through meta-prompted evaluation, refinement, and empirical validation. An initial "
    "evaluation using GPT-4o to judge and refine prompts targeting GPT-3.5-turbo reported per-task "
    "gains on Big-Bench Hard (BBH) but a pooled improvement that did not reach significance, and a "
    "later audit found its validation stage had leaked training data into scoring. To isolate whether "
    "the apparent gains reflected the refinement mechanism or the meta-model's capability advantage "
    "over its target, we reran the pilot with a single small open-weight model (Qwen3-1.7B) as target, "
    "optimizer, and refinement judge simultaneously, across three APO backbones on the three BBH "
    "tasks where the original gains were largest. The effect did not reproduce: MPIR's refined prompt "
    "was byte-identical to its base optimizer's prompt in all nine tested configurations, and pooled "
    "McNemar and GEE tests returned accuracy differences of 0.0000. Tracing the cause, we identify and "
    "fix a general defect in the refinement algorithm -- it never scored the pre-refinement prompt as "
    "a baseline -- plus six further correctness bugs found live. We report this as a boundary-condition "
    "finding: a cheap, post-hoc rubric-refinement layer of this kind appears to depend on the refining "
    "model's own capability, and its benefit is smaller than the original per-task deltas suggest."
)


def build():
    os.makedirs(LATEX_IEEE_DIR, exist_ok=True)
    body = latex_body_ieee()
    authors = author_block_ieee()
    keywords = ", ".join(sorted(k.strip() for k in C.KEYWORDS.split(",")))
    acknowledgment = ieee_acknowledgment()
    bios = author_bio_placeholders()

    doc = TEMPLATE.format(
        title=BL.esc(C.TITLE),
        shorttitle="Meta-Prompted Instruction Refinement: A Boundary-Condition Finding",
        authors=authors,
        abstract=BL.resolve_citations(IEEE_ABSTRACT),
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
\usepackage{{url}}

\begin{{document}}

\history{{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}}
\doi{{10.1109/ACCESS.2026.0000000}}

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

\EOD

\end{{document}}
"""

if __name__ == "__main__":
    build()
