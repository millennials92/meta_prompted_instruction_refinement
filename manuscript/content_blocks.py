# -*- coding: utf-8 -*-
# Manuscript content for the IEEE Access revision (Access-2026-44920). The first submission's text is
# preserved in archive/development/manuscript/content_blocks_v1_submitted.py. Block-building helpers live in blocks_api.py; the body is
# split into part modules that append to the same BLOCKS list, in reading order.
# Citation tokens use unicode guillemets, e.g. ‹9› or ‹9,10›, naming the numeric suffix of the
# reference_N keys in manuscript/latex/refkey_to_bibkey.json; build_latex_ieee.py turns them into \cite{}.

from blocks_api import BLOCKS  # noqa: F401  (re-exported for the build scripts)

TITLE = ("Does Rubric-Guided Refinement Improve Prompts from Small-Model Optimizers?")

AUTHORS = [
    {"name": "Linh Nguyen", "affil_idx": [1], "corresponding": False, "email": "s4021009@rmit.edu.vn"},
    {"name": "Quang-Vinh Dang", "affil_idx": [2], "corresponding": False},
    {"name": "Minh Ngoc Dinh", "affil_idx": [3], "corresponding": True, "email": "Minh.dinh@maeducation.com"},
    {"name": "Thuy Nguyen", "affil_idx": [1], "corresponding": False, "email": "thuy.nguyen43@rmit.edu.vn"},
]
AFFILIATIONS = [
    "School of Science, Engineering & Technology, RMIT University Vietnam, 702 Nguyen Van Linh Boulevard, Ho Chi Minh City 700000, Vietnam",
    "British University Vietnam, Hung Yen, Vietnam",
    "Millenia Education, Ho Chi Minh City, Vietnam",
]
CORRESPONDING_AUTHOR_EMAIL = "Minh.dinh@maeducation.com"
CORRESPONDING_AUTHOR_NOTE = (
    "*Corresponding author: Minh Ngoc Dinh, Millenia Education, Ho Chi Minh City, Vietnam. "
    f"Email: {CORRESPONDING_AUTHOR_EMAIL}"
)

KEYWORDS = ("Automatic prompt optimization, Big-Bench Hard, Large language models, Meta-prompting, "
            "Negative results, Prompt engineering")

from study_numbers import N  # noqa: E402,F401  (numbers computed by mpir_study/analyze.py)

# The abstract is defined in part modules' shared namespace once the numbers are loaded.
import part_abstract  # noqa: E402,F401
ABSTRACT = part_abstract.ABSTRACT

import part1_intro_related  # noqa: E402,F401
import part2_method_design  # noqa: E402,F401
import part3_results  # noqa: E402,F401
import part4_discussion_back  # noqa: E402,F401
import part5_appendix  # noqa: E402,F401
import part5b_appendix_g  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Final table numbering (accepted version): the tables added in the revision rounds carry hand-made
# labels R.1-R.6 and are interleaved with auto-numbered Tables 1-17. Renumber every table in reading
# order and rewrite all in-text references so the labels are consecutive.
import re as _re  # noqa: E402


def _renumber_tables(blocks):
    cap_re = _re.compile(r"^Table ((?:R\.)?\d+)\.")
    mapping, seq = {}, 0
    for b in blocks:
        if b["type"] == "table":
            m = cap_re.match(b["caption"])
            if m:
                seq += 1
                mapping[m.group(1)] = str(seq)
    single = _re.compile(r"Table ((?:R\.)?\d+)(?![\d.]*\d)")
    pair = _re.compile(r"Tables (\d+) and (\d+)")

    def fix(s):
        s = pair.sub(lambda m: f"Tables {mapping.get(m.group(1), m.group(1))} and {mapping.get(m.group(2), m.group(2))}", s)
        return single.sub(lambda m: f"Table {mapping.get(m.group(1), m.group(1))}", s)

    def walk(x):
        if isinstance(x, str):
            return fix(x)
        if isinstance(x, (list, tuple)):
            return type(x)(walk(i) for i in x)
        if isinstance(x, dict):
            return {k: (walk(v) if k != "type" else v) for k, v in x.items()}
        return x

    blocks[:] = [walk(b) for b in blocks]
    return mapping


TABLE_MAP = _renumber_tables(BLOCKS)
ABSTRACT = _re.sub(r"Table ((?:R\.)?\d+)", lambda m: f"Table {TABLE_MAP.get(m.group(1), m.group(1))}", ABSTRACT)
