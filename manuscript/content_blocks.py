# -*- coding: utf-8 -*-
# Manuscript content for the IEEE Access revision (Access-2026-44920). The first submission's text is
# preserved in content_blocks_v1_submitted.py. Block-building helpers live in blocks_api.py; the body is
# split into part modules that append to the same BLOCKS list, in reading order.
# Citation tokens use unicode guillemets, e.g. ‹9› or ‹9,10›, naming the numeric suffix of the
# reference_N keys in manuscript/latex/refkey_to_bibkey.json; build_latex_ieee.py turns them into \cite{}.

from blocks_api import BLOCKS  # noqa: F401  (re-exported for the build scripts)

TITLE = ("Does Rubric-Guided Refinement Improve Automatically Optimized Prompts? "
         "A Pre-Specified Negative-Result Study")

AUTHORS = [
    {"name": "Linh Nguyen", "affil_idx": [1], "corresponding": False, "email": "s4021009@rmit.edu.vn"},
    {"name": "Quang-Vinh Dang", "affil_idx": [2], "corresponding": True},
    {"name": "Minh Ngoc Dinh", "affil_idx": [3], "corresponding": False, "email": "Minh.dinh@maeducation.com"},
    {"name": "Thuy Nguyen", "affil_idx": [1], "corresponding": False, "email": "thuy.nguyen43@rmit.edu.vn"},
]
AFFILIATIONS = [
    "School of Science, Engineering & Technology, RMIT University Vietnam, 702 Nguyen Van Linh Boulevard, Ho Chi Minh City 700000, Vietnam",
    "British University Vietnam, Hung Yen, Vietnam",
    "Millenia Education, Ho Chi Minh City, Vietnam",
]
CORRESPONDING_AUTHOR_EMAIL = "vinh.dq4@buv.edu.vn"
CORRESPONDING_AUTHOR_NOTE = (
    "*Corresponding author: Quang-Vinh Dang, British University Vietnam, Hung Yen, Vietnam. "
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
