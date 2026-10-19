"""Prompt templates: per-technique evaluation wrappers (read from the repo's own yaml), rubric text
(original and corrected), refinement requests, and the rubric-free rewrite prompt."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
TECHNIQUE_DIR = REPO / "promptwizard" / "glue" / "promptopt" / "techniques"
TECHNIQUE_FOR = {"promptwizard": "critique_n_refine", "ape": "ape", "protegi": "protegi"}

TEXT_DELIMITER = re.compile(r"(?s)(?<=<START>)(.*?)(?=<END>)")
# Verbatim phrases from the rubric template that a genuine task prompt has no reason to contain.
ECHO_MARKERS = ("Senior Prompt Engineer", "Prompt Under Evaluation", "Seven-Criteria Evaluation Rubric",
                "Seven-Criteria", "I gave a friend an instruction")
ANSWER_FORMAT = "Wrap the final answer between <ANS_START> and <ANS_END>"


@lru_cache(maxsize=None)
def pool(technique_dir: str) -> dict:
    with open(TECHNIQUE_DIR / technique_dir / "prompt_pool.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def eval_template(optimizer: str) -> tuple[str, str]:
    """(eval_prompt, system_prompt) exactly as the base optimizer's own final evaluation uses them."""
    data = pool(TECHNIQUE_FOR[optimizer])
    return data["eval_prompt"], data["system_prompt"]


def final_prompt_template() -> str:
    return pool("heuristic")["final_prompt"]


ORIGINAL_RUBRIC = pool("heuristic")["prompt_evaluation"]

# The original template announces "five defined criteria", "all 6 criteria" and "Total Score: X/25"
# while listing seven criteria; the corrected text is otherwise identical.
CORRECTED_RUBRIC = (ORIGINAL_RUBRIC
                    .replace("against the five defined criteria", "against the seven defined criteria")
                    .replace("(continue for all 6 criteria)", "(continue for all 7 criteria)")
                    .replace("**Total Score: X/25**", "**Total Score: X/35**"))
assert CORRECTED_RUBRIC != ORIGINAL_RUBRIC

REFINEMENT_REQUEST = pool("heuristic")["prompt_refinement"]
IMPROVED_PROMPT = pool("heuristic")["improved_prompt"]

HISTORY_REQUEST = """Refine the prompt by applying all suggestions from the evaluation.
Earlier refinement attempts for this task were tried and did NOT beat the current best prompt on the
validation questions; do not repeat them, change the approach:

{history}

Make sure to wrap the refined prompt with <START> and <END>"""

FREE_REWRITE_PROMPT = """You are given a task-solving prompt generated for a Big-Bench Hard
task.

Rewrite the prompt to improve clarity, readability,
and instruction organization while preserving the
original task meaning.

Preserve the original task, answer format, and examples.
Do not change the meaning of the task.
Do not add unrelated content.
Do not use any prompt-evaluation rubric.
Do not mention rubric criteria, scoring, strengths,
weaknesses, or feedback.

Output only the rewritten task-solving prompt.

{variant_instruction}

Original prompt:
<START>
{prompt}
<END>"""

_TOTAL = re.compile(r"Total\s*Score\s*:?\**\s*\**\s*(\d+(?:\.\d+)?)\s*/\s*(\d+)", re.I)
_CRITERION = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*5\b")


def parse_rubric_total(text: str) -> tuple[float | None, int | None]:
    """(score, out_of) from the judge's 'Total Score: X/Y' line; falls back to the sum of per-criterion X/5."""
    match = _TOTAL.search(text)
    if match:
        return float(match.group(1)), int(match.group(2))
    parts = [float(x) for x in _CRITERION.findall(text)]
    return (sum(parts), 5 * len(parts)) if len(parts) >= 3 else (None, None)
