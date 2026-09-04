"""BBH-author chain-of-thought prompts, shared by the manual baseline
conditions (Zero-shot CoT, Expert few-shot CoT) in demos/baselines.ipynb.

Every file under BIG-Bench-Hard/cot-prompts/<task>.txt has the same shape: a
canary-GUID line, a "-----" separator, then the BBH authors' own one-line task
description, then three worked Q/A exemplars using "Let's think step by
step." reasoning. Extracting the task description from this file (rather than
hand-authoring a second copy) keeps the zero-shot and few-shot conditions
grounded in the same source the manuscript already cites for the few-shot
exemplars (REBUILD.md §4.0/§4.1, manuscript §4.3.2).
"""
import os
import re

_HEADER_LINES = 2  # canary GUID line + "-----" separator

# The raw BBH exemplars end each worked example in plain prose ("So the
# answer is (B)."), with no delimiter tags at all. Manuscript Appendix A.3's
# preserved example shows the actual methodology appends a bare
# "<ANS_START>(B)<ANS_END>" tag right after that sentence -- reasoning stays
# unwrapped prose, only the letter goes inside the tag. Without this, a model
# told separately to "wrap the final answer" in delimiters has no
# demonstration of what "the final answer" means relative to the exemplar
# style, and reasonably wraps its *entire* response instead of just the
# letter -- confirmed live: this caused expert_few_shot_cot to score exactly
# 0.0 on hyperbaton (every extracted "answer" was a full paragraph that could
# never match a bare "(A)"/"(B)" ground truth) despite the model's own prose
# reasoning frequently reaching the correct letter.
_CONCLUDING_SENTENCE_RE = re.compile(r"(So the answer is (\([A-Za-z0-9]+\)|\w+)\.)")


def _load_file(task_name: str, cot_prompts_dir: str) -> str:
    path = os.path.join(cot_prompts_dir, f"{task_name}.txt")
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    return "".join(lines[_HEADER_LINES:]).strip()


def task_description(task_name: str, cot_prompts_dir: str = "../BIG-Bench-Hard/cot-prompts") -> str:
    """The BBH authors' own one-line task description (first line after the
    canary/separator header)."""
    return _load_file(task_name, cot_prompts_dir).splitlines()[0].strip()


def expert_few_shot_prefix(task_name: str, cot_prompts_dir: str = "../BIG-Bench-Hard/cot-prompts") -> str:
    """Full BBH-author prompt: task description plus the three worked
    chain-of-thought exemplars (manuscript §4.3.2's "same three chain-of-thought
    exemplars written by the BBH benchmark authors"), each followed by a bare
    "<ANS_START>(letter)<ANS_END>" tag matching manuscript Appendix A.3's
    demonstrated format -- so the model has an explicit example of wrapping
    only the final letter, not its whole reasoning."""
    raw_text = _load_file(task_name, cot_prompts_dir)

    def _append_tag(match: re.Match) -> str:
        sentence, answer = match.group(1), match.group(2)
        return f"{sentence}\n<ANS_START>{answer}<ANS_END>"

    tagged_text, n_substitutions = _CONCLUDING_SENTENCE_RE.subn(_append_tag, raw_text)
    assert n_substitutions == 3, (
        f"Expected 3 worked examples with a 'So the answer is (X).' conclusion in "
        f"{task_name}.txt, found {n_substitutions} -- check the file's format hasn't changed.")
    return tagged_text
