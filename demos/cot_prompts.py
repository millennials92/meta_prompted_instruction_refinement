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

_HEADER_LINES = 2  # canary GUID line + "-----" separator


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
    exemplars written by the BBH benchmark authors")."""
    return _load_file(task_name, cot_prompts_dir)
