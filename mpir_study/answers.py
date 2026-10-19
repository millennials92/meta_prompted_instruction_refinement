"""Answer extraction and correctness, strict (as in the pilot) and lenient (secondary metric)."""
from __future__ import annotations

import re

ANS_START, ANS_END = "<ANS_START>", "<ANS_END>"
_OPTION_TOKEN = re.compile(r"\(([A-R])\)")


def extract_between(start: str, end: str, text: str) -> str:
    start_index = text.find(start)
    if start_index == -1:
        return ""
    start_index += len(start)
    end_index = text.find(end, start_index)
    if end_index == -1:
        return ""
    return text[start_index:end_index]


def extract_final_answer(output: str) -> str:
    return extract_between(ANS_START, ANS_END, output).strip()


def normalize(text: str) -> str:
    return text.strip().replace("(", "").replace(")", "").lower()


def is_correct_strict(output: str, gold: str) -> bool:
    """Exactly the pilot's rule: non-empty delimited span equal to gold after paren/case normalisation."""
    predicted = extract_final_answer(output)
    return bool(predicted) and normalize(predicted) == normalize(gold)


def contains_gold(output: str, gold: str) -> bool:
    """Upper bound: the correct option (or gold string) appears anywhere in the response."""
    gold_letter = _OPTION_TOKEN.fullmatch(gold.strip())
    if gold_letter:
        return gold.strip() in output
    return normalize(gold) != "" and normalize(gold) in output.lower()


def is_correct_lenient(output: str, gold: str) -> bool:
    """Lenient parse: the delimited span if present (else the last 300 characters) names exactly
    one option and it is the gold option; for non-lettered gold strings, the span must contain gold."""
    span = extract_final_answer(output) or output[-300:]
    gold_letter = _OPTION_TOKEN.fullmatch(gold.strip())
    if gold_letter:
        options = set(_OPTION_TOKEN.findall(span))
        return options == {gold_letter.group(1)}
    return normalize(gold) != "" and normalize(gold) in span.lower()
