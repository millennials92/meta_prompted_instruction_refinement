"""Strict vs lenient answer extraction, recomputed offline from saved raw outputs (protocol G7)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .answers import (ANS_END, ANS_START, contains_gold, extract_final_answer, is_correct_lenient,
                      is_correct_strict)
from .templates import REPO

SPAN_LONG_CHARS = 12  # a bare option letter or short answer is below this; a sentence is above it


def audit_rows(rows: list[dict]) -> dict:
    n = len(rows)
    has_delims = sum(1 for r in rows if ANS_START in r["llm_output"] and ANS_END in r["llm_output"])
    long_span = sum(1 for r in rows if len(extract_final_answer(r["llm_output"])) > SPAN_LONG_CHARS)
    return {
        "n": n,
        "strict": sum(is_correct_strict(r["llm_output"], r["actual"]) for r in rows) / n,
        "lenient": sum(is_correct_lenient(r["llm_output"], r["actual"]) for r in rows) / n,
        "contains_gold": sum(contains_gold(r["llm_output"], r["actual"]) for r in rows) / n,
        "has_delimiters": has_delims / n,
        "long_span": long_span / n,
    }


def audit_directory(directory: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for path in sorted(directory.glob("*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if rows:
            result[path.stem] = audit_rows(rows)
    return result


if __name__ == "__main__":
    out = audit_directory(REPO / "demos" / "results" / "predictions")
    target = REPO / "results" / "revision" / "parsing_audit_pilot.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name, row in out.items():
        print(f"{name:55s} strict {row['strict']:.3f} lenient {row['lenient']:.3f} "
              f"contains {row['contains_gold']:.3f} delims {row['has_delimiters']:.2f} long {row['long_span']:.2f}")
