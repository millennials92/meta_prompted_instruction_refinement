"""One-format-exemplar intervention on exemplar-free APE / ProTeGi prompts (protocol G7).

P0 is extended with the first worked example of the BBH authors' expert prompt (already carrying the bare
<ANS_START>(X)<ANS_END> tag); nothing else changes. The result is scored like any other prompt.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from .data import Cell, load_p0, tier1_cells, tier2_cells
from .scoring import Scorer
from .templates import REPO

sys.path.insert(0, str(REPO / "demos"))
from cot_prompts import expert_few_shot_prefix  # noqa: E402


def first_exemplar(task: str) -> str:
    """The first worked example (question, reasoning, tagged answer) from the BBH authors' prompt."""
    prefix = expert_few_shot_prefix(task, str(REPO / "BIG-Bench-Hard" / "cot-prompts"))
    blocks = re.split(r"\n\s*\n(?=Q:)", prefix)
    exemplars = [b for b in blocks if b.lstrip().startswith("Q:")]
    return exemplars[0].strip()


def with_exemplar(p0: str, task: str) -> str:
    return p0.rstrip() + "\n\nHere is a worked example of the required answer format:\n\n" + first_exemplar(task) + "\n"


def main() -> None:
    cells = [c for c in tier1_cells() + tier2_cells() if c.optimizer in ("ape", "protegi")]
    out_path = REPO / "results" / "revision" / "exemplar_intervention.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for cell in cells:
        if cell.key in results or not (REPO / "demos" / "results" / f"{cell.optimizer}_{cell.task}_seed{cell.seed}.pkl").exists():
            continue
        try:
            first_exemplar(cell.task)
        except AssertionError as error:
            print("skip", cell.key, error, flush=True)
            continue
        scorer = Scorer(cell)
        p0 = load_p0(cell)
        base = scorer.score(p0, "test", 0)
        extended = scorer.score(with_exemplar(p0, cell.task), "test", 0)
        results[cell.key] = {"p0": base.accuracy, "with_exemplar": extended.accuracy,
                             "paired": {"p0": list(base.correct), "ext": list(extended.correct)}}
        out_path.write_text(json.dumps(results))
        print("exemplar", cell.key, base.accuracy, "->", extended.accuracy, flush=True)


if __name__ == "__main__":
    main()
