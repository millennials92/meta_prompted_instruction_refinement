"""Positive control for the validation gate and the content-free bare-tag control (reviewer round 2).

    python -m mpir_study.controls

For every one of the 45 base prompts P0 this scores, on validation and test and in one serving session:
  P0                the base prompt;
  P0 + exemplar     P0 plus the first worked example (with chain-of-thought) of the BBH authors' prompt, the
                    known-good candidate of the positive control (the gate must accept it when it is better);
  P0 + bare tag     P0 plus one sentence showing only the final-answer tag, taken from that same example, with no
                    reasoning content, which separates the format effect from the worked-reasoning effect.
The gate rule is the one used everywhere else: strict improvement of the validation score over P0's.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from .analyze import REV
from .data import Cell, load_p0, tier1_cells, tier2_cells
from .exemplar import first_exemplar, with_exemplar
from .scoring import Scorer

OUT = REV / "controls.json"
TAGGED = re.compile(r"<ANS_START>.*?<ANS_END>", re.S)


def with_bare_tag(p0: str, task: str) -> str:
    tags = TAGGED.findall(first_exemplar(task))
    assert tags, task
    return p0.rstrip() + "\n\nExample of the required final-answer format: " + tags[-1].strip() + "\n"


def evaluate(cell: Cell) -> dict:
    """PromptWizard prompts already carry bare-tag worked examples, so for them only the gate check (validation
    scores of P0 and P0 + exemplar) is run; the bare-tag control and the test scoring cover the exemplar-free
    APE and ProTeGi prompts."""
    scorer = Scorer(cell)
    p0 = load_p0(cell)
    variants = {"p0": p0, "exemplar": with_exemplar(p0, cell.task)}
    if cell.optimizer != "promptwizard":
        variants["bare_tag"] = with_bare_tag(p0, cell.task)
    out = {}
    for name, prompt in variants.items():
        entry = {"val": scorer.score(prompt, "val", 0).accuracy}
        if cell.optimizer != "promptwizard":
            test = scorer.score(prompt, "test", 0)
            entry.update({"test": test.accuracy, "test_vector": list(test.correct)})
        out[name] = entry
    return out


def main() -> None:
    cells = [c for c in tier1_cells() + tier2_cells()]
    cells = [c for c in cells if (REV.parent.parent / "demos" / "results" /
                                  f"{c.optimizer}_{c.task}_seed{c.seed}.pkl").exists()]
    store = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [c for c in cells if c.key not in store]
    with ThreadPoolExecutor(3) as pool:
        for cell, result in zip(todo, pool.map(evaluate, todo)):
            store[cell.key] = result
            print("control", cell.key, {k: v["val"] for k, v in result.items()}, flush=True)
            OUT.write_text(json.dumps(store), encoding="utf-8")


if __name__ == "__main__":
    main()
