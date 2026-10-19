"""Serving-noise floor (protocol G5): score the same prompt repeatedly and see how much accuracy moves."""
from __future__ import annotations

import argparse
import json

from .data import Cell, load_p0, tier1_cells, tier2_cells
from .scoring import Scorer
from .templates import REPO

REPS = 5


def noise_for_cell(cell: Cell, reps: int = REPS) -> dict:
    scorer = Scorer(cell)
    prompt = load_p0(cell)
    runs = {split: [scorer.score(prompt, split, rep) for rep in range(reps)] for split in ("val", "test")}
    result = {"cell": cell.key}
    for split, scored in runs.items():
        accs = [s.accuracy for s in scored]
        pairs = [(a, b) for i, a in enumerate(scored) for b in scored[i + 1:]]
        discordant = [sum(x != y for x, y in zip(a.correct, b.correct)) / len(a.correct) for a, b in pairs]
        result[split] = {"accs": accs, "discordance_mean": sum(discordant) / len(discordant),
                         "discordance_max": max(discordant), "n": len(scored[0].correct)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", default="tier1-42", choices=["tier1-42", "all"])
    args = parser.parse_args()
    cells = [c for c in tier1_cells() if c.seed == 42] if args.cells == "tier1-42" else tier1_cells() + tier2_cells()
    out_path = REPO / "results" / "revision" / "noise_floor.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    for cell in cells:
        if not (REPO / "demos" / "results" / f"{cell.optimizer}_{cell.task}_seed{cell.seed}.pkl").exists():
            continue
        if cell.key in existing:
            continue
        existing[cell.key] = noise_for_cell(cell)
        out_path.write_text(json.dumps(existing, indent=1))
        print("noise", cell.key, existing[cell.key]["test"]["accs"], flush=True)


if __name__ == "__main__":
    main()
