"""Cells (task, optimizer, seed): base prompt P0 and the three-way split."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

from .templates import REPO

DEMOS = REPO / "demos"
OPTIMIZERS = ("promptwizard", "ape", "protegi")
TIER1_TASKS = ("hyperbaton", "ruin_names", "penguins_in_a_table")
TIER2_TASKS = ("temporal_sequences", "movie_recommendation", "disambiguation_qa",
               "geometric_shapes", "salient_translation_error_detection", "sports_understanding")


@dataclass(frozen=True)
class Cell:
    task: str
    optimizer: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.task}__{self.optimizer}__s{self.seed}"


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_path(task: str, seed: int, part: str) -> Path:
    return DEMOS / "data" / task / f"{part}_seed{seed}.jsonl"


def load_split(task: str, seed: int, part: str) -> list[dict]:
    """Examples as {'question', 'answer'} dicts (the framework's jsonl schema)."""
    return read_jsonl(split_path(task, seed, part))


def load_p0(cell: Cell) -> str:
    with open(DEMOS / "results" / f"{cell.optimizer}_{cell.task}_seed{cell.seed}.pkl", "rb") as handle:
        return pickle.load(handle)


def tier1_cells(seeds=(42, 43, 44)) -> list[Cell]:
    return [Cell(t, o, s) for s in seeds for t in TIER1_TASKS for o in OPTIMIZERS]


def tier2_cells(seed=42) -> list[Cell]:
    return [Cell(t, o, seed) for t in TIER2_TASKS for o in OPTIMIZERS]
