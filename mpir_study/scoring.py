"""Cached, parallel scoring of a prompt on a list of examples (target model, greedy)."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path

from .answers import is_correct_strict
from .client import TARGET, chat_many
from .data import Cell, load_split
from .templates import REPO, eval_template

CACHE_DIR = REPO / "results" / "revision" / "score_cache"
_LOCK = threading.Lock()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Scored:
    """Per-example outcome of one scoring pass (outputs kept for lenient re-parsing)."""
    correct: tuple[bool, ...]
    outputs: tuple[str, ...]

    @property
    def accuracy(self) -> float:
        return sum(self.correct) / len(self.correct) if self.correct else 0.0


class Scorer:
    """Scores prompts for one cell. A (prompt, split, rep) triple is computed once and persisted."""

    def __init__(self, cell: Cell, workers: int = 48):
        self.cell = cell
        self.workers = workers
        self.eval_prompt, self.system_prompt = eval_template(cell.optimizer)
        self.path = CACHE_DIR / f"{cell.key}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[tuple[str, str, int], Scored] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                self._cache[(row["hash"], row["split"], row["rep"])] = Scored(
                    tuple(row["correct"]), tuple(row["outputs"]))

    def score(self, prompt: str, split: str, rep: int = 0) -> Scored:
        key = (prompt_hash(prompt), split, rep)
        with _LOCK:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        examples = load_split(self.cell.task, self.cell.seed, split)
        batch = [[{"role": "system", "content": self.system_prompt},
                  {"role": "user", "content": self.eval_prompt.format(instruction=prompt, question=ex["question"])}]
                 for ex in examples]
        outputs = chat_many(TARGET, batch, workers=self.workers, temperature=0.0)
        result = Scored(tuple(is_correct_strict(out, ex["answer"]) for out, ex in zip(outputs, examples)),
                        tuple(outputs))
        with _LOCK:
            self._cache[key] = result
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"hash": key[0], "split": split, "rep": rep,
                                         "correct": list(result.correct), "outputs": list(result.outputs)}) + "\n")
        return result
