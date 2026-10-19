"""MPIR refinement variants with per-round logging (see REVISION_PROTOCOL.md section 1).

v0  legacy   Algorithm 1 as run in the pilot: every round refines P0, greedy decoding, original rubric text.
v1  indep    every round refines P0, sampled (T=0.7), corrected rubric text.
v2  chained  every round refines the incumbent and sees the rejected candidates with their scores.
v2l chained  as v2 with the original (internally inconsistent) rubric text.
v3  free     rubric-free rewrite of P0, sampled, same gate as v1.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .client import REFINERS, Endpoint, chat
from .data import Cell, load_p0
from .scoring import Scorer, prompt_hash
from .templates import (ANSWER_FORMAT, CORRECTED_RUBRIC, ECHO_MARKERS, FREE_REWRITE_PROMPT, HISTORY_REQUEST,
                        IMPROVED_PROMPT, ORIGINAL_RUBRIC, REFINEMENT_REQUEST, REPO, TEXT_DELIMITER,
                        final_prompt_template, parse_rubric_total)

ROUNDS_DIR = REPO / "results" / "revision" / "rounds"
GEN_DIR = REPO / "results" / "revision" / "gen_cache"
TEMPERATURE = 0.7  # plain temperature sampling; top-p was dropped (vLLM's non-flashinfer top-p path is ~3x slower)
EVAL_MAX_TOKENS, REFINE_MAX_TOKENS = 1500, 3000
HISTORY_KEEP, HISTORY_CHARS = 4, 800
VALIDATION_REPS = 3
VARIANTS = ("v0", "v1", "v2", "v2l", "v3")


@dataclass(frozen=True)
class RunSpec:
    cell: Cell
    variant: str
    refiner: str = "1.7b"
    mseed: int = 1
    rounds: int = 7
    val_reps: int = VALIDATION_REPS

    @property
    def key(self) -> str:
        return f"{self.cell.key}__{self.variant}__r{self.refiner}__m{self.mseed}"

    @property
    def path(self) -> Path:
        return ROUNDS_DIR / f"{self.key}.json"


def _seed(spec: RunSpec, round_no: int, call: int) -> int:
    return spec.mseed * 100_000 + round_no * 100 + call


def _extract_candidate(text: str) -> str | None:
    matches = TEXT_DELIMITER.findall(text)
    return matches[0] if matches else None


def _history_block(entries: list[dict]) -> str:
    lines = []
    for number, entry in enumerate(entries[-HISTORY_KEEP:], 1):
        shown = entry["text"].strip()[:HISTORY_CHARS]
        lines.append(f"Attempt {number} (validation accuracy {entry['val']:.2f} vs current best "
                     f"{entry['best']:.2f}):\n<<<\n{shown}\n>>>")
    return "\n\n".join(lines)


def _generate(spec: RunSpec, refiner: Endpoint, source: str, round_no: int, history: list[dict]) -> dict:
    """Cached wrapper: a generation is stored per (spec, round) so scoring can be decoupled from the refiner."""
    path = GEN_DIR / f"{spec.key}__t{round_no}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    generated = _generate_uncached(spec, refiner, source, round_no, history)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(generated), encoding="utf-8")
    return generated


def generate_independent(spec: RunSpec) -> None:
    """Generation-only pass for the independent variants (they never depend on validation scores), so a
    large refiner can be served alone and the target scored afterwards."""
    assert spec.variant in ("v0", "v1", "v3")
    p0 = load_p0(spec.cell)
    for round_no in range(1, spec.rounds + 1):
        _generate(spec, REFINERS[spec.refiner], p0, round_no, [])


def _generate_uncached(spec: RunSpec, refiner: Endpoint, source: str, round_no: int, history: list[dict]) -> dict:
    """One evaluate+refine (or free-rewrite) call chain; returns judge/refine texts and the raw candidate."""
    sampled = spec.variant != "v0"
    kwargs = {"temperature": TEMPERATURE if sampled else 0.0}
    if spec.variant == "v3":
        raw = chat(refiner, [{"role": "user", "content": FREE_REWRITE_PROMPT.format(prompt=source, variant_instruction="")}],
                   max_tokens=REFINE_MAX_TOKENS, seed=_seed(spec, round_no, 1), **kwargs)
        return {"judge_text": None, "refine_text": raw, "raw_candidate": raw.strip() or None}
    rubric = CORRECTED_RUBRIC if spec.variant in ("v1", "v2") else ORIGINAL_RUBRIC
    messages = [{"role": "user", "content": rubric.format(prompt=source)}]
    judge_text = chat(refiner, messages, max_tokens=EVAL_MAX_TOKENS, seed=_seed(spec, round_no, 1), **kwargs)
    request = (HISTORY_REQUEST.format(history=_history_block(history))
               if spec.variant in ("v2", "v2l") and history else REFINEMENT_REQUEST)
    messages = messages + [{"role": "assistant", "content": judge_text}, {"role": "user", "content": request}]
    refine_text = chat(refiner, messages, max_tokens=REFINE_MAX_TOKENS, seed=_seed(spec, round_no, 2), **kwargs)
    return {"judge_text": judge_text, "refine_text": refine_text, "raw_candidate": _extract_candidate(refine_text)}


def _classify(spec: RunSpec, raw_candidate: str | None, source: str, p0: str, seen: set[str]) -> tuple[str, str | None]:
    """(outcome, candidate_prompt). Outcomes: fallback, echo, duplicate, validated."""
    if not raw_candidate:
        return "fallback", None
    if spec.variant != "v3" and any(marker in raw_candidate for marker in ECHO_MARKERS):
        return "echo", None
    if spec.variant == "v3":
        prompt = final_prompt_template().format(instruction=raw_candidate.strip(), few_shot_examples="",
                                                answer_format=ANSWER_FORMAT)
    else:
        prompt = IMPROVED_PROMPT.format(instruction=raw_candidate)
    if prompt.strip() in (source.strip(), p0.strip()) or prompt_hash(prompt) in seen:
        return "duplicate", prompt
    return "validated", prompt


def run_refinement(spec: RunSpec, scorer: Scorer, oracle: bool = False) -> dict:
    """Run one refinement loop, log every round, score the final prompt (and, if oracle, all candidates) on test."""
    if spec.path.exists():
        return json.loads(spec.path.read_text(encoding="utf-8"))
    refiner = REFINERS[spec.refiner]
    p0 = load_p0(spec.cell)
    baseline = scorer.score(p0, "val", 0).accuracy
    best_prompt, best_val, seen, history, records = p0, baseline, {prompt_hash(p0)}, [], []

    for round_no in range(1, spec.rounds + 1):
        source = best_prompt if spec.variant in ("v2", "v2l") else p0
        generated = _generate(spec, refiner, source, round_no, history)
        outcome, candidate = _classify(spec, generated["raw_candidate"], source, p0, seen)
        judge_total, judge_max = parse_rubric_total(generated["judge_text"] or "")
        record = {"round": round_no, "outcome": outcome, "judge_total": judge_total, "judge_max": judge_max,
                  "source_hash": prompt_hash(source), "baseline_val": baseline, "best_val_before": best_val,
                  **generated, "candidate": candidate, "cand_hash": prompt_hash(candidate) if candidate else None,
                  "val": None, "val_reps": None, "accepted": False}
        if outcome == "validated":
            seen.add(record["cand_hash"])
            reps = [scorer.score(candidate, "val", rep).accuracy for rep in range(spec.val_reps)]
            record["val"], record["val_reps"] = reps[0], reps
            if reps[0] > best_val:
                record["accepted"] = True
                best_prompt, best_val = candidate, reps[0]
            history.append({"text": candidate, "val": reps[0], "best": record["best_val_before"]})
        elif outcome == "duplicate":
            history.append({"text": candidate, "val": record["best_val_before"], "best": record["best_val_before"]})
        records.append(record)

    final_test = scorer.score(best_prompt, "test", 0)
    p0_test = scorer.score(p0, "test", 0)
    if oracle:
        for record in records:
            if record["outcome"] == "validated":
                record["test"] = scorer.score(record["candidate"], "test", 0).accuracy
    result = {"spec": {**asdict(spec), "cell": asdict(spec.cell)}, "p0_hash": prompt_hash(p0),
              "baseline_val": baseline, "final_hash": prompt_hash(best_prompt),
              "final_is_p0": best_prompt == p0, "p0_test": p0_test.accuracy, "final_test": final_test.accuracy,
              "n_test": len(final_test.correct),
              "paired": {"p0": list(p0_test.correct), "final": list(final_test.correct)},
              "rounds": records}
    ROUNDS_DIR.mkdir(parents=True, exist_ok=True)
    spec.path.write_text(json.dumps(result), encoding="utf-8")
    return result
