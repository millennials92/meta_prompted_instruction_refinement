"""Shared BBH data preparation for every optimizer notebook (PromptWizard, APE,
ProTeGi, MPIR).

REBUILD.md §2.1 / HANDOFF-GPU.md §3: the rebuild requires a three-way partition
per (task, seed) -- optimizer-train, mpir-validation, test -- and every
condition compared against every other condition must score its test accuracy
on the exact same held-out examples, or the example-level paired analysis in
REBUILD.md §3.2 (McNemar on matched correct/incorrect pairs) is invalid. This
module is the single place that partition is computed, so every notebook that
imports it for a given (task, seed) gets the identical split instead of each
notebook reshuffling independently.

The split is written once per (task, seed) and re-loaded on every later call.
Every artifact -- train/val/test jsonl and the partition-index json -- is
named with the seed in it, not just the index file: an earlier version of this
module shared one train.jsonl/val.jsonl/test.jsonl per task across every seed,
so preparing a second seed for a task silently overwrote the first seed's data
files while its own now-stale partitions_seed<N>.json file still reported
"seed": N and passed the cache-hit check -- exactly the kind of silent data
corruption this rebuild exists to eliminate. Confirmed and fixed 2026-09-03.
"""
import json
import os
import random
from dataclasses import dataclass
from typing import List

OPTIMIZER_TRAIN_SIZE = 25
MPIR_VALIDATION_SIZE = 25


@dataclass(frozen=True)
class BBHSplitPaths:
    train_file_name: str
    val_file_name: str
    test_file_name: str
    partitions_file_name: str


def _format_data(samples: List[dict]) -> List[dict]:
    return [{"question": sample["input"], "answer": sample["target"]} for sample in samples]


def prepare_bbh_task_split(dataset_to_run: str, bbh_processor, seed: int = 42,
                           bbh_dir: str = "../BIG-Bench-Hard/bbh",
                           data_dir: str = "data") -> BBHSplitPaths:
    """
    Idempotently produce (and, on later calls, simply reuse) the three-way
    train/mpir-validation/test partition for one BBH task at one seed.

    :param dataset_to_run: BBH task name, matching a file under bbh_dir (e.g. "hyperbaton").
    :param bbh_processor: DatasetSpecificProcessing instance whose dataset_to_jsonl writes the
        train/val/test files in the framework's expected jsonl format.
    :param seed: Seed controlling the shuffle. Re-running with the same (task, seed) reuses the
        existing split rather than reshuffling; a different seed for the same task writes
        separate, seed-namespaced files rather than overwriting the first seed's.
    :param bbh_dir: Directory holding the raw BIG-Bench-Hard task json files.
    :param data_dir: Directory under which per-task train/val/test/partition files are written.
    :return: Paths to the train/val/test jsonl files and the partition-index json file.
    """
    task_dir = os.path.join(data_dir, dataset_to_run)
    os.makedirs(task_dir, exist_ok=True)

    train_file_name = os.path.join(task_dir, f"train_seed{seed}.jsonl")
    val_file_name = os.path.join(task_dir, f"val_seed{seed}.jsonl")
    test_file_name = os.path.join(task_dir, f"test_seed{seed}.jsonl")
    partitions_file_name = os.path.join(task_dir, f"partitions_seed{seed}.json")

    file_path = os.path.join(bbh_dir, f"{dataset_to_run}.json")
    with open(file_path, encoding="utf-8") as file:
        data = json.load(file)

    examples = data["examples"]
    assert len(examples) > OPTIMIZER_TRAIN_SIZE + MPIR_VALIDATION_SIZE, (
        f"Task '{dataset_to_run}' has only {len(examples)} examples, not enough for "
        f"OPTIMIZER_TRAIN_SIZE={OPTIMIZER_TRAIN_SIZE} + MPIR_VALIDATION_SIZE={MPIR_VALIDATION_SIZE} "
        f"plus a non-empty test set.")

    already_split = (os.path.exists(partitions_file_name) and os.path.exists(train_file_name)
                      and os.path.exists(val_file_name) and os.path.exists(test_file_name))
    if already_split:
        with open(partitions_file_name, encoding="utf-8") as partitions_file:
            recorded = json.load(partitions_file)
        # Beyond the seed, cross-check the inputs that determine the split's
        # content: a source-file change or a changed partition size would
        # otherwise silently keep serving a stale split that still passes a
        # bare seed check.
        cache_is_valid = (
            recorded.get("seed") == seed
            and recorded.get("num_examples") == len(examples)
            and recorded.get("optimizer_train_size") == OPTIMIZER_TRAIN_SIZE
            and recorded.get("mpir_validation_size") == MPIR_VALIDATION_SIZE
        )
        if cache_is_valid:
            return BBHSplitPaths(train_file_name, val_file_name, test_file_name, partitions_file_name)

    original_index_by_id = {id(example): idx for idx, example in enumerate(examples)}

    rng = random.Random(seed)
    shuffled = rng.sample(examples, len(examples))

    optimizer_train_samples = shuffled[:OPTIMIZER_TRAIN_SIZE]
    mpir_validation_samples = shuffled[OPTIMIZER_TRAIN_SIZE:OPTIMIZER_TRAIN_SIZE + MPIR_VALIDATION_SIZE]
    test_samples = shuffled[OPTIMIZER_TRAIN_SIZE + MPIR_VALIDATION_SIZE:]

    bbh_processor.dataset_to_jsonl(train_file_name, dataset=_format_data(optimizer_train_samples))
    bbh_processor.dataset_to_jsonl(val_file_name, dataset=_format_data(mpir_validation_samples))
    bbh_processor.dataset_to_jsonl(test_file_name, dataset=_format_data(test_samples))

    shuffled_original_indices = [original_index_by_id[id(example)] for example in shuffled]
    partitions = {
        "seed": seed,
        "num_examples": len(examples),
        "optimizer_train_size": OPTIMIZER_TRAIN_SIZE,
        "mpir_validation_size": MPIR_VALIDATION_SIZE,
        "optimizer_train_indices": shuffled_original_indices[:OPTIMIZER_TRAIN_SIZE],
        "mpir_validation_indices": shuffled_original_indices[
            OPTIMIZER_TRAIN_SIZE:OPTIMIZER_TRAIN_SIZE + MPIR_VALIDATION_SIZE],
        "test_indices": shuffled_original_indices[OPTIMIZER_TRAIN_SIZE + MPIR_VALIDATION_SIZE:],
    }
    with open(partitions_file_name, "w", encoding="utf-8") as partitions_file:
        json.dump(partitions, partitions_file, indent=2)

    return BBHSplitPaths(train_file_name, val_file_name, test_file_name, partitions_file_name)
