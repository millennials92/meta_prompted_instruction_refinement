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

The split is written once per (task, seed) and re-loaded on every later call,
so re-running any one notebook does not perturb a partition another notebook
already depends on.
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
        existing partition file rather than reshuffling.
    :param bbh_dir: Directory holding the raw BIG-Bench-Hard task json files.
    :param data_dir: Directory under which per-task train/val/test/partition files are written.
    :return: Paths to the train/val/test jsonl files and the partition-index json file.
    """
    task_dir = os.path.join(data_dir, dataset_to_run)
    os.makedirs(task_dir, exist_ok=True)

    train_file_name = os.path.join(task_dir, "train.jsonl")
    val_file_name = os.path.join(task_dir, "val.jsonl")
    test_file_name = os.path.join(task_dir, "test.jsonl")
    partitions_file_name = os.path.join(task_dir, f"partitions_seed{seed}.json")

    already_split = (os.path.exists(partitions_file_name) and os.path.exists(train_file_name)
                      and os.path.exists(val_file_name) and os.path.exists(test_file_name))
    if already_split:
        with open(partitions_file_name) as partitions_file:
            recorded = json.load(partitions_file)
        if recorded.get("seed") == seed:
            return BBHSplitPaths(train_file_name, val_file_name, test_file_name, partitions_file_name)

    file_path = os.path.join(bbh_dir, f"{dataset_to_run}.json")
    with open(file_path) as file:
        data = json.load(file)

    examples = data["examples"]
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
        "optimizer_train_indices": shuffled_original_indices[:OPTIMIZER_TRAIN_SIZE],
        "mpir_validation_indices": shuffled_original_indices[
            OPTIMIZER_TRAIN_SIZE:OPTIMIZER_TRAIN_SIZE + MPIR_VALIDATION_SIZE],
        "test_indices": shuffled_original_indices[OPTIMIZER_TRAIN_SIZE + MPIR_VALIDATION_SIZE:],
    }
    with open(partitions_file_name, "w") as partitions_file:
        json.dump(partitions, partitions_file, indent=2)

    return BBHSplitPaths(train_file_name, val_file_name, test_file_name, partitions_file_name)
