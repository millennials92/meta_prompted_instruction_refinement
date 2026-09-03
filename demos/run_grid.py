"""Grid runner for REBUILD.md §4.0 (pilot) and §4.1 (full grid).

Runs, per (task, seed): zero_shot_cot, expert_few_shot_cot, promptwizard,
promptwizard_mpir, ape, ape_mpir, protegi, protegi_mpir, free_rewrite. This
replaces manually re-running demos/*.ipynb once per task -- the notebooks
remain the documented, inspectable reference for what each condition does;
this script drives them unattended across many (task, seed) pairs.

Resumable: before running an optimizer search or a condition's evaluation, it
checks whether the artifact (results/<technique>_<task>_seed<seed>.pkl, or
results/predictions/<task>_<condition>_<seed>.jsonl) already exists and skips
recomputing it. A crashed run (e.g. LLMMgr.chat_completion exhausting its
retries -- see promptwizard/glue/common/llm/llm_mgr.py) can simply be
re-invoked with the same arguments.

Usage:
    python run_grid.py --tasks hyperbaton,ruin_names,penguins_in_a_table --seeds 42
    python run_grid.py --tasks hyperbaton --seeds 42,43,44
"""
import argparse
import json
import os
import pickle
import sys
import time
import traceback

sys.path.insert(0, "../")

from bbh_processor import BBH
from cot_prompts import expert_few_shot_prefix, task_description as get_task_description
from data_prep import prepare_bbh_task_split
from promptwizard.glue.common.llm.llm_mgr import LLMMgr
from promptwizard.glue.promptopt.instantiate import GluePromptOpt

RESULTS_DIR = "results"
PREDICTIONS_DIR = "results/predictions"

# Manuscript Appendix D's exact meta-prompt (preserved in content_blocks.py
# even though no implementing code ever existed elsewhere in the repo --
# REBUILD.md §2.0). {variant_instruction} has no surviving definition
# anywhere in the manuscript source; left empty here as the only defensible
# default. Note this prompt requests the rewritten prompt directly ("Output
# only the rewritten task-solving prompt"), not delimiter-wrapped -- so unlike
# every other generation-style prompt in this codebase, its response is used
# as-is rather than parsed via TEXT_DELIMITER_PATTERN.
FREE_REWRITE_PROMPT = """You are given a task-solving prompt generated for a Big-Bench Hard
task.

Rewrite the prompt to improve clarity, readability,
and instruction organization while preserving the
original task meaning.

Preserve the original task, answer format, and examples.
Do not change the meaning of the task.
Do not add unrelated content.
Do not use any prompt-evaluation rubric.
Do not mention rubric criteria, scoring, strengths,
weaknesses, or feedback.

Output only the rewritten task-solving prompt.

{variant_instruction}

Original prompt:
<START>
{prompt}
<END>"""

OPTIMIZER_CONFIGS = {
    "promptwizard": ("configs/promptwizard/promptopt_config_{task}.yaml", "configs/promptwizard/setup_config.yaml"),
    "ape": ("configs/ape/promptopt_config.yaml", "configs/ape/setup_config.yaml"),
    "protegi": ("configs/protegi/promptopt_config.yaml", "configs/protegi/setup_config.yaml"),
}


def pkl_path(technique: str, task: str, seed: int) -> str:
    return os.path.join(RESULTS_DIR, f"{technique}_{task}_seed{seed}.pkl")


def predictions_path(task: str, condition: str, seed: int) -> str:
    return os.path.join(PREDICTIONS_DIR, f"{task}_{condition}_{seed}.jsonl")


def condition_already_done(task: str, condition: str, seed: int) -> bool:
    return os.path.exists(predictions_path(task, condition, seed))


def save_pkl(technique: str, task: str, seed: int, prompt: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = pkl_path(technique, task, seed)
    with open(path, "wb") as f:
        pickle.dump(prompt, f)
    return path


def load_pkl(technique: str, task: str, seed: int) -> str:
    with open(pkl_path(technique, task, seed), "rb") as f:
        return pickle.load(f)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate_condition(gp: GluePromptOpt, test_file_name: str, task: str, condition: str,
                       seed: int, best_prompt: str, summary: list) -> None:
    if condition_already_done(task, condition, seed):
        log(f"SKIP  {task}/{condition}/seed{seed} -- predictions already exist")
        with open(predictions_path(task, condition, seed), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        accuracy = sum(r["is_correct"] for r in rows) / len(rows) if rows else 0.0
        summary.append({"task": task, "condition": condition, "seed": seed,
                        "accuracy": accuracy, "n": len(rows), "resumed": True})
        return

    gp.BEST_PROMPT = best_prompt
    start = time.time()
    accuracy = gp.evaluate(test_file_name, task_name=task, condition_name=condition,
                           seed=seed, predictions_dir=PREDICTIONS_DIR)
    elapsed = time.time() - start
    log(f"DONE  {task}/{condition}/seed{seed}: accuracy={accuracy:.4f} ({elapsed:.0f}s)")
    summary.append({"task": task, "condition": condition, "seed": seed,
                    "accuracy": accuracy, "elapsed_sec": elapsed, "resumed": False})


def run_optimizer(optimizer_name: str, task: str, seed: int, train_file_name: str,
                  bbh_processor: BBH) -> str:
    if os.path.exists(pkl_path(optimizer_name, task, seed)):
        log(f"SKIP  {task}/{optimizer_name}/seed{seed} search -- prompt already saved")
        return load_pkl(optimizer_name, task, seed)

    opt_config_template, setup_config = OPTIMIZER_CONFIGS[optimizer_name]
    opt_config = opt_config_template.format(task=task)
    if not os.path.exists(opt_config):
        raise FileNotFoundError(
            f"No config for optimizer='{optimizer_name}' task='{task}': {opt_config} does not exist. "
            f"Add a per-task promptopt_config_{{task}}.yaml under configs/{optimizer_name}/ "
            f"(see configs/promptwizard/promptopt_config_hyperbaton.yaml for the pattern).")

    log(f"START {task}/{optimizer_name}/seed{seed} search")
    start = time.time()
    gp_opt = GluePromptOpt(opt_config, setup_config, train_file_name, bbh_processor, seed=seed)
    if optimizer_name == "promptwizard":
        best_prompt, _ = gp_opt.get_best_prompt(use_examples=False, run_without_train_examples=False,
                                                  generate_synthetic_examples=False)
    else:
        best_prompt, _ = gp_opt.get_best_prompt()
    save_pkl(optimizer_name, task, seed, best_prompt)
    log(f"DONE  {task}/{optimizer_name}/seed{seed} search ({time.time() - start:.0f}s)")
    return best_prompt


def run_mpir(base_optimizer: str, base_prompt: str, task: str, seed: int,
            train_file_name: str, val_file_name: str, bbh_processor: BBH):
    """Returns (mpir_prompt, gp_mpir) in both the skip and fresh-run cases --
    gp_mpir is always (re)constructed since GluePromptOpt.__init__ makes no
    LLM calls and is cheap; only the expensive improve_prompt() call is
    skipped when a saved prompt already exists."""
    mpir_technique = f"{base_optimizer}_mpir"
    gp_mpir = GluePromptOpt("configs/heuristic/promptopt_config.yaml",
                            "configs/heuristic/setup_config.yaml",
                            train_file_name, bbh_processor,
                            validation_dataset_jsonl=val_file_name, seed=seed)

    if os.path.exists(pkl_path(mpir_technique, task, seed)):
        log(f"SKIP  {task}/{mpir_technique}/seed{seed} refinement -- prompt already saved")
        return load_pkl(mpir_technique, task, seed), gp_mpir

    log(f"START {task}/{mpir_technique}/seed{seed} refinement")
    start = time.time()
    mpir_prompt = gp_mpir.improve_prompt(base_prompt)
    save_pkl(mpir_technique, task, seed, mpir_prompt)
    log(f"DONE  {task}/{mpir_technique}/seed{seed} refinement ({time.time() - start:.0f}s)")
    return mpir_prompt, gp_mpir


def run_task(task: str, seed: int, summary: list) -> None:
    log(f"=== Task '{task}', seed {seed} ===")
    bbh_processor = BBH()
    split_paths = prepare_bbh_task_split(task, bbh_processor, seed=seed)
    test_file_name = split_paths.test_file_name

    # A lightweight GluePromptOpt vehicle for evaluate()'s logging on
    # non-optimizer conditions (zero-shot/few-shot/free-rewrite) -- reuses
    # the heuristic config purely because it has no task-specific fields.
    gp_vehicle = GluePromptOpt("configs/heuristic/promptopt_config.yaml",
                               "configs/heuristic/setup_config.yaml",
                               split_paths.train_file_name, bbh_processor, seed=seed)
    answer_format = gp_vehicle.prompt_opt_param.answer_format
    final_prompt_template = gp_vehicle.prompt_opt.prompt_pool.final_prompt

    task_desc = get_task_description(task)
    # Manuscript Appendix A.1 preserves the exact zero-shot prompt used for
    # penguins_in_a_table ("You are given a task that require [description].
    # Let's think step by step. For each question, wrap only the final letter
    # ... between <ANS_START> and <ANS_END> tags") -- the task-framing sentence
    # is adopted here, but the delimiter instruction stays the general
    # answer_format used by every other condition in this rebuild (a fixed
    # letter-set enumeration per task doesn't generalize across all 23 tasks'
    # varying answer shapes the way a general "wrap the final answer" does).
    zero_shot_instruction = f"You are given the following task: {task_desc}\nLet's think step by step."
    zero_shot_prompt = final_prompt_template.format(
        instruction=zero_shot_instruction, few_shot_examples="", answer_format=answer_format)
    evaluate_condition(gp_vehicle, test_file_name, task, "zero_shot_cot", seed, zero_shot_prompt, summary)

    few_shot_instruction = expert_few_shot_prefix(task)
    few_shot_prompt = final_prompt_template.format(
        instruction=few_shot_instruction, few_shot_examples="", answer_format=answer_format)
    evaluate_condition(gp_vehicle, test_file_name, task, "expert_few_shot_cot", seed, few_shot_prompt, summary)

    for optimizer_name in ["promptwizard", "ape", "protegi"]:
        best_prompt = run_optimizer(optimizer_name, task, seed, split_paths.train_file_name, bbh_processor)
        opt_config_template, setup_config = OPTIMIZER_CONFIGS[optimizer_name]
        gp_opt_eval = GluePromptOpt(opt_config_template.format(task=task), setup_config,
                                    split_paths.train_file_name, bbh_processor, seed=seed)
        evaluate_condition(gp_opt_eval, test_file_name, task, optimizer_name, seed, best_prompt, summary)

        mpir_prompt, gp_mpir = run_mpir(optimizer_name, best_prompt, task, seed,
                                        split_paths.train_file_name, split_paths.val_file_name, bbh_processor)
        evaluate_condition(gp_mpir, test_file_name, task, f"{optimizer_name}_mpir", seed, mpir_prompt, summary)

    # Free rewrite: unstructured control, rewriting the PromptWizard prompt
    # once with no rubric (manuscript §4.3.1 / Appendix D; REBUILD.md §4.0).
    if condition_already_done(task, "free_rewrite", seed):
        evaluate_condition(gp_vehicle, test_file_name, task, "free_rewrite", seed, None, summary)
    else:
        promptwizard_prompt = load_pkl("promptwizard", task, seed)
        rewrite_response = LLMMgr.chat_completion(
            [{"role": "user", "content": FREE_REWRITE_PROMPT.format(
                prompt=promptwizard_prompt, variant_instruction="")}])
        rewritten_instruction = rewrite_response.strip() or promptwizard_prompt
        free_rewrite_prompt = final_prompt_template.format(
            instruction=rewritten_instruction, few_shot_examples="", answer_format=answer_format)
        evaluate_condition(gp_vehicle, test_file_name, task, "free_rewrite", seed, free_rewrite_prompt, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="Comma-separated BBH task names")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds")
    parser.add_argument("--summary-out", default="results/grid_summary.jsonl",
                        help="Where to append the run summary (one JSON object per line)")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    summary = []
    failures = []
    for task in tasks:
        for seed in seeds:
            try:
                run_task(task, seed, summary)
            except Exception:
                log(f"FAILED task={task} seed={seed}:\n{traceback.format_exc()}")
                failures.append((task, seed))

    os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
    with open(args.summary_out, "a", encoding="utf-8") as f:
        for row in summary:
            f.write(json.dumps(row) + "\n")

    log(f"=== Grid run complete: {len(summary)} condition results, {len(failures)} task/seed failures ===")
    if failures:
        log(f"Failed: {failures}")
        log("Re-invoke this script with the same arguments to resume -- completed "
            "conditions are skipped automatically.")
        sys.exit(1)


if __name__ == "__main__":
    main()
