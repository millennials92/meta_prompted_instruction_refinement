"""Run optimizer searches (PromptWizard / APE / ProTeGi) for new (task, seed) cells.

Must be run with demos/ as the working directory (the optimizer configs and results/ are relative):
    python ../mpir_study/search.py --tasks hyperbaton --seeds 43,44 --optimizers ape,protegi,promptwizard
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "demos"))
os.chdir(REPO / "demos")

import run_grid  # noqa: E402  (loads demos/.env, defines run_optimizer)
from bbh_processor import BBH  # noqa: E402
from cot_prompts import task_description  # noqa: E402
from data_prep import prepare_bbh_task_split  # noqa: E402

PW_TEMPLATE = Path("configs/promptwizard/promptopt_config_hyperbaton.yaml")


def ensure_pw_config(task: str) -> None:
    """Per-task PromptWizard config: the hyperbaton config with the BBH authors' task description."""
    path = Path(f"configs/promptwizard/promptopt_config_{task}.yaml")
    if path.exists():
        return
    text = PW_TEMPLATE.read_text(encoding="utf-8")
    description = task_description(task).replace("'", "''")
    text = re.sub(r"task_description\s*:.*", f"task_description : '{description}'", text, count=1)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--optimizers", default="ape,protegi,promptwizard")
    args = parser.parse_args()
    processor = BBH()
    failures = []
    for task in args.tasks.split(","):
        ensure_pw_config(task)
        for seed in (int(s) for s in args.seeds.split(",")):
            paths = prepare_bbh_task_split(task, processor, seed=seed)
            for optimizer in args.optimizers.split(","):
                try:
                    run_grid.run_optimizer(optimizer, task, seed, paths.train_file_name, processor)
                except Exception:
                    print(f"FAILED {task}/{optimizer}/{seed}\n{traceback.format_exc()}", flush=True)
                    failures.append((task, optimizer, seed))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
