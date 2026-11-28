# Does Rubric-Guided Refinement Improve Prompts from Small-Model Optimizers?

Accepted in **IEEE Access**. [Read the paper](https://doi.org/10.1109/ACCESS.2026.3757210).

Code and results for Meta-Prompted Instruction Refinement (MPIR).

## Abstract

Meta-Prompted Instruction Refinement (MPIR) asks a language model to score an optimized prompt against a rubric, rewrite it, and keep the rewrite only if it beats the original on validation questions. We re-test it after an earlier evaluation was found to have a validation leak, with a repaired, logged loop and decision rules fixed in advance. The scope is narrow: Qwen3-1.7B as target, 45 base prompts from three small-model optimizers, nine Big-Bench Hard tasks. With the 1.7B model refining its own prompts, the original treatment changed the prompt in 1 of 80 runs, but 93-95% of its candidates had lost the answer format, so the test could not separate a failed idea from a failed treatment. With the format protected (positive control: the gate accepted a known-good candidate in 24 of 26 cells), the rubric-guided variant won in 1 cell and lost in 3 under a lenient metric validated on 400 AI-labelled responses, whereas a rubric-free rewrite with the same protection gained +3.8 points on average (11 winning cells of 45); the paired difference is +3.7 points [+1.1, +6.8] in its favour. Qwen3-235B and GPT-6 Luna improved the prompts under protection (+7.3 and +3.4 points, nine cells, one run each), but a rubric-free rewrite reached similar gains. One answer-format exemplar changed strict accuracy by +19.5 (APE) and +42.3 (ProTeGi) points, mostly tag format, and identical prompts disagree on 6.5% of examples. The rubric is not shown to help; conclusions are limited to Qwen3 targets, nine tasks and these optimizers.

## Code and data

- [Supplementary material](submission_final/Supplementary_Material.zip): code, experiment records, and protocol.
- [`mpir_study/`](mpir_study/): refinement methods, evaluation, and analysis.
- [`results/revision/`](results/revision/): experiment results.
- [`demos/data/`](demos/data/) and [`demos/results/`](demos/results/): data splits and saved prompts.
- [`BIG-Bench-Hard/`](BIG-Bench-Hard/): benchmark materials.

See [`run_study.py`](mpir_study/run_study.py) for experiment groups and [`client.py`](mpir_study/client.py) for model endpoint configuration.

## How to run

Use Linux or WSL2, Python 3.12, and a CUDA GPU supported by [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/). The examples use the paper's vLLM 0.28.0 serving version and the saved base prompts and data splits.

**1. Install the study runner.**

```bash
git clone https://github.com/millennials92/meta_prompted_instruction_refinement.git
cd meta_prompted_instruction_refinement
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install "vllm==0.28.0" openai python-dotenv tenacity PyYAML numpy scipy
```

These dependencies cover refinement from saved prompts; the older demo notebooks use the broader requirements.txt environment.

**2. Start the target/refiner server and leave it running.**

```bash
vllm serve Qwen/Qwen3-1.7B --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 8192 \
  --max-num-seqs 192 --gpu-memory-utilization 0.86
```

**3. In a second terminal, enter the repository, activate the same environment, and run:**

```bash
source .venv/bin/activate
export TARGET_URL=http://127.0.0.1:8000/v1
python -m mpir_study.run_study --help
python -m mpir_study.run_study --group g5-fix --mseeds 4 --workers 1
python -m mpir_study.run_study --group g7-v3f --mseeds 4 --workers 1
```

The groups run protected rubric-guided refinement (V2F) and its rubric-free control (V3F). Seed 4 starts new refinement trials; it does not reproduce the paper's reported seeds.

Completed runs are skipped. To rerun published seeds, use the seeds specified in the [protocol](archive/development/REVISION_PROTOCOL.md) in a separate working copy, with the corresponding saved runs and generation/scoring caches moved aside first. At the paper's 8192-token limit, one V2F cell exceeded the context window; increasing that limit changes the experimental setting.

Run records are saved in `results/revision/rounds/`, with generation and scoring caches in `gen_cache/` and `score_cache/` under the same directory. Inspect these JSON records for new runs. Published summaries are also in `results/revision/`; recomputing lenient analyses requires raw-output caches that are not all tracked in Git.

Other groups are listed in [run_study.py](mpir_study/run_study.py). The `g7-api` group additionally requires `OPENROUTER_API_KEY` in `demos/.env` and a running local target server.

## License

[MIT License](LICENSE). Bundled Big-Bench Hard and PromptWizard code retain their respective licenses.
