# Does Rubric-Guided Refinement Improve Prompts from Small-Model Optimizers?

**Linh Nguyen, Quang-Vinh Dang, Minh Ngoc Dinh, and Thuy Nguyen**

**Publication status:** Accepted for publication in **IEEE Access**.  
**Paper:** [IEEE Access article (DOI: 10.1109/ACCESS.2026.3757210)](https://doi.org/10.1109/ACCESS.2026.3757210)

This repository contains the code and study artifacts for Meta-Prompted Instruction Refinement (MPIR). The title and abstract below match the accepted author's version.

## Abstract

Meta-Prompted Instruction Refinement (MPIR) asks a language model to score an optimized prompt against a rubric, rewrite it, and keep the rewrite only if it beats the original on validation questions. We re-test it after an earlier evaluation was found to have a validation leak, with a repaired, logged loop and decision rules fixed in advance. The scope is narrow: Qwen3-1.7B as target, 45 base prompts from three small-model optimizers, nine Big-Bench Hard tasks. With the 1.7B model refining its own prompts, the original treatment changed the prompt in 1 of 80 runs, but 93-95% of its candidates had lost the answer format, so the test could not separate a failed idea from a failed treatment. With the format protected (positive control: the gate accepted a known-good candidate in 24 of 26 cells), the rubric-guided variant won in 1 cell and lost in 3 under a lenient metric validated on 400 AI-labelled responses, whereas a rubric-free rewrite with the same protection gained +3.8 points on average (11 winning cells of 45); the paired difference is +3.7 points [+1.1, +6.8] in its favour. Qwen3-235B and GPT-6 Luna improved the prompts under protection (+7.3 and +3.4 points, nine cells, one run each), but a rubric-free rewrite reached similar gains. One answer-format exemplar changed strict accuracy by +19.5 (APE) and +42.3 (ProTeGi) points, mostly tag format, and identical prompts disagree on 6.5% of examples. The rubric is not shown to help; conclusions are limited to Qwen3 targets, nine tasks and these optimizers.

## Study overview

MPIR scores an optimizer-generated prompt against a rubric, proposes a rewrite, and retains a candidate only when it improves validation performance. The accepted study evaluates repaired refinement variants, format-preserving treatments, rubric-free controls, and stronger refiners.

- **Evaluation:** Qwen3-1.7B as the target, 45 base prompts from three small-model optimizers, and nine Big-Bench Hard tasks.
- **Main finding:** the study does not establish a benefit from the rubric. Under format protection, rubric-free rewriting achieved similar or better results in the reported comparisons.
- **Diagnostics:** positive controls, strict and lenient answer scoring, candidate audits, and serving-noise measurements help distinguish refinement behaviour from answer-format effects.

The results are scoped to the models, tasks, and optimizer implementations evaluated in the paper.

## Repository guide

| Location | Contents |
| --- | --- |
| [`mpir_study/`](mpir_study/) | Refinement variants, experiment orchestration, scoring, controls, and analyses for the revised study. |
| [`results/revision/`](results/revision/) | Revision experiment records and analysis summaries. |
| [`demos/data/`](demos/data/) | Task data, saved train/validation/test splits, and partition records. |
| [`demos/results/`](demos/results/) | Saved base prompts and pilot artifacts. |
| [`REVISION_PROTOCOL.md`](REVISION_PROTOCOL.md) | Dated study protocol and amendments. |
| [`manuscript/`](manuscript/) | Manuscript sources, builders, and development records. |
| [`submission_revision2/upload_ready/`](submission_revision2/upload_ready/) | Archived second-revision submission package. |
| [`promptwizard/`](promptwizard/) | Framework code adapted from [Microsoft PromptWizard](https://github.com/microsoft/PromptWizard), with optimizer and MPIR implementations. |
| [`BIG-Bench-Hard/`](BIG-Bench-Hard/) | Benchmark data and original benchmark materials. |

### Using the revised study code

Start with [`mpir_study/run_study.py`](mpir_study/run_study.py) for experiment groups and [`mpir_study/client.py`](mpir_study/client.py) for model endpoint configuration. Local target inference uses an OpenAI-compatible Qwen3-1.7B endpoint (`TARGET_URL`, default `http://localhost:8000/v1`); the remote-refiner experiments use `OPENROUTER_API_KEY`. The `g5-fix`, `g7-v3f`, and `g7-api` groups cover the protected rubric-guided and rubric-free comparisons.

The runner uses saved base prompts and skips completed run files. Consult the protocol and the paper for the required models, seeds, serving configuration, and experiment scope. The legacy dependency list and notebooks below are not a complete environment specification for the revised study. Some raw-output caches are not tracked in Git; analysis scripts may require those caches or fresh model calls.

### Earlier evaluations and development notes

The September 2026 rejection and rebuild notes concern an earlier manuscript and evaluation. The accepted paper reports a subsequent repaired study and discusses the earlier evaluation as historical, exploratory work. The old results should not be substituted for the accepted study's results.

[`REBUILD.md`](REBUILD.md), [`HANDOFF-GPU.md`](HANDOFF-GPU.md), and [`REVISION2_STATUS.md`](REVISION2_STATUS.md) are retained as historical development records. Their dated plans and intermediate status statements are not the current publication status. Use the DOI-linked article for the accepted title, abstract, and conclusions; manuscript drafts and submission packages remain as versioned records.

## Citation

```bibtex
@article{nguyen2026rubric,
  author  = {Nguyen, Linh and Dang, Quang-Vinh and Dinh, Minh Ngoc and Nguyen, Thuy},
  title   = {Does Rubric-Guided Refinement Improve Prompts from Small-Model Optimizers?},
  journal = {IEEE Access},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3757210},
  url     = {https://doi.org/10.1109/ACCESS.2026.3757210}
}
```

## Legacy demo setup

The following setup is retained for the original PromptWizard/MPIR demo notebooks. For the revised study, use the entry points and protocol linked above.

### Requirements

- Python 3.9+
- OpenAI API key
- Required Python packages (see `requirements.txt`)

### 🛠️ Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### ⚙️ Configuration

#### Environment Variables

Copy `demos/.env.example` to `demos/.env` and fill in your own API credentials. Never commit the
resulting `.env` file.

```env
# OpenAI Configuration
USE_OPENAI_API_KEY="True"

OPENAI_API_KEY=""
OPENAI_MODEL_NAME =""

OPENAI_API_VERSION=""
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_DEPLOYMENT_NAME=""
```

#### Configuration Files

The project uses YAML configuration files stored in `demos/configs/`.  
Each subfolder contains configs for a specific technique:

- `demos/configs/promptwizard/` → configs for **PromptWizard**  
- `demos/configs/heuristic/` → configs for **MPIR**

### 🔬 Demo Notebooks

- `demos/promptwizard.ipynb`: PromptWizard optimization on the Hyperbaton task
- `demos/MPIR.ipynb`: MPIR integration on the Hyperbaton dataset using PromptWizard prompts

## License

This project is licensed under the MIT License (see `LICENSE`). It vendors `BIG-Bench-Hard/` and
code adapted from Microsoft's `PromptWizard`, each under its own MIT License.
