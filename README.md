# Meta-Prompted Instruction Refinement (MPIR)

## 📃 Abstract

Large language models (LLMs) are transforming artificial intelligence by enabling systems that can reason, write, and assist in complex tasks, and these capabilities that are increasingly important for science, education, and everyday applications. However, these models are critically dependent on the quality of their input prompts, making prompt design a central bottleneck. Manual prompt engineering, with techniques such as chain-of-thought reasoning and role assignment, can yield high performance, but requires expert knowledge and is not scalable. Automatic prompt optimization (APO) offers efficiency; however, its prompts often lack the structured guidance of human-designed heuristics. Here, we introduce Meta-Prompted Instruction Refinement (MPIR), a framework that refines APO-generated prompts using a seven-criteria rubric, meta-prompted evaluation and refinement, and empirical validation. MPIR outperforms APO in 16 of 23 tasks on the Big-Bench Hard (BBH) benchmark, with improvements of up to 20 percentage points on certain tasks. These results demonstrate that MPIR bridges human heuristics with automation, making prompt optimization more effective, scalable, and interpretable. Importantly, this approach democratizes AI by reducing the dependence on domain expertise, reducing labor efforts, and fostering more reliable and accessible systems.



## 🔬 Research Contributions

### Meta-Prompted Instruction Refinement (MPIR)

This project leverage the PromptWizard framework (https://github.com/microsoft/PromptWizard) to implement **Meta-Prompted Instruction Refinement (MPIR)**, a novel approach that leverages meta-prompting techniques to enhance prompt optimization:


#### MPIR Workflow:
1. **Initial Prompt Generation**: Uses PromptWizard's critique-and-refine technique
2. **Meta-Prompt Analysis**: Applies meta-prompting to analyze and critique the generated prompt
3. **Instruction Refinement**: Refines the prompt based on meta-prompt feedback
4. **Validation**: Validates the refined prompt on validation datasets
5. **Iterative Execution**: Steps 2–4 are repeated N times, creating an iterative refinement cycle.
6. **Prompt Selection**: Compare the refined prompts across validation results and select the one with the highest score as the final prompt.

## 📁 Project Structure
```python
MPIR/
├── Big-Bench-Hard
├── promptwizard/
│   └── glue/
│       └── promptopt/
│           └── techniques/ 
│               ├── critique_n_refine/      # PromptWizard technique
│               └── heuristic/              # MPIR technique  
├── demos/
│   ├── configs/
│   ├── prompt_wizard.ipynb
│   └── MPIR.ipynb
│   
├── results/
│    ├── Big_bench_hard
│    └── Albation
├── requirements.txt
└── README.md
```

## 📋 Requirements

- Python 3.9+
- OpenAI API key
- Required Python packages (see `requirements.txt`)

## 🛠️ Installation

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

## ⚙️ Configuration

### Environment Variables

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

### Configuration Files

The project uses YAML configuration files stored in `demos/configs/`.  
Each subfolder contains configs for a specific technique:

- `demos/configs/promptwizard/` → configs for **PromptWizard**  
- `demos/configs/heuristic/` → configs for **MPIR**

## 🔬 Demo Notebooks

- `demos/promptwizard.ipynb`: Promptwizard techniques with hyberbaton task
- `demos/MPIR.ipynb`: MPIR integration on the Hyperbaton dataset using PromptWizard prompts

## 📄 License

This project is licensed under the MIT License (see `LICENSE`). It vendors `BIG-Bench-Hard/` and
code adapted from Microsoft's `PromptWizard`, each under its own MIT License.