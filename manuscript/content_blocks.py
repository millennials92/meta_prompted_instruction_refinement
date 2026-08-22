# -*- coding: utf-8 -*-
# Final manuscript content for the AIA (Artificial Intelligence and Applications) docx.
# Citation tokens use unicode guillemets, e.g. ‘9‹ or ‹9,10›, referring to
# the bare numeric suffix of the original reference_N bib keys. These get resolved to
# bracketed, renumbered-by-first-appearance citations, e.g. [3] or [3, 7], by build_docx.py.

TITLE = "Meta-Prompted Instruction Refinement: Bridging Manual Prompting Techniques and Automatic Prompt Optimization"

AUTHORS = [
    {"name": "Linh Nguyen", "affil_idx": [1], "corresponding": False},
    {"name": "Quang-Vinh Dang", "affil_idx": [2], "corresponding": True},
    {"name": "Minh Ngoc Dinh", "affil_idx": [3], "corresponding": False},
    {"name": "Thuy Nguyen", "affil_idx": [1], "corresponding": False},
]
AFFILIATIONS = [
    "School of Science, Engineering & Technology, RMIT University Vietnam, 702 Nguyen Van Linh Boulevard, Ho Chi Minh City 700000, Vietnam",
    "British University Vietnam, Hung Yen, Vietnam",
    "School of Computing Technologies, RMIT University, Melbourne, Australia",
]
CORRESPONDING_AUTHOR_NOTE = (
    "*Corresponding author: Quang-Vinh Dang, British University Vietnam, Hung Yen, Vietnam. "
    "Email: [corresponding author email to be provided]"
)

ABSTRACT = (
    "Large language models (LLMs) are transforming artificial intelligence by enabling systems that can "
    "reason, write, and assist with complex tasks, capabilities that are increasingly important for "
    "science, education, and everyday applications. Yet these models remain critically dependent on the "
    "quality of their input prompts, making prompt design a central bottleneck. Manual prompt engineering, "
    "using techniques such as chain-of-thought reasoning and role assignment, can yield high performance "
    "but requires expert knowledge and does not scale. Automatic prompt optimization (APO) offers "
    "efficiency, but its outputs often lack the structured guidance that makes human-crafted prompts "
    "effective. This paper introduces Meta-Prompted Instruction Refinement (MPIR), a framework that "
    "refines APO-generated prompts through a seven-criteria rubric, meta-prompted evaluation and "
    "refinement, and empirical validation. Extensive experiments on the Big-Bench Hard (BBH) benchmark "
    "show that MPIR outperforms its PromptWizard baseline on 16 of 23 tasks, with gains of up to 20 "
    "percentage points on individual tasks, and that the same refinement layer improves two further APO "
    "methods (Iterative APE and ProTeGi) as well as a different underlying LLM family. These results "
    "demonstrate that MPIR bridges human heuristics with automation, making prompt optimization more "
    "effective, interpretable, and scalable, while reducing reliance on expert prompt engineering."
)

KEYWORDS = "Prompt engineering, meta-prompting, prompt optimization, large language models, automatic prompt optimization, Big-Bench Hard"

# ---------------------------------------------------------------------------
BLOCKS = []


def H1(text):
    BLOCKS.append({"type": "h1", "text": text})


def H2(text):
    BLOCKS.append({"type": "h2", "text": text})


def H3(text):
    BLOCKS.append({"type": "h3", "text": text})


def P(text):
    BLOCKS.append({"type": "p", "text": text})


def BULLETS(items):
    BLOCKS.append({"type": "bullets", "items": items})


def FIG(path, caption, full=True):
    BLOCKS.append({"type": "figure", "path": path, "caption": caption, "full": full})


def TABLE(caption, header, rows, full=True, note=None, colw=None):
    BLOCKS.append({"type": "table", "caption": caption, "header": header, "rows": rows,
                    "full": full, "note": note, "colw": colw})


def CODE(lines, caption=None, full=True):
    BLOCKS.append({"type": "code", "lines": lines, "caption": caption, "full": full})


def ALGO(title, require, ensure, steps, ret):
    BLOCKS.append({"type": "algo", "title": title, "require": require, "ensure": ensure,
                    "steps": steps, "ret": ret})


# ===========================================================================
H1("1. Introduction")

P("Large language models (LLMs) have rapidly advanced natural language processing, achieving strong "
  "performance on tasks such as translation ‹35›, summarization ‹34›, and question "
  "answering ‹36›, and are increasingly central to applied domains ranging from law to "
  "healthcare and decision support ‹37,38›. Their effectiveness, however, depends critically on "
  "the design of the input prompt: even small changes in phrasing can produce large differences in output "
  "accuracy, consistency, and reliability ‹33›. Prompt design has therefore become a major "
  "bottleneck to using LLMs reliably at scale.")

P("Two broad strategies dominate current approaches to prompt design: manual prompting and automatic "
  "prompt optimization (APO) ‹2›. Manual prompting relies on heuristics drawn from human "
  "intuition and experience, such as chain-of-thought reasoning ‹7›, role assignment "
  "‹12›, and few-shot demonstration ‹4›. These techniques are often effective and "
  "interpretable, but require domain expertise ‹3,32› and substantial trial-and-error "
  "‹26›, and are difficult to scale across the growing range of LLM applications. APO instead "
  "automates prompt generation and refinement through iterative search or model feedback ‹20,21›, "
  "improving efficiency and scalability but often forfeiting the structured guidance that makes "
  "human-crafted prompts effective. The trade-off between the effectiveness and interpretability of "
  "manual prompting and the efficiency and scalability of APO remains unresolved.")

P("Recent work on meta-prompting suggests a path forward: meta-prompts guide an LLM in generating, "
  "critiquing, or refining prompts, effectively adding a higher-order layer of reasoning to prompt "
  "optimization ‹25,26›. Yet most existing approaches integrate heuristics only loosely, "
  "either as initial seeds or as informal guidance during optimization, without systematic evaluation or "
  "explicit accounting for LLMs' sensitivity to subtle phrasing differences ‹33›. As a result, "
  "their effects are difficult to quantify and to generalize. What is missing is a principled framework "
  "that embeds human heuristics into APO in a structured, repeatable, and empirically validated way.")

P("This paper addresses that gap with Meta-Prompted Instruction Refinement (MPIR), a framework that "
  "integrates manual prompting heuristics into APO through a structured three-stage cycle of evaluation, "
  "refinement, and validation. MPIR functions as a modular layer on top of an existing APO system: it "
  "translates heuristics into explicit evaluation criteria, applies meta-prompted revision, and "
  "empirically validates each candidate prompt, thereby embedding human-guided design principles into "
  "APO-generated prompts without additional human intervention.")

P("Specifically, this study pursues four objectives:")
BULLETS([
    ("To develop a structured rubric for prompt evaluation.",
     "We formalize established manual prompting heuristics into a seven-criteria framework for "
     "systematically evaluating APO-generated prompts."),
    ("To evaluate whether meta-prompted refinement improves APO-generated prompts.",
     "MPIR applies rubric-guided feedback to iteratively refine prompts and empirically measures the "
     "resulting effectiveness on benchmark tasks."),
    ("To investigate the generalizability and modularity of heuristic-guided refinement.",
     "We examine whether MPIR functions as a modular refinement layer across different APO frameworks "
     "and across different underlying LLMs, while maintaining consistent gains."),
    ("To investigate how individual prompting heuristics contribute to performance.",
     "Through targeted ablations on representative BBH tasks, we quantify how each rubric criterion "
     "contributes to refinement outcomes."),
])

P("The remainder of this paper is organized as follows. Section 2 reviews related work on prompt "
  "engineering and automatic prompt optimization. Section 3 introduces the MPIR framework. Section 4 "
  "describes the implementation. Section 5 reports results and analysis. Section 6 concludes the paper.")

# ===========================================================================
H1("2. Related Work")

H2("2.1. Prompting as Human-AI Interaction")
P("Prompting is the primary mechanism for steering LLM behavior, leveraging in-context learning to shape "
  "outputs without updating model parameters ‹2›. It has been described as a form of natural "
  "language programming ‹3›, emphasizing scalability and efficiency. As a direct channel of "
  "user control, however, its effectiveness often depends on extensive trial and error, which has given "
  "rise to a wide range of heuristic techniques whose potential and limitations we review next.")

H2("2.2. Manual Prompting Heuristics")
P("Manual prompt engineering improves LLM performance through deliberate prompt design without modifying "
  "model parameters. Zero-shot prompting relies solely on task instructions ‹4›, while few-shot "
  "prompting introduces demonstrations that improve generalization and output consistency ‹5›; "
  "example formatting and label consistency further shape performance, underscoring the importance of "
  "prompt structure in activating latent model capabilities ‹6›.")

P("More advanced heuristics target reasoning structure and contextual alignment. Chain-of-thought (CoT) "
  "prompting improves multistep reasoning by decomposing problems into intermediate steps ‹7,8›, "
  "and coherent, relevant reasoning steps matter more than reasoning length alone ‹9,10›. Role "
  "prompting introduces contextual identities, such as “Act as a math teacher,” to guide "
  "reasoning behavior and domain alignment ‹11,12›, although poorly aligned personas can "
  "degrade reasoning quality ‹16›. Step-back prompting further improves knowledge-intensive "
  "reasoning by encouraging the model to first abstract key principles before solving a task "
  "‹13,17›.")

P("Prompt organization matters as well: because LLMs attend disproportionately to information placed "
  "near the beginning or end of a prompt ‹14,15›, strategically ordering instructions and "
  "constraints can improve reasoning accuracy and instruction-following. Collectively, these heuristics "
  "show that carefully designed instructions, reasoning structure, contextual framing, and exemplars can "
  "substantially improve LLM performance—but they remain labor-intensive, task-specific, and "
  "dependent on human expertise, which motivates automatic prompt optimization.")

H2("2.3. Automatic Prompt Optimization")
P("APO methods scale prompt refinement through iterative generation, evaluation, and selection "
  "‹47›. Continuous approaches optimize embeddings but require access to model internals, "
  "whereas discrete approaches edit natural language directly and produce interpretable outputs "
  "‹18›. Automatic Prompt Engineer (APE) uses an LLM to generate and refine instructions from "
  "input-output demonstrations ‹19›; ProTeGi mimics gradient descent with textual feedback to "
  "guide revisions ‹20›; EvoPrompt adapts evolutionary algorithms to mutate and recombine "
  "prompts ‹21›; and OPRO uses an LLM as a natural-language optimizer that iteratively "
  "generates and evaluates candidates conditioned on prior scores ‹22›. PromptWizard is a "
  "self-evolving, self-adapting framework that optimizes prompts through a feedback-driven critique and "
  "synthesis process ‹23›.")

P("Related frameworks pursue complementary strategies. DSPy treats LM pipelines as text-transformation "
  "graphs, compiling declarative modules into effective prompting, fine-tuning, and reasoning strategies "
  "‹51›. TextGrad performs automatic optimization by propagating natural-language feedback "
  "through a system, analogous to how PyTorch propagates gradients ‹52›. Self-Refine has an "
  "LLM iteratively critique and revise its own output, echoing how humans improve writing through "
  "successive drafts ‹53›. Reflexion lets an agent verbally reflect on its mistakes and store "
  "these reflections in memory to improve future decisions, rather than updating model weights "
  "‹54›. Because these frameworks treat optimization largely as black-box search, their outputs "
  "can be less robust and transparent, and by prioritizing automation they risk neglecting the structured "
  "heuristics that make manual prompting effective.")

H2("2.4. Meta-Prompting and Heuristic-Guided Automatic Prompt Optimization")
P("Meta-prompting guides an LLM to generate or refine prompts using performance feedback ‹18›, "
  "and several studies combine manual heuristics with automatic optimization. PE2 formalizes refinement "
  "through step-by-step reasoning templates and structured analysis ‹24›; dual-phase strategies "
  "combine a meta-instruction-guided initialization phase with iterative sentence-level optimization to "
  "accelerate convergence on high-quality prompts ‹25›; and PROPEL integrates "
  "a large set of expert-derived prompting principles as priors to guide refinement ‹26›. "
  "PromptWizard itself adopts heuristic-driven strategies to initialize prompts and self-generated "
  "reasoning chains to construct few-shot examples ‹23›.")

P("Important gaps remain, however. When heuristics are used at all, they are typically injected at "
  "initialization ‹25› or loosely during optimization ‹26› rather than treated as an "
  "independent stage, so their contribution is difficult to attribute to measurable gains. Current "
  "approaches also rarely address LLMs' sensitivity to subtle phrasing differences ‹33›, so "
  "reported improvements may not consistently transfer to task outcomes. Meta-prompting thus expands "
  "automation but still lacks a structured integration of heuristic knowledge. MPIR differs from its two "
  "closest relatives in this respect. PE2 is itself a complete, self-contained optimizer that formalizes "
  "one evaluation-refinement loop; it does not layer onto an already-optimized prompt from another APO "
  "system, nor does it select among candidates using a held-out validation score. PROPEL instead bakes a "
  "large set of expert-derived principles into a single refinement pass as implicit priors, rather than "
  "scoring a prompt against each principle explicitly and iterating multiple evaluation-refinement-"
  "validation cycles until an empirically best candidate is found. MPIR's contribution is precisely this "
  "combination—an explicit, criterion-by-criterion rubric, applied as a separate stage on top of an "
  "existing APO output, with iterative empirical validation deciding which candidate survives—rather than "
  "any single component in isolation.")

P("The literature reveals a persistent tension: manual prompting offers interpretability and "
  "effectiveness but does not scale, while automatic optimization offers efficiency and scalability but "
  "often neglects heuristic grounding. Emerging meta-prompting methods attempt to bridge this divide but "
  "lack systematic integration and principled evaluation. MPIR is designed to close this gap by "
  "formalizing manual heuristics into explicit rubrics, applying them as an independent stage of the "
  "optimization pipeline after an APO prompt has already been produced, and validating the resulting "
  "prompt's effectiveness on held-out data—combining human insight with automated scalability.")

# ===========================================================================
H1("3. Meta-Prompted Instruction Refinement Framework")

P("We propose Meta-Prompted Instruction Refinement (MPIR), a framework that extends APO by embedding "
  "manual prompting heuristics into a structured, iterative refinement process (Figure 1). MPIR uses "
  "meta-prompting to guide an LLM in systematically evaluating, refining, and validating APO-generated "
  "prompts, and consists of three components: (1) a heuristic rubric encoding principles such as role "
  "prompting and chain-of-thought reasoning; (2) an evaluation-refinement loop in which meta-prompts "
  "critique and revise candidate prompts; and (3) a validation stage that measures refined-prompt "
  "effectiveness on a held-out set using accuracy.")

P("These three stages echo concepts that appear under different names elsewhere in the prompt-optimization "
  "literature: evaluation relates to the critique-based, feedback-driven assessment used in frameworks "
  "such as ProTeGi and PromptWizard ‹20,23›; refinement aligns with the revision and "
  "self-improvement strategies common to meta-prompting ‹23,24›; and validation relates to the "
  "empirical, performance-based candidate selection used across APO frameworks ‹19,22,23›. By "
  "organizing these components into one rubric-driven cycle, MPIR provides a more structured integration "
  "of heuristic-guided prompt refinement than prior work.")

FIG("figure_3.png",
    "Overview of the Meta-Prompted Instruction Refinement (MPIR) cycle. An APO-generated prompt P0 "
    "enters a three-stage loop—Evaluation, Refinement, and Validation—guided by heuristic "
    "criteria. Each iteration uses P0 as the base and generates an improved candidate prompt "
    "(P1, P2, ..., PN). After N iterations, the prompt with the highest validation score is selected "
    "as the final prompt.")

H2("3.1. Problem Formulation")
P("Let (Qv, Av) = {(qi, ai)}, i = 1..M, denote a validation set of M question-answer pairs, where qi is "
  "the i-th input question and ai its ground-truth answer. An LLM L generates outputs with probability "
  "pL(ai | qi, P), where P is the prompt under evaluation, and its performance is measured by a "
  "task-specific metric A(L, P) (accuracy) computed on the validation set. The goal of MPIR is to refine "
  "the APO-generated prompt P0 into a prompt Pfinal that maximizes this metric:")
P("Pfinal = argmax over P' in R(P0) of A(L, P'),")
P("where R(P0) denotes the space of candidate refinements derived from P0.")

H2("3.2. MPIR Algorithm")
P("Algorithm 1 details the iterative process of evaluation, refinement, and validation that produces the "
  "final refined prompt.")

ALGO(
    "Algorithm 1. MPIR Algorithm",
    require=("L: target LLM; L′: LLM for evaluation and refinement; Meval: meta-prompt for rubric "
              "evaluation; Mrefine: meta-prompt for refinement; P0: APO-generated prompt; "
              "(Qv, Av) = {(qi, ai)}, i=1..M: validation set; R: seven-criteria rubric; N: refinement rounds"),
    ensure="Refined prompt Pfinal",
    steps=[
        "Initialize: Pfinal ← P0; Abest ← −∞",
        "for t = 1 to N do",
        "    S ← Evaluate(Meval, L′, P0, R)                    ▷ Rubric feedback",
        "    P′ ← Refine(Mrefine, L′, P0, S)                  ▷ Generate revised prompt",
        "    A′ ← Validate(L, P′, (Qv, Av))                    ▷ Performance of P′",
        "    if A′ ≥ Abest then",
        "        Pfinal ← P′; Abest ← A′",
        "    end if",
        "end for",
    ],
    ret="Pfinal",
)

P("Notation: P0 is the initial APO prompt used as the base for every refinement round; Pfinal is the "
  "best-performing prompt found across all rounds; P′ is the candidate produced in the current round; "
  "S is the textual rubric feedback returned by Evaluate(); Abest is the best validation score observed "
  "so far (initialized to −∞); and A′ is the validation score of the current candidate P′.")

H2("3.3. Seven-Criteria Evaluation Rubric")
P("The rubric combines insights from academic research, industry practice, and iterative "
  "experimentation. Research provides a diverse catalog of prompting strategies ‹44,45›, while "
  "industry contributes large-scale validation from deployed systems and user bases ‹40,41,42,43›; "
  "iterative testing then refined these insights into a form that generalizes across tasks. The resulting "
  "seven criteria are intentionally ordered as a coherent progression—from role framing and "
  "conceptual grounding, through structured reasoning and exemplification, to task closure—turning "
  "heuristics into a systematic framework for improving prompt quality:")

BULLETS([
    ("Role Prompting", "Assign a task-relevant role to condition model behavior."),
    ("Step-Back Reasoning", "Encourage abstraction and articulation of core concepts."),
    ("Guided Chain of Thought", "Structure reasoning into clear, sequential steps."),
    ("Instruction and Separation", "Separate task instructions from context to reduce ambiguity."),
    ("Output Format with Examples", "Specify the response schema and provide illustrative examples."),
    ("Worked Reasoning in Examples", "Include step-by-step reasoning within exemplars."),
    ("Conclusion", "Restate the task at the end to reinforce intent and coherence."),
])

P("Each criterion is grounded in an established prompting technique: Role Prompting in contextual role "
  "assignment ‹11,12›; Step-Back Reasoning in evidence that abstracting key principles improves "
  "reasoning ‹13,17›; Guided Chain-of-Thought in structured intermediate reasoning for complex "
  "tasks ‹7,8›; and Output Format with Examples and Worked Reasoning in Examples in few-shot and "
  "demonstration-based learning ‹5,6›. The Conclusion criterion follows from evidence of "
  "positional bias in LLMs, whereby information near the start or end of a prompt receives more attention "
  "‹14,15›. Consolidating these techniques operationalizes established manual heuristics into a "
  "systematic rubric that the next section applies to APO-generated prompts.")

H2("3.4. Evaluation and Refinement")
P("The evaluation-refinement stage extends manual prompting strategies into a systematic process for "
  "improving APO-generated prompts, mirroring how a human prompt engineer would diagnose a prompt's "
  "weaknesses and revise it using proven heuristics.")

P("The evaluation stage uses a meta-prompt (Figure 2) that directs the LLM to apply the seven-criteria "
  "rubric to an APO-generated prompt, scoring each criterion and identifying strengths, weaknesses, and "
  "actionable improvements. This produces a structured critique that combines quantitative scoring with "
  "qualitative insight, closely approximating expert human review, and reveals where the prompt departs "
  "from established heuristics.")

CODE([
    "### Task Overview",
    "You are a Senior Prompt Engineer with two main tasks:",
    "1. Evaluate prompts using a 7-criteria rubric.",
    "2. Refine prompts by applying evaluation feedback.",
    "Always remain objective, precise, and helpful.",
    "### Prompt Under Evaluation",
    "[Prompt]: {prompt}",
    "### Instructions",
    "1. Review Prompt: read the prompt to understand its purpose and structure.",
    "2. Apply Rubric: assess the prompt against the seven criteria below.",
    "3. Score & Justify: for each criterion, assign a score (1-5), state one",
    "   strength, one weakness, and a brief rationale.",
    "4. Calculate Total: sum all ratings for a cumulative score out of 35.",
    "5. Recommend Improvements: give 7-10 actionable suggestions.",
    "6. Report Findings: use the standardized output format below.",
    "### Seven-Criteria Evaluation Rubric",
    "1. Role Prompting  2. Step Back  3. Guided Chain of Thought",
    "4. Instruction & Separation  5. Output Format with Examples",
    "6. Worked Reasoning in Examples  7. Conclusion",
    "### Output Format",
    "1. Role Prompting: X/5 - Strength: ... - Improvement: ... - Rationale: ...",
    "(continue for the remaining six criteria)",
    "Total Score: X/35",
    "Refinement Summary (7-10 actionable suggestions): [Suggestion 1] ...",
], caption="Figure 2. Structured meta-prompt for evaluating a candidate prompt: a task overview, "
           "evaluation instructions, the seven-criteria rubric, and a standardized output format for "
           "consistent scoring and improvement suggestions.", full=True)

P("The refinement stage builds on this critique. A second meta-prompt (Figure 3) directs the LLM to "
  "revise the original prompt by systematically incorporating the evaluation's suggestions, with the "
  "evaluation itself preserved in the conversation history to keep the revision faithful to the rubric "
  "rather than an unconstrained rewrite.")

CODE([
    "Refine the prompt by applying all suggestions from the evaluation.",
    "Make sure to wrap the refined prompt with <START> and <END>",
], caption="Figure 3. Meta-prompt used to refine a candidate prompt by applying all evaluation "
           "feedback.", full=False)

P("Evaluation and refinement together form a feedback-driven process for improving prompts, but "
  "refinement alone does not guarantee performance gains; MPIR therefore adds a validation stage to "
  "confirm that improvements translate into measurable benefits.")

H2("3.5. Prompt Effectiveness Validation")
P("The validation stage ensures that rubric-guided refinements yield measurable performance gains. "
  "Because LLMs are sensitive to subtle phrasing variations, a rubric-aligned rewrite is not guaranteed "
  "to perform better; rather than treating this sensitivity as a limitation, MPIR treats it as a "
  "mechanism for iterative improvement by testing each refined candidate on a held-out subset of the "
  "target dataset and comparing model outputs against ground truth. This evaluate-refine-validate cycle "
  "repeats for N rounds, producing successive candidates with corresponding accuracy scores, and MPIR "
  "retains the prompt that achieves the highest accuracy—so the final prompt reflects both heuristic "
  "soundness and empirically verified effectiveness.")

# ===========================================================================
H1("4. Implementation")

H2("4.1. Datasets")
P("This paper uses Big-Bench Hard (BBH) as the evaluation benchmark ‹27›. BBH comprises 23 "
  "tasks (27 subtasks) spanning algorithmic and multistep arithmetic reasoning, natural language "
  "understanding, world knowledge, and multilingual reasoning, with about 6,511 examples in total "
  "(roughly 250 per task). Its tasks are instruction-following in nature and were selected specifically "
  "for their difficulty, making BBH well suited for measuring gains from prompt optimization.")

H2("4.2. Workflow")
P("The workflow has two stages (Figures 4 and 5). Stage 1 uses PromptWizard as the base APO to produce "
  "an initial optimized prompt together with few-shot examples and reasoning chains. Stage 2 applies "
  "MPIR to that optimized prompt, running N rounds of rubric-based evaluation, refinement, and validation "
  "to produce the final optimized prompt.")

FIG("PromptWizard.png",
    "Stage 1: workflow of the PromptWizard base APO, adapted from ‹23›. It performs iterative "
    "refinement of prompt instructions, diverse example selection, sequential optimization, and "
    "self-generated reasoning and validation to produce an optimized prompt with few-shot examples and "
    "reasoning chains.", full=False)

FIG("MPIR.png",
    "Stage 2: workflow of Meta-Prompted Instruction Refinement (MPIR). MPIR builds on the PromptWizard "
    "APO output and performs iterative evaluation, refinement, and validation over N rounds to further "
    "improve prompt quality.", full=False)

H3("4.2.1. Stage 1: PromptWizard")
P("PromptWizard is adopted as the base APO because it achieves strong performance across diverse "
  "benchmarks while remaining cost- and time-efficient ‹23›. It is primarily an APO framework "
  "—its core objective is iterative, automated prompt refinement and validation—but it also "
  "incorporates meta-prompting and heuristic-guided elements through self-generated reasoning chains, "
  "critique-based refinement, and heuristic-inspired prompt construction. Its modular design lets MPIR "
  "be layered on top without altering the underlying optimization process. Our implementation omits the "
  "synthesis components of the original framework ‹23›, since BBH's difficulty prevents the "
  "LLM from reliably generating them (for example, it often fails to produce a coherent expert persona). "
  "Algorithm 2 details the resulting procedure.")

ALGO(
    "Algorithm 2. PromptWizard Algorithm (adapted from ‹23›)",
    require=("L: large language model; D: problem description; S = {(qi, ai)}, i=1..n: training samples; "
              "T: thinking styles; k: few-shot count; N1: max_seq_iter; N2: mutate_refine_rounds; "
              "Pbase: base instruction (zero-shot CoT)"),
    ensure="Optimized prompt P̂opt and validated few-shot examples {(qfi, afi)}, i=1..k",
    steps=[
        "Initialize: P ← Pbase",
        "P̂ ← RefineInstructions(L, D, S, T, N2)",
        "Ediverse ← DiverseExampleSelection(L, D, S, P̂)",
        "P̂opt, Eopt ← SequentialOptimization(L, P̂, Ediverse, N1)",
        "Eopt,r ← ReasoningComponent(Eopt)                 ▷ Generate reasoning chains",
        "{(qfi, afi)} ← ValidateComponent(Eopt,r)              ▷ Validate examples",
    ],
    ret="P̂opt, {(qfi, afi)}, i=1..k",
)

H3("4.2.2. Stage 2: Meta-Prompted Instruction Refinement")
P("MPIR extends PromptWizard by applying the rubric-driven evaluation-refinement-validation loop "
  "(Algorithm 1) to its optimized prompt. In each round, the PromptWizard-generated prompt is evaluated "
  "against the seven-criteria rubric, refined using the meta-prompt feedback, and validated on a "
  "held-out set using accuracy; after N rounds, the best-performing prompt is selected.")

H2("4.3. Baselines")
P("Because the goal is to assess whether meta-prompting can enhance existing APO methods, PromptWizard "
  "is the primary baseline; additional reference points provide further context.")

H3("4.3.1. Main baseline")
BULLETS([
    ("APO baseline (PromptWizard).",
     "PromptWizard serves as a strong automated baseline without human-crafted heuristics; each final "
     "prompt includes three automatically generated in-context examples."),
    ("Free rewrite baseline.",
     "To isolate the contribution of MPIR's rubric from the general rewriting capability of GPT-4o, "
     "GPT-4o freely rewrites PromptWizard-generated prompts without rubric-guided evaluation or "
     "refinement, and the rewritten prompts are evaluated under the same BBH conditions as MPIR."),
])

H3("4.3.2. Extended APO baselines")
P("To further evaluate generalization, MPIR is also applied to APE ‹19› and ProTeGi "
  "‹20›. Since neither method generates few-shot reasoning examples the way PromptWizard does, "
  "both use the same three chain-of-thought exemplars written by the BBH benchmark authors "
  "‹29›; the resulting prompts are refined with MPIR and evaluated under identical conditions.")

H3("4.3.3. Reference points")
BULLETS([
    ("Manual prompting (zero-shot CoT).",
     "A manual reference point using only the task description plus “Let's think step by step” "
     "‹8›."),
    ("Expert-crafted prompting (few-shot CoT).",
     "An expert-augmented ceiling using the task description, “Let's think step by step,” and "
     "three chain-of-thought exemplars written by the BBH benchmark authors ‹27›; unlike MPIR, "
     "this approach is not scalable and requires substantial human effort."),
])

H2("4.4. Evaluation Metric")
P("Performance is measured with accuracy: the proportion of model outputs matching ground-truth labels "
  "on the full test set. For Ntest examples with inputs xj, ground-truth labels yj, and predictions "
  "ŷj, accuracy is (1/Ntest) Σ 1[ŷj = yj], the average indicator of a correct prediction.")

H2("4.5. Implementation Details")
P("GPT-3.5-turbo ‹48› serves as the target model for all benchmark tasks, both as the backbone "
  "of the PromptWizard baseline and for MPIR's validation stage, ensuring a consistent evaluation "
  "environment. MPIR's meta-prompting stages—evaluation and refinement—use the more capable "
  "GPT-4o ‹49› to provide expert-level feedback.")

P("PromptWizard randomly selects 25 training examples for prompt optimization and in-context example "
  "construction, with the remainder reserved for testing; MPIR uses the same 25 examples during "
  "validation to keep the comparison fair, with the remaining data again serving as the test set. "
  "Refinement runs for 7 rounds, balancing improvement opportunity against computational cost: each round "
  "issues 2 meta-prompting calls (evaluation and refinement, on GPT-4o) plus one validation call per "
  "held-out example (on GPT-3.5-turbo), so MPIR adds on the order of 200 additional API calls per task on "
  "top of the underlying APO method's own optimization cost—a real overhead that should be weighed "
  "against its per-task accuracy gains in latency- or cost-sensitive deployments. All "
  "models are accessed via API with temperature fixed at 0. Full PromptWizard hyperparameters are "
  "reported in Table 1 to support reproducibility. The implementation of MPIR is publicly available at "
  "https://github.com/millennials92/meta_prompted_instruction_refinement.")

TABLE("Table 1. Hyperparameter settings of PromptWizard.",
      header=["Hyperparameter", "Description", "Default"],
      rows=[
          ["mutate_refine_rounds", "Rounds of MutateComponent followed by refinement over the best prompt generated so far.", "3"],
          ["mutate_rounds", "Number of times MutateComponent is called.", "3"],
          ["style_variation", "Variations MutateComponent generates per call (one per thinking style).", "5"],
          ["min_example_correct_count", "Minimum questions ScoringComponent must answer correctly to qualify a prompt.", "3"],
          ["max_example_count", "Maximum attempts/questions given to ScoringComponent.", "6"],
          ["max_seq_iter", "Rounds of calls to CritiqueComponent.", "3"],
          ["few_shot_count", "Total few-shot examples included in the prompt.", "3"],
      ], full=True, colw=[0.28, 0.57, 0.15])

# ===========================================================================
H1("5. Results and Analysis")

H2("5.1. Results")
P("Table 2 reports accuracy on the full test sets of the 23 BBH tasks (Section 4.1).")

TABLE("Table 2. Accuracy (%) across 23 BBH tasks.",
      header=["Task", "Manual", "PromptWizard", "Free Rewrite", "MPIR", "Expert-crafted"],
      rows=[
          ["hyperbaton", "74.6", "62.2", "68.44", "84.0", "84.4"],
          ["disambiguation_qa", "55.1", "61.7", "44.44", "62.2", "69.3"],
          ["causal_judgement", "53.7", "59.3", "59.88", "56.8", "59.9"],
          ["date_understanding", "55.6", "71.1", "73.33", "74.2", "80.0"],
          ["penguins_in_a_table", "45.5", "75.0", "75.21", "82.6", "74.4"],
          ["boolean_expression", "81.8", "92.0", "89.78", "94.2", "91.6"],
          ["object_counting", "59.1", "91.5", "85.78", "92.4", "88.0"],
          ["word_sorting", "82.6", "80.0", "81.78", "81.3", "65.3"],
          ["logical_deduction", "45.3", "44.9", "39.11", "50.8", "59.4"],
          ["salient_translation_error_detection", "44.0", "50.7", "49.78", "52.0", "54.7"],
          ["geometric_shapes", "29.3", "57.8", "52.89", "49.3", "64.4"],
          ["snarks", "45.0", "63.3", "59.48", "65.4", "72.5"],
          ["temporal_sequences", "33.3", "44.4", "44.44", "38.2", "17.3"],
          ["web_of_lies", "47.5", "52.4", "52.89", "52.9", "80.9"],
          ["navigate", "49.3", "65.3", "74.67", "68.0", "93.8"],
          ["reasoning_about_colored_objects", "54.2", "56.0", "53.33", "68.4", "83.1"],
          ["sports_understanding", "78.2", "79.6", "89.33", "86.2", "93.8"],
          ["multistep_arithmetic_two", "33.3", "51.1", "52.00", "51.5", "84.9"],
          ["ruin_names", "54.7", "66.2", "56.00", "72.0", "69.3"],
          ["movie_recommendation", "66.2", "73.3", "48.00", "66.7", "79.1"],
          ["formal_fallacies", "53.8", "53.3", "31.11", "53.3", "51.6"],
          ["dyck_languages", "1.7", "14.2", "14.67", "12.9", "16.9"],
          ["tracking_shuffled_objects", "22.4", "69.8", "18.07", "65.2", "63.8"],
          ["Average", "50.7", "62.39", "57.15", "64.37", "69.5"],
      ], full=True, colw=[0.30, 0.14, 0.16, 0.14, 0.12, 0.14], note="bold_last_row")

P("Table 3 summarizes accuracy before and after MPIR refinement across two further APO methods, "
  "Iterative APE and ProTeGi. To keep the paper within its page limit, we report the average over all 23 "
  "tasks together with the two tasks where MPIR moves accuracy in the same direction under all three "
  "methods most strongly—hyperbaton and ruin_names (consistently positive)—and the two where it moves "
  "in the same direction most strongly the other way—geometric_shapes and tracking_shuffled_objects "
  "(consistently negative); full per-task results for all three APO methods are provided in the project "
  "repository (Section 4.5).")

TABLE("Table 3. Accuracy (%) before and after MPIR refinement, for three APO methods (average over all "
      "23 BBH tasks, plus the tasks with the most consistent gains and regressions across all three "
      "methods).",
      header=["Task", "APE (before)", "ProTeGi (before)", "PromptWizard (before)",
              "APE (after)", "ProTeGi (after)", "PromptWizard (after)"],
      rows=[
          ["hyperbaton", "82.67", "80.89", "62.2", "85.78", "88.89", "84.0"],
          ["ruin_names", "58.67", "58.67", "66.2", "68.44", "77.78", "72.0"],
          ["geometric_shapes", "62.67", "61.78", "57.8", "61.33", "60.89", "49.3"],
          ["tracking_shuffled_objects", "62.96", "61.63", "69.8", "58.22", "60.89", "65.5"],
          ["Average (23 tasks)", "70.16", "72.81", "62.39", "74.02", "74.26", "64.37"],
      ], full=True, colw=None, note="bold_last_row")

P("Beyond aggregate performance, concrete examples illustrate where MPIR improves on baseline APO "
  "methods. In one penguins_in_a_table case (Figure 6), PromptWizard incorrectly counts two penguins "
  "younger than 8 years old, mistakenly including an 8-year-old penguin, while MPIR's refinement produces "
  "a clearer reasoning chain that correctly counts only one qualifying penguin—illustrating how "
  "rubric-driven refinement can eliminate subtle reasoning errors by enforcing stricter interpretation of "
  "task rules.")

FIG("case.png",
    "Figure 6. Case study comparing PromptWizard and MPIR on the penguins_in_a_table task. PromptWizard "
    "incorrectly identifies two penguins younger than eight years old, while MPIR correctly counts only "
    "one.", full=True)

H2("5.2. Analysis")

H3("5.2.1. Overall Performance Improvements of MPIR")
P("Table 2 shows a directionally positive but not statistically significant improvement over "
  "PromptWizard: MPIR reaches an average accuracy of 64.37%, versus 62.39% for PromptWizard, a difference "
  "of 1.97 percentage points. A bootstrap resampling analysis across BBH tasks (10,000 iterations) gives "
  "a 95% confidence interval of [−0.46, 4.70] for this difference; because the interval includes zero, "
  "this task-level average improvement cannot be considered statistically significant at conventional "
  "thresholds, and we report it as such rather than as an established effect. The more robust evidence for "
  "MPIR comes from its consistency across individual tasks—it outperforms PromptWizard on 16 of 23 "
  "tasks, with several gains of 10-20 points—and from the free rewrite baseline, which reaches only "
  "57.15% on average, well below both PromptWizard and MPIR. Together, these results indicate that "
  "MPIR's per-task gains are attributable to its structured seven-criteria rubric rather than simply to "
  "GPT-4o's general rewriting ability, even though the pooled average improvement should be read as a "
  "promising trend rather than a confirmed effect; Section 6 revisits this as a limitation.")

H3("5.2.2. Evaluating MPIR Across Multiple APO Frameworks")
P("Table 3 shows that MPIR consistently improves average accuracy across APO frameworks: Iterative APE "
  "from 70.16% to 74.02%, ProTeGi from 72.81% to 74.26%, and PromptWizard from 62.39% to 64.37%. Although "
  "the magnitude of improvement varies across methods and tasks, the consistently positive trend supports "
  "MPIR functioning as a refinement layer across different APO frameworks rather than one tailored to "
  "PromptWizard-specific prompt structures.")

H3("5.2.3. Cross-Model Generalization")
P("To test MPIR under a different model family, we repeated the experiment with Gemini 3.5 Flash-Lite "
  "‹50› as both the target model and the meta-prompting model (Table 4). The baseline already "
  "achieves a high average accuracy of 92%, leaving limited room for improvement, and MPIR maintains the "
  "same rounded average after refinement. The most noticeable gains occur where baseline accuracy was "
  "around 70%: causal_judgement improves from 70% to 75%, disambiguation_qa from 77% to 80%, and "
  "dyck_languages from 72% to 75%. Tasks already near-perfect before refinement, such as "
  "sports_understanding and multistep_arithmetic_two, show little room for further gains and occasionally "
  "a slight decline—suggesting MPIR is most effective when the baseline prompt still has meaningful "
  "reasoning weaknesses to correct.")

TABLE("Table 4. Cross-model evaluation of MPIR on Gemini 3.5 Flash-Lite (average over all 23 BBH tasks, "
      "plus the tasks with the largest movement); full per-task results are in the project repository.",
      header=["Task", "PromptWizard (%)", "MPIR (%)", "Change (%)"],
      rows=[
          ["causal_judgement", "70", "75", "+5"],
          ["disambiguation_qa", "77", "80", "+3"],
          ["dyck_languages", "72", "75", "+3"],
          ["sports_understanding", "91", "86", "−5"],
          ["multistep_arithmetic_two", "97", "94", "−4"],
          ["Average (23 tasks)", "92", "92", "0"],
      ], full=False, note="bold_last_row")

H3("5.2.4. Clarity, Structure, and Task Difficulty")
P("MPIR also improves interpretability by embedding human-inspired structure, clarifying reasoning "
  "context, and filtering out details that could distract the model. In hyperbaton, PromptWizard's "
  "baseline prompt mixed background explanation with task directives in a single block, whereas MPIR "
  "separated it into a context section defining the role and purpose and an instruction section with "
  "explicit, numbered steps, helping the model distinguish framing from required actions. In web_of_lies, "
  "PromptWizard's vague instruction (“How can we systematically analyze the statements to determine "
  "the truthfulness of each individual in the scenario?”) became, after MPIR, an explicit role and "
  "goal statement (“You are a logical analyst tasked with evaluating the truthfulness of statements "
  "made by individuals in a given scenario...”), and similar edits elsewhere removed distracting, "
  "task-irrelevant instructions that had been reducing output quality.")

P("Across Tables 2-4, MPIR's strongest and most consistent gains occur on tasks governed by explicit "
  "structural or rule-based reasoning—hyperbaton, object_counting, boolean_expression, and "
  "temporal_sequences all improve repeatedly across APO frameworks and model families (e.g., hyperbaton "
  "rises from 62.2% to 84.0% under PromptWizard). This is consistent with prior evidence that CoT "
  "prompting helps models decompose complex problems into intermediate steps ‹7› and that "
  "step-back reasoning helps LLMs derive first principles before solving a task ‹13›. MPIR is "
  "comparatively less effective on tasks requiring precise symbolic manipulation, spatial abstraction, or "
  "extended sequential reasoning—geometric_shapes, tracking_shuffled_objects, and logical_deduction "
  "show smaller, less stable gains or occasional regressions (e.g., geometric_shapes falls from 57.8% to "
  "49.3% under PromptWizard). This pattern matches prior findings that coherent, relevant reasoning steps "
  "matter more than reasoning length ‹10›, and that LLMs remain highly sensitive to subtle "
  "prompt-phrasing changes ‹33›, which can disrupt the precise consistency these tasks demand.")

H3("5.2.5. Remaining Gap to Expert-Crafted Prompting")
P("Despite strong gains over automated baselines, MPIR still falls short of expert-crafted prompting on "
  "average (64.4% versus 69.5%). The gap concentrates in a small set of tasks—navigate, "
  "reasoning_about_colored_objects, web_of_lies, and multistep_arithmetic_two (Table 5)—where expert "
  "examples provide richer reasoning traces and more explicit state tracking. Because MPIR's examples are "
  "inherited from the PromptWizard baseline rather than newly generated, they reflect the same "
  "limitations present in PromptWizard's own outputs: MPIR's refinements are structurally consistent but "
  "tend to oversimplify example reasoning relative to expert-written traces.")

TABLE("Table 5. Tasks where MPIR underperforms expert-crafted prompting.",
      header=["Task", "MPIR (%)", "Expert (%)"],
      rows=[
          ["navigate", "68.0", "93.8"],
          ["reasoning_about_colored_objects", "68.4", "83.1"],
          ["web_of_lies", "52.9", "80.9"],
          ["multistep_arithmetic_two", "51.5", "84.9"],
      ], full=False)

H2("5.3. Ablation Studies")

H3("5.3.1. Which Rubric Criteria Matter Most")
P("To assess each rubric criterion's contribution, we ran an ablation on five representative BBH tasks, "
  "removing one criterion at a time (Table 6). Removing any single criterion reduces average accuracy "
  "from the full rubric's 80.0% to between 66.0% and 75.0%, so every criterion plays a meaningful role, "
  "though to different degrees. Four criteria prove particularly critical—Instruction & Separation "
  "(C4), Role Prompting (C1), Guided Chain of Thought (C3), and Step Back (C2)—each reducing average "
  "accuracy by over 11 points when removed, underscoring their role in structuring reasoning and reducing "
  "ambiguity. Removing Worked Reasoning in Examples (C6) produces a moderate but consistent drop, "
  "suggesting that step-by-step demonstrations reinforce structured reasoning patterns, whereas removing "
  "Output Format Specification (C5) or Conclusion (C7) causes only minor drops, indicating these criteria "
  "mainly enhance clarity rather than drive task accuracy. Overall, heuristics that organize reasoning and "
  "contextual framing appear to contribute most to refinement performance on these benchmark tasks.")

TABLE("Table 6. Effect of removing individual rubric criteria across five BBH tasks (accuracy, %). "
      "C1=Role Prompting, C2=Step Back, C3=Guided CoT, C4=Instruction & Separation, C5=Output Format, "
      "C6=Worked Reasoning, C7=Conclusion.",
      header=["Task", "ALL", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
      rows=[
          ["hyperbaton", "84.0", "56.0", "83.1", "67.6", "56.4", "59.1", "59.6", "79.1"],
          ["penguins_in_a_table", "82.6", "76.0", "67.8", "70.2", "45.5", "73.6", "79.3", "75.2"],
          ["ruin_names", "72.0", "52.4", "51.1", "50.2", "54.7", "67.1", "60.4", "57.3"],
          ["object_counting", "92.4", "90.2", "91.6", "88.4", "91.6", "92.0", "88.4", "92.0"],
          ["reasoning_about_colored_objects", "68.4", "59.1", "52.0", "68.9", "79.6", "62.7", "60.9", "69.8"],
          ["Average", "80.0", "67.0", "69.0", "69.0", "66.0", "71.0", "70.0", "75.0"],
      ], full=True, note="bold_last_row")

H3("5.3.2. Heuristic Rubric vs. Generic Rubric")
P("To test whether the rubric's specificity matters, we compared the full seven-criteria rubric against "
  "a generic rubric using broad criteria such as clarity, structure, and effectiveness, on the same five "
  "tasks, holding refinement and validation fixed (Table 7). The full rubric outperforms the generic one "
  "on four of five tasks, with average accuracy dropping from 79.9% to 72.2% under the generic rubric—"
  "an 8-point gap confirming that the seven-criteria rubric's specificity, not just the act of evaluation "
  "and refinement, drives MPIR's effectiveness. The heuristic rubric enforces step-by-step reasoning, "
  "context separation, and structured outputs, whereas the generic rubric's vaguer feedback often fails "
  "to correct task-specific weaknesses.")

H3("5.3.3. Importance of Prompt Effectiveness Validation")
P("Finally, we tested whether the validation stage itself is necessary by comparing the full framework "
  "(seven evaluation-refinement-validation cycles) against a validation-ablated variant that selects a "
  "prompt after a single evaluation-refinement cycle with no empirical testing, again on five "
  "representative tasks (Table 7). Removing validation drops average accuracy from 79.9% to 61.9%, an "
  "18-point decline showing that rubric alignment alone does not guarantee performance: an "
  "unvalidated prompt may look well structured yet perform worse empirically. Validation anchors MPIR's "
  "refinements in measured accuracy, ensuring gains are both real and task-oriented rather than purely "
  "heuristic.")

TABLE("Table 7. Effect of the generic rubric and of removing prompt-effectiveness validation, each "
      "measured against the same full-MPIR baseline, across five BBH tasks (accuracy, %).",
      header=["Task", "MPIR (full)", "Generic rubric", "No validation"],
      rows=[
          ["hyperbaton", "84.0", "66.2", "37.8"],
          ["penguins_in_a_table", "82.6", "78.5", "67.8"],
          ["ruin_names", "72.0", "66.2", "52.0"],
          ["object_counting", "92.4", "92.9", "91.6"],
          ["reasoning_about_colored_objects", "68.4", "57.3", "60.4"],
          ["Average", "79.9", "72.2", "61.9"],
      ], full=False, note="bold_last_row")

# ===========================================================================
H1("6. Conclusion")

P("This paper introduced Meta-Prompted Instruction Refinement (MPIR), a framework that integrates manual "
  "prompting heuristics into automatic prompt optimization through a structured cycle of evaluation, "
  "refinement, and validation. It asks whether human-inspired prompting heuristics can be systematically "
  "integrated into APO systems to improve prompt quality while preserving the scalability of automated "
  "optimization. Experiments on Big-Bench Hard show that rubric-guided meta-prompt refinement can improve "
  "APO-generated prompts across multiple tasks and APO frameworks, and that MPIR is particularly "
  "effective for structured, rule-based reasoning tasks, where heuristic-guided refinement yields clearer "
  "and more reliable reasoning.")

P("This study also has limitations. Most importantly, the average gain over the PromptWizard baseline is "
  "modest and, at the task-population level, not statistically significant: its 95% bootstrap confidence "
  "interval spans zero (Section 5.2.1). We therefore treat that pooled result as a promising trend rather "
  "than a confirmed effect, and rely more heavily on the per-task win rate (16 of 23 tasks) and on the "
  "consistent, replicated pattern across three APO frameworks and two model families as the stronger "
  "evidence for MPIR's contribution. Relatedly, all reported results come from a single run per condition "
  "at temperature 0 rather than repeated trials with varying seeds, so our confidence intervals capture "
  "variance across BBH tasks but not run-to-run variance in the optimization process itself; repeated-trial "
  "estimates of that variance are an important direction for follow-up work. To keep the comparison fair, "
  "MPIR also reuses PromptWizard's optimization examples during validation, which may introduce evaluation "
  "bias, and the seven-criteria rubric was developed in close proximity to the BBH tasks themselves, which "
  "may introduce construct dependence and limit generalizability to other benchmark families. The quality "
  "of MPIR's refined prompts remains partially bounded by the quality of the prompts and examples produced "
  "by the underlying APO framework. Finally, because MPIR depends on versioned commercial APIs "
  "(GPT-3.5-turbo, GPT-4o, Gemini 3.5 Flash-Lite), exact reproducibility is subject to provider-side model "
  "updates and deprecations outside the authors' control.")

P("Future work could extend MPIR along several directions: broader evaluation across additional datasets "
  "and reasoning domains; adaptive, task-specific rubrics; independent construct validation, by deriving "
  "a rubric on one benchmark family and evaluating it on a distinct one; and integration with "
  "complementary approaches such as symbolic reasoning modules, retrieval augmentation, or automated "
  "methods for learning prompting heuristics directly from empirical performance data.")

# ===========================================================================
H1("Acknowledgement")
P("The authors thank the reviewers for their constructive comments on earlier drafts of this manuscript.")

H1("Funding Support")
P("This research received no specific grant from any funding agency in the public, commercial, or "
  "not-for-profit sectors.")

H1("Ethical Statement")
P("This study did not involve human participants, human data, or animal subjects, and did not require "
  "ethical approval.")

H1("Conflicts of Interest")
P("The authors declare that they have no conflicts of interest.")

H1("Data Availability Statement")
P("The datasets used in this study are drawn from the publicly available Big-Bench Hard benchmark. Code, "
  "prompts, and experimental results are openly available at "
  "https://github.com/millennials92/meta_prompted_instruction_refinement.")

H1("Author Contribution Statement")
P("Linh Nguyen: Conceptualization, Methodology, Software, Investigation, Formal analysis, Data curation, "
  "Writing - original draft, Visualization. Quang-Vinh Dang: Conceptualization, Methodology, Supervision, "
  "Writing - review & editing, Project administration. Minh Ngoc Dinh: Methodology, Validation, Writing - "
  "review & editing. Thuy Nguyen: Investigation, Validation, Writing - review & editing.")

H1("Declaration of Generative AI and AI-Assisted Technologies in the Writing Process")
P("Generative AI tools were used solely to improve the grammar and readability of the text. All ideas, "
  "analyses, and conclusions are solely those of the authors. An AI-assisted coding tool was used to "
  "generate initial code snippets during development; all code was reviewed, verified, and adapted by "
  "the authors.")
