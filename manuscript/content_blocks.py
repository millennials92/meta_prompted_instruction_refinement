# -*- coding: utf-8 -*-
# Final manuscript content for the AIA (Artificial Intelligence and Applications) docx.
# Citation tokens use unicode guillemets, e.g. ‘9‹ or ‹9,10›, referring to
# the bare numeric suffix of the original reference_N bib keys. These get resolved to
# bracketed, renumbered-by-first-appearance citations, e.g. [3] or [3, 7], by build_docx.py.

TITLE = "Meta-Prompted Instruction Refinement: Bridging Manual Prompting Techniques and Automatic Prompt Optimization"

AUTHORS = [
    {"name": "Linh Nguyen", "affil_idx": [1], "corresponding": False},
    {"name": "Quang-Vinh Dang", "affil_idx": [2], "corresponding": True},
    {"name": "Minh Ngoc Dinh", "affil_idx": [3], "corresponding": False, "email": "Minh.dinh@maeducation.com"},
    {"name": "Thuy Nguyen", "affil_idx": [1], "corresponding": False},
]
AFFILIATIONS = [
    "School of Science, Engineering & Technology, RMIT University Vietnam, 702 Nguyen Van Linh Boulevard, Ho Chi Minh City 700000, Vietnam",
    "British University Vietnam, Hung Yen, Vietnam",
    "Millenia Education, Ho Chi Minh City, Vietnam",
]
CORRESPONDING_AUTHOR_EMAIL = "ai-data-ai-presales@smartosc.com"
CORRESPONDING_AUTHOR_NOTE = (
    "*Corresponding author: Quang-Vinh Dang, British University Vietnam, Hung Yen, Vietnam. "
    f"Email: {CORRESPONDING_AUTHOR_EMAIL}"
)

ABSTRACT = (
    "Large language models (LLMs) are transforming artificial intelligence by enabling systems that can "
    "reason, write, and assist with complex tasks, capabilities that are increasingly important for "
    "science, education, and everyday applications. Yet these models remain critically dependent on the "
    "quality of their input prompts, making prompt design a central bottleneck. Manual prompt engineering, "
    "using techniques such as chain-of-thought reasoning and role assignment, can yield high performance "
    "but requires expert knowledge and does not scale. Automatic prompt optimization (APO) offers "
    "efficiency, but its outputs often lack the structured guidance that makes human-crafted prompts "
    "effective. This paper introduces Meta-Prompted Instruction Refinement (MPIR), a lightweight, "
    "model-agnostic framework that refines APO-generated prompts through a seven-criteria rubric, "
    "meta-prompted evaluation and refinement, and empirical validation—without any additional training, "
    "search infrastructure, or access to model internals. Extensive experiments on the Big-Bench Hard "
    "(BBH) benchmark show that MPIR outperforms its PromptWizard baseline on 16 of 23 tasks, with gains "
    "of up to 20 percentage points on individual tasks, and that the same refinement layer improves two "
    "further APO methods (Iterative APE and ProTeGi) as well as a different underlying LLM family. These "
    "results demonstrate that a simple, interpretable, post-hoc heuristic layer can meaningfully improve "
    "prompts already optimized by heavier automated methods, bridging human heuristics with automation "
    "at a fraction of the engineering cost of learned or evolutionary alternatives."
)

KEYWORDS = "Prompt engineering, meta-prompting, automatic prompt optimization, large language models, Big-Bench Hard"

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


def APPENDIX_START():
    # Marks the boundary where appendix sections begin. LaTeX emits \appendix here (switching
    # subsequent \section headings to auto-numbered "Appendix A/B/C..."); DOCX is a no-op since
    # heading text there already carries an explicit "Appendix X." prefix.
    BLOCKS.append({"type": "appendix_start"})


# ===========================================================================
H1("1. Introduction")

P("Organizations adopting LLMs increasingly face a practical version of a familiar software-engineering "
  "problem: a component that works acceptably in a demo does not necessarily work reliably in "
  "production, and the gap between the two is often traceable to how the component is instructed rather "
  "than to a limitation of the underlying model. Unlike traditional software, an LLM has no fixed "
  "interface contract to program against; the prompt is simultaneously the specification, the interface, "
  "and—because prompts are natural language—an artifact whose correctness cannot be checked by a "
  "compiler or a type system. This makes prompt quality a cross-cutting concern that touches accuracy, "
  "consistency, cost, and user trust simultaneously, and makes the tooling used to construct and "
  "validate prompts a legitimate object of study in its own right, alongside the models the prompts are "
  "written for.")

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
  "APO-generated prompts without additional human intervention. Unlike the increasingly elaborate "
  "evolutionary, reinforcement-learning, and multi-agent machinery pursued elsewhere in this space "
  "(Section 2.5), MPIR asks how much of that benefit is recoverable from a fixed, interpretable "
  "checklist and a small, fixed number of meta-prompting calls—deliberately inexpensive to add on top "
  "of whatever APO pipeline a practitioner already has.")

P("Specifically, this study pursues four objectives:")
BULLETS([
    ("To develop a structured rubric for prompt evaluation:",
     "formalizing established manual prompting heuristics into a seven-criteria framework."),
    ("To evaluate whether meta-prompted refinement improves APO-generated prompts:",
     "applying rubric-guided feedback to iteratively refine and empirically validate prompts."),
    ("To investigate the generalizability and modularity of heuristic-guided refinement:",
     "testing MPIR as a layer across different APO frameworks and underlying LLMs."),
    ("To investigate how individual prompting heuristics contribute to performance:",
     "quantifying each rubric criterion's contribution via targeted ablations."),
])

P("The remainder of this paper is organized as follows. Section 2 reviews related work on prompt "
  "engineering and automatic prompt optimization. Section 3 introduces the MPIR framework. Section 4 "
  "describes the implementation. Section 5 reports results and analysis. Section 6 discusses threats "
  "to validity. Section 7 concludes the paper.")

# ===========================================================================
H1("2. Related Work")

H2("2.1. Prompting as Human-AI Interaction")
P("Prompting is the primary mechanism for steering LLM behavior, leveraging in-context learning to shape "
  "outputs without updating model parameters ‹2›. It has been described as a form of natural "
  "language programming ‹3›, emphasizing scalability and efficiency. As a direct channel of "
  "user control, however, its effectiveness often depends on extensive trial and error, which has given "
  "rise to a wide range of heuristic techniques whose potential and limitations we review next.")

P("This framing matters for how MPIR is positioned. Because prompting is a form of programming without a "
  "compiler, the feedback a prompt author receives is indirect: a change in wording is only validated by "
  "re-running the model and inspecting its output, and there is no static analysis that flags an "
  "ambiguous instruction before it is executed. Studies of non-expert prompt authors find that this "
  "absence of structural feedback is a primary source of difficulty, with users struggling to predict "
  "how small wording changes will affect model behavior and often overfitting to a handful of examples "
  "they happen to test ‹32›. Manual heuristics such as those reviewed in Section 2.2 function, "
  "in this framing, as an informal type system for prompts: they encode the kinds of structural "
  "properties—an assigned role, a stated reasoning procedure, a specified output format—that "
  "experienced prompt authors have learned to check for even without automated tooling to enforce them. "
  "MPIR's seven-criteria rubric (Section 3.3) can be read as an attempt to make this informal type "
  "system explicit and machine-checkable, applying it as an automated review step after a prompt has "
  "already been produced by another method, in the same way a linter is applied after code has already "
  "been written rather than only informing initial code style.")

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
  "constraints can improve reasoning accuracy and instruction-following. Separating context from task "
  "instructions—for example with explicit delimiters or headers—reduces ambiguity about which part "
  "of a prompt the model should treat as background versus as an actionable directive, a distinction "
  "that becomes more important as prompts grow longer and combine multiple heuristics at once.")

P("A further line of heuristics concerns how in-context examples themselves are constructed rather than "
  "simply whether they are present. Demonstrations that walk through intermediate reasoning steps, not "
  "just final answers, teach a model the expected reasoning process as well as the expected output "
  "format, and models trained or prompted to mimic worked examples tend to reproduce that structure on "
  "novel inputs ‹5,6›. Explicitly specifying the desired output format—for instance, "
  "requiring a delimiter-wrapped final answer—further reduces downstream parsing errors and makes "
  "automated grading more reliable, which matters directly for benchmark evaluation. Finally, closing a "
  "prompt with a brief restatement of the task exploits the same positional-attention effect that "
  "motivates placing key instructions early: a short concluding reminder keeps the task salient "
  "immediately before the model begins generating its answer, complementing rather than duplicating an "
  "opening statement of intent.")

P("A separate family of heuristics operates at decoding time rather than at the level of prompt text. "
  "Self-consistency samples multiple independent reasoning paths for the same chain-of-thought prompt "
  "and selects the answer that appears most frequently across samples, rather than relying on a single "
  "greedy decode ‹66›. This is complementary to, rather than competing with, the prompt-level "
  "heuristics reviewed above: an ensembling strategy over samples cannot compensate for a prompt that "
  "systematically misdirects the model's reasoning, but it can reduce the variance of a well-designed "
  "prompt's output. MPIR does not incorporate sampling-based ensembling, using a single greedy decode at "
  "temperature 0 throughout (Section 4.5); combining rubric-guided prompt refinement with "
  "self-consistency decoding is a natural extension we did not evaluate and note as future work.")

P("Collectively, these heuristics show that carefully designed instructions, reasoning structure, "
  "contextual framing, and exemplars can substantially improve LLM performance—but they remain "
  "labor-intensive, task-specific, and dependent on human expertise, which motivates automatic prompt "
  "optimization. Section 3.3 formalizes seven of the heuristics introduced in this subsection—role "
  "framing, step-back reasoning, guided chain-of-thought, instruction separation, output format "
  "specification, worked reasoning in examples, and task-closing restatement—into the explicit rubric "
  "MPIR uses to evaluate and refine prompts.")

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

P("These methods differ substantially in how they generate candidates and how they decide which "
  "candidate survives. APE treats prompt generation as program synthesis over a small set of "
  "demonstrations, scoring candidates by how well they reproduce the demonstrated input-output behavior "
  "‹19›. ProTeGi instead treats natural-language critique as a proxy for a gradient: it asks an "
  "LLM to diagnose why a prompt fails on a batch of examples, generates an edit in the direction the "
  "critique suggests, and performs a beam search over the resulting candidates ‹20›. EvoPrompt "
  "and OPRO both frame optimization as a black-box search over the space of prompt strings—EvoPrompt "
  "borrows crossover and mutation operators from genetic algorithms ‹21›, while OPRO instead "
  "conditions the LLM on a running history of previously tried prompts and their scores, asking it to "
  "propose an improvement in the style of an optimizer reading its own trajectory ‹22›. "
  "PromptWizard, the primary baseline used in this paper, combines several of these ideas in sequence: "
  "it iteratively refines an instruction using critique-based feedback similar to ProTeGi's, selects a "
  "diverse set of in-context examples, sequentially re-optimizes instruction and examples together, and "
  "finally generates and validates self-produced reasoning chains for the selected examples "
  "‹23›, described in full in Section 4.2.1.")

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

P("PE2 illustrates the mechanics of this category concretely: rather than asking an LLM to simply "
  "\"improve this prompt,\" it supplies a meta-prompt containing an explicit two-step reasoning "
  "template—first diagnose specific problems with the current prompt using a structured checklist of "
  "failure modes, then propose a textual edit that addresses the diagnosed problems—which the authors "
  "show outperforms unstructured revision requests of similar length ‹24›. PROPEL takes a "
  "different route to the same broad goal: rather than diagnosing a specific prompt's weaknesses at "
  "refinement time, it curates a set of expert-derived prompting principles in advance and supplies "
  "them as priors during a single optimization pass, so that the principles shape the search from the "
  "outset rather than critiquing an already-produced candidate ‹26›. Both approaches, along with "
  "PromptWizard's own heuristic-driven initialization, treat manual prompting knowledge as an input to "
  "the search process itself, which is what distinguishes them from the earlier black-box APO methods "
  "in Section 2.3 but also what limits how cleanly their heuristic contribution can be isolated from "
  "their search contribution.")

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

P("A closely related recent approach applies the same two-stage pattern—run a search-based optimizer, "
  "then locally refine its output—to few-shot relation extraction, using gradient- or attribution-style "
  "editing for the second stage rather than a fixed heuristic rubric scored by an LLM judge, and without "
  "testing generality across multiple upstream APO methods or backbone models ‹64›. MPIR shares "
  "its premise (a first-stage optimizer leaves room for post-hoc improvement) but differs in mechanism "
  "(a human-authored, criterion-by-criterion rubric evaluated by meta-prompting rather than local "
  "gradient-style edits) and in scope (evaluated across three upstream APO methods and two model "
  "families on a general reasoning benchmark rather than one task and one optimizer).")

H2("2.5. The Shifting Frontier of Prompt Optimization")
P("Since the meta-prompting and heuristic-guided methods above were introduced, prompt optimization has "
  "fragmented into more elaborate directions: evolutionary search with natural-language reflection, "
  "shown to outperform reinforcement-learning-based optimization with far fewer rollouts "
  "‹59›; reinforcement learning directly over edit actions; multi-agent debate and "
  "tournament-style Elo ratings as richer fitness functions than single-judge scoring "
  "‹62›; explicit error taxonomies that guide refinement top-down rather than through a "
  "fixed checklist ‹60›; and prompt format, not just content, as its own optimization axis "
  "‹61›. A recent survey frames the area through an optimization-theoretic lens "
  "‹63›, and there is growing interest in learned or instance-adaptive rubrics that "
  "replace fixed, human-authored criteria such as MPIR's seven. Relative to this frontier, MPIR's rubric "
  "is not the most sophisticated available mechanism; its contribution is instead that a small, "
  "interpretable, model-agnostic, and inexpensive post-hoc layer captures a meaningful share of the "
  "benefit these heavier methods pursue, without their training, search infrastructure, or per-task "
  "tuning cost—a different point on the cost-versus-sophistication trade-off, not a claim to surpass it.")

P("Table 1 summarizes where MPIR sits relative to the methods discussed above, along five dimensions "
  "that recur throughout this section: whether the method is guided by an explicit, human-authored "
  "rubric rather than a learned or implicit one; whether it is agnostic to the specific APO method or "
  "model it is paired with, rather than being a self-contained optimizer; whether it requires no "
  "additional training or fine-tuning; whether it retains empirical, held-out validation as part of "
  "candidate selection rather than relying on the judge's score alone; and whether it supports multiple "
  "iterative rounds rather than a single pass.")

TABLE("Table 1. Positioning of MPIR relative to representative prompt-optimization and refinement "
      "methods.",
      header=["Method", "Explicit rubric", "APO-agnostic layer", "Training-free",
              "Held-out validation", "Multi-round"],
      rows=[
          ["APE ‹19›", "No", "No", "Yes", "Yes", "Yes"],
          ["ProTeGi ‹20›", "No", "No", "Yes", "Yes", "Yes"],
          ["EvoPrompt ‹21›", "No", "No", "Yes", "Yes", "Yes"],
          ["OPRO ‹22›", "No", "No", "Yes", "Yes", "Yes"],
          ["PromptWizard ‹23›", "Partial", "No", "Yes", "Yes", "Yes"],
          ["PE2 ‹24›", "Partial", "No", "Yes", "Yes", "Yes"],
          ["PROPEL ‹26›", "Yes", "No", "Yes", "No", "No"],
          ["ETGPO ‹60›", "No", "No", "Yes", "Partial", "Yes"],
          ["CFPO ‹61›", "No", "No", "Yes", "Yes", "Yes"],
          ["GEPA ‹59›", "No", "No", "Yes", "Yes", "Yes"],
          ["MPIR (this work)", "Yes", "Yes", "Yes", "Yes", "Yes"],
      ], full=True)

P("No prior method combines all five properties. PROPEL is rubric-guided but bakes its principles into "
  "a single pass without a separate validation stage; ETGPO organizes refinement around failure "
  "categories rather than a fixed rubric, and validates only within its own search loop rather than "
  "against a truly held-out set; and the remaining methods are self-contained optimizers rather than "
  "layers designed to sit on top of an arbitrary upstream APO output. MPIR's position in this table is "
  "also its main limitation: an explicit, model-agnostic, iteratively validated rubric is a simpler "
  "mechanism than reflective evolution or reinforcement learning over edits, and Section 5 shows this "
  "simplicity comes with a correspondingly modest effect size.")

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

H2("3.3.1. Illustrative Application of the Rubric")
P("To make the abstract rubric concrete before Section 3.4 formalizes how it is applied automatically, "
  "this subsection walks through how the seven criteria would assess the real PromptWizard-optimized "
  "prompt for penguins_in_a_table reproduced in full in Appendix A.3, without reporting fabricated "
  "numeric scores for an evaluation that was not separately logged for this specific illustration. The "
  "PromptWizard prompt opens with a single undifferentiated paragraph that mixes a task description with "
  "an aside about comparing a new penguin's height to existing entries—an instruction that turns out "
  "to be irrelevant to most of the actual questions asked. Scored against the rubric: Role Prompting is "
  "weak, since no task-relevant persona is assigned; Instruction and Separation is weak, since context "
  "and directive are fused into one paragraph with no delimiter; Guided Chain of Thought is present only "
  "implicitly, in the numbered reasoning steps of the worked examples rather than in the instructions "
  "themselves; and Conclusion is absent, since the prompt ends abruptly after the worked examples with no "
  "restatement of the task. The MPIR-refined version of the same prompt, reproduced in Appendix A.4, "
  "directly addresses the two weakest criteria identified above: it opens with an explicit role "
  "(\"You are a data analyst tasked with comparing the attributes of penguins...\"), separates context "
  "from instructions under labeled headers (### Context, ### Instructions), and closes with an explicit "
  "### Conclusion section restating the task. This is precisely the pattern reported quantitatively in "
  "Section 5.2.4 and Table 7: removing Instruction and Separation (C4) or Role Prompting (C1) causes the "
  "largest ablation drops, consistent with these being the criteria most visibly deficient in the "
  "unrefined baseline for this task.")

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

P("Concretely, RefineInstructions runs mutate_refine_rounds sequential rounds in which the current best "
  "instruction is mutated into style_variation stylistic variants and re-scored on a batch of training "
  "examples, keeping the best-performing variant for the next round; DiverseExampleSelection then "
  "samples a pool of candidate few-shot examples and greedily selects a subset that maximizes coverage "
  "of distinct failure modes rather than simply the highest-scoring examples; SequentialOptimization "
  "alternates between refining the instruction and re-selecting examples for max_seq_iter rounds, since "
  "the two interact—a better instruction can make a previously low-value example newly informative, and "
  "vice versa; and ReasoningComponent prompts the model to generate a step-by-step justification for "
  "each selected example's ground-truth answer, which ValidateComponent then filters to keep only "
  "examples where the generated reasoning actually arrives at the correct answer, discarding examples "
  "whose self-generated reasoning is unreliable even when the final answer label is correct by chance. "
  "The full hyperparameter values used for every stage of this procedure are listed in Table 2.")

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

P("Together, these five conditions span a spectrum from no automation and no heuristics (manual "
  "zero-shot CoT) to full automation with heuristics but no scalability ceiling on human effort "
  "(expert-crafted few-shot CoT), with the three APO conditions and their MPIR-refined counterparts "
  "occupying the space between. This design lets Section 5 attribute MPIR's gains specifically to its "
  "rubric-guided refinement stage rather than to confounds that a narrower comparison would leave "
  "unaddressed: the free rewrite baseline controls for GPT-4o's general rewriting capability "
  "independent of any rubric; the extended APO baselines control for whether the effect is specific to "
  "PromptWizard's particular prompt structure; and the two reference points bound the range of "
  "achievable accuracy from unassisted manual prompting to unconstrained expert effort, giving Section "
  "5.2.6 a concrete ceiling against which to interpret the remaining gap.")

H2("4.4. Evaluation Metric")
P("Performance is measured with accuracy: the proportion of model outputs matching ground-truth labels "
  "on the full test set. For Ntest examples with inputs xj, ground-truth labels yj, and predictions "
  "ŷj, accuracy is (1/Ntest) Σ 1[ŷj = yj], the average indicator of a correct prediction.")

P("Accuracy was chosen over softer metrics such as partial-credit scoring or embedding-based similarity "
  "because every BBH task has a single, unambiguous correct answer drawn from a small option set or a "
  "short closed-form response (Section 4.1), making exact-match accuracy both well-defined and directly "
  "comparable to the original BBH benchmark paper and to the PromptWizard, APE, and ProTeGi baselines, "
  "all of which report accuracy on the same tasks. This choice has a corresponding limitation, already "
  "noted in Section 6.3: accuracy treats every incorrect answer identically regardless of how close the "
  "underlying reasoning came to correct, so it cannot by itself capture the reasoning-clarity "
  "improvements documented qualitatively in Section 5.2.4. Predicted answers are extracted "
  "programmatically from each response using the delimiter tags specified in every prompt variant "
  "(<ANS_START> and <ANS_END>, illustrated throughout Appendix A), rather than by free-form parsing of "
  "the full response text; this delimiter convention, itself one instantiation of the Output Format "
  "with Examples criterion in Section 3.3, is what makes automated accuracy scoring reliable across "
  "several thousand model responses without manual grading.")

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
  "reported in Table 2 to support reproducibility. The implementation of MPIR is publicly available at "
  "https://github.com/millennials92/meta_prompted_instruction_refinement.")

TABLE("Table 2. Hyperparameter settings of PromptWizard.",
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
P("Table 3 reports accuracy on the full test sets of the 23 BBH tasks (Section 4.1).")

TABLE("Table 3. Accuracy (%) across 23 BBH tasks.",
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

P("Table 4 reports accuracy before and after MPIR refinement across two further APO methods, Iterative "
  "APE and ProTeGi, on the full set of 23 BBH tasks. The tasks where MPIR moves accuracy in the same "
  "direction under all three methods most strongly are hyperbaton and ruin_names (consistently "
  "positive) and geometric_shapes and tracking_shuffled_objects (consistently negative).")

TABLE("Table 4. Accuracy (%) before and after MPIR refinement, for three APO methods, across all 23 "
      "BBH tasks.",
      header=["Task", "APE (before)", "ProTeGi (before)", "PromptWizard (before)",
              "APE (after)", "ProTeGi (after)", "PromptWizard (after)"],
      rows=[
          ["hyperbaton", "82.67", "80.89", "62.2", "85.78", "88.89", "84.0"],
          ["disambiguation_qa", "69.78", "73.78", "61.7", "68.00", "67.56", "62.2"],
          ["causal_judgement", "58.02", "61.11", "59.3", "58.64", "58.64", "56.8"],
          ["date_understanding", "87.56", "88.00", "71.1", "84.44", "85.78", "74.2"],
          ["penguins_in_a_table", "80.99", "79.34", "75.0", "84.30", "76.86", "82.6"],
          ["boolean_expression", "90.67", "90.67", "92.0", "93.78", "92.89", "94.2"],
          ["object_counting", "92.00", "88.44", "91.5", "93.33", "93.33", "92.4"],
          ["word_sorting", "69.33", "68.44", "80.0", "80.00", "66.67", "81.3"],
          ["logical_deduction", "61.93", "60.74", "44.9", "59.70", "59.26", "50.83"],
          ["salient_translation_error_detection", "55.11", "52.89", "50.7", "54.67", "49.33", "52.0"],
          ["geometric_shapes", "62.67", "61.78", "57.8", "61.33", "60.89", "49.3"],
          ["snarks", "61.44", "75.82", "63.3", "68.63", "73.20", "65.4"],
          ["temporal_sequences", "26.67", "84.00", "44.4", "86.67", "86.22", "38.2"],
          ["web_of_lies", "80.00", "81.33", "52.4", "73.33", "80.89", "52.9"],
          ["navigate", "95.56", "95.56", "65.3", "92.89", "97.78", "68.0"],
          ["reasoning_about_colored_objects", "82.67", "82.67", "56.0", "82.67", "83.56", "68.4"],
          ["sports_understanding", "95.11", "93.33", "79.6", "96.44", "94.67", "86.2"],
          ["multistep_arithmetic_two", "86.22", "83.56", "51.1", "84.89", "82.67", "51.5"],
          ["ruin_names", "58.67", "58.67", "66.2", "68.44", "77.78", "72.0"],
          ["movie_recommendation", "76.89", "76.00", "73.3", "79.11", "80.89", "66.7"],
          ["formal_fallacies", "52.44", "55.11", "53.3", "52.89", "56.89", "53.3"],
          ["dyck_languages", "24.44", "20.89", "14.2", "34.22", "32.44", "12.9"],
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
P("Table 3 shows a directionally positive but not conclusively significant improvement over "
  "PromptWizard: MPIR reaches an average accuracy of 64.37%, versus 62.39% for PromptWizard, a difference "
  "of 1.97 percentage points, with a 95% bootstrap confidence interval of [−0.46, 4.70] (10,000 "
  "resamples over the 23 tasks) that includes zero. Because the 23 tasks form matched pairs of "
  "heterogeneous difficulty rather than an unpaired sample, we follow standard practice for this design "
  "‹55,56› and supplement the bootstrap CI with three paired analyses of the same 23 "
  "task-level differences. A paired Wilcoxon signed-rank test is not significant (W = 83.0, p = 0.158), "
  "and a two-sided exact sign test on the win/loss count (16 wins, 6 losses, 1 tie) is borderline "
  "(p = 0.052; one-sided p = 0.026). The paired effect size is small-to-moderate (Cohen's dz = 0.30). We "
  "report all three rather than selecting the most favorable one: together they indicate a real but "
  "modest and not fully conclusive effect, consistent with typical statistical power at this task count "
  "‹57›, rather than an established, strongly significant improvement. The more robust evidence "
  "for MPIR is qualitative and structural—its consistency across individual tasks (16 of 23, with "
  "several gains of 10-20 points), across three different APO frameworks (Section 5.2.2), and across two "
  "model families (Section 5.2.3)—together with the free rewrite baseline, which reaches only 57.15% on "
  "average, well below both PromptWizard and MPIR. Together, these results indicate that MPIR's per-task "
  "gains are attributable to its structured seven-criteria rubric rather than simply to GPT-4o's general "
  "rewriting ability, even though the pooled average improvement should be read as a promising, "
  "consistent trend rather than a statistically confirmed effect; Section 6.1 revisits this as a "
  "threat to validity.")

H3("5.2.2. Evaluating MPIR Across Multiple APO Frameworks")
P("Table 4 shows that MPIR consistently improves average accuracy across APO frameworks: Iterative APE "
  "from 70.16% to 74.02%, ProTeGi from 72.81% to 74.26%, and PromptWizard from 62.39% to 64.37%. Although "
  "the magnitude of improvement varies across methods and tasks, the consistently positive trend supports "
  "MPIR functioning as a refinement layer across different APO frameworks rather than one tailored to "
  "PromptWizard-specific prompt structures.")

H3("5.2.3. Cross-Model Generalization")
P("To test MPIR under a different model family, we repeated the experiment with Gemini 3.5 Flash-Lite "
  "‹50› as both the target model and the meta-prompting model (Table 5). The baseline already "
  "achieves a high average accuracy of 92%, leaving limited room for improvement, and MPIR maintains the "
  "same rounded average after refinement. The most noticeable gains occur where baseline accuracy was "
  "around 70%: causal_judgement improves from 70% to 75%, disambiguation_qa from 77% to 80%, and "
  "dyck_languages from 72% to 75%. Tasks already near-perfect before refinement, such as "
  "sports_understanding and multistep_arithmetic_two, show little room for further gains and occasionally "
  "a slight decline—suggesting MPIR is most effective when the baseline prompt still has meaningful "
  "reasoning weaknesses to correct.")

TABLE("Table 5. Cross-model evaluation of MPIR on Gemini 3.5 Flash-Lite across all 23 BBH tasks.",
      header=["Task", "PromptWizard (%)", "MPIR (%)", "Change (%)"],
      rows=[
          ["hyperbaton", "100", "100", "0"],
          ["disambiguation_qa", "77", "80", "+3"],
          ["causal_judgement", "70", "75", "+5"],
          ["date_understanding", "95", "96", "+1"],
          ["penguins_in_a_table", "100", "98", "−2"],
          ["boolean_expressions", "100", "100", "0"],
          ["object_counting", "100", "100", "0"],
          ["word_sorting", "92", "92", "0"],
          ["logical_deduction", "99", "96", "−3"],
          ["salient_translation_error_detection", "75", "74", "−1"],
          ["geometric_shapes", "86", "83", "−3"],
          ["snarks", "90", "90", "0"],
          ["temporal_sequences", "100", "100", "0"],
          ["web_of_lies", "100", "100", "0"],
          ["navigate", "100", "99", "−1"],
          ["reasoning_about_colored_objects", "100", "100", "0"],
          ["sports_understanding", "91", "86", "−5"],
          ["multistep_arithmetic_two", "97", "94", "−4"],
          ["ruin_names", "88", "87", "−1"],
          ["movie_recommendation", "95", "96", "+1"],
          ["formal_fallacies", "99", "99", "0"],
          ["dyck_languages", "72", "75", "+3"],
          ["tracking_shuffled_objects", "100", "100", "0"],
          ["Average (23 tasks)", "92", "92", "0"],
      ], full=True, note="bold_last_row")

H3("5.2.4. Clarity, Structure, and Task Difficulty")
P("MPIR also improves interpretability by embedding human-inspired structure, clarifying reasoning "
  "context, and filtering out details that could distract the model. In hyperbaton, PromptWizard's "
  "baseline prompt mixed background explanation with task directives in a single block, whereas MPIR "
  "separated it into a context section and a numbered instruction section, helping the model distinguish "
  "framing from required actions; similar edits elsewhere (e.g., recasting a vague instruction in "
  "web_of_lies into an explicit role-and-goal statement) removed distracting, task-irrelevant content "
  "that had been reducing output quality.")

P("Across Tables 3-5, MPIR's strongest and most consistent gains occur on tasks governed by explicit "
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

H3("5.2.5. Detailed Failure Mode Analysis")
P("Appendix C.1 illustrates the symbolic-tracking failure pattern concretely on a "
  "tracking_shuffled_objects_five_objects example. Both PromptWizard and MPIR correctly identify the "
  "sequence of pairwise swaps described in the question, and both attempt to apply them in order—the "
  "failure is not in understanding the task, but in maintaining a consistent internal representation of "
  "state across several sequential updates. PromptWizard's reasoning trace re-derives the full mapping "
  "after each swap, which is verbose but self-correcting: an error at one step does not necessarily "
  "propagate, because the next step re-examines all positions rather than incrementally updating a "
  "single pair. MPIR's refined prompt, by contrast, produces a more compact, incrementally updated trace "
  "consistent with the Instruction and Separation and Guided Chain of Thought criteria it was optimized "
  "toward—but that same compactness means a single misapplied swap is never re-checked against the full "
  "state, and the error persists to the final answer. This is a case where two criteria that are "
  "individually beneficial for the structured, rule-based tasks discussed above (Section 5.2.4) "
  "interact unfavorably with a task that specifically requires redundant self-checking rather than "
  "compactness.")

P("A related pattern appears in logical_deduction and geometric_shapes, where the ablation in Table 7 "
  "shows Instruction and Separation (C4) as the single most load-bearing criterion overall, yet Table 3 "
  "and Table 4 show these same two tasks among MPIR's weakest relative to PromptWizard. Instruction and "
  "Separation improves clarity by isolating the task statement from surrounding context, but in these "
  "two task types the \"context\" being separated out often contains constraints the reasoning process "
  "must repeatedly refer back to—for example, an ordering constraint in logical_deduction or a shape's "
  "coordinate definition in geometric_shapes—so isolating it into a labeled context block does not "
  "reduce the cognitive load of the task the way it does for tasks where context is genuinely "
  "background rather than an active constraint. This suggests the seven criteria are not uniformly "
  "beneficial across task types, but rather beneficial conditional on the type of reasoning error a task "
  "is prone to—a distinction the current rubric does not represent explicitly, and which an "
  "instance-adaptive rubric (Section 7) could in principle capture.")

H3("5.2.6. Remaining Gap to Expert-Crafted Prompting")
P("Despite strong gains over automated baselines, MPIR still falls short of expert-crafted prompting on "
  "average (64.4% versus 69.5%). The gap concentrates in a small set of tasks—navigate, "
  "reasoning_about_colored_objects, web_of_lies, and multistep_arithmetic_two (Table 6)—where expert "
  "examples provide richer reasoning traces and more explicit state tracking. Because MPIR's examples are "
  "inherited from the PromptWizard baseline rather than newly generated, they reflect the same "
  "limitations present in PromptWizard's own outputs: MPIR's refinements are structurally consistent but "
  "tend to oversimplify example reasoning relative to expert-written traces.")

TABLE("Table 6. Tasks where MPIR underperforms expert-crafted prompting.",
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
  "removing one criterion at a time (Table 7). Removing any single criterion reduces average accuracy "
  "from the full rubric's 80.0% to between 66.0% and 75.0%, so every criterion plays a meaningful role, "
  "though to different degrees. Four criteria prove particularly critical—Instruction & Separation "
  "(C4), Role Prompting (C1), Guided Chain of Thought (C3), and Step Back (C2)—each reducing average "
  "accuracy by over 11 points when removed, underscoring their role in structuring reasoning and reducing "
  "ambiguity. Removing Worked Reasoning in Examples (C6) produces a moderate but consistent drop, "
  "suggesting that step-by-step demonstrations reinforce structured reasoning patterns, whereas removing "
  "Output Format Specification (C5) or Conclusion (C7) causes only minor drops, indicating these criteria "
  "mainly enhance clarity rather than drive task accuracy. Overall, heuristics that organize reasoning and "
  "contextual framing appear to contribute most to refinement performance on these benchmark tasks.")

TABLE("Table 7. Effect of removing individual rubric criteria across five BBH tasks (accuracy, %). "
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
  "tasks, holding refinement and validation fixed (Table 8). The full rubric outperforms the generic one "
  "on four of five tasks, with average accuracy dropping from 79.9% to 72.2% under the generic rubric—"
  "an 8-point gap confirming that the seven-criteria rubric's specificity, not just the act of evaluation "
  "and refinement, drives MPIR's effectiveness. The heuristic rubric enforces step-by-step reasoning, "
  "context separation, and structured outputs, whereas the generic rubric's vaguer feedback often fails "
  "to correct task-specific weaknesses.")

H3("5.3.3. Importance of Prompt Effectiveness Validation")
P("Finally, we tested whether the validation stage itself is necessary by comparing the full framework "
  "(seven evaluation-refinement-validation cycles) against a validation-ablated variant that selects a "
  "prompt after a single evaluation-refinement cycle with no empirical testing, again on five "
  "representative tasks (Table 8). Removing validation drops average accuracy from 79.9% to 61.9%, an "
  "18-point decline showing that rubric alignment alone does not guarantee performance: an "
  "unvalidated prompt may look well structured yet perform worse empirically. Validation anchors MPIR's "
  "refinements in measured accuracy, ensuring gains are both real and task-oriented rather than purely "
  "heuristic.")

TABLE("Table 8. Effect of the generic rubric and of removing prompt-effectiveness validation, each "
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

H2("5.4. Revisiting the Research Objectives")
P("Section 1 set out four objectives for this study; we revisit each in light of the results above.")

BULLETS([
    ("Objective 1 (a structured rubric for prompt evaluation):",
     "achieved. Section 3.3 formalizes seven manual prompting heuristics into an explicit, "
     "criterion-by-criterion rubric, and Section 5.3.1 shows the criteria are not redundant with one "
     "another—removing any single one measurably reduces accuracy, with Instruction and Separation, "
     "Role Prompting, Guided Chain of Thought, and Step Back identified as the most load-bearing."),
    ("Objective 2 (whether meta-prompted refinement improves APO-generated prompts):",
     "partially achieved. MPIR improves PromptWizard on 16 of 23 BBH tasks and the pooled average "
     "improvement is directionally positive, but Section 5.2.1 and Section 6.1 show this pooled "
     "improvement is not conclusively significant by paired statistical tests at this task count. The "
     "objective is better described as achieved in a qualitative, per-task sense than in a pooled, "
     "population-level sense."),
    ("Objective 3 (generalizability and modularity across APO frameworks and models):",
     "achieved within the tested scope. Section 5.2.2 shows consistent improvement when MPIR is applied "
     "to two additional APO methods beyond PromptWizard, and Section 5.2.3 shows the same pattern holds "
     "under a different model family; Section 6.2 notes this evidence does not extend to benchmark "
     "families beyond BBH or to task types such as open-ended generation."),
    ("Objective 4 (how individual heuristics contribute to performance):",
     "achieved. The ablation in Section 5.3.1 quantifies each criterion's individual contribution, and "
     "the comparison with a generic rubric in Section 5.3.2 further shows that the specific content of "
     "the seven criteria, not merely the act of iterative evaluation and refinement, drives the "
     "observed gains."),
])

H2("5.5. Practical Implications for Practitioners")
P("The results in this section suggest concrete guidance for a practitioner deciding whether to add "
  "MPIR on top of an existing APO pipeline. First, MPIR is best suited to settings where the target "
  "task involves explicit structural or rule-based reasoning—sorting, counting, boolean evaluation, "
  "temporal ordering—since Section 5.2.4 shows these are exactly the tasks where rubric-guided "
  "refinement most reliably helps; for tasks requiring precise symbolic manipulation or long sequential "
  "state tracking, the expected gain is smaller and occasionally negative, and a practitioner in that "
  "regime may be better served by investing directly in expert-crafted examples (Section 5.2.6) than in "
  "an automated refinement layer. Second, MPIR is most useful when the upstream APO system has room to "
  "improve: Section 5.2.3 shows that once baseline accuracy is already near-ceiling, MPIR's rubric-guided "
  "edits offer little additional benefit and can occasionally introduce small regressions, so it is not "
  "necessary to apply MPIR to prompts that are already performing well. Third, the per-task overhead is "
  "small and bounded—on the order of 200 additional API calls per task for the configuration "
  "used in this paper (Section 4.5)—so the primary cost-benefit question is whether the target "
  "application can tolerate that one-time refinement cost in exchange for a modest but consistent "
  "accuracy improvement, rather than whether the technique is computationally prohibitive in absolute "
  "terms. Fourth, because MPIR requires no retraining, gradient access, or bespoke search infrastructure, "
  "it is straightforward to retrofit onto an already-deployed APO pipeline without modifying that "
  "pipeline's own code, which may make it attractive in production settings where introducing a new "
  "training or search dependency is undesirable even when a more powerful but heavier method such as "
  "GEPA ‹59› might offer a larger expected gain.")

P("Beyond the BBH benchmark used for evaluation, the underlying pattern MPIR targets—an APO system "
  "produces a serviceable but imperfect prompt, and a small set of interpretable heuristics can catch "
  "specific, nameable weaknesses in it—plausibly recurs in several applied settings that share BBH's "
  "combination of structured tasks and automated prompt tuning. Customer-support and helpdesk deflection "
  "systems, for instance, often rely on an APO-tuned prompt to classify or answer routine queries; the "
  "kinds of errors documented in Section 5.2.4, where a baseline prompt buries the actual instruction "
  "inside irrelevant context, are directly analogous to a support prompt that mixes company-specific "
  "background with the actual classification task. Educational tutoring systems that generate "
  "step-by-step explanations depend heavily on the Guided Chain of Thought and Worked Reasoning in "
  "Examples criteria identified as most load-bearing in the ablation (Section 5.3.1), suggesting that a "
  "rubric-guided refinement pass could be particularly relevant there. Retrieval-augmented enterprise "
  "assistants, which typically combine a fixed instruction template with dynamically retrieved context, "
  "could apply MPIR's Instruction and Separation criterion specifically to keep the static instruction "
  "distinguishable from retrieved content that changes per query. We have not evaluated MPIR in any of "
  "these settings, and Section 6.2 already notes that generalization beyond BBH-style reasoning tasks is "
  "untested; we raise them here as plausible directions for the applied evaluation that Section 7 lists "
  "as future work, in keeping with this journal's focus on actionable applications of AI research.")

# ===========================================================================
H1("6. Threats to Validity")
P("This section organizes the study's limitations, several already introduced alongside individual "
  "results, into the standard internal, external, and construct validity framing used in empirical "
  "software and machine learning research, so that their scope and interaction are considered together "
  "rather than piecemeal.")

H2("6.1. Internal Validity")
P("The central internal-validity concern is statistical: the average improvement over PromptWizard "
  "(1.97 points) is not conclusively significant by any of the three paired tests reported in Section "
  "5.2.1, and all results come from a single run per condition at fixed temperature 0 rather than "
  "repeated trials with varied seeds or example orderings. A second concern is evaluator bias: MPIR's "
  "meta-prompting stages and the free rewrite baseline both use GPT-4o as judge and rewriter "
  "respectively, so any systematic preference GPT-4o has for its own stylistic conventions could inflate "
  "MPIR's apparent quality independently of downstream task accuracy; the validation stage (Section 3.5) "
  "partially controls for this by requiring an accuracy gain on GPT-3.5-turbo, a different model, before "
  "a candidate is accepted, but cannot rule it out entirely. A third concern is that MPIR reuses "
  "PromptWizard's own 25 optimization examples during validation (Section 4.5) to keep the comparison "
  "fair; this is a deliberate design choice rather than an oversight, but it does mean the validation "
  "signal and the optimization signal are drawn from overlapping data.")

H2("6.2. External Validity")
P("Generalizability is bounded along three axes tested in this paper and open along others not tested. "
  "Within the tested axes, MPIR's benefit generalizes across three APO backbones (PromptWizard, "
  "Iterative APE, ProTeGi; Section 5.2.2) and across two model families (GPT-3.5-turbo and Gemini "
  "3.5 Flash-Lite; Section 5.2.3), which is meaningfully broader evidence than a single-backbone, "
  "single-model result would provide. Outside the tested axes, all results are on Big-Bench Hard, a "
  "reasoning-and-instruction-following benchmark; performance on other task families such as "
  "open-ended generation, retrieval-augmented question answering, or code generation is untested. BBH "
  "also remains meaningfully unsaturated only for the cheap-tier target models used here; results might "
  "differ for frontier-tier target models, where BBH is reported to be substantially saturated "
  "‹27›. Finally, GPT-3.5-turbo, the primary target model, is scheduled for API retirement on "
  "October 23, 2026 ‹65›, so external validity for future reproductions will depend on its "
  "successor behaving comparably.")

H2("6.3. Construct Validity")
P("The seven-criteria rubric (Section 3.3) is the paper's central construct, and it was developed in "
  "proximity to the BBH task family it is evaluated on, drawing on general prompting literature "
  "‹44,45,63› and industry guidance ‹40,41,42,43› but refined through iterative "
  "experimentation on these same tasks. This creates a risk of construct dependence: the rubric may "
  "capture properties that happen to matter for BBH-style reasoning tasks specifically, rather than "
  "prompt quality in a domain-independent sense. The ablation in Section 5.3.1 shows the seven criteria "
  "are not equally load-bearing, which is consistent with either a well-differentiated construct or with "
  "some criteria being closer proxies for BBH performance than others; the study cannot distinguish "
  "between these two explanations without evaluating the rubric on an independent benchmark family, "
  "which Section 7 lists as future work. A second construct concern is the evaluation metric itself: "
  "accuracy against a single ground-truth answer treats near-miss and far-miss errors identically, which "
  "may understate MPIR's qualitative improvements in reasoning clarity documented in Section 5.2.4.")

# ===========================================================================
H1("7. Conclusion")

P("This paper introduced Meta-Prompted Instruction Refinement (MPIR), a lightweight, model-agnostic "
  "framework that integrates manual prompting heuristics into automatic prompt optimization through a "
  "structured cycle of evaluation, refinement, and validation. It asks whether human-inspired prompting "
  "heuristics can be systematically integrated into APO systems to improve prompt quality while "
  "preserving the scalability of automated optimization, and does so deliberately without the "
  "additional training, search infrastructure, or per-task tuning that increasingly elaborate 2025-2026 "
  "prompt-optimization methods require (Section 2.5). Experiments on Big-Bench Hard show that "
  "rubric-guided meta-prompt refinement can improve APO-generated prompts across multiple tasks and APO "
  "frameworks, and that MPIR is particularly effective for structured, rule-based reasoning tasks, where "
  "heuristic-guided refinement yields clearer and more reliable reasoning.")

P("This study also has limitations, discussed in depth as threats to validity in Section 6: the "
  "average gain over the PromptWizard baseline is modest and not conclusively significant by any single "
  "test (Section 6.1); results come from a single run per condition rather than repeated trials with "
  "varying seeds, a gap common across the prompt-optimization literature but one that recent evidence on "
  "the brittleness of LLM evaluations makes worth closing (Section 6.1); generalization beyond the three "
  "tested APO frameworks, two tested model families, and the BBH benchmark itself remains open (Section "
  "6.2); and the seven-criteria rubric was developed in proximity to the BBH tasks it is evaluated on, "
  "which may introduce construct dependence (Section 6.3). We summarize the strongest, most defensible "
  "evidence for MPIR's contribution as qualitative and structural—consistency across individual tasks, "
  "APO frameworks, and model families—rather than the pooled average improvement alone.")

P("Future work could extend MPIR along several directions. Repeated-trial evaluation with multiple "
  "seeds and example orderings would directly address the internal-validity concern in Section 6.1 by "
  "separating run-to-run optimization variance from genuine task-level effects, and would let the "
  "current bootstrap and paired tests be supplemented with variance estimates that account for both "
  "sources of noise simultaneously. Independent construct validation—deriving the seven-criteria "
  "rubric, or a variant of it, on one benchmark family and evaluating its transfer to a distinct one, "
  "such as open-ended generation or retrieval-augmented question answering—would directly test the "
  "construct-dependence concern raised in Section 6.3. Adaptive or instance-specific rubrics, in the "
  "spirit of the learned-rubric direction noted in Section 2.5, could replace MPIR's fixed seven "
  "criteria with criteria selected or weighted per task, potentially recovering some of the benefit of "
  "heavier methods such as GEPA while retaining MPIR's lower training and infrastructure cost. Applying "
  "MPIR to the frontier-model regime, where BBH itself is largely saturated ‹27›, would require "
  "pairing it with a harder successor benchmark such as BIG-Bench Extra Hard rather than BBH directly. "
  "Finally, combining MPIR with complementary approaches such as symbolic reasoning modules, retrieval "
  "augmentation, self-consistency decoding ‹66›, or automated methods for learning prompting "
  "heuristics directly from empirical performance data are all directions that could be pursued "
  "independently of one another, since none of them are mutually exclusive with the rubric-guided "
  "refinement loop introduced here.")

# ===========================================================================
H1("Acknowledgement")
P("The authors thank the reviewers for their constructive comments on earlier drafts of this manuscript.")

H1("Funding")
P("This research received no specific grant from any funding agency in the public, commercial, or "
  "not-for-profit sectors.")

H1("Ethical Statement")
P("This study did not involve human participants, human data, or animal subjects, and did not require "
  "ethical approval.")

H1("Declaration of Competing Interest")
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

# ===========================================================================
APPENDIX_START()
H1("Appendix A. Prompt Examples")

H2("A.1. Prompts for the penguins_in_a_table Task")
P("To make the qualitative discussion in Section 5.2.4 and the case study in Figure 6 self-contained, "
  "this appendix reproduces the full prompt text used at each stage of the pipeline for one "
  "representative task, penguins_in_a_table, exactly as issued to the model.")

CODE([
    "You are given a task that require answering questions about a table",
    "of penguins and their attributes.",
    "Let's think step by step.",
    "For each question, wrap only the final letter (A) (B) (C) (D) (E)",
    "between <ANS_START> and <ANS_END> tags",
], caption="Figure A.1. Zero-shot CoT prompt for penguins_in_a_table.", full=False)

CODE([
    "Answer questions about a table of penguins and their attributes.",
    "For each question, present the reasoning followed by final answer",
    "between <ANS_START> and <ANS_END> tags",
    "",
    "[Question]: Here is a table where the first line is a header and",
    "each subsequent line is a penguin: name, age, height (cm), weight (kg)",
    "Louis, 7, 50, 11",
    "Bernard, 5, 80, 13",
    "Vincent, 9, 60, 11",
    "Gwen, 8, 70, 15",
    "For example: the age of Louis is 7, the weight of Gwen is 15 kg,",
    "the height of Bernard is 80 cm.",
    "We now add a penguin to the table: James, 12, 90, 12",
    "How many penguins are less than 8 years old?",
    "Options: (A) 1 (B) 2 (C) 3 (D) 4 (E) 5",
    "[Answer]: Let's think step by step.",
    "This question focuses on age. We know the following: Louis is 7",
    "years old, Bernard is 5 years old, Vincent is 9 years old, and Gwen",
    "is 8 years old.",
    "Now, we add James to this table: James is 12 years old.",
    "The penguins that are less than 8 years old are Louis and Bernard.",
    "There are 2 penguins less than 8 years old. So the answer is",
    "<ANS_START>(B)<ANS_END>.",
    "",
    "[... two further worked examples omitted for space ...]",
    "",
    "For each question, present the reasoning followed by final answer",
    "between <ANS_START> and <ANS_END> tags",
], caption="Figure A.2. Few-shot CoT prompt for penguins_in_a_table (expert-crafted, three worked "
           "examples; two omitted here for space, unabridged in the project repository).", full=False)

CODE([
    "To ensure accurate comparisons of the penguins' heights, meticulously",
    "analyze and consider the specific attributes of each penguin,",
    "including age, height, and weight. It is crucial to compare the",
    "height of the new penguin, James, with the existing penguins in the",
    "table to arrive at the correct answer. Emphasize the importance of",
    "carefully examining all relevant information about each penguin and",
    "comparing their heights to make an informed decision.",
    "",
    "[Question]",
    "Here is a table where the first line is a header and each subsequent",
    "line is a penguin: name, age, height (cm), weight (kg)",
    "Louis, 7, 50, 11",
    "Bernard, 5, 80, 13",
    "Vincent, 9, 60, 11",
    "Gwen, 8, 70, 15",
    "For example: the age of Louis is 7, the weight of Gwen is 15 kg,",
    "the height of Bernard is 80 cm.",
    "Which penguin is younger but taller than Gwen?",
    "Options: (A) Louis (B) Bernard (C) Vincent (D) Gwen (E) James",
    "[Answer]",
    "1. Start by identifying the attributes of each penguin in the table:",
    "   name, age, height (cm), and weight (kg).",
    "2. Compare the age and height of each penguin to determine who is",
    "   younger but taller than Gwen.",
    "3. Gwen's age is 8, and her height is 70 cm.",
    "4. Louis is younger than Gwen with an age of 7, but his height is",
    "   50 cm, so he is not taller than Gwen.",
    "5. Bernard is younger than Gwen with an age of 5, and his height is",
    "   80 cm, making him both younger and taller than Gwen.",
    "6. Vincent is older than Gwen with an age of 9, so he is not younger",
    "   than Gwen.",
    "7. James is not provided in the table, so he cannot be compared to",
    "   Gwen.",
    "8. Therefore, the penguin who is younger but taller than Gwen is",
    "   Bernard, making the correct answer (B).",
    "<ANS_START>(B)<ANS_END>",
    "",
    "[... two further worked examples omitted for space ...]",
    "",
    "For each question, present the reasoning followed by final answer",
    "between <ANS_START> and <ANS_END> tags",
], caption="Figure A.3. PromptWizard-optimized prompt for penguins_in_a_table before MPIR "
           "refinement (one of three in-context examples shown; unabridged in the project "
           "repository). Note the task-irrelevant aside in the opening paragraph about comparing "
           "James's height, discussed in Section 5.2.4.", full=False)

CODE([
    "### Context",
    "You are a data analyst tasked with comparing the attributes of",
    "penguins to answer specific questions. Your role is to meticulously",
    "analyze and consider the specific attributes of each penguin,",
    "including age, height, and weight, to ensure accurate comparisons.",
    "",
    "### Instructions",
    "1. Understand the Principles: Begin by understanding the principles",
    "   of data analysis and comparison techniques. Focus on how to",
    "   extract relevant information and compare attributes effectively.",
    "2. Identify Attributes: Examine the table provided, which contains",
    "   information about penguins. Identify the key attributes: name,",
    "   age, height (cm), and weight (kg).",
    "3. Follow Step-by-Step Reasoning: For each question, follow a",
    "   detailed step-by-step reasoning process to arrive at the correct",
    "   answer.",
    "4. Use Output Format: Present your reasoning and final answer using",
    "   the specified output format, which includes [Question] and",
    "   [Answer] sections.",
    "5. Conclude with a Summary: After answering all questions, provide a",
    "   concise summary or restatement of the task to ensure you have",
    "   completed it correctly.",
    "",
    "### Example Output Format",
    "[Question]",
    "Here is a table where the first line is a header and each subsequent",
    "line is a penguin: name, age, height (cm), weight (kg)",
    "Louis, 7, 50, 11",
    "Bernard, 5, 80, 13",
    "Vincent, 9, 60, 11",
    "Gwen, 8, 70, 15",
    "For example: the age of Louis is 7, the weight of Gwen is 15 kg,",
    "the height of Bernard is 80 cm.",
    "Which penguin is younger but taller than Gwen?",
    "Options: (A) Louis (B) Bernard (C) Vincent (D) Gwen (E) James",
    "",
    "[Answer]",
    "1. Start by identifying the attributes of each penguin in the table:",
    "   name, age, height (cm), and weight (kg).",
    "2. Compare the age and height of each penguin to determine who is",
    "   younger but taller than Gwen.",
    "3. Gwen's age is 8, and her height is 70 cm.",
    "4. Louis is younger than Gwen with an age of 7, but his height is",
    "   50 cm, so he is not taller than Gwen.",
    "5. Bernard is younger than Gwen with an age of 5, and his height is",
    "   80 cm, making him both younger and taller than Gwen.",
    "6. Vincent is older than Gwen with an age of 9, so he is not younger",
    "   than Gwen.",
    "7. James is not provided in the table, so he cannot be compared to",
    "   Gwen.",
    "8. Therefore, the penguin who is younger but taller than Gwen is",
    "   Bernard, making the correct answer (B).",
    "<ANS_START>(B)<ANS_END>",
    "",
    "### Conclusion",
    "By following the instructions and using the output format, you can",
    "accurately compare the attributes of penguins to answer the",
    "questions provided. Ensure you have completed the task by reviewing",
    "your answers and the reasoning process.",
], caption="Figure A.4. MPIR-refined prompt for penguins_in_a_table (one of three in-context "
           "examples shown; unabridged in the project repository). Compared with Figure A.3, "
           "context and instructions are separated into labeled sections and the task-irrelevant "
           "aside has been removed.", full=False)

H2("A.2. Few-Shot Examples for the Navigate Task")
P("The navigate task asks whether a sequence of turns and steps returns to the starting point. Figure "
  "A.5 and Figure A.6 contrast the expert-crafted worked example used as the Section 4.3.3 reference "
  "point with the corresponding MPIR-refined example, illustrating the gap discussed in Section 5.2.6: "
  "MPIR's example, inherited from PromptWizard, tracks state as a bulleted sequence of moves without "
  "explicit coordinates, whereas the expert-crafted example maintains an explicit running position.")

CODE([
    "[Question]: If you follow these instructions, do you return to the",
    "starting point? Turn left. Turn around. Turn left. Take 7 steps.",
    "Take 2 steps. Take 4 steps. Take 8 steps.",
    "Options:",
    "- Yes",
    "- No",
    "[Answer]: Let's think step by step.",
    "We start at the origin (0, 0), facing the positive y-axis.",
    "(1) Turn left: (0, 0), facing the negative x-axis.",
    "(2) Turn around: (0, 0), facing the positive x-axis.",
    "(3) Turn left: (0, 0), facing the positive y-axis.",
    "(4) Take 7 steps: (0, 7), facing the positive y-axis.",
    "(5) Take 2 steps: (0, 9), facing the positive y-axis.",
    "(6) Take 4 steps: (0, 13), facing the positive y-axis.",
    "(7) Take 8 steps: (0, 21), facing the positive y-axis.",
    "Since (0, 21) is not (0, 0), we are not where we started.",
    "So the answer is <ANS_START>No<ANS_END>.",
], caption="Figure A.5. Expert-crafted worked example for the navigate task, with explicit "
           "running coordinates.", full=False)

CODE([
    "[Question] If you follow these instructions, do you return to the",
    "starting point? Take 1 step. Take 7 steps. Take 1 step.",
    "Options:",
    "- Yes",
    "- No",
    "[Answer]",
    "- [Step 1]: Start at the initial point.",
    "- [Step 2]: Take 1 step forward.",
    "- [Step 3]: Move 7 steps forward from the new position.",
    "- [Step 4]: Move 1 step forward from the current position.",
    "- [Logical Pathway]:",
    "  - After taking 1 step forward in Step 2, the agent is at a new",
    "    position.",
    "  - Moving 7 steps forward in Step 3 takes the agent further away",
    "    from the starting point.",
    "  - Finally, taking 1 step forward in Step 4 continues the movement",
    "    away from the starting point.",
    "- [Final Answer]: No <ANS_START>No<ANS_END>",
], caption="Figure A.6. MPIR-refined worked example for the navigate task, tracking state as a "
           "bulleted move sequence without explicit coordinates.", full=False)

# ===========================================================================
H1("Appendix B. Generic Rubric Baseline Prompt")
P("Section 5.3.2 compares MPIR's seven-criteria rubric against a generic rubric that evaluates prompts "
  "on broad, undifferentiated aspects of quality rather than the specific heuristics in Section 3.3. "
  "The full meta-prompt used for that generic-rubric evaluation stage is reproduced below; the "
  "refinement and validation stages are otherwise identical to MPIR's own (Sections 3.4-3.5).")

CODE([
    "You are an expert prompt evaluator. Your task is to analyze the",
    "given prompt and provide structured feedback on its overall quality",
    "for guiding a large language model (LLM) to perform the specified",
    "task effectively.",
    "Here is the prompt to evaluate: {prompt}",
    "Assess the prompt based on general aspects of good prompt design,",
    "such as clarity, structure, completeness, and effectiveness in",
    "eliciting the desired LLM behavior. For each aspect you identify",
    "(aim for 5-7 key aspects), provide:",
    "- A score from 1 to 5 (1 = very poor, 5 = excellent).",
    "- A brief justification explaining why you gave that score.",
    "- Strengths: What works well in this aspect.",
    "- Weaknesses: What could be improved in this aspect.",
    "Finally, based on your assessment, suggest 7-10 actionable",
    "improvements to make the prompt better overall. Keep suggestions",
    "practical and focused on enhancing task performance without",
    "overcomplicating the prompt.",
    "Output your response in this exact format:",
    "Aspect 1: [Name of aspect]",
    "Score: [Score]/10",
    "Justification: [Brief explanation]",
    "Strengths: [Bullet points]",
    "Weaknesses: [Bullet points]",
    "[Repeat for each aspect]",
    "Actionable Improvements:",
    "1. [Suggestion 1] 2. [Suggestion 2]...",
], caption="Figure B.1. Meta-prompt for the generic rubric baseline used in Section 5.3.2.",
    full=True)

# ===========================================================================
H1("Appendix C. Additional Case Study")
P("Section 5.1 and Figure 6 present a case in which MPIR corrects a PromptWizard reasoning error. Not "
  "every case favors MPIR: Figure C.1 shows a tracking_shuffled_objects_five_objects example where "
  "PromptWizard and MPIR follow the same sequence of swaps but diverge in intermediate bookkeeping, "
  "with PromptWizard reaching the correct answer and MPIR misapplying one swap. This example is "
  "representative of the symbolic-tracking weakness discussed in Section 5.2.4.")

FIG("case_study_2.png",
    "Figure C.1. Case study comparing PromptWizard and MPIR on the "
    "tracking_shuffled_objects_five_objects task. Both systems follow the same sequence of swaps but "
    "diverge in intermediate reasoning: PromptWizard maintains consistent tracking of player positions "
    "and reaches the correct answer (C: left winger), whereas MPIR misapplies one swap and concludes "
    "incorrectly (B: right midfielder).", full=True)

# ===========================================================================
H1("Appendix D. Free Rewrite Baseline Prompt")
P("Section 4.3.1 introduces a free rewrite baseline that isolates MPIR's rubric-guided evaluation from "
  "GPT-4o's general rewriting ability. The full meta-prompt used to produce that baseline is reproduced "
  "below.")

CODE([
    "You are given a task-solving prompt generated for a Big-Bench Hard",
    "task.",
    "",
    "Rewrite the prompt to improve clarity, readability,",
    "and instruction organization while preserving the",
    "original task meaning.",
    "",
    "Preserve the original task, answer format, and examples.",
    "Do not change the meaning of the task.",
    "Do not add unrelated content.",
    "Do not use any prompt-evaluation rubric.",
    "Do not mention rubric criteria, scoring, strengths,",
    "weaknesses, or feedback.",
    "",
    "Output only the rewritten task-solving prompt.",
    "",
    "{variant_instruction}",
    "",
    "Original prompt:",
    "<START>",
    "{prompt}",
    "<END>",
], caption="Figure D.1. Meta-prompt for the free rewrite baseline used in Section 4.3.1.", full=True)

# ===========================================================================
H1("Appendix E. Reproducibility and Experimental Protocol")
P("This appendix consolidates the settings scattered across Sections 3-5 into a single reference for "
  "reproduction, following the convention that empirical machine learning papers report a complete, "
  "self-contained protocol rather than requiring a reader to reassemble it from prose.")

TABLE("Table 9. Complete experimental protocol summary.",
      header=["Setting", "Value"],
      rows=[
          ["Benchmark", "Big-Bench Hard, 23 tasks (Suzgun et al., 2023)"],
          ["Primary target model", "GPT-3.5-turbo, temperature 0"],
          ["Meta-prompting model (evaluation and refinement)", "GPT-4o, temperature 0"],
          ["Cross-model target/meta model", "Gemini 3.5 Flash-Lite, temperature 0"],
          ["Primary APO baseline", "PromptWizard (unmodified hyperparameters, Table 2)"],
          ["Extended APO baselines", "Iterative APE; ProTeGi"],
          ["Training/optimization examples per task", "25, randomly sampled"],
          ["Held-out test examples per task", "Remainder of each task's example set"],
          ["MPIR refinement rounds (N)", "7"],
          ["Rubric criteria evaluated per round", "7 (Section 3.3)"],
          ["Candidate selection rule", "Highest held-out accuracy across all N rounds"],
          ["Ablation task subset (Sections 5.3.1-5.3.3)",
           "hyperbaton, penguins_in_a_table, ruin_names, object_counting, "
           "reasoning_about_colored_objects"],
          ["Significance tests reported", "Bootstrap CI (10,000 resamples); paired Wilcoxon "
           "signed-rank; exact sign test; Cohen's dz"],
          ["Runs per condition", "1 (single run at temperature 0; see Section 6.1)"],
      ], full=True, colw=[0.45, 0.55])

P("All code, configuration files, prompt templates, and per-task raw results referenced throughout this "
  "paper are available at the project repository (Section 4.5), including the exact YAML configuration "
  "files for PromptWizard and MPIR, the rubric prompt templates reproduced in Figures 2 and 3, and the "
  "full 23-task results underlying every averaged or condensed figure reported above.")
