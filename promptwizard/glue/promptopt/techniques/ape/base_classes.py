from dataclasses import dataclass

from ....common.base_classes import UniversalBaseClass
from ...constants import PromptOptimizationParams, PromptPool


@dataclass
class APEPromptPool(PromptPool):
    quest_reason_ans: str
    forward_gen_template: str
    resample_template: str
    ans_delimiter_instruction: str


@dataclass
class APEParams(PromptOptimizationParams, UniversalBaseClass):
    # Description of task. This will be fed to prompt.
    task_description: str
    # Optional seed instruction, appended to the induced candidate pool as-is
    # (APE itself induces instructions from demonstrations rather than
    # refining a starting instruction, but a seed lets it be compared
    # fairly against techniques that do start from one).
    base_instruction: str
    # Instruction for specifying answer format
    answer_format: str
    # Number of samples from dataset set aside as training data.
    seen_set_size: int
    # Number of examples to be given for few shots in the final prompt.
    few_shot_count: int
    # Number of input-output demonstrations shown per forward-generation call.
    num_demos_per_prompt: int
    # Number of candidate instructions to induce via forward generation.
    num_candidates: int
    # Size of the minibatch used to score each candidate by execution accuracy.
    num_scoring_examples: int
    # Number of iterative Monte Carlo search rounds (resampling around the
    # best-scoring candidates), per Zhou et al. 2023's iterative variant.
    iterations: int
    # Number of resampled (paraphrased) candidates generated per retained
    # candidate, each iteration.
    num_resamples_per_candidate: int
    # Number of top-scoring candidates retained for resampling each iteration.
    top_n: int
