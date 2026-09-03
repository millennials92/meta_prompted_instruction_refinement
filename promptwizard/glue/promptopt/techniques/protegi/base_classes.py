from dataclasses import dataclass

from ....common.base_classes import UniversalBaseClass
from ...constants import PromptOptimizationParams, PromptPool


@dataclass
class ProTeGiPromptPool(PromptPool):
    quest_reason_ans: str
    gradient_template: str
    edit_template: str
    paraphrase_template: str
    ans_delimiter_instruction: str


@dataclass
class ProTeGiParams(PromptOptimizationParams, UniversalBaseClass):
    # Description of task. This will be fed to prompt.
    task_description: str
    # Starting instruction that beam search refines.
    base_instruction: str
    # Instruction for specifying answer format
    answer_format: str
    # Number of samples from dataset set aside as training data.
    seen_set_size: int
    # Number of examples to be given for few shots in the final prompt.
    few_shot_count: int
    # Size of the minibatch run against a candidate to surface errors that
    # seed the next textual gradient.
    minibatch_size: int
    # Number of edited candidates requested per textual gradient.
    num_edits_per_gradient: int
    # Number of paraphrases generated per edited candidate, to broaden the
    # successor pool beyond literal gradient edits.
    num_paraphrases_per_edit: int
    # Size of the minibatch used to score every candidate before pruning the
    # beam each round.
    eval_batch_size: int
    # Number of candidates retained across rounds (beam width).
    beam_width: int
    # Number of gradient-descent-like rounds (expand + score + prune).
    num_steps: int
