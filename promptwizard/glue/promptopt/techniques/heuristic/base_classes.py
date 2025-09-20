from dataclasses import dataclass
from typing import List

from ....common.base_classes import UniversalBaseClass
from ...constants import PromptOptimizationParams, PromptPool


@dataclass
class HeuristicPromptPool(PromptPool):
    prompt_evaluation: str
    prompt_generic_rubric: str
    prompt_refinement: str
    improved_prompt: str
    final_prompt: str
    ans_delimiter_instruction: str

@dataclass
class HeuristicParams(PromptOptimizationParams, UniversalBaseClass):
    validation_round: int
    answer_format: str
    seen_set_size: int
    few_shot_count: int
    