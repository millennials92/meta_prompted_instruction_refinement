import random
import re
from os.path import join
from typing import Any, List

from ....paramlogger import ParamLogger
from ....paramlogger.constants import LogLiterals
from ....common.base_classes import SetupConfig, UniversalBaseClass
from ....common.llm.llm_mgr import LLMMgr
from ...constants import PromptOptimizationParams, SupportedPromptOpt
from ...techniques.common_logic import DatasetSpecificProcessing, PromptOptimizer
from ...techniques.ape.base_classes import APEPromptPool


class APE(PromptOptimizer, UniversalBaseClass):
    """
    Automatic Prompt Engineer (Zhou et al., 2023). Induces candidate instructions
    from input-output demonstrations via forward-mode generation, scores each
    candidate by execution accuracy on a minibatch, then runs the paper's
    iterative Monte Carlo search: resample (paraphrase) around the best-scoring
    candidates for a fixed number of rounds and keep whichever candidate scored
    highest overall.
    """

    TECHNIQUE_NAME = SupportedPromptOpt.APE.value

    class EvalLiterals:
        IS_CORRECT = "is_correct"
        PREDICTED_ANS = "predicted_ans"
        LLM_OUTPUT = "llm_output"

    # This has to be defined outside of constructor, so that it can be used as decorator.
    iolog = ParamLogger()

    def __init__(self, dataset: List, base_path: str, setup_config: SetupConfig,
                 prompt_pool: APEPromptPool, data_processor: DatasetSpecificProcessing, logger):
        self.dataset = dataset
        self.setup_config = setup_config
        self.data_processor = data_processor
        self.logger = logger
        self.prompt_pool = prompt_pool
        base_path = join(base_path, LogLiterals.DIR_NAME)
        self.iolog.reset_eval_glue(base_path)

    @iolog.log_io_params
    def chat_completion(self, user_prompt: str, system_prompt: str = None):
        """
        Make a chat completion request to the configured LLM.

        :param user_prompt: Text spoken by user in a conversation.
        :param system_prompt: Text spoken by system in a conversation.
        :return: Output of LLM
        """
        if not system_prompt:
            system_prompt = self.prompt_pool.system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = LLMMgr.chat_completion(messages)
        return response

    @iolog.log_io_params
    def generate_candidates(self, params: PromptOptimizationParams) -> List[str]:
        """
        Forward-mode generation (Zhou et al. 2023, §2.1): repeatedly sample a small
        set of input-output demonstrations from the training set and ask the LLM to
        induce an instruction that explains the mapping, building a pool of
        `params.num_candidates` candidate instructions.

        :param params: Hyperparameters for this optimization run.
        :return: List of candidate instruction strings.
        """
        candidates = []
        for _ in range(params.num_candidates):
            num_demos = min(params.num_demos_per_prompt, len(self.dataset))
            demos = random.sample(self.dataset, num_demos)
            demo_string = self.data_processor.collate_to_str(demos, self.prompt_pool.quest_reason_ans)
            gen_prompt = self.prompt_pool.forward_gen_template.format(demos=demo_string)
            response = self.chat_completion(gen_prompt)
            matches = re.findall(DatasetSpecificProcessing.TEXT_DELIMITER_PATTERN, response)
            if matches:
                candidates.append(matches[0].strip())
        return candidates

    @iolog.log_io_params
    def score_candidate(self, instruction: str, params: PromptOptimizationParams) -> float:
        """
        Execution accuracy of `instruction` over a random scoring minibatch drawn
        from the training set (Zhou et al. 2023, §2.2).

        :param instruction: Candidate instruction to score.
        :param params: Hyperparameters for this optimization run.
        :return: Fraction of the scoring minibatch answered correctly.
        """
        scoring_set = random.sample(self.dataset, min(params.num_scoring_examples, len(self.dataset)))
        if not scoring_set:
            return 0.0

        correct_count = 0
        for example in scoring_set:
            question = example[DatasetSpecificProcessing.QUESTION_LITERAL]
            actual_answer = example[DatasetSpecificProcessing.FINAL_ANSWER_LITERAL]
            eval_prompt = self.prompt_pool.eval_prompt.format(instruction=instruction, question=question)
            llm_output = self.chat_completion(eval_prompt)
            is_correct, _ = self.data_processor.access_answer(llm_output, actual_answer)
            correct_count += int(is_correct)

        return correct_count / len(scoring_set)

    @iolog.log_io_params
    def resample_candidate(self, instruction: str) -> str:
        """
        Generate a semantically similar paraphrase of `instruction`
        (Zhou et al. 2023, §2.3, iterative Monte Carlo search).

        :param instruction: Candidate instruction to paraphrase.
        :return: Paraphrased instruction, or the original if parsing fails.
        """
        resample_prompt = self.prompt_pool.resample_template.format(instruction=instruction)
        response = self.chat_completion(resample_prompt)
        matches = re.findall(DatasetSpecificProcessing.TEXT_DELIMITER_PATTERN, response)
        return matches[0].strip() if matches else instruction

    def get_best_prompt(self, params: PromptOptimizationParams, **kwargs) -> (str, Any):
        """
        Run forward generation, score the initial candidate pool, then run
        `params.iterations` rounds of resample-and-rescore around the top
        `params.top_n` candidates. Return the best-scoring instruction found,
        formatted with the framework's final_prompt template.

        :param params: Object of class APEParams with hyperparameters for this run.
        :return: (best_prompt, expert_profile)
        """
        candidates = self.generate_candidates(params)
        if params.base_instruction:
            candidates.append(params.base_instruction)
        if not candidates:
            candidates = [params.base_instruction or params.task_description]

        scored = [(candidate, self.score_candidate(candidate, params)) for candidate in candidates]

        for iteration in range(params.iterations):
            scored.sort(key=lambda item: item[1], reverse=True)
            top_candidates = scored[:params.top_n]

            resampled = []
            for candidate, _ in top_candidates:
                for _ in range(params.num_resamples_per_candidate):
                    resampled.append(self.resample_candidate(candidate))

            resampled_scored = [(candidate, self.score_candidate(candidate, params)) for candidate in resampled]
            scored = top_candidates + resampled_scored
            self.logger.info(f"APE iteration {iteration + 1}: "
                             f"best so far = {max(scored, key=lambda item: item[1])}")

        scored.sort(key=lambda item: item[1], reverse=True)
        best_instruction, best_score = scored[0]
        self.logger.info(f"APE final best score: {best_score:.2f}, instruction: {best_instruction}")

        final_best_prompt = self.prompt_pool.final_prompt.format(
            instruction=best_instruction,
            answer_format=params.answer_format,
            few_shot_examples="")

        return final_best_prompt, self.prompt_pool.system_prompt
