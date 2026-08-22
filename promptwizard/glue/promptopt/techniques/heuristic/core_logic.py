import re
from os.path import join
from typing import List

from ....paramlogger import ParamLogger
from ....paramlogger.constants import LogLiterals
from ....common.base_classes import SetupConfig, UniversalBaseClass
from ....common.llm.llm_mgr import LLMMgr
from ...constants import PromptOptimizationParams, SupportedPromptOpt
from ...techniques.common_logic import DatasetSpecificProcessing, PromptOptimizer
from ...techniques.heuristic.base_classes import HeuristicPromptPool


class Heuristic(PromptOptimizer, UniversalBaseClass):
    """
    MPIR (Meta-Prompted Instruction Refinement) technique. Refines an APO-generated
    prompt through iterative rounds of rubric-based evaluation, meta-prompted
    refinement, and validation against a held-out dataset, keeping the
    highest-scoring candidate across rounds.
    """

    TECHNIQUE_NAME = SupportedPromptOpt.HEURISTIC.value

    # Model used for the meta-prompting stages (rubric evaluation and refinement),
    # kept distinct from the target LLM used during validation.
    META_MODEL_NAME = "gpt-4o"

    class EvalLiterals:
        IS_CORRECT = "is_correct"
        PREDICTED_ANS = "predicted_ans"
        LLM_OUTPUT = "llm_output"

    class GetPromptScoreIndex:
        """
        Class to hold constants. Output of get_prompt_score() method is a list.
        This class stores mapping between output entity and its index in output of get_prompt_score() method.
        """
        PROMPT_STR = 0
        SCORE = 1
        DATASET = 2

    # This has to defined outside of constructor, so that it can be used as decorator.
    iolog = ParamLogger()

    def __init__(self, dataset: List, base_path: str, setup_config: SetupConfig,
                 prompt_pool: HeuristicPromptPool, data_processor: DatasetSpecificProcessing, logger):
        self.dataset = dataset
        self.setup_config = setup_config
        self.data_processor = data_processor
        self.logger = logger
        self.prompt_pool = prompt_pool
        base_path = join(base_path, LogLiterals.DIR_NAME)
        self.iolog.reset_eval_glue(base_path)
        self.conversation_history = []

    @iolog.log_io_params
    def chat_completion(self, user_prompt: str, system_prompt: str = None, model_name: str = None):
        """
        Make a chat completion request to the OpenAI API.

        :param user_prompt: Text spoken by user in a conversation.
        :param system_prompt: Text spoken by system in a conversation.
        :param model_name: The name of the model to use for the completion.
        :return: Output of LLM
        """
        if not system_prompt:
            system_prompt = self.prompt_pool.system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = LLMMgr.chat_completion(messages, model_name=model_name)
        return response

    def chat_completion_history(self, chat_history, system_prompt=None, remember=False, model_name: str = None):
        """
        Calls the LLM with a full chat history (list of dicts).
        Optionally prepends a system prompt.
        If remember=True, stores the conversation in self.conversation_history.
        """
        messages = chat_history.copy()
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        response = LLMMgr.chat_completion(messages, model_name=model_name)

        if remember:
            self.conversation_history = messages + [{"role": "assistant", "content": response}]

        return response

    def improve_prompt(self, current_prompt: str, params: PromptOptimizationParams) -> str:
        """
        Runs one evaluation-refinement cycle over a given prompt.

        Workflow:
        1. Evaluate the current prompt against the seven-criteria rubric.
        2. Use an LLM to generate rubric-based feedback on the prompt.
        3. Refine the prompt using that feedback.
        4. Extract the refined candidate and format it into the final improved version.
        """
        prompt_evaluation = self.prompt_pool.prompt_evaluation.format(prompt=current_prompt)

        chat_history = [
            {"role": "user", "content": prompt_evaluation}
        ]
        eval_response = self.chat_completion_history(chat_history, model_name=self.META_MODEL_NAME)
        chat_history.append({"role": "assistant", "content": eval_response})
        self.logger.info(f"Rubric evaluation: {eval_response}")

        prompt_refinement = self.prompt_pool.prompt_refinement
        chat_history.append({"role": "user", "content": prompt_refinement})
        refined_prompt = self.chat_completion_history(chat_history, model_name=self.META_MODEL_NAME)

        final_best_prompt = re.findall(DatasetSpecificProcessing.TEXT_DELIMITER_PATTERN, refined_prompt)
        final_improved_prompt = self.prompt_pool.improved_prompt.format(instruction=final_best_prompt[0])
        self.logger.info(f"Refined prompt: {final_improved_prompt}")
        return final_improved_prompt

    def improve_prompt_with_score_check(self, initial_prompt: str, params: PromptOptimizationParams) -> str:
        """
        Runs improve_prompt() for params.validation_round rounds, scores each
        refined candidate on the held-out dataset via validate_llm_answer(), and
        returns the best-scoring candidate across all rounds.

        :param initial_prompt: The original prompt to optimize.
        :param params: Parameters controlling the optimization process.
        :return: The best-performing prompt discovered.
        :rtype: str
        """
        best_prompt = initial_prompt
        best_score = float('-inf')

        for attempt in range(params.validation_round):
            self.conversation_history = []
            improved_prompt = self.improve_prompt(initial_prompt, params)

            num_correct = 0
            for example in self.dataset:
                question = example[DatasetSpecificProcessing.QUESTION_LITERAL]
                actual_answer = example[DatasetSpecificProcessing.FINAL_ANSWER_LITERAL]
                result = self.validate_llm_answer(improved_prompt, question, actual_answer)
                if result[self.EvalLiterals.IS_CORRECT]:
                    num_correct += 1

            score = num_correct / len(self.dataset) if self.dataset else 0.0
            self.logger.info(f"Attempt {attempt + 1}: Scored Improved Prompt = {score:.2f}")

            if score > best_score:
                best_score = score
                best_prompt = improved_prompt

        self.logger.info(f"Best score achieved: {best_score:.2f}")
        return best_prompt

    def validate_llm_answer(self, current_prompt: str, question: str, gt_answer: str) -> dict:
        """
        For the given input question, get answer to it from LLM

        :param question: Question to be asked to LLM, to solve
        :param gt_answer: Ground truth, final answer.
        :return:  (is_correct, predicted_ans, llm_output)
                is_correct -> Tells if prediction by LLM was correct.
                predicted_ans -> is the actual predicted answer by LLM.
                llm_output -> Output text generated by LLM for the given question
        :rtype: (bool, str, str)
        """
        final_prompt = self.prompt_pool.eval_prompt.format(instruction=current_prompt, question=question)
        llm_output = self.chat_completion(user_prompt=final_prompt)

        is_correct, predicted_ans = self.data_processor.access_answer(llm_output, gt_answer)
        return {self.EvalLiterals.IS_CORRECT: is_correct,
                self.EvalLiterals.PREDICTED_ANS: predicted_ans,
                self.EvalLiterals.LLM_OUTPUT: llm_output}
