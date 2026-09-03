import random
import re
from os.path import join
from typing import Any, List, Tuple

from ....paramlogger import ParamLogger
from ....paramlogger.constants import LogLiterals
from ....common.base_classes import SetupConfig, UniversalBaseClass
from ....common.llm.llm_mgr import LLMMgr
from ...constants import PromptOptimizationParams, SupportedPromptOpt
from ...techniques.common_logic import DatasetSpecificProcessing, PromptOptimizer
from ...techniques.protegi.base_classes import ProTeGiPromptPool


class ProTeGi(PromptOptimizer, UniversalBaseClass):
    """
    ProTeGi (Pryzant et al., 2023): treats a natural-language critique of a
    prompt's errors on a minibatch as a textual "gradient", edits the prompt in
    the direction the critique suggests, paraphrases each edit to widen the
    successor pool, and runs beam search -- scoring every successor on a
    validation minibatch each round and keeping the top `beam_width` -- for
    `num_steps` rounds.

    Simplification vs. the original paper: candidates are selected by scoring
    every successor on a full minibatch each round, rather than the paper's
    UCB-bandit sampling scheme. That scheme exists to cut evaluation cost over
    large candidate pools; at the pool sizes used here (num_edits_per_gradient x
    num_paraphrases_per_edit successors per beam member) full-batch scoring is
    tractable and keeps the selection logic auditable.
    """

    TECHNIQUE_NAME = SupportedPromptOpt.PROTEGI.value

    class EvalLiterals:
        IS_CORRECT = "is_correct"
        PREDICTED_ANS = "predicted_ans"
        LLM_OUTPUT = "llm_output"

    # This has to be defined outside of constructor, so that it can be used as decorator.
    iolog = ParamLogger()

    def __init__(self, dataset: List, base_path: str, setup_config: SetupConfig,
                 prompt_pool: ProTeGiPromptPool, data_processor: DatasetSpecificProcessing, logger):
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
    def evaluate_on_batch(self, instruction: str, batch: List[dict]) -> Tuple[float, List[dict]]:
        """
        Run `instruction` over every example in `batch`. Returns both the
        accuracy (used to score candidates) and the subset of examples answered
        incorrectly (used to seed the next textual gradient) from a single pass,
        so a round never queries the LLM twice for the same (instruction, batch).

        :param instruction: Instruction to evaluate.
        :param batch: Examples to evaluate over.
        :return: (accuracy, wrong_examples)
        """
        if not batch:
            return 0.0, []

        correct_count = 0
        wrong_examples = []
        for example in batch:
            question = example[DatasetSpecificProcessing.QUESTION_LITERAL]
            actual_answer = example[DatasetSpecificProcessing.FINAL_ANSWER_LITERAL]
            eval_prompt = self.prompt_pool.eval_prompt.format(instruction=instruction, question=question)
            llm_output = self.chat_completion(eval_prompt)
            is_correct, _ = self.data_processor.access_answer(llm_output, actual_answer)
            if is_correct:
                correct_count += 1
            else:
                wrong_examples.append(example)

        return correct_count / len(batch), wrong_examples

    @iolog.log_io_params
    def get_textual_gradient(self, instruction: str, error_examples: List[dict]) -> str:
        """
        Ask the LLM to diagnose why `instruction` fails on `error_examples`
        (Pryzant et al. 2023, §3.1's "gradient").

        :param instruction: Instruction being critiqued.
        :param error_examples: Examples the instruction answered incorrectly.
        :return: Natural-language critique.
        """
        error_string = self.data_processor.collate_to_str(error_examples, self.prompt_pool.quest_reason_ans)
        gradient_prompt = self.prompt_pool.gradient_template.format(instruction=instruction, errors=error_string)
        return self.chat_completion(gradient_prompt)

    @iolog.log_io_params
    def edit_with_gradient(self, instruction: str, gradient: str, params: PromptOptimizationParams) -> List[str]:
        """
        Edit `instruction` in the direction `gradient` suggests, producing
        `params.num_edits_per_gradient` candidate edits (Pryzant et al. 2023,
        §3.1's "edit").

        :param instruction: Instruction being edited.
        :param gradient: Critique produced by get_textual_gradient().
        :param params: Hyperparameters for this optimization run.
        :return: List of edited candidate instructions.
        """
        edit_prompt = self.prompt_pool.edit_template.format(
            instruction=instruction, gradient=gradient, num_edits=params.num_edits_per_gradient)
        response = self.chat_completion(edit_prompt)
        matches = re.findall(DatasetSpecificProcessing.TEXT_DELIMITER_PATTERN, response)
        return [match.strip() for match in matches] if matches else [instruction]

    @iolog.log_io_params
    def paraphrase(self, instruction: str) -> str:
        """
        Paraphrase `instruction`, widening the successor pool beyond literal
        gradient edits (Pryzant et al. 2023, §3.1's Monte Carlo paraphrasing).

        :param instruction: Instruction to paraphrase.
        :return: Paraphrased instruction, or the original if parsing fails.
        """
        paraphrase_prompt = self.prompt_pool.paraphrase_template.format(instruction=instruction)
        response = self.chat_completion(paraphrase_prompt)
        matches = re.findall(DatasetSpecificProcessing.TEXT_DELIMITER_PATTERN, response)
        return matches[0].strip() if matches else instruction

    @iolog.log_io_params
    def expand_candidate(self, instruction: str, params: PromptOptimizationParams) -> List[str]:
        """
        One full expansion step (Pryzant et al. 2023, §3.2's expand()): sample a
        minibatch, collect the errors `instruction` makes on it, turn those
        errors into a textual gradient, edit the instruction accordingly, and
        paraphrase each edit.

        :param instruction: Beam member being expanded.
        :param params: Hyperparameters for this optimization run.
        :return: List of successor candidate instructions (edits + paraphrases).
        """
        minibatch = random.sample(self.dataset, min(params.minibatch_size, len(self.dataset)))
        _, wrong_examples = self.evaluate_on_batch(instruction, minibatch)
        if not wrong_examples:
            return [instruction]

        gradient = self.get_textual_gradient(instruction, wrong_examples)
        edited_candidates = self.edit_with_gradient(instruction, gradient, params)

        expanded = list(edited_candidates)
        for candidate in edited_candidates:
            for _ in range(params.num_paraphrases_per_edit):
                expanded.append(self.paraphrase(candidate))

        return expanded

    def get_best_prompt(self, params: PromptOptimizationParams, **kwargs) -> (str, Any):
        """
        Run `params.num_steps` rounds of beam search: expand every beam member
        into successor candidates, score the union of beam + successors on a
        fresh validation minibatch, and keep the top `params.beam_width`. Return
        the best-scoring instruction seen across all rounds.

        :param params: Object of class ProTeGiParams with hyperparameters for this run.
        :return: (best_prompt, expert_profile)
        """
        beam = [params.base_instruction]
        best_instruction, best_score = params.base_instruction, float("-inf")

        for step in range(params.num_steps):
            candidates = list(beam)
            for instruction in beam:
                candidates.extend(self.expand_candidate(instruction, params))
            candidates = list(dict.fromkeys(candidates))  # de-duplicate, preserve order

            eval_batch = random.sample(self.dataset, min(params.eval_batch_size, len(self.dataset)))
            scored = [(candidate, self.evaluate_on_batch(candidate, eval_batch)[0]) for candidate in candidates]
            scored.sort(key=lambda item: item[1], reverse=True)

            beam = [candidate for candidate, _ in scored[:params.beam_width]]
            step_best_instruction, step_best_score = scored[0]
            if step_best_score > best_score:
                best_instruction, best_score = step_best_instruction, step_best_score
            self.logger.info(f"ProTeGi step {step + 1}: beam best score = {step_best_score:.2f}")

        self.logger.info(f"ProTeGi final best score: {best_score:.2f}, instruction: {best_instruction}")

        final_best_prompt = self.prompt_pool.final_prompt.format(
            instruction=best_instruction,
            answer_format=params.answer_format,
            few_shot_examples="")

        return final_best_prompt, self.prompt_pool.system_prompt
