import random
import re
from os.path import join
from tqdm import tqdm
from typing import Any, Dict, List
import json

from ....paramlogger import ParamLogger
from ....paramlogger.constants import LogLiterals
from ....common.base_classes import SetupConfig, UniversalBaseClass
from ....common.llm.llm_mgr import LLMMgr
from ....common.constants.log_strings import CommonLogsStr
from ...constants import PromptOptimizationParams, SupportedPromptOpt
from ...techniques.common_logic import DatasetSpecificProcessing, PromptOptimizer
from ...techniques.heuristic.base_classes import HeuristicPromptPool
from ....common.utils.file import read_jsonl, yaml_to_class, yaml_to_dict, read_jsonl_row


def extract_between(start, end, text):
    """
    Extracts the substring from 'text' that is between 'start' and 'end' strings.
    
    Parameters:
    - start (str): The starting delimiter string.
    - end (str): The ending delimiter string.
    - text (str): The text to search within.
    
    Returns:
    - str: The extracted substring between the start and end delimiters.
    """
    start_index = text.find(start)
    if start_index == -1:
        return '' 
    
    start_index += len(start)
    
    end_index = text.find(end, start_index)
    if end_index == -1:
        return ''  
    return text[start_index:end_index]


class Heuristic(PromptOptimizer, UniversalBaseClass):
    """
    TODO: Explain this method
    """

    TECHNIQUE_NAME = SupportedPromptOpt.HEURISTIC.value

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
        # Optionally prepend system prompt
        messages = chat_history.copy()
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        # Call the LLM manager with the full message list
        response = LLMMgr.chat_completion(messages, model_name=model_name)

        # Optionally remember the conversation
        if remember:
            if not hasattr(self, "conversation_history"):
                self.conversation_history = []
            self.conversation_history = messages + [{"role": "assistant", "content": response}]

        return response
    


     
    def get_best_prompt(self, params: PromptOptimizationParams) -> (str, str):
        """
        Placeholder method for getting the best prompt.
        """
        return None
    
    def improve_prompt(self, current_prompt: str,  params: PromptOptimizationParams) -> (str, str):
        """
        Iteratively improves a given prompt through evaluation and refinement.

        Workflow:
        1. Evaluate the current prompt using a predefined evaluation template.
        2. Use an LLM to generate feedback on the prompt.
        3. Refine the prompt using the feedback and a refinement template.
        4. Extract the best candidate prompt and format it into the final improved version.
        """
        # Evaluate the current prompt
        prompt_evaluation = self.prompt_pool.prompt_evaluation.\
            format(prompt=current_prompt)

        chat_history = [
            {"role": "user", "content": prompt_evaluation}
        ]
        # Generate feedback on the prompt.
        eval_response = self.chat_completion_history(chat_history,model_name="gpt-4o")
        chat_history.append({"role": "assistant", "content": eval_response})
        print("evaluation: ", eval_response)
         # Refine the prompt 
        prompt_refinement = self.prompt_pool.prompt_refinement
        chat_history.append({"role": "user", "content": prompt_refinement})
        refined_prompt = self.chat_completion_history(chat_history, model_name="gpt-4o")
        # Extract the best candidate prompt
        final_best_prompt = re.findall(DatasetSpecificProcessing.TEXT_DELIMITER_PATTERN,refined_prompt)
        final_improved_prompt = self.prompt_pool.improved_prompt.format(
                instruction=final_best_prompt[0])
        print(final_improved_prompt)
        return final_improved_prompt

        
    
       

    def improve_prompt_with_score_check(self, initial_prompt: str, params: PromptOptimizationParams,
                                    score_threshold: float = 0.5) -> str:
        """
        Iteratively improves a prompt while scoring it against a dataset to 
        select the best-performing version.

        Workflow:
        1. Start with an initial prompt.
        2. Repeatedly call improve_prompt() to generate refined versions.
        3. Evaluate each refined prompt against the dataset using validate_llm_answer().
        4. Track the best-scoring prompt across iterations.
        5. (Optional) Stop early if a score threshold is reached.

        :param initial_prompt: The original prompt to optimize.
        :param params: Parameters controlling the optimization process.
        :param score_threshold: Minimum acceptable score to stop early (default=0.5).
        :return: The best-performing prompt discovered.
        :rtype: str
        """
        best_prompt = initial_prompt
        best_score = float('-inf')

        for attempt in range(params.validation_round):
            self.conversation_history = []

            # Call original improve_prompt()
            improved_prompt = self.improve_prompt(initial_prompt, params)

            # Score the improved prompt using dataset sampling (like get_prompt_score)
            total_correct = 0
            total_questions = 0
            
            # Sample dataset and evaluate like in get_prompt_score
            for _ in range(1):
                #dataset_subset = random.sample(self.dataset, 25)
                dataset_subset = self.dataset
                
                
                # Evaluate each question individually
                batch_correct = 0
                for i, example in enumerate(dataset_subset):
                    question = example[DatasetSpecificProcessing.QUESTION_LITERAL]
                    actual_answer = example[DatasetSpecificProcessing.FINAL_ANSWER_LITERAL]
                    
                    # Use validate_llm_answer to evaluate individual question
                    result = self.validate_llm_answer(improved_prompt,question, actual_answer)
                    print(result)
                    if result[self.EvalLiterals.IS_CORRECT]:
                        batch_correct += 1
                
                num_correct = batch_correct
                num_questions = len(dataset_subset)
                
                total_correct += num_correct
                total_questions += num_questions
            
            score = total_correct / total_questions if total_questions > 0 else 0.0

            print(f"Attempt {attempt + 1}: Scored Improved Prompt = {score:.2f}")

            # Update best prompt if this one is better
            if score > best_score:
                best_score = score
                best_prompt = improved_prompt
            '''
            # Break early if score meets/exceeds threshold
            if best_score >= score_threshold:
                print(f"Threshold of {score_threshold:.2f} reached, stopping early.")
                break
            '''
        print(f"Best score achieved: {best_score:.2f}")
        return best_prompt
    
    def validate_llm_answer(self,current_prompt: str, question: str, gt_answer: str) -> (bool, str, str):
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
        final_prompt = self.prompt_pool.eval_prompt.format(instruction=current_prompt,
                                                           question=question)
        llm_output = self.chat_completion(user_prompt=final_prompt)
        
        is_correct, predicted_ans = self.data_processor.access_answer(llm_output, gt_answer)
        return {self.EvalLiterals.IS_CORRECT: is_correct,
                self.EvalLiterals.PREDICTED_ANS: predicted_ans,
                self.EvalLiterals.LLM_OUTPUT: llm_output}
    

    
    



