from os import makedirs, replace as os_replace
from os.path import dirname, join
import hashlib
import json
import pickle
import time
from typing import Any

from ..common.base_classes import LLMConfig, SetupConfig
from ..common.constants.log_strings import CommonLogsStr
from ..common.llm.llm_mgr import LLMMgr
from ..common.utils.logging import get_glue_logger, set_logging_config
from ..common.utils.file import read_jsonl, yaml_to_class, yaml_to_dict, read_jsonl_row
from ..paramlogger import ParamLogger
from ..promptopt.constants import PromptOptimizationLiterals
from ..promptopt.techniques.common_logic import DatasetSpecificProcessing
from ..promptopt.utils import get_promptopt_class
import random
import re


class GluePromptOpt:
    """
    This class is trigger point for any prompt optimization method. Different prompt optimization techniques are
    represented by different classes. This class collates all the user configs present in different yaml files and
    other boilerplate code. Any of supported prompt optimization techniques can be triggered by this class.
    """
    BEST_PROMPT = None
    EXPERT_PROFILE = None
    data_processor = None
    iolog = ParamLogger()

    class EvalLiterals:
        IS_CORRECT = "is_correct"
        PREDICTED_ANS = "predicted_ans"
        LLM_OUTPUT = "llm_output"

    def __init__(self,
                 prompt_config_path: str,
                 setup_config_path: str,
                 dataset_jsonl: str,
                 data_processor: DatasetSpecificProcessing,
                 dataset_processor_pkl_path: str = None,
                 prompt_pool_path: str = None,
                 validation_dataset_jsonl: str = None,
                 seed: int = None):
        """
        Collates all the configs present in different yaml files. Initialize logger, de-serialize pickle file that has
        class/method for dataset processing (for given dataset).

        :param llm_config_path: Path to yaml file that has LLM related configs.
        :param prompt_config_path: Path to yaml file that has prompt templates for the given techniques.
        :param setup_config_path: Path to yaml file that has user preferences.
        :param dataset_jsonl: Path to jsonl file that has dataset present in jsonl format. Used as the
        optimizer's own training set.
        :param data_processor: object of DatasetSpecificProcessing class, which has data handling methods which are
        specific to that dataset
        :param dataset_processor_pkl_path: Path to pickle file that has object of class DatasetSpecificProcessing
                                           serialized.
        :param prompt_pool_path: Path to yaml file that has prompts
        :param validation_dataset_jsonl: Path to jsonl file with a dataset disjoint from dataset_jsonl. When
        given, the technique is constructed against this set instead of the training set -- required for
        techniques (e.g. MPIR/heuristic) that score candidates on held-out data rather than on the
        optimizer's own training examples. When omitted, the technique falls back to dataset_jsonl, matching
        prior behavior.
        :param seed: When given, seeds the global `random` module before the technique is constructed.
        Every technique's own randomness (which demos/minibatch to sample, etc.) draws from the global
        `random` module rather than a per-instance RNG, so this is the one place that needs to seed it
        for a run's optimizer-side stochasticity to actually vary with -- and be reproducible from -- the
        seed recorded in results/predictions/*.jsonl (REBUILD.md §4.1, §5). Does not affect the seed used
        for the train/val/test split itself, which data_prep.py seeds independently via its own
        random.Random(seed) instance.
        """
        if seed is not None:
            random.seed(seed)

        if dataset_jsonl != None:
            if data_processor:
                self.data_processor = data_processor
            else:
                with open(dataset_processor_pkl_path, "rb") as file:
                    self.data_processor = pickle.load(file)  # datatype: class DatasetSpecificProcessing

        prompt_config_dict = yaml_to_dict(prompt_config_path)
        prompt_opt_cls, prompt_opt_hyperparam_cls, promptpool_cls = get_promptopt_class(
            prompt_config_dict[PromptOptimizationLiterals.PROMPT_TECHNIQUE_NAME])

        self.setup_config = yaml_to_class(setup_config_path, SetupConfig)
        self.prompt_opt_param = yaml_to_class(prompt_config_path, prompt_opt_hyperparam_cls)
        current_dir = dirname(__file__)
        default_yaml_path = join(current_dir,
                                 "techniques",
                                 prompt_config_dict[PromptOptimizationLiterals.PROMPT_TECHNIQUE_NAME],
                                 "prompt_pool.yaml")

        self.prompt_pool = yaml_to_class(prompt_pool_path, promptpool_cls, default_yaml_path)

        if dataset_jsonl != None:
            dataset = read_jsonl(dataset_jsonl)
        self.prompt_opt_param.answer_format += self.prompt_pool.ans_delimiter_instruction
        base_path = join(self.setup_config.dir_info.base_dir, self.setup_config.experiment_name)
        set_logging_config(join(base_path, self.setup_config.dir_info.log_dir_name),
                           self.setup_config.mode)
        self.logger = get_glue_logger(__name__)

        if dataset_jsonl != None:
            if len(dataset) < self.prompt_opt_param.seen_set_size:
                self.prompt_opt_param.seen_set_size = len(dataset)
                self.logger.info(f"Dataset has {len(dataset)} samples. However values for seen_set_size is "
                                f"{self.prompt_opt_param.seen_set_size}. Hence resetting seen_set_size"
                                f" to {len(dataset)}")

        if self.prompt_opt_param.few_shot_count > self.prompt_opt_param.seen_set_size:
            self.prompt_opt_param.few_shot_count = self.prompt_opt_param.seen_set_size
            self.logger.info(f"Value set for few_shot_count is {self.prompt_opt_param.few_shot_count}. "
                             f"However values for seen_set_size is {self.prompt_opt_param.seen_set_size}. "
                             f"Hence resetting few_shot_count to {self.prompt_opt_param.few_shot_count}")

        if dataset_jsonl != None:
            training_dataset = dataset[:self.prompt_opt_param.seen_set_size]
        else:
            training_dataset = None
        self.logger.info(f"Setup configurations parameters: {self.setup_config} \n{CommonLogsStr.LOG_SEPERATOR}")
        self.logger.info(f"Prompt Optimization parameters: {self.prompt_opt_param} \n{CommonLogsStr.LOG_SEPERATOR}")

        # This iolog is going to be used when doing complete evaluation over test-dataset
        self.iolog.reset_eval_glue(join(base_path, "evaluation"))

        if validation_dataset_jsonl is not None:
            technique_dataset = read_jsonl(validation_dataset_jsonl)
        else:
            technique_dataset = training_dataset

        self.prompt_opt = prompt_opt_cls(technique_dataset, base_path, self.setup_config,
                                         self.prompt_pool, self.data_processor, self.logger)

    def get_best_prompt(self,use_examples=False,run_without_train_examples=False,generate_synthetic_examples=False,resolve_tie_criteria="max") -> (str, Any):
        """
        Call get_best_prompt() method of class PromptOptimizer & return its value.
        :return: (best_prompt, expert_profile)
            best_prompt-> Best prompt for a given task description
            expert_profile-> Description of an expert who is apt to solve the task at hand. LLM would be asked to take
            identity of described in expert_profile.
        """
        start_time = time.time()
        self.BEST_PROMPT, self.EXPERT_PROFILE = self.prompt_opt.get_best_prompt(self.prompt_opt_param,use_examples=use_examples,run_without_train_examples=run_without_train_examples,generate_synthetic_examples=generate_synthetic_examples,resolve_tie_criteria=resolve_tie_criteria)
        self.logger.info(f"Time taken to find best prompt: {(time.time() - start_time)} sec")
        return self.BEST_PROMPT, self.EXPERT_PROFILE

    def evaluate(self, test_dataset_jsonl: str, task_name: str = None, condition_name: str = None,
                 seed: int = None, predictions_dir: str = "results/predictions") -> float:
        """
        Evaluate the performance of self.BEST_PROMPT over test dataset. Return the accuracy.

        Besides the existing per-run iolog dump (under the gitignored logs/ dir), this writes one
        tracked JSONL file per (task_name, condition_name, seed) under predictions_dir -- one row per
        example with the fields needed to recompute any reported number and to run example-level
        paired significance tests. task_name/condition_name/seed are optional so ad-hoc calls (e.g. a
        single demo run without a grid identity) keep working; they should be supplied for anything
        feeding into REBUILD.md's analysis.

        :param test_dataset_jsonl: Path to jsonl file that has test dataset
        :param task_name: BBH task group this run belongs to (e.g. "hyperbaton").
        :param condition_name: Grid condition identity (e.g. "protegi_mpir").
        :param seed: Seed used for this run, if the condition is seeded.
        :param predictions_dir: Tracked directory to write the per-example JSONL into.
        :return: Percentage accuracy
        """

        start_time = time.time()
        self.logger.info(f"Evaluation started {CommonLogsStr.LOG_SEPERATOR}")
        if not self.BEST_PROMPT:
            self.logger.error("BEST_PROMPT attribute is not set. Please set self.BEST_PROMPT attribute of this object, "
                              "either manually or by calling get_best_prompt() method.")
            return

        prompt_hash = hashlib.sha256(self.BEST_PROMPT.encode("utf-8")).hexdigest()[:16]

        makedirs(predictions_dir, exist_ok=True)
        predictions_file_name = "_".join(str(part) for part in
                                          [task_name, condition_name, seed] if part is not None) \
            or f"eval_result_{self.setup_config.experiment_name}"
        predictions_path = join(predictions_dir, f"{predictions_file_name}.jsonl")
        # Written to a temp path and renamed into place only after the full
        # loop completes -- a crash partway through (e.g. a call exhausting
        # LLMMgr's retries) previously left an empty/partial file at
        # predictions_path indistinguishable from a completed run, which a
        # resumable caller's "does the output file exist" check (e.g.
        # demos/run_grid.py) would then silently treat as done and skip
        # forever. Found when a real grid run crashed on its very first call
        # (a dotenv-loading bug in run_grid.py, since fixed) and left three
        # empty *.jsonl files that would have been silently skipped on retry.
        tmp_predictions_path = predictions_path + ".tmp"

        total_correct = 0
        total_count = 0
        with open(tmp_predictions_path, "w", encoding="utf-8") as predictions_file:
            for example_index, json_obj in enumerate(read_jsonl_row(test_dataset_jsonl)):
                answer = self.predict_and_access(json_obj[DatasetSpecificProcessing.QUESTION_LITERAL],
                                                 json_obj[DatasetSpecificProcessing.FINAL_ANSWER_LITERAL])

                total_correct += answer[self.EvalLiterals.IS_CORRECT]
                total_count += 1
                result = {"accuracy": f"{total_correct}/{total_count} : {total_correct/total_count*100.0}%",
                          "predicted": answer[self.EvalLiterals.PREDICTED_ANS],
                          "actual": json_obj[DatasetSpecificProcessing.FINAL_ANSWER_LITERAL],
                          "llm_output": answer[self.EvalLiterals.LLM_OUTPUT],
                          "question": json_obj[DatasetSpecificProcessing.QUESTION_LITERAL],}
                self.iolog.append_dict_to_chained_logs(result)
                self.logger.info(result)

                prediction_row = {
                    "task": task_name,
                    "condition": condition_name,
                    "seed": seed,
                    "example_index": example_index,
                    "prompt_hash": prompt_hash,
                    "question": json_obj[DatasetSpecificProcessing.QUESTION_LITERAL],
                    "llm_output": answer[self.EvalLiterals.LLM_OUTPUT],
                    "predicted": answer[self.EvalLiterals.PREDICTED_ANS],
                    "actual": json_obj[DatasetSpecificProcessing.FINAL_ANSWER_LITERAL],
                    "is_correct": bool(answer[self.EvalLiterals.IS_CORRECT]),
                }
                predictions_file.write(json.dumps(prediction_row) + "\n")

        # Only reached if the loop above completed without raising -- promote
        # the temp file to its real name so it's now safe for a caller to
        # treat predictions_path's existence as "this condition is done".
        os_replace(tmp_predictions_path, predictions_path)

        self.iolog.dump_chained_log_to_file(file_name=f"eval_result_{self.setup_config.experiment_name}")
        self.logger.info(f"Time taken for evaluation: {(time.time() - start_time)} sec")
        return total_correct / total_count

    @iolog.log_io_params
    def predict_and_access(self, question: str, gt_answer: str) -> (bool, str, str):
        """
        For the given input question, get answer to it from LLM, using the BEST_PROMPT & EXPERT_PROFILE
        computes earlier.

        :param question: Question to be asked to LLM, to solve
        :param gt_answer: Ground truth, final answer.
        :return:  (is_correct, predicted_ans, llm_output)
                is_correct -> Tells if prediction by LLM was correct.
                predicted_ans -> is the actual predicted answer by LLM.
                llm_output -> Output text generated by LLM for the given question
        :rtype: (bool, str, str)
        """
        final_prompt = self.prompt_pool.eval_prompt.format(instruction=self.BEST_PROMPT,
                                                           question=question)
        #print(final_prompt)
        llm_output = self.prompt_opt.chat_completion(user_prompt=final_prompt, system_prompt=self.EXPERT_PROFILE)
        #print(llm_output)
        
        is_correct, predicted_ans = self.data_processor.access_answer(llm_output, gt_answer)
        return {self.EvalLiterals.IS_CORRECT: is_correct,
                self.EvalLiterals.PREDICTED_ANS: predicted_ans,
                self.EvalLiterals.LLM_OUTPUT: llm_output}

    def evaluate_promt(self, current_prompt: str) -> (str, str):
        self.score = self.prompt_opt.evaluate_promt(current_prompt)
        return self.score

    def improve_prompt(self, current_prompt: str) -> (str, str):
        self.improved_prompt = self.prompt_opt.improve_prompt_with_score_check(current_prompt,self.prompt_opt_param)
        return self.improved_prompt



