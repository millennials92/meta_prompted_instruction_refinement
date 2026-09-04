"""Shared BBH DatasetSpecificProcessing implementation for every optimizer
notebook (PromptWizard, APE, ProTeGi, MPIR).

Previously this class (plus its extract_between/llm_eval helpers) was
duplicated verbatim in MPIR.ipynb and promptwizard.ipynb. Factored out here so
the exact-match-by-default fix (REBUILD.md §2.2) and the LLM-judge retarget
(HANDOFF-GPU.md §2) live in one place instead of drifting across copies as
more notebooks (ape.ipynb, protegi.ipynb) are added.
"""
from typing import Any

from tqdm import tqdm

from promptwizard.glue.common.llm.llm_mgr import call_api
from promptwizard.glue.promptopt.techniques.common_logic import DatasetSpecificProcessing
from promptwizard.glue.common.utils.file import save_jsonlist

# REBUILD.md §2.2 / HANDOFF-GPU.md §3: exact match is the default and primary
# metric. The manuscript's §4.4 describes exact-match accuracy; the LLM-judge
# branch below is kept for genuinely free-form tasks but must be opted into
# explicitly rather than silently driving the reported numbers.
LLM_AS_JUDGE_EVAL = False


def extract_between(start: str, end: str, text: str) -> str:
    """
    Extract the substring of `text` between the first occurrence of `start`
    and the following occurrence of `end`. Returns "" if either is absent.
    """
    start_index = text.find(start)
    if start_index == -1:
        return ''

    start_index += len(start)

    end_index = text.find(end, start_index)
    if end_index == -1:
        return ''
    return text[start_index:end_index]


def llm_eval(predicted_answer: str, gt_answer: str) -> bool:
    """
    LLM-as-judge equivalence check, used only when LLM_AS_JUDGE_EVAL is True.
    Routes through llm_mgr.call_api, which honors LOCAL_OPENAI_BASE_URL when
    set, so the judge follows the same retarget as everything else instead of
    a hardcoded OpenAI/Azure path.
    """
    eval_prompt = f"""Given the Predicted_Answer and Reference_Answer, compare them and check they mean the same.
                    If they mean the same then return True between <ANS_START> and <ANS_END> tags ,
                    If they differ in the meaning then return False between <ANS_START> and <ANS_END> tags
                    Following are the given :
                    Predicted_Answer: {predicted_answer}
                    Reference_Answer: {gt_answer}"""
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": eval_prompt}
    ]

    response = call_api(messages)
    final_judgement = extract_between(start="<ANS_START>", end="<ANS_END>", text=response)
    return final_judgement == "True"


class BBH(DatasetSpecificProcessing):

    def dataset_to_jsonl(self, dataset_jsonl: str, **kwargs: Any) -> None:
        def extract_answer_from_output(completion):
            return completion

        examples_set = []

        for _, sample in tqdm(enumerate(kwargs["dataset"]), desc="Evaluating samples"):
            example = {
                DatasetSpecificProcessing.QUESTION_LITERAL: sample['question'],
                DatasetSpecificProcessing.ANSWER_WITH_REASON_LITERAL: sample['answer'],
                DatasetSpecificProcessing.FINAL_ANSWER_LITERAL: extract_answer_from_output(sample["answer"])
            }
            examples_set.append(example)

        save_jsonlist(dataset_jsonl, examples_set, "w")

    def extract_final_answer(self, answer: str):
        # .strip() matters: models routinely wrap the answer with incidental
        # whitespace/newlines inside the delimiter tags (observed with Qwen3,
        # e.g. "<ANS_START>\n4\n<ANS_END>") -- without stripping, exact-match
        # comparison silently marks every such answer wrong regardless of
        # correctness, deflating every condition's measured accuracy uniformly
        # and invisibly. Found while validating against a live model server.
        return extract_between(text=answer, start="<ANS_START>", end="<ANS_END>").strip()

    @staticmethod
    def _normalize_for_comparison(text: str) -> str:
        # BBH multiple-choice ground truth is formatted "(A)", but a model
        # told to "wrap the final answer" sometimes wraps only the bare
        # letter ("A") instead of the full "(A)" option identifier --
        # observed live with Qwen3 on otherwise-correct reasoning. Since
        # parentheses are pure formatting around a multiple-choice letter and
        # never semantically meaningful BBH content, stripping them from both
        # sides before comparing fixes this without risking conflating
        # genuinely different answers (numeric/yes-no answers never contain
        # parentheses to begin with, so this is a no-op for them).
        return text.strip().replace("(", "").replace(")", "").lower()

    def access_answer(self, llm_output: str, gt_answer: str):
        predicted_answer = self.extract_final_answer(llm_output)
        if LLM_AS_JUDGE_EVAL:
            is_correct = llm_eval(predicted_answer, gt_answer)
        else:
            is_correct = bool(predicted_answer) and (
                self._normalize_for_comparison(predicted_answer) == self._normalize_for_comparison(gt_answer))

        return is_correct, predicted_answer
