import typing as tp

from torchtune.modules.tokenizers import ModelTokenizer

from ppotune.data.utils import PromptTemplate, PrefixSuffix
from ppotune.data.sets.qa import QAProblem, QATransform, QADataset


GSM8K_SYSTEM_PROMPT: str = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The "
    "reasoning process and answer are enclosed within <think></think> and "
    "<answer></answer> tags, respectively, i.e., <think>reasoning process "
    "here</think> <answer>answer here</answer>."
)

GSM8K_PROMPT_TEMPLATE: PromptTemplate = {
    "system": PrefixSuffix("", " "),
    "user":  PrefixSuffix("User: ", " "),
    "assistant":  PrefixSuffix("Assistant: ", "")
}

class GSM8K(QATransform):
    """
    Parses GSM8K record into question, chain of thoughts and answer fields.
    """
    def __call__(
        self,
        sample: tp.Mapping[str, tp.Any]
    ) -> QAProblem:

        question = sample["question"]
        solution = sample["answer"]
        cot, answer = solution.split("#### ")

        return QAProblem(
            question = question,
            cot = cot,
            answer = answer,
        )


class GSM8KEval(QATransform): # TODO: DEPRECATE
    """
    Keeps full answer as in the dataset.
    """
    def __call__(
        self,
        sample: tp.Mapping[str, tp.Any]
    ) -> QAProblem:

        question = sample["question"]
        answer = sample["answer"]

        return QAProblem(
            question = question,
            answer = answer,
        )


def gsm8k_dataset(
    split: str = "train",
    prompt_template: tp.Optional[PromptTemplate] = None,
    **load_dataset_kwargs: tp.Dict[str, tp.Any],
) -> QADataset:
    """
    GSM8k dataset from OpenAI.
    """
    return QADataset(
        source="openai/gsm8k",
        sample_transform=GSM8K(),
        system_prompt=GSM8K_SYSTEM_PROMPT,
        prompt_template=prompt_template,
        split=split,
        name="main",
        **load_dataset_kwargs,
    )

def chat_gsm8k_dataset(
    split: str = "train"
) -> QADataset:
    """
    GSM8K dataset for chat models.
    """
    return gsm8k_dataset(
        split=split,
        prompt_template=None
    )


def plain_gsm8k_dataset(
    split: str = "train"
) -> QADataset:
    """
    GSM8K dataset for non-chat models.
    """
    return gsm8k_dataset(
        split=split,
        prompt_template=GSM8K_PROMPT_TEMPLATE
    )


def eval_gsm8k_dataset( # TODO: DEPRECATE
    split: str = "test",
    **load_dataset_kwargs: tp.Dict[str, tp.Any],
) -> QADataset:
    """
    GSM8k dataset for evaluation.
    """
    return QADataset(
        source="openai/gsm8k",
        sample_transform=GSM8KEval(),
        system_prompt=GSM8K_SYSTEM_PROMPT,
        prompt_template=GSM8K_PROMPT_TEMPLATE,
        split=split,
        name="main",
        **load_dataset_kwargs,
    )
