import typing as tp

from datasets import load_dataset

import torch
from torch.utils.data import Dataset

from torchtune.data import Message, left_pad_sequence, truncate
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform


GSM8K_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The "
    "reasoning process and answer are enclosed within <think></think> and "
    "<answer></answer> tags, respectively, i.e., <think>reasoning process "
    "here</think> <answer>answer here</answer>."
)


class QAProblem(tp.TypedDict):
    question: str
    answer: str


class ReasoningProblem(QAProblem):
    path: str


class QATransform(Transform):
    def __call__(self, sample: tp.Mapping[str, tp.Any]) -> QAProblem:
        pass


class QAGSM8K(Transform):
    """
    Parses GSM8K record into question, chain of thoughts and answer fields.
    """
    def __call__(
        self,
        sample: tp.Mapping[str, tp.Any]
    ) -> QATransform:

        question = sample["question"]
        solution = sample["answer"]
        path, answer = solution.split("#### ")

        return ReasoningProblem(
            question = question,
            answer = answer,
            path = path
        )


class LeftPadCollator:
    def __init__(
        self,
        tokens_key: str = "tokens",
        pad_token: int = 0
    ) -> None:
        self.tokens_key = tokens_key
        self.pad_token = pad_token

    def __call__(
        self,
        batch: tp.List[tp.Dict[str, tp.List[int]]],
    ) -> tp.Dict[str, tp.Union[torch.Tensor, tp.List[str]]]:

        non_token_keys = [k for k in batch[0].keys() if k != self.tokens_key]
        collated_batch = {
            k : [row[k] for row in batch] for k in non_token_keys
        }
        collated_tokens = left_pad_sequence(
            [torch.tensor(row[self.tokens_key]) for row in batch],
            batch_first=True,
            padding_value=self.pad_token
        )
        collated_batch[self.tokens_key] = collated_tokens.long()
        return collated_batch


class QADataset(Dataset):
    """
    Question Answering dataset class.
    """
    def __init__(
        self,
        tokenizer: ModelTokenizer,
        *,
        source: str,
        sample_transform: QATransform,
        system_prompt: str = "",
        **load_dataset_kwargs: tp.Dict[str, tp.Any],
    ) -> None:

        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.sample_transform = sample_transform
        self.data = load_dataset(source, **load_dataset_kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> tp.Dict[str, tp.Any]:
        sample = self.sample_transform(self.data[index])
        question = sample["question"]
        answer = sample["answer"]

        messages = [
            Message(
                role="system",
                content=self.system_prompt,
                eot=True,
            ),
            Message(
                role="user",
                content=question,
                eot=True,
            ),
        ]

        max_seq_len = self.tokenizer.max_seq_len
        tokens = [self.tokenizer.bos_id]
        for message in messages:
            tokenized_message = self.tokenizer.tokenize_message(
                message, add_end_tokens=False
            )
            tokens = tokens + tokenized_message
            if max_seq_len and len(tokens) >= max_seq_len:
                break

        generation_prompt = self.tokenizer._tokenize_header(Message(
            role="assistant", content=""
        ))
        tokens = tokens + generation_prompt

        if max_seq_len:
            tokens = truncate(
                tokens, max_seq_len
            )
        return {"tokens": tokens, "answers": answer}


def gsm8k_dataset(
    tokenizer: ModelTokenizer,
    *,
    split: str = "train",
    **load_dataset_kwargs: tp.Dict[str, tp.Any],
) -> QADataset:
    """
    GSM8k dataset from OpenAI.
    """
    return QADataset(
        tokenizer=tokenizer,
        source="openai/gsm8k",
        system_prompt=GSM8K_SYSTEM_PROMPT,
        sample_transform=QAGSM8K(),
        split=split,
        name="main",
        **load_dataset_kwargs,
    )
