import typing as tp

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.data import Message
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform


class MultiHopProblem(tp.TypedDict):
    question: str
    answers: tp.List[str]
    final_answer: str
 

class MultihopTransform(Transform):
    def __call__(self, sample: tp.Mapping[str, tp.Any]) -> MultiHopProblem:
        ...


class MultiHopDataset(Dataset):
    def __init__(
        self,
        source: str,
        sample_transform: MultihopTransform,
        filter_fn: tp.Optional[tp.Callable] = None,
        system_prompt: tp.Optional[str] = None,
        **load_dataset_kwargs,
    ) -> None:
        self._data = load_dataset(path=source, **load_dataset_kwargs)
        self._sample_transform = sample_transform
        self._system_prompt = system_prompt

        if filter_fn is not None:
            self.data = self.data.filter(filter_fn)

    def setup(self, tokenizer: ModelTokenizer):
        self._tokenizer = tokenizer

    def _tokenize_question(self, question: str) -> tp.List[int]:
        """
        Tokenize a question possibly adding a system_prompt.
        """
        messages = []
        if self._system_prompt is not None:
            messages.append(
                Message(
                    role="system",
                    content=self._system_prompt,
                    eot=True
                )
            )
        messages.append(
            Message(
                role="user",
                content=question,
                eot=True,
            )
        )

        tokens = self._tokenizer.tokenize_messages(
            messages=messages,
            add_generation_prompt=True
        )

        return tokens

    def __getitem__(self, index) -> tp.Dict[str, tp.Any]:
        sample = self._sample_transform(self._data[index])
        tokens = self._tokenize_question(sample["question"])
        return {
            "tokens": tokens,
            "answers": sample["answers"],
            "final_answer": sample["final_answer"]
        }

    def __len__(self) -> int:
        return len(self._data)

