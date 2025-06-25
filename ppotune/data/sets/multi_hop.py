import typing as tp
from functools import partial

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.data import Message
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform

MULTI_HOP_SYSTEM_PROMPT = """You are a chain-of-thought language model. When the user asks a question you MUST reply in the structure below:
<think>
<question> <first self-generated sub-question> </question> <answer> <answer to the first sub-question> </answer>
<question> <second self-generated sub-question> </question> <answer> <answer to the second sub-question> </answer>
...
</think>
<answer> <final answer to the user’s original question> </answer>

Rules
1. Ask yourself sub-questions and answer them, wrap questions and answers in the indicated tags.
2. All inner tags (<question> / <answer>) live **inside** a single <think> ... </think> block.
3. After the </think> tag, output one—and only one—final answer to the user question, wrapped in its own outer <answer> ... </answer> tag.
4. Do not reveal any additional text, commentary, or tags outside those shown above.
5. Preserve the tag names and their order precisely as specified.
"""


class MultiHopProblem(tp.TypedDict):
    question: str
    answers: tp.List[str]
    final_answer: tp.List[str]
 

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


class TwoHopTransform(MultihopTransform):
    def __call__(self, sample: tp.Mapping[str, tp.Any]) -> MultiHopProblem:
        question = sample["generated_question"]
        answers = []
        answers.append(sample["second_entity_aliases"])
        answers.append(sample["third_entity_aliases"])
        final_answer = sample["third_entity_aliases"]

        return MultiHopProblem(question=question, answers=answers, final_answer=final_answer)


class ThreeHopTransform(MultihopTransform):
    def __call__(self, sample: tp.Mapping[str, tp.Any]) -> MultiHopProblem:
        question = sample["generated_question"]
        answers = []
        answers.append(sample["second_entity_aliases"])
        answers.append(sample["third_entity_aliases"])
        answers.append(sample["fourth_entity_aliases"])
        final_answer = sample["fourth_entity_aliases"]

        return MultiHopProblem(question=question, answers=answers, final_answer=final_answer)


two_hop_dataset = partial(
    MultiHopDataset,
    sample_transform=TwoHopTransform(),
    system_prompt=MULTI_HOP_SYSTEM_PROMPT,
)
three_hop_dataset = partial(
    MultiHopDataset,
    sample_transform=ThreeHopTransform(),
    system_prompt=MULTI_HOP_SYSTEM_PROMPT,
)

