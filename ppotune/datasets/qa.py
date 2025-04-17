import typing as tp
from torch.utils.data import Dataset

from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform

from torchtune.data import Message
from datasets import load_dataset

from ppotune.datasets.utils import PromptTemplate, apply_prompt_template


class QAProblem(tp.TypedDict):
    question: str
    answer: str

class QATransform(Transform):
    def __call__(self, sample: tp.Mapping[str, tp.Any]) -> QAProblem:
        pass

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
        system_prompt: tp.Optional[str] = None,
        prompt_template: tp.Optional[PromptTemplate] = None,
        add_generation_prompt: bool = False,
        **load_dataset_kwargs: tp.Dict[str, tp.Any],
    ) -> None:

        self.tokenizer = tokenizer
        self.sample_transform = sample_transform
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.add_generation_prompt = add_generation_prompt
        self.data = load_dataset(source, **load_dataset_kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> tp.Dict[str, tp.Any]:
        sample = self.sample_transform(self.data[index])
        question = sample["question"]
        answer = sample["answer"]

        messages = []
        if self.system_prompt is not None:
            messages.append(Message(
                role="system",
                content=self.system_prompt,
                eot=True,
            ))

        messages.append(Message(
            role="user",
            content=question,
            eot=True,
        ))

        tokens = []
        if self.prompt_template is None:
            tokens = self.tokenizer.tokenize_messages(
                messages=messages,
                add_eos=False,
                add_generation_prompt=True
            )
        else:
            text = apply_prompt_template(
                template=self.prompt_template,
                messages=messages,
                add_generation_prompt=True
            )
            tokens = self.tokenizer.encode(text, add_eos=False)

        return {"tokens": tokens, "answers": answer}
