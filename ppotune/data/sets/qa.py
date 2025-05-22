import typing as tp
from torch.utils.data import Dataset

from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform

from torchtune.data import Message
from datasets import load_dataset

from ppotune.data.utils import PromptTemplate, apply_prompt_template
from torchtune import utils

log = utils.get_logger("DEBUG")


class QAProblem(tp.TypedDict):
    question: str
    cot: tp.Optional[str] = None
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
        source: str,
        sample_transform: QATransform,
        filter_fn: tp.Optional[tp.Callable] = None,
        system_prompt: tp.Optional[str] = None,
        prompt_template: tp.Optional[PromptTemplate] = None,
        add_generation_prompt: bool = False,
        **load_dataset_kwargs: tp.Dict[str, tp.Any],
    ) -> None:

        self.sample_transform = sample_transform
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.add_generation_prompt = add_generation_prompt
        self.data = load_dataset(source, **load_dataset_kwargs)

        if filter_fn is not None:
            self.data = self.data.filter(filter_fn)
            log.debug(f"Dataset length after filtering: {self.__len__()}")

    def setup(self, tokenizer: ModelTokenizer) -> None:
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def _tokenize_question(self, question: str) -> tp.List[int]:
        """
        Tokenize a question according to dataset format defined in terms of
        optional system prompt and prompt template for non-chat models.
        """
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
                add_generation_prompt=True
            )
        else:
            text = apply_prompt_template(
                template=self.prompt_template,
                messages=messages,
                add_generation_prompt=True
            )
            tokens = self.tokenizer.encode(text, add_eos=False)
            tokens = tokens[:self.tokenizer.max_seq_len]

        return tokens

    def __getitem__(self, index: int) -> tp.Dict[str, tp.Any]:
        sample = self.sample_transform(self.data[index])
        tokens = self._tokenize_question(sample["question"])
        return {
            "tokens": tokens,
            "cots": sample["cot"],
            "answers": sample["answer"]
        }
