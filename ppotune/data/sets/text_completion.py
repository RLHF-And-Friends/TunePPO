import typing as tp
from torch.utils.data import Dataset

from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform

from datasets import load_dataset


class TextCompletion(tp.TypedDict):
    prompt: str
    completion: str


class TCTransform(Transform):
    def __call__(self, sample: tp.Mapping[str, tp.Any]) -> TextCompletion:
        pass


class TextCompletionDataset(Dataset):
    """
    Text Completion dataset class.
    """
    def __init__(
        self,
        tokenizer: ModelTokenizer,
        *,
        source: str,
        sample_transform: TCTransform,
        filter_fn: tp.Optional[tp.Callable] = None,
        **load_dataset_kwargs: tp.Dict[str, tp.Any],
    ) -> None:

        self.tokenizer = tokenizer
        self.sample_transform = sample_transform
        self.data = load_dataset(source, **load_dataset_kwargs)
        if filter_fn is not None:
            self.data = self.data.filter(filter_fn)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> tp.Dict[str, tp.Any]:
        sample = self.sample_transform(self.data[index])
        prompt = sample["prompt"]
        completion = sample["completion"]

        tokens = self.tokenizer.encode(prompt, add_eos=False)
        tokens = tokens[:self.tokenizer.max_seq_len]

        return {"tokens": tokens, "completion": completion}
