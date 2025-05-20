import typing as tp

from functools import partial
from torch.utils.data import Dataset

from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform
from torchtune import utils

from ppotune.data.utils import load_dataset_with_configurations

log = utils.get_logger("DEBUG")


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
        source: str,
        sample_transform: TCTransform,
        configurations: tp.Optional[str | tp.List[str] | tp.Dict[int, tp.List[str]]] = None,
        max_tokens: tp.Optional[int] = None,
        filter_fn: tp.Optional[tp.Callable] = None,
        **load_dataset_kwargs: tp.Dict[str, tp.Any],
    ) -> None:

        self.data = load_dataset_with_configurations(
            source,
            configurations,
            **load_dataset_kwargs
        )
        self.sample_transform = sample_transform
        self._max_tokens = max_tokens

        if filter_fn is not None:
            self.data = self.data.filter(filter_fn)
            log.debug(f"Dataset length after filtering: {self.__len__()}")

    def setup(self, tokenizer: ModelTokenizer) -> None:
        self.tokenizer = tokenizer

        if self._max_tokens is not None:
            filter_fn = partial(
                shorter_than_max_tokens,
                tokenizer=self.tokenizer,
                sample_transform=self._sample_transform,
                max_tokens=self._max_tokens
            )
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


# Filtering function
# =================================================================================================

def shorter_than_max_tokens(
    sample,
    tokenizer: ModelTokenizer,
    sample_transform: TCTransform,
    max_tokens: int
) -> None:
    prompt = sample_transform(sample)["prompt"]
    tokens = tokenizer.encode(prompt)
    return len(tokens) < max_tokens

