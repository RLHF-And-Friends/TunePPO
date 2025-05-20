import typing as tp

from ppotune.data.sets.text_completion import (
    TCTransform,
    TextCompletion,
    TextCompletionDataset
)
from torchtune.modules.transforms.tokenizers import BaseTokenizer


class WikiLinguaTransform(TCTransform):
    def __call__(self, sample: tp.Mapping[str, tp.Any]) -> TextCompletion:
        return TextCompletion(
            prompt=sample["text"],
            completion=sample["summary"]
        )


def wiki_lingua_dataset(
    tokenizer: BaseTokenizer,
    *,
    source: str,
    split: str = "train",
    max_tokens: tp.Optional[int] = 1024,
    configurations: tp.Optional[str | list[str] | tp.Dict[int, list[str]]] = None,
    **load_dataset_kwargs: tp.Dict[str, tp.Any],
) -> TextCompletionDataset:

    return TextCompletionDataset(
        tokenizer=tokenizer,
        source=source,
        configurations=configurations,
        sample_transform=WikiLinguaTransform(),
        max_tokens=max_tokens,
        split=split,
        **load_dataset_kwargs
    )
