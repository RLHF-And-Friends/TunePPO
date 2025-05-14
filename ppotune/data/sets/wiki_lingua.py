import typing as tp

from functools import partial

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


def filter_by_length(
    sample,
    tokenizer: BaseTokenizer,
    max_tokens: int
) -> None:
    tokens = tokenizer.encode(sample["text"])

    return len(tokens) < max_tokens


def wiki_lingua_dataset(
    tokenizer: BaseTokenizer,
    *,
    source: str,
    split: str = "train",
    max_input_tokens: tp.Optional[int] = 1024,
    configurations: tp.Optional[str | list[str] | tp.Dict[int, list[str]]] = None,
    **load_dataset_kwargs: tp.Dict[str, tp.Any],
) -> TextCompletionDataset:
    
    if max_input_tokens is not None:
        filter_fn = partial(
            filter_by_length,
            tokenizer=tokenizer,
            max_tokens=max_input_tokens
        )
    else:
        filter_fn = None

    return TextCompletionDataset(
        tokenizer=tokenizer,
        source=source,
        configurations=configurations,
        sample_transform=WikiLinguaTransform(),
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs
    )
