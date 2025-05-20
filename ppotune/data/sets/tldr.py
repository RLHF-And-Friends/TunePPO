import typing as tp

from ppotune.data.sets.text_completion import (
    TCTransform,
    TextCompletion,
    TextCompletionDataset
)


class TLDRTransform(TCTransform):
    def __call__(self, sample: tp.Mapping[str, tp.Any]) -> TextCompletion:
        return TextCompletion(
            prompt=sample["prompt"],
            completion=sample["completion"]
        )


def tldr_dataset(
    source: str = "trl-lib/tldr",
    split: str = "train",
    configurations: tp.Optional[str | list[str] | tp.Dict[int, list[str]]] = None,
    **load_dataset_kwargs: tp.Dict[str, tp.Any],
) -> TextCompletionDataset:

    return TextCompletionDataset(
        source=source,
        configurations=configurations,
        sample_transform=TLDRTransform(),
        split=split,
        **load_dataset_kwargs
    )
