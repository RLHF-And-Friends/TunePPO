import typing as tp

from ppotune.data.sets.qa import QADataset, QAProblem, QATransform


class AlpacaTransform(QATransform):
    def __call__(self, sample: tp.Mapping[str, tp.Any]) -> QAProblem:

        return QAProblem(
            question = sample["instruction"],
            answer = ""
        )


def alpaca_dataset(
    source: str,
    **load_dataset_kwargs: tp.Dict[str, tp.Any],
) -> QADataset:

    ds = QADataset(
        source=source,
        sample_transform=AlpacaTransform(),
        add_generation_prompt=True,
        filter_fn=lambda sample: sample["input"] == "", # keep samples with empty input only
        **load_dataset_kwargs,
    )

    return ds
