import typing as tp

import torch.distributed as dist

from torch.utils.data import Dataset

from datasets import (
    load_dataset,
    concatenate_datasets,
    get_dataset_config_names
)


class BaseDataset(Dataset):
    def __init__(
        self,
        source: str,
        configurations: tp.Optional[str | tp.List[str] | tp.Dict[int, tp.List[str]]] = None,
        **load_dataset_kwargs: tp.Dict[str, tp.Any],
    ):
        if isinstance(configurations, str):
            self.data = load_dataset(
                path=source,
                name=configurations,
                **load_dataset_kwargs
            )
        elif isinstance(configurations, list):
            self._load_concat_dataset(source, configurations, **load_dataset_kwargs)
        elif isinstance(configurations, dict):
            worker_configurations = configurations[dist.get_rank()]
            self._load_concat_dataset(source, worker_configurations, **load_dataset_kwargs)
        elif configurations is None:
            all_configurations = get_dataset_config_names(source)
            self._load_concat_dataset(source, all_configurations, **load_dataset_kwargs)
        else:
            raise Exception("Wrong configurations format!")

    def _load_concat_dataset(
        self,
        source: str,
        configurations: list[str],
        **load_dataset_kwargs
    ) -> None:
        self.data = concatenate_datasets([
            load_dataset(
                path=source,
                name=configuration,
                **load_dataset_kwargs
            ) for configuration in configurations
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
