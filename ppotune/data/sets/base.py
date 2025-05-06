import typing as tp

import torch.distributed as dist

from torch.utils.data import Dataset

from datasets import load_dataset, concatenate_datasets


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
            self.data = concatenate_datasets([
                load_dataset(
                    path=source,
                    name=configuration,
                    **load_dataset_kwargs
                ) for configuration in configurations
            ])
        elif isinstance(configurations, dict):
            worker_configurations = configurations[dist.get_rank()]
            self.data = concatenate_datasets([
                load_dataset(
                    path=source,
                    name=configuration,
                    **load_dataset_kwargs
                ) for configuration in worker_configurations
            ])
        elif configurations is None:
            self.data = load_dataset(
                path=source,
                **load_dataset_kwargs
            )
        else:
            raise Exception("Wrong configurations format!")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
