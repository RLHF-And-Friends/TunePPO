import typing as tp
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Subset


T = tp.TypeVar("T", covariant=True)

def subset(
    dataset: Dataset[T],
    num_samples: int,
    generator: torch.Generator = None,
) -> Subset[T]:
    """
    Selects random subset of the dataset.
    """
    assert num_samples <= len(dataset)

    indices = torch.randperm(len(dataset), generator=generator)[:num_samples]
    subset = Subset(dataset, indices.tolist())
    return subset


def distributed_subset(
    dataset: Dataset[T],
    num_samples: int,
    generator: torch.Generator,
) -> Subset[T]:
    """
    Split dataset for each rank evenly.
    """
    print(f"Distributed subset: {dist.get_world_size()} * {num_samples} <= {len(dataset)}")
    assert dist.get_world_size() * num_samples <= len(dataset)

    indices = torch.randperm(len(dataset), generator=generator)
    local_offset = dist.get_rank() * num_samples
    local_indices = indices[local_offset : local_offset + num_samples]
    local_subset = Subset(dataset, local_indices.tolist())
    return local_subset
