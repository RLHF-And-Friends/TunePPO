import torch
import torch.distributed as dist

from torch.utils.data import DataLoader, Dataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer

from ppotune.data.subsets import subset, distributed_subset
from ppotune.data.collators import LeftPadCollator
from ppotune.data.samplers import (
    RepeatedSampler,
    GroupedSampler,
    PermutationSampler
)


def dataloader(
    tokenizer: ModelTokenizer,
    dataset: Dataset,
    num_steps: int,
    batch_size: int,
    group_size: int = 1,
    num_epochs: int = 1,
    seed: int = 0,
) -> DataLoader:
    """
    Basic dataloader.
    """
    assert batch_size % group_size == 0
    raw_batch_size = batch_size // group_size
    num_samples = num_steps * raw_batch_size

    generator = torch.Generator().manual_seed(seed + dist.get_rank() + 1)
    dataset = subset(dataset, num_samples, generator)

    sampler = RepeatedSampler(
        GroupedSampler(
            PermutationSampler(num_samples, generator=generator),
            group_factor=group_size,
        ),
        num_epochs=num_epochs
    )
    collator = LeftPadCollator(
        tokens_key="tokens",
        pad_token=tokenizer.pad_id
    )
    return DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=collator
    )


def distributed_dataloader(
    tokenizer: ModelTokenizer,
    dataset: Dataset,
    num_steps: int,
    batch_size: int,
    group_size: int = 1,
    num_epochs: int = 1,
    seed: int = 0,
) -> DataLoader:
    """
    Dataloader for distributed setup.
    """
    assert batch_size % group_size == 0
    raw_batch_size = batch_size // group_size
    num_samples = num_steps * raw_batch_size

    generator = torch.Generator().manual_seed(seed)
    local_dataset = distributed_subset(dataset, num_samples, generator)

    return dataloader(
        tokenizer,
        local_dataset,
        num_steps=num_steps,
        batch_size=batch_size,
        group_size=group_size,
        num_epochs=num_epochs,
        seed=seed,
    )
