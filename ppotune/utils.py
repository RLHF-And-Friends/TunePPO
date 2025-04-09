import typing as tp
import torch

from torchtune.training import get_unmasked_sequence_lengths


@torch.no_grad()
def grad_norm(parameters: tp.Iterable[torch.Tensor]) -> torch.Tensor:
    """
    Computes gradient l2-norm of parameters given.
    """
    total_norm = torch.tensor(0.0)
    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def append_mask(mask: torch.Tensor) -> torch.Tensor: # B x S -> B x S
    """
    Append last entry in the mask
    """
    last_pos = get_unmasked_sequence_lengths(mask)
    after_last_pos = torch.where(
        (last_pos > 0) & (last_pos < mask.shape[1] - 1),
        last_pos + 1,
        last_pos,
    )
    appended_mask = mask.clone()
    appended_mask = appended_mask.scatter_(
        1, after_last_pos.unsqueeze(-1), False
    ) # unmask last entry
    return appended_mask
