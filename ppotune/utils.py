import typing as tp
import torch

from torchtune.modules.transforms.tokenizers import ModelTokenizer
from torchtune.models.llama3 import Llama3Tokenizer
from torchtune.models.mistral import MistralTokenizer
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

def liststrip(lst: list, element: tp.Any) -> list:
    start = 0
    while start < len(lst) and lst[start] == element:
        start += 1

    end = len(lst)
    while end > start and lst[end - 1] == element:
        end -= 1

    return lst[start:end]

def pretty_decode(
    tokenizer: ModelTokenizer,
    tokens: torch.Tensor, # Q + R
    response_pad_mask: torch.Tensor # R
) -> str:
    """
    Make it clean!
    """
    query_len = tokens.shape[0] - response_pad_mask.shape[0]
    query, response = tokens[:query_len], tokens[query_len:]

    response[response_pad_mask] = tokenizer.pad_id
    query_response = torch.cat([query, response]).tolist()
    completion_tokens = liststrip(query_response, tokenizer.pad_id)

    if isinstance(tokenizer, Llama3Tokenizer):
        return tokenizer.decode(
            completion_tokens,
            skip_special_tokens=False,
            truncate_at_eos=False
        )
    if isinstance(tokenizer, MistralTokenizer):
        return tokenizer.decode(
            completion_tokens
        )
    raise NotImplementedError(f"{type(tokenizer)} is not supported")
