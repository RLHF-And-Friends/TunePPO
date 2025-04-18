import typing as tp

import torch
import torch.distributed as dist

from torchtune.modules import TransformerDecoder


# ------------------ Distributed Communication Primitives ------------------- #
def all_gather_even(tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Performs torch.distributed.all_gather and returns the output list. Tensors
    have to be of the same shape.
    """
    peer_tensors = [
        torch.empty_like(tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(peer_tensors, tensor)
    return peer_tensors

def all_gather_shapes(tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Collects shapes of tensors in distributed system. Tensors have to be of
    the same number of dimensions.
    """
    shape = torch.tensor(tensor.shape, dtype=torch.int64, device=tensor.device)
    return all_gather_even(shape)

def all_gather_uneven(tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Collects unevenly shaped tensors in distributed system. Tensors have to be
    of the same number of dimensions.
    """
    peer_shapes = all_gather_shapes(tensor)
    peer_tensors = [
        torch.empty(shape.tolist(), dtype=tensor.dtype, device=tensor.device)
        for shape in peer_shapes
    ]
    dist.all_gather(peer_tensors, tensor)
    return peer_tensors

def all_to_all(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Peforms torch.distributed.all_to_all and returns the output list. Input
    tensor lists has to be of the same shape configuration across all
    processes i.e tensors[rank].shape == const for any rank.
    """
    reference = tensors[dist.get_rank()]
    output = [torch.empty_like(reference) for _ in tensors]
    dist.all_to_all(output, tensors)
    return output


# --------------------- Distributed Special Utilities ----------------------- #
def rank_preferred_mean(
    tensors: tp.Iterable[torch.Tensor],
    self_preference: tp.Optional[float] = None
) -> torch.Tensor:
    """
    Computes a weighted mean of a list of tensors, giving preference to the
    tensor corresponding to the current process rank.
    """
    assert len(tensors) == dist.get_world_size()

    if dist.get_world_size() == 1:
        return tensors[0]

    if not self_preference:
        return torch.stack(tensors).mean(dim=0)

    assert (0.0 <= self_preference) and (self_preference <= 1.0)

    rank = dist.get_rank()
    this = tensors[rank]
    others = tensors[0 : rank] + tensors[rank + 1 :]
    return self_preference * this + (1.0 - self_preference) * torch.stack(others).mean(dim=0)


# ----------------------- Distributed Policy Mixture ------------------------ #
class DistributedPolicyMixture:
    """
    Implements interprocess communication and delivers a mixtxture of policies.
    """
    def __init__(
        self,
        local_policy: TransformerDecoder,
        self_preference : tp.Optional[float] = None
    ) -> None:
        self._policy = local_policy
        self._preference = self_preference

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Follows the interface of TransformerDecoder forward method. Use it as a
        more comprehensive reference.
        Args:
            tokens (torch.Tensor): input tensor with shape ``[b x s]``
            mask (torch.Tensor): used to mask the scores after the query-key
                multiplication and before the softmax. This parameter is
                required during inference.
            input_pos (torch.Tensor): contains the position ids of each token.
                During training, this is used to indicate the positions of each
                token relative to its sample when packed, shape ``[b x s]``.
                During inference, this indicates the position of the current
                token. This parameter is required during inference.

        Returns:
            torch.Tensor: output tensor with shape ``[b x s x v]``

        Shape notation:
            - b: batch size
            - s: token sequence length
            - v: vocab size
        """
        peer_tokens = all_gather_uneven(tokens)
        peer_masks = all_gather_uneven(mask)
        peer_pos = all_gather_uneven(input_pos)

        peer_logits_requested = [
            self._policy(tokens, input_pos=pos, mask=mask)
            for tokens, pos, mask in zip(peer_tokens, peer_pos, peer_masks)
        ]
        peer_logits_responded = all_to_all(peer_logits_requested)
        return rank_preferred_mean(peer_logits_responded, self._preference)

    def __call__(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Executes forward pass.
        """
        return self.forward(tokens, mask, input_pos)
