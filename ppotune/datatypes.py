import torch
import typing as tp


class AdvantageTrajectoryStats(tp.NamedTuple):
    """
    Contains a collection of tensors stats collected at Advantage Model estimation stage
    """
    advantages: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor


class PPOTrajectoryStats(tp.NamedTuple):
    """
    Contains a collection of tensors stats collected at RLHF inference stage

    Attributes:
        query_responses (torch.Tensor): (query, response) pairs
            shape [b, context_length + max_generated_tokens]
        causal_mask (torch.Tensor): attention masks for input ids-generated responses pairs
            shape [b, context_length + max_generated_tokens, context_length + max_generated_tokens]
        position_ids (torch.Tensor): position IDs for input ids-generated responses pairs
            shape [b, context_length + max_generated_tokens]
        responses_pad_mask (torch.Tensor): padding masks for the truncated and padded generated responses
            shape [b, max_generated_tokens]
        gen_logprobs (torch.Tensor): log probabilities of the generated responses with
            shape [b, max_generated_tokens]
        ref_logprobs (torch.Tensor): log probs of the generated responses w.r.t. reference policy
            shape [b, max_generated_tokens]
        values (torch.Tensor): value estimates of the generated responses with
            shape [b, max_generated_tokens]
        returns (torch.Tensor): the estimated returns with
            shape: [b, response_len]
        advantages (torch.Tensor): the estimated advantages with
            shape: [b, response_len]
    """
    # generated trajectory
    query_responses: torch.Tensor
    causal_mask: torch.Tensor
    position_ids: torch.Tensor

    # response mask
    responses_pad_mask: torch.Tensor

    # logprob stats
    gen_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor

    # advantage model stats
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
