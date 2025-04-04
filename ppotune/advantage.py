from abc import ABC, abstractmethod

from torchtune import rlhf
from torchtune.modules import TransformerDecoder

from ppotune.datatypes import AdvantageTrajectoryStats
from ppotune.utils import append_mask

import torch


class IAdvantageModel(ABC):
    """
    Advantage Model Abstract Interface
    """
    @abstractmethod
    def __call__(
        self,
        rewards: torch.Tensor # B or B x R
    ) -> AdvantageTrajectoryStats:
        ...

    @abstractmethod
    def loss(
        self,
        **kwargs
    ) -> torch.Tensor:
        ...


class LLMCriticGAE(IAdvantageModel):
    """
    Generalized Advantage Estimation with LLM Value Model
    """
    def __init__(
        self,
        scorer: TransformerDecoder,
        gamma: float,
        lmbda: float,
        value_coeff: float,
        value_clip_range: float
    ) -> None:

        self.scorer = scorer
        self.gamma = gamma
        self.lmbda = lmbda
        self.value_coeff = value_coeff
        self.value_clip_range = value_clip_range

    def estimate_values(
        self,
        tokens:         torch.Tensor, # B x (Q + R)
        causal_mask:    torch.Tensor, # B x (Q + R) x (Q + R)
        position_ids:   torch.Tensor, # B x (Q + R)
        padding_mask:   torch.Tensor, # B x R
    ) -> torch.Tensor: # B x R
        """
        Value function inference
        """
        values = self.scorer(
            tokens,
            input_pos=position_ids,
            mask=causal_mask
        ).squeeze(-1) # single output at "vocab" dim
        queries_len = tokens.shape[1] - padding_mask.shape[1]
        values = values[:, queries_len - 1 : -1]
        values[padding_mask] = 0.0
        return values

    @torch.no_grad()
    def __call__(
        self,
        rewards:            torch.Tensor, # B or B x R
        tokens:             torch.Tensor, # B x (Q + R)
        causal_mask:        torch.Tensor, # B x (Q + R) x (Q + R)
        position_ids:       torch.Tensor, # B x (Q + R)
        responses_pad_mask: torch.Tensor, # B x R
    ) -> AdvantageTrajectoryStats:
        """
        Get advantage estimation
        """
        value_pad_mask = append_mask(
            responses_pad_mask,
        )
        values = self.estimate_values(
            tokens,
            causal_mask,
            position_ids,
            value_pad_mask
        )
        advantages, returns = rlhf.estimate_advantages(
            values,
            rewards,
            self.gamma,
            self.lmbda,
            masks=~responses_pad_mask
        )
        return AdvantageTrajectoryStats(
            advantages=advantages,
            values=values,
            returns=returns
        )

    def loss(
        self,
        tokens:             torch.Tensor, # B x (Q + R)
        causal_mask:        torch.Tensor, # B x (Q + R) x (Q + R)
        position_ids:       torch.Tensor, # B x (Q + R)
        responses_pad_mask: torch.Tensor, # B x R
        inference_values:   torch.Tensor, # B x R
        inference_returns:  torch.Tensor, # B x R
    ) -> torch.Tensor:
        """
        Execute value function optimisation step
        """
        value_pad_mask = append_mask(responses_pad_mask)
        values = self.estimate_values(
            tokens,
            causal_mask,
            position_ids,
            value_pad_mask
        )
        values_clipped = torch.clamp(
            values,
            inference_values - self.value_clip_range,
            inference_values + self.value_clip_range,
        )
        loss = torch.maximum(
            (values - inference_returns) ** 2, (values_clipped - inference_returns) ** 2
        )
        loss = 0.5 * rlhf.masked_mean(loss, ~value_pad_mask)

        return self.value_coeff * loss


class GRAE(IAdvantageModel):
    """
    Group Relative Advantage Model with LLM reward.
    """
    def __init__(self, group_size: int, **kwargs) -> None:
        self.group_size = group_size
        self.value_coeff = 1.0 # todo: remove

    @torch.no_grad()
    def __call__(self,
        rewards: torch.Tensor, # B
        **kwargs
    ) -> AdvantageTrajectoryStats:
        """
        Get advantage estimation
        """
        scores = rewards.reshape(-1, self.group_size)
        advantages = (scores - scores.mean(1, keepdim=True)) / (
            scores.std(1, keepdim=True) + 1e-4
        )
        advantages = advantages.flatten().unsqueeze(-1)

        return AdvantageTrajectoryStats(
            advantages=advantages,
            values=torch.zeros_like(advantages), # irrelevant
            returns=torch.zeros_like(advantages) # irrelevant
        )

    def loss(self, **kwargs) -> torch.Tensor:
        return torch.tensor(0)
