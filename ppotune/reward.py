from abc import ABC, abstractmethod

from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import disable_adapter
from torchtune.training import get_unmasked_sequence_lengths
from torchtune.rlhf import get_reward_penalty_mask, get_rewards_ppo

from ppotune.log import WandbLogger
from ppotune.utils import append_mask

import torch


logger = WandbLogger()

class IRewardModel(ABC):
    """
    Abstract Reward Model Interface
    """
    @abstractmethod
    def __call__(
        self,
        tokens:             torch.Tensor, # B x (Q + R)
        responses_pad_mask: torch.Tensor, # B x R
        **kwargs
    ) -> torch.Tensor: # B or B x R
        ...


class LLMRewardModel(IRewardModel):
    """
    LLM-based reward model
    """
    def __init__(
        self,
        scorer: TransformerDecoder,
        penalise_no_eos:    bool,
        reward_penalty:     int,
        min_response_len:   int,
    ) -> None:

        self.scorer = scorer
        self.penalise_no_eos = penalise_no_eos
        self.reward_penalty = reward_penalty
        self.min_response_len = min_response_len

    @torch.no_grad()
    def __call__(
        self,
        tokens:             torch.Tensor, # B x (Q + R)
        causal_mask:        torch.Tensor, # B x (Q + R) x (Q + R)
        position_ids:       torch.Tensor, # B x (Q + R)
        responses_pad_mask: torch.Tensor, # B x R
        **kwargs,
    ) -> torch.Tensor: # B

        queries_len = tokens.shape[1] - responses_pad_mask.shape[1]

        with disable_adapter(self.scorer): # in case it is a LoRA scorer
            scores = self.scorer(
                tokens,
                input_pos=position_ids,
                mask=causal_mask
            )

        # the scores from the reward model are the logits for the last non-padding token
        response_last_pos = get_unmasked_sequence_lengths(responses_pad_mask)
        scores = scores.gather(1, (response_last_pos + queries_len)[:, None, None]).squeeze(
            (-1, -2)
        )
        # apply penalties for no EOS or too short responses
        reward_penalty_mask = get_reward_penalty_mask(  # warn: seem to penalize generations with
            responses_pad_mask,                         # eos at the very end
            response_last_pos,
            self.penalise_no_eos,
            self.min_response_len,
        )
        scores[reward_penalty_mask] = self.reward_penalty

        logger.collect("scores", scores)
        return scores


class PerTokenKLPenalizedRewardModel(LLMRewardModel):
    """
    OpenAI-like reward model with injected per token KL-Penalty
    """
    def __init__(
        self,
        scorer: TransformerDecoder,
        penalise_no_eos:    bool,
        reward_penalty:     int,
        min_response_len:   int,
        kl_coeff:           float,
    ) -> None:

        super().__init__(
            scorer,
            penalise_no_eos,
            reward_penalty,
            min_response_len,
        )
        self.kl_coeff = kl_coeff

    @torch.no_grad()
    def __call__(
        self,
        tokens:             torch.Tensor, # B x (Q + R)
        causal_mask:        torch.Tensor, # B x (Q + R) x (Q + R)
        position_ids:       torch.Tensor, # B x (Q + R)
        responses_pad_mask: torch.Tensor, # B x R
        gen_logprobs:       torch.Tensor, # B x R
        ref_logprobs:       torch.Tensor, # B x R
    ) -> torch.Tensor: # B x R

        scores = super().__call__(
            tokens,
            causal_mask,
            position_ids,
            responses_pad_mask
        )
        mask_after_eos = append_mask(responses_pad_mask)
        pos_after_eos = get_unmasked_sequence_lengths(mask_after_eos)

        rewards, kl, kl_rewards = get_rewards_ppo(
            scores,
            gen_logprobs,
            ref_logprobs,
            self.kl_coeff,
            pos_after_eos
        )
        logger.collect_dict({
            "rlhf_reward": scores + kl_rewards.sum(1),
            "kl": kl.sum(1),
            "kl_reward": kl_rewards.sum(1),
        })
        return rewards
