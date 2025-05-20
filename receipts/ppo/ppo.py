# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import sys

from functools import partial
from itertools import chain
from typing import List

import torch
import torch.distributed as dist

from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

from torchtune import config, generation, rlhf, training, utils
from torchtune.modules.peft import (
    disable_adapter,
)
from torchtune.modules.transforms.tokenizers import ModelTokenizer
from torchtune.recipe_interfaces import FTRecipeInterface

from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from ppotune.advantage import IAdvantageModel
from ppotune.comm.mixture import DistributedMixture
from ppotune.config import instantiate
from ppotune.data.types import (
    PPOTrajectoryStats,
    AdvantageTrajectoryStats,
)
from ppotune.evaluation import Evaluator
from ppotune.loss import KLPenalty
from ppotune.log import WandbLogger
from ppotune.model import GenerativeLoRAModel
from ppotune.utils import grad_norm


log = utils.get_logger("DEBUG")
wandb_logger = WandbLogger()


class PPORecipe(FTRecipeInterface):
    """
    (Q)LoRA finetuning recipe for collaborative RLHF with PPO for dense transformer-based LLMs.
    This recipe is optimized for single GPU per agent training.

    This implementation is based on torchtune ppo full finetune recipe which in term derives from
    `tLearning to summarize from human feedback <https://arxiv.org/abs/2009.01325`_ and `Training a
    Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback
    <https://arxiv.org/abs/2204.05862`_.

    Features:
        - Activation Checkpointing helps reduce the memory footprint since we no longer
            keep activations in memory and instead recompute them during the backward pass. This is
            especially helpful for larger batch sizes when you're memory constrained. But these
            savings in memory come at the cost of training performance. In most cases training can
            slow-down quite a bit as a result of this activation recomputation.

        - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the
            ``dtype`` flag. When ``dtype=bf16``, all activations, gradients and optimizer states
            are in bfloat16. In most cases this should halve the memory footprint of full precision
            (fp32) training, without loss in model quality (will depend on the model, training data
            and other settings). For GPUs which do not support bfloat16, we fall back to fp32.
            Mixed precision training and fp16 precision are currently not supported.

        - Adjustable batch sizes. This recipe uses three different batch sizes:
            - ``batch_size`` controls the total number of samples which are sampled from the
                dataset for a single trajectory.
            - ``forward_batch_size`` controls the mini-batch size for trajectory generation. Since
                gradients are disabled during trajectory generation, memory consumption is lower
                and this can be higher than ``ppo_batch_size``.
            - ``ppo_batch_size`` controls the number of samples used for a single optimization step
                during PPO optimization. Since we're optimizing two models at once, adjusting this
                parameter can have a big impact during training.

        - Gradient Accumulation. You can simulate larger ``ppo_batch_size`` sizes by accumulating
            gradients. This is controlled using the ``gradient_accumulation_steps`` flag.

            For example: with ``ppo_batch_size``=32 and ``gradient_accumulation_steps``=16, each
            backward pass during PPO optimization uses a 'micro batch size' of 2.

            Gradient accumulation is especially useful when you are memory constrained. In this
            case, accumulating gradients might give you better training speed than enabling
            activation checkpointing.

        - Lower precision optimizers. This recipe supports lower-precision optimizers from the
            bitsandbytes library (https://huggingface.co/docs/bitsandbytes/main/en/index). We've
            tested the recipe with 8-bit AdamW and Paged AdamW. These optimizers are especially
            helpful when you are memory constrained since they help reduce the memory footprint
            associated with the optimizer states.

        - Checkpointing. Model weights are checkpointed only at the end of training. Resuming
            training is unsupported. For more details on the checkpointer, please take a look at
            torchtune checkpointer deepdive:
            https://pytorch.org/torchtune/main/deep_dives/checkpointer.html.

        - WandB Logging. Logs are piped into WandB directly.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    """
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize basic things.
        """
        # device and dtype
        self._device = utils.get_device("cuda")
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        # manually set up a generator
        self.seed = training.set_seed(seed=cfg.seed)
        self._rng = torch.Generator(self._device).manual_seed(self.seed)

        # ppo loss epsilon
        self._epsilon = cfg.epsilon

        # generation parameters
        self._empty_cache = cfg.get("empty_cache_after_generation", False)

    def setup(self, cfg: DictConfig) -> None:
        """
        Sets up the recipe state correctly.
        """
        # setup logger
        wandb_logger.setup(cfg.wandb_logger)
        wandb_logger.log_config(cfg)

        # setup data
        self._tokenizer: PreTrainedTokenizerBase | ModelTokenizer = instantiate(
            cfg.tokenizer
        )
        dataset: Dataset = instantiate(cfg.dataset)
        dataset.setup(self._tokenizer)
        self.dataloader: DataLoader = instantiate(
            cfg.dataloader,
            tokenizer=self._tokenizer,
            dataset=dataset,
            seed=self.seed
        )
        self._setup_batch_sizes(cfg)

        self.eval: Evaluator = instantiate(cfg.evaluator)
        self.eval.setup(
            tokenizer=self._tokenizer,
            seed=self.seed
        )

        # setup policy and advantage model
        with training.set_default_dtype(self._dtype), self._device:
            self.policy: GenerativeLoRAModel = instantiate(
                cfg.policy,
                tokenizer=self._tokenizer,
                rng=self._rng
            )
            self.advantage: IAdvantageModel = instantiate(cfg.advantage)

        self.policy.setup(cfg.policy)
        self.advantage.setup(cfg.advantage, tokenizer=self._tokenizer)

        # instantiate optimizer
        self._optimizer: Optimizer = instantiate(
            cfg.optimizer, chain(
                self.policy.parameters(),
                self.advantage.parameters()
        ))
        # instantiate reference policy
        self._ref_policy: DistributedMixture = instantiate(
            cfg.reference,
            local_policy=self.policy
        )
        # instantiate kl penalty module
        self.kl: KLPenalty = instantiate(cfg.kl_penalty)

    def _setup_batch_sizes(self, cfg: DictConfig) -> None:
        """
        Validates and sets up parameters for used during training and for tracking training state,
        batch sizes for model forward passes during trajectory generation, PPO minibatches,
        PPO microbatches for gradient accumulation.
        """
        self.batch_size = cfg.dataloader.batch_size
        self._forward_batch_size = cfg.forward_batch_size
        self._ppo_epochs = cfg.ppo_epochs
        self._ppo_batch_size = cfg.ppo_batch_size
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._ppo_backward_batch_size = (
            cfg.ppo_batch_size // self._gradient_accumulation_steps
        )
        assert self.batch_size % self._forward_batch_size == 0
        assert self.batch_size % self._ppo_batch_size == 0
        assert self._ppo_batch_size % self._gradient_accumulation_steps == 0

    @torch.no_grad()
    def generate_trajectory(self, batch: dict) -> PPOTrajectoryStats:
        """
        Generates a trajectory given the current policy and value models, the reference policy
        model, the reward model, and batch of inputs. This is done over the following steps:

        1: Generate responses, and logits corresponding to the responses using the current policy,
            generating (query, response) pairs.
        2. Estimate logprobs of the generated responses using the current policy.
        3. Estimate values from the generated responses using the current value function.
        4. Replace any tokens in the response after the first stop token (usually EOS token) with
           padding, producting truncated responses.
        5. Run the reward model on the (query, truncated-response) pairs.
        6. Mask out all the invalid values in the trajectory due to padding tokens.

        Args:
            batch (dict): dict of input data with "tokens" key containing tensor of input token
            IDs with shape [b, seq_length]

        Returns:
            PPOTrajectoryStats: An instance of :class:`ppotune.datatypes.PPOTrajectoryStats`
                comprising the current trajectory.
        """
        query_len = batch["tokens"].shape[1]

        generated = self.policy.generate(prompt=batch["tokens"])

        tokens_mask = generated.query_mask | generated.response_mask
        # create attention masks and position IDs for future forwards
        causal_mask = generation.get_causal_mask_from_padding_mask(
            tokens_mask
        )
        position_ids = generation.get_position_ids_from_padding_mask(
            tokens_mask
        )

        # generate reference logits
        with disable_adapter(self.policy):
            reference_logits = self._ref_policy(
                generated.tokens,
                input_pos=position_ids,
                mask=causal_mask
            )
        reference_logits = reference_logits[:, query_len - 1 : -1]

        responses = generated.tokens[:, query_len:]
        responses_pad_mask = ~ generated.response_mask[:, query_len:]

        # estimate logprobs of the responses w.r.t. generation policy
        gen_logprobs = self.policy.logits_to_logprobs(generated.logits, responses)
        gen_logprobs[responses_pad_mask] = 1.0

        # estimate logprobs of the responses w.r.t. reference policy
        ref_logprobs = self.policy.logits_to_logprobs(reference_logits, responses)
        ref_logprobs[responses_pad_mask] = 1.0

        # log generative | reference model divergence
        kl = (gen_logprobs - ref_logprobs).sum(1)
        wandb_logger.collect("kl", kl)

        advantage_trajectory: AdvantageTrajectoryStats = self.advantage(
            tokens          = generated.tokens,
            causal_mask     = causal_mask,
            position_ids    = position_ids,
            responses_pad_mask = responses_pad_mask,
            gen_logprobs = gen_logprobs,
            ref_logprobs = ref_logprobs,
            batch=batch
        )
        sample_completion = self._tokenizer.decode(
            generated.tokens[0][tokens_mask[0]].tolist(),
            skip_special_tokens=False
        )
        wandb_logger.collect_completion(
            sample_completion, advantage_trajectory.scores[0]
        )
        return PPOTrajectoryStats(
            query_responses     = generated.tokens,
            causal_mask         = causal_mask,
            position_ids        = position_ids,
            responses_pad_mask  = responses_pad_mask,
            gen_logprobs        = gen_logprobs,
            ref_logprobs        = ref_logprobs,
            advantages          = advantage_trajectory.advantages,
            values              = advantage_trajectory.values,
            returns             = advantage_trajectory.returns,
            scores              = advantage_trajectory.scores,
        )

    def generate_trajectory_batched(
        self,
        batch: dict,
        empty_cache: bool = False
    ) -> PPOTrajectoryStats:
        """
        Generates a ``self.batch_size`` batch of trajectories using `self._forward_batch_size`
        batch sizes. See ``generate_trajectory`` for more details.

        Args:
            input_ids (torch.Tensor): tensor of input token IDs with shape [b, seq_length]

        Returns:
            PPOTrajectoryStats: An instance of :class:`ppotune.datatypes.PPOTrajectoryStats`,
                comprising the current trajectory.
        """
        batch["tokens"] = batch["tokens"].to(self._device)
        trajectories: List[PPOTrajectoryStats] = []
        for batch_start in range(0, self.batch_size, self._forward_batch_size):
            subbatch = {}
            for key in batch.keys():
                subbatch[key] = batch[key][
                    batch_start : batch_start + self._forward_batch_size
                ]
            trajectories.append(self.generate_trajectory(subbatch))
            torch.cuda.empty_cache() if empty_cache else None

        trajectory = PPOTrajectoryStats(*map(torch.cat, zip(*trajectories)))
        wandb_logger.collect_dict({
            "num_stop_tokens": trajectory.responses_pad_mask.any(-1).sum().float(),
            "response_lengths": training.get_unmasked_sequence_lengths(
                trajectory.responses_pad_mask
            ).float(),
        })
        return trajectory

    def train(self) -> None:
        """
        The core training loop.
        """
        self._optimizer.zero_grad()

        self.eval(self.policy)
        wandb_logger.flush(step=0)

        for step, batch in tqdm(
            enumerate(self.dataloader, start=1),
            desc="Train",
            disable=dist.get_rank() != 0,
            total=len(self.dataloader)
        ):

            trajectory = self.generate_trajectory_batched(batch, self._empty_cache)
            # optimize with PPO objective over multiple epochs
            for _ in range(self._ppo_epochs):
                batch_idxs = torch.randperm(self.batch_size, device=self._device)
                for i in range(0, self.batch_size, self._ppo_batch_size):
                    mini_batch_idxs = batch_idxs[i : i + self._ppo_batch_size]

                    for j in range(
                        0, self._ppo_batch_size, self._ppo_backward_batch_size
                    ):
                        backward_batch_idxs = mini_batch_idxs[
                            j : j + self._ppo_backward_batch_size
                        ]

                        batch_trajectory = PPOTrajectoryStats(
                            *map(
                                partial(
                                    torch.index_select,
                                    dim=0,
                                    index=backward_batch_idxs,
                                ),
                                trajectory,
                            )
                        )
                        self.ppo_step(batch_trajectory)
                        del batch_trajectory

                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)

            self.eval(self.policy, step)

            self._ref_policy.gather(trajectory)
            self._ref_policy.update_at(step)

            wandb_logger.flush(step=step)
            self.cleanup_after_step(trajectory)

        self.policy.save_checkpoint()

    def ppo_step(self, trajectory: PPOTrajectoryStats) -> None:
        """
        Perform a single PPO optimisation step over a batch of trajectories.
        """
        queries_len = trajectory.query_responses.shape[1] - trajectory.responses_pad_mask.shape[1]
        # estimate logprobs from the policy at the current optimisation step
        logits = self.policy(
            trajectory.query_responses,
            input_pos=trajectory.position_ids,
            mask=trajectory.causal_mask,
        )

        logits = logits[:, queries_len - 1 : -1]
        logprobs = rlhf.logits_to_logprobs(
            logits, trajectory.query_responses[:, queries_len:], self.policy._temperature
        )
        logprobs[trajectory.responses_pad_mask] = 1.0

        del logits

        ratios = torch.exp(logprobs - trajectory.gen_logprobs)
        clipped_ratios = torch.clamp(ratios, 1.0 - self._epsilon, 1.0 + self._epsilon)

        policy_losses_clipped = -trajectory.advantages * clipped_ratios
        policy_losses_unclipped = -trajectory.advantages * ratios

        clipfrac = (policy_losses_clipped > policy_losses_unclipped).to(
            logprobs.dtype
        )
        clipfrac = rlhf.masked_mean(clipfrac, ~trajectory.responses_pad_mask)

        policy_loss = torch.maximum(policy_losses_clipped, policy_losses_unclipped)
        policy_loss = rlhf.masked_mean(policy_loss, ~trajectory.responses_pad_mask)

        value_loss = self.advantage.loss(
            tokens=trajectory.query_responses,
            causal_mask=trajectory.causal_mask,
            position_ids=trajectory.position_ids,
            responses_pad_mask=trajectory.responses_pad_mask,
            inference_values=trajectory.values,
            inference_returns=trajectory.returns
        )
        kl_penalty = self.kl(
            logprobs,
            trajectory.ref_logprobs,
            padding_masks=~trajectory.responses_pad_mask
        )

        loss = policy_loss + value_loss + kl_penalty
        loss /= self._gradient_accumulation_steps
        loss.backward()

        with torch.no_grad():
            approx_policy_kls = (
                0.5 * (logprobs - trajectory.gen_logprobs).pow(2)
            ).mean()

        wandb_logger.collect_dict({
            "loss": loss * self._gradient_accumulation_steps,
            "policy_loss": policy_loss,
            "ratios": ratios.mean(),
            "clipfrac": clipfrac,
            "approx_policy_kl": approx_policy_kls,
            **self._collect_grad_norm("policy", self.policy),
            **self._collect_grad_norm("value", self.advantage)
        })

    def _collect_grad_norm(self, name: str, module: nn.Module) -> dict[str, torch.Tensor]:
        return {
            f"{name}_lora_grad_norm":
                grad_norm([param for name, param in module.named_parameters() if "lora" in name]),
            f"{name}_base_grad_norm":
                grad_norm([param for name, param in module.named_parameters() if "lora" not in name]),
        }

    def cleanup_after_step(
        self,
        trajectory: PPOTrajectoryStats,
    ) -> None:
        """
        Cleanup tensors after each PPO step to free up memory.
        """
        # there shouldn't be any floating references to the individual tensors at the this point,
        # so gc can do its thing
        for v in trajectory:
            del v
        del trajectory

    def cleanup(self, **kwargs) -> None:
        wandb_logger.close()
        dist.destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config
        - Overwritten by arguments from the command-line
    """
    dist.init_process_group()
    config.log_config(recipe_name="PPORecipe", cfg=cfg)
    recipe = PPORecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
