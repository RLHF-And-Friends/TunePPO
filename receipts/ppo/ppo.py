# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys

from functools import partial
from itertools import chain
from typing import Any, Dict, List, Tuple
from warnings import warn

import torch
import torch.distributed as dist

from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchtune import config, generation, modules, rlhf, training, utils
from torchtune.data import padded_collate
from torchtune.modules import TransformerDecoder, local_kv_cache
from torchtune.modules.peft import (
    disable_adapter,
    get_adapter_params,
    set_trainable_params,
)
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training.checkpointing import Checkpointer
from torchtune.utils import log_rank_zero
from tqdm import tqdm

from ppotune.advantage import IAdvantageModel
from ppotune.datatypes import (
    PPOTrajectoryStats,
    AdvantageTrajectoryStats,
)
from ppotune.dist import DistributedPolicyMixture
from ppotune.loss import KLPenalty
from ppotune.log import WandbLogger
from ppotune.peft import (
    get_adapter_config,
    get_merged_adapter_state_dict,
    merge_lora_adapter,
    clear_lora_adapter,
)
from ppotune.reward import IRewardModel
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

        # checkpointing attributes
        self._output_dir = cfg.output_dir

        # manually set up a generator
        self.seed = training.set_seed(seed=cfg.seed)
        self._rng = torch.Generator(self._device).manual_seed(self.seed)

        # initialize step counters
        self.global_step    = 0
        self._total_steps   = 0
        self._steps_run     = 0
        self._total_epochs  = 0
        self._epochs_run    = 0

        # reference policy update schedule
        self._update_ref_policy_every_n_steps = cfg.get("update_ref_policy_every_n_steps", 1)

        # save adapter configs for checkpointing
        self._policy_adapter_config = get_adapter_config(cfg.policy)
        self._scorer_adpater_config = get_adapter_config(cfg.scorer)

    def setup(self, cfg: DictConfig) -> None:
        """
        Sets up the recipe state correctly.
        """
        # setup logger
        wandb_logger.setup(cfg.wandb_logger)
        wandb_logger.log_config(cfg)

        # setup checkpointers
        self._policy_checkpointer = self._setup_checkpointer(
            cfg.policy_checkpointer
        )
        self._scorer_checkpointer = self._setup_checkpointer(
            cfg.scorer_checkpointer
        )
        # load checkpoints
        policy_state_dict = self._policy_checkpointer.load_checkpoint()
        scorer_state_dict = self._scorer_checkpointer.load_checkpoint()

        # initialize models and load the state dict
        self._policy = self._setup_lora_model(
            cfg.policy,
            policy_state_dict[training.MODEL_KEY]
        )
        self._scorer = self._setup_lora_model(
            cfg.scorer,
            scorer_state_dict[training.MODEL_KEY]
        )

        # instantiate optimizer
        self._optimizer: Optimizer = config.instantiate(cfg.optimizer, chain(
            self._policy.parameters(),
            self._scorer.parameters()
        ))
        # initialize reference policy
        self._ref_policy: DistributedPolicyMixture = config.instantiate(
            cfg.reference_model,
            local_policy=self._policy
        )
        # initialize reward model
        self.rm: IRewardModel = config.instantiate(
            cfg.reward_model,
            scorer=self._scorer
        )
        # initialize advantage estimator
        self.ae: IAdvantageModel = config.instantiate(
            cfg.advantage_model,
            scorer=self._scorer
        )
        # instantiate kl penalty module
        self.kl: KLPenalty = config.instantiate(cfg.kl_penalty)

        # instantiate tokenizer
        self._tokenizer: ModelTokenizer = config.instantiate(cfg.tokenizer)
        # setup sampler and dataloader
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset = cfg.dataset,
            cfg_sampler = cfg.sampler,
            tokenizer   = self._tokenizer,
            shuffle     = cfg.shuffle,
            batch_size  = cfg.batch_size,
        )
        # set other parameters
        self._setup_batch_sizes(cfg)
        self._setup_hyperparameters(cfg)

        # setup a KV-caching context manager for trajectory generation
        self.cache_ctx_manager = lambda: local_kv_cache(
            self._policy,
            batch_size=self._forward_batch_size,
            dtype=self._dtype,
            decoder_max_seq_len=self._tokenizer.max_seq_len + self._max_generated_tokens,
            device=self._device,
        )

    def _setup_checkpointer(
        self,
        ckpt_cfg: DictConfig
    ) -> Checkpointer:
        """
        Sets up checkpointer.
        """
        # set different output dir names for each agent
        output_dir = ckpt_cfg.output_dir
        output_dir = f"{output_dir}-{dist.get_rank()}-of-{dist.get_world_size() - 1}"

        checkpointer: Checkpointer = config.instantiate(
            ckpt_cfg,
            resume_from_checkpoint=False,
            output_dir=output_dir
        )
        return checkpointer

    def _setup_hyperparameters(self, cfg: DictConfig) -> None:
        """
        Sets up some training hyperparameters that have not been wrapped up into some module yet.
        """
        # trajectory generation args
        self._temperature = cfg.temperature
        self._top_k = cfg.top_k
        self._max_generated_tokens = cfg.max_generated_tokens

        # loss params
        self._epsilon = cfg.epsilon

    def _setup_batch_sizes(self, cfg: DictConfig) -> None:
        """
        Validates and sets up parameters for used during training and for tracking training state,
        batch sizes for model forward passes during trajectory generation, PPO minibatches,
        PPO microbatches for gradient accumulation and GRPO group sizes.

        Raises
            - ValueError if:
                - batch_size is not divisible by forward_batch_size
                - batch_size is not divisible by ppo_batch_size
                - ppo_batch_size is not divisible by gradient_accumulation_steps
                - num_steps is less than batch_size
                - (grpo) batch_size is not divisible by group size
                - (grpo) forward_batch_size is not divisible by group_size
        """
        self.batch_size = cfg.batch_size
        self.group_size = cfg.group_size
        self._forward_batch_size = cfg.forward_batch_size
        self._ppo_epochs = cfg.ppo_epochs
        self._ppo_batch_size = cfg.ppo_batch_size
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._ppo_backward_batch_size = (
            cfg.ppo_batch_size // self._gradient_accumulation_steps
        )

        if self.batch_size % self._forward_batch_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be exactly divisible by "
                f"forward_batch_size ({self._forward_batch_size})."
            )
        if self.batch_size % self._ppo_batch_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be exactly divisible by "
                f"ppo_batch_size ({self._ppo_batch_size})."
            )
        if self._ppo_batch_size % self._gradient_accumulation_steps != 0:
            raise ValueError(
                f"ppo_batch_size ({self._ppo_batch_size}) must be exactly divisible "
                f"by gradient_accumulation_steps ({self._gradient_accumulation_steps})."
            )

        if self.batch_size % self.group_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be exactly divisible by "
                f"group_size({self.group_size})."
            )
        if self._forward_batch_size % self.group_size != 0:
            raise ValueError(
                f"forward_batch_size ({self._forward_batch_size}) must be exactly divisible by "
                f"group_size({self.group_size})."
            )

        self._total_steps = cfg.num_steps // (self.batch_size * dist.get_world_size())

        batches_per_epoch = max(
            1, len(self._dataloader)
        )  # when we only have a single batch in the dataset

        self._total_epochs = math.ceil(self._total_steps / batches_per_epoch)
        if self._total_steps == 0:
            raise ValueError(
                f"num_steps {cfg.num_steps} must be greater than the batch size {self.batch_size}."
            )
        if self._total_steps < len(self._dataloader):
            warn(
                f"There are fewer total steps ({self._total_steps}, (num_steps//batch_size) than"
                f"there are batches ({len(self._dataloader)}) in the dataset. Training will stop"
                f"after ({self._total_steps}) steps without saving intermediate checkpoints."
            )
        if (self._total_steps > batches_per_epoch) and (
            self._total_steps % batches_per_epoch != 0
        ):
            warn(
                f"num_steps ({cfg.num_steps}) is not exactly divisible by "
                f"the number of batches in the dataset ({batches_per_epoch}). "
                f"Intermediate checkpoints will only be saved every {batches_per_epoch} steps."
            )
        log_rank_zero(
            log, f"Total steps to run: {self._total_steps}, Total epochs: {self._total_epochs}"
        )

    def _setup_lora_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any]
    ) -> TransformerDecoder:
        """
        Sets up a LoRA model.
        """
        # note: ensure put.bias is not in scorer state dict
        with training.set_default_dtype(self._dtype), self._device:
            model: TransformerDecoder = config.instantiate(model_cfg)
            set_trainable_params(model, get_adapter_params(model))

        training.set_activation_checkpointing(
            model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
        )
        # warning: strict=False allows not all model params to be initialized
        # wich is dangerous but is necessary to load with no LoRAs predefined.
        model.load_state_dict(model_state_dict, strict=False)
        return model

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        cfg_sampler: DictConfig,
        tokenizer: ModelTokenizer,
        batch_size: int,
        shuffle: bool,
    ) -> Tuple[Sampler, DataLoader]:
        """
        All data related setup happens here.
        """
        dataset = config.instantiate(
            cfg_dataset,
            tokenizer=tokenizer
        )
        sampler: Sampler = config.instantiate(
            cfg_sampler,
            dataset=dataset,
            shuffle=shuffle,
            drop_last=True
        ) # better set seed here
        collator = partial(
            padded_collate,
            pad_direction="left",
            keys_to_pad=["tokens", "labels"],
            padding_idx=tokenizer.pad_id,
        )
        dataloader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=True,
            collate_fn=collator
        )
        return sampler, dataloader

    def generate_trajectory(self, input_ids: torch.Tensor) -> PPOTrajectoryStats:
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
            input_ids (torch.Tensor): tensor of input token IDs with shape [b, seq_length]

        Returns:
            PPOTrajectoryStats: An instance of :class:`ppotune.datatypes.PPOTrajectoryStats`
                comprising the current trajectory.
        """
        query_len = input_ids.shape[1]

        # generate responses and logits
        with self.cache_ctx_manager():
            tokens, logits = generation.generate(
                model=self._policy,
                prompt=input_ids,
                max_generated_tokens=self._max_generated_tokens,
                temperature=self._temperature,
                top_k=self._top_k,
                pad_id=self._tokenizer.pad_id,
                rng=self._rng,
            )

        tokens_pad_mask = tokens != self._tokenizer.pad_id

        responses = tokens[:, query_len:]
        # pad responses after eos token
        eos_mask = (responses == self._tokenizer.eos_id)
        seen_eos = torch.cumsum(eos_mask, dim=1)
        responses_pad_mask = (seen_eos > 1) | ((seen_eos == 1) & ~eos_mask)

        # create attention masks and position IDs for follow up generation
        causal_mask = generation.get_causal_mask_from_padding_mask(
            tokens_pad_mask
        )
        position_ids = generation.get_position_ids_from_padding_mask(
            tokens_pad_mask
        )

        # generate reference logits
        with disable_adapter(self._policy):
            ref_logits = self._ref_policy(
                tokens,
                input_pos=position_ids,
                mask=causal_mask
            )
        ref_logits = ref_logits[:, query_len - 1 : -1]

        # estimate logprobs of the responses w.r.t. generation policy
        gen_logprobs = rlhf.logits_to_logprobs(logits, responses, self._temperature)
        gen_logprobs[responses_pad_mask] = 1.0
        del logits

        # estimate logprobs of the responses w.r.t. reference policy
        ref_logprobs = rlhf.logits_to_logprobs(ref_logits, responses, self._temperature)
        ref_logprobs[responses_pad_mask] = 1.0
        del ref_logits

        rewards = self.rm(
            tokens,
            causal_mask,
            position_ids,
            responses_pad_mask,
            gen_logprobs=gen_logprobs,
            ref_logprobs=ref_logprobs
        )
        ae_trajectory: AdvantageTrajectoryStats = self.ae(
            rewards     = rewards,
            tokens          = tokens,
            causal_mask     = causal_mask,
            position_ids    = position_ids,
            responses_pad_mask  = responses_pad_mask,
        )
        return PPOTrajectoryStats(
            query_responses     = tokens,
            causal_mask         = causal_mask,
            position_ids        = position_ids,
            responses_pad_mask  = responses_pad_mask,
            gen_logprobs        = gen_logprobs,
            ref_logprobs        = ref_logprobs,
            values              = ae_trajectory.values,
            returns             = ae_trajectory.returns,
            advantages          = ae_trajectory.advantages,
        )

    def generate_trajectory_batched(self, input_ids: torch.Tensor) -> PPOTrajectoryStats:
        """
        Generates a ``self.batch_size`` batch of trajectories using `self._forward_batch_size`
        batch sizes. See ``generate_trajectory`` for more details.

        Args:
            input_ids (torch.Tensor): tensor of input token IDs with shape [b, seq_length]

        Returns:
            PPOTrajectoryStats: An instance of :class:`ppotune.datatypes.PPOTrajectoryStats`,
                comprising the current trajectory.
        """
        trajectories: List[PPOTrajectoryStats] = []
        with torch.no_grad():
            for batch_start in range(0, self.batch_size, self._forward_batch_size):
                batch_input_ids = input_ids[
                    batch_start : batch_start + self._forward_batch_size
                ]
                trajectories.append(self.generate_trajectory(batch_input_ids))
        return PPOTrajectoryStats(*map(torch.cat, zip(*trajectories)))

    def train(self) -> None:
        """
        The core training loop.
        """
        self._optimizer.zero_grad()

        training_completed = False
        pbar = tqdm(total=self._total_steps, initial=self._steps_run)
        for curr_epoch in range(self._epochs_run, self._total_epochs):
            # Ensure data is not reshuffled at new epoch so the agents are
            # trained on non-overlapping data.
            self._sampler.set_epoch(0)

            for _, batch in enumerate(self._dataloader):
                batch = batch["tokens"].to(self._device)

                # generate trajectories using:
                # - the current policy
                # - the current value function
                # - the reference model
                trajectory = self.generate_trajectory_batched(batch)

                wandb_logger.collect_dict({
                    "num_stop_tokens": trajectory.responses_pad_mask.any(-1).sum().float(),
                    "response_lengths": training.get_unmasked_sequence_lengths(
                        trajectory.responses_pad_mask
                    ).float(),
                })
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

                        wandb_logger.collect_dict({
                            **self._collect_grad_norm("policy", self._policy),
                            **self._collect_grad_norm("value", self._scorer)
                        })
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)

                        self.global_step += 1

                # step 5. profit
                self._steps_run += 1
                wandb_logger.flush(step=self.global_step)
                self.cleanup_after_step(trajectory)

                if self._steps_run % self._update_ref_policy_every_n_steps == 0:
                    # effectively update reference policy.
                    merge_lora_adapter(self._policy)
                    clear_lora_adapter(self._policy)

                pbar.update(1)
                if self._steps_run == self._total_steps:
                    training_completed = True
                    break

            self._epochs_run += 1

            if training_completed:
                self._policy_checkpointer.save_checkpoint(
                    state_dict=get_merged_adapter_state_dict(
                        module=self._policy,
                        module_adapter_config=self._policy_adapter_config
                    ),
                    epoch=curr_epoch
                )
                return

    def ppo_step(self, trajectory: PPOTrajectoryStats) -> None:
        """
        Perform a single PPO optimisation step over a batch of trajectories.
        """
        queries_len = trajectory.query_responses.shape[1] - trajectory.responses_pad_mask.shape[1]
        # estimate logprobs from the policy at the current optimisation step
        logits = self._policy(
            trajectory.query_responses,
            input_pos=trajectory.position_ids,
            mask=trajectory.causal_mask,
        )

        logits = logits[:, queries_len - 1 : -1]
        logprobs = rlhf.logits_to_logprobs(
            logits, trajectory.query_responses[:, queries_len:], self._temperature
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

        value_loss = self.ae.loss(
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
