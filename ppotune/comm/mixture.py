import torch

from abc import ABC
from torchtune.modules import TransformerDecoder
from ppotune.comm.primitives import (
    all_gather_uneven,
    all_gather_even,
    all_to_all
)
from ppotune.comm.protocols import CommProtocol
from ppotune.data.types import PPOTrajectoryStats
from ppotune.log import WandbLogger
from ppotune.peft import (
    get_lora_linears,
    merge_lora_adapter,
    clear_lora_adapter,
)


# TODO:
# * Split into:
#
#   InactiveLoRAPolicy          x       ActiveLoRaPolicy,
#   DistributedPolicyMixture    x       DistributedWeightMixture
#
#   And thus, their combinations.
#   Maybe do so in a form of chainable wrappers.
#


log = WandbLogger()
# --------------------------- Distributed Mixture --------------------------- #
#
class DistributedMixture(ABC):
    """
    Base mixture class.
    """
    def __init__(
        self,
        protocol: CommProtocol,
        local_policy: TransformerDecoder,
        update_interval: int = 1,   # TODO: VolatileInt support
    ) -> None:
        self._protocol = protocol
        self._policy = local_policy
        self._update_interval = update_interval
        self._tokenizer = None
        self._device = None
        self._forward_batch_size = 8  # Default, can be overridden
        self._empty_cache = False

    def setup_similarity_dataloader(self, similarity_dataloader, tokenizer, device, forward_batch_size=8, empty_cache=False):
        """
        Setup similarity dataloader for policy similarity protocols.
        This encapsulates the similarity computation logic within the mixture.
        """
        self._tokenizer = tokenizer
        self._device = device
        self._forward_batch_size = forward_batch_size
        self._empty_cache = empty_cache
        
        if hasattr(self._protocol, 'setup_similarity_dataloader'):
            self._protocol.setup_similarity_dataloader(similarity_dataloader)

    def forward(
        self,
        tokens:     torch.Tensor,
        mask:       torch.Tensor,
        input_pos:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns next token probabilities according to the mixture.
        Args:
            tokens (torch.Tensor): B x S  - input tensor
            mask (torch.Tensor): B x S x S - causal masks
            input_pos (torch.Tensor): B x S - position ids of each token

        Returns:
            torch.Tensor: B x S x V output tensor of next token probs

        Shape notation:
            - B: batch size
            - S: token sequence length
            - V: vocab size
        """
        ...

    def gather(
        self,
        stats: PPOTrajectoryStats
    ) -> None:
        """
        Gather policy statistics.
        """
        # Check if protocol needs policy generation
        if (hasattr(self._protocol, '_needs_policy_generation') and 
            hasattr(self._protocol, '_similarity_iter') and
            self._protocol._similarity_iter is not None and
            self._tokenizer is not None):
            
            # Generate policy batch internally
            test_batch = next(self._protocol._similarity_iter)
            policy_batch = self._protocol._generate_policy_batch(
                self._policy, test_batch, self._tokenizer, 
                self._device, self._forward_batch_size, self._empty_cache
            )
            
            # Create modified stats with policy batch
            stats_with_policy = stats._replace(policy_batch=policy_batch)
            self._protocol.gather(stats_with_policy)
            
            # Reset the flag
            self._protocol._needs_policy_generation = False
            
            if self._empty_cache:
                torch.cuda.empty_cache()
        else:
            self._protocol.gather(stats)

    def update(
        self
    ) -> None:
        """
        Update mixture.
        """
        ...

    def update_at(
        self,
        step: int,
    ) -> None:
        """
        Update at the step conditionally on update schedule.
        """
        if step % int(self._update_interval) != 0:
            return
        self.update()

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


class DistributedLoRAWeightMixture(DistributedMixture):
    """
    Local policy that updates with peer weights mixture.
    """
    def forward(
        self,
        tokens:     torch.Tensor,
        mask:       torch.Tensor,
        input_pos:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns next token probabilities w.r.t. local policy.
        """
        return self._policy(tokens, input_pos=input_pos, mask=mask)

    def update(self) -> None:
        """
        Updates local policy with averaged peers' policy weights and updates
        communication protocol.
        """
        for linear in get_lora_linears(self._policy):
            peer_weights = all_gather_even(linear.weight)
            linear.weight.data.copy_(self._protocol(peer_weights))

        merge_lora_adapter(self._policy)
        clear_lora_adapter(self._policy)
        self._protocol.update()


class DistributedLoRAPolicyMixture(DistributedMixture):
    """
    Peer policies mixture.
    """
    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns next token probabilities w.r.t. policy mixture.
        """
        peer_tokens = all_gather_uneven(tokens)
        peer_masks = all_gather_uneven(mask)
        peer_pos = all_gather_uneven(input_pos)

        peer_logits_requested = [
            self._policy(tokens, input_pos=pos, mask=mask)
            for tokens, pos, mask in zip(peer_tokens, peer_pos, peer_masks)
        ]
        peer_logits_responded = all_to_all(peer_logits_requested)
        return self._protocol(peer_logits_responded)

    def update(self) -> None:
        """
        Merges local policy LoRA and updates communication protocol.
        """
        merge_lora_adapter(self._policy)
        clear_lora_adapter(self._policy)
        self._protocol.update()



# ----------------------- Distributed Mixture Builders ---------------------- #
#
def distributed_weight_mixture(
    protocol: CommProtocol,
    local_policy: TransformerDecoder,
    update_every_n_steps: int,
) -> DistributedLoRAWeightMixture:
    """
    Builds DistributedLoRAWeightMixture instance.
    """
    return DistributedLoRAWeightMixture(
        protocol,
        local_policy,
        update_every_n_steps
    )

def distributed_policy_mixture(
    protocol: CommProtocol,
    local_policy: TransformerDecoder,
    update_every_n_steps: int,
) -> DistributedLoRAPolicyMixture:
    """
    Builds DistributedLoRAPolicyMixture instance.
    """
    return DistributedLoRAPolicyMixture(
        protocol,
        local_policy,
        update_every_n_steps
    )
