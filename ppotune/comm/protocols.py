import typing as tp
import torch
import torch.distributed as dist
import torch.nn.functional as F

from abc import ABC, abstractmethod
from ppotune.comm.weightage import Weightage
from ppotune.data.types import PPOTrajectoryStats
from ppotune.log import WandbLogger


log = WandbLogger()
# ------------------------ Communication Protocols -------------------------- #
#
class CommProtocol(ABC):
    """
    Communication Protocol reduces basically to peer tensors aggregation
    strategy driven by statisics gathered throughout training.
    """
    def __init__(
        self,
        weightage: Weightage
    ) -> None:
       self._weightage = weightage
       self._weights = self._weightage()

    @abstractmethod
    def gather(self, stats: PPOTrajectoryStats) -> None:
        """
        Gather statistics.
        """
        ...

    @abstractmethod
    def update(self) -> None:
        """
        Updates weights based on stats gathered and clears stats.
        """
        ...

    def __call__(
        self,
        tensors: tp.Iterable[torch.Tensor],
    ) -> torch.Tensor:
        """
        Reduces tensor list according to protocol weightage.
        """
        tensors = torch.stack(tensors)
        weights = self._weights.to(tensors[0].device)
        log.collect("self_preference", weights[dist.get_rank()])

        for _ in range(tensors.dim() - 1):
            weights = weights.unsqueeze(-1)
        return (weights * tensors).sum(dim=0)


class StaticProtocol(CommProtocol):
    """
    Executes static peer tensor reduction non-dependant on learning dynamics.
    """
    def __init__(
        self,
        weightage: Weightage,
    ) -> None:
        super().__init__(weightage)

    def gather(self, stats: PPOTrajectoryStats) -> None:
        """
        Ignores stats.
        """
        pass

    def update(self) -> None:
        """
        Clears stats gathered.
        """
        self._weights = self._weightage()


class ScoreBasedProtocol(CommProtocol):
    """
    Relies on scores obtained between communication rounds.
    """
    def __init__(
        self,
        weightage: Weightage,
    ) -> None:
        super().__init__(weightage)
        self._score = 0.0

    def gather(self, stats: PPOTrajectoryStats) -> None:
        """
        Gathers mean scores.
        """
        self._score += stats.scores.mean()

    def update(self) -> None:
        """
        Clears mean scores.
        """
        self._weights = self._weightage(self._score)
        self._score = 0.0


class BasePolicySimilarityProtocol(CommProtocol):
    """
    Base class for policy similarity protocols with shared functionality.
    """
    def __init__(self, weightage: Weightage):
        super().__init__(weightage)
        self._policy_batch = None
        self._similarity_dataloader = None
        self._similarity_iter = None
        self._needs_policy_generation = False
        
    def setup_similarity_dataloader(self, similarity_dataloader):
        """
        Setup the similarity dataloader for generating policy batches.
        """
        self._similarity_dataloader = similarity_dataloader
        self._similarity_iter = iter(similarity_dataloader)
        
    def _tokenize_completions(self, completions, tokenizer, device):
        """
        Tokenize completion strings into tensors.
        """
        completion_tokens_batch = []
        for completion in completions:
            completion_tokens = tokenizer.encode(completion, add_eos=True)
            completion_tokens_batch.append(torch.tensor(completion_tokens, device=device))
        return completion_tokens_batch
    
    def _create_padded_inputs(self, prompt_tokens_batch, completion_tokens_batch, tokenizer, device):
        """
        Create padded inputs from prompts and completions for batched processing.
        """
        # Create full input: prompt + correct completion
        full_inputs = []
        for prompt_tokens, completion_tokens in zip(prompt_tokens_batch, completion_tokens_batch):
            full_input = torch.cat([prompt_tokens, completion_tokens], dim=0)
            full_inputs.append(full_input)
        
        # Pad to same length for batching
        max_len = max(len(inp) for inp in full_inputs)
        padded_inputs = []
        attention_masks = []
        
        for inp in full_inputs:
            pad_len = max_len - len(inp)
            padded = torch.cat([
                torch.full((pad_len,), tokenizer.pad_id, device=device),
                inp
            ])
            mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.bool, device=device),
                torch.ones(len(inp), dtype=torch.bool, device=device)
            ])
            padded_inputs.append(padded)
            attention_masks.append(mask)
        
        return torch.stack(padded_inputs), torch.stack(attention_masks), full_inputs
    
    def _extract_completion_logprobs(self, logits, prompt_tokens_batch, completion_tokens_batch, full_inputs, max_len, device):
        """
        Extract logprobs for completion tokens from model output.
        """
        policy_batch = []
        
        for i, (prompt_tokens, completion_tokens) in enumerate(zip(prompt_tokens_batch, completion_tokens_batch)):
            prompt_len = len(prompt_tokens)
            completion_len = len(completion_tokens)
            
            # Get logits for positions corresponding to completion tokens
            start_pos = max_len - len(full_inputs[i]) + prompt_len
            end_pos = start_pos + completion_len - 1  # -1 because we predict next token
            
            if end_pos > start_pos:  # Ensure we have tokens to process
                completion_logits = logits[i, start_pos:end_pos]  # [seq_len-1, vocab_size]
                target_tokens = completion_tokens[1:]  # Skip first token (predict from 2nd onwards)
                
                # Convert to logprobs and extract for target tokens
                logprobs = torch.log_softmax(completion_logits, dim=-1)
                policy_logprobs = [
                    logprobs[j, token].item() 
                    for j, token in enumerate(target_tokens) 
                    if j < len(logprobs)
                ]
                
                if policy_logprobs:
                    policy_batch.append(torch.tensor(policy_logprobs, device=device))
        
        return policy_batch
        
    def _generate_policy_batch(self, policy_model, test_batch, tokenizer, device, forward_batch_size, empty_cache=False):
        """
        Generate policy logprobs batch for similarity calculation.
        Computes logprobs of the CORRECT answer tokens, not generated tokens.
        """
        test_batch["tokens"] = test_batch["tokens"].to(device)
        policy_batch = []
        
        # Process in batches to manage memory
        for batch_start in range(0, test_batch["tokens"].shape[0], forward_batch_size):
            # Extract subbatch
            subbatch = {
                key: test_batch[key][batch_start : batch_start + forward_batch_size]
                for key in test_batch.keys()
            }
            
            # Tokenize completions
            completion_tokens_batch = self._tokenize_completions(
                subbatch["completion"], tokenizer, device
            )
            
            # Create padded inputs
            batch_input, batch_mask, full_inputs = self._create_padded_inputs(
                subbatch["tokens"], completion_tokens_batch, tokenizer, device
            )
            
            # Create position_ids and causal_mask for proper forward pass
            from torchtune import generation
            position_ids = generation.get_position_ids_from_padding_mask(batch_mask)
            causal_mask = generation.get_causal_mask_from_padding_mask(batch_mask)
            
            # Forward pass through policy model
            with torch.no_grad():
                logits = policy_model(batch_input, input_pos=position_ids, mask=causal_mask)
            
            # Extract logprobs for completion tokens
            batch_policy_logprobs = self._extract_completion_logprobs(
                logits, subbatch["tokens"], completion_tokens_batch, 
                full_inputs, batch_input.shape[1], device
            )
            
            policy_batch.extend(batch_policy_logprobs)
            
            if empty_cache:
                torch.cuda.empty_cache()
        
        return policy_batch
        
    def gather(self, stats: PPOTrajectoryStats) -> None:
        """
        Gather policy batch from trajectory stats.
        """
        if stats.policy_batch is not None:
            self._policy_batch = stats.policy_batch
        elif self._similarity_iter is not None:
            # Signal that we need policy batch generation
            self._needs_policy_generation = True
        else:
            # No policy batch and no similarity dataloader - will use fallback weights
            self._policy_batch = None

    @abstractmethod
    def _compute_similarity(self, my_logprobs: torch.Tensor, peer_logprobs: torch.Tensor) -> float:
        """
        Compute similarity between two policy logprob sequences.
        Must be implemented by subclasses.
        """
        ...

    def update(self) -> None:
        """
        Update weights based on policy similarity.
        """
        if self._policy_batch is None:
            # Fallback to default weights if no policy batch
            self._weights = self._weightage()
            return
            
        all_agent_logprob_batches = [None] * dist.get_world_size()
        dist.all_gather_object(all_agent_logprob_batches, self._policy_batch)
        my_rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Compute similarities with all agents (including self)
        similarities = torch.zeros(world_size, dtype=torch.float32, device='cuda')
        for peer_rank in range(world_size):
            similarity_sum = 0.0
            for batch_idx in range(len(self._policy_batch)):
                my_logprobs = all_agent_logprob_batches[my_rank][batch_idx].to('cuda')
                peer_logprobs = all_agent_logprob_batches[peer_rank][batch_idx].to('cuda')
                min_len = min(len(my_logprobs), len(peer_logprobs))
                
                # Use subclass-specific similarity computation
                similarity = self._compute_similarity(
                    my_logprobs[:min_len], 
                    peer_logprobs[:min_len]
                )
                similarity_sum += similarity
            similarities[peer_rank] = similarity_sum / len(self._policy_batch)
        
        # Apply weightage computation
        self._weights = self._weightage(similarities)
        self._policy_batch = None  # clear after use


class PolicySimilarityProtocol(BasePolicySimilarityProtocol):
    """
    Protocol for policy similarity based on dot product of policy logprobs.
    """
    def _compute_similarity(self, my_logprobs: torch.Tensor, peer_logprobs: torch.Tensor) -> float:
        """
        Compute dot product similarity between logprob sequences.
        """
        return torch.dot(my_logprobs, peer_logprobs).item()


class KLDivergencePolicySimilarityProtocol(BasePolicySimilarityProtocol):
    """
    Protocol for policy similarity based on KL divergence between policy distributions.
    Uses KL(P_i || P_j) where P_i and P_j are probability distributions over tokens.
    """
    def _compute_kl_divergence(self, logprobs_p, logprobs_q):
        """
        Compute KL(P||Q) where P and Q are distributions defined by logprobs.
        KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
        
        Args:
            logprobs_p: log probabilities of distribution P
            logprobs_q: log probabilities of distribution Q
            
        Returns:
            KL divergence as a scalar
        """
        # Convert logprobs to probabilities
        probs_p = torch.exp(logprobs_p)
        probs_q = torch.exp(logprobs_q)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        probs_p = torch.clamp(probs_p, min=eps)
        probs_q = torch.clamp(probs_q, min=eps)
        
        # Compute KL divergence: KL(P||Q) = sum(P * log(P/Q))
        kl_div = torch.sum(probs_p * (torch.log(probs_p) - torch.log(probs_q)))
        return kl_div.item()
    
    def _compute_symmetric_kl_divergence(self, logprobs_p, logprobs_q):
        """
        Compute symmetric KL divergence: 0.5 * (KL(P||Q) + KL(Q||P))
        This is more stable and symmetric than one-directional KL.
        """
        kl_pq = self._compute_kl_divergence(logprobs_p, logprobs_q)
        kl_qp = self._compute_kl_divergence(logprobs_q, logprobs_p)
        return 0.5 * (kl_pq + kl_qp)

    def _compute_similarity(self, my_logprobs: torch.Tensor, peer_logprobs: torch.Tensor) -> float:
        """
        Compute KL divergence-based similarity between logprob sequences.
        Lower KL divergence = higher similarity, so we use exp(-KL).
        """
        kl_div = self._compute_symmetric_kl_divergence(my_logprobs, peer_logprobs)
        return torch.exp(-torch.tensor(kl_div, device=my_logprobs.device)).item()


# ------------------ Protocol Builders ------------------ #

def static_protocol(
    weightage: Weightage,
) -> StaticProtocol:
    return StaticProtocol(weightage)

def score_based_protocol(
    weightage: Weightage,
) -> ScoreBasedProtocol:
    return ScoreBasedProtocol(weightage)

def policy_similarity_protocol(
    weightage: Weightage,
) -> PolicySimilarityProtocol:
    return PolicySimilarityProtocol(weightage)

def kl_divergence_policy_similarity_protocol(
    weightage: Weightage,
) -> KLDivergencePolicySimilarityProtocol:
    return KLDivergencePolicySimilarityProtocol(weightage)
