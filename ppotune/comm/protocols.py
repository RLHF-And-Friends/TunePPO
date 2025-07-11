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


class BatchPolicySimilarityProtocol:
    def __init__(self, temperature=1.0, self_preference=None):
        self.temperature = temperature
        self.self_preference = self_preference
        self._policy_batch = None
        self._weights = torch.ones(dist.get_world_size()) / dist.get_world_size()

    def set_policy_batch(self, policy_batch):
        self._policy_batch = policy_batch

    def gather(self, stats):
        pass

    def update(self):
        if self._policy_batch is None:
            raise ValueError("Policy batch not set")
        peer_policies = [None] * dist.get_world_size()
        dist.all_gather_object(peer_policies, self._policy_batch)
        my_rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Compute similarities with all agents (including self)
        similarities = torch.zeros(world_size, dtype=torch.float32, device='cuda')
        for peer in range(world_size):
            sim = 0.0
            for k in range(len(self._policy_batch)):
                a = peer_policies[my_rank][k].to('cuda')
                b = peer_policies[peer][k].to('cuda')
                min_len = min(len(a), len(b))
                dot = torch.dot(a[:min_len], b[:min_len]).item()
                sim += dot
            similarities[peer] = sim / len(self._policy_batch)
        
        if self.self_preference is not None and world_size > 1:
            # Set fixed self preference
            self._weights = torch.zeros(world_size, dtype=torch.float32, device='cuda')
            self._weights[my_rank] = self.self_preference
            
            # Distribute remaining mass among other agents based on similarity
            remaining_mass = 1.0 - self.self_preference
            other_similarities = similarities.clone()
            other_similarities[my_rank] = 0.0  # exclude self from softmax
            
            # Apply softmax to other agents' similarities
            other_weights = torch.softmax(other_similarities / self.temperature, dim=0)
            other_weights[my_rank] = 0.0  # ensure self weight is 0 in other_weights
            
            # Scale other weights by remaining mass
            self._weights += remaining_mass * other_weights
        else:
            # Original behavior: pure similarity-based softmax
            self._weights = torch.softmax(similarities / self.temperature, dim=0)
            
        self._policy_batch = None  # clear

    def __call__(self, tensors):
        tensors = torch.stack(tensors)
        weights = self._weights.to(tensors[0].device)
        log.collect("self_preference", weights[dist.get_rank()])

        for _ in range(tensors.dim() - 1):
            weights = weights.unsqueeze(-1)
        return (weights * tensors).sum(dim=0)


# -------------------- Communication Protocol Builders ---------------------- #
#
def static_protocol(weightage: Weightage) -> StaticProtocol:
    return StaticProtocol(weightage)

def score_based_protocol(weightage: Weightage) -> ScoreBasedProtocol:
    return ScoreBasedProtocol(weightage)

def batch_policy_similarity_protocol(temperature=1.0, self_preference=None) -> BatchPolicySimilarityProtocol:
    return BatchPolicySimilarityProtocol(temperature, self_preference)
