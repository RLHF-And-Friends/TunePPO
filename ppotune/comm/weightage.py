import typing as tp
import torch
import torch.distributed as dist

from ppotune.volatile import VolatileFloat
from ppotune.comm.primitives import softmax, uniform


# --------------------- Distributed Weightage Utilities --------------------- #
#
class Weightage:
    def __init__(self) -> None:
        pass

    def __call__(self) -> torch.Tensor:
        pass


class Uniform(Weightage):
    """
    Computes uniform weightage over peers, with optional self-preference.
    """
    def __init__(
        self,
        self_preference: tp.Optional[float | VolatileFloat] = None
    ) -> None:
        if self_preference is None:
            self._preference = 1.0 / dist.get_world_size()
        else:
            self._preference = self_preference

    def __call__(self) -> torch.Tensor:
        self_preference = float(self._preference)
        return uniform(self_preference)


class Softmax(Weightage):
    """
    Computes softmax over peer weights.
    """
    def __init__(
        self,
        temperature: float | VolatileFloat = 1.0
    ) -> None:
        self.temperature = temperature

    def __call__(
        self,
        weight: tp.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        temp = float(self.temperature)
        return softmax(weight, temp)


class SoftmaxRefinedUniform(Weightage):
    """
    Composes softmax and uniform weightages.
    """
    def __init__(self, uni: Uniform, sm: Softmax) -> None:
        self._uni = uni
        self._sm = sm

    def __call__(
        self,
        weight: tp.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        uni_weights = self._uni()
        sm_weights = self._sm(weight)
        refined = uni_weights * sm_weights
        return refined / refined.sum()


class PolicySimilarity(Weightage):
    """
    Computes weights based on policy similarity.
    Uses either pure similarity normalization or composition with uniform self-preference.
    """
    def __init__(
        self,
        self_preference: tp.Optional[float | VolatileFloat] = None
    ) -> None:
        self._self_preference = self_preference
        if self_preference is not None:
            self._uniform = Uniform(self_preference)
        else:
            self._uniform = None

    def __call__(
        self,
        similarities: tp.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if similarities is None:
            # Fallback to uniform weights
            if self._uniform is not None:
                return self._uniform()
            else:
                return uniform()
        
        # Ensure similarities are on the correct device
        device = similarities.device
        
        if self._self_preference is not None:
            # Compose with uniform self-preference
            uniform_weights = self._uniform().to(device)  # Move to same device as similarities
            if similarities.sum() > 0:
                normalized_similarities = similarities / similarities.sum()
                # Blend: self_preference * uniform + (1 - self_preference) * similarity
                self_pref_val = float(self._self_preference)
                weights = self_pref_val * uniform_weights + (1 - self_pref_val) * normalized_similarities
                return weights / weights.sum()
            else:
                return uniform_weights
        else:
            # Simple case: just normalize similarities to sum to 1
            if similarities.sum() > 0:
                return similarities / similarities.sum()
            else:
                # If all similarities are 0, use uniform
                return uniform().to(device)


# ----------------- Distributed Weightage Utility Builders ------------------ #
#
def uniform_weightage(
    self_preference: tp.Optional[float | VolatileFloat] = None
) -> Uniform:
    return Uniform(self_preference)

def softmax_weightage(
    temperature: float | VolatileFloat = 1.0
) -> Softmax:
    return Softmax(temperature)

def softmax_refined_uniform_weightage(
    self_preference: tp.Optional[float | VolatileFloat] = None,
    temperature: float | VolatileFloat = 1.0
) -> SoftmaxRefinedUniform:
    return SoftmaxRefinedUniform(
        Uniform(self_preference),
        Softmax(temperature)
    )

def policy_similarity_weightage(
    self_preference: tp.Optional[float | VolatileFloat] = None
) -> PolicySimilarity:
    return PolicySimilarity(self_preference)
