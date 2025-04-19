from omegaconf import DictConfig
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp

from torchtune.training import (
    Checkpointer,
    MODEL_KEY,
    set_activation_checkpointing
)
from torchtune.modules import (
    TransformerDecoder,
    TransformerSelfAttentionLayer,
    local_kv_cache
)
from torchtune.modules.peft import (
    get_adapter_params,
    set_trainable_params,
)
from torchtune.modules.transforms.tokenizers import ModelTokenizer
from torchtune import generation
from torchtune import utils
from ppotune.peft import (
    get_merged_adapter_state_dict,
    get_adapter_config
)


@dataclass
class GenerationResult:
    tokens:         torch.IntTensor     # B x (Q + R)
    query_mask:     torch.BoolTensor    # B x (Q + R)
    response_mask:  torch.BoolTensor    # B x (Q + R)
    logits:         torch.FloatTensor   # B x R x V


class GenerativeModel(tp.Protocol):
    """
    A model that can generate response to a prompt
    """
    def generate(self, prompt: torch.Tensor) -> GenerationResult:
        ...


class LoRAModel(nn.Module):
    def __init__(
        self,
        ckpt: Checkpointer,
        model: TransformerDecoder,
    ) -> None:

        super().__init__()
        self.ckpt = ckpt
        self.model = model
        set_trainable_params(self.model, get_adapter_params(self.model))
        self._dtype=torch.get_default_dtype()
        self._device=utils.get_device("cuda")

    def setup(self, cfg: DictConfig) -> None:

        self._adapter_config = get_adapter_config(cfg.model)
        state_dict = self.ckpt.load_checkpoint()[MODEL_KEY]
        set_activation_checkpointing(
            self.model, auto_wrap_policy={TransformerSelfAttentionLayer}
        )
        # warning: strict=False allows not all model params to be initialized
        # wich is dangerous but is necessary to load with no LoRAs predefined.
        self.model.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:

        logits = self.model(
            tokens,
            mask=mask,
            input_pos=input_pos
        )
        return logits

    def save_checkpoint(self, epoch: int = 0) -> None:
        self.ckpt.save_checkpoint(
            state_dict=get_merged_adapter_state_dict(
                module=self.model,
                module_adapter_config=self._adapter_config
            ),
            epoch=epoch
        )


class GenerativeLoRAModel(LoRAModel, GenerativeModel):
    def __init__(
        self,
        ckpt: Checkpointer,
        model: TransformerDecoder,
        tokenizer: ModelTokenizer,
        max_response_len: int,
        generation_batch_size: int,
        temperature: float,
        top_k: tp.Optional[int] = None,
        rng: tp.Optional[torch.Generator] = None,
    ) -> None:

        super().__init__(ckpt, model)
        self._tokenizer = tokenizer
        self._max_response_len = max_response_len
        self._generation_batch_size = generation_batch_size
        self._temperature = temperature
        self._top_k = top_k
        self._rng = rng

    @property
    def max_query_len(self) -> int:
        return self._tokenizer.max_seq_len

    @property
    def max_response_len(self) -> int:
        return self._max_response_len

    @property
    def batch_size(self) -> int:
        return self._generation_batch_size

    @property
    def temperature(self) -> float:
        return self._temperature

    def setup(self, cfg: DictConfig) -> None:
        super().setup(cfg)
        self.cache_ctx_manager = lambda: local_kv_cache(
            self.model,
            batch_size=self.batch_size,
            dtype=self._dtype,
            decoder_max_seq_len=self.max_query_len + self.max_response_len,
            device=self._device
        )

    def generate(self, prompt: torch.Tensor) -> GenerationResult:
        with self.cache_ctx_manager():
            tokens, logits = generation.generate(
                model=self.model,
                prompt=prompt,
                max_generated_tokens=self.max_response_len,
                temperature=self._temperature,
                top_k=self._top_k,
                pad_id=self._tokenizer.pad_id,
                rng=self._rng,
            )

        query_len = prompt.shape[1]
        query_mask = tokens != self._tokenizer.pad_id
        query_mask[:, query_len:] = False # mask out responses

        responses = tokens[:, query_len:]
        eos_mask = (responses == self._tokenizer.eos_id)
        seen_eos = torch.cumsum(eos_mask, dim=1)
        responses_pad_mask = (seen_eos > 1) | ((seen_eos == 1) & ~eos_mask)

        response_mask = query_mask.clone()
        response_mask[:, :query_len] = False # mask out queries
        response_mask[:, query_len:] = ~responses_pad_mask

        return GenerationResult(
            tokens=tokens,
            query_mask=query_mask,
            response_mask=response_mask,
            logits=logits
        )

    def logits_to_logprobs(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Converts logits corresponding to a generated token sequence to logprobs
        Args:
            logits (torch.Tensor): B x L x V - logits
            tokens (torch.Tensor): B x L - corresponding tokens
        Returns:
            torch.Tensor: B x L - log probs corresponding to each token.
        """
        return torch.gather(
            F.log_softmax(logits / self.temperature, dim=-1),
            2,
            tokens.unsqueeze(-1),
        ).squeeze(-1)
