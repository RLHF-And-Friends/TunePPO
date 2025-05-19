import typing as tp
import torch
import torch.nn as nn

from omegaconf import DictConfig
from torchao.dtypes import NF4Tensor
from torchtune.modules.peft import LoRALinear
from torchtune.modules.peft import get_lora_module_names, get_merged_lora_ckpt
from torchtune.modules.peft.lora import to_nf4
from torchtune.training import MODEL_KEY


def get_adapter_config(cfg_model: DictConfig) -> tp.Dict[str, tp.Any]:
    """
    Retrieves adapter config from model config suitable for checkpointing.
    """
    return {
        "r": cfg_model.lora_rank,
        "lora_alpha": cfg_model.lora_alpha,
        "target_modules": get_lora_module_names(
            list(cfg_model.lora_attn_modules),
            getattr(cfg_model, "apply_lora_to_mlp", False),
            getattr(cfg_model, "apply_lora_to_output", False)
        ),
        "peft_type": "LORA",
    }

def get_merged_adapter_state_dict(
    module: nn.Module,
    module_adapter_config: dict[str, tp.Any]
) -> tp.Dict[str, tp.Any]:
    """
    Constructs model merged with adapter state dict.
    """
    return {
        MODEL_KEY: get_merged_lora_ckpt(
            state_dict  = {k: v.cpu() for k, v in module.state_dict().items()},
            rank        = module_adapter_config["r"],
            alpha       = module_adapter_config["lora_alpha"],
        )
    }

def get_lora_modules(model: nn.Module) -> tp.Iterator[LoRALinear]:
    """
    Returns an iterator over all LoRA modules in the network.
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield module

def get_lora_linears(model: nn.Module) -> tp.Iterator[nn.Linear]:
    """
    Returns an iterator over all LoRA A and B linear layers in the network.
    """
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield module.lora_a
            yield module.lora_b

@torch.no_grad()
def merge_lora_adapter(model: nn.Module) -> nn.Module:
    """
    Merges (Q)LoRA adapters into base model in-place.
    """
    for m in get_lora_modules(model):
        if isinstance(m.weight, NF4Tensor):
            dequantw = m.weight.get_original_weight()
            dequantw += (m.alpha / m.rank) * m.lora_b.weight @ m.lora_a.weight
            m.weight = nn.Parameter(to_nf4(dequantw))
        else:
            m.weight += (m.alpha / m.rank) * m.lora_b.weight @ m.lora_a.weight

    return model

@torch.no_grad()
def clear_lora_adapter(model: nn.Module) -> nn.Module:
    """
    Reinitialize LoRA adapters.
    """
    for m in get_lora_modules(model):
        m.initialize_parameters()

    return model
