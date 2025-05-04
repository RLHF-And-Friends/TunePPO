from typing import List
from functools import partial

from ppotune.models.llama3_2._component_builders import (
    llama3_2_classifier,
    lora_llama3_2_classifier
)

from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES


def llama3_2_1b_reward() -> TransformerDecoder:

    return llama3_2_classifier(
        num_classes=1,
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=131072,
        attn_dropout=0.0,
        rope_base=500_000,
        intermediate_dim=8192,
        norm_eps=1e-5,
        scale_factor=32,
    )


def llama3_2_3b_reward() -> TransformerDecoder:

    return llama3_2_classifier(
        num_classes=1,
        vocab_size=128_256,
        num_layers=28,
        num_heads=24,
        num_kv_heads=8,
        embed_dim=3072,
        max_seq_len=131072,
        attn_dropout=0.0,
        rope_base=500_000,
        intermediate_dim=8192,
        norm_eps=1e-5,
        scale_factor=32,
    )


def lora_llama3_2_1b_reward(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:

    return lora_llama3_2_classifier(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        num_classes=1,
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_llama3_2_3b_reward(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
           
    return lora_llama3_2_classifier(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        num_classes=1,
        vocab_size=128_256,
        num_layers=28,
        num_heads=24,
        num_kv_heads=8,
        embed_dim=3072,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )

qlora_llama3_2_1b_reward = partial(lora_llama3_2_1b_reward, quantize_base=True)

qlora_llama3_2_3b_reward = partial(lora_llama3_2_3b_reward, quantize_base=True)
