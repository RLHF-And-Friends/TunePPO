from typing import List, Optional
from functools import partial

import torchtune.models.llama3

from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES
from torchtune.data._prompt_templates import _get_prompt_template, _TemplateType

from ppotune.models.llama3_1._component_builders import (
    llama3_1_classifier,
    lora_llama3_1_classifier
)

from torchtune.models.llama3._tokenizer import Llama3Tokenizer


"""
Model builders build specific instantiations using component builders. 
For example the llama3_1_8b model builder uses the llama3 component builder 
to create the Llama3.1 8B model.
"""


def llama3_1_reward_8b() -> TransformerDecoder:
    """
    Builder for creating a reward model based on Llama3.1 model initialized w/ 
    the default 8b parameter values.

    Returns:
        TransformerDecoder: Instantiation of Llama3.1 classifier 8B model
    """
    return llama3_1_classifier(
        num_classes=1,
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=131072,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
    )


def lora_llama3_1_reward_8b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a reward model based on Llama3.1 classifier 8B model 
    with LoRA enabled.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): dropout probability for the low-rank approximation
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Llama3.1 8B model with LoRA applied
    """
    return lora_llama3_1_classifier(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        num_classes=1,
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=131072,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


qlora_llama3_1_reward_8b = partial(lora_llama3_1_reward_8b, quantize_base=True)

qlora_llama3_1_reward_8b.__doc__ = """
Builder for creating a Llama3.1 8B model with QLoRA enabled. Base model 
weights in linear layers that LoRA is applied to are quantized per the 
QLoRA paper: https://arxiv.org/abs/2305.14314. Please see `lora_llama3_1_8b` 
for full API arguments.
"""


def llama3_tokenizer(
    path: str,
    special_tokens_path: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    prompt_template: Optional[_TemplateType] = None,
    truncation_type: str = "right",
) -> Llama3Tokenizer:
    """
    Tokenizer for Llama3 with modified eos_id (for instruct models)

    Args:
        path (str): path to the tokenizer
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file
            structured similarly. Default is None to use the canonical Llama3 special tokens.
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags.
        truncation_type (str): type of truncation to apply, either "left" or "right".
            Default is "right".

    Returns:
        Llama3Tokenizer: Instantiation of the Llama3 tokenizer
    """

    tokenizer: Llama3Tokenizer = torchtune.models.llama3.llama3_tokenizer(
        path=path,
        special_tokens_path=special_tokens_path,
        max_seq_len=max_seq_len,
        prompt_template=prompt_template,
        truncation_type=truncation_type
    )

    tokenizer.eos_id = 128009 # "<|eot_id|>": 128009

    return tokenizer
