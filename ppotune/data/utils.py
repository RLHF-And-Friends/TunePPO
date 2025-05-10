import typing as tp
from torchtune.data import Message

import torch.distributed as dist

from datasets import (
    load_dataset,
    concatenate_datasets,
    get_dataset_config_names
)


class PrefixSuffix(tp.NamedTuple):
    """
    Prefix-Postfix pair to wrap-up messages
    """
    prefix: str
    suffix: str


class PromptTemplate(tp.TypedDict, total=False):
    """
    Chat Template alternative for non-instruct models and tasks. Contains
    prefix-suffix pair to wrap-up message for each role.
    """
    system: PrefixSuffix
    user: PrefixSuffix
    assistant: PrefixSuffix
    ipyhon: PrefixSuffix


def apply_prompt_template(
    template: PromptTemplate,
    messages: tp.List[Message],
    add_generation_prompt: bool = False
) -> str:
    """
    Applies prompt template to a list of messages resulting in single string.
    """
    result = ""
    for message in messages:
        result += (""
            + template[message.role].prefix
            + message.text_content
            + template[message.role].suffix
        )
    if add_generation_prompt:
        result += template["assistant"].prefix

    return result


def load_dataset_with_configurations(
    source: str,
    configurations: tp.Optional[str | tp.List[str] | tp.Dict[int, tp.List[str]]] = None,
    **load_dataset_kwargs: tp.Dict[str, tp.Any],
):

    if isinstance(configurations, str):
        data = load_concat_dataset(
            source,
            [configurations],
            **load_dataset_kwargs
        )
    elif isinstance(configurations, list):
        data = load_concat_dataset(
            source,
            configurations,
            **load_dataset_kwargs
        )
    elif isinstance(configurations, dict):
        worker_configurations = configurations[dist.get_rank()]
        data = load_concat_dataset(
            source,
            worker_configurations,
            **load_dataset_kwargs
        )
    elif configurations is None:
        all_configurations = get_dataset_config_names(source)
        data = load_concat_dataset(
            source,
            all_configurations,
            **load_dataset_kwargs
        )
    else:
        raise Exception("Wrong configurations format!")

    return data

  
def load_concat_dataset(
    source: str,
    configurations: list[str],
    **load_dataset_kwargs
):
    data = concatenate_datasets([
        load_dataset(
            path=source,
            name=configuration,
            **load_dataset_kwargs
        ) for configuration in configurations
    ])

    return data
