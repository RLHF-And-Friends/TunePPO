import typing as tp

from functools import partial

from torch.utils.data import Dataset

from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.data._messages import OpenAIToMessages, Transform

from ppotune.data.utils import load_dataset_with_configurations
from torchtune import utils

log = utils.get_logger("DEBUG")


class HelpsteerDataset(Dataset):
    def __init__(
        self,
        source: str,
        tokenizer: ModelTokenizer,
        message_transform: Transform,
        configurations: tp.Optional[str | tp.List[str] | tp.Dict[int, tp.List[str]]] = None,
        filter_fn: tp.Optional[tp.Callable] = None,
        **load_dataset_kwargs
    ) -> None:
        self.data = load_dataset_with_configurations(
            source,
            configurations,
            **load_dataset_kwargs
        )

        self._message_transform = message_transform
        self._tokenizer = tokenizer

        if filter_fn is not None:
            self.data = self.data.filter(filter_fn)
            log.debug(f"Dataset length after filtering: {self.__len__()}")

    def __getitem__(self, index):
        sample = self.data[index]
        response = sample["best_response"]
        
        messages = self._message_transform(sample)
        
        tokens = self._tokenizer.tokenize_messages(
            messages["messages"],
            add_generation_prompt = True,
        )
        
        return {"tokens": tokens, "completion": response}
    
    def __len__(self):
        return len(self.data)


# Filtering function
# =================================================================================================

def filter_by_length(
    sample,
    tokenizer: ModelTokenizer,
    message_transform: tp.Callable,
    max_tokens: int
) -> None:
    messages = message_transform(sample)
    tokens = tokenizer.tokenize_messages(messages["messages"])

    return len(tokens) < max_tokens


# Helpsteer builder
# =================================================================================================

def helpsteer_dataset(
    tokenizer: ModelTokenizer,
    source: str,
    configurations: tp.Optional[str | tp.List[str] | tp.Dict[int, tp.List[str]]] = None,
    max_input_tokens: tp.Optional[int] = None,
    **load_dataset_kwargs
) -> HelpsteerDataset:
    
    message_transform = OpenAIToMessages(
        train_on_input=True,
        column_map={"messages": "context"},
    )
    
    if max_input_tokens is not None:
        filter_fn = partial(
            filter_by_length,
            tokenizer=tokenizer,
            message_transform=message_transform,
            max_tokens=max_input_tokens
        )
    else:
        filter_fn = None

    return HelpsteerDataset(
        source=source,
        tokenizer=tokenizer,
        message_transform=message_transform,
        configurations=configurations,
        filter_fn=filter_fn,
        **load_dataset_kwargs
    )
        