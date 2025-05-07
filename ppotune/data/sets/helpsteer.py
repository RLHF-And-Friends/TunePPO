import typing as tp

from functools import partial

from ppotune.data.sets.base import BaseDataset

from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.data._messages import OpenAIToMessages


class HelpsteerDataset(BaseDataset):
    def __init__(
        self,
        tokenizer: ModelTokenizer,
        source: str,
        configurations: tp.Optional[str | tp.List[str] | tp.Dict[int, tp.List[str]]] = None,
        **load_dataset_kwargs
    ) -> None:
        super().__init__(source, configurations, **load_dataset_kwargs)
        
        self._message_transform = OpenAIToMessages(
            train_on_input=True,
            column_map={"messages": "context"},
        )
        self._tokenizer = tokenizer

    def __getitem__(self, index):
        sample = self.data[index]
        response = sample["best_response"]
        
        messages = self._message_transform(sample)
        
        tokens = self._tokenizer.tokenize_messages(messages)
        
        return {"tokens": tokens, "completion": response}


helpsteer_dataset = partial(HelpsteerDataset)
helpsteer_dataset.__doc__ = """
Builder for helpsteer dataset
"""