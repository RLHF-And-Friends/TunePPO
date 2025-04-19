import typing as tp

from copy import deepcopy
from torchtune.data import Message
from torchtune.models.mistral import MistralTokenizer as DefaultMistralTokenizer

def system_as_user(messages: tp.Iterable[Message]) -> tp.Iterator[Message]:
    """
    Interprets system role as user. Useful for Mistral where there is no such a
    thing as system role.
    """
    for message in messages:
        new_message = deepcopy(message)
        if new_message.role == "system":
            new_message.role = "user"
        yield new_message

def collapse(messages: tp.Iterable[Message]) -> tp.Iterator[Message]:
    """
    Concats neighbour messages of same role into one message.
    """
    iterator = iter(messages)
    try:
        cur = deepcopy(next(iterator))
    except StopIteration:
        return

    for nxt in iterator:
        if cur.role == nxt.role:
            cur.content.extend(deepcopy(nxt.content))
        else:
            yield cur
            cur = deepcopy(nxt)
    yield cur


class Mistral2Tokenizer(DefaultMistralTokenizer):
    """
    Default torchtune MistralTokenizer overload with altered tokenize messages
    logic. No need to customize eos token as there is only a single one
    """
    def tokenize_messages(
        self,
        messages: tp.List[Message],
        *,
        add_generation_prompt: bool = False,
    ) -> tp.Tuple[tp.List[int]]:

        messages = list(collapse(system_as_user(messages)))
        tokens, _ = super().tokenize_messages(
            messages,
            add_eos = not add_generation_prompt
        )
        return tokens

    def decode(
        self,
        token_ids: tp.List[int],
        skip_special_tokens: bool = False, # not usable. compatibility purpose.
    ) -> str:
        return super().decode(token_ids)


def mistral2_tokenizer(
    path: str,
    max_seq_len: tp.Optional[int] = None,
) -> Mistral2Tokenizer:
    return Mistral2Tokenizer(
        path=path,
        max_seq_len=max_seq_len
    )
