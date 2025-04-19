import typing as tp

from torchtune.data import Message
from torchtune.models.llama3 import Llama3Tokenizer as DefaultLlama3Tokenizer


class Llama3Tokenizer(DefaultLlama3Tokenizer):
    """
    Default torchtune Llama3Tokenizer overload with customizable eos token and
    alternated tokenize messages logic.
    """
    def __init__(
        self,
        path: str,
        special_tokens: tp.Optional[tp.Dict[str, int]] = None,
        max_seq_len: tp.Optional[int] = None,
        eos_token: str = "<|end_of_text|>",
    ):
        super().__init__(
            path,
            special_tokens,
            max_seq_len
        )
        self.eos_id = self.special_tokens[eos_token]

    def tokenize_messages(
        self,
        messages: tp.List[Message],
        *,
        add_generation_prompt: bool = False,
    ) -> tp.Tuple[tp.List[int]]:
        """
        Tokenize a list of messages into a list of tokens as defined by default
        chat template.

        Args:
            messages (List[Message]): The list of messages to tokenize.
            add_generation_prompt (bool): Whether to append generation prompt.
            add_eos (bool): Whether to append eos at the end.

        Returns:
            List[int] The list of token ids.
        """
        tokens = [self.bos_id]

        for message in messages:
            tokenized_message = self.tokenize_message(message)
            tokens = tokens + tokenized_message

        if add_generation_prompt:
            tokens = tokens + self.generation_prompt()

        if self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        return tokens

    def decode(
        self,
        token_ids: tp.List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        return super().decode(
            token_ids,
            truncate_at_eos=False,
            skip_special_tokens=skip_special_tokens
        )

    def generation_prompt(self) -> tp.List[int]:
        return self._tokenize_header(
            Message(role="assistant", content="")
        )

    def __call__(
        self, sample: tp.Mapping[str, tp.Any], inference: bool = False
    ) -> tp.Mapping[str, tp.Any]:
        raise NotImplementedError()


def llama3_tokenizer(
    path: str,
    max_seq_len: tp.Optional[int] = None,
    eos_token: str = "<|end_of_text|>",
) -> Llama3Tokenizer:
    """
    Llama3Tokenizer overload builder.

    Args:
        path (str): path to the tokenizer
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a
            single list of messages, after which the input will be truncated.
        eos_token: str = "<|end_of_text|>", Text representaion of special token
            used as eos

    Returns:
        Llama3Tokenizer: Instantiation of the Llama3 tokenizer
    """
    return Llama3Tokenizer(
        path=path,
        max_seq_len=max_seq_len,
        eos_token=eos_token
    )
