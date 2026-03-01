import typing as tp
from functools import partial

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.data import Message
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform

from ppotune.data.utils import PrefixSuffix, PromptTemplate, apply_prompt_template


def _to_python_list(obj: tp.Any) -> list[tp.Any]:
    """Convert HuggingFace Arrow types to plain Python list."""
    if obj is None:
        return []
    if hasattr(obj, "as_py"):
        return _to_python_list(obj.as_py())
    if isinstance(obj, list):
        return [_to_python_item(x) for x in obj]
    if isinstance(obj, str):
        import json

        try:
            return _to_python_list(json.loads(obj))
        except json.JSONDecodeError:
            return []
    return []


def _to_python_item(item: tp.Any) -> tp.Any:
    """Convert single item (dict/Arrow struct) to plain Python."""
    if hasattr(item, "as_py"):
        return _to_python_item(item.as_py())
    if isinstance(item, dict):
        return {k: _to_python_item(v) for k, v in item.items()}
    if isinstance(item, list):
        return [_to_python_item(x) for x in item]
    return item


# -------------------------------------------------------------------------------------------------
# System prompt with specific question-answer format
# -------------------------------------------------------------------------------------------------

MULTI_HOP_SYSTEM_PROMPT = """You are a chain-of-thought language model. When the user asks a question you MUST reply in the structure below:
<think>
<question> <first self-generated sub-question> </question> <answer> <answer to the first sub-question> </answer>
<question> <second self-generated sub-question> </question> <answer> <answer to the second sub-question> </answer>
...
</think>
<answer> <final answer to the user’s original question> </answer>

Rules
1. Ask yourself sub-questions and answer them, wrap questions and answers in the indicated tags.
2. All inner tags (<question> / <answer>) live inside a single <think> ... </think> block.
3. After the </think> tag, output one—and only one—final answer to the user question, wrapped in its own outer <answer> ... </answer> tag.
4. Inside answers tags give only answers to the questions without any additional text.
5. Preserve the tag names and their order precisely as specified. 
"""

# -------------------------------------------------------------------------------------------------
# System prompt with general resoning
# -------------------------------------------------------------------------------------------------

BASIC_REASONING_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The "
    "reasoning process and answer are enclosed within <think></think> and "
    "<answer></answer> tags, respectively, i.e., <think>reasoning process "
    "here</think> <answer>answer here</answer>."
)

MODIFIED_REASONING_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the Assistant solves it. Assistant's response consists of thinking and the answer. The "
    "thinking and answer are enclosed within <think></think> and "
    "<answer></answer> tags, respectively, i.e., <think>reasoning process</think>"
    "<answer>answer here</answer>. You need reply only by one <think>...</think> section and one <answer>...</answer> section. "
    "After first <answer>...</answer> section, you need to stop generating any text and return <|eot_id|> token. "
    "Answer section should be short and concise. Thinking section should be detailed and comprehensive.\n"
)

RULETAKER_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user provides a Context and a Question. "
    "Determine whether the Question follows from the Context (entailment) or not (not entailment). "
    "Assistant's response consists of thinking and the answer. The thinking and answer are enclosed "
    "within <think></think> and <answer></answer> tags. You must reply with exactly one <think>...</think> "
    "section and one <answer>...</answer> section. The text in answer section must be ONLY one of: entailment or not entailment. "
    "After the <answer>...</answer> section, stop and return <|eot_id|> token.\n"
)

REASONING_SYSTEM_PROMPT_V2 = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it using structured thinking.
Format your response as:
<think>
Known facts: [List specific facts you know]
Analysis: [Connect these facts logically]
Conclusion: [What follows from the facts]
</think>
<answer>[concise answer]</answer>

Rules:
- Present only factual information and logical reasoning
- No meta-commentary about your thinking process
- No "I need to search" or "I will consider" statements
- Stop after </answer> tag with <|eot_id|>
- Answer section should be short and concise. Thinking section should be detailed and comprehensive.
- You need reply only by one <answer>...</answer> section and one <think>...</think> section."""

# -------------------------------------------------------------------------------------------------
# Prompt tamplate for non-chat models
# -------------------------------------------------------------------------------------------------

BASIC_PROMPT_TEMPLATE: PromptTemplate = {
    "system": PrefixSuffix("System prompt: ", "\n"),
    "user": PrefixSuffix("User: ", "\n"),
    "assistant": PrefixSuffix("Assistant: ", "\n"),
}

# -------------------------------------------------------------------------------------------------


class MultiHopProblem(tp.TypedDict):
    question: str
    answers: tp.List[str]
    path: tp.Union[str, tp.List[tp.Any]]
    final_answer: tp.List[str]


class MultihopTransform(Transform):
    def __call__(self, sample: tp.Mapping[str, tp.Any]) -> MultiHopProblem: ...


class MultiHopDataset(Dataset):
    def __init__(
        self,
        source: str,
        sample_transform: MultihopTransform,
        filter_fn: tp.Optional[tp.Callable[..., tp.Any]] = None,
        system_prompt: tp.Optional[str] = None,
        prompt_template: tp.Optional[tp.Union[str, PromptTemplate]] = None,
        **load_dataset_kwargs: tp.Any,
    ) -> None:
        self._data = load_dataset(path=source, **load_dataset_kwargs)
        self._sample_transform = sample_transform
        self._system_prompt = system_prompt
        self._prompt_template = prompt_template

        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

    def setup(self, tokenizer: ModelTokenizer) -> None:
        self._tokenizer = tokenizer

    def _tokenize_question(self, question: str) -> tp.List[int]:
        messages = []
        if self._system_prompt is not None:
            messages.append(Message(role="system", content=self._system_prompt, eot=True))
        messages.append(
            Message(
                role="user",
                content=question,
                eot=True,
            )
        )

        if self._prompt_template is None:
            out = self._tokenizer.tokenize_messages(
                messages=messages, add_generation_prompt=True  # type: ignore[arg-type]
            )
            tokens: tp.List[int] = out[0] if isinstance(out, tuple) else out
        else:
            text = apply_prompt_template(
                template=tp.cast(PromptTemplate, self._prompt_template),
                messages=messages,
                add_generation_prompt=True,
            )
            tokens = list(self._tokenizer.encode(text, add_eos=False))  # type: ignore[attr-defined]
            tokens = list(tokens[: self._tokenizer.max_seq_len])

        return tokens

    def __getitem__(self, index: int) -> dict[str, tp.Any]:
        sample = self._sample_transform(self._data[index])
        tokens = self._tokenize_question(sample["question"])
        return {
            "tokens": tokens,
            "answers": sample["answers"],
            "path": sample["path"],
            "final_answer": sample["final_answer"],
            "question": sample["question"],
        }

    def __len__(self) -> int:
        return len(self._data)


class OneHopTransform(MultihopTransform):
    def __call__(self, sample: tp.Mapping[str, tp.Any]) -> MultiHopProblem:
        question = sample["generated_question"]  # use only first question in 2hop
        answers = [sample["first_entity_aliases"]]
        final_answer = sample["second_entity_aliases"]
        path = sample["path"]
        return MultiHopProblem(
            question=question, answers=answers, path=path, final_answer=final_answer
        )


class TwoHopTransform(MultihopTransform):
    def __call__(self, sample: tp.Mapping[str, tp.Any]) -> MultiHopProblem:
        question = sample["generated_question"]
        answers = []
        answers.append(sample["second_entity_aliases"])
        answers.append(sample["third_entity_aliases"])
        final_answer = sample["third_entity_aliases"]
        path = sample["path"]

        return MultiHopProblem(
            question=question, answers=answers, path=path, final_answer=final_answer
        )


class ThreeHopTransform(MultihopTransform):
    def __call__(self, sample: tp.Mapping[str, tp.Any]) -> MultiHopProblem:
        question = sample["generated_question"]
        answers = []
        answers.append(sample["second_entity_aliases"])
        answers.append(sample["third_entity_aliases"])
        answers.append(sample["fourth_entity_aliases"])
        final_answer = sample["fourth_entity_aliases"]
        path = sample["path"]

        return MultiHopProblem(
            question=question, answers=answers, path=path, final_answer=final_answer
        )

class MQuAKETransform(MultihopTransform):
    def __call__(self, sample: tp.Mapping[str, tp.Any]) -> MultiHopProblem:
        question = sample["question"]
        answers = sample["aliases"]
        final_answer = sample["aliases"][-1]
        path = sample["path"]

        return MultiHopProblem(
            question=question, answers=answers, path=path, final_answer=final_answer
        )


class RuletakerTransform(MultihopTransform):
    """
    Transform for ruletaker format (context, question, path, label).
    Path: list of {fact: {rel, src, tgt}}.
    """

    def __call__(self, sample: tp.Mapping[str, tp.Any]) -> MultiHopProblem:
        question = sample.get("question", "")
        context = sample.get("context", "")
        if context and question:
            question = f"Context: {context}\n\nQuestion: {question}"
        elif context:
            question = context
        raw_path = sample.get("path", sample.get("graph", []))
        path = _to_python_list(raw_path) if raw_path else []

        label = sample.get("label", "")
        if isinstance(label, str):
            final_answer: tp.List[str] = [label] if label else []
        else:
            final_answer = list(label) if label else []
        answers: tp.List[str] = []

        return MultiHopProblem(
            question=question, answers=answers, path=path, final_answer=final_answer
        )

class Math500Problem(tp.TypedDict):
    problem: str
    answer: str

class Math500Transform(Transform):
    def __call__(self, sample: tp.Mapping[str, tp.Any]) -> Math500Problem:
        return Math500Problem(problem=sample["problem"], answer=sample["answer"])

class Math500Dataset(Dataset):
    def __init__(
        self,
        source: str,
        sample_transform: Math500Transform,
        filter_fn: tp.Optional[tp.Callable[..., tp.Any]] = None,
        system_prompt: tp.Optional[str] = None,
        prompt_template: tp.Optional[tp.Union[str, PromptTemplate]] = None,
        **load_dataset_kwargs: tp.Any,
    ) -> None:
        self._data = load_dataset(path=source, **load_dataset_kwargs)
        self._sample_transform = sample_transform
        self._system_prompt = system_prompt
        self._prompt_template = prompt_template

        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

    def setup(self, tokenizer: ModelTokenizer) -> None:
        self._tokenizer = tokenizer

    def _tokenize_question(self, question: str) -> tp.List[int]:
        messages = []
        if self._system_prompt is not None:
            messages.append(Message(role="system", content=self._system_prompt, eot=True))
        messages.append(
            Message(
                role="user",
                content=question,
                eot=True,
            )
        )

        if self._prompt_template is None:
            out = self._tokenizer.tokenize_messages(
                messages=messages, add_generation_prompt=True  # type: ignore[arg-type]
            )
            tokens = out[0] if isinstance(out, tuple) else out
        else:
            text = apply_prompt_template(
                template=tp.cast(PromptTemplate, self._prompt_template),
                messages=messages,
                add_generation_prompt=True,
            )
            tokens = list(self._tokenizer.encode(text, add_eos=False))  # type: ignore[attr-defined]
            tokens = list(tokens[: self._tokenizer.max_seq_len])

        return tokens

    def __getitem__(self, index: int) -> tp.Dict[str, tp.Any]:
        sample = self._sample_transform(self._data[index])
        tokens = self._tokenize_question(sample["problem"])
        return {
            "tokens": tokens,
            "final_answer": [sample["answer"]],
            "path": "",
        }

    def __len__(self) -> int:
        return len(self._data)


one_hop_dataset = partial(
    MultiHopDataset,
    sample_transform=OneHopTransform(),
    # system_prompt=BASIC_REASONING_SYSTEM_PROMPT,
    system_prompt=MODIFIED_REASONING_SYSTEM_PROMPT,
    prompt_template=BASIC_PROMPT_TEMPLATE,
)

two_hop_dataset = partial(
    MultiHopDataset,
    sample_transform=TwoHopTransform(),
    system_prompt=MODIFIED_REASONING_SYSTEM_PROMPT,
    prompt_template=BASIC_PROMPT_TEMPLATE,
)
three_hop_dataset = partial(
    MultiHopDataset,
    sample_transform=ThreeHopTransform(),
    # system_prompt=BASIC_REASONING_SYSTEM_PROMPT,
    system_prompt=MODIFIED_REASONING_SYSTEM_PROMPT,
    prompt_template=BASIC_PROMPT_TEMPLATE,
)

mquake_dataset = partial(
    MultiHopDataset,
    sample_transform=MQuAKETransform(),
    system_prompt=MODIFIED_REASONING_SYSTEM_PROMPT,
    prompt_template=BASIC_PROMPT_TEMPLATE,
)

rultaker_dataset = partial(
    MultiHopDataset,
    sample_transform=RuletakerTransform(),
    system_prompt=RULETAKER_SYSTEM_PROMPT,
    prompt_template=BASIC_PROMPT_TEMPLATE,
)

math500_dataset = partial(
    Math500Dataset,
    sample_transform=Math500Transform(),
    system_prompt=MODIFIED_REASONING_SYSTEM_PROMPT,
    prompt_template=BASIC_PROMPT_TEMPLATE,
)
