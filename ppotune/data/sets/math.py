import typing as tp

from torch.utils.data import Dataset
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform
from torchtune.data import Message
from torchtune import utils

from datasets import load_dataset

from ppotune.data.utils import PromptTemplate, PrefixSuffix, apply_prompt_template

log = utils.get_logger("DEBUG")


# MATH dataset has 7 subjects - perfect for personalized multi-agent setup
MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


MATH_SYSTEM_PROMPT: str = (
    "A conversation between User and Assistant. The user asks a math question, and "
    "the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The "
    "reasoning process and answer are enclosed within <think></think> and "
    "<answer></answer> tags, respectively, i.e., <think>reasoning process "
    "here</think> <answer>answer here</answer>."
)

MATH_PROMPT_TEMPLATE: PromptTemplate = {
    "system": PrefixSuffix("", " "),
    "user": PrefixSuffix("User: ", " "),
    "assistant": PrefixSuffix("Assistant: ", "")
}


class MATHProblem(tp.TypedDict):
    question: str
    solution: tp.Optional[str]
    answer: str
    subject: str
    level: int


class MATHTransform(Transform):
    """
    Parses MATH dataset record into question, solution, answer, subject and level fields.
    """
    def __call__(self, sample: tp.Mapping[str, tp.Any]) -> MATHProblem:
        return MATHProblem(
            question=sample["problem"],
            solution=sample.get("solution", ""),
            answer=sample.get("answer", sample.get("expected_answer", "")),
            subject=sample.get("subject", sample.get("type", "")),
            level=int(sample.get("level", "0").replace("Level ", "")) if isinstance(sample.get("level"), str) else sample.get("level", 0),
        )


class MATHDataset(Dataset):
    """
    MATH dataset class for mathematical reasoning tasks.
    Supports filtering by subject (topic) and difficulty level for personalized training.
    """
    def __init__(
        self,
        source: str,
        sample_transform: MATHTransform,
        subjects: tp.Optional[tp.List[str] | tp.Dict[int, tp.List[str]]] = None,
        levels: tp.Optional[tp.List[int]] = None,
        system_prompt: tp.Optional[str] = None,
        prompt_template: tp.Optional[PromptTemplate] = None,
        _preloaded_data: tp.Optional[tp.Any] = None,
        **load_dataset_kwargs: tp.Dict[str, tp.Any],
    ) -> None:

        self.sample_transform = sample_transform
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template

        # Use preloaded data if provided, otherwise load from source
        if _preloaded_data is not None:
            self.data = _preloaded_data
        else:
            self.data = load_dataset(source, **load_dataset_kwargs)

        # Filter by subjects if specified
        if subjects is not None:
            if isinstance(subjects, dict):
                # Distributed setup: each rank gets different subjects
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else 0
                subject_list = subjects.get(rank, MATH_SUBJECTS)
            else:
                subject_list = subjects

            # Normalize subject names for comparison
            subject_list_normalized = [s.lower().replace(" ", "_").replace("&", "and") for s in subject_list]

            def subject_filter(sample):
                sample_subject = sample.get("subject", sample.get("type", "")).lower().replace(" ", "_").replace("&", "and")
                return sample_subject in subject_list_normalized

            self.data = self.data.filter(subject_filter)
            log.debug(f"Dataset length after subject filtering: {len(self.data)}")

        # Filter by difficulty levels if specified
        if levels is not None:
            def level_filter(sample):
                level_val = sample.get("level", "0")
                if isinstance(level_val, str):
                    level_val = int(level_val.replace("Level ", ""))
                return level_val in levels

            self.data = self.data.filter(level_filter)
            log.debug(f"Dataset length after level filtering: {len(self.data)}")

    def setup(self, tokenizer: ModelTokenizer) -> None:
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def _tokenize_question(self, question: str) -> tp.List[int]:
        """
        Tokenize a question according to dataset format.
        """
        messages = []
        if self.system_prompt is not None:
            messages.append(Message(
                role="system",
                content=self.system_prompt,
                eot=True,
            ))

        messages.append(Message(
            role="user",
            content=question,
            eot=True,
        ))

        tokens = []
        if self.prompt_template is None:
            tokens = self.tokenizer.tokenize_messages(
                messages=messages,
                add_generation_prompt=True
            )
        else:
            text = apply_prompt_template(
                template=self.prompt_template,
                messages=messages,
                add_generation_prompt=True
            )
            tokens = self.tokenizer.encode(text, add_eos=False)
            tokens = tokens[:self.tokenizer.max_seq_len]

        return tokens

    def __getitem__(self, index: int) -> tp.Dict[str, tp.Any]:
        sample = self.sample_transform(self.data[index])
        tokens = self._tokenize_question(sample["question"])
        return {
            "tokens": tokens,
            "solution": sample["solution"],
            "answers": sample["answer"],
            "subject": sample["subject"],
            "level": sample["level"],
        }


# Dataset builders
# =================================================================================================

def math_dataset(
    source: str = "hendrycks/competition_math",
    split: str = "train",
    subjects: tp.Optional[tp.List[str] | tp.Dict[int, tp.List[str]]] = None,
    levels: tp.Optional[tp.List[int]] = None,
    prompt_template: tp.Optional[PromptTemplate] = None,
    **load_dataset_kwargs: tp.Dict[str, tp.Any],
) -> MATHDataset:
    """
    Full MATH dataset (12.5K problems) for training.

    Args:
        source: HuggingFace dataset path
        split: Dataset split ("train" or "test")
        subjects: Filter by subject(s). Can be:
            - List of subjects: ["algebra", "geometry"]
            - Dict mapping rank to subjects: {0: ["algebra"], 1: ["geometry"], ...}
        levels: Filter by difficulty levels (1-5)
        prompt_template: Optional prompt template for non-chat models

    Returns:
        MATHDataset instance
    """
    return MATHDataset(
        source=source,
        sample_transform=MATHTransform(),
        subjects=subjects,
        levels=levels,
        system_prompt=MATH_SYSTEM_PROMPT,
        prompt_template=prompt_template,
        split=split,
        **load_dataset_kwargs,
    )


def hendrycks_math_dataset(
    source: str = "EleutherAI/hendrycks_math",
    split: str = "train",
    subjects: tp.Optional[tp.List[str] | tp.Dict[int, tp.List[str]]] = None,
    levels: tp.Optional[tp.List[int]] = None,
    prompt_template: tp.Optional[PromptTemplate] = None,
) -> MATHDataset:
    """
    Full Hendrycks MATH dataset (7.5K train, 5K test) with per-subject configs.

    This dataset requires loading each subject separately and concatenating.

    Args:
        source: HuggingFace dataset path (EleutherAI/hendrycks_math)
        split: Dataset split ("train" or "test")
        subjects: Filter by subject(s). Can be:
            - List of subjects: ["algebra", "geometry"]
            - Dict mapping rank to subjects: {0: ["algebra"], 1: ["geometry"], ...}
        levels: Filter by difficulty levels (1-5)
        prompt_template: Optional prompt template for non-chat models
    """
    from datasets import concatenate_datasets
    import torch.distributed as dist

    # Determine which subjects to load
    if subjects is None:
        subject_list = MATH_SUBJECTS
    elif isinstance(subjects, dict):
        rank = dist.get_rank() if dist.is_initialized() else 0
        subject_list = subjects.get(rank, MATH_SUBJECTS)
    else:
        subject_list = subjects

    # Load and concatenate datasets for each subject
    datasets = []
    for subject in subject_list:
        ds = load_dataset(source, subject, split=split)
        # Add subject field since it's not in the original data
        ds = ds.map(lambda x: {"type": subject, **x})
        datasets.append(ds)

    combined = concatenate_datasets(datasets)
    log.debug(f"Loaded {len(combined)} samples for subjects: {subject_list}")

    # Create dataset without subject filtering (already filtered by loading specific configs)
    return MATHDataset(
        source=source,
        sample_transform=MATHTransform(),
        subjects=None,  # Already filtered
        levels=levels,
        system_prompt=MATH_SYSTEM_PROMPT,
        prompt_template=prompt_template,
        split=split,
        _preloaded_data=combined,  # Pass pre-loaded data
    )


def plain_hendrycks_math_dataset(
    split: str = "train",
    subjects: tp.Optional[tp.List[str] | tp.Dict[int, tp.List[str]]] = None,
    levels: tp.Optional[tp.List[int]] = None,
) -> MATHDataset:
    """
    Hendrycks MATH dataset for base (non-chat) models with prompt template.
    """
    return hendrycks_math_dataset(
        split=split,
        subjects=subjects,
        levels=levels,
        prompt_template=MATH_PROMPT_TEMPLATE,
    )


def chat_math_dataset(
    split: str = "train",
    subjects: tp.Optional[tp.List[str] | tp.Dict[int, tp.List[str]]] = None,
    levels: tp.Optional[tp.List[int]] = None,
    **load_dataset_kwargs: tp.Dict[str, tp.Any],
) -> MATHDataset:
    """
    MATH dataset for chat/instruct models.
    """
    return math_dataset(
        split=split,
        subjects=subjects,
        levels=levels,
        prompt_template=None,
        **load_dataset_kwargs,
    )


def plain_math_dataset(
    split: str = "train",
    subjects: tp.Optional[tp.List[str] | tp.Dict[int, tp.List[str]]] = None,
    levels: tp.Optional[tp.List[int]] = None,
    **load_dataset_kwargs: tp.Dict[str, tp.Any],
) -> MATHDataset:
    """
    MATH dataset for base (non-chat) models with prompt template.
    """
    return math_dataset(
        split=split,
        subjects=subjects,
        levels=levels,
        prompt_template=MATH_PROMPT_TEMPLATE,
        **load_dataset_kwargs,
    )


def math500_dataset(
    source: str = "HuggingFaceH4/MATH-500",
    split: str = "test",
    subjects: tp.Optional[tp.List[str] | tp.Dict[int, tp.List[str]]] = None,
    levels: tp.Optional[tp.List[int]] = None,
    prompt_template: tp.Optional[PromptTemplate] = None,
    **load_dataset_kwargs: tp.Dict[str, tp.Any],
) -> MATHDataset:
    """
    MATH-500 dataset (500 problems) for evaluation.
    """
    return MATHDataset(
        source=source,
        sample_transform=MATHTransform(),
        subjects=subjects,
        levels=levels,
        system_prompt=MATH_SYSTEM_PROMPT,
        prompt_template=prompt_template,
        split=split,
        **load_dataset_kwargs,
    )


def eval_math_dataset(
    split: str = "test",
    **load_dataset_kwargs: tp.Dict[str, tp.Any],
) -> MATHDataset:
    """
    MATH dataset for evaluation (test split).
    """
    return math_dataset(
        split=split,
        **load_dataset_kwargs,
    )
