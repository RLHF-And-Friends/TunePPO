import typing as tp
import torch

from tqdm import tqdm
from torch import Generator
from torch.utils.data import RandomSampler, DataLoader, Dataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer
from ppotune.arbiters.pairwise_arbiter import PairwiseArbiter
from ppotune.datasets.utils import LeftPadCollator
from ppotune.log import WandbLogger
from ppotune.model import GenerativeModel


logger = WandbLogger()

class Evaluator(tp.Protocol):
    """
    Evaluator protocol.
    """
    def __call__(
        model: GenerativeModel,
        step: int
    ) -> None:
        """
        Performs evaluation at each n-th step and logs the result.
        """
        ...

class ReferenceCompletionEvaluator(Evaluator):
    """
    Evaluates model based on reference completion dataset w.r.t pairwise
    arbiter opinion.
    """
    def __init__(
        self,
        arbiter: PairwiseArbiter,
        num_samples: int,
        every_n_steps: int,
        *, # the rest has to be defined in the recipe
        dataset: Dataset,
        batch_size: int,
    ) -> None:

        self._arbiter = arbiter
        self._tokenizer: ModelTokenizer = dataset.tokenizer
        self._every_n_steps = every_n_steps

        sampler = RandomSampler(
            dataset,
            num_samples=num_samples,
        )
        collator = LeftPadCollator(
            tokens_key="tokens",
            pad_token=self._tokenizer.pad_id
        )
        self._dataloader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=True,
            collate_fn=collator
        )

    def __call__(
        self,
        model: GenerativeModel,
        step: int,
    ) -> None:

        if step % self._every_n_steps != 0:
            return

        prompts:        tp.List[str] = []
        completions:    tp.List[tp.Tuple[str, str]] = []

        decode = lambda tokens: self._tokenizer.decode(
            tokens.tolist(), skip_special_tokens=True
        )
        for batch in tqdm(self._dataloader, desc="Evaluation"):
            batch["tokens"] = batch["tokens"].to(model._device)
            generated = model.generate(prompt=batch["tokens"])

            queries = []
            responses = []
            for tokens, query_mask, response_mask in zip(
                generated.tokens, generated.query_mask, generated.response_mask
            ):
                queries.append(decode(tokens[query_mask]))
                responses.append(decode(tokens[response_mask]))

            prompts.extend(queries)
            completions.extend(list(zip(batch["completion"], responses)))

        wins = self._arbiter.judge(prompts, completions)
        valid = wins != -1
        winrate = torch.tensor(wins[valid]).float().mean()
        logger.collect("winrate", winrate)


def reference_completion_evaluator(
        arbiter: PairwiseArbiter,
        num_samples: int,
        every_n_steps: int,
        *, # the rest has to be defined in the recipe
        dataset: Dataset,
        batch_size: int,
) -> ReferenceCompletionEvaluator:

    return ReferenceCompletionEvaluator(
        arbiter=arbiter,
        num_samples=num_samples,
        every_n_steps=every_n_steps,
        dataset=dataset,
        batch_size=batch_size,
    )
