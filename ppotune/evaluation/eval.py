import typing as tp
import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.data import Dataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer
from ppotune.arbiters.pairwise_arbiter import PairwiseArbiter
from ppotune.data.loaders import DataloaderConfig, build_dataloader
from ppotune.log import WandbLogger
from ppotune.model import GenerativeModel


logger = WandbLogger()


class Evaluator(tp.Protocol):
    """
    Evaluator protocol.
    """
    def setup(
        self,
        tokenizer: ModelTokenizer,
        seed: int = 0
    ) -> None:
        ...

    def __call__(
        self,
        model: GenerativeModel,
        step: int
    ) -> None:
        """
        Performs evaluation at each n-th step and logs the result.
        """
        ...


class EvaluationGroup(Evaluator):
    """
    Collection of evaluators.
    """
    def __init__(self, evaluators: tp.List[Evaluator]) -> None:
        self._evaluators = evaluators

    def setup(self, tokenizer: ModelTokenizer, seed: int = 0) -> None:
        for eval in self._evaluators:
            eval.setup(tokenizer)

    def __call__(
        self,
        model: GenerativeModel,
        step: int = 0,
    ) -> None:
        """
        Triggers all evaluators.
        """
        for eval in self._evaluators:
            eval(model, step)


class ReferenceCompletionEvaluator(Evaluator):
    """
    Evaluates model based on reference completion dataset w.r.t pairwise
    arbiter opinion.
    """
    def __init__(
        self,
        arbiter: PairwiseArbiter,
        every_n_steps: int,
        dataset: Dataset,
        dataloader_config: DataloaderConfig,
        tag: str = "validation"
    ) -> None:
        self._arbiter = arbiter
        self._every_n_steps = every_n_steps
        self._dataset = dataset
        self._loader_config = dataloader_config
        self._tag = tag

    def setup(
        self,
        tokenizer: ModelTokenizer,
        seed: int = 0,
    ) -> None:

        self.decode = lambda tokens: tokenizer.decode(
            tokens.tolist(), skip_special_tokens=True
        )
        self._dataset.setup(tokenizer)
        self._dataloader = build_dataloader(
            dataset=self._dataset,
            seed=seed,
            **self._loader_config,
        )

    def __call__(
        self,
        model: GenerativeModel,
        step: int = 0,
    ) -> None:

        if step % self._every_n_steps != 0:
            return

        prompts:        tp.List[str] = []
        completions:    tp.List[tp.Tuple[str, str]] = []

        for batch in tqdm(
            self._dataloader,
            desc=f"Evaluation ({self._tag})",
            disable=dist.get_rank() != 0
        ):
            batch["tokens"] = batch["tokens"].to(model._device)
            generated = model.generate(prompt=batch["tokens"])

            queries = []
            responses = []
            for tokens, query_mask, response_mask in zip(
                generated.tokens, generated.query_mask, generated.response_mask
            ):
                queries.append(self.decode(tokens[query_mask]))
                responses.append(self.decode(tokens[response_mask]))

            prompts.extend(queries)
            completions.extend(list(zip(batch["completion"], responses)))

        wins = torch.tensor(self._arbiter.judge(prompts, completions))
        valid = wins != -1
        winrate = wins[valid].float().mean()
        logger.collect(f"{self._tag}-winrate", winrate)

        for idx in range(self._dataloader.batch_size):
            logger.collect_reference(
                reference=completions[idx][0],
                completion=completions[idx][1],
                chosen=wins[idx]
            )


def evaluation_group(evaluators: tp.List[Evaluator]) -> EvaluationGroup:
    return EvaluationGroup(evaluators)

def reference_completion_evaluator(
        arbiter: PairwiseArbiter,
        every_n_steps: int,
        dataset: Dataset,
        dataloader_config: DataloaderConfig,
        tag: str = "validation",
) -> ReferenceCompletionEvaluator:

    return ReferenceCompletionEvaluator(
        arbiter=arbiter,
        every_n_steps=every_n_steps,
        dataset=dataset,
        dataloader_config=dataloader_config,
        tag=tag
    )
