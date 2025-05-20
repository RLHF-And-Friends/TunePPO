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
        tag: str = "validation",
        num_logs: tp.Optional[int] = None,
        empty_cache_after_generation: bool = False,
    ) -> None:
        self._arbiter = arbiter
        self._every_n_steps = every_n_steps
        self._dataset = dataset
        self._loader_config = dataloader_config
        self._tag = tag
        self._num_logs = num_logs
        self._empty_cache = empty_cache_after_generation

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
        if self._num_logs is None:
            self._num_logs = self._dataloader.batch_size

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
            torch.cuda.empty_cache() if self._empty_cache else None

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
        logger.collect_table(f"{self._tag}-reference", {
            "reference":    [c[0] for c in completions[:self._num_logs]],
            "completion":   [c[1] for c in completions[:self._num_logs]],
            "chosen":       wins[0:self._num_logs]
        })


def evaluation_group(evaluators: tp.List[Evaluator]) -> EvaluationGroup:
    return EvaluationGroup(evaluators)

def reference_completion_evaluator(
        arbiter: PairwiseArbiter,
        every_n_steps: int,
        dataset: Dataset,
        dataloader_config: DataloaderConfig,
        tag: str = "validation",
        num_logs: tp.Optional[int] = None,
        empty_cache_after_generation: bool = False
) -> ReferenceCompletionEvaluator:

    return ReferenceCompletionEvaluator(
        arbiter=arbiter,
        every_n_steps=every_n_steps,
        dataset=dataset,
        dataloader_config=dataloader_config,
        tag=tag,
        num_logs=num_logs,
        empty_cache_after_generation=empty_cache_after_generation,
    )
