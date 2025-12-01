import typing as tp

import re

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.data import Dataset
from ppotune.arbiters.pairwise_arbiter import PairwiseArbiter
from ppotune.data.loaders import DataloaderConfig, build_dataloader
from ppotune.log import WandbLogger
from ppotune.model import GenerativeModel

from ppotune.evaluation.eval import ReferenceCompletionEvaluator

logger = WandbLogger()


class MathEvaluator(ReferenceCompletionEvaluator):
    """
    Evaluator for MATH dataset.
    Uses similar format to GSM8K evaluator but adapted for MATH dataset format.
    """
    def __call__(
        self,
        model: GenerativeModel,
        step: int = 0,
    ) -> None:

        if step % self._every_n_steps != 0:
            return

        prompts:     tp.List[str] = []
        completions: tp.List[tp.Tuple[str, str]] = []
        answers:     tp.List[tp.Tuple[str, str]] = []

        for batch in tqdm(self._dataloader, desc=f"Evaluation ({self._tag})", disable=dist.get_rank() != 0):
            batch["tokens"] = batch["tokens"].to(model._device)
            generated = model.generate(prompt=batch["tokens"])

            if self._empty_cache:
                torch.cuda.empty_cache()

            for tokens, query_mask, response_mask, reference_answer in zip(
                generated.tokens,
                generated.query_mask,
                generated.response_mask,
                batch["answers"]
            ):
                query = self.decode(tokens[query_mask])
                response = self.decode(tokens[response_mask])

                prompt = self._extract_prompt(query)
                if prompt is None:
                    continue

                reasoning_answer = self._extract_reasoning_answer(response)
                if reasoning_answer is None:
                    continue
                reasoning, answer = reasoning_answer

                completion = self._make_completion(reasoning, answer)
                reference_completion = self._make_reference_completion(reference_answer)

                prompts.append(prompt)
                completions.append((reference_completion, completion))
                answers.append((reference_answer, answer))

        if not completions:
            return

        wins = torch.tensor(self._arbiter.judge(prompts, completions))
        valid = wins != -1
        winrate = wins[valid].float().mean() if valid.any() else torch.tensor(0.0)

        # For MATH, use normalized answer comparison
        answer_accuracy = torch.tensor(
            sum([self._normalize_answer(ref) == self._normalize_answer(ans)
                 for ref, ans in answers]) / len(answers) if answers else 0.0
        )

        logger.collect_dict({
            f"{self._tag}_winrate": winrate,
            f"{self._tag}_answer_accuracy": answer_accuracy
        })

        logger.collect_table(f"{self._tag}-reference", {
            "reference":    [c[0] for c in completions[:self._num_logs]],
            "completion":   [c[1] for c in completions[:self._num_logs]],
            "chosen":       wins[0:self._num_logs]
        })

        return prompts, completions

    def _extract_prompt(self, query: str) -> str:
        # Try to extract from User: ... Assistant: format
        match = re.search(r'User:\s*(.+?)\s*Assistant:', query, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: use the whole query if no template found
        return query.strip() if query.strip() else None

    def _extract_reasoning_answer(self, response: str) -> tp.Optional[tp.Tuple[str, str]]:
        # Try <think>...</think> <answer>...</answer> format
        match = re.search(
            r'<think>(.*?)</think>\s*<answer>(.*?)</answer>',
            response,
            re.DOTALL
        )
        if match:
            reasoning = match.group(1).strip()
            answer = match.group(2).strip()
            return reasoning, answer

        # Try to extract boxed answer (common in MATH solutions)
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
        if boxed_match:
            answer = boxed_match.group(1).strip()
            reasoning = response[:boxed_match.start()].strip()
            return reasoning, answer

        # Fallback: use whole response as reasoning, empty answer
        if response.strip():
            return response.strip(), ""

        return None

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if answer is None:
            return ""
        # Remove whitespace and common LaTeX formatting
        normalized = answer.strip()
        normalized = re.sub(r'\\(?:text|mathrm|mathbf)\{([^}]+)\}', r'\1', normalized)
        normalized = re.sub(r'\s+', '', normalized)
        normalized = normalized.lower()
        return normalized

    def _make_completion(self, reasoning: str, answer: str) -> str:
        return f"Reasoning: {reasoning}\n\nAnswer: {answer}"

    def _make_reference_completion(self, reference_answer: str) -> str:
        return f"Reference answer: {reference_answer}"


def math_evaluator(
        arbiter: PairwiseArbiter,
        every_n_steps: int,
        dataset: Dataset,
        dataloader_config: DataloaderConfig,
        tag: str = "validation",
        num_logs: tp.Optional[int] = None,
        empty_cache_after_generation: bool = False
) -> MathEvaluator:
    """
    Builder for MATH dataset evaluator.
    """
    return MathEvaluator(
        arbiter=arbiter,
        every_n_steps=every_n_steps,
        dataset=dataset,
        dataloader_config=dataloader_config,
        tag=tag,
        num_logs=num_logs,
        empty_cache_after_generation=empty_cache_after_generation,
    )
