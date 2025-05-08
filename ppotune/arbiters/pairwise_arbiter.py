import typing as tp

import logging

from concurrent.futures import ThreadPoolExecutor

from abc import ABC, abstractmethod

import numpy as np

from openai import OpenAI

from ppotune.log import WandbLogger

logger = WandbLogger()


class PairwiseArbiter(ABC):
    """
    Interface for all pairwise arbiters.
    """
    @abstractmethod
    def judge(
        self,
        prompts: tp.List[str],
        completions: tp.List[tp.Tuple[str, str]],
        shuffle_order: bool = True
    ) -> tp.List:
        """
        Args:
            prompts:
                list of prompts.
            completions:
                list of tuple pairs with completions for given prompts.
            shuffle_order:
                whether to shuffle completions before passing them to the
                underlying model.
        Returns:
            list of [-1/0/1] where -1 denotes invalid result, 0 denotes win of
            0th completion and 1 denotes win of 1st completion in a tuple for
            each tuple in completions list.
        """
        raise NotImplementedError("Arbiters must override .judge(...) method.")


class RemotePairwiseArbiter(PairwiseArbiter):
    """
    Pairwise arbiter class for remote inference services like OpneAI or
    deepinfra.

    Args:
        system prompt:
            the model will be prompted to judge completions using this prompt.
        base_url:
            url for the inference service. If you do not want to use
            OpenAI set this to something else (or OpenAI will be used by
            default).
        model:
            which model to use. Defaults to "gpt-4o-mini".
    """
    def __init__(
        self,
        system_prompt: str,
        base_url: tp.Optional[str] = None,
        model: str = "gpt-4o-mini",
        concurrent: bool = True,
    ) -> None:
        self._client = OpenAI(base_url=base_url)
        self._model = model
        self._system_prompt = system_prompt
        self._concurrent = concurrent

    def judge(
        self,
        prompts: tp.List[str],
        completions: tp.List[tp.Tuple[str, str]],
        shuffle_order: bool = True
    ) -> list[int]:

        # Shuffle the order of the completions to avoid positional bias
        # -----------------------------------------------------------------------------------------
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [
                (pair[1], pair[0]) if flip else pair
                for flip, pair in zip(flip_mask, completions)
            ]

        # Call the completions concurrently or sequentially
        # -----------------------------------------------------------------------------------------

        if self._concurrent:
            with ThreadPoolExecutor() as executor:
                ranks = list(executor.map(self._get_rank, prompts, completions))
        else:
            ranks = [
                self._get_rank(
                    prompt,
                    completions_pair
                ) for prompt, completions_pair in zip(prompts, completions)
            ]

        # Flip back the ranks to the original order if needed
        # -----------------------------------------------------------------------------------------
        if shuffle_order:
            ranks = [ranks[i] if not flip else 1 - ranks[i] for i, flip in enumerate(flip_mask)]

        return ranks

    def _get_rank(self, prompt: str, candidates: tp.Tuple[str, str]) -> int:
        """
        Get the rank for a single prompt.
        """
        content = self._system_prompt.format(
            prompt=prompt,
            response0=candidates[0],
            response1=candidates[1]
        )
        messages = [{"role": "user", "content": content}]
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        response = completion.choices[0].message.content[-1]
        if response in ["0", "1"]:
            response = int(response)
        else:
            logging.warning(
                f"Invalid response from the judge model: '{response}'. Returning -1."
            )
            return -1

        return response
