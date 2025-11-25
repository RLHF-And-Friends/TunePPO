from functools import partial
import os
import typing as tp

from abc import ABC, abstractmethod
from omegaconf import DictConfig
from typing import Iterator, Tuple

from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import ast
import re
from pathlib import Path, PurePath

from torchtune.modules.peft import disable_adapter
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.training import get_unmasked_sequence_lengths
from torchtune.rlhf import get_reward_penalty_mask, get_rewards_ppo

from ppotune.log import WandbLogger
from ppotune.model import LoRAModel
from ppotune.utils import append_mask
from ppotune.volatile import VolatileFloat

from smart_thinking_llm.tools.graph_creator.base import GraphCreatorBase
from smart_thinking_llm.tools.graph_creator.graph_creator_with_llm_selector import (
    GraphCreatorWithLLMSelector,
)

import torch
from torch.nn import Parameter

from xml.etree import ElementTree


logger = WandbLogger()


class IRewardModel(ABC):
    """
    Abstract Reward Model Interface
    """

    @abstractmethod
    def __call__(
        self,
        tokens: torch.Tensor,  # B x (Q + R)
        responses_pad_mask: torch.Tensor,  # B x R
        **kwargs,
    ) -> torch.Tensor:  # B or B x R
        ...

    @abstractmethod
    def setup(self, cfg: DictConfig, **kwargs) -> None: ...

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]: ...


class LLMRewardModel(IRewardModel):
    """
    LLM-based reward model
    """

    def __init__(
        self,
        scorer: LoRAModel,
        penalise_no_eos: bool,
        reward_penalty: int,
        min_response_len: int,
    ) -> None:
        self.scorer = scorer
        self.penalise_no_eos = penalise_no_eos
        self.reward_penalty = reward_penalty
        self.min_response_len = min_response_len

    def setup(self, cfg: DictConfig, **kwargs) -> None:
        self.scorer.setup(cfg.scorer)

    @torch.no_grad()
    def __call__(
        self,
        tokens: torch.Tensor,  # B x (Q + R)
        causal_mask: torch.Tensor,  # B x (Q + R) x (Q + R)
        position_ids: torch.Tensor,  # B x (Q + R)
        responses_pad_mask: torch.Tensor,  # B x R
        **kwargs,
    ) -> torch.Tensor:  # B
        queries_len = tokens.shape[1] - responses_pad_mask.shape[1]

        with disable_adapter(self.scorer.model):  # in case it is a LoRA scorer
            scores = self.scorer.model(tokens, input_pos=position_ids, mask=causal_mask)

        # the scores from the reward model are the logits for the last non-padding token
        response_last_pos = get_unmasked_sequence_lengths(responses_pad_mask)
        scores = scores.gather(1, (response_last_pos + queries_len)[:, None, None]).squeeze(
            (-1, -2)
        )
        # apply penalties for no EOS or too short responses
        reward_penalty_mask = get_reward_penalty_mask(  # warn: seem to penalize generations with
            responses_pad_mask,  # eos at the very end
            response_last_pos,
            self.penalise_no_eos,
            self.min_response_len,
        )
        scores[reward_penalty_mask] = self.reward_penalty

        logger.collect("scores", scores)
        return scores

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        for name, param in self.scorer.named_parameters(prefix, recurse, remove_duplicate):
            yield name, param


class PerTokenKLPenalizedRewardModel(LLMRewardModel):
    """
    OpenAI-like reward model with injected per token KL-Penalty
    """

    def __init__(
        self,
        scorer: LoRAModel,
        penalise_no_eos: bool,
        reward_penalty: int,
        min_response_len: int,
        kl_coeff: float | VolatileFloat,
    ) -> None:
        super().__init__(
            scorer,
            penalise_no_eos,
            reward_penalty,
            min_response_len,
        )
        self._kl_coeff = kl_coeff

    @torch.no_grad()
    def __call__(
        self,
        tokens: torch.Tensor,  # B x (Q + R)
        causal_mask: torch.Tensor,  # B x (Q + R) x (Q + R)
        position_ids: torch.Tensor,  # B x (Q + R)
        responses_pad_mask: torch.Tensor,  # B x R
        gen_logprobs: torch.Tensor,  # B x R
        ref_logprobs: torch.Tensor,  # B x R
        **kwargs,
    ) -> torch.Tensor:  # B x R
        scores = super().__call__(tokens, causal_mask, position_ids, responses_pad_mask)
        mask_after_eos = append_mask(responses_pad_mask)
        pos_after_eos = get_unmasked_sequence_lengths(mask_after_eos)

        kl_coeff = float(self._kl_coeff)
        rewards, _, kl_rewards = get_rewards_ppo(
            scores, gen_logprobs, ref_logprobs, kl_coeff, pos_after_eos
        )
        logger.collect_dict(
            {
                "reward.kl_coeff": torch.tensor(kl_coeff),
                "reward.total": scores + kl_rewards.sum(1),
                "reward.kl_penalty": kl_rewards.sum(1),
            }
        )
        return rewards


class DeepSeekMathRewardModel(IRewardModel):
    """
    Rule-Based Reward Model as in DeepSeekMath.
    """

    def __init__(self) -> None:
        return

    def setup(self, cfg: DictConfig, tokenizer: ModelTokenizer, **kwargs) -> None:
        self.tokenizer = tokenizer

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        return iter([])

    @torch.no_grad()
    def __call__(
        self,
        tokens: torch.Tensor,  # B x (Q + R)
        causal_mask: torch.Tensor,  # B x (Q + R) x (Q + R)
        position_ids: torch.Tensor,  # B x (Q + R)
        responses_pad_mask: torch.Tensor,  # B x R
        batch: dict,
        **kwargs,
    ) -> torch.Tensor:  # B
        batch_size = tokens.shape[0]
        queries_len = tokens.shape[1] - responses_pad_mask.shape[1]
        response_tokens = tokens[:, queries_len:].clone()
        response_tokens[responses_pad_mask] = self.tokenizer.pad_id

        scores = torch.zeros_like(tokens[:, 0], dtype=torch.float32)
        successes = torch.zeros_like(tokens[:, 0], dtype=torch.float32)

        for i in range(batch_size):
            response = self.tokenizer.decode(response_tokens[i].tolist())
            answer = batch["answers"][i]
            scores[i], successes[i] = self.shaped_correctness_reward(
                answer=answer, completion=response
            )

        logger.collect_dict({"success_rate": successes, "scores": scores})
        return scores

    @staticmethod
    def shaped_correctness_reward(answer: str, completion: str) -> tuple[float, float]:
        """
        Reward function for verifiable rewards with some mild shaping.

        Args:
            answer (str): ground-truth answer to the current problem
            completion (str): model's completion, starting immediately after "Assistant: <think>"
        Returns:
            reward: (float) a shaped reward indicating the correct answer and the correct format
            success: (float) a binary measure of success (1 if the answer is correct and correctly
                formatted, 0 otherwise)
        """
        reward = 0.0
        success = 0.0

        try:
            tags = DeepSeekMathRewardModel.extract_tags(completion)
        except ElementTree.ParseError:
            tags = {"think": [], "answer": []}

        if len(tags["answer"]) == 1:
            reward += 5.0

        if len(tags["think"]) == 1:
            reward += 5.0

        if any(attempt == answer for attempt in tags["answer"]):
            # One of the answer tags has the right answer
            reward += 20.0

        if any((answer in attempt) for attempt in tags["answer"]):
            # One of the answer tags contains the right answer (might be e.g. $20 instead of 20)
            reward += 10.0

        if len(tags["answer"]) > 0 and tags["answer"][-1] == answer:
            reward = 100.0
            success = 1

        return reward, success

    @staticmethod
    def extract_tags(text: str) -> dict[str, list[str]]:
        """
        Parse XML-like tags from text. Returns a dictionary with keys 'think' and 'answer'.
        The values are lists of strings, with each string being the content of a tag.
        """
        xml_string = f"<root>{text}</root>"
        root = ElementTree.fromstring(xml_string)
        return {
            "think": [
                elem.text if elem.text is not None else "" for elem in root.findall("think")
            ],
            "answer": [
                elem.text if elem.text is not None else "" for elem in root.findall("answer")
            ],
        }


class MultiHopQAShapedReward(IRewardModel):
    """
    Our Rule-Based Reward Model for QA-Reasoning Format.
    """

    def __init__(self) -> None:
        return

    def setup(self, cfg: DictConfig, tokenizer: ModelTokenizer, **kwargs) -> None:
        self.tokenizer = tokenizer

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        return iter([])

    def __call__(
        self,
        tokens: torch.Tensor,  # B x (Q + R)
        causal_mask: torch.Tensor,  # B x (Q + R) x (Q + R)
        position_ids: torch.Tensor,  # B x (Q + R)
        responses_pad_mask: torch.Tensor,  # B x R
        batch: dict[str, torch.Tensor | str],
        **kwargs,
    ) -> torch.Tensor:  # B
        batch_size = tokens.shape[0]
        queries_len = tokens.shape[1] - responses_pad_mask.shape[1]
        response_tokens = tokens[:, queries_len:].clone()
        response_tokens[responses_pad_mask] = self.tokenizer.pad_id

        scores = torch.zeros_like(tokens[:, 0], dtype=torch.float32)
        successes = torch.zeros_like(tokens[:, 0], dtype=torch.float32)

        for i in range(batch_size):
            response = self.tokenizer.decode(response_tokens[i].tolist(), skip_special_tokens=True)
            answers = batch["answers"][i]
            final_answer = batch["final_answer"][i]
            scores[i], successes[i] = self.shaped_correctness_reward(
                answers=answers, final_answer=final_answer, completion=response
            )

        logger.collect_dict(
            {
                "success_rate": successes,
                "scores": scores,
            }
        )
        return scores

    @staticmethod
    def shaped_correctness_reward(
        answers: list[str], final_answer: str, completion: str
    ) -> tuple[float, float]:
        """
        Computes a shaped reward based on intermediate reasoning and final answer.

        Args:
            answers (List[str]): Expected intermediate answers (in order).
            final_answer (str): Expected final answer.
            completion (str): Model's output in structured format.

        Returns:
            Tuple[float, float]: (shaped reward, binary success)
        """
        reward = 0.0
        success = 0.0

        try:
            tags = MultiHopQAShapedReward.extract_tags(completion)
        except ElementTree.ParseError:
            return 0.0, 0.0

        if len(tags["answer"]) == 1:
            reward += 5.0

        if len(tags["think"]) == 1:
            reward += 5.0

        intermediates = (
            [step.get("answer", "").strip() for step in tags["think"][0]] if tags["think"] else []
        )

        if len(intermediates) == len(answers):
            reward += 5.0

        for i, expected in enumerate(answers):
            if i >= len(intermediates):
                break
            pred = intermediates[i]
            if pred in expected:
                reward += 10.0

        if any(attempt in final_answer for attempt in tags["answer"]):
            # One of the answer tags has the right answer
            reward += 20.0

        if len(tags["answer"]) > 0 and tags["answer"][-1] in final_answer:
            reward = 100.0
            success = 1

        return reward, success

    @staticmethod
    def extract_tags(text: str) -> dict[str, tp.Any]:
        """
        Expects intermediate <question>/<answer> reasoning format like:

        <think>
        <question>1st question</question>
        <answer>1st answer</answer>
        <question>2nd question</question>
        <answer>2nd answer</answer>
        </think>
        <answer>final answer</answer>

        and parses it into dictionary form:
        {
            "think": List[List[Dict[str, str]]],  # List of <think> blocks, each with q-a steps
            "answer": List[str],                  # All top-level <answer> contents
        }
        """
        result = {
            "think": [],
            "answer": [],
        }

        xml_string = f"<root>{text}</root>"
        root = ElementTree.fromstring(xml_string)

        for think_elem in root.findall("think"):
            steps = []
            children = list(think_elem)
            i = 0
            while i + 1 < len(children):
                if children[i].tag == "question" and children[i + 1].tag == "answer":
                    steps.append(
                        {
                            "question": (children[i].text or "").strip(),
                            "answer": (children[i + 1].text or "").strip(),
                        }
                    )
                    i += 2
                else:
                    i += 1  # Skip malformed or unexpected tags
            result["think"].append(steps)

        for answer_elem in root.findall("answer"):
            result["answer"].append((answer_elem.text or "").strip())

        return result


# -------------------------------------------------------------------------------------------------
# Reward using LLM to reasoning assessment
# -------------------------------------------------------------------------------------------------

TRIPLET_EXTRACTOR_PROMPT = """Analyze the following text step-by-step. For each logical statement in the text, perform the following actions:
1.  Identify the main subject of the statement.
2.  Identify the new piece of information (the answer) that the text provides about the subject.
3.  Formulate a question that links this subject and answer.
4.  Assemble the result into a triplet `(subject, question, answer)`.

After analyzing all statements, present the final result as a list of triplets. You need to provide your answer in the format of a list of triplets. Do not include any other text in your answer.

### Example for Analysis

**Source text:**
Donatus Djagom was a Roman Catholic bishop, and the headquarters of the Roman Catholic Church (the Holy See) is in Vatican City, an independent city-state enclaved within Rome, Italy.

**Reasoning:**
1.  First statement: "Donatus Djagom was a Roman Catholic bishop".
    *   Subject: "Donatus Djagom"
    *   Answer: "Catholicism"
    *   Question: "What is the religious affiliation of Donatus Djagom?"
    *   Triplet: ("Donatus Djagom", "What is the religious affiliation of Donatus Djagom?", "Catholicism")
2.  Second statement: "the headquarters of the Roman Catholic Church (the Holy See) is in Vatican City".
    *   Subject: "Catholicism"
    *   Answer: "Vatican City"
    *   Question: "Where is the headquarters of the Catholic Church located?"
    *   Triplet: ("Catholicism", "Where is the headquarters of the Catholic Church located?", "Vatican City")

**Final result as a list:**
[("Donatus Djagom", "What is the religious affiliation of Donatus Djagom?", "Catholicism"), ("Catholicism", "Where is the headquarters of the Catholic Church located?", "Vatican City")]

### Your Task

**Source text:**
{text}

**Final result as a list:**

You need to provide your answer in the format of a list of triplets. Do not include any other text in your answer.
"""


class LLMBasedMultiHopQAShapedReward(IRewardModel):
    def __init__(self, base_url: str, model: str, **api_request_kwargs) -> None:
        self._llm_api = OpenAI(base_url=base_url)
        self._model = model
        self._api_request_kwargs = api_request_kwargs

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        return iter([])

    def setup(self, cfg: DictConfig, tokenizer: ModelTokenizer, **kwargs) -> None:
        self._tokenizer = tokenizer

    def __call__(
        self,
        tokens: torch.Tensor,  # B x (Q + R)
        causal_mask: torch.Tensor,  # B x (Q + R) x (Q + R)
        position_ids: torch.Tensor,  # B x (Q + R)
        responses_pad_mask: torch.Tensor,  # B x R
        batch: dict[str, torch.Tensor | str],
        **kwargs,
    ) -> torch.Tensor:  # B
        queries_len = tokens.shape[1] - responses_pad_mask.shape[1]
        response_tokens = tokens[:, queries_len:].clone()
        response_tokens[responses_pad_mask] = self._tokenizer.pad_id

        responses = [
            self._tokenizer.decode(single_response_tokens.tolist(), skip_special_tokens=True)
            for single_response_tokens in response_tokens
        ]
        answers = batch["answers"]
        final_answers = batch["final_answer"]

        with ThreadPoolExecutor() as executor:
            scores, successes, reasonings, extractor_responses, interm_answers = zip(
                *executor.map(self.shaped_correctness_reward, answers, final_answers, responses)
            )

        logger.collect_table(
            name="LLM extractor",
            columns={
                "reasoning": reasonings,
                "LLM response": extractor_responses,
                "intermediates": [", ".join(answers) for answers in interm_answers],
            },
        )

        successes = torch.tensor(
            successes,
            dtype=torch.float32,
            device=tokens.device,
        ).unsqueeze(1)
        scores = torch.tensor(
            scores,
            dtype=torch.float32,
            device=tokens.device,
        ).unsqueeze(1)

        logger.collect_dict(
            {
                "success_rate": successes,
                "scores": scores,
            }
        )

        return scores

    def shaped_correctness_reward(
        self, answers: list[str], final_answer: str, completion: str
    ) -> tuple[float, float, str, str, list]:
        """
        Computes a shaped reward based on intermediate reasoning and final answer.

        Args:
            answers (List[str]): Expected intermediate answers (in order).
            final_answer (str): Expected final answer.
            completion (str): Model's output in structured format.

        Returns:
            Tuple[float, float, str, str, list]: (
                shaped reward,
                binary success,
                reasoning text,
                llm_extractor_response,
                list of intermediate answers
            )
        """
        reward = 0.0
        success = 0.0

        try:
            tags = self.extract_tags(completion)
        except ElementTree.ParseError:
            return 0.0, 0.0, "", "", []

        if len(tags["answer"]) == 1:
            reward += 5.0

        if len(tags["think"]) == 1:
            reward += 5.0

        if len(tags["think"]) > 0:
            reasoning = tags["think"][0]

            # print(f"Reasoning: {reasoning}")

            llm_response, intermediates = self.extract_answers_from_reasoning(reasoning)
        else:
            reasoning, llm_response, intermediates = "", "", []

        if len(intermediates) == len(answers):
            reward += 5.0

        for i, expected in enumerate(answers):
            if i >= len(intermediates):
                break
            pred = intermediates[i]
            if pred in expected:
                reward += 10.0

        if any(attempt in final_answer for attempt in tags["answer"]):
            # One of the answer tags has the right answer
            reward += 20.0

        if len(tags["answer"]) > 0 and tags["answer"][-1] in final_answer:
            reward = 100.0
            success = 1

        return reward, success, reasoning, llm_response, intermediates

    @staticmethod
    def extract_tags(text: str) -> dict[str, list[str]]:
        """
        Parse XML-like tags from text. Returns a dictionary with keys 'think' and 'answer'.
        The values are lists of strings, with each string being the content of a tag.
        """
        xml_string = f"<root>{text}</root>"
        root = ElementTree.fromstring(xml_string)
        return {
            "think": [
                elem.text if elem.text is not None else "" for elem in root.findall("think")
            ],
            "answer": [
                elem.text if elem.text is not None else "" for elem in root.findall("answer")
            ],
        }

    def extract_answers_from_reasoning(self, reasoning: str) -> tuple[str, list[str]]:
        """
        Ask LLM to extract intermediate answers from model thinking.
        """
        prompt = TRIPLET_EXTRACTOR_PROMPT.format(text=reasoning)
        messages = [{"role": "user", "content": prompt}]
        completion = self._llm_api.chat.completions.create(
            model=self._model,
            messages=messages,
            **self._api_request_kwargs,
        )
        completion_text = completion.choices[0].message.content

        completion_without_reasoning = self.remove_reasoning(completion_text)
        # print(f"Completion without reasoning: {completion_without_reasoning}")

        try:
            triplets = ast.literal_eval(completion_without_reasoning)
        except (SyntaxError, ValueError):
            return completion_text, []

        answers = [answer for _, _, answer in triplets]
        if not isinstance(answers, list) or not all(isinstance(answer, str) for answer in answers):
            answers = []
        # print(f"Answers: {answers}")

        return completion_text, answers

    @staticmethod
    def remove_reasoning(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class GraphMultihopQAReward(IRewardModel):
    def __init__(
        self,
        entity_aliases_filepath: str,
        relation_aliases_filepath: str,
        dataset_filepath: str,
        triplets_prompt_filepath: str,
        triplets_model: str,
        entity_description_filepath: str,
        llm_selector_prompt_filepath: str,
        base_url: str | None = None,
        norm_lev_threshold: float = 0.8,
        answer_tag_reward: float = 5.0,
        think_tag_reward: float = 5.0,
        correct_answer_reward: float = 100.0,
        similarity_reward: float = 20.0,
        format_penalty_reward: float = 3.0,
        reasoning_length_penalty_reward: float = 0.0,
        min_reasoning_length: int | None = 0,
        answer_length_penalty_reward: float = 0.0,
        min_answer_length: int | None = 0,
        embeddings_model: str = "Qwen/Qwen3-Embedding-4B-batch",
        llm_selector_model: str = "Qwen/Qwen2.5-72B-Instruct",
        # все в словах
        max_context_len: int = 100,
        max_reasoning_length: int = 100,
        max_answer_length: int = 5,
        max_similarity: float = 30.0,
        ban_penalty: float = 1000.0,  # если ответ модели выходит за рамки, описанные выше
        **triplets_generation_params,
    ) -> None:
        self.entity_aliases_filepath: PurePath = Path(entity_aliases_filepath)
        self.relation_aliases_filepath: PurePath = Path(relation_aliases_filepath)
        self.dataset_filepath: PurePath = Path(dataset_filepath)
        self.triplets_prompt_filepath: PurePath = Path(triplets_prompt_filepath)
        self.entity_description_filepath: PurePath = Path(entity_description_filepath)
        self.llm_selector_prompt_filepath: PurePath = Path(llm_selector_prompt_filepath)
        self.embeddings_model: str = embeddings_model
        self.llm_selector_model: str = llm_selector_model
        self.triplets_model: str = triplets_model
        self.base_url: str = base_url
        self.norm_lev_threshold: float = norm_lev_threshold
        self.triplets_generation_params = triplets_generation_params

        self.answer_tag_reward: float = answer_tag_reward
        self.think_tag_reward: float = think_tag_reward
        self.correct_answer_reward: float = correct_answer_reward
        self.similarity_reward: float = similarity_reward
        self.format_penalty_reward: float = format_penalty_reward
        self.reasoning_length_penalty_reward: float = reasoning_length_penalty_reward
        self.min_reasoning_length: int | None = min_reasoning_length
        self.answer_length_penalty_reward: float = answer_length_penalty_reward
        self.min_answer_length: int | None = min_answer_length

        self.max_context_len: int = max_context_len
        self.max_reasoning_length: int = max_reasoning_length
        self.max_answer_length: int = max_answer_length
        self.max_similarity: float = max_similarity
        self.ban_penalty: float = ban_penalty

        self.max_reward = self.compute_reward(
            self.correct_answer_reward,
            self.answer_tag_reward,
            self.think_tag_reward,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self.min_reward = self.compute_reward(
            0.0,
            0.0,
            0.0,
            self.ban_penalty,
            self.ban_penalty,
            self.ban_penalty,
            self.ban_penalty,
            self.ban_penalty,
            self.ban_penalty,
        )

        self._graph_creator = GraphCreatorWithLLMSelector(
            entity_aliases_filepath=self.entity_aliases_filepath,
            relation_aliases_filepath=self.relation_aliases_filepath,
            dataset_filepath=self.dataset_filepath,
            triplets_prompt_filepath=self.triplets_prompt_filepath,
            openai_client=OpenAI(base_url=self.base_url),
            triplets_model=self.triplets_model,
            norm_lev_threshold=self.norm_lev_threshold,
            parse_graph_strategy=self.triplets_generation_params["graph_mode"],
            entity_description_filepath=self.entity_description_filepath,
            llm_selector_model=self.llm_selector_model,
            llm_selector_prompt_filepath=self.llm_selector_prompt_filepath,
        )

    def compute_reward(
        self,
        correct_answer_reward: float,
        answer_tag_reward: float,
        think_tag_reward: float,
        similarity_penalty_reward: float,
        reasoning_length_penalty_reward: float,
        answer_length_penalty_reward: float,
        format_penalty_reward: float,
        many_answer_tags_penalty_reward: float,
        many_think_tags_penalty_reward: float,
    ) -> float:
        return (
            correct_answer_reward
            + answer_tag_reward
            + think_tag_reward
            - similarity_penalty_reward
            - reasoning_length_penalty_reward
            - answer_length_penalty_reward
            - format_penalty_reward
            - many_answer_tags_penalty_reward
            - many_think_tags_penalty_reward
        )

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        return iter([])

    def setup(self, cfg: DictConfig, tokenizer: ModelTokenizer, **kwargs) -> None:
        self._tokenizer = tokenizer

    @staticmethod
    def to_tensor(
        x: float | int | list[float | int] | tuple[float | int], device: torch.device
    ) -> torch.Tensor:
        if isinstance(x, list) or isinstance(x, tuple):
            return torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(1)
        elif isinstance(x, float) or isinstance(x, int):
            return torch.tensor([x], dtype=torch.float32, device=device).unsqueeze(1)
        else:
            raise ValueError(f"Unsupported type: {type(x)}")

    def __call__(
        self,
        tokens: torch.Tensor,  # B x (Q + R)
        causal_mask: torch.Tensor,  # B x (Q + R) x (Q + R)
        position_ids: torch.Tensor,  # B x (Q + R)
        responses_pad_mask: torch.Tensor,  # B x R
        batch: dict[str, torch.Tensor | str],
        **kwargs,
    ) -> torch.Tensor:  # B
        queries_len = tokens.shape[1] - responses_pad_mask.shape[1]
        response_tokens = tokens[:, queries_len:].clone()
        response_tokens[responses_pad_mask] = self._tokenizer.pad_id

        responses = [
            self._tokenizer.decode(single_response_tokens.tolist(), skip_special_tokens=True)
            for single_response_tokens in response_tokens
        ]
        final_answers = batch["final_answer"]
        paths = batch["path"]

        with ThreadPoolExecutor() as executor:
            scores, successes = zip(
                *executor.map(
                    partial(self.shaped_correctness_reward, self._graph_creator, tokens.device),
                    final_answers,
                    paths,
                    responses,
                )
            )

        successes = self.to_tensor(successes, device=tokens.device)
        scores = self.to_tensor(scores, device=tokens.device)

        logger.collect_dict(
            {
                "success_rate": successes,
                "scores": scores,
            }
        )

        return scores

    def shaped_correctness_reward(
        self,
        graph_creator: GraphCreatorBase,
        device: torch.device,
        final_answer: str,
        ground_truth_path: str,
        completion: str,
    ) -> tuple[float, float]:
        """
        Computes a shaped reward based on intermediate reasoning and final answer.

        Args:
            answers (List[str]): Expected intermediate answers (in order).
            final_answer (str): Expected final answer.
            ground_truth_path (str): Expected grapth path.
            completion (str): Model's output in structured format.
            device (torch.device): Device to use for calculations.

        Returns:
            Tuple[float, float]: (
                shaped reward,
                binary success,
            )
        """
        reward = 0.0
        success = 0.0

        try:
            tags, content_len, question = self.extract_tags_and_content_length(completion)
        except ElementTree.ParseError:
            return reward, success

        if len(tags["think"]) > 0:
            reasoning = tags["think"][0]
        else:
            reasoning = ""

        reasoning_length_penalty_reward = 0.0
        if self.min_reasoning_length is not None:
            reasoning_length = len(reasoning.split())
            # длина ризонинга не должна превышать некоторую границу
            reasoning_length_penalty_reward = (
                max(0, reasoning_length - self.min_reasoning_length)
                * self.reasoning_length_penalty_reward
            )
            reasoning_length_penalty_reward = min(
                reasoning_length_penalty_reward, self.ban_penalty
            )

        if len(tags["answer"]) > 0:
            answer = tags["answer"][0]
        else:
            answer = ""

        answer_length_penalty_reward = 0.0
        if self.min_answer_length is not None:
            answer_length = len(answer.split())
            # длина ответа должна быть небольшой
            answer_length_penalty_reward = (
                max(0, answer_length - self.min_answer_length) * self.answer_length_penalty_reward
            )
            answer_length_penalty_reward = min(answer_length_penalty_reward, self.ban_penalty)

        answer_tag_reward = 0.0
        if len(tags["answer"]) == 1:
            answer_tag_reward = self.answer_tag_reward

        many_answer_tags_penalty_reward = 0.0
        if len(tags["answer"]) > 1:
            many_answer_tags_penalty_reward = self.answer_tag_reward * (len(tags["answer"]) - 1)
            many_answer_tags_penalty_reward = min(
                many_answer_tags_penalty_reward, self.ban_penalty
            )

        think_tag_reward = 0.0
        if len(tags["think"]) == 1:
            think_tag_reward = self.think_tag_reward

        many_think_tags_penalty_reward = 0.0
        if len(tags["think"]) > 1:
            many_think_tags_penalty_reward = self.think_tag_reward * (len(tags["think"]) - 1)
            many_think_tags_penalty_reward = min(many_think_tags_penalty_reward, self.ban_penalty)

        correct_answer_reward = 0.0
        success = 0
        if len(tags["answer"]) == 1 and tags["answer"][0] in final_answer:
            correct_answer_reward = self.correct_answer_reward
            success = 1

        ground_truth_graph = graph_creator.get_graph_from_path(ground_truth_path)
        completion_graph = graph_creator(reasoning)
        similarity = completion_graph.compare_to(ground_truth_graph)
        ground_truth_graph_path = str(ground_truth_graph)
        completion_graph_path = str(completion_graph)

        similarity_penalty_reward = (
            similarity * self.similarity_reward
        )  # это graph_edit_distance - сколько действий произвести, чтобы графы сошлись
        similarity_penalty_reward = min(similarity_penalty_reward, self.ban_penalty)
        format_penalty_reward = content_len * self.format_penalty_reward  # вне think/answer тегов
        format_penalty_reward = min(format_penalty_reward, self.ban_penalty)

        reward = self.compute_reward(
            correct_answer_reward,
            answer_tag_reward,
            think_tag_reward,
            similarity_penalty_reward,
            reasoning_length_penalty_reward,
            answer_length_penalty_reward,
            format_penalty_reward,
            many_answer_tags_penalty_reward,
            many_think_tags_penalty_reward,
        )

        reward_range = self.max_reward - self.min_reward
        if reward_range > 0:
            scaled_reward = (reward - self.min_reward) / reward_range
        else:
            scaled_reward = 0.0

        logger.collect_completion_with_graph(
            question=question,
            completion=completion,
            reasoning=reasoning,
            answer=answer,
            ground_truth_graph_path=ground_truth_graph_path,
            completion_graph_path=completion_graph_path,
            answer_tag_reward=answer_tag_reward,
            think_tag_reward=think_tag_reward,
            correct_answer_reward=correct_answer_reward,
            similarity_penalty_reward=similarity_penalty_reward,
            reasoning_length_penalty_reward=reasoning_length_penalty_reward,
            answer_length_penalty_reward=answer_length_penalty_reward,
            format_penalty_reward=format_penalty_reward,
            score=reward,
        )

        logger.collect_dict(
            {
                "answer_length": self.to_tensor(len(answer.split()), device),
                "reasoning_length": self.to_tensor(len(reasoning.split()), device),
                "content_length": self.to_tensor(content_len, device),
                "similarity_penalty": self.to_tensor(similarity_penalty_reward, device),
                "scaled_scores": self.to_tensor(scaled_reward, device),
            }
        )

        return reward, success

    @staticmethod
    def extract_tags_and_content_length(text: str) -> tuple[str, int, str]:
        """
        Parse XML-like tags from text using regex. Returns a dictionary with keys 'think' and 'answer'.
        The values are lists of strings, with each string being the content of a tag.
        Also returns the length of content outside of tags (in words).
        Removes <system_prompt>...</system_prompt> tags and text before "Assistant:".
        """
        # Удаляем <system_prompt>...</system_prompt> сразу
        text = re.sub(r"<system_prompt>.*?</system_prompt>", "", text, flags=re.DOTALL)

        # Находим первое вхождение "User:" и берем вопрос после него
        user_pos = text.find("User:")
        assistant_pos = text.find("Assistant:")
        question = ""
        if user_pos != -1 and assistant_pos != -1:
            question = text[user_pos + len("User:") : assistant_pos].strip()

        # Находим первое вхождение "Assistant:" и берем текст после него
        if assistant_pos != -1:
            text = text[assistant_pos + len("Assistant:") :].strip()

        # Извлекаем все <think>...</think> теги
        think_pattern = r"<think>(.*?)</think>"
        think_matches = re.findall(think_pattern, text, re.DOTALL)

        # Извлекаем все <answer>...</answer> теги
        answer_pattern = r"<answer>(.*?)</answer>"
        answer_matches = re.findall(answer_pattern, text, re.DOTALL)

        tags = {
            "think": think_matches,
            "answer": answer_matches,
        }

        # Удаляем все теги из текста для подсчета контента вне тегов
        text_without_tags = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text_without_tags = re.sub(r"<answer>.*?</answer>", "", text_without_tags, flags=re.DOTALL)

        # Подсчитываем слова в оставшемся тексте
        extra_reasoning_len = sum(
            [len(reasoning) + len("<think></think>") for reasoning in think_matches[1:]]
        )
        extra_answer_len = sum(
            [len(answer) + len("<answer></answer>") for answer in answer_matches[1:]]
        )
        content_length = len(text_without_tags.split()) + extra_reasoning_len + extra_answer_len

        return tags, content_length, question
