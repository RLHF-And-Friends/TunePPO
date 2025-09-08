import typing as tp

from abc import ABC, abstractmethod
from omegaconf import DictConfig
from typing import Iterator, Tuple

from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import ast
import re
from pathlib import Path, PurePath
from functools import partial

from torchtune.modules.peft import disable_adapter
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.training import get_unmasked_sequence_lengths
from torchtune.rlhf import get_reward_penalty_mask, get_rewards_ppo

from ppotune.log import WandbLogger
from ppotune.model import LoRAModel
from ppotune.utils import append_mask
from ppotune.volatile import VolatileFloat

from smart_thinking_llm.tools.graph_creation import GraphCreator

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
        tokens:             torch.Tensor, # B x (Q + R)
        responses_pad_mask: torch.Tensor, # B x R
        **kwargs
    ) -> torch.Tensor: # B or B x R
        ...

    @abstractmethod
    def setup(self, cfg: DictConfig, **kwargs) -> None:
        ...

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        ...


class LLMRewardModel(IRewardModel):
    """
    LLM-based reward model
    """
    def __init__(
        self,
        scorer: LoRAModel,
        penalise_no_eos:    bool,
        reward_penalty:     int,
        min_response_len:   int,
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
        tokens:             torch.Tensor, # B x (Q + R)
        causal_mask:        torch.Tensor, # B x (Q + R) x (Q + R)
        position_ids:       torch.Tensor, # B x (Q + R)
        responses_pad_mask: torch.Tensor, # B x R
        **kwargs,
    ) -> torch.Tensor: # B

        queries_len = tokens.shape[1] - responses_pad_mask.shape[1]

        with disable_adapter(self.scorer.model): # in case it is a LoRA scorer
            scores = self.scorer.model(
                tokens,
                input_pos=position_ids,
                mask=causal_mask
            )

        # the scores from the reward model are the logits for the last non-padding token
        response_last_pos = get_unmasked_sequence_lengths(responses_pad_mask)
        scores = scores.gather(1, (response_last_pos + queries_len)[:, None, None]).squeeze(
            (-1, -2)
        )
        # apply penalties for no EOS or too short responses
        reward_penalty_mask = get_reward_penalty_mask(  # warn: seem to penalize generations with
            responses_pad_mask,                         # eos at the very end
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
        penalise_no_eos:    bool,
        reward_penalty:     int,
        min_response_len:   int,
        kl_coeff:           float | VolatileFloat,
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
        tokens:             torch.Tensor, # B x (Q + R)
        causal_mask:        torch.Tensor, # B x (Q + R) x (Q + R)
        position_ids:       torch.Tensor, # B x (Q + R)
        responses_pad_mask: torch.Tensor, # B x R
        gen_logprobs:       torch.Tensor, # B x R
        ref_logprobs:       torch.Tensor, # B x R
        **kwargs
    ) -> torch.Tensor: # B x R

        scores = super().__call__(
            tokens,
            causal_mask,
            position_ids,
            responses_pad_mask
        )
        mask_after_eos = append_mask(responses_pad_mask)
        pos_after_eos = get_unmasked_sequence_lengths(mask_after_eos)

        kl_coeff = float(self._kl_coeff)
        rewards, _, kl_rewards = get_rewards_ppo(
            scores,
            gen_logprobs,
            ref_logprobs,
            kl_coeff,
            pos_after_eos
        )
        logger.collect_dict({
            "reward.kl_coeff": torch.tensor(kl_coeff),
            "reward.total": scores + kl_rewards.sum(1),
            "reward.kl_penalty": kl_rewards.sum(1),
        })
        return rewards


class DeepSeekMathRewardModel(IRewardModel):
    """
    Rule-Based Reward Model as in DeepSeekMath.
    """
    def __init__(self) -> None:
        return

    def setup(
        self,
        cfg: DictConfig,
        tokenizer: ModelTokenizer,
        **kwargs
    ) -> None:
        self.tokenizer = tokenizer

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        return iter([])

    @torch.no_grad()
    def __call__(
        self,
        tokens:             torch.Tensor, # B x (Q + R)
        causal_mask:        torch.Tensor, # B x (Q + R) x (Q + R)
        position_ids:       torch.Tensor, # B x (Q + R)
        responses_pad_mask: torch.Tensor, # B x R
        batch:              dict,
        **kwargs,
    ) -> torch.Tensor: # B

        batch_size = tokens.shape[0]
        queries_len = tokens.shape[1] - responses_pad_mask.shape[1]
        response_tokens = tokens[:, queries_len:].clone()
        response_tokens[responses_pad_mask] = self.tokenizer.pad_id

        scores = torch.zeros_like(tokens[:,0], dtype=torch.float32)
        successes = torch.zeros_like(tokens[:,0], dtype=torch.float32)

        for i in range(batch_size):
            response = self.tokenizer.decode(response_tokens[i].tolist())
            answer = batch["answers"][i]
            scores[i], successes[i] = self.shaped_correctness_reward(
                answer=answer, completion=response
            )

        logger.collect_dict({
            "success_rate": successes,
            "scores": scores
        })
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

    def setup(
        self,
        cfg: DictConfig,
        tokenizer: ModelTokenizer,
        **kwargs
    ) -> None:
        self.tokenizer = tokenizer

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        return iter([])


    def __call__(
        self,
        tokens:             torch.Tensor, # B x (Q + R)
        causal_mask:        torch.Tensor, # B x (Q + R) x (Q + R)
        position_ids:       torch.Tensor, # B x (Q + R)
        responses_pad_mask: torch.Tensor, # B x R
        batch:              dict[str, torch.Tensor | str],
        **kwargs
    ) -> torch.Tensor: # B

        batch_size = tokens.shape[0]
        queries_len = tokens.shape[1] - responses_pad_mask.shape[1]
        response_tokens = tokens[:, queries_len:].clone()
        response_tokens[responses_pad_mask] = self.tokenizer.pad_id

        scores = torch.zeros_like(tokens[:,0], dtype=torch.float32)
        successes = torch.zeros_like(tokens[:,0], dtype=torch.float32)

        for i in range(batch_size):
            response = self.tokenizer.decode(
                response_tokens[i].tolist(),
                skip_special_tokens=True
            )
            answers = batch["answers"][i]
            final_answer = batch["final_answer"][i]
            scores[i], successes[i] = self.shaped_correctness_reward(
                answers=answers, final_answer=final_answer, completion=response
            )

        logger.collect_dict({
            "success_rate": successes,
            "scores": scores
        })
        return scores


    @staticmethod
    def shaped_correctness_reward(answers: list[str], final_answer: str, completion: str) -> tuple[float, float]:
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

        intermediates = [
            step.get("answer", "").strip()
            for step in tags["think"][0]
        ] if tags["think"] else []

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
                    steps.append({
                        "question": (children[i].text or "").strip(),
                        "answer": (children[i + 1].text or "").strip(),
                    })
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
        tokens:             torch.Tensor, # B x (Q + R)
        causal_mask:        torch.Tensor, # B x (Q + R) x (Q + R)
        position_ids:       torch.Tensor, # B x (Q + R)
        responses_pad_mask: torch.Tensor, # B x R
        batch:              dict[str, torch.Tensor | str],
        **kwargs
    ) -> torch.Tensor: # B

        queries_len = tokens.shape[1] - responses_pad_mask.shape[1]
        response_tokens = tokens[:, queries_len:].clone()
        response_tokens[responses_pad_mask] = self._tokenizer.pad_id

        responses = [
            self._tokenizer.decode(single_response_tokens.tolist(), skip_special_tokens=True) for
            single_response_tokens in response_tokens
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
            }
        )

        successes = torch.tensor(
            successes,
            dtype = torch.float32,
            device=tokens.device,
        ).unsqueeze(1)
        scores = torch.tensor(
            scores,
            dtype = torch.float32,
            device=tokens.device,
        ).unsqueeze(1)

        logger.collect_dict({
            "success_rate": successes,
            "scores": scores,
        })

        return scores

    def shaped_correctness_reward(
        self,
        answers: list[str],
        final_answer: str,
        completion: str
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
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


class GraphMultihopQAReward(IRewardModel):
    def __init__(
        self,
        entity_aliases_filepath: str,
        relation_aliases_filepath: str,
        dataset_filepath: str,
        triplets_prompt_filepath: str,
        triplets_model: str,
        base_url: str | None = None,
        norm_lev_threshold: float = 0.8,
        **triplets_generation_params,
    ) -> None:
        self.entity_aliases_filepath: PurePath = Path(entity_aliases_filepath)
        self.relation_aliases_filepath: PurePath = Path(relation_aliases_filepath)
        self.dataset_filepath: PurePath = Path(dataset_filepath)
        self.triplets_prompt_filepath: PurePath = Path(triplets_prompt_filepath)
        self.triplets_model: str = triplets_model
        self.base_url: str = base_url
        self.norm_lev_threshold: float = norm_lev_threshold
        self.triplets_generation_params = triplets_generation_params

        self._graph_creator = GraphCreator(
            entity_aliases_filepath=self.entity_aliases_filepath,
            relation_aliases_filepath=self.relation_aliases_filepath,
            dataset_filepath=self.dataset_filepath,
            triplets_prompt_filepath=self.triplets_prompt_filepath,
            openai_client=OpenAI(base_url=self.base_url),
            triplets_model=self.triplets_model,
            norm_lev_threshold=self.norm_lev_threshold,
            **self.triplets_generation_params,
        )

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        return iter([])

    def setup(self, cfg: DictConfig, tokenizer: ModelTokenizer, **kwargs) -> None:
        self._tokenizer = tokenizer

    def __call__(
        self,
        tokens:             torch.Tensor, # B x (Q + R)
        causal_mask:        torch.Tensor, # B x (Q + R) x (Q + R)
        position_ids:       torch.Tensor, # B x (Q + R)
        responses_pad_mask: torch.Tensor, # B x R
        batch:              dict[str, torch.Tensor | str],
        **kwargs
    ) -> torch.Tensor: # B

        queries_len = tokens.shape[1] - responses_pad_mask.shape[1]
        response_tokens = tokens[:, queries_len:].clone()
        response_tokens[responses_pad_mask] = self._tokenizer.pad_id

        responses = [
            self._tokenizer.decode(single_response_tokens.tolist(), skip_special_tokens=True) for
            single_response_tokens in response_tokens
        ]
        final_answers = batch["final_answer"]
        paths = batch["path"]

        with ThreadPoolExecutor() as executor:
            scores, successes = zip(
                *executor.map(
                    partial(self.shaped_correctness_reward, self._graph_creator),
                    final_answers,
                    paths,
                    responses
                )
            )

        successes = torch.tensor(
            successes,
            dtype = torch.float32,
            device=tokens.device,
        ).unsqueeze(1)
        scores = torch.tensor(
            scores,
            dtype = torch.float32,
            device=tokens.device,
        ).unsqueeze(1)

        logger.collect_dict({
            "success_rate": successes,
            "scores": scores,
        })

        return scores

    def shaped_correctness_reward(
        self,
        graph_creator,
        final_answer: str,
        ground_truth_path: str,
        completion: str
    ) -> tuple[float, float]:
        """
        Computes a shaped reward based on intermediate reasoning and final answer.

        Args:
            answers (List[str]): Expected intermediate answers (in order).
            final_answer (str): Expected final answer.
            ground_truth_path (str): Expected grapth path.
            completion (str): Model's output in structured format.

        Returns:
            Tuple[float, float]: (
                shaped reward,
                binary success,
            )
        """
        reward = 0.0
        success = 0.0

        try:
            tags = self.extract_tags(completion)
        except ElementTree.ParseError:
            return reward, success

        if len(tags["think"]) > 0:
            reasoning = tags["think"][0]
        else:
            reasoning = ""

        if len(tags["answer"]) == 1:
            reward += 5.0

        if len(tags["think"]) == 1:
            reward += 5.0

        ground_truth_graph = graph_creator.get_graph_from_path(ground_truth_path)
        graph = graph_creator(reasoning)

        similarity = graph.compare_to(ground_truth_graph)

        reward += similarity

        if len(tags["answer"]) > 0 and tags["answer"][-1] in final_answer:
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
