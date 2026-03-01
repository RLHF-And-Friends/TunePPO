from functools import partial
import json
import logging
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
from smart_thinking_llm.tools.graph import Graph
from smart_thinking_llm.utils import make_openai_request

import networkx as nx
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


RULETAKER_TRIPLET_PROMPT = """Extract all subject-relation-object triplets from the text. Use short entity names (e.g. "cat", "cow"). Return ONLY a Python list of tuples: [("src", "rel", "tgt"), ...].

Text:
{text}

List of triplets:"""


class RuletakerPathGraph:
    """
    Lightweight graph from ruletaker path format: list of {fact: {rel, src, tgt}}.
    Uses string labels for nodes/edges, comparable via graph edit distance.
    """

    def __init__(self, path_data: list[dict[str, tp.Any]]) -> None:
        self.graph = nx.DiGraph()
        for item in path_data:
            fact = item.get("fact") or item
            if not isinstance(fact, dict):
                continue
            rel = fact.get("rel", "")
            src = str(fact.get("src", "")).lower()
            tgt = str(fact.get("tgt", "")).lower()
            if not rel or not src or not tgt:
                continue
            if fact.get("negated"):
                rel = "negated " + rel
            self.graph.add_node(src, label=src)
            self.graph.add_node(tgt, label=tgt)
            self.graph.add_edge(src, tgt, label=rel)

    def compare_to(
        self,
        other: "RuletakerPathGraph",
        node_del_cost: float = 1.0,
        node_ins_cost: float = 1.0,
        edge_del_cost: float = 1.0,
        edge_ins_cost: float = 1.0,
    ) -> float:
        def node_match(n1: dict, n2: dict) -> bool:
            return n1.get("label") == n2.get("label")

        def edge_match(e1: dict, e2: dict) -> bool:
            return e1.get("label") == e2.get("label")

        node_del = lambda _: node_del_cost
        node_ins = lambda _: node_ins_cost
        edge_del = lambda _: edge_del_cost
        edge_ins = lambda _: edge_ins_cost

        return float(
            nx.graph_edit_distance(
                self.graph,
                other.graph,
                node_match=node_match,
                edge_match=edge_match,
                node_del_cost=node_del,
                node_ins_cost=node_ins,
                edge_del_cost=edge_del,
                edge_ins_cost=edge_ins,
            )
            or 0.0
        )

    def __str__(self) -> str:
        if self.graph.number_of_nodes() == 0:
            return "RuletakerPathGraph(empty)"
        lines = []
        for u, v, d in self.graph.edges(data=True):
            lines.append(f"{u} --[{d.get('label', '')}]--> {v}")
        return "\n".join(lines) if lines else "RuletakerPathGraph(empty)"


class ReasoningGraphMatcher:
    """
    Extracts subgraph from model reasoning and compares it with ground truth path.
    Supports: Wikidata "Q1-P1-Q2", B21 "Q1 -> P1 -> Q2", ruletaker [{"fact": {rel, src, tgt}}].
    """

    @staticmethod
    def is_ruletaker_path(path: tp.Any) -> bool:
        if path is None:
            return False
        if isinstance(path, str):
            try:
                path = json.loads(path)
            except json.JSONDecodeError:
                return False
        try:
            if not hasattr(path, "__len__") or len(path) == 0:
                return False
            first = path[0]
            if hasattr(first, "as_py"):
                first = first.as_py()
            if isinstance(first, dict):
                return "fact" in first or "rel" in first
            return False
        except (TypeError, IndexError, KeyError, AttributeError, ValueError):
            return False

    @staticmethod
    def normalize_path(path: str) -> str:
        """Convert B21 ' -> ' format to hyphen format for get_graph_from_path."""
        if not path or not path.strip():
            return path
        return re.sub(r"\s*->\s*", "-", path.strip())

    @staticmethod
    def extract_graph_from_reasoning(
        graph_creator: GraphCreatorBase, reasoning: str
    ) -> Graph | None:
        """Extract graph from reasoning text via LLM triplet extraction."""
        if not reasoning or not reasoning.strip():
            return None
        try:
            return graph_creator(reasoning)
        except Exception:
            return None

    @staticmethod
    def get_graph_from_path(
        graph_creator: GraphCreatorBase, path: str
    ) -> Graph | None:
        """Build graph from path string (supports B21 and hyphen formats)."""
        if not path or not path.strip():
            return None
        normalized = ReasoningGraphMatcher.normalize_path(path)
        try:
            return graph_creator.get_graph_from_path(normalized)
        except (KeyError, ValueError, TypeError):
            return None

    @staticmethod
    def _to_python_obj(obj: tp.Any) -> tp.Any:
        """Convert Arrow types to plain Python (HuggingFace datasets)."""
        if hasattr(obj, "as_py"):
            return ReasoningGraphMatcher._to_python_obj(obj.as_py())
        if isinstance(obj, dict):
            return {k: ReasoningGraphMatcher._to_python_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ReasoningGraphMatcher._to_python_obj(x) for x in obj]
        return obj

    @staticmethod
    def normalize_path_for_ruletaker(path: tp.Any) -> list[tp.Any] | None:
        """Convert path to plain Python list of dicts (handles HuggingFace Arrow)."""
        if path is None:
            return None
        if isinstance(path, str):
            try:
                path = json.loads(path)
            except json.JSONDecodeError:
                return None
        try:
            path = ReasoningGraphMatcher._to_python_obj(path)
            if not isinstance(path, list):
                return None
            return path if path else None
        except (TypeError, ValueError, AttributeError):
            return None

    @staticmethod
    def get_ruletaker_graph_from_path(path: tp.Any) -> RuletakerPathGraph | None:
        """Build RuletakerPathGraph from path list or JSON string."""
        normalized = ReasoningGraphMatcher.normalize_path_for_ruletaker(path)
        if not normalized:
            return None
        try:
            return RuletakerPathGraph(normalized)
        except Exception:
            return None

    @staticmethod
    def extract_ruletaker_graph_from_reasoning(
        reasoning: str,
        openai_client: tp.Any,
        model: str,
    ) -> RuletakerPathGraph | None:
        """Extract triplets from reasoning via LLM, build RuletakerPathGraph."""
        if not reasoning or not reasoning.strip():
            return None
        prompt = RULETAKER_TRIPLET_PROMPT.format(text=reasoning)
        try:
            response = make_openai_request(
                openai_client, model, prompt, logging.getLogger(__name__)
            )
            triplets = ast.literal_eval(response.strip())
            if not isinstance(triplets, list):
                return None
            path_data = []
            for tup in triplets:
                if isinstance(tup, (list, tuple)) and len(tup) >= 3:
                    s = str(tup[0]).lower()
                    r_raw = str(tup[1]).strip().lower()
                    t = str(tup[2]).lower()
                    if r_raw in ("is not", "not") or r_raw.startswith("is not "):
                        rel, negated = "is", True
                    else:
                        rel = r_raw.replace("not ", "").strip() or r_raw
                        negated = False
                    path_data.append({"fact": {"src": s, "rel": rel, "tgt": t, "negated": negated}})
            if not path_data:
                return None
            return RuletakerPathGraph(path_data)
        except (ValueError, SyntaxError, IndexError, TypeError):
            return None

    @staticmethod
    def compare_graphs(
        completion_graph: Graph,
        ground_truth_graph: Graph,
        node_del_cost: float = 0.0,
        edge_del_cost: float = 0.0,
    ) -> float:
        """Return graph edit distance (lower = more similar)."""
        return completion_graph.compare_to(
            ground_truth_graph,
            node_del_cost=node_del_cost,
            edge_del_cost=edge_del_cost,
        )


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
        similarity_penalty: float,
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
            - similarity_penalty
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
        full_tokens = tokens.clone()
        full_tokens[:, queries_len:][responses_pad_mask] = self._tokenizer.pad_id

        responses = [
            self._tokenizer.decode(t.tolist(), skip_special_tokens=True) for t in full_tokens
        ]
        final_answers = batch["final_answer"]
        paths = batch["path"]
        questions = batch.get("question", [None] * len(responses))
        if not isinstance(questions, (list, tuple)):
            questions = [questions] * len(responses)

        with ThreadPoolExecutor() as executor:
            scores, successes = zip(
                *executor.map(
                    partial(self.shaped_correctness_reward, self._graph_creator, tokens.device),
                    final_answers,
                    paths,
                    questions,
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
        final_answer: str | list[str],
        ground_truth_path: str | list[tp.Any],
        question_from_batch: str | None,
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
            tags, content_len, parsed_question = self.extract_tags_and_content_length(
                completion
            )
        except ElementTree.ParseError:
            return reward, success

        question = (
            question_from_batch if question_from_batch is not None else parsed_question
        )

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
        fa_list = (
            list(final_answer)
            if isinstance(final_answer, (list, tuple))
            else [final_answer]
        )
        fa_normalized = [str(x).strip().lower() for x in fa_list]
        if len(tags["answer"]) == 1 and tags["answer"][0].strip().lower() in fa_normalized:
            correct_answer_reward = self.correct_answer_reward
            success = 1

        similarity_penalty = 0.0
        ground_truth_graph_path = ""
        completion_graph_path = ""
        path_valid = (
            ground_truth_path is not None
            and (
                (isinstance(ground_truth_path, str) and ground_truth_path.strip())
                or (isinstance(ground_truth_path, list) and len(ground_truth_path) > 0)
            )
        )
        if path_valid and ReasoningGraphMatcher.is_ruletaker_path(ground_truth_path):
            gt_graph = ReasoningGraphMatcher.get_ruletaker_graph_from_path(
                ground_truth_path
            )
            comp_graph = ReasoningGraphMatcher.extract_ruletaker_graph_from_reasoning(
                reasoning,
                graph_creator.openai_client,
                self.triplets_model,
            )
            if gt_graph is not None:
                ground_truth_graph_path = str(gt_graph)
            if comp_graph is not None:
                completion_graph_path = str(comp_graph)
            if gt_graph is not None and comp_graph is not None:
                similarity = comp_graph.compare_to(
                    gt_graph, node_del_cost=0.0, edge_del_cost=0.0
                )
                similarity_penalty = similarity * self.similarity_reward
                similarity_penalty = min(similarity_penalty, self.ban_penalty)
        elif path_valid and isinstance(ground_truth_path, str):
            ground_truth_graph = ReasoningGraphMatcher.get_graph_from_path(
                graph_creator, ground_truth_path
            )
            completion_graph = ReasoningGraphMatcher.extract_graph_from_reasoning(
                graph_creator, reasoning
            )
            if ground_truth_graph is not None and completion_graph is not None:
                similarity = ReasoningGraphMatcher.compare_graphs(
                    completion_graph,
                    ground_truth_graph,
                    node_del_cost=0.0,
                    edge_del_cost=0.0,
                )
                similarity_penalty = similarity * self.similarity_reward
                similarity_penalty = min(similarity_penalty, self.ban_penalty)
                ground_truth_graph_path = str(ground_truth_graph)
                completion_graph_path = str(completion_graph)

        format_penalty_val = content_len * self.format_penalty_reward
        format_penalty_val = min(format_penalty_val, self.ban_penalty)

        reward = self.compute_reward(
            correct_answer_reward,
            answer_tag_reward,
            think_tag_reward,
            similarity_penalty,
            reasoning_length_penalty_reward,
            answer_length_penalty_reward,
            format_penalty_val,
            many_answer_tags_penalty_reward,
            many_think_tags_penalty_reward,
        )

        reward_range = self.max_reward - self.min_reward
        if reward_range > 0:
            scaled_reward = (reward - self.min_reward) / reward_range
        else:
            scaled_reward = 0.0

        ground_truth_answer = (
            final_answer[0] if isinstance(final_answer, (list, tuple)) else final_answer
        )

        logger.collect_completion_with_graph(
            question=question,
            completion=completion,
            reasoning=reasoning,
            answer=answer,
            ground_truth_answer=ground_truth_answer,
            ground_truth_graph_path=ground_truth_graph_path,
            completion_graph_path=completion_graph_path,
            answer_tag_reward=answer_tag_reward,
            think_tag_reward=think_tag_reward,
            correct_answer_reward=correct_answer_reward,
            similarity_penalty_reward=similarity_penalty,
            reasoning_length_penalty_reward=reasoning_length_penalty_reward,
            answer_length_penalty_reward=answer_length_penalty_reward,
            format_penalty_reward=format_penalty_val,
            score=reward,
        )

        logger.collect_dict(
            {
                "answer_length": self.to_tensor(len(answer.split()), device),
                "reasoning_length": self.to_tensor(len(reasoning.split()), device),
                "content_length": self.to_tensor(content_len, device),
                "similarity_penalty": self.to_tensor(similarity_penalty, device),
                "scaled_scores": self.to_tensor(scaled_reward, device),
            }
        )

        return reward, success

    @staticmethod
    def extract_tags_and_content_length(
        text: str,
    ) -> tuple[dict[str, list[str]], int, str]:
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
