import json
import re
from functools import lru_cache
from typing import Any

from openai import OpenAI


def extract_tags_and_content_length(text: str) -> tuple[dict[str, list[str]], int]:
    """
    Parse XML-like tags from text with regex.
    Returns a dictionary with keys 'think' and 'answer',
    plus the total word count of content outside these tags.
    """
    text = re.sub(r"<system_prompt>.*?</system_prompt>", "", text, flags=re.DOTALL)
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(r"<response>.*?</response>", "", text, flags=re.DOTALL)

    think_inner_patterns = (
        r"<think>(.*?)</think>",
        r"<redacted_thinking>(.*?)</redacted_thinking>",
    )
    think_strip_patterns = (
        r"<think>.*?</think>",
        r"<redacted_thinking>.*?</redacted_thinking>",
    )

    think_blocks: list[str] = []
    for inner_pat in think_inner_patterns:
        think_blocks.extend(re.findall(inner_pat, text, flags=re.DOTALL))

    tags = {
        "think": think_blocks,
        "answer": re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL),
    }

    text_without_tags = text
    for strip_pat in think_strip_patterns:
        text_without_tags = re.sub(strip_pat, "", text_without_tags, flags=re.DOTALL)
    text_without_tags = re.sub(
        r"<answer>.*?</answer>", "", text_without_tags, flags=re.DOTALL
    )
    content_length = len(text_without_tags.split())

    return tags, content_length


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("_", " ")
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_triplet(src: Any, rel: Any, tgt: Any) -> dict[str, str]:
    return {
        "src": str(src).strip(),
        "rel": str(rel).strip(),
        "tgt": str(tgt).strip(),
    }


def format_triplet(triplet: dict[str, str]) -> str:
    return f"{triplet['src']} {triplet['rel']} {triplet['tgt']}"


def normalize_triplet(triplet: dict[str, str]) -> dict[str, str]:
    return {
        "src": normalize_text(triplet["src"]),
        "rel": normalize_text(triplet["rel"]),
        "tgt": normalize_text(triplet["tgt"]),
    }


def deduplicate_triplets(triplets: list[dict[str, str]]) -> list[dict[str, str]]:
    unique_triplets: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for triplet in triplets:
        normalized_triplet = normalize_triplet(triplet)
        key = (
            normalized_triplet["src"],
            normalized_triplet["rel"],
            normalized_triplet["tgt"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique_triplets.append(triplet)

    return unique_triplets


def triplet_from_string(triplet: str) -> dict[str, str] | None:
    parts = str(triplet).strip().split(maxsplit=2)
    if len(parts) != 3:
        return None
    return make_triplet(src=parts[0], rel=parts[1], tgt=parts[2])


def parse_triplet_item(item: Any) -> dict[str, str] | None:
    if isinstance(item, dict):
        src = item.get("src")
        rel = item.get("rel")
        tgt = item.get("tgt")
        if src is None or rel is None or tgt is None:
            return None
        return make_triplet(src=src, rel=rel, tgt=tgt)

    if isinstance(item, (list, tuple)) and len(item) == 3:
        return make_triplet(src=item[0], rel=item[1], tgt=item[2])

    if isinstance(item, str):
        return triplet_from_string(item)

    return None


def get_ground_truth_triplets(
    ground_truth: dict[str, Any],
) -> tuple[list[str], list[dict[str, str]]]:
    structured_triplets: list[dict[str, str]] = []
    raw_structured_triplets = ground_truth.get("proof_triplets_structured", [])
    if isinstance(raw_structured_triplets, list):
        for item in raw_structured_triplets:
            parsed_triplet = parse_triplet_item(item)
            if parsed_triplet is not None:
                structured_triplets.append(parsed_triplet)

    proof_triplets = ground_truth.get("proof_triplets", [])
    text_triplets = [str(item).strip() for item in proof_triplets if str(item).strip()]

    if not structured_triplets and text_triplets:
        for triplet in text_triplets:
            parsed_triplet = triplet_from_string(triplet)
            if parsed_triplet is not None:
                structured_triplets.append(parsed_triplet)

    structured_triplets = deduplicate_triplets(structured_triplets)

    if not text_triplets and structured_triplets:
        text_triplets = [format_triplet(triplet) for triplet in structured_triplets]

    return text_triplets, structured_triplets


def compute_heuristic_reasoning_coverage(
    reasoning: str, proof_triplets: list[str]
) -> tuple[float, dict[str, Any]]:
    if not proof_triplets:
        return 1.0, {
            "coverage_mode": "heuristic_empty_ground_truth",
            "matched_triplets": [],
        }

    normalized_reasoning = normalize_text(reasoning)
    matched_triplets: list[str] = []
    matched = 0

    for triplet in proof_triplets:
        normalized_triplet = normalize_text(triplet)
        if not normalized_triplet:
            matched += 1
            matched_triplets.append(triplet)
            continue

        if normalized_triplet in normalized_reasoning:
            matched += 1
            matched_triplets.append(triplet)

    return matched / len(proof_triplets), {
        "coverage_mode": "heuristic",
        "matched_triplets": matched_triplets,
    }


def get_answer_aliases() -> dict[str, str]:
    return {
        "entailment": "entailment",
        "entailed": "entailment",
        "true": "entailment",
        "yes": "entailment",
        "follows": "entailment",
        "not entailment": "not entailment",
        "not entailed": "not entailment",
        "false": "not entailment",
        "no": "not entailment",
        "contradiction": "not entailment",
        "does not follow": "not entailment",
    }


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def optional_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    return text


@lru_cache(maxsize=8)
def get_openai_client(base_url: str | None, timeout: float) -> OpenAI:
    client_kwargs: dict[str, Any] = {
        "timeout": timeout,
        "max_retries": 0,
    }
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def build_triplet_extractor_prompt(reasoning: str) -> str:
    return f"""
Extract atomic reasoning triplets from the text below.

Return only a JSON array.
Each item must be an object with exactly these string keys:
- "src"
- "rel"
- "tgt"

Rules:
- Copy facts that are explicitly stated in the reasoning text.
- Do not infer missing facts.
- Keep entity names and relations concise.
- If there are no factual triplets, return [].

Reasoning text:
{reasoning}
""".strip()


def extract_json_payload(text: str) -> Any:
    cleaned_text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "")
    cleaned_text = re.sub(r"<think>.*?</think>", "", cleaned_text, flags=re.DOTALL).strip()

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass

    array_match = re.search(r"\[.*\]", cleaned_text, flags=re.DOTALL)
    if array_match is None:
        raise ValueError("LLM extractor response does not contain a JSON array")

    return json.loads(array_match.group(0))


def parse_triplets_from_response(raw_response: str) -> list[dict[str, str]]:
    payload = extract_json_payload(raw_response)
    items = payload.get("triplets") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("LLM extractor JSON payload must be a list of triplets")

    parsed_triplets: list[dict[str, str]] = []
    for item in items:
        parsed_triplet = parse_triplet_item(item)
        if parsed_triplet is None:
            continue
        if not any(parsed_triplet.values()):
            continue
        parsed_triplets.append(parsed_triplet)

    return deduplicate_triplets(parsed_triplets)


def extract_triplets_with_llm(reasoning: str, **kwargs: Any) -> tuple[list[dict[str, str]], dict[str, Any]]:
    extractor_model = optional_text(kwargs.get("llm_graph_extractor_model"))
    extractor_base_url = optional_text(kwargs.get("llm_graph_extractor_base_url"))
    extractor_timeout = float(kwargs.get("llm_graph_extractor_timeout", 30.0))
    extractor_temperature = float(kwargs.get("llm_graph_extractor_temperature", 0.0))
    extractor_max_retries = int(kwargs.get("llm_graph_extractor_max_retries", 2))

    metadata: dict[str, Any] = {
        "extractor_model": extractor_model,
        "extractor_base_url": extractor_base_url,
        "extractor_request_ok": False,
        "extractor_parse_ok": False,
        "extractor_raw_response": "",
        "extractor_error": None,
        "extractor_attempts": 0,
    }

    if extractor_model is None:
        metadata["extractor_error"] = "llm_graph_extractor_model is not configured"
        return [], metadata

    try:
        client = get_openai_client(base_url=extractor_base_url, timeout=extractor_timeout)
    except Exception as exc:
        metadata["extractor_error"] = f"{type(exc).__name__}: {exc}"
        return [], metadata

    for attempt in range(extractor_max_retries + 1):
        metadata["extractor_attempts"] = attempt + 1
        try:
            completion = client.chat.completions.create(
                model=extractor_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You extract factual triplets and return strict JSON only.",
                    },
                    {
                        "role": "user",
                        "content": build_triplet_extractor_prompt(reasoning=reasoning),
                    },
                ],
                temperature=extractor_temperature,
            )
            raw_response = completion.choices[0].message.content or ""
            metadata["extractor_request_ok"] = True
            metadata["extractor_raw_response"] = raw_response
        except Exception as exc:
            metadata["extractor_error"] = f"{type(exc).__name__}: {exc}"
            continue

        try:
            predicted_triplets = parse_triplets_from_response(raw_response)
        except Exception as exc:
            metadata["extractor_error"] = f"{type(exc).__name__}: {exc}"
            return [], metadata

        metadata["extractor_parse_ok"] = True
        return predicted_triplets, metadata

    return [], metadata


def compute_triplet_match_score(
    predicted_triplet: dict[str, str], ground_truth_triplet: dict[str, str]
) -> float:
    matched_fields = sum(
        predicted_triplet[key] == ground_truth_triplet[key] for key in ("src", "rel", "tgt")
    )

    if matched_fields == 3:
        return 1.0
    if matched_fields == 2:
        return 2.0 / 3.0
    return 0.0


def compute_structured_triplet_coverage(
    predicted_triplets: list[dict[str, str]],
    ground_truth_triplets: list[dict[str, str]],
) -> tuple[float, dict[str, Any]]:
    if not ground_truth_triplets:
        return 1.0, {
            "coverage_mode": "structured_empty_ground_truth",
            "matched_triplets": [],
            "triplet_match_scores": [],
        }

    normalized_ground_truth = [normalize_triplet(triplet) for triplet in ground_truth_triplets]
    normalized_predicted = [normalize_triplet(triplet) for triplet in predicted_triplets]

    remaining_predicted = normalized_predicted.copy()
    matched_triplets: list[str] = []
    triplet_match_scores: list[float] = []

    for ground_truth_triplet in normalized_ground_truth:
        best_index: int | None = None
        best_score = 0.0

        for idx, predicted_triplet in enumerate(remaining_predicted):
            current_score = compute_triplet_match_score(
                predicted_triplet=predicted_triplet,
                ground_truth_triplet=ground_truth_triplet,
            )
            if current_score > best_score:
                best_score = current_score
                best_index = idx

        if best_index is not None and best_score > 0.0:
            matched_triplets.append(format_triplet(ground_truth_triplet))
            remaining_predicted.pop(best_index)

        triplet_match_scores.append(best_score)

    coverage = sum(triplet_match_scores) / len(normalized_ground_truth)
    return coverage, {
        "coverage_mode": "structured_llm",
        "matched_triplets": matched_triplets,
        "triplet_match_scores": triplet_match_scores,
        "ground_truth_triplets_normalized": normalized_ground_truth,
        "predicted_triplets_normalized": normalized_predicted,
    }


def compute_reasoning_coverage(
    reasoning: str,
    proof_triplets: list[str],
    ground_truth_triplets: list[dict[str, str]],
    **kwargs: Any,
) -> tuple[float, dict[str, Any]]:
    if not proof_triplets and not ground_truth_triplets:
        return 1.0, {
            "coverage_mode": "empty_ground_truth",
            "matched_triplets": [],
        }

    if (
        to_bool(kwargs.get("llm_graph_extractor_enabled", False))
        and reasoning
        and ground_truth_triplets
    ):
        predicted_triplets, extractor_metadata = extract_triplets_with_llm(reasoning, **kwargs)
        if extractor_metadata["extractor_request_ok"] and extractor_metadata["extractor_parse_ok"]:
            coverage, coverage_metadata = compute_structured_triplet_coverage(
                predicted_triplets=predicted_triplets,
                ground_truth_triplets=ground_truth_triplets,
            )
            return coverage, {
                **coverage_metadata,
                **extractor_metadata,
                "predicted_triplets": predicted_triplets,
                "ground_truth_triplets": ground_truth_triplets,
            }

        coverage, coverage_metadata = compute_heuristic_reasoning_coverage(
            reasoning=reasoning,
            proof_triplets=proof_triplets,
        )
        return coverage, {
            **coverage_metadata,
            **extractor_metadata,
            "coverage_mode": "heuristic_fallback",
            "predicted_triplets": [],
            "ground_truth_triplets": ground_truth_triplets,
        }

    coverage, coverage_metadata = compute_heuristic_reasoning_coverage(
        reasoning=reasoning,
        proof_triplets=proof_triplets,
    )
    return coverage, {
        **coverage_metadata,
        "extractor_model": optional_text(kwargs.get("llm_graph_extractor_model")),
        "extractor_request_ok": False,
        "extractor_parse_ok": False,
        "extractor_raw_response": "",
        "predicted_triplets": [],
        "ground_truth_triplets": ground_truth_triplets,
    }


def normalize_answer(answer: str) -> str | None:
    answer = answer.strip().lower()
    answer = re.sub(r"[.,!?;:\"']", "", answer).strip()

    aliases = get_answer_aliases()
    if answer in aliases:
        return aliases[answer]

    if "not entailment" in answer or "not entailed" in answer:
        return "not entailment"
    if "entailment" in answer or "entailed" in answer:
        return "entailment"

    return None


def zero_components() -> dict[str, float]:
    return {
        "score": 0.0,
        "acc": 0.0,
        "success_rate": 0.0,
        "coverage": 0.0,
        "reward_think_tags": 0.0,
        "reward_answer_tags": 0.0,
        "reward_correct_answer": 0.0,
        "reward_reasoning": 0.0,
        "reward_format": 0.0,
    }


def append_reward_log(
    log_file: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None,
    reward: float,
    **details: Any,
) -> None:
    with open(log_file, "a", encoding="utf-8") as f:
        entry = {
            "solution_str": solution_str,
            "ground_truth": str(ground_truth)[:200],
            "reward": reward,
            **details,
        }
        if extra_info:
            entry["question"] = extra_info.get("question", "")
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def compute_score(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    answer_tag_reward = float(kwargs.get("answer_tag_reward", 5.0))
    think_tag_reward = float(kwargs.get("think_tag_reward", 5.0))
    correct_answer_reward = float(kwargs.get("correct_answer_reward", 100.0))
    graph_coverage_reward = float(kwargs.get("graph_coverage_reward", 50.0))
    graph_coverage_scale = float(kwargs.get("graph_coverage_scale", 0.0))
    format_penalty = float(kwargs.get("format_penalty", 10.0))
    log_file = kwargs.get("log_file", None)

    try:
        tags, content_len = extract_tags_and_content_length(solution_str)
    except Exception as exc:
        if log_file:
            append_reward_log(
                log_file,
                solution_str,
                ground_truth,
                extra_info,
                0.0,
                parse_error=True,
                parse_error_type=type(exc).__name__,
            )
        return zero_components()

    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    gt_label = gt["label"]
    proof_triplets, ground_truth_triplets = get_ground_truth_triplets(gt)

    reasoning = tags["think"][0] if tags["think"] else ""
    answer = tags["answer"][-1] if tags["answer"] else ""

    reward_think_tags = 0.0
    if len(tags["think"]) == 1:
        reward_think_tags = think_tag_reward
    elif len(tags["think"]) > 1:
        reward_think_tags = -(len(tags["think"]) - 1) * think_tag_reward

    reward_answer_tags = 0.0
    if len(tags["answer"]) == 1:
        reward_answer_tags = answer_tag_reward
    elif len(tags["answer"]) > 1:
        reward_answer_tags = -(len(tags["answer"]) - 1) * answer_tag_reward

    normalized = normalize_answer(answer)
    is_correct = normalized is not None and normalized == gt_label
    reward_correct_answer = correct_answer_reward if is_correct else 0.0

    coverage = 0.0
    coverage_details: dict[str, Any] = {
        "coverage_mode": "no_reasoning",
        "matched_triplets": [],
        "ground_truth_triplets": ground_truth_triplets,
        "predicted_triplets": [],
        "extractor_model": optional_text(kwargs.get("llm_graph_extractor_model")),
        "extractor_request_ok": False,
        "extractor_parse_ok": False,
        "extractor_raw_response": "",
    }
    if reasoning:
        coverage, coverage_details = compute_reasoning_coverage(
            reasoning=reasoning,
            proof_triplets=proof_triplets,
            ground_truth_triplets=ground_truth_triplets,
            **kwargs,
        )

    reward_reasoning = coverage * graph_coverage_reward * graph_coverage_scale

    reward_format = -min(content_len * format_penalty, 100.0)

    reward = (
        reward_think_tags
        + reward_answer_tags
        + reward_correct_answer
        + reward_reasoning
        + reward_format
    )

    acc = float(is_correct)
    if log_file:
        append_reward_log(
            log_file,
            solution_str,
            ground_truth,
            extra_info,
            reward,
            reasoning=reasoning,
            answer=answer,
            is_correct=is_correct,
            coverage=coverage,
            coverage_mode=coverage_details.get("coverage_mode"),
            matched_triplets=coverage_details.get("matched_triplets"),
            ground_truth_triplets=coverage_details.get("ground_truth_triplets"),
            predicted_triplets=coverage_details.get("predicted_triplets"),
            triplet_match_scores=coverage_details.get("triplet_match_scores"),
            ground_truth_triplets_normalized=coverage_details.get(
                "ground_truth_triplets_normalized"
            ),
            predicted_triplets_normalized=coverage_details.get("predicted_triplets_normalized"),
            extractor_model=coverage_details.get("extractor_model"),
            extractor_request_ok=coverage_details.get("extractor_request_ok"),
            extractor_parse_ok=coverage_details.get("extractor_parse_ok"),
            extractor_raw_response=coverage_details.get("extractor_raw_response"),
            extractor_error=coverage_details.get("extractor_error"),
            extractor_attempts=coverage_details.get("extractor_attempts"),
            reward_think_tags=reward_think_tags,
            reward_answer_tags=reward_answer_tags,
            reward_correct_answer=reward_correct_answer,
            reward_reasoning=reward_reasoning,
            reward_format=reward_format,
        )

    return {
        "score": reward,
        "acc": acc,
        "success_rate": acc,
        "coverage": coverage,
        "reward_think_tags": reward_think_tags,
        "reward_answer_tags": reward_answer_tags,
        "reward_correct_answer": reward_correct_answer,
        "reward_reasoning": reward_reasoning,
        "reward_format": reward_format,
    }
