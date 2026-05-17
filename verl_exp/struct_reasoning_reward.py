import json
import re
from typing import Any


def extract_tags_and_content_length(text: str) -> tuple[dict[str, list[str]], int]:
    """
    Parse XML-like tags from text with regex.
    Returns a dictionary with keys 'think' and 'answer',
    plus the total word count of content outside these tags.

    The <answer> block now carries BOTH the reasoning graph (JSON array of
    triplets) AND the final label ('entailment' / 'not entailment').
    Splitting those two is done downstream in `parse_answer_block`.
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


def parse_answer_block(
    answer_text: str,
) -> tuple[list[dict[str, str]], str, dict[str, Any]]:
    """Parse the merged <answer> block into (predicted_triplets, label_text, metadata).

    Expected format inside <answer>:
        [{"src": "...", "rel": "...", "tgt": "..."}, ...]
        entailment | not entailment

    Accepted graph formats inside <answer> (tried in order):
      1. JSON array of objects:  [{"src": "...", "rel": "...", "tgt": "..."}, ...]
      2. JSON array of arrays:   [["src", "rel", "tgt"], ...]
      3. JSON object with key:   {"triplets": [ ... ]}
      4. Same as above wrapped in ```json ... ``` fences.
      5. Plain text fallback:    one "src rel tgt" per line (label on its own line).

    label_text is whatever sits OUTSIDE the JSON array (or the non-triplet lines
    in the fallback case) — that is what we feed to `normalize_answer`.

    No external LLM is called.
    """
    metadata: dict[str, Any] = {
        "graph_parse_ok": False,
        "graph_parse_mode": None,
        "graph_parse_error": None,
        "graph_raw": "",
    }

    if answer_text is None or not str(answer_text).strip():
        metadata["graph_parse_error"] = "empty answer block"
        return [], "", metadata

    cleaned = re.sub(r"```(?:json)?", "", answer_text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    # Greedy: from first '[' to last ']' — captures multi-line JSON arrays.
    array_match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)

    if array_match is None:
        # Fallback: maybe the model wrote "src rel tgt" per line.
        line_triplets: list[dict[str, str]] = []
        non_triplet_lines: list[str] = []
        for line in cleaned.splitlines():
            parsed_triplet = triplet_from_string(line)
            if parsed_triplet is not None:
                line_triplets.append(parsed_triplet)
            else:
                non_triplet_lines.append(line)

        if line_triplets:
            metadata["graph_parse_ok"] = True
            metadata["graph_parse_mode"] = "lines"
            metadata["graph_raw"] = "\n".join(
                format_triplet(t) for t in line_triplets
            )
            label_text = "\n".join(non_triplet_lines).strip()
            return deduplicate_triplets(line_triplets), label_text, metadata

        metadata["graph_parse_error"] = "no JSON array or 'src rel tgt' lines found"
        return [], cleaned, metadata

    graph_text = array_match.group(0)
    metadata["graph_raw"] = graph_text
    # Anything outside the array is the label.
    label_text = (
        cleaned[: array_match.start()] + " " + cleaned[array_match.end():]
    ).strip()

    try:
        payload = json.loads(graph_text)
        metadata["graph_parse_mode"] = "json"
    except json.JSONDecodeError as exc:
        metadata["graph_parse_error"] = f"json: {exc}"
        return [], label_text, metadata

    items = payload.get("triplets") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        metadata["graph_parse_error"] = "payload is not a list of triplets"
        return [], label_text, metadata

    parsed_triplets: list[dict[str, str]] = []
    for item in items:
        parsed_triplet = parse_triplet_item(item)
        if parsed_triplet is None:
            continue
        if not any(parsed_triplet.values()):
            continue
        parsed_triplets.append(parsed_triplet)

    metadata["graph_parse_ok"] = True
    return deduplicate_triplets(parsed_triplets), label_text, metadata


def compute_triplet_match_score(
    predicted_triplet: dict[str, str], ground_truth_triplet: dict[str, str]
) -> float:
    matched_fields = sum(
        predicted_triplet[key] == ground_truth_triplet[key]
        for key in ("src", "rel", "tgt")
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

    normalized_ground_truth = [
        normalize_triplet(triplet) for triplet in ground_truth_triplets
    ]
    normalized_predicted = [
        normalize_triplet(triplet) for triplet in predicted_triplets
    ]

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
        "coverage_mode": "structured_self_reported",
        "matched_triplets": matched_triplets,
        "triplet_match_scores": triplet_match_scores,
        "ground_truth_triplets_normalized": normalized_ground_truth,
        "predicted_triplets_normalized": normalized_predicted,
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
        "graph_parse_ok": 0.0,
        "reward_think_tags": 0.0,
        "reward_answer_tags": 0.0,
        "reward_graph_parse": 0.0,
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
    graph_coverage_reward = float(kwargs.get("graph_coverage_reward", 100.0))
    graph_coverage_scale = float(kwargs.get("graph_coverage_scale", 1.0))
    graph_parse_bonus = float(kwargs.get("graph_parse_bonus", 5.0))
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
    _, ground_truth_triplets = get_ground_truth_triplets(gt)

    reasoning = tags["think"][0] if tags["think"] else ""
    answer_text = tags["answer"][-1] if tags["answer"] else ""

    # Tag rewards: encourage exactly one <think> and one <answer>.
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
    else:
        # No <answer> at all — graph + label are both missing, so hit harder.
        reward_answer_tags = -answer_tag_reward * 10

    # Split <answer> into (graph triplets, label remainder, parse metadata).
    predicted_triplets, label_text, parse_metadata = parse_answer_block(answer_text)

    # Answer correctness derived from the text outside the JSON array.
    normalized = normalize_answer(label_text) if label_text else None
    is_correct = normalized is not None and normalized == gt_label
    reward_correct_answer = correct_answer_reward if is_correct else 0.0

    reward_graph_parse = (
        graph_parse_bonus if parse_metadata["graph_parse_ok"] else 0.0
    )

    coverage = 0.0
    coverage_details: dict[str, Any] = {
        "coverage_mode": "no_graph",
        "matched_triplets": [],
        "ground_truth_triplets": ground_truth_triplets,
        "predicted_triplets": predicted_triplets,
    }

    if not ground_truth_triplets:
        coverage = 1.0
        coverage_details = {
            "coverage_mode": "structured_empty_ground_truth",
            "matched_triplets": [],
            "triplet_match_scores": [],
            "ground_truth_triplets": ground_truth_triplets,
            "predicted_triplets": predicted_triplets,
        }
    elif predicted_triplets:
        coverage, structured_details = compute_structured_triplet_coverage(
            predicted_triplets=predicted_triplets,
            ground_truth_triplets=ground_truth_triplets,
        )
        coverage_details = {
            **structured_details,
            "ground_truth_triplets": ground_truth_triplets,
            "predicted_triplets": predicted_triplets,
        }

    reward_reasoning = coverage * graph_coverage_reward * graph_coverage_scale

    reward_format = -min(content_len * format_penalty, 100.0)

    reward = (
        reward_think_tags
        + reward_answer_tags
        + reward_graph_parse
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
            answer_raw=answer_text,
            graph_raw=parse_metadata.get("graph_raw"),
            label_text=label_text,
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
            predicted_triplets_normalized=coverage_details.get(
                "predicted_triplets_normalized"
            ),
            graph_parse_ok=parse_metadata.get("graph_parse_ok"),
            graph_parse_mode=parse_metadata.get("graph_parse_mode"),
            graph_parse_error=parse_metadata.get("graph_parse_error"),
            reward_think_tags=reward_think_tags,
            reward_answer_tags=reward_answer_tags,
            reward_graph_parse=reward_graph_parse,
            reward_correct_answer=reward_correct_answer,
            reward_reasoning=reward_reasoning,
            reward_format=reward_format,
        )

    return {
        "score": reward,
        "acc": acc,
        "success_rate": acc,
        "coverage": coverage,
        "graph_parse_ok": float(bool(parse_metadata.get("graph_parse_ok"))),
        "reward_think_tags": reward_think_tags,
        "reward_answer_tags": reward_answer_tags,
        "reward_graph_parse": reward_graph_parse,
        "reward_correct_answer": reward_correct_answer,
        "reward_reasoning": reward_reasoning,
        "reward_format": reward_format,
    }