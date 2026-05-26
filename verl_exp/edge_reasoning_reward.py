import json
import re
from collections import defaultdict
from itertools import count
from typing import Any

_log_counter = count()


def extract_tags_and_content_length(text: str) -> tuple[dict[str, list[str]], int]:
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
    for pat in think_inner_patterns:
        think_blocks.extend(re.findall(pat, text, flags=re.DOTALL))

    tags = {
        "think": think_blocks,
        "answer": re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL),
    }

    text_without_tags = text
    for pat in think_strip_patterns:
        text_without_tags = re.sub(pat, "", text_without_tags, flags=re.DOTALL)
    text_without_tags = re.sub(
        r"<answer>.*?</answer>", "", text_without_tags, flags=re.DOTALL
    )
    content_length = len(text_without_tags.split())

    return tags, content_length


def normalize_claim(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_answer_block(
    answer_text: str,
) -> tuple[list[list[str]], str, dict[str, Any]]:
    """Parse <answer> into (edges, label_text, metadata).

    Expected:
        [[ev1, concl1], [ev2, concl2], ...]
        entailment | not entailment
    """
    metadata: dict[str, Any] = {
        "graph_parse_ok": False,
        "graph_parse_error": None,
    }

    if not answer_text or not answer_text.strip():
        metadata["graph_parse_error"] = "empty answer block"
        return [], "", metadata

    cleaned = re.sub(r"```(?:json)?", "", answer_text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    array_match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if array_match is None:
        metadata["graph_parse_error"] = "no JSON array found"
        return [], cleaned, metadata

    json_text = array_match.group(0)
    label_text = (
        cleaned[: array_match.start()] + " " + cleaned[array_match.end() :]
    ).strip()

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        metadata["graph_parse_error"] = f"json: {exc}"
        return [], label_text, metadata

    if not isinstance(payload, list):
        metadata["graph_parse_error"] = "payload is not a list"
        return [], label_text, metadata

    edges: list[list[str]] = []
    for item in payload:
        if isinstance(item, list) and len(item) == 2:
            ev = str(item[0]).strip()
            concl = str(item[1]).strip()
            if ev and concl:
                edges.append([ev, concl])

    metadata["graph_parse_ok"] = True
    return edges, label_text, metadata


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


def validate_subgraph(
    predicted_edges: list[list[str]],
    proof_rules: dict[str, list[list[str]]],
    question_text: str,
) -> tuple[float, dict[str, Any]]:
    """Check if predicted edges form a valid subgraph of the proof graph.

    Returns (score, details) where score is fraction of valid nodes.
    """
    if not proof_rules:
        if not predicted_edges:
            return 1.0, {"validation": "empty_gt_empty_pred"}
        return 0.5, {"validation": "empty_gt_nonempty_pred"}

    if not predicted_edges:
        return 0.0, {"validation": "nonempty_gt_empty_pred"}

    norm_rules: dict[str, list[set[str]]] = {}
    for concl, variants in proof_rules.items():
        nc = normalize_claim(concl)
        norm_rules[nc] = [
            {normalize_claim(c) for c in variant} for variant in variants
        ]

    by_conclusion: dict[str, set[str]] = defaultdict(set)
    for ev, concl in predicted_edges:
        by_conclusion[normalize_claim(concl)].add(normalize_claim(ev))

    norm_question = normalize_claim(question_text)

    visited: set[str] = set()
    valid_nodes = 0
    total_nodes = 0
    invalid_details: list[str] = []

    def check_node(node: str) -> bool:
        nonlocal valid_nodes, total_nodes
        if node in visited:
            return True
        visited.add(node)
        total_nodes += 1

        evidences = by_conclusion.get(node)
        if evidences is None:
            valid_nodes += 1
            return True

        gt_variants = norm_rules.get(node)
        if gt_variants is None:
            invalid_details.append(f"no gt rules for '{node}'")
            return False

        matched = False
        for variant in gt_variants:
            if evidences == variant:
                matched = True
                break

        if not matched:
            invalid_details.append(
                f"'{node}': pred={sorted(evidences)}, "
                f"gt_variants={[sorted(v) for v in gt_variants]}"
            )
            valid_nodes += 1 * 0.5
            for ev in evidences:
                check_node(ev)
            return False

        valid_nodes += 1
        for ev in evidences:
            check_node(ev)
        return True

    check_node(norm_question)
    score = valid_nodes / total_nodes if total_nodes > 0 else 0.0

    return score, {
        "validation": "checked",
        "valid_nodes": valid_nodes,
        "total_nodes": total_nodes,
        "invalid_details": invalid_details,
    }


def zero_components() -> dict[str, float]:
    return {
        "score": 0.0,
        "acc": 0.0,
        "success_rate": 0.0,
        "graph_coverage": 0.0,
        "graph_parse_ok": 0.0,
        "reward_think_tags": 0.0,
        "reward_answer_tags": 0.0,
        "reward_graph_parse": 0.0,
        "reward_correct_answer": 0.0,
        "reward_graph_validity": 0.0,
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
            "solution_str": solution_str[:500],
            "ground_truth": str(ground_truth)[:300],
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
    think_tag_reward = float(kwargs.get("think_tag_reward", 5.0))
    answer_tag_reward = float(kwargs.get("answer_tag_reward", 5.0))
    correct_answer_reward = float(kwargs.get("correct_answer_reward", 100.0))
    graph_validity_reward = float(kwargs.get("graph_validity_reward", 100.0))
    graph_validity_scale = float(kwargs.get("graph_validity_scale", 1.0))
    graph_parse_bonus = float(kwargs.get("graph_parse_bonus", 5.0))
    format_penalty = float(kwargs.get("format_penalty", 10.0))
    log_file = kwargs.get("log_file", None)
    log_sample_rate = int(kwargs.get("log_sample_rate", 100))
    should_log = log_file and (next(_log_counter) % log_sample_rate == 0)

    try:
        tags, content_len = extract_tags_and_content_length(solution_str)
    except Exception:
        if should_log:
            append_reward_log(
                log_file, solution_str, ground_truth, extra_info, 0.0,
                parse_error=True,
            )
        return zero_components()

    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    gt_label = gt["label"]
    gt_proof_rules = gt.get("proof_rules", {})

    answer_text = tags["answer"][-1] if tags["answer"] else ""

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
        reward_answer_tags = -answer_tag_reward * 10

    predicted_edges, label_text, parse_meta = parse_answer_block(answer_text)

    normalized = normalize_answer(label_text) if label_text else None
    is_correct = normalized is not None and normalized == gt_label
    reward_correct_answer = correct_answer_reward if is_correct else 0.0

    reward_graph_parse = graph_parse_bonus if parse_meta["graph_parse_ok"] else 0.0

    question_text = ""
    if extra_info:
        question_text = extra_info.get("question", "")
    question_text = re.sub(
        r"^(The |A |An )", "", question_text, flags=re.IGNORECASE,
    ).rstrip(".")

    graph_coverage = 0.0
    validity_details: dict[str, Any] = {}
    if gt_label == "not entailment":
        if not predicted_edges:
            graph_coverage = 1.0
            validity_details = {"validation": "not_entailment_empty_ok"}
        else:
            graph_coverage = 0.5
            validity_details = {"validation": "not_entailment_has_edges"}
    elif gt_proof_rules:
        graph_coverage, validity_details = validate_subgraph(
            predicted_edges=predicted_edges,
            proof_rules=gt_proof_rules,
            question_text=question_text,
        )
    elif not predicted_edges:
        graph_coverage = 1.0
        validity_details = {"validation": "both_empty"}

    reward_graph_validity = (
        graph_coverage * graph_validity_reward * graph_validity_scale
    )

    reward_format = -min(content_len * format_penalty, 100.0)

    reward = (
        reward_think_tags
        + reward_answer_tags
        + reward_graph_parse
        + reward_correct_answer
        + reward_graph_validity
        + reward_format
    )

    acc = float(is_correct)

    if should_log:
        append_reward_log(
            log_file, solution_str, ground_truth, extra_info, reward,
            is_correct=is_correct,
            label_text=label_text,
            graph_coverage=graph_coverage,
            graph_parse_ok=parse_meta.get("graph_parse_ok"),
            graph_parse_error=parse_meta.get("graph_parse_error"),
            predicted_edges=predicted_edges,
            validity=validity_details,
            reward_think_tags=reward_think_tags,
            reward_answer_tags=reward_answer_tags,
            reward_graph_parse=reward_graph_parse,
            reward_correct_answer=reward_correct_answer,
            reward_graph_validity=reward_graph_validity,
            reward_format=reward_format,
        )

    return {
        "score": reward,
        "acc": acc,
        "success_rate": acc,
        "graph_coverage": graph_coverage,
        "graph_parse_ok": float(bool(parse_meta.get("graph_parse_ok"))),
        "reward_think_tags": reward_think_tags,
        "reward_answer_tags": reward_answer_tags,
        "reward_graph_parse": reward_graph_parse,
        "reward_correct_answer": reward_correct_answer,
        "reward_graph_validity": reward_graph_validity,
        "reward_format": reward_format,
    }
