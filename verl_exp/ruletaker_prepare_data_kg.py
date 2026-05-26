import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a logical reasoning assistant. "
    "Given a context containing facts and rules about entities, "
    "determine whether a given statement can be logically derived "
    "from the provided information.\n\n"
    "Analyze the context step by step in <think> tags, then give your "
    "answer in <answer> tags.\n\n"
    "The <answer> block MUST contain:\n"
    "1. A JSON array of reasoning edges — each edge is "
    "[\"evidence\", \"conclusion\"], meaning \"evidence supports conclusion\".\n"
    "2. On a separate line: exactly 'entailment' or 'not entailment'.\n\n"
    "Rules for building edges:\n"
    "- Each edge is [\"A\", \"B\"] where A is a fact or derived claim that "
    "supports B.\n"
    "- Facts (from the context) are leaves — they appear only as evidence, "
    "never as conclusion.\n"
    "- The statement you are checking is the root — it appears only as "
    "conclusion, never as evidence.\n"
    "- Write claims as \"subject relation object\" (e.g. \"cow chases lion\").\n"
    "- For entailment: pick ONE rule per step and include ALL its conditions.\n"
    "- For not entailment: show that NO rule works — for each possible rule, "
    "show ONE condition that fails.\n"
    "- Do NOT wrap the JSON in markdown code fences.\n\n"
    "Example:\n"
    "<think>cow chases lion is a fact. Rule says: if cow chases lion then "
    "lion needs tiger. So lion needs tiger.</think>\n"
    "<answer>\n"
    "[[\"cow chases lion\", \"lion needs tiger\"]]\n"
    "entailment\n"
    "</answer>"
)

SYSTEM_PROMPT_BASELINE = (
    "You are a logical reasoning assistant. "
    "Given a context containing facts and rules about entities, "
    "determine whether a given statement can be logically derived "
    "from the provided information.\n\n"
    "Analyze the context step by step, identifying which facts and "
    "rules are relevant to prove or disprove the statement. "
    "Put your reasoning in <think> tags and your final answer in <answer> tags. "
    "Your answer must be exactly \"entailment\" if the statement follows from "
    "the context, or \"not entailment\" if it does not."
)

USER_TEMPLATE_BASELINE = (
    "Context:\n{context}\n\n"
    "Statement: {question}\n\n"
    "Can this statement be derived from the facts and rules in the context? "
    "Reply with <think>...</think><answer>entailment OR not entailment</answer>."
)

USER_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Statement: {question}\n\n"
    "Can this statement be derived from the facts and rules in the context? "
    "Reply with <think>...</think><answer>[[edges...]]\\n"
    "entailment OR not entailment</answer>."
)


def has_negation(item: dict) -> bool:
    """Return True if any fact or rule condition involves a negation relation."""
    for s in item.get("structured", []):
        for t in s.get("consequents", []) + s.get("conditions", []):
            if "not" in t.get("rel", "").lower():
                return True
    return False


def _claim_text(node: dict) -> str | None:
    if node.get("type") in ("question", "condition"):
        src = node.get("src", "").strip()
        rel = node.get("rel", "").strip()
        tgt = node.get("tgt", "").strip()
        if src and rel and tgt:
            return f"{src} {rel} {tgt}"
    return None


def extract_proof_edges(
    nodes: list[dict], adj_lists: list[dict],
) -> list[list[str]]:
    """Extract [evidence, conclusion] edges by collapsing RULE vertices.

    RUS graph alternates: QUESTION/CONDITION → RULE → CONDITION.
    We collapse to: CONDITION → QUESTION/CONDITION (skipping RULE nodes).
    Direction is reversed: evidence → conclusion (bottom-up).
    """
    node_map = {n["node_id"]: n for n in nodes}
    parent_of: dict[int, int] = {}
    for adj in adj_lists:
        parent_of[adj["to"]] = adj["from"]

    edges: list[list[str]] = []
    for node in nodes:
        if node["type"] != "condition":
            continue
        rule_nid = parent_of.get(node["node_id"])
        if rule_nid is None:
            continue
        rule_node = node_map.get(rule_nid)
        if rule_node is None or rule_node["type"] != "rule":
            continue
        gp_nid = parent_of.get(rule_nid)
        if gp_nid is None:
            continue
        gp_node = node_map.get(gp_nid)
        if gp_node is None:
            continue

        c_text = _claim_text(node)
        p_text = _claim_text(gp_node)
        if c_text and p_text:
            edges.append([c_text, p_text])

    return edges


def extract_proof_rules(
    nodes: list[dict], adj_lists: list[dict],
) -> dict[str, list[list[str]]]:
    """For each conclusion, list valid rule variants (sets of conditions).

    Returns: {conclusion_text: [[cond1, cond2], [cond3], ...]}
    Each inner list = all conditions of one rule that derives this conclusion.
    """
    node_map = {n["node_id"]: n for n in nodes}
    children_of: dict[int, list[int]] = defaultdict(list)
    for adj in adj_lists:
        children_of[adj["from"]].append(adj["to"])

    rules: dict[str, list[list[str]]] = defaultdict(list)

    for node in nodes:
        if node["type"] not in ("question", "condition"):
            continue
        p_text = _claim_text(node)
        if not p_text:
            continue
        for rule_nid in children_of[node["node_id"]]:
            rule_node = node_map.get(rule_nid)
            if rule_node is None or rule_node["type"] != "rule":
                continue
            conds: set[str] = set()
            for cond_nid in children_of[rule_nid]:
                cond_node = node_map.get(cond_nid)
                if cond_node and cond_node["type"] == "condition":
                    ct = _claim_text(cond_node)
                    if ct:
                        conds.add(ct)
            if conds:
                rules[p_text].append(sorted(conds))

    for key in rules:
        seen: set[tuple[str, ...]] = set()
        deduped: list[list[str]] = []
        for variant in rules[key]:
            t = tuple(variant)
            if t not in seen:
                seen.add(t)
                deduped.append(variant)
        rules[key] = deduped

    return dict(rules)


def make_verl_row(item: dict, idx: int) -> dict:
    context = item["context"]
    question = item["question"]
    label = item["label"]
    config = item["config"]
    nodes = item["nodes"]
    adj_lists = item["adj_lists"]

    user_content = USER_TEMPLATE_BASELINE.format(context=context, question=question)
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT_BASELINE},
        {"role": "user", "content": user_content},
    ]

    proof_edges = extract_proof_edges(nodes=nodes, adj_lists=adj_lists)
    proof_rules = extract_proof_rules(nodes=nodes, adj_lists=adj_lists)

    ground_truth = json.dumps(
        {
            "label": label,
            "proof_edges": proof_edges,
            "proof_rules": proof_rules,
        },
        ensure_ascii=False,
    )

    return {
        "data_source": "ruletaker_kg",
        "prompt": prompt,
        "ability": "logical_reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "index": idx,
            "config": config,
            "question": question,
            "context": context,
            "raw_prompt": prompt,
        },
    }


def load_raw_data(args) -> list[dict]:
    if args.input:
        with open(args.input) as f:
            return json.load(f)
    from datasets import load_dataset
    ds = load_dataset(args.hf_dataset, split=args.split)
    return [dict(row) for row in ds]


def main():
    parser = argparse.ArgumentParser(description="Prepare KGReasoning dataset for Verl training")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--hf_dataset", type=str, default="rusnarziev35/KGReasoning")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--output_dir", type=str, default="./data/ruletaker_kg")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_depth",
        type=str,
        default=None,
        help="Filter by config prefix(es), comma-separated. E.g. 'depth-1,depth-2,depth-3'",
    )
    parser.add_argument(
        "--filter_negations",
        action="store_true",
        default=True,
        help="Skip examples with negation relations (is not, does not, etc.)",
    )
    parser.add_argument("--no_filter_negations", dest="filter_negations", action="store_false")
    parser.add_argument(
        "--filter_natlang",
        action="store_true",
        default=True,
        help="Skip NatLang configs (less structured, harder for graph extraction)",
    )
    parser.add_argument("--no_filter_natlang", dest="filter_natlang", action="store_false")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    raw_data = load_raw_data(args)
    print(f"Loaded {len(raw_data)} examples")

    if args.filter_negations:
        before = len(raw_data)
        raw_data = [d for d in raw_data if not has_negation(d)]
        print(f"After negation filter: {len(raw_data)} (removed {before - len(raw_data)})")

    if args.filter_natlang:
        before = len(raw_data)
        raw_data = [d for d in raw_data if "NatLang" not in d["config"]]
        print(f"After NatLang filter: {len(raw_data)} (removed {before - len(raw_data)})")

    if args.max_depth:
        prefixes = [p.strip() for p in args.max_depth.split(",")]
        before = len(raw_data)
        raw_data = [d for d in raw_data if any(d["config"].startswith(p) for p in prefixes)]
        print(f"After depth filter ({args.max_depth}): {len(raw_data)} (removed {before - len(raw_data)})")

    from collections import Counter
    print(f"Configs: {dict(Counter(d['config'] for d in raw_data))}")
    print(f"Labels:  {dict(Counter(d['label'] for d in raw_data))}")

    random.seed(args.seed)
    random.shuffle(raw_data)

    verl_data = [make_verl_row(item, idx) for idx, item in enumerate(raw_data)]

    split_idx = int(len(verl_data) * args.train_ratio)
    train_data = verl_data[:split_idx]
    val_data = verl_data[split_idx:]

    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"

    with open(train_path, "w") as f:
        json.dump(train_data, f, ensure_ascii=False)
    with open(val_path, "w") as f:
        json.dump(val_data, f, ensure_ascii=False)

    print(f"Train: {len(train_data)} examples -> {train_path}")
    print(f"Val:   {len(val_data)} examples -> {val_path}")


if __name__ == "__main__":
    main()
