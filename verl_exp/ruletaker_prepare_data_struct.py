import argparse
import json
import random
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a logical reasoning assistant. "
    "Given a context containing facts and rules about entities, "
    "determine whether a given statement can be logically derived "
    "from the provided information.\n\n"
    "Analyze the context step by step, identifying which facts and "
    "rules are relevant to prove or disprove the statement.\n\n"
    "Respond in EXACTLY this format, in this order, with no extra text "
    "outside the tags:\n"
    "<think>your step-by-step reasoning</think>\n"
    "<answer>\n"
    "<a JSON array of every fact and rule application you used>\n"
    "entailment OR not entailment\n"
    "</answer>\n\n"
    "Rules for the <answer> block:\n"
    "- It MUST contain a valid JSON array on the first lines, "
    "then the final label on a separate line.\n"
    "- Each array element MUST be an object with exactly three string keys: "
    "\"src\", \"rel\", \"tgt\".\n"
    "- Include one triplet for every fact you relied on and every "
    "consequence/condition of every rule you applied. Cover the full chain "
    "from premises to conclusion.\n"
    "- Use concise entity names and relations, copied verbatim from the "
    "context where possible.\n"
    "- Do NOT wrap the JSON in markdown code fences.\n"
    "- The final line of <answer> MUST be exactly 'entailment' or 'not entailment'.\n"
    "- If you cannot derive the statement, still list the facts and rule "
    "applications you considered.\n\n"
    "Example of the required output format:\n"
    "<think>Alice is kind (fact). Rule: kind people are happy. So Alice is happy.</think>\n"
    "<answer>\n"
    "[{\"src\": \"Alice\", \"rel\": \"is\", \"tgt\": \"kind\"}, "
    "{\"src\": \"kind people\", \"rel\": \"are\", \"tgt\": \"happy\"}, "
    "{\"src\": \"Alice\", \"rel\": \"is\", \"tgt\": \"happy\"}]\n"
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

USER_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Statement: {question}\n\n"
    "Can this statement be derived from the facts and rules in the context? "
    "Reply with <think>...</think><answer>[...]\\nentailment OR not entailment</answer>."
)


def make_triplet(src: str, rel: str, tgt: str) -> dict[str, str]:
    return {
        "src": src.strip(),
        "rel": rel.strip(),
        "tgt": tgt.strip(),
    }


def format_triplet(triplet: dict[str, str]) -> str:
    return f"{triplet['src']} {triplet['rel']} {triplet['tgt']}"


def extract_structured_proof_triplets(
    nodes: list[dict[str, str]], structured: list[dict]
) -> list[dict[str, str]]:
    structured_map = {s["id"]: s for s in structured}
    triplets: set[tuple[str, str, str]] = set()

    for node in nodes:
        node_type = node.get("type", "")

        if node_type == "fact":
            fact_id = node.get("fact_id")
            fact = structured_map.get(fact_id)
            if fact:
                for t in fact.get("consequents", []):
                    triplets.add((t["src"].strip(), t["rel"].strip(), t["tgt"].strip()))

        elif node_type == "rule":
            rule_id = node.get("rule_id")
            rule = structured_map.get(rule_id)
            if rule:
                for t in rule.get("conditions", []):
                    triplets.add((t["src"].strip(), t["rel"].strip(), t["tgt"].strip()))
                for t in rule.get("consequences", []):
                    triplets.add((t["src"].strip(), t["rel"].strip(), t["tgt"].strip()))

        elif node_type == "condition":
            src = node.get("src", "")
            rel = node.get("rel", "")
            tgt = node.get("tgt", "")
            if src and rel and tgt:
                triplets.add((src.strip(), rel.strip(), tgt.strip()))

    return [make_triplet(src=src, rel=rel, tgt=tgt) for src, rel, tgt in sorted(triplets)]


def extract_proof_triplets(nodes: list[dict], structured: list[dict]) -> list[str]:
    structured_triplets = extract_structured_proof_triplets(nodes=nodes, structured=structured)
    return [format_triplet(triplet) for triplet in structured_triplets]


def make_verl_row(item: dict, idx: int) -> dict:
    context = item["context"]
    question = item["question"]
    label = item["label"]
    config = item["config"]
    nodes = item["nodes"]
    structured = item["structured"]

    user_content = USER_TEMPLATE.format(context=context, question=question)
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT_BASELINE},
        {"role": "user", "content": user_content},
    ]

    proof_triplets_structured = extract_structured_proof_triplets(nodes=nodes, structured=structured)
    proof_triplets = [format_triplet(triplet) for triplet in proof_triplets_structured]

    ground_truth = json.dumps(
        {
            "label": label,
            "proof_triplets": proof_triplets,
            "proof_triplets_structured": proof_triplets_structured,
        },
        ensure_ascii=False,
    )

    return {
        "data_source": "ruletaker",
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

    ds = load_dataset(args.hf_dataset, args.hf_name, split=args.split)
    return [row for row in ds]


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ruletaker dataset for Verl training"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to local JSON file (alternative to HF dataset)",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="VavGreg/wikidata_b21_dataset",
    )
    parser.add_argument(
        "--hf_name",
        type=str,
        default="ruletaker_dev_with_reasoning_graph",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="./data/ruletaker")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_depth",
        type=str,
        default=None,
        help="Filter by config prefix, e.g. 'depth-1' or 'depth-3'",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data...")
    raw_data = load_raw_data(args)
    print(f"Loaded {len(raw_data)} examples")

    if args.max_depth:
        raw_data = [d for d in raw_data if d["config"].startswith(args.max_depth)]
        print(f"Filtered to {len(raw_data)} examples with config={args.max_depth}*")

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