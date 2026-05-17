#!/usr/bin/env python3
"""Convert ruletaker Arrow dataset to JSONL for verl PPO training."""

import json
from datasets import load_from_disk

SYSTEM_PROMPT = (
    "You are a logical reasoning assistant. Your task is to determine whether a given "
    "statement logically follows from a set of rules and facts (the context).\n\n"
    "You MUST structure your response exactly as follows:\n"
    "1. Put your chain-of-thought reasoning inside <think>...</think> tags.\n"
    "2. Put your final answer inside <answer>...</answer> tags.\n"
    "   Inside <answer>, first output the reasoning graph as a JSON array of triplets "
    '[{"src": "subject", "rel": "relation", "tgt": "object"}, ...], '
    'then on the next line output either "entailment" or "not entailment".\n\n'
    "Example:\n"
    "<think>\nLet me analyze the context step by step...\n</think>\n"
    '<answer>\n[{"src": "Bob", "rel": "is", "tgt": "kind"}]\nentailment\n</answer>'
)

ds = load_from_disk("data/ruletaker")

for split in ["train", "test", "dev"]:
    data = ds[split]
    output_path = f"data/ruletaker/{split}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        for row in data:
            context = row["context"]
            question = row["question"]
            label = row["label"]
            config = row.get("config", "")

            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question: {question}\n\n"
                        'Determine if the question is entailed by the context. '
                        'Answer "entailment" or "not entailment".'
                    ),
                },
            ]

            item = {
                "prompt": prompt,
                "data_source": f"ruletaker/{config}",
                "reward_model": {
                    "ground_truth": {
                        "label": label,
                        "proof_triplets": [],
                        "proof_triplets_structured": [],
                    }
                },
                "extra_info": {
                    "question": question,
                    "context": context,
                    "config": config,
                },
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(data)} rows to {output_path}")

# val.json = dev.json (as used in the shell script)
import shutil
shutil.copy("data/ruletaker/dev.json", "data/ruletaker/val.json")
print("Copied dev.json -> val.json")
