#!/usr/bin/env python3
"""Pre-filter ruletaker dataset: keep only prompts ≤ 1024 tokens for Qwen3-0.6B."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "verl"))

import datasets
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
MAX_LENGTH = 1024
INPUT_DIR = "data/ruletaker"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

for split, fname in [("train", "train_v2.parquet"), ("val", "val_v2.parquet")]:
    path = os.path.join(INPUT_DIR, fname)
    if not os.path.exists(path):
        print(f"SKIP {path} — not found")
        continue

    ds = datasets.load_dataset("parquet", data_files=path)["train"]
    print(f"{split}: loaded {len(ds)} examples")

    def token_len(example):
        prompt = example["prompt"]  # list of {role, content}
        text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        return {"n_tokens": len(tokenizer(text, add_special_tokens=False)["input_ids"])}

    n_proc = min(16, os.cpu_count() or 4)
    ds = ds.map(token_len, num_proc=n_proc, desc=f"Tokenizing {split}")

    ds_filtered = ds.filter(lambda x: x["n_tokens"] <= MAX_LENGTH,
                            num_proc=n_proc, desc=f"Filtering {split}")
    ds_filtered = ds_filtered.remove_columns(["n_tokens"])

    out_path = os.path.join(INPUT_DIR, f"{split}_filtered.parquet")
    ds_filtered.to_parquet(out_path)
    print(f"{split}: {len(ds_filtered)} / {len(ds)} kept → {out_path}")
    print()
