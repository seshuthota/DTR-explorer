import json
import random
import re

import numpy as np
import torch
from datasets import load_dataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_answer_from_text(text):
    # Primary rule used across this repo.
    boxed = re.search(r"\\boxed\{(.+?)\}", text)
    if boxed:
        return boxed.group(1).strip()

    # Fallback for non-boxed outputs.
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if nums:
        return nums[-1].strip()
    return None


def normalize_answer(value):
    if value is None:
        return None
    text = str(value).strip()
    # GSM8K canonical answer format often ends with "#### 42"
    if "####" in text:
        text = text.split("####")[-1].strip()
    text = text.replace(",", "")
    text = text.strip("$")
    return text


def majority_vote(predictions):
    counts = {}
    for p in predictions:
        if p is None or p == "":
            continue
        counts[p] = counts.get(p, 0) + 1
    if not counts:
        return None
    # Stable tie-break: highest count, then lexical.
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def load_qa_samples(
    dataset="gsm8k",
    split="test",
    limit=20,
    dataset_jsonl="",
    question_field="question",
    answer_field="answer",
    dataset_config=None,
):
    if dataset_jsonl:
        records = []
        with open(dataset_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if question_field not in row or answer_field not in row:
                    raise KeyError(
                        f"JSONL row missing required fields: "
                        f"'{question_field}', '{answer_field}'"
                    )
                records.append(
                    {
                        "question": str(row[question_field]),
                        "answer": normalize_answer(row[answer_field]),
                    }
                )
        return records[:limit]

    if dataset == "gsm8k":
        ds = load_dataset("gsm8k", "main", split=split)
        selected = ds.select(range(limit))
        return [
            {
                "question": str(sample["question"]),
                "answer": normalize_answer(sample["answer"]),
            }
            for sample in selected
        ]

    # Generic Hugging Face dataset mode for AIME/custom sets.
    if dataset_config:
        ds = load_dataset(dataset, dataset_config, split=split)
    else:
        ds = load_dataset(dataset, split=split)
    selected = ds.select(range(limit))
    return [
        {
            "question": str(sample[question_field]),
            "answer": normalize_answer(sample[answer_field]),
        }
        for sample in selected
    ]
