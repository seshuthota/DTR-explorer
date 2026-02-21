"""
DTR vs Accuracy Correlation
===========================
Runs repeated samples per GSM8K question, then reports:
  - Pearson correlation: DTR vs correctness
  - Pearson correlation: token length vs correctness
  - Accuracy by DTR quintile bins
"""
import sys
import os
import re
import csv
import argparse
import random
import numpy as np
import torch
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtr.model import DTRModel
from dtr.calculator import DTRCalculator


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_answer(text):
    boxed = re.search(r"\\boxed\{(.+?)\}", text)
    if boxed:
        return boxed.group(1).strip()

    # Fallback: use the last number-looking span.
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if nums:
        return nums[-1].strip()
    return None


def normalize_num(s):
    if s is None:
        return None
    cleaned = s.replace(",", "").strip()
    cleaned = cleaned.strip("$")
    return cleaned


def pearson(x, y):
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size == 0 or y_arr.size == 0:
        return np.nan
    if np.std(x_arr) == 0.0 or np.std(y_arr) == 0.0:
        return np.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=20, help="Number of GSM8K test questions")
    parser.add_argument("--samples-per-question", type=int, default=8, help="Samples per question")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k for sampling")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Max new tokens per sample")
    parser.add_argument("--threshold-g", type=float, default=0.60, help="DTR settling threshold g")
    parser.add_argument("--depth-rho", type=float, default=0.85, help="Deep-thinking depth fraction rho")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--csv-out", type=str, default="", help="Optional CSV output path")
    args = parser.parse_args()

    print("Initializing model...")
    dtr_model = DTRModel()
    calculator = DTRCalculator(
        lm_head=dtr_model.lm_head,
        final_norm=dtr_model.final_norm,
        threshold_g=args.threshold_g,
        depth_fraction_rho=args.depth_rho,
    )

    print(f"Loading GSM8K test split (first {args.questions} questions)...")
    dataset = load_dataset("gsm8k", "main", split="test").select(range(args.questions))

    rows = []
    total = args.questions * args.samples_per_question
    idx = 0

    for q_idx, sample in enumerate(dataset):
        question = sample["question"]
        gold = normalize_num(sample["answer"].split("####")[-1])
        prompt = (
            "Please reason step by step, and put your final numerical answer within \\boxed{}. "
            f"Question: {question}"
        )

        for s_idx in range(args.samples_per_question):
            idx += 1
            run_seed = args.seed + (q_idx * 1000) + s_idx
            set_seed(run_seed)

            outputs, prompt_length, _ = dtr_model.generate_with_hidden_states(
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_k=args.top_k,
                return_prompt_metadata=True,
            )

            hidden_states_list = dtr_model.extract_generated_hidden_states(outputs)
            dtr_value = calculator.calculate_dtr_for_sequence(hidden_states_list)

            new_tokens = outputs.sequences[0][prompt_length:]
            decoded = dtr_model.tokenizer.decode(new_tokens, skip_special_tokens=True)
            pred = normalize_num(extract_answer(decoded))
            correct = int(pred == gold)

            rows.append(
                {
                    "q_idx": q_idx,
                    "sample_idx": s_idx,
                    "seed": run_seed,
                    "dtr": float(dtr_value),
                    "length": int(len(hidden_states_list)),
                    "correct": correct,
                    "pred": pred if pred is not None else "",
                    "gold": gold if gold is not None else "",
                }
            )
            print(f"[{idx:4d}/{total}] q={q_idx:02d} s={s_idx:02d} dtr={dtr_value:.3f} len={len(hidden_states_list):3d} correct={correct}")

    dtr_vals = [r["dtr"] for r in rows]
    len_vals = [r["length"] for r in rows]
    y_vals = [r["correct"] for r in rows]

    corr_dtr = pearson(dtr_vals, y_vals)
    corr_len = pearson(len_vals, y_vals)
    corr_rev_len = pearson([-x for x in len_vals], y_vals)
    acc = float(np.mean(y_vals)) if y_vals else float("nan")

    print("\n=== Correlation Summary ===")
    print(f"Samples: {len(rows)}")
    print(f"Overall accuracy: {acc:.3f}")
    print(f"Pearson(DTR, correctness): {corr_dtr:.3f}")
    print(f"Pearson(length, correctness): {corr_len:.3f}")
    print(f"Pearson(-length, correctness): {corr_rev_len:.3f}")

    print("\n=== DTR Quintile Bins ===")
    quantiles = np.quantile(dtr_vals, [0.2, 0.4, 0.6, 0.8])
    edges = [-np.inf, quantiles[0], quantiles[1], quantiles[2], quantiles[3], np.inf]
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        in_bin = [r for r in rows if lo < r["dtr"] <= hi] if i > 0 else [r for r in rows if lo <= r["dtr"] <= hi]
        if not in_bin:
            print(f"Bin {i+1}: empty")
            continue
        bin_acc = float(np.mean([r["correct"] for r in in_bin]))
        print(f"Bin {i+1}: n={len(in_bin):3d}, dtr_range=({lo:.3f}, {hi:.3f}], acc={bin_acc:.3f}")

    if args.csv_out:
        out_dir = os.path.dirname(args.csv_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved sample-level outputs to: {args.csv_out}")


if __name__ == "__main__":
    main()
