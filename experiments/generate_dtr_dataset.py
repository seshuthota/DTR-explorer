"""
Generate DTR-Filtered SFT Dataset
=================================
Builds an SFT dataset by sampling multiple responses per question, scoring each
with DTR, and keeping high-DTR traces (optionally requiring correctness).
"""
import os
import sys
import json
import csv
import argparse
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtr.model import DTRModel
from dtr.calculator import DTRCalculator
from experiments.common import (
    set_seed,
    extract_answer_from_text,
    normalize_answer,
    load_qa_samples,
)


def build_prompt(question):
    return (
        "Please reason step by step, and put your final numerical answer within "
        f"\\boxed{{}}. Question: {question}"
    )


def save_jsonl(path, rows):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path, rows):
    if not rows:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gsm8k", help="HF dataset name (or gsm8k)")
    parser.add_argument("--dataset-config", type=str, default="main", help="HF dataset config (ignored for JSONL)")
    parser.add_argument("--dataset-jsonl", type=str, default="", help="Optional JSONL dataset path")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--question-field", type=str, default="question")
    parser.add_argument("--answer-field", type=str, default="answer")
    parser.add_argument("--questions", type=int, default=200)
    parser.add_argument("--samples-per-q", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--threshold-g", type=float, default=0.60)
    parser.add_argument("--depth-rho", type=float, default=0.85)
    parser.add_argument("--min-dtr", type=float, default=0.32)
    parser.add_argument("--keep-per-q", type=int, default=4)
    parser.add_argument("--require-correct", action="store_true", help="Keep only traces with correct final answer")
    parser.add_argument("--fallback-best-correct", action="store_true", help="If no trace passes filters, keep best correct trace")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--output", type=str, default="data/dtr_filtered_sft.jsonl")
    parser.add_argument("--candidates-out", type=str, default="data/dtr_candidates_debug.csv")
    args = parser.parse_args()

    if args.dataset == "gsm8k" and args.dataset_config == "main":
        dataset_config = "main"
    elif args.dataset == "gsm8k":
        dataset_config = args.dataset_config
    else:
        dataset_config = args.dataset_config if args.dataset_config else None

    print("Loading source questions...")
    samples = load_qa_samples(
        dataset=args.dataset,
        split=args.split,
        limit=args.questions,
        dataset_jsonl=args.dataset_jsonl,
        question_field=args.question_field,
        answer_field=args.answer_field,
        dataset_config=dataset_config,
    )
    print(f"Loaded {len(samples)} questions.")

    print("Initializing model + DTR calculator...")
    dtr_model = DTRModel()
    calculator = DTRCalculator(
        lm_head=dtr_model.lm_head,
        final_norm=dtr_model.final_norm,
        threshold_g=args.threshold_g,
        depth_fraction_rho=args.depth_rho,
    )

    kept_rows = []
    candidate_rows = []
    all_dtrs = []
    kept_dtrs = []
    kept_counts = []

    total_to_run = len(samples) * args.samples_per_q
    done = 0

    for q_idx, sample in enumerate(samples):
        question = sample["question"]
        gold = normalize_answer(sample["answer"])
        prompt = build_prompt(question)
        q_candidates = []

        for s_idx in range(args.samples_per_q):
            done += 1
            run_seed = args.seed + (q_idx * 10000) + s_idx
            set_seed(run_seed)

            outputs, prompt_length, _ = dtr_model.generate_with_hidden_states(
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                system_prompt=args.system_prompt,
                return_prompt_metadata=True,
                output_hidden_states=True,
            )

            hidden_states_list = dtr_model.extract_generated_hidden_states(outputs)
            dtr_value = float(calculator.calculate_dtr_for_sequence(hidden_states_list))
            all_dtrs.append(dtr_value)

            new_ids = outputs.sequences[0][prompt_length:]
            completion = dtr_model.tokenizer.decode(new_ids, skip_special_tokens=True)
            pred = normalize_answer(extract_answer_from_text(completion))
            correct = int((pred is not None) and (gold is not None) and (pred == gold))
            length_tokens = int(new_ids.shape[0])

            row = {
                "question_idx": q_idx,
                "sample_idx": s_idx,
                "seed": run_seed,
                "dtr": dtr_value,
                "length_tokens": length_tokens,
                "correct": correct,
                "gold": gold if gold is not None else "",
                "pred": pred if pred is not None else "",
                "prompt": prompt,
                "completion": completion,
            }
            q_candidates.append(row)
            candidate_rows.append(
                {
                    "question_idx": q_idx,
                    "sample_idx": s_idx,
                    "seed": run_seed,
                    "dtr": dtr_value,
                    "length_tokens": length_tokens,
                    "correct": correct,
                    "gold": row["gold"],
                    "pred": row["pred"],
                }
            )
            print(
                f"[{done:5d}/{total_to_run}] q={q_idx:03d} s={s_idx:02d} "
                f"dtr={dtr_value:.3f} len={length_tokens:3d} correct={correct}"
            )

        # Apply filters and retain top candidates per question.
        filtered = [
            r for r in q_candidates
            if (r["dtr"] >= args.min_dtr) and ((not args.require_correct) or r["correct"] == 1)
        ]
        filtered.sort(key=lambda r: r["dtr"], reverse=True)

        if (not filtered) and args.fallback_best_correct:
            correct_only = [r for r in q_candidates if r["correct"] == 1]
            correct_only.sort(key=lambda r: r["dtr"], reverse=True)
            filtered = correct_only[:1]

        keep = filtered[: max(0, args.keep_per_q)]
        kept_counts.append(len(keep))

        for r in keep:
            kept_dtrs.append(r["dtr"])
            kept_rows.append(
                {
                    "prompt": r["prompt"],
                    "completion": r["completion"],
                    "dtr": r["dtr"],
                    "length_tokens": r["length_tokens"],
                    "correct": r["correct"],
                    "gold": r["gold"],
                    "pred": r["pred"],
                    "question_idx": r["question_idx"],
                    "sample_idx": r["sample_idx"],
                    "seed": r["seed"],
                }
            )

        print(
            f"  -> kept {len(keep)}/{args.samples_per_q} "
            f"(min_dtr={args.min_dtr:.3f}, require_correct={args.require_correct})"
        )

    save_jsonl(args.output, kept_rows)
    save_csv(args.candidates_out, candidate_rows)

    mean_all_dtr = statistics.mean(all_dtrs) if all_dtrs else float("nan")
    mean_kept_dtr = statistics.mean(kept_dtrs) if kept_dtrs else float("nan")
    keep_rate = (len(kept_rows) / len(candidate_rows)) if candidate_rows else 0.0
    mean_keep_per_q = statistics.mean(kept_counts) if kept_counts else 0.0
    kept_acc = (
        sum(int(r["correct"]) for r in kept_rows) / len(kept_rows)
        if kept_rows else 0.0
    )

    print("\n=== DTR Dataset Summary ===")
    print(f"Questions: {len(samples)}")
    print(f"Generated candidates: {len(candidate_rows)}")
    print(f"Kept completions: {len(kept_rows)}")
    print(f"Keep rate: {keep_rate:.2%}")
    print(f"Mean kept per question: {mean_keep_per_q:.2f}")
    print(f"Mean DTR (all): {mean_all_dtr:.3f}")
    print(f"Mean DTR (kept): {mean_kept_dtr:.3f}")
    print(f"Kept correctness rate: {kept_acc:.2%}")
    print(f"SFT output: {args.output}")
    print(f"Debug candidates CSV: {args.candidates_out}")


if __name__ == "__main__":
    main()
