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
import re
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

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_jsonl(path, rows):
    if not rows:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_csv(path, rows, fieldnames):
    if not rows:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def save_state(path, state):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def load_state(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_hidden_states_for_batch_item(outputs, batch_idx):
    """
    Extract per-token hidden states for one item in a batched generation output.
    """
    hidden_states_list = []
    if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
        return hidden_states_list

    for step_hidden_states in outputs.hidden_states:
        # Keep only the current batch item and the most recent token position.
        extracted_layers = [layer[batch_idx:batch_idx + 1, -1:, :] for layer in step_hidden_states]
        hidden_states_list.append(extracted_layers)
    return hidden_states_list


def has_boxed_answer(text):
    return bool(re.search(r"\\boxed\{.+?\}", text))


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
    parser.add_argument("--sample-batch-size", type=int, default=1, help="Number of samples generated in parallel per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--threshold-g", type=float, default=0.60)
    parser.add_argument("--depth-rho", type=float, default=0.85)
    parser.add_argument("--min-dtr", type=float, default=0.32)
    parser.add_argument("--keep-per-q", type=int, default=4)
    parser.add_argument("--require-correct", action="store_true", help="Keep only traces with correct final answer")
    parser.add_argument("--require-boxed", action="store_true", help="Keep only traces that include a boxed final answer")
    parser.add_argument("--exclude-truncated", action="store_true", help="Exclude traces that hit max_new_tokens")
    parser.add_argument("--fallback-best-correct", action="store_true", help="If no trace passes filters, keep best correct trace")
    parser.add_argument("--dtr-max-tokens", type=int, default=0, help="If >0, compute DTR only on the first N generated tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--output", type=str, default="data/dtr_filtered_sft.jsonl")
    parser.add_argument("--candidates-out", type=str, default="data/dtr_candidates_debug.csv")
    parser.add_argument("--state-path", type=str, default="", help="Checkpoint file path (defaults to <output>.state.json)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint and existing output files")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing outputs/state and start fresh")
    parser.add_argument("--log-every", type=int, default=1, help="Print sample progress every N generations")
    args = parser.parse_args()

    if args.dataset == "gsm8k" and args.dataset_config == "main":
        dataset_config = "main"
    elif args.dataset == "gsm8k":
        dataset_config = args.dataset_config
    else:
        dataset_config = args.dataset_config if args.dataset_config else None

    if args.log_every < 1:
        raise ValueError("--log-every must be >= 1")
    if args.sample_batch_size < 1:
        raise ValueError("--sample-batch-size must be >= 1")
    if args.dtr_max_tokens < 0:
        raise ValueError("--dtr-max-tokens must be >= 0")

    state_path = args.state_path or f"{args.output}.state.json"

    if args.overwrite:
        for p in [args.output, args.candidates_out, state_path]:
            if os.path.exists(p):
                os.remove(p)
        print("Removed existing output/state files (overwrite mode).")

    if (not args.resume) and (not args.overwrite):
        existing = [p for p in [args.output, args.candidates_out, state_path] if os.path.exists(p)]
        if existing:
            raise FileExistsError(
                "Existing output files found. Use --resume to continue or --overwrite to restart:\n"
                + "\n".join(existing)
            )

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

    kept_rows = load_jsonl(args.output) if args.resume else []
    candidate_rows = load_csv(args.candidates_out) if args.resume else []

    # Normalize resumed numeric fields for summary computations.
    for row in candidate_rows:
        if isinstance(row.get("dtr"), str):
            row["dtr"] = float(row["dtr"])
        if isinstance(row.get("length_tokens"), str):
            row["length_tokens"] = int(row["length_tokens"])
        if isinstance(row.get("correct"), str):
            row["correct"] = int(row["correct"])
        if isinstance(row.get("question_idx"), str):
            row["question_idx"] = int(row["question_idx"])
        if isinstance(row.get("sample_idx"), str):
            row["sample_idx"] = int(row["sample_idx"])
        if isinstance(row.get("has_boxed"), str):
            row["has_boxed"] = int(row["has_boxed"])
        if isinstance(row.get("truncated"), str):
            row["truncated"] = int(row["truncated"])

    for row in kept_rows:
        if isinstance(row.get("dtr"), str):
            row["dtr"] = float(row["dtr"])
        if isinstance(row.get("length_tokens"), str):
            row["length_tokens"] = int(row["length_tokens"])
        if isinstance(row.get("correct"), str):
            row["correct"] = int(row["correct"])
        if isinstance(row.get("question_idx"), str):
            row["question_idx"] = int(row["question_idx"])
        if isinstance(row.get("sample_idx"), str):
            row["sample_idx"] = int(row["sample_idx"])
        if isinstance(row.get("has_boxed"), str):
            row["has_boxed"] = int(row["has_boxed"])
        if isinstance(row.get("truncated"), str):
            row["truncated"] = int(row["truncated"])

    state = load_state(state_path) if args.resume else None
    completed_questions = set(state.get("completed_questions", [])) if state else set()
    if args.resume:
        print(f"Resume mode: {len(completed_questions)} completed questions already recorded.")
        print(f"Loaded {len(candidate_rows)} existing candidates and {len(kept_rows)} kept rows.")

    total_to_run = len(samples) * args.samples_per_q
    done = 0

    for q_idx, sample in enumerate(samples):
        if q_idx in completed_questions:
            done += args.samples_per_q
            print(f"[{done:5d}/{total_to_run}] q={q_idx:03d} skipped (already completed)")
            continue

        question = sample["question"]
        gold = normalize_answer(sample["answer"])
        prompt = build_prompt(question)
        prompt_formatted = dtr_model._format_prompt(prompt, system_prompt=args.system_prompt)
        prompt_length = dtr_model.tokenizer(prompt_formatted, return_tensors="pt").input_ids.shape[1]
        q_candidates = []
        q_candidate_debug_rows = []

        for s_start in range(0, args.samples_per_q, args.sample_batch_size):
            batch_n = min(args.sample_batch_size, args.samples_per_q - s_start)
            batch_seed = args.seed + (q_idx * 10000) + s_start
            set_seed(batch_seed)

            batch_text = [prompt_formatted] * batch_n
            model_inputs = dtr_model.tokenizer(
                batch_text,
                return_tensors="pt",
                padding=True,
            ).to(dtr_model.device)

            outputs = dtr_model.generate_from_model_inputs(
                model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                output_hidden_states=True,
            )

            for b_idx in range(batch_n):
                s_idx = s_start + b_idx
                done += 1
                run_seed = args.seed + (q_idx * 10000) + s_idx

                hidden_states_list = extract_hidden_states_for_batch_item(outputs, b_idx)
                dtr_value = float(
                    calculator.calculate_dtr_for_sequence(
                        hidden_states_list,
                        max_tokens=(args.dtr_max_tokens if args.dtr_max_tokens > 0 else None),
                    )
                )

                new_ids = outputs.sequences[b_idx][prompt_length:]
                completion = dtr_model.tokenizer.decode(new_ids, skip_special_tokens=True)
                pred = normalize_answer(extract_answer_from_text(completion))
                correct = int((pred is not None) and (gold is not None) and (pred == gold))
                length_tokens = int(new_ids.shape[0])
                has_boxed = int(has_boxed_answer(completion))
                truncated = int(length_tokens >= args.max_new_tokens)

                row = {
                    "question_idx": q_idx,
                    "sample_idx": s_idx,
                    "seed": run_seed,
                    "dtr": dtr_value,
                    "length_tokens": length_tokens,
                    "correct": correct,
                    "has_boxed": has_boxed,
                    "truncated": truncated,
                    "gold": gold if gold is not None else "",
                    "pred": pred if pred is not None else "",
                    "prompt": prompt,
                    "completion": completion,
                }
                q_candidates.append(row)
                q_candidate_debug_rows.append(
                    {
                        "question_idx": q_idx,
                        "sample_idx": s_idx,
                        "seed": run_seed,
                        "dtr": dtr_value,
                        "length_tokens": length_tokens,
                        "correct": correct,
                        "has_boxed": has_boxed,
                        "truncated": truncated,
                        "gold": row["gold"],
                        "pred": row["pred"],
                    }
                )
                if (s_idx + 1) % args.log_every == 0 or (s_idx + 1) == args.samples_per_q:
                    print(
                        f"[{done:5d}/{total_to_run}] q={q_idx:03d} s={s_idx:02d} "
                        f"dtr={dtr_value:.3f} len={length_tokens:3d} "
                        f"correct={correct} boxed={has_boxed} trunc={truncated}"
                    )

        # Apply filters and retain top candidates per question.
        filtered = [
            r for r in q_candidates
            if (r["dtr"] >= args.min_dtr)
            and ((not args.require_correct) or r["correct"] == 1)
            and ((not args.require_boxed) or r["has_boxed"] == 1)
            and ((not args.exclude_truncated) or r["truncated"] == 0)
        ]
        filtered.sort(key=lambda r: r["dtr"], reverse=True)

        if (not filtered) and args.fallback_best_correct:
            correct_only = [
                r for r in q_candidates
                if (r["correct"] == 1)
                and ((not args.require_boxed) or r["has_boxed"] == 1)
                and ((not args.exclude_truncated) or r["truncated"] == 0)
            ]
            correct_only.sort(key=lambda r: r["dtr"], reverse=True)
            filtered = correct_only[:1]

        keep = filtered[: max(0, args.keep_per_q)]

        question_kept_rows = []
        for r in keep:
            question_kept_rows.append(
                {
                    "prompt": r["prompt"],
                    "completion": r["completion"],
                    "dtr": r["dtr"],
                    "length_tokens": r["length_tokens"],
                    "correct": r["correct"],
                    "has_boxed": r["has_boxed"],
                    "truncated": r["truncated"],
                    "gold": r["gold"],
                    "pred": r["pred"],
                    "question_idx": r["question_idx"],
                    "sample_idx": r["sample_idx"],
                    "seed": r["seed"],
                }
            )

        # Persist question-level outputs atomically (for resume safety).
        append_csv(
            args.candidates_out,
            q_candidate_debug_rows,
            fieldnames=[
                "question_idx",
                "sample_idx",
                "seed",
                "dtr",
                "length_tokens",
                "correct",
                "has_boxed",
                "truncated",
                "gold",
                "pred",
            ],
        )
        append_jsonl(args.output, question_kept_rows)

        candidate_rows.extend(q_candidate_debug_rows)
        kept_rows.extend(question_kept_rows)

        completed_questions.add(q_idx)
        save_state(
            state_path,
            {
                "completed_questions": sorted(completed_questions),
                "questions_total": len(samples),
                "samples_per_q": args.samples_per_q,
                "output": args.output,
                "candidates_out": args.candidates_out,
            },
        )

        print(
            f"  -> kept {len(keep)}/{args.samples_per_q} "
            f"(min_dtr={args.min_dtr:.3f}, require_correct={args.require_correct}, "
            f"require_boxed={args.require_boxed}, exclude_truncated={args.exclude_truncated})"
        )

    all_dtrs = [float(r["dtr"]) for r in candidate_rows]
    kept_dtrs = [float(r["dtr"]) for r in kept_rows]
    kept_counts_by_q = {}
    for r in kept_rows:
        q = int(r["question_idx"])
        kept_counts_by_q[q] = kept_counts_by_q.get(q, 0) + 1

    mean_all_dtr = statistics.mean(all_dtrs) if all_dtrs else float("nan")
    mean_kept_dtr = statistics.mean(kept_dtrs) if kept_dtrs else float("nan")
    keep_rate = (len(kept_rows) / len(candidate_rows)) if candidate_rows else 0.0
    mean_keep_per_q = (
        statistics.mean(kept_counts_by_q.values())
        if kept_counts_by_q else 0.0
    )
    kept_acc = (
        sum(int(r["correct"]) for r in kept_rows) / len(kept_rows)
        if kept_rows else 0.0
    )
    boxed_rate = (
        sum(int(r.get("has_boxed", 0)) for r in kept_rows) / len(kept_rows)
        if kept_rows else 0.0
    )
    kept_trunc_rate = (
        sum(int(r.get("truncated", 0)) for r in kept_rows) / len(kept_rows)
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
    print(f"Kept boxed-answer rate: {boxed_rate:.2%}")
    print(f"Kept truncated rate: {kept_trunc_rate:.2%}")
    print(f"SFT output: {args.output}")
    print(f"Debug candidates CSV: {args.candidates_out}")
    print(f"State checkpoint: {state_path}")


if __name__ == "__main__":
    main()
