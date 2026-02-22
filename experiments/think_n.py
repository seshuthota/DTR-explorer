"""
Think@n Experiment
==================
Implements DTR-based early pruning:
1) Generate N short prefixes.
2) Rank prefixes by DTR.
3) Early-stop bottom samples.
4) Continue top samples and majority-vote.

Compares against full self-consistency (Cons@n) with token-cost accounting.
"""
import os
import sys
import csv
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtr.model import DTRModel
from dtr.calculator import DTRCalculator
from experiments.common import (
    set_seed,
    extract_answer_from_text,
    normalize_answer,
    majority_vote,
    load_qa_samples,
)


def build_prompt(question):
    return (
        "Please reason step by step, and put your final numerical answer within "
        f"\\boxed{{}}. Question: {question}"
    )


def decode_prediction(tokenizer, token_ids):
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return normalize_answer(extract_answer_from_text(text)), text


def run_full_consistency(
    dtr_model,
    prompt,
    n,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    seed_base,
    system_prompt,
    return_traces=False,
):
    predictions = []
    generated_lengths = []
    traces = [] if return_traces else None

    for i in range(n):
        set_seed(seed_base + i)
        outputs, prompt_length, _ = dtr_model.generate_with_hidden_states(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            system_prompt=system_prompt,
            return_prompt_metadata=True,
            output_hidden_states=False,
        )
        new_ids = outputs.sequences[0][prompt_length:]
        pred, text = decode_prediction(dtr_model.tokenizer, new_ids)
        predictions.append(pred)
        generated_lengths.append(int(new_ids.shape[0]))
        if return_traces:
            traces.append(
                {
                    "idx": i,
                    "seed": seed_base + i,
                    "prediction": pred if pred is not None else "",
                    "length_tokens": int(new_ids.shape[0]),
                    "text": text,
                }
            )

    vote = majority_vote(predictions)
    result = {
        "vote": vote,
        "predictions": predictions,
        "token_cost": int(sum(generated_lengths)),
        "mean_len": float(np.mean(generated_lengths)) if generated_lengths else 0.0,
    }
    if return_traces:
        result["traces"] = traces
    return result


def run_think_n(
    dtr_model,
    calculator,
    prompt,
    n,
    keep_ratio,
    prefix_tokens,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    seed_base,
    system_prompt,
    return_traces=False,
):
    inputs, prompt_len, _ = dtr_model.prepare_inputs(prompt, system_prompt=system_prompt)
    prompt_ids = inputs["input_ids"]
    prompt_attn = inputs.get("attention_mask", None)

    prefix_rows = []
    prefix_token_cost = 0

    # Stage 1: generate short prefixes for all N candidates.
    for i in range(n):
        set_seed(seed_base + i)
        model_inputs = {"input_ids": prompt_ids}
        if prompt_attn is not None:
            model_inputs["attention_mask"] = prompt_attn

        prefix_outputs = dtr_model.generate_from_model_inputs(
            model_inputs,
            max_new_tokens=prefix_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            output_hidden_states=True,
        )

        prefix_ids = prefix_outputs.sequences[0][prompt_len:].detach().cpu()
        prefix_len = int(prefix_ids.shape[0])
        prefix_token_cost += prefix_len

        hidden_states_list = dtr_model.extract_generated_hidden_states(prefix_outputs)
        prefix_dtr = float(calculator.calculate_dtr_for_sequence(hidden_states_list))

        prefix_rows.append(
            {
                "idx": i,
                "seed": seed_base + i,
                "prefix_ids": prefix_ids,
                "prefix_len": prefix_len,
                "prefix_dtr": prefix_dtr,
            }
        )

    keep_n = max(1, int(round(n * keep_ratio)))
    keep_n = min(keep_n, n)
    selected = sorted(prefix_rows, key=lambda x: x["prefix_dtr"], reverse=True)[:keep_n]
    selected_idx_set = {r["idx"] for r in selected}

    # Stage 2: continue only selected candidates to full budget.
    continued_predictions = []
    continuation_token_cost = 0
    final_lengths = []
    selected_traces = [] if return_traces else None

    for row in selected:
        prefix_ids = row["prefix_ids"].to(prompt_ids.device)
        prefix_len = int(row["prefix_len"])
        remaining = max(0, max_new_tokens - prefix_len)

        continuation_ids = torch.empty(0, dtype=torch.long, device=prompt_ids.device)
        if remaining > 0:
            context_ids = torch.cat([prompt_ids[0], prefix_ids], dim=0).unsqueeze(0)
            context_inputs = {
                "input_ids": context_ids,
                "attention_mask": torch.ones_like(context_ids),
            }
            # Offset seed so continuation randomness is stable and independent.
            set_seed(seed_base + 100000 + row["idx"])
            cont_outputs = dtr_model.generate_from_model_inputs(
                context_inputs,
                max_new_tokens=remaining,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                output_hidden_states=False,
            )
            continuation_ids = cont_outputs.sequences[0][context_ids.shape[1]:]

        final_ids = torch.cat([prefix_ids, continuation_ids], dim=0)
        pred, text = decode_prediction(dtr_model.tokenizer, final_ids)
        continued_predictions.append(pred)
        continuation_token_cost += int(continuation_ids.shape[0])
        final_lengths.append(int(final_ids.shape[0]))
        if return_traces:
            selected_traces.append(
                {
                    "idx": int(row["idx"]),
                    "seed": int(row["seed"]),
                    "prefix_dtr": float(row["prefix_dtr"]),
                    "prediction": pred if pred is not None else "",
                    "length_tokens": int(final_ids.shape[0]),
                    "text": text,
                }
            )

    vote = majority_vote(continued_predictions)
    result = {
        "vote": vote,
        "selected_n": keep_n,
        "prefix_cost": int(prefix_token_cost),
        "continuation_cost": int(continuation_token_cost),
        "token_cost": int(prefix_token_cost + continuation_token_cost),
        "mean_selected_prefix_dtr": float(np.mean([r["prefix_dtr"] for r in selected])) if selected else 0.0,
        "mean_len": float(np.mean(final_lengths)) if final_lengths else 0.0,
    }
    if return_traces:
        prefix_traces = []
        for r in prefix_rows:
            prefix_traces.append(
                {
                    "idx": int(r["idx"]),
                    "seed": int(r["seed"]),
                    "prefix_len": int(r["prefix_len"]),
                    "prefix_dtr": float(r["prefix_dtr"]),
                    "selected": int(r["idx"] in selected_idx_set),
                }
            )
        result["prefix_traces"] = prefix_traces
        result["selected_traces"] = selected_traces
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gsm8k", help="HF dataset name (or gsm8k)")
    parser.add_argument("--dataset-config", type=str, default="main", help="HF dataset config (ignored for JSONL)")
    parser.add_argument("--dataset-jsonl", type=str, default="", help="Optional JSONL dataset path")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--question-field", type=str, default="question")
    parser.add_argument("--answer-field", type=str, default="answer")
    parser.add_argument("--questions", type=int, default=30)
    parser.add_argument("--question-offset", type=int, default=0, help="Start index within selected split")
    parser.add_argument("--n", type=int, default=24, help="Samples for Cons@n / Think@n")
    parser.add_argument("--keep-ratio", type=float, default=0.5, help="Fraction kept after prefix ranking")
    parser.add_argument("--prefix-tokens", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--threshold-g", type=float, default=0.60)
    parser.add_argument("--depth-rho", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--csv-out", type=str, default="")
    parser.add_argument("--trace-jsonl", type=str, default="", help="Optional per-question detailed trace output")
    args = parser.parse_args()

    if args.dataset == "gsm8k" and args.dataset_config == "main":
        dataset_config = "main"
    elif args.dataset == "gsm8k":
        dataset_config = args.dataset_config
    else:
        dataset_config = args.dataset_config if args.dataset_config else None

    print("Loading samples...")
    if args.question_offset < 0:
        raise ValueError("--question-offset must be >= 0")

    raw_samples = load_qa_samples(
        dataset=args.dataset,
        split=args.split,
        limit=args.question_offset + args.questions,
        dataset_jsonl=args.dataset_jsonl,
        question_field=args.question_field,
        answer_field=args.answer_field,
        dataset_config=dataset_config,
    )
    samples = raw_samples[args.question_offset: args.question_offset + args.questions]
    print(f"Loaded {len(samples)} questions.")

    print("Initializing model and DTR calculator...")
    dtr_model = DTRModel()
    calculator = DTRCalculator(
        lm_head=dtr_model.lm_head,
        final_norm=dtr_model.final_norm,
        threshold_g=args.threshold_g,
        depth_fraction_rho=args.depth_rho,
    )

    rows = []
    trace_rows = []
    cons_correct = []
    think_correct = []
    cons_costs = []
    think_costs = []

    for local_idx, sample in enumerate(samples):
        q_idx = args.question_offset + local_idx
        question = sample["question"]
        gold = normalize_answer(sample["answer"])
        prompt = build_prompt(question)
        question_seed = args.seed + (q_idx * 10000)

        cons = run_full_consistency(
            dtr_model=dtr_model,
            prompt=prompt,
            n=args.n,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed_base=question_seed,
            system_prompt=args.system_prompt,
            return_traces=bool(args.trace_jsonl),
        )

        think = run_think_n(
            dtr_model=dtr_model,
            calculator=calculator,
            prompt=prompt,
            n=args.n,
            keep_ratio=args.keep_ratio,
            prefix_tokens=args.prefix_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed_base=question_seed,
            system_prompt=args.system_prompt,
            return_traces=bool(args.trace_jsonl),
        )

        cons_ok = int(cons["vote"] == gold)
        think_ok = int(think["vote"] == gold)
        cons_cost = int(cons["token_cost"])
        think_cost = int(think["token_cost"])
        savings = 1.0 - (think_cost / cons_cost) if cons_cost > 0 else float("nan")

        cons_correct.append(cons_ok)
        think_correct.append(think_ok)
        cons_costs.append(cons_cost)
        think_costs.append(think_cost)

        print(
            f"[Q{q_idx:03d}] Cons@n={cons_ok} Think@n={think_ok} "
            f"cost_cons={cons_cost} cost_think={think_cost} savings={savings:.2%}"
        )

        rows.append(
            {
                "question_idx": q_idx,
                "gold": gold if gold is not None else "",
                "cons_vote": cons["vote"] if cons["vote"] is not None else "",
                "think_vote": think["vote"] if think["vote"] is not None else "",
                "cons_correct": cons_ok,
                "think_correct": think_ok,
                "cons_token_cost": cons_cost,
                "think_token_cost": think_cost,
                "token_savings_ratio": savings,
                "selected_n": think["selected_n"],
                "prefix_cost": think["prefix_cost"],
                "continuation_cost": think["continuation_cost"],
                "mean_selected_prefix_dtr": think["mean_selected_prefix_dtr"],
            }
        )
        if args.trace_jsonl:
            trace_rows.append(
                {
                    "question_idx": q_idx,
                    "question": question,
                    "gold": gold if gold is not None else "",
                    "cons": {
                        "vote": cons["vote"] if cons["vote"] is not None else "",
                        "correct": cons_ok,
                        "token_cost": cons_cost,
                        "samples": cons.get("traces", []),
                    },
                    "think": {
                        "vote": think["vote"] if think["vote"] is not None else "",
                        "correct": think_ok,
                        "token_cost": think_cost,
                        "selected_n": think["selected_n"],
                        "prefix_cost": think["prefix_cost"],
                        "continuation_cost": think["continuation_cost"],
                        "mean_selected_prefix_dtr": think["mean_selected_prefix_dtr"],
                        "prefixes": think.get("prefix_traces", []),
                        "selected_samples": think.get("selected_traces", []),
                    },
                }
            )

    cons_acc = float(np.mean(cons_correct)) if cons_correct else float("nan")
    think_acc = float(np.mean(think_correct)) if think_correct else float("nan")
    mean_cons_cost = float(np.mean(cons_costs)) if cons_costs else float("nan")
    mean_think_cost = float(np.mean(think_costs)) if think_costs else float("nan")
    mean_savings = 1.0 - (mean_think_cost / mean_cons_cost) if mean_cons_cost > 0 else float("nan")

    print("\n=== Aggregate Results ===")
    print(f"Questions: {len(samples)}")
    print(f"Cons@{args.n} accuracy:  {cons_acc:.3f}")
    print(f"Think@{args.n} accuracy: {think_acc:.3f}")
    print(f"Accuracy delta (Think-Cons): {think_acc - cons_acc:+.3f}")
    print(f"Mean token cost Cons@{args.n}:  {mean_cons_cost:.1f}")
    print(f"Mean token cost Think@{args.n}: {mean_think_cost:.1f}")
    print(f"Mean token savings ratio: {mean_savings:.2%}")

    if args.csv_out:
        out_dir = os.path.dirname(args.csv_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved detailed rows to: {args.csv_out}")

    if args.trace_jsonl:
        out_dir = os.path.dirname(args.trace_jsonl)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.trace_jsonl, "w", encoding="utf-8") as f:
            for row in trace_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved trace rows to: {args.trace_jsonl}")


if __name__ == "__main__":
    main()
