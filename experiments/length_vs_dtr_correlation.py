"""
Length vs DTR Correlation
=========================
For each question, sample many responses and compute:
  - correlation(length, correctness)
  - correlation(DTR, correctness)
Then aggregate and plot binned accuracy for both signals.
"""
import os
import sys
import csv
import argparse
import numpy as np

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


def pearson(x, y):
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size == 0 or y_arr.size == 0:
        return np.nan
    if np.std(x_arr) == 0.0 or np.std(y_arr) == 0.0:
        return np.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def quantile_bin_stats(values, targets, bins=8):
    values = np.asarray(values, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if values.size == 0:
        return []

    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    # Enforce monotonic edges for stable binning when there are ties.
    edges = np.maximum.accumulate(edges)

    stats = []
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == 0:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values > lo) & (values <= hi)

        idx = np.where(mask)[0]
        if idx.size == 0:
            stats.append(
                {
                    "bin": i + 1,
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": 0,
                    "acc": np.nan,
                    "mean_x": np.nan,
                }
            )
            continue
        stats.append(
            {
                "bin": i + 1,
                "lo": float(lo),
                "hi": float(hi),
                "n": int(idx.size),
                "acc": float(np.mean(targets[idx])),
                "mean_x": float(np.mean(values[idx])),
            }
        )
    return stats


def maybe_plot(length_bins, dtr_bins, plot_out):
    if not plot_out:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting optional
        print(f"Skipping plot (matplotlib unavailable): {exc}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    lx = [b["mean_x"] for b in length_bins if not np.isnan(b["mean_x"])]
    ly = [b["acc"] for b in length_bins if not np.isnan(b["acc"])]
    dx = [b["mean_x"] for b in dtr_bins if not np.isnan(b["mean_x"])]
    dy = [b["acc"] for b in dtr_bins if not np.isnan(b["acc"])]

    axes[0].plot(lx, ly, marker="o")
    axes[0].set_title("Accuracy vs Length (binned)")
    axes[0].set_xlabel("Mean length in bin")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.3)

    axes[1].plot(dx, dy, marker="o")
    axes[1].set_title("Accuracy vs DTR (binned)")
    axes[1].set_xlabel("Mean DTR in bin")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_dir = os.path.dirname(plot_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(plot_out, dpi=150)
    print(f"Saved plot to: {plot_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gsm8k", help="HF dataset name (or gsm8k)")
    parser.add_argument("--dataset-config", type=str, default="main", help="HF dataset config (ignored for JSONL)")
    parser.add_argument("--dataset-jsonl", type=str, default="", help="Optional JSONL dataset path")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--question-field", type=str, default="question")
    parser.add_argument("--answer-field", type=str, default="answer")
    parser.add_argument("--questions", type=int, default=15)
    parser.add_argument("--samples-per-question", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--threshold-g", type=float, default=0.60)
    parser.add_argument("--depth-rho", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--csv-out", type=str, default="")
    parser.add_argument("--plot-out", type=str, default="")
    args = parser.parse_args()

    if args.dataset == "gsm8k" and args.dataset_config == "main":
        dataset_config = "main"
    elif args.dataset == "gsm8k":
        dataset_config = args.dataset_config
    else:
        dataset_config = args.dataset_config if args.dataset_config else None

    print("Loading samples...")
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

    print("Initializing model...")
    dtr_model = DTRModel()
    calculator = DTRCalculator(
        lm_head=dtr_model.lm_head,
        final_norm=dtr_model.final_norm,
        threshold_g=args.threshold_g,
        depth_fraction_rho=args.depth_rho,
    )

    all_rows = []
    q_corr_len = []
    q_corr_dtr = []
    total = len(samples) * args.samples_per_question
    done = 0

    for q_idx, sample in enumerate(samples):
        question = sample["question"]
        gold = normalize_answer(sample["answer"])
        prompt = build_prompt(question)

        lengths = []
        dtrs = []
        ys = []

        for s_idx in range(args.samples_per_question):
            done += 1
            run_seed = args.seed + q_idx * 10000 + s_idx
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

            new_ids = outputs.sequences[0][prompt_length:]
            length = int(new_ids.shape[0])
            text = dtr_model.tokenizer.decode(new_ids, skip_special_tokens=True)
            pred = normalize_answer(extract_answer_from_text(text))
            correct = int(pred == gold)

            lengths.append(length)
            dtrs.append(dtr_value)
            ys.append(correct)
            all_rows.append(
                {
                    "question_idx": q_idx,
                    "sample_idx": s_idx,
                    "seed": run_seed,
                    "length": length,
                    "dtr": dtr_value,
                    "correct": correct,
                    "gold": gold if gold is not None else "",
                    "pred": pred if pred is not None else "",
                }
            )
            print(
                f"[{done:4d}/{total}] q={q_idx:02d} s={s_idx:02d} "
                f"len={length:3d} dtr={dtr_value:.3f} correct={correct}"
            )

        q_corr_len.append(pearson(lengths, ys))
        q_corr_dtr.append(pearson(dtrs, ys))

    pooled_lengths = [r["length"] for r in all_rows]
    pooled_dtrs = [r["dtr"] for r in all_rows]
    pooled_y = [r["correct"] for r in all_rows]

    pooled_corr_len = pearson(pooled_lengths, pooled_y)
    pooled_corr_dtr = pearson(pooled_dtrs, pooled_y)

    finite_len = [v for v in q_corr_len if not np.isnan(v)]
    finite_dtr = [v for v in q_corr_dtr if not np.isnan(v)]
    mean_q_corr_len = float(np.mean(finite_len)) if finite_len else np.nan
    mean_q_corr_dtr = float(np.mean(finite_dtr)) if finite_dtr else np.nan

    length_bins = quantile_bin_stats(pooled_lengths, pooled_y, bins=args.bins)
    dtr_bins = quantile_bin_stats(pooled_dtrs, pooled_y, bins=args.bins)

    print("\n=== Correlation Summary ===")
    print(f"Total samples: {len(all_rows)}")
    print(f"Pooled corr(length, correct): {pooled_corr_len:.3f}")
    print(f"Pooled corr(DTR, correct):    {pooled_corr_dtr:.3f}")
    print(f"Mean question corr(length, correct): {mean_q_corr_len:.3f}")
    print(f"Mean question corr(DTR, correct):    {mean_q_corr_dtr:.3f}")

    print("\n=== Length Bins ===")
    for b in length_bins:
        print(
            f"bin={b['bin']:02d} n={b['n']:4d} "
            f"range=({b['lo']:.3f}, {b['hi']:.3f}] acc={b['acc']:.3f}"
        )

    print("\n=== DTR Bins ===")
    for b in dtr_bins:
        print(
            f"bin={b['bin']:02d} n={b['n']:4d} "
            f"range=({b['lo']:.3f}, {b['hi']:.3f}] acc={b['acc']:.3f}"
        )

    if args.csv_out:
        out_dir = os.path.dirname(args.csv_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved sample-level rows to: {args.csv_out}")

    maybe_plot(length_bins, dtr_bins, args.plot_out)


if __name__ == "__main__":
    main()
