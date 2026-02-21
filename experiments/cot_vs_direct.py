"""
Prompt Shuffling Control Experiment
====================================
Tests the same GSM8K questions with two different prompt styles:
  A) Chain-of-thought: "Solve step by step... \\boxed{}"
  B) Direct answer:    "Give only the final answer in \\boxed{}"

If DTR measures intrinsic task difficulty, Math should remain higher than Easy
regardless of whether chain-of-thought verbosity is requested.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from datasets import load_dataset
from dtr.model import DTRModel
from dtr.calculator import DTRCalculator
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_condition(dtr_model, calculator, condition_name, prompts, max_new_tokens=150):
    all_dtrs = []
    all_depths = []
    
    print(f"\n--- {condition_name} ({len(prompts)} prompts) ---")
    
    for i, prompt in enumerate(prompts):
        set_seed(42 + i)
        
        outputs = dtr_model.generate_with_hidden_states(
            prompt,
            max_new_tokens=max_new_tokens,
            system_prompt="You are a helpful assistant."
        )
        
        hidden_states_list = dtr_model.extract_generated_hidden_states(outputs)
                
        dtr_value, depths = calculator.calculate_dtr_for_sequence(hidden_states_list, return_depths=True)
        all_dtrs.append(dtr_value)
        all_depths.extend(depths)
        
    mean_dtr = np.mean(all_dtrs)
    std_dtr = np.std(all_dtrs)
    median_depth = np.median(all_depths)
    p25 = np.percentile(all_depths, 25)
    p75 = np.percentile(all_depths, 75)
    iqr = p75 - p25
    
    print(f"  DTR: {mean_dtr:.2%} ± {std_dtr:.2%}")
    print(f"  Median Settling Depth: {median_depth:.1f} [IQR: {p25:.0f}–{p75:.0f}, width={iqr:.1f}]")
    print(f"  Total tokens analyzed: {len(all_depths)}")
    
    return mean_dtr, std_dtr, median_depth, iqr

def main():
    print("Initializing Model...")
    dtr_model = DTRModel()
    
    calculator = DTRCalculator(
        lm_head=dtr_model.lm_head, 
        final_norm=dtr_model.final_norm,
        threshold_g=0.60, 
        depth_fraction_rho=0.85
    )
    
    print("Loading GSM8K...")
    dataset = load_dataset("gsm8k", "main", split="test")
    samples = dataset.select(range(20))
    
    # Build two prompt variants for the same questions
    cot_prompts = [
        f"Please reason step by step, and put your final numerical answer within \\boxed{{}}. Question: {s['question']}"
        for s in samples
    ]
    direct_prompts = [
        f"Give only the final numerical answer in \\boxed{{}}. Do not show any work. Question: {s['question']}"
        for s in samples
    ]
    
    # Easy baseline
    easy_prompts = [
        "Write a haiku about a cat.", "What is the capital of France?",
        "Explain how a rainbow is formed.", "Write a short polite email declining a meeting.",
        "List three common types of apples.", "What color is the sky?",
        "Who wrote Harry Potter?", "Write a 3 sentence story about a knight.",
        "What is the chemical symbol for water?", "Write a joke about a programmer.",
        "What is 10 + 10?", "Name the planets in our solar system.",
        "How many legs does a spider have?", "Translate 'Hello' to Spanish.",
        "What is the largest ocean?", "Write a poem about the moon.",
        "What is the powerhouse of the cell?", "Name a primary color.",
        "What sound does a cow make?", "Summarize Romeo and Juliet in one sentence."
    ]
    
    # Run all three conditions
    easy_res = run_condition(dtr_model, calculator, "Easy Text (Baseline)", easy_prompts)
    cot_res = run_condition(dtr_model, calculator, "Math + Chain-of-Thought", cot_prompts)
    direct_res = run_condition(dtr_model, calculator, "Math + Direct Answer Only", direct_prompts)
    
    # Effect sizes
    delta_cot = cot_res[0] - easy_res[0]
    delta_direct = direct_res[0] - easy_res[0]
    
    print("\n\n=== FINAL COMPARISON ===")
    print(f"{'Condition':<30} | {'DTR':>12} | {'Median Depth':>12} | {'IQR':>5} | {'Δ vs Easy':>10}")
    print("-" * 80)
    print(f"{'Easy Text':<30} | {easy_res[0]:>10.2%}   | {easy_res[2]:>10.1f}   | {easy_res[3]:>5.1f} | {'—':>10}")
    print(f"{'Math + CoT':<30} | {cot_res[0]:>10.2%}   | {cot_res[2]:>10.1f}   | {cot_res[3]:>5.1f} | {delta_cot:>+9.2%}")
    print(f"{'Math + Direct Answer':<30} | {direct_res[0]:>10.2%}   | {direct_res[2]:>10.1f}   | {direct_res[3]:>5.1f} | {delta_direct:>+9.2%}")

if __name__ == "__main__":
    main()
