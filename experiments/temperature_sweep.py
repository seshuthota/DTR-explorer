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

def run_experiment_on_dataset(dtr_model, calculator, dataset_name, prompts, temperature, do_sample, max_new_tokens=150):
    all_dtrs = []
    all_jsd_depths = []
    all_topk_depths = []
    
    print(f"\n--- Running {dataset_name} at T={temperature} (Sample={do_sample}) ---")
    
    for i, prompt in enumerate(prompts):
        set_seed(42 + i)  # Specific seed per prompt for consistency across conditions
        outputs = dtr_model.generate_with_hidden_states(
            prompt, 
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            system_prompt="You are a helpful assistant."
        )
        
        hidden_states_list = dtr_model.extract_generated_hidden_states(outputs)
                
        dtr_value, jsd_depths, topk_depths = calculator.calculate_dtr_for_sequence(hidden_states_list, return_depths=True, return_top_k_depths=True)
        
        all_dtrs.append(dtr_value)
        all_jsd_depths.extend(jsd_depths)
        all_topk_depths.extend(topk_depths)
        
    mean_dtr = np.mean(all_dtrs)
    std_dtr = np.std(all_dtrs)
    median_jsd_depth = np.median(all_jsd_depths)
    iqr_jsd_depth = np.percentile(all_jsd_depths, 75) - np.percentile(all_jsd_depths, 25)
    median_topk_depth = np.median(all_topk_depths)
    
    print(f"  DTR: {mean_dtr:.2%} ± {std_dtr:.2%}")
    print(f"  JSD Median Settling Depth:  {median_jsd_depth:.1f} (IQR: {iqr_jsd_depth:.1f})")
    print(f"  Top-K Median Settling Depth: {median_topk_depth:.1f}")
    
    return mean_dtr, median_jsd_depth, median_topk_depth

def main():
    print("Initializing Model...")
    dtr_model = DTRModel()
    
    # We keep g=0.60 fixed to anchor the metric
    calculator = DTRCalculator(
        lm_head=dtr_model.lm_head, 
        final_norm=dtr_model.final_norm,
        threshold_g=0.60, 
        depth_fraction_rho=0.85,
        top_k_agreement_k=10,
        top_k_agreement_threshold=0.9
    )
    
    print("Loading data...")
    dataset = load_dataset("gsm8k", "main", split="test")
    # Take 20 samples to combat noise
    gsm8k_samples = dataset.select(range(20))
    gsm8k_prompts = [f"Please reason step by step, and put your final numerical answer within \\boxed{{}}. Question: {sample['question']}" for sample in gsm8k_samples]
    
    # Simple prompts
    easy_prompts = [
        "Write a haiku about a cat.",
        "What is the capital of France?",
        "Explain how a rainbow is formed.",
        "Write a short polite email declining a meeting request.",
        "List three common types of apples.",
        "What color is the sky on a clear day?",
        "Who is the author of Harry Potter?",
        "Write a 3 sentence story about a brave knight.",
        "What is the chemical symbol for water?",
        "Write a funny joke about a programmer.",
        "What is 10 + 10?",
        "Name the planets in our solar system.",
        "How many legs does a spider have?",
        "Translate 'Hello' to Spanish.",
        "What is the largest ocean on Earth?",
        "Write a short poem about the moon.",
        "What is the powerhouse of the cell?",
        "Name a primary color.",
        "What sound does a cow make?",
        "Write a one-sentence summary of Romeo and Juliet."
    ]
    
    temperatures = [0.0, 0.4, 0.8] # 0.0 means greedy
    
    results = {}
    
    for temp in temperatures:
        do_sample = temp > 0.0
        # Pass slightly higher temp internally to avoid 0.0 exactly for sample=True if we decided to run it that way
        run_temp = temp if do_sample else None
        
        easy_res = run_experiment_on_dataset(dtr_model, calculator, "Easy Text", easy_prompts, run_temp, do_sample)
        math_res = run_experiment_on_dataset(dtr_model, calculator, "Math Reasoning", gsm8k_prompts, run_temp, do_sample)
        
        results[temp] = {"easy": easy_res, "math": math_res}
        
    print("\n\n=== FINAL SUMMARY ===")
    print("Temperature | Easy DTR | Math DTR | Easy Top-K Depth | Math Top-K Depth")
    print("-" * 75)
    for temp in temperatures:
        easy_res = results[temp]["easy"]
        math_res = results[temp]["math"]
        print(f"{temp:11.1f} | {easy_res[0]:7.2%} | {math_res[0]:7.2%} | {easy_res[2]:16.1f} | {math_res[2]:16.1f}")

if __name__ == "__main__":
    main()
