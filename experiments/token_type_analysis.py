import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import defaultdict
from dtr.model import DTRModel
from dtr.calculator import DTRCalculator

def categorize_token(tok_str):
    # Remove leading/trailing spaces HuggingFace tokenizer might add
    clean_str = tok_str.strip()
    
    if not clean_str:
        return "Whitespace/Punctuation"
    if any(char.isdigit() for char in clean_str):
        return "Number/Digit"
    if not clean_str.isalnum():
        return "Whitespace/Punctuation"
    
    function_words = {"the", "and", "to", "of", "a", "in", "is", "it", "you", "that", "for", "with", "on", "as", "at"}
    if clean_str.lower() in function_words:
        return "Common Function Word"
        
    return "Content/Rare Word"

def run_token_breakdown(dtr_model, name, prompt, g_threshold=0.60):
    print(f"\n======================================")
    print(f"Running Token Breakdown for: {name} (Threshold g={g_threshold})")
    
    calculator = DTRCalculator(
        lm_head=dtr_model.lm_head, 
        final_norm=dtr_model.final_norm,
        threshold_g=g_threshold, 
        depth_fraction_rho=0.85
    )
    
    outputs, prompt_length, _ = dtr_model.generate_with_hidden_states(
        prompt,
        max_new_tokens=150,
        system_prompt="You are a helpful assistant.",
        return_prompt_metadata=True
    )
    
    new_tokens = outputs.sequences[0][prompt_length:]
    
    hidden_states_list = dtr_model.extract_generated_hidden_states(outputs)
            
    # Get settling depths for each token
    _, depths = calculator.calculate_dtr_for_sequence(hidden_states_list, return_depths=True)
    
    # Track depths by category
    category_depths = defaultdict(list)
    
    for t in range(len(depths)):
        token_id = new_tokens[t]
        token_str = dtr_model.tokenizer.decode([token_id])
        category = categorize_token(token_str)
        category_depths[category].append(depths[t])
        
    print("\n--- Average Settling Depth by Token Type ---")
    for category, dep_list in sorted(category_depths.items()):
        avg_depth = np.mean(dep_list)
        count = len(dep_list)
        print(f"{category:25s}: {avg_depth:>5.1f} layers (based on {count:3d} tokens)")

def main():
    print("Initializing Model...")
    dtr_model = DTRModel()
    
    easy_prompt = "Write a short haiku about a cat sleeping."
    hard_prompt = "Please reason step by step, and put your final numerical answer within \\boxed{}. Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    
    # We use our calibrated threshold of 0.60
    run_token_breakdown(dtr_model, "Easy Text", easy_prompt, 0.60)
    run_token_breakdown(dtr_model, "Math Reasoning", hard_prompt, 0.60)

if __name__ == "__main__":
    main()
