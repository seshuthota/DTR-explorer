import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
from dtr.model import DTRModel
from dtr.calculator import DTRCalculator

def print_histogram(depths, max_layer):
    counts = Counter(depths)
    print("\n--- Settling Depth (c_t) Histogram ---")
    for l in range(1, max_layer + 1):
        c = counts.get(l, 0)
        # 1 block character per token, scale it down slightly if too many
        scale = 1
        bar = "█" * (c // scale)
        print(f"Layer {l:2d}: {c:3d} | {bar}")
    print("--------------------------------------")

def run_prompt(dtr_model, name, prompt, thresholds):
    print(f"\n======================================")
    print(f"Running scenario: {name}")
    print(f"Prompt: {prompt}")
    
    outputs, prompt_length, _ = dtr_model.generate_with_hidden_states(
        prompt,
        max_new_tokens=150,
        system_prompt="You are a helpful assistant.",
        return_prompt_metadata=True
    )
    
    new_tokens = outputs.sequences[0][prompt_length:]
    decoded_resp = dtr_model.tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"\nResponse:\n{decoded_resp}\n")
    
    hidden_states_list = dtr_model.extract_generated_hidden_states(outputs)
            
    for g in thresholds:
        print(f"\n>>>> Threshold g = {g:.2f} <<<<")
        calculator = DTRCalculator(
            lm_head=dtr_model.lm_head, 
            final_norm=dtr_model.final_norm,
            threshold_g=g, 
            depth_fraction_rho=0.85
        )
        dtr_value, depths = calculator.calculate_dtr_for_sequence(hidden_states_list, return_depths=True)
        max_l = len(hidden_states_list[0]) - 1 if len(hidden_states_list) > 0 else 15
        print_histogram(depths, max_layer=max_l)
        print(f"DTR for '{name}': {dtr_value:.2%}\n")

def main():
    print("Initializing Model...")
    dtr_model = DTRModel()
    
    easy_prompt = "Write a haiku about cats."
    hard_prompt = "Please reason step by step, and put your final numerical answer within \\boxed{}. Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    
    thresholds = [0.5, 0.55, 0.6, 0.65]
    
    run_prompt(dtr_model, "Easy Text", easy_prompt, thresholds)
    run_prompt(dtr_model, "Math Reasoning", hard_prompt, thresholds)

if __name__ == "__main__":
    main()
