import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtr.model import DTRModel
from dtr.calculator import DTRCalculator
import torch.nn.functional as F

def main():
    print("Initializing DTR Model...")
    dtr_model = DTRModel()
    
    calculator = DTRCalculator(lm_head=dtr_model.lm_head, final_norm=dtr_model.final_norm, threshold_g=0.5, depth_fraction_rho=0.85)
    
    prompt = "Please reason step by step. Question: What is 2 + 2?"
    
    print("\nGenerating short response...")
    outputs, prompt_length, _ = dtr_model.generate_with_hidden_states(
        prompt,
        max_new_tokens=10,
        return_prompt_metadata=True
    )
    
    generated_sequence = outputs.sequences[0]
    new_tokens = generated_sequence[prompt_length:]
    
    # Get hidden states for generated tokens
    hidden_states_list = dtr_model.extract_generated_hidden_states(outputs)

    print("\n--- JSD Analysis per Generated Token ---")
    
    for t in range(len(hidden_states_list)):
        token_id = new_tokens[t]
        token_str = dtr_model.tokenizer.decode([token_id])
        
        token_hidden_states = hidden_states_list[t]
        L = len(token_hidden_states) - 1
        
        # Final layer distribution
        h_t_L = token_hidden_states[-1].squeeze(0).squeeze(0)
        h_t_L_normed = dtr_model.final_norm(h_t_L)
        p_t_L = F.softmax(calculator.lm_head(h_t_L_normed), dim=-1)
        
        jsd_values = []
        for l in range(1, L + 1):
            h_t_l = token_hidden_states[l].squeeze(0).squeeze(0)
            h_t_l_normed = dtr_model.final_norm(h_t_l)
            p_t_l = F.softmax(calculator.lm_head(h_t_l_normed), dim=-1)
            jsd_values.append(calculator.compute_jsd(p_t_l, p_t_L))
            
        print(f"Token '{token_str}' (ID: {token_id})")
        print(f"  Layer 1/4/8/12/... JSD: {[round(jsd_values[i], 3) for i in range(0, L, max(1, L//8))]}")
        print(f"  Min JSD reached: {min(jsd_values):.3f} (Max possible JSD is ~0.693)")

if __name__ == "__main__":
    main()
