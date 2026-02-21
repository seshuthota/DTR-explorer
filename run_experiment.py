import sys
import re
from datasets import load_dataset
from dtr.model import DTRModel
from dtr.calculator import DTRCalculator

def extract_answer(text):
    """
    Helper function to parse the numerical answer out of the model's generated text.
    The benchmark (GSM8K) expects the final answer to be wrapped in a LaTeX box, 
    like: \boxed{42}
    """
    # Use regular expressions to find anything inside \boxed{...}
    match = re.search(r"\\boxed\{(.+?)\}", text)
    if match: 
        return match.group(1).strip()
    return None

def main():
    print("Loading GSM8K...")
    # Load the GSM8K dataset from Hugging Face. GSM8K contains grade-school math word problems.
    # We load the "test" split to evaluate the model's performance on unseen questions.
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # For demonstration and speed, we only take the first 5 samples from the dataset.
    # To run a full evaluation, you would remove this line and loop over the whole `dataset`.
    samples = dataset.select(range(5))

    print("Initializing DTR Model...")
    # Initialize our custom model wrapper that loads the LFM2.5-1.2B model onto the GPU 
    # and ensures it outputs intermediate hidden states during generation.
    dtr_model = DTRModel()
    
    # Initialize the core DTR Calculator.
    # We pass it the model's unembedding matrix (lm_head) so it can project hidden states into vocab probabilities.
    # threshold_g (0.5) and depth_fraction_rho (0.85) are the default parameters from the DTR paper.
    calculator = DTRCalculator(
        lm_head=dtr_model.lm_head, 
        final_norm=dtr_model.final_norm,
        threshold_g=0.5,            # INCREASE this (e.g. 1.0 or 1.5). Allow tokens to "settle" even if they are further from the final distribution.
        depth_fraction_rho=0.85     # DECREASE this. Consider anything past the 60% depth mark to be "deep thinking" instead of 85%.
    )
    
    # Keep track of statistics to compute accuracy and average DTR later
    results = []
    
    for i, sample in enumerate(samples):
        # Extract the question text
        question = sample['question']
        
        # The GSM8K answer format often looks like "Step 1... Step 2... #### 42"
        # We split by '####' to grab the actual final number (the ground truth).
        ground_truth = sample['answer'].split("####")[-1].strip()
        
        # We construct a prompt telling the model to "reason step by step" (Chain-of-Thought)
        # and to place its final answer inside \boxed{}.
        prompt = f"Please reason step by step, and put your final numerical answer within \\boxed{{}}. Question: {question}"
        
        print(f"\n--- Sample {i+1}/5 ---")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        # Generate the response. 
        # Crucially, this returns the generated token IDs *and* a massive tuple of all hidden states.
        outputs, prompt_length, _ = dtr_model.generate_with_hidden_states(
            prompt,
            max_new_tokens=200,
            return_prompt_metadata=True
        )
        
        # Extract just the newly generated tokens (slice off the input prompt)
        generated_sequence = outputs.sequences[0]
        new_tokens = generated_sequence[prompt_length:]
        
        # Decode the tokens back into readable text
        decoded_resp = dtr_model.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Check if the model got the answer right
        ans = extract_answer(decoded_resp)
        is_correct = (ans == ground_truth)
        
        # --- PREPARING HIDDEN STATES FOR DTR ---
        # The outputs.hidden_states object is a bit complex. 
        # It's a tuple where outputs.hidden_states[t] = the hidden states for generation step `t`.
        hidden_states_list = dtr_model.extract_generated_hidden_states(outputs)
                
        # --- CALCULATING DEEP THINKING RATIO ---
        # We pass the clean list of per-token hidden states into our math module
        dtr_value = calculator.calculate_dtr_for_sequence(hidden_states_list)
        seq_len = len(hidden_states_list)
        
        print(f"Generated {seq_len} tokens")
        print(f"Predicted Answer: {ans} | Ground Truth: {ground_truth}")
        print(f"Correct: {is_correct} | DTR: {dtr_value:.2%}")
        
        # Save results for potential aggregate metrics later
        results.append({
            "is_correct": is_correct,
            "dtr": dtr_value,
            "length": seq_len
        })

if __name__ == "__main__":
    main()
