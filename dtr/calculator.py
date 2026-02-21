import torch
import torch.nn.functional as F
import math

class DTRCalculator:
    """
    This class implements the Deep-Thinking Ratio (DTR) calculation as described in the paper
    'Think Deep, Not Just Long'. It measures how much computational 'effort' a model exerts 
    on each generated token by analyzing how its internal predictions evolve across hidden layers.
    """
    def __init__(self, lm_head, final_norm, threshold_g=0.5, depth_fraction_rho=0.85, top_k_agreement_k=10, top_k_agreement_threshold=0.9):
        """
        Args:
            lm_head: The final linear layer (unembedding matrix W_U) of the language model.
            final_norm: The final layer normalization module that is applied before the lm_head.
            threshold_g: The acceptable "error/distance" threshold.
            depth_fraction_rho: The cut-off percentage (default 0.85 = 85%).
            top_k_agreement_k: The K value for the Top-K agreement metric (default 10).
            top_k_agreement_threshold: The threshold for top-K agreement to be considered 'settled' (default 0.9).
        """
        self.lm_head = lm_head
        self.final_norm = final_norm
        self.threshold_g = threshold_g
        self.depth_fraction_rho = depth_fraction_rho
        self.top_k_agreement_k = top_k_agreement_k
        self.top_k_agreement_threshold = top_k_agreement_threshold

    def compute_jsd(self, p, q):
        """
        Computes the Jensen-Shannon Divergence (JSD).
        """
        # Compute JSD in float32 for numerical stability.
        p = p.float()
        q = q.float()
        m = 0.5 * (p + q)
        log_m = torch.log(m.clamp(min=1e-12))
        
        kl_p_m = F.kl_div(log_m, p, reduction='sum')
        kl_q_m = F.kl_div(log_m, q, reduction='sum')
        
        jsd = 0.5 * kl_p_m + 0.5 * kl_q_m
        return jsd.item()
        
    def calculate_dtr_for_sequence(self, hidden_states_list, return_depths=False, return_top_k_depths=False):
        """
        Calculates the overall Deep-Thinking Ratio for a generated response.
        """
        T = len(hidden_states_list)
        if T == 0:
            if return_depths and return_top_k_depths:
                return 0.0, [], []
            elif return_depths:
                return 0.0, []
            return 0.0
            
        deep_thinking_count = 0
        valid_token_count = 0
        settling_depths = []
        top_k_settling_depths = []
            
        for t in range(T):
            token_hidden_states = hidden_states_list[t]
            L = len(token_hidden_states) - 1
            if L <= 0:
                continue
            valid_token_count += 1
            
            # Get the model's FINAL answer for this token
            h_t_L = token_hidden_states[-1].squeeze(0).squeeze(0)
            
            # Apply final normalization and pass through the unembedding matrix
            h_t_L_normed = self.final_norm(h_t_L)
            z_t_L = self.lm_head(h_t_L_normed)
            p_t_L = F.softmax(z_t_L, dim=-1)
            
            # Calculate Top-K sets for the final layer
            _, top_k_indices_L = torch.topk(p_t_L, k=self.top_k_agreement_k)
            set_L = set(top_k_indices_L.tolist())
            
            min_D_tl = float('inf')
            settling_depth_c_t = L
            top_k_settling_depth = L
            found_top_k_settling = False
            
            # Iterate through every intermediate block
            for l in range(1, L + 1):
                h_t_l = token_hidden_states[l].squeeze(0).squeeze(0)
                
                # Apply normalization and project to vocabulary space
                h_t_l_normed = self.final_norm(h_t_l)
                z_t_l = self.lm_head(h_t_l_normed)
                p_t_l = F.softmax(z_t_l, dim=-1)
                
                # STEP 3: Compare this layer's prediction (p_t_l) to the final prediction (p_t_L)
                D_tl = self.compute_jsd(p_t_l, p_t_L)
                
                # The paper defines 'settling' using the minimum divergence seen *up to* this point
                min_D_tl = min(min_D_tl, D_tl)
                
                # If the difference drops below our threshold (0.5), the prediction has "settled"!
                # The model is confident enough early on, so it doesn't need the remaining layers to think.
                if min_D_tl <= self.threshold_g:
                    # We capture the first layer where it settles, but we MUST keep iterating 
                    # for the Top-K agreement metric to be evaluated fully across all layers
                    if settling_depth_c_t == L:
                        settling_depth_c_t = l
                
                # Evaluate Top-K agreement
                if not found_top_k_settling:
                    _, top_k_indices_l = torch.topk(p_t_l, k=self.top_k_agreement_k)
                    set_l = set(top_k_indices_l.tolist())
                    agreement = len(set_l.intersection(set_L)) / self.top_k_agreement_k
                    if agreement >= self.top_k_agreement_threshold:
                        top_k_settling_depth = l
                        found_top_k_settling = True
                    
            # STEP 4: Determine if this token required "Deep Thinking"
            # Calculate the layer index that marks the start of the "late" regime (e.g., 85% depth)
            deep_thinking_regime_start = math.ceil(self.depth_fraction_rho * L)
            
            # Record the settling depth for this token
            settling_depths.append(settling_depth_c_t)
            top_k_settling_depths.append(top_k_settling_depth)
            
            # If the token didn't settle until it reached this late regime, it required deep thinking!
            if settling_depth_c_t >= deep_thinking_regime_start:
                deep_thinking_count += 1
                
        # STEP 5: Calculate the final DTR percentage (Deep thinking tokens / Total tokens)
        if valid_token_count == 0:
            dtr = 0.0
        else:
            dtr = deep_thinking_count / valid_token_count
        
        if return_depths and return_top_k_depths:
            return dtr, settling_depths, top_k_settling_depths
        elif return_depths:
            return dtr, settling_depths
            
        return dtr
