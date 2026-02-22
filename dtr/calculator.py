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
        log_p = torch.log(p.clamp(min=1e-12))
        log_q = torch.log(q.clamp(min=1e-12))

        kl_p_m = (p * (log_p - log_m)).sum()
        kl_q_m = (q * (log_q - log_m)).sum()

        jsd = 0.5 * kl_p_m + 0.5 * kl_q_m
        return float(jsd.detach().cpu().item())
        
    def calculate_dtr_for_sequence(
        self,
        hidden_states_list,
        return_depths=False,
        return_top_k_depths=False,
        max_tokens=None,
    ):
        """
        Calculates the overall Deep-Thinking Ratio for a generated response.
        """
        T = len(hidden_states_list)
        if max_tokens is not None and max_tokens > 0:
            T = min(T, int(max_tokens))
            hidden_states_list = hidden_states_list[:T]
        if T == 0:
            if return_depths and return_top_k_depths:
                return 0.0, [], []
            elif return_depths:
                return 0.0, []
            return 0.0
            
        deep_thinking_count = 0
        valid_token_count = 0
        settling_depths = [] if return_depths else None
        top_k_settling_depths = [] if return_top_k_depths else None
            
        with torch.no_grad():
            for t in range(T):
                token_hidden_states = hidden_states_list[t]
                L = len(token_hidden_states) - 1
                if L <= 0:
                    continue
                valid_token_count += 1

                # Stack all layers once: [L+1, hidden]
                layers_hidden = torch.stack(
                    [h.squeeze(0).squeeze(0) for h in token_hidden_states],
                    dim=0,
                )

                # Project all layers to vocab in one pass on device.
                layers_hidden_normed = self.final_norm(layers_hidden)
                logits = self.lm_head(layers_hidden_normed)
                probs = F.softmax(logits.float(), dim=-1)

                p_final = probs[-1]      # [vocab]
                p_mid = probs[1:]        # [L, vocab]
                p_final_b = p_final.unsqueeze(0)

                # Vectorized JSD for all intermediate layers vs final layer.
                m = 0.5 * (p_mid + p_final_b)
                log_m = torch.log(m.clamp(min=1e-12))
                log_p_mid = torch.log(p_mid.clamp(min=1e-12))
                log_p_final = torch.log(p_final_b.clamp(min=1e-12))
                kl_p_m = (p_mid * (log_p_mid - log_m)).sum(dim=-1)
                kl_q_m = (p_final_b * (log_p_final - log_m)).sum(dim=-1)
                jsd_values = 0.5 * (kl_p_m + kl_q_m)  # [L]

                # Paper settling rule: first layer where min divergence-so-far <= g.
                min_so_far = torch.cummin(jsd_values, dim=0).values
                settled_mask = min_so_far <= self.threshold_g
                if torch.any(settled_mask):
                    settling_depth_c_t = int(torch.argmax(settled_mask.int()).item()) + 1
                else:
                    settling_depth_c_t = L

                top_k_settling_depth = L
                if return_top_k_depths:
                    # Only compute Top-K agreement when explicitly requested.
                    top_k_indices_L = torch.topk(p_final, k=self.top_k_agreement_k).indices  # [K]
                    top_k_indices_layers = torch.topk(
                        p_mid,
                        k=self.top_k_agreement_k,
                        dim=-1
                    ).indices  # [L, K]
                    agreements = (
                        (top_k_indices_layers.unsqueeze(-1) == top_k_indices_L.unsqueeze(0).unsqueeze(0))
                        .any(dim=-1)
                        .sum(dim=-1)
                        .float()
                        / float(self.top_k_agreement_k)
                    )  # [L]
                    topk_mask = agreements >= self.top_k_agreement_threshold
                    if torch.any(topk_mask):
                        top_k_settling_depth = int(torch.argmax(topk_mask.int()).item()) + 1

                deep_thinking_regime_start = math.ceil(self.depth_fraction_rho * L)

                if return_depths:
                    settling_depths.append(settling_depth_c_t)
                if return_top_k_depths:
                    top_k_settling_depths.append(top_k_settling_depth)

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
