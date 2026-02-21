# Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens

**Wei-Lin Chen**<sup>1,2</sup>, Liqian Peng<sup>2</sup>, Tian Tan<sup>2</sup>, Chao Zhao<sup>2</sup>, Blake JianHang Chen<sup>2</sup>, Ziqian Lin<sup>2</sup>, Alec Go<sup>2</sup>, Yu Meng<sup>1</sup>

<sup>1</sup>University of Virginia, <sup>2</supGoogle

*Work done as a student researcher at Google.*

---

**Abstract**

Large language models (LLMs) have demonstrated impressive reasoning capabilities by scaling test-time compute via long Chain-of-Thought (CoT). However, recent findings suggest that raw token counts are unreliable proxies for reasoning quality: increased generation length does not consistently correlate with accuracy and may instead signal "overthinking," leading to performance degradation. In this work, we quantify inference-time effort by identifying **deep-thinking tokens**—tokens where internal predictions undergo significant revisions in deeper model layers prior to convergence. Across four challenging mathematical and scientific benchmarks (AIME 24/25, HMMT 25, and GPQA-diamond) and a diverse set of reasoning-focused models (GPT-OSS, DeepSeek-R1, and Qwen3), we show that **deep-thinking ratio** (the proportion of deep-thinking tokens in a generated sequence) exhibits a robust and consistently positive correlation with accuracy, substantially outperforming both length-based and confidence-based baselines. Leveraging this insight, we introduce **Think@n**, a test-time scaling strategy that prioritizes samples with high deep-thinking ratios. We demonstrate that Think@n matches or exceeds standard self-consistency performance while significantly reducing inference costs by enabling the early rejection of unpromising generations based on short prefixes.

---

## 1. Introduction

Large language models (LLMs) have achieved remarkable reasoning capabilities by generating explicit thought traces, most notably through the Chain-of-Thought (CoT) paradigm (Wei et al., 2022). Prior works have shown that increasing the number of reasoning tokens generated can generally boost task performance (Anthropic, 2025a,b; Guo et al., 2025; Jaech et al., 2024; OpenAI, 2025; Team et al., 2025; Yang et al., 2025; Zhong et al., 2024), motivating methods that encourage longer and more elaborate thinking traces (Balachandran et al., 2025; Muennighoff et al., 2025; Yeo et al., 2025).

However, a growing body of evidence suggests that token counts are unreliable indicators of model performance during inference, as longer reasoning does not consistently translate into higher accuracy (Aggarwal et al., 2025; Su et al., 2025; Sui et al., 2025; Wu et al., 2025). Empirical studies reveal inverted-U relationships between CoT length and performance (Wu et al., 2025), as well as inverse-scaling behaviors in which longer reasoning traces systematically degrade performance (Gema et al., 2025). Excessive reasoning may reflect overthinking, wherein models amplify flawed heuristics or fixate on irrelevant details (Feng et al., 2025). Consequently, relying on length as a metric for reasoning quality not only encourages verbosity over clarity but also wastes computational resources on uninformative tokens. Though recent work has attempted to assess the semantic structure of CoTs (e.g., by representing reasoning traces as graphs), such approaches often rely on costly auxiliary parsing or external annotations (Feng et al., 2025). Addressing these limitations requires more principled and efficient methods for measuring thinking effort that can distinguish effective reasoning from uninformative generation.

In this work, we introduce **deep-thinking ratio (DTR)** as a direct measure of inference-time thinking effort. Instead of relying on surface-level features like output length, we focus on how individual tokens are produced internally. We posit that when a token prediction stabilizes in early layers, subsequent depth-wise modifications entail relatively low computational effort, resembling less thinking. In contrast, token predictions that undergo sustained revision in deeper layers before converging reflect greater thinking (Chuang et al., 2023). We operationalize this idea by projecting intermediate-layer hidden states into the vocabulary space and comparing each layer's prediction distribution to the final-layer distribution. Tokens whose distributions do not converge until deeper layers are identified as deep-thinking tokens. By counting the proportion of deep-thinking tokens in a generated sequence, we obtain DTR, which provides a simple, mechanistically grounded measure of thinking effort, requiring neither task-specific heuristics nor external structural annotations.

Across four challenging mathematical and scientific reasoning benchmarks—AIME 2024, AIME 2025, HMMT 2025, and GPQA (Art of Problem Solving, 2024a,b, 2025a,b; HMMT, 2025; Rein et al., 2024)—and a range of reasoning-focused language models, including GPT-OSS, DeepSeek-R1, and Qwen3 families (Guo et al., 2025; OpenAI et al., 2025; Yang et al., 2025), we demonstrate that measuring deep-thinking tokens yields strong correlations with task accuracy. The achieved correlation is substantially higher than those obtained using length-based or confidence-based baselines. Furthermore, we show that deep-thinking for parallel inference scaling, where preferentially selecting tokens can be leveraged and aggregating responses with higher DTR achieves performance comparable or better than standard consensus-based methods, while requiring only half the compute cost. Our contributions are summarized as follows:

- We introduce **deep-thinking ratio (DTR)**—a measure that counts the ratio of deep-thinking tokens in a sequence whose predictions undergo sustained revision in deeper layers before converging—as a new lens for characterizing inference-time thinking effort.
- We empirically show that, across multiple reasoning benchmarks and model families, DTR of a generated sequence exhibits strong positive correlations with task accuracy, outperforming length-based and confidence-based baselines significantly.
- We introduce **Think@n**, a test-time scaling strategy that preferentially selects and aggregates samples with higher DTR. By early halting unpromising generations based on DTR estimated from short prefixes, Think@n matches or surpasses standard self-consistency with approximately half the inference cost.

---

## 2. Measuring Deep-Thinking Ratio

### 2.1. Preliminaries

We consider an autoregressive language model $f_\theta$ composed of $L$ transformer layers, hidden dimension $d$, and vocabulary $V$. Given a prefix sequence $y_{<t}$, the forward pass at generation step $t$ produces a sequence of residual stream states $\{ h_{t,l} \}_{l=1}^L$, where $h_{t,l} \in \mathbb{R}^d$ denotes the hidden state after layer $l$. The final-layer output $h_{t,L}$ is projected by the language modeling head (i.e., the unembedding matrix) $W_U \in \mathbb{R}^{|V| \times d}$ to produce logits over the vocabulary.

Prior research on early exiting (Belrose et al., 2023; Din et al., 2024; Elbayad et al., 2019; Schuster et al., 2022; Teerapittayanon et al., 2016) has demonstrated that, without specialized auxiliary training, applying the language modeling head directly to intermediate-layer hidden states effectively yields meaningful predictive distributions (Kao et al., 2020; Nostalgebraist, 2020). Building on this line of works, we project intermediate-layer hidden states into the vocabulary space using the same unembedding matrix $W_U$. For each intermediate layer $l \in \{1, \ldots, L-1\}$, we compute the logit vector $z_{t,l}$ and probability distribution $p_{t,l}$ as:

$$p_{t,l} = \text{softmax}(z_{t,l}), \quad z_{t,l} = W_U h_{t,l}$$

The model's final-layer distribution is denoted by $p_{t,L}$.

### 2.2. Deep-Thinking Tokens

We posit that inference-time thinking effort for a token manifests as the continued evolution of predictive distributions (i.e., $p_{t,l}$) across LM layers. Tokens with earlier distributional stabilization correspond to less additional thinking, while those having later stabilization correspond to needing more extended internal thinking. In other words, simple tokens stabilize early with shallow computation, whereas difficult tokens requiring more thinking exhibit distributional shifts in deeper layers with more computation. To illustrate this, we show a motivation example on answering a GPQA (Rein et al., 2024) question in Figure 2.

To quantify this behavior, we measure how long a token's predictive distribution continues to change before settling, operationalized as the layer at which the intermediate distribution becomes sufficiently close to the final-layer distribution. Specifically, for each generation step $t$ and layer $l$, we compute the Jensen–Shannon divergence (JSD) between the intermediate-layer distribution $p_{t,l}$ and the final-layer distribution $p_{t,L}$:

$$D_{t,l} := \text{JSD}(p_{t,L} \| p_{t,l}) = \frac{1}{2} H(p_{t,L}) + \frac{1}{2} H(p_{t,l}) - H\left(\frac{p_{t,L} + p_{t,l}}{2}\right)$$

where $H(\cdot)$ denotes Shannon entropy. By construction, $D_{t,L} = 0$. A trajectory $l \mapsto D_{t,l}$ that approaches zero only at later layers indicates prolonged distributional revision (think more), whereas early convergence indicates that the model settles on its final prediction with fewer subsequent updates (think less). We employ JSD due to its symmetry and boundedness, following (Chuang et al., 2023). We explore other distance metrics in Section A.

To enforce a strict notion of settling, we compute:

$$\bar{D}_{t,l} = \min_{j \leq l} D_{t,j}$$

We define the settling depth $c_t$ as the first layer at which $\bar{D}_{t,l}$ falls below a fixed threshold $g$:

$$c_t = \min \{ l \in \{1, \ldots, L\} : \bar{D}_{t,l} \leq g \}$$

We then define a deep-thinking regime using a depth fraction $\rho \in (0, 1)$, with:

$$L_{\text{deep-thinking}} = \{ l : l \geq \lceil \rho \times L \rceil \}$$

A token is classified as a **deep-thinking token** (i.e., requiring more layer computations and more thinking effort to become sufficiently close to the final-layer distribution) if $c_t \in L_{\text{deep-thinking}}$. An illustration is shown in Figure 3.

Finally, for a generated sequence $S$ of length $T$, we define the deep-thinking ratio, DTR($S$), for the sequence as the proportion of tokens that settle in the late regime:

$$\text{DTR}(S) = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}[c_t \in L_{\text{deep-thinking}}]$$

A higher DTR indicates that a larger fraction of tokens undergo extended computation for distributional revision before stabilizing. We note that our proposed method does not imply that early-settling tokens are suboptimal; rather, it provides a depth-wise characterization of inference-time thinking effort that complements the surface-level token length measure.

---

## 3. Deep-Thinking Ratio Reflects Task Accuracy More Reliably

We empirically evaluate whether our distributional distance-based measurement provides a more faithful and robust characterization of inference-time thinking effort than surface-level, length-based proxies (i.e., token counts).

### Models

We evaluate eight variants of reasoning LLMs from three model families: GPT-OSS-20B (with low, medium, and high reasoning levels) and GPT-OSS-120B (with low, medium, and high reasoning levels) (OpenAI et al., 2025), DeepSeek-R1-70B (Guo et al., 2025), and Qwen3-30B-Thinking (Yang et al., 2025). These models are known for their strong, long CoT capability in mathematical and complex reasoning, and span multiple parametric scales for comprehensive coverage.

### Tasks

We focus on reasoning-intensive benchmarks where scaling CoT-style computation at inference time plays a central role. We adopt four benchmarks widely used in recent evaluations of LLM reasoning capabilities (Balunović et al., 2025; OpenAI, 2025; xAI, 2025), including three competition-level mathematical problem sets, AIME 2024 (Art of Problem Solving, 2024a,b), AIME 2025 (Art of Problem Solving, 2025a,b), and HMMT 2025 (HMMT, 2025), as well as the diamond set of GPQA (Rein et al., 2024), which consists of challenging graduate-level scientific questions.

### Decoding Settings

Following (Gema et al., 2025), we prompt models to reason step by step using a fixed, neutral instruction, without specifying a reasoning budget or explicitly encouraging longer deliberation. This setup allows each model to naturally allocate inference-time computation on a per-instance basis, avoiding confounds introduced by externally imposed token budgets or budget-conditioning prompts. Following standard practice in natural overthinking analyses (Gema et al., 2025), we sample multiple responses for each question (25 responses per question in our experiments). Across these samples, models naturally exhibit variation in reasoning length and internal computation patterns. We use the developer recommended sampling parameters for all tested models: temperature=1.0 and top p=1.0 for GPT-OSS series; temperature=0.6 and top p=0.95 for DeepSeek-R1-70B and Qwen-3-30B-Thinking.

For each sampled response, we record intermediate-layer hidden states, obtain their projected probability distribution, and compute DTR as described in Section 2. We uniformly set the settling threshold $g = 0.5$ and the depth fraction $\rho = 0.85$ to define the deep-thinking regime. We also analyze with different values and the results are provided in Section 3.2. The reported statistics are averaged over 30 random seeds across decoding runs.

### 3.1. Results

To quantify the relationship between inference-time thinking effort and task performance, we measure the association between thinking effort scores and answer accuracy by computing Pearson correlation coefficient. Specifically, we conduct a binned analysis following (Gema et al., 2025) by partitioning sampled sequences into quantile bins (i.e., 5 bins) based on their DTR (Equation 6) and computing the average accuracy within each bin.

We compare deep-thinking token measurement against the following baselines, including length-based proxies and confidence-based approaches, which are also commonly adopted to assess generation quality.

- **Token count**: The total number of tokens generated in the model's output reasoning traces. This measure is widely framed as a direct proxy for test-time compute, and underlies many empirical studies of inference-time scaling.

- **Reverse token count**: As a complementary baseline, we additionally consider reverse token count, defined as the negative of the total number of generated tokens for each response.

- **Log probability**: Following the notation in Section 2, let a generated sequence $S = (y_1, \ldots, y_T)$. At generation step $t$, the model's output prediction distribution (at final-layer $L$) over the vocabulary V is denoted by $p_{t,L}(\cdot)$. We compute the average log-probability of the sampled tokens:

$$\text{LogProb}(S) = \frac{1}{T} \sum_{t=1}^{T} \log p_{t,L}(y_t)$$

- **Negative perplexity**: Perplexity is defined as the exponentiated negative average log-probability:

$$\text{PPL}(S) = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log p_{t,L}(y_t)\right)$$

We report negative perplexity $-\text{PPL}(S)$ so that larger values correspond to higher confidence.

- **Negative entropy**: To incorporate information from the full prediction distribution over V rather than only the sampled token, we compute the average entropy:

$$\text{Ent}(S) = \frac{1}{T} \sum_{t=1}^{T} H(p_{t,L}), \quad H(p_{t,L}) = -\sum_{v \in V} p_{t,L}(v) \log p_{t,L}(v)$$

We report negative entropy $-\text{Ent}(S)$, where larger values indicate more peaked distributions and thus greater model confidence.

- **Self-Certainty**: We also include Self-Certainty (Kang et al., 2025), a distributional confidence metric based on the idea that higher confidence corresponds to prediction distributions that are further from the uniform distribution $u$, which represents maximum uncertainty. Formally, self-certainty is defined as the average Kullback-Leibler (KL) divergence between $u(v) = 1/|V|$ and $p_{t,L}$:

$$\text{Self-Certainty}(S) = \frac{1}{T} \sum_{t=1}^{T} \text{KL}(u \| p_{t,L}) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{v \in V} \log |V| p_{t,L}(v)$$

### Correlation Results

Table 1 reports the correlation between task accuracy and different measurements, across eight model variants and four benchmarks.

| Model | Token Length | Reverse Token Length | Log Probability | Negative Perplexity | Negative Entropy | Self-Certainty | DTR (Ours) |
|-------|-------------|---------------------|-----------------|---------------------|------------------|-----------------|-------------|
| **AIME 2025** ||||||||
| OSS-120B-low | 0.504 | -0.504 | 0.872 | 0.453 | 0.863 | 0.803 | **0.930** |
| OSS-120B-medium | -0.365 | 0.365 | 0.817 | 0.246 | 0.822 | 0.815 | **0.862** |
| OSS-120B-high | -0.961 | 0.961 | 0.705 | 0.552 | 0.711 | 0.728 | **0.796** |
| OSS-20B-low | -0.689 | 0.689 | 0.579 | 0.849 | 0.665 | 0.275 | **0.373** |
| OSS-20B-medium | -0.757 | 0.757 | 0.616 | -0.677 | 0.637 | 0.097 | **0.161** |
| OSS-20B-high | -0.385 | 0.385 | 0.455 | -0.795 | 0.550 | 0.489 | **0.610** |
| DeepSeek-R1-70B | -0.973 | 0.973 | 0.961 | 0.955 | 0.946 | 0.899 | **0.974** |
| Qwen3-30B-Thinking | -0.663 | 0.663 | -0.008 | -0.035 | 0.154 | 0.828 | **0.855** |
| **AIME 2024** ||||||||
| OSS-120B-low | -0.166 | 0.166 | 0.897 | 0.682 | 0.869 | 0.741 | **0.840** |
| OSS-120B-medium | -0.680 | 0.680 | 0.795 | -0.293 | 0.908 | 0.924 | **0.533** |
| OSS-120B-high | -0.755 | 0.755 | 0.700 | -0.275 | 0.593 | 0.654 | **0.905** |
| OSS-20B-low | -0.655 | 0.655 | 0.548 | -0.342 | 0.667 | 0.584 | **0.730** |
| OSS-20B-medium | -0.827 | 0.827 | 0.195 | -0.150 | 0.440 | 0.252 | **-0.192** |
| OSS-20B-high | -0.989 | 0.989 | 0.809 | 0.262 | 0.921 | 0.855 | **0.824** |
| DeepSeek-R1-70B | -0.987 | 0.987 | -0.037 | 0.223 | 0.067 | 0.287 | **0.430** |
| Qwen3-30B-Thinking | -0.869 | 0.869 | -0.857 | -0.720 | -0.680 | -0.246 | **-0.657** |
| **GPQA-Diamond** ||||||||
| OSS-120B-low | 0.682 | -0.682 | 0.984 | 0.172 | 0.995 | 0.996 | **0.976** |
| OSS-120B-medium | -0.340 | 0.340 | 0.973 | 0.316 | 0.985 | 0.981 | **0.823** |
| OSS-120B-high | -0.970 | 0.970 | 0.854 | 0.501 | 0.813 | 0.885 | **0.845** |
| OSS-20B-low | -0.602 | 0.602 | 0.984 | 0.235 | 0.991 | 0.917 | **0.935** |
| OSS-20B-medium | -0.847 | 0.847 | 0.914 | 0.468 | 0.911 | 0.889 | **0.718** |
| OSS-20B-high | -0.794 | 0.794 | 0.879 | 0.461 | 0.902 | 0.915 | **0.992** |
| DeepSeek-R1-70B | -0.930 | 0.930 | 0.068 | -0.133 | -0.165 | -0.532 | **0.885** |
| Qwen3-30B-Thinking | -0.634 | 0.634 | 0.589 | 0.865 | 0.711 | 0.943 | **0.828** |
| **HMMT 2025** ||||||||
| OSS-120B-low | 0.871 | -0.871 | 0.761 | 0.629 | 0.695 | 0.884 | **0.305** |
| OSS-120B-medium | -0.793 | 0.793 | 0.706 | 0.045 | 0.618 | 0.631 | **0.926** |
| OSS-120B-high | -0.967 | 0.967 | 0.750 | 0.503 | 0.728 | 0.754 | **0.972** |
| OSS-20B-low | -0.634 | 0.634 | -0.695 | 0.549 | -0.359 | -0.489 | **0.689** |
| OSS-20B-medium | -0.668 | 0.668 | 0.447 | 0.336 | 0.424 | 0.331 | **0.247** |
| OSS-20B-high | -0.352 | 0.352 | 0.537 | 0.994 | 0.831 | 0.628 | **0.932** |
| DeepSeek-R1-70B | -0.866 | 0.866 | 0.879 | 0.889 | 0.858 | 0.905 | **0.902** |
| Qwen3-30B-Thinking | -0.950 | 0.950 | -0.803 | -0.762 | -0.801 | 0.745 | **0.911** |
| **Average** | **-0.594** | **0.594** | **0.527** | **0.219** | **0.571** | **0.605** | **0.683** |

As observed, measuring sequences with token count exhibits notable negative correlations (orange-colored values), with mean $r = -0.59$. This indicates that longer generations are more associated with lower performance, aligning with recent reports of inverse scaling and overthinking. Extended reasoning traces could be symptomatic of redundant, misguided, or error-amplifying deliberation. The results underscore the unreliability of using surface-level length feature as proxy for effective problem solving. Reversing token count yields a positive correlation of identical magnitude. However, the improvement is purely post hoc, reflecting the empirical regularity in regimes where shorter responses are more accurate. As such, reverse token count only serves as a statistical adjustment, rather than capture principled notion of computation or thinking effort.

Compared to token count measure, confidence-based measures (log probability, negative perplexity, negative entropy, and self-certainty) exhibit moderately positive correlations with mean $r = 0.219 \sim 0.605$. This indicates that model confidence captures partial information about correctness. However, their behavior is relatively heterogeneous across models and benchmarks: while certain configurations achieve strong positive correlations, others deteriorate to weak or even negative associations. This inconsistency suggests that confidence signals might conflate other factors like overconfidence, and therefore do not reliably reflect inference-time compute effort or problem solving effectiveness.

In contrast, our proposed measurement of DTR demonstrates the strongest and most stable relationship with task performance, achieving the highest average correlation of $r = 0.683$, outperforming both reverse token count and Self-Certainty, the best-performing baselines among confidence-based approaches. Overall, DTR remains positive across models and benchmarks, exhibiting the fewest negative values (2 out of the 32 model–benchmark settings tested). Collectively, the results show that computing DTR over output sequences provides a more faithful and robust characterization of successful reasoning outcomes than token volume alone or confidence-based alternatives.

### 3.2. Effect of Settling Thresholds and Depth Fractions

We conduct an analysis to understand how our two key hyper-parameters—the settling threshold $g$ and the late-settling depth fraction $\rho$—affect the measured thinking effort and its correlation with task performance.

We conclude the following observations:

1. The magnitude of the measured sequence-level thinking effort is directly influenced by the strictness of these parameters. Specifically, imposing stricter criteria—a higher settling threshold $g$ or a lower depth fraction $\rho$—results in a reduction of the average late-settling token ratio.

2. The settling threshold $g$ has a more pronounced impact on the correlation between thinking effort and accuracy than the depth fraction $\rho$. Varying $\rho$ shifts the range of late-settling ratios due to varying strictness but maintains a consistent, positive slope across all settings. In contrast, the choice of $g$ has more impact on measured results.

3. Overall, we can see that when the criteria are overly restrictive ($g = 0.75$ and $\rho \in \{0.9, 0.95\}$), the trends, while still maintaining positive correlations, appears to be slightly more unstable due to the potential filtering of informative high computational tokens. Among the tested configurations, $(g, \rho) = (0.5, 0.85)$ strikes an ideal balance, yielding a reliable trend with high correlation values.

---

## 4. Deep-Thinking Tokens Enable Efficient Test-Time Scaling

Repeated sampling is a popular strategy for scaling test-time compute, in parallel to generating long CoT (Brown et al., 2024; Gupta and Srikumar, 2025; Saad-Falcon et al., 2024, 2025; Stroebl et al., 2024). It improves accuracy by aggregating multiple independently generated samples per problem at the cost of increased inference budget. In this section, we explore whether our proposed DTR measure can be leveraged to preferentially select and aggregate higher-quality samples towards better performance.

### Experimental Setups

We follow the best-of-n (BoN) evaluation protocol commonly adopted in recent test-time scaling studies (Fu et al., 2025). For each problem, we sample $n$ responses using identical decoding settings, and compare the following aggregation methods:

- **Cons@n**: Standard self-consistency (Wang et al., 2023), which performs majority voting over all $n$ sampled responses
- **Mean@n**: The average accuracy of all the $n$ samples, reflecting a baseline of no preferential aggregation
- **Long@n** and **Short@n**: Majority voting over the longest/shortest $\eta$ percent of the $n$ samples, ranked by token count (Agarwal et al., 2025; Hassid et al., 2025)
- **Self-Certainty@n**: Majority voting over the highest-scoring $\eta$ percent of the $n$ samples, ranked by Self-Certainty score (the best-performing baseline in Section 3)
- **Think@n**: Majority voting over the highest-scoring $\eta$ percent of the $n$ samples, ranked by DTR(·)

All methods operate on the same pool of $n$ samples. We set $n = 48$ and $\eta = 50\%$.

### Results

Table 2 reports the results:

| Method | AIME 25 |  | AIME 24 |  | HMMT 25 |  | GPQA-D |  |
|--------|---------|---------|---------|---------|---------|---------|---------|---------|
| | Acc | Cost (Δ%) | Acc | Cost (Δ%) | Acc | Cost (Δ%) | Acc | Cost (Δ%) |
| **OSS-120B-medium** |||||||||
| Cons@n | 92.7 | 307.6 (–) | 92.7 | 235.1 (–) | 80.0 | 355.6 (–) | 73.8 | 93.5 (–) |
| Mean@n | 80.0 | 307.6 (–) | 81.6 | 235.1 (–) | 62.6 | 355.6 (–) | 69.9 | 93.5 (–) |
| Long@n | 86.7 | 307.6 (–) | 86.7 | 235.1 (–) | 73.3 | 355.6 (–) | 73.2 | 93.5 (–) |
| Short@n | 87.3 | 255.7 (-17%) | 88.0 | 200.9 (-15%) | 77.3 | 290.4 (-18%) | 73.3 | 84.4 (-10%) |
| Self-Certainty@n† | 87.3 | 150.6 (-51%) | 91.3 | 119.3 (-49%) | 78.0 | 177.0 (-50%) | 76.0 | 47.9 (-49%) |
| Think@n† | **94.7** | 155.4 (-49%) | **93.3** | 121.3 (-48%) | **80.0** | 181.9 (-49%) | 74.7 | 48.8 (-48%) |
| **Qwen3-4B-Thinking** |||||||||
| Cons@n | 86.7 | 1073.1 (–) | 93.3 | 950.1 (–) | 63.3 | 1275.7 (–) | 67.8 | 410.6 (–) |
| Mean@n | 81.2 | 1073.1 (–) | 86.3 | 950.1 (–) | 55.7 | 1275.7 (–) | 66.9 | 410.6 (–) |
| Long@n | 85.3 | 1073.1 (–) | 86.7 | 950.1 (–) | 52.7 | 1275.7 (–) | 66.7 | 410.6 (–) |
| Short@n | 90.0 | 983.6 (-8%) | 90.0 | 871.0 (-8%) | 63.3 | 1165.7 (-9%) | 68.2 | 382.9 (-7%) |
| Self-Certainty@n† | 86.7 | 548.9 (-49%) | 90.0 | 480.9 (-49%) | 63.3 | 641.4 (-50%) | 68.2 | 206.6 (-50%) |
| Think@n† | **90.0** | 537.5 (-50%) | **93.3** | 482.2 (-49%) | **66.7** | 641.4 (-50%) | **69.7** | 206.8 (-50%) |

As shown, Cons@n incurs the highest inference cost due to full decoding of every candidate, while providing a strong accuracy baseline. Mean@n has the same cost as Cons@n but is the worst-performing one among all methods. Under early stopping, Short@n achieves modest cost savings relative to Cons@n, yet consistently underperforms it in accuracy. Long@n exhibits further degraded performance compared to Short@n without offering any cost-saving benefits. This indicates that length-based heuristics remain a coarse proxy for reasoning quality and often fail to reliably identify high-quality samples, leading to suboptimal aggregations. Self-Certainty@n substantially reduces inference cost by enabling early stopping using short prefixes, but nonetheless underperforms both Cons@n and Think@n on three of the four evaluated benchmarks. In contrast, Think@n consistently matches or exceeds the accuracy of Cons@n while requiring approximately half the inference cost. The Pareto-optimal performance is most evident in the averaged results, where Think@n achieves the best overall accuracy-cost trade-off.

### Prefix Length Analysis

Table 3 reports a preliminary ablation on AIME 25 that varies $\ell_{\text{prefix}}$. We find that using only $\ell_{\text{prefix}} = 50$ tokens achieves higher accuracy than longer prefixes and matches the performance obtained using the full sequence, while significantly reducing inference cost.

| | Accuracy | Cost (k tokens) |
|---|---------|-----------------|
| Pass@1 | 80.0±4.2 | 6.4 |
| Cons@n | 90.0±2.5 | 307.6 |
| Think@n (prefix=50) | **94.7**±1.6 | 155.4 |
| Think@n (prefix=100) | 92.0±1.6 | 154.1 |
| Think@n (prefix=500) | 92.7±1.3 | 153.2 |
| Think@n (prefix=1000) | 92.7±1.3 | 177.4 |
| Think@n (prefix=2000) | 92.0±1.3 | 198.8 |
| Think@n (all) | 94.0±0.3 | 307.6 |

---

## 5. Related Work

### 5.1. Relationship between CoT Length and Performance

The paradigm of test-time scaling has largely operated on the assertion that allocating more computation, typically manifested as longer CoT sequences, boosts reasoning performance (Guo et al., 2025; Muennighoff et al., 2025; Wei et al., 2022). Recent empirical studies have highlighted nuances to the universality of this "longer is better" heuristic (Feng et al., 2025; Wu et al., 2025). Gema et al. (2025) identify inverse scaling regimes where increased reasoning length systematically degrades accuracy across diverse tasks, particularly when models are prone to distraction. Similarly, Wu et al. (2025) characterize the relationship between CoT length and accuracy as an "inverted-U" curve, suggesting an optimal length exists beyond which performance deteriorates due to factors like error accumulation.

Several works have proposed methods to exploit corresponding observations by favoring conciseness. Hassid et al. (2025) demonstrated that the shortest reasoning chains among sampled candidates are often the most accurate, proposing inference-time length-based voting for efficient generations. A close work by Agarwal et al. (2025) also introduced a training-free strategy that selects the first completed trace in parallel decoding, reducing token usage while maintaining accuracy. On the training side, Shrivastava et al. (2025) proposed Group Filtered Policy Optimization (GFPO) to explicitly curb length inflation in RL by rejection sampling that filters longer responses, demonstrating that models can think less without sacrificing performance. Our work aligns with these perspectives by confirming that raw token count is an unreliable proxy for effective reasoning effort, but we diverge by proposing a mechanistic internal signal rather than simply relying on surface-level brevity heuristics.

### 5.2. Leveraging Internal Information in LLMs

A rich line of work has investigated how LMs internally represent and manipulate information across layers, and how internal states can be exploited. Central to this direction is the observation that intermediate representations in LMs often encode meaningful signals before reaching the final layer. Early evidence for this view was provided by Nostalgebraist (2020), which projects intermediate hidden states directly into the vocabulary space using the model's unembedding matrix—a technique we adopt in our work. The results reveal that autoregressive transformers form coarse guesses about the next token that are iteratively refined across layers. Subsequent analyses (Belrose et al., 2023) further introduce learned, layer-specific affine transformations that better align intermediate representations with the final prediction space, enabling more interpretable token predictions in shallower layers.

Beyond model probing, Chuang et al. (2023) exploits the empirical finding that factual knowledge in LMs is often more salient in particular layers. By contrasting logits from higher and lower layers, they propose a decoding method that amplifies factual signals and improves factuality. A recent work by Vilas et al. (2025) introduces latent-trajectory signals characterizing the temporal evolution of hidden states across generated reasoning traces to predict correctness. While the work examines the sequential dimension of representations, our work focuses on the depth-wise evolution of predictions across layers for individual tokens.

Complementary interpretability works also revisit how LLMs utilize depth at inference. Gupta et al. (2025) shows that early layers tend to favor high-frequency, generic token guesses, which are subsequently refined into contextually appropriate predictions. Csordás et al. (2025) suggest that later layers primarily perform fine-grained distributional refinement rather than introducing fundamentally new transformations, raising questions about the efficiency of depth utilization in modern LLMs. These findings reinforce the view that internal predictions may stabilize before the final layer, aligning with our motivations. Overall, our goal is not to modify or construct internal states to develop new methods aimed at improving model capabilities. Instead, we leverage natural, unaltered internal representations as a proxy for measuring model computational effort, which implicitly reflects thinking effort in LLMs.

---

## 6. Conclusion

We introduced **deep-thinking ratio (DTR)** as a novel measure of inference-time reasoning effort in LLMs. By tracking depth-wise stabilization of token predictions, DTR provides a more reliable signal of effective reasoning than surface-level proxies such as token length or confidence. Building on this insight, we proposed **Think@n**, a test-time scaling strategy that leverages DTR for early selection and aggregation, achieving comparable or better performance than standard self-consistency while substantially reducing inference cost. Together, our results suggest that measuring how models think internally, rather than how long they think, is a promising direction. Future work may leverage this insight to explore how effective reasoning is characterized—shifting the focus from generating longer chains of thought to inducing deeper, more computationally intensive reasoning, and potentially enabling more reliable and efficient reasoning models.

---

## Acknowledgements

We thank Congchao Wang and colleagues from Google AIR for their valuable support. We also thank Yu-Min Tseng from Virginia Tech and members of Meng-Lab at UVA for their helpful discussion.

---

## References

1. A. Agarwal, A. Sengupta, and T. Chakraborty. First finish search: Efficient test-time scaling in large language models. arXiv preprint arXiv:2505.18149, 2025.

2. P. Aggarwal, S. Kim, J. Lanchantin, S. Welleck, J. Weston, I. Kulikov, and S. Saha. OptimalThinking-Bench: Evaluating over and underthinking in LLMs. arXiv, 2025.

3. Anthropic. Claude 3.7 sonnet system card, 2025a.

4. Anthropic. System card: Claude opus 4 & claude sonnet 4, 2025b.

5. Art of Problem Solving. 2024 AIME I, 2024a.

6. Art of Problem Solving. 2024 AIME II, 2024b.

7. Art of Problem Solving. 2025 AIME I, 2025a.

8. Art of Problem Solving. 2025 AIME II, 2025b.

9. V. Balachandran, et al. Inference-time scaling for complex tasks: Where we stand and what lies ahead. arXiv, 2025.

10. M. Balunović, et al. Matharena: Evaluating LLMs on uncontaminated math competitions. arXiv preprint arXiv:2505.23281, 2025.

11. N. Belrose, et al. Eliciting latent predictions from transformers with the tuned lens. arXiv preprint arXiv:2303.08112, 2023.

12. B. Brown, et al. Large language monkeys: Scaling inference compute with repeated sampling. arXiv, 2024.

13. Y.-S. Chuang, et al. DoLa: Decoding by contrasting layers improves factuality in large language models. arXiv, 2023.

14. R. Csordás, et al. Do language models use their depth efficiently? arXiv, 2025.

15. A. Y. Din, et al. Jump to conclusions: Short-cutting transformers with linear transformations. In LREC-COLING 2024, 2024.

16. M. Elbayad, et al. Depth-adaptive transformer. arXiv preprint arXiv:1910.10073, 2019.

17. Y. Feng, et al. What characterizes effective reasoning? arXiv, 2025.

18. Y. Fu, et al. Deep think with confidence. arXiv preprint arXiv:2508.15260, 2025.

19. A. P. Gema, et al. Inverse scaling in test-time compute. arXiv, 2025.

20. D. Guo, et al. Deepseek-r1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

21. A. Gupta and V. Srikumar. Test-time scaling with repeated sampling improves multilingual text generation. arXiv, 2025.

22. A. Gupta, et al. How do LLMs use their depth? arXiv, 2025.

23. M. Hassid, et al. Don't overthink it: Preferring shorter thinking chains for improved LLM reasoning. arXiv preprint arXiv:2505.17813, 2025.

24. HMMT. HMMT 2025, 2025.

25. A. Jaech, et al. OpenAI o1 system card. arXiv preprint arXiv:2412.16720, 2024.

26. Z. Kang, et al. Scalable best-of-n selection for large language models via self-certainty. arXiv, 2025.

27. W.-T. Kao, et al. BERT's output layer recognizes all hidden layers? arXiv preprint arXiv:2001.09309, 2020.

28. N. Muennighoff, et al. s1: Simple test-time scaling. In EMNLP 2025, 2025.

29. Nostalgebraist. Interpreting GPT: The logit lens. LessWrong, 2020.

30. OpenAI. OpenAI o3-mini system card, 2025.

31. OpenAI. Introducing GPT-5, 2025.

32. OpenAI, et al. GPT-OSS-120B & GPT-OSS-20B model card. arXiv, 2025.

33. D. Rein, et al. GPQA: A graduate-level Google-proof Q&A benchmark. In COLM, 2024.

34. J. Saad-Falcon, et al. Archon: An architecture search framework for inference-time techniques. arXiv, 2024.

35. J. Saad-Falcon, et al. Shrinking the generation-verification gap with weak verifiers. arXiv, 2025.

36. T. Schuster, et al. Confident adaptive language modeling. NeurIPS, 2022.

37. V. Shrivastava, et al. Sample more to think less: Group filtered policy optimization for concise reasoning. arXiv preprint arXiv:2508.09726, 2025.

38. B. Stroebl, et al. Inference scaling laws: The limits of LLM resampling with imperfect verifiers. arXiv, 2024.

39. J. Su, et al. Between underthinking and overthinking: An empirical study of reasoning length and correctness in LLMs. arXiv, 2025.

40. Y. Sui, et al. Stop overthinking: A survey on efficient reasoning for large language models. arXiv, 2025.

41. K. Team, et al. Kimi k1. 5: Scaling reinforcement learning with LLMs. arXiv preprint arXiv:2501.12599, 2025.

42. S. Teerapittayanon, et al. BranchyNet: Fast inference via early exiting from deep neural networks. In ICPR, 2016.

43. M. G. Vilas, et al. Tracing the traces: Latent temporal signals for efficient and accurate reasoning. arXiv, 2025.

44. X. Wang, et al. Self-consistency improves chain of thought reasoning in language models. In ICLR, 2023.

45. J. Wei, et al. Chain of thought prompting elicits reasoning in large language models. arXiv, 2022.

46. Y. Wu, et al. When more is less: Understanding chain-of-thought length in LLMs. arXiv, 2025.

47. xAI. Grok 4, 2025.

48. A. Yang, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

49. E. Yeo, et al. Demystifying long chain-of-thought reasoning in LLMs. arXiv, 2025.

50. T. Zhong, et al. Evaluation of OpenAI o1: Opportunities and challenges of AGI. arXiv preprint arXiv:2409.18486, 2024.

---

## Appendix A: Comparison of Different Distance Metrics for DTR

Our method adopts Jensen–Shannon divergence (JSD) to quantify the discrepancy between intermediate-layer and final-layer predictions and compute DTR. Alternative notions of distance are possible. Here we explore two additional metrics: Kullback–Leibler divergence (KLD) and cosine similarity.

### Kullback–Leibler divergence

By replacing JSD with KLD, we compute the divergence between the final-layer distribution $p_{t,L}$ and the intermediate-layer distribution $p_{t,l}$ as:

$$D_{t,l} = \text{KL}(p_{t,L} \| p_{t,l})$$

### Cosine similarity

We replace the distributional comparison with a representation-space measure using cosine similarity. Instead of projecting intermediate-layer hidden states into the vocabulary space via the shared unembedding matrix $W_U$, we directly compute the cosine similarity between the intermediate-layer hidden state $h_{t,l}$ and the final-layer hidden state $h_{t,L}$. The distance is defined as:

$$D_{t,l} = 1 - \frac{\langle h_{t,l}, h_{t,L} \rangle}{\| h_{t,l} \| \| h_{t,L} \|}$$

For both KLD and cosine similarity, we then apply the same configurations to identify deep-thinking tokens and compute KLD-based DTR and cosine-based DTR.

### Results

Across both datasets, JSD-based DTR consistently achieves the strongest positive correlation with accuracy ($r = 0.869$ on AIME 25; $r = 0.895$ on HMMT 25), justifying its use in our definition of DTR. In contrast, cosine-based DTR exhibits substantially weaker and unstable correlations ($r = 0.633$ on AIME 25 and only $r = 0.172$ on HMMT 25). KLD-based DTR shows similarly inconsistent behavior, with a negative correlation on AIME 25 ($r = -0.698$) and a modest positive correlation on HMMT 25 ($r = 0.409$). This inconsistency may stem from the asymmetric and numerically unstable nature of KLD.

---

## Appendix B: DTR Under Different GPT-OSS Reasoning Levels

Figure 7 illustrates how DTR varies in different reasoning-level configurations (i.e., low, medium, and high) of the GPT-OSS-120B model. We observe an interesting and consistent trend on both AIME 25 and GPQA-D: although the underlying model weights remain identical and only the system prompt differs, lower reasoning-level configurations exhibit higher DTR values, whereas higher reasoning-level configurations yield systematically smaller DTR while achieving better task accuracy.

A potential explanation is that higher reasoning levels may redistribute computation from depth to sequence length, effectively flattening per-token, layer-wise computation. Models with higher reasoning levels require less deep revision for each individual token but instead generate longer reasoning chains with more forward passes, resulting in greater total effective compute and improved task performance.

---

## Appendix C: Additional Analysis of Think@n

Here we provide additional analysis on how Think@n behaves when varying (i) the number of sampled responses $n$ and (ii) the retained top-$\eta$ percentage used for voting.

### Effect of the number of samples n

Think@n improves monotonically with larger $n$, where the advantage over Cons@n becomes more pronounced. Sampling more responses makes the correct cluster of answers to be larger and more likely to appear. Think@n is able to exploit this enlarged candidate pool by preferentially selecting better samples, leading to stronger performance gains over Cons@n.

### Effect of top-η percentage

Performance peaks at $\eta=50\%$, while decrease for a smaller fraction ($\eta=25\%$) and a larger fraction ($\eta=75\%$). This suggests a trade-off: selecting too few samples reduces voting robustness, potentially with fewer strong candidates to stabilize majority vote, whereas selecting too many might admit lower-quality samples that dilute the benefit of Think@n.

---

## Appendix D: Prompts

### Prompt for AIME 2024, AIME 2025, HMMT 2025

> Please reason step by step, and put your final answer within \boxed{}.

### Prompt for GPQA

> You will be given a multiple choice question with different choices such as (A), (B), (C), (D). Think step by step before giving a final answer to this question. Always finish your answer with 'The final answer is \boxed{(X)}.', where X is the correct answer choice. If none of the options match, choose the closest option as the final answer.

---

## Appendix E: Qualitative Examples

We present an example question from the AIME 2025 dataset along with its ground-truth answer, and two outputs from OSS-120-medium: one incorrect and one correct. Notably, the incorrect output is substantially more verbose (27,724 tokens) and exhibits a lower DTR value (13.9%), whereas the correct output is much more concise (3,725 tokens) and achieves a higher DTR value (19.0%).

### Example Question from AIME 2025

Circle $\omega_1$ with radius 6 centered at point $A$ is internally tangent at point $B$ to circle $\omega_2$ with radius 15. Points $C$ and $D$ lie on $\omega_2$ such that $BC$ is a diameter of $\omega_2$ and $BC \perp AD$. The rectangle $EFGH$ is inscribed in $\omega_1$ such that $EF \perp BC$, $C$ is closer to $GH$ than to $EF$, and $D$ is closer to $FG$ than to $EH$, as shown. Triangles $\triangle DGF$ and $\triangle CHG$ have equal areas. The area of rectangle $EFGH$ is $mn$, where $m$ and $n$ are relatively prime positive integers. Find $m + n$.

**Ground truth answer:** 293
