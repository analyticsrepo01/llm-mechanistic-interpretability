# Mechanistic Interpretability: Mathematical & Theoretical Foundations

This research repository implements core diagnostic and intervention techniques designed to reverse-engineer transformer hidden states, residual streams, and attention mechanisms.

---

## 1. Residual Stream Formulation

Following the transformer circuits framework (Elhage et al., Anthropic 2021), a decoder-only transformer operates as a shared communication channel called the **residual stream** $x \in \mathbb{R}^{d_{model}}$. 

At each transformer block $l \in \{1, \dots, L\}$, attention and MLP sub-layers write their outputs additively into the stream:

$$x_{l} = x_{l-1} + a_l(x_{l-1}) + m_l(x_{l-1} + a_l(x_{l-1}))$$

where:
* $a_l(\cdot)$ is the multi-head attention block output.
* $m_l(\cdot)$ is the multi-layer perceptron (SwiGLU / GeLU) output.
* The final logit distribution is obtained after root-mean-square normalization (RMSNorm) and projection by the unembedding matrix $W_U \in \mathbb{R}^{|V| \times d_{model}}$:

$$\text{logits} = \text{RMSNorm}(x_L) W_U^T$$

---

## 2. The Logit Lens

First introduced by nostalgebraist (2020) and formalized by Belrose et al. (2023), the **Logit Lens** decodes intermediate representations at layer $l$ into vocabulary logits without waiting for the full network depth:

$$\text{logits}_l = \text{RMSNorm}(x_l) W_U^T$$

$$\hat{y}_l = \text{softmax}(\text{logits}_l)$$

By computing the top-1 prediction and entropy $H(\hat{y}_l) = -\sum_v \hat{y}_{l,v} \log \hat{y}_{l,v}$ at each layer, we map the exact layer where semantic abstractions crystallize into confident token selections.

---

## 3. Outlier Feature Emergence (LLM.int8())

Dettmers et al. (NeurIPS 2022) showed that once language models exceed ~6.7B parameters, a small subset of hidden dimensions ($\approx 0.1\%$ of $d_{model}$) develop extreme activation magnitudes (often exceeding $100\times$ the standard deviation):

$$\mathcal{O}_l = \{ k \in \{1, \dots, d_{model}\} \mid \max_{t} |x_{l,t,k}| > \mu_k + 6\sigma_k \}$$

These outlier features act as internal coordinate anchors for multi-head attention and token tracking. Standard uniform INT8 quantization clips or truncates these features, causing severe perplexity degradation.

---

## 4. Centered Kernel Alignment (CKA)

To measure representational similarity across distinct layers without assuming coordinate-wise alignment, we compute linear **Centered Kernel Alignment** (Kornblith et al., ICML 2019):

$$\text{CKA}(X, Y) = \frac{\|Y^T X\|_F^2}{\|X^T X\|_F \|Y^T Y\|_F}$$

where $X \in \mathbb{R}^{N \times d}$ and $Y \in \mathbb{R}^{N \times d}$ are centered representation matrices across $N$ sample tokens. 

### Properties:
1. **Orthogonal Invariance:** $\text{CKA}(X U, Y V) = \text{CKA}(X, Y)$ for any orthogonal matrices $U, V$.
2. **Isotropic Scaling Invariance:** $\text{CKA}(\alpha X, \beta Y) = \text{CKA}(X, Y)$ for scalar constants $\alpha, \beta > 0$.
3. **Layer Redundancy Detection:** Adjacent layers exhibiting $\text{CKA} \ge 0.95$ share redundant representations, providing candidate layers for pruning (Men et al., ShortGPT 2024).

---

## 5. Direct Logit Attribution (DLA)

Because the residual stream is strictly additive, any target token logit $L_v$ can be decomposed into the exact sum of contributions from the embedding layer, each attention head, and each MLP block:

$$L_v = \left( x_0 + \sum_{l=1}^L a_l + \sum_{l=1}^L m_l \right) \cdot W_{U,v}$$

Direct Logit Attribution computes the scalar projection of each component's output onto the target token's unembedding direction $W_{U,v}$, quantifying whether that layer promoted ($> 0$) or suppressed ($< 0$) the final token choice.

---

## 6. Contrastive Activation Addition (CAA) & Neural Steering

Activation steering modulates high-level concepts (e.g. sentiment, persona, truthfulness) at test-time without modifying model weights (Rimsky et al., 2023; Zou et al., 2023).

Given a set of contrastive prompt pairs $(P^+_i, P^-_i)$, we capture the residual activations at layer $l$:

$$v_l = \frac{1}{M} \sum_{i=1}^M x_l(P^+_i) - \frac{1}{M} \sum_{i=1}^M x_l(P^-_i)$$

$$\hat{v}_l = \frac{v_l}{\|v_l\|_2}$$

At inference time, we intervene on the residual stream at layer $l$ using steering coefficient $\alpha$:

$$x'_l = x_l + \alpha \cdot \hat{v}_l$$

This shifts model behavior dynamically during autoregressive token generation.

---

## 7. Key Literature & References

1. **Transformer Circuits:** Elhage et al., *"A Mathematical Framework for Transformer Circuits"*, Anthropic (2021).
2. **Logit Lens:** nostalgebraist, *"interpreting GPT: the logit lens"* (2020).
3. **Tuned Lens:** Belrose et al., *"Eliciting Latent Predictions from Transformers with the Tuned Lens"*, arXiv:2303.08112 (2023).
4. **Outlier Features:** Dettmers et al., *"LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"*, NeurIPS (2022).
5. **CKA Similarity:** Kornblith et al., *"Similarity of Neural Network Representations Revisited"*, ICML (2019).
6. **ShortGPT (Layer Pruning):** Men et al., *"ShortGPT: Layers in Large Language Models are More Redundant Than You Expect"*, arXiv:2403.03853 (2024).
7. **Representation Engineering:** Zou et al., *"Representation Engineering: A Top-Down Approach to AI Transparency"*, arXiv:2310.01405 (2023).
8. **Activation Addition (CAA):** Rimsky et al., *"Steering Llama 2 via Contrastive Activation Addition"*, arXiv:2312.06681 (2023).
