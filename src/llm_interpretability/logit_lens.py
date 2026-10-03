"""
Logit Lens & Intermediate Prediction Analysis
=============================================
Unembeds intermediate hidden state representations directly into vocabulary space
to observe when concepts and next-token predictions crystallize across network depth.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LogitLensResult:
    """Prediction distribution at a specific layer."""
    layer_name: str
    top_tokens: List[str]
    top_token_ids: List[int]
    top_probabilities: List[float]
    entropy: float


class LogitLens:
    """
    Applies the model's final normalization and language model head to intermediate layer outputs.
    """

    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

        # Locate final layer norm and LM head
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            self.final_norm = model.model.norm
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            self.final_norm = model.transformer.ln_f
        else:
            self.final_norm = nn.Identity()

        if hasattr(model, "lm_head"):
            self.lm_head = model.lm_head
        else:
            raise AttributeError("Model must have an lm_head to apply logit lens.")

    @torch.no_grad()
    def inspect_layer(
        self,
        hidden_state: torch.Tensor,
        layer_name: str = "layer",
        token_index: int = -1,
        top_k: int = 5,
    ) -> LogitLensResult:
        """
        Unembeds a single layer's hidden state for a specific token position.
        
        Args:
            hidden_state: Tensor of shape (batch, seq_len, hidden_dim)
            layer_name: Name of current layer
            token_index: Position in sequence (default -1 for next-token prediction)
            top_k: Number of candidate tokens to return
        """
        # Ensure device alignment
        device = next(self.lm_head.parameters()).device
        dtype = next(self.lm_head.parameters()).dtype

        # Slice target token: (batch, hidden_dim)
        if hidden_state.dim() == 3:
            h = hidden_state[:, token_index, :].to(device=device, dtype=dtype)
        else:
            h = hidden_state.to(device=device, dtype=dtype)

        # Apply final norm + LM head projection
        normed = self.final_norm(h)
        logits = self.lm_head(normed)

        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

        # Calculate prediction entropy
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean().item()

        first_sample_indices = top_indices[0].tolist()
        first_sample_probs = top_probs[0].tolist()
        tokens = [self.tokenizer.decode([idx]) for idx in first_sample_indices]

        return LogitLensResult(
            layer_name=layer_name,
            top_tokens=tokens,
            top_token_ids=first_sample_indices,
            top_probabilities=first_sample_probs,
            entropy=entropy,
        )

    def analyze_trajectory(
        self,
        activations: Dict[str, torch.Tensor],
        token_index: int = -1,
        top_k: int = 3,
    ) -> List[LogitLensResult]:
        """
        Runs Logit Lens across all captured layer activations.
        """
        results = []
        for name, tensor in activations.items():
            res = self.inspect_layer(tensor, layer_name=name, token_index=token_index, top_k=top_k)
            results.append(res)
        return results
