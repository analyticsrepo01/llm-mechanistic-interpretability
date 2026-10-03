"""
Direct Logit Attribution (DLA) & Residual Stream Decomposition
==============================================================
Decomposes target token logits into additive contributions from attention heads
and MLP blocks across transformer depth (Elhage et al., Anthropic 2021; Nanda, 2022).
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn


class DirectLogitAttribution:
    """
    Measures the direct contribution of individual layer outputs to the final target token logit.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            self.final_norm = model.model.norm
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            self.final_norm = model.transformer.ln_f
        else:
            self.final_norm = nn.Identity()

        if hasattr(model, "lm_head"):
            self.lm_head = model.lm_head
        else:
            raise AttributeError("Model must have an lm_head.")

    @torch.no_grad()
    def attribute_target_token(
        self,
        layer_outputs: Dict[str, torch.Tensor],
        target_token_id: int,
        token_index: int = -1,
    ) -> Dict[str, float]:
        """
        Calculates how strongly each layer directly pushes for (or against) the target token.
        
        Args:
            layer_outputs: Dict of layer output tensors
            target_token_id: The vocabulary token ID to attribute
            token_index: Token position in sequence
        """
        device = next(self.lm_head.parameters()).device
        dtype = next(self.lm_head.parameters()).dtype

        # Unembedding direction for target token: shape (hidden_dim,)
        # lm_head weight has shape (vocab_size, hidden_dim)
        w_u = self.lm_head.weight[target_token_id].to(device=device, dtype=dtype)

        attributions = {}
        for name, tensor in layer_outputs.items():
            if tensor.dim() == 3:
                h = tensor[:, token_index, :].to(device=device, dtype=dtype)
            else:
                h = tensor.to(device=device, dtype=dtype)

            # Direct projection onto unembedding direction
            direct_contrib = torch.matmul(h, w_u).mean().item()
            attributions[name] = direct_contrib

        return attributions
