"""
Outlier Feature Detection & Quantization Bottlenecks
====================================================
Detects high-magnitude activation dimensions in transformer residual streams
as described by Dettmers et al. (LLM.int8(), NeurIPS 2022).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch


@dataclass
class OutlierReport:
    """Outlier dimensions identified within a layer's activations."""
    layer_name: str
    outlier_dimensions: List[int]
    max_magnitudes: List[float]
    global_max: float
    threshold: float
    outlier_fraction: float


class OutlierDetector:
    """
    Scans activations to identify outlier coordinates that dominate attention and MLP computations.
    """

    def __init__(self, std_threshold: float = 6.0, absolute_threshold: float = 6.0):
        self.std_threshold = std_threshold
        self.absolute_threshold = absolute_threshold

    def analyze_layer(self, activation: torch.Tensor, layer_name: str = "layer") -> OutlierReport:
        """
        Identifies outlier dimensions across sequence length and batch tokens.
        
        Args:
            activation: Tensor of shape (batch, seq_len, hidden_dim) or (tokens, hidden_dim)
            layer_name: Name of the layer
        """
        if activation.dim() == 3:
            # Flatten to (batch * seq_len, hidden_dim)
            flat = activation.reshape(-1, activation.shape[-1]).abs()
        else:
            flat = activation.abs()

        # Compute max magnitude per dimension
        max_per_dim, _ = flat.max(dim=0)
        mean_val = max_per_dim.mean().item()
        std_val = max_per_dim.std().item()

        threshold = max(self.absolute_threshold, mean_val + self.std_threshold * std_val)

        outlier_mask = max_per_dim > threshold
        outlier_indices = torch.where(outlier_mask)[0].tolist()
        outlier_values = max_per_dim[outlier_mask].tolist()

        return OutlierReport(
            layer_name=layer_name,
            outlier_dimensions=outlier_indices,
            max_magnitudes=outlier_values,
            global_max=flat.max().item(),
            threshold=threshold,
            outlier_fraction=len(outlier_indices) / flat.shape[-1],
        )

    def scan_network(self, activations: Dict[str, torch.Tensor]) -> Dict[str, OutlierReport]:
        """Scans all layers for persistent outlier dimensions across model depth."""
        return {name: self.analyze_layer(t, layer_name=name) for name, t in activations.items()}
