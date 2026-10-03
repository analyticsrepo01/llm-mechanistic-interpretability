"""
Centered Kernel Alignment (CKA) & Layer Redundancy Analysis
===========================================================
Measures representational similarity across transformer layers using linear CKA
(Kornblith et al., ICML 2019; Men et al., ShortGPT 2024).
"""

from typing import Dict, List, Tuple
import numpy as np
import torch


class CKAAnalyzer:
    """
    Computes Linear Centered Kernel Alignment (CKA) to evaluate representation similarity
    between layer representations and identify candidate layers for pruning.
    """

    @staticmethod
    def _centering_matrix(n: int, device: torch.device) -> torch.Tensor:
        """Constructs the centering matrix H = I - (1/n) * 11^T."""
        eye = torch.eye(n, device=device)
        ones = torch.ones((n, n), device=device) / n
        return eye - ones

    @classmethod
    def linear_cka(cls, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Computes Linear CKA between two representation matrices.
        
        Args:
            x: Tensor of shape (n_samples, d1)
            y: Tensor of shape (n_samples, d2)
            
        Returns:
            Scalar similarity score in [0.0, 1.0]
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Sample count must match: {x.shape[0]} vs {y.shape[0]}")

        x = x.float()
        y = y.float()

        # Center columns
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

        # Frobenious inner products of Gram matrices: Tr(X X^T Y Y^T) = ||Y^T X||_F^2
        similarity = torch.norm(torch.matmul(y.t(), x), p="fro") ** 2
        norm_x = torch.norm(torch.matmul(x.t(), x), p="fro")
        norm_y = torch.norm(torch.matmul(y.t(), y), p="fro")

        if norm_x == 0 or norm_y == 0:
            return 0.0

        cka_score = (similarity / (norm_x * norm_y)).item()
        return float(np.clip(cka_score, 0.0, 1.0))

    def compute_similarity_matrix(
        self,
        activations: Dict[str, torch.Tensor],
        token_index: int = -1,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Computes the pairwise CKA similarity matrix across all recorded layers.
        
        Returns:
            (layer_names, similarity_matrix)
        """
        layer_names = list(activations.keys())
        n_layers = len(layer_names)
        matrix = np.zeros((n_layers, n_layers), dtype=np.float32)

        # Extract 2D representations (n_samples, hidden_dim)
        reps = []
        for name in layer_names:
            act = activations[name]
            if act.dim() == 3:
                # Use specified token position across batch, or flatten
                rep = act[:, token_index, :].reshape(act.shape[0], -1)
            else:
                rep = act.reshape(act.shape[0], -1)
            reps.append(rep)

        for i in range(n_layers):
            matrix[i, i] = 1.0
            for j in range(i + 1, n_layers):
                score = self.linear_cka(reps[i], reps[j])
                matrix[i, j] = score
                matrix[j, i] = score

        return layer_names, matrix
