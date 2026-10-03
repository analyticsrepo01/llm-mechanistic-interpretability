"""
Representation Geometry & Manifold Analysis
===========================================
Performs dimensionality reduction (PCA, Cosine distance) across model layers
to visualize semantic clustering and concept separation across transformer depth.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch


class RepresentationGeometry:
    """
    Analyzes high-dimensional manifold geometry across transformer layers.
    """

    @staticmethod
    def compute_layer_pca(
        activation_matrix: np.ndarray,
        n_components: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes PCA projection using SVD.
        
        Args:
            activation_matrix: Shape (n_samples, hidden_dim)
            n_components: Number of principal components
            
        Returns:
            (projected_points, explained_variance_ratio)
        """
        # Center data
        mean = np.mean(activation_matrix, axis=0)
        centered = activation_matrix - mean

        # SVD
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:n_components]
        projected = np.dot(centered, components.T)

        variance = (s ** 2) / (len(activation_matrix) - 1)
        total_var = np.sum(variance)
        explained_variance_ratio = variance[:n_components] / total_var if total_var > 0 else np.zeros(n_components)

        return projected, explained_variance_ratio

    @staticmethod
    def compute_consecutive_cosine_similarity(
        layer_activations: List[torch.Tensor],
        token_index: int = -1,
    ) -> List[float]:
        """
        Computes cosine similarity between layer l and layer l+1 to track token trajectory velocity.
        """
        similarities = []
        for i in range(len(layer_activations) - 1):
            act_i = layer_activations[i]
            act_next = layer_activations[i + 1]

            if act_i.dim() == 3:
                v_i = act_i[:, token_index, :].reshape(act_i.shape[0], -1).float()
                v_next = act_next[:, token_index, :].reshape(act_next.shape[0], -1).float()
            else:
                v_i = act_i.reshape(act_i.shape[0], -1).float()
                v_next = act_next.reshape(act_next.shape[0], -1).float()

            cos_sim = torch.cosine_similarity(v_i, v_next, dim=-1).mean().item()
            similarities.append(float(cos_sim))

        return similarities
