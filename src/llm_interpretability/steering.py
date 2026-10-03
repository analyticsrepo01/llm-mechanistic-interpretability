"""
Activation Steering & Representation Engineering
=================================================
Extracts persona and behavioral steering vectors via contrastive activations
and injects them at test-time to modulate model outputs without fine-tuning
(Rimsky et al., 2023; Turner et al., 2023; Zou et al., 2023).
"""

from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn


class SteeringHook:
    """Removable hook that adds a steering vector to a target layer's residual stream."""

    def __init__(self, steering_vector: torch.Tensor, coefficient: float = 1.0):
        self.steering_vector = steering_vector
        self.coefficient = coefficient
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def __call__(self, module: nn.Module, input: Any, output: Any):
        vec = (self.coefficient * self.steering_vector).to(
            device=output[0].device if isinstance(output, tuple) else output.device,
            dtype=output[0].dtype if isinstance(output, tuple) else output.dtype,
        )

        if isinstance(output, tuple):
            steered_tensor = output[0] + vec
            return (steered_tensor,) + output[1:]
        return output + vec

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class ActivationSteering:
    """
    Computes contrastive steering directions and injects them into model layers.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.active_hooks: List[SteeringHook] = []

    def compute_steering_vector(
        self,
        positive_activations: torch.Tensor,
        negative_activations: torch.Tensor,
        token_index: int = -1,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Computes the contrastive steering vector: v = mean(A_pos) - mean(A_neg).
        """
        if positive_activations.dim() == 3:
            pos = positive_activations[:, token_index, :].mean(dim=0)
        else:
            pos = positive_activations.mean(dim=0)

        if negative_activations.dim() == 3:
            neg = negative_activations[:, token_index, :].mean(dim=0)
        else:
            neg = negative_activations.mean(dim=0)

        diff = pos - neg
        if normalize:
            norm = diff.norm(p=2)
            if norm > 0:
                diff = diff / norm
        return diff

    def attach_steering(
        self,
        layer: nn.Module,
        steering_vector: torch.Tensor,
        coefficient: float = 1.0,
    ) -> SteeringHook:
        """
        Attaches a steering vector intervention to a specific transformer layer.
        """
        hook = SteeringHook(steering_vector=steering_vector, coefficient=coefficient)
        hook.handle = layer.register_forward_hook(hook)
        self.active_hooks.append(hook)
        return hook

    def clear_steering(self):
        """Removes all runtime steering hooks."""
        for hook in self.active_hooks:
            hook.remove()
        self.active_hooks.clear()
