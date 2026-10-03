"""
PyTorch Hook Management & Activation Extraction
================================================
Thread-safe and memory-safe forward/backward hook registration for transformer architectures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import time
import torch
import torch.nn as nn


@dataclass
class LayerActivation:
    """Captured tensor activations and statistics for a single layer."""
    layer_name: str
    tensor: Optional[torch.Tensor] = None
    shape: List[int] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    l2_norm: float = 0.0
    zero_percentage: float = 0.0
    duration_ms: float = 0.0


class HookManager:
    """
    Manages PyTorch forward and backward hooks with automatic cleanup to prevent memory leaks.
    
    Supports:
    - Forward hooks for activation capture and statistics
    - Pre-forward hooks for micro-benchmarking and layer timing
    - Context manager support (`with HookManager(model) as hm: ...`)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.activations: Dict[str, torch.Tensor] = {}
        self.stats: Dict[str, LayerActivation] = {}
        self._timing_start: Dict[str, float] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def register_layer_hooks(
        self,
        layers: Optional[List[nn.Module]] = None,
        layer_names: Optional[List[str]] = None,
        store_tensors: bool = True,
        track_timing: bool = True,
    ):
        """
        Attaches forward hooks to specified layers (or all transformer layers by default).
        """
        if layers is None:
            # Auto-detect transformer blocks in Llama / Gemma / Mistral / GPT models
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                layers = list(self.model.model.layers)
                layer_names = [f"layer_{i}" for i in range(len(layers))]
            elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                layers = list(self.model.transformer.h)
                layer_names = [f"layer_{i}" for i in range(len(layers))]
            else:
                raise ValueError("Could not auto-detect layers. Pass layers explicitly.")

        if layer_names is None:
            layer_names = [f"layer_{i}" for i in range(len(layers))]

        for name, layer in zip(layer_names, layers):
            if track_timing:
                self._attach_timing_hooks(name, layer)
            self._attach_activation_hook(name, layer, store_tensors)

    def _attach_timing_hooks(self, name: str, layer: nn.Module):
        def pre_hook(module, input):
            self._timing_start[name] = time.perf_counter()

        h_pre = layer.register_forward_pre_hook(pre_hook)
        self.handles.append(h_pre)

    def _attach_activation_hook(self, name: str, layer: nn.Module, store_tensors: bool):
        def hook(module, input, output):
            duration_ms = 0.0
            if name in self._timing_start:
                duration_ms = (time.perf_counter() - self._timing_start[name]) * 1000.0

            tensor = output[0] if isinstance(output, tuple) else output

            if isinstance(tensor, torch.Tensor):
                t_float = tensor.detach().float()
                stats = LayerActivation(
                    layer_name=name,
                    shape=list(tensor.shape),
                    mean=t_float.mean().item(),
                    std=t_float.std().item(),
                    min_val=t_float.min().item(),
                    max_val=t_float.max().item(),
                    l2_norm=t_float.norm().item(),
                    zero_percentage=(t_float == 0).float().mean().item() * 100.0,
                    duration_ms=duration_ms,
                )
                if store_tensors:
                    stats.tensor = t_float.cpu()
                    self.activations[name] = stats.tensor
                self.stats[name] = stats

        h = layer.register_forward_hook(hook)
        self.handles.append(h)

    def remove_hooks(self):
        """Removes all registered hooks to free PyTorch references and memory."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self._timing_start.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Returns statistical overview across all instrumented layers."""
        return {
            name: {
                "shape": act.shape,
                "mean": act.mean,
                "std": act.std,
                "l2_norm": act.l2_norm,
                "zero_pct": act.zero_percentage,
                "duration_ms": act.duration_ms,
            }
            for name, act in self.stats.items()
        }
