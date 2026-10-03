#!/usr/bin/env python3
"""
CLI: Compute Layer Similarity Matrix via Centered Kernel Alignment (CKA)
========================================================================
Usage:
    python scripts/compute_cka_matrix.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_interpretability.hooks import HookManager
from llm_interpretability.cka import CKAAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Compute CKA layer similarity matrix.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-plot", type=str, default="outputs/cka_matrix.png")
    args = parser.parse_args()

    print(f"Loading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Deep learning architectures rely on backpropagation and matrix multiplication.",
        "The history of artificial intelligence spans decades of symbolic and connectionist methods.",
        "Quantum computing exploits superposition and entanglement to perform complex computations.",
    ]

    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)

    print("Extracting layer activations across batch prompts...")
    with HookManager(model) as hm:
        hm.register_layer_hooks(track_timing=False)
        with torch.no_grad():
            _ = model(**inputs)
        activations = hm.activations

    print("Computing pairwise Linear CKA scores across all layers...")
    analyzer = CKAAnalyzer()
    layer_names, cka_matrix = analyzer.compute_similarity_matrix(activations)

    os.makedirs(os.path.dirname(args.output_plot) or ".", exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(cka_matrix, cmap="magma", vmin=0.0, vmax=1.0, origin="lower")
    plt.colorbar(label="Linear CKA Similarity")
    plt.title(f"Layer Representation Similarity (CKA)\nModel: {args.model}", fontsize=12, fontweight="bold")
    plt.xlabel("Layer Index")
    plt.ylabel("Layer Index")
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=300)
    plt.close()

    print(f"Saved CKA heatmap to: {args.output_plot}")


if __name__ == "__main__":
    main()
