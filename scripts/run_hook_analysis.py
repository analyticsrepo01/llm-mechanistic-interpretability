#!/usr/bin/env python3
"""
CLI: Run PyTorch Forward Hooks and Extract Layer Statistics
===========================================================
Usage:
    python scripts/run_hook_analysis.py --model meta-llama/Llama-3.1-8B-Instruct --prompt "Hello world"
"""

import argparse
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_interpretability.hooks import HookManager


def main():
    parser = argparse.ArgumentParser(description="Inspect transformer activations layer-by-layer.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--prompt", type=str, default="Explain how attention works in transformers.", help="Input text prompt")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (auto, cuda, cpu)")
    parser.add_argument("--output-json", type=str, default="outputs/activation_stats.json", help="Path to save stats JSON")
    args = parser.parse_args()

    print(f"Loading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=args.device,
    )
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)

    print(f"Executing forward pass with hooks attached...")
    with HookManager(model) as hm:
        hm.register_layer_hooks(track_timing=True)
        with torch.no_grad():
            _ = model(**inputs)

        summary = hm.get_summary()

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCaptured {len(summary)} layers successfully!")
    print(f"Saved statistics to: {args.output_json}")


if __name__ == "__main__":
    main()
