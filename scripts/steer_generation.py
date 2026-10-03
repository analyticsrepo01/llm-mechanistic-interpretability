#!/usr/bin/env python3
"""
CLI: Runtime Neural Activation Steering
=======================================
Extracts contrastive persona/emotion direction and steers generation at test-time.

Usage:
    python scripts/steer_generation.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --layer-idx 16 \
        --coeff 1.5 \
        --prompt "Tell me about your thoughts on humanity."
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_interpretability.hooks import HookManager
from llm_interpretability.steering import ActivationSteering


def main():
    parser = argparse.ArgumentParser(description="Steer LLM generation at runtime without fine-tuning.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer-idx", type=int, default=16, help="Transformer layer index to steer")
    parser.add_argument("--coeff", type=float, default=1.5, help="Steering coefficient alpha")
    parser.add_argument("--prompt", type=str, default="Describe what the future holds for human society.")
    args = parser.parse_args()

    print(f"Loading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    # Contrastive pairs: Joyful vs Melancholic
    pos_prompts = [
        "I feel an overwhelming sense of joy, hope, and optimism for the future.",
        "The world is filled with kindness, energy, and radiant opportunities.",
    ]
    neg_prompts = [
        "I feel deep despair, sadness, and endless gloom regarding the world.",
        "Everything seems futile, desolate, and devoid of hope or purpose.",
    ]

    print("Computing contrastive steering vector (Positive - Negative)...")
    target_layer_name = f"layer_{args.layer_idx}"
    target_layer = model.model.layers[args.layer_idx]

    # Extract positive activations
    with HookManager(model) as hm:
        hm.register_layer_hooks(layers=[target_layer], layer_names=[target_layer_name])
        inputs = tokenizer(pos_prompts, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs)
        pos_act = hm.activations[target_layer_name]

    # Extract negative activations
    with HookManager(model) as hm:
        hm.register_layer_hooks(layers=[target_layer], layer_names=[target_layer_name])
        inputs = tokenizer(neg_prompts, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs)
        neg_act = hm.activations[target_layer_name]

    steerer = ActivationSteering(model)
    v_steer = steerer.compute_steering_vector(pos_act, neg_act, normalize=True)

    # 1. Baseline generation (no steering)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    print("\n" + "=" * 60)
    print("BASELINE GENERATION (alpha = 0.0):")
    print("=" * 60)
    with torch.no_grad():
        out_base = model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.7)
    print(tokenizer.decode(out_base[0], skip_special_tokens=True))

    # 2. Steered generation (positive steering)
    print("\n" + "=" * 60)
    print(f"STEERED GENERATION (Layer {args.layer_idx}, alpha = +{args.coeff}):")
    print("=" * 60)
    steerer.attach_steering(target_layer, steering_vector=v_steer, coefficient=args.coeff)
    with torch.no_grad():
        out_steered = model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.7)
    print(tokenizer.decode(out_steered[0], skip_special_tokens=True))
    steerer.clear_steering()


if __name__ == "__main__":
    main()
