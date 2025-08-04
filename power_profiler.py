#!/usr/bin/env python3
"""
Simplified Power Profiler - Inference Only Mode
For Jetson and other edge devices
"""

import os
import sys
import time
import torch
import argparse
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import configuration
from config import (
    MODEL_NAME, W_STAR_VALUES, SAMPLES_PER_W_STAR, MAX_NEW_TOKENS,
    DEVICE_CONFIGS, TEMPERATURE, TOP_P, TOP_K, PLATFORM
)

# Edge inference function embedded for standalone operation

def simulate_edge_inference(model, tokenizer, prompt, w_star, temperature=0.7, max_new_tokens=700):
    """Simulate edge device inference up to w* tokens"""
    device = next(model.parameters()).device
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device) if "attention_mask" in inputs else None
    
    prompt_length = input_ids.shape[1]
    
    print(f"  [Edge] Starting inference: prompt_length={prompt_length}, w_star={w_star}")
    
    # Initialize generation
    generated_ids = input_ids
    past_key_values = None
    tokens_generated = 0
    completed = False
    
    edge_start = time.time()
    
    # Generate up to w* tokens
    with torch.no_grad():
        for i in range(w_star):
            # Forward pass - like slicer_client_metrics.py, don't pass attention_mask with past_key_values
            if past_key_values:
                outputs = model(
                    input_ids=generated_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )
            else:
                outputs = model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    use_cache=True
                )
            
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            logits = logits / temperature
            
            # Sample next token
            if temperature > 0:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            tokens_generated += 1
            
            # Progress indicator every 50 tokens
            if (i + 1) % 50 == 0:
                print(f"  [Edge] Generated {i+1}/{w_star} tokens...", end='\r')
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                completed = True
                break
    
    edge_time = time.time() - edge_start
    
    print(f"\n  [Edge] Completed: {tokens_generated} tokens in {edge_time:.2f}s ({tokens_generated/edge_time:.1f} tok/s)")
    
    # Clean up KV cache
    del past_key_values
    torch.cuda.empty_cache()
    
    return {
        "type": "edge_complete" if completed else "edge_incomplete",
        "edge_tokens": tokens_generated,
        "edge_time": edge_time,
        "prompt_length": prompt_length,
        "total_length": generated_ids.shape[1],
        "completed_at_edge": completed,
        "tokens_per_second": tokens_generated / edge_time if edge_time > 0 else 0
    }

def get_test_prompts(num_samples: int) -> List[Tuple[str, Tuple[str, str]]]:
    """Get test prompts for inference"""
    prompts = [
        ("coding", (
            "You are an expert Python programmer. Solve the given problem with clean, well-documented code.",
            "Write a function to find the k-th largest element in an unsorted array using a min-heap approach."
        )),
        ("reasoning", (
            "You are a logical reasoning expert. Analyze the problem step by step.",
            "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning."
        )),
        ("creative", (
            "You are a creative writer with vivid imagination.",
            "Write a short story about a world where gravity works in reverse, but only on Tuesdays."
        )),
        ("technical", (
            "You are a technical expert. Provide detailed technical explanations.",
            "Explain how a transformer neural network processes sequential data, including the attention mechanism."
        )),
        ("analytical", (
            "You are a data analyst. Provide insights based on the given scenario.",
            "A company's sales increased by 20% but profits decreased by 5%. What could be the possible reasons?"
        ))
    ]
    
    # Repeat prompts if needed
    selected_prompts = []
    for i in range(num_samples):
        selected_prompts.append((f"prompt_{i+1}", prompts[i % len(prompts)]))
    
    return selected_prompts

def format_phi_prompt(system: str, user: str) -> str:
    """Format prompt for Phi models"""
    return f"<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n"

def format_llama_prompt(system: str, user: str) -> str:
    """Format prompt for Llama models"""
    return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"

def run_inference_only(device_indices=None, w_star_values=None, samples=None):
    """Run inference without power profiling"""
    print("=" * 80)
    print("INFERENCE ONLY MODE (No Power Profiling)")
    print("=" * 80)
    
    # Select devices
    if device_indices is None:
        test_devices = DEVICE_CONFIGS[:1]  # Just first device
    else:
        test_devices = [DEVICE_CONFIGS[i] for i in device_indices if i < len(DEVICE_CONFIGS)]
    
    # Select w* values - use config defaults if not specified
    if w_star_values is None:
        w_star_values = W_STAR_VALUES  # Use values from config.py
    
    # Number of samples - use config default if not specified
    if samples is None:
        samples = SAMPLES_PER_W_STAR  # Use value from config.py
    
    # Print configuration being used
    print("\nConfiguration:")
    print(f"Platform: {PLATFORM}")
    print(f"Model: {MODEL_NAME}")
    print(f"W* values: {w_star_values}")
    print(f"Samples per W*: {samples}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Devices: {[d['name'] for d in test_devices]}")
    print("=" * 80)
    
    # Load test prompts
    test_prompts = get_test_prompts(samples)
    
    for device_config in test_devices:
        print(f"\n--- Device: {device_config['name']} ---")
        
        # Set device
        device = torch.device(f"cuda:{device_config['gpu_id']}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device_config['gpu_id'])
        
        # Load model
        print(f"\nLoading model...")
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            device_config['model'],
            torch_dtype=device_config.get('torch_dtype', torch.float16),
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": device_config['gpu_id']}
        ).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(device_config['model'], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Model loaded in {time.time() - start_time:.1f}s")
        
        # Test each w* value
        for w_star in w_star_values:
            print(f"\n[w* = {w_star}]")
            
            for i, (prompt_type, (system, user)) in enumerate(test_prompts):
                # Format prompt correctly
                if "phi" in device_config['model'].lower():
                    full_prompt = format_phi_prompt(system, user)
                elif "llama" in device_config['model'].lower():
                    full_prompt = format_llama_prompt(system, user)
                else:
                    full_prompt = f"{system}\n\n{user}"
                
                print(f"\nSample {i+1}/{samples}:")
                print(f"Prompt: {full_prompt[:80]}...")
                
                # Run edge inference
                result = simulate_edge_inference(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=full_prompt,
                    w_star=w_star,
                    temperature=TEMPERATURE,
                    max_new_tokens=MAX_NEW_TOKENS
                )
                
                print(f"Generated {result['edge_tokens']} tokens in {result['edge_time']:.2f}s "
                      f"({result['tokens_per_second']:.1f} tok/s)")
                
                if result['completed_at_edge']:
                    print("→ Completed at edge (found EOS)")
                else:
                    print("→ Would need server continuation")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Power profiler for edge devices (inference only)")
    parser.add_argument("--devices", type=int, nargs="+", help="Device indices to test")
    parser.add_argument("--w-star", type=int, nargs="+", help="w* values to test")
    parser.add_argument("--samples", type=int, help="Number of samples per w*")
    
    args = parser.parse_args()
    
    # Run inference only mode
    run_inference_only(
        device_indices=args.devices,
        w_star_values=args.w_star,
        samples=args.samples
    )

if __name__ == "__main__":
    main()