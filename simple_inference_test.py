#!/usr/bin/env python3
"""
Simple inference test - just generate w* tokens without profiling
"""

import torch
import time
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def simple_inference(model_name: str = "microsoft/Phi-3.5-mini-instruct", 
                    w_star: int = 50,
                    prompt: str = None):
    """Run simple inference without any profiling"""
    
    print("=" * 60)
    print(f"Simple Inference Test")
    print(f"Model: {model_name}")
    print(f"w*: {w_star} tokens")
    print("=" * 60)
    
    # Default prompt if none provided
    if prompt is None:
        prompt = "Write a Python function to calculate the factorial of a number."
    
    # 1. Load model and tokenizer
    print("\n1. Loading model...")
    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    load_time = time.time() - start_time
    print(f"   Model loaded in {load_time:.1f}s")
    
    # 2. Tokenize input
    print(f"\n2. Input prompt: '{prompt[:50]}...'")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(device)
    prompt_length = input_ids.shape[1]
    print(f"   Prompt length: {prompt_length} tokens")
    
    # 3. Generate w* tokens
    print(f"\n3. Generating {w_star} tokens...")
    inference_start = time.time()
    
    generated_ids = input_ids
    with torch.no_grad():
        for i in range(w_star):
            # Simple generation without KV cache optimization
            outputs = model(input_ids=generated_ids)
            logits = outputs.logits[:, -1, :]
            
            # Greedy decoding
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                print(f"\n   [EOS reached at token {i+1}]")
                break
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"   Generated {i+1}/{w_star} tokens...", end='\r')
    
    inference_time = time.time() - inference_start
    tokens_generated = generated_ids.shape[1] - prompt_length
    
    # 4. Decode and display result
    print(f"\n\n4. Results:")
    print(f"   Tokens generated: {tokens_generated}")
    print(f"   Inference time: {inference_time:.2f}s")
    print(f"   Speed: {tokens_generated/inference_time:.1f} tokens/s")
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\n5. Generated text:")
    print("-" * 60)
    print(generated_text)
    print("-" * 60)
    
    # 5. Memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n6. Peak GPU memory: {memory_used:.2f} GB")

def main():
    parser = argparse.ArgumentParser(description="Simple inference test")
    parser.add_argument("--model", default="microsoft/Phi-3.5-mini-instruct", help="Model name")
    parser.add_argument("--w-star", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--prompt", type=str, help="Custom prompt")
    
    args = parser.parse_args()
    
    simple_inference(
        model_name=args.model,
        w_star=args.w_star,
        prompt=args.prompt
    )

if __name__ == "__main__":
    main()