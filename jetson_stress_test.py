#!/usr/bin/env python3
"""
Jetson GPU Stress Test - Maximum GPU Utilization
Run continuous inference to maximize GPU usage
"""

import torch
import time
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import threading
import os

def continuous_inference(model, tokenizer, device, prompt, duration_seconds=300):
    """Run continuous inference for specified duration"""
    
    print(f"Running continuous inference for {duration_seconds} seconds...")
    start_time = time.time()
    inference_count = 0
    total_tokens = 0
    
    while (time.time() - start_time) < duration_seconds:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        
        # Generate with aggressive settings
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=500,
                temperature=0.9,
                top_p=0.95,
                top_k=100,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        
        tokens_generated = outputs.shape[1] - input_ids.shape[1]
        total_tokens += tokens_generated
        inference_count += 1
        
        # No cooldown - immediate next inference
        elapsed = time.time() - start_time
        print(f"[{elapsed:.1f}s] Inference #{inference_count}: {tokens_generated} tokens, "
              f"Avg: {total_tokens/elapsed:.1f} tok/s", end='\r')
    
    print(f"\n\nCompleted {inference_count} inferences, {total_tokens} total tokens")
    print(f"Average throughput: {total_tokens/duration_seconds:.1f} tokens/second")

def parallel_inference_threads(model, tokenizer, device, num_threads=2):
    """Run multiple inference threads to maximize GPU usage"""
    
    prompts = [
        "Explain quantum computing in detail with examples and applications.",
        "Write a comprehensive guide on machine learning algorithms.",
        "Describe the history and future of artificial intelligence.",
        "Explain blockchain technology and its various use cases.",
    ]
    
    threads = []
    for i in range(num_threads):
        prompt = prompts[i % len(prompts)]
        t = threading.Thread(
            target=continuous_inference,
            args=(model, tokenizer, device, prompt, 60)  # 60 seconds per thread
        )
        threads.append(t)
        t.start()
        print(f"Started inference thread {i+1}")
    
    # Wait for all threads
    for t in threads:
        t.join()

def main():
    parser = argparse.ArgumentParser(description="Jetson GPU stress test")
    parser.add_argument("--model", default="microsoft/Phi-3.5-mini-instruct", help="Model name")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds")
    parser.add_argument("--threads", type=int, default=1, help="Number of parallel threads")
    parser.add_argument("--continuous", action="store_true", help="Run continuous generation")
    
    args = parser.parse_args()
    
    # Disable tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("=" * 60)
    print("JETSON GPU STRESS TEST")
    print(f"Model: {args.model}")
    print(f"Duration: {args.duration}s")
    print(f"Threads: {args.threads}")
    print("=" * 60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load model with maximum memory usage
    print("\nLoading model...")
    start_time = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"  # Let it use all available memory
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded in {time.time() - start_time:.1f}s")
    
    # Show GPU memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {memory_used:.1f}/{memory_total:.1f} GB ({memory_used/memory_total*100:.1f}%)")
    
    # Run stress test
    if args.threads > 1:
        print(f"\nRunning {args.threads} parallel inference threads...")
        parallel_inference_threads(model, tokenizer, device, args.threads)
    else:
        prompt = "Write a detailed technical analysis of GPU architecture, parallel computing principles, and CUDA programming."
        continuous_inference(model, tokenizer, device, prompt, args.duration)
    
    print("\nStress test complete!")

if __name__ == "__main__":
    main()