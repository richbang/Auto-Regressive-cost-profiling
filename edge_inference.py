"""
Edge Inference Module for Power Testing
Simplified version that focuses on edge processing only
"""

import torch
import time
from typing import Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

def safe_check_eos(token_ids, tokenizer) -> bool:
    """Safely check if any token is EOS"""
    if tokenizer.eos_token_id is None:
        return False
    
    if isinstance(token_ids, torch.Tensor):
        if token_ids.dim() == 0:  # scalar
            return token_ids.item() == tokenizer.eos_token_id
        else:  # tensor
            return tokenizer.eos_token_id in token_ids.flatten().tolist()
    elif isinstance(token_ids, (list, tuple)):
        return tokenizer.eos_token_id in token_ids
    elif isinstance(token_ids, (int, float)):
        return int(token_ids) == tokenizer.eos_token_id
    else:
        return False

def edge_inference(
    model, 
    tokenizer, 
    device: Optional[torch.device],
    prompt: str, 
    w_star: int,
    temperature: float = 0.7,
    max_new_tokens: int = 700
) -> Dict:
    """
    Run edge inference up to w* tokens
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        device: CUDA device
        prompt: Input prompt
        w_star: Number of tokens to generate on edge
        temperature: Sampling temperature
        max_new_tokens: Maximum new tokens to generate
    
    Returns:
        Dictionary with inference results
    """
    edge_start = time.time()
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    # For INT8 models (when device is None), the model is already on the correct device
    # Only move input_ids if device is provided (non-quantized models)
    if device is None:
        # INT8 model - don't move input_ids, let the model handle it
        input_ids = inputs["input_ids"]
    else:
        # Regular model - move to device
        input_ids = inputs["input_ids"].to(device)
    prompt_length = input_ids.shape[1]
    
    print(f"  [Edge] Starting inference: prompt_length={prompt_length}, w_star={w_star}")
    
    # Initialize generation
    generated_ids = input_ids
    past_key_values = None
    
    # Generate up to w* tokens
    tokens_generated = 0
    completed = False
    
    with torch.no_grad():
        for i in range(min(w_star, max_new_tokens)):
            # Forward pass with KV cache
            outputs = model(
                input_ids=generated_ids[:, -1:] if past_key_values else generated_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            
            # Sample next token
            if temperature > 0:
                logits = logits / temperature
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
            if safe_check_eos(next_token, tokenizer):
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

def format_phi_prompt(system: str, user: str) -> str:
    """Format prompt for Phi models"""
    return f"<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n"

def format_llama_prompt(system: str, user: str) -> str:
    """Format prompt for Llama models"""
    return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"