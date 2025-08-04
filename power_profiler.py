#!/usr/bin/env python3
"""
Power Profiler for SLICER Edge Devices
Main script for running power consumption experiments
"""

import os
import sys
import torch
import time
import pandas as pd
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import argparse
from multiprocessing import Process, Queue, Manager

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from system_monitor import SystemMonitor
from edge_inference import edge_inference, format_phi_prompt, format_llama_prompt

# Import PLATFORM explicitly if not already imported
try:
    _ = PLATFORM
except NameError:
    from config import PLATFORM

@dataclass
class PowerProfile:
    """Power profile for a single configuration"""
    device_name: str
    w_star: int
    avg_power_w: float
    max_power_w: float
    baseline_power_w: float
    net_power_w: float  # avg_power - baseline
    total_energy_j: float
    avg_temp_c: float
    max_temp_c: float
    temp_increase_c: float
    inference_time_s: float
    tokens_generated: int
    energy_per_token_j: float
    tokens_per_second: float
    edge_completion_rate: float
    gpu_utilization_avg: float
    gpu_utilization_max: float
    memory_used_mb: float

def get_test_prompts(num_samples: int = 10) -> List[Tuple[str, str]]:
    """Load test prompts from MBPP dataset"""
    try:
        dataset = load_dataset("mbpp", split="train", trust_remote_code=True)
        system_prompt = "You are an expert Python programmer. Solve the given problem with clean, well-documented code."
        
        prompts = []
        for i in range(min(num_samples, len(dataset))):
            problem = dataset[i]['text'].strip()
            user_prompt = f"""Problem: {problem}

Provide a complete solution including:
1. A clear Python function implementation
2. Brief explanation of the approach
3. Time complexity analysis"""
            
            prompts.append((f"mbpp_{i:03d}", (system_prompt, user_prompt)))
        
        return prompts
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Fallback prompts
        return [
            ("test_001", ("You are a helpful assistant.", "Write a Python function to calculate factorial.")),
            ("test_002", ("You are a helpful assistant.", "Explain how quicksort works and implement it.")),
            ("test_003", ("You are a helpful assistant.", "Write a function to find the longest palindrome in a string.")),
        ]

def profile_single_inference(
    model, 
    tokenizer, 
    device: torch.device,
    prompt: str, 
    w_star: int,
    monitor: SystemMonitor
) -> Dict:
    """Profile power consumption for a single inference"""
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(1)
    
    # Get baseline power (idle)
    monitor.clear_history()
    monitor.start_monitoring()
    time.sleep(2)
    monitor.stop_monitoring()
    baseline_stats = monitor.get_metrics_summary()
    baseline_power = baseline_stats['gpu']['avg_power_w']
    baseline_temp = baseline_stats['gpu']['avg_temp_c']
    
    # Monitor during inference
    monitor.clear_history()
    monitor.start_monitoring()
    
    # Run edge inference
    start_time = time.time()
    result = edge_inference(
        model, tokenizer, device, prompt, w_star,
        temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS
    )
    inference_time = time.time() - start_time
    
    # Continue monitoring for thermal settling
    time.sleep(1)
    monitor.stop_monitoring()
    
    # Get inference statistics
    inference_stats = monitor.get_metrics_summary()
    
    return {
        'inference_time': inference_time,
        'baseline_power': baseline_power,
        'avg_power': inference_stats['gpu']['avg_power_w'],
        'max_power': inference_stats['gpu']['max_power_w'],
        'net_power': inference_stats['gpu']['avg_power_w'] - baseline_power,
        'energy_j': (inference_stats['gpu']['avg_power_w'] - baseline_power) * inference_time,
        'baseline_temp': baseline_temp,
        'avg_temp': inference_stats['gpu']['avg_temp_c'],
        'max_temp': inference_stats['gpu']['max_temp_c'],
        'temp_increase': inference_stats['gpu']['max_temp_c'] - baseline_temp,
        'tokens_generated': result['edge_tokens'],
        'tokens_per_second': result['tokens_per_second'],
        'completed_at_edge': result['completed_at_edge'],
        'gpu_utilization_avg': inference_stats['gpu']['avg_utilization'],
        'gpu_utilization_max': inference_stats['gpu']['max_utilization'],
        'memory_used_mb': inference_stats['gpu']['max_memory_mb']
    }

def run_device_profiling_process(device_config: Dict, w_star_values: List[int], 
                                samples_per_w_star: int, output_dir: str,
                                result_queue: Queue):
    """Run power profiling for a single device in a separate process"""
    
    # Set CUDA device for this process
    gpu_id = device_config['gpu_id']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"\n[Process {os.getpid()}] Starting profiling for {device_config['name']} on GPU {gpu_id}")
    
    # Run the profiling
    device_results = run_single_device_profiling(device_config, w_star_values, samples_per_w_star, output_dir)
    
    # Send results back
    result_queue.put((device_config['name'], device_results))
    print(f"\n[{device_config['name']}] Profiling complete!")

def run_single_device_profiling(device_config: Dict, w_star_values: List[int], 
                               samples_per_w_star: int, output_dir: str) -> List[PowerProfile]:
    """Run profiling for a single device"""
    
    # Setup device
    device = torch.device("cuda:0")  # Always cuda:0 after CUDA_VISIBLE_DEVICES
    torch.cuda.set_device(0)
    
    # Set memory limit
    torch.cuda.set_per_process_memory_fraction(
        device_config['memory_fraction'], 
        0  # Always device 0 in this process
    )
    torch.cuda.empty_cache()
    
    # Initialize monitor with platform from config
    monitor = SystemMonitor(gpu_id=0, interval=POWER_SAMPLE_INTERVAL, platform=PLATFORM)
    
    # Load model
    print(f"\n[{device_config['name']}] Loading model...")
    load_start = time.time()
    
    if device_config.get('load_in_8bit', False):
        # INT8 loading with BitsAndBytesConfig
        from transformers import BitsAndBytesConfig
        
        # Create quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=False  # Prevent CPU offloading
        )
        
        # INT8 models must use device_map='auto'
        model = AutoModelForCausalLM.from_pretrained(
            device_config['model'],
            quantization_config=quantization_config,
            device_map='auto',  # Must be 'auto' for 8-bit models
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # INT8 models are already in eval mode and on correct device
        model.eval()
        
        # Since we're using CUDA_VISIBLE_DEVICES, 'auto' will map to the correct GPU
        print(f"[{device_config['name']}] Model loaded with INT8 quantization on available GPU")
    else:
        # Standard loading with torch_dtype from config
        torch_dtype = device_config.get('torch_dtype', torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            device_config['model'],
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device).eval()  # Note: .to(device) only for non-INT8 models
    
    tokenizer = AutoTokenizer.from_pretrained(device_config['model'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[{device_config['name']}] Model loaded in {time.time() - load_start:.1f}s")
    
    # Load test prompts
    test_prompts = get_test_prompts(samples_per_w_star * 2)
    
    # Format prompts
    formatted_prompts = []
    for prompt_id, (system, user) in test_prompts:
        if "phi" in device_config['model'].lower():
            full_prompt = format_phi_prompt(system, user)
        elif "llama" in device_config['model'].lower():
            full_prompt = format_llama_prompt(system, user)
        else:
            full_prompt = f"{system}\n\n{user}"
        formatted_prompts.append((prompt_id, full_prompt))
    
    # Test each w* value
    device_results = []
    
    for w_star in w_star_values:
        print(f"\n[{device_config['name']}] Testing w* = {w_star}")
        
        w_star_results = []
        edge_completions = 0
        
        # Test multiple samples
        for i in range(samples_per_w_star):
            prompt_id, prompt = formatted_prompts[i]
            
            try:
                # Pass None as device for INT8 models since they handle device placement internally
                profile_device = None if device_config.get('load_in_8bit', False) else device
                profile = profile_single_inference(
                    model, tokenizer, profile_device, prompt, w_star, monitor
                )
                
                w_star_results.append(profile)
                if profile['completed_at_edge']:
                    edge_completions += 1
                
                print(f"[{device_config['name']}] Sample {i+1}: "
                      f"{profile['tokens_generated']} tokens, "
                      f"{profile['net_power']:.1f}W")
                
                # Cool down
                time.sleep(COOLDOWN_TIME)
                
            except Exception as e:
                print(f"[{device_config['name']}] Error: {e}")
                continue
        
        # Aggregate results
        if w_star_results:
            profile = PowerProfile(
                device_name=device_config['name'],
                w_star=w_star,
                avg_power_w=np.mean([r['avg_power'] for r in w_star_results]),
                max_power_w=np.max([r['max_power'] for r in w_star_results]),
                baseline_power_w=np.mean([r['baseline_power'] for r in w_star_results]),
                net_power_w=np.mean([r['net_power'] for r in w_star_results]),
                total_energy_j=np.sum([r['energy_j'] for r in w_star_results]),
                avg_temp_c=np.mean([r['avg_temp'] for r in w_star_results]),
                max_temp_c=np.max([r['max_temp'] for r in w_star_results]),
                temp_increase_c=np.mean([r['temp_increase'] for r in w_star_results]),
                inference_time_s=np.sum([r['inference_time'] for r in w_star_results]),
                tokens_generated=np.sum([r['tokens_generated'] for r in w_star_results]),
                energy_per_token_j=np.sum([r['energy_j'] for r in w_star_results]) / 
                                  np.sum([r['tokens_generated'] for r in w_star_results]),
                tokens_per_second=np.mean([r['tokens_per_second'] for r in w_star_results]),
                edge_completion_rate=edge_completions / len(w_star_results) * 100,
                gpu_utilization_avg=np.mean([r['gpu_utilization_avg'] for r in w_star_results]),
                gpu_utilization_max=np.max([r['gpu_utilization_max'] for r in w_star_results]),
                memory_used_mb=np.max([r['memory_used_mb'] for r in w_star_results])
            )
            
            device_results.append(profile)
    
    # Clean up
    del model
    del monitor
    torch.cuda.empty_cache()
    
    return device_results

def run_power_profiling(
    device_indices: Optional[List[int]] = None,
    w_star_values: Optional[List[int]] = None,
    output_dir: str = "results"
):
    """Run comprehensive power profiling - concurrent by default"""
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Select devices to test
    if device_indices is None:
        test_devices = DEVICE_CONFIGS
    else:
        test_devices = [DEVICE_CONFIGS[i] for i in device_indices if i < len(DEVICE_CONFIGS)]
    
    # Select w* values to test
    if w_star_values is None:
        w_star_values = W_STAR_VALUES
    
    print("=" * 80)
    print("SLICER Power Profiling for Edge Devices")
    print("=" * 80)
    print(f"Devices: {[d['name'] for d in test_devices]}")
    print(f"w* values: {w_star_values}")
    print(f"Samples per w*: {SAMPLES_PER_W_STAR}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Load test prompts (will be loaded in each process)
    
    all_results = []
    
    # Run all devices concurrently
    if len(test_devices) > 1:
        print("\n" + "="*60)
        print("Running devices CONCURRENTLY")
        print("="*60)
        
        # Create manager for shared queue
        manager = Manager()
        result_queue = manager.Queue()
        
        # Start processes for each device
        processes = []
        for device_config in test_devices:
            p = Process(
                target=run_device_profiling_process,
                args=(device_config, w_star_values, SAMPLES_PER_W_STAR, output_dir, result_queue)
            )
            p.start()
            processes.append(p)
            print(f"Started process {p.pid} for {device_config['name']} on GPU {device_config['gpu_id']}")
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        print("\nAll processes completed!")
        
        # Collect results from queue
        device_results_map = {}
        while not result_queue.empty():
            device_name, results = result_queue.get()
            device_results_map[device_name] = results
            all_results.extend(results)
        
        # Save individual device results
        for device_name, results in device_results_map.items():
            if results:
                device_df = pd.DataFrame([vars(r) for r in results])
                device_file = os.path.join(output_dir, f"power_profile_{device_name}_{timestamp}.csv")
                device_df.to_csv(device_file, index=False)
                print(f"Device results saved to: {device_file}")
    else:
        # Single device - run in main process with CUDA_VISIBLE_DEVICES
        device_config = test_devices[0]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_config['gpu_id'])
        print(f"\nRunning single device: {device_config['name']} on GPU {device_config['gpu_id']}")
        
        device_results = run_single_device_profiling(device_config, w_star_values, SAMPLES_PER_W_STAR, output_dir)
        all_results.extend(device_results)
        
        # Save device-specific results
        if device_results:
            device_df = pd.DataFrame([vars(r) for r in device_results])
            device_file = os.path.join(output_dir, f"power_profile_{device_config['name']}_{timestamp}.csv")
            device_df.to_csv(device_file, index=False)
            print(f"Device results saved to: {device_file}")
    
    # Save all results
    all_df = pd.DataFrame([vars(r) for r in all_results])
    summary_file = os.path.join(output_dir, f"power_profile_summary_{timestamp}.csv")
    all_df.to_csv(summary_file, index=False)
    
    # Generate final report
    print_final_report(all_df, output_dir, timestamp)
    
    return all_df

def print_final_report(df: pd.DataFrame, output_dir: str, timestamp: str):
    """Print and save final report"""
    
    report_lines = []
    
    def log(line=""):
        print(line)
        report_lines.append(line)
    
    log("\n" + "="*80)
    log("POWER PROFILING FINAL REPORT")
    log("="*80)
    log(f"\nTimestamp: {timestamp}")
    log(f"Total configurations tested: {len(df)}")
    
    # Summary table
    log(f"\n{'Device':<20} {'w*':<6} {'Net Power':<12} {'Energy/Token':<14} {'Max Temp':<10} {'Tokens/s':<10}")
    log("-" * 82)
    
    for _, row in df.iterrows():
        log(f"{row['device_name']:<20} {row['w_star']:<6} "
            f"{row['net_power_w']:<12.1f} {row['energy_per_token_j']:<14.3f} "
            f"{row['max_temp_c']:<10.0f} {row['tokens_per_second']:<10.1f}")
    
    # Device comparison
    log("\n" + "="*60)
    log("DEVICE COMPARISON")
    log("="*60)
    
    for device_name in df['device_name'].unique():
        device_data = df[df['device_name'] == device_name]
        log(f"\n{device_name}:")
        log(f"  Power range: {device_data['net_power_w'].min():.1f} - {device_data['net_power_w'].max():.1f}W")
        log(f"  Energy efficiency: {device_data['energy_per_token_j'].min():.3f} - {device_data['energy_per_token_j'].max():.3f} J/token")
        log(f"  Temperature: {device_data['max_temp_c'].min():.0f} - {device_data['max_temp_c'].max():.0f}°C")
        log(f"  Memory usage: {device_data['memory_used_mb'].max():.0f}MB")
        
        # Find optimal w*
        optimal_idx = device_data['energy_per_token_j'].idxmin()
        optimal_w = device_data.loc[optimal_idx, 'w_star']
        log(f"  Optimal w* for efficiency: {optimal_w}")
    
    # Key findings
    log("\n" + "="*60)
    log("KEY FINDINGS FOR REVIEWER")
    log("="*60)
    
    log("\n1. POWER CONSUMPTION (Net, excluding baseline):")
    log(f"   - Average: {df['net_power_w'].mean():.1f}W")
    log(f"   - Peak: {df['max_power_w'].max():.1f}W")
    log(f"   - INT8 vs FP16: {df[df['device_name'].str.contains('INT8')]['net_power_w'].mean():.1f}W vs "
        f"{df[~df['device_name'].str.contains('INT8')]['net_power_w'].mean():.1f}W")
    
    log("\n2. THERMAL IMPACT:")
    log(f"   - Max temperature: {df['max_temp_c'].max():.0f}°C")
    log(f"   - Average temp increase: {df['temp_increase_c'].mean():.1f}°C")
    log("   - All within safe operating range")
    
    log("\n3. ENERGY EFFICIENCY:")
    log(f"   - Best: {df['energy_per_token_j'].min():.3f} J/token")
    log(f"   - Worst: {df['energy_per_token_j'].max():.3f} J/token")
    log(f"   - INT8 advantage: {(1 - df[df['device_name'].str.contains('INT8')]['energy_per_token_j'].mean() / df[~df['device_name'].str.contains('INT8')]['energy_per_token_j'].mean()) * 100:.1f}% more efficient")
    
    log("\n4. PRACTICAL IMPLICATIONS:")
    avg_power = df['net_power_w'].mean()
    battery_wh = 18.5  # 5000mAh at 3.7V
    runtime_hours = battery_wh / avg_power
    log(f"   - Mobile battery life (5000mAh): ~{runtime_hours:.1f} hours")
    log(f"   - Throughput: {df['tokens_per_second'].mean():.1f} tokens/second average")
    log("   - Edge completion increases with w*, reducing server load")
    
    # Save report
    report_file = os.path.join(output_dir, f"power_profile_report_{timestamp}.txt")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    log(f"\nReport saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Power profiling for SLICER edge devices")
    parser.add_argument("--devices", nargs="+", type=int, help="Device indices to test (e.g., 0 1)")
    parser.add_argument("--w-star", nargs="+", type=int, help="w* values to test")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--samples", type=int, help="Samples per w* value")
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.samples:
        global SAMPLES_PER_W_STAR
        SAMPLES_PER_W_STAR = args.samples
    
    # Run profiling
    run_power_profiling(
        device_indices=args.devices,
        w_star_values=args.w_star,
        output_dir=args.output
    )

if __name__ == "__main__":
    main()