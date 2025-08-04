#!/usr/bin/env python3
"""
Simple monitoring for Docker environments
Reads basic metrics without requiring special services
"""

import time
import os

def get_simple_metrics():
    """Get basic metrics that work in Docker"""
    metrics = {
        'temperature': 0.0,
        'gpu_load': 0.0,
        'memory_used_mb': 0.0,
        'memory_total_mb': 0.0,
    }
    
    # 1. Temperature
    try:
        with open('/sys/class/thermal/thermal_zone2/temp', 'r') as f:
            metrics['temperature'] = int(f.read().strip()) / 1000.0
    except:
        try:
            # Try other zones
            for i in range(10):
                try:
                    with open(f'/sys/class/thermal/thermal_zone{i}/temp', 'r') as f:
                        temp = int(f.read().strip()) / 1000.0
                        if temp > metrics['temperature']:
                            metrics['temperature'] = temp
                except:
                    pass
        except:
            pass
    
    # 2. GPU Load
    try:
        with open('/sys/devices/gpu.0/load', 'r') as f:
            metrics['gpu_load'] = int(f.read().strip()) / 10.0
    except:
        pass
    
    # 3. Memory
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    metrics['memory_total_mb'] = int(line.split()[1]) / 1024.0
                elif line.startswith('MemAvailable:'):
                    available = int(line.split()[1]) / 1024.0
                    metrics['memory_used_mb'] = metrics['memory_total_mb'] - available
    except:
        pass
    
    return metrics

def monitor_loop(duration=10):
    """Monitor for specified duration"""
    print(f"Monitoring for {duration} seconds...")
    print(f"{'Time':>6} {'Temp(°C)':>10} {'GPU(%)':>8} {'Mem(MB)':>10}")
    print("-" * 40)
    
    start_time = time.time()
    while time.time() - start_time < duration:
        metrics = get_simple_metrics()
        elapsed = time.time() - start_time
        print(f"{elapsed:6.1f} {metrics['temperature']:10.1f} "
              f"{metrics['gpu_load']:8.1f} "
              f"{metrics['memory_used_mb']:10.0f}/{metrics['memory_total_mb']:.0f}")
        time.sleep(1)

if __name__ == "__main__":
    print("Simple Jetson Monitor (Docker-friendly)")
    print("=" * 40)
    
    # Single read
    metrics = get_simple_metrics()
    print(f"Current Temperature: {metrics['temperature']:.1f}°C")
    print(f"Current GPU Load: {metrics['gpu_load']:.1f}%")
    print(f"Current Memory: {metrics['memory_used_mb']:.0f}/{metrics['memory_total_mb']:.0f}MB")
    
    print("\n" + "=" * 40)
    
    # Continuous monitoring
    monitor_loop(10)