"""
System Monitor - Simplified version for power testing
Monitors GPU power, temperature, and utilization
Supports both desktop GPUs (nvidia-smi) and Jetson devices (tegrastats)
"""

import os
import subprocess
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import re

@dataclass
class GPUMetrics:
    """GPU metrics at a point in time"""
    timestamp: float
    memory_used_mb: float
    memory_total_mb: float
    utilization_percent: float
    temperature_c: float
    power_w: float

class SystemMonitor:
    """Monitor GPU resources for power profiling"""
    
    def __init__(self, gpu_id: int = 0, interval: float = 0.2, platform: str = "desktop"):
        self.gpu_id = gpu_id
        self.interval = interval
        self.platform = platform  # "desktop" or "jetson"
        self.metrics_history: List[GPUMetrics] = []
        self.monitoring = False
        self._stop_flag = False
        self._monitor_thread = None
        self._tegrastats_process = None
        
    def get_gpu_metrics(self) -> Optional[GPUMetrics]:
        """Get current GPU metrics"""
        if self.platform == "jetson":
            return self._get_jetson_metrics()
        else:
            return self._get_desktop_metrics()
    
    def _get_jetson_metrics(self) -> Optional[GPUMetrics]:
        """Get metrics from Jetson using tegrastats"""
        try:
            # Run tegrastats once and parse output
            cmd = ['tegrastats', '--once']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                output = result.stdout.strip()
                
                # Parse tegrastats output
                # Example: RAM 2810/7850MB ... GPU@54C ... VDD_GPU_SOC 3924mW
                
                # Extract temperature
                temp_match = re.search(r'GPU@(\d+)C', output)
                temperature = float(temp_match.group(1)) if temp_match else 0.0
                
                # Extract power (total system power on Jetson)
                power_match = re.search(r'VDD_GPU_SOC (\d+)mW', output)
                if not power_match:
                    power_match = re.search(r'VDD_IN (\d+)mW', output)
                power_mw = float(power_match.group(1)) if power_match else 0.0
                power_w = power_mw / 1000.0
                
                # Extract GPU utilization
                gpu_match = re.search(r'GR3D_FREQ (\d+)%', output)
                utilization = float(gpu_match.group(1)) if gpu_match else 0.0
                
                # Extract memory
                mem_match = re.search(r'RAM (\d+)/(\d+)MB', output)
                if mem_match:
                    mem_used = float(mem_match.group(1))
                    mem_total = float(mem_match.group(2))
                else:
                    mem_used = mem_total = 0.0
                
                return GPUMetrics(
                    timestamp=time.time(),
                    memory_used_mb=mem_used,
                    memory_total_mb=mem_total,
                    utilization_percent=utilization,
                    temperature_c=temperature,
                    power_w=power_w
                )
                
        except Exception as e:
            print(f"Error getting Jetson metrics: {e}")
        return None
    
    def _get_desktop_metrics(self) -> Optional[GPUMetrics]:
        """Get metrics from desktop GPU using nvidia-smi"""
        try:
            # When CUDA_VISIBLE_DEVICES is set, nvidia-smi uses actual GPU IDs
            # So we need to map back to the real GPU ID
            actual_gpu_id = self.gpu_id
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                visible_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
                if len(visible_devices) == 1:
                    # Single GPU visible, use that actual ID
                    actual_gpu_id = int(visible_devices[0])
            
            # Query GPU properties
            query = "memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw"
            cmd = [
                'nvidia-smi',
                '--query-gpu=' + query,
                '--format=csv,nounits,noheader',
                '-i', str(actual_gpu_id)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                if len(values) >= 5:
                    # Handle [N/A] for power on some GPUs
                    power_val = values[4]
                    power = float(power_val) if power_val != '[N/A]' else 0.0
                    
                    return GPUMetrics(
                        timestamp=time.time(),
                        memory_used_mb=float(values[0]),
                        memory_total_mb=float(values[1]),
                        utilization_percent=float(values[2]),
                        temperature_c=float(values[3]),
                        power_w=power
                    )
        except Exception as e:
            print(f"Error getting GPU metrics: {e}")
        return None
    
    def start_monitoring(self):
        """Start continuous monitoring in background"""
        self.monitoring = True
        self._stop_flag = False
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        self._stop_flag = True
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring and not self._stop_flag:
            metrics = self.get_gpu_metrics()
            if metrics:
                self.metrics_history.append(metrics)
            time.sleep(self.interval)
    
    def clear_history(self):
        """Clear metrics history"""
        self.metrics_history = []
    
    def get_metrics_summary(self) -> Dict:
        """Get summary statistics from collected metrics"""
        if not self.metrics_history:
            return {
                'gpu': {
                    'avg_power_w': 0,
                    'max_power_w': 0,
                    'avg_temp_c': 0,
                    'max_temp_c': 0,
                    'avg_utilization': 0,
                    'max_utilization': 0,
                    'avg_memory_mb': 0,
                    'max_memory_mb': 0
                }
            }
        
        powers = [m.power_w for m in self.metrics_history]
        temps = [m.temperature_c for m in self.metrics_history]
        utils = [m.utilization_percent for m in self.metrics_history]
        mems = [m.memory_used_mb for m in self.metrics_history]
        
        return {
            'gpu': {
                'avg_power_w': np.mean(powers),
                'max_power_w': np.max(powers),
                'avg_temp_c': np.mean(temps),
                'max_temp_c': np.max(temps),
                'avg_utilization': np.mean(utils),
                'max_utilization': np.max(utils),
                'avg_memory_mb': np.mean(mems),
                'max_memory_mb': np.max(mems),
                'sample_count': len(self.metrics_history)
            }
        }
    
    def get_current_metrics(self) -> Optional[Dict]:
        """Get current GPU metrics as dictionary"""
        metrics = self.get_gpu_metrics()
        if metrics:
            return asdict(metrics)
        return None