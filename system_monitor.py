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
        """Get metrics from Jetson using tegrastats or system files"""
        try:
            # Try different possible locations for tegrastats
            tegrastats_paths = [
                'tegrastats',
                '/usr/bin/tegrastats',
                '/home/nvidia/tegrastats',
                '/usr/local/bin/tegrastats'
            ]
            
            tegrastats_cmd = None
            for path in tegrastats_paths:
                if subprocess.run(['which', path], capture_output=True).returncode == 0:
                    tegrastats_cmd = path
                    break
                elif os.path.exists(path):
                    tegrastats_cmd = path
                    break
            
            if not tegrastats_cmd:
                # Fallback to reading from system files directly
                return self._get_jetson_metrics_sysfs()
            
            # Run tegrastats once and parse output
            cmd = [tegrastats_cmd, '--once'] if '--once' in subprocess.run([tegrastats_cmd, '--help'], capture_output=True, text=True).stdout else [tegrastats_cmd]
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
    
    
    def _get_jetson_metrics_sysfs(self) -> Optional[GPUMetrics]:
        """Get metrics from Jetson by reading system files directly"""
        try:
            # Initialize default values
            temperature = 0.0
            power_mw = 0.0
            gpu_util = 0.0
            mem_used = 0.0
            mem_total = 0.0
            
            # 1. Try to get temperature from thermal zones
            thermal_zones = [
                "/sys/devices/virtual/thermal/thermal_zone0/temp",
                "/sys/devices/virtual/thermal/thermal_zone1/temp",
                "/sys/class/thermal/thermal_zone0/temp",
                "/sys/class/thermal/thermal_zone1/temp"
            ]
            
            for zone in thermal_zones:
                if os.path.exists(zone):
                    try:
                        with open(zone, 'r') as f:
                            temp = int(f.read().strip()) / 1000.0
                            if temp > temperature:
                                temperature = temp
                    except:
                        pass
            
            # 2. Try to get power consumption from hwmon
            # First, find hwmon path dynamically
            hwmon_base = "/sys/bus/i2c/drivers/ina3221/7-0040/hwmon"
            hwmon_path = None
            
            if os.path.exists(hwmon_base):
                # Find hwmon* directory
                for item in os.listdir(hwmon_base):
                    if item.startswith('hwmon'):
                        hwmon_path = os.path.join(hwmon_base, item)
                        break
            
            if hwmon_path:
                # Read voltage and current to calculate power
                try:
                    # Read all channels
                    for channel in range(1, 4):  # Channels 1, 2, 3
                        voltage_file = os.path.join(hwmon_path, f"in{channel}_input")
                        current_file = os.path.join(hwmon_path, f"curr{channel}_input")
                        
                        if os.path.exists(voltage_file) and os.path.exists(current_file):
                            with open(voltage_file, 'r') as f:
                                voltage_mv = float(f.read().strip())
                            with open(current_file, 'r') as f:
                                current_ma = float(f.read().strip())
                            
                            # Calculate power in mW (V * I)
                            channel_power_mw = (voltage_mv / 1000.0) * current_ma
                            power_mw += channel_power_mw
                except:
                    pass
            
            # Fallback to old paths if needed
            if power_mw == 0.0:
                power_files = [
                    "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input",
                    "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device0/in_power0_input"
                ]
            
            for pfile in power_files:
                if os.path.exists(pfile):
                    try:
                        with open(pfile, 'r') as f:
                            power_mw += float(f.read().strip())
                    except:
                        pass
            
            # 3. Try to get GPU utilization from sysfs
            gpu_load_files = [
                "/sys/devices/gpu.0/load",
                "/sys/devices/17000000.gv11b/load",
                "/sys/devices/17000000.ga10b/load"
            ]
            
            for gfile in gpu_load_files:
                if os.path.exists(gfile):
                    try:
                        with open(gfile, 'r') as f:
                            gpu_util = float(f.read().strip()) / 10.0  # Often in per-mille
                    except:
                        pass
            
            # 4. Get memory info from /proc/meminfo
            try:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    
                mem_total_kb = 0
                mem_free_kb = 0
                
                for line in meminfo.split('\n'):
                    if line.startswith('MemTotal:'):
                        mem_total_kb = int(line.split()[1])
                    elif line.startswith('MemAvailable:'):
                        mem_free_kb = int(line.split()[1])
                
                mem_total = mem_total_kb / 1024.0  # Convert to MB
                mem_used = (mem_total_kb - mem_free_kb) / 1024.0
            except:
                pass
            
            # 5. If still no GPU util, try nvidia-smi (some Jetsons might have it)
            if gpu_util == 0.0:
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, timeout=1)
                    if result.returncode == 0:
                        gpu_util = float(result.stdout.strip())
                except:
                    pass
            
            # Convert power from mW to W
            power_w = power_mw / 1000.0
            
            # If we got at least some data, return it
            if temperature > 0 or power_w > 0 or mem_total > 0:
                return GPUMetrics(
                    timestamp=time.time(),
                    memory_used_mb=mem_used,
                    memory_total_mb=mem_total,
                    utilization_percent=gpu_util,
                    temperature_c=temperature,
                    power_w=power_w
                )
            else:
                print("Warning: Unable to read Jetson metrics from sysfs")
                # Return minimal metrics to keep the profiler running
                return GPUMetrics(
                    timestamp=time.time(),
                    memory_used_mb=0,
                    memory_total_mb=0,
                    utilization_percent=0,
                    temperature_c=0,
                    power_w=0
                )
                
        except Exception as e:
            print(f"Error reading Jetson sysfs metrics: {e}")
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