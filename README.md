# SLICER Edge Device Power Profiling

Standalone power profiling framework for measuring energy consumption and thermal impact of Large Language Models (LLMs) on edge devices.

## Purpose

This tool addresses critical concerns about energy consumption and thermal costs on edge devices by providing concrete measurements including power draw, temperature impact, and energy efficiency metrics. It was developed to provide empirical data for the SLICER paper review process.

## Key Features

- **Concurrent multi-device profiling** - Test multiple GPUs simultaneously
- **Comprehensive metrics** - Power, thermal, efficiency, and performance measurements
- **Memory simulation** - Simulate different device capacities on larger GPUs
- **Automated reporting** - Generate detailed reports for analysis

## What It Measures

| Metric | Description | Unit |
|--------|-------------|------|
| Power Consumption | Actual GPU power draw (baseline-adjusted) | Watts (W) |
| Energy Efficiency | Energy cost per generated token | Joules/token (J/tok) |
| Thermal Impact | Temperature increase during inference | Celsius (°C) |
| Throughput | Token generation speed | tokens/second |
| Memory Usage | GPU memory consumption | MB |
| Battery Life | Estimated runtime on mobile device | hours |

## Requirements

### Desktop/Server GPUs
- NVIDIA GPU with `nvidia-smi` support
- Python 3.8 or higher
- CUDA-compatible environment
- PyTorch 2.0+

### Jetson Devices
- Jetson Nano/TX2/Xavier/Orin
- Python 3.8 or higher
- PyTorch for Jetson
- Access to /sys filesystem for power/thermal monitoring

## Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run with default settings (tests both 8GB and 16GB configurations):
```bash
python power_profiler.py
```

Or use the convenience script:
```bash
./run_power_test.sh
```

## Configuration

Edit `config.py` to customize:

```python
# Platform settings
PLATFORM = "desktop"  # Use "jetson" for Jetson devices

# Model configuration
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"  # Change model here

# Device configurations
DEVICE_CONFIGS = [
    {
        "name": "8GB_Device_FP16",
        "memory_gb": 8,
        "memory_fraction": 0.34,  # Simulates 8GB on 24GB GPU
        "gpu_id": 0,
        "model": MODEL_NAME,
        "torch_dtype": torch.float16,
        "load_in_8bit": False,  # Set True for INT8 (if supported)
    },
    # Add more device configurations...
]

# Test parameters
W_STAR_VALUES = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700]
SAMPLES_PER_W_STAR = 5
```

### Jetson-specific Configuration

For Jetson devices, set `PLATFORM = "jetson"` in config.py. The tool will read metrics directly from system files:
- Power: `/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/` (voltage × current)
- Temperature: `/sys/class/thermal/thermal_zone*/temp`
- GPU Load: `/sys/devices/gpu.0/load`
- Memory: `/proc/meminfo`

Note: Power measurements are calculated from INA3221 sensor readings (voltage × current for each rail).

## Advanced Usage

### Test specific devices
```bash
# Test only device 0
python power_profiler.py --devices 0

# Test devices 0 and 1
python power_profiler.py --devices 0 1
```

### Test specific w* values
```bash
python power_profiler.py --w-star 50 100 200
```

### Adjust sample size
```bash
python power_profiler.py --samples 10
```

### Custom output directory
```bash
python power_profiler.py --output results/experiment_2024
```

### Isolated GPU testing
```bash
# Force specific GPU
CUDA_VISIBLE_DEVICES=2 python power_profiler.py --devices 0
```

## Output Structure

```
results/
├── power_profile_8GB_Device_FP16_20240804_164226.csv    # Device-specific data
├── power_profile_16GB_Device_FP16_20240804_164226.csv   # Device-specific data
├── power_profile_summary_20240804_164226.csv            # Combined results
└── power_profile_report_20240804_164226.txt             # Human-readable report
```

## Interpreting Results

### Sample Output
```
Device               w*     Net Power    Energy/Token   Max Temp   Tokens/s  
--------------------------------------------------------------------------------
8GB_Device_FP16      50     33.4         1.535          45         22.0      
8GB_Device_FP16      100    45.9         1.981          48         23.3      
16GB_Device_FP16     50     29.8         1.341          44         22.3      
16GB_Device_FP16     100    45.4         1.946          48         23.4
```

### Key Insights
- **Lower w*** = Better energy efficiency but more server offloading
- **Higher w*** = More edge processing but higher energy cost
- **Optimal w*** typically around 50-100 for efficiency

## Files Description

- `power_profiler.py` - Main profiling script
- `config.py` - Configuration file
- `system_monitor.py` - GPU monitoring module
- `edge_inference.py` - Edge inference implementation
- `run_power_test.sh` - Convenience script
- `cleanup.sh` - Process cleanup utility
- `requirements.txt` - Python dependencies

## Troubleshooting

### CUDA Out of Memory
Reduce `memory_fraction` in `config.py`:
```python
"memory_fraction": 0.25,  # Use only 25% of GPU memory
```

### Hanging Processes
Run cleanup script:
```bash
./cleanup.sh
```

### INT8 Loading Fails
Some models may not support INT8 quantization. Set `load_in_8bit: False` in config.

### Permission Denied
Make scripts executable:
```bash
chmod +x run_power_test.sh cleanup.sh
```

## Extending the Framework

### Adding New Metrics
Modify `system_monitor.py` to collect additional GPU metrics.

### Supporting New Models
Update `MODEL_NAME` in `config.py` and adjust prompt formatting in `edge_inference.py`.

### Custom Device Configurations
Add new entries to `DEVICE_CONFIGS` in `config.py`.

## Notes

- Results may vary based on GPU model, driver version, and system configuration
- Ensure consistent GPU cooling for thermal measurements
- Close other GPU applications for accurate measurements
- The tool simulates memory constraints using PyTorch's memory fraction limits
- For Jetson devices:
  - Power is calculated from INA3221 sensor (V × I)
  - Temperature readings from thermal zones
  - Memory readings are system-wide
  - Requires read access to /sys filesystem
  - Consider thermal throttling at high temperatures

## Citation

If you use this tool in your research, please cite:
```
[SLICER paper citation]
```