"""
Configuration for Power Testing
Standalone configuration that doesn't depend on external files
"""

import torch

# Model Configuration
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"  # 3.8B model

# Test Configuration
W_STAR_VALUES = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700]
SAMPLES_PER_W_STAR = 5  # Number of samples to test per w* value
MAX_NEW_TOKENS = 700

# Device Configurations - simplified for power testing
DEVICE_CONFIGS = [
    {
        "name": "8GB_Device_FP16",  # Changed from INT8 to FP16 due to compatibility issues
        "memory_gb": 8,
        "memory_fraction": 0.34,  # Simulates 8GB on 24GB GPU
        "gpu_id": 0,  # Default to GPU 0
        "model": MODEL_NAME,
        "torch_dtype": torch.float16,
        "load_in_8bit": False,  # Disabled INT8 due to library issues
    },
    {
        "name": "16GB_Device_FP16",
        "memory_gb": 16,
        "memory_fraction": 0.67,  # Simulates 16GB on 24GB GPU
        "gpu_id": 1,  # Default to GPU 1
        "model": MODEL_NAME,
        "torch_dtype": torch.float16,
        "load_in_8bit": False,  # FP16 (no quantization)
    }
]

# Generation settings
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50

# Monitoring settings
POWER_SAMPLE_INTERVAL = 0.2  # seconds between power measurements
COOLDOWN_TIME = 3  # seconds between tests

# Platform settings
PLATFORM = "jetson"  # Options: "desktop" (nvidia-smi) or "jetson" (tegrastats)
# For Jetson, install: sudo pip3 install jetson-stats