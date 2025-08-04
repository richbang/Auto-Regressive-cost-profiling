#!/bin/bash
# Power testing script for SLICER edge devices

echo "Starting power profiling experiment..."
echo "This will measure actual power consumption and thermal impact"
echo ""

# Create results directory
mkdir -p results

# Test both devices
echo "Testing 8GB INT8 device on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python power_profiler.py --devices 0 --output results/8gb_int8

echo ""
echo "Testing 16GB FP16 device on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python power_profiler.py --devices 1 --output results/16gb_fp16

echo ""
echo "Power profiling complete! Check results/ directory for reports."