#!/bin/bash
# Cleanup script to stop any remaining power profiler processes

echo "Cleaning up power profiler processes..."

# Find and kill power_profiler processes
PIDS=$(ps aux | grep -E "python.*power_profiler" | grep -v grep | awk '{print $2}')

if [ -n "$PIDS" ]; then
    echo "Found processes: $PIDS"
    for PID in $PIDS; do
        echo "Killing process $PID"
        kill -9 $PID 2>/dev/null
    done
    echo "Cleanup completed"
else
    echo "No power profiler processes found"
fi

# Also check for any orphaned python processes using significant GPU
echo ""
echo "Checking for orphaned GPU processes..."
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader

echo ""
echo "If you see any python processes above using GPU memory, you can kill them with:"
echo "kill -9 <PID>"