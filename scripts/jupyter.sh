#!/bin/bash
# This script initializes the environment and starts an interactive Jupyter server.
# Designed for use on ETH Zurich's Euler cluster with GPU support.
# Adjust srun resources and Jupyter options as needed.

# If USE_GPUS is true or not set, request GPU resources
if [ "$USE_GPUS" == "" ] || [ "$USE_GPUS" == "true" ]; then
  echo "Requesting GPU resources..."
  GPU_OPTIONS="--gpus=1 --gres=gpumem:64G --mem-per-cpu=32G"
else
  GPU_OPTIONS=""
fi

srun --time=02:00:00 $GPU_OPTIONS bash <<'EOF'
  
# Initialize environment (loads modules + uv sync)
source scripts/env.sh

# check if kernel is installed
if ! jupyter kernelspec list | grep -q "vl_saliency"; then
  echo "Kernel 'vl_saliency' not found. Installing kernel..."
  python -m ipykernel install --user --name="vl_saliency" --display-name "Python (vl_saliency)"
fi

# Start Jupyter
uv run jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
EOF