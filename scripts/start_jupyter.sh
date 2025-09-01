#!/bin/bash
# This script sets up the environment and starts a Jupyter server.
# Designed for use on ETH Zurich's Euler cluster with GPU support.
# You may need to adjust the module load commands based on your environment.

source scripts/setup_env.sh

jupyter notebook --no-browser --ip=0.0.0.0 --port=8888

deactivate