#!/bin/bash

module load stack/2024-06 gcc/12.2.0 python/3.11.6 eth_proxy

VENV_PATH="$SCRATCH/saliency/.venv"
export HF_HOME="$SCRATCH/saliency/cache"

# export RESET_ENV="true"

# RESET_ENV can also be set to "true" to force a fresh environment
if [ ! -d "$VENV_PATH" ]; then
  RESET_ENV="true"
fi

if [ "$RESET_ENV" == "true" ]; then
  rm -rf "$VENV_PATH"
  python3 -m venv "$VENV_PATH"
  echo "Virtual environment created at $VENV_PATH at $(date)"
fi

source "$VENV_PATH/bin/activate"

if [ "$RESET_ENV" == "true" ]; then
  pip install --upgrade pip --quiet
  pip install -e .[dev] --quiet
  echo "Dependencies installed at $(date)"
else
  echo "Using existing virtual environment at $VENV_PATH"
fi