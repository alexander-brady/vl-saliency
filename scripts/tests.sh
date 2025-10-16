#!/bin/bash
# Run test suite with coverage reporting

echo "Running tests at $(date)"

source scripts/env.sh

COVERAGE_FILE=.pytest_cache/.coverage pytest -v --tb=short --maxfail=3 \
  --cov=vl_saliency --cov-report=term-missing --color=yes

deactivate