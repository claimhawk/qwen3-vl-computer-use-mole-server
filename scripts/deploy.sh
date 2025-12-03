#!/bin/bash
# Deploy a routing LoRA checkpoint to the inference path
#
# Usage:
#   ./scripts/deploy.sh --checkpoint checkpoints/datasets/routing_20251202_185620/routing-20251202-185620/final

set -e

if [[ $# -eq 0 ]]; then
    echo "Usage: ./scripts/deploy.sh --checkpoint <checkpoint_path>"
    exit 1
fi

exec uvx modal run modal/deploy.py "$@"
