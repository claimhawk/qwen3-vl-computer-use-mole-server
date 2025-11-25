#!/bin/bash
# Evaluate routing LoRA on held-out eval data
#
# Usage:
#   ./scripts/eval.sh --run-name <NAME> --dataset-name <NAME>

set -e

RUN_NAME=""
DATASET_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --run-name <NAME> --dataset-name <NAME>"
            exit 1
            ;;
    esac
done

if [[ -z "$RUN_NAME" ]]; then
    echo "Error: --run-name is required"
    exit 1
fi

if [[ -z "$DATASET_NAME" ]]; then
    echo "Error: --dataset-name is required"
    exit 1
fi

echo "=================================="
echo "Routing LoRA Evaluation"
echo "=================================="
echo ""
echo "Run: $RUN_NAME"
echo "Dataset: $DATASET_NAME"
echo ""

uvx modal run modal/eval.py --run-name "$RUN_NAME" --dataset-name "$DATASET_NAME"
