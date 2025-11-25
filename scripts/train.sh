#!/bin/bash
# Train Qwen3-VL with Modal
# This script ensures training always runs in DETACHED mode (-d)
# so it continues even if you disconnect.

set -e  # Exit on error

# Parse arguments
FAST_FLAG=""
TEST_FLAG=""
RUN_NAME=""
DATASET_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)
            FAST_FLAG="--fast"
            shift
            ;;
        --test)
            TEST_FLAG="--test-mode"
            shift
            ;;
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
            echo "Usage: $0 --run-name <NAME> --dataset-name <NAME> [--fast] [--test]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$RUN_NAME" ]]; then
    echo "Error: --run-name is required"
    echo "Usage: $0 --run-name <NAME> --dataset-name <NAME> [--fast] [--test]"
    exit 1
fi

if [[ -z "$DATASET_NAME" ]]; then
    echo "Error: --dataset-name is required"
    echo "Usage: $0 --run-name <NAME> --dataset-name <NAME> [--fast] [--test]"
    exit 1
fi

# Build modal command with -d (ALWAYS DETACHED) via uvx
MODAL_CMD="uvx modal run -d modal/training.py --run-name $RUN_NAME --dataset-name $DATASET_NAME $FAST_FLAG $TEST_FLAG"

echo "=================================="
echo "Starting Modal Training (DETACHED)"
echo "=================================="
echo "Command: $MODAL_CMD"
echo ""

# Execute
$MODAL_CMD
