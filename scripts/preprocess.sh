#!/bin/bash
# Preprocess Qwen3-VL datasets on Modal, then auto-start training.
#
# Pipeline: generate_dataset.sh -> upload_dataset.sh -> preprocess.sh -> train_router.sh

set -euo pipefail

DATASET_NAME=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-name)
            DATASET_NAME="${2:-}"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$DATASET_NAME" ]]; then
    echo "Error: --dataset-name <NAME> is required"
    exit 1
fi

echo "========================================"
echo "STAGE 3: Preprocess Dataset"
echo "========================================"
echo ""
echo "Dataset: $DATASET_NAME"
echo ""

CMD="uvx modal run modal/preprocess.py --dataset-name $DATASET_NAME"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=" ${EXTRA_ARGS[*]}"
fi

echo "Command: $CMD"
echo ""

eval "$CMD"

if [[ $? -ne 0 ]]; then
    echo ""
    echo "❌ Preprocessing failed"
    exit 1
fi

echo ""
echo "✅ Preprocessing complete: $DATASET_NAME"
echo ""
echo "========================================"
echo "Auto-starting LoRA training..."
echo "========================================"
echo ""

# Extract just the dataset folder name (e.g., "routing_20251124_172521" from "datasets/routing_20251124_172521")
DATASET_FOLDER=$(basename "$DATASET_NAME")

# Auto-start training
exec ./scripts/train_router.sh "$DATASET_FOLDER"
