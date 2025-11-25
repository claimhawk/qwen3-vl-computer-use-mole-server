#!/bin/bash
# Preprocess Qwen3-VL datasets on Modal.
#
# Pipeline: generate_dataset.sh -> preprocess.sh -> train.sh

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
echo "STAGE 2: Preprocess Dataset"
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
echo "Next: ./scripts/train.sh --dataset-name $DATASET_NAME --run-name <run_name>"
