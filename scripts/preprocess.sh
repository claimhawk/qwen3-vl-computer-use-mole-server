#!/bin/bash
# Preprocess Qwen3-VL datasets on Modal.
#
# Pipeline: generate_dataset.sh -> preprocess.sh -> train.sh

set -euo pipefail

DATASET_NAME=""
DRY_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-name)
            DATASET_NAME="${2:-}"
            shift 2
            ;;
        --dry)
            DRY_RUN=true
            shift
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

# Extract dataset timestamp for run name (e.g., routing_20251125_045148 -> routing-20251125-045148)
RUN_NAME=$(basename "$DATASET_NAME" | tr '_' '-')

echo "========================================"
echo "STAGE 2: Preprocess Dataset"
echo "========================================"
echo ""
echo "Dataset: $DATASET_NAME"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would execute:"
    echo "  uvx modal run modal/preprocess.py --dataset-name $DATASET_NAME ${EXTRA_ARGS[*]:-}"
    echo ""
    echo "Next step (run without --dry to auto-continue):"
    echo "  ./scripts/train.sh --dataset-name $DATASET_NAME --run-name $RUN_NAME"
    exit 0
fi

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
echo "Auto-starting training..."
echo "========================================"
echo ""

exec ./scripts/train.sh --dataset-name "$DATASET_NAME" --run-name "$RUN_NAME"
