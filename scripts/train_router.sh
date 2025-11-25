#!/usr/bin/env bash
set -euo pipefail

# Train router as LoRA adapter using Modal training infrastructure
#
# Usage:
#   ./scripts/train_router.sh [--dry] routing_20251124_150334

DRY_RUN=false
DATASET_NAME=""

# Parse args
for arg in "$@"; do
  if [[ "$arg" == "--dry" ]]; then
    DRY_RUN=true
  else
    DATASET_NAME="$arg"
  fi
done

if [[ -z "$DATASET_NAME" ]]; then
  echo "❌ Error: Dataset name required"
  echo "Usage: $0 [--dry] <dataset_name>"
  exit 1
fi

echo "========================================"
echo "STAGE 3: Train Router LoRA"
echo "========================================"
echo ""
echo "Dataset: $DATASET_NAME"
echo "Train data: /data/datasets/$DATASET_NAME/train.jsonl"
echo "Eval data: /data/datasets/$DATASET_NAME/eval.jsonl"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY RUN] Would train router LoRA using Modal training infrastructure:"
  echo ""
  echo "  ./scripts/train.sh \\"
  echo "    --run-name router-classifier \\"
  echo "    --dataset-name $DATASET_NAME"
  echo ""
  echo "The model will learn to output: 'calendar', 'claim-window', or 'provider-select'"
  exit 0
fi

echo "Starting LoRA training on Modal..."
echo ""

# Train router LoRA using the Modal training infrastructure
./scripts/train.sh --run-name router-classifier --dataset-name "$DATASET_NAME"

if [[ $? -ne 0 ]]; then
  echo ""
  echo "❌ Training failed"
  exit 1
fi

echo ""
echo "✅ Router LoRA training started!"
echo ""
echo "Training is running in DETACHED mode on Modal."
echo ""
echo "Monitor progress:"
echo "  - Check Modal dashboard: https://modal.com"
echo "  - View logs: uvx modal app logs router-classifier"
echo ""
echo "When complete, checkpoint will be saved to: /checkpoints/router-classifier"
echo ""
echo "Next steps:"
echo "  1. Wait for training to complete"
echo "  2. Test router on validation set"
echo "  3. Integrate into inference pipeline (see plans/router-lora-guide.md)"
echo "  4. Router will output: 'calendar', 'claim-window', or 'provider-select'"
