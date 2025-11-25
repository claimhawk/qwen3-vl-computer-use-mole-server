#!/usr/bin/env bash
set -euo pipefail

# Preprocess routing dataset on Modal CPU and optionally auto-start training
#
# Usage:
#   ./scripts/preprocess_dataset.sh [--dry] routing_20251124_133254

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
echo "STAGE 3: Preprocess Dataset on Modal"
echo "========================================"
echo ""
echo "Dataset: $DATASET_NAME"
echo "Modal path: /data/datasets/$DATASET_NAME"
echo "Output: /data/preprocessed/$DATASET_NAME"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY RUN] Would execute:"
  echo "  uvx modal run modal/router_preprocess.py --dataset-dir $DATASET_NAME"
  echo ""
  echo "[DRY RUN] On success, would run:"
  echo "  ./scripts/train_router.sh $DATASET_NAME"
  exit 0
fi

# Execute preprocessing
uvx modal run modal/router_preprocess.py --dataset-dir "$DATASET_NAME"

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

# Auto-start training (without --dry)
exec ./scripts/train_router.sh "$DATASET_NAME"
