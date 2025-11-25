#!/usr/bin/env bash
set -euo pipefail

# Upload routing dataset, convert to classification format, and auto-start preprocessing
#
# Usage:
#   ./scripts/upload_dataset.sh [--dry] routing_20251124_133254

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

DATASET_DIR="datasets/$DATASET_NAME"

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "❌ Error: Dataset directory not found: $DATASET_DIR"
  exit 1
fi

echo "========================================"
echo "STAGE 2: Convert & Upload Dataset"
echo "========================================"
echo ""
echo "Dataset: $DATASET_NAME"
echo "Local path: $DATASET_DIR"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY RUN] Would execute:"
  echo "  1. python3 scripts/convert_routing_to_classification.py $DATASET_DIR"
  echo "  2. uvx modal volume put --force claimhawk-training-data $DATASET_DIR datasets/$DATASET_NAME"
  echo ""
  echo "[DRY RUN] On success, would run:"
  echo "  ./scripts/preprocess.sh --dataset-name $DATASET_NAME"
  exit 0
fi

# Step 1: Convert to classification format (overwrites original files)
echo "Converting to classification format..."
python3 scripts/convert_routing_to_classification.py "$DATASET_DIR"

if [[ $? -ne 0 ]]; then
  echo ""
  echo "❌ Conversion failed"
  exit 1
fi
echo "✅ Converted to classification format"
echo ""

# Step 2: Upload dataset (including classification files)
echo "Uploading to Modal volume..."
uvx modal volume put --force moe-lora-data "$DATASET_DIR" "datasets/$DATASET_NAME"

if [[ $? -ne 0 ]]; then
  echo ""
  echo "❌ Upload failed"
  exit 1
fi

echo ""
echo "✅ Dataset uploaded: $DATASET_NAME"
echo ""
echo "========================================"
echo "Auto-starting preprocessing..."
echo "========================================"
echo ""

# Auto-start preprocessing (without --dry)
exec ./scripts/preprocess.sh --dataset-name "datasets/$DATASET_NAME"
