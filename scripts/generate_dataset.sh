#!/usr/bin/env bash
set -euo pipefail

# Generate routing dataset and optionally auto-start upload
#
# Usage:
#   ./scripts/generate_dataset.sh [--dry] --train-images 100 \
#     --calendar-dataset /path/to/cal/train.jsonl \
#     --claim-window-dataset /path/to/claim/train.jsonl \
#     --provider-select-dataset /path/to/provider/train.jsonl

DRY_RUN=false
SCRIPT_ARGS=()

# Parse args for --dry flag
for arg in "$@"; do
  if [[ "$arg" == "--dry" ]]; then
    DRY_RUN=true
  else
    SCRIPT_ARGS+=("$arg")
  fi
done

echo "========================================"
echo "STAGE 1: Generate Routing Dataset"
echo "========================================"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY RUN] Would execute:"
  echo "  python3 scripts/generate_routing_data.py ${SCRIPT_ARGS[*]}"
  echo ""
  echo "[DRY RUN] On success, would run:"
  echo "  ./scripts/upload_dataset.sh <dataset_name>"
  exit 0
fi

# Execute dataset generation
python3 scripts/generate_routing_data.py "${SCRIPT_ARGS[@]}"

if [[ $? -ne 0 ]]; then
  echo ""
  echo "❌ Dataset generation failed"
  exit 1
fi

# Extract dataset name from most recent dataset directory
DATASET_DIR=$(ls -td datasets/routing_* 2>/dev/null | head -1)

if [[ -z "$DATASET_DIR" ]]; then
  echo ""
  echo "❌ Could not find generated dataset directory"
  exit 1
fi

DATASET_NAME=$(basename "$DATASET_DIR")

echo ""
echo "✅ Dataset generated: $DATASET_NAME"
echo ""
echo "========================================"
echo "Auto-starting upload..."
echo "========================================"
echo ""

# Auto-start upload (without --dry)
exec ./scripts/upload_dataset.sh "$DATASET_NAME"
