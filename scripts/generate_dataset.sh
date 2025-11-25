#!/usr/bin/env bash
set -euo pipefail

# Generate routing dataset on Modal from existing datasets on Modal volumes
# Auto-downloads locally and auto-starts preprocessing
#
# Usage:
#   ./scripts/generate_dataset.sh [--dry] --train-tasks 1000 --eval-tasks 100

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
  echo "  uvx modal run modal/generate_routing.py ${SCRIPT_ARGS[*]:-}"
  echo ""
  echo "[DRY RUN] On success, would auto-start:"
  echo "  ./scripts/preprocess.sh --dataset-name datasets/<dataset_name>"
  exit 0
fi

# Execute dataset generation on Modal (downloads locally too)
OUTPUT=$(uvx modal run modal/generate_routing.py "${SCRIPT_ARGS[@]:-}" 2>&1 | tee /dev/stderr)

if [[ $? -ne 0 ]]; then
  echo ""
  echo "❌ Dataset generation failed"
  exit 1
fi

# Extract dataset name from output
DATASET_NAME=$(echo "$OUTPUT" | grep -oE 'routing_[0-9]{8}_[0-9]{6}' | tail -1)

if [[ -z "$DATASET_NAME" ]]; then
  echo ""
  echo "❌ Could not extract dataset name from output"
  exit 1
fi

echo ""
echo "✅ Dataset generated: $DATASET_NAME"
echo ""
echo "========================================"
echo "Auto-starting preprocessing..."
echo "========================================"
echo ""

exec ./scripts/preprocess.sh --dataset-name "datasets/$DATASET_NAME"
