#!/usr/bin/env bash
set -euo pipefail

# Generate routing dataset on Modal from existing datasets on Modal volumes
# Auto-downloads locally and auto-starts preprocessing
#
# Usage:
#   ./scripts/generate_dataset.sh [--dry]

# Production defaults:
# - 1000 train total (balanced across 3 adapters)
# - 100 val total (for early stopping during training)
# - 100 eval total (held-out for final accuracy testing after training)
TRAIN_TASKS=1000
VAL_TASKS=100
EVAL_TASKS=100

DRY_RUN=false

# Parse args
for arg in "$@"; do
  if [[ "$arg" == "--dry" ]]; then
    DRY_RUN=true
  fi
done

echo "========================================"
echo "STAGE 1: Generate Routing Dataset"
echo "========================================"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY RUN] Would execute:"
  echo "  uvx modal run modal/generate_routing.py --train-tasks $TRAIN_TASKS --val-tasks $VAL_TASKS --eval-tasks $EVAL_TASKS"
  echo ""
  echo "Next step (run without --dry to auto-continue):"
  echo "  ./scripts/preprocess.sh --dataset-name datasets/<dataset_name>"
  exit 0
fi

# Set env var so Modal scripts know they were called from shell script
export FROM_SCRIPT=1

# Execute dataset generation on Modal (downloads locally too)
OUTPUT=$(uvx modal run modal/generate_routing.py --train-tasks "$TRAIN_TASKS" --val-tasks "$VAL_TASKS" --eval-tasks "$EVAL_TASKS" 2>&1 | tee /dev/stderr)

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
