#!/usr/bin/env bash
# Upload local dataset to Modal volume and auto-chain to preprocess
#
# Usage:
#   ./scripts/upload.sh [dataset_dir]       # Upload and auto-preprocess
#   ./scripts/upload.sh --dry [dataset_dir] # Dry run, show what would be uploaded

set -euo pipefail

DRY_RUN=false
DATASET_DIR=""

# Parse args
for arg in "$@"; do
    if [[ "$arg" == "--dry" ]]; then
        DRY_RUN=true
    elif [[ -z "$DATASET_DIR" && ! "$arg" =~ ^-- ]]; then
        DATASET_DIR="$arg"
    fi
done

echo "========================================"
echo "STAGE 2: Upload Dataset to Modal"
echo "========================================"
echo ""

if [[ -z "$DATASET_DIR" ]]; then
    # Find most recent routing dataset (supports both routing_ and routing-- formats)
    DATASET_DIR=$(ls -td datasets/routing* 2>/dev/null | head -1)
    if [[ -z "$DATASET_DIR" ]]; then
        echo "No dataset directory found. Specify path or run ./scripts/generate.sh first."
        exit 1
    fi
fi

# Remove trailing slash
DATASET_DIR="${DATASET_DIR%/}"
DATASET_NAME=$(basename "$DATASET_DIR")

echo "Dataset: $DATASET_NAME"
echo "Path: $DATASET_DIR"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would upload:"
    echo "  $DATASET_DIR/*.jsonl -> moe-lora-data:datasets/$DATASET_NAME/"
    echo ""
    echo "Then auto-preprocess:"
    echo "  ./scripts/preprocess.sh --dataset-name datasets/$DATASET_NAME"
    exit 0
fi

# Upload jsonl files to Modal volume
echo "Uploading to Modal volume moe-lora-data..."
for f in "$DATASET_DIR"/*.jsonl; do
    fname=$(basename "$f")
    echo "  Uploading $fname..."
    uvx modal volume put moe-lora-data "$f" "datasets/$DATASET_NAME/$fname" --force
done

# Also upload metadata.json if it exists
if [[ -f "$DATASET_DIR/metadata.json" ]]; then
    echo "  Uploading metadata.json..."
    uvx modal volume put moe-lora-data "$DATASET_DIR/metadata.json" "datasets/$DATASET_NAME/metadata.json" --force
fi

echo ""
echo "Upload complete: $DATASET_NAME"
echo ""
echo "========================================"
echo "Auto-starting preprocessing..."
echo "========================================"
echo ""

exec ./scripts/preprocess.sh --dataset-name "datasets/$DATASET_NAME"
