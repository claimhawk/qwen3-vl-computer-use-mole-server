#!/usr/bin/env bash
# Copyright (c) 2025 Tylt LLC. All rights reserved.
# Derivative works may be released by researchers,
# but original files may not be redistributed or used beyond research purposes.

# Generate MoE routing dataset locally (reads from local generator datasets)
# Auto-chains to upload.sh
#
# Usage:
#   ./scripts/generate.sh [options]        # Generate and auto-upload
#   ./scripts/generate.sh --dry [options]  # Generate only, no upload

set -euo pipefail

DRY_RUN=false
EXTRA_ARGS=()

# Parse args - extract --dry, pass everything else through
for arg in "$@"; do
    if [[ "$arg" == "--dry" ]]; then
        DRY_RUN=true
    else
        EXTRA_ARGS+=("$arg")
    fi
done

echo "========================================"
echo "STAGE 1: Generate MoE Routing Dataset"
echo "========================================"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Will generate but NOT upload"
    echo ""
fi

# Run the dataset generation locally
# Set env var so generator.py knows it was called from this script
export FROM_SCRIPT=1

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    uv run --script generator.py "${EXTRA_ARGS[@]}"
else
    uv run --script generator.py
fi

if [[ $? -ne 0 ]]; then
    echo ""
    echo "Dataset generation failed"
    exit 1
fi

# Find the most recently created routing dataset directory
LATEST_DATASET=$(ls -td datasets/routing* 2>/dev/null | head -1)

if [[ -z "$LATEST_DATASET" ]]; then
    echo ""
    echo "No dataset directory found"
    exit 1
fi

DATASET_NAME=$(basename "$LATEST_DATASET")
echo ""
echo "Generated dataset: $DATASET_NAME"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "[DRY RUN] Skipping upload"
    echo ""
    echo "To upload manually:"
    echo "  ./scripts/upload.sh datasets/$DATASET_NAME"
    exit 0
fi

echo ""
echo "========================================"
echo "Auto-starting upload..."
echo "========================================"
echo ""

exec ./scripts/upload.sh "datasets/$DATASET_NAME"
