#!/usr/bin/env bash
# Copyright (c) 2025 Tylt LLC. All rights reserved.
# Derivative works may be released by researchers,
# but original files may not be redistributed or used beyond research purposes.

# Generate MoE routing dataset on Modal
# All data operations happen on Modal volumes - no local downloads/uploads
#
# Steps:
#   1. Verify all expert datasets exist on Modal
#   2. Assemble routing dataset from expert datasets
#   3. Preprocess (tokenize with Qwen processor)
#   4. Train router LoRA
#
# Usage:
#   ./scripts/generate.sh                          # Full pipeline
#   ./scripts/generate.sh --skip-train             # Generate dataset only
#   ./scripts/generate.sh --samples-per-adapter 500 --test-per-adapter 25

set -euo pipefail

echo "========================================"
echo "MoE Routing Dataset Generator (Modal)"
echo "========================================"
echo ""

# Run the Modal function with all arguments
uvx modal run modal/generate.py "$@"
