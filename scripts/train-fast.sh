#!/bin/bash
# Fast training mode for Router trainer
# Runs 20 steps (first checkpoint + eval)
# Use this for quick iteration and testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pass all arguments plus --fast to train.sh
exec "$SCRIPT_DIR/train.sh" --fast "$@"
