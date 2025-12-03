#!/bin/bash
# Clean up old TensorBoard logs for Router trainer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TB_SERVER="$PROJECT_ROOT/projects/tensorboard-server"

cd "$TB_SERVER"
uvx modal run scripts/cleanup.py --trainer router
