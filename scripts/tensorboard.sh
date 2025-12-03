#!/bin/bash
# Start TensorBoard for Router trainer
# This deploys the standalone TensorBoard server (scales to zero when idle)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TB_SERVER="$PROJECT_ROOT/projects/tensorboard-server"

echo "=================================="
echo "TensorBoard for Router Trainer"
echo "=================================="
echo ""

# Check if tensorboard-server exists
if [[ ! -d "$TB_SERVER" ]]; then
    echo "Error: tensorboard-server not found at $TB_SERVER"
    exit 1
fi

cd "$TB_SERVER"

# Deploy (creates stable URL) or serve (temporary)
MODE="${1:-deploy}"

if [[ "$MODE" == "serve" ]]; then
    echo "Starting TensorBoard (temporary - Ctrl+C to stop)..."
    echo ""
    uvx modal serve modal/tb_server.py
else
    echo "Deploying TensorBoard (stable URL, scales to zero when idle)..."
    echo ""
    uvx modal deploy modal/tb_server.py
    echo ""
    echo "TensorBoard is deployed. Access via Modal dashboard or:"
    echo "  https://<your-workspace>--tensorboard-server-tensorboard-router.modal.run"
    echo ""
    echo "To clean up old runs:"
    echo "  modal run scripts/cleanup.py --trainer router"
fi
