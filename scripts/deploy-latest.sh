#!/bin/bash
# Deploy the latest routing LoRA checkpoint to the inference path
#
# Usage:
#   ./scripts/deploy-latest.sh

set -e
exec uvx modal run modal/deploy.py --latest
