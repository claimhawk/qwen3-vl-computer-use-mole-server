#!/usr/bin/env bash
# Copy a LoRA checkpoint into a Modal volume for sharing across apps.
# Updates config/loras.json to track LoRAs locally.
#
# Usage:
#   # upload local -> volume
#   scripts/save_lora.sh --checkpoint /path/to/ckpt --name calendar-tasks [--dataset datasets/xyz] [--volume moe-lora-checkpoints]
#   # copy within Modal volumes (no local checkpoint)
#   scripts/save_lora.sh --from-volume claimhawk-checkpoints --checkpoint path/to/checkpoint-40 --name calendar-tasks [--dataset datasets/xyz]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$ROOT_DIR/config/loras.json"

VOLUME="moe-lora-checkpoints"
DEST=""
NAME=""
FROM_VOLUME=""
DATASET=""
CHECKPOINT=""

usage() {
    echo "Usage: $0 --checkpoint <path_or_volume_path> --name <id> [--dataset <dataset_path>] [--volume ${VOLUME}] [--dest subdir] [--from-volume <vol>]" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT="${2:-}"
            shift 2
            ;;
        --from-volume)
            FROM_VOLUME="${2:-}"
            shift 2
            ;;
        --name)
            NAME="${2:-}"
            shift 2
            ;;
        --dataset)
            DATASET="${2:-}"
            shift 2
            ;;
        --volume)
            VOLUME="${2:-}"
            shift 2
            ;;
        --dest)
            DEST="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            ;;
    esac
done

if [[ -z "${CHECKPOINT:-}" ]]; then
    echo "Error: --checkpoint is required." >&2
    usage
fi

if [[ -z "${NAME:-}" ]]; then
    echo "Error: --name is required (e.g., calendar-tasks)." >&2
    usage
fi

if command -v modal >/dev/null 2>&1; then
    MODAL_BIN="modal"
elif command -v uvx >/dev/null 2>&1; then
    MODAL_BIN="uvx modal"
else
    echo "Error: modal CLI not found. Install via 'pip install modal-client' or ensure 'uvx modal' is available." >&2
    exit 1
fi

if [[ -z "$DEST" ]]; then
    DEST="$NAME"
fi

# Ensure config file exists
mkdir -p "$(dirname "$CONFIG_FILE")"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo '{"loras": {}}' > "$CONFIG_FILE"
fi

update_config() {
    local name="$1"
    local volume="$2"
    local dest="$3"
    local checkpoint="$4"
    local dataset="${5:-}"
    local from_vol="${6:-}"
    local timestamp
    timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

    # Build the JSON entry
    local source_checkpoint="$checkpoint"
    if [[ -n "$from_vol" ]]; then
        source_checkpoint="$from_vol:$checkpoint"
    fi

    # Use Python to update JSON (more reliable than jq for complex updates)
    python3 << EOF
import json
from pathlib import Path

config_path = Path("$CONFIG_FILE")
config = json.loads(config_path.read_text())

config["loras"]["$name"] = {
    "volume": "$volume",
    "path": "$dest",
    "source_checkpoint": "$source_checkpoint",
    "dataset": "$dataset" if "$dataset" else None,
    "saved_at": "$timestamp"
}

# Remove None values
config["loras"]["$name"] = {k: v for k, v in config["loras"]["$name"].items() if v is not None}

config_path.write_text(json.dumps(config, indent=2) + "\n")
print(f"Updated {config_path}")
EOF
}

if [[ -n "$FROM_VOLUME" ]]; then
    TMP_DIR="$(mktemp -d)"
    echo "Copying checkpoint via get/put:"
    echo "  source volume/path: $FROM_VOLUME:$CHECKPOINT"
    echo "  dest volume:        $VOLUME"
    echo "  name:               $NAME"
    echo "  dest:               $DEST"
    echo
    $MODAL_BIN volume get "$FROM_VOLUME" "$CHECKPOINT" "$TMP_DIR"
    SRC_PATH="$TMP_DIR/$(basename "$CHECKPOINT")"
    if [[ ! -e "$SRC_PATH" ]]; then
        SRC_PATH="$TMP_DIR"
    fi
    $MODAL_BIN volume put "$VOLUME" "$SRC_PATH" "$DEST"
    rm -rf "$TMP_DIR"

    # Update config
    update_config "$NAME" "$VOLUME" "$DEST" "$CHECKPOINT" "$DATASET" "$FROM_VOLUME"

    echo "Done."
    exit 0
fi

# Local upload path validation
CHECKPOINT="$(realpath "${CHECKPOINT}")"
if [[ ! -d "$CHECKPOINT" && ! -f "$CHECKPOINT" ]]; then
    echo "Error: checkpoint path does not exist: $CHECKPOINT" >&2
    exit 1
fi

echo "Uploading checkpoint:"
echo "  source:   $CHECKPOINT"
echo "  volume:   $VOLUME"
echo "  name:     $NAME"
echo "  dest:     $DEST"
echo

$MODAL_BIN volume put "$VOLUME" "$CHECKPOINT" "$DEST"

# Update config
update_config "$NAME" "$VOLUME" "$DEST" "$CHECKPOINT" "$DATASET" ""

echo "Done."
