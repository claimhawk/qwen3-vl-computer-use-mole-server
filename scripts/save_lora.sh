#!/usr/bin/env bash
# Copy a LoRA checkpoint into a Modal volume for sharing across apps.
# Supports uploading from local disk or copying from one Modal volume to another.
# Usage:
#   # upload local -> volume
#   scripts/save_lora.sh --checkpoint /path/to/ckpt --name calendar-tasks [--volume claimhawk-checkpoints] [--dest subdir/in/volume]
#   # copy within Modal volumes (no local checkpoint)
#   scripts/save_lora.sh --from-volume claimhawk-checkpoints --checkpoint mike-im-day-clicks-system-prompt-8B_20251120_180854/mike-day-clicks-graduated-8b/checkpoint-40 --name calendar-tasks [--volume claimhawk-checkpoints] [--dest subdir]

set -euo pipefail

VOLUME="moe-lora-checkpoints"
DEST=""
NAME=""
FROM_VOLUME=""

usage() {
    echo "Usage: $0 --checkpoint <path_or_volume_path> --name <id> [--volume ${VOLUME}] [--dest subdir/in/volume] [--from-volume <vol>]" >&2
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

echo "Done."
