#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
MODEL="${1:-"$SCRIPT_DIR/models/model.gguf"}"

if ! command -v llama-server >/dev/null 2>&1; then
    echo "llama-server is not installed or not on PATH"
    exit 1
fi

llama-server \
    --model "$MODEL" \
    --ctx-size 4096
