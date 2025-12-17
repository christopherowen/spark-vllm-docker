#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DEST_FILE="$ROOT_DIR/bench/ShareGPT_V3_unfiltered_cleaned_split.json"

if [[ ! -s "$DEST_FILE" ]]; then
  mkdir -p -- "$(dirname -- "$DEST_FILE")"
  wget -O "$DEST_FILE" \
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
fi
