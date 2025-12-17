#!/usr/bin/env bash
set -euo pipefail

# Directory where this script lives (scripts/ or bin/)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Repo root = one level up from that directory
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$ROOT_DIR/bench"

wget -O "$ROOT_DIR/bench/ShareGPT_V3_unfiltered_cleaned_split.json" \
  "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
