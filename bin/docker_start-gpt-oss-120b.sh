#!/usr/bin/env bash
set -euo pipefail

# Directory where this script lives (scripts/ or bin/)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Repo root = one level up from that directory
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

$ROOT_DIR/bin/get_bench_data.sh

docker compose -f $ROOT_DIR/docker-compose-gpt-oss-120b.yml up -d vllm
