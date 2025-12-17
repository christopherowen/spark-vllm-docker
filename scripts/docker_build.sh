#!/usr/bin/env bash
set -euo pipefail

# Directory where this script lives (scripts/ or bin/)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Repo root = one level up from that directory
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

mkdir $ROOT_DIR/logs

DOCKER_BUILDKIT=1 docker build --progress=plain -t spark-vllm:latest . 2>&1 | tee "$ROOT_DIR/logs/docker_build.$(date +%Y%m%d-%H%M%S).log"
