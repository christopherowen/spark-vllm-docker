#!/usr/bin/env bash
set -euo pipefail

log() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# Repo root (one level up from this script directory)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

COMPOSE_SERVICE="vllm"

BASE_URL="http://127.0.0.1:8000"
MODEL="gpt-oss-120b"
TOKENIZER_MODEL="openai/gpt-oss-120b"
OUTPUT_LEN="1024"
NUM_PROMPTS_LIST="1 10"
MAX_CONCURRENCY="2"

DATASET_PATH="/bench/ShareGPT_V3_unfiltered_cleaned_split.json"

# find the container id by name or default to vllm
COMPOSE_SERVICE="vllm"
CONTAINER_ID="$(docker compose ps -q vllm 2>/dev/null || true)"
CONTAINER_ID="${CONTAINER_ID:-$COMPOSE_SERVICE}"

# ---- Warmup ----
log "[warmup] tiny request -> $BASE_URL"
curl -fsS --max-time 15 "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\":\"user\",\"content\":\"Say hi.\"}],
    \"max_tokens\": 16,
    \"temperature\": 0
  }" >/dev/null

# ---- Bench loop ----
for NUM_PROMPTS in $NUM_PROMPTS_LIST; do
  log ""
  log "================ BENCH ================="
  log "model          : $MODEL"
  log "tokenizer      : $TOKENIZER_MODEL"
  log "base_url       : $BASE_URL"
  log "dataset_path   : $DATASET_PATH"
  log "num_prompts    : $NUM_PROMPTS"
  log "output_len     : $OUTPUT_LEN"
  log "max_concurrency: $MAX_CONCURRENCY"
  log "========================================"
  log ""

  # tqdm/progress output often goes to stderr; keep it visible (2>&1)
  docker exec -i "${TTY_ARGS[@]}" "$CONTAINER_ID" bash -lc "
    vllm bench serve \
      --backend openai-chat \
      --base-url '${BASE_URL}' \
      --endpoint /v1/chat/completions \
      --model '${TOKENIZER_MODEL}' \
      --served-model-name '${MODEL}' \
      --temperature 1.0 \
      --top-p 1.0 \
      --top-k 0 \
      --dataset-name sharegpt \
      --dataset-path '${DATASET_PATH}' \
      --num-prompts '${NUM_PROMPTS}' \
      --sharegpt-output-len '${OUTPUT_LEN}' \
      --max-concurrency '${MAX_CONCURRENCY}'
  " 2>&1
done

