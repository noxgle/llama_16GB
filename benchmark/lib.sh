#!/usr/bin/env bash
# Shared helpers for benchmark scripts.
# Run on the server, inside the repo dir (e.g. /opt/llama16).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8089}"
OUT_DIR="${OUT_DIR:-$ROOT/benchmark}"
mkdir -p "$OUT_DIR"

now() { date +%Y%m%d-%H%M%S; }

vram_used() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '
}

vram_total() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' '
}

wait_health() {
    local tries="${1:-90}"
    for _ in $(seq 1 "$tries"); do
        if curl -sf "http://localhost:$PORT/health" 2>/dev/null | grep -q '"ok"'; then
            return 0
        fi
        sleep 2
    done
    return 1
}

# recreate stack with env overrides: recreate CTX=65536 CACHE_TYPE_K=q4_0 ...
recreate() {
    docker compose down >/dev/null 2>&1 || true
    env "$@" docker compose up -d --force-recreate >/dev/null
}

abort_if_cpu() {
    local v
    v=$(vram_used)
    if [[ -z "$v" || "$v" -lt 8000 ]]; then
        echo "ABORT: VRAM ${v:-0} MiB — model is not on GPU (CPU fallback?)" >&2
        exit 1
    fi
}

# extract a numeric field from llama-server timings JSON
json_num() {
    local doc="$1" key="$2"
    grep -o "\"${key}\": *[0-9.]*" <<<"$doc" | head -1 | sed 's/.*: *//'
}

chat() {
    local prompt="$1" max_tokens="${2:-256}"
    curl -s "http://localhost:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(printf '{"model":"%s","messages":[{"role":"user","content":%s}],"max_tokens":%s,"temperature":0.2}' \
            "${ALIAS:-qwen3.8-27b}" "$(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" "$max_tokens")"
}

csv_row() {
    local file="$1"; shift
    printf '%s\n' "$(IFS=,; echo "$*")" >> "$file"
}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
