#!/usr/bin/env bash
# start-router.sh — start (or resume) llama-server in router mode.
#
# Single source of truth for the router container flags. Used by
# deploy/install-llama.sh and deploy/systemd/llama16-router.service.
#
# Env overrides:
#   LLAMA_IMAGE            image (default: pinned stable-b12716-v1)
#   LLAMA_DIR              install dir (default: directory containing this script's parent)
#   LLAMA_PORT             host/container port (default: 8089)
#   LLAMA_ROUTER_CONTAINER container name (default: llama16-router)
#   LLAMA_MODELS_MAX       presets resident at once (default: 1 for 16 GB)

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$DIR}"
IMAGE="${LLAMA_IMAGE:-ghcr.io/noxgle/llama-server-16gb:stable-b12716-v1}"
PORT="${LLAMA_PORT:-8089}"
NAME="${LLAMA_ROUTER_CONTAINER:-llama16-router}"
MODELS_MAX="${LLAMA_MODELS_MAX:-1}"
PRESET="$LLAMA_DIR/configs/router.ini"

[[ -f "$PRESET" ]] || { echo "ERROR: router config missing: $PRESET" >&2; exit 1; }

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "router: $NAME already running"
    exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "router: starting existing container $NAME"
    docker start "$NAME" >/dev/null
else
    # Port 8089 can only have one owner. The single-model compose stack is
    # the fallback path, so it must step aside when router mode is selected.
    if docker ps -a --format '{{.Names}}' | grep -qx 'llama16-llama-server-1'; then
        echo "router: removing fallback compose container llama16-llama-server-1"
        docker rm -f llama16-llama-server-1 >/dev/null
    fi
    echo "router: creating $NAME from $IMAGE"
    docker run -d --name "$NAME" --restart unless-stopped \
        --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e GGML_CUDA_GRAPH_OPT=1 \
        -p "$PORT:$PORT" \
        -v "$LLAMA_DIR/models:/models:ro" \
        -v "$PRESET:/router.ini:ro" \
        "$IMAGE" \
        --models-preset /router.ini \
        --models-max "$MODELS_MAX" --models-memory-margin 512 \
        --host 0.0.0.0 --port "$PORT" --alias router --jinja --threads-http 2 --fit off \
        --metrics \
        >/dev/null
fi

for _ in $(seq 1 60); do
    if curl -sf "http://localhost:$PORT/health" 2>/dev/null | grep -q '"ok"'; then
        echo "router: healthy on port $PORT"
        exit 0
    fi
    sleep 2
done

echo "ERROR: router did not become healthy — check: docker logs $NAME" >&2
exit 1
