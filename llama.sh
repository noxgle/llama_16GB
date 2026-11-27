#!/usr/bin/env bash
# llama.sh — model control via `docker run` (dev/testing).
# Production runs through docker-compose.yml.
#
# Usage:
#   ./llama.sh start q38          # default: UD-IQ3_S + MTP, 48K
#   ./llama.sh start q38-q3kxl    # quality: UD-Q3_K_XL, 32K
#   ./llama.sh start q38-long     # long context: UD-IQ3_S, 128K
#   ./llama.sh start inkling2     # Inkling-2-Mini, MoE + CPU experts, 128K
#   ./llama.sh status | stop | logs [model] | pull | health
#
# Env:
#   LLAMA_IMAGE  override image (default: pinned stable tag)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${LLAMA_IMAGE:-ghcr.io/noxgle/llama-server-16gb:stable-b12716-v1}"
NAME_PREFIX="llama16"

GREEN='\033[0;32m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'
info() { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*" >&2; }

config_for() {
    case "$1" in
        q38)       echo "configs/qwen3.8-27b-iq3s-mtp.env" ;;
        q38-q3kxl) echo "configs/qwen3.8-27b-q3kxl.env" ;;
        q38-long)  echo "configs/qwen3.8-27b-iq3s-longctx.env" ;;
        inkling2)  echo "configs/inkling-2-mini.env" ;;
        *) return 1 ;;
    esac
}

stop_all() {
    local names
    names=$(docker ps -a --filter "name=${NAME_PREFIX}-" --format '{{.Names}}' || true)
    for n in $names; do
        info "stopping $n"
        docker rm -f "$n" >/dev/null 2>&1 || true
    done
}

start() {
    local model="$1" cfg
    cfg=$(config_for "$model") || { err "unknown model: $model (q38|q38-q3kxl|q38-long|inkling2)"; exit 1; }
    [[ -f "$ROOT/$cfg" ]] || { err "config not found: $cfg"; exit 1; }

    set -a
    # shellcheck disable=SC1090
    source "$ROOT/$cfg"
    set +a

    stop_all

    local args=(
        --alias "${ALIAS:-qwen3.8-27b}"
        --jinja
        -c "${CTX:-49152}"
        -n "${N_PREDICT:--1}"
        --port "${PORT:-8089}"
        --host "${HOST:-0.0.0.0}"
        -ngl "${NGLAYERS:-999}"
        -fa "${FLASHATTN:-on}"
        -b "${BATCH:-1024}"
        -ub "${UBATCH:-512}"
        -t "${THREADS:-4}"
        --threads-batch "${THREADS_BATCH:-4}"
        --parallel "${PARALLEL:-1}"
        --poll "${POLL:-50}"
        --load-mode "${LOAD_MODE:-mlock}"
        --fit off
        -ctk "${CACHE_TYPE_K:-q8_0}"
        -ctv "${CACHE_TYPE_V:-q8_0}"
        --ctx-checkpoints "${CTX_CHECKPOINTS:-4}"
        --reasoning "${REASONING:-auto}"
        --cache-ram "${CACHE_RAM:-4096}"
        --cache-reuse "${CACHE_REUSE:-256}"
        --chat-template-kwargs "{\"preserve_thinking\": ${PRESERVE_THINKING:-false}, \"reasoning_effort\": \"${REASONING_EFFORT:-medium}\"}"
        --threads-http "${THREADS_HTTP:-2}"
        --metrics
        --spec-type "${SPEC_TYPE:-none}"
        --spec-draft-n-max "${SPEC_DRAFT_N_MAX:-2}"
        --gpu-layers-draft "${GPU_LAYERS_DRAFT:-0}"
        --spec-draft-n-min "${SPEC_DRAFT_N_MIN:-0}"
        --spec-draft-p-min "${SPEC_DRAFT_P_MIN:-0.0}"
        --repeat-penalty "${REPEAT_PENALTY:-1.0}"
        --temp "${TEMPERATURE:-1.0}"
        --top-p "${TOP_P:-0.95}"
        --top-k "${TOP_K:-20}"
        --min-p "${MIN_P:-0.05}"
    )

    if [[ -n "${CPU_MOE_FLAG:-}" ]]; then
        args+=("$CPU_MOE_FLAG")
    fi
    if [[ -n "${OP_OFFLOAD_FLAG:-}" ]]; then
        args+=("$OP_OFFLOAD_FLAG")
    fi
    if [[ -n "${MOE_CACHE_MIB:-}" ]]; then
        args+=(--moe-cache-mib "$MOE_CACHE_MIB")
    fi
    if [[ -n "${MOE_CACHE_CPU_SHARE:-}" ]]; then
        args+=(--moe-cache-cpu-share "$MOE_CACHE_CPU_SHARE")
    fi

    if [[ -n "${MMPROJ:-}" ]]; then
        args+=(--mmproj "$MMPROJ")
    else
        args+=(--no-mmproj)
    fi
    if [[ -n "${DRAFT_MODEL:-}" ]]; then
        args+=(-md "$DRAFT_MODEL")
    fi

    info "starting $model ($(basename "$cfg")) on port ${PORT:-8089}"
    docker run -d \
        --name "${NAME_PREFIX}-${model}" \
        --restart unless-stopped \
        --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e GGML_CUDA_GRAPH_OPT=1 \
        -p "${PORT:-8089}:${PORT:-8089}" \
        -v "$ROOT/models:/models:ro" \
        "$IMAGE" \
        -m "$MODEL" "${args[@]}" >/dev/null

    info "waiting for health..."
    for _ in $(seq 1 60); do
        if curl -sf "http://localhost:${PORT:-8089}/health" | grep -q '"ok"'; then
            ok "server ready: http://localhost:${PORT:-8089}"
            return 0
        fi
        sleep 5
    done
    err "server did not become healthy — check: ./llama.sh logs $model"
    exit 1
}

case "${1:-help}" in
    start)
        [[ -n "${2:-}" ]] || { err "usage: $0 start {q38|q38-q3kxl|q38-long|inkling2}"; exit 1; }
        start "$2" ;;
    stop)   stop_all ;;
    status)
        docker ps -a --filter "name=${NAME_PREFIX}-" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader ;;
    logs)
        docker logs -f --tail=100 "${NAME_PREFIX}-${2:-q38}" ;;
    pull)
        docker pull "$IMAGE" ;;
    health)
        curl -s "http://localhost:${PORT:-8089}/health"; echo ;;
    help|--help|-h)
        sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//' ;;
    *)
        err "unknown command: $1"; exit 1 ;;
esac
