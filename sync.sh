#!/usr/bin/env bash
# sync.sh — workstation <-> server helper for llama_16GB.
#
# Usage: ./sync.sh <command>
#
# Env:
#   LLAMA_HOST  (default: root@192.168.200.38)
#   LLAMA_DIR   (default: /opt/llama16)
#
# Commands: push pull deploy rebuild start stop restart logs status health ssh config

set -euo pipefail

SERVER="${LLAMA_HOST:-root@192.168.200.38}"
REMOTE_DIR="${LLAMA_DIR:-/opt/llama16}"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8089}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err()  { echo -e "${RED}[ERROR]${NC} $1"; }

ssh_exec() {
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$SERVER" "$1"
}

rsync_push() {
    log_info "sync: local -> $SERVER:$REMOTE_DIR"
    rsync -az --info=progress2 \
        -e "ssh -o StrictHostKeyChecking=accept-new" \
        --exclude=".git/" \
        --exclude=".env" \
        --exclude="models/" \
        --exclude="*.log" \
        --exclude="build/" \
        --exclude="benchmark/*.csv" \
        --exclude="benchmark/*.json" \
        --exclude="benchmark/*.txt" \
        --exclude=".opencode/" \
        "$LOCAL_DIR/" "$SERVER:$REMOTE_DIR/"
    log_ok "sync done"
}

rsync_pull() {
    log_info "sync: $SERVER:$REMOTE_DIR -> local (configs + results)"
    rsync -az --info=progress2 \
        -e "ssh -o StrictHostKeyChecking=accept-new" \
        --include="configs/" --include="*.env" \
        --include="benchmark/" --include="benchmark/*.csv" --include="benchmark/*.json" \
        --include="*.md" \
        --exclude="*" \
        "$SERVER:$REMOTE_DIR/" "$LOCAL_DIR/"
    log_ok "pull done"
}

cmd_deploy() {
    rsync_push
    log_info "recreating stack (down + up to reload .env)..."
    ssh_exec "cd $REMOTE_DIR && docker compose down && docker compose up -d"
    log_ok "deployed"
}

cmd_status() {
    echo -e "${BLUE}=== containers ===${NC}"
    ssh_exec "cd $REMOTE_DIR && docker compose ps"
    echo ""
    echo -e "${BLUE}=== GPU ===${NC}"
    ssh_exec "nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader || echo 'no GPU'"
}

cmd_health() {
    echo -e "${BLUE}=== health ===${NC}"
    local code
    code=$(ssh_exec "curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null || echo failed")
    [[ "$code" == "200" ]] && log_ok "HTTP 200" || log_err "not healthy (HTTP: $code)"
    echo ""
    echo -e "${BLUE}=== VRAM / RAM ===${NC}"
    ssh_exec "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; free -h | head -2"
    echo ""
    echo -e "${BLUE}=== container usage ===${NC}"
    ssh_exec "docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}' 2>/dev/null || true"
}

COMMAND="${1:-help}"
case "$COMMAND" in
    push)     rsync_push ;;
    pull)     rsync_pull ;;
    deploy)   cmd_deploy ;;
    rebuild)  rsync_push; ssh_exec "cd $REMOTE_DIR && LLAMA_NATIVE=ON docker compose up -d --build" ;;
    start)    ssh_exec "cd $REMOTE_DIR && docker compose up -d" ;;
    stop)     ssh_exec "cd $REMOTE_DIR && docker compose down" ;;
    restart)  ssh_exec "cd $REMOTE_DIR && docker compose down && docker compose up -d" ;;
    logs)     ssh_exec "cd $REMOTE_DIR && docker compose logs --tail=100 -f" ;;
    status)   cmd_status ;;
    health)   cmd_health ;;
    ssh)      ssh "$SERVER" ;;
    config)   ssh_exec "cd $REMOTE_DIR && cat .env" ;;
    help|--help|-h)
        sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//' ;;
    *) log_err "unknown command: $COMMAND"; exit 1 ;;
esac
