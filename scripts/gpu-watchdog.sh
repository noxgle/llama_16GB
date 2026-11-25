#!/usr/bin/env bash
# GPU watchdog for the llama_16GB stack.
# Detects CPU fallback (CUDA not initialized / no VRAM used) and self-heals.
# Intended to run from systemd timer every 2 minutes.
#
# Env overrides:
#   WATCHDOG_CONTAINER  (default: llama16-llama-server-1)
#   WATCHDOG_PORT       (default: 8089)
#   WATCHDOG_MIN_VRAM   (default: 2000 MiB — model must occupy several GiB)

set -euo pipefail

CONTAINER="${WATCHDOG_CONTAINER:-llama16-llama-server-1}"
PORT="${WATCHDOG_PORT:-8089}"
MIN_VRAM="${WATCHDOG_MIN_VRAM:-2000}"
LOG_FILE="/var/log/llama16-gpu-watchdog.log"
LOCK_FILE="/tmp/llama16-watchdog.lock"
COUNTER_FILE="/tmp/llama16-watchdog-attempts"
COOLDOWN_FILE="/tmp/llama16-watchdog-cooldown"
MAX_ATTEMPTS=2
COOLDOWN_MINUTES=30
HEALTH_URL="http://localhost:${PORT}/health"

log() {
    local msg
    msg="[$(date +'%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG_FILE" | logger -t llama16-gpu-watchdog 2>/dev/null || true
}

lock() {
    exec 200>"$LOCK_FILE"
    if ! flock -n 200; then
        log "WARN: another instance is running, exiting"
        exit 0
    fi
}

is_cpu_fallback() {
    if [[ ! -e /dev/nvidia-uvm ]]; then
        log "DETECT: /dev/nvidia-uvm missing (LXC passthrough or host module issue)"
        return 0
    fi

    local vram
    vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "")
    if [[ -n "${vram}" && "${vram}" -lt "${MIN_VRAM}" ]]; then
        log "DETECT: VRAM usage ${vram} MiB < ${MIN_VRAM} MiB"
        return 0
    fi

    if docker logs --tail=60 "$CONTAINER" 2>/dev/null | grep -q "ggml_cuda_init: failed"; then
        log "DETECT: ggml_cuda_init failed in recent logs"
        return 0
    fi

    if docker logs --tail=60 "$CONTAINER" 2>/dev/null | grep -q "no usable GPU found"; then
        log "DETECT: 'no usable GPU found' in recent logs"
        return 0
    fi

    return 1
}

wait_health() {
    local tries="${1:-60}"
    for _ in $(seq 1 "$tries"); do
        local code
        code=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null || echo "000")
        [[ "$code" == "200" ]] && return 0
        sleep 2
    done
    return 1
}

attempt_repair() {
    log "REPAIR: restarting container $CONTAINER"
    docker restart "$CONTAINER" >/dev/null 2>&1 || true
    if wait_health 60; then
        log "REPAIR: health OK after restart"
        return 0
    fi
    log "REPAIR: health NOT OK after restart"
    return 1
}

deep_repair() {
    log "DEEP-REPAIR: restarting docker"
    systemctl restart docker || true
    docker start "$CONTAINER" >/dev/null 2>&1 || true
    if wait_health 90; then
        log "DEEP-REPAIR: health OK after docker restart"
        return 0
    fi
    log "DEEP-REPAIR: health NOT OK after docker restart"
    return 1
}

main() {
    lock

    if [[ -f "$COOLDOWN_FILE" ]]; then
        local ts now elapsed
        ts=$(cat "$COOLDOWN_FILE" 2>/dev/null || echo 0)
        now=$(date +%s)
        elapsed=$(( now - ts ))
        if (( elapsed < COOLDOWN_MINUTES * 60 )); then
            log "COOLDOWN: $(( elapsed / 60 ))min of ${COOLDOWN_MINUTES}min"
            exit 0
        fi
        rm -f "$COOLDOWN_FILE"
        log "COOLDOWN: expired, resuming"
    fi

    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
        log "ERROR: container $CONTAINER is not running"
        exit 1
    fi

    local health
    health=$(curl -s "$HEALTH_URL" 2>/dev/null || echo '{"status":"error"}')
    if ! echo "$health" | grep -q '"status":"ok"'; then
        log "WARN: health not ok: $health"
    fi

    if ! is_cpu_fallback; then
        log "OK: GPU in use"
        exit 0
    fi

    if [[ ! -e /dev/nvidia-uvm ]]; then
        log "HARD-FAIL: /dev/nvidia-uvm missing — fix Proxmox host passthrough"
        date +%s > "$COOLDOWN_FILE"
        exit 1
    fi

    local attempts=0
    [[ -f "$COUNTER_FILE" ]] && attempts=$(cat "$COUNTER_FILE" 2>/dev/null || echo 0)

    if (( attempts >= MAX_ATTEMPTS )); then
        log "HARD-FAIL: max attempts reached, cooling down ${COOLDOWN_MINUTES}min"
        date +%s > "$COOLDOWN_FILE"
        rm -f "$COUNTER_FILE"
        exit 1
    fi

    attempts=$(( attempts + 1 ))
    echo "$attempts" > "$COUNTER_FILE"
    log "REPAIR: attempt $attempts/$MAX_ATTEMPTS"

    if attempt_repair; then
        sleep 5
        if is_cpu_fallback; then
            if deep_repair; then
                sleep 8
                if is_cpu_fallback; then
                    log "DEEP-REPAIR: still CPU fallback, next cycle will retry"
                    exit 1
                fi
                log "DEEP-REPAIR: SUCCESS"
                rm -f "$COUNTER_FILE"
                exit 0
            fi
            log "DEEP-REPAIR: failed"
            exit 1
        fi
        log "REPAIR: SUCCESS"
        rm -f "$COUNTER_FILE"
        exit 0
    fi

    log "REPAIR: restart failed"
    exit 1
}

main "$@"
