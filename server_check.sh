#!/usr/bin/env bash
# server_check.sh — one-shot diagnostic for a llama_16GB host.
# Read-only. Safe to run from cron or by hand after a reboot.

set -uo pipefail

PORT="${PORT:-8089}"
CONTAINER="${CONTAINER:-llama16-llama-server-1}"

echo "== host =="
hostname; date; uptime

echo ""
echo "== gpu =="
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv || echo "nvidia-smi failed"

echo ""
echo "== gpu processes =="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true

echo ""
echo "== container =="
docker ps -a --filter "name=llama16" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true

echo ""
echo "== health =="
curl -s -o /dev/null -w "HTTP %{http_code}\n" "http://localhost:$PORT/health" || true

echo ""
echo "== cuda / device lines (last 200) =="
docker logs --tail=200 "$CONTAINER" 2>&1 | grep -iE "cuda|device|offload|error|fail" | tail -15 || true

echo ""
echo "== ram =="
free -h | head -3

echo ""
echo "== disk =="
df -h / /opt 2>/dev/null | tail -2
