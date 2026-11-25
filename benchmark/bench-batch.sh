#!/usr/bin/env bash
# Prefill sweep: measure prompt-processing throughput for BATCH/UBATCH pairs.
#
# Generation speed is mostly unaffected by b/ub; the interesting metric here is
# prompt_per_second on a large prompt. Keep the context fixed across runs.
#
# Usage:
#   ./benchmark/bench-batch.sh                        # default pairs
#   ./benchmark/bench-batch.sh 512:256 2048:512       # explicit batch:ubatch
#   CTX=65536 TARGET_TOKENS=24000 ./benchmark/bench-batch.sh
#
# Output: benchmark/batch-<timestamp>.csv

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

PAIRS=("$@")
[[ ${#PAIRS[@]} -gt 0 ]] || PAIRS=(512:256 1024:512 2048:512 2048:1024)

CTX="${CTX:-32768}"
TARGET_TOKENS="${TARGET_TOKENS:-12000}"
STAMP="$(now)"
CSV="$OUT_DIR/batch-$STAMP.csv"
PAYLOAD="$(mktemp /tmp/llama16-batch.XXXXXX.json)"
trap 'rm -f "$PAYLOAD"' EXIT

csv_row "$CSV" batch ubatch prefill_tps gen_tps prompt_tokens gen_tokens total_s

make_payload() {
    local target="$1"
    python3 - "$target" "$PAYLOAD" <<'PY'
import json, sys, time

target, path = int(sys.argv[1]), sys.argv[2]
body = (
    "Prefill throughput depends on matrix multiplication shapes, which are driven "
    "by the batch size and the micro-batch size used by llama.cpp. Larger batches "
    "improve GPU utilization but consume more compute buffer memory. "
)
text = body * max(1, (target * 4) // len(body))
prompt = (
    f"[run {time.time_ns()}] Read this reference text and summarise it in 5 bullets.\n\n"
    + text
)
with open(path, "w") as f:
    json.dump({
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.0,
        "reasoning_effort": "low",
    }, f)
PY
}

make_payload "$TARGET_TOKENS"

for pair in "${PAIRS[@]}"; do
    b="${pair%%:*}"
    ub="${pair##*:}"
    [[ "$b" != "$ub" || -n "$b" ]] || { echo "bad pair: $pair" >&2; exit 1; }

    log "batch=$b ubatch=$ub ctx=$CTX — recreating stack"
    recreate CTX="$CTX" BATCH="$b" UBATCH="$ub"

    if ! wait_health 120; then
        echo "FAIL: server not healthy for batch=$b/$ub" >&2
        docker compose logs --tail=40 >&2 || true
        exit 1
    fi
    abort_if_cpu

    t0=$(date +%s%3N)
    response=$(curl -s --max-time 900 "http://localhost:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' -d @"$PAYLOAD")
    t1=$(date +%s%3N)

    prefill=$(json_num "$response" prompt_per_second)
    gen=$(json_num "$response" predicted_per_second)
    prompt_n=$(json_num "$response" prompt_n)
    gen_n=$(json_num "$response" predicted_n)

    if [[ -z "$prefill" ]]; then
        echo "FAIL: no timings for batch=$b/$ub — $(echo "$response" | head -c 300)" >&2
        continue
    fi

    total=$(awk -v a="$t0" -v z="$t1" 'BEGIN { printf "%.1f", (z - a) / 1000 }')
    log "batch=$b/$ub prefill=${prefill} tok/s gen=${gen} tok/s total=${total}s"
    csv_row "$CSV" "$b" "$ub" "$prefill" "$gen" "$prompt_n" "$gen_n" "$total"
done

echo ""
log "results: $CSV"
column -s, -t < "$CSV" 2>/dev/null || cat "$CSV"
awk -F, 'NR > 1 && $4 > best { best = $4; line = $0 } END { if (line) print "\nbest prefill: " line }' "$CSV"

echo ""
log "restore the default profile with: docker compose down && docker compose up -d"
