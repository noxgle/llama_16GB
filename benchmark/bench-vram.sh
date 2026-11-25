#!/usr/bin/env bash
# Context sweep: measure VRAM peak, prefill and generation speed per context size.
#
# Recreates the compose stack for each CTX value, fills roughly 70% of the
# context with a synthetic prompt, then records timings and peak VRAM.
#
# Usage:
#   ./benchmark/bench-vram.sh                    # default sweep 16384 32768 49152 65536 131072
#   ./benchmark/bench-vram.sh 32768 49152        # explicit list
#   KV=q4_0 ./benchmark/bench-vram.sh 131072     # override KV cache type
#
# Output: benchmark/vram-<timestamp>.csv + a summary on stdout.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SIZES=("$@")
[[ ${#SIZES[@]} -gt 0 ]] || SIZES=(16384 32768 49152 65536 131072)

KV_K="${CACHE_TYPE_K:-q8_0}"
KV_V="${CACHE_TYPE_V:-q8_0}"
STAMP="$(now)"
CSV="$OUT_DIR/vram-$STAMP.csv"

PAYLOAD="$(mktemp /tmp/llama16-payload.XXXXXX.json)"
PEAK_FILE="$(mktemp /tmp/llama16-vram.XXXXXX)"
trap 'rm -f "$PAYLOAD" "$PEAK_FILE"' EXIT

csv_row "$CSV" ctx kv vram_peak_mib vram_after_mib prefill_tps gen_tps ttft_s ttft_ms prompt_tokens gen_tokens

# Background VRAM sampler: writes the highest value seen to $PEAK_FILE.
# Killing the sampler PID makes nvidia-smi die on SIGPIPE.
start_vram_sampler() {
    : > "$PEAK_FILE"
    python3 - "$PEAK_FILE" <<'PY' &
import sys, subprocess

peak, f = 0, open(sys.argv[1], "w")
proc = subprocess.Popen(
    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "250"],
    stdout=subprocess.PIPE, text=True)
for line in proc.stdout:
    try:
        v = int(line.strip())
    except ValueError:
        continue
    if v > peak:
        peak = v
        f.write(f"{peak}\n")
        f.flush()
PY
    SAMPLER_PID=$!
}

stop_vram_sampler() {
    kill "$SAMPLER_PID" 2>/dev/null || true
    pkill -P "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
    tail -1 "$PEAK_FILE" 2>/dev/null | tr -d ' ' || true
}

# Build a synthetic prompt of ~target_tokens tokens (~4 chars/token for English).
make_payload() {
    local target_tokens="$1"
    python3 - "$target_tokens" "$PAYLOAD" <<'PY'
import json, sys

target, path = int(sys.argv[1]), sys.argv[2]
body = (
    "Distributed systems coordinate independent machines through message passing. "
    "Consensus protocols such as Paxos and Raft let replicas agree on an ordered log "
    "even when nodes fail. Replication improves availability but introduces latency "
    "and consistency trade-offs. Sharding partitions data across nodes to scale writes. "
    "Observability relies on metrics, structured logs and distributed traces. "
)
text = body * max(1, (target * 4) // len(body))
prompt = (
    "Read the following reference text and answer the question at the end.\n\n"
    + text
    + "\n\nQuestion: list five distinct topics mentioned above, one per line."
)
payload = {
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 200,
    "temperature": 0.0,
    "reasoning_effort": "low",
}
with open(path, "w") as f:
    json.dump(payload, f)
print(len(prompt))
PY
}

for ctx in "${SIZES[@]}"; do
    log "ctx=$ctx kv=$KV_K/$KV_V — recreating stack"
    recreate CTX="$ctx" CACHE_TYPE_K="$KV_K" CACHE_TYPE_V="$KV_V"

    if ! wait_health 120; then
        echo "FAIL: server not healthy at ctx=$ctx" >&2
        docker compose logs --tail=40 >&2 || true
        exit 1
    fi
    abort_if_cpu

    target=$(( ctx * 70 / 100 ))
    make_payload "$target"

    start_vram_sampler
    t0=$(date +%s%3N)
    response=$(curl -s --max-time 1200 "http://localhost:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' -d @"$PAYLOAD")
    t1=$(date +%s%3N)
    peak=$(stop_vram_sampler)
    elapsed_ms=$(( t1 - t0 ))

    prefill=$(json_num "$response" prompt_per_second)
    gen=$(json_num "$response" predicted_per_second)
    prompt_n=$(json_num "$response" prompt_n)
    gen_n=$(json_num "$response" predicted_n)
    ttft_s=$(awk -v ms="$elapsed_ms" -v p="${prefill:-0}" -v n="${prompt_n:-0}" \
        'BEGIN { if (p > 0) printf "%.2f", n / p; else printf "%.2f", ms / 1000 }')

    if [[ -z "$prefill" ]]; then
        echo "FAIL: no timings at ctx=$ctx — $(echo "$response" | head -c 300)" >&2
        docker compose logs --tail=30 >&2 || true
        exit 1
    fi

    log "ctx=$ctx peak=${peak}MiB prefill=${prefill} tok/s gen=${gen} tok/s ttft=${ttft_s}s"
    csv_row "$CSV" "$ctx" "$KV_K" "${peak:-0}" "$(vram_used)" \
        "$prefill" "$gen" "$ttft_s" "${elapsed_ms}" "$prompt_n" "$gen_n"
done

echo ""
log "results: $CSV"
column -s, -t < "$CSV" 2>/dev/null || cat "$CSV"
echo ""
log "restore the default profile with: docker compose down && docker compose up -d"
