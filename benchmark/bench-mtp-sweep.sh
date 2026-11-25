#!/usr/bin/env bash
# MTP sweep: measure speculative decoding benefit for --spec-draft-n-max 0..4.
#
# Requires the MTP draft model (configs/qwen3.8-27b-iq3s-mtp.env).
# n_max=0 disables drafting at runtime, which gives the non-MTP baseline
# without reloading a different model.
#
# Usage:
#   ./benchmark/bench-mtp-sweep.sh                 # sweep 0 1 2 3 4
#   ./benchmark/bench-mtp-sweep.sh 2 3             # explicit values
#   CTX=65536 ./benchmark/bench-mtp-sweep.sh       # different context
#
# Output: benchmark/mtp-sweep-<timestamp>.csv

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

NS=("$@")
[[ ${#NS[@]} -gt 0 ]] || NS=(0 1 2 3 4)

if ! grep -q '^DRAFT_MODEL=/models/.' .env 2>/dev/null; then
    echo "FAIL: the active .env has no DRAFT_MODEL. Activate the MTP profile first:" >&2
    echo "      cp configs/qwen3.8-27b-iq3s-mtp.env .env && docker compose down && docker compose up -d" >&2
    exit 1
fi

CTX="${CTX:-32768}"
PROBES="${PROBES:-4}"
STAMP="$(now)"
CSV="$OUT_DIR/mtp-sweep-$STAMP.csv"
PAYLOAD="$(mktemp /tmp/llama16-mtp.XXXXXX.json)"
trap 'rm -f "$PAYLOAD"' EXIT

csv_row "$CSV" n_max probe gen_tps prefill_tps draft_n draft_accepted accept_rate vram_peak_mib

# Short, varied generation tasks: draft acceptance depends on the workload.
PROMPTS=(
    "Write a bash function that retries a command three times with exponential backoff. Return only the code."
    "Explain in about 150 words why KV cache quantization reduces VRAM usage in transformer inference."
    "List 10 common causes of CUDA out-of-memory errors when serving LLMs, one per line, no explanations."
    "Write a Python dataclass for an HTTP request with method, url, headers and body, then a helper that serializes it to JSON."
)

for n in "${NS[@]}"; do
    log "n_max=$n ctx=$CTX — recreating stack"
    recreate CTX="$CTX" SPEC_TYPE=draft-mtp SPEC_DRAFT_N_MAX="$n"

    if ! wait_health 120; then
        echo "FAIL: server not healthy at n_max=$n" >&2
        docker compose logs --tail=40 >&2 || true
        exit 1
    fi
    abort_if_cpu

    for i in "${!PROMPTS[@]}"; do
        python3 - "$i" "${PROMPTS[$i]}" "$PAYLOAD" <<'PY'
import json, sys, time

idx, prompt, path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
if idx == 0:
    prompt = f"[run {time.time_ns()}] {prompt}"  # defeat prompt cache reuse
payload = {
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 300,
    "temperature": 0.0,
    "reasoning_effort": "low",
}
with open(path, "w") as f:
    json.dump(payload, f)
PY

        response=$(curl -s --max-time 600 "http://localhost:$PORT/v1/chat/completions" \
            -H 'Content-Type: application/json' -d @"$PAYLOAD")

        gen=$(json_num "$response" predicted_per_second)
        prefill=$(json_num "$response" prompt_per_second)
        draft_n=$(json_num "$response" draft_n)
        draft_acc=$(json_num "$response" draft_n_accepted)

        if [[ -z "$gen" ]]; then
            echo "FAIL: no timings at n_max=$n probe=$i — $(echo "$response" | head -c 300)" >&2
            continue
        fi

        accept=$(awk -v a="${draft_acc:-0}" -v b="${draft_n:-0}" \
            'BEGIN { if (b > 0) printf "%.3f", a / b; else printf "0" }')

        log "n_max=$n probe=$((i + 1)) gen=${gen} tok/s prefill=${prefill} accept=${accept}"
        csv_row "$CSV" "$n" "$((i + 1))" "$gen" "$prefill" "${draft_n:-0}" "${draft_acc:-0}" "$accept" "$(vram_used)"
    done
done

echo ""
log "results: $CSV"
column -s, -t < "$CSV" 2>/dev/null || cat "$CSV"

echo ""
log "per-n averages:"
awk -F, 'NR > 1 { g[$1] += $3; a[$1] += $8; c[$1]++ }
         END {
             printf "  %-6s %-12s %-14s %s\n", "n_max", "gen tok/s", "accept rate", "probes"
             for (k in g) printf "  %-6s %-12.1f %-14.3f %d\n", k, g[k] / c[k], a[k] / c[k], c[k]
         }' "$CSV" | sort -k1,1n

echo ""
log "restore the default profile with: docker compose down && docker compose up -d"
