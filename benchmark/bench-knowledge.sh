#!/usr/bin/env bash
# Knowledge / instruction-following smoke suite.
#
# The suite is deliberately small and deterministic: each item has expected
# substrings and an optional forbidden substring. It is a regression guard when
# switching quantization (UD-IQ3_S vs UD-Q3_K_XL), not a public benchmark.
#
# Usage:
#   ./benchmark/bench-knowledge.sh            # run against the running stack
#   PROFILE_NOTE=iq3s ./benchmark/bench-knowledge.sh
#
# Output: benchmark/knowledge-<timestamp>.md (answers + PASS/FAIL matrix)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

STAMP="$(now)"
OUT="$OUT_DIR/knowledge-$STAMP.md"
NOTE="${PROFILE_NOTE:-$(grep -m1 '^MODEL=' .env 2>/dev/null | cut -d= -f2 || echo unknown)}"
PAYLOAD="$(mktemp /tmp/llama16-kb.XXXXXX.json)"
trap 'rm -f "$PAYLOAD"' EXIT

wait_health 60 || { echo "FAIL: server not healthy" >&2; exit 1; }
abort_if_cpu

# id|question|must contain (case-insensitive, ; separated)|must not contain
ITEMS=(
  "arith|What is 17 * 23 + 19? Answer with the number only.|410|"
  "prime|Is 91 prime? Answer yes or no, then one sentence why.|no|yes"
  "bash|Write a bash one-liner that prints the 3 largest files under /var/log by size.|du;sort;head|"
  "sql|Given tables users(id, name) and orders(id, user_id, total), write a query returning each user name and their total order sum, including users with no orders.|left join|"
  "python|Write a Python function chunk(lst, n) that splits a list into consecutive chunks of size n (last chunk may be shorter). Return only code.|def chunk|"
  "gguf|In one paragraph, explain what a GGUF quantization type like Q4_K_M describes.|quant|"
  "ctx|A model has 64 layers, 4 KV heads, head_dim 256, and only 16 layers keep a KV cache. How many bytes per token does an fp16 KV cache use? Show the arithmetic.|byte|"
  "json|Return exactly this JSON and nothing else: {\"ok\": true, \"n\": 3}|\"ok\": true|"
)

{
    echo "# Knowledge suite — $(date -Iseconds)"
    echo ""
    echo "- model/config: \`$NOTE\`"
    echo "- endpoint: \`http://localhost:$PORT\`"
    echo "- image: \`$(docker inspect --format '{{.Config.Image}}' llama16-llama-server-1 2>/dev/null || echo unknown)\`"
    echo ""
    echo "| id | result | note |"
    echo "|---|---|---|"
} > "$OUT"

pass=0
fail=0

for item in "${ITEMS[@]}"; do
    IFS='|' read -r id question must must_not <<< "$item"

    python3 - "$question" "$PAYLOAD" <<'PY'
import json, sys, time

question, path = sys.argv[1], sys.argv[2]
with open(path, "w") as f:
    json.dump({
        "model": "qwen3.8-27b",
        "messages": [
            {"role": "system", "content": "Be precise and concise. Follow the requested output format exactly."},
            {"role": "user", "content": question},
        ],
        "max_tokens": 700,
        "temperature": 0.0,
        "reasoning_effort": "medium",
    }, f)
PY

    response=$(curl -s --max-time 600 "http://localhost:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' -d @"$PAYLOAD")

    answer=$(printf '%s' "$response" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print(d["choices"][0]["message"].get("content") or "")
except Exception as exc:
    print(f"__ERROR__: {exc}")
')

    verdict="PASS"
    note=""
    if [[ "$answer" == __ERROR__* ]]; then
        verdict="FAIL"; note="request failed"
    else
        lower=$(printf '%s' "$answer" | tr '[:upper:]' '[:lower:]')
        IFS=';' read -ra needles <<< "$must"
        for needle in "${needles[@]}"; do
            [[ -z "$needle" ]] && continue
            if ! grep -qF -- "$(printf '%s' "$needle" | tr '[:upper:]' '[:lower:]')" <<< "$lower"; then
                verdict="FAIL"; note+="missing: $needle; "
            fi
        done
        if [[ -n "$must_not" ]]; then
            if grep -qiF -- "$must_not" <<< "$answer"; then
                verdict="FAIL"; note+="forbidden: $must_not; "
            fi
        fi
    fi

    if [[ "$verdict" == PASS ]]; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
    fi

    log "$id: $verdict ${note:+($note)}"
    {
        printf '| %s | %s | %s |\n' "$id" "$verdict" "${note:-ok}"
        printf '\n<details><summary>%s — answer</summary>\n\n```\n%s\n```\n\n</details>\n\n' "$id" "$answer"
    } >> "$OUT"
done

{
    echo "## Summary"
    echo ""
    echo "- PASS: $pass"
    echo "- FAIL: $fail"
    echo "- total: $((pass + fail))"
} >> "$OUT"

echo ""
log "results: $OUT"
log "PASS=$pass FAIL=$fail"
