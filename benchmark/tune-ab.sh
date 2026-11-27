#!/usr/bin/env bash
# A/B verification: tuned vs baseline preset (router mode).
# Requires the `qwen-iq3s-mtp` preset (draft-mtp,ngram-mod + cache-ram 8192 +
# 8 checkpoints) in router.ini. Measures copy/edit and prompt-cache return.
#
# Part A: two copy/edit workloads (long source file reproduced with one change).
# Part B: 39K-token chat -> foreign request -> return; reports prompt_n/cached_tokens.
# Run on the server: bash benchmark/tune-ab.sh
# A/B verification of tuned vs baseline preset: copy/edit throughput and prompt-cache return.
set -uo pipefail
HOST=http://localhost:8089

ask() {
    local model="$1" payload="$2" tag="$3" resp
    resp="/tmp/tune-$tag-resp.json"
    local t0 t1
    t0=$(date +%s)
    curl -s --max-time 900 "$HOST/v1/chat/completions" -H 'Content-Type: application/json' \
        -d @"$payload" -o "$resp"
    t1=$(date +%s)
    python3 - "$resp" "$model" "$tag" "$((t1-t0))" <<'PY'
import json, sys
resp, model, tag, elapsed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    d = json.load(open(resp))
    t = d.get("timings", {})
    u = d.get("usage", {})
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", "-")
    print(f"[{tag}] model={model} gen={round(t.get('predicted_per_second', 0), 1)} "
          f"prefill={round(t.get('prompt_per_second', 0), 1)} draft={t.get('draft_n')} "
          f"accepted={t.get('draft_n_accepted')} prompt_n={t.get('prompt_n')} "
          f"cached={cached} elapsed={elapsed}s")
except Exception as exc:
    print(f"[{tag}] ERROR: {str(exc)[:120]}: {open(resp).read()[:160]}")
PY
}

# ---------- A: copy/edit workload ----------
python3 - <<'PY'
import json
def code(seed, version):
    lines = [f"# generated module {seed} — do not reorder", f"VERSION = {version}", "", "def transform(items):"]
    for i in range(160):
        lines.append(f"    item_{i} = [x * {i+1} for x in items if x % {i+2} == 0]")
        lines.append(f"    cache_{i} = {{'id': {i}, 'seed': '{seed}', 'value': sum(item_{i})}}")
    lines += ["    return cache_0", ""]
    return "\n".join(lines)
for n, seed in enumerate(("alpha", "bravo")):
    file = code(seed, 1)
    prompt = ("Below is a source file. Output the ENTIRE file verbatim, changing only "
              "VERSION = 1 to VERSION = 2. No explanation, no fences.\n\n" + file)
    json.dump({"model": "PLACEHOLDER", "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 1200, "temperature": 0.0, "reasoning_effort": "low"},
              open(f"/tmp/tune-copy-{n}.json", "w"))
print("copy payloads ready")
PY

for model in qwen-iq3s-mtp; do
    for n in 0 1; do
        sed "s/PLACEHOLDER/$model/" "/tmp/tune-copy-$n.json" > "/tmp/tune-copy-$model-$n.json"
        ask "$model" "/tmp/tune-copy-$model-$n.json" "copy-$model-$n"
    done
done

# ---------- B: return to a long chat ----------
python3 - <<'PY'
import json
unit = ("The distributed cache stores telemetry from independent worker nodes. "
        "Each worker publishes metrics through a shared message bus. "
        "Engineers review dashboards to detect latency regressions. ")
text = "SECRET CODE: ZEBRA-7781. " + unit * 1300
prompt = "Read this reference text carefully.\n\n" + text + "\n\nQuestion: summarise it in one sentence."
json.dump({"model": "PLACEHOLDER", "messages": [{"role": "user", "content": prompt}],
           "max_tokens": 64, "temperature": 0.0, "reasoning_effort": "low"},
          open("/tmp/tune-long.json", "w"))
json.dump({"model": "PLACEHOLDER", "messages": [{"role": "user", "content": "What is 2+2? Reply with the number only."}],
           "max_tokens": 16, "temperature": 0.0, "reasoning_effort": "low"},
          open("/tmp/tune-foreign.json", "w"))
print("long prompt chars:", len(prompt))
PY

for model in qwen-iq3s-mtp; do
    sed "s/PLACEHOLDER/$model/" /tmp/tune-long.json > "/tmp/tune-long-$model.json"
    sed "s/PLACEHOLDER/$model/" /tmp/tune-foreign.json > "/tmp/tune-foreign-$model.json"
    ask "$model" "/tmp/tune-long-$model.json" "long-$model"
    python3 - "$model" <<'PY'
import json, sys
model = sys.argv[1]
resp = json.load(open(f"/tmp/tune-long-{model}-resp.json"))
answer = resp["choices"][0]["message"].get("content") or ""
prompt = json.load(open(f"/tmp/tune-long-{model}.json"))["messages"][0]["content"]
payload = {"model": model,
           "messages": [{"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                        {"role": "user", "content": "What was the secret code? Reply with only the code."}],
           "max_tokens": 32, "temperature": 0.0, "reasoning_effort": "low"}
json.dump(payload, open(f"/tmp/tune-return-{model}.json", "w"))
PY
    ask "$model" "/tmp/tune-foreign-$model.json" "foreign-$model"
    ask "$model" "/tmp/tune-return-$model.json" "return-$model"
    python3 - "$model" <<'PY'
import json, sys
model = sys.argv[1]
r = json.load(open(f"/tmp/tune-return-{model}-resp.json"))
print(f"[return-{model}] answer:", repr(r["choices"][0]["message"].get("content")))
PY
done

echo "DONE"
