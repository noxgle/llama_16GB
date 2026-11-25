# llama_16GB

Dockerized, OpenAI-compatible [`llama-server`](https://github.com/ggml-org/llama.cpp)
tuned for **16 GB NVIDIA GPUs** — default target: **RTX 5060 Ti 16 GB (Blackwell,
sm_120)** — serving **Qwen3.8-27B** (dense 27B, hybrid Gated DeltaNet / attention)
in 3-bit GGUF quantizations.

The model is **dense**: everything runs in VRAM. There is no MoE expert offload,
no partial layer split, no CPU fallback. If it does not fit, you lower the
context, quantize the KV cache harder, or drop the MTP head.

Forked in structure from [`noxgle/llama_6GB`](https://github.com/noxgle/llama_6GB),
but the operating model is different — see [Not a 6 GB fork](#not-a-6-gb-fork).

---

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Profiles](#profiles)
- [Router mode (experimental)](#router-mode-experimental)
- [Memory budget](#memory-budget)
- [Commands](#commands)
- [Benchmarking](#benchmarking)
- [Troubleshooting](#troubleshooting)
- [Not a 6 GB fork](#not-a-6-gb-fork)
- [License](#license)

---

## Requirements

| Item | Minimum | Notes |
|---|---|---|
| GPU | 16 GB VRAM, sm_120 or newer | RTX 5060 Ti / 5070 / 4060 Ti 16 GB also works (sm_89 needs `GGML_CUDA_ARCH=89`) |
| Driver | 580.x | must support CUDA 12.9 runtime |
| Docker | 26.0+ | `deploy.resources.reservations.devices` must work |
| NVIDIA Container Toolkit | any recent | `nvidia-ctk` |
| Disk | ~25 GB | 12–14 GB of weights + image |
| RAM | 24 GB, 30 GB comfortable | `--load-mode mlock` keeps the model resident |

Models are not bundled. They are downloaded from
[`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
(pinned revision) by `deploy/download-model.sh`.

## Quick start

On a fresh Debian/Ubuntu host (bare metal or a Proxmox LXC with GPU passthrough):

```bash
git clone https://github.com/noxgle/llama_16GB.git /opt/llama16
cd /opt/llama16
bash deploy/install-llama.sh                        # router mode (default)
# or explicitly single-model: bash deploy/install-llama.sh q38
```

The script installs Docker + nvidia-container-toolkit if missing, verifies the
GPU inside a container, pulls the pinned image, downloads the weights, writes
`.env` from the selected profile, and waits for `/health`.

Manual equivalent:

```bash
cd /opt/llama16
bash deploy/download-model.sh --quant iq3s             # ~13.4 GB with MTP
cp configs/qwen3.8-27b-iq3s-mtp.env .env
docker compose pull
docker compose down && docker compose up -d            # down+up, never restart
curl -s localhost:8089/health
```

First token after a cold start takes 60–180 s (weights are read from disk into
RAM/VRAM).

> The dev box (`192.168.200.38`) currently runs **router mode by default** —
> see [Router mode (experimental)](#router-mode-experimental). The commands
> above describe the single-model fallback stack; stop `llama16-router` before
> using compose.

## Profiles

One profile = one complete `.env`. Copy it to `.env` and recreate the stack.
For the default router deployment, use `configs/router.ini` instead.

| File | Quant | MTP | Context | KV | Est. VRAM |
|---|---|---|---|---|---|
| `configs/qwen3.8-27b-iq3s-mtp.env` **(default)** | UD-IQ3_S (11.2 GiB) | `n_max=2` | 48K | q8_0 | ~15.1 GiB |
| `configs/qwen3.8-27b-q3kxl.env` | UD-Q3_K_XL (12.2 GiB) | off | 32K | q8_0 | ~14.4 GiB |
| `configs/qwen3.8-27b-iq3s-longctx.env` | UD-IQ3_S (11.2 GiB) | off | 128K | q4_0 | ~14.6 GiB |

`UD-Q3_K_XL` is the quality option: it is ~1 GiB larger and leaves no room for
the MTP draft head (another 1.28 GiB). `iq3s-longctx` trades KV precision and
MTP for 128K of context.

Fallbacks when a profile OOMs: 48K → 40K (default), 32K → 24K (q3kxl), or switch
`CACHE_TYPE_K/V` to `q4_0`. Do **not** add MoE flags — the model has no experts.

## Router mode (experimental)

llama.cpp `b12716` can run as a **router**: one `llama-server` started *without*
`-m`, using `--models-preset` to expose several named configurations. Each
request is dispatched by its `model` field; the router starts and stops child
`llama-server` processes on demand. **Docker is not restarted** — only the
child processes come and go.

`configs/router.ini` defines three presets matching the standalone profiles:
`iq3s-mtp` (48K + MTP), `q3kxl` (32K), and `longctx` (128K, q4_0 KV).

Start it (stop the compose stack first — both use port 8089):

```bash
cd /opt/llama16
docker compose down
docker run -d --name llama16-router --restart unless-stopped \
    --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e GGML_CUDA_GRAPH_OPT=1 \
    -p 8089:8089 \
    -v /opt/llama16/models:/models:ro \
    -v /opt/llama16/configs/router.ini:/router.ini:ro \
    ghcr.io/noxgle/llama-server-16gb:stable-b12716-v1 \
    --models-preset /router.ini --models-max 1 --models-memory-margin 512 \
    --host 0.0.0.0 --port 8089 --alias router --jinja --threads-http 2 --fit off
```

Use it:

```bash
curl -s localhost:8089/models | jq '.data[] | {id, status: .status.value}'

curl -s localhost:8089/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"iq3s-mtp","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'

curl -s -X POST localhost:8089/models/unload -H 'Content-Type: application/json' \
  -d '{"model":"iq3s-mtp"}'
# Wait until GET /models reports status "unloaded" before /models/load,
# otherwise you get 500 "model limit reached".
```

Back to the normal single-model stack:

```bash
docker rm -f llama16-router
cp configs/qwen3.8-27b-iq3s-mtp.env .env
docker compose up -d
```

Measured on the 5060 Ti (2026-11-25), same Docker container throughout:

| Preset | VRAM | gen tok/s | Switch time |
|---|---|---|---|
| `iq3s-mtp` | 14400 MiB | 63.2 (draft 19/20) | ~5 s |
| `q3kxl` | 13340 MiB | 29.0 | ~41 s cold |
| `longctx` | 13984 MiB | 30.4 | ~5 s |

Caveats: only one model fits in 16 GB at a time (`--models-max 1`); a cold
switch re-reads 12–13 GB of weights; the router itself never touches the GPU.
This mode is opt-in and is **not** used by `docker-compose.yml`, `.env` or
`llama.sh`.

## Memory budget

VRAM = weights + optional MTP head + KV cache + ~0.15 GiB recurrent state +
~0.9 GiB CUDA context/compute buffers. Usable VRAM on a 16311 MiB card is
~15.9 GiB, not 16.

KV is allocated **only for the 16 full-attention layers** — the 48 Gated DeltaNet
layers keep a fixed-size recurrent state (~150 MiB total, independent of context):

| KV precision | Bytes/token | 32K | 64K | 128K |
|---|---|---|---|---|
| `q8_0` | ~34 KiB | 1.06 GiB | 2.13 GiB | 4.25 GiB |
| `q4_0` | ~18 KiB | 0.56 GiB | 1.13 GiB | 2.25 GiB |

That is why 48K with q8_0 KV fits next to an 11 GiB model, and why 128K is
reachable at all on a 16 GB card.

## Commands

```bash
# workstation → server (defaults: root@192.168.200.38:/opt/llama16)
./sync.sh push                 # rsync repo (excludes .env, models/)
./sync.sh deploy               # push + docker compose down && up -d
./sync.sh rebuild              # push + local build on the server (60-120 min)
./sync.sh health               # HTTP status + VRAM/RAM
./sync.sh status | logs | config | stop | start | restart | ssh

# dev-only docker run wrapper (does not touch .env)
./llama.sh start q38           # IQ3_S + MTP, 48K
./llama.sh start q38-q3kxl     # Q3_K_XL, 32K
./llama.sh start q38-long      # IQ3_S, 128K
./llama.sh status | stop | logs q38 | health

# API
curl -s localhost:8089/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}],"max_tokens":64}' \
  | jq '{gen: .timings.predicted_per_second, prefill: .timings.prompt_per_second, draft: .timings.draft_n, accepted: .timings.draft_n_accepted}'
```

Systemd units (installed with `install-llama.sh --with-systemd`):

| Unit | Purpose |
|---|---|
| `llama16-router.service` | starts `llama16-router` (default mode) via `deploy/start-router.sh` |
| `llama16-compose.service` | single-model fallback stack (`Conflicts=` with router) |
| `llama16-gpu-watchdog.timer` | every 2 min: detect CPU fallback / dead container and self-heal |

`install-llama.sh --with-systemd` installs the unit matching the selected mode.

## Benchmarking

Scripts run **on the server** (they recreate the compose stack). They refuse to
run when the model is not on the GPU.

```bash
./benchmark/bench-vram.sh                     # ctx sweep: VRAM peak, prefill, gen
./benchmark/bench-mtp-sweep.sh                # SPEC_DRAFT_N_MAX 0..4, acceptance rate
./benchmark/bench-batch.sh                    # BATCH/UBATCH prefill sweep
./benchmark/bench-knowledge.sh                # small deterministic quality suite
```

Raw CSVs/Markdown land in `benchmark/` (gitignored except for the scripts).
Measured numbers go into the table below — only real measurements, no estimates.

Hardware for the table: RTX 5060 Ti 16 GB (driver 580.95.05, CUDA 12.9.2,
sm_120a, llama.cpp `b12716`, image `ghcr.io/noxgle/llama-server-16gb:stable-b12716-v1`).

| Config | Gen tok/s | Prefill tok/s | VRAM peak | Image |
|---|---|---|---|---|
| `qwen3.8-27b-iq3s-mtp.env`, ctx=16384 | 59.1 | 921.5 | 13040 MiB | `stable-b12716-v1` |
| `qwen3.8-27b-iq3s-mtp.env`, ctx=32768 | 56.5 | 907.3 | 13728 MiB | `stable-b12716-v1` |
| `qwen3.8-27b-iq3s-mtp.env`, ctx=49152 (default) | 51.1 | 878.2 | 14424 MiB | `stable-b12716-v1` |
| `qwen3.8-27b-iq3s-mtp.env`, ctx=65536 (near-full prompt) | 48.3 | 845.2 | 15110 MiB | `stable-b12716-v1` |

Raw CSVs: `benchmark/vram-20261125-113852.csv` (script output printed to the
console log as well). Measured near-full-context 64K prefill via `probe-ctx.sh 65536 0.90`:
prefill 819.7 tok/s, gen 38.1 tok/s, VRAM 15112 MiB.

MTP sweep (CTX=32768, `benchmark/bench-mtp-sweep.sh 1 2 3 4`,
CSV `benchmark/mtp-sweep-20261125-120054.csv`):

| n_max | avg gen tok/s | avg accept rate |
|---|---|---|
| 1 | 48.6 | 0.82 |
| 2 | 55.2 | 0.71 |
| 3 | 62.0 | 0.63 |
| 4 | 65.7 | 0.58 |

Default `n_max=2` sits in the sweet spot; `n_max=3` buys a bit more gen speed.

Expected order of magnitude for 448 GB/s memory bandwidth: 25–30 tok/s without
MTP, 35–45 tok/s with MTP. Treat that as a hypothesis until measured.

## Troubleshooting

**CUDA / sm_120** — startup log must show `CUDA0 ... sm_120` and `nvidia-smi`
must show >11 GiB used by the container. If the model runs on CPU, the image was
built without sm_120 kernels: rebuild with `GGML_CUDA_ARCH=120a` on CUDA ≥ 12.9.

**`ggml_cuda_init: failed` / `no usable GPU found`** — the container did not get
the GPU. On Docker 26.x use the compose `deploy.resources.reservations.devices`
block; `--gpus all` can degrade to CPU-serialized CUDA JIT.

**OOM at model load** — check the profile's estimated VRAM, lower `CTX`, switch
KV to `q4_0`, or disable MTP. `--fit off` is intentional: it must fail loudly
instead of silently shrinking the context.

**`common_download_get_hf_plan: no GGUF files found`** — you used `-hf`. The MTP
head lives in `MTP/` and the HF planner does not recurse; use
`deploy/download-model.sh` and `-m` / `-md` with local paths.

**Empty responses** — Qwen3.8 defaults to `reasoning_effort=xhigh`; with a low
`max_tokens` the budget is consumed by thinking. Raise `N_PREDICT`/`max_tokens`
or set `REASONING_EFFORT=medium` in `.env`.

**`.env` changes have no effect** — `docker compose restart` does not re-read
`.env`. Use `down && up -d`.

## Not a 6 GB fork

| | llama_6GB | llama_16GB |
|---|---|---|
| Model | MoE (35B A3B) | dense 27B |
| Strategy | experts on CPU, attention on GPU (`-ot exps=CPU`) | everything in VRAM |
| KV cache | full attention, all layers | 16 attention layers only (DeltaNet is recurrent) |
| CUDA | 12.4, sm_89/PTX | 12.9, sm_120 |
| Model source | `-hf` | local GGUFs (`-m`), because MTP lives in a subdirectory |
| Context | 8–32K | 48K default, 128K profile |

Carrying over 6 GB flags (`CPUMOE`, `-ot exps=CPU`, `--n-cpu-moe`) silently
produces a broken configuration on a dense model.

## License

MIT — see [LICENSE](LICENSE).
