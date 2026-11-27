# llama_16GB

Dockerized, OpenAI-compatible [`llama-server`](https://github.com/ggml-org/llama.cpp)
tuned for **16 GB NVIDIA GPUs** — default target: **RTX 5060 Ti 16 GB (Blackwell,
sm_120)** — serving two models:
- **Qwen3.8-27B** — dense 27B, hybrid Gated DeltaNet / attention, 3-bit GGUFs;
- **Inkling-2-Mini** — 26B MoE with looped layers, `UD-Q4_K_M`.

The Qwen profiles are dense: everything runs in VRAM, no expert offload.
Inkling is MoE: non-looped experts live in host RAM while looped experts and
all non-expert weights stay on the GPU. See [Profiles](#profiles).

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
| Disk | ~45 GB | both model sets (~25 GB) + image (10.5 GB) |
| RAM | 24 GB, 30 GB comfortable | `--load-mode mlock` keeps the model resident |

Models are not bundled. `deploy/download-model.sh` downloads pinned revisions
of [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
and [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)
(the latter via `--model inkling2`, using the `Mainline_Support` layout).

## Quick start

On a fresh Debian/Ubuntu host (bare metal or a Proxmox LXC with GPU passthrough):

```bash
git clone https://github.com/noxgle/llama_16GB.git /opt/llama16
cd /opt/llama16
bash deploy/install-llama.sh                        # router mode (default)
# or explicitly single-model: bash deploy/install-llama.sh q38
```

The script installs Docker + nvidia-container-toolkit if missing, verifies the
GPU inside a container, pulls the pinned image, downloads the weights (all
three GGUFs in router mode), writes `.env` for the fallback profile and waits
for `/health`. With `--with-systemd` it also installs the router unit and the
GPU watchdog timer.

Manual equivalent:

```bash
cd /opt/llama16
bash deploy/download-model.sh --quant iq3s             # ~13.4 GB with MTP
bash deploy/download-model.sh --quant q3kxl --no-mtp   # second preset
bash deploy/download-model.sh --model inkling2         # Inkling Q4_K_M (~16,6 GB)
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

| File | Quant | MTP | Context | KV | Measured VRAM |
|---|---|---|---|---|---|
| `configs/qwen3.8-27b-iq3s-mtp.env` (fallback) | UD-IQ3_S (11.2 GiB) | `n_max=2` | 48K | q8_0 | ~15.1 GiB |
| `configs/qwen3.8-27b-q3kxl.env` | UD-Q3_K_XL (12.2 GiB) | off | 32K | q8_0 | ~14.4 GiB |
| `configs/qwen3.8-27b-iq3s-longctx.env` | UD-IQ3_S (11.2 GiB) | off | 128K | q4_0 | ~14.6 GiB |
| `configs/inkling-2-mini.env` | Inkling UD-Q4_K_M (MoE, 15.5 GiB) | off | 128K | q8_0 | 13442 MiB |

`UD-Q3_K_XL` is the quality option: it is ~1 GiB larger and leaves no room for
the MTP draft head (another 1.28 GiB). `qwen-iq3s-longctx` trades KV precision and
MTP for 128K of context.

All profiles use `CTX_CHECKPOINTS=8` and `REPEAT_PENALTY=1.0`. Qwen profiles use
`CACHE_RAM=8192`; `qwen-iq3s-mtp` uses `draft-mtp,ngram-mod` and the no-MTP profiles
use `ngram-mod`. Inkling uses `CACHE_RAM=4096`, `load-mode none` (mmap),
`REASONING_BUDGET=8192` and the message `" thinking budget exceeded, answer now."`.

Acceptance targets: 16 GB VRAM, 128K context, `q8_0` KV, generation >= 20 tok/s,
prefill >= 200 tok/s; Q4 quant preferred, Q3 acceptable. Current status is in
`AGENTS.md` → "Project targets".

Fallbacks for the **dense Qwen** profiles: 48K → 40K, 32K → 24K, or KV → `q4_0`.
Do **not** add MoE flags to Qwen — it has no experts (Inkling is the exception).

## Router mode (experimental)

llama.cpp `b12811` can run as a **router**: one `llama-server` started *without*
`-m`, using `--models-preset` to expose several named configurations. Each
request is dispatched by its `model` field; the router starts and stops child
`llama-server` processes on demand. **Docker is not restarted** — only the
child processes come and go.

`configs/router.ini` defines six presets: `qwen-iq3s-mtp` (48K + MTP, tuned),
`qwen-q3kxl` (32K), `qwen-iq3s-longctx` (128K, q4_0 KV), `inkling2-mini` (MoE, 128K, q8_0,
selective CPU experts — primary), `inkling2-mini-balanced` and
`inkling2-mini-dspark`.

Start it (stop the compose stack first — both use port 8089):

```bash
cd /opt/llama16
docker compose down
bash deploy/start-router.sh        # idempotent; image + flags pinned in the script
```

Use it:

```bash
curl -s localhost:8089/models | jq '.data[] | {id, status: .status.value}'

curl -s localhost:8089/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen-iq3s-mtp","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'

curl -s 'localhost:8089/metrics?model=qwen-iq3s-mtp' | grep llamacpp   # Prometheus
curl -s 'localhost:8089/slots?model=qwen-iq3s-mtp' | jq '.[0].n_ctx'    # slot info

curl -s -X POST localhost:8089/models/unload -H 'Content-Type: application/json' \
  -d '{"model":"qwen-iq3s-mtp"}'
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
| `qwen-iq3s-mtp` | 14400 MiB | 63.2 (draft 19/20) | ~5 s |
| `qwen-q3kxl` | 13340 MiB | 29.0 | ~41 s cold |
| `qwen-iq3s-longctx` | 13984 MiB | 30.4 | ~5 s |

These gen numbers were measured **before** the `ngram-mod`/cache tuning; they
are still a valid baseline. See [Benchmarking](#benchmarking) for current
throughput (copy/edit workloads reach ~280–310 tok/s).

Caveats: only one model fits in 16 GB at a time (`--models-max 1`); a cold
switch re-reads 12–13 GB of weights; the router itself never touches the GPU.
This is the **default deployment mode** on the dev box and the default of
`install-llama.sh`; `docker-compose.yml`, `.env` and `llama.sh` are the
single-model fallback path. Changing router CLI flags requires recreating the
container (`docker rm -f llama16-router && bash deploy/start-router.sh`) —
`docker restart` keeps the old args.

## Memory budget

VRAM = weights + optional MTP head + KV cache + ~0.15 GiB recurrent state +
~0.9 GiB CUDA context/compute buffers. Usable VRAM on a 16311 MiB card is
~15.9 GiB, not 16.

For Qwen, KV is allocated **only for the 16 full-attention layers** — the 48
Gated DeltaNet layers keep a fixed-size recurrent state (~150 MiB, independent
of context). Inkling uses sliding-window + looped attention instead; its KV is
smaller than the table suggests:

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
./llama.sh start inkling2      # Inkling Q4_K_M, MoE, 128K
./llama.sh status | stop | logs q38 | health

# API — router mode: "model" selects one of six presets:
# qwen-iq3s-mtp | qwen-q3kxl | qwen-iq3s-longctx | inkling2-mini | inkling2-mini-balanced | inkling2-mini-dspark
curl -s localhost:8089/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-iq3s-mtp","messages":[{"role":"user","content":"hi"}],"max_tokens":64}' \
  | jq '{gen: .timings.predicted_per_second, prefill: .timings.prompt_per_second, draft: .timings.draft_n, accepted: .timings.draft_n_accepted}'
# single-model fallback uses "model":"qwen3.8-27b" instead
```

Systemd units (installed with `install-llama.sh --with-systemd`):

| Unit | Purpose |
|---|---|
| `llama16-router.service` | starts `llama16-router` (default mode) via `deploy/start-router.sh` |
| `llama16-compose.service` | single-model fallback stack (`Conflicts=` with router) |
| `llama16-gpu-watchdog.timer` | every 2 min: detect CPU fallback / dead container and self-heal |

`install-llama.sh --with-systemd` installs the unit matching the selected mode.

## Benchmarking

Most scripts run **on the server** and recreate the compose stack; they refuse
to run when the model is not on the GPU. `tune-ab.sh` is the exception: it needs
router mode and the `qwen-iq3s-mtp` preset.

```bash
./benchmark/bench-vram.sh                     # ctx sweep: VRAM peak, prefill, gen
./benchmark/bench-mtp-sweep.sh                # SPEC_DRAFT_N_MAX 0..4, acceptance rate
./benchmark/bench-batch.sh                    # BATCH/UBATCH prefill sweep
./benchmark/bench-knowledge.sh                # small deterministic quality suite
./benchmark/tune-ab.sh                        # router copy/edit + cache test (qwen-iq3s-mtp)
```

Raw CSVs/Markdown land in `benchmark/` (gitignored except for the scripts).
Measured numbers go into the table below — only real measurements, no estimates.

Hardware for the table: RTX 5060 Ti 16 GB (driver 580.95.05, CUDA 12.9.2,
sm_120a, measured on llama.cpp `b12716` / `stable-b12716-v1`; new pin is `b12811` / `stable-b12811-v1`).

| Config | Gen tok/s | Prefill tok/s | VRAM peak | Image |
|---|---|---|---|---|
| `qwen3.8-27b-iq3s-mtp.env`, ctx=16384 | 59.1 | 921.5 | 13040 MiB | `stable-b12716-v1` |
| `qwen3.8-27b-iq3s-mtp.env`, ctx=32768 | 56.5 | 907.3 | 13728 MiB | `stable-b12716-v1` |
| `qwen3.8-27b-iq3s-mtp.env`, ctx=49152 (default) | 51.1 | 878.2 | 14424 MiB | `stable-b12716-v1` |
| `qwen3.8-27b-iq3s-mtp.env`, ctx=65536 (near-full prompt) | 48.3 | 845.2 | 15110 MiB | `stable-b12716-v1` |

The sweep above was run with the pre-tuning settings (MTP only, 4096/4 cache).
Current default `qwen-iq3s-mtp` (ngram-mod) measured 60.7 gen tok/s on a short
prompt and 282.6–308.5 tok/s on copy/edit workloads (see below).

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

`draft-mtp,ngram-mod` vs MTP alone (`benchmark/tune-ab.sh`, same GPU, two
copy/edit prompts, 2026-11-25):

| Preset | gen tok/s (copy 1 / 2) | draft accepted |
|---|---|---|
| `qwen-iq3s-mtp` (tuned) | 282.6 / 308.5 | 1155/1158, 1165/1206 |
| MTP only (old baseline, preset removed) | 63.3 / 63.1 | 798/801, 797/803 |

DSpark on b12811 is available as optional preset `inkling2-mini-dspark`
(drafter: `Mainline_Support/dspark-Inkling-2-Mini-Q8_0.gguf_file` -> `.gguf`):
repeated-text gen 40,8 tok/s (accept 206/276) vs 35,8 without, but no gain on
short prompts and VRAM 15764 vs 14602 MiB — kept optional. No regression on
Qwen (60,1 tok/s). `n_max=2` was slower than 3.

DEFAULT `inkling2-mini` keeps 128K + q8_0 KV and reaches **249–255 tok/s
prefill** with gen 29–32 tok/s (ubatch 256, MoE cache off, VRAM 13442 MiB;
verified with a 39,6K prompt). `inkling2-mini-balanced` gives 138 tok/s prefill
with gen 33 and an active expert cache.

Prefill experiments with q4_0 KV or 48K context reached 138–252 tok/s, but
were removed from `router.ini` after the primary reached 249–255 tok/s while
keeping 128K + q8_0.

UBatch 256 needs the MoE cache off; with 1,5 GiB cache it OOMs at load
(~13 GiB compute buffer). Batch 1024 is not feasible at all: compute buffer
alone needs ~25,7 GiB. A full 128K prompt is allocated but still takes ~7 min
to prefill at ~250 tok/s.

A 39K-token chat followed by a foreign request returned with 39065 cached
tokens in ~2 s on **both** cache settings; the 8192/8 benefit is expected on
65K+ chats (as measured in the field guide linked in `AGENTS.md`).

Measured ranges (hardware above): ~19 tok/s without MTP at 128K, 51–63 tok/s
with MTP on short prompts, and up to ~310 tok/s on copy/edit workloads when
`ngram-mod` finds a long match in the context. Generation on pure reasoning is
unchanged by `ngram-mod`.

## Troubleshooting

**CUDA / sm_120** — startup log must show `CUDA0 ... sm_120` and `nvidia-smi`
must show >11 GiB used by the container. If the model runs on CPU, the image was
built without sm_120 kernels: rebuild with `GGML_CUDA_ARCH=120a` on CUDA ≥ 12.9.

**`ggml_cuda_init: failed` / `no usable GPU found`** — the container did not get
the GPU. On Docker 26.x use the compose `deploy.resources.reservations.devices`
block; `--gpus all` can degrade to CPU-serialized CUDA JIT.

**OOM at model load** — check the profile's estimated VRAM, lower `CTX`, switch
KV to `q4_0`, or disable MTP. For Inkling: `UBATCH=256` works only with
`MOE_CACHE_MIB=0` (with a 1,5 GiB cache the compute buffer OOMs); download
`Mainline_Support/*.gguf_file` renamed to `.gguf` — the root GGUF is rejected by
mainline llama.cpp. `--fit off` is intentional: it fails loudly instead of
silently shrinking the context.

**`common_download_get_hf_plan: no GGUF files found`** — you used `-hf`. The MTP
head lives in `MTP/` and the HF planner does not recurse; use
`deploy/download-model.sh` and `-m` / `-md` with local paths.

**Empty responses** — Qwen3.8 defaults to `reasoning_effort=xhigh`; with a low
`max_tokens` the budget is consumed by thinking. Raise `N_PREDICT`/`max_tokens`,
set `REASONING_EFFORT=medium` in `.env` (single model) or `reasoning_effort` in
`configs/router.ini` (router). Valid Qwen values are `xhigh`, `medium`, `low` —
`high` returns HTTP 500. Inkling uses a different scale (`none|minimal|low|medium|
high|xhigh|max` or 0–0.99) and a hard `REASONING_BUDGET=8192`; allow more than
8192 `max_tokens` for hard tasks, otherwise the answer is cut off.

**`.env` changes have no effect** — `docker compose restart` does not re-read
`.env`. Use `down && up -d`.

## Not a 6 GB fork

| | llama_6GB | llama_16GB |
|---|---|---|
| Model | MoE (35B A3B) | dense Qwen 27B / MoE Inkling 26B |
| Strategy | experts on CPU, attention on GPU (`-ot exps=CPU`) | Qwen: all in VRAM; Inkling: selective `-ot` experts in RAM |
| KV cache | full attention, all layers | 16 attention layers only (DeltaNet is recurrent) |
| CUDA | 12.4, sm_89/PTX | 12.9, sm_120 |
| Model source | `-hf` | local GGUFs (`-m`), because MTP lives in a subdirectory |
| Context | 8–32K | 48K Qwen default, 128K Inkling default |

Carrying over 6 GB flags (`CPUMOE`, `-ot exps=CPU`, `--n-cpu-moe`) silently
produces a broken configuration on a dense model.

## License

MIT — see [LICENSE](LICENSE).
