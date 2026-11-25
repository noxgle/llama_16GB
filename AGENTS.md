# Agent Instructions — llama_16GB

Operational reference for agents working on this repo. Read this before changing
configs, compose, or anything on the servers.

## What this project is

Docker-based, OpenAI-compatible `llama-server` (llama.cpp + CUDA) tuned for
**16 GB NVIDIA GPUs**, serving **`unsloth/Qwen3.8-27B-GGUF`** (dense 27B, hybrid
DeltaNet/attention) in 2 quantizations: `UD-IQ3_S` (default) and `UD-Q3_K_XL`.

Forked in structure from `noxgle/llama_6GB`, but the operating model is
**different**: 6 GB boxes run MoE with experts on CPU; a 16 GB box runs a dense
model **fully in VRAM**. Do not carry over 6 GB assumptions.

Repo status (2026-11-25): scaffold complete and pushed to
`github.com/noxgle/llama_16GB`. CI builds `ghcr.io/noxgle/llama-server-16gb`
from llama.cpp `b12716` (CUDA 12.9.2, `GGML_CUDA_ARCH=120a`). The dev box has
the IQ3_S weights + MTP head in `/opt/llama16/models`; no deployment has been
validated on hardware yet, so every VRAM number in this file is an estimate.

Release flow: `master` builds `:latest` + `:sha-<commit>` from the pinned ref;
annotated-style tags named `stable-b<NNNNN>-v1` build the release tag used in
`configs/` and `install-llama.sh`. Bump the pin in the workflow, Dockerfile and
profiles together.

## Hardware and hosts

| Role | Host | Access | GPU | Notes |
|---|---|---|---|---|
| **Dev** | `root@192.168.200.38` | SSH key, passwordless (`BatchMode=yes` OK) | **RTX 5060 Ti 16 GB** (16311 MiB), driver 580.95.05 | LXC on Proxmox, hostname `dev-llama-gpu` |
| Prod | TBD | TBD | 16 GB | not provisioned |

Dev box facts (verified 2026-11-25): 4 vCPU, 30 GB RAM, 177 GB disk (**72 GB
free**), Docker 26.1.5, `/opt` on `pve-vm-1004-disk-0`.

```bash
ssh -o BatchMode=yes root@192.168.200.38 'hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
# dev-llama-gpu
# NVIDIA GeForce RTX 5060 Ti, 16311 MiB
```

### Legacy stack — removed 2026-11-25

The old **llama_6GB** deployment on the dev box (`/opt/llama`,
`llama-compose.service`, `llama-postboot-restart.service`,
`llama-gpu-watchdog.timer`, container `llama-llama-server-1` on `:b10665`) was
removed on 2026-11-25. Configs, `.env*`, benchmark results and `.git` are
archived on the host:

```
root@192.168.200.38:/root/llama_6gb_removal/opt-llama-state-20261125-0939.tar.gz
# 111 MB, models/ excluded
```

Do not recreate `/opt/llama`. Leftovers from that era, safe to purge (not used
by this project): `llama_hf-cache` docker volume (37.7 GB: Qwen3.6-35B,
Gemma4 26B/E2B) and the `ghcr.io/noxgle/llama-server:*` images (built without
sm_120 kernels).

Install layout for this repo:

| Element | Value |
|---|---|
| Install dir | `/opt/llama16` |
| Compose project | `llama16` (`COMPOSE_PROJECT_NAME`) |
| Port | `8089` (free since the legacy stack was removed) |
| Systemd | `llama16-compose.service`, `llama16-gpu-watchdog.{service,timer}` |
| Models | `/opt/llama16/models` (bind-mounted, not an HF cache volume) |

## Model facts (measured, not assumed)

Source: `unsloth/Qwen3.8-27B-GGUF` @ commit `4af51b3542f775b0a8e4b310bfb3f17e01c306c9`.

| Property | Value |
|---|---|
| Arch (GGUF) | `qwen35` |
| Type | **dense 27B** — no MoE, no experts |
| Layers | 64 = 16 × (3 × Gated DeltaNet + 1 × Gated Attention) |
| Attention | 24 Q heads, **4 KV heads**, head_dim 256 |
| Native context | 262 144 |
| Vision | yes (`mmproj-F16.gguf`, `clip.projector_type = qwen3vl_merger`) — **out of scope for v1** |
| MTP | yes, separate file `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`, `nextn_predict_layers = 1` |

Exact files and sizes:

| File | Size |
|---|---|
| `Qwen3.8-27B-UD-IQ3_S.gguf` | 12.04 GB (11.21 GiB) |
| `Qwen3.8-27B-UD-Q3_K_XL.gguf` | 13.15 GB (12.25 GiB) |
| `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` | 1.37 GB (1.28 GiB) |
| `mmproj-F16.gguf` | 0.93 GB (0.87 GiB) |

### KV cache math

KV is allocated **only for the 16 full-attention layers** (DeltaNet layers are
recurrent, fixed state ≈ 150 MiB total regardless of context). This is why long
context is cheap on this model — 6 GB/35B assumptions do not apply.

| KV precision | Bytes/token | 32K | 64K | 128K |
|---|---|---|---|---|
| `q8_0` | ~34 KiB | 1.06 GiB | 2.13 GiB | 4.25 GiB |
| `q4_0` | ~18 KiB | 0.56 GiB | 1.13 GiB | 2.25 GiB |

VRAM budget = weights + optional MTP + KV + ~0.15 GiB recurrent state +
~0.9 GiB CUDA context/compute buffers. Usable VRAM is ~15.9 GiB, not 16.

## Supported profiles

Common: `NGLAYERS=999`, `--load-mode mlock`, `--fit off`, `PARALLEL=1`,
`CTX_CHECKPOINTS=4`, `CACHE_RAM=4096`, `--alias qwen3.8-27b`.

| Profile | Quant | MTP | CTX | KV | Est. VRAM |
|---|---|---|---|---|---|
| `qwen3.8-27b-iq3s-mtp.env` **(default)** | UD-IQ3_S | `n_max=2` | 48K | q8_0 | ~15.1 GiB |
| `qwen3.8-27b-q3kxl.env` | UD-Q3_K_XL | off | 32K | q8_0 | ~14.4 GiB |
| `qwen3.8-27b-iq3s-longctx.env` | UD-IQ3_S | off | 128K | q4_0 | ~14.6 GiB |

Estimates are to be validated on hardware. OOM fallbacks: default 48K → 40K,
q3kxl 32K → 24K. Never "fix" OOM by offloading experts — there are none.

## Repository layout (target)

```
.github/workflows/build.yml     CI → ghcr.io/noxgle/llama-server-16gb
benchmark/                      bench-vram, bench-mtp-sweep, bench-batch, bench-knowledge
configs/                        model profiles (*.env, sourced as bash)
deploy/install-llama.sh         single-command provisioning
deploy/download-model.sh        HF download + GGUF verification
deploy/systemd/                 llama16-compose, llama16-gpu-watchdog
scripts/gpu-watchdog.sh
Dockerfile                      llama-server build (CUDA 12.9, sm_120)
docker-compose.yml              production deployment
llama.sh                        docker run wrapper
sync.sh                         local ↔ /opt/llama16 sync
AGENTS.md                       this file
```

## Commands

```bash
# --- local (repo root) ---
./sync.sh push                 # local → root@192.168.200.38:/opt/llama16
./sync.sh deploy               # push + docker compose down && up -d
./sync.sh health               # HTTP 200 + VRAM + RAM
./sync.sh status               # container + GPU processes
./sync.sh logs                 # tail remote logs
./sync.sh config               # show remote .env

# --- remote ---
ssh root@192.168.200.38
cd /opt/llama16
cp configs/<profile>.env .env && docker compose down && docker compose up -d

# model control (docker run path, dev only)
./llama.sh start q38           # default: IQ3_S + MTP
./llama.sh start q38-q3kxl     # quality, no MTP
./llama.sh start q38-long      # 128K context
./llama.sh status | stop | logs q38

# --- health / throughput ---
curl -s http://localhost:8089/health
curl -s http://localhost:8089/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Write ~500 chars."}],"max_tokens":500}' \
  | jq '{gen: .timings.predicted_per_second, prefill: .timings.prompt_per_second}'

# --- VRAM peak during a run ---
nvidia-smi --query-gpu=memory.used --format=csv,noheader -lms 500
```

## Deployment gotchas (read before touching servers)

### CUDA 12.8+ and `GGML_CUDA_ARCH=120a` are mandatory

RTX 5060 Ti is Blackwell **sm_120**; llama.cpp needs CUDA ≥ 12.8 and an
`sm_120a` build (CUDA 12.9). Images built on CUDA 12.4 (the 6GB default) either fail to run or fall
back to CPU — the legacy `.38` container restarting on `:b10665` is the proof.
Every build must pass `-DCMAKE_CUDA_ARCHITECTURES=120a` (CI build-arg
`GGML_CUDA_ARCH`; plain `120` also runs, but `120a` enables Blackwell
architecture-specific instructions). Verify after deploy: startup log shows `CUDA0 ... sm_120`
and `nvidia-smi` shows > 11 GiB for the llama process.

### Never use `--gpus all` on Docker 26.1.5

The dev box runs exactly 26.1.5. `--gpus all` triggers CPU-serialized CUDA JIT
after boot (≈1.5 tok/s). Use `deploy.resources.reservations.devices` in compose,
or `--runtime=nvidia` + `NVIDIA_VISIBLE_DEVICES=all` in `llama.sh`.

### No MoE offload flags

The model is dense. `-ot exps=CPU` / `--n-cpu-moe` / `CPUMOE` do not apply and
silently produce nonsense. If VRAM is short: lower `CTX`, switch KV to `q4_0`,
or drop MTP. Do not partially offload layers unless a profile explicitly does it
with measured numbers.

### MTP is a separate GGUF and `-hf` cannot fetch it

`-hf` with files in subdirectories (`MTP/`) fails with
`common_download_get_hf_plan: no GGUF files found` (same upstream quirk as in
llama_6GB). Always download to `/opt/llama16/models` and pass:

```
MODEL_FLAG=-m  MODEL=/models/Qwen3.8-27B-UD-IQ3_S.gguf
DRAFT_FLAG=-md DRAFT_MODEL=/models/mtp-Qwen3.8-27B-Q4_0.gguf
SPEC_TYPE=draft-mtp
GPU_LAYERS_DRAFT=99
```

`-md` is only accepted with `--spec-type draft-mtp` (llama.cpp rejects other
spec types with a draft model).

### `.env` changes require `down` + `up`, not `restart`

`docker compose restart` does not re-read `.env`. Always
`docker compose down && docker compose up -d`.

### `.env` is never committed or synced

`.env` is gitignored and excluded by `sync.sh push`. Changing
`configs/*.env` locally has no effect until copied to `.env` on the host.

### Context checkpoints cost RAM, and MTP used to multiply it

Hybrid recurrent models keep checkpoints at a fixed ≈150 MiB. Older builds grew
them by 4 KiB/token with MTP (upstream #30502, fixed by #30514). Keep
`CTX_CHECKPOINTS=4` (max 8) and inspect `created context checkpoint` lines
after changing MTP settings.

### `--cache-reuse` is probably a no-op here

Expect `not supported by this context` / full prompt reprocessing on hybrid
recurrent models. Harmless, but do not tune it. `--cache-ram` is still useful
and counts against the 30 GB host RAM budget, not VRAM.

### RAM budget on the dev box is 30 GB total

`--load-mode mlock` pins the 12–13 GB model in RAM; add `CACHE_RAM=4096`,
checkpoints and page cache. Do not raise `CACHE_RAM` or `CTX_CHECKPOINTS`
without checking `free -g` on the host afterwards.

### Qwen3.8 thinking control

Default `reasoning_effort=xhigh`. Set it per deployment via a single JSON:

```
--chat-template-kwargs '{"preserve_thinking": false, "reasoning_effort": "medium"}'
```

Values: `xhigh` (default), `medium`, `low`. Empty Qwen responses are usually a
reasoning-budget issue — raise `max_tokens` or lower `reasoning_effort`, do not
disable thinking globally for coding profiles.

### Do not deploy `:latest`

`:latest` tracks untested upstream master. Pin `stable-b<tag>-v1` images; keep
the previous tag on disk for rollback.

## Configuration conventions

- `configs/*.env` files are **sourced as bash** by `llama.sh`. Keep `KEY=value`,
  no spaces around `=`, double-quote any value with spaces.
- One profile = one complete `.env`. Do not introduce shared defaults that only
  exist in compose.
- Model paths in profiles are **container paths** (`/models/...`), never host
  paths.
- Compose passes optional values through empty-filtered args; new flags must be
  added as `${VAR:-}` and filtered, otherwise llama.cpp rejects empty argv
  elements.
- Keep `--fit off`. `--fit on` silently rewrites context/offload and makes
  benchmark numbers non-reproducible (use it only for first-run sizing, then
  hardcode results).

## Benchmarking and tuning protocol

1. Never benchmark on a box where `nvidia-smi` shows another llama process.
2. Abort if VRAM < 8 GiB or the log contains `ggml_cuda_init: failed` (CPU fallback).
3. Measure in this order: ctx sweep (16K→128K) → MTP sweep (`n_max` 0–4) →
   batch sweep (`BATCH`/`UBATCH`) → prefill → soak (≥30 min, watch VRAM/RSS).
4. Record: gen tok/s, prefill tok/s, TTFT, VRAM peak, RAM peak, draft acceptance
   (`n_draft`/`n_accept` from `/metrics` or timings), and the exact image tag.
5. Store raw runs in `benchmark/` and update the table in `README.md` only with
   measured numbers. Mark estimates as such.

Expected order of magnitude on 448 GB/s: ~25–30 tok/s without MTP, ~35–45 with
MTP. Treat as a hypothesis, not a baseline.

### Measured context range on RTX 5060 Ti 16 GB (measured, not estimated)

Default profile (IQ3_S + MTP n_max=2, q8_0 KV) — see `benchmark/vram-20261125-112709.csv`
(sampled with 250 ms interval):

| ctx | gen tok/s | prefill tok/s | TTFT | VRAM used after |
|---|---|---|---|---|
| 16384 | 59.0 | 921.1 | 8.3 s | 13040 MiB |
| 32768 | 56.5 | 906.3 | 16.9 s | 13728 MiB |
| 49152 | 51.1 | 877.0 | 26.1 s | 14424 MiB |
| 65536 | 48.3 | 843.2 | 36.2 s | 15110 MiB |

- 64K near-full prefill (≈39K tokens, a custom curl probe at 90% of the context): prefill 819.7
  tok/s, gen 38.1 tok/s, VRAM 15112 MiB — fits, but leaves ~1.9 GiB headroom.
- 128K + MTP OOMs during prefill (llama-server aborts with CUDA OOM at
  `ggml_cuda_mul_mat_q`). Use `configs/qwen3.8-27b-iq3s-longctx.env` (MTP off,
  q4_0 KV) and validate there.
- MTP is worth it here: gen ≈ 51 tok/s at 48K vs ≈ 25–30 without (hypothesis).
- 128K longctx profile (IQ3_S, SPEC_TYPE=none, q4_0 KV, CTX=131072) probe
  (`probe-ctx.sh 131072 0.40`): prefill 881.1 tok/s, gen 19.4 tok/s,
  VRAM after 13992 MiB — fits on the 16 GB card.

## Upstream issues to watch

| Issue | Impact |
|---|---|
| #30502 / #30514 | MTP checkpoint RAM blowup — verify fix present in pinned tag |
| #31302 | MTP + mmproj graph reallocation after an image (vision, v2) |
| #32475 / #32560 | MTP on multi-GPU / HIP — not applicable to single CUDA |
| `qwen35` CPU backend loops | never fall back to CPU; abort benchmarks instead |

## Constraints for agents

- `deploy/install-llama.sh` is the shared provisioning script. Do not edit it
  for local experiments; config changes belong in `configs/*.env`.
- Never commit `models/`, `.env`, `*.gguf`, or benchmark binaries.
- Never recreate `/opt/llama` on the dev box; the legacy 6 GB stack is removed
  and archived (see Hardware and hosts).
- Do not raise `NGLAYERS` above the model's layer count or add MoE flags.
- Do not use `--gpus all`, `docker compose restart` after `.env` edits, or
  `:latest` images.
- Report changes with: profile touched, measured VRAM/RAM, gen/prefill tok/s,
  and rollback image tag.

## TODO

- [x] Validate 48K / 32K / 128K contexts on the 5060 Ti; record real VRAM peaks
      (measured 2026-11-25; see "Measured context range" above).
- [x] Validate 128K on the 5060 Ti with the longctx profile; confirm it fits
      (measured 2026-11-25: fits, VRAM after ≈13992 MiB, prefill ≈881 tok/s).
      Note: without MTP gen ≈ 19 tok/s; quality comparison open (TODO above).
- [ ] Confirm whether `UD-Q3_K_XL` fits 32K and whether it beats `IQ3_S` on the
      knowledge suite (quality vs context trade-off).
- [x] Sweep `SPEC_DRAFT_N_MAX` 0–4; pick per-profile default (run
      `benchmark/bench-mtp-sweep.sh`; CSVs in `benchmark/mtp-sweep-*.csv`).
- [ ] Sweep `BATCH`/`UBATCH` (512/256 … 2048/512) for prefill
      (`benchmark/bench-batch.sh`).
- [ ] Verify `--cache-reuse` behaviour in logs on `qwen35`.
- [ ] Vision profile (mmproj) — deferred to v2, needs #31302 resolved.
- [ ] Decide prod host(s) and add them to the scope table.
