#!/usr/bin/env bash
# install-llama.sh — provision a Debian/Ubuntu machine (LXC or bare metal)
# with Docker, NVIDIA GPU support and the llama_16GB stack.
#
# Usage:
#   bash deploy/install-llama.sh                    # router mode (default)
#   bash deploy/install-llama.sh router             # same, explicit
#   bash deploy/install-llama.sh q38                # single-model fallback
#   bash deploy/install-llama.sh q38-q3kxl          # UD-Q3_K_XL, no MTP
#   bash deploy/install-llama.sh q38-long           # UD-IQ3_S, 128K context
#   bash deploy/install-llama.sh router --build-local  # compile on this host
#
# Environment:
#   LLAMA_REPO   git repo (default: https://github.com/noxgle/llama_16GB.git)
#   LLAMA_DIR    install dir (default: /opt/llama16)
#   LLAMA_IMAGE  image (default: ghcr.io/noxgle/llama-server-16gb:stable-b12811-v1)
#
# Prerequisites: root, Debian 12+/Ubuntu 22.04+, GPU visible on the host.
# For Proxmox LXC GPU passthrough see the error message in step 3.

set -euo pipefail

MODEL="${1:-router}"
BUILD_LOCAL=false
SKIP_DOWNLOAD=false
WITH_SYSTEMD=false

shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-local)   BUILD_LOCAL=true ;;
        --skip-download) SKIP_DOWNLOAD=true ;;
        --with-systemd)  WITH_SYSTEMD=true ;;
        -h|--help)       MODEL="--help" ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

REPO_URL="${LLAMA_REPO:-https://github.com/noxgle/llama_16GB.git}"
INSTALL_DIR="${LLAMA_DIR:-/opt/llama16}"
LLAMA_IMAGE="${LLAMA_IMAGE:-ghcr.io/noxgle/llama-server-16gb:stable-b12811-v1}"
LLAMA_REF="${LLAMA_REF:-b12811}"
GGML_CUDA_ARCH="${GGML_CUDA_ARCH:-120a}"

case "$MODEL" in
    router)    MODE=router;  CONFIG_NAME="qwen3.8-27b-iq3s-mtp.env"; QUANT="iq3s";  MTP=1 ;;
    q38)       MODE=single;  CONFIG_NAME="qwen3.8-27b-iq3s-mtp.env"; QUANT="iq3s";  MTP=1 ;;
    q38-q3kxl) MODE=single;  CONFIG_NAME="qwen3.8-27b-q3kxl.env";    QUANT="q3kxl"; MTP=0 ;;
    q38-long)  MODE=single;  CONFIG_NAME="qwen3.8-27b-iq3s-longctx.env"; QUANT="iq3s"; MTP=0 ;;
    inkling2)  MODE=single;  CONFIG_NAME="inkling-2-mini.env";             QUANT="q4km"; MTP=0 ;;
    *)
        cat >&2 <<USAGE
Usage: $(basename "$0") [router|q38|q38-q3kxl|q38-long] [--build-local] [--skip-download] [--with-systemd]

  router     llama-server router mode with configs/router.ini  (default)
             presets: iq3s-mtp | q3kxl | longctx, routed by the "model" field
  q38        single-model fallback: UD-IQ3_S + MTP, 48K context
  q38-q3kxl  single-model: UD-Q3_K_XL, 32K context (quality, no MTP)
  q38-long   single-model: UD-IQ3_S, 128K context (no MTP, q4_0 KV)
  inkling2   single-model: Inkling-2-Mini UD-Q4_K_M, MoE + CPU experts, 128K

Options:
  --build-local    build the image on this host (GGML_NATIVE=ON, 60-120 min)
  --skip-download  do not download GGUF weights (use existing ./models)
  --with-systemd   install the matching runtime unit + GPU watchdog timer
USAGE
        exit 1 ;;
esac
ROUTER=false
[[ "$MODE" == router ]] && ROUTER=true

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "  ${GREEN}*${NC} $*"; }
warn()  { echo -e "  ${YELLOW}WARN${NC} $*" >&2; }
die()   { echo -e "  ${RED}ERROR${NC} $*" >&2; exit 1; }
heading() { echo ""; echo -e "${BOLD}── $* ──${NC}"; }

heading "Pre-flight"
[[ "$EUID" -eq 0 ]] || die "must run as root (or via sudo)"
command -v apt-get >/dev/null || die "apt-get required (Debian/Ubuntu)"
info "host:   $(hostname)"
info "mode:   $MODE ($( $ROUTER && echo "presets: iq3s-mtp | q3kxl | longctx" || echo "$QUANT, mtp=$MTP"))"
info "repo:   $REPO_URL -> $INSTALL_DIR"
info "image:  $LLAMA_IMAGE"
info "build:  $($BUILD_LOCAL && echo "local (sm_$GGML_CUDA_ARCH, native)" || echo 'pull from GHCR')"

heading "Step 1/10 — Docker Engine"
if command -v docker >/dev/null 2>&1; then
    info "Docker already installed ($(docker --version))"
else
    apt-get update -qq
    apt-get install -y -qq ca-certificates curl git gnupg
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
    . /etc/os-release
    case "$ID/$VERSION_CODENAME" in
        debian/trixie|debian/forky) DOCKER_CODENAME="bookworm" ;;
        ubuntu/plucky)              DOCKER_CODENAME="noble" ;;
        *)                          DOCKER_CODENAME="$VERSION_CODENAME" ;;
    esac
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/${ID} ${DOCKER_CODENAME} stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    systemctl enable --now docker
    info "Docker installed"
fi

heading "Step 2/10 — NVIDIA Container Toolkit"
if command -v nvidia-ctk >/dev/null 2>&1; then
    info "nvidia-container-toolkit already installed"
else
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    info "nvidia-container-toolkit installed"
fi

heading "Step 3/10 — GPU verification"
if [[ ! -c /dev/nvidia0 && ! -c /dev/nvidiactl ]]; then
    cat >&2 <<EOF

  ${RED}GPU DEVICES NOT FOUND${NC}
  No /dev/nvidia* devices. For Proxmox LXC add to /etc/pve/lxc/<VMID>.conf
  on the host and restart the container:

    lxc.cgroup2.devices.allow: c 195:* rwm
    lxc.cgroup2.devices.allow: c 510:* rwm
    lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file
    lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file
    lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file
    lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file
    lxc.mount.entry: /dev/nvidia-caps dev/nvidia-caps none bind,optional,create=dir

  Bare metal: install NVIDIA drivers and reboot.
EOF
    exit 1
fi

docker pull -q nvidia/cuda:12.9.2-base-ubuntu24.04 >/dev/null 2>&1 || true
GPU_OUTPUT=$(docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    nvidia/cuda:12.9.2-base-ubuntu24.04 nvidia-smi -L 2>&1) || true
echo "$GPU_OUTPUT" | grep -q "GPU [0-9]" || die "GPU not accessible in Docker: $GPU_OUTPUT"
info "GPU: $(echo "$GPU_OUTPUT" | head -1)"

heading "Step 4/10 — Repository"
if [[ -d "$INSTALL_DIR/.git" ]]; then
    info "repo exists at $INSTALL_DIR, pulling..."
    git -C "$INSTALL_DIR" pull --ff-only --quiet || warn "git pull failed, continuing"
elif [[ -f "$INSTALL_DIR/llama.sh" ]]; then
    # Directory pushed with sync.sh instead of git clone: use it as-is.
    info "using existing directory $INSTALL_DIR (not a git checkout)"
else
    mkdir -p "$(dirname "$INSTALL_DIR")"
    git clone --quiet "$REPO_URL" "$INSTALL_DIR"
fi
[[ -f "$INSTALL_DIR/llama.sh" ]] || die "llama.sh missing in $INSTALL_DIR"
chmod +x "$INSTALL_DIR/llama.sh" "$INSTALL_DIR/deploy/"*.sh "$INSTALL_DIR/scripts/"*.sh 2>/dev/null || true

heading "Step 5/10 — Storage and disk space"
mkdir -p "$INSTALL_DIR/models"
MIN_GB=40
$BUILD_LOCAL && MIN_GB=$((MIN_GB + 15))
AVAIL_GB=$(df -BG / | awk 'NR==2{gsub(/G/,"",$4); print $4+0}')
info "free on /: ${AVAIL_GB}G (need ~${MIN_GB}G)"
[[ "$AVAIL_GB" -ge "$MIN_GB" ]] || die "not enough disk: ${AVAIL_GB}G < ${MIN_GB}G"

heading "Step 6/10 — Server image"
if $BUILD_LOCAL; then
    LLAMA_IMAGE="ghcr.io/noxgle/llama-server-16gb:local"
    info "building locally (LLAMA_REF=$LLAMA_REF, sm_$GGML_CUDA_ARCH, native)..."
    info "this takes 60-120 min on 4 vCPU"
    cd "$INSTALL_DIR"
    LLAMA_IMAGE="$LLAMA_IMAGE" LLAMA_REF="$LLAMA_REF" LLAMA_NATIVE=ON GGML_CUDA_ARCH="$GGML_CUDA_ARCH" \
        docker compose build
else
    info "pulling $LLAMA_IMAGE ..."
    if ! docker pull "$LLAMA_IMAGE"; then
        cat >&2 <<EOF

  ${RED}IMAGE NOT AVAILABLE${NC}
  docker pull failed and no local image is present.
  Either wait for the CI build (ghcr.io/noxgle/llama-server-16gb) or run:
      bash deploy/install-llama.sh $MODEL --build-local
EOF
        exit 1
    fi
fi

heading "Step 7/10 — Model weights"
if $SKIP_DOWNLOAD; then
    info "skipped (--skip-download)"
elif $ROUTER; then
    # Router presets reference all three GGUFs.
    DEST="$INSTALL_DIR/models" bash "$INSTALL_DIR/deploy/download-model.sh" \
        --quant iq3s --dest "$INSTALL_DIR/models"
    DEST="$INSTALL_DIR/models" bash "$INSTALL_DIR/deploy/download-model.sh" \
        --quant q3kxl --no-mtp --dest "$INSTALL_DIR/models"
elif [[ "$MODEL" == inkling2 ]]; then
    DEST="$INSTALL_DIR/models" bash "$INSTALL_DIR/deploy/download-model.sh" \
        --model inkling2 --dest "$INSTALL_DIR/models"
else
    DL_ARGS=(--quant "$QUANT" --dest "$INSTALL_DIR/models")
    [[ "$MTP" -eq 1 ]] || DL_ARGS+=(--no-mtp)
    DEST="$INSTALL_DIR/models" bash "$INSTALL_DIR/deploy/download-model.sh" "${DL_ARGS[@]}"
fi
if $ROUTER; then
    REQUIRED=(Qwen3.8-27B-UD-IQ3_S.gguf mtp-Qwen3.8-27B-Q4_0.gguf Qwen3.8-27B-UD-Q3_K_XL.gguf)
else
    MODEL_FILE=$(grep -m1 '^MODEL=' "$INSTALL_DIR/configs/$CONFIG_NAME" | cut -d= -f2 | sed 's#^/models/##')
    REQUIRED=("$MODEL_FILE")
fi
for f in "${REQUIRED[@]}"; do
    [[ -f "$INSTALL_DIR/models/$f" ]] || die "GGUF missing: $INSTALL_DIR/models/$f (run download-model.sh)"
done

heading "Step 8/10 — Configuration"
cp "$INSTALL_DIR/configs/$CONFIG_NAME" "$INSTALL_DIR/.env"
# Image pin must match what was actually obtained
sed -i "s#^LLAMA_IMAGE=.*#LLAMA_IMAGE=$LLAMA_IMAGE#" "$INSTALL_DIR/.env"
grep -q '^LLAMA_IMAGE=' "$INSTALL_DIR/.env" || echo "LLAMA_IMAGE=$LLAMA_IMAGE" >> "$INSTALL_DIR/.env"
grep -q '^LLAMA_REF=' "$INSTALL_DIR/.env" || echo "LLAMA_REF=$LLAMA_REF" >> "$INSTALL_DIR/.env"
grep -q '^GGML_CUDA_ARCH=' "$INSTALL_DIR/.env" || echo "GGML_CUDA_ARCH=$GGML_CUDA_ARCH" >> "$INSTALL_DIR/.env"
info ".env created from $CONFIG_NAME"
PORT=$(grep -m1 '^PORT=' "$INSTALL_DIR/.env" | cut -d= -f2)
PORT="${PORT:-8089}"

if $WITH_SYSTEMD; then
    if $ROUTER; then
        systemctl disable --now llama16-compose.service >/dev/null 2>&1 || true
        cp "$INSTALL_DIR/deploy/systemd/llama16-router.service" /etc/systemd/system/
        RUNTIME_UNIT=llama16-router.service
    else
        systemctl disable --now llama16-router.service >/dev/null 2>&1 || true
        cp "$INSTALL_DIR/deploy/systemd/llama16-compose.service" /etc/systemd/system/
        RUNTIME_UNIT=llama16-compose.service
    fi
    cp "$INSTALL_DIR/deploy/systemd/llama16-gpu-watchdog.service" /etc/systemd/system/
    cp "$INSTALL_DIR/deploy/systemd/llama16-gpu-watchdog.timer" /etc/systemd/system/
    sed -i "s#/opt/llama16#$INSTALL_DIR#g" /etc/systemd/system/llama16-*.service
    sed -i "s#/opt/llama16#$INSTALL_DIR#g" /etc/systemd/system/llama16-gpu-watchdog.timer
    sed -i "s#stable-b12811-v1#${LLAMA_IMAGE##*:}#g" "/etc/systemd/system/$RUNTIME_UNIT"
    systemctl daemon-reload
    systemctl enable --now "$RUNTIME_UNIT"
    systemctl enable --now llama16-gpu-watchdog.timer
    info "systemd: $RUNTIME_UNIT enabled + GPU watchdog timer active"
fi

heading "Step 9/10 — Starting server"
if $ROUTER; then
    LLAMA_IMAGE="$LLAMA_IMAGE" LLAMA_DIR="$INSTALL_DIR" LLAMA_PORT="$PORT" \
        bash "$INSTALL_DIR/deploy/start-router.sh"
else
    docker rm -f llama16-router >/dev/null 2>&1 || true
    cd "$INSTALL_DIR"
    docker compose down 2>/dev/null || true
    docker compose up -d
fi

info "waiting for /health (model load can take 60-180 s)..."
READY=false
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${PORT}/health" 2>/dev/null | grep -q '"ok"'; then
        READY=true
        break
    fi
    (( i % 6 == 0 )) && echo "  still loading (${i}0 s)..."
    sleep 5
done
$READY || die "server did not become healthy — check: docker logs llama16-router"

heading "Step 10/10 — Verification"
if $ROUTER; then
    info "presets: $(curl -s "http://localhost:${PORT}/models" | python3 -c 'import json,sys; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))')"
    info "loading default preset iq3s-mtp with a probe request..."
    curl -s --max-time 600 "http://localhost:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"iq3s-mtp","messages":[{"role":"user","content":"Reply with OK"}],"max_tokens":32,"reasoning_effort":"low"}' \
        | python3 -c 'import json,sys; d=json.load(sys.stdin); print("  content:", repr(d["choices"][0]["message"].get("content")))' || true
else
    docker compose ps
fi
VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
info "health: OK"
info "VRAM used: ${VRAM} MiB of $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MiB"

cat <<EOF

$(echo -e "${GREEN}Installation complete${NC}")

  mode:    $MODE
  config:  $( $ROUTER && echo configs/router.ini || echo "$CONFIG_NAME -> .env" )
  port:    ${PORT}
  dir:     $INSTALL_DIR
  image:   $LLAMA_IMAGE

Manage:
  $( $ROUTER && echo "bash $INSTALL_DIR/deploy/start-router.sh   # idempotent start" || echo "cd $INSTALL_DIR && docker compose down && docker compose up -d" )
  ./sync.sh deploy          # from your workstation
  curl http://localhost:${PORT}/health
  curl http://localhost:${PORT}/models   # router presets + status

Reboot: the container has restart: unless-stopped; with --with-systemd the
matching unit and the GPU watchdog timer are enabled.
EOF
