#!/usr/bin/env bash
# Download Qwen3.8-27B GGUF files into ./models and verify them.
#
# Why not `llama-server -hf`: the MTP head lives in a subdirectory (MTP/) and
# the HF downloader fails with "no GGUF files found" for such repos. Local
# files + -m/-md are reliable.
#
# Usage:
#   deploy/download-model.sh                      # UD-IQ3_S + MTP (default)
#   deploy/download-model.sh --quant q3kxl --no-mtp
#   deploy/download-model.sh --quant iq2s --with-mmproj
#
# Options:
#   --quant iq3s|q3kxl|iq2s   quantization (default: iq3s)
#   --dest DIR                target dir (default: <repo>/models)
#   --no-mtp                  skip the MTP draft head
#   --with-mmproj             also download the vision projector
#   --revision REV            HF revision (default: pinned commit)

set -euo pipefail

REPO="unsloth/Qwen3.8-27B-GGUF"
REVISION="4af51b3542f775b0a8e4b310bfb3f17e01c306c9"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${DEST:-$SCRIPT_DIR/../models}"

QUANT="iq3s"
WITH_MTP=1
WITH_MMPROJ=0

MTP_PATH="MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
MTP_SIZE=1369590656
MMPROJ_PATH="mmproj-F16.gguf"
MMPROJ_SIZE=927607488

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quant)      QUANT="${2:?}"; shift 2 ;;
        --dest)       DEST="${2:?}"; shift 2 ;;
        --revision)   REVISION="${2:?}"; shift 2 ;;
        --no-mtp)     WITH_MTP=0; shift ;;
        --with-mmproj) WITH_MMPROJ=1; shift ;;
        -h|--help)    usage ;;
        *)            echo "Unknown option: $1" >&2; usage ;;
    esac
done

case "$QUANT" in
    iq3s)  MAIN_PATH="Qwen3.8-27B-UD-IQ3_S.gguf";  MAIN_SIZE=12040883104 ;;
    q3kxl) MAIN_PATH="Qwen3.8-27B-UD-Q3_K_XL.gguf"; MAIN_SIZE=13146393504 ;;
    iq2s)  MAIN_PATH="Qwen3.8-27B-UD-IQ2_S.gguf";  MAIN_SIZE=8422719488 ;;
    *) echo "Unknown quant: $QUANT (use iq3s|q3kxl|iq2s)" >&2; exit 1 ;;
esac

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "  ${GREEN}*${NC} $*"; }
warn() { echo -e "  ${YELLOW}WARN${NC} $*" >&2; }
die()  { echo -e "  ${RED}ERROR${NC} $*" >&2; exit 1; }

mkdir -p "$DEST"

HF_CLI=""
if command -v hf >/dev/null 2>&1; then
    HF_CLI="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI="huggingface-cli"
fi

verify() {
    local file="$1" size="$2"
    [[ -f "$file" ]] || die "missing file: $file"
    local magic
    magic=$(head -c 4 "$file" | od -An -tx1 | tr -d ' \n')
    [[ "$magic" == "47475546" ]] || die "not a GGUF file: $file (magic=$magic)"
    local actual
    actual=$(stat -c%s "$file")
    if [[ "$actual" -ne "$size" ]]; then
        die "size mismatch for $file: $actual != $size (delete and re-run)"
    fi
    info "verified $(basename "$file") ($(numfmt --to=iec "$actual"))"
}

fetch() {
    local path="$1" size="$2"
    local target
    target="$DEST/$(basename "$path")"

    if [[ -f "$target" ]] && [[ "$(stat -c%s "$target")" -eq "$size" ]]; then
        info "already present: $(basename "$target")"
        return 0
    fi

    echo ""
    info "downloading $path ($(numfmt --to=iec "$size"))"

    if [[ -n "$HF_CLI" ]]; then
        local tmp="$DEST/_hf_download"
        rm -rf "$tmp"
        "$HF_CLI" download "$REPO" --revision "$REVISION" --local-dir "$tmp" --include "$path"
        mv "$tmp/$path" "$target"
        rm -rf "$tmp"
    else
        warn "huggingface_hub CLI not found, using curl (resumable)"
        curl -L -C - --retry 5 --retry-delay 3 --progress-bar \
            -o "$target.part" \
            "https://huggingface.co/$REPO/resolve/$REVISION/$path"
        mv "$target.part" "$target"
    fi

    verify "$target" "$size"
}

echo ""
echo "Qwen3.8-27B download"
echo "  repo:     $REPO"
echo "  revision: $REVISION"
echo "  quant:    $QUANT"
echo "  dest:     $DEST"
echo "  mtp:      $([[ $WITH_MTP -eq 1 ]] && echo yes || echo no)"
echo "  mmproj:   $([[ $WITH_MMPROJ -eq 1 ]] && echo yes || echo no)"

fetch "$MAIN_PATH" "$MAIN_SIZE"
[[ "$WITH_MTP" -eq 1 ]] && fetch "$MTP_PATH" "$MTP_SIZE"
[[ "$WITH_MMPROJ" -eq 1 ]] && fetch "$MMPROJ_PATH" "$MMPROJ_SIZE"

echo ""
info "done. files in $DEST:"
ls -lh "$DEST"/*.gguf 2>/dev/null || true
