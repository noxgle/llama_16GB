# syntax=docker/dockerfile:1

# ──────────────────────────────────────────────
# Stage 1: Build llama-server
# ──────────────────────────────────────────────
# CUDA >= 12.8 is required for Blackwell / sm_120 (RTX 50-series).
# Building on 12.4 yields an image that cannot run on the target GPU.
FROM nvidia/cuda:12.9.2-devel-ubuntu24.04 AS builder

ARG LLAMA_REPO=https://github.com/ggml-org/llama.cpp.git
ARG LLAMA_REF=b12811
ARG LLAMA_NATIVE=OFF
ARG GGML_CUDA_ARCH=120a
ARG BUILD_JOBS=4

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential libcurl4-openssl-dev libssl-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone pinned llama.cpp reference (tag, branch, or commit hash).
# HTTP archive is used because `git fetch --depth 1` does not work for arbitrary hashes.
RUN mkdir -p /tmp/llama-cpp-build && \
    curl -L "${LLAMA_REPO%.git}/archive/${LLAMA_REF}.tar.gz" | \
    tar -xz --strip-components=1 -C /tmp/llama-cpp-build

WORKDIR /tmp/llama-cpp-build

# CUDA driver stubs: link against them, real driver comes from the host at runtime
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1 && \
    ldconfig
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# GGML_CUDA_ARCH is explicit so CI (no GPU) still produces sm_120 kernels.
RUN cmake -B build \
        -DLLAMA_CURL=ON \
        -DLLAMA_CUDA=ON \
        -DLLAMA_OPENSSL=ON \
        -DGGML_CUDA_NCCL=OFF \
        -DGGML_NATIVE=${LLAMA_NATIVE} \
        -DCMAKE_CUDA_ARCHITECTURES=${GGML_CUDA_ARCH} \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath,/usr/local/cuda/lib64" \
    && cmake --build build --config Release -j${BUILD_JOBS} -- llama-server

# ──────────────────────────────────────────────
# Stage 2: Runtime
# ──────────────────────────────────────────────
# devel (not runtime) keeps cuBLAS/cuDART available without chasing package lists.
FROM nvidia/cuda:12.9.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

COPY --from=builder /tmp/llama-cpp-build/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /tmp/llama-cpp-build/build/bin/*.so* /usr/local/lib/

RUN ldconfig

# Models are bind-mounted at /models (read-only); no HF cache layer here.
RUN mkdir -p /models

EXPOSE 8089

ENTRYPOINT ["llama-server"]
