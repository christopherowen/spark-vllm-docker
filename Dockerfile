# syntax=docker/dockerfile:1.6

ARG BUILD_JOBS=15

# =========================================================
# STAGE 1: Base Image (Installs Dependencies)
# =========================================================
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS base

# Try to keep the system from trashing during builds
ARG BUILD_JOBS
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"

# Set non-interactive frontend to prevent apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# Allow pip to install globally on Ubuntu 24.04 without a venv
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Set the base directory environment variable
ENV VLLM_BASE_DIR=/workspace/vllm

# 1. Install Build Dependencies & Ccache
# Added ccache to enable incremental compilation caching
RUN apt update && apt upgrade -y \
    && apt install -y --allow-change-held-packages --no-install-recommends \
    curl vim cmake build-essential ninja-build \
    libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
    python3 python3-dev python3-pip python3-venv git wget \
    libnccl-dev libnccl2 libibverbs1 libibverbs-dev rdma-core \
    libopenmpi3 libopenblas0-pthread libnuma1 \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Configure Ccache for CUDA/C++
ENV PATH=/usr/lib/ccache:$PATH
ENV CCACHE_DIR=/root/.ccache
# Tell CMake to use ccache for compilation
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# Setup Workspace
WORKDIR $VLLM_BASE_DIR

# 2. Set Environment Variables
ENV TORCH_CUDA_ARCH_LIST=12.1a
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# 3. Install Python Dependencies with Cache Mounts
# Using --mount=type=cache ensures that even if this layer invalidates, 
# pip reuses previously downloaded wheels.

# Set pip cache directory
ENV PIP_CACHE_DIR=/root/.cache/pip

# Copy Patches
COPY patches/ /tmp/patches/

# =========================================================
# STAGE 2: PyTorch Builder (Builds wheel independently)
# =========================================================
FROM base AS pytorch-builder

ARG PYTORCH_REPO=https://github.com/pytorch/pytorch.git
ARG PYTORCH_REF=ffcbb7fd6109df6b65e96fe07287255e387f0123
ARG TORCH_CUDA_ARCH_LIST_DEFAULT="12.1a"

ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST_DEFAULT}"
ENV USE_CCACHE=1
ENV CCACHE_DIR=/root/.ccache

# Clone/update PyTorch using a persistent repo cache
RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    set -eux; \
    cd /repo-cache; \
    if [ ! -d pytorch ]; then \
      echo "Cache miss: cloning PyTorch..."; \
      git clone --recursive "${PYTORCH_REPO}" pytorch; \
    fi; \
    cd pytorch; \
    git fetch --all; \
    git checkout "${PYTORCH_REF}"; \
    if [ "${PYTORCH_REF}" = "main" ]; then \
      git reset --hard origin/main; \
    fi; \
    git submodule sync; \
    git submodule update --init --recursive; \
    rm -rf "${VLLM_BASE_DIR}/pytorch"; \
    cp -a /repo-cache/pytorch "${VLLM_BASE_DIR}/pytorch"

WORKDIR $VLLM_BASE_DIR/pytorch
ENV PYTORCH_VENV=${VLLM_BASE_DIR}/pytorch/.venv-build
ENV PATH=${PYTORCH_VENV}/bin:$PATH

RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    set -eux; \
    python3 -m venv "$PYTORCH_VENV"; \
    pip install -U pip setuptools wheel

# Build torch wheel (with a wheelhouse cache) and export it like Triton does
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=torch-wheelhouse,target=/wheelhouse \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      build-essential ninja-build cmake ccache clang lld patchelf pkg-config \
      libssl-dev libffi-dev libopenblas-dev libomp-dev libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*; \
    \
    patch="pytorch-sm121.patch"; \
    echo "==> Applying $patch"; \
    (patch --dry-run -p1 < "/tmp/patches/$patch" && patch -p1 < "/tmp/patches/$patch"); \
    \
    WHEEL_CACHE_DIR="/wheelhouse/pytorch/${PYTORCH_REF}/sm121a/aarch64/cu130"; \
    mkdir -p "$WHEEL_CACHE_DIR" /workspace/wheels; \
    \
    PYTORCH_WHEEL="$(ls -1 "$WHEEL_CACHE_DIR"/torch-*.whl 2>/dev/null | head -n1 || true)"; \
    if [ -z "$PYTORCH_WHEEL" ]; then \
      echo "==> Wheel cache miss: building torch wheel"; \
      rm -f "$WHEEL_CACHE_DIR"/torch-*.whl; \
      pip install -r requirements.txt; \
      export USE_CUDA=1 USE_DISTRIBUTED=1 BUILD_TEST=0 USE_KINETO=0 USE_ITT=0 USE_MKLDNN=0; \
      python setup.py bdist_wheel; \
      cp dist/torch-*.whl "$WHEEL_CACHE_DIR"/; \
      PYTORCH_WHEEL="$(ls -1 "$WHEEL_CACHE_DIR"/torch-*.whl | head -n1)"; \
    else \
      echo "==> Wheel cache hit: $PYTORCH_WHEEL"; \
    fi; \
    \
    cp -a "$WHEEL_CACHE_DIR"/*.whl /workspace/wheels/; \
    rm -rf "$PYTORCH_VENV"

# =========================================================
# STAGE 3: Triton Builder (Builds wheel independently)
# =========================================================
FROM base AS triton-builder

ARG TRITON_REPO=https://github.com/triton-lang/triton.git
# Set to v3.5.1 tag by default (pass from CLI to change)
ARG TRITON_REF=v3.5.1

# Clone/update Triton using the same persistent repo cache approach
RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    set -eux; \
    cd /repo-cache; \
    if [ ! -d triton ]; then \
      echo "Cache miss: cloning Triton..."; \
      git clone --recursive "${TRITON_REPO}" triton; \
    fi; \
    cd triton; \
    git fetch --all --tags; \
    git checkout "${TRITON_REF}"; \
    if [ "${TRITON_REF}" = "main" ]; then \
      git reset --hard origin/main; \
    fi; \
    git submodule sync; \
    git submodule update --init --recursive; \
    rm -rf "${VLLM_BASE_DIR}/triton"; \
    cp -a /repo-cache/triton "${VLLM_BASE_DIR}/triton"

WORKDIR $VLLM_BASE_DIR/triton
ENV TRITON_VENV=${VLLM_BASE_DIR}/triton/.venv-build
ENV PATH=${TRITON_VENV}/bin:$PATH

RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    set -eux; \
    python3 -m venv "$TRITON_VENV"; \
    pip install -U pip setuptools wheel

# Build (or reuse) Triton wheels from a persistent wheelhouse cache
RUN --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    --mount=type=cache,id=triton-wheelhouse,target=/wheelhouse \
    set -eux; \
    pip install -r python/requirements.txt; \
    \
    # Key the wheel cache by ref + environment (good enough given fixed base image)
    WHEEL_CACHE_DIR="/wheelhouse/triton/${TRITON_REF}/cp312/aarch64/cu130"; \
    mkdir -p "$WHEEL_CACHE_DIR" /workspace/wheels; \
    \
    TRITON_WHEEL="$(ls -1 "$WHEEL_CACHE_DIR"/triton-*.whl 2>/dev/null | head -n1 || true)"; \
    KERNELS_WHEEL="$(ls -1 "$WHEEL_CACHE_DIR"/triton_kernels-*.whl 2>/dev/null | head -n1 || true)"; \
    \
    if [ -z "$TRITON_WHEEL" ] || [ -z "$KERNELS_WHEEL" ]; then \
      echo "==> Wheel cache miss: building Triton wheels"; \
      pip wheel --no-build-isolation . --wheel-dir="$WHEEL_CACHE_DIR" -v; \
      pip wheel --no-build-isolation python/triton_kernels --no-deps --wheel-dir="$WHEEL_CACHE_DIR"; \
    else \
      echo "==> Wheel cache hit: $TRITON_WHEEL and $KERNELS_WHEEL"; \
    fi; \
    \
    cp -a "$WHEEL_CACHE_DIR"/*.whl /workspace/wheels/; \
    rm -rf "$TRITON_VENV"

# =========================================================
# STAGE 4: FlashInfer Builder (Nightly wheels; force JIT for sm_121a)
# =========================================================
FROM base AS flashinfer-builder

# Match Shaun: nightly index + cu130 JIT cache, skip flashinfer-cubin to force JIT SM121a
ARG FLASHINFER_INDEX_BASE=https://flashinfer.ai/whl/nightly/
ARG FLASHINFER_JIT_INDEX_CU130=https://flashinfer.ai/whl/nightly/cu130
ARG FLASHINFER_WHEEL_KEY=nightly
ARG FLASHINFER_CUDA_TAG=cu130

WORKDIR $VLLM_BASE_DIR
ENV FLASHINFER_VENV=${VLLM_BASE_DIR}/flashinfer/.venv-build
ENV PATH=${FLASHINFER_VENV}/bin:$PATH

RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    set -eux; \
    python3 -m venv "$FLASHINFER_VENV"; \
    pip install -U pip setuptools wheel

# Download (or reuse) wheels into a persistent wheelhouse cache, then export to /workspace/wheels
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    --mount=type=cache,id=flashinfer-wheelhouse,target=/wheelhouse \
    set -eux; \
    \
    # Sanity check: nvcc supports sm_121a (fail fast if CUDA toolchain can't target Spark)
    echo '__global__ void k(){}' | nvcc -arch=sm_121a -x cu -c - -o /dev/null; \
    \
    WHEEL_CACHE_DIR="/wheelhouse/flashinfer/${FLASHINFER_WHEEL_KEY}/cp312/aarch64/${FLASHINFER_CUDA_TAG}"; \
    mkdir -p "$WHEEL_CACHE_DIR" /workspace/wheels; \
    \
    # Cache hit check (be flexible about filenames)
    if ls -1 "$WHEEL_CACHE_DIR"/flashinfer*python*.whl >/dev/null 2>&1 && \
       ls -1 "$WHEEL_CACHE_DIR"/flashinfer*jit*cache*.whl >/dev/null 2>&1; then \
      echo "==> Wheel cache hit: FlashInfer (${FLASHINFER_WHEEL_KEY}, ${FLASHINFER_CUDA_TAG})"; \
    else \
      echo "==> Wheel cache miss: downloading FlashInfer nightly wheels"; \
      rm -f "$WHEEL_CACHE_DIR"/flashinfer*.whl; \
      \
      # flashinfer-python (nightly) - no deps
      pip download --no-deps --pre \
        --index-url "${FLASHINFER_INDEX_BASE}" \
        -d "$WHEEL_CACHE_DIR" \
        flashinfer-python; \
      \
      # flashinfer-jit-cache (nightly/cu130) - no deps
      pip download --no-deps --pre \
        --index-url "${FLASHINFER_JIT_INDEX_CU130}" \
        -d "$WHEEL_CACHE_DIR" \
        flashinfer-jit-cache; \
    fi; \
    \
    cp -a "$WHEEL_CACHE_DIR"/*.whl /workspace/wheels/; \
    rm -rf "$FLASHINFER_VENV"


# =========================================================
# STAGE 5: vLLM Builder (Builds vLLM and dependancies wheels)
# =========================================================
FROM base AS vllm-builder

# Git reference (branch, tag, or SHA) to checkout
ARG VLLM_REF=main

# 4. Smart Git Clone (Fetch changes instead of full re-clone)
# We mount a cache at /repo-cache. This directory persists on your host machine.
RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    set -eux; \
    cd /repo-cache; \
    if [ ! -d "vllm" ]; then \
        echo "Cache miss: Cloning vLLM from scratch..." && \
        git clone --recursive https://github.com/vllm-project/vllm.git vllm; \
    fi; \
    cd vllm; \
    git fetch --all; \
    git checkout ${VLLM_REF}; \
    if [ "${VLLM_REF}" = "main" ]; then \
        git reset --hard origin/main; \
    fi; \
    git submodule update --init --recursive; \
    rm -rf "${VLLM_BASE_DIR}/vllm"; \
    cp -a /repo-cache/vllm "${VLLM_BASE_DIR}/"

WORKDIR $VLLM_BASE_DIR/vllm
ENV VLLM_VENV=${VLLM_BASE_DIR}/vllm/.venv-build
ENV PATH=${VLLM_VENV}/bin:$PATH

RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    set -eux; \
    python3 -m venv "$VLLM_VENV"; \
    pip install -U pip setuptools wheel

# Install custom PyTorch, Triton before vLLM tooling
COPY --from=pytorch-builder /workspace/wheels/. /workspace/wheels/
COPY --from=triton-builder  /workspace/wheels/. /workspace/wheels/
COPY --from=flashinfer-builder /workspace/wheels/. /workspace/wheels/
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install /workspace/wheels/*.whl && rm -rf /workspace/wheels

# Install additional dependencies
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install apache-tvm-ffi nvidia-cudnn-frontend nvidia-cutlass-dsl nvidia-ml-py tabulate

# Prepare build requirements
# prefer our flashinfer and pytorch
# xgrammar pulls in pytorch 2.9.1, so install it manually in the runner
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    sed -i -E '/^(torch|torchaudio|torchvision)==/d' requirements/cuda.txt && \
    sed -i -E '/^xgrammar([<=> ].*)?$/d' requirements/common.txt && \
    pip install -r requirements/build.txt

# Apply Patches
# Performance boost for spark: https://github.com/vllm-project/vllm/pull/28099
RUN set -eux; \
    patch="vllm-pr-28099.diff"; \
    echo "==> Applying $patch"; \
    cd "$VLLM_BASE_DIR/vllm"; \
    (patch --dry-run -p1 < "/tmp/patches/$patch" && patch -p1 < "/tmp/patches/$patch")

# fastsafetensors loading in cluster setup - tracking https://github.com/foundation-model-stack/fastsafetensors/issues/36
RUN set -eux; \
    patch="fastsafetensors-issue-36.patch"; \
    echo "==> Applying $patch"; \
    cd "$VLLM_BASE_DIR/vllm"; \
    (patch --dry-run -p1 < "/tmp/patches/$patch" && patch -p1 < "/tmp/patches/$patch")

# vLLM Compilation
# Build vLLM and deps wheels
RUN --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    set -eux; \
    mkdir -p /workspace/wheels && \
    pip wheel --no-build-isolation --no-deps . -w /workspace/wheels -v && \
    pip wheel -r requirements/cuda.txt -w /workspace/wheels; \
    rm -rf "$VLLM_VENV"

#xgrammar pulls in torch.  nuke it.
RUN set -eux; \
    rm -f /workspace/wheels/torch-*.whl \
          /workspace/wheels/torchaudio-*.whl \
          /workspace/wheels/torchvision-*.whl

# =========================================================
# STAGE 6: Runner (Transfers only necessary artifacts)
# =========================================================
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV VLLM_BASE_DIR=/workspace/vllm

# Set pip cache directory
ENV PIP_CACHE_DIR=/root/.cache/pip

# Install minimal runtime dependencies (NCCL, Python)
# Note: "devel" tools like cmake/gcc are NOT installed here to save space
RUN apt update && apt upgrade -y \
    && apt install -y --allow-change-held-packages --no-install-recommends \
    python3 python3-pip python3-dev vim curl git wget \
    libcudnn9-cuda-13 \
    libnccl-dev libnccl2 libibverbs1 libibverbs-dev rdma-core \
    libopenmpi3 libopenblas0-pthread libnuma1 \
    gnuplot-nox \
    && rm -rf /var/lib/apt/lists/*

# Set final working directory
WORKDIR $VLLM_BASE_DIR

# Cache + download tiktoken encodings (won't re-download across builds)
RUN --mount=type=cache,id=tiktoken-encodings,target=/root/.cache/tiktoken_encodings \
    set -eux; \
    mkdir -p /root/.cache/tiktoken_encodings; \
    for f in o200k_base.tiktoken cl100k_base.tiktoken; do \
      if [ ! -s "/root/.cache/tiktoken_encodings/$f" ]; then \
        echo "Downloading $f..."; \
        wget -q -O "/root/.cache/tiktoken_encodings/$f" \
          "https://openaipublic.blob.core.windows.net/encodings/$f"; \
      else \
        echo "Cache hit: $f"; \
      fi; \
    done; \
    mkdir -p "$VLLM_BASE_DIR/tiktoken_encodings"; \
    cp -a /root/.cache/tiktoken_encodings/. "$VLLM_BASE_DIR/tiktoken_encodings/"

# Install from wheels
COPY --from=pytorch-builder /workspace/wheels/. /workspace/wheels/
COPY --from=triton-builder /workspace/wheels/. /workspace/wheels/
COPY --from=flashinfer-builder /workspace/wheels/. /workspace/wheels/
COPY --from=vllm-builder  /workspace/wheels/. /workspace/wheels/

# Install the built wheels
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    set -eux; \
    pip install /workspace/wheels/*.whl; \
    rm -rf /workspace/wheels

# Install additional runtime dependencies
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install --no-deps xgrammar fastsafetensors  && \
    pip install apache-tvm-ffi nvidia-cudnn-frontend nvidia-cutlass-dsl nvidia-ml-py tabulate

# Setup Env for Runtime
ENV TORCH_CUDA_ARCH_LIST=12.1a
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TIKTOKEN_ENCODINGS_BASE=$VLLM_BASE_DIR/tiktoken_encodings
ENV PATH=$VLLM_BASE_DIR:$PATH

# Copy scripts
COPY run-cluster-node.sh $VLLM_BASE_DIR/
RUN chmod +x $VLLM_BASE_DIR/run-cluster-node.sh

# Final extra deps
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install ray[default] termplotlib
