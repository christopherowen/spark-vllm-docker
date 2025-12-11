FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

# Set non-interactive frontend to prevent apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# Allow pip to install globally on Ubuntu 24.04 without a venv
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Set the base directory environment variable
ENV VLLM_BASE_DIR=/workspace/vllm

# 1. Install System Dependencies
RUN apt update && apt upgrade -y && apt install -y --allow-change-held-packages \
    curl \
    vim \
    cmake \
    build-essential \
    ninja-build \
    python3-dev \
    python3-pip \
    git \
    wget \
    gnuplot \
    libnccl-dev \
    libnccl2 \
    libibverbs1 \
    libibverbs-dev \
    rdma-core \
    && rm -rf /var/lib/apt/lists/*

# Setup Workspace
WORKDIR $VLLM_BASE_DIR

# 2. Download Tiktoken files
RUN mkdir -p tiktoken_encodings && \
    wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

# 3. Set Environment Variables
ENV TORCH_CUDA_ARCH_LIST=12.1a
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TIKTOKEN_ENCODINGS_BASE=$VLLM_BASE_DIR/tiktoken_encodings

# --- CACHE BUSTER ---
# Change this argument to force a re-download of PyTorch/FlashInfer
ARG CACHEBUST_DEPS=1

# 4. Install Python Dependencies 

# Install PyTorch for CUDA 13.0
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install Helper libraries
RUN pip install xgrammar triton termplotlib

# Install FlashInfer
RUN pip install flashinfer-python --no-deps --index-url https://flashinfer.ai/whl && \
    pip install flashinfer-cubin --index-url https://flashinfer.ai/whl && \
    pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu130 && \
    pip install apache-tvm-ffi nvidia-cudnn-frontend nvidia-cutlass-dsl nvidia-ml-py tabulate

# Install fast safetensors to improve loading speeds
RUN pip install fastsafetensors

# --- VLLM SOURCE CACHE BUSTER ---
# Change THIS argument to force a fresh git clone and rebuild of vLLM
# without re-installing the dependencies above.
ARG CACHEBUST_VLLM=1

# 5. Clone and Build vLLM
RUN git clone --recursive https://github.com/vllm-project/vllm.git
WORKDIR $VLLM_BASE_DIR/vllm

# Prepare build requirements
RUN python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    pip install -r requirements/build.txt

# TEMPORARY PATCH for fastsafetensors loading in cluster setup - tracking https://github.com/foundation-model-stack/fastsafetensors/issues/36
COPY fastsafetensors.patch .
RUN patch -p1 < fastsafetensors.patch

# Final Build
# Uses --no-build-isolation to respect the pre-installed Torch/FlashInfer
RUN pip install --no-build-isolation . -v

# Set the final workdir
WORKDIR $VLLM_BASE_DIR

# Copy clustering script
COPY run-cluster-node.sh $VLLM_BASE_DIR/
RUN chmod +x $VLLM_BASE_DIR/run-cluster-node.sh

# Install additional modules for Ray dashboard support
RUN pip install ray[default]

