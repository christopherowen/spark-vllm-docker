
# vLLM Ray Cluster Node Docker for DGX Spark

This repository contains the Docker configuration and startup scripts to run a multi-node vLLM inference cluster using Ray. It supports InfiniBand/RDMA (NCCL) and custom environment configuration for high-performance setups.

## DISCLAIMER

This repository is not affiliated with NVIDIA or their subsidiaries. The content is provided as a reference material only, not intended for production use.
Some of the steps and parameters may be unnecessary, and some may be missing. This is a work in progress. Use at your own risk!

The Dockerfile builds from the main branch of VLLM, so depending on when you run the build process, it may not be in fully functioning state.

## 1\. Building the Docker Image

The Dockerfile includes specific **Build Arguments** to allow you to selectively rebuild layers (e.g., update the vLLM source code without re-downloading PyTorch).

### Option A: Standard Build (First Time)

```bash
docker build -t vllm-node .
```

### Option B: Fast Rebuild (Update vLLM Source Only)

Use this if you want to pull the latest code from GitHub but keep the heavy dependencies (Torch, FlashInfer, system deps) cached.

```bash
docker build \
  --build-arg CACHEBUST_VLLM=$(date +%s) \
  -t vllm-node .
```

### Option C: Full Rebuild (Update All Dependencies)

Use this to force a re-download of PyTorch, FlashInfer, and system packages.

```bash
docker build \
  --build-arg CACHEBUST_DEPS=$(date +%s) \
  -t vllm-node .
```

### Copying the container to another Spark node

To avoid extra network overhead, you can copy the image directly to your second Spark node via ConnectX 7 interface by using the following command:

```bash
docker save vllm-node | ssh your_username@another_spark_hostname_or_ip "docker load"
```

-----

## 2\. Running the Container

Ray and NCCL require specific Docker flags to function correctly across multiple nodes (Shared memory, Network namespace, and Hardware access).

```bash
docker run -it --rm \
  --gpus all \
  --net=host \
  --ipc=host \
  --privileged \
  --name vllm_node \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-node bash
```

Or if you want to start the cluster node (head or regular), you can launch with the run-cluster.sh script (see details below):

**On head node:**

```bash
docker run --privileged --gpus all -it --rm \
  --ipc=host \
  --network host \
  --name vllm_node \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-node ./run-cluster-node.sh \
    --role head \
    --host-ip 192.168.177.11 \
    --eth-if enp1s0f1np1 \
    --ib-if rocep1s0f1 
```

**On worker node**

```bash
docker run --privileged --gpus all -it --rm \
  --ipc=host \
  --network host \
  --name vllm_node \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-node ./run-cluster-node.sh \
    --role node \
    --host-ip 192.168.177.12 \
    --eth-if enp1s0f1np1 \
    --ib-if rocep1s0f1 \
    --head-ip 192.168.177.11
```


**Flags Explained:**

  * `--net=host`: **Required.** Ray and NCCL need full access to host network interfaces.
  * `--ipc=host`: **Recommended.** Allows shared memory access for PyTorch/NCCL. As an alternative, you can set it via `--shm-size=16g`.
  * `--privileged`: **Recommended for InfiniBand.** Grants the container access to RDMA devices (`/dev/infiniband`). As an alternative, you can pass `--ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/infiniband`.

-----

## 3\. Using `run-cluster-node.sh`

The script is used to configure the environment and launch Ray either in head or node mode.

Normally you would start it with the container like in the example above, but you can launch it inside the Docker session manually if needed (but make sure it's not already running).

### Syntax

```bash
./run-cluster-node.sh [OPTIONS]
```

| Flag | Long Flag | Description | Required? |
| :--- | :--- | :--- | :--- |
| `-r` | `--role` | Role of the machine: `head` or `node`. | **Yes** |
| `-h` | `--host-ip` | The IP address of **this** specific machine (IB or Eth IP). | **Yes** |
| `-e` | `--eth-if` | Ethernet interface name (e.g., `eth0`, `enp3s0`). | **Yes** |
| `-i` | `--ib-if` | InfiniBand interface name (e.g., `ib0`, `rocep1s0f1`). | **Yes** |
| `-m` | `--head-ip` | The IP address of the **Head Node**. | Only if role is `node` |


**Hint**: to decide which interfaces to use, you can run `ibdev2netdev`. You will see an output like this:

```
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
```

Each physical port on Spark has two pairs of logical interfaces in Linux. 
Current NVIDIA guidance recommends using only one of them, in this case it would be `enp1s0f1np1` for Ethernet and `rocep1s0f1` for IB.

You need to make sure you allocate IP addresses to them (no need to allocate IP to their "twins").

### Example: Starting inside the Head Node

```bash
./run-cluster-node.sh \
  --role head \
  --host-ip 192.168.177.11 \
  --eth-if enp1s0f1np1 \
  --ib-if rocep1s0f1
```

### Example: Starting inside a Worker Node

```bash
./run-cluster-node.sh \
  --role node \
  --host-ip 192.168.177.12 \
  --eth-if enp1s0f1np1 \
  --ib-if rocep1s0f1 \
  --head-ip 192.168.177.11
```

-----

## 4\. Configuration Details

### Environment Persistence

The script automatically appends exported variables to `~/.bashrc`. If you need to open a second terminal into the running container for debugging, simply run:

```bash
docker exec -it vllm_node bash
```

All environment variables (NCCL, Ray, vLLM config) set by the startup script will be loaded automatically in this new session.

## 5.\. Using cluster mode for inference

First, start follow the instructions above to start the head container on your first Spark, and node container on the second Spark.
Then, on the first Spark, run vllm like this:

```bash
docker exec -it vllm_node bash -i -c "vllm serve RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4 --port 8888 --host 0.0.0.0 --gpu-memory-utilization 0.7 -tp 2 --distributed-executor-backend ray --max-model-len 32768"
```

Alternatively, run an interactive shell first:

```bash
docker exec -it vllm_node
```

And execute vllm command inside.


### Hardware Architecture

**Note:** The Dockerfile defaults to `TORCH_CUDA_ARCH_LIST=12.1a` (NVIDIA GB10). If you are using different hardware, update the `ENV` variable in the Dockerfile before building.

