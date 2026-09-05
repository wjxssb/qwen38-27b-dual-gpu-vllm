#!/usr/bin/env bash
set -euo pipefail

GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"

echo -e "${BLUE}==============================================================${NC}"
echo -e "${BLUE}  Qwen3.8-27B-NVFP4 Dual-GPU Production Server Launcher      ${NC}"
echo -e "${BLUE}  Verified Stack: 262K Native Context | TP=2 | MTP K=3        ${NC}"
echo -e "${BLUE}==============================================================${NC}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${DIR}/config.env" ]; then
    echo -e "${GREEN}[*] Loading configuration from config.env${NC}"
    # shellcheck disable=SC1091
    source "${DIR}/config.env"
fi

CONTAINER_NAME="${CONTAINER_NAME:-qwen38-27b-vllm}"
HOST_PORT="${HOST_PORT:-18094}"
GPU_DEVICES="${GPU_DEVICES:-all}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.8-27B-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-unsloth/Qwen3.8-27B-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-57926baca9a82b4d6906b43f2750d55315f5b10f}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4}"
MTP_SPEC_TOKENS="${MTP_SPEC_TOKENS:-3}"
HOST_MEM_LIMIT="${HOST_MEM_LIMIT:-48g}"

if [ -z "${HF_CACHE_DIR:-}" ]; then
    if [ -d "${HOME}/ai/hf" ]; then
        HF_CACHE_DIR="${HOME}/ai/hf"
    else
        HF_CACHE_DIR="${HOME}/.cache/huggingface"
    fi
fi
mkdir -p "${HF_CACHE_DIR}"

if ! command -v docker >/dev/null 2>&1; then
    echo -e "${RED}[ERROR] Docker is not installed or not in PATH.${NC}" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo -e "${RED}[ERROR] nvidia-smi not found. NVIDIA drivers are required.${NC}" >&2
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "${GPU_COUNT}" -lt 2 ]; then
    echo -e "${YELLOW}[WARNING] Detected ${GPU_COUNT} GPU(s). Tensor Parallel (TP=2) requires 2 GPUs.${NC}"
fi

if docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}[*] Container '${CONTAINER_NAME}' is already running.${NC}"
    echo -e "    API Endpoint : http://127.0.0.1:${HOST_PORT}/v1"
    echo -e "    View logs    : docker logs -f ${CONTAINER_NAME}"
    echo -e "    Stop server  : ./stop.sh"
    exit 0
fi

if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}[*] Removing existing stopped container '${CONTAINER_NAME}'...${NC}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1
fi

if [ -z "${DOCKER_IMAGE:-}" ]; then
    # Auto-detect local custom SM120 image if present, otherwise use official vLLM
    LOCAL_IMAGE=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "vllm-nvfp4-kv-sm120" | head -n 1 || true)
    if [ -n "${LOCAL_IMAGE}" ]; then
        DOCKER_IMAGE="${LOCAL_IMAGE}"
    elif docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^vllm/vllm-openai:latest$"; then
        DOCKER_IMAGE="vllm/vllm-openai:latest"
    else
        DOCKER_IMAGE="vllm/vllm-openai:latest"
        echo -e "${YELLOW}[*] Pulling official vLLM image '${DOCKER_IMAGE}'...${NC}"
        docker pull "${DOCKER_IMAGE}"
    fi
fi

echo -e "${GREEN}[*] Container Name : ${CONTAINER_NAME}${NC}"
echo -e "${GREEN}[*] Docker Image   : ${DOCKER_IMAGE}${NC}"
echo -e "${GREEN}[*] Target Model   : ${MODEL_NAME}${NC}"
echo -e "${GREEN}[*] Model Revision : ${MODEL_REVISION}${NC}"
echo -e "${GREEN}[*] Host Port      : ${HOST_PORT}${NC}"
echo -e "${GREEN}[*] HF Cache Dir   : ${HF_CACHE_DIR}${NC}"
echo -e "${GREEN}[*] Max Context    : ${MAX_MODEL_LEN} tokens${NC}"
echo -e "${GREEN}[*] MTP Speculative: K=${MTP_SPEC_TOKENS}${NC}"
echo -e "${GREEN}[*] Starting vLLM serving engine in background...${NC}"

docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart unless-stopped \
    --memory "${HOST_MEM_LIMIT}" \
    --memory-swap "${HOST_MEM_LIMIT}" \
    --gpus "${GPU_DEVICES}" \
    --ipc host \
    --ulimit memlock=-1:-1 \
    --publish "127.0.0.1:${HOST_PORT}:8000" \
    --volume "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    --env HF_HOME=/root/.cache/huggingface \
    --env VLLM_NO_USAGE_STATS=1 \
    --env PYTHONUNBUFFERED=1 \
    --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --env NCCL_P2P_DISABLE=1 \
    --env NCCL_SHM_DISABLE=0 \
    --env VLLM_KV_CACHE_LAYOUT=HND \
    --env VLLM_USE_V2_MODEL_RUNNER=0 \
    --env VLLM_SM120_NVFP4_K3_NATIVE=1 \
    --env VLLM_SM120_NVFP4_K3_GRAPH=1 \
    "${DOCKER_IMAGE}" \
    "${MODEL_NAME}" \
    --revision "${MODEL_REVISION}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --distributed-executor-backend mp \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs 1 \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --gpu-memory-utilization 0.914 \
    --linear-backend auto \
    --disable-custom-all-reduce \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --language-model-only \
    --attention-config '{"backend":"FLASHINFER","use_trtllm_attention":false}' \
    --block-size 16 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --spec-method mtp \
    --spec-tokens "${MTP_SPEC_TOKENS}" \
    --no-enforce-eager \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4],"max_cudagraph_capture_size":4}' \
    --kv-cache-memory-bytes 3152594944

echo -e "${GREEN}==============================================================${NC}"
echo -e "${GREEN}  Service successfully launched in container '${CONTAINER_NAME}'!${NC}"
echo -e "${GREEN}==============================================================${NC}"
echo -e "API Base URL : http://127.0.0.1:${HOST_PORT}/v1"
echo -e "Live Logs    : docker logs -f ${CONTAINER_NAME}"
echo -e "Health Check : ./test_api.sh"
echo -e "Stop Service : ./stop.sh"
echo -e ""
echo -e "${YELLOW}Note: If model weights are not pre-cached, vLLM will automatically download them from Hugging Face.${NC}"
echo -e "${YELLOW}Loading weights and CUDA graph capture take ~2-3 minutes on startup.${NC}"
