#!/usr/bin/env bash
set -euo pipefail

# ANSI color codes
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
NC="\033[0m"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${DIR}/config.env" ]; then
    # shellcheck disable=SC1091
    source "${DIR}/config.env"
fi

MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.8-27B-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-57926baca9a82b4d6906b43f2750d55315f5b10f}"

if [ -z "${HF_CACHE_DIR:-}" ]; then
    if [ -d "${HOME}/ai/hf" ]; then
        HF_CACHE_DIR="${HOME}/ai/hf"
    else
        HF_CACHE_DIR="${HOME}/.cache/huggingface"
    fi
fi
mkdir -p "${HF_CACHE_DIR}"

echo -e "${BLUE}==============================================================${NC}"
echo -e "${BLUE}  Pre-downloading Qwen3.8-27B-NVFP4 Model Weights             ${NC}"
echo -e "${BLUE}==============================================================${NC}"
echo -e "Model Name : ${MODEL_NAME}"
echo -e "Revision   : ${MODEL_REVISION}"
echo -e "Target Dir : ${HF_CACHE_DIR}"
echo ""

# If user has huggingface-cli installed locally
if command -v huggingface-cli >/dev/null 2>&1; then
    echo -e "${GREEN}[*] Using local huggingface-cli to download...${NC}"
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
        "${MODEL_NAME}" \
        --revision "${MODEL_REVISION}" \
        --cache-dir "${HF_CACHE_DIR}"
else
    echo -e "${GREEN}[*] Running lightweight download container to pull weights into ${HF_CACHE_DIR}...${NC}"
    docker run --rm -it \
        --volume "${HF_CACHE_DIR}:/root/.cache/huggingface" \
        --env HF_HOME=/root/.cache/huggingface \
        python:3.11-slim bash -c "
            pip install -q huggingface_hub && \
            python3 -c '
from huggingface_hub import snapshot_download
print(\"Downloading ${MODEL_NAME} at revision ${MODEL_REVISION}...\")
snapshot_download(repo_id=\"${MODEL_NAME}\", revision=\"${MODEL_REVISION}\")
print(\"Model download complete!\")
'
        "
fi

echo -e "${GREEN}[+] Model weights verified in ${HF_CACHE_DIR}!${NC}"
