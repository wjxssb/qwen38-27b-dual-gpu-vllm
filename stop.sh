#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-qwen38-27b-vllm}"

echo "[*] Stopping container '${CONTAINER_NAME}'..."
if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    echo "[+] Container '${CONTAINER_NAME}' stopped and removed."
else
    echo "[-] No container named '${CONTAINER_NAME}' found."
fi
