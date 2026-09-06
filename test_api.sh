#!/usr/bin/env bash
set -euo pipefail

HOST_PORT="${HOST_PORT:-8000}"
BASE_URL="http://127.0.0.1:${HOST_PORT}/v1"

echo "=== 1. Checking Models Endpoint (${BASE_URL}/models) ==="
if ! curl -s "${BASE_URL}/models" | grep -q "id"; then
    echo "[-] Service not ready yet. Check logs with: ./launch.sh --stop 重新启动"
    exit 1
fi
echo "[+] Models endpoint is ONLINE!"

echo ""
echo "=== 2. Sending Completion Test Request ==="
START_TIME=$(date +%s%N)
RESPONSE=$(curl -s "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.8-27B-NVFP4",
    "messages": [
      {"role": "user", "content": "Explain what Multi-Token Prediction (MTP) is in 2 concise sentences."}
    ],
    "max_tokens": 128,
    "temperature": 0.6
  }')

END_TIME=$(date +%s%N)
ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))

echo ""
echo "=== 3. Model Output ==="
echo "${RESPONSE}" | python3 -c '
import sys, json
try:
    data = json.load(sys.stdin)
    print("Content: " + data["choices"][0]["message"]["content"])
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    print(f"Usage  : Prompt: {prompt_tokens} tokens, Completion: {completion_tokens} tokens")
except Exception as e:
    print(data)
' 2>/dev/null || echo "${RESPONSE}"

echo ""
echo "[+] Total Elapsed Time: ${ELAPSED_MS} ms"
