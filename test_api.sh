#!/usr/bin/env bash
set -euo pipefail
DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ ! -f "$DIR/runtime/graph006/api_smoke.py" ]; then
  echo "api_smoke not found: the local runtime/ state is not distributed with this repo." >&2
  echo "Run on the host that owns the runtime state (see README positioning note)." >&2
  exit 1
fi
exec /usr/bin/python3 -I "$DIR/runtime/graph006/api_smoke.py"
