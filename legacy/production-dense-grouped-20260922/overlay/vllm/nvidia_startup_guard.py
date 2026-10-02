"""Bounded CPU polling of startup failure receipts and local process death."""
import json
import os
from pathlib import Path
import time

STARTUP_METHODS = frozenset({'determine_available_memory', 'get_kv_cache_spec',
    'initialize_from_config', 'compile_or_warm_up_model', 'get_supported_tasks'})
POLL_SECONDS = 0.25


def raise_if_startup_failed(method, ranks):
    if method not in STARTUP_METHODS:
        return
    directory = os.environ.get('VLLM_STABILITY_TELEMETRY_DIR')
    if not directory:
        return
    root = Path(directory).parent
    for rank in ranks:
        role = f'worker-rank-{rank}'
        path = root/f'first-failure-{role}.json'
        try:
            with path.open('rb') as f:
                raw = f.read(65537)
        except FileNotFoundError:
            continue
        if len(raw) > 65536:
            raise RuntimeError('STARTUP_FAILURE_RECEIPT_TOO_LARGE')
        record = json.loads(raw)
        if record.get('process_role') != role:
            raise RuntimeError('STARTUP_FAILURE_RECEIPT_ROLE_MISMATCH')
        if (record.get('hard_stop') or record.get('fatal_profile') or
                record.get('method') in STARTUP_METHODS):
            raise RuntimeError(f'STARTUP_WORKER_FAILED rank={rank} method={method}: '
                               +str(record.get('message',''))[:512])


def wait_for_engine_ready(socket, manager, timeout_seconds):
    deadline = time.monotonic()+timeout_seconds
    while True:
        if manager is not None and (finished := manager.finished_procs()):
            raise RuntimeError(f'ENGINE_EXITED_BEFORE_READY: {finished}')
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            return False
        if socket.poll(timeout=max(1,int(min(POLL_SECONDS,remaining)*1000))):
            if manager is not None and (finished := manager.finished_procs()):
                raise RuntimeError(f'ENGINE_EXITED_BEFORE_READY: {finished}')
            return True
