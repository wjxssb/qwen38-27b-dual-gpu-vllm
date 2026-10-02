#!/usr/bin/python3
"""Request the sealed supervisor's stop; reconcile one lost startup SIGTERM."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path('/home/frank/nvidia-dense-runtime')
CANDIDATE = Path('/home/frank/nvidia-dense-runtime/production-dense-recovery-20260923')
MANIFEST = '053466c0ea4101f3c53bc22d4550e3474e3b3a431f99b62bda995737e17350ef'
DOCKER = ['/usr/bin/docker', '--host', 'unix:///var/run/docker.sock']


def retry_eligible(intent, elapsed, already_recorded):
    return (elapsed >= 120 and not already_recorded
            and intent.get('stop_dispatched') is True
            and not intent.get('cleanup_verified')
            and not intent.get('hardware_fault_latched'))


def http(port, path):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=5) as response:
        return response.status, response.read()


def main():
    raw = (CANDIDATE / 'MANIFEST.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST:
        raise RuntimeError('Stop refuses changed NVIDIA manifest')
    manifest = json.loads(raw)
    path = CANDIDATE / 'launcher.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['files']['launcher.py']['sha256']:
        raise RuntimeError('Stop refuses changed launcher')
    spec = importlib.util.spec_from_file_location('owned_launcher', path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    launcher.static()
    request = launcher.own_stop('graph-prefix')
    run = CANDIDATE / 'runs' / request['instance_id']
    supervisor = launcher.supervisor_module()
    receipt = run / 'startup-stop-graceful-retry.json'
    started = time.monotonic()
    while True:
        intent = launcher.read(run / 'intent.json')
        if intent.get('cleanup_verified'):
            print(json.dumps({'state': 'OWNED_STOP_CLEANUP_VERIFIED', 'instance_id': request['instance_id']}))
            return
        if retry_eligible(intent, time.monotonic() - started, receipt.exists()):
            # Only the original live supervisor may own this exact container.
            pid = intent['pid']
            main_pid = launcher.command(['systemctl', '--user', 'show', 'qwen27b.service', '-p', 'MainPID', '--value'])
            if main_pid != str(pid) or supervisor.proc_ticks(pid) != intent['supervisor_start_ticks']:
                raise RuntimeError('Supervisor identity changed; graceful retry refused')
            row = json.loads(launcher.command(DOCKER + ['inspect', intent['container_id']]))[0]
            supervisor.exact_owned(intent, row)
            if row['State']['Running']:
                try:
                    ready = http(intent['port'], '/health')[0] == 200
                    metrics = http(intent['port'], '/metrics')[1].decode() if ready else ''
                    counts = [s for s in metrics.splitlines() if s.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
                    quiet = len(counts) >= 2 and all(float(s.rsplit(' ', 1)[-1]) == 0 for s in counts)
                except (OSError, ValueError):
                    quiet = False
                if quiet:
                    # A signal received before uvicorn installed its handlers can
                    # be consumed by initialization. Retry once after readiness.
                    # Keep the original pending docker stop and its supervisor.
                    supervisor.durable(receipt, {'state': 'DISPATCHING_GRACEFUL_SIGTERM', 'epoch': time.time(),
                        'instance_id': intent['instance_id'], 'container_id': intent['container_id'],
                        'signal': 'SIGTERM', 'force_kill': False, 'active_requests': 0})
                    result = subprocess.run(DOCKER + ['kill', '--signal=SIGTERM', intent['container_id']], capture_output=True, text=True, timeout=20)
                    supervisor.durable(receipt, {'state': 'GRACEFUL_RETRY_RETURNED_EXIT_PENDING', 'epoch': time.time(),
                        'instance_id': intent['instance_id'], 'container_id': intent['container_id'],
                        'signal': 'SIGTERM', 'force_kill': False, 'returncode': result.returncode,
                        'stdout': result.stdout, 'stderr': result.stderr})
        time.sleep(2)


if __name__ == '__main__':
    main()
