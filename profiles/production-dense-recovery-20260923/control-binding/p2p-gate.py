#!/usr/bin/python3
"""Pre-start P2P check bound to the sealed NVIDIA production workload."""
import hashlib
import argparse
import importlib.util
import json
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/frank/nvidia-dense-runtime')
CANDIDATE = Path('/home/frank/nvidia-dense-runtime/production-dense-recovery-20260923')
EXPECTED_MANIFEST = '053466c0ea4101f3c53bc22d4550e3474e3b3a431f99b62bda995737e17350ef'


def validate_environment(actual, required):
    if any(actual.get(k) != v for k, v in required.items()):
        raise RuntimeError('Sealed NVIDIA NCCL environment differs from P2P proof')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt-only', action='store_true')
    parser.add_argument('--wait', type=int, default=0)
    args = parser.parse_args()
    raw = (CANDIDATE / 'MANIFEST.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_MANIFEST:
        raise RuntimeError('NVIDIA workload manifest differs')
    manifest = json.loads(raw)
    source = CANDIDATE / 'launcher.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['files']['launcher.py']['sha256']:
        raise RuntimeError('NVIDIA launcher differs')
    spec = importlib.util.spec_from_file_location('nvidia_launcher', source)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    launcher.static()
    required = launcher.read(Path('/home/frank/p2p-stable/manifest/STABLE-P2P-MANIFEST.json'))['nccl_production_env']
    validate_environment(launcher.read(CANDIDATE / 'environment.json'), required)
    if args.receipt_only:
        deadline = time.monotonic() + args.wait
        while True:
            try:
                receipt = launcher.verify_p2p()
                break
            except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        print(json.dumps({'status': 'CURRENT_BOOT_NVIDIA_P2P_RECEIPT_PASS', 'boot_id': receipt['boot_id'],
                          'manifest_sha256': EXPECTED_MANIFEST, 'CUDA_probe_run': False}))
        return
    with launcher.leases():
        receipt = launcher.verify_p2p()
        if launcher.command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']):
            raise RuntimeError('GPU owner present before P2P pre-start probe')
        launcher.validate_gpu_rows(launcher.command(['nvidia-smi', '--query-gpu=uuid,memory.free,memory.used,utilization.gpu,gpu_recovery_action', '--format=csv,noheader,nounits']))
        peer = launcher.command(['/home/frank/p2p-stable/scripts/cuda_peer_probe'])
    print(json.dumps({'status': 'CURRENT_BOOT_NVIDIA_P2P_GATE_PASS', 'manifest_sha256': EXPECTED_MANIFEST,
                      'boot_id': receipt['boot_id'], 'peer_probe': peer}))


if __name__ == '__main__':
    main()
