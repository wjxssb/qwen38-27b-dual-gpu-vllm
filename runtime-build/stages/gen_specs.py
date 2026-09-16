#!/usr/bin/env python3
"""Generate immutable stage specs (and a fresh kernel baseline) for run_stage.py."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
QUALIFICATION = ROOT.parent / "qualification"
GATE = QUALIFICATION / "quality_gate.py"
MODEL_HOST = Path("/home/frank/ai/hf/hub")
GPU_UUIDS = [
    "GPU-0506b796-f8ea-b40c-616d-5e9d43a9e175",
    "GPU-8897f327-ceb9-4fab-b84e-d34b24f0b584",
]
FAULT = re.compile(
    r"NVRM: Xid|oom-kill:|Out of memory:|Memory cgroup out of memory:|"
    r"fallen off the bus|EXT4-fs error|I/O error, dev",
    re.I,
)

STAGES = {
    "stage-a": {
        "manifest": HERE / "manifest-stage-a.json",
        "mode": "eager-baseline",
        "port": 18001,
    },
    "stage-b": {
        "manifest": HERE / "manifest-stage-b.json",
        "mode": "mtp3-graph",
        "port": 18002,
    },
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kernel_baseline() -> dict:
    rows = subprocess.run(
        ["journalctl", "-k", "-b", "--no-pager", "--output=short-iso"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    faults = [row for row in rows if FAULT.search(row)]
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return {"boot_id": boot, "acknowledged_kernel_faults": faults}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--stages", nargs="+", choices=sorted(STAGES), required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_id):
        parser.error("--image-id must be a sha256 image ID")
    contract = json.loads((ROOT / "build/release.json").read_text())
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    baseline_path = HERE / "kernel-baseline.json"
    baseline_path.write_text(json.dumps(kernel_baseline(), indent=2) + "\n")
    for name in args.stages:
        stage = STAGES[name]
        manifest = stage["manifest"]
        run_id = f"qwen38-pub-{name.replace('-', '')}-{stamp}"
        run_dir = HERE / "runs" / run_id
        profile = contract["modes"][stage["mode"]]
        port = stage["port"]
        create = [
            "docker", "create",
            "--name", run_id,
            "--restart", "no",
            "--init",
            "--network", "bridge",
            "--ipc", "private",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--cpus", str(profile["cpus"]),
            "--memory", str(profile["host_memory_bytes"]),
            "--memory-swap", str(profile["host_memory_bytes"]),
            "--shm-size", str(profile["shm_bytes"]),
            "--ulimit", "memlock=" + str(profile["pinned_memory_bytes"]),
            "--pids-limit", "1024",
            "--stop-timeout", "30",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=268435456",
            "--gpus", '"device=' + ",".join(GPU_UUIDS) + '"',
            "--publish", f"127.0.0.1:{port}:8000",
            "--label", "io.qwen38.qualification=" + run_id,
            "--mount", f"type=bind,src={MODEL_HOST},dst=/models/hub,readonly",
            "--mount", f"type=bind,src={run_dir}/cache,dst=/cache",
            "--mount", f"type=bind,src={run_dir}/results,dst=/results",
            "--workdir", "/results",
        ]
        env = dict(profile["environment"])
        env.update({
            "HOME": "/cache/home", "XDG_CACHE_HOME": "/cache/xdg",
            "HF_HOME": "/cache/huggingface", "CUDA_CACHE_PATH": "/cache/cuda",
            "TRITON_CACHE_DIR": "/cache/triton", "TORCH_EXTENSIONS_DIR": "/cache/torch",
            "NVIDIA_VISIBLE_DEVICES": ",".join(GPU_UUIDS),
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        })
        for key in sorted(env):
            create += ["--env", f"{key}={env[key]}"]
        create += ["--entrypoint", profile["argv"][0], args.image_id, *profile["argv"][1:]]
        spec = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "image": args.image_id,
            "gpu_uuids": GPU_UUIDS,
            "gpu_locks": [f"/tmp/qwen38-runtime-gpu-locks-v1/{uid}.lock" for uid in GPU_UUIDS],
            "required_available_bytes": 30000000000,
            "kernel_baseline": str(baseline_path),
            "file_hashes": {
                str(manifest): sha(manifest),
                str(GATE): sha(GATE),
                str(HERE / "run_quality.py"): sha(HERE / "run_quality.py"),
                str(ROOT / "build/release.json"): sha(ROOT / "build/release.json"),
            },
            "docker_create": create,
            "create_dirs": [str(run_dir / "cache"), str(run_dir / "results")],
            "health_url": f"http://127.0.0.1:{port}/health",
            "startup_timeout_seconds": profile["startup_timeout_s"],
            "deadline_seconds": 9000,
            "client_argv": [
                "/usr/bin/python3", str(HERE / "run_quality.py"),
                "--result", str(run_dir / "quality-result.json"),
                "--report-dir", str(run_dir / "quality-report"),
                "--", "/usr/bin/python3", str(GATE),
                "--base-url", f"http://127.0.0.1:{port}",
                "--model", contract["model"]["repo_id"],
                "--stage-id", json.loads(manifest.read_text())["stage_id"],
                "--stage-manifest", str(manifest),
                "--output", str(run_dir / "quality-report"),
                "--windows", "8192,65536,131072,196608,262144",
                "--repeats", "2",
                "--max-tokens", "512",
                "--timeout", "1800",
                "--deadline", "6000",
                "--seed", "dense-public-quality-v1",
            ],
            "quality_result": str(run_dir / "quality-result.json"),
        }
        out = HERE / f"{name}-spec.json"
        out.write_text(json.dumps(spec, indent=2) + "\n")
        print(json.dumps({"stage": name, "spec": str(out), "run_id": run_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
