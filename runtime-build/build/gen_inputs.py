#!/usr/bin/env python3
"""Generate the byte-pinned build input manifest for the public runtime image.

Every pin is a hard build input: the image refuses to build if any byte drifts.
flashinfer pins must equal the files actually shipped inside the pinned base
image (verified against the running container at pin time).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

VLLM_COMMIT = "99a10304dce8945119bd0b1a072297803c52a749"
FLASHINFER_COMMIT = "9dc1b2495b40314dec8a8cde8cd7faf5c5206702"
CUTLASS_COMMIT = "da5e086dab31d63815acafdac9a9c5893b1c69e2"
BASE_IMAGE_DIGEST = (
    "sha256:96a70bbb56acb6c4d22b3153b090ca322da927361c48a3129a1a258f4c702e73"
)

# sha256 of /usr/local/cuda-12.9/targets/x86_64-linux/lib/libnvrtc.so.12
# inside the pinned base image.
NVRTC_SHA256 = "c3430e8b6b0d8d3a74f5f839118a85e13e3ad6e4d6203ea1b6a27a0c0b4fe10a"

# flashinfer source files whose installed bytes inside the base image must
# equal the pinned git objects. mapping: repo-relative -> installed sha256
# (recorded from the base image container at pin time).
FLASHINFER_INSTALLED_SHA256 = {
    "flashinfer/nvfp4_attention_sm120.py": "07ef34538d2b6e275554b6d7af379857435c10ca9fc370881714c02fb56b5f52",
    "flashinfer/jit/nvfp4_attention_sm120.py": "bcae2efa6c36cc8aa4f0c2c8ccd803af3b0a55f8300e227157680c4e5296d115",
    "flashinfer/trace/templates/nvfp4_attention_sm120.py": "e257eac0ce18e327642749f89e3b7e14976d6c2cc30dd30ac4bf0a9c9fcd73bf",
    "flashinfer/utils.py": "912466a826fec0f0bdc15e7dd8caa02edb304374d2aade147f112ff09d058136",
    "flashinfer/page.py": "4a20f05410442ff415a44337c7c6246d90777e4c177852b2830fae3cf00eab04",
    "flashinfer/prefill.py": "7b83362805dabf5d6da80738ab8910be4d0ce60b5dadcbbec2ebdc64594f6525",
    "flashinfer/decode.py": "f67c5704d8ee1c7130b97adde8df8079faa42a11f37a08afa33ab26da71c1fe7",
}

VLLM_FILES = [
    "CMakeLists.txt",
    "csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu",
    "tests/kernels/attention/test_cache.py",
    "tests/kernels/attention/test_flashinfer_trtllm_attention.py",
    "tests/kernels/quantization/nvfp4_utils.py",
    "vllm/v1/attention/backends/flashinfer.py",
    "vllm/utils/torch_utils.py",
]

OVERLAY_FILES = [
    "vllm/distributed/device_communicators/shm_broadcast.py",
    "vllm/stability_telemetry.py",
    "vllm/v1/attention/backends/flashinfer.py",
    "vllm/v1/engine/async_llm.py",
    "vllm/v1/engine/core.py",
    "vllm/v1/executor/multiproc_executor.py",
    "vllm/v1/executor/tp_lifecycle.py",
    "vllm/v1/spec_decode/llm_base_proposer.py",
    "vllm/v1/spec_decode/sm120_mtp_decode_graph.py",
    "vllm/v1/worker/gpu_model_runner.py",
    "vllm/v1/worker/gpu_worker.py",
    "vllm/v1/worker/sm120_graph_validation.py",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    files: dict[str, dict[str, object]] = {}
    for rel in VLLM_FILES:
        path = ROOT / "checkouts/vllm" / rel
        files[f"checkouts/vllm/{rel}"] = {
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
    for rel in OVERLAY_FILES:
        path = ROOT / "overlay-graph006" / rel
        files[f"overlay-graph006/{rel}"] = {
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
    critical: dict[str, str] = {}
    for repo_rel, installed_sha in FLASHINFER_INSTALLED_SHA256.items():
        path = ROOT / "checkouts/flashinfer" / repo_rel
        actual = digest(path)
        if actual != installed_sha:
            print(
                f"flashinfer drift for {repo_rel}: checkout {actual} != "
                f"base-image {installed_sha}",
                file=sys.stderr,
            )
            return 1
        key = f"checkouts/flashinfer/{repo_rel}"
        files[key] = {"bytes": path.stat().st_size, "sha256": actual}
        critical[key] = repo_rel
    manifest = {
        "schema": 1,
        "vllm_commit": VLLM_COMMIT,
        "flashinfer_commit": FLASHINFER_COMMIT,
        "cutlass_commit": CUTLASS_COMMIT,
        "base_image_digest": BASE_IMAGE_DIGEST,
        "packages": {
            "flashinfer-python": "0.6.16.post3",
            "torch": "2.13.0+cu129",
        },
        "nvrtc_sha256": NVRTC_SHA256,
        "files": files,
        "flashinfer_critical_sources": critical,
    }
    out = ROOT / "build/inputs.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "INPUTS_GENERATED",
                "files": len(files),
                "manifest_sha256": digest(out),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
