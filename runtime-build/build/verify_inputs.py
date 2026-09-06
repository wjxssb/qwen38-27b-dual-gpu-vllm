"""Byte and ABI inputs checked before any native compilation or GPU use."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((ROOT / "build/inputs.json").read_text())
    for relative, pin in manifest["files"].items():
        path = ROOT / relative
        assert path.resolve() == path and path.is_file(), relative
        assert path.stat().st_size == pin["bytes"], relative
        assert digest(path) == pin["sha256"], relative
    for package, version in manifest["packages"].items():
        assert importlib.metadata.version(package) == version, package
    fi = importlib.metadata.distribution("flashinfer-python")
    for source, installed in manifest["flashinfer_critical_sources"].items():
        # Keep the already pinned package only if its actual Python/header
        # sources match the specified Git object. A version string is not proof.
        actual = Path(fi.locate_file(installed))
        assert actual.is_file(), str(actual)
        assert digest(actual) == manifest["files"][source]["sha256"], installed
    native = Path("/usr/local/cuda-12.9/targets/x86_64-linux/lib/libnvrtc.so.12")
    assert digest(native) == manifest["nvrtc_sha256"]
    import torch
    assert torch.__version__ == "2.13.0+cu129" and torch.version.cuda == "12.9"
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status="INPUTS_VERIFIED", files=len(manifest["files"]),
                          cuda_initialized=False)), flush=True)


if __name__ == "__main__":
    main()
