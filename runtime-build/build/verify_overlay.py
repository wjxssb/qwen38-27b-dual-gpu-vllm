"""Verify installed Graph006 overlay files equal the pinned build-context bytes."""
import hashlib
import json
from pathlib import Path
import sys

PKG = Path("/usr/local/lib/python3.12/dist-packages")
MANIFEST = Path("/opt/qwen38-runtime/build/inputs.json")
PREFIX = "overlay-graph006/"
EXPECTED = 12


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads(MANIFEST.read_text())
    checked = 0
    for key, pin in manifest["files"].items():
        if not key.startswith(PREFIX):
            continue
        rel = key[len(PREFIX):]
        installed = PKG / rel
        if not installed.is_file():
            raise SystemExit(f"overlay file not installed: {rel}")
        actual = digest(installed)
        if actual != pin["sha256"]:
            raise SystemExit(
                f"overlay drift: {rel} installed {actual} != pinned {pin['sha256']}"
            )
        checked += 1
    if checked != EXPECTED:
        raise SystemExit(f"overlay pin count {checked} != {EXPECTED}")
    print(f"OVERLAY_VERIFIED files={checked}")


if __name__ == "__main__":
    sys.exit(main())
