#!/usr/bin/env python3
"""Run the HTTP quality gate, then emit the PASS/FAIL result file run_stage.py consumes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--result", type=Path, required=True)
parser.add_argument("--report-dir", type=Path, required=True)

raw = sys.argv[1:]
head, tail = raw, []
if "--" in raw:
    cut = raw.index("--")
    head, tail = raw[:cut], raw[cut + 1:]
if not tail:
    raise SystemExit("expected: -- <python> <quality_gate.py> [gate args...]")
args = parser.parse_args(head)
gate_argv = tail

proc = subprocess.run(gate_argv, stdin=subprocess.DEVNULL)
report = args.report_dir / "quality-report.json"
try:
    detail = json.loads(report.read_text())
except Exception:
    detail = {}
status = "PASS" if proc.returncode == 0 and detail.get("status") == "QUALITY_GATE_PASS" else "FAIL"
tmp = args.result.with_suffix(".tmp")
tmp.write_text(json.dumps({"status": status, "exit_code": proc.returncode}, indent=2) + "\n")
tmp.replace(args.result)
print(json.dumps({"quality": status}), flush=True)
sys.exit(0 if status == "PASS" else 1)
