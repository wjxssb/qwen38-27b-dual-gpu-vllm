"""Passive host-load context for later benchmark interpretation; no process control."""
import json,time
from pathlib import Path

root=Path('/home/frank/dense-opt-repair-20260922/qualification/evidence')
with (root/'host-pressure.jsonl').open('x') as output:
    deadline=time.monotonic()+6*3600
    while time.monotonic()<deadline:
        output.write(json.dumps({'epoch':time.time(),'loadavg':Path('/proc/loadavg').read_text().strip(),
          'cpu_pressure':Path('/proc/pressure/cpu').read_text().strip(),
          'memory_pressure':Path('/proc/pressure/memory').read_text().strip(),
          'cpu_ticks':Path('/proc/stat').read_text().splitlines()[0]})+'\n');output.flush();time.sleep(1)
