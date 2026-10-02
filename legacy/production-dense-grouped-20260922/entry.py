#!/usr/bin/python3
import hashlib,json,os,sys
from pathlib import Path
for row in json.loads(Path('/candidate/overlay-integrity.json').read_text())+json.loads(Path('/candidate/inspection-integrity.json').read_text()):
 p=Path(row['target']);raw=p.read_bytes()
 if p.is_symlink() or len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise RuntimeError('candidate overlay hash mismatch: '+str(p))
if sys.argv[1:4]!=['/usr/bin/python3','-m','vllm.entrypoints.openai.api_server']:raise RuntimeError('entrypoint mismatch')
print('NVIDIA_CANDIDATE_OVERLAY_INTEGRITY_PASS',flush=True)
os.execv(sys.argv[1],[sys.argv[1],'-I',*sys.argv[2:]])
