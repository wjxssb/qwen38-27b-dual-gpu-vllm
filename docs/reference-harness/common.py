import hashlib, importlib.util, json, os, subprocess, time, urllib.request
from pathlib import Path

Q=Path('/home/frank/dense-opt-repair-20260922/qualification')
NV=Path('/home/frank/nvidia-dense-runtime')
BASE=NV/'candidate-v31-sm120-fp8-compat'
PROD=Path('/home/frank/local-inference-production')
UNITS=Path('/home/frank/.config/systemd/user')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def cmd(args,timeout=30):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=timeout).strip()
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def http(path,port=18080,timeout=10):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}',timeout=timeout) as r:return r.read()
def status():return json.loads(http('/switch/status'))
def active(candidate=BASE):return read(candidate/'state/graph-prefix/active.json')
def unit(name):return dict(line.split('=',1) for line in cmd(['systemctl','--user','show',name,'-p','MainPID','-p','ActiveState']).splitlines())
def identity(candidate=BASE):
    a=active(candidate);row=json.loads(cmd(['docker','inspect',a['container_id']]))[0]
    l=module('identity_launcher',candidate/'launcher.py');l.static();s=l.supervisor_module();s.exact_owned(a,row)
    assert a['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert unit('qwen27b.service')['MainPID']==str(a['pid'])
    assert s.proc_ticks(a['pid'])==a['supervisor_start_ticks'] and row['State']['Running']
    assert a['runtime_manifest_sha256']==sha(candidate/'MANIFEST.json')
    return a,row
def assert_idle(candidate=BASE):
    a,row=identity(candidate);st=status()
    assert st['state']=='ready' and st['active_requests']==0 and st['queued_requests']==0, 'WAITING_FOR_IDLE'
    metrics=http('/metrics',18096).decode()
    counts=[float(line.rsplit(' ',1)[1]) for line in metrics.splitlines() if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
    assert len(counts)==2 and not any(counts),'WAITING_FOR_IDLE'
    return a,row,st
def ticks(pid):
    try:
        raw=Path(f'/proc/{pid}/stat').read_text();return int(raw[raw.rfind(')')+2:].split()[19])
    except FileNotFoundError:return None
