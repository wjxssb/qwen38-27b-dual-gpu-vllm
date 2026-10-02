"""Temporary sealed release bindings with exact-file rollback; preserve baseline sources."""
import argparse, re, shutil
from common import *

OVERRIDE='99-dense-release.conf'
SERVICES=['qwen27b.service','qwen27b-relay.service','local-gpu-recovery.service']

def bindings(candidate):
    from cache_release import prepare
    cache=prepare(candidate)
    root=PROD/'runtime-bindings' if candidate.name.startswith('production-') else Q/'bindings'
    root.mkdir(exist_ok=True)
    revisions=read(Q/'control-source-revisions.json') if (Q/'control-source-revisions.json').exists() else {}
    control_files=revisions.get('files',{})
    for path,digest in control_files.items():assert sha(path)==digest,'Qualified control source drift'
    suffix='-control-'+sha(PROD/'scripts/recovery.py')[:12] if control_files else ''
    out=root/(candidate.name+suffix)
    if out.exists():
        receipt=read(out/'binding.json')
        assert receipt['manifest']==sha(candidate/'MANIFEST.json')
        for name,digest in receipt['files'].items():assert sha(out/name)==digest
        return out
    out.mkdir()
    original=read(Q/'evidence/frozen-baseline.json')
    for row in original['files']:
        if row['path'].startswith(str(PROD/'scripts')):assert sha(row['path'])==control_files.get(row['path'],row['sha256']),'Canonical control source drift'
    source=(PROD/'scripts/recovery.py').read_text()
    source=source.replace('ROOT=Path(__file__).resolve().parents[1]',f'ROOT=Path({str(PROD)!r})').replace(str(BASE),str(candidate)).replace(sha(BASE/'MANIFEST.json'),sha(candidate/'MANIFEST.json'))
    source=source.replace('/home/frank/nvidia-dense-runtime/nvidia-owned-stop.py',str(out/'owned-stop.py'))
    (out/'recovery.py').write_text(source)
    launcher=(PROD/'scripts/nvidia-launch.py').read_text()
    needle="BLESSED_CACHE_DIR = Path('/home/frank/nvidia-dense-runtime/cache-blessed')"
    assert launcher.count(needle)==1
    launcher=launcher.replace(needle,f'BLESSED_CACHE_DIR = Path({str(cache)!r})')
    (out/'nvidia-launch.py').write_text(launcher)
    for name,key in [('nvidia-owned-stop.py','MANIFEST'),('nvidia-p2p-gate.py','EXPECTED_MANIFEST')]:
        source=(NV/name).read_text();source=re.sub(r'^ROOT = .*$',f'ROOT = Path({str(NV)!r})',source,flags=re.M)
        source=re.sub(r'^CANDIDATE = .*$',f'CANDIDATE = Path({str(candidate)!r})',source,flags=re.M)
        source=re.sub(r'^'+key+r' = .*$',f'{key} = {sha(candidate/"MANIFEST.json")!r}',source,flags=re.M)
        (out/('owned-stop.py' if name=='nvidia-owned-stop.py' else 'p2p-gate.py')).write_text(source)
    for name in ['relay.mjs','relay-supervisor.mjs']:shutil.copy2(NV/'relay'/name,out/name)
    p=out/'relay.mjs';text=p.read_text();text,n=re.subn(r"statePath: '[^']+/state/graph-prefix/active.json'",f"statePath: '{candidate}/state/graph-prefix/active.json'",text);assert n==1;p.write_text(text)
    units={
      'qwen27b.service':f'[Service]\nExecStartPre=\nExecStartPre=/usr/bin/python3 -B -I {out}/p2p-gate.py\nExecStart=\nExecStart=/usr/bin/python3 -B -I {out}/nvidia-launch.py run --profile graph-prefix\nExecStop=\nExecStop=/usr/bin/python3 -B -I {out}/owned-stop.py\n',
      'qwen27b-relay.service':f'[Service]\nExecStart=\nExecStart=/home/frank/.local/share/node-v22.16.0-linux-x64/bin/node {out}/relay-supervisor.mjs\n',
      'local-gpu-recovery.service':f'[Service]\nExecStart=\nExecStart=/usr/bin/python3 -B -I {out}/recovery.py\n'}
    for unit,body in units.items():(out/(unit+'.conf')).write_text(body)
    for p in out.glob('*.py'):compile(p.read_text(),str(p),'exec')
    save(out/'binding.json',{'candidate':str(candidate),'manifest':sha(candidate/'MANIFEST.json'),'cache_directory':str(cache),'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'logic_changes':'none; only explicit root, release path, manifest digest, owned-stop path, and proven identical per-release cache bindings'})
    return out

def selected():
    p=Q/'current.json';return Path(read(p)['candidate']) if p.exists() else BASE

def accepted():
    p=Q/'canonical.json'
    if not p.exists():return BASE
    record=read(p);candidate=Path(record['candidate'])
    assert record['state']=='PROMOTED' and sha(candidate/'MANIFEST.json')==record['manifest']
    return candidate

def wait_unit(service,wanted,seconds):
    deadline=time.monotonic()+seconds
    while True:
        value=unit(service)
        if value['ActiveState'] in wanted:return value
        if time.monotonic()>deadline:raise TimeoutError(f'{service}: {value}')
        time.sleep(2)

def stop_current(current,record):
    # An unavailable HTTP endpoint may be stopped, but an identity mismatch
    # must never be interpreted as permission to stop a replacement owner.
    recovery=module('qualification_stop_owner',bindings(current)/'recovery.py' if current!=BASE else PROD/'scripts/recovery.py')
    owned=recovery.nv_intent()
    def owner_guard():
        if owned is not None:recovery.check_stop_owner(owned)
        elif int(unit('qwen27b.service')['MainPID']):raise RuntimeError('No exact owner proof for the active service')
    owner_guard()
    gpu_pids={int(line) for line in cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).splitlines() if line.strip()}
    process_pids=set()
    if owned is not None and owned[2]['State']['Running']:
        process_pids={int(line.strip()) for line in cmd(['docker','top',owned[1]['container_id'],'-eo','pid']).splitlines()[1:]}
    if gpu_pids-process_pids:raise RuntimeError('WAITING_FOR_GPU_IDLE: foreign GPU process present')
    try:assert_idle(current)
    except (AssertionError,OSError):
        # A failed test may already have lost its API. Never mistake an
        # observable active request from another client for an idle boundary.
        try:
            st=status()
            if st['active_requests'] or st['queued_requests']:raise RuntimeError('WAITING_FOR_IDLE')
        except urllib.error.URLError:pass
    cmd(['systemctl','--user','stop','--no-block','local-model-switch.service'])
    wait_unit('local-model-switch.service',{'inactive','failed'},90)
    owner_guard()
    cmd(['systemctl','--user','stop','--no-block','qwen27b.service'])
    try:wait_unit('qwen27b.service',{'inactive','failed'},180)
    except TimeoutError:
        # Exact current owner only, after an already expired graceful stop.
        recovery=module('bounded_stop',bindings(current)/'recovery.py' if current!=BASE else PROD/'scripts/recovery.py')
        import uuid
        run=PROD/'recovery/runs'/uuid.uuid4().hex;run.mkdir()
        # Keep the captured owner through escalation. Re-reading here could
        # authorize stopping a replacement that appeared during the wait.
        recovery.check_stop_owner(owned)
        recovery.stop_owned(run,owned=owned,deadline=time.monotonic())
        with (NV/'state/gpu.lock').open('r+') as lock:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);recovery.proof(run)
        record['escalation']=str(run)
    for service in ['local-gpu-recovery.service','qwen27b-relay.service']:
        cmd(['systemctl','--user','stop','--no-block',service]);wait_unit(service,{'inactive','failed'},60)
    assert not cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])

def write_bindings(candidate,current):
    old=bindings(current) if current!=BASE else None
    new=bindings(candidate) if candidate!=BASE else None
    for service in SERVICES:
        path=UNITS/(service+'.d')/OVERRIDE
        if path.exists():
            assert old is not None and path.read_bytes()==(old/(service+'.conf')).read_bytes(),'Unexpected override drift'
        elif old is not None:raise RuntimeError('Expected override missing')
    for service in SERVICES:
        path=UNITS/(service+'.d')/OVERRIDE
        if new:
            path.parent.mkdir(exist_ok=True);temp=path.with_suffix('.pending');temp.write_bytes((new/(service+'.conf')).read_bytes());temp.replace(path)
        elif path.exists():path.unlink()
    save(Q/'current.json',{'candidate':str(candidate),'manifest':sha(candidate/'MANIFEST.json'),'epoch':time.time()})
    cmd(['systemctl','--user','daemon-reload'])

def start(candidate,record):
    cmd(['systemctl','--user','reset-failed','qwen27b.service'])
    cmd(['systemctl','--user','start','local-gpu-recovery.service','qwen27b-relay.service'])
    cmd(['systemctl','--user','start','--no-block','local-model-switch.service'])
    deadline=time.monotonic()+1000
    while time.monotonic()<deadline:
        try:
            st=status()
            if st['state']=='ready':
                a,row=identity(candidate)
                if st['pid']==a['pid']:
                    record['ready_epoch']=time.time();record['identity']={key:a[key] for key in ['instance_id','container_id','pid','supervisor_start_ticks']};return
        except (OSError,ValueError,AssertionError,subprocess.SubprocessError):pass
        time.sleep(3)
    raise TimeoutError('Candidate failed readiness within 1000s')

def activate(candidate):
    current=selected();record={'from':str(current),'to':str(candidate),'started_epoch':time.time(),'state':'STOPPING'}
    output=Q/'evidence'/('transition-'+str(time.time_ns())+'.json');save(output,record)
    if candidate!=BASE:bindings(candidate)
    try:
        stop_current(current,record);write_bindings(candidate,current);record['state']='STARTING';save(output,record);start(candidate,record)
        record['state']='READY';save(output,record)
    except BaseException as error:
        record.update(state='FAILED_ROLLBACK_REQUIRED',error=repr(error));save(output,record)
        raise
    print(json.dumps(record),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','activate','restore']);p.add_argument('--candidate',type=Path,default=BASE);a=p.parse_args()
    if a.action=='prepare':print(bindings(a.candidate))
    else:activate(BASE if a.action=='restore' else a.candidate)
