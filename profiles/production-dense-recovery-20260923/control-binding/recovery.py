"""Single recovery worker. Never starts a model; Gateway remains sole scheduler.
Original NVIDIA payload and incident records are immutable. Recovery proof admits
only an exactly exited run after fresh CUDA/P2P checks and a quiet 60s interval.
"""
import sys,os,json,time,signal,subprocess,fcntl,uuid,hashlib,importlib.util,re,urllib.request
from pathlib import Path
ROOT=Path('/home/frank/local-inference-production');STATE=ROOT/'gateway/state';NV=Path('/home/frank/nvidia-dense-runtime/production-dense-recovery-20260923')
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
progress=module('production_rpc_progress',ROOT/'scripts/rpc_progress.py');forensics=module('production_wedge_forensics',ROOT/'scripts/wedge_forensics.py')
h=module('production_hardware',ROOT/'scripts/hardware.py');l=module('production_frozen_launcher',NV/'launcher.py');s=l.supervisor_module();r=module('frozen_recovery_validator',NV/'restart_app_recovery.py')
UNITS=['qwen27b.service']
FAST_FAULT_REASONS={'OWNED_NVIDIA_RPC_STALLED_90S','OWNED_NVIDIA_TP_FAILED'}
FAULT_STOP_GRACE_S=15
FAULT_FORENSICS_BUDGET_S=8
def cmd(args,timeout=25):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=timeout).strip()
def status(state,**kw):
    # A retry must retain the failed instance, including across controller
    # restarts. Otherwise INCOMPLETE_RECOVERY could stop its healthy successor.
    if state in {'RECOVERING','RETRY_WAIT'} and 'recovery_owner' not in kw:
        previous=h.read(STATE/'recovery-status.json',{})
        if previous.get('boot_id')==h.boot() and previous.get('state') in {'RECOVERING','RETRY_WAIT'}:
            if previous.get('recovery_owner'):kw['recovery_owner']=previous['recovery_owner']
    h.save(STATE/'recovery-status.json',dict(state=state,boot_id=h.boot(),epoch=time.time(),**kw))
def unit(name):
    raw=cmd(['systemctl','--user','show',name,'-p','MainPID','-p','ActiveState']);return dict(x.split('=',1) for x in raw.splitlines())
def ticks(pid):
    try:return s.proc_ticks(pid)
    except (OSError,ProcessLookupError):return None

def signal_exact(pid,start,sig):
    if start is None:return
    if ticks(pid)==start:
        # pidfd prevents PID reuse between verification and signal delivery.
        try:fd=os.pidfd_open(pid)
        except ProcessLookupError:return
        try:
            if ticks(pid)!=start:raise RuntimeError('PID identity changed')
            signal.pidfd_send_signal(fd,sig)
        finally:os.close(fd)

def nv_intent():
    active=h.read(NV/'state/graph-prefix/active.json',{})
    if not active:return None
    path=NV/'runs'/active['instance_id']/'intent.json';intent=h.read(path)
    if intent['boot_id']!=h.boot():return None
    if intent['runtime_manifest_sha256']!='053466c0ea4101f3c53bc22d4550e3474e3b3a431f99b62bda995737e17350ef':raise RuntimeError('Unexpected NVIDIA manifest')
    row=json.loads(cmd(l.DOCKER+['inspect',intent['container_id']]))[0];s.exact_owned(intent,row)
    return path,intent,row

def stopped(row):
    st=row['State'];return s.exited_after_start(row) and st['Pid']==0 and not st.get('Paused') and not st.get('Restarting')

def same_owner(left,right):
    keys=('boot_id','instance_id','container_id','pid','supervisor_start_ticks')
    return all(left.get(key) is not None and left.get(key)==right.get(key) for key in keys)

class RecoveryOwnerChanged(RuntimeError):pass

def check_stop_owner(owned):
    """Never let an old recovery attempt stop a replacement service instance."""
    if owned is None:return
    current=nv_intent()
    if current is None or not same_owner(owned[1],current[1]):raise RecoveryOwnerChanged('Recovery owner changed')
    service=unit('qwen27b.service');pid=int(service['MainPID'])
    if pid not in (0,owned[1]['pid']):raise RecoveryOwnerChanged('Recovery service PID changed')
    current_ticks=ticks(pid) if pid else None
    if pid and current_ticks!=owned[1]['supervisor_start_ticks']:
        if current_ticks is not None:raise RecoveryOwnerChanged('Recovery supervisor identity changed')
        # systemd may still show our just-killed supervisor until it reaps it.
        # Absence is acceptable only after the exact old container has exited.
        if not stopped(current[2]):raise RuntimeError('Supervisor absent before owned container exit')

def stop_cpu_waiter(owned):
    command=['systemctl','--user','show','qwen27b.service','-p','ControlPID','--value']
    pid=int(cmd(command))
    if pid<=1:return
    start=ticks(pid)
    if start is None:return
    try:args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,ProcessLookupError):return
    if b'/home/frank/local-inference-production/runtime-bindings/production-dense-recovery-20260923-control-4767c3b893fe/owned-stop.py' not in args:return
    check_stop_owner(owned)
    if int(cmd(command))!=pid:return
    signal_exact(pid,start,signal.SIGKILL)

def stop_owned(run,*,owned=None,deadline=None):
    owned=owned or nv_intent()
    check_stop_owner(owned)
    for name in UNITS:cmd(['systemctl','--user','stop','--no-block',name])
    if deadline is None:deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        check_stop_owner(owned)
        if all(unit(n)['ActiveState'] in ['inactive','failed'] for n in UNITS):
            current=nv_intent()
            nv_gone=not current or stopped(current[2])
            if nv_gone:return
            break
        time.sleep(2)
    # Only a positively identified crashed/hung owned backend may escalate.
    if owned:
        check_stop_owner(owned)
        path,intent,row=owned;fresh=json.loads(cmd(l.DOCKER+['inspect',intent['container_id']]))[0];s.exact_owned(intent,fresh)
        if fresh['State']['Running']:
            h.save(run/'nvidia-force-stop.json',{'instance_id':intent['instance_id'],'container_id':intent['container_id'],'signal':'SIGKILL','epoch':time.time()})
            cmd(l.DOCKER+['kill','--signal=SIGKILL',intent['container_id']])
        fresh=json.loads(cmd(l.DOCKER+['inspect',intent['container_id']]))[0];s.exact_owned(intent,fresh)
        if not stopped(fresh):raise RuntimeError('Owned NVIDIA container exit not proven')
        if ticks(intent['pid'])==intent['supervisor_start_ticks']:
            # Original supervisor intentionally quarantines after any Xid.
            # Stop only this CPU supervisor after exact container exit proof.
            signal_exact(intent['pid'],intent['supervisor_start_ticks'],signal.SIGKILL)
    # Stop lingering original ExecStop CPU waiter only once the GPU owner exited.
    if owned:
        check_stop_owner(owned)
        path,intent,_=owned
        row=json.loads(cmd(l.DOCKER+['inspect',intent['container_id']]))[0];s.exact_owned(intent,row)
        if not stopped(row):raise RuntimeError('NVIDIA exit changed')
        stop_cpu_waiter(owned)
    deadline=time.monotonic()+100
    while any(unit(n)['ActiveState'] not in ['inactive','failed'] for n in UNITS):
        check_stop_owner(owned)
        if time.monotonic()>deadline:raise RuntimeError('Owned service still stopping; retry later')
        time.sleep(2)

def observation_idle():
    ob=h.observe()
    if len(ob['gpus'])!=2 or {x['uuid'] for x in ob['gpus']}!=h.UUIDS:raise RuntimeError('GPU pair unavailable; retry later')
    if any(x['recovery'].lower()!='none' or int(x['used'])>128 for x in ob['gpus']):raise RuntimeError('GPU idle/recovery not clear')
    if cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']):raise RuntimeError('GPU owner still present')
    return ob

def pci_identity():
    rows=[]
    for line in cmd(['nvidia-smi','--query-gpu=uuid,pci.bus_id,driver_version','--format=csv,noheader,nounits']).splitlines():
        gpu,bus,driver=[v.strip() for v in line.split(',')]
        domain,device,function=bus.lower().split(':');bus=f'{int(domain,16):04x}:{device}:{function}'
        path=Path('/sys/bus/pci/devices')/bus
        row={'uuid':gpu,'bus':bus,'driver_version':driver,'vendor':(path/'vendor').read_text().strip(),'device':(path/'device').read_text().strip(),'driver':(path/'driver').resolve().name,'link_width':int((path/'current_link_width').read_text()),'link_speed':(path/'current_link_speed').read_text().strip()}
        with (path/'config').open('rb') as config:row['config_vendor']=config.read(2).hex()
        if row['vendor']!='0x10de' or row['config_vendor']!='de10' or row['driver']!='nvidia' or row['link_width']<=0 or float(row['link_speed'].split()[0])<=0:raise RuntimeError('PCI identity/link inaccessible')
        rows.append(row)
    if len(rows)!=2 or {v['uuid'] for v in rows}!=h.UUIDS:raise RuntimeError('PCI pair differs')
    return rows

def same_pci_identity(before,after):
    # Current link speed is power-managed; idle 32 -> 2.5 GT/s is not a
    # replaced/disconnected device. Retain UUID/BDF/driver/config/width checks.
    def stable(rows):
        for row in rows:
            if float(row['link_speed'].split()[0])<=0:raise RuntimeError('PCI link inaccessible')
        return [{key:value for key,value in row.items() if key!='link_speed'} for row in rows]
    return stable(before)==stable(after)

def proof(run):
    first=observation_idle();before=first['journal'];pci=pci_identity();h.save(run/'pci-before.json',pci)
    kernel=cmd(['journalctl','-k','-b','--show-cursor','--no-pager']);cursor=re.findall(r'-- cursor: (.+)',kernel)[-1]
    (run/'kernel-before.log').write_text(kernel)
    if re.search(r'Xid[^\n]*:\s*8\b',before):time.sleep(30)
    started=time.time();env=os.environ.copy();(ROOT/'recovery/jit').mkdir(parents=True,exist_ok=True);(ROOT/'recovery/tmp').mkdir(parents=True,exist_ok=True);env['CUDA_CACHE_PATH']=str(ROOT/'recovery/jit');env['TMPDIR']=str(ROOT/'recovery/tmp')
    probe=subprocess.run(['/usr/bin/python3','-B','-I',str(NV/'cuda_recovery_probe.py')],env=env,text=True,capture_output=True,timeout=120)
    h.save(run/'cuda-probe.json',dict(returncode=probe.returncode,stdout=probe.stdout,stderr=probe.stderr,source_sha256=l.sha(NV/'cuda_recovery_probe.py'),started_epoch=started,finished_epoch=time.time()))
    if probe.returncode:raise RuntimeError('Fresh CUDA/P2P probe failed')
    r.validate_probe(json.loads(probe.stdout),l.GPU_UUIDS)
    samples=[];begin=time.monotonic()
    while True:
        ob=observation_idle();new=cmd(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'])
        if s.KERNEL_ERRORS.search(new):raise RuntimeError('New fault during recovery proof')
        samples.append({'monotonic':time.monotonic(),'gpus':ob['gpus']})
        if time.monotonic()-begin>=60:break
        time.sleep(10)
    after_pci=pci_identity();h.save(run/'pci-after.json',after_pci)
    if not same_pci_identity(pci,after_pci):raise RuntimeError('PCI identity/link changed during proof')
    p2p=l.verify_p2p();h.save(run/'idle-proof.json',dict(samples=samples,elapsed_seconds=time.monotonic()-begin,p2p=p2p))
    kernel=cmd(['journalctl','-k','-b','--show-cursor','--no-pager']);(run/'kernel-after.log').write_text(kernel)
    cursor=re.findall(r'-- cursor: (.+)',kernel)[-1]
    owned=nv_intent()
    receipts=[]
    if owned:
        path,intent,row=owned
        if not stopped(row) or ticks(intent['pid'])==intent['supervisor_start_ticks']:raise RuntimeError('Old NVIDIA lifecycle not conclusively stopped')
        h.save(run/'old-intent.json',intent);h.save(run/'old-container.json',row)
        receipt={'boot_id':h.boot(),'state':'PASS','old_intent_sha256':l.sha(path),'container_id':intent['container_id'],'container_finished_at':row['State']['FinishedAt'],'run':str(run),'kernel_cursor':cursor,'epoch':time.time(),'old_incident_preserved':True,'model_qualification':'NOT_PROVEN'}
        receipts.append((STATE/'reconciled'/f"{intent['instance_id']}.json",receipt))
    seal={p.name:l.sha(p) for p in run.iterdir() if p.is_file()};h.save(run/'proof-manifest.json',seal)
    for path,receipt in receipts:
        path.parent.mkdir(exist_ok=True);receipt['proof_manifest_sha256']=l.sha(run/'proof-manifest.json');h.save(path,receipt)
    h.save(STATE/'hardware-cursor.json',{'boot_id':h.boot(),'cursor':cursor})
    status('RECOVERED',run=str(run),probe='PASS',observation_seconds=60)

def validate_receipt(intent,path):
    receipt=h.read(STATE/'reconciled'/f"{intent['instance_id']}.json",{})
    if receipt.get('state')!='PASS' or receipt.get('boot_id')!=h.boot() or receipt.get('old_intent_sha256')!=l.sha(path):return False
    run=Path(receipt['run']);manifest=run/'proof-manifest.json'
    if not run.resolve().is_relative_to((ROOT/'recovery/runs').resolve()) or l.sha(manifest)!=receipt['proof_manifest_sha256']:raise RuntimeError('Recovery proof path/digest changed')
    for name,digest in h.read(manifest).items():
        if Path(name).name!=name or l.sha(run/name)!=digest:raise RuntimeError('Recovery proof artifact changed')
    evidence=h.read(run/'cuda-probe.json');r.validate_probe(json.loads(evidence['stdout']),l.GPU_UUIDS)
    samples=h.read(run/'idle-proof.json')['samples']
    if len(samples)<3 or samples[-1]['monotonic']-samples[0]['monotonic']<60:raise RuntimeError('Recovery idle proof too short')
    fresh=json.loads(cmd(l.DOCKER+['inspect',intent['container_id']]))[0];s.exact_owned(intent,fresh)
    if not stopped(fresh) or fresh['State']['FinishedAt']!=receipt['container_finished_at'] or ticks(intent['pid'])==intent['supervisor_start_ticks']:raise RuntimeError('Reconciled NVIDIA identity changed')
    return True

def reset_if_required(ob,run):
    if ob['state']=='GPU_RESET_REQUIRED':
        if cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']):raise RuntimeError('Foreign GPU process blocks reset')
        args=['sudo','-n','/usr/bin/nvidia-smi','--gpu-reset','-i',','.join(sorted(h.UUIDS))]
        reset=subprocess.run(args,capture_output=True,text=True,timeout=90);h.save(run/'gpu-reset.json',dict(argv=args,returncode=reset.returncode,stdout=reset.stdout,stderr=reset.stderr))
        if reset.returncode:raise RuntimeError('Driver reset not available yet; retry with backoff')

def cycle(reason):
    owned=None
    pending=h.read(STATE/'recovery-status.json',{})
    bound=pending.get('recovery_owner') if pending.get('boot_id')==h.boot() and pending.get('state') in {'RECOVERING','RETRY_WAIT'} else None
    if bound:
        owned=nv_intent()
        if not owned or not same_owner(bound,owned[1]):raise RecoveryOwnerChanged('Recovery owner changed before retry')
        check_stop_owner(owned)
    if reason in FAST_FAULT_REASONS:
        # Re-read progress/failure evidence before publishing a drain. A worker
        # may have resumed, or Gateway may have admitted a replacement instance.
        fault=owned_rpc_stall()
        if not fault or fault['reason']!=reason:return
        owned=nv_intent()
        if not owned or not same_owner(fault,owned[1]):return
        check_stop_owner(owned)
    root=ROOT/'recovery/runs';root.mkdir(parents=True,exist_ok=True);run=root/uuid.uuid4().hex;run.mkdir()
    owner_fields={key:owned[1][key] for key in ('boot_id','instance_id','container_id','pid','supervisor_start_ticks')} if owned else None
    status('RECOVERING',run=str(run),reason=reason,recovery_owner=owner_fields);h.save(run/'incident.json',dict(h.observe(),controller_reason=reason))
    if reason in FAST_FAULT_REASONS:h.save(run/'rpc-stall.json',fault)
    # Dispatch stop before capture. For TP faults both share one short grace
    # window; ordinary hardware recovery retains its established policy.
    if owned:check_stop_owner(owned)
    stop_started=time.monotonic()
    for name in UNITS:cmd(['systemctl','--user','stop','--no-block',name])
    try:
        if owned:
            # Share the grace window with capture, instead of adding up to 100s
            # of forensics before starting another 120s graceful-stop wait.
            forensics.capture(run,reason,budget_s=FAULT_FORENSICS_BUDGET_S)
        else:forensics.capture(run,reason)
    except Exception as exc:print(json.dumps({'forensics_error':str(exc),'epoch':time.time()}),flush=True)
    if owned:stop_owned(run,owned=owned,deadline=stop_started+FAULT_STOP_GRACE_S)
    else:stop_owned(run)
    with Path('/home/frank/nvidia-dense-runtime/state/gpu.lock').open('r+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if owned:check_stop_owner(owned)
        ob=h.observe()
        if ob['state']=='REBOOT_REQUIRED':raise RuntimeError('REBOOT_REQUIRED')
        reset_if_required(ob,run)
        proof(run)

TRANSIENT_SUSTAIN_S=30
SELF_HEALING_REASONS={'GPU_UNAVAILABLE_RETRY','OWNED_NVIDIA_QUARANTINE'}
def sustained(reason,since,now):
    """Self-healing transients must never kill a healthy startup on first sight.
    GPU_UNAVAILABLE_RETRY is a single failed nvidia-smi/journal query.
    OWNED_NVIDIA_QUARANTINE is the supervisor's own reconcile window, which it
    clears itself within QUARANTINE_MAX_RECHECK_SECONDS=30. Both act only after
    ~30s of consecutive ticks; every other state still cycles immediately."""
    if reason in (None, 'HEALTHY'):
        return None, None
    if reason in SELF_HEALING_REASONS:
        since = since or now
        return (reason if now - since >= TRANSIENT_SUSTAIN_S else None), since
    return reason, None
def admitted_backend_health():
    active=h.read(STATE/'active-model.json',{})
    if active.get('boot_id')!=h.boot() or active.get('service') not in UNITS:return None
    current=unit(active['service'])
    if current['ActiveState']!='active' or int(current['MainPID'])!=active.get('pid'):return None
    expected={'qwen27b.service':'http://127.0.0.1:18082'}[active['service']]
    if active.get('backend')!=expected:raise RuntimeError('Admitted backend endpoint differs')
    try:
        with urllib.request.urlopen(expected+'/health',timeout=8) as response:return response.status==200
    except Exception:return False

def fatal_tp_marker(directory,supervisor_start_ticks,now_ns):
    """Immutable first-fault markers bypass the stall timer, never old logs.

    A unique run directory plus its boot/supervisor/container checks binds this
    marker even when the crashed child's container-namespace PID no longer exists.
    """
    allowed={'engine-core':{'TP_WORKER_EXIT','TP_LIFECYCLE_FAIL_CLOSED','ENGINE_CORE_EXCEPTION'},
             'worker-rank-0':{'WORKER_RPC_FAILED','ASYNC_OUTPUT_THREAD_FAILED'},
             'worker-rank-1':{'WORKER_RPC_FAILED','ASYNC_OUTPUT_THREAD_FAILED'}}
    start_ns=int(supervisor_start_ticks)*1_000_000_000//os.sysconf('SC_CLK_TCK')
    for role,reasons in allowed.items():
        path=directory/f'first-failure-{role}.json'
        try:
            with path.open('rb') as stream:raw=stream.read(16385)
            if len(raw)>16384:continue
            row=json.loads(raw)
            if not isinstance(row,dict):continue
            stamp=row.get('monotonic_ns')
            if (row.get('schema_version')!=1 or row.get('process_role')!=role or
                    row.get('reason') not in reasons or type(stamp) is not int or
                    not start_ns<=stamp<=now_ns or type(row.get('pid')) is not int or row['pid']<=0):continue
            return {'reason':'OWNED_NVIDIA_TP_FAILED','failure_marker':str(path),'failure':row}
        except (OSError,ValueError,TypeError):continue
    return None

def owned_rpc_stall():
    """Bind unfinished worker telemetry to the currently admitted exact owner."""
    active=h.read(STATE/'active-model.json',{})
    if active.get('boot_id')!=h.boot() or active.get('service')!='qwen27b.service':return None
    current=unit('qwen27b.service')
    if current['ActiveState'] not in {'active','deactivating'} or int(current['MainPID'])!=active.get('pid'):return None
    latest=h.read(NV/'state/graph-prefix/active.json',{})
    run_id=latest.get('instance_id','')
    if not re.fullmatch('[0-9a-f]{32}',run_id) or latest.get('boot_id')!=h.boot() or latest.get('pid')!=active['pid']:return None
    if ticks(latest['pid'])!=latest.get('supervisor_start_ticks'):return None
    logs=NV/'runs'/run_id/'logs';now=time.monotonic_ns()
    evidence=fatal_tp_marker(logs,latest['supervisor_start_ticks'],now)
    if not evidence:
        if current['ActiveState']!='active':return None
        evidence=progress.stalled_workers(logs/'stability-telemetry',now)
    if not evidence:return None
    # Recheck container ownership only once a stall is positively established.
    owned=nv_intent()
    if not owned or not same_owner(latest,owned[1]):return None
    if not owned[2]['State']['Running']:
        if evidence['reason']!='OWNED_NVIDIA_TP_FAILED' or not stopped(owned[2]):return None
    return dict(evidence,instance_id=run_id,boot_id=h.boot(),pid=latest['pid'],
                supervisor_start_ticks=latest['supervisor_start_ticks'],container_id=owned[1]['container_id'],epoch=time.time())

def main():
    (ROOT/'recovery').mkdir(exist_ok=True)
    with (ROOT/'recovery/controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempt=0;next_try=0;unhealthy_since=None;transient_since=None;quarantine_since=None
        h.save(ROOT/'recovery/live.json',{'pid':os.getpid(),'epoch':time.time(),'source_sha256':l.sha(Path(__file__)),
            'modules':{name:l.sha(ROOT/'scripts'/name) for name in ('rpc_progress.py','wedge_forensics.py')},
            'fault_stop_grace_seconds':FAULT_STOP_GRACE_S,'fault_forensics_budget_seconds':FAULT_FORENSICS_BUDGET_S})
        if not (STATE/'recovery-status.json').exists():status('IDLE')
        while True:
            try:
                ob=h.observe();pending=h.read(STATE/'recovery-status.json',{})
                hw=ob['state'] if ob['state']!='HEALTHY' else None
                hw,transient_since=sustained(hw,transient_since,time.monotonic())
                reason=hw
                if pending.get('boot_id')==h.boot() and pending.get('state') not in (None,'IDLE','RECOVERED'):reason=reason or 'INCOMPLETE_RECOVERY'
                reason=prefer_owned_fatal(reason)
                # Probe the last admitted process directly: Gateway drain/queue
                # state must not hide a hung backend. A new cold-start PID is exempt.
                if not reason:
                    # Check durable failure/progress before the HTTP health
                    # probe: a responsive API says nothing about a dead rank.
                    stalled=owned_rpc_stall()
                    if stalled:
                        h.save(STATE/'rpc-stall.json',stalled)
                        reason=stalled['reason']
                if not reason:
                    health=admitted_backend_health()
                    if health is False:
                        unhealthy_since=unhealthy_since or time.monotonic()
                        if time.monotonic()-unhealthy_since>=120:reason='BACKEND_UNRESPONSIVE_120S'
                    else:unhealthy_since=None
                latest=h.read(NV/'state/graph-prefix/active.json',{})
                if latest.get('boot_id')==h.boot():
                    durable=h.read(NV/'runs'/latest['instance_id']/'intent.json',{})
                    qcond=str(durable.get('state','')).startswith('QUARANTINE') and ticks(durable.get('pid',0))==durable.get('supervisor_start_ticks')
                    qraw,quarantine_since=sustained('OWNED_NVIDIA_QUARANTINE' if qcond else None,quarantine_since,time.monotonic())
                    if qraw:reason=reason or qraw
                    elif not durable.get('cleanup_verified') and ticks(durable.get('pid',0))!=durable.get('supervisor_start_ticks') and not validate_receipt(durable,NV/'runs'/latest['instance_id']/'intent.json'):
                        reason=reason or 'OWNED_NVIDIA_ORPHAN'
                if reason=='REBOOT_REQUIRED':status('REBOOT_REQUIRED',reason=reason)
                elif reason and time.time()>=next_try:
                    try:cycle(reason);attempt=0;unhealthy_since=None;transient_since=None;quarantine_since=None
                    except RecoveryOwnerChanged as exc:
                        # This recovery belongs to an obsolete instance. Clear
                        # its retry state; a new fault must establish new proof.
                        attempt=0;next_try=0;unhealthy_since=None;transient_since=None;quarantine_since=None
                        status('IDLE',reason='STALE_RECOVERY_OWNER_CHANGED',detail=str(exc))
                    except Exception as exc:
                        attempt+=1;seconds=min(300,30*2**min(attempt-1,4));next_try=time.time()+seconds
                        status('RETRY_WAIT',reason=str(exc),attempt=attempt,next_retry_epoch=next_try)
                elif not reason:attempt=0
            except Exception as exc:
                # Transient monitoring faults are retried, never a permanent latch.
                print(json.dumps({'monitor_retry':str(exc),'epoch':time.time()}),flush=True)
            time.sleep(10)
def prefer_owned_fatal(reason):
    # An attributable fatal + Xid must still use the bounded, owner-bound TP
    # path. That path rechecks hardware and still requires probe + idle proof.
    # Explicit reboot and an existing recovery attempt retain their precedence.
    if reason not in {'RESTART_APP_PROBES_REQUIRED','GPU_RESET_REQUIRED'}:return reason
    fault=owned_rpc_stall()
    if fault and fault['reason']=='OWNED_NVIDIA_TP_FAILED':
        h.save(STATE/'rpc-stall.json',dict(fault,simultaneous_hardware_reason=reason))
        return fault['reason']
    return reason

if __name__=='__main__':main()
