"""Authorized, exact-identity single-worker exit; production recovery owns teardown."""
import argparse, signal,contextlib
from common import *

def smoke():
    payload={'model':'qwen38-27b-dense','messages':[{'role':'user','content':'Return exactly RECOVERY_OK.'}],'max_tokens':32,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
    req=urllib.request.Request('http://127.0.0.1:18080/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=15) as response:body=json.load(response)
    assert 'RECOVERY_OK' in (body['choices'][0]['message'].get('content') or ''),body
    return body

def _fault_test(candidate,label,after_monitor_poll=False,workload=None):
    out=Q/'evidence'/label;out.mkdir();a,row,st=assert_idle(candidate)
    oldrun=candidate/'runs'/a['instance_id'];l=module('fault_owned',candidate/'launcher.py');s=l.supervisor_module()
    raw=cmd(['docker','top',a['container_id'],'-eo','pid,ppid,comm'])
    workers=[int(line.split()[0]) for line in raw.splitlines()[1:] if 'Worker_TP' in line]
    assert len(workers)==2,raw
    victim=workers[1];start=ticks(victim);assert start is not None
    receipt={'state':'PREPARED','candidate':str(candidate),'old_identity':{key:a[key] for key in ['instance_id','container_id','pid','supervisor_start_ticks','boot_id','runtime_manifest_sha256']},'victim_pid':victim,'victim_start_ticks':start,'peer_pid':workers[0],'processes_before':raw,'timeline':{},'method':'SIGKILL one exact TP Worker via pidfd; no GPU reset or physical action'}
    save(out/'receipt.json',receipt)
    if after_monitor_poll:
        monitor_pid=int(unit('local-gpu-recovery.service')['MainPID']);monitor_ticks=ticks(monitor_pid)
        assert monitor_pid>1 and monitor_ticks is not None
        deadline=time.monotonic()+25;seen_awake=False
        while time.monotonic()<deadline:
            assert ticks(monitor_pid)==monitor_ticks,'Recovery monitor changed before fault injection'
            waiting=Path(f'/proc/{monitor_pid}/wchan').read_text().strip()
            asleep='nanosleep' in waiting
            if not asleep:seen_awake=True
            if seen_awake and asleep:
                receipt['injection_alignment']={'monitor_pid':monitor_pid,'start_ticks':monitor_ticks,'sleep_observed_epoch':time.time(),'method':'observed completed monitor poll followed by nanosleep; inject at start of its next polling interval'}
                break
            time.sleep(.02)
        else:raise RuntimeError('Could not establish monitor polling phase; no worker signalled')
    if workload is not None:
        workload.start();workload.await_executing();receipt['inflight_request']=workload.snapshot()
        fresh,freshrow=identity(candidate)
        st=status();assert st['active_requests']==1 and st['queued_requests']==0
        assert not workload.done.is_set()
    else:fresh,freshrow,_=assert_idle(candidate)
    assert fresh['instance_id']==a['instance_id'];s.exact_owned(a,freshrow)
    assert ticks(victim)==start and str(victim) in [line.strip() for line in cmd(['docker','top',a['container_id'],'-eo','pid']).splitlines()]
    fd=os.pidfd_open(victim)
    try:
        assert ticks(victim)==start
        t0=time.time();receipt['timeline']['worker_failure']=t0
        signal.pidfd_send_signal(fd,signal.SIGKILL)
    finally:os.close(fd)
    receipt['state']='RUNNING';save(out/'receipt.json',receipt);deadline=time.monotonic()+1000;last=None;success_pid=None;stable_since=None
    with (out/'timeline.jsonl').open('x') as log:
        while time.monotonic()<deadline:
            now=time.time();sample={'epoch':now}
            if workload is not None:receipt['inflight_request']=workload.snapshot()
            try:
                service=unit('qwen27b.service');sample['service']=service
                if ticks(a['pid'])!=a['supervisor_start_ticks']:receipt['timeline'].setdefault('old_supervisor_gone',now)
                if int(service['MainPID']) not in (0,a['pid']):receipt['timeline'].setdefault('new_supervisor_started',now)
                state=read(PROD/'gateway/state/recovery-status.json');sample['recovery']=state
                if state.get('reason')=='OWNED_NVIDIA_TP_FAILED' and state.get('recovery_owner',{}).get('instance_id')==a['instance_id']:
                    receipt['timeline'].setdefault('detected',state['epoch']);receipt['recovery_run']=state['run']
                intent=read(oldrun/'intent.json')
                if intent.get('stop_dispatched_epoch'):receipt['timeline'].setdefault('stop_issued',intent['stop_dispatched_epoch'])
                if 'stop_issued' not in receipt['timeline'] and 'detected' in receipt['timeline']:
                    from stop_timeline import observe
                    observe(receipt,out)
                if all(ticks(pid) is None for pid in workers):receipt['timeline'].setdefault('old_group_gone',now)
                marker=oldrun/'logs/first-failure-engine-core.json'
                if marker.exists():
                    value=read(marker);receipt.setdefault('first_failure',value);save(out/'first-failure-engine-core.json',value)
                reconciled=PROD/'gateway/state/reconciled'/f'{a["instance_id"]}.json'
                if reconciled.exists():
                    rec=read(reconciled);receipt['reconciliation']=rec
                    receipt['timeline'].setdefault('observation_complete',rec['epoch']);save(out/'reconciliation.json',rec)
                try:
                    gateway=status();sample['gateway']={key:gateway.get(key) for key in ['state','pid','active_requests','queued_requests','last_error']}
                    if gateway['state']=='ready' and gateway.get('pid') not in (None,0,a['pid']):
                        receipt['timeline'].setdefault('model_ready',now)
                        if success_pid is None:
                            body=smoke();save(out/'successful-completion.json',body);receipt['timeline']['first_successful_completion']=time.time();success_pid=gateway['pid'];stable_since=time.monotonic()
                        elif gateway['pid']!=success_pid:raise RuntimeError('replacement instance changed during observation')
                        if stable_since and time.monotonic()-stable_since>=60 and receipt.get('reconciliation'):
                            assert receipt.get('first_failure',{}).get('reason')=='TP_WORKER_EXIT',receipt.get('first_failure')
                            assert receipt['timeline'].get('detected'), 'Fast failure marker was not detected'
                            assert receipt['timeline']['detected']-t0<30,'Fatal detection waited too long'
                            new,_=identity(candidate);receipt['new_identity']={key:new[key] for key in ['instance_id','container_id','pid','supervisor_start_ticks']}
                            run=Path(receipt['recovery_run']);proof=read(Path(receipt['reconciliation']['run'])/'idle-proof.json')
                            assert proof['elapsed_seconds']>=60
                            forensic=read(run/'wedge/manifest.json')
                            assert forensic['budget_seconds']==8
                            assert forensic['finished_epoch']-forensic['started_epoch']<=10,'Forensics exceeded bounded budget'
                            receipt['forensics']=forensic
                            fault=read(run/'rpc-stall.json')
                            assert all(fault[key]==a[key] for key in ['instance_id','container_id','pid','supervisor_start_ticks','boot_id'])
                            probe=read(run/'cuda-probe.json');assert probe['returncode']==0
                            receipt['cuda_p2p_probe']=probe
                            assert 'stop_issued' in receipt['timeline'] and 'old_group_gone' in receipt['timeline']
                            receipt['replacement_processes']=cmd(['docker','top',new['container_id'],'-eo','pid,ppid,comm'])
                            assert sum('Worker_TP' in line for line in receipt['replacement_processes'].splitlines())==2
                            if workload is not None:receipt['inflight_request']=workload.verify_interrupted(t0)
                            receipt.update(state='PASS',finished_epoch=time.time(),latencies={key:value-t0 for key,value in receipt['timeline'].items()},new_instance_survived_seconds=time.monotonic()-stable_since)
                            save(out/'receipt.json',receipt);print(json.dumps(receipt['latencies']),flush=True);return receipt
                except urllib.error.URLError as error:sample['gateway_error']=str(error)
            except (FileNotFoundError,subprocess.SubprocessError,json.JSONDecodeError) as error:sample['observation_error']=str(error)
            log.write(json.dumps(sample)+'\n');log.flush();save(out/'receipt.json',receipt);time.sleep(1)
    receipt.update(state='FAIL',error='Recovery did not meet all gates before deadline',finished_epoch=time.time());save(out/'receipt.json',receipt);raise RuntimeError(receipt['error'])

def fault_test(candidate,label,after_monitor_poll=False,inflight=False):
    from fault_workload import InflightRequest
    manager=InflightRequest(candidate,label) if inflight else contextlib.nullcontext(None)
    with manager as workload:return _fault_test(candidate,label,after_monitor_poll,workload)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,default=BASE);p.add_argument('--label',required=True);p.add_argument('--after-monitor-poll',action='store_true');p.add_argument('--inflight',action='store_true');a=p.parse_args()
    try:fault_test(a.candidate,a.label,a.after_monitor_poll,a.inflight)
    except BaseException as error:
        path=Q/'evidence'/a.label/'receipt.json'
        if path.exists():
            receipt=read(path);receipt.update(state='FAIL',error=repr(error),finished_epoch=time.time());save(path,receipt)
        raise
