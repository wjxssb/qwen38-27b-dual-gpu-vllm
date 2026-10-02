"""Same-boot Xid 8 reconciliation. Historical incidents are never cleared."""
import hashlib
import json
import math
import re
import time
from pathlib import Path

SOURCE = 'https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html'
XID = re.compile(r'NVRM: Xid\s*\([^)]*\):\s*(\d+)', re.I)
OTHER = re.compile(r'AER:.*(?:error|fatal)|fallen off|illegal memory access|misaligned address|CUDA error', re.I)

def require(ok, message):
    if not ok: raise RuntimeError(message)

def faults(journal):
    result=[]
    for row in journal:
        msg=row.get('MESSAGE', '')
        if XID.search(msg) or OTHER.search(msg):
            result.append({k:row[k] for k in ('__CURSOR','_BOOT_ID','__MONOTONIC_TIMESTAMP','MESSAGE')})
    return result

def policy(events):
    for row in events:
        msg=row['MESSAGE']; m=XID.search(msg)
        if (m and int(m[1]) == 79) or (m and int(m[1]) == 154 and re.search(r'Node Reboot',msg,re.I)) or 'fallen off' in msg.lower():
            return 'REQUEST_NODE_REBOOT'
    if events and all(XID.search(r['MESSAGE']) and int(XID.search(r['MESSAGE'])[1]) == 8 for r in events):
        return 'RESTART_APP_PROBES_REQUIRED'
    return 'STOP_FOR_DIAGNOSIS'  # An unfamiliar event is not an automatic reboot request.

def validate_probe(p, uuids):
    require(p.get('status')=='PASS' and p.get('contexts_destroyed') is True, 'CUDA probe did not complete cleanup')
    devs=p.get('devices',[])
    require(len(devs)==2 and {r['uuid'] for r in devs}==set(uuids) and {r['dev'] for r in devs}=={0,1}, 'probe GPU identity differs')
    cases=p.get('cases',[])
    expected={('local_alloc_kernel_write_read',s,s):4 for s in range(2)}
    expected.update({(k,s,1-s):8 for s in range(2) for k in ('peer_copy','direct_peer_kernel_write')})
    counts={}; seeds={}
    for c in cases:
        key=(c.get('kind'),c.get('source'),c.get('destination'))
        require(c.get('bytes')==1048576 and c.get('mismatch_count')==0 and re.fullmatch('[a-f0-9]{64}',c.get('sha256','')), 'probe integrity data missing')
        counts[key]=counts.get(key,0)+1; seeds.setdefault(key,set()).add(c.get('seed'))
    require(counts==expected and all(len(seeds[k])==v for k,v in expected.items()),'probe direction/repeat coverage differs')

def validate_evidence(l, intent, evidence):
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(evidence['old_intent']==intent and intent['boot_id']==boot, 'recovery old intent or boot differs')
    old=evidence['old_container']; l.supervisor_module().exact_owned(intent,old)
    require(l.supervisor_module().exited_after_start(old) and old['State']['Pid']==0 and not old['State']['Paused'] and not old['State']['Restarting'], 'old application not definitively exited')
    require(evidence['old_supervisor_stopped'] is True, 'old CPU supervisor not stopped')
    try: ticks=l.supervisor_module().proc_ticks(intent['pid'])
    except (OSError,ProcessLookupError): ticks=None
    require(ticks!=intent['supervisor_start_ticks'], 'old CPU supervisor remains alive')
    before=faults(evidence['kernel_before']); after=faults(evidence['kernel_after'])
    require(policy(before)=='RESTART_APP_PROBES_REQUIRED' and before==after, 'fault set changed or not isolated Xid 8')
    require(all(r['_BOOT_ID']==boot.replace('-','') for r in evidence['kernel_before']+evidence['kernel_after']), 'journal boot differs')
    require(evidence['kernel_before'] and evidence['kernel_after'], 'full boot journal absent')
    probe=evidence['probe']; require(probe['returncode']==0, 'CUDA probe process failed')
    validate_probe(json.loads(probe['stdout']),l.GPU_UUIDS)
    require(probe['source_sha256']==l.sha(l.ROOT/'cuda_recovery_probe.py'), 'recovery probe source binding differs')
    for phase in ['pci_before','pci_after']:
        rows=evidence[phase]
        require(len(rows)==2 and {r['uuid'] for r in rows}==set(l.GPU_UUIDS), 'PCI pair differs')
        for r in rows:
            require(r['vendor']=='0x10de' and r['device'].startswith('0x') and r['driver']=='nvidia' and r['config_vendor']=='10de' and r['link_width']>0 and r['link_speed']>0, 'PCI inaccessible or link invalid')
        if phase=='pci_after': require(rows==evidence['pci_before'], 'PCI identity/link changed during recovery')
    samples=evidence['idle_samples']; times=[r['monotonic'] for r in samples]
    require(len(times)>=3 and all(math.isfinite(t) for t in times) and times[-1]-times[0]>=60 and all(0<b-a<=30 for a,b in zip(times,times[1:])), 'continuous 60-second idle observation absent')
    for sample in [evidence['nvml_before'],*samples]:
        require(sample['compute_csv'].strip()=='', 'GPU compute owner present')
        rows=l.validate_gpu_rows(sample['gpu_csv'])
        require(all(r['used_mib']<=128 for r in rows),'GPU memory not restored')
    require(evidence['current_p2p']==l.verify_p2p(), 'root P2P proof changed')
    require(evidence['completed_epoch']>=probe['finished_epoch']>=probe['started_epoch']>=evidence['started_epoch'], 'recovery chronology invalid')

def validate(l, intent, boot, path, inspect_current=None):
    receipt,seal=l.immutable_receipt_json(path)
    require(receipt.get('kind')=='SAME_BOOT_RESTART_APP_RECOVERY' and receipt.get('status')=='PASS' and receipt.get('boot_id')==boot==intent['boot_id'] and receipt.get('old_instance_id')==intent['instance_id'], 'recovery receipt scope differs')
    require(receipt.get('old_incident_preserved') is True and receipt.get('model_qualification')=='NOT_PROVEN','recovery cannot claim model qualification')
    e,es=l.supervisor_module().read_immutable_mapping(path.parent,'evidence.json')
    require(es['sha256']==receipt['evidence_sha256'], 'recovery evidence digest differs')
    validate_evidence(l,intent,e)
    fresh=inspect_current(intent['container_id']) if inspect_current else json.loads(l.command(l.DOCKER+['inspect',intent['container_id']]))[0]
    l.supervisor_module().exact_owned(intent,fresh)
    require(fresh['State']==e['old_container']['State'], 'old container state changed since recovery')
    journal=l.command(['journalctl','-k','-b','--no-pager','-o','json'])
    require(faults([json.loads(r) for r in journal.splitlines()])==faults(e['kernel_after']), 'new kernel fault since recovery')
    l.supervisor_module().verify_mapping_seals([seal,es])
    return {'receipt':str(path),'sha256':seal['sha256'],'scope':'ALLOW_FRESH_INSTANCE_SAME_BOOT_AFTER_XID8_RECOVERY','old_run_qualification':'FAILED_PRESERVED'}
