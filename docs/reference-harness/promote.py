"""Copy a qualified payload into a durable production release, activate, verify."""
import argparse,shutil
from common import *
import lifecycle
from bench import completion
from fault_test import smoke

def check_gate_identity(candidate,summary,fault):
    digest=sha(candidate/'MANIFEST.json')
    assert summary['state']=='PASS' and summary['candidate']==str(candidate)
    assert summary['rows'] and all(r['state']=='PASS' and r['candidate']==str(candidate) and r['manifest']==digest for r in summary['rows']), 'Benchmark receipt payload differs from candidate'
    assert fault['state']=='PASS' and fault['candidate']==str(candidate)
    assert fault['old_identity']['runtime_manifest_sha256']==digest, 'Fault receipt payload differs from candidate'

def restore_ledgers(previous,release):
    for name,original in previous.items():
        path=Path(name);current=path.read_bytes() if path.exists() else None
        if current==original:continue
        if current is not None:
            record=json.loads(current)
            assert record.get('candidate',record.get('runtime'))==str(release),'Another owner changed the promotion ledger'
        if original is None:
            if path.exists():path.unlink()
        else:
            temporary=path.with_suffix('.rollback-pending');temporary.write_bytes(original);temporary.replace(path)

def promote(candidate,label,fault_label,name):
    comparison=read(Q/'evidence'/f'{label}-comparison.json');assert comparison['decision']=='PASS'
    summary=read(Q/'evidence'/label/'summary.json');assert summary['state']=='PASS'
    assert summary['candidate']==str(candidate)
    fault=read(Q/'evidence'/fault_label/'receipt.json');assert fault['state']=='PASS' and fault['candidate']==str(candidate)
    check_gate_identity(candidate,summary,fault)
    assert lifecycle.selected()==candidate
    assert_idle(candidate)
    assert name.startswith('production-') and Path(name).name==name
    for rel in ['PROFILES.json','graph-prefix.json','environment.json']:
        assert sha(candidate/rel)==sha(BASE/rel),('Frozen inference parameter drift',rel)
    fallback=lifecycle.accepted();dest=NV/name;assert not dest.exists()
    out=Q/'evidence'/(name+'-promotion');out.mkdir()
    previous_ledgers={str(path):path.read_bytes() if path.exists() else None for path in [Q/'canonical.json',PROD/'dense-production-release.json']}
    save(out/'previous-production-ledgers.json',{path:raw.decode() if raw is not None else None for path,raw in previous_ledgers.items()})
    before,_=identity(candidate)
    record={'state':'PREPARING','candidate':str(candidate),'production':str(dest),'fallback':str(fallback),'before_identity':before,'qualified_manifest':sha(candidate/'MANIFEST.json'),'qualification_comparison':str(Q/'evidence'/f'{label}-comparison.json'),'fault_receipt':str(Q/'evidence'/fault_label/'receipt.json'),'started_epoch':time.time()}
    save(out/'receipt.json',record)
    manifest=read(candidate/'MANIFEST.json');dest.mkdir()
    for rel,entry in manifest['files'].items():
        assert sha(candidate/rel)==entry['sha256']
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(candidate/rel,target)
        assert sha(target)==entry['sha256']
    manifest.update(state='SEALED_PRODUCTION',qualified_payload_manifest_sha256=sha(candidate/'MANIFEST.json'),promotion_receipt=str(out/'receipt.json'))
    save(dest/'MANIFEST.json',manifest);module('promoted_static',dest/'launcher.py').static()
    record['production_manifest']=sha(dest/'MANIFEST.json');record['payload_files_identical']=True;save(out/'receipt.json',record)
    try:
        lifecycle.activate(dest);record['smoke']=smoke();save(out/'receipt.json',record)
        from loaded_identity import collect
        collect(dest,out)
        record['long_context']=completion(dest,258048,256,out/'maximum-context-production')
        reference=[r for r in summary['rows'] if r['count']==258048 and r['requested_tokens']==256 and not r['warmup']]
        assert reference and record['long_context']['output_hash'] in {r['output_hash'] for r in reference}
        delta=min(max(abs(a-b) for a,b in zip(record['long_context']['raw_logits'],r['raw_logits'])) for r in reference)
        assert delta<=1e-6,('Promoted copy differs from its qualified numerical result',delta)
        record['promoted_copy_logit_max_abs_difference']=delta
        active_before,_=identity(dest);start=time.monotonic()
        while time.monotonic()-start<60:
            current,_=identity(dest);assert current['instance_id']==active_before['instance_id']
            assert not list((dest/'runs'/current['instance_id']/'logs').glob('first-failure-*.json'))
            time.sleep(5)
        record['smoke_after_observation']=smoke()
        record.update(state='PROMOTED',finished_epoch=time.time(),after_identity=active_before,observation_seconds=time.monotonic()-start)
        save(out/'receipt.json',record)
        save(Q/'canonical.json',{'state':'PROMOTED','candidate':str(dest),'manifest':sha(dest/'MANIFEST.json'),'receipt':str(out/'receipt.json'),'qualified_benchmark_label':label,'epoch':time.time()})
        save(PROD/'dense-production-release.json',{'state':'PROMOTED','runtime':str(dest),'manifest_sha256':sha(dest/'MANIFEST.json'),'bindings':str(lifecycle.bindings(dest)),'receipt':str(out/'receipt.json'),'previous_runtime':str(fallback),'epoch':time.time()})
    except BaseException as error:
        record.update(state='FAILED_RESTORING',error=repr(error));save(out/'receipt.json',record)
        try:
            from safe_restore import restore
            record['rollback']=restore(fallback)
            record['rollback_smoke']=record['rollback']['smoke'];restore_ledgers(previous_ledgers,dest);record['state']='REJECTED_ROLLBACK_PASS'
        except BaseException as rollback:record.update(state='ROLLBACK_NEEDS_REPAIR',rollback_error=repr(rollback))
        save(out/'receipt.json',record);raise
    print(json.dumps({'state':record['state'],'runtime':str(dest),'manifest':record['production_manifest']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--fault-label',required=True);p.add_argument('--name',required=True);a=p.parse_args();promote(a.candidate,a.label,a.fault_label,a.name)
