"""Read worker-reported classes/config plus immutable process-namespace mounts."""
import uuid
from common import *

def collect(candidate,out):
    a,row,_=assert_idle(candidate)
    startup=candidate/'runs'/a['instance_id']/'logs/container.jsonl'
    text=startup.read_text()
    assert 'OOM detected' not in text and 'memory allocation failed with OOM' not in text,'Startup autotune OOM invalidates controlled cache comparison'
    payload={'identity':{key:a[key] for key in ['instance_id','boot_id','model_revision','runtime_manifest_sha256']},'request_id':'qualification_'+uuid.uuid4().hex,'read_scalars':False,'quiescent_asserted':True}
    req=urllib.request.Request('http://127.0.0.1:18096/nvidia-runtime-inspection',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    attempts=[];deadline=time.monotonic()+30
    while True:
        try:
            with urllib.request.urlopen(req,timeout=150) as response:body=json.load(response)
            break
        except urllib.error.HTTPError as error:
            detail=error.read().decode(errors='replace');attempts.append({'epoch':time.time(),'status':error.code,'detail':detail});save(out/'inspection-admission-attempts.json',attempts)
            transient=('candidate HTTP work already in progress','native frontend still owns unfinished requests','inspection already in progress')
            if error.code!=409 or not any(message in detail for message in transient) or time.monotonic()>=deadline:
                raise RuntimeError('Inspection admission refused: '+detail) from error
            time.sleep(.25)
            fresh,_=identity(candidate);assert fresh['instance_id']==a['instance_id']
    assert body['status']=='COLLECTED_UNQUALIFIED' and len(body['results'])==2
    assert {r['rank'] for r in body['results']}=={0,1}
    for worker in body['results']:
        assert worker['identity']==payload['identity']
        config=worker['effective_config']
        assert config['parallel']['tensor_parallel_size']==2
        assert config['scheduler']['max_num_batched_tokens']==4096
        assert config['model']['max_model_len']==262144
        assert config['runner_num_spec_tokens']==3
    save(out/'worker-runtime-inspection.json',body)
    raw=cmd(['docker','top',a['container_id'],'-eo','pid,comm']);workers=[int(line.split()[0]) for line in raw.splitlines()[1:] if 'Worker_TP' in line]
    assert len(workers)==2
    files=[]
    for entry in read(candidate/'overlay-integrity.json'):
        source=candidate/entry['path'];assert sha(source)==entry['sha256']
        mount=next(m for m in row['Mounts'] if m['Destination']==entry['target'])
        assert mount['Source']==str(source) and mount['RW'] is False
        installed={str(pid):sha(Path('/proc')/str(pid)/'root'/entry['target'].lstrip('/')) for pid in workers}
        assert all(digest==entry['sha256'] for digest in installed.values())
        files.append({'source':str(source),'installed_and_process_namespace_path':entry['target'],'source_sha256':entry['sha256'],'actual_worker_namespace_sha256':installed,'readonly_mount':True})
    fresh,_=identity(candidate);assert fresh['instance_id']==a['instance_id']
    save(out/'loaded-source-identity.json',{'state':'PASS','identity':payload['identity'],'workers':[{key:w[key] for key in ['pid','rank','worker_class','runner_class','model_class']} for w in body['results']],'host_workers':workers,'executables':{str(pid):str((Path('/proc')/str(pid)/'exe').resolve()) for pid in workers},'files':files,'scope':'Actual worker class/config RPC and source bytes visible in each live worker namespace, readonly mounts, sealed entrypoint verifies before imports. Loaded function bytecode is separately sampled by diagnostic profile release.'})
    if candidate!=BASE:
        cache=candidate/'cache-blessed';proof=read(cache/'cache-provenance.json')
        assert proof['target_manifest']==sha(candidate/'MANIFEST.json')
        save(out/'cache-compatibility.json',{'state':'PASS','provenance_path':str(cache/'cache-provenance.json'),'provenance_sha256':sha(cache/'cache-provenance.json'),'source_bless_sha256':proof['source_bless_sha256'],'files':len(proof['files']),'startup_autotune_oom':False})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir();collect(a.candidate,a.output)
