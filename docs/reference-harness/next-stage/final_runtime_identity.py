"""Read-only source, installation and current-process identities, separately."""
from common import *
import lifecycle

def collect(release,output):
    owner,container,_=assert_idle(release)
    binding=lifecycle.bindings(release);services={}
    for name in ['qwen27b.service','local-gpu-recovery.service','qwen27b-relay.service','local-model-switch.service']:
        state=unit(name);pid=int(state['MainPID']);assert pid and state['ActiveState']=='active'
        root=Path('/proc')/str(pid);start=ticks(pid)
        argv=[v.decode() for v in (root/'cmdline').read_bytes().split(b'\0') if v]
        services[name]={'pid':pid,'start_ticks':start,'exe':str((root/'exe').resolve()),'argv':argv,'script_files':{arg:sha(arg) for arg in argv if arg.endswith(('.py','.mjs')) and Path(arg).is_file()}}
        assert start is not None and ticks(pid)==start
    binding_record=read(binding/'binding.json')
    for rel,digest in binding_record['files'].items():assert sha(binding/rel)==digest
    assert str(binding/'recovery.py') in services['local-gpu-recovery.service']['script_files']
    assert str(binding/'nvidia-launch.py') in services['qwen27b.service']['script_files']
    live=read(PROD/'recovery/live.json')
    assert live['pid']==services['local-gpu-recovery.service']['pid'] and live['source_sha256']==sha(binding/'recovery.py')
    assert all(digest==sha(PROD/'scripts'/name) for name,digest in live['modules'].items())
    processes=[]
    for line in cmd(['docker','top',owner['container_id'],'-eo','pid,ppid,comm']).splitlines()[1:]:
        pid,ppid,comm=line.split(maxsplit=2);pid=int(pid);root=Path('/proc')/str(pid);start=ticks(pid)
        row={'pid':pid,'ppid':int(ppid),'comm':comm,'start_ticks':start,'exe':str((root/'exe').resolve()),'argv':[v.decode() for v in (root/'cmdline').read_bytes().split(b'\0') if v]}
        if comm!='docker-init':
            row['readonly_overlay_files']=[]
            for entry in read(release/'overlay-integrity.json'):
                source=release/entry['path'];target=entry['target']
                mount=next(m for m in container['Mounts'] if m['Destination']==target)
                assert mount['Source']==str(source) and mount['RW'] is False
                observed=sha(root/'root'/target.lstrip('/'));assert observed==entry['sha256']==sha(source)
                row['readonly_overlay_files'].append({'source':str(source),'installed':target,'process_namespace':str(root/'root'/target.lstrip('/')),'sha256':observed})
        assert start is not None and ticks(pid)==start;processes.append(row)
    host_files={str(release/rel):sha(release/rel) for rel in ['launcher.py','inventory_cache.py','model-inventory-cache.json'] if (release/rel).is_file()}
    profiles=read(release/'PROFILES.json')
    model_mount=next(m for m in container['Mounts'] if m['Destination']=='/candidate-model')
    assert Path(model_mount['Source']).resolve()==Path(profiles['model_host_path']).resolve()
    assert model_mount['RW'] is False
    model_identity={'source':model_mount['Source'],'installed':'/candidate-model','readonly':True,'process_namespace_identities':[]}
    if (release/'model-inventory-cache.json').exists():
        cache=module('final_model_identity_cache',release/'inventory_cache.py')
        receipt=read(release/'model-inventory-cache.json')
        for process in processes:
            if 'Worker_TP' not in process['comm']:continue
            root=Path('/proc')/str(process['pid'])/'root/candidate-model';files={}
            for rel,row in receipt['files'].items():
                observed=cache.identity((root/rel).stat())
                assert observed=={k:row[k] for k in cache.FIELDS},('Worker model file identity differs from sealed model',rel)
                files[rel]=observed
            model_identity['process_namespace_identities'].append({'pid':process['pid'],'start_ticks':process['start_ticks'],'files':files})
        assert len(model_identity['process_namespace_identities'])==2
    final,_=identity(release);assert final['instance_id']==owner['instance_id']
    result={'state':'PASS','epoch':time.time(),'runtime':str(release),'manifest':sha(release/'MANIFEST.json'),'owner':owner,'canonical_sources':{str(PROD/'scripts'/name):sha(PROD/'scripts'/name) for name in ['recovery.py','rpc_progress.py','wedge_forensics.py','nvidia-launch.py']},'generated_binding':str(binding),'binding_manifest':binding_record,'host_runtime_files':host_files,'live_services':services,'recovery_process_startup_receipt':live,'model_mount_identity':model_identity,'container':{'id':container['Id'],'image':container['Image'],'processes':processes},'scope':'Actual process executable/argv/PID/start ticks, readonly namespace source bytes and sealed startup verification. Worker class/config RPC is separate. No claim of full Python bytecode introspection.'}
    save(output,result);return {'path':str(output),'sha256':sha(output),'instance_id':owner['instance_id']}

if __name__=='__main__':collect(lifecycle.selected(),Q/'evidence/final-control-identity.json')
