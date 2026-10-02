"""Exclusive bounded diagnostic containers; restore accepted release on exit."""
from common import *
import lifecycle,fcntl,uuid
from fault_test import smoke

SOURCE=Q/'xid/source-candidate'
IMAGE=read(lifecycle.accepted()/'PROFILES.json')['image']
LABEL='org.frank.qwen38.nextstage'

def exact_container(cid,tag):
    row=json.loads(cmd(['docker','inspect',cid]))[0]
    assert row['Id']==cid and row['Image']==IMAGE
    assert row['Config']['Labels'].get(LABEL)==tag
    return row

def main():
    fallback=lifecycle.accepted();assert lifecycle.selected()==fallback
    out=Q/'xid/evidence/minimal-matrix-r2';out.mkdir()
    manifest={'state':'DIAGNOSTIC_ONLY','image':IMAGE,'files':{name:sha(SOURCE/name) for name in ['cupti_modes.cpp','cupti_modes.so','minimal_graph_repro.py','libcupti.so.12']},'bounded_policy':'One fresh process per feature A-H. Stop immediately on any new Xid or probe/P2P failure. No retry of a failed GPU configuration. No production profiling.','operations':'Persistent tensors; CUDA graph with GEMM and NCCL; three replays/step; pinned index copy and explicit wait; 64 steps. Not a full Qwen context/MTP reproduction.'}
    save(out/'manifest.json',manifest);tag=sha(out/'manifest.json')
    result={'state':'RUNNING','fallback':str(fallback),'started_epoch':time.time(),'rows':[]};save(out/'receipt.json',result)
    active_cid=None
    try:
        lifecycle.stop_current(fallback,result)
        with (NV/'state/gpu.lock').open('r+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert not cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])
            for mode in 'ABCDEFGH':
                work=out/mode;work.mkdir();started=time.time()
                args=['docker','create','--name','qwen38-xid-'+uuid.uuid4().hex,'--init','--gpus','all','--network','none','--cpus','8','--memory','8g','--shm-size','1g','--user',str(os.getuid())+':'+str(os.getgid()),'--label',LABEL+'='+tag,'--mount','type=bind,src='+str(SOURCE)+',dst=/source,readonly','--mount','type=bind,src='+str(work)+',dst=/results','--env','NCCL_PROTO=Simple','--env','NCCL_P2P_DISABLE=0','--env','NCCL_DEBUG=INFO','--env','OMP_NUM_THREADS=1','--entrypoint','/usr/bin/python3',IMAGE,'-I','-m','torch.distributed.run','--nnodes=1','--node-rank=0','--master-addr=127.0.0.1','--master-port=29500','--nproc-per-node=2','/source/minimal_graph_repro.py',mode]
                cid=cmd(args);active_cid=cid;row=exact_container(cid,tag);save(work/'container-created.json',row);save(work/'command.json',args)
                cmd(['docker','start',cid]);timed_out=False
                try:exit_code=int(cmd(['docker','wait',cid],timeout=180))
                except subprocess.TimeoutExpired:
                    timed_out=True;exact_container(cid,tag);cmd(['docker','stop','--time','5',cid],timeout=20);exit_code=int(cmd(['docker','wait',cid],timeout=10))
                row=exact_container(cid,tag);assert not row['State']['Running'];save(work/'container-exited.json',row);active_cid=None
                (work/'container.log').write_text(cmd(['docker','logs','--timestamps',cid]))
                kernel=cmd(['journalctl','-k','-b','--since','@'+str(started),'--no-pager']);(work/'kernel.log').write_text(kernel)
                xid='NVRM: Xid' in kernel or 'GPU has fallen' in kernel
                payloads=[read(work/f'{mode}-rank{r}.json') if (work/f'{mode}-rank{r}.json').exists() else {'state':'MISSING'} for r in [0,1]]
                entry={'mode':mode,'started_epoch':started,'finished_epoch':time.time(),'exit_code':exit_code,'timed_out':timed_out,'new_xid':xid,'container':cid,'ranks':payloads,'state':'PASS_NO_REPRODUCTION' if exit_code==0 and not xid and all(p['state']=='PASS_NO_REPRODUCTION' for p in payloads) else 'FAIL_OR_UNSUPPORTED'}
                result['rows'].append(entry);save(out/'receipt.json',result)
                print(json.dumps({k:entry[k] for k in ['mode','state','exit_code','new_xid']}),flush=True)
                if xid or timed_out or exit_code:
                    result['stop_reason']='BOUNDED_SAFETY_STOP_ON_XID_OR_EXECUTION_FAILURE';break
                assert not cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])
            proof=out/'post-diagnostic-hardware-proof';proof.mkdir()
            recovery=module('minimal_recovery',lifecycle.bindings(fallback)/'recovery.py')
            recovery.proof(proof);result['hardware_proof']=str(proof)
        lifecycle.start(fallback,result);result['restored_completion']=smoke()
        result['state']='COMPLETED_RESTORED' if len(result['rows'])==8 and all(x['state']=='PASS_NO_REPRODUCTION' for x in result['rows']) else 'BOUNDED_STOP_RESTORED'
    except BaseException as error:
        result.update(state='FAIL_REQUIRES_REVIEW',error=repr(error))
        if active_cid:
            row=exact_container(active_cid,tag)
            if row['State']['Running']:cmd(['docker','stop','--time','5',active_cid],timeout=20)
        # Unknown hardware failures never bypass fresh proof. Caller diagnoses before restart.
        raise
    finally:result['finished_epoch']=time.time();save(out/'receipt.json',result)

if __name__=='__main__':main()
