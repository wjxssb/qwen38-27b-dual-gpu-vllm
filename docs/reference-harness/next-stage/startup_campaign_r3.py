"""One instrumented startup and real in-flight recovery, then exact fallback."""
from common import *
import lifecycle, threading
from fault_test import fault_test,smoke
from safe_restore import restore
from bench import sampler,completion

class StartupSampler(sampler.Sampler):
    def __init__(self,path,candidate):
        super().__init__(path,period=1.0);self.candidate=candidate
    def sample(self):
        pids=[];owner=None
        try:
            a=active(self.candidate);owner={k:a.get(k) for k in ['instance_id','pid','container_id']}
            container=a.get('container_id')
            if isinstance(container,str) and len(container)==64:
                pids=[int(v.split()[0]) for v in cmd(['docker','top',container,'-eo','pid']).splitlines()[1:]]
            if isinstance(a.get('pid'),int) and a['pid']>1:pids.append(a['pid'])
        except (OSError,ValueError,subprocess.SubprocessError):pass
        self.pids=pids;row=super().sample();row['owner']=owner
        row['io']={}
        for pid in pids:
            try:
                row['io'][str(pid)]={'start_ticks':ticks(pid),'counters':dict((k,int(v)) for k,v in (line.split(':',1) for line in Path(f'/proc/{pid}/io').read_text().splitlines()))}
            except (OSError,ValueError):pass
        row['nvme_diskstats']=[line.split() for line in Path('/proc/diskstats').read_text().splitlines() if 'nvme' in line and line.split()[2][-1].isdigit()]
        row['cpu_pressure']=Path('/proc/pressure/cpu').read_text().strip()
        return row

def main():
    candidate=NV/'candidate-v42-startup-spawn-observer-20260923';fallback=lifecycle.accepted()
    out=Q/'recovery/evidence/observer-campaign-r3';out.mkdir()
    result={'state':'RUNNING','candidate':str(candidate),'fallback':str(fallback),'started_epoch':time.time()}
    save(out/'receipt.json',result)
    try:
        with StartupSampler(out/'metrics.jsonl',candidate):
            lifecycle.activate(candidate)
            result['initial_identity']=identity(candidate)[0];result['first_smoke']=smoke()
            marker_dir=candidate/'runs'/result['initial_identity']['instance_id']/'logs/startup-timeline'
            assert len(list(marker_dir.glob('*.jsonl')))>=4, 'Full process startup instrumentation absent'
            marker_text=''.join(p.read_text() for p in marker_dir.glob('*.jsonl'))
            assert 'GPUModelRunner.capture_model' in marker_text and 'Worker.init_device' in marker_text, 'Required startup phases missing'
            result['normal_completion']=completion(candidate,1024,256,out/'observer-1k')
            save(out/'receipt.json',result)
            result['spawn_observation']='STARTUP_ONLY_NO_ADDITIONAL_FAULT'
            result['replacement_identity']=identity(candidate)[0]
        result['rollback']=restore(fallback)
        result['state']='PASS_OBSERVATION_AND_RESTORE'
    except BaseException as error:
        result.update(state='FAIL',error=repr(error));save(out/'receipt.json',result)
        try:
            result['rollback']=restore(fallback)
            result['state']='FAIL_RESTORED'
        except BaseException as rollback:
            result['rollback_error']=repr(rollback)
        raise
    finally:
        result['finished_epoch']=time.time();save(out/'receipt.json',result)

if __name__=='__main__':main()
