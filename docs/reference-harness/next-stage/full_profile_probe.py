"""One bounded original-profiler session in isolated current-math release."""
from common import *
import lifecycle,gzip,collections
from bench import completion
from safe_restore import restore
from loaded_identity import collect

def control(action):
 req=urllib.request.Request('http://127.0.0.1:18096/'+action,data=b'',method='POST')
 with urllib.request.urlopen(req,timeout=60) as r:return r.read().decode()
def main():
 minimal=read(Q/'xid/evidence/minimal-matrix-r3/receipt.json')
 assert minimal['state']=='COMPLETED_RESTORED' and not any(r['new_xid'] for r in minimal['rows']),'No further active reproduction after safety/feature failure'
 candidate=NV/'candidate-v44-xid-original-profiler-20260923';fallback=lifecycle.accepted();out=Q/'xid/evidence/full-original-35k';out.mkdir()
 result={'state':'RUNNING','started_epoch':time.time(),'candidate':str(candidate),'manifest':sha(candidate/'MANIFEST.json'),'budget':'Exactly one fresh 35K session. No retry; stop on first new Xid or execution failure.','performance_scope':'DIAGNOSTIC_ONLY_NOT_PRODUCTION_THROUGHPUT','timeline':[]};save(out/'receipt.json',result)
 def event(name):result['timeline'].append({'name':name,'realtime_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns()});save(out/'receipt.json',result)
 try:
  lifecycle.activate(candidate);a,_=identity(candidate);result['identity']=a;collect(candidate,out)
  result['unprofiled']=completion(candidate,35840,256,out/'cupti-off-35k');save(out/'receipt.json',result)
  event('start_profile_dispatch');result['start_profile_response']=control('start_profile');event('start_profile_return')
  raw=cmd(['docker','top',a['container_id'],'-eo','pid,comm']);workers=[int(line.split()[0]) for line in raw.splitlines()[1:] if 'Worker_TP' in line]
  assert len(workers)==2;result['actual_profiling_libraries']=[]
  for pid in workers:
   maps=Path(f'/proc/{pid}/maps').read_text();paths=sorted({line.split()[-1] for line in maps.splitlines() if any(n in line for n in ['libcupti.so','libcuda.so','libnccl.so','libcudart.so'])})
   result['actual_profiling_libraries'].append({'pid':pid,'start_ticks':ticks(pid),'paths':paths,'cupti_sha256':{path:sha(Path(f'/proc/{pid}/root')/path.lstrip('/')) for path in paths if 'libcupti.so' in path},'monotonic_ns':time.monotonic_ns()})
  save(out/'receipt.json',result)
  try:result['profiled']=completion(candidate,35840,256,out/'original-profile-35k')
  except BaseException:
   event('request_failed');raise
  event('stop_profile_dispatch');result['stop_profile_response']=control('stop_profile');event('stop_profile_return')
  traces=list((candidate/'runs'/a['instance_id']/'logs/decode-profile').rglob('*.pt.trace.json*'));assert len(traces)>=2
  result['traces']=[]
  for p in traces:
   opener=gzip.open if p.suffix=='.gz' else open
   with opener(p,'rt') as f:d=json.load(f)
   cats=collections.Counter(e.get('cat') for e in d.get('traceEvents',[]));assert cats['kernel']>0
   result['traces'].append({'path':str(p),'sha256':sha(p),'categories':dict(cats),'base_time_ns':d.get('baseTimeNanoseconds')})
  assert result['unprofiled']['output_hash']==result['profiled']['output_hash'] and result['unprofiled']['raw_logits']==result['profiled']['raw_logits']
  result['result']='PASS_NO_REPRODUCTION'
 except BaseException as error:
  result.update(result='BOUNDED_DIAGNOSTIC_FAILURE',error=repr(error));event('diagnostic_stopped')
 finally:
  kernel=cmd(['journalctl','-k','-b','--since','@'+str(result['started_epoch']),'--no-pager']);(out/'kernel.log').write_text(kernel);result['new_xid']='NVRM: Xid' in kernel or 'GPU has fallen' in kernel
  save(out/'receipt.json',result)
  try:
   result['rollback']=restore(fallback);result['state']='COMPLETED_RESTORED'
  except BaseException as error:
   result.update(state='FAIL_CLOSED_REQUIRES_REVIEW',rollback_error=repr(error));raise
  finally:result['finished_epoch']=time.time();save(out/'receipt.json',result)
if __name__=='__main__':main()
