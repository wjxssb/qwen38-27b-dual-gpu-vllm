"""Single-variable recovery A/B, no startup observer in either runtime."""
from common import *
import lifecycle,statistics
from bench import completion
from fault_test import fault_test
from safe_restore import restore
from loaded_identity import collect

from functional_checks import functional

def main():
 baseline=lifecycle.accepted();candidate=NV/'candidate-v45-renderer-handshake-overlap-20260923';out=Q/'recovery/evidence/renderer-campaign-r2';out.mkdir()
 result={'state':'RUNNING','started_epoch':time.time(),'baseline':str(baseline),'candidate':str(candidate),'design':'A3 then B3 real 66K in-flight worker loss; every replacement fresh, fixed cache provenance, polling-phase aligned. Descriptive n=3; block order disclosed. No startup observer in either runtime.','faults':{},'inference':{}};save(out/'receipt.json',result)
 try:
  assert lifecycle.selected()==baseline
  for label,runtime in [('baseline',baseline),('candidate',candidate)]:
   if lifecycle.selected()!=runtime:lifecycle.activate(runtime)
   path=out/label;path.mkdir();collect(runtime,path);result['inference'][label]=[]
   if label=='baseline':
    prior=Q/'recovery/evidence/renderer-campaign/receipt.json'
    old=read(prior);assert len(old['inference']['baseline'])==4 and all(r['state']=='PASS' and r['manifest']==sha(baseline/'MANIFEST.json') for r in old['inference']['baseline'])
    result['inference'][label]=old['inference']['baseline'];result['numerical_reference']={'path':str(prior),'sha256':sha(prior),'scope':'Completed baseline correctness requests reused; no inference-throughput A/B claim. New recovery trials use fresh current instances.'}
   else:
    for count in [1024,16384,67584,258048]:
     row=completion(runtime,count,256,path/f'{label}-{count}-256');result['inference'][label].append(row);save(out/'receipt.json',result)
   result[label+'_functional']=functional(runtime,path/'functional');save(out/'receipt.json',result)
   if label=='candidate':
    for a,b in zip(result['inference']['baseline'],result['inference']['candidate']):assert a['count']==b['count'] and a['output_hash']==b['output_hash'] and a['raw_logits']==b['raw_logits'],('Numerical mismatch',a['count'])
    result['numerics']='PASS_EXACT_TOKEN_HASH_AND_SELECTED_RAW_LOGITS'
   result['faults'][label]=[]
   for repetition in range(3):
    receipt=fault_test(runtime,f'renderer-r2-{label}-fault-{repetition+1}',after_monitor_poll=True,inflight=True);result['faults'][label].append(receipt);save(out/'receipt.json',result)
  result['rollback']=restore(baseline)
  def stats(label):
   rows=result['faults'][label];a=[r['latencies']['first_successful_completion'] for r in rows];b=[r['latencies']['model_ready']-r['latencies']['new_supervisor_started'] for r in rows]
   return {'completion_s':a,'reload_s':b,'completion_median_s':statistics.median(a),'reload_median_s':statistics.median(b),'completion_range_s':[min(a),max(a)],'reload_range_s':[min(b),max(b)]}
  result['summary']={label:stats(label) for label in ['baseline','candidate']};result['state']='PASS_MEASURED_RESTORED_NOT_PROMOTED'
 except BaseException as error:
  result.update(state='FAIL',error=repr(error));save(out/'receipt.json',result)
  try:result['rollback']=restore(baseline);result['state']='FAIL_RESTORED'
  except BaseException as error:result['rollback_error']=repr(error)
  raise
 finally:result['finished_epoch']=time.time();save(out/'receipt.json',result)
if __name__=='__main__':main()
