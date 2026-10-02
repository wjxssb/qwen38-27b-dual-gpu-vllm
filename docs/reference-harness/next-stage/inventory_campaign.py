from common import *
import lifecycle,statistics
from bench import completion
from fault_test import fault_test
from safe_restore import restore
from loaded_identity import collect
from functional_checks import functional

def main():
 fallback=lifecycle.accepted();candidate=NV/'candidate-v46-renderer-inventory-cache-20260923';out=Q/'recovery/evidence/inventory-cache-campaign';out.mkdir()
 prior_path=Q/'recovery/evidence/renderer-campaign-r2/receipt.json';prior=read(prior_path);assert prior['state']=='PASS_MEASURED_RESTORED_NOT_PROMOTED'
 result={'state':'RUNNING','started_epoch':time.time(),'candidate':str(candidate),'manifest':sha(candidate/'MANIFEST.json'),'parent_comparison':{'path':str(prior_path),'sha256':sha(prior_path),'baseline':'candidate-v45 renderer overlap, three fresh real replacements'},'single_variable':'Sealed model inventory cache with full SHA fallback; all renderer/model/GPU/ownership paths identical to qualified parent.','inference':[],'faults':[]};save(out/'receipt.json',result)
 try:
  lifecycle.activate(candidate);collect(candidate,out)
  for count in [1024,16384,67584,258048]:
   row=completion(candidate,count,256,out/f'cached-{count}-256');reference=next(r for r in prior['inference']['candidate'] if r['count']==count)
   assert row['output_hash']==reference['output_hash'] and row['raw_logits']==reference['raw_logits'],('Numerical mismatch',count)
   result['inference'].append(row);save(out/'receipt.json',result)
  result['functional']=functional(candidate,out/'functional');save(out/'receipt.json',result)
  for repetition in range(3):
   receipt=fault_test(candidate,f'inventory-cache-fault-{repetition+1}',after_monitor_poll=True,inflight=True);result['faults'].append(receipt);save(out/'receipt.json',result)
  result['rollback']=restore(fallback)
  result['summary']={'completion_s':[r['latencies']['first_successful_completion'] for r in result['faults']],'reload_s':[r['latencies']['model_ready']-r['latencies']['new_supervisor_started'] for r in result['faults']]}
  result['state']='PASS_MEASURED_RESTORED_NOT_PROMOTED'
 except BaseException as error:
  result.update(state='FAIL',error=repr(error));save(out/'receipt.json',result)
  try:result['rollback']=restore(fallback);result['state']='FAIL_RESTORED'
  except BaseException as error:result['rollback_error']=repr(error)
  raise
 finally:result['finished_epoch']=time.time();save(out/'receipt.json',result)
if __name__=='__main__':main()
