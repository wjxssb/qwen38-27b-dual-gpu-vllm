"""Single-variable recovery A/B, no startup observer in either runtime."""
from common import *
import lifecycle,statistics
from bench import completion
from fault_test import fault_test
from safe_restore import restore
from loaded_identity import collect

def functional(candidate,out):
 h=module('historical_functional',NV/'mtp-k3-closeout-20260914/host-client/prod_http_harness.py');plan=h.build_plan()
 cases=next(v for v in plan.values() if isinstance(v,list) and v and isinstance(v[0],dict) and 'payload' in v[0]);rows=[]
 for caseid in ['A_geometry_repeat1','A_geometry_repeat2','tool_simple_arguments_json','tool_nested_object_json','text_after_shape_switch']:
  case=next(c for c in cases if c['id']==caseid);a,_,_=assert_idle(candidate)
  payload=dict(case['payload'],model='qwen38-27b-dense',stream=False,cache_salt='renderer-'+str(time.time_ns()));payload.pop('stream_options',None)
  req=urllib.request.Request('http://127.0.0.1:18080/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=300) as r:response=json.load(r)
  checks=h.assess(case,200,response);record={'case':caseid,'instance':a['instance_id'],'response':response,'assessment':checks,'asset_sha256':case.get('asset_sha256')};save(out/(caseid+'.json'),record)
  assert checks['mechanical_status']=='PASS',checks
  if case['kind']=='vision':
   answer=h.strict_json(response['choices'][0]['message']['content']);assert answer==case['expected'][0]['expected'],answer
   record['semantic_status']='PASS_EXACT_GEOMETRY'
  elif case['kind']=='tools':record['semantic_status']='PASS_EXACT_SCHEMA_AND_ARGUMENTS'
  else:record['semantic_status']='PASS_EXACT_STATE_TEXT'
  assert identity(candidate)[0]['instance_id']==a['instance_id'];rows.append(record);save(out/(caseid+'.json'),record)
 return rows

def main():
 baseline=lifecycle.accepted();candidate=NV/'candidate-v43-renderer-overlap-20260923';out=Q/'recovery/evidence/renderer-campaign';out.mkdir()
 result={'state':'RUNNING','started_epoch':time.time(),'baseline':str(baseline),'candidate':str(candidate),'design':'A3 then B3 real 66K in-flight worker loss; every replacement fresh, fixed cache provenance, polling-phase aligned. Descriptive n=3; block order disclosed. No startup observer in either runtime.','faults':{},'inference':{}};save(out/'receipt.json',result)
 try:
  assert lifecycle.selected()==baseline
  for label,runtime in [('baseline',baseline),('candidate',candidate)]:
   if lifecycle.selected()!=runtime:lifecycle.activate(runtime)
   path=out/label;path.mkdir();collect(runtime,path);result['inference'][label]=[]
   for count in [1024,16384,67584,258048]:
    row=completion(runtime,count,256,path/f'{label}-{count}-256');result['inference'][label].append(row);save(out/'receipt.json',result)
   result[label+'_functional']=functional(runtime,path/'functional');save(out/'receipt.json',result)
   if label=='candidate':
    for a,b in zip(result['inference']['baseline'],result['inference']['candidate']):assert a['count']==b['count'] and a['output_hash']==b['output_hash'] and a['raw_logits']==b['raw_logits'],('Numerical mismatch',a['count'])
    result['numerics']='PASS_EXACT_TOKEN_HASH_AND_SELECTED_RAW_LOGITS'
   result['faults'][label]=[]
   for repetition in range(3):
    receipt=fault_test(runtime,f'renderer-{label}-fault-{repetition+1}',after_monitor_poll=True,inflight=True);result['faults'][label].append(receipt);save(out/'receipt.json',result)
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
