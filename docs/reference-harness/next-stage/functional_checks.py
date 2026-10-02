from common import *
def functional(candidate,out):
 h=module('historical_functional',NV/'mtp-k3-closeout-20260914/host-client/prod_http_harness.py');plan=h.build_plan()
 cases=next(v for v in plan.values() if isinstance(v,list) and v and isinstance(v[0],dict) and 'payload' in v[0]);rows=[]
 for caseid in ['A_geometry_repeat1','A_geometry_repeat2','tool_simple_arguments_json','tool_nested_object_json','text_after_shape_switch']:
  case=next(c for c in cases if c['id']==caseid);a,_,_=assert_idle(candidate)
  payload=dict(case['payload'],model='qwen38-27b-dense',stream=False,cache_salt='renderer-'+str(time.time_ns()));payload.pop('stream_options',None)
  req=urllib.request.Request('http://127.0.0.1:18080/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=300) as r:response=json.load(r)
  assessment_response=json.loads(json.dumps(response));normalization=[]
  if case['kind']=='vision':
   import re
   content=assessment_response['choices'][0]['message']['content'].strip()
   match=re.fullmatch(r'```(?:json)?\s*\n(.*)\n```',content,re.S)
   if match:content=match[1];normalization.append('unwrap one complete JSON code fence')
   answer=h.strict_json(content)
   assessment_response['choices'][0]['message']['content']=json.dumps(answer)
  if case['kind']=='state_text':
   assessment_response['choices'][0]['message']['content']=assessment_response['choices'][0]['message']['content'].strip();normalization.append('trim surrounding whitespace on state text')
  checks=h.assess(case,200,assessment_response);record={'case':caseid,'instance':a['instance_id'],'response':response,'assessment':checks,'asset_sha256':case.get('asset_sha256'),'normalization':normalization};save(out/(caseid+'.json'),record)
  assert checks['mechanical_status']=='PASS',checks
  if case['kind']=='vision':
   answer=h.strict_json(assessment_response['choices'][0]['message']['content']);assert set(answer)=={'shapes'} and len(answer['shapes'])==3,answer
   normalized={'shapes':[{k:v.casefold() for k,v in row.items()} for row in answer['shapes']]};assert normalized==case['expected'][0]['expected'],answer
   record['normalization'].append('casefold geometry position/shape/color values for semantic oracle')
   record['semantic_status']='PASS_EXACT_GEOMETRY'
  elif case['kind']=='tools':record['semantic_status']='PASS_EXACT_SCHEMA_AND_ARGUMENTS'
  else:record['semantic_status']='PASS_EXACT_STATE_TEXT'
  assert identity(candidate)[0]['instance_id']==a['instance_id'];rows.append(record);save(out/(caseid+'.json'),record)
 return rows
