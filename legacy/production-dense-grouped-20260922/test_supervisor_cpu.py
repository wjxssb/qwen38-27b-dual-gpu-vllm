#!/usr/bin/env python3
"""Mocked state-machine checks; no Docker, service, GPU or production access."""
import copy,hashlib,importlib.util,json,os,signal,tempfile,time,unittest
from unittest.mock import patch
from pathlib import Path
P=Path(__file__).with_name('supervisor.py');spec=importlib.util.spec_from_file_location('supervisor_test_target',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class FakeIO:
 def __init__(self,run):self.run=run;self.row=None;self.calls=[];self.journal='-- No entries --\n';self.outcomes={};self.inspect_error=None;self.monitor_error=False;self.compute='';self.gpu='GPU-a, 100, 16000, 0, 35, 12, 210, None, DRIVER\nGPU-b, 100, 16000, 0, 35, 12, 210, None, DRIVER\n'
 def dispatch(self,kind,args,timeout=45):
  state=json.loads((self.run/'intent.json').read_text());assert state['state'] in ['CREATE_INTENT_DURABLE','START_INTENT_DURABLE','STOP_INTENT_DURABLE']
  self.calls.append((kind,args));self.outcomes[kind]={'returncode':None,'timed_out':False,'outcome':'PENDING'}
 def poll(self):return self.outcomes
 def inspect(self,name):
  if self.inspect_error:raise RuntimeError(self.inspect_error)
  return self.row
 def read(self,args,timeout=15):
  if self.monitor_error:raise RuntimeError('telemetry unavailable')
  if args[0]=='journalctl':return self.journal
  if args[0]=='nvidia-smi':return self.compute if args[1].startswith('--query-compute-apps=') else self.gpu
  return ''
 def append(self,name,row):self.calls.append(('evidence',name,row))
class StateMachine(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'logs').mkdir()
  self.intent={'instance_id':'b'*32,'run_id':'b'*32,'container_name':'nvidia-dense-'+'b'*32,'container_id':None,'container_labels':{m.LABEL+'.campaign':'/campaign',m.LABEL+'.run':'b'*32},'image':'sha256:'+'e'*64,'model_revision':'d'*40,'runtime_manifest_sha256':'c'*64,'boot_id':'boot','logprobs_mode':'raw_logits','speculative_decoding':False,'startup_receipt':str(self.root/'startup-receipt.json'),'port':18097,'model_alias':'candidate','profile':'graph-prefix','kernel_cursor':'CURSOR0','cleanup_verified':False,'expected_gpu_uuids':['GPU-a','GPU-b'],'expected_driver':'DRIVER','idle_gpu_baseline':[{'uuid':u,'used_mib':100,'free_mib':16000} for u in ['GPU-a','GPU-b']]}
  self.io=FakeIO(self.root);self.s=m.Supervisor(self.intent,self.io,self.root,self.root/'active.json');self.now=100;self.s.clock=lambda:self.now
 def tearDown(self):self.tmp.cleanup()
 def row(self,state='created'):
  return {'Id':'a'*64,'Image':self.intent['image'],'Name':'/'+self.intent['container_name'],'Config':{'Labels':dict(self.intent['container_labels'])},'State':{'Running':state=='running','Status':state,'Pid':0,'ExitCode':0,'OOMKilled':False,'StartedAt':'0001-01-01T00:00:00Z' if state=='created' else '2026-09-13T01:00:00Z','FinishedAt':'2026-09-13T02:00:00Z' if state=='exited' else '0001-01-01T00:00:00Z'}}
 def begin(self):self.s.begin(m.DOCKER+['create','--name',self.intent['container_name'],'image'])
 def observe_exit(self):
  self.s.tick();self.now+=30;self.s.tick();self.now+=30;self.s.tick()
 def test_intent_fsync_precedes_create(self):self.begin();self.assertEqual(self.io.calls[0][0],'create');self.assertTrue((self.root/'intent.json').exists())
 def test_create_timeout_absence_keeps_claim_and_does_not_retry(self):
  self.begin();self.io.outcomes['create']={'returncode':-15,'timed_out':True,'outcome':'UNKNOWN'};self.s.tick();self.s.tick()
  self.assertFalse(self.s.released);self.assertEqual(sum(c[0]=='create' for c in self.io.calls),1);self.assertIn('QUARANTINE',self.s.intent['state'])
 def test_late_create_reconciles_exact_name_then_starts_once(self):
  self.begin();self.s.tick();self.io.row=self.row();self.s.tick();self.s.tick()
  self.assertEqual(sum(c[0]=='start' for c in self.io.calls),1);self.assertEqual(self.s.intent['container_id'],'a'*64);self.assertFalse(self.s.released)
 def test_foreign_labels_never_stopped(self):
  self.begin();self.io.row=self.row('running');self.io.row['Config']['Labels'][m.LABEL+'.campaign']='foreign';self.s.tick()
  self.assertFalse(self.s.released);self.assertFalse(any(c[0]=='stop' for c in self.io.calls));self.assertTrue(self.s.quarantine)
 def test_docker_outage_quarantines_then_exact_cleanup(self):
  self.begin();self.io.inspect_error='daemon unavailable';self.s.tick();self.assertFalse(self.s.released)
  self.io.inspect_error=None;self.io.row=self.row();self.observe_exit();self.assertTrue(self.s.released);self.assertFalse(any(c[0]=='start' for c in self.io.calls))
 def test_uncertain_start_created_state_is_not_exit_proof(self):
  self.begin();self.io.row=self.row();self.s.tick();self.io.outcomes['start']={'returncode':-15,'timed_out':True,'outcome':'UNKNOWN'};self.s.tick();self.assertFalse(self.s.released)
 def test_xid_requests_only_owned_graceful_stop_and_keeps_lease(self):
  self.begin();self.io.row=self.row('running');self.io.journal='NVRM: Xid (PCI:1): 43\n-- cursor: NEXT\n';self.s.tick()
  stop=[c for c in self.io.calls if c[0]=='stop'];self.assertEqual(len(stop),1);self.assertEqual(stop[0][1],m.DOCKER+['stop','--timeout','-1','a'*64]);self.assertFalse(self.s.released)
  receipt=json.loads((self.root/'first-incident.json').read_text());self.assertEqual(receipt['root_cause'],'NOT_ATTRIBUTED');self.assertFalse(receipt['reset'])
 def test_monitor_loss_stops_owned_candidate_without_releasing(self):
  self.begin();self.io.row=self.row('running');self.io.monitor_error=True;self.s.tick();self.assertTrue(any(c[0]=='stop' for c in self.io.calls));self.assertFalse(self.s.released)
 def test_empty_journal_retains_cursor_without_false_fault(self):
  self.begin();self.io.row=self.row();self.s.tick();self.assertEqual(self.s.cursor,'CURSOR0');self.assertFalse(self.s.stop_requested)
 def test_confirmed_exit_preserves_container_and_releases(self):
  self.begin();self.io.row=self.row();self.s.tick();self.io.row=self.row('exited');self.observe_exit();self.assertTrue(self.s.released);self.assertTrue(self.s.intent['cleanup_verified']);self.assertFalse(any(c[0] in ['rm','kill'] for c in self.io.calls))
 def test_health_never_sets_mapping_or_ready(self):
  self.begin();self.io.row=self.row('running');self.s.tick();r=json.loads((self.root/'startup-receipt.json').read_text());self.assertEqual(r['state'],'RUNNING_UNQUALIFIED');self.assertFalse(r['runtime_mapping_recorded']);self.assertEqual(r['container_id'],'a'*64)
 def test_stop_before_dispatch_creates_no_container(self):self.s.stop_requested=True;self.begin();self.assertTrue(self.s.released);self.assertFalse(any(c[0]=='create' for c in self.io.calls))
 def test_resume_created_intent_never_starts(self):self.s.stop_requested=True;self.io.row=self.row();self.observe_exit();self.assertTrue(self.s.released);self.assertFalse(any(c[0]=='start' for c in self.io.calls))
 def test_exit_requires_complete_post_exit_observation(self):
  self.s.stop_requested=True;self.io.row=self.row();self.s.tick();self.now+=59;self.s.tick()
  self.assertFalse(self.s.released);self.assertFalse(self.s.intent.get('cleanup_verified'))
  self.now+=1;self.s.tick();self.assertTrue(self.s.released)
 def test_nonzero_exit_is_failed_after_safe_cleanup(self):
  self.s.intent['start_dispatched']=True;self.io.row=self.row('exited');self.io.row['State']['ExitCode']=137;self.observe_exit()
  self.assertTrue(self.s.released);self.assertEqual(self.s.outcome_code(),2);self.assertEqual(self.s.intent['qualification'],'NOT_PROVEN');self.assertIn('137',self.s.intent['failure_reason'])
 def test_oom_is_failed_even_if_exit_code_zero(self):
  self.s.intent['start_dispatched']=True;self.io.row=self.row('exited');self.io.row['State']['OOMKilled']=True;self.observe_exit()
  self.assertTrue(self.s.released);self.assertEqual(self.s.outcome_code(),2)
 def test_incident_revokes_ready_preserving_historical_receipt(self):
  self.io.row=self.row('running');self.s.tick();ready=json.loads((self.root/'startup-receipt.json').read_text());ready.update(state='READY',runtime_mapping_recorded=True);m.durable(self.root/'startup-receipt.json',ready)
  self.s.incident('test fault');current=json.loads((self.root/'startup-receipt.json').read_text())
  self.assertEqual(current['state'],'INCIDENT_UNQUALIFIED');self.assertFalse(current['runtime_mapping_recorded']);self.assertEqual(json.loads((self.root/'qualification-before-revocation.json').read_text()),ready);self.assertEqual(self.s.outcome_code(),2)
 def test_stop_revokes_ready_before_stop_dispatch(self):
  self.io.row=self.row('running');self.s.tick();ready=json.loads((self.root/'startup-receipt.json').read_text());ready.update(state='READY',runtime_mapping_recorded=True);m.durable(self.root/'startup-receipt.json',ready)
  self.s.stop_requested=True;self.s.tick();self.assertEqual(json.loads((self.root/'startup-receipt.json').read_text())['state'],'STOPPING_UNQUALIFIED')
 def test_monitor_loss_after_exit_retains_claim_and_restarts_window(self):
  self.s.stop_requested=True;self.io.row=self.row();self.s.tick();self.now+=30;self.io.monitor_error=True;self.s.tick();self.assertFalse(self.s.released)
  self.io.monitor_error=False;self.now+=30;self.s.tick();self.assertFalse(self.s.released);self.observe_exit();self.assertTrue(self.s.released);self.assertEqual(self.s.outcome_code(),2)
 def test_compute_after_exit_retains_claim(self):
  self.s.stop_requested=True;self.io.row=self.row();self.io.compute='GPU-a, 777, orphan worker';self.observe_exit();self.assertFalse(self.s.released);self.assertIn('compute process remains',self.s.intent['failure_reason'])
 def test_stale_vram_without_compute_after_exit_retains_claim(self):
  self.s.stop_requested=True;self.io.row=self.row();self.io.gpu=self.io.gpu.replace('100, 16000, 0','14000, 2000, 0');self.observe_exit();self.assertFalse(self.s.released);self.assertIn('idle baseline',self.s.intent['failure_reason'])
 def test_unreleased_memory_below_generic_limit_retains_claim(self):
  self.s.stop_requested=True;self.io.row=self.row();self.io.gpu=self.io.gpu.replace('100, 16000, 0','229, 15871, 0');self.observe_exit();self.assertFalse(self.s.released);self.assertIn('128MiB',self.s.intent['failure_reason'])
 def test_explicit_idle_memory_tolerance_boundary_accepted(self):
  self.s.stop_requested=True;self.io.row=self.row();self.io.gpu=self.io.gpu.replace('100, 16000, 0','228, 15872, 0');self.observe_exit();self.assertTrue(self.s.released)
 def test_busy_gpu_without_compute_after_exit_retains_claim(self):
  self.s.stop_requested=True;self.io.row=self.row();self.io.gpu=self.io.gpu.replace('100, 16000, 0','100, 16000, 80');self.observe_exit();self.assertFalse(self.s.released)
 def test_changed_uuid_after_exit_retains_claim(self):
  self.s.stop_requested=True;self.io.row=self.row();self.io.gpu=self.io.gpu.replace('GPU-b','GPU-c');self.observe_exit();self.assertFalse(self.s.released)
 def test_recovery_action_after_exit_retains_claim(self):
  self.s.stop_requested=True;self.io.row=self.row();self.io.gpu=self.io.gpu.replace('None','Reset');self.observe_exit();self.assertFalse(self.s.released)
 def test_changed_driver_after_exit_retains_claim(self):
  self.s.stop_requested=True;self.io.row=self.row();self.io.gpu=self.io.gpu.replace('DRIVER','OTHER');self.observe_exit();self.assertFalse(self.s.released)
 def test_large_sample_gap_restarts_window(self):
  self.s.stop_requested=True;self.io.row=self.row();self.s.tick();self.now+=90;self.s.tick();self.assertEqual(self.s.intent['post_exit_observation']['elapsed_seconds'],0);self.assertFalse(self.s.released)
 def test_fault_outcome_survives_successful_container_exit(self):
  self.s.intent['start_dispatched']=True;self.io.row=self.row('exited');self.s.incident('new kernel fault');self.observe_exit();self.assertTrue(self.s.released);self.assertEqual(self.s.outcome_code(),2)
 def test_failed_kernel_append_does_not_advance_cursor(self):
  self.io.row=self.row('running');self.io.journal='quiet\n-- cursor: NEXT\n';original=self.io.append
  def reject(name,row):
   if name=='kernel.jsonl':raise OSError('disk full')
   original(name,row)
  self.io.append=reject;self.s.tick();self.assertEqual(self.s.cursor,'CURSOR0');self.assertFalse(self.s.released)
 def test_local_append_fsyncs_file_and_directory(self):
  local=m.LocalIO(self.root)
  with patch.object(m.os,'fsync',wraps=os.fsync) as fsync:local.append('proof.jsonl',{'proof':True});self.assertEqual(fsync.call_count,2)
 def test_rotation_preserves_old_segment(self):
  p=self.root/'logs'/'proof.jsonl';p.write_bytes(b'x'*(8*1024**2+1));local=m.LocalIO(self.root);local.append('proof.jsonl',{'new':True})
  segments=list((self.root/'logs').glob('proof.jsonl.segment-*'));self.assertEqual(len(segments),1);self.assertEqual(segments[0].stat().st_size,8*1024**2+1)
 def mapping_attestation(self):
  self.io.row=self.row('running')
  if not self.s.intent.get('running_observed'):self.s.tick()
  identity={k:self.s.intent[k] for k in m.IDENTITY_FIELDS}
  attestation={'schema':1,'kind':'NVIDIA_RUNTIME_MAPPING_ATTESTATION','scope':'RUNTIME_COLLECTION_ONLY','runtime_mapping_recorded':True,'numerical_qualification':'NOT_PROVEN','identity':identity,'request_id':'mapping-request-1','collector_sha256':'f'*64,'artifacts':[]}
  for rank in [0,1]:
   report={'schema':1,'scope':'ACTUAL_RUNTIME_OBSERVATION_ONLY','status':'COLLECTED_UNQUALIFIED','numerical_correctness':'NOT_PROVEN','p2p_data_integrity':'NOT_PROVEN','identity':{k:self.s.intent[k] for k in m.COLLECTOR_IDENTITY_FIELDS},'request_id':attestation['request_id'],'collector_sha256':attestation['collector_sha256'],'rank':rank,'local_rank':rank,'tensor_parallel':{'rank':rank,'rank_in_group':rank,'world_size':2,'ranks':[0,1]},'device':{'uuid':['GPU-a','GPU-b'][rank]},'effective_config':{'speculative_config_is_none':True,'parallel':{'tensor_parallel_size':2}},'parameters':{'weight':{'dtype':'torch.uint8'}},'modules':{'model':{'class':'model'}}}
   path=self.root/f'mapping-rank-{rank}.json';self.sealed_json(path,report)
   attestation['artifacts'].append({'rank':rank,'path':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
  self.sealed_json(self.root/'mapping-attestation.json',attestation)
  return attestation
 def sealed_json(self,path,value):
  if path.exists():path.chmod(0o644)
  path.write_text(json.dumps(value)+'\n');path.chmod(0o444)
 def mutate_attestation(self,change):
  path=self.root/'mapping-attestation.json';record=json.loads(path.read_text());change(record);self.sealed_json(path,record)
 def mutate_rank(self,rank,change,reseal_hash=True):
  path=self.root/f'mapping-rank-{rank}.json';record=json.loads(path.read_text());change(record);self.sealed_json(path,record)
  if reseal_hash:
   def update(a):a['artifacts'][rank].update(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
   self.mutate_attestation(update)
 def assert_mapping_rejected(self):
  self.s.tick();receipt=json.loads((self.root/'startup-receipt.json').read_text())
  self.assertNotEqual(receipt['state'],'READY');self.assertFalse(receipt['runtime_mapping_recorded']);self.assertTrue(self.s.stop_requested);self.assertIn('mapping',self.s.intent['failure_reason']);self.assertTrue(any(c[0]=='stop' for c in self.io.calls))
 def test_valid_attestation_publishes_collection_ready_once(self):
  self.mapping_attestation();self.s.tick();receipt=json.loads((self.root/'startup-receipt.json').read_text())
  self.assertEqual(receipt['state'],'READY');self.assertTrue(receipt['runtime_mapping_recorded']);self.assertEqual(receipt['numerical_qualification'],'NOT_PROVEN');self.assertEqual(self.s.intent['state'],'RUNNING_UNQUALIFIED')
  self.assertTrue(self.s.intent['mapping_ready_published']);self.assertTrue(self.s.intent['runtime_mapping_recorded'])
  first=(self.root/'startup-receipt.json').stat().st_mtime_ns;self.s.tick();self.assertEqual(first,(self.root/'startup-receipt.json').stat().st_mtime_ns)
 def test_ready_without_validated_mapping_argument_refused(self):
  self.io.row=self.row('running');self.s.tick()
  with self.assertRaisesRegex(RuntimeError,'requires'):self.s.identity('READY')
 def test_mapping_attestation_full_identity_bound(self):
  for field in m.IDENTITY_FIELDS:
   with self.subTest(field=field):
    self.mapping_attestation();self.mutate_attestation(lambda a:a['identity'].update({field:'different'}))
    with self.assertRaisesRegex(RuntimeError,'full identity'):m.validate_mapping_attestation(self.root,self.s.intent)
 def test_mapping_cannot_claim_numerical_pass(self):
  self.mapping_attestation();self.mutate_attestation(lambda a:a.update(numerical_qualification='PASS'));self.assert_mapping_rejected()
 def test_mapping_unvalidated_top_level_claim_refused(self):
  self.mapping_attestation();self.mutate_attestation(lambda a:a.update(production_verified=True));self.assert_mapping_rejected()
 def test_mapping_mutable_file_refused(self):
  self.mapping_attestation();(self.root/'mapping-rank-0.json').chmod(0o644);self.assert_mapping_rejected()
 def test_mapping_symlink_file_refused(self):
  self.mapping_attestation();path=self.root/'mapping-rank-0.json';saved=path.with_name('saved.json');path.rename(saved);path.symlink_to(saved);self.assert_mapping_rejected()
 def test_mapping_path_escape_refused(self):
  self.mapping_attestation();self.mutate_attestation(lambda a:a['artifacts'][0].update(path='../other.json'));self.assert_mapping_rejected()
 def test_mapping_wrong_hash_refused(self):
  self.mapping_attestation();self.mutate_rank(0,lambda r:r['parameters'].update(other={}),reseal_hash=False);self.assert_mapping_rejected()
 def test_mapping_duplicate_rank_refused(self):
  self.mapping_attestation();self.mutate_attestation(lambda a:a['artifacts'][1].update(rank=0));self.assert_mapping_rejected()
 def test_mapping_wrong_actual_rank_world_size_refused(self):
  self.mapping_attestation();self.mutate_rank(1,lambda r:r['tensor_parallel'].update(world_size=1));self.assert_mapping_rejected()
 def test_mapping_wrong_rank_identity_refused(self):
  self.mapping_attestation();self.mutate_rank(1,lambda r:r['identity'].update(boot_id='other'));self.assert_mapping_rejected()
 def test_mapping_rank_request_and_source_hash_bound(self):
  self.mapping_attestation()
  for field in ['request_id','collector_sha256']:
   with self.subTest(field=field):
    path=self.root/'mapping-rank-1.json';original=json.loads(path.read_text());self.mutate_rank(1,lambda r:r.update({field:'different'}))
    with self.assertRaisesRegex(RuntimeError,'report scope/identity'):m.validate_mapping_attestation(self.root,self.s.intent)
    self.mutate_rank(1,lambda r:r.update(original))
 def test_mapping_rank_numerical_pass_refused(self):
  self.mapping_attestation();self.mutate_rank(1,lambda r:r.update(numerical_correctness='PASS'));self.assert_mapping_rejected()
 def test_mapping_duplicate_gpu_refused(self):
  self.mapping_attestation();self.mutate_rank(1,lambda r:r['device'].update(uuid='GPU-a'));self.assert_mapping_rejected()
 def test_mapping_speculative_runtime_refused(self):
  self.mapping_attestation();self.mutate_rank(1,lambda r:r['effective_config'].update(speculative_config_is_none=False));self.assert_mapping_rejected()
 def test_mapping_missing_real_collection_refused(self):
  self.mapping_attestation();self.mutate_rank(1,lambda r:r.update(parameters={}));self.assert_mapping_rejected()
 def test_stop_before_attestation_never_publishes_ready(self):
  self.mapping_attestation();m.durable(self.root/'stop-request.json',{'instance_id':self.s.intent['instance_id']});self.s.tick()
  self.assertNotEqual(json.loads((self.root/'startup-receipt.json').read_text())['state'],'READY');self.assertFalse(self.s.intent.get('mapping_ready_published',False))
 def test_failure_before_attestation_never_publishes_ready(self):
  self.mapping_attestation();self.s.incident('prior fault');self.s.tick();self.assertFalse(self.s.intent.get('mapping_ready_published',False))
 def test_fresh_monitor_fault_forbids_mapping_ready(self):
  self.mapping_attestation();self.io.journal='NVRM: Xid 8\n-- cursor: NEXT\n';self.s.tick()
  self.assertTrue(self.s.stop_requested);self.assertFalse(self.s.intent.get('mapping_ready_published',False));self.assertNotEqual(json.loads((self.root/'startup-receipt.json').read_text())['state'],'READY')
 def test_fresh_monitor_loss_forbids_mapping_ready(self):
  self.mapping_attestation();self.io.monitor_error=True;self.s.tick()
  self.assertTrue(self.s.stop_requested);self.assertFalse(self.s.intent.get('mapping_ready_published',False));self.assertFalse(self.s.intent['runtime_mapping_recorded'])
 def test_not_running_unqualified_state_cannot_publish(self):
  self.mapping_attestation();self.s.persist('QUARANTINE_RECONCILING');self.s.tick();self.assertFalse(self.s.intent.get('mapping_ready_published',False))
 def test_fresh_inspect_exit_forbids_mapping_ready(self):
  self.mapping_attestation();calls=[];original=self.io.inspect
  def inspect(name):
   calls.append(name);return self.row('exited') if len(calls)>1 else original(name)
  self.io.inspect=inspect;self.assert_mapping_rejected()
 def test_stop_during_ready_publication_revokes_before_return(self):
  self.mapping_attestation();original=m.durable
  def stop_during_write(path,value):
   original(path,value)
   if Path(path).name=='startup-receipt.json' and value.get('state')=='READY':original(self.root/'stop-request.json',{'instance_id':self.s.intent['instance_id']})
  with patch.object(m,'durable',side_effect=stop_during_write):self.s.tick()
  self.assertTrue(self.s.stop_requested);self.assertNotEqual(json.loads((self.root/'startup-receipt.json').read_text())['state'],'READY');self.assertFalse(self.s.intent['runtime_mapping_recorded'])
  self.assertEqual(json.loads((self.root/'qualification-before-revocation.json').read_text())['state'],'READY')
 def test_signal_during_ready_publication_revokes_before_return(self):
  self.mapping_attestation();original=m.durable;old=signal.signal(signal.SIGTERM,lambda signum,frame:setattr(self.s,'stop_requested',True))
  def signal_during_write(path,value):
   original(path,value)
   if Path(path).name=='startup-receipt.json' and value.get('state')=='READY':os.kill(os.getpid(),signal.SIGTERM)
  try:
   with patch.object(m,'durable',side_effect=signal_during_write):self.s.tick()
  finally:signal.signal(signal.SIGTERM,old)
  self.assertTrue(self.s.stop_requested);self.assertNotEqual(json.loads((self.root/'startup-receipt.json').read_text())['state'],'READY');self.assertFalse(self.s.intent['runtime_mapping_recorded'])
 def test_published_mapping_mutation_revokes_ready(self):
  self.mapping_attestation();self.s.tick();self.mutate_rank(1,lambda r:r.update(pid=999),reseal_hash=False);self.s.tick()
  self.assertTrue(self.s.stop_requested);self.assertFalse(self.s.intent['runtime_mapping_recorded']);self.assertNotEqual(json.loads((self.root/'startup-receipt.json').read_text())['state'],'READY')
 def test_exit_revokes_mapping_before_post_exit_window(self):
  self.mapping_attestation();self.s.tick();self.s.intent['start_dispatched']=True;self.io.row=self.row('exited');self.s.tick()
  self.assertFalse(self.s.released);self.assertFalse(self.s.intent['runtime_mapping_recorded']);self.assertEqual(json.loads((self.root/'startup-receipt.json').read_text())['state'],'EXITED_UNQUALIFIED')
if __name__=='__main__':unittest.main()
