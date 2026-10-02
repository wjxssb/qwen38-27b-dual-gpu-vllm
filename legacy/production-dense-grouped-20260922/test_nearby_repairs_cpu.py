#!/usr/bin/env python3
"""Regression cases from Xid79, monitor loss, quarantine spam and version copies."""
import copy, hashlib, json, math, os, subprocess, tempfile, time, unittest
from pathlib import Path
from unittest.mock import patch
import test_supervisor_cpu as existing
import test_launcher_cpu as launch_existing
m=existing.m;l=launch_existing.m

class SupervisorRegressions(unittest.TestCase):
 setUp=existing.StateMachine.setUp
 tearDown=existing.StateMachine.tearDown
 row=existing.StateMachine.row
 begin=existing.StateMachine.begin
 def test_first_fault_excerpt_survives_long_backtrace(self):
  self.begin();self.io.row=self.row('running')
  self.io.journal='NVRM: Xid (PCI:0000:03:00): 79, GPU has fallen off the bus.\n'+'backtrace filler\n'*3000+'-- cursor: NEXT\n'
  self.s.tick();first=json.loads((self.root/'first-incident.json').read_text())
  self.assertIn('Xid',first['reason']);self.assertIn('79',first['reason']);self.assertEqual(first['evidence']['text_sha256'],hashlib.sha256(self.io.journal.encode()).hexdigest())
  self.assertTrue(self.s.intent['hardware_fault_latched']);self.assertFalse(self.s.released)
 def test_nvml_loss_does_not_hide_final_container_traceback(self):
  self.begin();self.io.row=self.row('running');original=self.io.read;calls=[]
  def read(args,timeout=15):
   calls.append(args)
   if args[0]=='nvidia-smi':raise RuntimeError('NVML unavailable')
   if 'logs' in args:return 'CUDA error: device lost\nfinal stack trace\n'
   return original(args,timeout)
  self.io.read=read;self.s.tick()
  logs=[c for c in self.io.calls if c[0:2]==('evidence','container.jsonl')]
  self.assertIn('final stack trace',logs[0][2]['text']);self.assertTrue(any('logs' in x for x in calls))
  self.assertTrue(any(c[0]=='stop' for c in self.io.calls));self.assertFalse(self.s.released)
 def test_journal_failure_does_not_hide_gpu_or_container_evidence(self):
  self.begin();self.io.row=self.row('running');original=self.io.read
  def read(args,timeout=15):
   if args[0]=='journalctl':raise RuntimeError('journal missing')
   return original(args,timeout)
  self.io.read=read;self.s.tick()
  channels={c[1] for c in self.io.calls if c[0]=='evidence'}
  self.assertTrue({'gpu.jsonl','container.jsonl'}<=channels);self.assertFalse(self.s.released)
 def test_repeated_incident_is_counted_without_rewriting_first_or_duplicating_log(self):
  self.s.intent['container_id']='a'*64
  for _ in range(100):self.s.incident('same fault')
  events=[c for c in self.io.calls if c[0:2]==('evidence','incidents.jsonl')]
  self.assertEqual(len(events),1);self.assertEqual(sum(self.s.intent['incident_counts'].values()),100)
  before=(self.root/'first-incident.json').read_bytes();self.s.incident('later distinct fault')
  self.assertEqual((self.root/'first-incident.json').read_bytes(),before)
  self.assertEqual(len([c for c in self.io.calls if c[0:2]==('evidence','incidents.jsonl')]),2)
 def test_hardware_latch_cannot_release_after_idle_appears(self):
  self.s.intent['start_dispatched']=True;self.io.row=self.row('exited');self.s.intent['container_id']='a'*64
  self.s.incident('new kernel fault: NVRM: Xid 79',hardware_fault=True)
  for i in range(30):self.now+=30;self.s.tick()
  self.assertFalse(self.s.released);self.assertFalse(self.s.intent['cleanup_verified']);self.assertEqual(self.s.intent['state'],'QUARANTINE_HARDWARE_FAULT_LATCHED')
 def test_changed_gpu_identity_latches_even_if_pair_returns(self):
  self.s.intent['start_dispatched']=True;self.io.row=self.row('exited');self.io.gpu=self.io.gpu.replace('GPU-b','GPU-other');self.s.tick()
  self.io.gpu=self.io.gpu.replace('GPU-other','GPU-b')
  for i in range(5):self.now+=30;self.s.tick()
  self.assertTrue(self.s.intent['hardware_fault_latched']);self.assertFalse(self.s.released)
 def test_resumed_legacy_kernel_incident_stays_latched(self):
  self.s.intent['container_id']='a'*64;m.durable(self.root/'first-incident.json',{'reason':'new kernel fault: truncated backtrace','reset':False})
  resumed=m.Supervisor(self.s.intent,self.io,self.root,self.root/'active.json',stop_requested=True)
  self.assertTrue(resumed.hardware_fault_latched)
 def test_quarantine_rechecks_back_off_but_lease_and_heartbeat_persist(self):
  self.s.intent['start_dispatched']=True;self.io.row=self.row('exited');self.io.monitor_error=True
  polls=[];original=self.io.inspect
  def inspect(name):polls.append(self.now);return original(name)
  self.io.inspect=inspect
  for i in range(61):self.now=100+i*2;self.s.tick()
  self.assertLess(len(polls),12);self.assertGreater(len(polls),3)
  self.assertFalse(self.s.released);self.assertFalse(self.s.intent['cleanup_verified'])
  self.assertTrue(all(b-a<=30 for a,b in zip(polls,polls[1:])))
  self.assertEqual(len([c for c in self.io.calls if c[0:2]==('evidence','incidents.jsonl')]),1)
 def test_finished_owned_client_handle_is_closed(self):
  local=m.LocalIO(self.root)
  local.dispatch('cpu-only',['/usr/bin/true'])
  local.jobs['cpu-only']['process'].wait(timeout=5);local.poll()
  self.assertTrue(local.jobs['cpu-only']['handle'].closed)

class Reconciliation(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.boot='11111111-1111-1111-1111-111111111111';self.oldboot='22222222-2222-2222-2222-222222222222'
  self.intent={'instance_id':'a'*32,'container_id':'b'*64,'container_name':'nvidia-dense-'+'a'*32,'container_labels':{m.LABEL+'.campaign':str(self.root),m.LABEL+'.run':'a'*32},'image':'sha256:'+'c'*64,'runtime_manifest_sha256':'d'*64,'boot_id':self.oldboot,'pid':123,'supervisor_start_ticks':456,'cleanup_verified':False}
  self.old={'Id':self.intent['container_id'],'Name':'/'+self.intent['container_name'],'Image':self.intent['image'],'Config':{'Labels':self.intent['container_labels']},'State':{'Running':False,'Status':'exited','StartedAt':'2026-09-13T19:00:00Z','FinishedAt':'2026-09-13T19:27:53Z','ExitCode':1,'OOMKilled':False,'Pid':0}}
  self.path=l.reconciliation_path(self.intent,self.boot,self.root);self.path.parent.mkdir(parents=True)
  self.p2p={'status':'VERIFIED_PASS','boot_id':self.boot,'gpus':l.GPU_UUIDS,'driver':'610','manifest_sha256':'f'*64,'kernel':'TEST-KERNEL'}
  p2p_sha=self.write('current_p2p',self.p2p)['sha256']
  csv='\n'.join(u+', 15866, 15, 0, None' for u in l.GPU_UUIDS)
  self.obs={'schema':1,'current_boot_id':self.boot,'old_boot_id':self.oldboot,'old_supervisor_alive_on_recorded_boot':False,'old_supervisor_identity':{k:self.intent[k] for k in ('boot_id','pid','supervisor_start_ticks')},'elapsed_seconds':60,'kernel':'TEST-KERNEL','samples':[{'monotonic_seconds':t,'boot_id':self.boot,'compute_csv':'','gpu_csv':csv,'driver':'610','kernel_fault':False,'kernel_cursor':'CURSOR'+str(t)} for t in (100,120,140,160)],'p2p_receipt_provenance':{'path':'/run/p2p-stable/verified','uid':0,'regular_file':True,'group_or_world_writable':False,'sha256':p2p_sha}}
  self.receipt={'schema':1,'kind':'PRIOR_BOOT_RECONCILIATION','scope':'ALLOW_FRESH_INSTANCE_ON_CURRENT_BOOT_ONLY','old_boot_cleanup':'NOT_PROVEN','old_incident_preserved':True,'current_boot_id':self.boot,'old_identity':{k:self.intent[k] for k in l.PRIOR_BOOT_IDENTITY_FIELDS},'artifacts':{kind:self.write(kind,obj) for kind,obj in [('old_intent',self.intent),('old_container',self.old),('current_observation',self.obs),('current_p2p',self.p2p)]}}
  self.seal_receipt()
 def tearDown(self):self.tmp.cleanup()
 def write(self,name,obj):
  path=self.path.parent/(name+'.json')
  if path.exists():path.chmod(0o600)
  path.write_text(json.dumps(obj)+'\n');path.chmod(0o444)
  return {'path':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 def seal_receipt(self):
  if self.path.exists():self.path.chmod(0o600)
  self.path.write_text(json.dumps(self.receipt)+'\n');self.path.chmod(0o444)
 def validate(self):return l.validate_prior_boot_reconciliation(self.intent,self.boot,self.path,lambda cid:copy.deepcopy(self.old))
 def change(self,kind,obj):self.receipt['artifacts'][kind]=self.write(kind,obj);self.seal_receipt()
 def global_capture(self):
  source={'boot_id':self.boot,'old_boot_id':self.oldboot,'elapsed_seconds':64.1157,'samples':[]}
  self.obs['elapsed_seconds']=source['elapsed_seconds'];self.obs['samples']=[]
  for i,t in enumerate((0.1241,16.104,32.102,48.099,64.1153)):
   gpu='\n'.join(u+', 15, 15866, 0, None, 610' for u in l.GPU_UUIDS)
   source['samples'].append({'epoch':1000+t,'elapsed_seconds':t,'gpu':{'argv':['nvidia-smi','--query-gpu=uuid,memory.used,memory.free,utilization.gpu,gpu_recovery_action,driver_version','--format=csv,noheader,nounits'],'returncode':0,'stdout':gpu},'compute':{'argv':['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],'returncode':0,'stdout':''}})
   self.obs['samples'].append({'epoch':1000+t,'monotonic_seconds':t,'boot_id':self.boot,'gpu_csv':'\n'.join(u+', 15866, 15, 0, None' for u in l.GPU_UUIDS),'compute_csv':'','driver':'610'})
  cursor='s=a;b='+self.boot.replace('-','')+';i=1'
  self.obs['kernel_coverage']={'scope':'FULL_CURRENT_BOOT_AFTER_OBSERVATION','boot_id':self.boot,'captured_epoch':1065,'kernel_cursor':cursor,'command':{'argv':['journalctl','-k','-b','--show-cursor','--no-pager'],'returncode':0,'stdout':'boot quiet\n-- cursor: '+cursor+'\n'}}
  self.receipt['artifacts']['source_observation']=self.write('source_observation',source)
  self.obs['source_observation_sha256']=self.receipt['artifacts']['source_observation']['sha256'];self.change('current_observation',self.obs)
  return source
 def test_actual_format_global_boot_journal_strict_sample_span_accepted(self):
  self.global_capture();self.assertEqual(self.validate()['old_boot_cleanup'],'NOT_PROVEN')
 def test_global_kernel_capture_before_interval_end_rejected(self):
  self.global_capture();self.obs['kernel_coverage']['captured_epoch']=1059;self.change('current_observation',self.obs)
  with self.assertRaisesRegex(RuntimeError,'global kernel coverage'):self.validate()
 def test_global_kernel_wrong_boot_cursor_rejected(self):
  self.global_capture();coverage=self.obs['kernel_coverage'];coverage['command']['stdout']=coverage['command']['stdout'].replace(self.boot.replace('-',''),self.oldboot.replace('-',''));self.change('current_observation',self.obs)
  with self.assertRaisesRegex(RuntimeError,'global kernel fault/cursor'):self.validate()
 def test_global_kernel_xid_rejected(self):
  self.global_capture();self.obs['kernel_coverage']['command']['stdout']='NVRM: Xid 79\n'+self.obs['kernel_coverage']['command']['stdout'];self.change('current_observation',self.obs)
  with self.assertRaisesRegex(RuntimeError,'global kernel fault/cursor'):self.validate()
 def test_adapter_cannot_invent_successful_sample(self):
  self.global_capture();self.obs['samples'][2]['gpu_csv']=self.obs['samples'][2]['gpu_csv'].replace('15866','15867');self.change('current_observation',self.obs)
  with self.assertRaisesRegex(RuntimeError,'differs from raw sample'):self.validate()
 def test_raw_failed_command_cannot_be_relabelled_clean(self):
  source=self.global_capture();source['samples'][2]['gpu']['returncode']=1;self.receipt['artifacts']['source_observation']=self.write('source_observation',source);self.obs['source_observation_sha256']=self.receipt['artifacts']['source_observation']['sha256'];self.change('current_observation',self.obs)
  with self.assertRaisesRegex(RuntimeError,'raw observation command'):self.validate()
 def test_59991_sample_span_is_not_rounded_up(self):
  source=self.global_capture();source['elapsed_seconds']=60.1157;source['samples'][-1]['epoch']=1060.1153;source['samples'][-1]['elapsed_seconds']=60.1153
  self.obs['elapsed_seconds']=60.1157;self.obs['samples'][-1]['epoch']=1060.1153;self.obs['samples'][-1]['monotonic_seconds']=60.1153
  self.receipt['artifacts']['source_observation']=self.write('source_observation',source);self.obs['source_observation_sha256']=self.receipt['artifacts']['source_observation']['sha256'];self.change('current_observation',self.obs)
  with self.assertRaisesRegex(RuntimeError,'timing/gaps'):self.validate()
 def test_legacy_cleanup_true_does_not_mask_recorded_kernel_fault(self):
  intent=copy.deepcopy(self.intent);intent['cleanup_verified']=True
  old=self.root/'candidate-v4'/'runs'/intent['instance_id']/'intent.json';old.parent.mkdir(parents=True);old.write_text(json.dumps(intent))
  (old.parent/'first-incident.json').write_text(json.dumps({'reason':'new kernel fault: truncated'}));self.path.unlink()
  with self.assertRaisesRegex(RuntimeError,'unresolved prior-boot incident'):l.reject_unresolved_campaign_intents(self.boot,self.root)
 def test_foreign_campaign_record_cannot_grant_admission(self):
  intent=copy.deepcopy(self.intent);intent['container_labels'][m.LABEL+'.campaign']='/other'
  old=self.root/'candidate-v4'/'runs'/intent['instance_id']/'intent.json';old.parent.mkdir(parents=True);old.write_text(json.dumps(intent))
  with self.assertRaisesRegex(RuntimeError,'campaign identity'):l.reject_unresolved_campaign_intents(self.boot,self.root)
 def test_extra_unexpected_gpu_rejected(self):
  self.obs['samples'][1]['gpu_csv']+='\nGPU-foreign, 15866, 15, 0, None';self.change('current_observation',self.obs)
  with self.assertRaisesRegex(RuntimeError,'unexpected GPU'):self.validate()
 def test_old_boot_stop_refuses_before_writing_request(self):
  state=self.root/'state/eager-no-cache';state.mkdir(parents=True);(state/'active.json').write_text(json.dumps(self.intent));(self.root/'PROFILES.json').write_text(json.dumps({'profiles':{'eager-no-cache':{}}}))
  with patch.object(l,'ROOT',self.root),self.assertRaisesRegex(RuntimeError,'stale-boot'):l.own_stop('eager-no-cache')
  self.assertEqual(list(self.root.glob('runs/*/stop-request.json')),[])
 def test_shared_raw_current_boot_observation_needs_no_old_boot_field(self):
  source=self.global_capture();del source['old_boot_id'];self.receipt['artifacts']['source_observation']=self.write('source_observation',source);self.obs['source_observation_sha256']=self.receipt['artifacts']['source_observation']['sha256'];self.change('current_observation',self.obs)
  self.assertEqual(self.validate()['old_boot_cleanup'],'NOT_PROVEN')
 def test_publisher_preserves_old_evidence_and_publishes_receipt_last(self):
  import publish_prior_boot_reconciliation as publisher
  from types import SimpleNamespace
  source=self.root/'inputs';self.path.parent.rename(source)
  args=SimpleNamespace(campaign=str(self.root),current_boot=self.boot,source_observation=None,**{kind:str(source/(kind+'.json')) for kind in ['old_intent','old_container','current_observation','current_p2p']})
  before=(source/'old_intent.json').read_bytes()
  with patch.object(publisher.launcher,'STORAGE_ROOT',self.root), patch.object(publisher.launcher,'CAMPAIGN',self.root):result=publisher.publish(args)
  self.assertEqual(result['status'],'PRIOR_BOOT_RECEIPT_PUBLISHED');self.assertEqual((source/'old_intent.json').read_bytes(),before);self.validate()
  with patch.object(publisher.launcher,'STORAGE_ROOT',self.root), patch.object(publisher.launcher,'CAMPAIGN',self.root), self.assertRaisesRegex(RuntimeError,'overwrite refused'):publisher.publish(args)
 def test_publisher_rejects_invalid_evidence_without_admission_receipt(self):
  import publish_prior_boot_reconciliation as publisher
  from types import SimpleNamespace
  source=self.root/'inputs';self.path.parent.rename(source);path=source/'current_observation.json';path.chmod(0o644);bad=json.loads(path.read_text());bad['elapsed_seconds']=1;path.write_text(json.dumps(bad))
  args=SimpleNamespace(campaign=str(self.root),current_boot=self.boot,source_observation=None,**{kind:str(source/(kind+'.json')) for kind in ['old_intent','old_container','current_observation','current_p2p']})
  with patch.object(publisher.launcher,'STORAGE_ROOT',self.root), patch.object(publisher.launcher,'CAMPAIGN',self.root), self.assertRaisesRegex(RuntimeError,'60-second'):publisher.publish(args)
  self.assertFalse(self.path.exists())
 def test_nan_elapsed_rejected(self):
  self.obs['elapsed_seconds']=float('nan');self.change('current_observation',self.obs)
  with self.assertRaisesRegex(RuntimeError,'60-second'):self.validate()
 def test_complete_prior_boot_receipt_allows_only_current_boot_without_old_cleanup_change(self):
  result=self.validate();self.assertEqual(result['old_boot_cleanup'],'NOT_PROVEN');self.assertFalse(self.intent['cleanup_verified'])
 def test_same_boot_receipt_is_never_an_escape(self):
  with self.assertRaisesRegex(RuntimeError,'same-boot'):l.validate_prior_boot_reconciliation(self.intent,self.oldboot,self.path)
 def test_other_current_boot_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'scope/current boot'):l.validate_prior_boot_reconciliation(self.intent,'33333333-3333-3333-3333-333333333333',self.path)
 def test_mutated_old_identity_rejected(self):
  self.receipt['old_identity']['container_id']='e'*64;self.seal_receipt()
  with self.assertRaisesRegex(RuntimeError,'old identity'):self.validate()
 def test_mutated_evidence_hash_rejected(self):
  self.write('old_container',dict(self.old,Name='/another'))
  with self.assertRaisesRegex(RuntimeError,'hash/size'):self.validate()
 def test_mutable_receipt_rejected(self):
  self.path.chmod(0o644)
  with self.assertRaisesRegex(RuntimeError,'immutable'):self.validate()
 def test_short_observation_rejected(self):
  self.obs['elapsed_seconds']=59;self.change('current_observation',self.obs)
  with self.assertRaisesRegex(RuntimeError,'60-second'):self.validate()
 def test_large_sample_gap_rejected(self):
  self.obs['samples'][1]['monotonic_seconds']=101;self.obs['samples'][2]['monotonic_seconds']=159;self.change('current_observation',self.obs)
  with self.assertRaisesRegex(RuntimeError,'timing/gaps'):self.validate()
 def test_any_compute_process_rejected(self):
  self.obs['samples'][1]['compute_csv']='GPU-x, 123, worker';self.change('current_observation',self.obs)
  with self.assertRaisesRegex(RuntimeError,'compute'):self.validate()
 def test_any_fault_sample_rejected(self):
  self.obs['samples'][1]['kernel_fault']=True;self.change('current_observation',self.obs)
  with self.assertRaisesRegex(RuntimeError,'kernel coverage'):self.validate()
 def test_recovery_action_rejected(self):
  self.obs['samples'][1]['gpu_csv']=self.obs['samples'][1]['gpu_csv'].replace('None','Reboot');self.change('current_observation',self.obs)
  with self.assertRaisesRegex(RuntimeError,'recovery'):self.validate()
 def test_missing_gpu_pair_rejected(self):
  self.obs['samples'][1]['gpu_csv']=self.obs['samples'][1]['gpu_csv'].splitlines()[0];self.change('current_observation',self.obs)
  with self.assertRaisesRegex(RuntimeError,'absent/duplicated'):self.validate()
 def test_non_root_p2p_receipt_rejected(self):
  self.obs['p2p_receipt_provenance']['uid']=1000;self.change('current_observation',self.obs)
  with self.assertRaisesRegex(RuntimeError,'provenance'):self.validate()
 def test_stale_p2p_boot_rejected(self):
  self.p2p['boot_id']=self.oldboot;self.change('current_p2p',self.p2p)
  with self.assertRaisesRegex(RuntimeError,'P2P receipt'):self.validate()
 def test_live_or_restarted_old_container_rejected(self):
  for change in ({'Running':True,'Status':'running'},{'StartedAt':'2026-09-13T21:00:00Z'},{'FinishedAt':'2026-09-13T21:00:00Z'}):
   fresh=copy.deepcopy(self.old);fresh['State'].update(change)
   with self.subTest(change=change),self.assertRaisesRegex(RuntimeError,'changed or restarted'):l.validate_prior_boot_reconciliation(self.intent,self.boot,self.path,lambda cid:fresh)
 def test_sibling_candidate_claim_cannot_be_bypassed_by_new_version(self):
  old=self.root/'candidate-v4'/'runs'/self.intent['instance_id']/'intent.json';old.parent.mkdir(parents=True);old.write_text(json.dumps(self.intent))
  self.path.unlink()
  with self.assertRaisesRegex(RuntimeError,'unresolved prior-boot incident'):l.reject_unresolved_campaign_intents(self.boot,self.root)
 def test_valid_reconciled_sibling_claim_is_admitted_without_rewrite(self):
  old=self.root/'candidate-v4'/'runs'/self.intent['instance_id']/'intent.json';old.parent.mkdir(parents=True);old.write_text(json.dumps(self.intent));before=old.read_bytes()
  self.assertEqual(len(l.reject_unresolved_campaign_intents(self.boot,self.root,lambda cid:copy.deepcopy(self.old))),1);self.assertEqual(old.read_bytes(),before)
 def test_pid_reuse_on_new_boot_does_not_count_as_live_old_supervisor(self):
  self.obs['current_pid_reused']=True;self.change('current_observation',self.obs)
  self.assertEqual(self.validate()['old_boot_cleanup'],'NOT_PROVEN')

if __name__=='__main__':unittest.main()
