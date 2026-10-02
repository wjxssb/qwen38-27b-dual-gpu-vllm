#!/usr/bin/env python3
"""Owned candidate supervisor. Never infer daemon rollback from a CLI failure."""
import datetime,hashlib,json,os,re,signal,stat,subprocess,time
from pathlib import Path
DOCKER=['/usr/bin/docker','--host','unix:///var/run/docker.sock']
LABEL='org.frank.nvidia-migration'
POST_EXIT_OBSERVATION_SECONDS=60
POST_EXIT_MEMORY_TOLERANCE_MIB=128
POST_EXIT_MAX_SAMPLE_GAP_SECONDS=60
KERNEL_ERRORS=re.compile(r'NVRM: Xid|AER:.*(?:error|fatal)|fallen off|illegal memory access|misaligned address|NCCL.*(?:error|watchdog|timeout)|CUDA error|CUBLAS.*error|CUDNN.*error|device-side assert',re.I)
STORAGE_ERRORS=re.compile(r'controller is down|Device not ready|I/O (?:error|Error)|I/O Cmd.*Error|EXT4-fs (?:error|warning)|Aborting journal|Remounting filesystem read-only|reset failed|Removing after probe failure',re.I)

def storage_fault_lines(journal,devices):
 # Match only backing devices recorded at admission. A recovered error on an
 # unrelated disk is not a CUDA/Xid fault and must not request a GPU reboot.
 names=[re.escape(name) for name in devices if re.fullmatch(r'nvme\d+(?:n\d+(?:p\d+)?)?',name)]
 if len(names)!=len(devices):raise RuntimeError('invalid admitted storage device name')
 if not names:return []
 device=re.compile(r'(?<![A-Za-z0-9])(?:'+'|'.join(names)+r')(?![A-Za-z0-9])')
 return [line for line in journal.splitlines() if device.search(line) and STORAGE_ERRORS.search(line)]

def storage_devices_for_paths(paths):
 root_device=os.stat('/home/frank').st_dev
 names=set()
 for path in paths:
  dev=os.stat(path).st_dev
  if dev!=root_device:raise RuntimeError('active model/runtime/container storage is not on the admitted system disk: '+str(path))
  device=(Path('/sys/dev/block')/(str(os.major(dev))+':'+str(os.minor(dev)))).resolve(strict=True)
  fields=dict(line.split('=',1) for line in (device/'uevent').read_text().splitlines() if '=' in line)
  name=fields['DEVNAME']
  match=re.fullmatch(r'(nvme\d+)(n\d+)(p\d+)?',name)
  if not match:raise RuntimeError('expected NVMe-backed system disk')
  names.update([name,match[1],match[1]+match[2]])
 return sorted(names)
IDENTITY_FIELDS=('container_id','container_labels','instance_id','model_revision','runtime_manifest_sha256','boot_id','logprobs_mode','speculative_decoding','startup_receipt','port','model_alias','image','profile')
COLLECTOR_IDENTITY_FIELDS=('instance_id','boot_id','model_revision','runtime_manifest_sha256')
STOP_SIGNALS={signal.SIGTERM,signal.SIGINT}
MAX_MAPPING_ARTIFACT_BYTES=64*1024**2
QUARANTINE_MAX_RECHECK_SECONDS=30
QUARANTINE_HEARTBEAT_SECONDS=10

class HardwareIdentityError(RuntimeError):
 """An admitted GPU/driver/recovery identity no longer holds."""

def fault_evidence(text,source):
 """Retain the first actual matching line, never just the end of a backtrace."""
 lines=text.splitlines();matches=[i for i,line in enumerate(lines) if KERNEL_ERRORS.search(line)]
 first=matches[0] if matches else 0
 excerpt='\n'.join(lines[max(0,first-2):first+9])[:4000]
 return excerpt,{'source':source,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                 'text_bytes':len(text.encode()),'line_count':len(lines),
                 'first_match_line':first+1 if matches else None,
                 'matching_lines':[{'line':i+1,'text':lines[i][:2000]} for i in matches[:32]],
                 'match_count':len(matches),'full_text_preserved':True}


def mapping_fingerprint(info):
 return [info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns,info.st_mode,info.st_uid]

def read_immutable_mapping(root,relative):
 """Read a sealed local artifact without following symlinks or mutable files."""
 root=Path(root).resolve();name=Path(relative)
 if name.is_absolute() or not name.parts or any(p in ['.','..'] for p in name.parts):raise RuntimeError('mapping artifact path must stay relative to run directory')
 path=root
 for part in name.parts:
  path=path/part
  if path.is_symlink():raise RuntimeError('mapping artifact symlink refused')
 if not path.resolve().is_relative_to(root):raise RuntimeError('mapping artifact escaped run directory')
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb') as f:
  before=os.fstat(f.fileno())
  if not stat.S_ISREG(before.st_mode) or before.st_uid!=os.getuid() or before.st_mode&0o222:raise RuntimeError('mapping artifact must be owner-matched immutable regular file')
  if not 0<before.st_size<=MAX_MAPPING_ARTIFACT_BYTES:raise RuntimeError('mapping artifact size outside bounded limit')
  raw=f.read(MAX_MAPPING_ARTIFACT_BYTES+1);after=os.fstat(f.fileno())
 if len(raw)!=before.st_size or mapping_fingerprint(before)!=mapping_fingerprint(after) or mapping_fingerprint(path.lstat())!=mapping_fingerprint(after):raise RuntimeError('mapping artifact changed while read')
 return json.loads(raw),{'path':str(path),'root':str(root),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'fingerprint':mapping_fingerprint(after)}

def verify_mapping_seals(seals):
 for seal in seals:
  path=Path(seal['path'])
  root=Path(seal['root']);walk=root
  for part in path.relative_to(root).parts:
   walk=walk/part
   if walk.is_symlink():raise RuntimeError('immutable mapping evidence path became a symlink')
  if path.is_symlink() or mapping_fingerprint(path.stat())!=seal['fingerprint']:raise RuntimeError('immutable mapping evidence changed: '+str(path))

def validate_mtp_mapping(report):
    config = report.get('effective_config') or {}
    mtp = report.get('mtp') or {}
    expected = {
        'method': 'mtp', 'k': 3, 'use_v2_model_runner': False,
        'target_model': '/candidate-model', 'draft_model': '/candidate-model',
        'draft_model_class': 'vllm.model_executor.models.qwen3_5_mtp.Qwen3_5MTP',
        'draft_tp': 2, 'rejection_method': 'standard',
        'graph_captured': True, 'boundary_version': 1,
    }
    if any(type(mtp.get(k)) is not type(v) or mtp.get(k) != v for k,v in expected.items()):
        raise ValueError('MTP native K3 model/graph/phase contract mismatch')
    if report.get('runner_class') != 'vllm.v1.worker.gpu_model_runner.GPUModelRunner':
        raise ValueError('MTP requires inherited V1 runner')
    if config.get('speculative_config_is_none') is not False or config.get('runner_num_spec_tokens') != 3:
        raise ValueError('MTP no-spec fallback or wrong K')
    if config.get('runner_drafter_class') != 'vllm.v1.spec_decode.eagle.EagleProposer':
        raise ValueError('MTP unexpected proposer')
    if (config.get('compilation') or {}).get('cudagraph_capture_sizes') != [4]:
        raise ValueError('MTP historical target graph shape missing')
    if (config.get('cache') or {}).get('enable_prefix_caching') is not True:
        raise ValueError('MTP prefix disabled')
    if mtp.get('supports_mm_inputs') is not True:
        raise ValueError('MTP Vision embedding path missing')
    params = mtp.get('draft_parameters') or {}
    if not params or 'model.fc.weight' not in params:
        raise ValueError('MTP actual parameter collection missing')
    target = report.get('parameters') or {}
    shared_head = {'lm_head.'+suffix for suffix in ('weight', 'weight_scale', 'input_global_scale', 'weight_global_scale', 'alpha', 'input_global_scale_inv')}
    if not shared_head.issubset(params) or 'model.embed_tokens.weight' not in params:
        raise ValueError('MTP shared head/embedding collection missing')
    for name, row in params.items():
        if not row.get('device','').startswith('cuda:'):
            raise ValueError('MTP draft parameter placement/dtype mismatch: '+name)
        if name.startswith('lm_head.') or name == 'model.embed_tokens.weight':
            peer = target.get('language_model.'+name) or {}
            storage = row.get('storage') or {}
            if (type(storage.get('data_ptr')) is not int or storage['data_ptr'] <= 0
                    or storage.get('bytes',0) <= 0
                    or any(row.get(k) != peer.get(k) for k in ('shape','stride','dtype','device','storage_offset','storage'))):
                raise ValueError('MTP head/embedding is not the exact target-owned tensor: '+name)
        elif row.get('dtype') != 'torch.bfloat16':
            raise ValueError('MTP native layer dtype mismatch: '+name)


def validate_mapping_attestation(run_dir,intent):
 """Validate collection evidence only; this never establishes numerical PASS."""
 attestation,seal=read_immutable_mapping(run_dir,'mapping-attestation.json');seals=[seal]
 expected={'schema':1,'kind':'NVIDIA_RUNTIME_MAPPING_ATTESTATION','scope':'RUNTIME_COLLECTION_ONLY','runtime_mapping_recorded':True,'numerical_qualification':'NOT_PROVEN'}
 if not isinstance(attestation,dict) or any(type(attestation.get(k)) is not type(v) or attestation.get(k)!=v for k,v in expected.items()):raise RuntimeError('mapping attestation scope/status differs')
 if set(attestation)!=set(expected)|{'identity','request_id','collector_sha256','artifacts'}:raise RuntimeError('mapping attestation top-level fields differ')
 identity={k:intent[k] for k in IDENTITY_FIELDS}
 if attestation.get('identity')!=identity:raise RuntimeError('mapping attestation full identity differs')
 request_id=attestation.get('request_id');collector_sha=attestation.get('collector_sha256')
 if not isinstance(request_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{8,128}',request_id):raise RuntimeError('mapping request identity missing')
 if not isinstance(collector_sha,str) or not re.fullmatch('[0-9a-f]{64}',collector_sha):raise RuntimeError('mapping collector source hash missing')
 artifacts=attestation.get('artifacts')
 if not isinstance(artifacts,list) or len(artifacts)!=2:raise RuntimeError('mapping must contain exactly two rank artifacts')
 ranks=set();devices=set();paths=set()
 for artifact in artifacts:
  if not isinstance(artifact,dict) or set(artifact)!={'rank','path','bytes','sha256'}:raise RuntimeError('mapping artifact descriptor fields differ')
  rank=artifact['rank']
  if type(rank) is not int or rank not in [0,1] or rank in ranks:raise RuntimeError('mapping rank absent/duplicated')
  ranks.add(rank)
  if not isinstance(artifact['path'],str):raise RuntimeError('mapping artifact path missing')
  report,report_seal=read_immutable_mapping(run_dir,artifact['path']);seals.append(report_seal)
  if report_seal['path'] in paths:raise RuntimeError('mapping rank artifact duplicated')
  paths.add(report_seal['path'])
  if type(artifact['bytes']) is not int or artifact['bytes']!=report_seal['bytes'] or artifact['sha256']!=report_seal['sha256']:raise RuntimeError('mapping artifact hash/size differs')
  required={'schema':1,'scope':'ACTUAL_RUNTIME_OBSERVATION_ONLY','status':'COLLECTED_UNQUALIFIED','numerical_correctness':'NOT_PROVEN','p2p_data_integrity':'NOT_PROVEN','rank':rank,'local_rank':rank,'request_id':request_id,'collector_sha256':collector_sha}
  if not isinstance(report,dict) or any(type(report.get(k)) is not type(v) or report.get(k)!=v for k,v in required.items()):raise RuntimeError('mapping rank report scope/identity differs')
  if report.get('identity')!={k:intent[k] for k in COLLECTOR_IDENTITY_FIELDS}:raise RuntimeError('mapping rank run identity differs')
  tp=report.get('tensor_parallel') or {}
  if any(type(tp.get(k)) is not int or tp.get(k)!=v for k,v in {'rank':rank,'rank_in_group':rank,'world_size':2}.items()) or tp.get('ranks')!=[0,1] or any(type(r) is not int for r in tp['ranks']):raise RuntimeError('mapping actual TP2 rank group differs')
  device=(report.get('device') or {}).get('uuid','')
  if not isinstance(device,str):raise RuntimeError('mapping actual GPU UUID missing')
  device=device.removeprefix('GPU-').lower()
  expected_devices={u.removeprefix('GPU-').lower() for u in intent['expected_gpu_uuids']}
  if device not in expected_devices or device in devices:raise RuntimeError('mapping GPU UUID absent/duplicated/foreign')
  devices.add(device)
  config=report.get('effective_config') or {}
  if (config.get('parallel') or {}).get('tensor_parallel_size')!=2:raise RuntimeError('mapping runtime is not TP2')
  if intent.get('speculative_decoding') is True:validate_mtp_mapping(report)
  elif config.get('speculative_config_is_none') is not True:raise RuntimeError('mapping runtime is not no-spec TP2')
  if not isinstance(report.get('parameters'),dict) or not report['parameters'] or not isinstance(report.get('modules'),dict) or not report['modules']:raise RuntimeError('mapping actual parameter/module collection absent')
 verify_mapping_seals(seals)
 return {'attestation_sha256':seal['sha256'],'request_id':request_id,'collector_sha256':collector_sha,'rank_artifacts':artifacts,'seals':seals,'scope':'RUNTIME_COLLECTION_ONLY','numerical_qualification':'NOT_PROVEN'}

def durable(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(path);fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)

def exact_owned(intent,row):
 if row.get('Name','').lstrip('/')!=intent['container_name']:raise RuntimeError('owned container name differs')
 if row.get('Image')!=intent['image']:raise RuntimeError('owned image differs')
 if any(row.get('Config',{}).get('Labels',{}).get(k)!=v for k,v in intent['container_labels'].items()):raise RuntimeError('owned labels differ')
 cid=row.get('Id','')
 if not re.fullmatch('[0-9a-f]{64}',cid):raise RuntimeError('invalid container ID')
 if intent.get('container_id') and intent['container_id']!=cid:raise RuntimeError('container ID changed')
 return cid

def exited_after_start(row):
 s=row['State']
 return (not s.get('Running') and s.get('Status') in ['exited','dead']
         and not str(s.get('StartedAt','0001')).startswith('0001')
         and not str(s.get('FinishedAt','0001')).startswith('0001'))

def proc_ticks(pid):
 s=Path(f'/proc/{pid}/stat').read_text();return int(s[s.rfind(')')+2:].split()[19])

class LocalIO:
 def __init__(self,run_dir):self.run_dir=Path(run_dir);self.jobs={};self.handles=[]
 def read(self,args,timeout=15):
  r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
  if r.returncode:raise RuntimeError(f'command failed {args[:4]}: {r.stdout[-2000:]}')
  return r.stdout
 def inspect(self,name):
  r=subprocess.run(DOCKER+['container','inspect',name],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=15)
  if r.returncode:
   if re.search(r'No such (?:object|container):',r.stdout):return None
   raise RuntimeError('Docker unavailable/inspect uncertain: '+r.stdout[-2000:])
  rows=json.loads(r.stdout)
  if len(rows)!=1:raise RuntimeError('ambiguous inspect result')
  return rows[0]
 def dispatch(self,kind,args,timeout=45):
  # Only own CLI processes; daemon outcome is reconciled independently.
  previous=self.jobs.get(kind)
  if previous and previous['process'].poll() is None:raise RuntimeError('owned CLI still pending: '+kind)
  if previous and not previous['handle'].closed:previous['handle'].close()
  f=(self.run_dir/'logs'/(kind+'-cli.log')).open('ab')
  try:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT)
  except Exception:f.close();raise
  self.jobs[kind]={'process':p,'handle':f,'started':time.monotonic(),'timeout':timeout,'timed_out':False}
 def poll(self):
  result={}
  for kind,j in self.jobs.items():
   code=j['process'].poll()
   if code is not None and not j['handle'].closed:j['handle'].close()
   if code is None and not j['timed_out'] and time.monotonic()-j['started']>j['timeout']:
    j['timed_out']=True
    # Terminate only this supervisor's own Docker client. This is NOT a
    # container stop and does not imply any daemon-side operation rolled back.
    try:j['process'].terminate()
    except ProcessLookupError:pass
   result[kind]={'returncode':code,'timed_out':j['timed_out'],'outcome':'UNKNOWN' if j['timed_out'] else ('PENDING' if code is None else 'CLIENT_FINISHED')}
  return result
 def append(self,name,row):
  p=self.run_dir/'logs'/name
  if p.exists() and p.stat().st_size>8*1024**2:
   # Preserve the full campaign record; only bound individual segment size.
   segment=p.with_name(p.name+'.segment-'+str(time.time_ns()))
   if segment.exists():raise RuntimeError('telemetry segment collision')
   p.replace(segment)
  with p.open('a') as f:
   f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
  fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)

class PostExitUtilizationSettling(RuntimeError):pass

class Supervisor:
 def __init__(self,intent,io,run_dir,active_path,stop_requested=False):
  self.intent=dict(intent);self.io=io;self.run_dir=Path(run_dir);self.active_path=Path(active_path)
  self.clock=time.monotonic;self.post_exit_start=None;self.post_exit_last=None;self.post_exit_samples=0
  self.stop_requested=stop_requested;self.released=False;self.last_monitor=0;self.log_since=intent.get('log_since');self.cursor=intent['kernel_cursor'];self.quarantine=False
  self.mapping_seals=None
  self.incident_counts=dict(intent.get('incident_counts',{}))
  self.quarantine_delay=2;self.next_quarantine_check=0;self.last_quarantine_heartbeat=0
  self.hardware_fault_latched=bool(intent.get('hardware_fault_latched'))
  # Resume never turns a historical hardware fault into a clean release.
  first=self.run_dir/'first-incident.json'
  if first.exists():
   historical=json.loads(first.read_text())
   if historical.get('hardware_fault') or str(historical.get('reason','')).startswith('new kernel fault:'):
    self.hardware_fault_latched=True
 def persist(self,state=None,**fields):
  if state:self.intent['state']=state
  self.intent.update(fields,updated_epoch=time.time());durable(self.run_dir/'intent.json',self.intent);durable(self.active_path,self.intent)
 def identity(self,state,mapping=None):
  if state=='READY' and mapping is None:raise RuntimeError('READY requires supervisor-validated mapping attestation')
  row={k:self.intent[k] for k in IDENTITY_FIELDS}
  row.update(state=state,runtime_mapping_recorded=state=='READY',numerical_qualification='NOT_PROVEN',qualification='NOT_PROVEN')
  if mapping is not None:
   row.update(mapping_attestation_sha256=mapping['attestation_sha256'],mapping_request_id=mapping['request_id'],mapping_scope=mapping['scope'],mapping_rank_artifacts=mapping['rank_artifacts'])
  durable(self.run_dir/'identity.json',row)
  # Qualification is historical once an incident/stop/exit occurs. Preserve
  # evidence separately, but revoke the current receipt used for admission.
  receipt=self.run_dir/'startup-receipt.json'
  if receipt.exists():
   previous=json.loads(receipt.read_text())
   if state!='READY' and (previous.get('state')=='READY' or previous.get('runtime_mapping_recorded')):
    history=self.run_dir/'qualification-before-revocation.json'
    if not history.exists():durable(history,previous)
  durable(receipt,row)
  if state!='READY':self.intent['runtime_mapping_recorded']=False
 def poll_stop_request(self):
  path=self.run_dir/'stop-request.json'
  if os.path.lexists(path):
   if path.is_symlink():raise RuntimeError('stop request symlink refused')
   request=json.loads(path.read_text())
   if request.get('instance_id')!=self.intent['instance_id']:raise RuntimeError('stop request identity mismatch')
   self.stop_requested=True
 def readiness_blocked(self):
  self.poll_stop_request()
  if signal.sigpending()&STOP_SIGNALS:self.stop_requested=True
  return (self.stop_requested or self.intent.get('state')!='RUNNING_UNQUALIFIED'
          or any(self.intent.get(k) for k in ['failure_reason','error','cleanup_verified','stop_dispatched'])
          or os.path.lexists(self.run_dir/'first-incident.json'))
 def maybe_publish_mapping_ready(self,row):
  """Only this supervisor publishes READY; an inspector supplies evidence only."""
  if self.readiness_blocked():return False
  if not row['State'].get('Running') or row['State'].get('Paused') or row['State'].get('Restarting'):raise RuntimeError('mapping publication requires normally running owned container')
  if self.intent.get('mapping_ready_published'):
   if self.mapping_seals is None:raise RuntimeError('mapping publication has no live supervisor seal cache')
   verify_mapping_seals(self.mapping_seals);return True
  if not os.path.lexists(self.run_dir/'mapping-attestation.json'):return False
  try:
   mapping=validate_mapping_attestation(self.run_dir,self.intent)
   # Slow parsing/hashing must precede fresh monitoring and exact Docker state.
   self.monitor(row,force=True)
   fresh=self.io.inspect(self.intent['container_name'])
   if fresh is None:raise RuntimeError('owned container absent before mapping publication')
   exact_owned(self.intent,fresh)
   if not fresh['State'].get('Running') or fresh['State'].get('Paused') or fresh['State'].get('Restarting'):raise RuntimeError('owned container not running normally before mapping publication')
   verify_mapping_seals(mapping['seals'])
   if self.readiness_blocked():return False
   # Python signal handlers run only outside this publication region. A signal
   # already pending forbids publication; one arriving within it is observed
   # immediately afterwards and revokes the receipt before this method returns.
   old_mask=signal.pthread_sigmask(signal.SIG_BLOCK,STOP_SIGNALS)
   try:
    if self.readiness_blocked():return False
    self.persist(runtime_mapping_recorded=True,mapping_ready_published=True,
                 mapping_attestation_sha256=mapping['attestation_sha256'],
                 mapping_request_id=mapping['request_id'],mapping_scope='RUNTIME_COLLECTION_ONLY',
                 numerical_qualification='NOT_PROVEN',qualification='NOT_PROVEN')
    if self.readiness_blocked():
     self.persist(runtime_mapping_recorded=False,mapping_ready_published=False);return False
    self.identity('READY',mapping)
    self.mapping_seals=mapping['seals']
   finally:signal.pthread_sigmask(signal.SIG_SETMASK,old_mask)
   if self.readiness_blocked():
    self.persist(runtime_mapping_recorded=False)
    self.identity('STOPPING_UNQUALIFIED')
    self.io.append('readiness.jsonl',{'at':time.time(),'event':'MAPPING_READY_REVOKED_DURING_PUBLICATION','numerical_qualification':'NOT_PROVEN'})
    return False
   self.io.append('readiness.jsonl',{'at':time.time(),'event':'RUNTIME_COLLECTION_READY_FOR_BASELINE_PROBES','attestation_sha256':mapping['attestation_sha256'],'numerical_qualification':'NOT_PROVEN'})
   # An external stop file can arrive during the notification write as well.
   if self.readiness_blocked():
    self.persist(runtime_mapping_recorded=False);self.identity('STOPPING_UNQUALIFIED');return False
   return True
  except Exception as error:
   self.incident('mapping attestation/publication rejected: '+str(error));return False
 def incident(self,reason,evidence=None,hardware_fault=False):
  self.stop_requested=True;self.post_exit_start=None;self.post_exit_last=None;self.post_exit_samples=0
  reason=str(reason);key=hashlib.sha256(reason.encode()).hexdigest()
  count=self.incident_counts.get(key,0)+1;self.incident_counts[key]=count
  self.hardware_fault_latched=self.hardware_fault_latched or hardware_fault
  self.persist(failure_reason=self.intent.get('failure_reason') or reason,qualification='NOT_PROVEN',
               runtime_mapping_recorded=False,incident_counts=self.incident_counts,
               hardware_fault_latched=self.hardware_fault_latched)
  first=self.run_dir/'first-incident.json'
  row={'at':time.time(),'reason':reason,'reason_sha256':key,'root_cause':'NOT_ATTRIBUTED',
       'hardware_fault':hardware_fault,'reset':False}
  if evidence is not None:row['evidence']=evidence
  if not first.exists():durable(first,row)
  # Repeated observation remains an unresolved failure, but does not overwrite
  # the first record or append the same fault hundreds of times.
  if count==1:
   self.identity('INCIDENT_UNQUALIFIED')
   self.io.append('incidents.jsonl',dict(row,action='GRACEFUL_STOP_OWN_CANDIDATE_ONLY'))
 def defer_quarantine_observation(self):
  self.next_quarantine_check=self.clock()+self.quarantine_delay
  self.quarantine_delay=min(QUARANTINE_MAX_RECHECK_SECONDS,self.quarantine_delay*2)
  self.persist(quarantine_recheck_in_seconds=self.next_quarantine_check-self.clock())
 def begin(self,command):
  if self.stop_requested:
   self.persist('NOT_DISPATCHED_STOP_REQUESTED',cleanup_verified=True);self.released=True;return
  self.persist('CREATE_INTENT_DURABLE',create_dispatched=True,create_command=command)
  self.io.dispatch('create',command)
  self.persist('CREATE_OUTCOME_PENDING')
 def monitor_kernel(self):
  journal=self.io.read(['journalctl','-k','-b','--after-cursor',self.cursor,'--show-cursor','--no-pager','-o','short-iso'])
  matches=re.findall(r'^-- cursor: (.+)$',journal,re.M)
  if not matches and journal.strip() not in ['', '-- No entries --']:raise RuntimeError('kernel journal continuity unproven')
  storage_faults=storage_fault_lines(journal,self.intent.get('storage_devices',[]))
  if storage_faults:
   # Latch stop before attempting telemetry I/O on the affected filesystem.
   self.stop_requested=True
   self.incident('owned storage fault: '+storage_faults[0],evidence={'matching_lines':storage_faults[:8],'after_cursor':self.cursor,'storage_devices':self.intent['storage_devices']},hardware_fault=False)
  # Persist telemetry before advancing durable cursor, so a crash repeats
  # records instead of losing the first kernel fault.
  self.io.append('kernel.jsonl',{'at':time.time(),'after_cursor':self.cursor,'text':journal})
  if KERNEL_ERRORS.search(journal):
   excerpt,evidence=fault_evidence(journal,'logs/kernel.jsonl');evidence['after_cursor']=self.cursor
   self.incident('new kernel fault: '+excerpt,evidence=evidence,hardware_fault=True)
  if matches:self.cursor=matches[-1]
  self.persist(kernel_cursor=self.cursor)
 def monitor_gpu(self):
  gpu=self.io.read(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.sm,gpu_recovery_action,driver_version','--format=csv,noheader,nounits'])
  self.io.append('gpu.jsonl',{'at':time.time(),'csv':gpu})
  gpu_rows=[[x.strip() for x in line.split(',')] for line in gpu.splitlines()]
  expected=self.intent['expected_gpu_uuids']
  if len(gpu_rows)!=len(expected) or {r[0] for r in gpu_rows}!=set(expected):raise HardwareIdentityError('current GPU UUID pair absent/duplicated')
  for values in gpu_rows:
   if len(values)!=9:raise RuntimeError('GPU telemetry schema differs')
   if values[-2].lower()!='none':raise HardwareIdentityError('GPU recovery action: '+','.join(values))
   if values[-1]!=self.intent['expected_driver']:raise HardwareIdentityError('current GPU driver differs from admitted P2P identity')
  return gpu_rows
 def monitor_container(self,row):
  if row is not None and self.intent.get('container_id'):
   cid=self.intent['container_id'];now=datetime.datetime.now(datetime.timezone.utc).isoformat()
   args=DOCKER+['logs','--timestamps','--tail','2000']
   if self.log_since:args+=['--since',self.log_since]
   logs=self.io.read(args+[cid]);self.io.append('container.jsonl',{'at':time.time(),'text':logs})
   if len(logs.splitlines())>=2000:self.incident('container log window saturated; monitoring coverage incomplete')
   if KERNEL_ERRORS.search(logs):
    excerpt,evidence=fault_evidence(logs,'logs/container.jsonl')
    self.incident('candidate runtime error: '+excerpt,evidence=evidence)
   self.log_since=now;self.persist(log_since=now)
   pid=row['State'].get('Pid',0);rss=None
   if pid:
    try:rss=[l for l in Path(f'/proc/{pid}/status').read_text().splitlines() if l.startswith(('VmRSS:','VmSwap:'))]
    except OSError:pass
   workers={}
   if row['State'].get('Running'):
    for token in self.io.read(DOCKER+['top',cid,'-eo','pid']).split():
     if token.isdigit():
      try:workers[token]=[l for l in Path('/proc',token,'status').read_text().splitlines() if l.startswith(('VmRSS:','VmSwap:'))]
      except OSError:workers[token]=['EXITED_DURING_SAMPLE']
   host=[l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemAvailable:','SwapFree:','SwapTotal:'))]
   self.io.append('process.jsonl',{'at':time.time(),'pid':pid,'status':row['State'],'host_pid_memory':rss,'container_process_memory':workers,'host_memory':host})
 def monitor(self,row,force=False):
  if not force and self.clock()-self.last_monitor<3:return
  self.last_monitor=self.clock();errors=[];gpu_rows=None
  # Failure in one channel must not suppress the others. In particular NVML
  # loss after Xid must still preserve the owned container's final traceback.
  for channel,operation in [('kernel',self.monitor_kernel),('gpu',self.monitor_gpu),
                            ('container',lambda:self.monitor_container(row))]:
   try:
    result=operation()
    if channel=='gpu':gpu_rows=result
   except Exception as error:
    if isinstance(error,HardwareIdentityError):
     self.incident('hardware monitoring fault: '+str(error),hardware_fault=True)
    errors.append(channel+': '+str(error))
  if errors:raise RuntimeError('; '.join(errors))
  return gpu_rows
 def request_stop_if_needed(self,row):
  if not self.stop_requested or not row['State'].get('Running'):return
  if self.intent.get('stop_dispatched'):
   if time.time()-self.intent.get('stop_dispatched_epoch',time.time())>120 and not self.intent.get('stop_wait_deadline_recorded'):
    self.incident('owned graceful stop exceeded120s; container exit remains unproven; no force-kill escalation')
    self.persist(stop_wait_deadline_recorded=True,cleanup_verified=False)
   previous=self.intent.get('cli_outcomes',{}).get('stop')
   if not previous or previous.get('returncode') is None or previous.get('returncode')==0:return
   if time.time()-self.intent.get('stop_dispatched_epoch',0)<30:return
   # Retry only idempotent graceful stop after fresh exact ownership and
   # running state. Never escalate to kill or retry create/start.
   self.persist(stop_dispatched=False)
  exact_owned(self.intent,row)
  self.persist('STOP_INTENT_DURABLE',stop_dispatched=True,stop_dispatched_epoch=time.time(),runtime_mapping_recorded=False)
  self.identity('STOPPING_UNQUALIFIED')
  self.io.dispatch('stop',DOCKER+['stop','--timeout','-1',self.intent['container_id']],timeout=10**12)
 def post_exit(self,row):
  state=row['State']
  if self.intent.get('runtime_mapping_recorded'):
   self.persist(runtime_mapping_recorded=False)
   self.identity('EXITED_UNQUALIFIED')
  if state.get('Status') in ['exited','dead']:
   if 'ExitCode' not in state:raise RuntimeError('container exit code unavailable')
   if state.get('OOMKilled') or state['ExitCode']!=0:
    reason='owned container exited: ExitCode='+str(state['ExitCode'])+' OOMKilled='+str(state.get('OOMKilled',False))
    if not self.intent.get('exit_failure_recorded'):
     self.incident(reason);self.persist(exit_failure_recorded=True)
  # Container exit alone is insufficient after delayed-fault history. Require
  # a fresh contiguous interval with journal coverage and no compute process.
  try:
   gpu_rows=self.monitor(row,force=True)
   compute=self.io.read(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'])
   self.io.append('post-exit.jsonl',{'at':time.time(),'compute_csv':compute,'container_state':state,'kernel_cursor':self.cursor})
   if compute.strip():raise RuntimeError('GPU compute process remains after owned container exit: '+compute)
   baseline={g['uuid']:g for g in self.intent['idle_gpu_baseline']}
   if set(baseline)!=set(self.intent['expected_gpu_uuids']):raise RuntimeError('admitted idle GPU baseline identity differs')
   for gpu in gpu_rows:
    admitted=baseline[gpu[0]]
    if int(gpu[1])>admitted['used_mib']+POST_EXIT_MEMORY_TOLERANCE_MIB or int(gpu[2])<admitted['free_mib']-POST_EXIT_MEMORY_TOLERANCE_MIB:raise RuntimeError('GPU memory not restored to admitted idle baseline within128MiB: '+','.join(gpu))
    if int(gpu[1])>1800 or int(gpu[2])<14500:raise RuntimeError('GPU idle memory/utilization not restored after exit: '+','.join(gpu))
   for gpu in gpu_rows:
    if int(gpu[3])>5:
     # NVML utilization covers a sampling interval and can still include the
     # just-exited workload. Only allow this bounded settling interval when
     # memory is already restored, all compute PIDs are absent, and no hardware
     # fault is latched. The subsequent full60s idle window is unchanged.
     finished=datetime.datetime.fromisoformat(state['FinishedAt'].replace('Z','+00:00')).timestamp()
     age=time.time()-finished
     if not self.hardware_fault_latched and 0<=age<10:
      raise PostExitUtilizationSettling('post-exit utilization sampling interval settling: '+','.join(gpu))
     raise RuntimeError('GPU idle memory/utilization not restored after exit: '+','.join(gpu))
  except PostExitUtilizationSettling as error:
   self.post_exit_start=None;self.post_exit_last=None;self.post_exit_samples=0
   self.io.append('post-exit-settling.jsonl',{'at':time.time(),'reason':str(error),'gpu_rows':gpu_rows,'compute_csv':compute,'container_state':state})
   self.persist('POST_EXIT_UTILIZATION_SETTLING',cleanup_verified=False,post_exit_settling_limit_seconds=10)
   return
  except Exception as error:
   self.incident('post-exit observation unavailable/unclear: '+str(error))
   self.quarantine=True;self.persist('QUARANTINE_POST_EXIT_OBSERVATION',cleanup_verified=False)
   self.defer_quarantine_observation();return
  if self.hardware_fault_latched:
   self.quarantine=True;self.persist('QUARANTINE_HARDWARE_FAULT_LATCHED',cleanup_verified=False,hardware_fault_latched=True)
   self.defer_quarantine_observation();return
  self.quarantine_delay=2;self.next_quarantine_check=0
  now=self.clock()
  if self.post_exit_last is not None and now-self.post_exit_last>POST_EXIT_MAX_SAMPLE_GAP_SECONDS:
   self.post_exit_start=None;self.post_exit_samples=0
  if self.post_exit_start is None:self.post_exit_start=now
  self.post_exit_last=now;self.post_exit_samples+=1
  proof={'elapsed_seconds':now-self.post_exit_start,'required_seconds':POST_EXIT_OBSERVATION_SECONDS,'samples':self.post_exit_samples,'gpu_uuids':self.intent['expected_gpu_uuids'],'driver':self.intent['expected_driver'],'kernel_cursor':self.cursor,'all_compute_processes_absent':True,'recovery_actions_clear':True,'idle_memory_utilization_restored':True,'gpu_rows':gpu_rows,'idle_gpu_baseline':self.intent['idle_gpu_baseline'],'memory_tolerance_mib':POST_EXIT_MEMORY_TOLERANCE_MIB}
  self.persist('POST_EXIT_OBSERVING',cleanup_verified=False,post_exit_observation=proof,container_state=state)
  if proof['elapsed_seconds']<POST_EXIT_OBSERVATION_SECONDS or self.post_exit_samples<3:return
  self.persist('EXITED_PRESERVED',cleanup_verified=True,qualification='NOT_PROVEN',outcome='FAILED' if self.intent.get('failure_reason') or self.intent.get('error') else 'STOPPED_UNQUALIFIED')
  self.identity('EXITED_PRESERVED');self.released=True
 def outcome_code(self):
  return 2 if self.intent.get('failure_reason') or self.intent.get('error') else 0
 def tick(self):
  try:
   self.poll_stop_request()
   if self.intent.get('state') in ['QUARANTINE_POST_EXIT_OBSERVATION','QUARANTINE_HARDWARE_FAULT_LATCHED'] and self.clock()<self.next_quarantine_check:
    if self.clock()-self.last_quarantine_heartbeat>=QUARANTINE_HEARTBEAT_SECONDS:
     self.persist(cleanup_verified=False);self.last_quarantine_heartbeat=self.clock()
    return
   outcomes=self.io.poll();self.persist(cli_outcomes=outcomes)
   row=self.io.inspect(self.intent['container_name'])
   if row is None:
    # Never infer rollback from absence after create dispatch. A daemon may
    # complete an already-submitted create after a timed-out CLI disappears.
    self.quarantine=True;self.persist('QUARANTINE_CREATE_OUTCOME_UNKNOWN')
    self.monitor(None);return
   cid=exact_owned(self.intent,row)
   if not self.intent.get('container_id'):self.persist('CREATED_VERIFIED',container_id=cid);self.identity('CREATED_NOT_STARTED')
   # Non-running cleanup performs its own mandatory fresh monitor sample.
   if row['State'].get('Running') or not self.stop_requested and not self.intent.get('start_dispatched'):
    try:self.monitor(row)
    except Exception as error:self.incident('monitor unavailable: '+str(error))
   if row['State'].get('Running'):
    if not self.intent.get('running_observed'):
     self.persist('RUNNING_UNQUALIFIED',running_observed=True);self.identity('RUNNING_UNQUALIFIED')
    self.maybe_publish_mapping_ready(row)
    self.request_stop_if_needed(row);return
   if not self.intent.get('start_dispatched'):
    if self.stop_requested:
     self.post_exit(row);return
    self.persist('START_INTENT_DURABLE',start_dispatched=True)
    self.io.dispatch('start',DOCKER+['start',cid]);return
   if exited_after_start(row):
    self.post_exit(row);return
   # A 'created' state after an uncertain start is not proof it will never
   # run. Quarantine holds both GPU leases until a definitive state is seen.
   self.quarantine=True;self.persist('QUARANTINE_START_OUTCOME_UNKNOWN',container_state=row['State'])
  except Exception as error:
   self.quarantine=True;self.stop_requested=True
   try:self.incident(error);self.persist('QUARANTINE_RECONCILING',error=str(error))
   except Exception:pass
   # No exception may release the caller's leases after Docker dispatch.
 def loop(self):
  def flag(signum,frame):self.stop_requested=True
  old={s:signal.signal(s,flag) for s in [signal.SIGTERM,signal.SIGINT]}
  try:
   while not self.released:self.tick();time.sleep(2)
  finally:
   for s,handler in old.items():signal.signal(s,handler)
  return self.outcome_code()
