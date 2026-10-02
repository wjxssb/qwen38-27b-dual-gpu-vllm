#!/usr/bin/env python3
"""Staged NVIDIA candidate. Default describe is read-only; run is fail-closed."""
import argparse,contextlib,fcntl,hashlib,json,math,os,re,signal,socket,stat,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STORAGE_ROOT=ROOT.parent
CAMPAIGN=Path('/mnt/storage/ai/qwen38-27b-nvidia-migration-20260913')  # Preserve the existing campaign identity across storage relocation.
LABEL='org.frank.nvidia-migration'
DOCKER=['/usr/bin/docker','--host','unix:///var/run/docker.sock']
LOCKS=[STORAGE_ROOT/'state/gpu.lock']  # Sole production GPU lease on the verified WD filesystem.
GPU_UUIDS=['GPU-8897f327-ceb9-4fab-b84e-d34b24f0b584','GPU-0506b796-f8ea-b40c-616d-5e9d43a9e175']
def read(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def command(args):return subprocess.check_output(args,stderr=subprocess.STDOUT,text=True,timeout=20).strip()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def verify_files(root,entries):
 for name,row in entries.items():
  p=root/name
  if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:
   raise RuntimeError('manifest mismatch: '+name)
def static():
 m=read(ROOT/'MANIFEST.json');verify_files(ROOT,m['files'])
 if sha(STORAGE_ROOT/'PATCH_APPLICABILITY_MATRIX.json')!=m['patch_matrix_sha256']:raise RuntimeError('patch applicability matrix changed')
 p=read(ROOT/'PROFILES.json')
 if p['image']!=m['image'] or p['model_revision']!=m['model_revision']:raise RuntimeError('pinned identity changed')
 return p,m,sha(ROOT/'MANIFEST.json')
def _full_verify_inventory(profiles):
 inv=read(STORAGE_ROOT/'NVIDIA_MODEL_INVENTORY.json')
 if inv['file_integrity']!='PASS' or inv['verified_file_count']!=inv['expected_file_count'] or inv['revision']!=profiles['model_revision']:raise RuntimeError('target payload SHA256 inventory incomplete or revision differs')
 model=Path(profiles['model_host_path']).resolve()
 if Path(inv['model_directory']).resolve()!=model:raise RuntimeError('model directory differs')
 rows={}
 for row in inv['files']:
  path=Path(row['path'])
  if row.get('integrity')!='PASS' or not re.fullmatch('[0-9a-f]{64}',row.get('sha256','')) or not path.resolve().is_relative_to(model):raise RuntimeError('invalid model hash receipt')
  rows[str(path.relative_to(Path(inv['model_directory']))) ]={'bytes':row['size_bytes'],'sha256':row['sha256']}
 verify_files(model,rows)
 return sha(STORAGE_ROOT/'NVIDIA_MODEL_INVENTORY.json')
def verify_inventory(profiles):
 # static() has already verified the helper and receipt against this release.
 import importlib.util
 spec=importlib.util.spec_from_file_location('sealed_model_inventory_cache',ROOT/'inventory_cache.py')
 cache=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache)
 p2p=read('/run/p2p-stable/verified')
 inputs={'inventory_sha256':sha(STORAGE_ROOT/'NVIDIA_MODEL_INVENTORY.json'),'profiles_sha256':sha(ROOT/'PROFILES.json'),'environment_sha256':sha(ROOT/'environment.json'),'model_revision':profiles['model_revision'],'image':profiles['image'],'driver':p2p['driver'],'kernel':p2p['kernel'],'p2p_manifest':p2p['manifest_sha256'],'gpus':p2p['gpus']}
 started=time.monotonic()
 hit=cache.matches(profiles['model_host_path'],read(ROOT/'model-inventory-cache.json'),inputs)
 result=inputs['inventory_sha256'] if hit else _full_verify_inventory(profiles)
 print('MODEL_INVENTORY_VERIFY='+json.dumps({'mode':'SEALED_STAT_IDENTITY' if hit else 'FULL_SHA_FALLBACK','elapsed_seconds':time.monotonic()-started,'inventory_sha256':result}),flush=True)
 return result

def reject_processes(processes):
 for pid,args in processes:
  if not args:continue
  exe=Path(args[0]).name; executed=args[0]
  if exe in ['python','python3','python3.12','bash','sh']:
   # Interpret only an actual script argument. Tool shells whose -c body merely
   # mentions a protected path are not the controller being inspected.
   if '-c' in args[1:] or '-m' in args[1:]:continue
   candidates=[a for a in args[1:] if not a.startswith('-')]
   executed=candidates[0] if candidates else ''
  protected=('qwen38-flash-next' in executed and any(x in executed for x in ['final-validation-','model-audit','llama-server','maintenance']))
  if protected:raise RuntimeError('protected Flash workload/controller active: PID '+str(pid))
def process_list():
 rows=[]
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   try:rows.append((int(p.name),[b.decode(errors='replace') for b in (p/'cmdline').read_bytes().split(b'\0') if b]))
   except (OSError,PermissionError):pass
 return rows
def validate_p2p_hardware(receipt,raw):
 rows=[[x.strip() for x in line.split(',')] for line in raw.splitlines()]
 if len(rows)!=len(GPU_UUIDS) or any(len(row)!=2 for row in rows) or {row[0] for row in rows}!=set(GPU_UUIDS):raise RuntimeError('current P2P GPU identity differs')
 if len(receipt.get('gpus',[]))!=len(GPU_UUIDS) or set(receipt.get('gpus',[]))!=set(GPU_UUIDS):raise RuntimeError('P2P receipt GPU identity differs')
 if not receipt.get('driver') or any(row[1]!=receipt['driver'] for row in rows):raise RuntimeError('current P2P driver differs')
def validate_resume_boot(intent,current_boot=None):
 current_boot=current_boot or Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 if intent.get('boot_id')!=current_boot:raise RuntimeError('stale-boot intent: automatic resume/stop refused; independent reconciliation required')
def verify_p2p():
 p=Path('/run/p2p-stable/verified');s=p.lstat();r=read(p);manifest=Path('/home/frank/p2p-stable/manifest/STABLE-P2P-MANIFEST.json')
 if not stat.S_ISREG(s.st_mode) or s.st_uid!=0 or s.st_mode&0o022:raise RuntimeError('P2P receipt ownership/type invalid')
 if r.get('status')!='VERIFIED_PASS' or r.get('boot_id')!=Path('/proc/sys/kernel/random/boot_id').read_text().strip() or r.get('kernel')!=os.uname().release or r.get('manifest_sha256')!=sha(manifest):raise RuntimeError('current boot P2P proof absent/stale')
 validate_p2p_hardware(r,command(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader,nounits']))
 for unit in ['disable-acs-p2p.service','p2p-stable-verify.service']:
  if command(['systemctl','is-active',unit])!='active':raise RuntimeError('P2P unit inactive: '+unit)
 return r
@contextlib.contextmanager
def leases():
 with contextlib.ExitStack() as stack:
  for p in LOCKS:
   # Existing coordination files only; never create or replace an external lock.
   f=stack.enter_context(p.open('r+'))
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:raise RuntimeError('GPU residency owner holds '+str(p))
  yield

def validate_gpu_rows(raw):
 rows=[]
 for line in raw.splitlines():
  u,free,used,util,recovery=[x.strip() for x in line.split(',')]
  if u not in GPU_UUIDS:raise RuntimeError('unexpected GPU UUID in admitted pair: '+u)
  row={'uuid':u,'free_mib':int(free),'used_mib':int(used),'utilization':int(util),'gpu_recovery_action':recovery};rows.append(row)
  if recovery.lower()!='none':raise RuntimeError('GPU recovery action not clear: '+line)
  if row['free_mib']<14500 or row['used_mib']>1800 or row['utilization']>5:raise RuntimeError('GPU not free for candidate: '+line)
 if len(rows)!=2 or {r['uuid'] for r in rows}!=set(GPU_UUIDS):raise RuntimeError('expected GPU UUID pair absent/duplicated')
 return rows

def preflight(profiles,profile):
 storage_devices=supervisor_module().storage_devices_for_paths([ROOT,Path(profiles['model_host_path']),Path('/var/lib/docker')])
 reject_processes(process_list())
 for unit in ['qwen38-flash-next.service','qwen27b.service']:
  result=subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True,timeout=10)
  if result.stdout.strip() in ['active','activating','deactivating','reloading']:
   # The production unit directly owns this launcher; another live unit or
   # another invocation is still rejected. The authoritative GPU lease stays mandatory.
   own = unit == 'qwen27b.service' and result.stdout.strip() in ['active','activating']
   if own:
    unit_pid = command(['systemctl','--user','show',unit,'-p','MainPID','--value'])
    invocation = command(['systemctl','--user','show',unit,'-p','InvocationID','--value'])
    own = unit_pid == str(os.getpid()) and bool(invocation) and invocation == os.environ.get('INVOCATION_ID')
   if not own:raise RuntimeError('backend service owns residency: '+unit)
 running=command(DOCKER+['ps','--format','{{.ID}}'])
 if running:
  rows=json.loads(command(DOCKER+['inspect',*running.split()]))
  if any(x['HostConfig'].get('DeviceRequests') for x in rows):raise RuntimeError('GPU container already running')
 compute=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'])
 if compute.strip():raise RuntimeError('GPU compute processes present: '+compute)
 gpu_rows=validate_gpu_rows(command(['nvidia-smi','--query-gpu=uuid,memory.free,memory.used,utilization.gpu,gpu_recovery_action','--format=csv,noheader,nounits']))
 with socket.socket() as s:
  s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
  try:s.bind(('127.0.0.1',profile['port']))
  except OSError:raise RuntimeError('candidate port occupied')
 available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
 if available<22*1024**3:raise RuntimeError('host available memory below22GiB admission')
 p2p=verify_p2p();inventory_sha=verify_inventory(profiles)
 image=json.loads(command(DOCKER+['image','inspect',profiles['image']]))[0]
 if image['Id']!=profiles['image']:raise RuntimeError('pinned image identity differs')
 # Recheck ownership after slow model hashing before Docker mutation.
 reject_processes(process_list())
 if command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip():raise RuntimeError('GPU ownership changed during preflight')
 gpu_rows=validate_gpu_rows(command(['nvidia-smi','--query-gpu=uuid,memory.free,memory.used,utilization.gpu,gpu_recovery_action','--format=csv,noheader,nounits']))
 validate_p2p_hardware(p2p,command(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader,nounits']))
 journal=command(['journalctl','-k','-b','--show-cursor','--no-pager'])
 if '-- cursor:' not in journal:raise RuntimeError('fresh kernel journal cursor unavailable')
 if supervisor_module().storage_fault_lines(journal,storage_devices):raise RuntimeError('current boot has unresolved faults on active storage devices')
 return {'storage_devices':storage_devices,'kernel_journal_before':journal,'launch_status':'EXTERNAL_ADMISSION_AND_RUNTIME_QUALIFICATION_REQUIRED','status':'PREFLIGHT_ONLY_PASS','gpu':gpu_rows,'p2p_boot_receipt':p2p,'model_inventory_sha256':inventory_sha,'utc_epoch':time.time(),'note':'No model startup, transport checksum or mathematical correctness implied'}
def validate_owned(record,row):
 if not re.fullmatch('[0-9a-f]{64}',record.get('container_id','')) or row.get('Id')!=record['container_id']:raise RuntimeError('container ID mismatch')
 expected={LABEL+'.campaign':str(CAMPAIGN),LABEL+'.run':record['run_id'],LABEL+'.manifest':record['manifest_sha256'],LABEL+'.profile':record['profile']}
 if any(row['Config'].get('Labels',{}).get(k)!=v for k,v in expected.items()) or row.get('Image')!=record['image']:raise RuntimeError('container ownership mismatch; stop refused')
PRIOR_BOOT_IDENTITY_FIELDS=('instance_id','container_id','container_labels','image','runtime_manifest_sha256','boot_id','pid','supervisor_start_ticks')

def campaign_intents(campaign=None):
 """Version copies do not discard this campaign's durable ownership claims."""
 campaign=Path(campaign or STORAGE_ROOT).resolve();seen=set()
 for candidate in sorted(campaign.glob('candidate*')):
  if not candidate.is_dir() or not (candidate/'runs').exists():continue
  if candidate.is_symlink() or not candidate.resolve().is_relative_to(campaign):raise RuntimeError('candidate ownership path escaped campaign')
  for path in sorted((candidate/'runs').glob('*/intent.json')):
   if path.is_symlink() or not path.resolve().is_relative_to(candidate.resolve()):raise RuntimeError('owned intent path escaped candidate')
   resolved=path.resolve()
   if resolved not in seen:seen.add(resolved);yield resolved

def immutable_receipt_json(path):
 module=supervisor_module();path=Path(path)
 return module.read_immutable_mapping(path.parent,path.name)

def reconciliation_path(intent,current_boot,campaign=None):
 if not re.fullmatch('[0-9a-f]{32}',intent.get('instance_id','')):raise RuntimeError('invalid archived instance ID')
 if not re.fullmatch('[0-9a-f-]{36}',current_boot):raise RuntimeError('invalid current boot ID')
 return Path(campaign or STORAGE_ROOT)/'prior-boot-reconciliations'/intent['instance_id']/current_boot/'reconciliation.json'

def validate_prior_boot_reconciliation(intent,current_boot,receipt_path,inspect_current=None):
 """A new-boot admission exception, never a claim that old cleanup passed."""
 if intent.get('boot_id')==current_boot:raise RuntimeError('same-boot unresolved ownership cannot use prior-boot reconciliation')
 receipt,seal=immutable_receipt_json(receipt_path)
 required={'schema':1,'kind':'PRIOR_BOOT_RECONCILIATION','scope':'ALLOW_FRESH_INSTANCE_ON_CURRENT_BOOT_ONLY',
           'old_boot_cleanup':'NOT_PROVEN','old_incident_preserved':True,'current_boot_id':current_boot}
 if any(type(receipt.get(k)) is not type(v) or receipt.get(k)!=v for k,v in required.items()):raise RuntimeError('prior-boot reconciliation scope/current boot differs')
 expected={k:intent[k] for k in PRIOR_BOOT_IDENTITY_FIELDS}
 if receipt.get('old_identity')!=expected:raise RuntimeError('prior-boot reconciliation old identity differs')
 artifacts=receipt.get('artifacts')
 if not isinstance(artifacts,dict) or set(artifacts) not in ({'old_intent','old_container','current_observation','current_p2p'},{'old_intent','old_container','current_observation','current_p2p','source_observation'}):raise RuntimeError('prior-boot reconciliation artifacts differ')
 reports={};seals=[seal]
 for kind,descriptor in artifacts.items():
  if not isinstance(descriptor,dict) or set(descriptor)!={'path','bytes','sha256'}:raise RuntimeError('reconciliation artifact descriptor differs')
  report,evidence=supervisor_module().read_immutable_mapping(Path(receipt_path).parent,descriptor['path'])
  if descriptor['sha256']!=evidence['sha256'] or type(descriptor['bytes']) is not int or descriptor['bytes']!=evidence['bytes']:raise RuntimeError('reconciliation artifact hash/size differs')
  reports[kind]=report;seals.append(evidence)
 if reports['old_intent']!=intent:raise RuntimeError('archived intent differs from immutable reconciliation snapshot')
 old=reports['old_container']
 if isinstance(old,list) and len(old)==1:old=old[0]
 supervisor_module().exact_owned(intent,old)
 if not supervisor_module().exited_after_start(old):raise RuntimeError('prior-boot container not definitively stopped after start')
 if old['State'].get('Paused') or old['State'].get('Restarting') or old['State'].get('Pid')!=0:raise RuntimeError('prior-boot container state is ambiguous')
 obs=reports['current_observation'];p2p=reports['current_p2p']
 if obs.get('schema')!=1 or obs.get('current_boot_id')!=current_boot or obs.get('old_boot_id')!=intent['boot_id'] or obs.get('old_supervisor_alive_on_recorded_boot') is not False:raise RuntimeError('current observation boot/supervisor identity differs')
 if obs.get('old_supervisor_identity')!={k:intent[k] for k in ('boot_id','pid','supervisor_start_ticks')}:raise RuntimeError('archived supervisor identity differs')
 if type(obs.get('elapsed_seconds')) not in (int,float) or not math.isfinite(obs['elapsed_seconds']) or obs['elapsed_seconds']<60 or not isinstance(obs.get('samples'),list) or len(obs['samples'])<3:raise RuntimeError('full 60-second idle observation absent')
 global_kernel=obs.get('kernel_coverage')
 if global_kernel is not None:
  raw=reports.get('source_observation')
  if raw is None or obs.get('source_observation_sha256')!=artifacts['source_observation']['sha256']:raise RuntimeError('raw observation source binding absent')
  if raw.get('boot_id')!=current_boot or ('old_boot_id' in raw and raw['old_boot_id']!=intent['boot_id']) or raw.get('elapsed_seconds')!=obs['elapsed_seconds'] or len(raw.get('samples',[]))!=len(obs['samples']):raise RuntimeError('raw observation boot/timing/sample count differs')
  for source,sample in zip(raw['samples'],obs['samples']):
   gpu=source.get('gpu') or {};compute=source.get('compute') or {}
   if gpu.get('returncode')!=0 or compute.get('returncode')!=0 or gpu.get('argv')!=['nvidia-smi','--query-gpu=uuid,memory.used,memory.free,utilization.gpu,gpu_recovery_action,driver_version','--format=csv,noheader,nounits'] or compute.get('argv')!=['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits']:raise RuntimeError('raw observation command/status differs')
   rows=[[v.strip() for v in line.split(',')] for line in gpu['stdout'].splitlines()]
   if len(rows)!=2 or any(len(row)!=6 for row in rows):raise RuntimeError('raw GPU telemetry schema differs')
   normalized='\n'.join(', '.join([r[0],r[2],r[1],r[3],r[4]]) for r in rows)
   if normalized.strip()!=sample['gpu_csv'].strip() or any(r[5]!=sample['driver'] for r in rows) or source['elapsed_seconds']!=sample['monotonic_seconds'] or source['epoch']!=sample.get('epoch') or compute['stdout']!=sample['compute_csv']:raise RuntimeError('normalized observation differs from raw sample')
  command=global_kernel.get('command') or {};args=command.get('argv',[]);journal=command.get('stdout','')
  if global_kernel.get('scope')!='FULL_CURRENT_BOOT_AFTER_OBSERVATION' or global_kernel.get('boot_id')!=current_boot or global_kernel.get('captured_epoch',0)<max(x['epoch'] for x in raw['samples']) or command.get('returncode')!=0:raise RuntimeError('global kernel coverage binding failed')
  if not args or args[0]!='journalctl' or '-k' not in args or '-b' not in args or '--show-cursor' not in args or any(x in args for x in ['--since','--after-cursor','-n','--grep','-g']):raise RuntimeError('global kernel capture is not a full boot journal')
  cursors=re.findall(r'^-- cursor: (.+)$',journal,re.M)
  if not cursors or global_kernel.get('kernel_cursor')!=cursors[-1] or ';b='+current_boot.replace('-','')+';' not in ';'+cursors[-1]+';' or supervisor_module().KERNEL_ERRORS.search(journal):raise RuntimeError('global kernel fault/cursor coverage unproven')
 timestamps=[]
 for sample in obs['samples']:
  if sample.get('boot_id')!=current_boot or sample.get('compute_csv','MISSING').strip():raise RuntimeError('observation boot/compute process check failed')
  rows=validate_gpu_rows(sample['gpu_csv'])
  if any(row['used_mib']>128 or row['free_mib']<14500 or row['utilization']>5 for row in rows):raise RuntimeError('prior-boot observation GPU not fully idle')
  if sample.get('driver')!=p2p.get('driver'):raise RuntimeError('observation driver differs from P2P proof')
  if global_kernel is None and (sample.get('kernel_fault') is not False or not sample.get('kernel_cursor')):raise RuntimeError('observation kernel coverage unproven')
  timestamps.append(sample['monotonic_seconds'])
 if any(type(t) not in (int,float) or not math.isfinite(t) for t in timestamps) or timestamps[-1]-timestamps[0]<60 or any(not 0<b-a<=30 for a,b in zip(timestamps,timestamps[1:])):raise RuntimeError('idle observation timing/gaps invalid')
 if global_kernel is None and abs(obs['elapsed_seconds']-(timestamps[-1]-timestamps[0]))>0.001:raise RuntimeError('idle observation elapsed interval differs')
 if global_kernel is not None and not 0<=timestamps[0]<timestamps[-1]<=obs['elapsed_seconds']:raise RuntimeError('raw observation endpoints exceed controller interval')
 if p2p.get('status')!='VERIFIED_PASS' or p2p.get('boot_id')!=current_boot or set(p2p.get('gpus',[]))!=set(GPU_UUIDS) or len(p2p.get('gpus',[]))!=2:raise RuntimeError('current root P2P receipt differs')
 provenance=obs.get('p2p_receipt_provenance') or {}
 if provenance!={'path':'/run/p2p-stable/verified','uid':0,'regular_file':True,'group_or_world_writable':False,'sha256':artifacts['current_p2p']['sha256']}:raise RuntimeError('root P2P receipt provenance absent')
 if not re.fullmatch('[0-9a-f]{64}',p2p.get('manifest_sha256','')) or p2p.get('kernel')!=obs.get('kernel'):raise RuntimeError('P2P kernel/manifest identity absent')
 if inspect_current is not None:
  fresh=inspect_current(intent['container_id']);supervisor_module().exact_owned(intent,fresh)
  if not supervisor_module().exited_after_start(fresh) or fresh['State'].get('Pid')!=0 or fresh['State'].get('Paused') or fresh['State'].get('Restarting') or any(fresh['State'].get(k)!=old['State'].get(k) for k in ('StartedAt','FinishedAt','ExitCode','OOMKilled')):raise RuntimeError('archived container changed or restarted since reconciliation')
 supervisor_module().verify_mapping_seals(seals)
 return {'receipt':str(receipt_path),'sha256':seal['sha256'],'scope':required['scope'],'old_boot_cleanup':'NOT_PROVEN'}

def reject_unresolved_campaign_intents(current_boot=None,campaign=None,inspect_current=None):
 current_boot=current_boot or Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 accepted=[]
 for path in campaign_intents(campaign):
  intent=read(path)
  if intent.get('container_labels',{}).get(LABEL+'.campaign')!=(str(Path(campaign).resolve()) if campaign is not None else str(CAMPAIGN)):raise RuntimeError('owned intent campaign identity differs: '+str(path))
  first=path.parent/'first-incident.json'
  historical=read(first) if first.exists() else {}
  fatal_history=bool(historical.get('hardware_fault')) or str(historical.get('reason','')).startswith(('new kernel fault:','hardware monitoring fault:'))
  if intent.get('cleanup_verified') is True and not intent.get('hardware_fault_latched') and not fatal_history:continue
  receipt=reconciliation_path(intent,current_boot,campaign)
  if intent.get('boot_id')==current_boot:
   import importlib.util
   spec=importlib.util.spec_from_file_location('restart_app_recovery',ROOT/'restart_app_recovery.py');recovery=importlib.util.module_from_spec(spec);spec.loader.exec_module(recovery)
   receipt=Path(campaign or STORAGE_ROOT)/'same-boot-reconciliations'/intent['instance_id']/current_boot/'reconciliation.json'
   if not receipt.exists():raise RuntimeError('same-boot unresolved incident requires RESTART_APP probes and receipt: '+str(path))
   accepted.append(recovery.validate(sys.modules[__name__],intent,current_boot,receipt,inspect_current))
  else:
   if not receipt.exists():raise RuntimeError('unresolved prior-boot incident requires reconciliation: '+str(path))
   accepted.append(validate_prior_boot_reconciliation(intent,current_boot,receipt,inspect_current))
 return accepted

def own_stop(profile):
 if profile not in read(ROOT/'PROFILES.json')['profiles']:raise RuntimeError('unknown profile; stop refused')
 record=read(ROOT/'state'/profile/'active.json')
 validate_resume_boot(record)
 if record.get('profile')!=profile or record.get('container_labels',{}).get(LABEL+'.campaign')!=str(CAMPAIGN):raise RuntimeError('stop request ownership differs')
 run_id=record['instance_id']
 if not re.fullmatch('[0-9a-f]{32}',run_id):raise RuntimeError('invalid instance')
 run_dir=ROOT/'runs'/run_id
 durable_intent=read(run_dir/'intent.json')
 if any(durable_intent.get(k)!=record.get(k) for k in ['instance_id','container_id','container_labels','image','profile','boot_id','runtime_manifest_sha256']):raise RuntimeError('stop active/durable identity differs')
 # Request the owning supervisor, never dispatch stop from an unrelated process.
 supervisor_module().durable(run_dir/'stop-request.json',{'instance_id':run_id,'requested_epoch':time.time()})
 return {'status':'OWNED_SUPERVISOR_STOP_REQUESTED','instance_id':run_id,'container_id':record.get('container_id'),'note':'If supervisor is absent, resume reconciles this exact run while holding leases'}
def planned_create_command(profiles,manifest_sha,name,run_id='STAGED_TEMPLATE_NOT_AN_INSTANCE'):
 profile=profiles['profiles'][name]
 run_dir=ROOT/'runs'/run_id
 cmd=DOCKER+['create','--name','nvidia-dense-'+run_id,'--init','--gpus','"device='+','.join(GPU_UUIDS)+'"','--network','bridge','--publish',f'127.0.0.1:{profile["port"]}:8000','--cpus','8','--memory',str(45*1024**3),'--memory-swap',str(45*1024**3),'--shm-size',str(1024**3),'--ulimit','memlock=67108864:67108864','--user',f'{os.getuid()}:{os.getgid()}','--log-driver','json-file','--log-opt','max-size=32m','--log-opt','max-file=4','--entrypoint','/usr/bin/python3']
 labels={LABEL+'.campaign':str(CAMPAIGN),LABEL+'.run':run_id,LABEL+'.manifest':manifest_sha,LABEL+'.profile':name}
 for k,v in labels.items():cmd+=['--label',k+'='+v]
 for k,v in read(ROOT/'environment.json').items():cmd+=['--env',k+'='+v]
 binding={'instance_id':run_id,'model_revision':profiles['model_revision'],'runtime_manifest_sha256':manifest_sha,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
 cmd+=['--env','NVIDIA_AUDIT_BINDING_JSON='+json.dumps(binding,separators=(',',':'))]
 mounts=[(profiles['model_host_path'],'/candidate-model',True),(str(ROOT/'entry.py'),'/candidate/entry.py',True),(str(ROOT/'overlay-integrity.json'),'/candidate/overlay-integrity.json',True),(str(run_dir/'cache'),'/cache',False),(str(run_dir/'logs'),'/results',False)]
 for row in read(ROOT/'overlay-integrity.json'):mounts.append((str(ROOT/row['path']),row['target'],True))
 for name in ['nvidia_runtime_inspection','nvidia_runtime_inspection-0.1.0.dist-info']:
  mounts.append((str(ROOT/'inspection'/name),'/usr/local/lib/python3.12/dist-packages/'+name,True))
 mounts.append((str(ROOT/'inspection-integrity.json'),'/candidate/inspection-integrity.json',True))
 for source,target,ro in mounts:cmd+=['--mount',f'type=bind,src={source},dst={target}'+(',readonly' if ro else '')]
 cmd += [profiles['image'],'-I','/candidate/entry.py',*profile['argv']]
 return cmd

def supervisor_module():
 import importlib.util
 spec=importlib.util.spec_from_file_location('candidate_owned_supervisor',ROOT/'supervisor.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def run(profiles,manifest,manifest_sha,name,resume=False):
 module=supervisor_module();active=ROOT/'state'/name/'active.json'
 with leases():
  if resume:
   intent=read(active)
   if intent.get('runtime_manifest_sha256')!=manifest_sha or intent.get('profile')!=name or intent.get('container_labels',{}).get(LABEL+'.campaign')!=str(CAMPAIGN):raise RuntimeError('resume identity/manifest differs')
   run_id=intent['instance_id']
   if not re.fullmatch('[0-9a-f]{32}',run_id):raise RuntimeError('invalid resumed instance')
   run_dir=ROOT/'runs'/run_id
   durable_intent=read(run_dir/'intent.json')
   if any(durable_intent.get(k)!=intent.get(k) for k in ['instance_id','runtime_manifest_sha256','container_labels','image','profile']):raise RuntimeError('active and durable intent identity disagree')
   intent=durable_intent
   validate_resume_boot(intent)
   intent.update(pid=os.getpid(),supervisor_start_ticks=module.proc_ticks(os.getpid()),stop_dispatched=False)
   supervisor=module.Supervisor(intent,module.LocalIO(run_dir),run_dir,active,stop_requested=True)
  else:
   # A previous uncertain create is a durable ownership claim even if no GPU
   # process is presently visible. Never create a replacement until reconciled.
   def inspect_archived(cid):return json.loads(command(DOCKER+['container','inspect',cid]))[0]
   reconciliations=reject_unresolved_campaign_intents(inspect_current=inspect_archived)
   proof=preflight(profiles,profiles['profiles'][name])
   proof['prior_boot_reconciliations']=reconciliations
   # Slow admission hashing must not hide a newly unresolved sibling run.
   reject_unresolved_campaign_intents(inspect_current=inspect_archived)
   run_id=uuid.uuid4().hex;run_dir=ROOT/'runs'/run_id
   for d in ['cache','logs','state']:(run_dir/d).mkdir(parents=True,exist_ok=True)
   module.durable(run_dir/'preflight.json',proof)
   cursor=re.findall(r'^-- cursor: (.+)$',proof['kernel_journal_before'],re.M)[-1]
   labels={LABEL+'.campaign':str(CAMPAIGN),LABEL+'.run':run_id,LABEL+'.manifest':manifest_sha,LABEL+'.profile':name}
   intent={'instance_id':run_id,'run_id':run_id,'container_name':'nvidia-dense-'+run_id,'container_id':None,'container_labels':labels,'profile':name,'manifest_sha256':manifest_sha,'runtime_manifest_sha256':manifest_sha,'image':profiles['image'],'model_revision':profiles['model_revision'],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'logprobs_mode':'raw_logits','speculative_decoding':profiles['profiles'][name]['speculative_decoding'],'port':profiles['profiles'][name]['port'],'model_alias':'unsloth/Qwen3.8-27B-NVFP4','startup_receipt':str(run_dir/'startup-receipt.json'),'runtime_mapping_recorded':False,'run_dir':str(run_dir),'pid':os.getpid(),'supervisor_start_ticks':module.proc_ticks(os.getpid()),'state':'INTENT_NOT_DISPATCHED','kernel_cursor':cursor,'cleanup_verified':False,'expected_gpu_uuids':GPU_UUIDS,'expected_driver':proof['p2p_boot_receipt']['driver'],'idle_gpu_baseline':proof['gpu']}
   intent['storage_devices']=proof['storage_devices']
   supervisor=module.Supervisor(intent,module.LocalIO(run_dir),run_dir,active)
  # Install signal flags before any external mutation; handlers never invoke Docker.
  old_handlers={sig:signal.signal(sig,lambda signum,frame:setattr(supervisor,'stop_requested',True)) for sig in [signal.SIGTERM,signal.SIGINT]}
  try:
   if not resume:
    try:supervisor.begin(planned_create_command(profiles,manifest_sha,name,run_id))
    except Exception as error:
     # Dispatch may have reached Docker. Hold leases and reconcile unique identity.
     supervisor.stop_requested=True
     try:supervisor.persist('QUARANTINE_CREATE_DISPATCH',error=str(error))
     except Exception:pass
   return supervisor.loop()
  finally:
   for sig,handler in old_handlers.items():signal.signal(sig,handler)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',nargs='?',choices=['describe','preflight','run','resume','stop'],default='describe');ap.add_argument('--profile',default='eager-no-cache');args=ap.parse_args()
 if args.action=='stop':print(json.dumps(own_stop(args.profile)));return 0
 profiles,manifest,manifest_sha=static()
 if args.profile not in profiles['profiles']:raise RuntimeError('unknown profile')
 if args.action=='describe':print(json.dumps({'state':'STAGED_NOT_LAUNCHED','profile':args.profile,'configuration':profiles['profiles'][args.profile],'manifest_sha256':manifest_sha,'overlay_count':len(read(ROOT/'overlay-integrity.json')),'launch_status':'EXTERNAL_ADMISSION_AND_RUNTIME_QUALIFICATION_REQUIRED','create_command_template':planned_create_command(profiles,manifest_sha,args.profile)},indent=2));return 0
 if args.action=='preflight':
  with leases():print(json.dumps(preflight(profiles,profiles['profiles'][args.profile]),indent=2))
  return 0
 return run(profiles,manifest,manifest_sha,args.profile,resume=args.action=='resume')
if __name__=='__main__':
 try:sys.exit(main())
 except Exception as error:print(json.dumps({'status':'BLOCKED','error':str(error),'production_changed':False}),file=sys.stderr);sys.exit(2)
