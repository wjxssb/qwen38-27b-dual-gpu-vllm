"""Derive nested startup spans; never add overlapping worker/parent ranges."""
from common import *
import statistics,collections
candidate=NV/'candidate-v41-startup-observer-r2-20260923'
fault=read(Q/'evidence/recovery-observer-r2-fault-1/receipt.json')
a=fault['new_identity'];t=fault['timeline'];total=t['model_ready']-t['new_supervisor_started']
paths=list((candidate/'runs'/a['instance_id']/'logs/startup-timeline').glob('*.jsonl'))+[Q/'recovery/evidence/host-markers'/f"{a['pid']}.jsonl"]
metrics=[json.loads(l) for l in (Q/'recovery/evidence/observer-campaign-r2/metrics.jsonl').read_text().splitlines()]
rows=[];missing=[];points=[]
for path in paths:
 begins={}
 for line in path.read_text().splitlines():
  x=json.loads(line)
  if x['kind']=='begin':begins[x['span']]=x
  elif x['kind']=='unavailable':missing.append(x)
  elif x['kind']=='point':points.append(x)
  elif x['kind']=='end' and x['span'] in begins:
   b=begins[x['span']];seconds=(x['monotonic_ns']-b['monotonic_ns'])/1e9
   samples=[m for m in metrics if b['monotonic_ns']/1e9<=m['monotonic']<=x['monotonic_ns']/1e9]
   cpusec=x['cpu_user_s']+x['cpu_system_s']-b['cpu_user_s']-b['cpu_system_s']
   row={'stage':x['label'],'pid_namespace':x['pid'],'begin_monotonic_ns':b['monotonic_ns'],'end_monotonic_ns':x['monotonic_ns'],'begin_realtime_ns':b['realtime_ns'],'end_realtime_ns':x['realtime_ns'],'duration_s':seconds,'percent_of_historical_109s':seconds/109.009675*100,'percent_of_observed_reload':seconds/total*100,'process_cpu_s':cpusec,'process_cpu_cores':cpusec/seconds,'rss_peak_kib':x['rss_peak_kib'],'block_read_bytes':(x['read_blocks']-b['read_blocks'])*512,'source':str(path),'samples':len(samples),'succeeded':x['succeeded']}
   row['gpu_metrics']=[{key:{'mean':statistics.mean(m['gpus'][i][key] for m in samples),'max':max(m['gpus'][i][key] for m in samples)} for key in ['gpu_util','vram_bytes','pcie_tx_kb_s','pcie_rx_kb_s']} for i in range(2)] if samples else None
   row['nvme_read_bytes']=None
   if len(samples)>=2:
    before={v[2]:int(v[5])*512 for v in samples[0]['nvme_diskstats'] if v[2] in ['nvme0n1','nvme1n1']}
    after={v[2]:int(v[5])*512 for v in samples[-1]['nvme_diskstats'] if v[2] in before}
    row['nvme_read_bytes']={k:after[k]-v for k,v in before.items()}
    row['nvme_host_throughput_bytes_s']={k:v/(samples[-1]['monotonic']-samples[0]['monotonic']) for k,v in row['nvme_read_bytes'].items()}
   rows.append(row)
rows.sort(key=lambda x:-x['duration_s'])
policy=[
 ('WorkerProc.make_worker_process',True,False,True,False,'Not proven CPU-only preparation; currently after gate','Observe spawn pipe writes before changing startup','Fresh spawn and inherited-fd cleanup must remain'),
 ('BaseRenderer.warmup',True,False,True,False,'Not qualified before hardware gate','Consider CPU renderer warmup concurrent with independent EngineCore','Shared renderer cache and thread state require a join before admission'),
 ('host.verify_inventory',True,True,True,False,'Potential, not implemented','Sealed stat identity cache with full-hash fallback','Never remove integrity verification'),
 ('EngineCore._initialize_kv_caches',True,False,False,True,'No','Keep allocation/profile/warmup order','KV or first-request regression'),
 ('GPUModelRunner.capture_model',True,False,False,True,'No','Retain FULL_DECODE_ONLY capture4; 1.8s not worth eager/lazy complexity','Unvalidated eager path or concurrent capture'),
 ('GPUModelRunner.load_model',True,False,False,True,'No','Do not assume NVMe-bound; inspect nested weight iteration','Live device allocations cannot survive poisoned context'),
 ('init_distributed_environment',True,False,True,True,'No','Separate peer startup wait from communicator time','No reuse of stale NCCL communicator'),
]
for row in rows:
 row.update(required='YES_OR_PARENT_CONTAINER',cacheable='NOT_ESTABLISHED',parallelizable='NOT_ESTABLISHED',gpu_touching='UNKNOWN',safe_overlap_hardware_observation='NOT_QUALIFIED',candidate_optimization='No change without attribution',risk='Nested spans overlap and are not additive')
 for key,required,cacheable,parallel,gpu,overlap,proposal,risk in policy:
  if row['stage'].endswith(key):row.update(required=required,cacheable=cacheable,parallelizable=parallel,gpu_touching=gpu,safe_overlap_hardware_observation=overlap,candidate_optimization=proposal,risk=risk)
result={'state':'MEASURED_DIAGNOSTIC_ONLY','historical_reload_s':109.009675,'observed_reload_s':total,'fault_completion_s':fault['latencies']['first_successful_completion'],'method':'Monotonic startup boundary spans plus 1Hz host/NVML samples. Nested spans and TP ranks overlap; percentages are individual contributions, not an exclusive sum. Old 109s is reference only, not same-process comparison. Timestamped supervisor detection may lag process start by polling.','instance':a['instance_id'],'fault_source':str(Q/'evidence/recovery-observer-r2-fault-1/receipt.json'),'source_hashes':{str(p):sha(p) for p in paths},'stages':rows,'missing_hooks':missing,'points':points,'limitations':['No CUDA API tracing in startup observer. CUDA context/device placement/quantization nested in worker load/init, not independently timed.','1Hz telemetry undersamples short stages; RSS is cumulative process high-water. Disk stats are host-wide and not exclusive NVMe attribution; block counts measure actual block I/O, not page-cache reads.','PCIe telemetry is sampled host throughput; no claim of byte-accurate H2D timing.','Cache hit/miss must come from cache logs/inventory, not inferred solely from startup duration.','Observer effect not yet quantified; compare changes with matching observers and qualify production without observer.']}
save(Q/'recovery/RECOVERY-RELOAD-BREAKDOWN.json',result)
lines=['# Recovery reload breakdown','',result['method'],'',f'MEASURED: observation reload {total:.3f}s; historical reference 109.010s. No recovery optimization promoted.','', '| Stage (nested, per process) | Seconds | % of historical 109s | CPU cores | Block reads |','|---|---:|---:|---:|---:|']
for row in rows:
 if row['duration_s']>=.1:lines.append(f"| {row['pid_namespace']} {row['stage']} | {row['duration_s']:.3f} | {row['percent_of_historical_109s']:.2f}% | {row['process_cpu_cores']:.2f} | {row['block_read_bytes']} |")
lines+=['','## Candidate decisions','']
for p in policy:lines.append(f'- {p[0]}: required={p[1]}, cacheable={p[2]}, parallelizable={p[3]}, GPU-touching={p[4]}. Hardware overlap: {p[5]}. {p[6]}. Risk: {p[7]}.')
lines+=['','## Measurement limits','']+['- '+s for s in result['limitations']]
(Q/'recovery/RECOVERY-RELOAD-BREAKDOWN.md').write_text('\n'.join(lines)+'\n')
print({'reload_s':total,'spans':len(rows),'missing':len(missing)})
