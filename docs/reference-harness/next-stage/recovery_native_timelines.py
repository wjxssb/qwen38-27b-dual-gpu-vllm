"""Extract native startup events for each fault without changing timed runtimes."""
from common import *
import re,datetime,calendar

def main():
    paths=sorted((Q/'evidence').glob('renderer-r2-*-fault-*/receipt.json'))+sorted((Q/'evidence').glob('inventory-cache-fault-*/receipt.json'))
    final=Q/'evidence/final-production-worker-loss/receipt.json'
    if final.exists():paths.append(final)
    rows=[]
    patterns={'engine_initialization_begin':'Initializing a V1 LLM engine',
              'model_load_begin':'Starting to load model',
              'weight_load_complete':'Loading weights took',
              'model_load_complete':'Model loading took',
              'graph_ready':'Graph capturing finished',
              'engine_initialization_complete':'init engine (profile',
              'multimodal_warmup_complete':'multi-modal warmup completed',
              'application_admission_ready':'Application startup complete',
              'autotune_cache_path':'Using FlashInfer autotune cache file',
              'autotune_cache_load':re.compile(r'\[Autotuner\]:\s+Loaded\s+\d+\s+configs\s+from\b',re.IGNORECASE),
              'warmup_begin':'Warming up Qwen Triton kernels'}
    for path in paths:
        d=read(path)
        if d['state']!='PASS':continue
        log=Path(d['candidate'])/'runs'/d['new_identity']['instance_id']/'logs/container.jsonl'
        events=[]
        boundary_ns=round(d['timeline']['model_ready']*1e9)
        for record in log.read_text().splitlines():
            for line in json.loads(record).get('text','').splitlines():
                match=re.match(r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\.(\d+)Z (.*)',line)
                if not match:continue
                ns=calendar.timegm(datetime.datetime.strptime(match[1],'%Y-%m-%dT%H:%M:%S').timetuple())*10**9+int(match[2].ljust(9,'0')[:9])
                if ns>boundary_ns:continue
                hits=[name for name,value in patterns.items() if (value.search(match[3]) if isinstance(value,re.Pattern) else value.casefold() in match[3].casefold())]
                if hits:events.append({'events':hits,'timestamp':match[1]+'.'+match[2]+'Z','realtime_ns':ns,'seconds_from_new_supervisor':ns/1e9-d['timeline']['new_supervisor_started'],'text':match[3]})
        journal=cmd(['journalctl','--user','-b','_PID='+str(d['new_identity']['pid']),'--since','@'+str(d['timeline']['new_supervisor_started']-2),'--until','@'+str(d['timeline']['model_ready']+2),'--no-pager','-o','json'])
        verifier=[]
        for line in journal.splitlines():
            if not line.startswith('{'):continue
            j=json.loads(line)
            if 'MODEL_INVENTORY_VERIFY=' in j.get('MESSAGE',''):
                verifier.append({'realtime_us':j['__REALTIME_TIMESTAMP'],'monotonic_us':j.get('__MONOTONIC_TIMESTAMP'),'record':json.loads(j['MESSAGE'].split('MODEL_INVENTORY_VERIFY=',1)[1])})
        rows.append({'fault_receipt':str(path),'fault_receipt_sha256':sha(path),'runtime':d['candidate'],'new_identity':d['new_identity'],'container_log':str(log),'container_log_sha256':sha(log),'events':events,'model_inventory_verification':verifier,'missing_native_event_kinds':[name for name in patterns if not any(name in e['events'] for e in events)],'note':'Native logger timestamps; absent rank/event entries are not synthesized. Detailed nested phase timings come from separate startup observer, not these uninstrumented trials.'})
    save(Q/'recovery/RECOVERY-NATIVE-FAULT-TIMELINES.json',{'state':'EXTRACTED','rows':rows})
    print(json.dumps([{'receipt':r['fault_receipt'],'events':len(r['events']),'inventory':r['model_inventory_verification']} for r in rows],indent=2))

if __name__=='__main__':main()
