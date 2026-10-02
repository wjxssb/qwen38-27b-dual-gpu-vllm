"""Offline reconstruction; unknown CUDA calls remain unknown, not invented."""
from common import *
import datetime,re,shutil

OLD=Path('/home/frank/dense-opt-repair-20260922/qualification')
CASES=[('incident-1','candidate-v35-telemetry-observer-r2-20260922','c157297ed7c74ff3a83e9a5cc8d57ae7',16384),('incident-2','candidate-v35-profile-hoststages-r2-20260922','5860d54035ad4fc8ac75075799ba65ea',35840)]

def main():
    results=[]
    for label,release,instance,count in CASES:
        root=NV/release/'runs'/instance;logs=root/'logs';out=Q/'xid/evidence'/label;out.mkdir(exist_ok=True)
        events=[];sources={};markers=[]
        def source(path):
            sources[str(path)]={'sha256':sha(path),'bytes':path.stat().st_size};return path
        for p in logs.glob('first-failure-*.json'):
            d=read(source(p));shutil.copy2(p,out/p.name);markers.append(d)
            events.append({'kind':'first_failure','wall_ns':d['wall_time_ns'],'monotonic_ns':d['monotonic_ns'],'source':str(p),'role':d.get('process_role'),'reason':d.get('reason'),'message':d.get('message')})
        t=min(d['wall_time_ns'] for d in markers)/1e9
        offset=statistics_median([d['wall_time_ns']-d['monotonic_ns'] for d in markers])
        journal=cmd(['journalctl','-k','-b','--since','@'+str(t-150),'--until','@'+str(t+180),'-o','json','--no-pager'],timeout=30)
        (out/'kernel-window.jsonl').write_text(journal+'\n');source(out/'kernel-window.jsonl')
        for line in journal.splitlines():
            d=json.loads(line);message=d.get('MESSAGE','')
            if re.search(r'NVRM|Xid|PCIe|AER|nvidia',message,re.I):
                events.append({'kind':'kernel','wall_ns':int(d['__REALTIME_TIMESTAMP'])*1000,'monotonic_ns':int(d['__MONOTONIC_TIMESTAMP'])*1000,'source':str(out/'kernel-window.jsonl'),'message':message})
        source(logs/'container.jsonl');selected=[]
        for raw in (logs/'container.jsonl').open():
            d=json.loads(raw)
            if not t-300<=d['at']<=t+30:continue
            for line in d.get('text','').splitlines():
                if not re.search(r'profil|cupti|cuda|sampler|mtp|fatal|engine.*fail|error|xid',line,re.I):continue
                match=re.match(r'^(\d{4}-\d\d-\d\dT\S+Z)',line)
                if match:
                    value=match[1].removesuffix('Z');seconds,_,fraction=value.partition('.')
                    stamp_ns=int(datetime.datetime.fromisoformat(seconds+'+00:00').timestamp())*1000000000+int((fraction+'000000000')[:9])
                else:stamp_ns=int(d['at']*1e9)
                row={'kind':'application','wall_ns':stamp_ns,'monotonic_ns_derived':stamp_ns-offset,'source':str(logs/'container.jsonl'),'message':line,'timestamp_scope':'docker line realtime; monotonic derived from same-boot first-failure clock pair'}
                events.append(row);selected.append(row)
        save(out/'application-window.json',selected)
        for p in (logs/'stability-telemetry').glob('*.jsonl*'):
            source(p);last=[];profile=[]
            for line in p.open():
                try:d=json.loads(line)
                except ValueError:continue
                wall=d.get('wall_time_ns',0)/1e9
                if t-3<=wall<=t+3:last.append(d)
                if t-300<=wall<=t+3 and ('profil' in str(d.get('method','')).lower() or 'profil' in str(d.get('rpc_method','')).lower()):profile.append(d)
            save(out/(p.name+'.tail.json'),last[-300:]);save(out/(p.name+'.profile.json'),profile)
            for d in last[-12:]:events.append({'kind':'worker_or_engine_boundary','wall_ns':d.get('wall_time_ns'),'monotonic_ns':d.get('monotonic_ns'),'source':str(p),'event':d.get('event'),'stage':d.get('stage'),'rpc_method':d.get('rpc_method'),'fields':d})
        profile=read(NV/release/'PROFILES.json');argv=profile['profiles']['graph-prefix']['argv'];i=argv.index('--profiler-config')
        result={'id':label,'confidence':'NOT_PROVEN','runtime':str(NV/release),'manifest_sha256':sha(NV/release/'MANIFEST.json'),'instance':instance,'context':count,'tp':2,'mtp_k':3,'graph':'FULL_DECODE_ONLY capture4','profiler_configuration':json.loads(argv[i+1]),'first_failure_epoch':t,'clock_alignment':{'source':'same-boot first-failure realtime/monotonic pairs plus kernel journal exact clocks','median_realtime_minus_monotonic_ns':offset,'range_ns':max(d['wall_time_ns']-d['monotonic_ns'] for d in markers)-min(d['wall_time_ns']-d['monotonic_ns'] for d in markers)},'events':sorted(events,key=lambda d:d.get('wall_ns') or 0),'sources':sources,'unknown':['Exact CUPTI enable timestamp not emitted separately from start_profile RPC/HTTP.','No complete trace from failing request: last successful CUDA API, first anomalous CUDA return and exact failing graph/kernel cannot be established.','Telemetry submission boundaries do not establish completed GPU operations.'],'interpretation':'Error surfaced during sampler index wait after previous stream work. Neither memcpy nor CUPTI is proven root cause by this stack.'}
        save(out/'timeline.json',result);results.append(result)
    save(Q/'xid/XID13-HISTORICAL-TIMELINE.json',{'state':'RECONSTRUCTED_WITH_EXPLICIT_GAPS','incidents':results,'historical_count':2,'new_reproductions':0,'root_cause':'NOT_PROVEN'})
    print(json.dumps({'incidents':len(results),'event_counts':[len(r['events']) for r in results]}))

def statistics_median(xs):
    xs=sorted(xs);mid=len(xs)//2
    return xs[mid] if len(xs)%2 else (xs[mid-1]+xs[mid])//2

if __name__=='__main__':main()
