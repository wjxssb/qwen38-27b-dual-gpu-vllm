"""Fixed-token, cold-cache, Gateway-only GPU observations. No lifecycle writes."""
import argparse, math, re, statistics, threading,sys,shutil
from http.client import HTTPConnection
from common import *

sampler=module('nvml_sampler',NV/'prefill-tp2-closeout-20260914/telemetry.py')
def metric(raw,name):
    return sum(float(line.rsplit(' ',1)[1]) for line in raw.splitlines() if re.match(r'^vllm:'+re.escape(name)+r'(\{|\s)',line))
def quantile(xs,q):
    if not xs:return None
    xs=sorted(xs);at=(len(xs)-1)*q;lo=int(at);return xs[lo]+(xs[min(lo+1,len(xs)-1)]-xs[lo])*(at-lo)
def observation(samples):
    rows=[json.loads(line) for line in samples.read_text().splitlines()]
    result={'samples':len(rows),'gpus':[]}
    for i in (0,1):
        gpu={}
        for key in ['gpu_util','vram_bytes','pcie_tx_kb_s','pcie_rx_kb_s','sm_clock_mhz','power_mw']:
            xs=[r['gpus'][i][key] for r in rows if r['gpus'][i][key] is not None]
            gpu[key]={'mean':statistics.mean(xs),'max':max(xs)} if xs else None
        result['gpus'].append(gpu)
    elapsed=rows[-1]['monotonic']-rows[0]['monotonic'] if len(rows)>1 else 0
    if elapsed:
        delta=sum(max(0,rows[-1]['cpu'][pid]['cpu_ticks']-value['cpu_ticks']) for pid,value in rows[0]['cpu'].items() if value and rows[-1]['cpu'].get(pid))
        result['cpu_cores']=delta/os.sysconf('SC_CLK_TCK')/elapsed
    return result
def completion(candidate,count,tokens,out):
    out.mkdir(parents=True,exist_ok=False)
    a,row,_=assert_idle(candidate)
    pids=[int(x.split()[0]) for x in cmd(['docker','top',a['container_id'],'-eo','pid']).splitlines()[1:]]
    before=http('/metrics',18096).decode();(out/'metrics-before.txt').write_text(before)
    fixture=read(Q/'fixtures'/f'tokens-{count:06d}.json')
    salt='dense-qualification-'+out.name+'-'+str(time.time_ns())
    payload={'model':'qwen38-27b-dense','prompt':fixture['prompt_token_ids'],'max_tokens':tokens,'min_tokens':tokens,'ignore_eos':True,'temperature':0,'seed':0,'add_special_tokens':False,'return_token_ids':True,'logprobs':1,'return_tokens_as_token_ids':True,'stream':True,'stream_options':{'include_usage':True,'continuous_usage_stats':True},'cache_salt':salt}
    raw=json.dumps(payload).encode();save(out/'request-metadata.json',{'prompt_sha256':hashlib.sha256(json.dumps(fixture['prompt_token_ids']).encode()).hexdigest(),'request_sha256':hashlib.sha256(raw).hexdigest(),'cache_salt':salt,'prompt_tokens':count,'decode_tokens':tokens})
    result={'state':'RUNNING','started_epoch':time.time(),'instance_id':a['instance_id'],'candidate':str(candidate),'manifest':sha(candidate/'MANIFEST.json'),'count':count,'requested_tokens':tokens}
    save(out/'result.json',result)
    events=[];ids=[];times=[];texts=[];logs=[];done=False;usage=None;finish=[]
    connection=HTTPConnection('127.0.0.1',18080,timeout=600);start=time.monotonic()
    try:
        with sampler.Sampler(out/'nvml.jsonl',pids,period=.2):
            connection.request('POST','/v1/completions',raw,{'Content-Type':'application/json','X-Request-Id':out.name})
            response=connection.getresponse()
            if response.status!=200:raise RuntimeError(f'HTTP {response.status}: {response.read()[:500]}')
            while True:
                line=response.readline();elapsed=time.monotonic()-start
                if not line:break
                if not line.startswith(b'data:'):continue
                data=line[5:].strip()
                if data==b'[DONE]':done=True;continue
                obj=json.loads(data);events.append({'elapsed':elapsed,'event':obj})
                if obj.get('error'):raise RuntimeError(str(obj['error']))
                if obj.get('usage'):usage=obj['usage']
                for choice in obj.get('choices',[]):
                    tokenids=choice.get('token_ids') or []
                    if tokenids:times.extend([elapsed]*len(tokenids));ids.extend(tokenids)
                    texts.append(choice.get('text') or '')
                    logs.extend(x for x in (choice.get('logprobs') or {}).get('token_logprobs',[]) if x is not None)
                    if choice.get('finish_reason'):finish.append(choice['finish_reason'])
        after=http('/metrics',18096).decode();(out/'metrics-after.txt').write_text(after)
        assert done and finish==['length'],(done,finish)
        assert usage and usage['prompt_tokens']==count and usage['completion_tokens']==tokens and len(ids)==tokens,(usage,len(ids))
        assert len(logs)==tokens and all(math.isfinite(x) for x in logs),'nonfinite/missing logits'
        text=''.join(texts);correct=[]
        for line in text.splitlines()[:-1]:
            m=re.fullmatch(r'(\d{4})\|item=(\w+)\|square=(\d+)\|status=retained',line)
            if m:
                n=int(m[1]);correct.append(n==len(correct)+1 and m[2]==['quartz','cobalt','willow','ember'][(n-1)%4] and int(m[3])==n*n)
        required=10 if tokens>=256 else 1
        assert len(correct)>=required and all(correct),('semantic oracle failed',correct,text[:500])
        deltas={name:metric(after,name)-metric(before,name) for name in ['request_prefill_time_seconds_sum','request_decode_time_seconds_sum','prefix_cache_hits_total','spec_decode_num_accepted_tokens_total','spec_decode_num_draft_tokens_total','spec_decode_num_drafts_total','generation_tokens_total']}
        assert deltas['prefix_cache_hits_total']==0,'cold-cache invalid'
        if tokens>=256:assert deltas['spec_decode_num_draft_tokens_total']>0,'MTP not active'
        current,_=identity(candidate);assert current['instance_id']==a['instance_id'],'runtime changed during request'
        changes=cmd(['journalctl','-k','-b','--since','@'+str(result['started_epoch']),'--no-pager'])
        (out/'kernel.txt').write_text(changes)
        assert not re.search(r'NVRM: Xid|GPU has fallen',changes),'kernel GPU fault'
        failures=list((candidate/'runs'/a['instance_id']/'logs').glob('first-failure-*.json'))
        assert not failures,('runtime fatal',failures)
        unique=sorted(set(times));gaps=[y-x for x,y in zip(unique,unique[1:])]
        result.update(state='PASS',finished_epoch=time.time(),ttft_seconds=times[0],elapsed_seconds=time.monotonic()-start,decode_tok_s=(len(ids)-1)/(times[-1]-times[0]),prefill_tok_s=count/deltas['request_prefill_time_seconds_sum'] if deltas['request_prefill_time_seconds_sum']>0 else None,metrics_delta=deltas,mtp_acceptance_rate=deltas['spec_decode_num_accepted_tokens_total']/deltas['spec_decode_num_draft_tokens_total'] if deltas['spec_decode_num_draft_tokens_total'] else None,stream_gap_p50=quantile(gaps,.5),stream_gap_p95=quantile(gaps,.95),token_delivery_gap_p50=quantile([b-a for a,b in zip(times,times[1:])],.5),token_delivery_gap_p95=quantile([b-a for a,b in zip(times,times[1:])],.95),itl_scope='client token delivery; batched MTP tokens share timestamp; not per-token GPU execution latency',output_ids=ids,output_hash=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),raw_logits=logs,correct_records=len(correct),finite_logits=len(logs),hardware=observation(out/'nvml.jsonl'),usage=usage)
        (out/'output.txt').write_text(text)
    except BaseException as error:
        result.update(state='FAIL',error=repr(error),finished_epoch=time.time());raise
    finally:
        connection.close();save(out/'events.json',events);save(out/'result.json',result)
    from log_observation import observe
    save(out/'synchronization-observed.json',observe(candidate/'runs'/a['instance_id']/'logs/stability-telemetry',result['started_epoch'],result['finished_epoch']))
    print(json.dumps({key:result.get(key) for key in ['state','count','requested_tokens','ttft_seconds','prefill_tok_s','decode_tok_s','mtp_acceptance_rate']}),flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,default=BASE);p.add_argument('--label',required=True);p.add_argument('--counts',type=int,nargs='+',default=[256,1024,16384,35840,67584,131072]);p.add_argument('--repeats',type=int,default=3);p.add_argument('--tokens',type=int,default=256);p.add_argument('--short-decode',action='store_true');a=p.parse_args()
    out=Q/'evidence'/a.label;out.mkdir()
    source_dir=out/'measurement-sources';source_dir.mkdir()
    for source in [Path(__file__),Q/'scripts/common.py',Q/'scripts/log_observation.py',Q/'scripts/loaded_identity.py',NV/'prefill-tp2-closeout-20260914/telemetry.py']:
        shutil.copy2(source,source_dir/source.name)
    save(out/'measurement-source-manifest.json',{'argv':sys.argv,'files':{p.name:sha(p) for p in source_dir.iterdir()},'fixture_inventory':sha(Q/'fixtures/inventory.json')})
    summary={'state':'RUNNING','candidate':str(a.candidate),'rows':[],'label':a.label};save(out/'summary.json',summary)
    try:
        from loaded_identity import collect
        collect(a.candidate,out)
        for count in a.counts:
            for rep in range(a.repeats+1):
                row=completion(a.candidate,count,a.tokens,out/f'{a.label}-{count}-{rep}-{a.tokens}');row.update(warmup=rep==0,rep=rep);summary['rows'].append(row);save(out/'summary.json',summary)
            if a.short_decode and count>=35840:
                row=completion(a.candidate,count,32,out/f'{a.label}-{count}-short-32');row.update(warmup=False,rep='short');summary['rows'].append(row);save(out/'summary.json',summary)
        summary['state']='PASS'
    except BaseException as error:summary.update(state='FAIL',error=repr(error));raise
    finally:save(out/'summary.json',summary)

if __name__=='__main__':main()
