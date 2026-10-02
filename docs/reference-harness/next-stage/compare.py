import argparse, itertools, statistics
from common import *

def logs(row,label):
    if 'raw_logits' in row:return row['raw_logits']
    path=Q/'evidence'/label/f'{label}-{row["count"]}-{row["rep"]}-{row["requested_tokens"]}'/'events.json'
    return [v for event in read(path) for choice in event['event'].get('choices',[]) for v in (choice.get('logprobs') or {}).get('token_logprobs',[]) if v is not None]
def compare(base_label,label,kind='telemetry'):
    base=read(Q/'evidence'/base_label/'summary.json');candidate=read(Q/'evidence'/label/'summary.json')
    assert base['state']==candidate['state']=='PASS'
    result={'baseline':base_label,'candidate':label,'kind':kind,'rows':[],'decision':'PASS','criterion':'paired fixed token workload; cold salts; >=3 formal samples; no >3% median throughput regression; synchronization needs >=3% prefill benefit at two long contexts'}
    for count in sorted({r['count'] for r in candidate['rows']}):
        b=[r for r in base['rows'] if r['count']==count and r['requested_tokens']==256 and not r['warmup']]
        c=[r for r in candidate['rows'] if r['count']==count and r['requested_tokens']==256 and not r['warmup']]
        assert len(b)>=3 and len(c)>=3,('Insufficient formal matched samples',count,len(b),len(c))
        hashes={r['output_hash'] for r in b};correct=all(r['output_hash'] in hashes for r in c)
        bl=[logs(r,base_label) for r in b];cl=[logs(r,label) for r in c]
        assert all(len(values)==256 for values in bl+cl),('Missing selected-token logits',count)
        noise=max((max(abs(x-y) for x,y in zip(a,z)) for a,z in itertools.combinations(bl,2)),default=0)
        delta=max(min(max(abs(x-y) for x,y in zip(a,z)) for a in bl) for z in cl)
        row={'count':count,'baseline_samples':len(b),'candidate_samples':len(c),'deterministic_hash_match':correct,'baseline_logit_repeat_max_abs':noise,'candidate_nearest_baseline_logit_max_abs':delta,'logit_gate':delta<=max(noise,1e-6),'medians':{}}
        for key in ['ttft_seconds','prefill_tok_s','decode_tok_s','mtp_acceptance_rate']:
            x=statistics.median(r[key] for r in b);y=statistics.median(r[key] for r in c)
            row['medians'][key]={'baseline':x,'candidate':y,'change_percent':100*(y/x-1) if x else None}
        row['no_meaningful_regression']=all(row['medians'][key]['change_percent']>=-3 for key in ['prefill_tok_s','decode_tok_s'])
        if not correct or not row['logit_gate'] or not row['no_meaningful_regression']:result['decision']='REVIEW_OR_REJECT'
        result['rows'].append(row)
    assert result['rows'],'No matched workloads'
    result['short_decode_correctness']=[]
    for c in candidate['rows']:
        if c['requested_tokens']!=32:continue
        b=[r for r in base['rows'] if r['count']==c['count'] and r['requested_tokens']==32]
        assert b,('Missing short decode baseline',c['count'])
        correct=c['output_hash'] in {r['output_hash'] for r in b}
        result['short_decode_correctness'].append({'count':c['count'],'deterministic_hash_match':correct})
        if not correct:result['decision']='REVIEW_OR_REJECT'
    if kind!='telemetry':
        wins=[r for r in result['rows'] if r['count']>=35840 and r['medians']['prefill_tok_s']['change_percent']>=3]
        if len(wins)<2 and result['decision']=='PASS':result['decision']='REJECT_NO_MEASURABLE_PREFILL_BENEFIT'
    save(Q/'evidence'/f'{label}-comparison.json',result);print(json.dumps(result,indent=2),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('candidate');p.add_argument('--kind',default='telemetry');a=p.parse_args();compare(a.baseline,a.candidate,a.kind)
