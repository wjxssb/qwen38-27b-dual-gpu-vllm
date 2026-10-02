"""Derive recovery comparisons from preserved real fault receipts."""
from common import *
import statistics

def summarize(rows):
    result={}
    names={'failure_to_completion':lambda t:t['first_successful_completion'],
           'supervisor_to_ready':lambda t:t['model_ready']-t['new_supervisor_started'],
           'old_supervisor_to_new_supervisor':lambda t:t['new_supervisor_started']-t['old_supervisor_gone'],
           'failure_to_detected':lambda t:t['detected'],
           'failure_to_old_supervisor_gone':lambda t:t['old_supervisor_gone']}
    for name,fn in names.items():
        values=[fn(row['latencies']) for row in rows]
        result[name]={'samples_seconds':values,'median_seconds':statistics.median(values),'range_seconds':[min(values),max(values)],'n':len(values)}
    return result

def main():
    renderer_path=Q/'recovery/evidence/renderer-campaign-r2/receipt.json'
    inventory_path=Q/'recovery/evidence/inventory-cache-campaign/receipt.json'
    renderer=read(renderer_path);inventory=read(inventory_path)
    assert renderer['state']==inventory['state']=='PASS_MEASURED_RESTORED_NOT_PROMOTED'
    groups={'A_fallback':renderer['faults']['baseline'],'B_renderer':renderer['faults']['candidate'],'C_renderer_inventory':inventory['faults']}
    result={'state':'MEASURED','design':'A3 then B3 then C3, fresh replacement for every fault; single variable A->B and B->C. Fixed polling phase, exact pidfd worker loss during real 67584-token prefill, matching compatible compile caches. Block order disclosed; descriptive n=3, no statistical population guarantee. No startup observer on measured runtime. Root agent performed light file/document work; no concurrent GPU workload or benchmark.','historical_v1':{'failure_to_completion_seconds':203.28818345069885,'supervisor_to_ready_seconds':109.00967526435852,'evidence':'/home/frank/dense-opt-repair-20260922/qualification/evidence/final-inflight-peer-loss/receipt.json'},'groups':{},'evidence':{str(p):sha(p) for p in [renderer_path,inventory_path]},'limitations':['Formal latencies use existing fault harness wall timestamps; stage observer spans use monotonic timestamps. Wall-clock discontinuities were not independently instrumented in that harness; cross-clock incident conversion is separately documented.','Model throughput requests here are correctness checks, not a new matched throughput qualification.','Selected raw logits and token hashes are exact; no full-vocabulary logit equivalence claim.','Safety includes fresh CUDA/P2P proof, full 60-second hardware observation and replacement observation. No replacement-model GPU work starts before gate; the recovery-only probe is itself part of that gate.']}
    timelines=[]
    for group,rows in groups.items():
        assert len(rows)==3
        result['groups'][group]=summarize(rows)
        for i,row in enumerate(rows,1):
            assert row['state']=='PASS' and row['new_instance_survived_seconds']>=60
            timelines.append({'group':group,'repetition':i,'candidate':row['candidate'],'timeline_epoch':row['timeline'],'latencies_seconds':row['latencies'],'replacement_observation_seconds':row['new_instance_survived_seconds'],'old_identity':row['old_identity'],'new_identity':row['new_identity'],'recovery_run':row['recovery_run'],'probe_returncode':row['cuda_p2p_probe']['returncode']})
    for before,after in [('A_fallback','B_renderer'),('B_renderer','C_renderer_inventory'),('A_fallback','C_renderer_inventory')]:
        result.setdefault('comparisons',{})[before+'__'+after]={}
        for metric in ['failure_to_completion','supervisor_to_ready']:
            a=result['groups'][before][metric]['median_seconds'];b=result['groups'][after][metric]['median_seconds']
            result['comparisons'][before+'__'+after][metric]={'reduction_seconds':a-b,'reduction_percent':100*(a-b)/a}
    result['timelines']=timelines
    # Preserve original v1 numbers exactly from their receipt.
    old=read(result['historical_v1']['evidence'])['latencies']
    result['historical_v1']['failure_to_completion_seconds']=old['first_successful_completion']
    result['historical_v1']['supervisor_to_ready_seconds']=old['model_ready']-old['new_supervisor_started']
    final=Q/'evidence/final-production-worker-loss/receipt.json'
    if final.exists() and read(final)['state']=='PASS':
        row=read(final);result['production_copy_fault']={'evidence':str(final),'sha256':sha(final),'metrics':summarize([row]),'latencies_seconds':row['latencies'],'replacement_observation_seconds':row['new_instance_survived_seconds']}
    save(Q/'recovery/RECOVERY-OPTIMIZATION-RESULTS.json',result)
    lines=['# Recovery optimization results','',result['design'],'','MEASURED (seconds; median [min, max]):','','| Release | Full completion | Reload | Old supervisor gone → new supervisor |','|---|---:|---:|---:|']
    for name,g in result['groups'].items():
        cells=[]
        for metric in ['failure_to_completion','supervisor_to_ready','old_supervisor_to_new_supervisor']:
            x=g[metric];cells.append(f"{x['median_seconds']:.3f} [{x['range_seconds'][0]:.3f}, {x['range_seconds'][1]:.3f}]")
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['','DERIVED: reductions are from matched-current repeated baselines, not subtraction from the older single 203.288s receipt. Historical v1 remains intact.','',*['- '+s for s in result['limitations']]]
    (Q/'recovery/RECOVERY-OPTIMIZATION-RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result['comparisons'],indent=2))

if __name__=='__main__':main()
