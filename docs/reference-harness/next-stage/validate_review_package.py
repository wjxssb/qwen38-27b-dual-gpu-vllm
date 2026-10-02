"""Read-only final evidence recomputation; never runs inference or lifecycle."""
from common import *
import re,statistics
P=PROD/'docs/qwen38-optimization'

def main():
    required=['START-HERE.md','CURRENT-PRODUCTION.json','QWEN38-OPTIMIZATION-HISTORY.md','PERFORMANCE-BASELINES.json','RECOVERY-ARCHITECTURE.md','RECOVERY-OPTIMIZATION.md','XID13-INVESTIGATION.md','FAILURES-AND-REJECTED-IDEAS.md','TRANSFERABLE-LESSONS.md','QWEN4-MIGRATION-PLAYBOOK.md','QWEN4-AGENT-BOOTSTRAP.md','EVIDENCE-INDEX.json','SOURCE-MANIFEST.json']
    assert all((P/name).is_file() for name in required)
    manifest=read(P/'SOURCE-MANIFEST.json');checked=0
    for row in manifest['files']+manifest['durable_files']:
        assert sha(row['path'])==row['sha256'],row['path'];checked+=1
        if row.get('durable_copy'):assert sha(row['durable_copy'])==row['durable_sha256']
    index=read(P/'EVIDENCE-INDEX.json');assert index['source_manifest_sha256']==sha(P/'SOURCE-MANIFEST.json')
    missing=[]
    for p in [*(P.glob('*.md')),Q/'FINAL-REPORT.md']:
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            if target.startswith(('http://','https://','#')):continue
            target=target.split('#',1)[0]
            if not (p.parent/target).exists():missing.append({'document':str(p),'target':target})
    assert not missing,missing
    r=read(Q/'recovery/RECOVERY-OPTIMIZATION-RESULTS.json')
    a=read(Q/'recovery/evidence/renderer-campaign-r2/receipt.json')
    c=read(Q/'recovery/evidence/inventory-cache-campaign/receipt.json')
    for label,rows in [('A_fallback',a['faults']['baseline']),('B_renderer',a['faults']['candidate']),('C_renderer_inventory',c['faults'])]:
        assert all(x['state']=='PASS' and x['new_instance_survived_seconds']>=60 for x in rows)
        total=[x['timeline']['first_successful_completion']-x['timeline']['worker_failure'] for x in rows]
        reload=[x['timeline']['model_ready']-x['timeline']['new_supervisor_started'] for x in rows]
        assert statistics.median(total)==r['groups'][label]['failure_to_completion']['median_seconds']
        assert statistics.median(reload)==r['groups'][label]['supervisor_to_ready']['median_seconds']
    for b,cc in zip(a['inference']['candidate'],c['inference']):
        assert b['count']==cc['count'] and b['output_hash']==cc['output_hash'] and b['raw_logits']==cc['raw_logits']
    prod=NV/'production-dense-recovery-20260923';candidate=NV/'candidate-v46-renderer-inventory-cache-20260923';fallback=NV/'production-dense-grouped-20260922'
    assert sha(fallback/'MANIFEST.json')=='87971548b51369f2b028f18bef4275196d70d7079aec3ef830d725257dafb0a5'
    pm=read(prod/'MANIFEST.json');cm=read(candidate/'MANIFEST.json');assert pm['files']==cm['files'] and pm.get('diagnostic_only') is False
    assert pm['qualified_payload_manifest_sha256']==sha(candidate/'MANIFEST.json')
    for rel in ['PROFILES.json','graph-prefix.json','environment.json']:assert sha(prod/rel)==sha(fallback/rel)
    for rel,entry in pm['files'].items():assert sha(prod/rel)==entry['sha256']
    assert not any('startup_observer' in x or 'cupti_modes' in x for x in pm['files'])
    stage=read(Q/'evidence/production-copy-qualification/receipt.json')
    assert stage['state']=='QUALIFIED_ACTIVE_PENDING_INDEPENDENT_REVIEW' and stage['fault']['state']=='PASS'
    for row in stage['inference']:
        ref=next(x for x in c['inference'] if x['count']==row['count'])
        assert row['output_hash']==ref['output_hash'] and row['raw_logits']==ref['raw_logits']
    assert stage['rollback_identity']['runtime_manifest_sha256']==sha(fallback/'MANIFEST.json')
    native=read(Q/'recovery/RECOVERY-NATIVE-FAULT-TIMELINES.json')
    fast=[x for x in native['rows'] if 'inventory-cache-fault-' in x['fault_receipt'] or 'final-production-worker-loss' in x['fault_receipt']]
    assert len(fast)==4 and all(x['model_inventory_verification'] and all(y['record']['mode']=='SEALED_STAT_IDENTITY' for y in x['model_inventory_verification']) for x in fast)
    result={'state':'PASS','epoch':time.time(),'required_documents':len(required),'verified_hash_entries':checked,'broken_top_level_markdown_links':0,'recomputed_fault_groups':3,'real_faults':10,'exact_candidate_and_production_payload':True,'baseline_fallback_manifest_unchanged':True,'formal_inventory_fast_paths':len(fast),'scope':'Root pre-review recomputation; independent reviewer still required.'}
    save(Q/'evidence/root-review-package-check.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
