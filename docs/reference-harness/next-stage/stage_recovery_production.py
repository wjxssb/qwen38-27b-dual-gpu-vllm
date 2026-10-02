"""Qualify an immutable production copy; canonical commit waits for review.

This is an authorized GPU workflow, invoked explicitly after diagnostic restore.
It never modifies the accepted fallback or commits canonical ledgers.
"""
from common import *
import shutil, statistics
import lifecycle
from safe_restore import restore
from bench import completion
from fault_test import fault_test, smoke
from loaded_identity import collect
from functional_checks import functional

FALLBACK = NV/'production-dense-grouped-20260922'
CANDIDATE = NV/'candidate-v46-renderer-inventory-cache-20260923'
DEST = NV/'production-dense-recovery-20260923'

def main():
    continuation=read(Q/'evidence/continuation-ledger.json')
    assert continuation['state']=='QUALIFICATION_FINISHED_RESTORED_PENDING_ROOT_REVIEW'
    assert lifecycle.accepted()==FALLBACK and lifecycle.selected()==FALLBACK
    assert_idle(FALLBACK)
    renderer=read(Q/'recovery/evidence/renderer-campaign-r2/receipt.json')
    cached=read(Q/'recovery/evidence/inventory-cache-campaign/receipt.json')
    assert renderer['state']==cached['state']=='PASS_MEASURED_RESTORED_NOT_PROMOTED'
    assert cached['candidate']==str(CANDIDATE) and cached['manifest']==sha(CANDIDATE/'MANIFEST.json')
    cpu=read(Q/'recovery/evidence/combined-cpu-regression.json')
    assert cpu['state']=='PASS' and cpu['tested_source_mapping']['state']=='PASS'
    assert cpu['tested_source_mapping']['candidate_manifest']==cached['manifest']
    for path,digest in cpu['tested_source_mapping']['files'].items():assert sha(path)==digest
    assert len(cached['faults'])==3 and all(r['state']=='PASS' for r in cached['faults'])
    for r in cached['faults']:
        assert r['old_identity']['runtime_manifest_sha256']==cached['manifest']
    final_reload=statistics.median(cached['summary']['reload_s'])
    final_total=statistics.median(cached['summary']['completion_s'])
    # Explicit meaningful-gain screen. This does not replace safety/review gates.
    assert renderer['summary']['baseline']['reload_median_s']-final_reload>=15
    assert renderer['summary']['baseline']['completion_median_s']-final_total>=15
    manifest=read(CANDIDATE/'MANIFEST.json')
    assert manifest.get('diagnostic_only') is False
    assert manifest['qualification_variable']=='renderer-overlap-inventory-cache'
    for rel in ['PROFILES.json','graph-prefix.json','environment.json']:
        assert sha(CANDIDATE/rel)==sha(FALLBACK/rel)
    out=Q/'evidence/production-copy-qualification';out.mkdir()
    assert not DEST.exists()
    ledgers={str(p):{'sha256':sha(p),'content':read(p)} for p in [Q/'canonical.json',PROD/'dense-production-release.json']}
    save(out/'previous-canonical-ledgers.json',ledgers)
    record={'state':'BUILDING_PROVISIONAL_COPY','candidate':str(CANDIDATE),'production':str(DEST),'fallback':str(FALLBACK),'qualified_manifest':sha(CANDIDATE/'MANIFEST.json'),'started_epoch':time.time(),'canonical_committed':False}
    save(out/'receipt.json',record)
    DEST.mkdir()
    for rel,entry in manifest['files'].items():
        assert sha(CANDIDATE/rel)==entry['sha256']
        target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(CANDIDATE/rel,target)
        assert sha(target)==entry['sha256']
    manifest.update(state='SEALED_PRODUCTION',qualified_candidate_path=str(CANDIDATE),qualified_payload_manifest_sha256=sha(CANDIDATE/'MANIFEST.json'),promotion_receipt=str(out/'receipt.json'))
    save(DEST/'MANIFEST.json',manifest)
    module('production_static',DEST/'launcher.py').static()
    record.update(production_manifest=sha(DEST/'MANIFEST.json'),payload_files_identical=True)
    save(out/'receipt.json',record)
    try:
        lifecycle.activate(DEST)
        collect(DEST,out)
        record['inference']=[]
        for count in [1024,258048]:
            row=completion(DEST,count,256,out/f'production-{count}-256')
            reference=next(r for r in cached['inference'] if r['count']==count)
            assert row['output_hash']==reference['output_hash'] and row['raw_logits']==reference['raw_logits']
            record['inference'].append(row);save(out/'receipt.json',record)
        record['functional']=functional(DEST,out/'functional');save(out/'receipt.json',record)
        record['fault']=fault_test(DEST,'final-production-worker-loss',after_monitor_poll=True,inflight=True)
        save(out/'receipt.json',record)
        record['rollback']=restore(FALLBACK)
        record['rollback_identity']=identity(FALLBACK)[0]
        save(out/'receipt.json',record)
        lifecycle.activate(DEST)
        record['reactivated_smoke']=smoke()
        final_out=out/'final-live';final_out.mkdir();collect(DEST,final_out)
        from final_runtime_identity import collect as collect_final
        record['final_identity']=collect_final(DEST,Q/'evidence/final-control-identity.json')
        for path,old in ledgers.items():assert sha(path)==old['sha256'],'Canonical ledger changed before review'
        record.update(state='QUALIFIED_ACTIVE_PENDING_INDEPENDENT_REVIEW',finished_epoch=time.time())
        save(out/'receipt.json',record)
        save(Q/'PROMOTION-LEDGER.json',{'state':record['state'],'canonical':str(FALLBACK),'active_provisional':str(DEST),'manifest':record['production_manifest'],'receipt':str(out/'receipt.json'),'canonical_commit_gate':'Independent review and closure PASS','epoch':time.time()})
    except BaseException as error:
        record.update(state='FAILED_RESTORING',error=repr(error));save(out/'receipt.json',record)
        try:record['rollback']=restore(FALLBACK);record['state']='REJECTED_ROLLBACK_PASS'
        except BaseException as failure:record.update(state='ROLLBACK_NEEDS_REPAIR',rollback_error=repr(failure))
        save(out/'receipt.json',record);raise

if __name__=='__main__':main()
